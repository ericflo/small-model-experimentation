#!/usr/bin/env python3
"""Trace-bootstrap retention for a frozen Qwen structured latent executor.

The bridge first learns a program-symbol interface from trace supervision. The
central test is whether that interface survives when trace loss is removed and
training continues with only final-answer supervision through the executor.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
    try:
        from transformers import AutoModelForMultimodalLM  # type: ignore
    except Exception:
        AutoModelForMultimodalLM = None  # type: ignore
except Exception as exc:
    raise SystemExit(f"transformers is required: {exc}")


OP_NAMES = ["ADD", "SUB", "MUL"]


@dataclass
class ProgramExample:
    prompt: str
    modulus: int
    length: int
    init_value: int
    ops: List[int]
    args: List[int]
    answer: int
    init_pos: int
    step_op_pos: List[int]
    step_arg_pos: List[int]
    answer_pos: int
    input_ids: List[int]


@dataclass
class FeatureSet:
    init_h: torch.Tensor
    op_h: torch.Tensor
    arg_h: torch.Tensor
    answer_h: torch.Tensor
    lengths: torch.Tensor
    init_value: torch.Tensor
    ops: torch.Tensor
    args: torch.Tensor
    answer: torch.Tensor
    prompts: List[str]


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif getattr(tokenizer, "unk_token", None) is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)


def tokenize_no_special(tokenizer: Any, text: str) -> List[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
    return list(ids)


def apply_op(x: int, op: int, arg: int, modulus: int) -> int:
    if op == 0:
        return (x + arg) % modulus
    if op == 1:
        return (x - arg) % modulus
    if op == 2:
        return (x * arg) % modulus
    raise ValueError(op)


class TextProgramGenerator:
    def __init__(self, tokenizer: Any, modulus: int, max_steps: int, seed: int) -> None:
        self.tokenizer = tokenizer
        self.modulus = int(modulus)
        self.max_steps = int(max_steps)
        self.rng = random.Random(seed)

    def make(self, min_len: int, max_len: int) -> ProgramExample:
        length = self.rng.randint(min_len, max_len)
        init_value = self.rng.randrange(self.modulus)
        x = init_value
        ops: List[int] = []
        args: List[int] = []
        step_specs: List[Tuple[str, str, str]] = []
        for _ in range(length):
            op = self.rng.randrange(3)
            if op == 2:
                arg = self.rng.randint(2, min(self.modulus - 1, 12))
            else:
                arg = self.rng.randint(1, min(self.modulus - 1, 40))
            ops.append(op)
            args.append(arg)
            x = apply_op(x, op, arg, self.modulus)
            if op == 0:
                step_specs.append(("Step: add ", str(arg), "."))
            elif op == 1:
                step_specs.append(("Step: subtract ", str(arg), "."))
            else:
                step_specs.append(("Step: multiply by ", str(arg), "."))
        answer = x

        input_ids: List[int] = []
        prompt_parts: List[str] = []
        init_pos = -1
        step_op_pos: List[int] = []
        step_arg_pos: List[int] = []
        answer_pos = -1

        def add_text(text: str) -> int:
            ids = tokenize_no_special(self.tokenizer, text)
            if not ids:
                raise RuntimeError(f"empty tokenization for text {text!r}")
            input_ids.extend(ids)
            prompt_parts.append(text)
            return len(input_ids) - 1

        add_text(f"Compute a hidden value modulo {self.modulus}.\n")
        add_text("Initial x = ")
        init_pos = add_text(str(init_value))
        add_text(".\n")
        for prefix, number, suffix in step_specs:
            op_pos = add_text(prefix)
            arg_pos = add_text(number)
            add_text(suffix + "\n")
            step_op_pos.append(op_pos)
            step_arg_pos.append(arg_pos)
        add_text("Return x after the final step.\n")
        answer_pos = add_text("Answer:\n")
        prompt = "".join(prompt_parts).rstrip("\n")
        return ProgramExample(
            prompt=prompt,
            modulus=self.modulus,
            length=length,
            init_value=init_value,
            ops=ops,
            args=args,
            answer=answer,
            init_pos=init_pos,
            step_op_pos=step_op_pos,
            step_arg_pos=step_arg_pos,
            answer_pos=answer_pos,
            input_ids=input_ids,
        )

    def dataset(self, n: int, min_len: int, max_len: int) -> List[ProgramExample]:
        return [self.make(min_len, max_len) for _ in range(n)]


def dtype_from_string(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(name)


def load_tokenizer_and_model(args: argparse.Namespace) -> Tuple[Any, nn.Module, str]:
    tokenizer = None
    try:
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
    except Exception as exc:
        if args.verbose:
            print(f"[load] AutoProcessor failed: {exc}", flush=True)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"

    quantization_config = None
    torch_dtype = dtype_from_string(args.torch_dtype)
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype if torch_dtype != torch.float32 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    common: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        common["device_map"] = args.device_map
    if quantization_config is not None:
        common["quantization_config"] = quantization_config

    classes: List[Tuple[str, Any]] = []
    if args.model_class == "causal":
        classes = [("AutoModelForCausalLM", AutoModelForCausalLM)]
    elif args.model_class == "multimodal":
        if AutoModelForMultimodalLM is None:
            raise RuntimeError("AutoModelForMultimodalLM is unavailable")
        classes = [("AutoModelForMultimodalLM", AutoModelForMultimodalLM)]
    else:
        if "qwen3.5" in args.model_id.lower() and AutoModelForMultimodalLM is not None:
            classes.append(("AutoModelForMultimodalLM", AutoModelForMultimodalLM))
        classes.append(("AutoModelForCausalLM", AutoModelForCausalLM))
        if AutoModelForMultimodalLM is not None and classes[0][0] != "AutoModelForMultimodalLM":
            classes.append(("AutoModelForMultimodalLM", AutoModelForMultimodalLM))

    last_exc: Optional[BaseException] = None
    for name, cls in classes:
        try:
            print(f"[load] Loading {args.model_id} with {name}", flush=True)
            model = cls.from_pretrained(args.model_id, **common)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            return tokenizer, model, name
        except Exception as exc:
            last_exc = exc
            print(f"[load] {name} failed: {type(exc).__name__}: {exc}", flush=True)
    if args.text_model_fallback and args.text_model_fallback != args.model_id:
        old_id = args.model_id
        args.model_id = args.text_model_fallback
        try:
            return load_tokenizer_and_model(args)
        finally:
            args.model_id = old_id
    raise RuntimeError(f"could not load model: {last_exc}")


def extract_hidden(outputs: Any) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]
    last = getattr(outputs, "last_hidden_state", None)
    if torch.is_tensor(last):
        return last
    if isinstance(outputs, dict):
        for key in ["hidden_states", "last_hidden_state", "language_model_outputs"]:
            if key in outputs:
                val = outputs[key]
                if isinstance(val, (tuple, list)):
                    return val[-1]
                if torch.is_tensor(val):
                    return val
                try:
                    return extract_hidden(val)
                except Exception:
                    pass
    for attr in ["language_model_outputs", "text_model_output", "model_output"]:
        val = getattr(outputs, attr, None)
        if val is not None:
            return extract_hidden(val)
    raise RuntimeError("model did not return hidden states")


def input_device(model: nn.Module) -> torch.device:
    try:
        return next(model.get_input_embeddings().parameters()).device
    except Exception:
        return next(model.parameters()).device


@torch.no_grad()
def extract_features(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[ProgramExample],
    max_steps: int,
    batch_size: int,
    max_length: int,
    store_dtype: torch.dtype,
) -> FeatureSet:
    device = input_device(model)
    pad_id = int(tokenizer.pad_token_id)
    init_hs: List[torch.Tensor] = []
    op_hs: List[torch.Tensor] = []
    arg_hs: List[torch.Tensor] = []
    answer_hs: List[torch.Tensor] = []
    lengths: List[int] = []
    init_values: List[int] = []
    ops_all: List[List[int]] = []
    args_all: List[List[int]] = []
    answers: List[int] = []
    prompts: List[str] = []

    for start in range(0, len(examples), batch_size):
        chunk = list(examples[start : start + batch_size])
        max_len = min(max(len(ex.input_ids) for ex in chunk), max_length)
        ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        init_pos = []
        answer_pos = []
        op_pos = []
        arg_pos = []
        for i, ex in enumerate(chunk):
            cur = ex.input_ids[:max_len]
            ids[i, : len(cur)] = torch.tensor(cur, dtype=torch.long, device=device)
            mask[i, : len(cur)] = 1
            if (
                ex.answer_pos >= max_len
                or ex.init_pos >= max_len
                or any(p >= max_len for p in ex.step_op_pos)
                or any(p >= max_len for p in ex.step_arg_pos)
            ):
                raise RuntimeError("max_length truncated a required supervision position")
            init_pos.append(ex.init_pos)
            answer_pos.append(ex.answer_pos)
            padded_op = ex.step_op_pos + [-1] * (max_steps - len(ex.step_op_pos))
            padded_arg = ex.step_arg_pos + [-1] * (max_steps - len(ex.step_arg_pos))
            op_pos.append(padded_op[:max_steps])
            arg_pos.append(padded_arg[:max_steps])

        outputs = model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True, return_dict=True)
        hidden = extract_hidden(outputs).detach()
        rows = torch.arange(len(chunk), device=hidden.device)
        init_h = hidden[rows, torch.tensor(init_pos, device=hidden.device)]
        answer_h = hidden[rows, torch.tensor(answer_pos, device=hidden.device)]
        op_tensor = torch.zeros(len(chunk), max_steps, hidden.shape[-1], dtype=hidden.dtype, device=hidden.device)
        arg_tensor = torch.zeros(len(chunk), max_steps, hidden.shape[-1], dtype=hidden.dtype, device=hidden.device)
        for i, poss in enumerate(op_pos):
            for j, pos in enumerate(poss):
                if pos >= 0:
                    op_tensor[i, j] = hidden[i, pos]
        for i, poss in enumerate(arg_pos):
            for j, pos in enumerate(poss):
                if pos >= 0:
                    arg_tensor[i, j] = hidden[i, pos]

        init_hs.append(init_h.to("cpu", dtype=store_dtype))
        answer_hs.append(answer_h.to("cpu", dtype=store_dtype))
        op_hs.append(op_tensor.to("cpu", dtype=store_dtype))
        arg_hs.append(arg_tensor.to("cpu", dtype=store_dtype))
        for ex in chunk:
            lengths.append(ex.length)
            init_values.append(ex.init_value)
            ops_all.append(ex.ops + [-100] * (max_steps - len(ex.ops)))
            args_all.append(ex.args + [-100] * (max_steps - len(ex.args)))
            answers.append(ex.answer)
            prompts.append(ex.prompt)

    return FeatureSet(
        init_h=torch.cat(init_hs, dim=0),
        op_h=torch.cat(op_hs, dim=0),
        arg_h=torch.cat(arg_hs, dim=0),
        answer_h=torch.cat(answer_hs, dim=0),
        lengths=torch.tensor(lengths, dtype=torch.long),
        init_value=torch.tensor(init_values, dtype=torch.long),
        ops=torch.tensor(ops_all, dtype=torch.long),
        args=torch.tensor(args_all, dtype=torch.long),
        answer=torch.tensor(answers, dtype=torch.long),
        prompts=prompts,
    )


class DirectAnswerHead(nn.Module):
    def __init__(self, hidden_dim: int, modulus: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, width),
            nn.SiLU(),
            nn.Linear(width, modulus),
        )

    def forward(self, answer_h: torch.Tensor) -> torch.Tensor:
        return self.net(answer_h.float())


class ProgramCompiler(nn.Module):
    def __init__(self, hidden_dim: int, modulus: int, width: int) -> None:
        super().__init__()
        self.init_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.SiLU(), nn.Linear(width, modulus))
        self.op_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.SiLU(), nn.Linear(width, len(OP_NAMES)))
        self.arg_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.SiLU(), nn.Linear(width, modulus))

    def forward(self, init_h: torch.Tensor, op_h: torch.Tensor, arg_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, steps, dim = op_h.shape
        init_logits = self.init_head(init_h.float())
        op_flat = op_h.float().view(bsz * steps, dim)
        arg_flat = arg_h.float().view(bsz * steps, dim)
        op_logits = self.op_head(op_flat).view(bsz, steps, len(OP_NAMES))
        arg_logits = self.arg_head(arg_flat).view(bsz, steps, -1)
        return init_logits, op_logits, arg_logits


class TransitionExecutor(nn.Module):
    def __init__(self, modulus: int, device: torch.device) -> None:
        super().__init__()
        table = torch.zeros(len(OP_NAMES), modulus, modulus, modulus, dtype=torch.float32)
        for op in range(len(OP_NAMES)):
            for arg in range(modulus):
                for old in range(modulus):
                    new = apply_op(old, op, arg, modulus)
                    table[op, arg, old, new] = 1.0
        self.register_buffer("table", table.to(device))
        self.modulus = modulus

    def soft_forward(
        self,
        init_logits: torch.Tensor,
        op_logits: torch.Tensor,
        arg_logits: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        state = F.softmax(init_logits.float(), dim=-1)
        op_probs = F.softmax(op_logits.float(), dim=-1)
        arg_probs = F.softmax(arg_logits.float(), dim=-1)
        lengths = lengths.to(state.device)
        for t in range(op_logits.shape[1]):
            cand = torch.einsum("bp,oapq->boaq", state, self.table)
            weights = op_probs[:, t, :, None] * arg_probs[:, t, None, :]
            next_state = (cand * weights[:, :, :, None]).sum(dim=(1, 2))
            active = (lengths > t).float().unsqueeze(-1)
            state = active * next_state + (1.0 - active) * state
        return state.clamp_min(1e-9)


def batch_from_features(features: FeatureSet, idx: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "init_h": features.init_h[idx].to(device),
        "op_h": features.op_h[idx].to(device),
        "arg_h": features.arg_h[idx].to(device),
        "answer_h": features.answer_h[idx].to(device),
        "lengths": features.lengths[idx].to(device),
        "init_value": features.init_value[idx].to(device),
        "ops": features.ops[idx].to(device),
        "args": features.args[idx].to(device),
        "answer": features.answer[idx].to(device),
    }


def trace_losses(
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    mask = batch["ops"].ne(-100)
    init_loss = F.cross_entropy(init_logits, batch["init_value"])
    op_loss = F.cross_entropy(op_logits[mask], batch["ops"][mask])
    arg_loss = F.cross_entropy(arg_logits[mask], batch["args"][mask])
    return init_loss + op_loss + arg_loss, {
        "init_loss": float(init_loss.detach().cpu()),
        "op_loss": float(op_loss.detach().cpu()),
        "arg_loss": float(arg_loss.detach().cpu()),
    }


@torch.no_grad()
def argmax_execute(init_pred: torch.Tensor, op_pred: torch.Tensor, arg_pred: torch.Tensor, lengths: torch.Tensor, modulus: int) -> torch.Tensor:
    outs: List[int] = []
    for i in range(init_pred.shape[0]):
        x = int(init_pred[i].item())
        for t in range(int(lengths[i].item())):
            x = apply_op(x, int(op_pred[i, t].item()), int(arg_pred[i, t].item()), modulus)
        outs.append(x)
    return torch.tensor(outs, dtype=torch.long, device=init_pred.device)


@torch.no_grad()
def evaluate_variant(
    variant: str,
    direct: Optional[DirectAnswerHead],
    compiler: Optional[ProgramCompiler],
    executor: TransitionExecutor,
    features: FeatureSet,
    device: torch.device,
    batch_size: int,
    modulus: int,
) -> Dict[str, float]:
    if direct is not None:
        direct.eval()
    if compiler is not None:
        compiler.eval()
    total = 0
    direct_correct = 0
    executor_correct = 0
    init_correct = 0
    op_correct = 0
    arg_correct = 0
    op_total = 0
    program_exact = 0
    mass_sum = 0.0
    direct_mass_sum = 0.0
    n = features.answer.shape[0]
    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n))
        b = batch_from_features(features, idx, device)
        total += b["answer"].numel()
        if direct is not None:
            direct_logits = direct(b["answer_h"])
            direct_probs = F.softmax(direct_logits, dim=-1)
            direct_correct += int(direct_logits.argmax(dim=-1).eq(b["answer"]).sum().item())
            direct_mass_sum += float(direct_probs.gather(1, b["answer"].view(-1, 1)).sum().item())
        if compiler is not None:
            init_logits, op_logits, arg_logits = compiler(b["init_h"], b["op_h"], b["arg_h"])
            probs = executor.soft_forward(init_logits, op_logits, arg_logits, b["lengths"])
            mass_sum += float(probs.gather(1, b["answer"].view(-1, 1)).sum().item())
            init_pred = init_logits.argmax(dim=-1)
            op_pred = op_logits.argmax(dim=-1)
            arg_pred = arg_logits.argmax(dim=-1)
            pred_answer = argmax_execute(init_pred, op_pred, arg_pred, b["lengths"], modulus)
            executor_correct += int(pred_answer.eq(b["answer"]).sum().item())
            init_ok = init_pred.eq(b["init_value"])
            init_correct += int(init_ok.sum().item())
            mask = b["ops"].ne(-100)
            op_ok = op_pred.eq(b["ops"]) | ~mask
            arg_ok = arg_pred.eq(b["args"]) | ~mask
            op_correct += int(op_pred[mask].eq(b["ops"][mask]).sum().item())
            arg_correct += int(arg_pred[mask].eq(b["args"][mask]).sum().item())
            op_total += int(mask.sum().item())
            program_exact += int((init_ok & op_ok.all(dim=1) & arg_ok.all(dim=1)).sum().item())
    return {
        "variant": variant,
        "n": float(total),
        "direct_accuracy": direct_correct / total if direct is not None else math.nan,
        "direct_target_mass": direct_mass_sum / total if direct is not None else math.nan,
        "executor_accuracy": executor_correct / total if compiler is not None else math.nan,
        "executor_target_mass": mass_sum / total if compiler is not None else math.nan,
        "init_accuracy": init_correct / total if compiler is not None else math.nan,
        "op_accuracy": op_correct / op_total if compiler is not None and op_total else math.nan,
        "arg_accuracy": arg_correct / op_total if compiler is not None and op_total else math.nan,
        "program_exact": program_exact / total if compiler is not None else math.nan,
    }


def train_variant(
    variant: str,
    bootstrap_train: FeatureSet,
    answer_train: FeatureSet,
    eval_sets: Dict[str, FeatureSet],
    hidden_dim: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    seed_offset = 0 if variant in {"compiler_trace_then_answer", "compiler_trace_then_answer_low_lr"} else sum(
        ord(ch) for ch in variant
    )
    torch.manual_seed(args.seed + seed_offset)
    direct: Optional[DirectAnswerHead] = None
    compiler: Optional[ProgramCompiler] = None
    modules: List[nn.Module] = []
    if variant in {"direct"}:
        direct = DirectAnswerHead(hidden_dim, args.modulus, args.head_width).to(device)
        modules.append(direct)
    if variant in {
        "compiler_trace",
        "compiler_answer_only",
        "compiler_trace_then_answer",
        "compiler_trace_then_answer_low_lr",
    }:
        compiler = ProgramCompiler(hidden_dim, args.modulus, args.head_width).to(device)
        modules.append(compiler)
    if not modules:
        raise ValueError(variant)
    executor = TransitionExecutor(args.modulus, device)
    opt = torch.optim.AdamW([p for m in modules for p in m.parameters()], lr=args.lr, weight_decay=args.weight_decay)
    base_lrs = [float(group["lr"]) for group in opt.param_groups]
    log_rows: List[Dict[str, Any]] = []
    t0 = time.time()

    def run_stage(
        stage_name: str,
        train_features: FeatureSet,
        steps: int,
        use_trace_loss: bool,
        global_offset: int,
        lr_scale: float = 1.0,
    ) -> int:
        for group, base_lr in zip(opt.param_groups, base_lrs):
            group["lr"] = base_lr * lr_scale
        n = train_features.answer.shape[0]
        for local_step in range(1, steps + 1):
            global_step = global_offset + local_step
            idx = torch.randint(0, n, (args.train_batch_size,))
            b = batch_from_features(train_features, idx, device)
            loss = torch.tensor(0.0, device=device)
            aux: Dict[str, float] = {}
            if direct is not None:
                direct_logits = direct(b["answer_h"])
                direct_loss = F.cross_entropy(direct_logits, b["answer"])
                loss = loss + direct_loss
                aux["direct_loss"] = float(direct_loss.detach().cpu())
            if compiler is not None:
                init_logits, op_logits, arg_logits = compiler(b["init_h"], b["op_h"], b["arg_h"])
                probs = executor.soft_forward(init_logits, op_logits, arg_logits, b["lengths"])
                exec_loss = F.nll_loss(probs.log(), b["answer"])
                loss = loss + args.executor_loss_weight * exec_loss
                aux["executor_loss"] = float(exec_loss.detach().cpu())
                if use_trace_loss:
                    tr_loss, tr_aux = trace_losses(init_logits, op_logits, arg_logits, b)
                    loss = loss + args.trace_loss_weight * tr_loss
                    aux.update(tr_aux)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for m in modules for p in m.parameters()], args.grad_clip)
            opt.step()

            should_eval = (
                local_step == 1
                or local_step == steps
                or global_step % args.eval_every == 0
                or local_step % args.stage_eval_every == 0
            )
            if not should_eval:
                continue
            row: Dict[str, Any] = {
                "stage": stage_name,
                "local_step": local_step,
                "step": global_step,
                "loss": float(loss.detach().cpu()),
                "trace_loss_active": use_trace_loss,
                "lr_scale": lr_scale,
                **aux,
            }
            for name, feats in eval_sets.items():
                metrics = evaluate_variant(variant, direct, compiler, executor, feats, device, args.eval_batch_size, args.modulus)
                for k, v in metrics.items():
                    if k not in {"variant", "n"}:
                        row[f"{name}_{k}"] = v
            log_rows.append(row)
            msg = f"[{variant}] {stage_name} step {global_step} loss={row['loss']:.4f}"
            for name in eval_sets:
                key = f"{name}_executor_accuracy" if compiler is not None else f"{name}_direct_accuracy"
                if key in row and not math.isnan(row[key]):
                    msg += f" {name}={row[key]*100:.1f}%"
            print(msg, flush=True)
        return global_offset + steps

    global_step = 0
    if variant == "direct":
        global_step = run_stage("answer_only", answer_train, args.bootstrap_steps + args.answer_steps, False, global_step)
    elif variant == "compiler_answer_only":
        global_step = run_stage("answer_only", answer_train, args.bootstrap_steps + args.answer_steps, False, global_step)
    elif variant == "compiler_trace":
        global_step = run_stage("trace_bootstrap", bootstrap_train, args.bootstrap_steps, True, global_step)
        global_step = run_stage("trace_continuation", answer_train, args.answer_steps, True, global_step)
    elif variant == "compiler_trace_then_answer":
        global_step = run_stage("trace_bootstrap", bootstrap_train, args.bootstrap_steps, True, global_step)
        global_step = run_stage("answer_retention", answer_train, args.answer_steps, False, global_step)
    elif variant == "compiler_trace_then_answer_low_lr":
        global_step = run_stage("trace_bootstrap", bootstrap_train, args.bootstrap_steps, True, global_step)
        global_step = run_stage(
            "answer_retention_low_lr",
            answer_train,
            args.answer_steps,
            False,
            global_step,
            lr_scale=args.answer_lr_scale,
        )
    else:
        raise ValueError(variant)

    final_metrics = {
        name: evaluate_variant(variant, direct, compiler, executor, feats, device, args.eval_batch_size, args.modulus)
        for name, feats in eval_sets.items()
    }
    ckpt_root = checkpoint_dir(args) / variant
    ckpt_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_root / "checkpoint_final.pt"
    torch.save(
        {
            "variant": variant,
            "args": vars(args),
            "direct": direct.state_dict() if direct is not None else None,
            "compiler": compiler.state_dict() if compiler is not None else None,
            "hidden_dim": hidden_dim,
        },
        ckpt_path,
    )
    return {
        "variant": variant,
        "train_seconds": time.time() - t0,
        "checkpoints": [str(ckpt_path)],
        "train_log": log_rows,
        "final_metrics": final_metrics,
    }


def checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    return Path("large_artifacts/qwen_trace_bootstrap_retention/checkpoints") / Path(args.output_dir).name


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect_metadata(args: argparse.Namespace, loader_name: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "transformers_loader": loader_name,
        "peft_installed": importlib.util.find_spec("peft") is not None,
    }
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
    return meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Frozen Qwen trace-bootstrap retention experiment")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-4B")
    p.add_argument("--text_model_fallback", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--model_class", type=str, default="auto", choices=["auto", "causal", "multimodal"])
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch_dtype", type=str, default="bf16")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--modulus", type=int, default=97)
    p.add_argument("--max_steps", type=int, default=12)
    p.add_argument("--train_min_len", type=int, default=1)
    p.add_argument("--train_max_len", type=int, default=4)
    p.add_argument("--answer_train_min_len", type=int, default=1)
    p.add_argument("--answer_train_max_len", type=int, default=8)
    p.add_argument("--eval_lengths", type=str, default="4,8,12")
    p.add_argument("--train_size", type=int, default=512)
    p.add_argument("--answer_train_size", type=int, default=512)
    p.add_argument("--eval_size", type=int, default=128)
    p.add_argument("--feature_batch_size", type=int, default=8)
    p.add_argument("--train_batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--bootstrap_steps", type=int, default=400)
    p.add_argument("--answer_steps", type=int, default=400)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--stage_eval_every", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--answer_lr_scale", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--head_width", type=int, default=512)
    p.add_argument("--trace_loss_weight", type=float, default=1.0)
    p.add_argument("--executor_loss_weight", type=float, default=1.0)
    p.add_argument(
        "--variants",
        type=str,
        default="direct,compiler_trace,compiler_answer_only,compiler_trace_then_answer,compiler_trace_then_answer_low_lr",
    )
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output_dir", type=str, default="experiments/qwen_trace_bootstrap_retention/runs/default")
    p.add_argument("--checkpoint_dir", type=str, default="")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, loader_name = load_tokenizer_and_model(args)
    metadata = collect_metadata(args, loader_name)
    gen_bootstrap_train = TextProgramGenerator(tokenizer, args.modulus, args.max_steps, args.seed)
    bootstrap_train_examples = gen_bootstrap_train.dataset(args.train_size, args.train_min_len, args.train_max_len)
    print(f"[features] extracting bootstrap_train={len(bootstrap_train_examples)}", flush=True)
    bootstrap_train_features = extract_features(
        model, tokenizer, bootstrap_train_examples, args.max_steps, args.feature_batch_size, args.max_length, torch.float16
    )
    gen_answer_train = TextProgramGenerator(tokenizer, args.modulus, args.max_steps, args.seed + 177)
    answer_train_examples = gen_answer_train.dataset(
        args.answer_train_size, args.answer_train_min_len, args.answer_train_max_len
    )
    print(f"[features] extracting answer_train={len(answer_train_examples)}", flush=True)
    answer_train_features = extract_features(
        model, tokenizer, answer_train_examples, args.max_steps, args.feature_batch_size, args.max_length, torch.float16
    )
    eval_sets: Dict[str, FeatureSet] = {}
    for length in [int(x.strip()) for x in args.eval_lengths.split(",") if x.strip()]:
        gen_eval = TextProgramGenerator(tokenizer, args.modulus, args.max_steps, args.seed + 1000 + length)
        examples = gen_eval.dataset(args.eval_size, length, length)
        print(f"[features] extracting eval_len{length}={len(examples)}", flush=True)
        eval_sets[f"len{length}"] = extract_features(
            model, tokenizer, examples, args.max_steps, args.feature_batch_size, args.max_length, torch.float16
        )
    hidden_dim = int(bootstrap_train_features.init_h.shape[-1])
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results: Dict[str, Any] = {
        "args": vars(args),
        "metadata": metadata,
        "hidden_dim": hidden_dim,
        "dataset": {
            "bootstrap_train_size": args.train_size,
            "answer_train_size": args.answer_train_size,
            "eval_size": args.eval_size,
            "bootstrap_train_lengths": [args.train_min_len, args.train_max_len],
            "answer_train_lengths": [args.answer_train_min_len, args.answer_train_max_len],
            "eval_lengths": [int(x.strip()) for x in args.eval_lengths.split(",") if x.strip()],
        },
        "variants": {},
    }
    flat_rows: List[Dict[str, Any]] = []
    for variant in [x.strip() for x in args.variants.split(",") if x.strip()]:
        variant_result = train_variant(
            variant,
            bootstrap_train_features,
            answer_train_features,
            eval_sets,
            hidden_dim,
            args,
            device,
        )
        results["variants"][variant] = variant_result
        for row in variant_result["train_log"]:
            flat_rows.append({"variant": variant, **row})
    with (out / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    write_csv(out / "train_log.csv", flat_rows)
    print(f"[done] wrote {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
