#!/usr/bin/env python3
"""Structural latent compiler expansion for Qwen.

The experiment trains one executable latent program compiler. There is no beam
search, selector, text program decoding, or reranker. The intervention is
structural expansion: train a short-slot compiler, expand its register slots,
continue training on longer chains, and test whether executable accuracy
survives the expansion.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"transformers is required: {exc}")

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"peft is required: {exc}")


ROOT = Path("/workspace/experiments/qwen_structural_latent_compiler_expansion")
CHECKPOINT_ROOT = Path("/workspace/large_artifacts/qwen_structural_latent_compiler_expansion/checkpoints")
OP_NAMES = ["ADD", "SUB", "MUL"]


@dataclass
class ProgramSpec:
    length: int
    init_value: int
    ops: List[int]
    args: List[int]
    states: List[int]
    answer: int


@dataclass
class ProgramExample:
    prompt: str
    length: int
    init_value: int
    ops: List[int]
    args: List[int]
    states: List[int]
    answer: int
    answer_pos: int
    register_pos: List[int]
    input_ids: List[int]
    template_mode: str


@dataclass
class ExampleSet:
    examples: List[ProgramExample]
    paired: bool = False
    pair_size: int = 1

    def __len__(self) -> int:
        return len(self.examples)


@dataclass
class StageSpec:
    name: str
    max_steps: int
    train_min_len: int
    train_max_len: int
    steps: int


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.convert_ids_to_tokens(0)


def tokenize_no_special(tokenizer: Any, text: str) -> List[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
    ids = list(ids)
    if not ids:
        raise RuntimeError(f"empty tokenization for {text!r}")
    return ids


def apply_op(x: int, op: int, arg: int, modulus: int) -> int:
    if op == 0:
        return (x + arg) % modulus
    if op == 1:
        return (x - arg) % modulus
    if op == 2:
        return (x * arg) % modulus
    raise ValueError(op)


def parse_int_list(raw: str, name: str) -> List[int]:
    out = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not out:
        raise ValueError(f"{name} must contain at least one integer")
    return out


def parse_stage_specs(args: argparse.Namespace) -> List[StageSpec]:
    max_steps = parse_int_list(args.stage_max_steps, "stage_max_steps")
    stage_steps = parse_int_list(args.stage_steps, "stage_steps")
    min_lens = parse_int_list(args.stage_min_lengths, "stage_min_lengths")
    max_lens = parse_int_list(args.stage_train_max_lengths, "stage_train_max_lengths")
    n = len(max_steps)
    for name, values in [
        ("stage_steps", stage_steps),
        ("stage_min_lengths", min_lens),
        ("stage_train_max_lengths", max_lens),
    ]:
        if len(values) != n:
            raise ValueError(f"{name} has {len(values)} entries but stage_max_steps has {n}")
    stages: List[StageSpec] = []
    for i, (slot_max, min_len, max_len, steps) in enumerate(zip(max_steps, min_lens, max_lens, stage_steps), start=1):
        if max_len > slot_max:
            raise ValueError(f"stage {i} train_max_len={max_len} exceeds compiler max_steps={slot_max}")
        if min_len < 1 or min_len > max_len:
            raise ValueError(f"stage {i} has invalid train length range {min_len}..{max_len}")
        stages.append(StageSpec(name=f"stage{i}_max{slot_max}", max_steps=slot_max, train_min_len=min_len, train_max_len=max_len, steps=steps))
    return stages


class TextProgramGenerator:
    def __init__(self, tokenizer: Any, modulus: int, max_steps: int, seed: int, template_mode: str) -> None:
        self.tokenizer = tokenizer
        self.modulus = int(modulus)
        self.max_steps = int(max_steps)
        self.rng = random.Random(seed)
        self.template_mode = template_mode

    def _choice_for_mode(self, mode: str, standard: Sequence[Any], paraphrase: Sequence[Any]) -> Any:
        if mode == "standard":
            return standard[0]
        if mode == "paraphrase":
            return self.rng.choice(list(paraphrase))
        if mode == "mixed":
            return self.rng.choice(list(standard) + list(paraphrase))
        raise ValueError(mode)

    def make_spec(self, min_len: int, max_len: int) -> ProgramSpec:
        length = self.rng.randint(min_len, max_len)
        init_value = self.rng.randrange(self.modulus)
        x = init_value
        ops: List[int] = []
        args: List[int] = []
        states: List[int] = []
        for _ in range(length):
            op = self.rng.randrange(len(OP_NAMES))
            if op == 2:
                arg = self.rng.randint(2, min(self.modulus - 1, 12))
            else:
                arg = self.rng.randint(1, min(self.modulus - 1, 40))
            ops.append(op)
            args.append(arg)
            x = apply_op(x, op, arg, self.modulus)
            states.append(x)
        return ProgramSpec(length=length, init_value=init_value, ops=ops, args=args, states=states, answer=x)

    def render_spec(self, spec: ProgramSpec, template_mode: str) -> ProgramExample:
        if spec.length > self.max_steps:
            raise ValueError(f"spec length {spec.length} exceeds max_steps {self.max_steps}")
        input_ids: List[int] = []
        parts: List[str] = []

        def add_text(text: str) -> int:
            ids = tokenize_no_special(self.tokenizer, text)
            input_ids.extend(ids)
            parts.append(text)
            return len(input_ids) - 1

        intro = self._choice_for_mode(
            template_mode,
            [f"Compute a hidden value modulo {self.modulus}.\n"],
            [
                f"Work in arithmetic modulo {self.modulus}.\n",
                f"Track x using modulus {self.modulus}.\n",
                f"Every update below is modulo {self.modulus}.\n",
            ],
        )
        init_prefix = self._choice_for_mode(
            template_mode,
            ["Initial x = "],
            ["Start with x equal to ", "Let x be ", "The starting value of x is "],
        )
        init_suffix = self._choice_for_mode(template_mode, [".\n"], [".\n", " before any updates.\n", " at the start.\n"])
        final_line = self._choice_for_mode(
            template_mode,
            ["Return x after the final step.\n"],
            ["Report x after all updates.\n", "Give the final value of x.\n", "What is x after the listed updates?\n"],
        )
        answer_marker = self._choice_for_mode(template_mode, ["Answer:\n"], ["Final answer:\n", "Result:\n", "Value:\n"])

        add_text(intro)
        add_text(init_prefix)
        add_text(str(spec.init_value))
        add_text(init_suffix)
        for op, arg in zip(spec.ops, spec.args):
            if op == 0:
                prefix, op_word, between, number, suffix = self._choice_for_mode(
                    template_mode,
                    [("Step: ", "add", " ", str(arg), ".")],
                    [
                        ("Next, ", "increase", " x by ", str(arg), "."),
                        ("Now ", "add", " ", str(arg), " to x."),
                        ("Use an ", "add", " update of ", str(arg), "."),
                    ],
                )
            elif op == 1:
                prefix, op_word, between, number, suffix = self._choice_for_mode(
                    template_mode,
                    [("Step: ", "subtract", " ", str(arg), ".")],
                    [
                        ("Next, ", "decrease", " x by ", str(arg), "."),
                        ("Now ", "subtract", " ", str(arg), " from x."),
                        ("Use a ", "subtract", " update of ", str(arg), "."),
                    ],
                )
            else:
                prefix, op_word, between, number, suffix = self._choice_for_mode(
                    template_mode,
                    [("Step: ", "multiply", " by ", str(arg), ".")],
                    [
                        ("Next, ", "multiply", " x by ", str(arg), "."),
                        ("Now ", "scale", " x by ", str(arg), "."),
                        ("Use a ", "multiply", " update of ", str(arg), "."),
                    ],
                )
            add_text(prefix)
            add_text(op_word)
            add_text(between)
            add_text(number)
            add_text(suffix + "\n")
        add_text(final_line)
        answer_pos = add_text(answer_marker)

        register_pos: List[int] = []
        add_text("\nLatent executable program registers:\n")
        add_text("INIT ")
        register_pos.append(add_text("<LC_INIT>"))
        add_text("\n")
        for step in range(self.max_steps):
            add_text(f"{step:02d} OP ")
            register_pos.append(add_text(f"<LC_OP_{step:02d}>"))
            add_text(" ARG ")
            register_pos.append(add_text(f"<LC_ARG_{step:02d}>"))
            add_text("\n")

        return ProgramExample(
            prompt="".join(parts).rstrip("\n"),
            length=spec.length,
            init_value=spec.init_value,
            ops=list(spec.ops),
            args=list(spec.args),
            states=list(spec.states),
            answer=spec.answer,
            answer_pos=answer_pos,
            register_pos=register_pos,
            input_ids=input_ids,
            template_mode=template_mode,
        )

    def make(self, min_len: int, max_len: int) -> ProgramExample:
        return self.render_spec(self.make_spec(min_len, max_len), self.template_mode)

    def dataset(self, n: int, min_len: int, max_len: int) -> ExampleSet:
        return ExampleSet([self.make(min_len, max_len) for _ in range(n)])

    def paired_dataset(self, n_pairs: int, min_len: int, max_len: int, template_modes: Sequence[str]) -> ExampleSet:
        examples: List[ProgramExample] = []
        for _ in range(n_pairs):
            spec = self.make_spec(min_len, max_len)
            for mode in template_modes:
                examples.append(self.render_spec(spec, mode))
        return ExampleSet(examples, paired=True, pair_size=len(template_modes))


def collate_examples(
    examples: Sequence[ProgramExample],
    pad_id: int,
    max_steps: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    seq_len = min(max(len(ex.input_ids) for ex in examples), max_length)
    ids = torch.full((len(examples), seq_len), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(examples), seq_len), dtype=torch.long, device=device)
    reg_count = 1 + 2 * max_steps
    lengths: List[int] = []
    init_values: List[int] = []
    answers: List[int] = []
    answer_pos: List[int] = []
    register_pos: List[List[int]] = []
    ops: List[List[int]] = []
    args: List[List[int]] = []
    states: List[List[int]] = []
    for row, ex in enumerate(examples):
        if len(ex.register_pos) != reg_count:
            raise RuntimeError(f"example has {len(ex.register_pos)} registers but expected {reg_count}")
        required = [ex.answer_pos] + list(ex.register_pos)
        if max(required) >= seq_len:
            raise RuntimeError(
                f"max_length={max_length} truncated a required register/answer position; need {max(required) + 1} tokens"
            )
        cur = ex.input_ids[:seq_len]
        ids[row, : len(cur)] = torch.tensor(cur, dtype=torch.long, device=device)
        mask[row, : len(cur)] = 1
        lengths.append(ex.length)
        init_values.append(ex.init_value)
        answers.append(ex.answer)
        answer_pos.append(ex.answer_pos)
        register_pos.append(ex.register_pos)
        ops.append(ex.ops + [-100] * (max_steps - len(ex.ops)))
        args.append(ex.args + [-100] * (max_steps - len(ex.args)))
        states.append(ex.states + [-100] * (max_steps - len(ex.states)))
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
        "init_value": torch.tensor(init_values, dtype=torch.long, device=device),
        "ops": torch.tensor(ops, dtype=torch.long, device=device),
        "args": torch.tensor(args, dtype=torch.long, device=device),
        "states": torch.tensor(states, dtype=torch.long, device=device),
        "answer": torch.tensor(answers, dtype=torch.long, device=device),
        "answer_pos": torch.tensor(answer_pos, dtype=torch.long, device=device),
        "register_pos": torch.tensor(register_pos, dtype=torch.long, device=device),
    }


def sample_batch(
    dataset: ExampleSet,
    batch_size: int,
    pad_id: int,
    max_steps: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    idxs = torch.randint(0, len(dataset), (batch_size,)).tolist()
    return collate_examples([dataset.examples[i] for i in idxs], pad_id, max_steps, max_length, device)


def dtype_from_string(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(name)


class StructuralLatentCompiler(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        modulus: int,
        max_steps: int,
        width: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.modulus = int(modulus)
        self.max_steps = int(max_steps)
        self.width = int(width)
        self.layers = int(layers)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.register_count = 1 + 2 * self.max_steps
        self.input = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width))
        self.slot_pos = nn.Parameter(torch.randn(self.register_count, width) * 0.02)
        if layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        else:
            self.encoder = nn.Identity()
        self.init_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, modulus))
        self.op_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, len(OP_NAMES)))
        self.arg_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, modulus))

    def config_dict(self) -> Dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "modulus": self.modulus,
            "max_steps": self.max_steps,
            "width": self.width,
            "layers": self.layers,
            "heads": self.heads,
            "dropout": self.dropout,
        }

    def forward(self, register_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # register_h: [batch, register, hidden]
        bsz, regs, _ = register_h.shape
        if regs != self.register_count:
            raise RuntimeError(f"expected {self.register_count} registers, got {regs}")
        x = self.input(register_h.float())
        x = x + self.slot_pos[None, :, :]
        x = self.encoder(x)
        init = x[:, 0]
        step = x[:, 1:].reshape(bsz, self.max_steps, 2, x.shape[-1])
        op_h = step[:, :, 0]
        arg_h = step[:, :, 1]
        return self.init_head(init), self.op_head(op_h), self.arg_head(arg_h)


class DirectAnswerHead(nn.Module):
    def __init__(self, hidden_dim: int, modulus: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.SiLU(), nn.Linear(width, modulus))

    def forward(self, answer_h: torch.Tensor) -> torch.Tensor:
        return self.net(answer_h.float())


def expand_compiler(old: StructuralLatentCompiler, new_max_steps: int, noise_scale: float) -> StructuralLatentCompiler:
    if new_max_steps <= old.max_steps:
        return old
    new = StructuralLatentCompiler(
        hidden_dim=old.hidden_dim,
        modulus=old.modulus,
        max_steps=new_max_steps,
        width=old.width,
        layers=old.layers,
        heads=old.heads,
        dropout=old.dropout,
    ).to(next(old.parameters()).device)
    old_sd = old.state_dict()
    new_sd = new.state_dict()
    with torch.no_grad():
        for key, value in old_sd.items():
            if key == "slot_pos":
                continue
            if key in new_sd and new_sd[key].shape == value.shape:
                new_sd[key].copy_(value)
        new.slot_pos[0].copy_(old.slot_pos[0])
        for step in range(new_max_steps):
            new_op_idx = 1 + 2 * step
            new_arg_idx = 2 + 2 * step
            src_step = min(step, old.max_steps - 1)
            old_op_idx = 1 + 2 * src_step
            old_arg_idx = 2 + 2 * src_step
            new.slot_pos[new_op_idx].copy_(old.slot_pos[old_op_idx])
            new.slot_pos[new_arg_idx].copy_(old.slot_pos[old_arg_idx])
            if step >= old.max_steps and noise_scale > 0:
                new.slot_pos[new_op_idx].add_(torch.randn_like(new.slot_pos[new_op_idx]) * noise_scale)
                new.slot_pos[new_arg_idx].add_(torch.randn_like(new.slot_pos[new_arg_idx]) * noise_scale)
    return new


class TransitionExecutor(nn.Module):
    def __init__(self, modulus: int, device: torch.device) -> None:
        super().__init__()
        table = torch.zeros(len(OP_NAMES), modulus, modulus, modulus, dtype=torch.float32)
        for op in range(len(OP_NAMES)):
            for arg in range(modulus):
                for old in range(modulus):
                    table[op, arg, old, apply_op(old, op, arg, modulus)] = 1.0
        self.register_buffer("table", table.to(device))

    def soft_trajectory(self, init_logits: torch.Tensor, op_logits: torch.Tensor, arg_logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        state = F.softmax(init_logits.float(), dim=-1)
        op_probs = F.softmax(op_logits.float(), dim=-1)
        arg_probs = F.softmax(arg_logits.float(), dim=-1)
        states: List[torch.Tensor] = []
        for step in range(op_logits.shape[1]):
            cand = torch.einsum("bp,oapq->boaq", state, self.table)
            weights = op_probs[:, step, :, None] * arg_probs[:, step, None, :]
            next_state = (cand * weights[:, :, :, None]).sum(dim=(1, 2))
            active = (lengths > step).float().unsqueeze(-1)
            state = active * next_state + (1.0 - active) * state
            states.append(state.clamp_min(1e-9))
        return torch.stack(states, dim=1)

    def soft_forward(self, init_logits: torch.Tensor, op_logits: torch.Tensor, arg_logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        states = self.soft_trajectory(init_logits, op_logits, arg_logits, lengths)
        idx = (lengths.clamp_min(1) - 1).view(-1, 1, 1).expand(-1, 1, states.shape[-1])
        final = states.gather(1, idx).squeeze(1)
        no_step = lengths.eq(0).view(-1, 1)
        if no_step.any():
            final = torch.where(no_step, F.softmax(init_logits.float(), dim=-1).clamp_min(1e-9), final)
        return final.clamp_min(1e-9)


def extract_hidden(outputs: Any) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]
    last = getattr(outputs, "last_hidden_state", None)
    if torch.is_tensor(last):
        return last
    raise RuntimeError("model did not return hidden states")


def forward_hidden(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    return extract_hidden(outputs)


def gather_registers(hidden: torch.Tensor, register_pos: torch.Tensor) -> torch.Tensor:
    bsz, regs = register_pos.shape
    b_idx = torch.arange(bsz, device=hidden.device).view(bsz, 1).expand(bsz, regs)
    return hidden[b_idx, register_pos]


def gather_answer_hidden(hidden: torch.Tensor, answer_pos: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, answer_pos]


def trace_losses(
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    active = batch["ops"].ne(-100)
    init_loss = F.cross_entropy(init_logits, batch["init_value"])
    op_loss = F.cross_entropy(op_logits[active], batch["ops"][active])
    arg_loss = F.cross_entropy(arg_logits[active], batch["args"][active])
    total = args.init_trace_loss_weight * init_loss + args.op_trace_loss_weight * op_loss + args.arg_trace_loss_weight * arg_loss
    return total, {
        "init_loss": float(init_loss.detach().cpu()),
        "op_loss": float(op_loss.detach().cpu()),
        "arg_loss": float(arg_loss.detach().cpu()),
    }


def state_ladder_loss(state_probs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    active = batch["states"].ne(-100)
    loss = F.nll_loss(state_probs[active].log(), batch["states"][active])
    pred = state_probs.argmax(dim=-1)
    acc = pred[active].eq(batch["states"][active]).float().mean()
    return loss, {"state_loss": float(loss.detach().cpu()), "state_train_accuracy": float(acc.detach().cpu())}


@torch.no_grad()
def argmax_execute_with_states(
    init_pred: torch.Tensor,
    op_pred: torch.Tensor,
    arg_pred: torch.Tensor,
    lengths: torch.Tensor,
    modulus: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    outs: List[int] = []
    state_rows: List[List[int]] = []
    max_steps = int(op_pred.shape[1])
    for row in range(init_pred.shape[0]):
        x = int(init_pred[row].item())
        states = [-100] * max_steps
        for step in range(int(lengths[row].item())):
            x = apply_op(x, int(op_pred[row, step].item()), int(arg_pred[row, step].item()), modulus)
            states[step] = x
        outs.append(x)
        state_rows.append(states)
    return torch.tensor(outs, dtype=torch.long, device=init_pred.device), torch.tensor(state_rows, dtype=torch.long, device=init_pred.device)


def loss_for_batch(
    model: nn.Module,
    compiler: StructuralLatentCompiler,
    direct_head: Optional[DirectAnswerHead],
    executor: TransitionExecutor,
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    hidden = forward_hidden(model, batch)
    reg_h = gather_registers(hidden, batch["register_pos"])
    init_logits, op_logits, arg_logits = compiler(reg_h)
    trace_loss, aux = trace_losses(init_logits, op_logits, arg_logits, batch, args)
    state_probs = executor.soft_trajectory(init_logits, op_logits, arg_logits, batch["lengths"])
    final_probs = executor.soft_forward(init_logits, op_logits, arg_logits, batch["lengths"])
    answer_loss = F.nll_loss(final_probs.log(), batch["answer"])
    ladder_loss, ladder_aux = state_ladder_loss(state_probs, batch)
    loss = args.trace_loss_weight * trace_loss + args.executor_loss_weight * answer_loss + args.state_loss_weight * ladder_loss
    aux.update(ladder_aux)
    aux["executor_loss"] = float(answer_loss.detach().cpu())
    if direct_head is not None and args.direct_head_weight > 0:
        direct_logits = direct_head(gather_answer_hidden(hidden, batch["answer_pos"]))
        direct_loss = F.cross_entropy(direct_logits, batch["answer"])
        loss = loss + args.direct_head_weight * direct_loss
        aux["direct_loss"] = float(direct_loss.detach().cpu())
    aux["loss"] = float(loss.detach().cpu())
    return loss, aux


@torch.no_grad()
def evaluate(
    split: str,
    stage_name: str,
    model: nn.Module,
    compiler: StructuralLatentCompiler,
    direct_head: Optional[DirectAnswerHead],
    executor: TransitionExecutor,
    dataset: ExampleSet,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    compiler.eval()
    if direct_head is not None:
        direct_head.eval()
    total = 0
    exec_correct = 0
    exec_mass = 0.0
    direct_correct = 0
    init_correct = 0
    op_correct = 0
    arg_correct = 0
    op_total = 0
    program_exact = 0
    state_correct = 0
    state_total = 0
    state_all_exact = 0
    prefix_correct_total = 0
    prefix_possible_total = 0
    answers_all: List[torch.Tensor] = []
    lengths_all: List[torch.Tensor] = []
    exec_preds_all: List[torch.Tensor] = []
    direct_preds_all: List[torch.Tensor] = []
    init_preds_all: List[torch.Tensor] = []
    op_preds_all: List[torch.Tensor] = []
    arg_preds_all: List[torch.Tensor] = []
    state_preds_all: List[torch.Tensor] = []
    pad_id = int(tokenizer.pad_token_id)
    for start in range(0, len(dataset), args.eval_batch_size):
        chunk = dataset.examples[start : start + args.eval_batch_size]
        batch = collate_examples(chunk, pad_id, compiler.max_steps, args.max_length, device)
        hidden = forward_hidden(model, batch)
        reg_h = gather_registers(hidden, batch["register_pos"])
        init_logits, op_logits, arg_logits = compiler(reg_h)
        final_probs = executor.soft_forward(init_logits, op_logits, arg_logits, batch["lengths"])
        exec_mass += float(final_probs.gather(1, batch["answer"].view(-1, 1)).sum().item())
        init_pred = init_logits.argmax(dim=-1)
        op_pred = op_logits.argmax(dim=-1)
        arg_pred = arg_logits.argmax(dim=-1)
        pred_answer, pred_states = argmax_execute_with_states(init_pred, op_pred, arg_pred, batch["lengths"], args.modulus)
        total += pred_answer.shape[0]
        exec_correct += int(pred_answer.eq(batch["answer"]).sum().item())
        init_ok = init_pred.eq(batch["init_value"])
        init_correct += int(init_ok.sum().item())
        active = batch["ops"].ne(-100)
        op_ok = op_pred.eq(batch["ops"]) | ~active
        arg_ok = arg_pred.eq(batch["args"]) | ~active
        op_correct += int(op_pred[active].eq(batch["ops"][active]).sum().item())
        arg_correct += int(arg_pred[active].eq(batch["args"][active]).sum().item())
        op_total += int(active.sum().item())
        program_exact += int((init_ok & op_ok.all(dim=1) & arg_ok.all(dim=1)).sum().item())
        state_active = batch["states"].ne(-100)
        state_ok = pred_states.eq(batch["states"]) | ~state_active
        state_correct += int(pred_states[state_active].eq(batch["states"][state_active]).sum().item())
        state_total += int(state_active.sum().item())
        state_all_exact += int(state_ok.all(dim=1).sum().item())
        for row in range(pred_states.shape[0]):
            length = int(batch["lengths"][row].item())
            prefix = 0
            for step in range(length):
                if bool(state_ok[row, step].item()):
                    prefix += 1
                else:
                    break
            prefix_correct_total += prefix
            prefix_possible_total += length
        if direct_head is not None:
            direct_logits = direct_head(gather_answer_hidden(hidden, batch["answer_pos"]))
            direct_pred = direct_logits.argmax(dim=-1)
            direct_correct += int(direct_pred.eq(batch["answer"]).sum().item())
            direct_preds_all.append(direct_pred.detach().cpu())
        answers_all.append(batch["answer"].detach().cpu())
        lengths_all.append(batch["lengths"].detach().cpu())
        exec_preds_all.append(pred_answer.detach().cpu())
        init_preds_all.append(init_pred.detach().cpu())
        op_preds_all.append(op_pred.detach().cpu())
        arg_preds_all.append(arg_pred.detach().cpu())
        state_preds_all.append(pred_states.detach().cpu())
    out: Dict[str, Any] = {
        "stage": stage_name,
        "split": split,
        "n": total,
        "max_steps": compiler.max_steps,
        "executor_accuracy": exec_correct / total if total else math.nan,
        "executor_target_mass": exec_mass / total if total else math.nan,
        "direct_accuracy": direct_correct / total if direct_head is not None and total else math.nan,
        "init_accuracy": init_correct / total if total else math.nan,
        "op_accuracy": op_correct / op_total if op_total else math.nan,
        "arg_accuracy": arg_correct / op_total if op_total else math.nan,
        "program_exact": program_exact / total if total else math.nan,
        "state_accuracy": state_correct / state_total if state_total else math.nan,
        "state_all_exact": state_all_exact / total if total else math.nan,
        "state_prefix_fraction": prefix_correct_total / prefix_possible_total if prefix_possible_total else math.nan,
        "state_mean_correct_prefix": prefix_correct_total / total if total else math.nan,
    }
    if dataset.paired and dataset.pair_size == 2 and total >= 2:
        usable = (total // 2) * 2
        pair_count = usable // 2
        answers = torch.cat(answers_all)[:usable].view(pair_count, 2)
        lengths = torch.cat(lengths_all)[:usable].view(pair_count, 2)
        exec_preds = torch.cat(exec_preds_all)[:usable].view(pair_count, 2)
        init_preds = torch.cat(init_preds_all)[:usable].view(pair_count, 2)
        op_preds = torch.cat(op_preds_all)[:usable].view(pair_count, 2, -1)
        arg_preds = torch.cat(arg_preds_all)[:usable].view(pair_count, 2, -1)
        state_preds = torch.cat(state_preds_all)[:usable].view(pair_count, 2, -1)
        program_same: List[bool] = []
        state_same: List[bool] = []
        for pair_idx in range(pair_count):
            length = int(lengths[pair_idx, 0].item())
            same = bool(init_preds[pair_idx, 0].eq(init_preds[pair_idx, 1]).item())
            same = same and bool(op_preds[pair_idx, 0, :length].eq(op_preds[pair_idx, 1, :length]).all().item())
            same = same and bool(arg_preds[pair_idx, 0, :length].eq(arg_preds[pair_idx, 1, :length]).all().item())
            program_same.append(same)
            state_same.append(bool(state_preds[pair_idx, 0, :length].eq(state_preds[pair_idx, 1, :length]).all().item()))
        out["executor_pair_answer_consistency"] = float(exec_preds[:, 0].eq(exec_preds[:, 1]).float().mean().item())
        out["executor_pair_both_correct"] = float(exec_preds.eq(answers).all(dim=1).float().mean().item())
        out["compiler_pair_program_consistency"] = sum(program_same) / max(1, len(program_same))
        out["compiler_pair_state_consistency"] = sum(state_same) / max(1, len(state_same))
        if direct_preds_all:
            direct_preds = torch.cat(direct_preds_all)[:usable].view(pair_count, 2)
            out["direct_pair_answer_consistency"] = float(direct_preds[:, 0].eq(direct_preds[:, 1]).float().mean().item())
            out["direct_pair_both_correct"] = float(direct_preds.eq(answers).all(dim=1).float().mean().item())
    return out


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, nn.Module, str]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"
    dtype = dtype_from_string(args.torch_dtype)
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    common: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": args.device_map if torch.cuda.is_available() else None,
    }
    if quantization_config is not None:
        common["quantization_config"] = quantization_config
    common = {k: v for k, v in common.items() if v is not None}
    print(f"[load] Loading {args.model_id} with AutoModelForCausalLM", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **common)
    model.config.use_cache = False
    if args.use_lora and args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.use_lora:
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        if args.lora_target_modules == "all-linear":
            target_modules: str | List[str] = "all-linear"
        else:
            target_modules = [part.strip() for part in args.lora_target_modules.split(",") if part.strip()]
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        for p in model.parameters():
            p.requires_grad_(False)
    return tokenizer, model, "AutoModelForCausalLM"


def optimizer_for(params: List[nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer:
    if not params:
        raise RuntimeError("no trainable parameters")
    if args.optimizer == "paged_adamw_8bit":
        try:
            import bitsandbytes as bnb  # type: ignore

            return bnb.optim.PagedAdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay)
        except Exception as exc:
            print(f"[optim] PagedAdamW8bit unavailable ({exc}); falling back to AdamW", flush=True)
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_hidden_dim(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for attr in ["hidden_size", "n_embd", "d_model"]:
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    base = getattr(model, "base_model", None)
    config = getattr(base, "config", None)
    for attr in ["hidden_size", "n_embd", "d_model"]:
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise RuntimeError("could not infer hidden size from model config")


def collect_metadata(args: argparse.Namespace, loader: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "transformers_loader": loader,
        "peft_installed": importlib.util.find_spec("peft") is not None,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "model_id": args.model_id,
    }
    if torch.cuda.is_available():
        out["gpu_name"] = torch.cuda.get_device_name(0)
        out["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
    return out


def make_eval_sets(tokenizer: Any, compiler_max_steps: int, args: argparse.Namespace, seed_offset: int) -> Dict[str, ExampleSet]:
    lengths = [x for x in parse_int_list(args.eval_lengths, "eval_lengths") if x <= compiler_max_steps]
    out: Dict[str, ExampleSet] = {}
    for i, length in enumerate(lengths):
        std_gen = TextProgramGenerator(tokenizer, args.modulus, compiler_max_steps, args.seed + seed_offset + 1000 + i, "standard")
        para_gen = TextProgramGenerator(tokenizer, args.modulus, compiler_max_steps, args.seed + seed_offset + 2000 + i, "paraphrase")
        pair_gen = TextProgramGenerator(tokenizer, args.modulus, compiler_max_steps, args.seed + seed_offset + 3000 + i, "mixed")
        out[f"standard_L{length}"] = std_gen.dataset(args.eval_examples, length, length)
        out[f"paraphrase_L{length}"] = para_gen.dataset(args.eval_examples, length, length)
        out[f"paired_L{length}"] = pair_gen.paired_dataset(args.paired_eval_pairs, length, length, ["standard", "paraphrase"])
    return out


def checkpoint_dir(args: argparse.Namespace, run_name: str) -> Path:
    return Path(args.checkpoint_dir) if args.checkpoint_dir else CHECKPOINT_ROOT / run_name


def save_checkpoint(
    run_name: str,
    tag: str,
    stage: StageSpec,
    global_step: int,
    model: nn.Module,
    compiler: StructuralLatentCompiler,
    direct_head: Optional[DirectAnswerHead],
    args: argparse.Namespace,
    metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    ckpt = checkpoint_dir(args, run_name) / tag
    ckpt.mkdir(parents=True, exist_ok=True)
    if args.use_lora and hasattr(model, "save_pretrained"):
        adapter_dir = ckpt / "adapter"
        model.save_pretrained(adapter_dir)
        model_path = str(adapter_dir)
    else:
        marker = ckpt / "frozen_backbone.txt"
        marker.write_text(f"Frozen backbone: {args.model_id}\n")
        model_path = str(marker)
    heads_path = ckpt / "heads.pt"
    torch.save(
        {
            "args": vars(args),
            "stage": stage.__dict__,
            "global_step": int(global_step),
            "compiler_config": compiler.config_dict(),
            "compiler": compiler.state_dict(),
            "direct_head": direct_head.state_dict() if direct_head is not None else None,
            "metrics": metrics or {},
        },
        heads_path,
    )
    record: Dict[str, Any] = {
        "run": run_name,
        "tag": tag,
        "stage": stage.name,
        "global_step": int(global_step),
        "compiler_max_steps": compiler.max_steps,
        "checkpoint_dir": str(ckpt),
        "model_checkpoint": model_path,
        "heads_checkpoint": str(heads_path),
    }
    if metrics:
        for key in ["selection_split", "selection_metric", "selection_value", "executor_accuracy", "program_exact", "state_prefix_fraction"]:
            if key in metrics:
                record[key] = metrics[key]
    return record


def choose_selection(metrics_rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    split = args.selection_split
    metric = args.selection_metric
    candidates = [row for row in metrics_rows if row.get("split") == split]
    if not candidates:
        candidates = list(metrics_rows)
    best: Optional[Dict[str, Any]] = None
    best_value = -float("inf")
    for row in candidates:
        value = row.get(metric)
        if isinstance(value, (float, int)) and value == value and float(value) > best_value:
            best = row
            best_value = float(value)
    if best is None:
        return {"selection_split": split, "selection_metric": metric, "selection_value": math.nan}
    out = dict(best)
    out["selection_split"] = best.get("split", split)
    out["selection_metric"] = metric
    out["selection_value"] = best_value
    return out


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model, loader = load_model_and_tokenizer(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.load_in_4bit and args.device_map in {"", "none", "None"}:
        model.to(device)
    hidden_dim = read_hidden_dim(model)
    compiler: Optional[StructuralLatentCompiler] = None
    direct_head: Optional[DirectAnswerHead] = DirectAnswerHead(hidden_dim, args.modulus, args.head_width).to(device) if args.direct_head_weight > 0 else None
    executor = TransitionExecutor(args.modulus, device)
    stages = parse_stage_specs(args)
    train_rows: List[Dict[str, Any]] = []
    metrics_rows: List[Dict[str, Any]] = []
    checkpoint_records: List[Dict[str, Any]] = []
    global_step = 0
    pad_id = int(tokenizer.pad_token_id)
    t0 = time.time()

    for stage_idx, stage in enumerate(stages, start=1):
        if compiler is None:
            compiler = StructuralLatentCompiler(
                hidden_dim=hidden_dim,
                modulus=args.modulus,
                max_steps=stage.max_steps,
                width=args.head_width,
                layers=args.compiler_layers,
                heads=args.compiler_heads,
                dropout=args.compiler_dropout,
            ).to(device)
            expansion_event = "init"
        elif stage.max_steps > compiler.max_steps:
            old_steps = compiler.max_steps
            compiler = expand_compiler(compiler, stage.max_steps, args.expansion_noise)
            expansion_event = f"expand_{old_steps}_to_{stage.max_steps}"
        elif stage.max_steps == compiler.max_steps:
            expansion_event = "same"
        else:
            raise ValueError("stage max_steps cannot shrink")

        train_gen = TextProgramGenerator(tokenizer, args.modulus, compiler.max_steps, args.seed + 17 * stage_idx, args.train_template_mode)
        train_set = train_gen.dataset(args.train_examples, stage.train_min_len, stage.train_max_len)
        params = [p for p in model.parameters() if p.requires_grad] + [p for p in compiler.parameters() if p.requires_grad]
        if direct_head is not None:
            params += [p for p in direct_head.parameters() if p.requires_grad]
        opt = optimizer_for(params, args)
        print(
            f"[stage] {stage.name} event={expansion_event} train_len={stage.train_min_len}..{stage.train_max_len} "
            f"examples={len(train_set)} steps={stage.steps}",
            flush=True,
        )
        opt.zero_grad(set_to_none=True)
        for local_step in range(1, stage.steps + 1):
            model.train()
            compiler.train()
            if direct_head is not None:
                direct_head.train()
            batch = sample_batch(train_set, args.batch_size, pad_id, compiler.max_steps, args.max_length, device)
            loss, aux = loss_for_batch(model, compiler, direct_head, executor, batch, args)
            (loss / args.grad_accum).backward()
            if local_step % args.grad_accum == 0 or local_step == stage.steps:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
            global_step += 1
            if local_step == 1 or local_step % args.log_interval == 0 or local_step == stage.steps:
                row: Dict[str, Any] = {
                    "run": args.run_name,
                    "stage": stage.name,
                    "stage_idx": stage_idx,
                    "local_step": local_step,
                    "global_step": global_step,
                    "compiler_max_steps": compiler.max_steps,
                    "elapsed_sec": round(time.time() - t0, 3),
                    "expansion_event": expansion_event,
                }
                row.update(aux)
                train_rows.append(row)
                write_csv(run_dir / "train_log.csv", train_rows)
                print(
                    f"[train] {stage.name} {local_step}/{stage.steps} g={global_step} "
                    f"loss={row.get('loss', math.nan):.4f} exec={row.get('executor_loss', math.nan):.4f} "
                    f"state_acc={row.get('state_train_accuracy', math.nan):.3f}",
                    flush=True,
                )
        eval_sets = make_eval_sets(tokenizer, compiler.max_steps, args, 10000 * stage_idx)
        stage_metric_rows: List[Dict[str, Any]] = []
        for split, dataset in eval_sets.items():
            metrics = evaluate(split, stage.name, model, compiler, direct_head, executor, dataset, tokenizer, args, device)
            metrics["run"] = args.run_name
            metrics["global_step"] = global_step
            metrics["stage_idx"] = stage_idx
            metrics_rows.append(metrics)
            stage_metric_rows.append(metrics)
            print(
                f"[eval] {stage.name} {split}: executor={metrics['executor_accuracy']:.3f} "
                f"program={metrics['program_exact']:.3f} prefix={metrics['state_prefix_fraction']:.3f}",
                flush=True,
            )
        write_csv(run_dir / "metrics.csv", metrics_rows)
        selection = choose_selection(stage_metric_rows, args)
        if args.save_checkpoints:
            record = save_checkpoint(args.run_name, f"{stage.name}_step{global_step:05d}", stage, global_step, model, compiler, direct_head, args, selection)
            checkpoint_records.append(record)
            write_csv(run_dir / "checkpoint_manifest.csv", checkpoint_records)
            write_csv(ROOT / "checkpoint_manifest.csv", checkpoint_records)
        status = {
            "run": args.run_name,
            "last_stage": stage.name,
            "global_step": global_step,
            "compiler_max_steps": compiler.max_steps,
            "latest_selection": selection,
        }
        (run_dir / "latest_status.json").write_text(json.dumps(status, indent=2))

    results = {
        "run": args.run_name,
        "args": vars(args),
        "metadata": collect_metadata(args, loader),
        "stages": [stage.__dict__ for stage in stages],
        "train_log": train_rows,
        "metrics": metrics_rows,
        "checkpoints": checkpoint_records,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_csv(run_dir / "train_log.csv", train_rows)
    write_csv(run_dir / "metrics.csv", metrics_rows)
    if checkpoint_records:
        write_csv(run_dir / "checkpoint_manifest.csv", checkpoint_records)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--model_id", default="Qwen/Qwen3-4B")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--modulus", type=int, default=97)
    p.add_argument("--stage_max_steps", default="8,16,24")
    p.add_argument("--stage_min_lengths", default="1,8,16")
    p.add_argument("--stage_train_max_lengths", default="8,16,24")
    p.add_argument("--stage_steps", default="80,80,80")
    p.add_argument("--train_examples", type=int, default=128)
    p.add_argument("--train_template_mode", choices=["standard", "paraphrase", "mixed"], default="mixed")
    p.add_argument("--eval_lengths", default="8,16,24")
    p.add_argument("--eval_examples", type=int, default=48)
    p.add_argument("--paired_eval_pairs", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--head_width", type=int, default=256)
    p.add_argument("--compiler_layers", type=int, default=2)
    p.add_argument("--compiler_heads", type=int, default=4)
    p.add_argument("--compiler_dropout", type=float, default=0.05)
    p.add_argument("--expansion_noise", type=float, default=0.005)
    p.add_argument("--trace_loss_weight", type=float, default=1.0)
    p.add_argument("--executor_loss_weight", type=float, default=1.0)
    p.add_argument("--state_loss_weight", type=float, default=0.25)
    p.add_argument("--init_trace_loss_weight", type=float, default=1.0)
    p.add_argument("--op_trace_loss_weight", type=float, default=1.0)
    p.add_argument("--arg_trace_loss_weight", type=float, default=1.0)
    p.add_argument("--direct_head_weight", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adamw", "paged_adamw_8bit"], default="paged_adamw_8bit")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--torch_dtype", default="bf16")
    p.add_argument("--load_in_4bit", type=int, default=1)
    p.add_argument("--device_map", default="auto")
    p.add_argument("--use_lora", type=int, default=1)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="all-linear")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_checkpoints", type=int, default=1)
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--selection_split", default="paired_L24")
    p.add_argument("--selection_metric", default="executor_pair_both_correct")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.load_in_4bit = bool(args.load_in_4bit)
    args.use_lora = bool(args.use_lora)
    args.gradient_checkpointing = bool(args.gradient_checkpointing)
    args.save_checkpoints = bool(args.save_checkpoints)
    results = run_experiment(args)
    print(json.dumps({"run": results["run"], "elapsed_sec": results["elapsed_sec"], "metrics_rows": len(results["metrics"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
