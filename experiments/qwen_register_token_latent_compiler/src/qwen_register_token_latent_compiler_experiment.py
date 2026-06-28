#!/usr/bin/env python3
"""Qwen register-token latent compiler experiment.

A prompt is followed by a fixed bank of register markers. The compiler may read
only the hidden states at those register markers, then an invisible executor runs
the predicted program. The experiment tests whether Qwen can expose a stable
program-writing interface at fixed latent positions.
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
except Exception as exc:
    raise SystemExit(f"transformers is required: {exc}")

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception as exc:
    raise SystemExit(f"peft is required: {exc}")


ROOT = Path("experiments/qwen_register_token_latent_compiler")
CHECKPOINT_ROOT = Path("large_artifacts/qwen_register_token_latent_compiler/checkpoints")
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
    reg_init_pos: int
    reg_op_pos: List[int]
    reg_arg_pos: List[int]
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
class CurriculumStage:
    name: str
    min_len: int
    max_len: int
    steps: int


def parse_curriculum_stages(spec: str, fallback_min: int, fallback_max: int, fallback_steps: int) -> List[CurriculumStage]:
    if not spec.strip():
        return [CurriculumStage("train", fallback_min, fallback_max, fallback_steps)]
    stages: List[CurriculumStage] = []
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 4:
            raise ValueError(f"curriculum stage must be name:min_len:max_len:steps, got {part!r}")
        name, min_len, max_len, steps = fields
        stages.append(CurriculumStage(name=name, min_len=int(min_len), max_len=int(max_len), steps=int(steps)))
    if not stages:
        raise ValueError("curriculum_stages produced no stages")
    return stages


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


class TextProgramGenerator:
    def __init__(
        self,
        tokenizer: Any,
        modulus: int,
        max_steps: int,
        seed: int,
        template_mode: str,
        register_style: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.modulus = int(modulus)
        self.max_steps = int(max_steps)
        self.rng = random.Random(seed)
        self.template_mode = template_mode
        self.register_style = register_style

    def _choice_for_mode(self, mode: str, standard: Sequence[Any], paraphrase: Sequence[Any]) -> Any:
        if mode == "standard":
            return standard[0]
        if mode == "paraphrase":
            return self.rng.choice(list(paraphrase))
        if mode == "mixed":
            return self.rng.choice(list(standard) + list(paraphrase))
        raise ValueError(mode)

    def _choice(self, standard: Sequence[Any], paraphrase: Sequence[Any]) -> Any:
        return self._choice_for_mode(self.template_mode, standard, paraphrase)

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
        inline_reg_init_pos = -1
        inline_reg_op_pos: List[int] = []
        inline_reg_arg_pos: List[int] = []
        if self.register_style == "inline":
            add_text("Initial register: ")
            inline_reg_init_pos = add_text("<REG_INIT>")
            add_text("\n")
        for step_index, (op, arg) in enumerate(zip(spec.ops, spec.args)):
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
            if self.register_style == "inline":
                add_text(f"Step {step_index:02d} registers: ")
                inline_reg_op_pos.append(add_text(f"<REG_OP_{step_index:02d}>"))
                add_text(" ")
                inline_reg_arg_pos.append(add_text(f"<REG_ARG_{step_index:02d}>"))
                add_text("\n")
        add_text(final_line)
        answer_pos = add_text(answer_marker)

        reg_op_pos: List[int] = []
        reg_arg_pos: List[int] = []
        if self.register_style == "bare":
            add_text("\nRegister bank:\n")
            reg_init_pos = add_text("<REG_INIT>")
            add_text("\n")
            for step in range(self.max_steps):
                reg_op_pos.append(add_text(f"<REG_OP_{step:02d}>"))
                add_text(" ")
                reg_arg_pos.append(add_text(f"<REG_ARG_{step:02d}>"))
                add_text("\n")
        elif self.register_style == "named":
            add_text("\nFixed register bank for the program above.\n")
            add_text("Initial value register: ")
            reg_init_pos = add_text("<REG_INIT>")
            add_text("\n")
            for step in range(self.max_steps):
                add_text(f"Step {step:02d} operation register: ")
                reg_op_pos.append(add_text(f"<REG_OP_{step:02d}>"))
                add_text("  argument register: ")
                reg_arg_pos.append(add_text(f"<REG_ARG_{step:02d}>"))
                add_text("\n")
        elif self.register_style == "typed":
            add_text("\nThe following fixed markers are latent program slots.\n")
            add_text("Slot INIT stores the starting numeric value: ")
            reg_init_pos = add_text("<REG_INIT>")
            add_text("\n")
            for step in range(self.max_steps):
                add_text(f"Slot OP {step:02d} stores add/subtract/multiply: ")
                reg_op_pos.append(add_text(f"<REG_OP_{step:02d}>"))
                add_text(f" Slot ARG {step:02d} stores the numeric update argument: ")
                reg_arg_pos.append(add_text(f"<REG_ARG_{step:02d}>"))
                add_text("\n")
        elif self.register_style == "inline":
            reg_init_pos = inline_reg_init_pos
            reg_op_pos = list(inline_reg_op_pos)
            reg_arg_pos = list(inline_reg_arg_pos)
            if reg_init_pos < 0:
                raise RuntimeError("inline init register was not created")
            for step in range(spec.length, self.max_steps):
                add_text(f"Unused step {step:02d} registers: ")
                reg_op_pos.append(add_text(f"<REG_OP_{step:02d}>"))
                add_text(" ")
                reg_arg_pos.append(add_text(f"<REG_ARG_{step:02d}>"))
                add_text("\n")
        else:
            raise ValueError(f"unknown register_style={self.register_style!r}")
        prompt = "".join(parts).rstrip("\n")
        return ProgramExample(
            prompt=prompt,
            length=spec.length,
            init_value=spec.init_value,
            ops=list(spec.ops),
            args=list(spec.args),
            states=list(spec.states),
            answer=spec.answer,
            answer_pos=answer_pos,
            reg_init_pos=reg_init_pos,
            reg_op_pos=reg_op_pos,
            reg_arg_pos=reg_arg_pos,
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


def dtype_from_string(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(name)


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
    lengths: List[int] = []
    init_values: List[int] = []
    answers: List[int] = []
    answer_pos: List[int] = []
    reg_pos: List[List[int]] = []
    ops: List[List[int]] = []
    args: List[List[int]] = []
    states: List[List[int]] = []
    for row, ex in enumerate(examples):
        required = [ex.answer_pos, ex.reg_init_pos, *ex.reg_op_pos, *ex.reg_arg_pos]
        if max(required) >= seq_len:
            raise RuntimeError(
                f"max_length={max_length} truncated a required answer/register position; "
                f"needed {max(required) + 1} tokens"
            )
        cur = ex.input_ids[:seq_len]
        ids[row, : len(cur)] = torch.tensor(cur, dtype=torch.long, device=device)
        mask[row, : len(cur)] = 1
        lengths.append(ex.length)
        init_values.append(ex.init_value)
        answers.append(ex.answer)
        answer_pos.append(ex.answer_pos)
        reg_pos.append([ex.reg_init_pos] + [p for pair in zip(ex.reg_op_pos, ex.reg_arg_pos) for p in pair])
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
        "register_pos": torch.tensor(reg_pos, dtype=torch.long, device=device),
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


def sample_paired_batch(
    dataset: ExampleSet,
    batch_size: int,
    pair_size: int,
    pad_id: int,
    max_steps: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if pair_size < 2:
        raise ValueError("pair_size must be at least 2")
    if batch_size % pair_size != 0:
        raise ValueError("batch size must be divisible by pair size")
    n_pairs = len(dataset) // pair_size
    pair_idxs = torch.randint(0, n_pairs, (batch_size // pair_size,)).tolist()
    examples: List[ProgramExample] = []
    for pair_idx in pair_idxs:
        start = pair_idx * pair_size
        examples.extend(dataset.examples[start : start + pair_size])
    return collate_examples(examples, pad_id, max_steps, max_length, device)


class DirectAnswerHead(nn.Module):
    def __init__(self, hidden_dim: int, modulus: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.SiLU(), nn.Linear(width, modulus))

    def forward(self, answer_h: torch.Tensor) -> torch.Tensor:
        return self.net(answer_h.float())


class RegisterProgramCompiler(nn.Module):
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
        self.max_steps = int(max_steps)
        self.modulus = int(modulus)
        self.register_count = 1 + 2 * self.max_steps
        self.input = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width))
        self.pos = nn.Parameter(torch.randn(self.register_count, width) * 0.02)
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

    def forward(self, register_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input(register_h.float()) + self.pos[None, :, :]
        x = self.encoder(x)
        init = x[:, 0]
        step = x[:, 1:].view(x.shape[0], self.max_steps, 2, x.shape[-1])
        op_h = step[:, :, 0]
        arg_h = step[:, :, 1]
        return self.init_head(init), self.op_head(op_h), self.arg_head(arg_h)


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
            init_state = F.softmax(init_logits.float(), dim=-1).clamp_min(1e-9)
            final = torch.where(no_step, init_state, final)
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
    b_idx = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    return hidden[b_idx, register_pos]


def trace_losses(
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    init_weight: float,
    op_weight: float,
    arg_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    active = batch["ops"].ne(-100)
    init_loss = F.cross_entropy(init_logits, batch["init_value"])
    op_loss = F.cross_entropy(op_logits[active], batch["ops"][active])
    arg_loss = F.cross_entropy(arg_logits[active], batch["args"][active])
    return init_weight * init_loss + op_weight * op_loss + arg_weight * arg_loss, {
        "init_loss": float(init_loss.detach().cpu()),
        "op_loss": float(op_loss.detach().cpu()),
        "arg_loss": float(arg_loss.detach().cpu()),
    }


def state_ladder_loss(state_probs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    active = batch["states"].ne(-100)
    if not active.any():
        zero = state_probs.sum() * 0.0
        return zero, {"state_loss": 0.0, "state_train_accuracy": 0.0}
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
    return (
        torch.tensor(outs, dtype=torch.long, device=init_pred.device),
        torch.tensor(state_rows, dtype=torch.long, device=init_pred.device),
    )


@torch.no_grad()
def evaluate(
    variant: str,
    model: nn.Module,
    direct: Optional[DirectAnswerHead],
    compiler: Optional[RegisterProgramCompiler],
    executor: TransitionExecutor,
    dataset: ExampleSet,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    if direct is not None:
        direct.eval()
    if compiler is not None:
        compiler.eval()
    total = 0
    direct_correct = 0
    direct_mass = 0.0
    exec_correct = 0
    exec_mass = 0.0
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
    answer_values: List[torch.Tensor] = []
    length_values: List[torch.Tensor] = []
    direct_pred_values: List[torch.Tensor] = []
    exec_pred_values: List[torch.Tensor] = []
    init_pred_values: List[torch.Tensor] = []
    op_pred_values: List[torch.Tensor] = []
    arg_pred_values: List[torch.Tensor] = []
    state_pred_values: List[torch.Tensor] = []

    for start in range(0, len(dataset), args.eval_batch_size):
        chunk = dataset.examples[start : start + args.eval_batch_size]
        batch = collate_examples(chunk, int(tokenizer.pad_token_id), args.max_steps, args.max_length, device)
        hidden = forward_hidden(model, batch)
        rows = torch.arange(hidden.shape[0], device=device)
        total += hidden.shape[0]
        answer_values.append(batch["answer"].detach().cpu())
        length_values.append(batch["lengths"].detach().cpu())
        if direct is not None:
            logits = direct(hidden[rows, batch["answer_pos"]])
            probs = F.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            direct_pred_values.append(pred.detach().cpu())
            direct_correct += int(pred.eq(batch["answer"]).sum().item())
            direct_mass += float(probs.gather(1, batch["answer"].view(-1, 1)).sum().item())
        if compiler is not None:
            reg_h = gather_registers(hidden, batch["register_pos"])
            init_logits, op_logits, arg_logits = compiler(reg_h)
            probs = executor.soft_forward(init_logits, op_logits, arg_logits, batch["lengths"])
            exec_mass += float(probs.gather(1, batch["answer"].view(-1, 1)).sum().item())
            init_pred = init_logits.argmax(dim=-1)
            op_pred = op_logits.argmax(dim=-1)
            arg_pred = arg_logits.argmax(dim=-1)
            pred_answer, pred_states = argmax_execute_with_states(init_pred, op_pred, arg_pred, batch["lengths"], args.modulus)
            exec_pred_values.append(pred_answer.detach().cpu())
            init_pred_values.append(init_pred.detach().cpu())
            op_pred_values.append(op_pred.detach().cpu())
            arg_pred_values.append(arg_pred.detach().cpu())
            state_pred_values.append(pred_states.detach().cpu())
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
            for row_idx in range(hidden.shape[0]):
                length = int(batch["lengths"][row_idx].item())
                prefix = 0
                for step in range(length):
                    if bool(state_ok[row_idx, step].item()):
                        prefix += 1
                    else:
                        break
                prefix_correct_total += prefix
                prefix_possible_total += length

    metrics: Dict[str, float] = {
        "variant": variant,  # type: ignore[dict-item]
        "n": float(total),
        "direct_accuracy": direct_correct / total if direct is not None else math.nan,
        "direct_target_mass": direct_mass / total if direct is not None else math.nan,
        "executor_accuracy": exec_correct / total if compiler is not None else math.nan,
        "executor_target_mass": exec_mass / total if compiler is not None else math.nan,
        "init_accuracy": init_correct / total if compiler is not None else math.nan,
        "op_accuracy": op_correct / op_total if compiler is not None and op_total else math.nan,
        "arg_accuracy": arg_correct / op_total if compiler is not None and op_total else math.nan,
        "program_exact": program_exact / total if compiler is not None else math.nan,
        "state_accuracy": state_correct / state_total if compiler is not None and state_total else math.nan,
        "state_all_exact": state_all_exact / total if compiler is not None else math.nan,
        "state_prefix_fraction": prefix_correct_total / prefix_possible_total if compiler is not None and prefix_possible_total else math.nan,
    }
    if dataset.paired and dataset.pair_size == 2 and total >= 2:
        usable = (total // 2) * 2
        pair_count = usable // 2
        answers = torch.cat(answer_values)[:usable].view(pair_count, 2)
        lengths = torch.cat(length_values)[:usable].view(pair_count, 2)
        metrics["pair_true_answer_consistency"] = float(answers[:, 0].eq(answers[:, 1]).float().mean().item())
        if direct_pred_values:
            direct_preds = torch.cat(direct_pred_values)[:usable].view(pair_count, 2)
            metrics["direct_pair_answer_consistency"] = float(direct_preds[:, 0].eq(direct_preds[:, 1]).float().mean().item())
            metrics["direct_pair_both_correct"] = float(direct_preds.eq(answers).all(dim=1).float().mean().item())
        if exec_pred_values:
            exec_preds = torch.cat(exec_pred_values)[:usable].view(pair_count, 2)
            init_preds = torch.cat(init_pred_values)[:usable].view(pair_count, 2)
            op_preds = torch.cat(op_pred_values)[:usable].view(pair_count, 2, -1)
            arg_preds = torch.cat(arg_pred_values)[:usable].view(pair_count, 2, -1)
            state_preds = torch.cat(state_pred_values)[:usable].view(pair_count, 2, -1)
            program_same: List[bool] = []
            state_same: List[bool] = []
            for pair_idx in range(pair_count):
                length = int(lengths[pair_idx, 0].item())
                same = bool(init_preds[pair_idx, 0].eq(init_preds[pair_idx, 1]).item())
                same = same and bool(op_preds[pair_idx, 0, :length].eq(op_preds[pair_idx, 1, :length]).all().item())
                same = same and bool(arg_preds[pair_idx, 0, :length].eq(arg_preds[pair_idx, 1, :length]).all().item())
                program_same.append(same)
                state_same.append(bool(state_preds[pair_idx, 0, :length].eq(state_preds[pair_idx, 1, :length]).all().item()))
            metrics["executor_pair_answer_consistency"] = float(exec_preds[:, 0].eq(exec_preds[:, 1]).float().mean().item())
            metrics["executor_pair_both_correct"] = float(exec_preds.eq(answers).all(dim=1).float().mean().item())
            metrics["compiler_pair_program_consistency"] = sum(program_same) / len(program_same)
            metrics["compiler_pair_state_consistency"] = sum(state_same) / len(state_same)
    return metrics


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
        target_modules: str | List[str]
        if args.lora_target_modules == "all-linear":
            target_modules = "all-linear"
        else:
            target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
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


def train_variant(
    variant: str,
    model: nn.Module,
    hidden_dim: int,
    train_sets: List[Tuple[CurriculumStage, ExampleSet]],
    answer_set: ExampleSet,
    eval_sets: Dict[str, ExampleSet],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    train_seed = args.train_seed if args.train_seed >= 0 else args.seed + sum(ord(ch) for ch in variant)
    torch.manual_seed(train_seed)
    direct: Optional[DirectAnswerHead] = None
    compiler: Optional[RegisterProgramCompiler] = None
    modules: List[nn.Module] = [model]
    if variant == "direct":
        direct = DirectAnswerHead(hidden_dim, args.modulus, args.head_width).to(device)
        modules.append(direct)
    else:
        compiler = RegisterProgramCompiler(
            hidden_dim,
            args.modulus,
            args.max_steps,
            args.register_width,
            args.register_layers,
            args.register_heads,
            args.register_dropout,
        ).to(device)
        modules.append(compiler)
    executor = TransitionExecutor(args.modulus, device)
    model_trainable = any(p.requires_grad for p in model.parameters())
    trainable = [p for module in modules for p in module.parameters() if p.requires_grad]
    opt = optimizer_for(trainable, args)
    log_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    pad_id = int(tokenizer.pad_token_id)

    def run_stage(
        stage: str,
        dataset: ExampleSet,
        steps: int,
        trace_active: bool,
        executor_active: bool,
        state_active: bool,
        lr_scale: float,
        offset: int,
    ) -> int:
        for group in opt.param_groups:
            group["lr"] = args.lr * lr_scale
        for local_step in range(1, steps + 1):
            global_step = offset + local_step
            if model_trainable:
                model.train()
            else:
                model.eval()
            if direct is not None:
                direct.train()
            if compiler is not None:
                compiler.train()
            if args.paired_train and args.paired_batches:
                batch = sample_paired_batch(dataset, args.train_batch_size, args.pair_size, pad_id, args.max_steps, args.max_length, device)
            else:
                batch = sample_batch(dataset, args.train_batch_size, pad_id, args.max_steps, args.max_length, device)
            if model_trainable:
                hidden = forward_hidden(model, batch)
            else:
                with torch.no_grad():
                    hidden = forward_hidden(model, batch)
            rows = torch.arange(hidden.shape[0], device=device)
            loss = torch.tensor(0.0, device=device)
            aux: Dict[str, float] = {}
            if direct is not None:
                logits = direct(hidden[rows, batch["answer_pos"]])
                direct_loss = F.cross_entropy(logits, batch["answer"])
                loss = loss + direct_loss
                aux["direct_loss"] = float(direct_loss.detach().cpu())
            if compiler is not None:
                reg_h = gather_registers(hidden, batch["register_pos"])
                init_logits, op_logits, arg_logits = compiler(reg_h)
                state_probs: Optional[torch.Tensor] = None
                if executor_active or state_active:
                    state_probs = executor.soft_trajectory(init_logits, op_logits, arg_logits, batch["lengths"])
                    gather_idx = (batch["lengths"].clamp_min(1) - 1).view(-1, 1, 1).expand(-1, 1, state_probs.shape[-1])
                    answer_probs = state_probs.gather(1, gather_idx).squeeze(1).clamp_min(1e-9)
                    if executor_active:
                        exec_loss = F.nll_loss(answer_probs.log(), batch["answer"])
                        loss = loss + args.executor_loss_weight * exec_loss
                        aux["executor_loss"] = float(exec_loss.detach().cpu())
                if state_active and state_probs is not None:
                    ladder_loss, ladder_aux = state_ladder_loss(state_probs, batch)
                    loss = loss + args.state_loss_weight * ladder_loss
                    aux.update(ladder_aux)
                if trace_active:
                    tr_loss, tr_aux = trace_losses(
                        init_logits,
                        op_logits,
                        arg_logits,
                        batch,
                        args.init_trace_loss_weight,
                        args.op_trace_loss_weight,
                        args.arg_trace_loss_weight,
                    )
                    loss = loss + args.trace_loss_weight * tr_loss
                    aux.update(tr_aux)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()

            if local_step == 1 or local_step == steps or global_step % args.eval_every == 0 or local_step % args.stage_eval_every == 0:
                row: Dict[str, Any] = {
                    "variant": variant,
                    "stage": stage,
                    "local_step": local_step,
                    "step": global_step,
                    "loss": float(loss.detach().cpu()),
                    "trace_loss_active": trace_active,
                    "executor_loss_active": executor_active,
                    "state_loss_active": state_active,
                    "lr_scale": lr_scale,
                    **aux,
                }
                for split, eval_set in eval_sets.items():
                    metrics = evaluate(variant, model, direct, compiler, executor, eval_set, tokenizer, args, device)
                    for key, val in metrics.items():
                        if key not in {"variant", "n"}:
                            row[f"{split}_{key}"] = val
                log_rows.append(row)
                msg = f"[{variant}] {stage} step {global_step} loss={row['loss']:.4f}"
                for split in eval_sets:
                    key = f"{split}_direct_accuracy" if direct is not None else f"{split}_executor_accuracy"
                    if key in row and not math.isnan(float(row[key])):
                        msg += f" {split}={100.0 * float(row[key]):.1f}%"
                print(msg, flush=True)
        return offset + steps

    def run_curriculum(label: str, trace_active: bool, executor_active: bool, state_active: bool, offset: int) -> int:
        for stage_spec, stage_set in train_sets:
            offset = run_stage(label + "_" + stage_spec.name, stage_set, stage_spec.steps, trace_active, executor_active, state_active, 1.0, offset)
        return offset

    step = 0
    curriculum_steps = sum(stage.steps for stage, _ in train_sets)
    answer_steps = args.bootstrap_steps + args.answer_steps if args.bootstrap_steps + args.answer_steps > 0 else curriculum_steps
    if variant == "direct":
        step = run_stage("answer_only", answer_set, answer_steps, False, False, False, 1.0, step)
    elif variant == "register_answer_only":
        step = run_stage("answer_only", answer_set, answer_steps, False, True, False, 1.0, step)
    elif variant == "register_trace":
        step = run_curriculum("trace", True, True, False, step)
    elif variant == "register_trace_state":
        step = run_curriculum("trace_state", True, True, True, step)
    elif variant == "register_trace_then_answer":
        step = run_curriculum("trace", True, True, False, step)
        step = run_stage("answer_retention", answer_set, args.answer_steps, False, True, False, args.answer_lr_scale, step)
    elif variant == "register_trace_state_then_answer":
        step = run_curriculum("trace_state", True, True, True, step)
        step = run_stage("answer_retention", answer_set, args.answer_steps, False, True, False, args.answer_lr_scale, step)
    else:
        raise ValueError(variant)

    final_metrics = {
        split: evaluate(variant, model, direct, compiler, executor, eval_set, tokenizer, args, device)
        for split, eval_set in eval_sets.items()
    }
    ckpt_root = checkpoint_dir(args) / variant
    ckpt_root.mkdir(parents=True, exist_ok=True)
    if args.use_lora and hasattr(model, "save_pretrained"):
        adapter_dir = ckpt_root / "adapter"
        model.save_pretrained(adapter_dir)
        model_ckpt = str(adapter_dir)
    else:
        marker = ckpt_root / "frozen_backbone.txt"
        marker.write_text(f"Frozen backbone: {args.model_id}\n")
        model_ckpt = str(marker)
    heads_path = ckpt_root / "heads.pt"
    torch.save(
        {
            "variant": variant,
            "args": vars(args),
            "direct": direct.state_dict() if direct is not None else None,
            "compiler": compiler.state_dict() if compiler is not None else None,
            "hidden_dim": hidden_dim,
            "register_count": 1 + 2 * args.max_steps,
        },
        heads_path,
    )
    return {
        "variant": variant,
        "train_seconds": time.time() - t0,
        "checkpoints": [model_ckpt, str(heads_path)],
        "train_log": log_rows,
        "final_metrics": final_metrics,
    }


def checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir)
    return CHECKPOINT_ROOT / Path(args.output_dir).name


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


def flatten_final_metrics(run_name: str, results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant, result in results.get("variants", {}).items():
        for split, metrics in result.get("final_metrics", {}).items():
            rows.append({"run": run_name, "variant": variant, "split": split, **metrics})
    return rows


def collect_metadata(args: argparse.Namespace, loader: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "transformers_loader": loader,
        "peft_installed": importlib.util.find_spec("peft") is not None,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
    }
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
    return meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Qwen register-token latent compiler experiment")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch_dtype", type=str, default="bf16")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="all-linear")
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--optimizer", type=str, default="paged_adamw_8bit", choices=["adamw", "paged_adamw_8bit"])
    p.add_argument("--modulus", type=int, default=97)
    p.add_argument("--max_steps", type=int, default=24)
    p.add_argument("--train_min_len", type=int, default=1)
    p.add_argument("--train_max_len", type=int, default=12)
    p.add_argument("--curriculum_stages", type=str, default="short:1:4:150,medium:1:8:150,train:1:12:150,long:8:24:150")
    p.add_argument("--answer_train_min_len", type=int, default=1)
    p.add_argument("--answer_train_max_len", type=int, default=24)
    p.add_argument("--eval_lengths", type=str, default="4,8,12,24")
    p.add_argument("--train_template_mode", type=str, default="mixed", choices=["standard", "mixed", "paraphrase"])
    p.add_argument("--paired_train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--paired_batches", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--paired_template_modes", type=str, default="standard,paraphrase")
    p.add_argument("--pair_size", type=int, default=2)
    p.add_argument("--answer_train_template_mode", type=str, default="mixed", choices=["standard", "mixed", "paraphrase"])
    p.add_argument("--eval_template_modes", type=str, default="standard,paraphrase")
    p.add_argument("--paired_eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--paired_eval_template_modes", type=str, default="standard,paraphrase")
    p.add_argument("--train_size", type=int, default=512)
    p.add_argument("--answer_train_size", type=int, default=512)
    p.add_argument("--eval_size", type=int, default=128)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--bootstrap_steps", type=int, default=0)
    p.add_argument("--answer_steps", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=150)
    p.add_argument("--stage_eval_every", type=int, default=150)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--answer_lr_scale", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--head_width", type=int, default=512)
    p.add_argument("--register_width", type=int, default=512)
    p.add_argument("--register_layers", type=int, default=1)
    p.add_argument("--register_heads", type=int, default=4)
    p.add_argument("--register_dropout", type=float, default=0.05)
    p.add_argument("--register_style", type=str, default="bare", choices=["bare", "named", "typed", "inline"])
    p.add_argument("--trace_loss_weight", type=float, default=1.0)
    p.add_argument("--init_trace_loss_weight", type=float, default=4.0)
    p.add_argument("--op_trace_loss_weight", type=float, default=1.0)
    p.add_argument("--arg_trace_loss_weight", type=float, default=4.0)
    p.add_argument("--executor_loss_weight", type=float, default=1.0)
    p.add_argument("--state_loss_weight", type=float, default=0.25)
    p.add_argument("--variants", type=str, default="register_trace")
    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train_seed", type=int, default=-1)
    p.add_argument("--output_dir", type=str, default=str(ROOT / "runs/default"))
    p.add_argument("--checkpoint_dir", type=str, default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    if args.use_lora and len(variants) > 1:
        raise SystemExit("Run one LoRA variant per process so variants start from the same base model.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, loader = load_model_and_tokenizer(args)
    if not args.load_in_4bit:
        model.to(device)
    metadata = collect_metadata(args, loader)

    curriculum_stages = parse_curriculum_stages(args.curriculum_stages, args.train_min_len, args.train_max_len, args.bootstrap_steps)
    paired_template_modes = [x.strip() for x in args.paired_template_modes.split(",") if x.strip()]
    train_sets: List[Tuple[CurriculumStage, ExampleSet]] = []
    for stage_index, stage in enumerate(curriculum_stages):
        gen_train = TextProgramGenerator(
            tokenizer,
            args.modulus,
            args.max_steps,
            args.seed + 31 * stage_index,
            args.train_template_mode,
            args.register_style,
        )
        if args.paired_train:
            if len(paired_template_modes) != args.pair_size:
                raise ValueError("paired_template_modes count must match pair_size")
            stage_set = gen_train.paired_dataset(args.train_size, stage.min_len, stage.max_len, paired_template_modes)
        else:
            stage_set = gen_train.dataset(args.train_size, stage.min_len, stage.max_len)
        train_sets.append((stage, stage_set))
    gen_answer = TextProgramGenerator(
        tokenizer,
        args.modulus,
        args.max_steps,
        args.seed + 177,
        args.answer_train_template_mode,
        args.register_style,
    )
    answer_set = gen_answer.dataset(args.answer_train_size, args.answer_train_min_len, args.answer_train_max_len)
    eval_sets: Dict[str, ExampleSet] = {}
    eval_modes = [x.strip() for x in args.eval_template_modes.split(",") if x.strip()]
    paired_eval_modes = [x.strip() for x in args.paired_eval_template_modes.split(",") if x.strip()]
    eval_lengths = [int(x.strip()) for x in args.eval_lengths.split(",") if x.strip()]
    for mode_index, mode in enumerate(eval_modes):
        for length in eval_lengths:
            gen = TextProgramGenerator(
                tokenizer,
                args.modulus,
                args.max_steps,
                args.seed + 1000 + 97 * mode_index + length,
                mode,
                args.register_style,
            )
            eval_sets[f"{mode}_len{length}"] = gen.dataset(args.eval_size, length, length)
    if args.paired_eval:
        if len(paired_eval_modes) != 2:
            raise ValueError("paired_eval currently expects exactly two template modes")
        for length in eval_lengths:
            gen = TextProgramGenerator(
                tokenizer,
                args.modulus,
                args.max_steps,
                args.seed + 2000 + length,
                "mixed",
                args.register_style,
            )
            eval_sets[f"paired_len{length}"] = gen.paired_dataset(args.eval_size, length, length, paired_eval_modes)

    probe_batch = collate_examples(train_sets[0][1].examples[:1], int(tokenizer.pad_token_id), args.max_steps, args.max_length, device)
    with torch.no_grad():
        hidden_dim = int(forward_hidden(model, probe_batch).shape[-1])
    train_examples = sum(len(stage_set) for _, stage_set in train_sets)
    print(
        f"[setup] hidden_dim={hidden_dim} registers={1 + 2 * args.max_steps} "
        f"train={train_examples} answer={len(answer_set)} eval_splits={len(eval_sets)}",
        flush=True,
    )

    results: Dict[str, Any] = {
        "args": vars(args),
        "metadata": metadata,
        "hidden_dim": hidden_dim,
        "register_count": 1 + 2 * args.max_steps,
        "dataset": {
            "train_size": args.train_size,
            "train_examples": train_examples,
            "curriculum_stages": [stage.__dict__ for stage in curriculum_stages],
            "answer_train_size": args.answer_train_size,
            "eval_size": args.eval_size,
            "eval_lengths": eval_lengths,
            "train_template_mode": args.train_template_mode,
            "paired_train": args.paired_train,
            "paired_template_modes": paired_template_modes,
            "answer_train_template_mode": args.answer_train_template_mode,
            "eval_template_modes": eval_modes,
            "paired_eval": args.paired_eval,
            "paired_eval_template_modes": paired_eval_modes,
            "register_style": args.register_style,
        },
        "variants": {},
    }
    flat_rows: List[Dict[str, Any]] = []
    for variant in variants:
        result = train_variant(variant, model, hidden_dim, train_sets, answer_set, eval_sets, tokenizer, args, device)
        results["variants"][variant] = result
        flat_rows.extend(result["train_log"])
    with (out / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    write_csv(out / "train_log.csv", flat_rows)
    metric_rows = flatten_final_metrics(out.name, results)
    write_csv(out / "metrics.csv", metric_rows)
    print(f"[done] wrote {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
