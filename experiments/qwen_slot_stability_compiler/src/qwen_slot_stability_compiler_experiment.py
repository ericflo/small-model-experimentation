#!/usr/bin/env python3
"""Paired slot-stability Qwen numeric-copy compiler experiment.

The compiler learns token roles and ordered program slots. Numeric and operator
symbols are copied from deterministic token maps. The stability condition trains
on paired paraphrases of the same underlying program and penalizes drift in the
copied slot distributions.
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


OP_NAMES = ["ADD", "SUB", "MUL"]
VERY_NEG = -1e4


@dataclass
class ProgramSpec:
    length: int
    init_value: int
    ops: List[int]
    args: List[int]
    answer: int


@dataclass
class ProgramExample:
    prompt: str
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
    num_values: List[int]
    op_values: List[int]


@dataclass
class ExampleSet:
    examples: List[ProgramExample]
    paired: bool = False
    pair_size: int = 1

    def __len__(self) -> int:
        return len(self.examples)


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.convert_ids_to_tokens(0)


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

    def _choice(self, standard: Sequence[Any], paraphrase: Sequence[Any]) -> Any:
        return self._choice_for_mode(self.template_mode, standard, paraphrase)

    def make_spec(self, min_len: int, max_len: int) -> ProgramSpec:
        length = self.rng.randint(min_len, max_len)
        init_value = self.rng.randrange(self.modulus)
        x = init_value
        ops: List[int] = []
        args: List[int] = []
        for _ in range(length):
            op = self.rng.randrange(3)
            if op == 2:
                arg = self.rng.randint(2, min(self.modulus - 1, 12))
            else:
                arg = self.rng.randint(1, min(self.modulus - 1, 40))
            ops.append(op)
            args.append(arg)
            x = apply_op(x, op, arg, self.modulus)
        return ProgramSpec(length=length, init_value=init_value, ops=ops, args=args, answer=x)

    def render_spec(self, spec: ProgramSpec, template_mode: str) -> ProgramExample:
        step_specs: List[Tuple[str, str, str, str, str]] = []
        for op, arg in zip(spec.ops, spec.args):
            if op == 0:
                step_specs.append(
                    self._choice_for_mode(
                        template_mode,
                        [("Step: ", "add", " ", str(arg), ".")],
                        [
                            ("Next, ", "increase", " x by ", str(arg), "."),
                            ("Now ", "add", " ", str(arg), " to x."),
                            ("Use an ", "add", " update of ", str(arg), "."),
                        ],
                    )
                )
            elif op == 1:
                step_specs.append(
                    self._choice_for_mode(
                        template_mode,
                        [("Step: ", "subtract", " ", str(arg), ".")],
                        [
                            ("Next, ", "decrease", " x by ", str(arg), "."),
                            ("Now ", "subtract", " ", str(arg), " from x."),
                            ("Use a ", "subtract", " update of ", str(arg), "."),
                        ],
                    )
                )
            else:
                step_specs.append(
                    self._choice_for_mode(
                        template_mode,
                        [("Step: ", "multiply", " by ", str(arg), ".")],
                        [
                            ("Next, ", "multiply", " x by ", str(arg), "."),
                            ("Now ", "scale", " x by ", str(arg), "."),
                            ("Use a ", "multiply", " update of ", str(arg), "."),
                        ],
                    )
                )

        input_ids: List[int] = []
        num_values: List[int] = []
        op_values: List[int] = []
        parts: List[str] = []

        def add_text(text: str, num_value: int = -1, op_value: int = -1) -> int:
            ids = tokenize_no_special(self.tokenizer, text)
            if not ids:
                raise RuntimeError(f"empty tokenization for {text!r}")
            input_ids.extend(ids)
            num_values.extend([num_value] * len(ids))
            op_values.extend([op_value] * len(ids))
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
        init_pos = add_text(str(spec.init_value), num_value=spec.init_value)
        add_text(init_suffix)
        op_pos: List[int] = []
        arg_pos: List[int] = []
        for step_index, (prefix, op_word, between, number, suffix) in enumerate(step_specs):
            add_text(prefix)
            op_pos.append(add_text(op_word, op_value=spec.ops[step_index]))
            add_text(between)
            arg_pos.append(add_text(number, num_value=spec.args[step_index]))
            add_text(suffix + "\n")
        add_text(final_line)
        answer_pos = add_text(answer_marker)
        return ProgramExample(
            prompt="".join(parts).rstrip("\n"),
            length=spec.length,
            init_value=spec.init_value,
            ops=spec.ops,
            args=spec.args,
            answer=spec.answer,
            init_pos=init_pos,
            step_op_pos=op_pos,
            step_arg_pos=arg_pos,
            answer_pos=answer_pos,
            input_ids=input_ids,
            num_values=num_values,
            op_values=op_values,
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
    if name in {"fp16", "float16"}:
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
    init_pos: List[int] = []
    answer_pos: List[int] = []
    num_values: List[List[int]] = []
    op_values: List[List[int]] = []
    ops: List[List[int]] = []
    args: List[List[int]] = []
    op_pos: List[List[int]] = []
    arg_pos: List[List[int]] = []
    for i, ex in enumerate(examples):
        if ex.answer_pos >= seq_len:
            raise RuntimeError("max_length truncates a required position")
        cur = ex.input_ids[:seq_len]
        ids[i, : len(cur)] = torch.tensor(cur, dtype=torch.long, device=device)
        mask[i, : len(cur)] = 1
        num_values.append(ex.num_values[:seq_len] + [-1] * (seq_len - len(cur)))
        op_values.append(ex.op_values[:seq_len] + [-1] * (seq_len - len(cur)))
        lengths.append(ex.length)
        init_values.append(ex.init_value)
        answers.append(ex.answer)
        init_pos.append(ex.init_pos)
        answer_pos.append(ex.answer_pos)
        ops.append(ex.ops + [-100] * (max_steps - len(ex.ops)))
        args.append(ex.args + [-100] * (max_steps - len(ex.args)))
        op_pos.append(ex.step_op_pos + [-100] * (max_steps - len(ex.step_op_pos)))
        arg_pos.append(ex.step_arg_pos + [-100] * (max_steps - len(ex.step_arg_pos)))
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "hidden_mask": mask.bool(),
        "lengths": torch.tensor(lengths, dtype=torch.long, device=device),
        "init_value": torch.tensor(init_values, dtype=torch.long, device=device),
        "ops": torch.tensor(ops, dtype=torch.long, device=device),
        "args": torch.tensor(args, dtype=torch.long, device=device),
        "answer": torch.tensor(answers, dtype=torch.long, device=device),
        "init_pos": torch.tensor(init_pos, dtype=torch.long, device=device),
        "answer_pos": torch.tensor(answer_pos, dtype=torch.long, device=device),
        "num_values": torch.tensor(num_values, dtype=torch.long, device=device),
        "op_values": torch.tensor(op_values, dtype=torch.long, device=device),
        "op_pos": torch.tensor(op_pos, dtype=torch.long, device=device),
        "arg_pos": torch.tensor(arg_pos, dtype=torch.long, device=device),
    }


def sample_batch(dataset: ExampleSet, batch_size: int, pad_id: int, max_steps: int, max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
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
        raise ValueError("pair_size must be at least 2 for paired batches")
    if batch_size % pair_size != 0:
        raise ValueError(f"train_batch_size={batch_size} must be divisible by pair_size={pair_size}")
    n_pairs = len(dataset) // pair_size
    if n_pairs <= 0:
        raise ValueError("paired dataset is empty")
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


def copy_logits_from_token_map(scores: torch.Tensor, token_values: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert position or slot scores into class logits using a lexical token map."""
    neg = -1e4
    classes = torch.arange(num_classes, device=scores.device)
    value_matches = token_values[:, :, None].eq(classes[None, None, :])
    value_matches = value_matches & token_values[:, :, None].ge(0)
    if scores.dim() == 2:
        masked = scores[:, :, None].masked_fill(~value_matches, neg)
        return torch.logsumexp(masked, dim=1)
    if scores.dim() == 3:
        masked = scores[:, :, :, None].masked_fill(~value_matches[:, None, :, :], neg)
        return torch.logsumexp(masked, dim=2)
    raise ValueError(f"expected 2D or 3D scores, got {tuple(scores.shape)}")


class ProgramCompiler(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        modulus: int,
        width: int,
        max_steps: int,
        rank_temperature: float,
        arg_reader_mode: str,
        arg_window: int,
        arg_distance_temperature: float,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.rank_temperature = float(rank_temperature)
        self.arg_reader_mode = arg_reader_mode
        self.arg_window = int(arg_window)
        self.arg_distance_temperature = float(arg_distance_temperature)
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.token_ff = nn.Sequential(nn.Linear(hidden_dim, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.role_head = nn.Linear(width, 3)
        nn.init.constant_(self.role_head.bias, -2.5)
        self.modulus = int(modulus)

    def _monotonic_slot_scores(self, role_logits: torch.Tensor, hidden_mask: torch.Tensor) -> torch.Tensor:
        mask = hidden_mask.bool()
        role_prob = torch.sigmoid(role_logits).masked_fill(~mask, 0.0)
        cumulative = torch.cumsum(role_prob, dim=1)
        centers = torch.arange(1, self.max_steps + 1, device=role_logits.device, dtype=role_logits.dtype)
        distance = cumulative[:, None, :] - centers[None, :, None]
        scores = torch.log(role_prob.clamp_min(1e-6))[:, None, :] - distance.square() / self.rank_temperature
        return scores.masked_fill(~mask[:, None, :], VERY_NEG)

    def _after_op_arg_scores(self, arg_role_logits: torch.Tensor, hidden_mask: torch.Tensor, op_slot_scores: torch.Tensor) -> torch.Tensor:
        mask = hidden_mask.bool()
        seq_len = arg_role_logits.shape[1]
        min_value = VERY_NEG
        arg_role_score = torch.log(torch.sigmoid(arg_role_logits).clamp_min(1e-6)).masked_fill(~mask, min_value)
        op_log_w = F.log_softmax(op_slot_scores, dim=-1)
        positions = torch.arange(seq_len, device=arg_role_logits.device)
        distance = positions[None, :] - positions[:, None]
        valid = (distance > 0) & (distance <= self.arg_window)
        distance_penalty = -((distance.to(arg_role_logits.dtype) - 1.0).square() / self.arg_distance_temperature)
        distance_penalty = distance_penalty.masked_fill(~valid, min_value)
        anchored_scores = torch.logsumexp(op_log_w[:, :, :, None] + distance_penalty[None, None, :, :], dim=2)
        return (anchored_scores + arg_role_score[:, None, :]).masked_fill(~mask[:, None, :], min_value)

    def forward(
        self,
        hidden: torch.Tensor,
        hidden_mask: torch.Tensor,
        num_values: torch.Tensor,
        op_values: torch.Tensor,
        return_scores: bool = False,
    ) -> Any:
        x = self.token_ff(self.hidden_norm(hidden.float()))
        mask = hidden_mask.bool()
        role_logits = self.role_head(x)
        init_scores = role_logits[:, :, 0].masked_fill(~mask, VERY_NEG)
        op_slot_scores = self._monotonic_slot_scores(role_logits[:, :, 1], hidden_mask)
        if self.arg_reader_mode == "after_op":
            arg_slot_scores = self._after_op_arg_scores(role_logits[:, :, 2], hidden_mask, op_slot_scores)
        else:
            arg_slot_scores = self._monotonic_slot_scores(role_logits[:, :, 2], hidden_mask)
        init_logits = copy_logits_from_token_map(init_scores, num_values, self.modulus)
        op_logits = copy_logits_from_token_map(op_slot_scores, op_values, len(OP_NAMES))
        arg_logits = copy_logits_from_token_map(arg_slot_scores, num_values, self.modulus)
        if return_scores:
            return init_logits, op_logits, arg_logits, {
                "role_logits": role_logits,
                "init_scores": init_scores,
                "op_slot_scores": op_slot_scores,
                "arg_slot_scores": arg_slot_scores,
            }
        return init_logits, op_logits, arg_logits


class TransitionExecutor(nn.Module):
    def __init__(self, modulus: int, device: torch.device) -> None:
        super().__init__()
        table = torch.zeros(len(OP_NAMES), modulus, modulus, modulus, dtype=torch.float32)
        for op in range(len(OP_NAMES)):
            for arg in range(modulus):
                for old in range(modulus):
                    table[op, arg, old, apply_op(old, op, arg, modulus)] = 1.0
        self.register_buffer("table", table.to(device))

    def soft_forward(self, init_logits: torch.Tensor, op_logits: torch.Tensor, arg_logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        state = F.softmax(init_logits.float(), dim=-1)
        op_probs = F.softmax(op_logits.float(), dim=-1)
        arg_probs = F.softmax(arg_logits.float(), dim=-1)
        for t in range(op_logits.shape[1]):
            cand = torch.einsum("bp,oapq->boaq", state, self.table)
            weights = op_probs[:, t, :, None] * arg_probs[:, t, None, :]
            next_state = (cand * weights[:, :, :, None]).sum(dim=(1, 2))
            active = (lengths > t).float().unsqueeze(-1)
            state = active * next_state + (1.0 - active) * state
        return state.clamp_min(1e-9)


def extract_hidden(outputs: Any) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]
    last = getattr(outputs, "last_hidden_state", None)
    if torch.is_tensor(last):
        return last
    raise RuntimeError("model did not return hidden states")


def trace_losses(init_logits: torch.Tensor, op_logits: torch.Tensor, arg_logits: torch.Tensor, batch: Dict[str, torch.Tensor], init_weight: float, op_weight: float, arg_weight: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    active = batch["ops"].ne(-100)
    init_loss = F.cross_entropy(init_logits, batch["init_value"])
    op_loss = F.cross_entropy(op_logits[active], batch["ops"][active])
    arg_loss = F.cross_entropy(arg_logits[active], batch["args"][active])
    return init_weight * init_loss + op_weight * op_loss + arg_weight * arg_loss, {
        "init_loss": float(init_loss.detach().cpu()),
        "op_loss": float(op_loss.detach().cpu()),
        "arg_loss": float(arg_loss.detach().cpu()),
    }


def selection_losses(scores: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, float]]:
    init_loss = F.cross_entropy(scores["init_scores"], batch["init_pos"])
    op_mask = batch["op_pos"].ne(-100)
    arg_mask = batch["arg_pos"].ne(-100)
    op_loss = F.cross_entropy(scores["op_slot_scores"][op_mask], batch["op_pos"][op_mask])
    arg_loss = F.cross_entropy(scores["arg_slot_scores"][arg_mask], batch["arg_pos"][arg_mask])
    role_targets = torch.zeros_like(scores["role_logits"])
    rows = torch.arange(role_targets.shape[0], device=role_targets.device)
    role_targets[rows, batch["init_pos"], 0] = 1.0
    for col, pos in [(1, batch["op_pos"]), (2, batch["arg_pos"])]:
        mask = pos.ne(-100)
        b_idx, _ = mask.nonzero(as_tuple=True)
        role_targets[b_idx, pos[mask], col] = 1.0
    valid = batch["hidden_mask"].bool()
    role_logits = scores["role_logits"][valid]
    role_targets_flat = role_targets[valid]
    pos_weight = torch.full((3,), args.role_pos_weight, dtype=role_logits.dtype, device=role_logits.device)
    role_loss = F.binary_cross_entropy_with_logits(role_logits, role_targets_flat, pos_weight=pos_weight)
    total = (
        args.init_selection_loss_weight * init_loss
        + args.op_selection_loss_weight * op_loss
        + args.arg_selection_loss_weight * arg_loss
        + role_loss
    )
    return total, {
        "init_attn_loss": float(init_loss.detach().cpu()),
        "op_attn_loss": float(op_loss.detach().cpu()),
        "arg_attn_loss": float(arg_loss.detach().cpu()),
        "role_loss": float(role_loss.detach().cpu()),
    }


def symmetric_kl_from_logits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    log_a = F.log_softmax(a.float(), dim=-1)
    log_b = F.log_softmax(b.float(), dim=-1)
    prob_a = log_a.exp()
    prob_b = log_b.exp()
    # Avoid F.kl_div's target.log() path: masked lexical classes can have
    # exactly zero probability, which gives finite forward values but NaN
    # gradients through 0 * log(0).
    kl_ab = (prob_a * (log_a - log_b)).sum(dim=-1)
    kl_ba = (prob_b * (log_b - log_a)).sum(dim=-1)
    return 0.5 * (kl_ab + kl_ba)


def stability_losses(
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    answer_probs: Optional[torch.Tensor],
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    pair_size = int(args.pair_size)
    if pair_size != 2:
        raise ValueError("stability loss currently expects pair_size=2")
    if init_logits.shape[0] % pair_size != 0:
        raise ValueError("stability batch size must be divisible by pair_size")
    a = torch.arange(0, init_logits.shape[0], pair_size, device=init_logits.device)
    b = a + 1
    init_loss = symmetric_kl_from_logits(init_logits[a], init_logits[b]).mean()
    active = batch["ops"][a].ne(-100) & batch["ops"][b].ne(-100)
    op_pair = symmetric_kl_from_logits(op_logits[a], op_logits[b])
    arg_pair = symmetric_kl_from_logits(arg_logits[a], arg_logits[b])
    op_loss = op_pair[active].mean() if active.any() else op_pair.mean() * 0.0
    arg_loss = arg_pair[active].mean() if active.any() else arg_pair.mean() * 0.0
    answer_loss = torch.tensor(0.0, device=init_logits.device)
    if answer_probs is not None and args.answer_stability_weight > 0.0:
        answer_loss = symmetric_kl_from_logits(answer_probs[a].log(), answer_probs[b].log()).mean()
    total = (
        args.init_stability_weight * init_loss
        + args.op_stability_weight * op_loss
        + args.arg_stability_weight * arg_loss
        + args.answer_stability_weight * answer_loss
    )
    return total, {
        "init_stability_loss": float(init_loss.detach().cpu()),
        "op_stability_loss": float(op_loss.detach().cpu()),
        "arg_stability_loss": float(arg_loss.detach().cpu()),
        "answer_stability_loss": float(answer_loss.detach().cpu()),
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


def forward_hidden(model: nn.Module, batch: Dict[str, torch.Tensor], output_hidden_states: bool = True) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    return extract_hidden(outputs)


@torch.no_grad()
def evaluate(
    variant: str,
    model: nn.Module,
    direct: Optional[DirectAnswerHead],
    compiler: Optional[ProgramCompiler],
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
    init_correct = init_pos_correct = 0
    op_correct = arg_correct = op_pos_correct = arg_pos_correct = op_total = 0
    program_exact = 0
    answer_values: List[torch.Tensor] = []
    length_values: List[torch.Tensor] = []
    direct_pred_values: List[torch.Tensor] = []
    exec_pred_values: List[torch.Tensor] = []
    init_pred_values: List[torch.Tensor] = []
    op_pred_values: List[torch.Tensor] = []
    arg_pred_values: List[torch.Tensor] = []
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
            direct_pred = logits.argmax(dim=-1)
            direct_pred_values.append(direct_pred.detach().cpu())
            direct_correct += int(direct_pred.eq(batch["answer"]).sum().item())
            direct_mass += float(probs.gather(1, batch["answer"].view(-1, 1)).sum().item())
        if compiler is not None:
            init_logits, op_logits, arg_logits, scores = compiler(
                hidden,
                batch["hidden_mask"],
                batch["num_values"],
                batch["op_values"],
                return_scores=True,
            )
            probs = executor.soft_forward(init_logits, op_logits, arg_logits, batch["lengths"])
            exec_mass += float(probs.gather(1, batch["answer"].view(-1, 1)).sum().item())
            init_pred = init_logits.argmax(dim=-1)
            op_pred = op_logits.argmax(dim=-1)
            arg_pred = arg_logits.argmax(dim=-1)
            pred_answer = argmax_execute(init_pred, op_pred, arg_pred, batch["lengths"], args.modulus)
            exec_pred_values.append(pred_answer.detach().cpu())
            init_pred_values.append(init_pred.detach().cpu())
            op_pred_values.append(op_pred.detach().cpu())
            arg_pred_values.append(arg_pred.detach().cpu())
            exec_correct += int(pred_answer.eq(batch["answer"]).sum().item())
            init_ok = init_pred.eq(batch["init_value"])
            init_correct += int(init_ok.sum().item())
            active = batch["ops"].ne(-100)
            op_ok = op_pred.eq(batch["ops"]) | ~active
            arg_ok = arg_pred.eq(batch["args"]) | ~active
            init_pos_correct += int(scores["init_scores"].argmax(dim=-1).eq(batch["init_pos"]).sum().item())
            op_pos_correct += int(scores["op_slot_scores"].argmax(dim=-1)[active].eq(batch["op_pos"][active]).sum().item())
            arg_pos_correct += int(scores["arg_slot_scores"].argmax(dim=-1)[active].eq(batch["arg_pos"][active]).sum().item())
            op_correct += int(op_pred[active].eq(batch["ops"][active]).sum().item())
            arg_correct += int(arg_pred[active].eq(batch["args"][active]).sum().item())
            op_total += int(active.sum().item())
            program_exact += int((init_ok & op_ok.all(dim=1) & arg_ok.all(dim=1)).sum().item())
    metrics = {
        "variant": variant,
        "n": float(total),
        "direct_accuracy": direct_correct / total if direct is not None else math.nan,
        "direct_target_mass": direct_mass / total if direct is not None else math.nan,
        "executor_accuracy": exec_correct / total if compiler is not None else math.nan,
        "executor_target_mass": exec_mass / total if compiler is not None else math.nan,
        "init_accuracy": init_correct / total if compiler is not None else math.nan,
        "init_pos_accuracy": init_pos_correct / total if compiler is not None else math.nan,
        "op_accuracy": op_correct / op_total if compiler is not None and op_total else math.nan,
        "arg_accuracy": arg_correct / op_total if compiler is not None and op_total else math.nan,
        "op_pos_accuracy": op_pos_correct / op_total if compiler is not None and op_total else math.nan,
        "arg_pos_accuracy": arg_pos_correct / op_total if compiler is not None and op_total else math.nan,
        "program_exact": program_exact / total if compiler is not None else math.nan,
    }
    if dataset.paired and dataset.pair_size == 2 and total >= 2:
        usable = (total // 2) * 2
        pair_count = usable // 2
        answers = torch.cat(answer_values)[:usable].view(pair_count, 2)
        lengths = torch.cat(length_values)[:usable].view(pair_count, 2)
        metrics["pair_true_answer_consistency"] = float(answers[:, 0].eq(answers[:, 1]).float().mean().item())
        metrics["pair_true_length_consistency"] = float(lengths[:, 0].eq(lengths[:, 1]).float().mean().item())
        if direct_pred_values:
            direct_preds = torch.cat(direct_pred_values)[:usable].view(pair_count, 2)
            metrics["direct_pair_answer_consistency"] = float(direct_preds[:, 0].eq(direct_preds[:, 1]).float().mean().item())
            metrics["direct_pair_both_correct"] = float(direct_preds.eq(answers).all(dim=1).float().mean().item())
        if exec_pred_values:
            exec_preds = torch.cat(exec_pred_values)[:usable].view(pair_count, 2)
            init_preds = torch.cat(init_pred_values)[:usable].view(pair_count, 2)
            op_preds = torch.cat(op_pred_values)[:usable].view(pair_count, 2, -1)
            arg_preds = torch.cat(arg_pred_values)[:usable].view(pair_count, 2, -1)
            program_same = []
            for pair_idx in range(pair_count):
                length = int(lengths[pair_idx, 0].item())
                same = bool(init_preds[pair_idx, 0].eq(init_preds[pair_idx, 1]).item())
                same = same and bool(op_preds[pair_idx, 0, :length].eq(op_preds[pair_idx, 1, :length]).all().item())
                same = same and bool(arg_preds[pair_idx, 0, :length].eq(arg_preds[pair_idx, 1, :length]).all().item())
                program_same.append(same)
            metrics["executor_pair_answer_consistency"] = float(exec_preds[:, 0].eq(exec_preds[:, 1]).float().mean().item())
            metrics["executor_pair_both_correct"] = float(exec_preds.eq(answers).all(dim=1).float().mean().item())
            metrics["compiler_pair_program_consistency"] = sum(program_same) / len(program_same)
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
        raise RuntimeError("no trainable parameters; enable LoRA or a trainable head")
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
    train_set: ExampleSet,
    answer_set: ExampleSet,
    eval_sets: Dict[str, ExampleSet],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    train_seed = args.train_seed if args.train_seed >= 0 else args.seed + sum(ord(ch) for ch in variant)
    torch.manual_seed(train_seed)
    direct: Optional[DirectAnswerHead] = None
    compiler: Optional[ProgramCompiler] = None
    modules: List[nn.Module] = [model]
    if variant == "direct":
        direct = DirectAnswerHead(hidden_dim, args.modulus, args.head_width).to(device)
        modules.append(direct)
    else:
        compiler = ProgramCompiler(
            hidden_dim,
            args.modulus,
            args.head_width,
            args.max_steps,
            args.rank_temperature,
            args.arg_reader_mode,
            args.arg_window,
            args.arg_distance_temperature,
        ).to(device)
        modules.append(compiler)
    executor = TransitionExecutor(args.modulus, device)
    model_trainable = any(p.requires_grad for p in model.parameters())
    trainable = [p for m in modules for p in m.parameters() if p.requires_grad]
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
        stability_active: bool = False,
        lr_scale: float = 1.0,
        offset: int = 0,
    ) -> int:
        for group in opt.param_groups:
            group["lr"] = args.lr * lr_scale
        if stability_active and not args.paired_train:
            raise ValueError("stability_active requires --paired_train")
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
                init_logits, op_logits, arg_logits, scores = compiler(
                    hidden,
                    batch["hidden_mask"],
                    batch["num_values"],
                    batch["op_values"],
                    return_scores=True,
                )
                answer_probs: Optional[torch.Tensor] = None
                if executor_active:
                    answer_probs = executor.soft_forward(init_logits, op_logits, arg_logits, batch["lengths"])
                    exec_loss = F.nll_loss(answer_probs.log(), batch["answer"])
                    loss = loss + args.executor_loss_weight * exec_loss
                    aux["executor_loss"] = float(exec_loss.detach().cpu())
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
                    sel_loss, sel_aux = selection_losses(scores, batch, args)
                    loss = loss + args.trace_loss_weight * tr_loss + args.selection_loss_weight * sel_loss
                    aux.update(tr_aux)
                    aux.update(sel_aux)
                elif args.selection_loss_weight > 0 and not executor_active:
                    sel_loss, sel_aux = selection_losses(scores, batch, args)
                    loss = loss + args.selection_loss_weight * sel_loss
                    aux.update(sel_aux)
                if stability_active and args.stability_loss_weight > 0.0:
                    st_loss, st_aux = stability_losses(init_logits, op_logits, arg_logits, answer_probs, batch, args)
                    loss = loss + args.stability_loss_weight * st_loss
                    aux.update(st_aux)
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
                    if key in row and not math.isnan(row[key]):
                        msg += f" {split}={100.0 * row[key]:.1f}%"
                print(msg, flush=True)
        return offset + steps

    step = 0
    if variant == "direct":
        step = run_stage("answer_only", answer_set, args.bootstrap_steps + args.answer_steps, False, False, False, 1.0, step)
    elif variant == "copy_tagger":
        step = run_stage("tagger", train_set, args.bootstrap_steps, True, False, False, 1.0, step)
    elif variant == "copy_trace":
        step = run_stage("trace", train_set, args.bootstrap_steps, True, True, False, 1.0, step)
    elif variant == "copy_trace_stability":
        step = run_stage("trace_stability", train_set, args.bootstrap_steps, True, True, True, 1.0, step)
    elif variant == "copy_answer_only":
        step = run_stage("answer_only", answer_set, args.bootstrap_steps + args.answer_steps, False, True, False, 1.0, step)
    elif variant == "copy_trace_then_answer":
        step = run_stage("trace", train_set, args.bootstrap_steps, True, True, False, 1.0, step)
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
    return Path("large_artifacts/qwen_slot_stability_compiler/checkpoints") / Path(args.output_dir).name


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
    p = argparse.ArgumentParser(description="Paired slot-stability Qwen numeric-copy compiler experiment")
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
    p.add_argument("--answer_train_min_len", type=int, default=1)
    p.add_argument("--answer_train_max_len", type=int, default=12)
    p.add_argument("--eval_lengths", type=str, default="4,8,12,24")
    p.add_argument("--train_template_mode", type=str, default="mixed", choices=["standard", "mixed", "paraphrase"])
    p.add_argument("--paired_train", action=argparse.BooleanOptionalAction, default=False)
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
    p.add_argument("--bootstrap_steps", type=int, default=400)
    p.add_argument("--answer_steps", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--stage_eval_every", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--answer_lr_scale", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--head_width", type=int, default=512)
    p.add_argument("--rank_temperature", type=float, default=1.0)
    p.add_argument("--arg_reader_mode", type=str, default="after_op", choices=["monotonic", "after_op"])
    p.add_argument("--arg_window", type=int, default=8)
    p.add_argument("--arg_distance_temperature", type=float, default=2.0)
    p.add_argument("--trace_loss_weight", type=float, default=1.0)
    p.add_argument("--init_trace_loss_weight", type=float, default=4.0)
    p.add_argument("--op_trace_loss_weight", type=float, default=1.0)
    p.add_argument("--arg_trace_loss_weight", type=float, default=4.0)
    p.add_argument("--selection_loss_weight", type=float, default=1.0)
    p.add_argument("--role_pos_weight", type=float, default=20.0)
    p.add_argument("--init_selection_loss_weight", type=float, default=1.0)
    p.add_argument("--op_selection_loss_weight", type=float, default=1.0)
    p.add_argument("--arg_selection_loss_weight", type=float, default=4.0)
    p.add_argument("--executor_loss_weight", type=float, default=1.0)
    p.add_argument("--stability_loss_weight", type=float, default=0.25)
    p.add_argument("--init_stability_weight", type=float, default=0.25)
    p.add_argument("--op_stability_weight", type=float, default=1.0)
    p.add_argument("--arg_stability_weight", type=float, default=1.0)
    p.add_argument("--answer_stability_weight", type=float, default=0.25)
    p.add_argument("--variants", type=str, default="copy_trace_stability")
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--train_seed", type=int, default=-1)
    p.add_argument("--output_dir", type=str, default="experiments/qwen_slot_stability_compiler/runs/default")
    p.add_argument("--checkpoint_dir", type=str, default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, loader = load_model_and_tokenizer(args)
    if not args.load_in_4bit:
        model.to(device)
    metadata = collect_metadata(args, loader)

    gen_train = TextProgramGenerator(tokenizer, args.modulus, args.max_steps, args.seed, args.train_template_mode)
    paired_template_modes = [x.strip() for x in args.paired_template_modes.split(",") if x.strip()]
    if args.paired_train:
        if len(paired_template_modes) != args.pair_size:
            raise ValueError("paired_template_modes count must match pair_size")
        train_set = gen_train.paired_dataset(args.train_size, args.train_min_len, args.train_max_len, paired_template_modes)
    else:
        train_set = gen_train.dataset(args.train_size, args.train_min_len, args.train_max_len)
    gen_answer = TextProgramGenerator(tokenizer, args.modulus, args.max_steps, args.seed + 177, args.answer_train_template_mode)
    answer_set = gen_answer.dataset(args.answer_train_size, args.answer_train_min_len, args.answer_train_max_len)
    eval_sets: Dict[str, ExampleSet] = {}
    eval_modes = [x.strip() for x in args.eval_template_modes.split(",") if x.strip()]
    paired_eval_modes = [x.strip() for x in args.paired_eval_template_modes.split(",") if x.strip()]
    eval_lengths = [int(x.strip()) for x in args.eval_lengths.split(",") if x.strip()]
    for mode_index, mode in enumerate(eval_modes):
        for length in eval_lengths:
            gen = TextProgramGenerator(tokenizer, args.modulus, args.max_steps, args.seed + 1000 + 97 * mode_index + length, mode)
            eval_sets[f"{mode}_len{length}"] = gen.dataset(args.eval_size, length, length)
    if args.paired_eval:
        if len(paired_eval_modes) != 2:
            raise ValueError("paired_eval currently expects exactly two template modes")
        for length in eval_lengths:
            gen = TextProgramGenerator(tokenizer, args.modulus, args.max_steps, args.seed + 2000 + length, "mixed")
            eval_sets[f"paired_len{length}"] = gen.paired_dataset(args.eval_size, length, length, paired_eval_modes)

    probe_batch = collate_examples(train_set.examples[:1], int(tokenizer.pad_token_id), args.max_steps, args.max_length, device)
    with torch.no_grad():
        hidden_dim = int(forward_hidden(model, probe_batch).shape[-1])
    print(f"[setup] hidden_dim={hidden_dim} train={len(train_set)} answer={len(answer_set)} eval_splits={len(eval_sets)}", flush=True)

    results: Dict[str, Any] = {
        "args": vars(args),
        "metadata": metadata,
        "hidden_dim": hidden_dim,
        "dataset": {
            "train_size": args.train_size,
            "train_examples": len(train_set),
            "answer_train_size": args.answer_train_size,
            "eval_size": args.eval_size,
            "train_lengths": [args.train_min_len, args.train_max_len],
            "answer_train_lengths": [args.answer_train_min_len, args.answer_train_max_len],
            "eval_lengths": eval_lengths,
            "train_template_mode": args.train_template_mode,
            "paired_train": args.paired_train,
            "paired_batches": args.paired_batches,
            "paired_template_modes": paired_template_modes,
            "answer_train_template_mode": args.answer_train_template_mode,
            "eval_template_modes": eval_modes,
            "paired_eval": args.paired_eval,
            "paired_eval_template_modes": paired_eval_modes,
        },
        "variants": {},
    }
    flat_rows: List[Dict[str, Any]] = []
    for variant in [x.strip() for x in args.variants.split(",") if x.strip()]:
        # Each variant starts from the same loaded base only when launched in a
        # separate run. Keep multi-variant runs for closely related continuations.
        variant_result = train_variant(variant, model, hidden_dim, train_set, answer_set, eval_sets, tokenizer, args, device)
        results["variants"][variant] = variant_result
        flat_rows.extend(variant_result["train_log"])
    with (out / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    write_csv(out / "train_log.csv", flat_rows)
    print(f"[done] wrote {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
