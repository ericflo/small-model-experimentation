#!/usr/bin/env python3
"""Tail-repair critic for length-24 executable modular programs.

This standalone experiment freezes several source compilers, enumerates local
tail edits around their generated programs, labels candidate repairs with exact
execution checks, and trains a critic to select repair candidates without using
target answers or target states at inference.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import importlib.util
import json
import math
import platform
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path("/workspace/experiments/qwen_tail_repair_stability_critic")
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_tail_repair_stability_critic")
CACHE_ROOT = LARGE_ROOT / "candidate_groups"
CHECKPOINT_ROOT = LARGE_ROOT / "checkpoints"
SOURCE_MODULE_PATH = Path("/workspace/experiments/qwen_compiler_multiseed_reattribution/src/qwen_compiler_multiseed_reattribution.py")
SOURCE_CHECKPOINT_ROOT = Path("/workspace/large_artifacts/qwen_compiler_multiseed_reattribution/checkpoints")
OP_NAMES = ["ADD", "SUB", "MUL"]


FEATURE_NAMES = [
    "bias",
    "candidate_prior_per_slot",
    "prior_delta_per_slot",
    "prior_delta_per_edit",
    "base_prior_per_slot",
    "edit_count",
    "changed",
    "init_edit",
    "op_edit_count",
    "arg_edit_count",
    "edit_fraction",
    "first_edit_pos_norm",
    "last_edit_pos_norm",
    "mean_edit_pos_norm",
    "edit_span_norm",
    "mean_tail_distance_norm",
    "mean_new_rank",
    "max_new_rank",
    "mean_logp_delta",
    "min_logp_delta",
    "max_logp_delta",
    "mean_changed_margin",
    "min_changed_margin",
    "mean_changed_entropy",
    "base_init_margin",
    "base_op_margin_mean",
    "base_op_margin_min",
    "base_arg_margin_mean",
    "base_arg_margin_min",
    "base_op_entropy_mean",
    "base_arg_entropy_mean",
    "candidate_answer_norm",
    "base_answer_norm",
    "answer_equals_base",
    "candidate_final_soft_logp",
    "candidate_state_soft_logp_mean",
    "candidate_state_soft_logp_min",
    "base_final_soft_logp",
    "base_state_soft_logp_mean",
    "state_same_fraction_as_base",
    "state_prefix_same_as_base",
    "state_mean_abs_delta_from_base_norm",
    "candidate_add_frac",
    "candidate_sub_frac",
    "candidate_mul_frac",
    "base_add_frac",
    "base_sub_frac",
    "base_mul_frac",
    "candidate_arg_mean_norm",
    "candidate_arg_std_norm",
    "base_arg_mean_norm",
    "base_arg_std_norm",
]


@dataclass
class Candidate:
    init_value: int
    ops: List[int]
    args: List[int]
    answer: int
    states: List[int]
    prior: float
    features: List[float]
    answer_exact: bool
    state_exact: bool
    program_exact: bool
    changed: bool
    edit_count: int


@dataclass
class CandidateGroup:
    source_run: str
    source_seed: int
    split: str
    example_index: int
    length: int
    answer: int
    init_value: int
    ops: List[int]
    args: List[int]
    states: List[int]
    base_index: int
    candidates: List[Candidate]
    features: torch.Tensor
    answer_labels: torch.Tensor
    state_labels: torch.Tensor
    program_labels: torch.Tensor
    priors: torch.Tensor
    context: torch.Tensor


class RepairCritic(nn.Module):
    def __init__(self, input_dim: int, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def log(message: str) -> None:
    print(message, flush=True)


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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default))


def parse_csv_list(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def source_seed_from_run(run: str) -> int:
    match = re.search(r"_seed(\d+)$", run)
    return int(match.group(1)) if match else -1


def load_source_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("source_compiler_module", SOURCE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load source compiler module from {SOURCE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.convert_ids_to_tokens(0)


def dtype_from_string(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(name)


def apply_op(x: int, op: int, arg: int, modulus: int) -> int:
    if op == 0:
        return (x + arg) % modulus
    if op == 1:
        return (x - arg) % modulus
    if op == 2:
        return (x * arg) % modulus
    raise ValueError(op)


def execute_program(init_value: int, ops: Sequence[int], args: Sequence[int], length: int, max_steps: int, modulus: int) -> Tuple[int, List[int]]:
    x = int(init_value)
    states = [-100] * max_steps
    for step in range(int(length)):
        x = apply_op(x, int(ops[step]), int(args[step]), modulus)
        states[step] = x
    return x, states


def correct_prefix_fraction(pred_states: Sequence[int], target_states: Sequence[int], length: int) -> float:
    if length <= 0:
        return 1.0
    prefix = 0
    for step in range(length):
        if int(pred_states[step]) != int(target_states[step]):
            break
        prefix += 1
    return prefix / float(length)


def same_prefix_fraction(states_a: Sequence[int], states_b: Sequence[int], length: int) -> float:
    if length <= 0:
        return 1.0
    prefix = 0
    for step in range(length):
        if int(states_a[step]) != int(states_b[step]):
            break
        prefix += 1
    return prefix / float(length)


def same_fraction(states_a: Sequence[int], states_b: Sequence[int], length: int) -> float:
    if length <= 0:
        return 1.0
    return sum(int(states_a[i]) == int(states_b[i]) for i in range(length)) / float(length)


def entropy_from_logits(logits: torch.Tensor) -> float:
    probs = F.softmax(logits.float(), dim=-1)
    logp = F.log_softmax(logits.float(), dim=-1)
    return float(-(probs * logp).sum().item())


def top2_margin(logits: torch.Tensor) -> float:
    values = torch.topk(F.log_softmax(logits.float(), dim=-1), k=min(2, logits.numel())).values
    if values.numel() < 2:
        return 0.0
    return float((values[0] - values[1]).item())


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else float(default)


def safe_min(values: Sequence[float], default: float = 0.0) -> float:
    return float(min(values)) if values else float(default)


def safe_max(values: Sequence[float], default: float = 0.0) -> float:
    return float(max(values)) if values else float(default)


def safe_std(values: Sequence[float], default: float = 0.0) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64))) if values else float(default)


def top_options_with_rank(logits: torch.Tensor, topk: int) -> Dict[int, Tuple[float, int]]:
    logp = F.log_softmax(logits.float(), dim=-1)
    k = max(1, min(int(topk), int(logp.numel())))
    values, indices = torch.topk(logp, k=k)
    return {int(idx.item()): (float(val.item()), int(rank)) for rank, (val, idx) in enumerate(zip(values, indices))}


def op_fractions(ops: Sequence[int], length: int) -> Tuple[float, float, float]:
    if length <= 0:
        return 0.0, 0.0, 0.0
    return tuple(sum(int(op) == idx for op in ops[:length]) / float(length) for idx in range(3))  # type: ignore[return-value]


def log_prob_at(probs: torch.Tensor, index: int) -> float:
    return float(torch.log(probs[int(index)].clamp_min(1e-9)).item())


def candidate_feature_vector(
    candidate_prior: float,
    base_prior: float,
    edit_records: Sequence[Dict[str, float]],
    init_value: int,
    ops: Sequence[int],
    args_values: Sequence[int],
    answer: int,
    states: Sequence[int],
    base_answer: int,
    base_states: Sequence[int],
    base_ops: Sequence[int],
    base_args: Sequence[int],
    source_state_probs: torch.Tensor,
    source_final_probs: torch.Tensor,
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    length: int,
    max_steps: int,
    modulus: int,
) -> List[float]:
    length = int(length)
    denom = float(1 + 2 * max(1, length))
    edit_count = len(edit_records)
    changed = 1.0 if edit_count else 0.0
    positions = [float(row["pos"]) for row in edit_records if row.get("pos", -1) >= 0]
    logp_deltas = [float(row.get("logp_delta", 0.0)) for row in edit_records]
    ranks = [float(row.get("new_rank", 0.0)) for row in edit_records]
    margins = [float(row.get("margin", 0.0)) for row in edit_records]
    entropies = [float(row.get("entropy", 0.0)) for row in edit_records]
    first_pos = min(positions) if positions else -1.0
    last_pos = max(positions) if positions else -1.0
    mean_pos = safe_mean(positions, -1.0)
    tail_dists = [(length - 1 - pos) / max(1.0, float(length)) for pos in positions]
    op_edit_count = sum(row.get("kind") == "op" for row in edit_records)
    arg_edit_count = sum(row.get("kind") == "arg" for row in edit_records)
    init_edit = sum(row.get("kind") == "init" for row in edit_records)
    candidate_add, candidate_sub, candidate_mul = op_fractions(ops, length)
    base_add, base_sub, base_mul = op_fractions(base_ops, length)
    active_args = [float(x) / max(1.0, float(modulus - 1)) for x in args_values[:length]]
    base_active_args = [float(x) / max(1.0, float(modulus - 1)) for x in base_args[:length]]
    state_deltas = [
        abs(float(states[i]) - float(base_states[i])) / max(1.0, float(modulus - 1))
        for i in range(length)
    ]
    state_logps = [log_prob_at(source_state_probs[i], int(states[i])) for i in range(length)]
    base_state_logps = [log_prob_at(source_state_probs[i], int(base_states[i])) for i in range(length)]
    base_op_margins = [top2_margin(op_logits[i]) for i in range(length)]
    base_arg_margins = [top2_margin(arg_logits[i]) for i in range(length)]
    base_op_entropies = [entropy_from_logits(op_logits[i]) for i in range(length)]
    base_arg_entropies = [entropy_from_logits(arg_logits[i]) for i in range(length)]
    return [
        1.0,
        float(candidate_prior) / denom,
        float(candidate_prior - base_prior) / denom,
        float(candidate_prior - base_prior) / max(1.0, float(edit_count)),
        float(base_prior) / denom,
        float(edit_count),
        changed,
        float(init_edit),
        float(op_edit_count),
        float(arg_edit_count),
        float(edit_count) / denom,
        first_pos / max(1.0, float(length - 1)) if first_pos >= 0 else -1.0,
        last_pos / max(1.0, float(length - 1)) if last_pos >= 0 else -1.0,
        mean_pos / max(1.0, float(length - 1)) if mean_pos >= 0 else -1.0,
        (last_pos - first_pos) / max(1.0, float(length - 1)) if positions else 0.0,
        safe_mean(tail_dists, 0.0),
        safe_mean(ranks, 0.0),
        safe_max(ranks, 0.0),
        safe_mean(logp_deltas, 0.0),
        safe_min(logp_deltas, 0.0),
        safe_max(logp_deltas, 0.0),
        safe_mean(margins, 0.0),
        safe_min(margins, 0.0),
        safe_mean(entropies, 0.0),
        top2_margin(init_logits),
        safe_mean(base_op_margins, 0.0),
        safe_min(base_op_margins, 0.0),
        safe_mean(base_arg_margins, 0.0),
        safe_min(base_arg_margins, 0.0),
        safe_mean(base_op_entropies, 0.0),
        safe_mean(base_arg_entropies, 0.0),
        float(answer) / max(1.0, float(modulus - 1)),
        float(base_answer) / max(1.0, float(modulus - 1)),
        1.0 if int(answer) == int(base_answer) else 0.0,
        log_prob_at(source_final_probs, int(answer)),
        safe_mean(state_logps, 0.0),
        safe_min(state_logps, 0.0),
        log_prob_at(source_final_probs, int(base_answer)),
        safe_mean(base_state_logps, 0.0),
        same_fraction(states, base_states, length),
        same_prefix_fraction(states, base_states, length),
        safe_mean(state_deltas, 0.0),
        candidate_add,
        candidate_sub,
        candidate_mul,
        base_add,
        base_sub,
        base_mul,
        safe_mean(active_args, 0.0),
        safe_std(active_args, 0.0),
        safe_mean(base_active_args, 0.0),
        safe_std(base_active_args, 0.0),
    ]


def make_candidate_group(
    source_run: str,
    source_seed: int,
    split: str,
    example_index: int,
    example: Any,
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    context: torch.Tensor,
    executor: Any,
    args: argparse.Namespace,
) -> CandidateGroup:
    max_steps = int(args.max_steps)
    modulus = int(args.modulus)
    length = int(example.length)
    init_logits = init_logits.detach().cpu().float()
    op_logits = op_logits.detach().cpu().float()
    arg_logits = arg_logits.detach().cpu().float()
    with torch.no_grad():
        lengths = torch.tensor([length], dtype=torch.long)
        state_probs = executor.soft_trajectory(init_logits[None, :], op_logits[None, :, :], arg_logits[None, :, :], lengths)[0].cpu()
        final_probs = executor.soft_forward(init_logits[None, :], op_logits[None, :, :], arg_logits[None, :, :], lengths)[0].cpu()

    init_options = top_options_with_rank(init_logits, args.repair_topk)
    op_options = [top_options_with_rank(op_logits[t], args.repair_topk) for t in range(max_steps)]
    arg_options = [top_options_with_rank(arg_logits[t], args.repair_topk) for t in range(max_steps)]
    base_init = int(init_logits.argmax(dim=-1).item())
    base_ops = [int(x) for x in op_logits.argmax(dim=-1).tolist()]
    base_args = [int(x) for x in arg_logits.argmax(dim=-1).tolist()]
    base_prior = float(init_options[base_init][0])
    for step in range(length):
        base_prior += float(op_options[step][base_ops[step]][0])
        base_prior += float(arg_options[step][base_args[step]][0])
    base_answer, base_states = execute_program(base_init, base_ops, base_args, length, max_steps, modulus)
    target_states = [int(x) for x in example.states] + [-100] * (max_steps - len(example.states))
    target_ops = [int(x) for x in example.ops]
    target_args = [int(x) for x in example.args]
    target_answer = int(example.answer)

    candidates: List[Candidate] = []
    seen: set[Tuple[int, Tuple[int, ...], Tuple[int, ...]]] = set()

    def add_candidate(init_value: int, ops: List[int], args_values: List[int], prior: float, edit_records: Sequence[Dict[str, float]]) -> None:
        key = (int(init_value), tuple(int(x) for x in ops[:length]), tuple(int(x) for x in args_values[:length]))
        if key in seen:
            return
        seen.add(key)
        answer, states = execute_program(init_value, ops, args_values, length, max_steps, modulus)
        answer_exact = int(answer) == target_answer
        state_exact = answer_exact and [int(x) for x in states[:length]] == [int(x) for x in target_states[:length]]
        program_exact = (
            int(init_value) == int(example.init_value)
            and [int(x) for x in ops[:length]] == target_ops[:length]
            and [int(x) for x in args_values[:length]] == target_args[:length]
        )
        features = candidate_feature_vector(
            prior,
            base_prior,
            edit_records,
            init_value,
            ops,
            args_values,
            answer,
            states,
            base_answer,
            base_states,
            base_ops,
            base_args,
            state_probs,
            final_probs,
            init_logits,
            op_logits,
            arg_logits,
            length,
            max_steps,
            modulus,
        )
        candidates.append(
            Candidate(
                init_value=int(init_value),
                ops=[int(x) for x in ops],
                args=[int(x) for x in args_values],
                answer=int(answer),
                states=[int(x) for x in states],
                prior=float(prior),
                features=features,
                answer_exact=bool(answer_exact),
                state_exact=bool(state_exact),
                program_exact=bool(program_exact),
                changed=bool(edit_records),
                edit_count=len(edit_records),
            )
        )

    add_candidate(base_init, base_ops, base_args, base_prior, [])
    tail_start = max(0, length - int(args.tail_window))
    tail_slots = list(range(tail_start, length))
    single_arg_edits: List[Tuple[int, int, float, int, Dict[str, float]]] = []

    for step in tail_slots:
        for value, (logp, rank) in op_options[step].items():
            if value == base_ops[step]:
                continue
            old_logp = op_options[step][base_ops[step]][0]
            rec = {
                "kind": "op",
                "pos": float(step),
                "old": float(base_ops[step]),
                "new": float(value),
                "new_rank": float(rank),
                "logp_delta": float(logp - old_logp),
                "margin": top2_margin(op_logits[step]),
                "entropy": entropy_from_logits(op_logits[step]),
            }
            ops = list(base_ops)
            ops[step] = int(value)
            add_candidate(base_init, ops, list(base_args), base_prior + float(logp - old_logp), [rec])
        for value, (logp, rank) in arg_options[step].items():
            if value == base_args[step]:
                continue
            old_logp = arg_options[step][base_args[step]][0]
            rec = {
                "kind": "arg",
                "pos": float(step),
                "old": float(base_args[step]),
                "new": float(value),
                "new_rank": float(rank),
                "logp_delta": float(logp - old_logp),
                "margin": top2_margin(arg_logits[step]),
                "entropy": entropy_from_logits(arg_logits[step]),
            }
            args_values = list(base_args)
            args_values[step] = int(value)
            delta = float(logp - old_logp)
            single_arg_edits.append((step, int(value), delta, int(rank), rec))
            add_candidate(base_init, list(base_ops), args_values, base_prior + delta, [rec])

    for step in tail_slots:
        for op_value, (op_logp, op_rank) in op_options[step].items():
            if op_value == base_ops[step]:
                continue
            for arg_value, (arg_logp, arg_rank) in arg_options[step].items():
                if arg_value == base_args[step]:
                    continue
                old_op_logp = op_options[step][base_ops[step]][0]
                old_arg_logp = arg_options[step][base_args[step]][0]
                rec1 = {
                    "kind": "op",
                    "pos": float(step),
                    "old": float(base_ops[step]),
                    "new": float(op_value),
                    "new_rank": float(op_rank),
                    "logp_delta": float(op_logp - old_op_logp),
                    "margin": top2_margin(op_logits[step]),
                    "entropy": entropy_from_logits(op_logits[step]),
                }
                rec2 = {
                    "kind": "arg",
                    "pos": float(step),
                    "old": float(base_args[step]),
                    "new": float(arg_value),
                    "new_rank": float(arg_rank),
                    "logp_delta": float(arg_logp - old_arg_logp),
                    "margin": top2_margin(arg_logits[step]),
                    "entropy": entropy_from_logits(arg_logits[step]),
                }
                ops = list(base_ops)
                args_values = list(base_args)
                ops[step] = int(op_value)
                args_values[step] = int(arg_value)
                add_candidate(base_init, ops, args_values, base_prior + float(op_logp - old_op_logp) + float(arg_logp - old_arg_logp), [rec1, rec2])

    pair_arg_edits = single_arg_edits
    if int(args.max_pair_arg_candidates) > 0 and len(pair_arg_edits) > int(args.max_pair_arg_candidates):
        pair_arg_edits = sorted(pair_arg_edits, key=lambda row: row[2], reverse=True)[: int(args.max_pair_arg_candidates)]
    for i in range(len(pair_arg_edits)):
        step1, value1, delta1, _rank1, rec1 = pair_arg_edits[i]
        for j in range(i + 1, len(pair_arg_edits)):
            step2, value2, delta2, _rank2, rec2 = pair_arg_edits[j]
            if step1 == step2:
                continue
            args_values = list(base_args)
            args_values[step1] = int(value1)
            args_values[step2] = int(value2)
            add_candidate(base_init, list(base_ops), args_values, base_prior + delta1 + delta2, [rec1, rec2])

    features = torch.tensor([cand.features for cand in candidates], dtype=torch.float32)
    answer_labels = torch.tensor([cand.answer_exact for cand in candidates], dtype=torch.bool)
    state_labels = torch.tensor([cand.state_exact for cand in candidates], dtype=torch.bool)
    program_labels = torch.tensor([cand.program_exact for cand in candidates], dtype=torch.bool)
    priors = torch.tensor([cand.prior for cand in candidates], dtype=torch.float32)
    return CandidateGroup(
        source_run=source_run,
        source_seed=source_seed,
        split=split,
        example_index=example_index,
        length=length,
        answer=target_answer,
        init_value=int(example.init_value),
        ops=target_ops + [-100] * (max_steps - len(target_ops)),
        args=target_args + [-100] * (max_steps - len(target_args)),
        states=target_states,
        base_index=0,
        candidates=candidates,
        features=features,
        answer_labels=answer_labels,
        state_labels=state_labels,
        program_labels=program_labels,
        priors=priors,
        context=context.detach().cpu().float(),
    )


def group_to_plain(group: CandidateGroup) -> Dict[str, Any]:
    return {
        "source_run": group.source_run,
        "source_seed": group.source_seed,
        "split": group.split,
        "example_index": group.example_index,
        "length": group.length,
        "answer": group.answer,
        "init_value": group.init_value,
        "ops": group.ops,
        "args": group.args,
        "states": group.states,
        "base_index": group.base_index,
        "features": group.features,
        "answer_labels": group.answer_labels,
        "state_labels": group.state_labels,
        "program_labels": group.program_labels,
        "priors": group.priors,
        "context": group.context,
        "candidates": [cand.__dict__ for cand in group.candidates],
    }


def plain_to_group(row: Dict[str, Any]) -> CandidateGroup:
    candidates = [Candidate(**cand) for cand in row["candidates"]]
    return CandidateGroup(
        source_run=row["source_run"],
        source_seed=int(row["source_seed"]),
        split=row["split"],
        example_index=int(row["example_index"]),
        length=int(row["length"]),
        answer=int(row["answer"]),
        init_value=int(row["init_value"]),
        ops=[int(x) for x in row["ops"]],
        args=[int(x) for x in row["args"]],
        states=[int(x) for x in row["states"]],
        base_index=int(row["base_index"]),
        candidates=candidates,
        features=row["features"].float(),
        answer_labels=row["answer_labels"].bool(),
        state_labels=row["state_labels"].bool(),
        program_labels=row["program_labels"].bool(),
        priors=row["priors"].float(),
        context=row["context"].float(),
    )


def build_datasets(tokenizer: Any, source: ModuleType, args: argparse.Namespace) -> Dict[str, Any]:
    max_steps = int(args.max_steps)
    seed = int(args.dataset_seed)
    train_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 10, "mixed")
    val_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 20, "mixed")
    standard_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 30, "standard")
    paraphrase_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 40, "paraphrase")
    heldout_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 50, "heldout")
    pair_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 60, "mixed")
    heldout_pair_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 70, "mixed")
    return {
        "train_mixed_L24": train_gen.dataset(args.train_examples, 24, 24),
        "val_mixed_L24": val_gen.dataset(args.val_examples, 24, 24),
        "standard_L24": standard_gen.dataset(args.eval_examples, 24, 24),
        "paraphrase_L24": paraphrase_gen.dataset(args.eval_examples, 24, 24),
        "heldout_L24": heldout_gen.dataset(args.eval_examples, 24, 24),
        "paired_L24": pair_gen.paired_dataset(args.paired_eval_pairs, 24, 24, ["standard", "paraphrase"]),
        "paired_heldout_L24": heldout_pair_gen.paired_dataset(args.paired_eval_pairs, 24, 24, ["standard", "heldout"]),
    }


def source_final_checkpoint(source_run: str) -> Tuple[Path, Path]:
    run_root = SOURCE_CHECKPOINT_ROOT / source_run
    heads = sorted(run_root.glob("stage*_step*/heads.pt"))
    if not heads:
        raise FileNotFoundError(f"no heads.pt found under {run_root}")
    def step_num(path: Path) -> int:
        match = re.search(r"step(\d+)", str(path.parent.name))
        return int(match.group(1)) if match else -1
    head_path = sorted(heads, key=step_num)[-1]
    adapter_dir = head_path.parent / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter dir missing: {adapter_dir}")
    return head_path, adapter_dir


def load_source_model(source: ModuleType, source_run: str, args: argparse.Namespace) -> Tuple[Any, Any, Any, Any, torch.device]:
    head_path, adapter_dir = source_final_checkpoint(source_run)
    ckpt = torch.load(head_path, map_location="cpu")
    model_id = ckpt.get("args", {}).get("model_id", args.model_id)
    dtype = dtype_from_string(args.torch_dtype)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if args.load_in_4bit else None
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"
    common: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": args.device_map if torch.cuda.is_available() else None,
    }
    if quantization_config is not None:
        common["quantization_config"] = quantization_config
    common = {key: value for key, value in common.items() if value is not None}
    log(f"[load] {source_run}: base={model_id} adapter={adapter_dir}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **common)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compiler = source.StructuralLatentCompiler(**ckpt["compiler_config"]).to(device)
    compiler.load_state_dict(ckpt["compiler"])
    compiler.eval()
    executor = source.TransitionExecutor(int(args.modulus), torch.device("cpu"))
    return tokenizer, model, compiler, executor, device


@torch.no_grad()
def collect_groups_for_source(source: ModuleType, source_run: str, args: argparse.Namespace, run_dir: Path) -> List[CandidateGroup]:
    cache_name = args.cache_run_name if args.cache_run_name else args.run_name
    cache_path = CACHE_ROOT / cache_name / f"{source_run}_groups.pt"
    if cache_path.exists() and not args.rebuild_cache:
        log(f"[cache] loading {cache_path}")
        raw = torch.load(cache_path, map_location="cpu")
        return [plain_to_group(row) for row in raw["groups"]]

    tokenizer, model, compiler, executor, device = load_source_model(source, source_run, args)
    datasets = build_datasets(tokenizer, source, args)
    all_groups: List[CandidateGroup] = []
    source_seed = source_seed_from_run(source_run)
    pad_id = int(tokenizer.pad_token_id)
    for split, dataset in datasets.items():
        log(f"[groups] {source_run} {split} n={len(dataset)}")
        for start in range(0, len(dataset), args.eval_batch_size):
            chunk = dataset.examples[start : start + args.eval_batch_size]
            batch = source.collate_examples(chunk, pad_id, int(args.max_steps), int(args.max_length), device)
            hidden = source.forward_hidden(model, batch)
            reg_h = source.gather_registers(hidden, batch["register_pos"])
            init_logits, op_logits, arg_logits = compiler(reg_h)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            answer_context = hidden[rows, batch["answer_pos"]].detach().cpu().float()
            for offset, ex in enumerate(chunk):
                group = make_candidate_group(
                    source_run,
                    source_seed,
                    split,
                    start + offset,
                    ex,
                    init_logits[offset],
                    op_logits[offset],
                    arg_logits[offset],
                    answer_context[offset],
                    executor,
                    args,
                )
                all_groups.append(group)
            log(f"[groups] {source_run} {split} {min(start + len(chunk), len(dataset))}/{len(dataset)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_run": source_run,
            "args": vars(args),
            "feature_names": FEATURE_NAMES,
            "groups": [group_to_plain(group) for group in all_groups],
        },
        cache_path,
    )
    write_json(run_dir / "cache_manifest.json", {"last_written": str(cache_path), "source_run": source_run, "groups": len(all_groups)})
    del model, compiler, tokenizer, executor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_groups


def label_tensor(group: CandidateGroup, label_mode: str) -> torch.Tensor:
    if label_mode == "state":
        return group.state_labels
    if label_mode == "answer":
        return group.answer_labels
    if label_mode == "program":
        return group.program_labels
    raise ValueError(label_mode)


def split_groups(groups: Sequence[CandidateGroup], split: str) -> List[CandidateGroup]:
    return [group for group in groups if group.split == split]


def detail_features_for_group(group: CandidateGroup, tail_detail_window: int, modulus: int) -> torch.Tensor:
    base = group.candidates[group.base_index]
    start = max(0, int(group.length) - int(tail_detail_window))
    rows: List[List[float]] = []
    for cand in group.candidates:
        values: List[float] = []
        for step in range(start, int(group.length)):
            cand_op = int(cand.ops[step])
            base_op = int(base.ops[step])
            values.extend([1.0 if cand_op == idx else 0.0 for idx in range(3)])
            values.extend([1.0 if base_op == idx else 0.0 for idx in range(3)])
            cand_arg = float(cand.args[step]) / max(1.0, float(modulus - 1))
            base_arg = float(base.args[step]) / max(1.0, float(modulus - 1))
            values.extend(
                [
                    cand_arg,
                    base_arg,
                    cand_arg - base_arg,
                    1.0 if cand_op != base_op else 0.0,
                    1.0 if int(cand.args[step]) != int(base.args[step]) else 0.0,
                ]
            )
        rows.append(values)
    return torch.tensor(rows, dtype=torch.float32)


def input_for_group(
    group: CandidateGroup,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    use_context: bool,
    tail_detail_window: int,
    modulus: int,
) -> torch.Tensor:
    x = group.features
    if tail_detail_window > 0:
        x = torch.cat([x, detail_features_for_group(group, tail_detail_window, modulus)], dim=1)
    if use_context:
        ctx = group.context.view(1, -1).expand(x.shape[0], -1)
        x = torch.cat([x, ctx], dim=1)
    x = x.to(device)
    return (x - mean.to(device)) / std.to(device).clamp_min(1e-6)


def fit_feature_stats(groups: Sequence[CandidateGroup], use_context: bool, tail_detail_window: int, modulus: int) -> Tuple[torch.Tensor, torch.Tensor]:
    total = 0
    sum_x: Optional[torch.Tensor] = None
    sumsq_x: Optional[torch.Tensor] = None
    for group in groups:
        x = group.features
        if tail_detail_window > 0:
            x = torch.cat([x, detail_features_for_group(group, tail_detail_window, modulus)], dim=1)
        if use_context:
            ctx = group.context.view(1, -1).expand(x.shape[0], -1)
            x = torch.cat([x, ctx], dim=1)
        x = x.float()
        if sum_x is None:
            sum_x = torch.zeros(x.shape[1], dtype=torch.float64)
            sumsq_x = torch.zeros(x.shape[1], dtype=torch.float64)
        sum_x += x.double().sum(dim=0)
        sumsq_x += x.double().square().sum(dim=0)
        total += x.shape[0]
    if total <= 0 or sum_x is None or sumsq_x is None:
        raise RuntimeError("no candidate features for stats")
    mean = (sum_x / total).float()
    var = (sumsq_x / total - mean.double().square()).clamp_min(1e-8).float()
    std = var.sqrt().clamp_min(1e-4)
    return mean, std


def group_loss(scores: torch.Tensor, labels: torch.Tensor, base_index: int, no_positive_base_weight: float) -> torch.Tensor:
    labels = labels.to(scores.device)
    if bool(labels.any().item()):
        return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[labels], dim=0)
    target = torch.tensor([int(base_index)], dtype=torch.long, device=scores.device)
    return no_positive_base_weight * F.cross_entropy(scores.view(1, -1), target)


def shuffled_labels_for_group(group: CandidateGroup, label_mode: str, rng: random.Random) -> torch.Tensor:
    labels = label_tensor(group, label_mode)
    count = int(labels.sum().item())
    out = torch.zeros_like(labels)
    if count <= 0:
        return out
    indices = list(range(len(labels)))
    rng.shuffle(indices)
    for idx in indices[:count]:
        out[idx] = True
    return out


def focused_train_groups(groups: Sequence[CandidateGroup], label_mode: str, args: argparse.Namespace, seed: int) -> List[CandidateGroup]:
    if args.train_focus == "all":
        return list(groups)
    rng = random.Random(seed + 2718)
    recoverable: List[CandidateGroup] = []
    preserve: List[CandidateGroup] = []
    impossible: List[CandidateGroup] = []
    for group in groups:
        base_correct = bool(group.candidates[group.base_index].answer_exact)
        has_positive = bool(label_tensor(group, label_mode).any().item())
        if (not base_correct) and has_positive:
            recoverable.append(group)
        elif base_correct:
            preserve.append(group)
        else:
            impossible.append(group)
    if args.train_focus == "recoverable_only":
        return recoverable
    if args.train_focus != "recoverable_balanced":
        raise ValueError(args.train_focus)
    rng.shuffle(preserve)
    rng.shuffle(impossible)
    keep_preserve = min(len(preserve), max(1, int(round(len(recoverable) * float(args.preserve_ratio)))))
    keep_impossible = min(len(impossible), int(round(len(recoverable) * float(args.impossible_ratio))))
    out = recoverable + preserve[:keep_preserve] + impossible[:keep_impossible]
    rng.shuffle(out)
    return out


def evaluate_selection(
    groups: Sequence[CandidateGroup],
    selected: Sequence[int],
    label_mode: str,
    prefix: str,
) -> Dict[str, float]:
    n = len(groups)
    if n == 0:
        return {}
    correct = 0
    state_exact = 0
    program_exact = 0
    changed = 0
    damage = 0
    recovered = 0
    base_correct = 0
    positive = 0
    prefix_sum = 0.0
    edit_sum = 0.0
    prefix_buckets: Dict[str, List[int]] = {
        "lt70": [0, 0],
        "70_80": [0, 0],
        "80_90": [0, 0],
        "90_100": [0, 0],
        "100": [0, 0],
    }
    for group, raw_idx in zip(groups, selected):
        idx = int(raw_idx)
        cand = group.candidates[idx]
        base = group.candidates[group.base_index]
        cand_correct = bool(cand.answer_exact)
        base_is_correct = bool(base.answer_exact)
        correct += int(cand_correct)
        state_exact += int(cand.state_exact)
        program_exact += int(cand.program_exact)
        changed += int(cand.changed)
        damage += int(base_is_correct and not cand_correct)
        recovered += int((not base_is_correct) and cand_correct)
        base_correct += int(base_is_correct)
        positive += int(bool(label_tensor(group, label_mode).any().item()))
        prefix_value = correct_prefix_fraction(cand.states, group.states, group.length)
        prefix_sum += prefix_value
        edit_sum += float(cand.edit_count)
        base_prefix = correct_prefix_fraction(base.states, group.states, group.length)
        if base_prefix >= 1.0:
            bucket = "100"
        elif base_prefix >= 0.9:
            bucket = "90_100"
        elif base_prefix >= 0.8:
            bucket = "80_90"
        elif base_prefix >= 0.7:
            bucket = "70_80"
        else:
            bucket = "lt70"
        prefix_buckets[bucket][0] += int(cand_correct)
        prefix_buckets[bucket][1] += 1
    out: Dict[str, float] = {
        f"{prefix}_executor_accuracy": correct / n,
        f"{prefix}_state_exact": state_exact / n,
        f"{prefix}_program_exact": program_exact / n,
        f"{prefix}_changed_fraction": changed / n,
        f"{prefix}_damage_rate": damage / max(1, base_correct),
        f"{prefix}_recovery_rate": recovered / max(1, n - base_correct),
        f"{prefix}_has_positive_fraction": positive / n,
        f"{prefix}_state_prefix_fraction": prefix_sum / n,
        f"{prefix}_avg_edits": edit_sum / n,
    }
    for bucket, (ok, total) in prefix_buckets.items():
        out[f"{prefix}_bucket_{bucket}_n"] = float(total)
        out[f"{prefix}_bucket_{bucket}_accuracy"] = ok / total if total else math.nan
    return out


def pair_metrics(groups: Sequence[CandidateGroup], selected: Sequence[int], prefix: str) -> Dict[str, float]:
    if not groups:
        return {}
    usable = (len(groups) // 2) * 2
    if usable <= 0:
        return {}
    answer_same = both_correct = program_same = state_same = 0
    pair_count = usable // 2
    for i in range(0, usable, 2):
        g0, g1 = groups[i], groups[i + 1]
        c0 = g0.candidates[int(selected[i])]
        c1 = g1.candidates[int(selected[i + 1])]
        length = min(g0.length, g1.length)
        answer_same += int(c0.answer == c1.answer)
        both_correct += int(c0.answer_exact and c1.answer_exact)
        program_same += int(c0.init_value == c1.init_value and c0.ops[:length] == c1.ops[:length] and c0.args[:length] == c1.args[:length])
        state_same += int(c0.states[:length] == c1.states[:length])
    return {
        f"{prefix}_pair_answer_consistency": answer_same / pair_count,
        f"{prefix}_pair_both_correct": both_correct / pair_count,
        f"{prefix}_pair_program_consistency": program_same / pair_count,
        f"{prefix}_pair_state_consistency": state_same / pair_count,
    }


def select_oracle(groups: Sequence[CandidateGroup], label_mode: str) -> List[int]:
    out: List[int] = []
    for group in groups:
        labels = label_tensor(group, label_mode)
        if bool(labels.any().item()):
            masked = group.priors.clone()
            masked[~labels] = -float("inf")
            out.append(int(masked.argmax().item()))
        else:
            out.append(group.base_index)
    return out


def select_base(groups: Sequence[CandidateGroup]) -> List[int]:
    return [group.base_index for group in groups]


def select_prior(groups: Sequence[CandidateGroup]) -> List[int]:
    return [int(group.priors.argmax().item()) for group in groups]


@torch.no_grad()
def select_learned(
    model: RepairCritic,
    groups: Sequence[CandidateGroup],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    use_context: bool,
    tail_detail_window: int,
    modulus: int,
) -> List[int]:
    model.eval()
    out: List[int] = []
    for group in groups:
        x = input_for_group(group, mean, std, device, use_context, tail_detail_window, modulus)
        scores = model(x).detach().cpu()
        out.append(int(scores.argmax().item()))
    return out


def evaluate_arm(
    groups: Sequence[CandidateGroup],
    selected: Sequence[int],
    label_mode: str,
    arm: str,
    split: str,
    critic_seed: Optional[int],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "arm": arm,
        "split": split,
        "label_mode": label_mode,
        "critic_seed": critic_seed if critic_seed is not None else -1,
        "n": len(groups),
    }
    row.update(evaluate_selection(groups, selected, label_mode, arm))
    if split.startswith("paired"):
        row.update(pair_metrics(groups, selected, arm))
    return row


def train_critic(
    train_groups: Sequence[CandidateGroup],
    val_groups: Sequence[CandidateGroup],
    label_mode: str,
    critic_seed: int,
    args: argparse.Namespace,
    run_dir: Path,
    shuffled: bool,
) -> Tuple[RepairCritic, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    rng = random.Random(critic_seed + (100000 if shuffled else 0))
    torch.manual_seed(critic_seed)
    random.seed(critic_seed)
    use_context = bool(args.use_context)
    train_groups = focused_train_groups(train_groups, label_mode, args, critic_seed)
    if not train_groups:
        raise RuntimeError(f"train_focus={args.train_focus} produced no groups for label_mode={label_mode}")
    mean, std = fit_feature_stats(train_groups, use_context, int(args.tail_detail_window), int(args.modulus))
    input_dim = int(mean.numel())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RepairCritic(input_dim, int(args.critic_width), float(args.critic_dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.critic_lr), weight_decay=float(args.critic_weight_decay))
    train_labels: Dict[int, torch.Tensor] = {}
    if shuffled:
        for idx, group in enumerate(train_groups):
            train_labels[idx] = shuffled_labels_for_group(group, label_mode, rng)
    log_rows: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = -float("inf")
    order = list(range(len(train_groups)))
    for epoch in range(1, int(args.critic_epochs) + 1):
        rng.shuffle(order)
        model.train()
        total_loss = 0.0
        positive_groups = 0
        for idx in order:
            group = train_groups[idx]
            labels = train_labels[idx] if shuffled else label_tensor(group, label_mode)
            positive_groups += int(bool(labels.any().item()))
            x = input_for_group(group, mean, std, device, use_context, int(args.tail_detail_window), int(args.modulus))
            scores = model(x)
            loss = group_loss(scores, labels, group.base_index, float(args.no_positive_base_weight))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            opt.step()
            total_loss += float(loss.detach().cpu().item())
        val_selected = select_learned(model, val_groups, mean, std, device, use_context, int(args.tail_detail_window), int(args.modulus))
        val_metrics = evaluate_selection(val_groups, val_selected, label_mode, "val")
        if args.checkpoint_metric == "utility":
            val_score = (
                val_metrics.get("val_executor_accuracy", 0.0)
                + float(args.checkpoint_recovery_bonus) * val_metrics.get("val_recovery_rate", 0.0)
                - float(args.checkpoint_damage_penalty) * val_metrics.get("val_damage_rate", 0.0)
            )
        else:
            val_score = val_metrics.get("val_executor_accuracy", 0.0)
        row: Dict[str, Any] = {
            "label_mode": label_mode,
            "critic_seed": critic_seed,
            "shuffled": shuffled,
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_groups)),
            "train_focus": args.train_focus,
            "focused_train_groups": len(train_groups),
            "positive_train_groups": positive_groups,
            "val_selection_score": val_score,
            "val_executor_accuracy": val_metrics.get("val_executor_accuracy", math.nan),
            "val_state_exact": val_metrics.get("val_state_exact", math.nan),
            "val_changed_fraction": val_metrics.get("val_changed_fraction", math.nan),
            "val_damage_rate": val_metrics.get("val_damage_rate", math.nan),
            "val_recovery_rate": val_metrics.get("val_recovery_rate", math.nan),
        }
        log_rows.append(row)
        if val_score > best_val:
            best_val = float(val_score)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        log(
            f"[critic] mode={label_mode} seed={critic_seed} shuffled={int(shuffled)} "
            f"epoch={epoch} loss={row['train_loss']:.4f} "
            f"val_exec={row['val_executor_accuracy']:.3f} score={val_score:.3f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    tag = f"{label_mode}_seed{critic_seed}_{'shuffled' if shuffled else 'true'}"
    ckpt_dir = CHECKPOINT_ROOT / args.run_name / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "args": vars(args),
            "label_mode": label_mode,
            "critic_seed": critic_seed,
            "shuffled": shuffled,
            "feature_names": FEATURE_NAMES,
            "context_dim": int(train_groups[0].context.numel()) if train_groups and use_context else 0,
            "tail_detail_window": int(args.tail_detail_window),
            "train_focus": args.train_focus,
            "checkpoint_metric": args.checkpoint_metric,
        },
        ckpt_dir / "critic.pt",
    )
    write_csv(run_dir / f"training_{tag}.csv", log_rows)
    return model, mean, std, log_rows


def aggregate_by_source(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    for keys, sub in metrics.groupby(["arm", "split", "critic_seed", "label_mode"], dropna=False):
        arm, split, critic_seed, label_mode = keys
        acc_col = f"{arm}_executor_accuracy"
        if acc_col not in sub.columns:
            continue
        vals = sub[acc_col].astype(float)
        rows.append(
            {
                "arm": arm,
                "split": split,
                "label_mode": label_mode,
                "critic_seed": critic_seed,
                "source_seed_count": len(vals),
                "mean_executor_accuracy": vals.mean(),
                "std_executor_accuracy": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                "min_executor_accuracy": vals.min(),
                "max_executor_accuracy": vals.max(),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    use = df.head(max_rows).copy()
    for col in use.columns:
        if use[col].dtype.kind in "fc":
            if "accuracy" in col or "fraction" in col or "rate" in col or "exact" in col or "consistency" in col:
                use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{100 * float(x):.1f}%")
            else:
                use[col] = use[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    return use.to_markdown(index=False)


def save_figures(run_dir: Path, metrics: pd.DataFrame, training: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        plot = summary[summary["split"].isin(["standard_L24", "heldout_L24", "paired_L24"])].copy()
        if not plot.empty:
            arms = ["base", "critic_answer", "critic_state", "shuffled_answer", "oracle_answer", "oracle_state"]
            arms = [arm for arm in arms if arm in set(plot["arm"])]
            splits = [split for split in ["standard_L24", "heldout_L24", "paired_L24"] if split in set(plot["split"])]
            x = np.arange(len(splits))
            width = min(0.13, 0.85 / max(1, len(arms)))
            plt.figure(figsize=(11.5, 5.8))
            for i, arm in enumerate(arms):
                arm_rows = plot[plot["arm"] == arm]
                if arm_rows.empty:
                    continue
                arm_seed = arm_rows["critic_seed"].min()
                sub = arm_rows[arm_rows["critic_seed"] == arm_seed].set_index("split")
                means = [float(sub.loc[split, "mean_executor_accuracy"]) if split in sub.index else np.nan for split in splits]
                stds = [float(sub.loc[split, "std_executor_accuracy"]) if split in sub.index else 0.0 for split in splits]
                plt.bar(x + (i - (len(arms) - 1) / 2) * width, means, width, yerr=stds, capsize=3, label=arm)
            plt.title("Tail Repair Accuracy by Split")
            plt.ylabel("Executable accuracy")
            plt.xticks(x, splits, rotation=15, ha="right")
            plt.ylim(0, 1.05)
            plt.grid(axis="y", alpha=0.25)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(FIGURES / "accuracy_by_split.png", dpi=180)
            plt.close()

    if not metrics.empty and "source_seed" in metrics.columns:
        rows = metrics[metrics["split"] == "standard_L24"].copy()
        if not rows.empty:
            plt.figure(figsize=(10.5, 5.5))
            arms = [arm for arm in ["base", "critic_answer", "critic_state", "shuffled_answer", "oracle_answer"] if arm in set(rows["arm"])]
            seeds = sorted(rows["source_seed"].dropna().unique().tolist())
            x = np.arange(len(seeds))
            width = min(0.16, 0.82 / max(1, len(arms)))
            for i, arm in enumerate(arms):
                vals: List[float] = []
                for seed in seeds:
                    sub = rows[(rows["arm"] == arm) & (rows["source_seed"] == seed)]
                    acc_col = f"{arm}_executor_accuracy"
                    vals.append(float(sub[acc_col].mean()) if not sub.empty and acc_col in sub else np.nan)
                plt.bar(x + (i - (len(arms) - 1) / 2) * width, vals, width, label=arm)
            plt.title("Standard L24 Accuracy by Source Seed")
            plt.ylabel("Executable accuracy")
            plt.xticks(x, [str(int(seed)) for seed in seeds])
            plt.ylim(0, 1.05)
            plt.grid(axis="y", alpha=0.25)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(FIGURES / "standard_accuracy_by_source_seed.png", dpi=180)
            plt.close()

    if not metrics.empty:
        rows = metrics[metrics["split"] == "standard_L24"].copy()
        if not rows.empty:
            arm_cols = []
            for arm in rows["arm"].unique():
                for metric in ["damage_rate", "recovery_rate", "changed_fraction"]:
                    col = f"{arm}_{metric}"
                    if col in rows:
                        arm_cols.append((arm, metric, col))
            if arm_cols:
                data: List[Dict[str, Any]] = []
                for arm, metric, col in arm_cols:
                    data.append({"arm": arm, "metric": metric, "value": rows[col].astype(float).mean()})
                df = pd.DataFrame(data)
                pivot = df.pivot(index="arm", columns="metric", values="value").fillna(0.0)
                pivot.plot(kind="bar", figsize=(10, 5.5))
                plt.title("Standard L24 Repair Behavior")
                plt.ylabel("Fraction")
                plt.ylim(0, 1.05)
                plt.grid(axis="y", alpha=0.25)
                plt.tight_layout()
                plt.savefig(FIGURES / "repair_behavior_standard.png", dpi=180)
                plt.close()

    if not training.empty:
        plt.figure(figsize=(10.5, 5.5))
        for key, sub in training.groupby(["label_mode", "critic_seed", "shuffled"], dropna=False):
            label_mode, critic_seed, shuffled = key
            label = f"{label_mode}/seed{critic_seed}/{'shuf' if shuffled else 'true'}"
            plt.plot(sub["epoch"], sub["val_executor_accuracy"], marker="o", label=label)
        plt.title("Critic Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Validation executable accuracy")
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(FIGURES / "training_curves.png", dpi=180)
        plt.close()


def write_report(run_dir: Path, args: argparse.Namespace) -> None:
    metrics_path = run_dir / "selection_metrics.csv"
    training_paths = sorted(run_dir.glob("training_*.csv"))
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    training = pd.concat([pd.read_csv(path) for path in training_paths], ignore_index=True, sort=False) if training_paths else pd.DataFrame()
    summary = aggregate_by_source(metrics)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not summary.empty:
        summary.to_csv(REPORTS / "summary_by_source_seed.csv", index=False)
    if not metrics.empty:
        metrics.to_csv(REPORTS / "selection_metrics.csv", index=False)
    if not training.empty:
        training.to_csv(REPORTS / "training_log.csv", index=False)
    save_figures(run_dir, metrics, training, summary)

    standard_summary = summary[summary["split"] == "standard_L24"].copy() if not summary.empty else pd.DataFrame()
    learned_rows = standard_summary[standard_summary["arm"].isin(["critic_answer", "critic_state"])] if not standard_summary.empty else pd.DataFrame()
    best_learned = learned_rows.sort_values("mean_executor_accuracy", ascending=False).head(1) if not learned_rows.empty else pd.DataFrame()
    verdict = "The learned critic did not produce a confirmed stability win."
    if not best_learned.empty:
        base = standard_summary[standard_summary["arm"] == "base"]
        if not base.empty:
            b = base.iloc[0]
            c = best_learned.iloc[0]
            if float(c["mean_executor_accuracy"]) > float(b["mean_executor_accuracy"]) and float(c["std_executor_accuracy"]) < float(b["std_executor_accuracy"]):
                verdict = "The learned critic improved mean accuracy and reduced source-seed spread on the standard split."

    lines = [
        "# Qwen Tail Repair Stability Critic Report",
        "",
        "## Summary",
        "",
        verdict,
        "",
        f"This experiment freezes {len(parse_csv_list(args.source_runs))} independent length-24 modular-program source compiler snapshot(s), enumerates candidate tail edits around each generated program, and trains a critic to select a repair without using target answers or target states at inference. The success criterion is two-part: improve exact executable accuracy and reduce source-seed spread.",
        "",
        "## What Ran",
        "",
        f"- Source compiler snapshots: `{', '.join(parse_csv_list(args.source_runs))}`.",
        f"- Length and modulus: `24` steps, modulus `{args.modulus}`.",
        f"- Candidate search: tail window `{args.tail_window}`, top-k `{args.repair_topk}`, pair-arg cap `{args.max_pair_arg_candidates}`.",
        f"- Train/validation/eval examples per source: `{args.train_examples}` / `{args.val_examples}` / `{args.eval_examples}` plus paired splits.",
        f"- Critic seeds: `{args.critic_seeds}`.",
        f"- Context features: `{'enabled' if args.use_context else 'disabled'}`.",
        f"- Tail detail window: `{args.tail_detail_window}`.",
        f"- Training focus: `{args.train_focus}`.",
        f"- Checkpoint metric: `{args.checkpoint_metric}`.",
        "",
        "## Headline Metrics",
        "",
        "Mean and standard deviation are computed across source compiler seeds.",
        "",
        markdown_table(standard_summary[["arm", "split", "label_mode", "critic_seed", "source_seed_count", "mean_executor_accuracy", "std_executor_accuracy", "min_executor_accuracy", "max_executor_accuracy"]] if not standard_summary.empty else pd.DataFrame()),
        "",
        "## Split Summary",
        "",
        markdown_table(summary[["arm", "split", "label_mode", "critic_seed", "source_seed_count", "mean_executor_accuracy", "std_executor_accuracy", "min_executor_accuracy", "max_executor_accuracy"]] if not summary.empty else pd.DataFrame(), max_rows=120),
        "",
        "![Accuracy by split](figures/accuracy_by_split.png)",
        "",
        "![Standard accuracy by source seed](figures/standard_accuracy_by_source_seed.png)",
        "",
        "![Repair behavior](figures/repair_behavior_standard.png)",
        "",
        "![Training curves](figures/training_curves.png)",
        "",
        "## Interpretation",
        "",
        "The oracle rows show the recoverability ceiling of the candidate set. A learned row is meaningful only if it separates from both the no-repair baseline and the shuffled-label control while reducing seed spread. Damage rate is especially important: a repair critic that frequently breaks already-correct programs is not a stable method, even if it recovers some failures.",
        "",
        "## Artifacts",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Candidate cache and critic checkpoints: `{LARGE_ROOT}`",
        f"- Metrics: `{REPORTS / 'selection_metrics.csv'}`",
        f"- HTML report: `{REPORTS / 'qwen_tail_repair_stability_critic_report.html'}`",
    ]
    report_md = REPORTS / "qwen_tail_repair_stability_critic_report.md"
    report_md.write_text("\n".join(lines) + "\n")
    body = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'><title>Qwen Tail Repair Stability Critic</title>",
            "<style>body{font-family:Inter,Arial,sans-serif;max-width:980px;margin:32px auto;line-height:1.45;color:#1f2933}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccd3db;padding:5px 7px}th{background:#eef2f6}code{background:#f4f6f8;padding:1px 4px;border-radius:4px}img{max-width:100%;border:1px solid #d8dee6;margin:12px 0}</style>",
            "</head><body>",
            markdown_to_html("\n".join(lines)),
            "</body></html>",
        ]
    )
    (REPORTS / "qwen_tail_repair_stability_critic_report.html").write_text(body)


def markdown_to_html(md: str) -> str:
    out: List[str] = []
    in_table = False
    for line in md.splitlines():
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            out.append(f"<p>&bull; {html.escape(line[2:])}</p>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt = line[2:].split("]", 1)[0]
            src = line.split("(", 1)[1][:-1]
            out.append(f"<img alt='{html.escape(alt)}' src='{html.escape(src)}'>")
        elif line.startswith("|"):
            cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
            if set(cells[0]) <= {"-", ":"}:
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table>")
                in_table = True
            out.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>")
        else:
            if in_table:
                out.append("</table>")
                in_table = False
            if line.strip():
                escaped = html.escape(line)
                escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
                out.append(f"<p>{escaped}</p>")
    if in_table:
        out.append("</table>")
    return "\n".join(out)


def run_suite(args: argparse.Namespace) -> None:
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.joinpath(args.run_name).mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.joinpath(args.run_name).mkdir(parents=True, exist_ok=True)
    source = load_source_module()
    t0 = time.time()
    write_json(
        run_dir / "run_config.json",
        {
            "args": vars(args),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "feature_names": FEATURE_NAMES,
        },
    )
    all_groups: List[CandidateGroup] = []
    source_manifest: List[Dict[str, Any]] = []
    for source_run in parse_csv_list(args.source_runs):
        head_path, adapter_dir = source_final_checkpoint(source_run)
        source_manifest.append({"source_run": source_run, "heads": str(head_path), "adapter": str(adapter_dir)})
        groups = collect_groups_for_source(source, source_run, args, run_dir)
        all_groups.extend(groups)
        log(f"[groups] loaded {len(groups)} groups for {source_run}")
    write_json(run_dir / "source_manifest.json", source_manifest)

    group_rows: List[Dict[str, Any]] = []
    for split in sorted({group.split for group in all_groups}):
        sub = split_groups(all_groups, split)
        group_rows.append(
            {
                "split": split,
                "groups": len(sub),
                "avg_candidates": safe_mean([len(group.candidates) for group in sub]),
                "answer_positive_fraction": safe_mean([float(group.answer_labels.any().item()) for group in sub]),
                "state_positive_fraction": safe_mean([float(group.state_labels.any().item()) for group in sub]),
                "base_executor_accuracy": safe_mean([float(group.candidates[group.base_index].answer_exact) for group in sub]),
            }
        )
    write_csv(run_dir / "candidate_group_summary.csv", group_rows)

    train_groups = split_groups(all_groups, "train_mixed_L24")
    val_groups = split_groups(all_groups, "val_mixed_L24")
    eval_splits = ["standard_L24", "paraphrase_L24", "heldout_L24", "paired_L24", "paired_heldout_L24"]
    label_modes = parse_csv_list(args.label_modes)
    critic_seeds = [int(seed) for seed in parse_csv_list(args.critic_seeds)]
    metrics_rows: List[Dict[str, Any]] = []
    training_rows: List[Dict[str, Any]] = []

    for split in eval_splits:
        groups = split_groups(all_groups, split)
        for source_seed in sorted({group.source_seed for group in groups}):
            seed_groups = [group for group in groups if group.source_seed == source_seed]
            for arm, selected in [
                ("base", select_base(seed_groups)),
                ("prior", select_prior(seed_groups)),
            ]:
                row = evaluate_arm(seed_groups, selected, "answer", arm, split, None)
                row["source_seed"] = source_seed
                metrics_rows.append(row)

    for label_mode in label_modes:
        for split in eval_splits:
            groups = split_groups(all_groups, split)
            for source_seed in sorted({group.source_seed for group in groups}):
                seed_groups = [group for group in groups if group.source_seed == source_seed]
                for arm, selected in [
                    (f"oracle_{label_mode}", select_oracle(seed_groups, label_mode)),
                ]:
                    row = evaluate_arm(seed_groups, selected, label_mode, arm, split, None)
                    row["source_seed"] = source_seed
                    metrics_rows.append(row)

        for critic_seed in critic_seeds:
            for shuffled in [False, True]:
                model, mean, std, logs = train_critic(train_groups, val_groups, label_mode, critic_seed, args, run_dir, shuffled)
                training_rows.extend(logs)
                arm_name = f"{'shuffled' if shuffled else 'critic'}_{label_mode}"
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                for split in eval_splits:
                    groups = split_groups(all_groups, split)
                    for source_seed in sorted({group.source_seed for group in groups}):
                        seed_groups = [group for group in groups if group.source_seed == source_seed]
                        selected = select_learned(
                            model,
                            seed_groups,
                            mean,
                            std,
                            device,
                            bool(args.use_context),
                            int(args.tail_detail_window),
                            int(args.modulus),
                        )
                        row = evaluate_arm(seed_groups, selected, label_mode, arm_name, split, critic_seed)
                        row["source_seed"] = source_seed
                        metrics_rows.append(row)
                write_csv(run_dir / "selection_metrics.csv", metrics_rows)
                write_csv(run_dir / "training_log.csv", training_rows)

    write_csv(run_dir / "selection_metrics.csv", metrics_rows)
    write_csv(run_dir / "training_log.csv", training_rows)
    write_json(run_dir / "run_summary.json", {"elapsed_sec": round(time.time() - t0, 3), "groups": len(all_groups), "metrics_rows": len(metrics_rows)})
    write_report(run_dir, args)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="main_tail_repair_critic")
    p.add_argument("--suite", choices=["smoke", "main"], default="main")
    p.add_argument("--source_runs", default="main_expand_copy_seed123,main_expand_copy_seed456,main_expand_copy_seed789")
    p.add_argument("--model_id", default="Qwen/Qwen3-4B")
    p.add_argument("--modulus", type=int, default=97)
    p.add_argument("--max_steps", type=int, default=24)
    p.add_argument("--dataset_seed", type=int, default=4242)
    p.add_argument("--train_examples", type=int, default=160)
    p.add_argument("--val_examples", type=int, default=48)
    p.add_argument("--eval_examples", type=int, default=64)
    p.add_argument("--paired_eval_pairs", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--tail_window", type=int, default=8)
    p.add_argument("--repair_topk", type=int, default=3)
    p.add_argument("--max_pair_arg_candidates", type=int, default=24)
    p.add_argument("--label_modes", default="answer,state")
    p.add_argument("--critic_seeds", default="101,202,303")
    p.add_argument("--critic_epochs", type=int, default=18)
    p.add_argument("--critic_width", type=int, default=256)
    p.add_argument("--critic_dropout", type=float, default=0.05)
    p.add_argument("--critic_lr", type=float, default=1e-3)
    p.add_argument("--critic_weight_decay", type=float, default=1e-4)
    p.add_argument("--no_positive_base_weight", type=float, default=0.25)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--use_context", type=int, default=1)
    p.add_argument("--tail_detail_window", type=int, default=8)
    p.add_argument("--train_focus", choices=["all", "recoverable_only", "recoverable_balanced"], default="all")
    p.add_argument("--preserve_ratio", type=float, default=1.0)
    p.add_argument("--impossible_ratio", type=float, default=0.5)
    p.add_argument("--checkpoint_metric", choices=["accuracy", "utility"], default="accuracy")
    p.add_argument("--checkpoint_recovery_bonus", type=float, default=0.25)
    p.add_argument("--checkpoint_damage_penalty", type=float, default=1.0)
    p.add_argument("--cache_run_name", default="")
    p.add_argument("--torch_dtype", default="bf16")
    p.add_argument("--load_in_4bit", type=int, default=1)
    p.add_argument("--device_map", default="auto")
    p.add_argument("--rebuild_cache", action="store_true")
    return p


def apply_suite_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.suite == "smoke":
        args.run_name = args.run_name if args.run_name != "main_tail_repair_critic" else "smoke_tail_repair_critic"
        args.source_runs = "main_expand_copy_seed789"
        args.train_examples = min(args.train_examples, 24)
        args.val_examples = min(args.val_examples, 12)
        args.eval_examples = min(args.eval_examples, 16)
        args.paired_eval_pairs = min(args.paired_eval_pairs, 8)
        args.tail_window = min(args.tail_window, 4)
        args.repair_topk = min(args.repair_topk, 2)
        args.max_pair_arg_candidates = min(args.max_pair_arg_candidates, 8)
        args.critic_seeds = "101"
        args.critic_epochs = min(args.critic_epochs, 3)
    args.use_context = bool(args.use_context)
    args.load_in_4bit = bool(args.load_in_4bit)
    return args


def main() -> None:
    args = apply_suite_defaults(build_arg_parser().parse_args())
    run_suite(args)


if __name__ == "__main__":
    main()
