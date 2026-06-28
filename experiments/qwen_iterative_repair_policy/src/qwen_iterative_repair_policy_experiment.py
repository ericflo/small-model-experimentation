#!/usr/bin/env python3
"""Train an iterative hidden-program repair policy for Qwen-compiled programs.

The experiment freezes a QLoRA-attached Qwen numeric compiler, builds local
execution traces for compiled hidden programs, and trains a small policy to
apply sparse slot repairs repeatedly. The policy receives the current hidden
program trace at each iteration and decides which init/op/arg slots to rewrite.
Evaluation sweeps K repair iterations without access to target answers or target
states.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from qwen_iterative_repair_policy_core import (
    OP_NAMES,
    DirectAnswerHead,
    ExampleSet,
    ProgramCompiler,
    TextProgramGenerator,
    TransitionExecutor,
    apply_op,
    argmax_execute_with_states,
    collate_examples,
    dtype_from_string,
    ensure_pad_token,
    execute_program_python,
    forward_hidden,
)


ROOT = Path("experiments/qwen_iterative_repair_policy")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_iterative_repair_policy/checkpoints")
DEFAULT_COMPILER_CHECKPOINT = CHECKPOINT_ROOT / "fixed_compiler_step00800"


@dataclass
class CandidateBudgetStage:
    name: str
    topk: int
    max_edits: int
    max_pair_arg_slots: int
    epochs: int


FEATURE_NAMES = [
    "bias",
    "candidate_prior",
    "prior_delta",
    "candidate_prior_per_slot",
    "prior_delta_per_edit",
    "base_prior",
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
    "mean_new_rank",
    "max_new_rank",
    "mean_changed_margin",
    "min_changed_margin",
    "mean_changed_entropy",
    "mean_logp_delta",
    "min_logp_delta",
    "max_logp_delta",
    "base_init_margin",
    "base_op_margin_mean",
    "base_op_margin_min",
    "base_arg_margin_mean",
    "base_arg_margin_min",
    "base_op_entropy_mean",
    "base_arg_entropy_mean",
    "candidate_answer_norm",
    "candidate_final_soft_logp",
    "candidate_state_soft_logp_mean",
    "candidate_state_soft_logp_min",
    "base_final_soft_logp",
    "base_state_soft_logp_mean",
    "answer_equals_base",
    "state_same_fraction_as_base",
    "state_prefix_same_as_base",
    "mean_old_value_norm",
    "mean_new_value_norm",
    "mean_abs_value_delta_norm",
    "arg_new_value_mean_norm",
    "arg_abs_delta_mean_norm",
    "op_new_add_frac",
    "op_new_sub_frac",
    "op_new_mul_frac",
    "candidate_add_frac",
    "candidate_sub_frac",
    "candidate_mul_frac",
    "candidate_arg_mean_norm",
    "candidate_arg_std_norm",
    "base_add_frac",
    "base_sub_frac",
    "base_mul_frac",
]

TRACE_FEATURE_NAMES = [
    "is_global",
    "is_step",
    "active",
    "step_pos_norm",
    "candidate_prior",
    "prior_delta",
    "edit_count",
    "init_value_norm",
    "init_changed",
    "answer_norm",
    "final_soft_logp",
    "base_final_soft_logp",
    "answer_equals_base",
    "op_add",
    "op_sub",
    "op_mul",
    "arg_norm",
    "state_norm",
    "state_soft_logp",
    "base_op_add",
    "base_op_sub",
    "base_op_mul",
    "base_arg_norm",
    "base_state_norm",
    "base_state_soft_logp",
    "op_changed",
    "arg_changed",
    "state_same_as_base",
    "op_logp",
    "arg_logp",
    "op_logp_delta",
    "arg_logp_delta",
    "op_margin",
    "arg_margin",
    "op_entropy",
    "arg_entropy",
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
    trace: List[List[float]]
    state_exact: bool
    program_exact: bool
    changed: bool
    edit_count: int


@dataclass
class CandidateGroup:
    split: str
    length: int
    answer: int
    init_value: int
    ops: List[int]
    args: List[int]
    states: List[int]
    base_index: int
    candidates: List[Candidate]
    features: torch.Tensor
    traces: torch.Tensor
    state_labels: torch.Tensor
    program_labels: torch.Tensor
    priors: torch.Tensor
    init_logits: torch.Tensor
    op_logits: torch.Tensor
    arg_logits: torch.Tensor
    state_probs: torch.Tensor
    final_probs: torch.Tensor

    def has_positive(self) -> bool:
        return bool(self.state_labels.any().item())


class CandidateTraceVerifier(nn.Module):
    def __init__(
        self,
        trace_dim: int,
        feature_dim: int,
        d_model: int,
        layers: int,
        heads: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.trace_proj = nn.Linear(trace_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pos = nn.Parameter(torch.zeros(1, 25, d_model))
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.score = nn.Sequential(nn.LayerNorm(d_model * 2), nn.Linear(d_model * 2, d_model), nn.SiLU(), nn.Linear(d_model, 1))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, trace_x: torch.Tensor, feature_x: torch.Tensor) -> torch.Tensor:
        x = self.trace_proj(trace_x)
        x = x + self.pos[:, : x.shape[1], :]
        encoded = self.encoder(x)
        pooled = encoded[:, 0, :]
        feat = self.feature_proj(feature_x)
        return self.score(torch.cat([pooled, feat], dim=-1)).squeeze(-1)


class SlotRepairEditor(nn.Module):
    def __init__(
        self,
        trace_dim: int,
        modulus: int,
        max_steps: int,
        d_model: int,
        layers: int,
        heads: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.trace_proj = nn.Linear(trace_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pos = nn.Parameter(torch.zeros(1, max_steps + 1, d_model))
        self.init_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, modulus))
        self.op_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, len(OP_NAMES)))
        self.arg_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, modulus))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, trace_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.trace_proj(trace_x)
        x = x + self.pos[:, : x.shape[1], :]
        encoded = self.encoder(x)
        init_logits = self.init_head(encoded[:, 0, :])
        step_h = encoded[:, 1 : 1 + self.max_steps, :]
        op_logits = self.op_head(step_h)
        arg_logits = self.arg_head(step_h)
        return init_logits, op_logits, arg_logits


class SlotEditPolicy(nn.Module):
    def __init__(
        self,
        trace_dim: int,
        modulus: int,
        max_steps: int,
        d_model: int,
        layers: int,
        heads: int,
        ff_mult: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.trace_proj = nn.Linear(trace_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pos = nn.Parameter(torch.zeros(1, max_steps + 1, d_model))
        self.init_value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, modulus))
        self.op_value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, len(OP_NAMES)))
        self.arg_value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, modulus))
        self.init_gate_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))
        self.op_gate_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))
        self.arg_gate_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)
        for head in [self.init_gate_head, self.op_gate_head, self.arg_gate_head]:
            nn.init.constant_(head[-1].bias, -2.0)

    def forward(
        self, trace_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.trace_proj(trace_x)
        x = x + self.pos[:, : x.shape[1], :]
        encoded = self.encoder(x)
        init_h = encoded[:, 0, :]
        step_h = encoded[:, 1 : 1 + self.max_steps, :]
        init_gate_logits = self.init_gate_head(init_h).squeeze(-1)
        op_gate_logits = self.op_gate_head(step_h).squeeze(-1)
        arg_gate_logits = self.arg_gate_head(step_h).squeeze(-1)
        init_logits = self.init_value_head(init_h)
        op_logits = self.op_value_head(step_h)
        arg_logits = self.arg_value_head(step_h)
        return init_gate_logits, op_gate_logits, arg_gate_logits, init_logits, op_logits, arg_logits


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


def finite_float(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=finite_float))


def parse_candidate_budget_stages(spec: str, args: argparse.Namespace) -> List[CandidateBudgetStage]:
    """Parse ``name:topk:max_edits:max_pair_arg_slots:epochs`` stage specs."""
    raw = spec.strip()
    if not raw:
        return [
            CandidateBudgetStage(
                "flat",
                int(args.repair_topk),
                int(args.repair_max_edits),
                int(args.repair_max_pair_arg_slots),
                int(args.verifier_epochs),
            )
        ]
    stages: List[CandidateBudgetStage] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        fields = item.split(":")
        if len(fields) != 5:
            raise ValueError(
                "candidate_curriculum entries must be "
                "name:topk:max_edits:max_pair_arg_slots:epochs"
            )
        name, topk, max_edits, max_pair_arg_slots, epochs = fields
        stage = CandidateBudgetStage(
            name=name,
            topk=int(topk),
            max_edits=int(max_edits),
            max_pair_arg_slots=int(max_pair_arg_slots),
            epochs=int(epochs),
        )
        if stage.topk < 1 or stage.max_edits < 0 or stage.epochs < 1:
            raise ValueError(f"invalid candidate curriculum stage: {item!r}")
        stages.append(stage)
    if not stages:
        raise ValueError("candidate_curriculum produced no stages")
    return stages


def stage_args(args: argparse.Namespace, stage: CandidateBudgetStage) -> argparse.Namespace:
    cloned = SimpleNamespace(**vars(args))
    cloned.repair_topk = int(stage.topk)
    cloned.repair_max_edits = int(stage.max_edits)
    cloned.repair_max_pair_arg_slots = int(stage.max_pair_arg_slots)
    return cloned


def is_full_budget(args: argparse.Namespace, stage: CandidateBudgetStage) -> bool:
    return (
        int(stage.topk) == int(args.repair_topk)
        and int(stage.max_edits) == int(args.repair_max_edits)
        and int(stage.max_pair_arg_slots) == int(args.repair_max_pair_arg_slots)
    )


def top_options_with_rank(logits: torch.Tensor, topk: int) -> Dict[int, Tuple[float, int]]:
    logp = F.log_softmax(logits.float(), dim=-1)
    k = max(1, min(int(topk), int(logp.numel())))
    values, indices = torch.topk(logp, k=k)
    return {int(idx.item()): (float(val.item()), rank) for rank, (val, idx) in enumerate(zip(values, indices))}


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


def correct_prefix_fraction(pred_states: Sequence[int], target_states: Sequence[int], length: int) -> float:
    if length <= 0:
        return 1.0
    prefix = 0
    for t in range(length):
        if int(pred_states[t]) != int(target_states[t]):
            break
        prefix += 1
    return prefix / length


def same_prefix_fraction(states_a: Sequence[int], states_b: Sequence[int], length: int) -> float:
    if length <= 0:
        return 1.0
    prefix = 0
    for t in range(length):
        if int(states_a[t]) != int(states_b[t]):
            break
        prefix += 1
    return prefix / length


def program_matches(
    init_value: int,
    ops: Sequence[int],
    args: Sequence[int],
    target_init: int,
    target_ops: Sequence[int],
    target_args: Sequence[int],
    length: int,
) -> bool:
    if int(init_value) != int(target_init):
        return False
    return list(map(int, ops[:length])) == list(map(int, target_ops[:length])) and list(map(int, args[:length])) == list(
        map(int, target_args[:length])
    )


def state_exact(answer: int, states: Sequence[int], target_answer: int, target_states: Sequence[int], length: int) -> bool:
    return int(answer) == int(target_answer) and list(map(int, states[:length])) == list(map(int, target_states[:length]))


def feature_for_candidate(
    *,
    init_value: int,
    ops: List[int],
    args: List[int],
    answer: int,
    states: List[int],
    prior: float,
    edit_records: Sequence[Dict[str, float]],
    base_prior: float,
    base_answer: int,
    base_states: Sequence[int],
    length: int,
    max_steps: int,
    modulus: int,
    slot_count: int,
    init_margin: float,
    op_margins: Sequence[float],
    arg_margins: Sequence[float],
    op_entropies: Sequence[float],
    arg_entropies: Sequence[float],
    state_log_probs: Sequence[float],
    final_log_probs: Sequence[float],
    base_state_soft_logp_mean: float,
    base_final_soft_logp: float,
) -> List[float]:
    changed = bool(edit_records)
    positions = [float(rec["pos"]) for rec in edit_records if rec["pos"] >= 0]
    new_ranks = [float(rec["new_rank"]) for rec in edit_records]
    margins = [float(rec["margin"]) for rec in edit_records]
    entropies = [float(rec["entropy"]) for rec in edit_records]
    deltas = [float(rec["delta"]) for rec in edit_records]
    old_value_norms = [float(rec["old_value_norm"]) for rec in edit_records]
    new_value_norms = [float(rec["new_value_norm"]) for rec in edit_records]
    abs_value_deltas = [abs(float(rec["new_value_norm"]) - float(rec["old_value_norm"])) for rec in edit_records]
    arg_new_values = [float(rec["new_value_norm"]) for rec in edit_records if rec["kind"] == 2]
    arg_abs_deltas = [abs(float(rec["new_value_norm"]) - float(rec["old_value_norm"])) for rec in edit_records if rec["kind"] == 2]
    op_new_values = [int(rec["new_value"]) for rec in edit_records if rec["kind"] == 1]
    init_edit = sum(1 for rec in edit_records if rec["kind"] == 0)
    op_edit = sum(1 for rec in edit_records if rec["kind"] == 1)
    arg_edit = sum(1 for rec in edit_records if rec["kind"] == 2)
    norm_den = max(1.0, float(length - 1))
    state_same = sum(1 for t in range(length) if int(states[t]) == int(base_states[t])) / max(1, length)
    active_ops = list(map(int, ops[:length]))
    active_args = [float(x) for x in args[:length]]
    add_frac = active_ops.count(0) / max(1, length)
    sub_frac = active_ops.count(1) / max(1, length)
    mul_frac = active_ops.count(2) / max(1, length)
    arg_mean = safe_mean(active_args) / max(1, modulus - 1)
    arg_std = (sum((x - safe_mean(active_args)) ** 2 for x in active_args) / max(1, len(active_args))) ** 0.5 / max(1, modulus - 1)
    # The base program is the candidate with zero edits. For edited candidates,
    # approximate base operation mix by undoing changed operation edits.
    base_like_ops = list(active_ops)
    for rec in edit_records:
        if rec["kind"] == 1 and 0 <= int(rec["pos"]) < length:
            base_like_ops[int(rec["pos"])] = int(rec["old_value"])
    base_add_frac = base_like_ops.count(0) / max(1, length)
    base_sub_frac = base_like_ops.count(1) / max(1, length)
    base_mul_frac = base_like_ops.count(2) / max(1, length)
    return [
        1.0,
        float(prior),
        float(prior - base_prior),
        float(prior / max(1, slot_count)),
        float((prior - base_prior) / max(1, len(edit_records))),
        float(base_prior),
        float(len(edit_records)),
        1.0 if changed else 0.0,
        float(init_edit),
        float(op_edit),
        float(arg_edit),
        float(len(edit_records) / max(1, slot_count)),
        float(safe_min(positions, -1.0) / norm_den) if positions else -1.0,
        float(safe_max(positions, -1.0) / norm_den) if positions else -1.0,
        float(safe_mean(positions, -1.0) / norm_den) if positions else -1.0,
        float((safe_max(positions, 0.0) - safe_min(positions, 0.0)) / norm_den) if positions else 0.0,
        safe_mean(new_ranks),
        safe_max(new_ranks),
        safe_mean(margins),
        safe_min(margins),
        safe_mean(entropies),
        safe_mean(deltas),
        safe_min(deltas),
        safe_max(deltas),
        float(init_margin),
        safe_mean(list(op_margins)[:length]),
        safe_min(list(op_margins)[:length]),
        safe_mean(list(arg_margins)[:length]),
        safe_min(list(arg_margins)[:length]),
        safe_mean(list(op_entropies)[:length]),
        safe_mean(list(arg_entropies)[:length]),
        float(answer / max(1, modulus - 1)),
        float(final_log_probs[int(answer)]),
        safe_mean([float(state_log_probs[t * modulus + int(states[t])]) for t in range(length)]),
        safe_min([float(state_log_probs[t * modulus + int(states[t])]) for t in range(length)]),
        float(base_final_soft_logp),
        float(base_state_soft_logp_mean),
        1.0 if int(answer) == int(base_answer) else 0.0,
        float(state_same),
        same_prefix_fraction(states, base_states, length),
        safe_mean(old_value_norms),
        safe_mean(new_value_norms),
        safe_mean(abs_value_deltas),
        safe_mean(arg_new_values),
        safe_mean(arg_abs_deltas),
        op_new_values.count(0) / max(1, len(op_new_values)),
        op_new_values.count(1) / max(1, len(op_new_values)),
        op_new_values.count(2) / max(1, len(op_new_values)),
        add_frac,
        sub_frac,
        mul_frac,
        arg_mean,
        arg_std,
        base_add_frac,
        base_sub_frac,
        base_mul_frac,
    ]


def trace_for_candidate(
    *,
    init_value: int,
    ops: List[int],
    args: List[int],
    answer: int,
    states: List[int],
    prior: float,
    edit_records: Sequence[Dict[str, float]],
    base_prior: float,
    base_answer: int,
    base_ops: Sequence[int],
    base_args: Sequence[int],
    base_states: Sequence[int],
    length: int,
    max_steps: int,
    modulus: int,
    op_options: Sequence[Dict[int, Tuple[float, int]]],
    arg_options: Sequence[Dict[int, Tuple[float, int]]],
    op_margins: Sequence[float],
    arg_margins: Sequence[float],
    op_entropies: Sequence[float],
    arg_entropies: Sequence[float],
    state_log_probs: Sequence[float],
    final_log_probs: Sequence[float],
    base_final_soft_logp: float,
) -> List[List[float]]:
    edit_by_step: Dict[int, Dict[str, bool]] = {}
    init_changed = 0.0
    for rec in edit_records:
        kind = int(rec["kind"])
        pos = int(rec["pos"])
        if kind == 0:
            init_changed = 1.0
        elif pos >= 0:
            edit_by_step.setdefault(pos, {"op": False, "arg": False})
            if kind == 1:
                edit_by_step[pos]["op"] = True
            if kind == 2:
                edit_by_step[pos]["arg"] = True
    prior_delta = float(prior - base_prior)
    answer_norm = float(answer / max(1, modulus - 1))
    init_norm = float(init_value / max(1, modulus - 1))
    global_values = [
        1.0,
        0.0,
        1.0,
        -1.0,
        float(prior),
        prior_delta,
        float(len(edit_records)),
        init_norm,
        init_changed,
        answer_norm,
        float(final_log_probs[int(answer)]),
        float(base_final_soft_logp),
        1.0 if int(answer) == int(base_answer) else 0.0,
    ]
    rows: List[List[float]] = [global_values + [0.0] * (len(TRACE_FEATURE_NAMES) - len(global_values))]
    norm_den = max(1.0, float(length - 1))
    for t in range(max_steps):
        active = 1.0 if t < length else 0.0
        op = int(ops[t])
        arg = int(args[t])
        state = int(states[t]) if t < len(states) else -100
        base_op = int(base_ops[t])
        base_arg = int(base_args[t])
        base_state = int(base_states[t]) if t < len(base_states) else -100
        op_logp = op_options[t].get(op, (float("-20.0"), -1))[0]
        arg_logp = arg_options[t].get(arg, (float("-20.0"), -1))[0]
        base_op_logp = op_options[t].get(base_op, (float("-20.0"), -1))[0]
        base_arg_logp = arg_options[t].get(base_arg, (float("-20.0"), -1))[0]
        state_soft = float(state_log_probs[t * modulus + state]) if active and state >= 0 else 0.0
        base_state_soft = float(state_log_probs[t * modulus + base_state]) if active and base_state >= 0 else 0.0
        changed = edit_by_step.get(t, {"op": False, "arg": False})
        row = [
            0.0,
            1.0,
            active,
            float(t / norm_den) if active else 0.0,
            float(prior),
            prior_delta,
            float(len(edit_records)),
            init_norm,
            init_changed,
            answer_norm,
            float(final_log_probs[int(answer)]),
            float(base_final_soft_logp),
            1.0 if int(answer) == int(base_answer) else 0.0,
            1.0 if active and op == 0 else 0.0,
            1.0 if active and op == 1 else 0.0,
            1.0 if active and op == 2 else 0.0,
            float(arg / max(1, modulus - 1)) if active else 0.0,
            float(state / max(1, modulus - 1)) if active and state >= 0 else 0.0,
            state_soft,
            1.0 if active and base_op == 0 else 0.0,
            1.0 if active and base_op == 1 else 0.0,
            1.0 if active and base_op == 2 else 0.0,
            float(base_arg / max(1, modulus - 1)) if active else 0.0,
            float(base_state / max(1, modulus - 1)) if active and base_state >= 0 else 0.0,
            base_state_soft,
            1.0 if changed["op"] else 0.0,
            1.0 if changed["arg"] else 0.0,
            1.0 if active and state == base_state else 0.0,
            float(op_logp) if active else 0.0,
            float(arg_logp) if active else 0.0,
            float(op_logp - base_op_logp) if active else 0.0,
            float(arg_logp - base_arg_logp) if active else 0.0,
            float(op_margins[t]) if active else 0.0,
            float(arg_margins[t]) if active else 0.0,
            float(op_entropies[t]) if active else 0.0,
            float(arg_entropies[t]) if active else 0.0,
        ]
        rows.append(row)
    return rows


def build_candidate_group(
    *,
    split: str,
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    state_probs: torch.Tensor,
    final_probs: torch.Tensor,
    length: int,
    target_init: int,
    target_ops: Sequence[int],
    target_args: Sequence[int],
    target_answer: int,
    target_states: Sequence[int],
    modulus: int,
    max_steps: int,
    topk: int,
    max_edits: int,
    max_pair_arg_slots: int,
) -> CandidateGroup:
    length = int(length)
    target_answer = int(target_answer)
    target_states_list = [int(x) for x in target_states[:length]]
    init_options = top_options_with_rank(init_logits, topk)
    op_options = [top_options_with_rank(op_logits[t], topk) for t in range(max_steps)]
    arg_options = [top_options_with_rank(arg_logits[t], topk) for t in range(max_steps)]

    base_init = int(init_logits.argmax(dim=-1).item())
    base_ops = [int(x) for x in op_logits.argmax(dim=-1).tolist()]
    base_args = [int(x) for x in arg_logits.argmax(dim=-1).tolist()]
    slot_count = 1 + 2 * length
    base_prior = init_options[base_init][0]
    for t in range(length):
        base_prior += op_options[t][base_ops[t]][0] + arg_options[t][base_args[t]][0]
    base_answer, base_states = execute_program_python(base_init, base_ops, base_args, length, max_steps, modulus)

    state_log_probs_tensor = state_probs.float().clamp_min(1e-12).log().reshape(-1)
    final_log_probs_tensor = final_probs.float().clamp_min(1e-12).log()
    state_log_probs = [float(x) for x in state_log_probs_tensor.tolist()]
    final_log_probs = [float(x) for x in final_log_probs_tensor.tolist()]
    base_state_soft_logp_mean = safe_mean([state_log_probs[t * modulus + int(base_states[t])] for t in range(length)])
    base_final_soft_logp = final_log_probs[int(base_answer)]

    init_margin = top2_margin(init_logits)
    op_margins = [top2_margin(op_logits[t]) for t in range(max_steps)]
    arg_margins = [top2_margin(arg_logits[t]) for t in range(max_steps)]
    init_entropy = entropy_from_logits(init_logits)
    op_entropies = [entropy_from_logits(op_logits[t]) for t in range(max_steps)]
    arg_entropies = [entropy_from_logits(arg_logits[t]) for t in range(max_steps)]

    def edit(
        kind: int,
        pos: int,
        old_value: int,
        new_value: int,
        old_logp: float,
        new_logp: float,
        new_rank: int,
        margin: float,
        entropy: float,
    ) -> Dict[str, float]:
        value_den = 2.0 if kind == 1 else float(max(1, modulus - 1))
        return {
            "kind": float(kind),
            "pos": float(pos),
            "old_value": float(old_value),
            "new_value": float(new_value),
            "old_value_norm": float(old_value) / value_den,
            "new_value_norm": float(new_value) / value_den,
            "delta": float(new_logp - old_logp),
            "new_rank": float(new_rank),
            "margin": float(margin),
            "entropy": float(entropy),
        }

    candidates: List[Candidate] = []
    seen: set[Tuple[int, Tuple[int, ...], Tuple[int, ...]]] = set()

    def add_candidate(init_value: int, ops: List[int], args_values: List[int], prior: float, edit_records: Sequence[Dict[str, float]]) -> None:
        key = (int(init_value), tuple(map(int, ops[:length])), tuple(map(int, args_values[:length])))
        if key in seen:
            return
        seen.add(key)
        answer, states = execute_program_python(init_value, ops, args_values, length, max_steps, modulus)
        features = feature_for_candidate(
            init_value=init_value,
            ops=ops,
            args=args_values,
            answer=answer,
            states=states,
            prior=prior,
            edit_records=edit_records,
            base_prior=base_prior,
            base_answer=base_answer,
            base_states=base_states,
            length=length,
            max_steps=max_steps,
            modulus=modulus,
            slot_count=slot_count,
            init_margin=init_margin,
            op_margins=op_margins,
            arg_margins=arg_margins,
            op_entropies=op_entropies,
            arg_entropies=arg_entropies,
            state_log_probs=state_log_probs,
            final_log_probs=final_log_probs,
            base_state_soft_logp_mean=base_state_soft_logp_mean,
            base_final_soft_logp=base_final_soft_logp,
        )
        trace = trace_for_candidate(
            init_value=init_value,
            ops=ops,
            args=args_values,
            answer=answer,
            states=states,
            prior=prior,
            edit_records=edit_records,
            base_prior=base_prior,
            base_answer=base_answer,
            base_ops=base_ops,
            base_args=base_args,
            base_states=base_states,
            length=length,
            max_steps=max_steps,
            modulus=modulus,
            op_options=op_options,
            arg_options=arg_options,
            op_margins=op_margins,
            arg_margins=arg_margins,
            op_entropies=op_entropies,
            arg_entropies=arg_entropies,
            state_log_probs=state_log_probs,
            final_log_probs=final_log_probs,
            base_final_soft_logp=base_final_soft_logp,
        )
        candidates.append(
            Candidate(
                init_value=int(init_value),
                ops=list(map(int, ops)),
                args=list(map(int, args_values)),
                answer=int(answer),
                states=list(map(int, states)),
                prior=float(prior),
                features=features,
                trace=trace,
                state_exact=state_exact(answer, states, target_answer, target_states_list, length),
                program_exact=program_matches(init_value, ops, args_values, target_init, target_ops, target_args, length),
                changed=bool(edit_records),
                edit_count=len(edit_records),
            )
        )

    add_candidate(base_init, base_ops, base_args, base_prior, [])

    if max_edits >= 1:
        for value, (logp, rank) in init_options.items():
            if value == base_init:
                continue
            rec = edit(0, -1, base_init, value, init_options[base_init][0], logp, rank, init_margin, init_entropy)
            add_candidate(value, base_ops, base_args, base_prior - init_options[base_init][0] + logp, [rec])

    single_arg_edits: List[Tuple[int, int, float, Dict[str, float]]] = []
    if max_edits >= 1:
        for t in range(length):
            for value, (logp, rank) in op_options[t].items():
                if value == base_ops[t]:
                    continue
                ops = list(base_ops)
                ops[t] = value
                old_logp = op_options[t][base_ops[t]][0]
                rec = edit(1, t, base_ops[t], value, old_logp, logp, rank, op_margins[t], op_entropies[t])
                add_candidate(base_init, ops, base_args, base_prior - old_logp + logp, [rec])
            for value, (logp, rank) in arg_options[t].items():
                if value == base_args[t]:
                    continue
                args_values = list(base_args)
                args_values[t] = value
                old_logp = arg_options[t][base_args[t]][0]
                delta = logp - old_logp
                rec = edit(2, t, base_args[t], value, old_logp, logp, rank, arg_margins[t], arg_entropies[t])
                single_arg_edits.append((t, value, delta, rec))
                add_candidate(base_init, base_ops, args_values, base_prior + delta, [rec])

    if max_edits >= 2:
        for t in range(length):
            for op_value, (op_logp, op_rank) in op_options[t].items():
                if op_value == base_ops[t]:
                    continue
                old_op_logp = op_options[t][base_ops[t]][0]
                op_rec = edit(1, t, base_ops[t], op_value, old_op_logp, op_logp, op_rank, op_margins[t], op_entropies[t])
                for arg_value, (arg_logp, arg_rank) in arg_options[t].items():
                    if arg_value == base_args[t]:
                        continue
                    old_arg_logp = arg_options[t][base_args[t]][0]
                    arg_rec = edit(2, t, base_args[t], arg_value, old_arg_logp, arg_logp, arg_rank, arg_margins[t], arg_entropies[t])
                    ops = list(base_ops)
                    args_values = list(base_args)
                    ops[t] = op_value
                    args_values[t] = arg_value
                    add_candidate(
                        base_init,
                        ops,
                        args_values,
                        base_prior - old_op_logp - old_arg_logp + op_logp + arg_logp,
                        [op_rec, arg_rec],
                    )

        if max_pair_arg_slots > 0:
            allowed_slots = set(range(min(length, max_pair_arg_slots)))
            pair_edits = [item for item in single_arg_edits if item[0] in allowed_slots]
        else:
            pair_edits = single_arg_edits
        for i in range(len(pair_edits)):
            t1, value1, delta1, rec1 = pair_edits[i]
            for j in range(i + 1, len(pair_edits)):
                t2, value2, delta2, rec2 = pair_edits[j]
                if t1 == t2:
                    continue
                args_values = list(base_args)
                args_values[t1] = value1
                args_values[t2] = value2
                add_candidate(base_init, base_ops, args_values, base_prior + delta1 + delta2, [rec1, rec2])

    features = torch.tensor([cand.features for cand in candidates], dtype=torch.float32)
    traces = torch.tensor([cand.trace for cand in candidates], dtype=torch.float32)
    state_labels = torch.tensor([cand.state_exact for cand in candidates], dtype=torch.bool)
    program_labels = torch.tensor([cand.program_exact for cand in candidates], dtype=torch.bool)
    priors = torch.tensor([cand.prior for cand in candidates], dtype=torch.float32)
    return CandidateGroup(
        split=split,
        length=length,
        answer=target_answer,
        init_value=int(target_init),
        ops=list(map(int, target_ops)),
        args=list(map(int, target_args)),
        states=list(map(int, target_states)),
        base_index=0,
        candidates=candidates,
        features=features,
        traces=traces,
        state_labels=state_labels,
        program_labels=program_labels,
        priors=priors,
        init_logits=init_logits.detach().clone().float(),
        op_logits=op_logits.detach().clone().float(),
        arg_logits=arg_logits.detach().clone().float(),
        state_probs=state_probs.detach().clone().float(),
        final_probs=final_probs.detach().clone().float(),
    )


def load_fixed_compiler(args: argparse.Namespace, device: torch.device) -> Tuple[Any, nn.Module, Optional[DirectAnswerHead], ProgramCompiler, Dict[str, Any]]:
    checkpoint_dir = Path(args.compiler_checkpoint)
    heads_path = checkpoint_dir / "heads.pt"
    adapter_dir = checkpoint_dir / "adapter"
    if not heads_path.exists():
        raise FileNotFoundError(heads_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(adapter_dir)
    heads = torch.load(heads_path, map_location=device)
    run_args = dict(heads["args"])
    run_args["eval_batch_size"] = args.qwen_batch_size
    run_args["repair_topk"] = args.repair_topk
    run_args["repair_max_edits"] = args.repair_max_edits
    run_args["repair_max_pair_arg_slots"] = args.repair_max_pair_arg_slots
    run_args["max_length"] = args.max_length

    tokenizer = AutoTokenizer.from_pretrained(run_args["model_id"], trust_remote_code=True, use_fast=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"
    dtype = dtype_from_string(run_args.get("torch_dtype", "bf16"))
    quantization_config = None
    if run_args.get("load_in_4bit", False):
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
        "device_map": run_args.get("device_map", "auto") if torch.cuda.is_available() else None,
    }
    if quantization_config is not None:
        common["quantization_config"] = quantization_config
    common = {k: v for k, v in common.items() if v is not None}
    print(f"[load] {run_args['model_id']} + {adapter_dir}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(run_args["model_id"], **common)
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()

    direct = None
    if heads.get("direct") is not None:
        direct = DirectAnswerHead(heads["hidden_dim"], run_args["modulus"], run_args["head_width"]).to(device)
        direct.load_state_dict(heads["direct"])
        direct.eval()
    compiler = ProgramCompiler(
        heads["hidden_dim"],
        run_args["modulus"],
        run_args["head_width"],
        run_args["max_steps"],
        run_args["rank_temperature"],
        run_args["arg_reader_mode"],
        run_args["arg_window"],
        run_args["arg_distance_temperature"],
    ).to(device)
    compiler.load_state_dict(heads["compiler"])
    compiler.eval()
    return tokenizer, model, direct, compiler, run_args


def clean_render_dataset(
    tokenizer: Any,
    run_args: Dict[str, Any],
    n_examples: int,
    length: int,
    seed: int,
    template_modes: Sequence[str],
) -> ExampleSet:
    gen = TextProgramGenerator(tokenizer, run_args["modulus"], run_args["max_steps"], seed, "standard")
    examples = []
    modes = list(template_modes)
    if not modes:
        raise ValueError("at least one template mode is required")
    for i in range(n_examples):
        spec = gen.make_spec(length, length)
        examples.append(gen.render_spec(spec, modes[i % len(modes)]))
    return ExampleSet(examples)


def make_datasets(tokenizer: Any, run_args: Dict[str, Any], args: argparse.Namespace) -> Dict[str, ExampleSet]:
    datasets: Dict[str, ExampleSet] = {}
    train_modes = [x.strip() for x in args.train_template_modes.split(",") if x.strip()]
    datasets["train_len24"] = clean_render_dataset(
        tokenizer, run_args, args.train_examples, args.train_length, args.seed + 11, train_modes
    )
    datasets["val_len24"] = clean_render_dataset(
        tokenizer, run_args, args.val_examples, args.eval_length, args.seed + 29, train_modes
    )
    if int(args.val_pairs) > 0:
        gen_val_pair = TextProgramGenerator(tokenizer, run_args["modulus"], run_args["max_steps"], args.seed + 503, "mixed")
        datasets[f"val_paired_len{args.eval_length}"] = gen_val_pair.paired_dataset(
            args.val_pairs, args.eval_length, args.eval_length, ["standard", "paraphrase"]
        )
    for mode_index, mode in enumerate(["standard", "paraphrase"]):
        gen = TextProgramGenerator(tokenizer, run_args["modulus"], run_args["max_steps"], args.eval_seed + 101 * mode_index, mode)
        datasets[f"fresh_{mode}_len{args.eval_length}"] = gen.dataset(args.eval_examples, args.eval_length, args.eval_length)
    gen_pair = TextProgramGenerator(tokenizer, run_args["modulus"], run_args["max_steps"], args.eval_seed + 503, "mixed")
    datasets[f"fresh_paired_len{args.eval_length}"] = gen_pair.paired_dataset(
        args.eval_pairs, args.eval_length, args.eval_length, ["standard", "paraphrase"]
    )
    return datasets


@torch.no_grad()
def build_groups_for_dataset(
    split: str,
    dataset: ExampleSet,
    tokenizer: Any,
    model: nn.Module,
    compiler: ProgramCompiler,
    executor: TransitionExecutor,
    run_args: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> List[CandidateGroup]:
    groups: List[CandidateGroup] = []
    pad_id = int(tokenizer.pad_token_id)
    for start in range(0, len(dataset), args.qwen_batch_size):
        chunk = dataset.examples[start : start + args.qwen_batch_size]
        batch = collate_examples(chunk, pad_id, run_args["max_steps"], args.max_length, device)
        hidden = forward_hidden(model, batch)
        init_logits, op_logits, arg_logits = compiler(
            hidden,
            batch["hidden_mask"],
            batch["num_values"],
            batch["op_values"],
            return_scores=False,
        )
        state_probs = executor.soft_trajectory(init_logits, op_logits, arg_logits, batch["lengths"])
        gather_idx = (batch["lengths"].clamp_min(1) - 1).view(-1, 1, 1).expand(-1, 1, state_probs.shape[-1])
        final_probs = state_probs.gather(1, gather_idx).squeeze(1).clamp_min(1e-12)
        for i in range(len(chunk)):
            groups.append(
                build_candidate_group(
                    split=split,
                    init_logits=init_logits[i].detach().cpu(),
                    op_logits=op_logits[i].detach().cpu(),
                    arg_logits=arg_logits[i].detach().cpu(),
                    state_probs=state_probs[i].detach().cpu(),
                    final_probs=final_probs[i].detach().cpu(),
                    length=int(batch["lengths"][i].item()),
                    target_init=int(batch["init_value"][i].item()),
                    target_ops=[int(x) for x in batch["ops"][i].detach().cpu().tolist()],
                    target_args=[int(x) for x in batch["args"][i].detach().cpu().tolist()],
                    target_answer=int(batch["answer"][i].item()),
                    target_states=[int(x) for x in batch["states"][i].detach().cpu().tolist()],
                    modulus=int(run_args["modulus"]),
                    max_steps=int(run_args["max_steps"]),
                    topk=int(args.repair_topk),
                    max_edits=int(args.repair_max_edits),
                    max_pair_arg_slots=int(args.repair_max_pair_arg_slots),
                )
            )
        print(f"[candidates] {split} {min(start + len(chunk), len(dataset))}/{len(dataset)}", flush=True)
    return groups


def fit_feature_stats(groups: Sequence[CandidateGroup]) -> Tuple[torch.Tensor, torch.Tensor]:
    all_features = torch.cat([group.features for group in groups], dim=0)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0).clamp_min(1e-5)
    std[0] = 1.0
    mean[0] = 0.0
    return mean, std


def fit_trace_stats(groups: Sequence[CandidateGroup]) -> Tuple[torch.Tensor, torch.Tensor]:
    all_traces = torch.cat([group.traces.reshape(-1, group.traces.shape[-1]) for group in groups], dim=0)
    mean = all_traces.mean(dim=0)
    std = all_traces.std(dim=0).clamp_min(1e-5)
    return mean, std


def normalized_features(group: CandidateGroup, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> torch.Tensor:
    return ((group.features.to(device) - mean.to(device)) / std.to(device)).float()


def normalized_traces(group: CandidateGroup, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> torch.Tensor:
    return ((group.traces.to(device) - mean.to(device)) / std.to(device)).float()


def group_loss(scores: torch.Tensor, labels: torch.Tensor, base_index: int, no_positive_base_weight: float) -> torch.Tensor:
    labels = labels.to(scores.device)
    if labels.any():
        return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[labels], dim=0)
    return no_positive_base_weight * F.cross_entropy(scores.view(1, -1), torch.tensor([base_index], dtype=torch.long, device=scores.device))


def group_loss_weight(group: CandidateGroup, args: argparse.Namespace) -> float:
    if bool(group.state_labels[group.base_index].item()):
        return float(args.base_positive_group_weight)
    if group.has_positive():
        return float(args.repairable_group_weight)
    return float(args.no_positive_group_weight)


@torch.no_grad()
def select_index_with_model(
    verifier: CandidateTraceVerifier,
    group: CandidateGroup,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    trace_mean: torch.Tensor,
    trace_std: torch.Tensor,
    device: torch.device,
) -> Tuple[int, torch.Tensor]:
    scores = verifier(
        normalized_traces(group, trace_mean, trace_std, device),
        normalized_features(group, feature_mean, feature_std, device),
    ).detach().cpu()
    return int(scores.argmax().item()), scores


def train_verifier(
    train_groups: Sequence[CandidateGroup],
    val_groups: Sequence[CandidateGroup],
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
    train_stages: Optional[Sequence[Tuple[CandidateBudgetStage, Sequence[CandidateGroup]]]] = None,
) -> Tuple[CandidateTraceVerifier, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    feature_mean, feature_std = fit_feature_stats(train_groups)
    trace_mean, trace_std = fit_trace_stats(train_groups)
    verifier = CandidateTraceVerifier(
        len(TRACE_FEATURE_NAMES),
        len(FEATURE_NAMES),
        args.trace_d_model,
        args.trace_layers,
        args.trace_heads,
        args.trace_ff_mult,
        args.verifier_dropout,
    ).to(device)
    opt = torch.optim.AdamW(verifier.parameters(), lr=args.verifier_lr, weight_decay=args.verifier_weight_decay)
    rng = random.Random(args.seed + 700)
    rows: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = -1.0
    best_epoch = 0
    if train_stages is None:
        train_stages = [
            (
                CandidateBudgetStage(
                    "flat",
                    int(args.repair_topk),
                    int(args.repair_max_edits),
                    int(args.repair_max_pair_arg_slots),
                    int(args.verifier_epochs),
                ),
                train_groups,
            )
        ]
    epoch = 0
    for stage, stage_groups in train_stages:
        for stage_epoch in range(1, int(stage.epochs) + 1):
            epoch += 1
            verifier.train()
            order = list(range(len(stage_groups)))
            rng.shuffle(order)
            total_loss = 0.0
            trained = 0
            for idx in order:
                group = stage_groups[idx]
                trace_x = normalized_traces(group, trace_mean, trace_std, device)
                feature_x = normalized_features(group, feature_mean, feature_std, device)
                labels = group.state_labels.to(device)
                scores = verifier(trace_x, feature_x)
                loss = group_loss(scores, labels, group.base_index, args.no_positive_base_weight)
                loss = group_loss_weight(group, args) * loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(verifier.parameters(), args.grad_clip)
                opt.step()
                total_loss += float(loss.detach().cpu())
                trained += 1
            val_metrics = evaluate_groups(
                val_groups,
                verifier,
                feature_mean,
                feature_std,
                trace_mean,
                trace_std,
                device,
                split_name=str(getattr(args, "selection_split", "val_len24")),
            )
            row = {
                "epoch": epoch,
                "stage": stage.name,
                "stage_epoch": stage_epoch,
                "stage_topk": stage.topk,
                "stage_max_edits": stage.max_edits,
                "stage_max_pair_arg_slots": stage.max_pair_arg_slots,
                "train_loss": total_loss / max(1, trained),
                **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
            }
            rows.append(row)
            print(
                f"[verifier] epoch={epoch} stage={stage.name} loss={row['train_loss']:.4f} "
                f"val_learned={100.0 * val_metrics['learned_executor_accuracy']:.1f}% "
                f"val_oracle={100.0 * val_metrics['oracle_executor_accuracy']:.1f}%",
                flush=True,
            )
            selection_value = float(val_metrics.get("learned_pair_both_correct", val_metrics["learned_executor_accuracy"]))
            if selection_value > best_val:
                best_val = selection_value
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in verifier.state_dict().items()}
    if best_state is not None:
        verifier.load_state_dict(best_state)
    ckpt_path = CHECKPOINT_ROOT / run_dir.name / "candidate_trace_verifier.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": verifier.state_dict(),
            "feature_names": FEATURE_NAMES,
            "trace_feature_names": TRACE_FEATURE_NAMES,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "trace_mean": trace_mean,
            "trace_std": trace_std,
            "best_epoch": best_epoch,
            "best_val_learned_executor_accuracy": best_val,
            "args": vars(args),
        },
        ckpt_path,
    )
    return verifier, feature_mean, feature_std, trace_mean, trace_std, rows


def oracle_candidate_index(group: CandidateGroup) -> int:
    if group.state_labels.any():
        positive = torch.where(group.state_labels)[0]
        return int(positive[group.priors[positive].argmax()].item())
    return group.base_index


def base_trace(group: CandidateGroup) -> torch.Tensor:
    return group.traces[group.base_index]


def fit_editor_trace_stats(groups: Sequence[CandidateGroup]) -> Tuple[torch.Tensor, torch.Tensor]:
    traces = torch.stack([base_trace(group) for group in groups], dim=0)
    flat = traces.reshape(-1, traces.shape[-1])
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp_min(1e-5)
    return mean, std


def normalized_editor_trace(group: CandidateGroup, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = base_trace(group).to(device)
    return ((x - mean.to(device)) / std.to(device)).float().unsqueeze(0)


def target_program_for_group(group: CandidateGroup, target_mode: str) -> Tuple[int, List[int], List[int], str]:
    if target_mode == "gold":
        return group.init_value, list(group.ops), list(group.args), "gold"
    if group.state_labels.any():
        cand = group.candidates[oracle_candidate_index(group)]
        return cand.init_value, list(cand.ops), list(cand.args), "oracle"
    if target_mode == "oracle_or_gold":
        return group.init_value, list(group.ops), list(group.args), "gold_fallback"
    if target_mode == "oracle_or_base":
        cand = group.candidates[group.base_index]
        return cand.init_value, list(cand.ops), list(cand.args), "base_fallback"
    raise ValueError(f"unknown editor_target_mode {target_mode!r}")


def add_base_copy_bias(
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    group: CandidateGroup,
    bias: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if bias <= 0:
        return init_logits, op_logits, arg_logits
    base = group.candidates[group.base_index]
    init_logits = init_logits.clone()
    op_logits = op_logits.clone()
    arg_logits = arg_logits.clone()
    init_logits[0, int(base.init_value)] += float(bias)
    for t in range(group.length):
        op_logits[0, t, int(base.ops[t])] += float(bias)
        arg_logits[0, t, int(base.args[t])] += float(bias)
    return init_logits, op_logits, arg_logits


def slot_loss_for_group(
    editor: SlotRepairEditor,
    group: CandidateGroup,
    trace_mean: torch.Tensor,
    trace_std: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    init_target, op_targets, arg_targets, source = target_program_for_group(group, args.editor_target_mode)
    init_logits, op_logits, arg_logits = editor(normalized_editor_trace(group, trace_mean, trace_std, device))
    init_logits, op_logits, arg_logits = add_base_copy_bias(init_logits, op_logits, arg_logits, group, args.base_copy_logit_bias)
    base = group.candidates[group.base_index]
    active_len = int(group.length)
    op_t = torch.tensor(op_targets[:active_len], dtype=torch.long, device=device)
    arg_t = torch.tensor(arg_targets[:active_len], dtype=torch.long, device=device)
    init_t = torch.tensor([init_target], dtype=torch.long, device=device)

    init_weight = args.changed_slot_weight if int(init_target) != int(base.init_value) else 1.0
    init_loss = F.cross_entropy(init_logits, init_t) * init_weight

    if active_len > 0:
        op_loss_each = F.cross_entropy(op_logits[0, :active_len, :], op_t, reduction="none")
        arg_loss_each = F.cross_entropy(arg_logits[0, :active_len, :], arg_t, reduction="none")
        op_weights = torch.tensor(
            [args.changed_slot_weight if int(op_targets[t]) != int(base.ops[t]) else 1.0 for t in range(active_len)],
            dtype=op_loss_each.dtype,
            device=device,
        )
        arg_weights = torch.tensor(
            [args.changed_slot_weight if int(arg_targets[t]) != int(base.args[t]) else 1.0 for t in range(active_len)],
            dtype=arg_loss_each.dtype,
            device=device,
        )
        op_loss = (op_loss_each * op_weights).mean()
        arg_loss = (arg_loss_each * arg_weights).mean()
        changed_slots = int(init_weight > 1.0) + int((op_weights > 1.0).sum().item()) + int((arg_weights > 1.0).sum().item())
    else:
        op_loss = op_logits.sum() * 0.0
        arg_loss = arg_logits.sum() * 0.0
        changed_slots = int(init_weight > 1.0)
    loss = args.init_loss_weight * init_loss + args.op_loss_weight * op_loss + args.arg_loss_weight * arg_loss
    return loss, {
        "init_loss": float(init_loss.detach().cpu()),
        "op_loss": float(op_loss.detach().cpu()),
        "arg_loss": float(arg_loss.detach().cpu()),
        "target_changed_slots": float(changed_slots),
        "target_is_oracle": 1.0 if source == "oracle" else 0.0,
        "target_is_gold_fallback": 1.0 if source == "gold_fallback" else 0.0,
        "target_is_base_fallback": 1.0 if source == "base_fallback" else 0.0,
    }


@torch.no_grad()
def editor_prediction(
    editor: SlotRepairEditor,
    group: CandidateGroup,
    trace_mean: torch.Tensor,
    trace_std: torch.Tensor,
    device: torch.device,
    base_copy_logit_bias: float,
) -> Dict[str, Any]:
    editor.eval()
    init_logits, op_logits, arg_logits = editor(normalized_editor_trace(group, trace_mean, trace_std, device))
    init_logits, op_logits, arg_logits = add_base_copy_bias(init_logits, op_logits, arg_logits, group, base_copy_logit_bias)
    init_value = int(init_logits.argmax(dim=-1).item())
    ops = [int(x) for x in op_logits[0].argmax(dim=-1).detach().cpu().tolist()]
    args_values = [int(x) for x in arg_logits[0].argmax(dim=-1).detach().cpu().tolist()]
    answer, states = execute_program_python(init_value, ops, args_values, group.length, len(ops), 97)
    return {
        "init_value": init_value,
        "ops": ops,
        "args": args_values,
        "answer": int(answer),
        "states": list(map(int, states)),
    }


def candidate_to_prediction(cand: Candidate) -> Dict[str, Any]:
    return {
        "init_value": cand.init_value,
        "ops": list(cand.ops),
        "args": list(cand.args),
        "answer": cand.answer,
        "states": list(cand.states),
    }


def prediction_program_exact(pred: Dict[str, Any], group: CandidateGroup) -> bool:
    return (
        int(pred["init_value"]) == int(group.init_value)
        and list(map(int, pred["ops"][: group.length])) == list(map(int, group.ops[: group.length]))
        and list(map(int, pred["args"][: group.length])) == list(map(int, group.args[: group.length]))
    )


def prediction_state_exact(pred: Dict[str, Any], group: CandidateGroup) -> bool:
    return int(pred["answer"]) == int(group.answer) and list(map(int, pred["states"][: group.length])) == list(
        map(int, group.states[: group.length])
    )


def prediction_prefix_fraction(pred: Dict[str, Any], group: CandidateGroup) -> float:
    return correct_prefix_fraction(pred["states"], group.states, group.length)


def prediction_edit_count(pred: Dict[str, Any], group: CandidateGroup) -> int:
    base = group.candidates[group.base_index]
    edits = int(int(pred["init_value"]) != int(base.init_value))
    for t in range(group.length):
        edits += int(int(pred["ops"][t]) != int(base.ops[t]))
        edits += int(int(pred["args"][t]) != int(base.args[t]))
    return edits


def prediction_in_candidate_set(pred: Dict[str, Any], group: CandidateGroup) -> bool:
    key = (int(pred["init_value"]), tuple(map(int, pred["ops"][: group.length])), tuple(map(int, pred["args"][: group.length])))
    for cand in group.candidates:
        cand_key = (int(cand.init_value), tuple(map(int, cand.ops[: group.length])), tuple(map(int, cand.args[: group.length])))
        if key == cand_key:
            return True
    return False


def summarize_predictions(groups: Sequence[CandidateGroup], preds: Sequence[Dict[str, Any]], prefix: str) -> Dict[str, float]:
    n = len(groups)
    exec_correct = 0
    program_correct = 0
    prefix_sum = 0.0
    changed = 0
    edit_sum = 0.0
    in_candidates = 0
    for group, pred in zip(groups, preds):
        exec_correct += int(prediction_state_exact(pred, group))
        program_correct += int(prediction_program_exact(pred, group))
        prefix_sum += prediction_prefix_fraction(pred, group)
        edits = prediction_edit_count(pred, group)
        changed += int(edits > 0)
        edit_sum += edits
        in_candidates += int(prediction_in_candidate_set(pred, group))
    return {
        f"{prefix}_executor_accuracy": exec_correct / max(1, n),
        f"{prefix}_program_exact": program_correct / max(1, n),
        f"{prefix}_state_prefix_fraction": prefix_sum / max(1, n),
        f"{prefix}_changed_fraction": changed / max(1, n),
        f"{prefix}_avg_edits": edit_sum / max(1, n),
        f"{prefix}_in_candidate_set_fraction": in_candidates / max(1, n),
    }


def prediction_pair_metrics(groups: Sequence[CandidateGroup], preds: Sequence[Dict[str, Any]], prefix: str) -> Dict[str, float]:
    usable = (len(groups) // 2) * 2
    if usable < 2:
        return {}
    pair_count = usable // 2
    answer_same = 0
    both_correct = 0
    program_same = 0
    state_same = 0
    for i in range(0, usable, 2):
        g0 = groups[i]
        p0 = preds[i]
        p1 = preds[i + 1]
        length = g0.length
        answer_same += int(int(p0["answer"]) == int(p1["answer"]))
        both_correct += int(prediction_state_exact(p0, groups[i]) and prediction_state_exact(p1, groups[i + 1]))
        same_program = (
            int(p0["init_value"]) == int(p1["init_value"])
            and list(map(int, p0["ops"][:length])) == list(map(int, p1["ops"][:length]))
            and list(map(int, p0["args"][:length])) == list(map(int, p1["args"][:length]))
        )
        program_same += int(same_program)
        state_same += int(list(map(int, p0["states"][:length])) == list(map(int, p1["states"][:length])))
    return {
        f"{prefix}_pair_answer_consistency": answer_same / pair_count,
        f"{prefix}_pair_both_correct": both_correct / pair_count,
        f"{prefix}_pair_program_consistency": program_same / pair_count,
        f"{prefix}_pair_state_consistency": state_same / pair_count,
    }


@torch.no_grad()
def evaluate_editor_groups(
    groups: Sequence[CandidateGroup],
    editor: Optional[SlotRepairEditor],
    trace_mean: Optional[torch.Tensor],
    trace_std: Optional[torch.Tensor],
    device: torch.device,
    split_name: str,
    base_copy_logit_bias: float = 5.0,
) -> Dict[str, float]:
    base_preds = [candidate_to_prediction(group.candidates[group.base_index]) for group in groups]
    oracle_preds = [candidate_to_prediction(group.candidates[oracle_candidate_index(group)]) for group in groups]
    out: Dict[str, float] = {
        "n": float(len(groups)),
        "oracle_found_fraction": sum(group.has_positive() for group in groups) / max(1, len(groups)),
        "avg_candidates": safe_mean([float(len(group.candidates)) for group in groups]),
        "avg_positive_candidates": safe_mean([float(group.state_labels.sum().item()) for group in groups]),
    }
    out.update(summarize_predictions(groups, base_preds, "base"))
    out.update(summarize_predictions(groups, oracle_preds, "oracle"))
    if editor is not None and trace_mean is not None and trace_std is not None:
        editor_preds = [editor_prediction(editor, group, trace_mean, trace_std, device, base_copy_logit_bias) for group in groups]
        out.update(summarize_predictions(groups, editor_preds, "editor"))
        denom = out["oracle_executor_accuracy"] - out["base_executor_accuracy"]
        out["editor_oracle_gap_recovered"] = (
            (out["editor_executor_accuracy"] - out["base_executor_accuracy"]) / denom if abs(denom) > 1e-9 else math.nan
        )
        if "paired" in split_name:
            out.update(prediction_pair_metrics(groups, base_preds, "base"))
            out.update(prediction_pair_metrics(groups, oracle_preds, "oracle"))
            out.update(prediction_pair_metrics(groups, editor_preds, "editor"))
    return out


def train_editor(
    train_groups: Sequence[CandidateGroup],
    val_groups: Sequence[CandidateGroup],
    args: argparse.Namespace,
    run_dir: Path,
    run_args: Dict[str, Any],
    device: torch.device,
) -> Tuple[SlotRepairEditor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    trace_mean, trace_std = fit_editor_trace_stats(train_groups)
    editor = SlotRepairEditor(
        len(TRACE_FEATURE_NAMES),
        int(run_args["modulus"]),
        int(run_args["max_steps"]),
        args.editor_d_model,
        args.editor_layers,
        args.editor_heads,
        args.editor_ff_mult,
        args.editor_dropout,
    ).to(device)
    opt = torch.optim.AdamW(editor.parameters(), lr=args.editor_lr, weight_decay=args.editor_weight_decay)
    rng = random.Random(args.seed + 1100)
    rows: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = -1.0
    best_epoch = 0
    for epoch in range(1, args.editor_epochs + 1):
        editor.train()
        order = list(range(len(train_groups)))
        rng.shuffle(order)
        totals: Dict[str, float] = {
            "loss": 0.0,
            "init_loss": 0.0,
            "op_loss": 0.0,
            "arg_loss": 0.0,
            "target_changed_slots": 0.0,
            "target_is_oracle": 0.0,
            "target_is_gold_fallback": 0.0,
            "target_is_base_fallback": 0.0,
        }
        trained = 0
        for idx in order:
            group = train_groups[idx]
            loss, parts = slot_loss_for_group(editor, group, trace_mean, trace_std, device, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(editor.parameters(), args.grad_clip)
            opt.step()
            totals["loss"] += float(loss.detach().cpu())
            for key, value in parts.items():
                totals[key] += float(value)
            trained += 1
        val_metrics = evaluate_editor_groups(val_groups, editor, trace_mean, trace_std, device, "val_len24", args.base_copy_logit_bias)
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(1, trained),
            "train_init_loss": totals["init_loss"] / max(1, trained),
            "train_op_loss": totals["op_loss"] / max(1, trained),
            "train_arg_loss": totals["arg_loss"] / max(1, trained),
            "train_target_changed_slots": totals["target_changed_slots"] / max(1, trained),
            "train_target_oracle_fraction": totals["target_is_oracle"] / max(1, trained),
            "train_target_gold_fallback_fraction": totals["target_is_gold_fallback"] / max(1, trained),
            "train_target_base_fallback_fraction": totals["target_is_base_fallback"] / max(1, trained),
            **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
        }
        rows.append(row)
        print(
            f"[editor] epoch={epoch} loss={row['train_loss']:.4f} "
            f"val_editor={100.0 * val_metrics['editor_executor_accuracy']:.1f}% "
            f"val_base={100.0 * val_metrics['base_executor_accuracy']:.1f}% "
            f"val_oracle={100.0 * val_metrics['oracle_executor_accuracy']:.1f}%",
            flush=True,
        )
        if val_metrics["editor_executor_accuracy"] > best_val:
            best_val = float(val_metrics["editor_executor_accuracy"])
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in editor.state_dict().items()}
    if best_state is not None:
        editor.load_state_dict(best_state)
    ckpt_path = CHECKPOINT_ROOT / run_dir.name / "slot_repair_editor.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": editor.state_dict(),
            "trace_feature_names": TRACE_FEATURE_NAMES,
            "trace_mean": trace_mean,
            "trace_std": trace_std,
            "best_epoch": best_epoch,
            "best_val_editor_executor_accuracy": best_val,
            "args": vars(args),
            "compiler_args": run_args,
        },
        ckpt_path,
    )
    return editor, trace_mean, trace_std, rows


def parse_threshold_grid(spec: str) -> List[float]:
    vals: List[float] = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        vals.append(min(1.0, max(0.0, float(item))))
    if not vals:
        vals = [0.5]
    return sorted(set(vals))


def edit_slot_targets(
    group: CandidateGroup,
    target_mode: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int]:
    init_target, op_targets, arg_targets, source = target_program_for_group(group, target_mode)
    base = group.candidates[group.base_index]
    active_len = int(group.length)
    init_t = torch.tensor([int(init_target)], dtype=torch.long, device=device)
    op_t = torch.tensor(list(map(int, op_targets[:active_len])), dtype=torch.long, device=device)
    arg_t = torch.tensor(list(map(int, arg_targets[:active_len])), dtype=torch.long, device=device)
    init_changed = torch.tensor([float(int(init_target) != int(base.init_value))], dtype=torch.float32, device=device)
    op_changed = torch.tensor(
        [float(int(op_targets[t]) != int(base.ops[t])) for t in range(active_len)],
        dtype=torch.float32,
        device=device,
    )
    arg_changed = torch.tensor(
        [float(int(arg_targets[t]) != int(base.args[t])) for t in range(active_len)],
        dtype=torch.float32,
        device=device,
    )
    changed_slots = int(init_changed.item()) + int(op_changed.sum().item()) + int(arg_changed.sum().item())
    return init_t, op_t, arg_t, init_changed, op_changed, arg_changed, source, changed_slots


def prediction_from_program(group: CandidateGroup, init_value: int, ops: Sequence[int], args_values: Sequence[int]) -> Dict[str, Any]:
    ops_full = list(map(int, ops))
    args_full = list(map(int, args_values))
    answer, states = execute_program_python(int(init_value), ops_full, args_full, group.length, len(ops_full), 97)
    return {
        "init_value": int(init_value),
        "ops": ops_full,
        "args": args_full,
        "answer": int(answer),
        "states": list(map(int, states)),
    }


def target_prediction_for_group(group: CandidateGroup, target_mode: str) -> Tuple[Dict[str, Any], str]:
    init_value, ops, args_values, source = target_program_for_group(group, target_mode)
    return prediction_from_program(group, init_value, ops, args_values), source


def logprob_value(logits: torch.Tensor, value: int) -> float:
    return float(F.log_softmax(logits.float(), dim=-1)[int(value)].detach().cpu())


def program_prior(group: CandidateGroup, pred: Dict[str, Any]) -> float:
    score = logprob_value(group.init_logits, int(pred["init_value"]))
    for t in range(group.length):
        score += logprob_value(group.op_logits[t], int(pred["ops"][t]))
        score += logprob_value(group.arg_logits[t], int(pred["args"][t]))
    return float(score)


def edit_record_for_change(group: CandidateGroup, kind: int, pos: int, old_value: int, new_value: int) -> Dict[str, float]:
    if kind == 0:
        logits = group.init_logits
        margin = top2_margin(logits)
        entropy = entropy_from_logits(logits)
        value_den = float(max(1, int(group.final_probs.numel()) - 1))
    elif kind == 1:
        logits = group.op_logits[pos]
        margin = top2_margin(logits)
        entropy = entropy_from_logits(logits)
        value_den = 2.0
    else:
        logits = group.arg_logits[pos]
        margin = top2_margin(logits)
        entropy = entropy_from_logits(logits)
        value_den = float(max(1, int(group.final_probs.numel()) - 1))
    old_logp = logprob_value(logits, old_value)
    new_logp = logprob_value(logits, new_value)
    rank_order = torch.argsort(logits.float(), descending=True).detach().cpu().tolist()
    try:
        new_rank = rank_order.index(int(new_value))
    except ValueError:
        new_rank = len(rank_order)
    return {
        "kind": float(kind),
        "pos": float(pos),
        "old_value": float(old_value),
        "new_value": float(new_value),
        "old_value_norm": float(old_value) / value_den,
        "new_value_norm": float(new_value) / value_den,
        "delta": float(new_logp - old_logp),
        "new_rank": float(new_rank),
        "margin": float(margin),
        "entropy": float(entropy),
    }


def edit_records_from_base(group: CandidateGroup, pred: Dict[str, Any]) -> List[Dict[str, float]]:
    base = group.candidates[group.base_index]
    records: List[Dict[str, float]] = []
    if int(pred["init_value"]) != int(base.init_value):
        records.append(edit_record_for_change(group, 0, -1, int(base.init_value), int(pred["init_value"])))
    for t in range(group.length):
        if int(pred["ops"][t]) != int(base.ops[t]):
            records.append(edit_record_for_change(group, 1, t, int(base.ops[t]), int(pred["ops"][t])))
        if int(pred["args"][t]) != int(base.args[t]):
            records.append(edit_record_for_change(group, 2, t, int(base.args[t]), int(pred["args"][t])))
    return records


def trace_for_prediction(group: CandidateGroup, pred: Dict[str, Any]) -> torch.Tensor:
    base = group.candidates[group.base_index]
    modulus = int(group.final_probs.numel())
    max_steps = len(base.ops)
    op_options = [top_options_with_rank(group.op_logits[t], int(group.op_logits.shape[-1])) for t in range(max_steps)]
    arg_options = [top_options_with_rank(group.arg_logits[t], int(group.arg_logits.shape[-1])) for t in range(max_steps)]
    op_margins = [top2_margin(group.op_logits[t]) for t in range(max_steps)]
    arg_margins = [top2_margin(group.arg_logits[t]) for t in range(max_steps)]
    op_entropies = [entropy_from_logits(group.op_logits[t]) for t in range(max_steps)]
    arg_entropies = [entropy_from_logits(group.arg_logits[t]) for t in range(max_steps)]
    trace = trace_for_candidate(
        init_value=int(pred["init_value"]),
        ops=list(map(int, pred["ops"])),
        args=list(map(int, pred["args"])),
        answer=int(pred["answer"]),
        states=list(map(int, pred["states"])),
        prior=program_prior(group, pred),
        edit_records=edit_records_from_base(group, pred),
        base_prior=float(base.prior),
        base_answer=int(base.answer),
        base_ops=list(map(int, base.ops)),
        base_args=list(map(int, base.args)),
        base_states=list(map(int, base.states)),
        length=int(group.length),
        max_steps=max_steps,
        modulus=modulus,
        op_options=op_options,
        arg_options=arg_options,
        op_margins=op_margins,
        arg_margins=arg_margins,
        op_entropies=op_entropies,
        arg_entropies=arg_entropies,
        state_log_probs=[float(x) for x in group.state_probs.float().clamp_min(1e-12).log().reshape(-1).tolist()],
        final_log_probs=[float(x) for x in group.final_probs.float().clamp_min(1e-12).log().tolist()],
        base_final_soft_logp=float(group.final_probs.float().clamp_min(1e-12).log()[int(base.answer)].item()),
    )
    return torch.tensor(trace, dtype=torch.float32)


def normalized_prediction_trace(
    group: CandidateGroup,
    pred: Dict[str, Any],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    x = trace_for_prediction(group, pred).to(device)
    return ((x - mean.to(device)) / std.to(device)).float().unsqueeze(0)


def diff_targets_for_prediction(
    group: CandidateGroup,
    current: Dict[str, Any],
    target: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    active_len = int(group.length)
    init_t = torch.tensor([int(target["init_value"])], dtype=torch.long, device=device)
    op_t = torch.tensor(list(map(int, target["ops"][:active_len])), dtype=torch.long, device=device)
    arg_t = torch.tensor(list(map(int, target["args"][:active_len])), dtype=torch.long, device=device)
    init_changed = torch.tensor([float(int(target["init_value"]) != int(current["init_value"]))], dtype=torch.float32, device=device)
    op_changed = torch.tensor(
        [float(int(target["ops"][t]) != int(current["ops"][t])) for t in range(active_len)],
        dtype=torch.float32,
        device=device,
    )
    arg_changed = torch.tensor(
        [float(int(target["args"][t]) != int(current["args"][t])) for t in range(active_len)],
        dtype=torch.float32,
        device=device,
    )
    changed_slots = int(init_changed.item()) + int(op_changed.sum().item()) + int(arg_changed.sum().item())
    return init_t, op_t, arg_t, init_changed, op_changed, arg_changed, changed_slots


def repair_case_loss(
    policy: SlotEditPolicy,
    group: CandidateGroup,
    current: Dict[str, Any],
    target: Dict[str, Any],
    trace_mean: torch.Tensor,
    trace_std: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    init_t, op_t, arg_t, init_changed, op_changed, arg_changed, changed_slots = diff_targets_for_prediction(
        group, current, target, device
    )
    (
        init_gate_logits,
        op_gate_logits,
        arg_gate_logits,
        init_logits,
        op_logits,
        arg_logits,
    ) = policy(normalized_prediction_trace(group, current, trace_mean, trace_std, device))
    active_len = int(group.length)
    pos_weight = torch.tensor([float(args.edit_gate_pos_weight)], dtype=torch.float32, device=device)
    init_gate_loss = F.binary_cross_entropy_with_logits(init_gate_logits, init_changed, pos_weight=pos_weight)
    if active_len > 0:
        op_gate_loss = F.binary_cross_entropy_with_logits(op_gate_logits[0, :active_len], op_changed, pos_weight=pos_weight)
        arg_gate_loss = F.binary_cross_entropy_with_logits(arg_gate_logits[0, :active_len], arg_changed, pos_weight=pos_weight)
        op_weights = torch.where(
            op_changed > 0.5,
            torch.full_like(op_changed, float(args.changed_slot_weight)),
            torch.full_like(op_changed, float(args.unchanged_value_loss_weight)),
        )
        arg_weights = torch.where(
            arg_changed > 0.5,
            torch.full_like(arg_changed, float(args.changed_slot_weight)),
            torch.full_like(arg_changed, float(args.unchanged_value_loss_weight)),
        )
        op_value_loss = weighted_ce_loss(op_logits[0, :active_len, :], op_t, op_weights)
        arg_value_loss = weighted_ce_loss(arg_logits[0, :active_len, :], arg_t, arg_weights)
    else:
        op_gate_loss = op_gate_logits.sum() * 0.0
        arg_gate_loss = arg_gate_logits.sum() * 0.0
        op_value_loss = op_logits.sum() * 0.0
        arg_value_loss = arg_logits.sum() * 0.0
    init_weight_value = float(args.changed_slot_weight) if float(init_changed.item()) > 0.5 else float(args.unchanged_value_loss_weight)
    init_value_weight = torch.tensor([init_weight_value], dtype=torch.float32, device=device)
    init_value_loss = weighted_ce_loss(init_logits, init_t, init_value_weight)
    gate_loss = args.init_loss_weight * init_gate_loss + args.op_loss_weight * op_gate_loss + args.arg_loss_weight * arg_gate_loss
    value_loss = args.init_loss_weight * init_value_loss + args.op_loss_weight * op_value_loss + args.arg_loss_weight * arg_value_loss
    loss = args.edit_gate_loss_weight * gate_loss + args.edit_value_loss_weight * value_loss
    active_slots = 1 + 2 * active_len
    return loss, {
        "gate_loss": float(gate_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "target_changed_slots": float(changed_slots),
        "target_changed_fraction": float(changed_slots / max(1, active_slots)),
    }


def weighted_ce_loss(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if target.numel() == 0 or float(weights.sum().detach().cpu()) <= 0.0:
        return logits.sum() * 0.0
    loss_each = F.cross_entropy(logits, target, reduction="none")
    return (loss_each * weights).sum() / weights.sum().clamp_min(1e-6)


def edit_policy_loss_for_group(
    policy: SlotEditPolicy,
    group: CandidateGroup,
    trace_mean: torch.Tensor,
    trace_std: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    init_t, op_t, arg_t, init_changed, op_changed, arg_changed, source, changed_slots = edit_slot_targets(
        group, args.editor_target_mode, device
    )
    (
        init_gate_logits,
        op_gate_logits,
        arg_gate_logits,
        init_logits,
        op_logits,
        arg_logits,
    ) = policy(normalized_editor_trace(group, trace_mean, trace_std, device))
    active_len = int(group.length)
    pos_weight = torch.tensor([float(args.edit_gate_pos_weight)], dtype=torch.float32, device=device)
    init_gate_loss = F.binary_cross_entropy_with_logits(init_gate_logits, init_changed, pos_weight=pos_weight)
    if active_len > 0:
        op_gate_loss = F.binary_cross_entropy_with_logits(op_gate_logits[0, :active_len], op_changed, pos_weight=pos_weight)
        arg_gate_loss = F.binary_cross_entropy_with_logits(arg_gate_logits[0, :active_len], arg_changed, pos_weight=pos_weight)
        op_weights = torch.where(
            op_changed > 0.5,
            torch.full_like(op_changed, float(args.changed_slot_weight)),
            torch.full_like(op_changed, float(args.unchanged_value_loss_weight)),
        )
        arg_weights = torch.where(
            arg_changed > 0.5,
            torch.full_like(arg_changed, float(args.changed_slot_weight)),
            torch.full_like(arg_changed, float(args.unchanged_value_loss_weight)),
        )
        op_value_loss = weighted_ce_loss(op_logits[0, :active_len, :], op_t, op_weights)
        arg_value_loss = weighted_ce_loss(arg_logits[0, :active_len, :], arg_t, arg_weights)
    else:
        op_gate_loss = op_gate_logits.sum() * 0.0
        arg_gate_loss = arg_gate_logits.sum() * 0.0
        op_value_loss = op_logits.sum() * 0.0
        arg_value_loss = arg_logits.sum() * 0.0
    init_weight_value = float(args.changed_slot_weight) if float(init_changed.item()) > 0.5 else float(args.unchanged_value_loss_weight)
    init_value_weight = torch.tensor([init_weight_value], dtype=torch.float32, device=device)
    init_value_loss = weighted_ce_loss(init_logits, init_t, init_value_weight)
    gate_loss = args.init_loss_weight * init_gate_loss + args.op_loss_weight * op_gate_loss + args.arg_loss_weight * arg_gate_loss
    value_loss = args.init_loss_weight * init_value_loss + args.op_loss_weight * op_value_loss + args.arg_loss_weight * arg_value_loss
    loss = args.edit_gate_loss_weight * gate_loss + args.edit_value_loss_weight * value_loss
    active_slots = 1 + 2 * active_len
    return loss, {
        "gate_loss": float(gate_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "init_gate_loss": float(init_gate_loss.detach().cpu()),
        "op_gate_loss": float(op_gate_loss.detach().cpu()),
        "arg_gate_loss": float(arg_gate_loss.detach().cpu()),
        "init_value_loss": float(init_value_loss.detach().cpu()),
        "op_value_loss": float(op_value_loss.detach().cpu()),
        "arg_value_loss": float(arg_value_loss.detach().cpu()),
        "target_changed_slots": float(changed_slots),
        "target_changed_fraction": float(changed_slots / max(1, active_slots)),
        "target_is_oracle": 1.0 if source == "oracle" else 0.0,
        "target_is_gold_fallback": 1.0 if source == "gold_fallback" else 0.0,
        "target_is_base_fallback": 1.0 if source == "base_fallback" else 0.0,
    }


@torch.no_grad()
def edit_policy_prediction(
    policy: SlotEditPolicy,
    group: CandidateGroup,
    trace_mean: torch.Tensor,
    trace_std: torch.Tensor,
    device: torch.device,
    threshold: float,
    modulus: int,
) -> Dict[str, Any]:
    policy.eval()
    (
        init_gate_logits,
        op_gate_logits,
        arg_gate_logits,
        init_logits,
        op_logits,
        arg_logits,
    ) = policy(normalized_editor_trace(group, trace_mean, trace_std, device))
    base = group.candidates[group.base_index]
    init_value = int(base.init_value)
    ops = list(map(int, base.ops))
    args_values = list(map(int, base.args))
    init_prob = float(torch.sigmoid(init_gate_logits[0]).detach().cpu())
    op_probs = [float(x) for x in torch.sigmoid(op_gate_logits[0]).detach().cpu().tolist()]
    arg_probs = [float(x) for x in torch.sigmoid(arg_gate_logits[0]).detach().cpu().tolist()]
    init_gate = init_prob >= float(threshold)
    op_gates = [False for _ in range(len(ops))]
    arg_gates = [False for _ in range(len(args_values))]
    if init_gate:
        init_value = int(init_logits.argmax(dim=-1).item())
    op_pred = [int(x) for x in op_logits[0].argmax(dim=-1).detach().cpu().tolist()]
    arg_pred = [int(x) for x in arg_logits[0].argmax(dim=-1).detach().cpu().tolist()]
    for t in range(group.length):
        op_gates[t] = bool(op_probs[t] >= float(threshold))
        arg_gates[t] = bool(arg_probs[t] >= float(threshold))
        if op_gates[t]:
            ops[t] = int(op_pred[t])
        if arg_gates[t]:
            args_values[t] = int(arg_pred[t])
    answer, states = execute_program_python(init_value, ops, args_values, group.length, len(ops), int(modulus))
    return {
        "init_value": init_value,
        "ops": ops,
        "args": args_values,
        "answer": int(answer),
        "states": list(map(int, states)),
        "init_gate": bool(init_gate),
        "op_gates": op_gates,
        "arg_gates": arg_gates,
    }


def edit_gate_metrics(groups: Sequence[CandidateGroup], preds: Sequence[Dict[str, Any]], target_mode: str, prefix: str) -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    pred_pos = 0
    true_pos = 0
    total = 0
    for group, pred in zip(groups, preds):
        target_init, target_ops, target_args, _ = target_program_for_group(group, target_mode)
        base = group.candidates[group.base_index]
        true_init = int(target_init) != int(base.init_value)
        pred_init = bool(pred.get("init_gate", False))
        pairs = [(pred_init, true_init)]
        for t in range(group.length):
            pairs.append((bool(pred.get("op_gates", [])[t]), int(target_ops[t]) != int(base.ops[t])))
            pairs.append((bool(pred.get("arg_gates", [])[t]), int(target_args[t]) != int(base.args[t])))
        for pred_changed, true_changed in pairs:
            pred_pos += int(pred_changed)
            true_pos += int(true_changed)
            tp += int(pred_changed and true_changed)
            fp += int(pred_changed and not true_changed)
            fn += int((not pred_changed) and true_changed)
            total += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        f"{prefix}_gate_precision": precision,
        f"{prefix}_gate_recall": recall,
        f"{prefix}_gate_f1": 2.0 * precision * recall / max(1e-9, precision + recall),
        f"{prefix}_gate_positive_rate": pred_pos / max(1, total),
        f"{prefix}_gate_target_positive_rate": true_pos / max(1, total),
    }


@torch.no_grad()
def evaluate_edit_policy_groups(
    groups: Sequence[CandidateGroup],
    policy: Optional[SlotEditPolicy],
    trace_mean: Optional[torch.Tensor],
    trace_std: Optional[torch.Tensor],
    device: torch.device,
    split_name: str,
    threshold: float,
    modulus: int,
    target_mode: str,
) -> Dict[str, float]:
    base_preds = [candidate_to_prediction(group.candidates[group.base_index]) for group in groups]
    oracle_preds = [candidate_to_prediction(group.candidates[oracle_candidate_index(group)]) for group in groups]
    out: Dict[str, float] = {
        "n": float(len(groups)),
        "oracle_found_fraction": sum(group.has_positive() for group in groups) / max(1, len(groups)),
        "avg_candidates": safe_mean([float(len(group.candidates)) for group in groups]),
        "avg_positive_candidates": safe_mean([float(group.state_labels.sum().item()) for group in groups]),
    }
    out.update(summarize_predictions(groups, base_preds, "base"))
    out.update(summarize_predictions(groups, oracle_preds, "oracle"))
    if policy is not None and trace_mean is not None and trace_std is not None:
        editor_preds = [
            edit_policy_prediction(policy, group, trace_mean, trace_std, device, threshold, modulus)
            for group in groups
        ]
        out.update(summarize_predictions(groups, editor_preds, "editor"))
        out.update(edit_gate_metrics(groups, editor_preds, target_mode, "editor"))
        out["editor_threshold"] = float(threshold)
        denom = out["oracle_executor_accuracy"] - out["base_executor_accuracy"]
        out["editor_oracle_gap_recovered"] = (
            (out["editor_executor_accuracy"] - out["base_executor_accuracy"]) / denom if abs(denom) > 1e-9 else math.nan
        )
        if "paired" in split_name:
            out.update(prediction_pair_metrics(groups, base_preds, "base"))
            out.update(prediction_pair_metrics(groups, oracle_preds, "oracle"))
            out.update(prediction_pair_metrics(groups, editor_preds, "editor"))
    return out


def candidate_prediction(cand: Candidate) -> Dict[str, Any]:
    return {
        "init_value": int(cand.init_value),
        "ops": list(map(int, cand.ops)),
        "args": list(map(int, cand.args)),
        "answer": int(cand.answer),
        "states": list(map(int, cand.states)),
    }


def base_prediction(group: CandidateGroup) -> Dict[str, Any]:
    return candidate_prediction(group.candidates[group.base_index])


def fit_iterative_trace_stats(groups: Sequence[CandidateGroup], stats_candidate_traces: int) -> Tuple[torch.Tensor, torch.Tensor]:
    traces: List[torch.Tensor] = []
    for group in groups:
        traces.append(base_trace(group))
        if stats_candidate_traces > 0:
            ranked = sorted(group.candidates, key=lambda c: float(c.prior), reverse=True)
            for cand in ranked[: int(stats_candidate_traces)]:
                traces.append(torch.tensor(cand.trace, dtype=torch.float32))
    stacked = torch.stack(traces, dim=0)
    flat = stacked.reshape(-1, stacked.shape[-1])
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp_min(1e-5)
    return mean, std


def training_currents_for_group(
    group: CandidateGroup,
    rng: random.Random,
    target: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    currents = [base_prediction(group)]
    if bool(args.include_target_noedit_case):
        currents.append(target)
    pool = [cand for cand in group.candidates if not cand.state_exact]
    if bool(args.augment_repairable_only):
        pool = [cand for cand in pool if cand.edit_count > 0]
    if pool and int(args.augment_cases_per_group) > 0:
        ranked = sorted(pool, key=lambda c: (candidate_prefix_fraction(c, group), float(c.prior)), reverse=True)
        take_top = min(len(ranked), int(args.augment_top_pool))
        subpool = ranked[:take_top] if take_top > 0 else ranked
        for cand in rng.sample(subpool, k=min(len(subpool), int(args.augment_cases_per_group))):
            currents.append(candidate_prediction(cand))
    return currents


@torch.no_grad()
def apply_policy_step(
    policy: SlotEditPolicy,
    group: CandidateGroup,
    current: Dict[str, Any],
    trace_mean: torch.Tensor,
    trace_std: torch.Tensor,
    device: torch.device,
    threshold: float,
    max_edits_per_iteration: int,
) -> Tuple[Dict[str, Any], int]:
    policy.eval()
    (
        init_gate_logits,
        op_gate_logits,
        arg_gate_logits,
        init_logits,
        op_logits,
        arg_logits,
    ) = policy(normalized_prediction_trace(group, current, trace_mean, trace_std, device))
    proposals: List[Tuple[float, str, int, int]] = []
    init_prob = float(torch.sigmoid(init_gate_logits[0]).detach().cpu())
    init_value = int(init_logits.argmax(dim=-1).item())
    if init_prob >= float(threshold) and init_value != int(current["init_value"]):
        proposals.append((init_prob, "init", -1, init_value))
    op_probs = torch.sigmoid(op_gate_logits[0]).detach().cpu()
    arg_probs = torch.sigmoid(arg_gate_logits[0]).detach().cpu()
    op_pred = op_logits[0].argmax(dim=-1).detach().cpu().tolist()
    arg_pred = arg_logits[0].argmax(dim=-1).detach().cpu().tolist()
    for t in range(group.length):
        op_prob = float(op_probs[t])
        arg_prob = float(arg_probs[t])
        if op_prob >= float(threshold) and int(op_pred[t]) != int(current["ops"][t]):
            proposals.append((op_prob, "op", t, int(op_pred[t])))
        if arg_prob >= float(threshold) and int(arg_pred[t]) != int(current["args"][t]):
            proposals.append((arg_prob, "arg", t, int(arg_pred[t])))
    proposals.sort(reverse=True, key=lambda x: x[0])
    if max_edits_per_iteration > 0:
        proposals = proposals[: int(max_edits_per_iteration)]
    next_init = int(current["init_value"])
    next_ops = list(map(int, current["ops"]))
    next_args = list(map(int, current["args"]))
    edits = 0
    for _, kind, pos, value in proposals:
        if kind == "init":
            if value != next_init:
                next_init = value
                edits += 1
        elif kind == "op":
            if value != next_ops[pos]:
                next_ops[pos] = value
                edits += 1
        elif kind == "arg":
            if value != next_args[pos]:
                next_args[pos] = value
                edits += 1
    if edits <= 0:
        return current, 0
    return prediction_from_program(group, next_init, next_ops, next_args), edits


@torch.no_grad()
def iterative_policy_prediction(
    policy: SlotEditPolicy,
    group: CandidateGroup,
    trace_mean: torch.Tensor,
    trace_std: torch.Tensor,
    device: torch.device,
    threshold: float,
    k_steps: int,
    max_edits_per_iteration: int,
) -> Dict[str, Any]:
    current = base_prediction(group)
    total_edits = 0
    steps_used = 0
    for _ in range(int(k_steps)):
        nxt, edits = apply_policy_step(
            policy,
            group,
            current,
            trace_mean,
            trace_std,
            device,
            threshold,
            max_edits_per_iteration,
        )
        if edits <= 0:
            break
        current = nxt
        total_edits += edits
        steps_used += 1
    current = dict(current)
    current["iter_steps_used"] = int(steps_used)
    current["iter_total_edits"] = int(total_edits)
    return current


def iterative_extra_metrics(preds: Sequence[Dict[str, Any]], prefix: str) -> Dict[str, float]:
    if not preds:
        return {f"{prefix}_steps_used_mean": 0.0, f"{prefix}_total_iter_edits_mean": 0.0}
    return {
        f"{prefix}_steps_used_mean": safe_mean([float(pred.get("iter_steps_used", 0)) for pred in preds]),
        f"{prefix}_total_iter_edits_mean": safe_mean([float(pred.get("iter_total_edits", 0)) for pred in preds]),
    }


@torch.no_grad()
def evaluate_iterative_policy_groups(
    groups: Sequence[CandidateGroup],
    policy: Optional[SlotEditPolicy],
    trace_mean: Optional[torch.Tensor],
    trace_std: Optional[torch.Tensor],
    device: torch.device,
    split_name: str,
    threshold: float,
    k_steps: int,
    max_edits_per_iteration: int,
) -> Dict[str, float]:
    base_preds = [base_prediction(group) for group in groups]
    oracle_preds = [candidate_prediction(group.candidates[oracle_candidate_index(group)]) for group in groups]
    out: Dict[str, float] = {
        "n": float(len(groups)),
        "k_steps": float(k_steps),
        "threshold": float(threshold),
        "oracle_found_fraction": sum(group.has_positive() for group in groups) / max(1, len(groups)),
        "avg_candidates": safe_mean([float(len(group.candidates)) for group in groups]),
        "avg_positive_candidates": safe_mean([float(group.state_labels.sum().item()) for group in groups]),
    }
    out.update(summarize_predictions(groups, base_preds, "base"))
    out.update(summarize_predictions(groups, oracle_preds, "oracle"))
    if policy is not None and trace_mean is not None and trace_std is not None:
        iter_preds = [
            iterative_policy_prediction(
                policy,
                group,
                trace_mean,
                trace_std,
                device,
                threshold,
                k_steps,
                max_edits_per_iteration,
            )
            for group in groups
        ]
        out.update(summarize_predictions(groups, iter_preds, "iter"))
        out.update(iterative_extra_metrics(iter_preds, "iter"))
        denom = out["oracle_executor_accuracy"] - out["base_executor_accuracy"]
        out["iter_oracle_gap_recovered"] = (
            (out["iter_executor_accuracy"] - out["base_executor_accuracy"]) / denom if abs(denom) > 1e-9 else math.nan
        )
        if "paired" in split_name:
            out.update(prediction_pair_metrics(groups, base_preds, "base"))
            out.update(prediction_pair_metrics(groups, oracle_preds, "oracle"))
            out.update(prediction_pair_metrics(groups, iter_preds, "iter"))
    return out


def metric_selection_key(metrics: Dict[str, float]) -> Tuple[float, float, float, float]:
    primary = float(metrics.get("iter_pair_both_correct", metrics.get("iter_executor_accuracy", -1.0)))
    consistency = float(metrics.get("iter_pair_state_consistency", metrics.get("iter_state_prefix_fraction", -1.0)))
    return (
        primary,
        float(metrics.get("iter_executor_accuracy", -1.0)) + 0.1 * consistency,
        -float(metrics.get("iter_total_iter_edits_mean", 1e9)),
        -float(metrics.get("k_steps", 1e9)),
    )


def parse_int_grid(spec: str) -> List[int]:
    vals: List[int] = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        vals.append(max(0, int(item)))
    return sorted(set(vals)) or [0]


def train_iterative_policy(
    train_groups: Sequence[CandidateGroup],
    val_groups: Sequence[CandidateGroup],
    args: argparse.Namespace,
    run_dir: Path,
    run_args: Dict[str, Any],
    device: torch.device,
) -> Tuple[SlotEditPolicy, torch.Tensor, torch.Tensor, List[Dict[str, Any]], float, int, int]:
    trace_mean, trace_std = fit_iterative_trace_stats(train_groups, int(args.stats_candidate_traces))
    policy = SlotEditPolicy(
        len(TRACE_FEATURE_NAMES),
        int(run_args["modulus"]),
        int(run_args["max_steps"]),
        args.editor_d_model,
        args.editor_layers,
        args.editor_heads,
        args.editor_ff_mult,
        args.editor_dropout,
    ).to(device)
    thresholds = parse_threshold_grid(args.edit_threshold_grid)
    train_k_values = parse_int_grid(args.train_select_k_values)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.editor_lr, weight_decay=args.editor_weight_decay)
    rng = random.Random(args.seed + 2700)
    rows: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_threshold = thresholds[0]
    best_k = train_k_values[0]
    best_epoch = 0
    best_key = (-1.0, -1.0, -1e9, -1e9)
    best_val = -1.0
    for epoch in range(1, args.editor_epochs + 1):
        policy.train()
        order = list(range(len(train_groups)))
        rng.shuffle(order)
        totals = {
            "loss": 0.0,
            "gate_loss": 0.0,
            "value_loss": 0.0,
            "target_changed_slots": 0.0,
            "target_changed_fraction": 0.0,
        }
        trained = 0
        cases = 0
        for idx in order:
            group = train_groups[idx]
            target, _ = target_prediction_for_group(group, args.editor_target_mode)
            currents = training_currents_for_group(group, rng, target, args)
            for current in currents:
                loss, parts = repair_case_loss(policy, group, current, target, trace_mean, trace_std, device, args)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                opt.step()
                totals["loss"] += float(loss.detach().cpu())
                for key in ["gate_loss", "value_loss", "target_changed_slots", "target_changed_fraction"]:
                    totals[key] += float(parts[key])
                cases += 1
            trained += 1
        threshold_metrics: List[Dict[str, float]] = []
        for threshold in thresholds:
            for k_steps in train_k_values:
                threshold_metrics.append(
                    evaluate_iterative_policy_groups(
                        val_groups,
                        policy,
                        trace_mean,
                        trace_std,
                        device,
                        "val_len24",
                        threshold,
                        k_steps,
                        int(args.max_edits_per_iteration),
                    )
                )
        val_metrics = max(threshold_metrics, key=metric_selection_key)
        val_threshold = float(val_metrics["threshold"])
        val_k = int(val_metrics["k_steps"])
        row = {
            "epoch": epoch,
            "train_groups": trained,
            "train_cases": cases,
            "train_loss": totals["loss"] / max(1, cases),
            "train_gate_loss": totals["gate_loss"] / max(1, cases),
            "train_value_loss": totals["value_loss"] / max(1, cases),
            "train_target_changed_slots": totals["target_changed_slots"] / max(1, cases),
            "train_target_changed_fraction": totals["target_changed_fraction"] / max(1, cases),
            "val_selected_threshold": val_threshold,
            "val_selected_k": val_k,
            **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
        }
        rows.append(row)
        print(
            f"[iter_policy] epoch={epoch} loss={row['train_loss']:.4f} "
            f"k={val_k} thr={val_threshold:.2f} "
            f"val_iter={100.0 * val_metrics['iter_executor_accuracy']:.1f}% "
            f"val_base={100.0 * val_metrics['base_executor_accuracy']:.1f}% "
            f"val_oracle={100.0 * val_metrics['oracle_executor_accuracy']:.1f}% "
            f"steps={val_metrics['iter_steps_used_mean']:.2f} edits={val_metrics['iter_total_iter_edits_mean']:.2f}",
            flush=True,
        )
        key = metric_selection_key(val_metrics)
        if key > best_key:
            best_key = key
            best_val = float(val_metrics["iter_executor_accuracy"])
            best_epoch = epoch
            best_threshold = val_threshold
            best_k = val_k
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    if best_state is not None:
        policy.load_state_dict(best_state)
    ckpt_path = CHECKPOINT_ROOT / run_dir.name / "iterative_repair_policy.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "editor_type": "iterative_sparse_edit_policy",
            "trace_feature_names": TRACE_FEATURE_NAMES,
            "trace_mean": trace_mean,
            "trace_std": trace_std,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "best_k": best_k,
            "best_val_iter_executor_accuracy": best_val,
            "args": vars(args),
            "compiler_args": run_args,
        },
        ckpt_path,
    )
    return policy, trace_mean, trace_std, rows, best_threshold, best_k, best_epoch


def train_edit_policy(
    train_groups: Sequence[CandidateGroup],
    val_groups: Sequence[CandidateGroup],
    args: argparse.Namespace,
    run_dir: Path,
    run_args: Dict[str, Any],
    device: torch.device,
) -> Tuple[SlotEditPolicy, torch.Tensor, torch.Tensor, List[Dict[str, Any]], float, int]:
    trace_mean, trace_std = fit_editor_trace_stats(train_groups)
    policy = SlotEditPolicy(
        len(TRACE_FEATURE_NAMES),
        int(run_args["modulus"]),
        int(run_args["max_steps"]),
        args.editor_d_model,
        args.editor_layers,
        args.editor_heads,
        args.editor_ff_mult,
        args.editor_dropout,
    ).to(device)
    thresholds = parse_threshold_grid(args.edit_threshold_grid)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.editor_lr, weight_decay=args.editor_weight_decay)
    rng = random.Random(args.seed + 1700)
    rows: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_threshold = thresholds[0]
    best_epoch = 0
    best_key = (-1.0, -1.0, -1e9)
    best_val = -1.0
    for epoch in range(1, args.editor_epochs + 1):
        policy.train()
        order = list(range(len(train_groups)))
        rng.shuffle(order)
        totals: Dict[str, float] = {
            "loss": 0.0,
            "gate_loss": 0.0,
            "value_loss": 0.0,
            "init_gate_loss": 0.0,
            "op_gate_loss": 0.0,
            "arg_gate_loss": 0.0,
            "init_value_loss": 0.0,
            "op_value_loss": 0.0,
            "arg_value_loss": 0.0,
            "target_changed_slots": 0.0,
            "target_changed_fraction": 0.0,
            "target_is_oracle": 0.0,
            "target_is_gold_fallback": 0.0,
            "target_is_base_fallback": 0.0,
        }
        trained = 0
        for idx in order:
            group = train_groups[idx]
            loss, parts = edit_policy_loss_for_group(policy, group, trace_mean, trace_std, device, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            opt.step()
            totals["loss"] += float(loss.detach().cpu())
            for key, value in parts.items():
                totals[key] += float(value)
            trained += 1
        threshold_metrics = [
            evaluate_edit_policy_groups(
                val_groups,
                policy,
                trace_mean,
                trace_std,
                device,
                "val_len24",
                threshold,
                int(run_args["modulus"]),
                args.editor_target_mode,
            )
            for threshold in thresholds
        ]
        val_metrics = max(threshold_metrics, key=metric_selection_key)
        val_threshold = float(val_metrics["editor_threshold"])
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(1, trained),
            "train_gate_loss": totals["gate_loss"] / max(1, trained),
            "train_value_loss": totals["value_loss"] / max(1, trained),
            "train_init_gate_loss": totals["init_gate_loss"] / max(1, trained),
            "train_op_gate_loss": totals["op_gate_loss"] / max(1, trained),
            "train_arg_gate_loss": totals["arg_gate_loss"] / max(1, trained),
            "train_init_value_loss": totals["init_value_loss"] / max(1, trained),
            "train_op_value_loss": totals["op_value_loss"] / max(1, trained),
            "train_arg_value_loss": totals["arg_value_loss"] / max(1, trained),
            "train_target_changed_slots": totals["target_changed_slots"] / max(1, trained),
            "train_target_changed_fraction": totals["target_changed_fraction"] / max(1, trained),
            "train_target_oracle_fraction": totals["target_is_oracle"] / max(1, trained),
            "train_target_gold_fallback_fraction": totals["target_is_gold_fallback"] / max(1, trained),
            "train_target_base_fallback_fraction": totals["target_is_base_fallback"] / max(1, trained),
            "val_selected_threshold": val_threshold,
            **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
        }
        rows.append(row)
        print(
            f"[edit_policy] epoch={epoch} loss={row['train_loss']:.4f} "
            f"thr={val_threshold:.2f} val_editor={100.0 * val_metrics['editor_executor_accuracy']:.1f}% "
            f"val_base={100.0 * val_metrics['base_executor_accuracy']:.1f}% "
            f"val_oracle={100.0 * val_metrics['oracle_executor_accuracy']:.1f}% "
            f"edits={val_metrics['editor_avg_edits']:.2f}",
            flush=True,
        )
        key = metric_selection_key(val_metrics)
        if key > best_key:
            best_key = key
            best_val = float(val_metrics["editor_executor_accuracy"])
            best_epoch = epoch
            best_threshold = val_threshold
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    if best_state is not None:
        policy.load_state_dict(best_state)
    ckpt_path = CHECKPOINT_ROOT / run_dir.name / "slot_repair_editor.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "editor_type": "gated_edit_policy",
            "trace_feature_names": TRACE_FEATURE_NAMES,
            "trace_mean": trace_mean,
            "trace_std": trace_std,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "best_val_editor_executor_accuracy": best_val,
            "args": vars(args),
            "compiler_args": run_args,
        },
        ckpt_path,
    )
    return policy, trace_mean, trace_std, rows, best_threshold, best_epoch


def candidate_prefix_fraction(candidate: Candidate, group: CandidateGroup) -> float:
    return correct_prefix_fraction(candidate.states, group.states, group.length)


def summarize_selected(groups: Sequence[CandidateGroup], selected_indices: Sequence[int], prefix: str) -> Dict[str, float]:
    n = len(groups)
    correct = 0
    program_exact_count = 0
    changed = 0
    prefix_sum = 0.0
    edit_sum = 0.0
    for group, idx in zip(groups, selected_indices):
        cand = group.candidates[int(idx)]
        correct += int(cand.state_exact)
        program_exact_count += int(cand.program_exact)
        changed += int(cand.changed)
        prefix_sum += candidate_prefix_fraction(cand, group)
        edit_sum += cand.edit_count
    return {
        f"{prefix}_executor_accuracy": correct / max(1, n),
        f"{prefix}_program_exact": program_exact_count / max(1, n),
        f"{prefix}_state_prefix_fraction": prefix_sum / max(1, n),
        f"{prefix}_changed_fraction": changed / max(1, n),
        f"{prefix}_avg_edits": edit_sum / max(1, n),
    }


def pair_metrics(groups: Sequence[CandidateGroup], selected_indices: Sequence[int], prefix: str) -> Dict[str, float]:
    usable = (len(groups) // 2) * 2
    if usable < 2:
        return {}
    answer_same = 0
    both_correct = 0
    program_same = 0
    state_same = 0
    pair_count = usable // 2
    for i in range(0, usable, 2):
        g0, g1 = groups[i], groups[i + 1]
        c0 = g0.candidates[int(selected_indices[i])]
        c1 = g1.candidates[int(selected_indices[i + 1])]
        length = g0.length
        answer_same += int(c0.answer == c1.answer)
        both_correct += int(c0.state_exact and c1.state_exact)
        same_program = c0.init_value == c1.init_value and c0.ops[:length] == c1.ops[:length] and c0.args[:length] == c1.args[:length]
        program_same += int(same_program)
        state_same += int(c0.states[:length] == c1.states[:length])
    return {
        f"{prefix}_pair_answer_consistency": answer_same / pair_count,
        f"{prefix}_pair_both_correct": both_correct / pair_count,
        f"{prefix}_pair_program_consistency": program_same / pair_count,
        f"{prefix}_pair_state_consistency": state_same / pair_count,
    }


@torch.no_grad()
def pair_consistency_indices(
    groups: Sequence[CandidateGroup],
    learned_scores: Sequence[torch.Tensor],
    top_m: int,
    bonus: float,
) -> List[int]:
    selected: List[int] = []
    usable = (len(groups) // 2) * 2
    for i in range(0, usable, 2):
        g0, g1 = groups[i], groups[i + 1]
        s0, s1 = learned_scores[i], learned_scores[i + 1]
        k0 = min(top_m, s0.numel())
        k1 = min(top_m, s1.numel())
        idx0 = torch.topk(s0, k=k0).indices.tolist()
        idx1 = torch.topk(s1, k=k1).indices.tolist()
        best = (float("-inf"), int(idx0[0]), int(idx1[0]))
        length = g0.length
        for a in idx0:
            c0 = g0.candidates[int(a)]
            for b in idx1:
                c1 = g1.candidates[int(b)]
                same_program = c0.init_value == c1.init_value and c0.ops[:length] == c1.ops[:length] and c0.args[:length] == c1.args[:length]
                same_states = c0.states[:length] == c1.states[:length]
                same_answer = c0.answer == c1.answer
                score = float(s0[a] + s1[b])
                if same_answer:
                    score += 0.25 * bonus
                if same_states:
                    score += 0.75 * bonus
                if same_program:
                    score += bonus
                if score > best[0]:
                    best = (score, int(a), int(b))
        selected.extend([best[1], best[2]])
    if len(groups) > usable:
        selected.append(int(learned_scores[-1].argmax().item()))
    return selected


@torch.no_grad()
def evaluate_groups(
    groups: Sequence[CandidateGroup],
    verifier: Optional[CandidateTraceVerifier],
    feature_mean: Optional[torch.Tensor],
    feature_std: Optional[torch.Tensor],
    trace_mean: Optional[torch.Tensor],
    trace_std: Optional[torch.Tensor],
    device: torch.device,
    split_name: str,
    pair_top_m: int = 25,
    pair_bonus: float = 2.0,
) -> Dict[str, float]:
    n = len(groups)
    base_indices = [group.base_index for group in groups]
    prior_indices = [int(group.priors.argmax().item()) for group in groups]
    oracle_indices = [
        int(torch.where(group.state_labels)[0][group.priors[group.state_labels].argmax()].item()) if group.state_labels.any() else group.base_index
        for group in groups
    ]
    soft_trace_indices = [int(group.features[:, FEATURE_NAMES.index("candidate_state_soft_logp_mean")].argmax().item()) for group in groups]
    out: Dict[str, float] = {
        "n": float(n),
        "oracle_found_fraction": sum(group.has_positive() for group in groups) / max(1, n),
        "avg_candidates": safe_mean([float(len(group.candidates)) for group in groups]),
        "avg_positive_candidates": safe_mean([float(group.state_labels.sum().item()) for group in groups]),
    }
    out.update(summarize_selected(groups, base_indices, "base"))
    out.update(summarize_selected(groups, prior_indices, "prior"))
    out.update(summarize_selected(groups, soft_trace_indices, "soft_trace"))
    out.update(summarize_selected(groups, oracle_indices, "oracle"))
    if verifier is not None and feature_mean is not None and feature_std is not None and trace_mean is not None and trace_std is not None:
        learned_indices: List[int] = []
        learned_scores: List[torch.Tensor] = []
        verifier.eval()
        for group in groups:
            idx, scores = select_index_with_model(verifier, group, feature_mean, feature_std, trace_mean, trace_std, device)
            learned_indices.append(idx)
            learned_scores.append(scores)
        out.update(summarize_selected(groups, learned_indices, "learned"))
        denom = out["oracle_executor_accuracy"] - out["base_executor_accuracy"]
        out["learned_oracle_gap_recovered"] = (
            (out["learned_executor_accuracy"] - out["base_executor_accuracy"]) / denom if abs(denom) > 1e-9 else math.nan
        )
        if "paired" in split_name:
            pair_indices = pair_consistency_indices(groups, learned_scores, pair_top_m, pair_bonus)
            out.update(summarize_selected(groups, pair_indices, "pair_rerank"))
            out.update(pair_metrics(groups, learned_indices, "learned"))
            out.update(pair_metrics(groups, pair_indices, "pair_rerank"))
            out.update(pair_metrics(groups, oracle_indices, "oracle"))
            out.update(pair_metrics(groups, base_indices, "base"))
    return out


def candidate_edit_distance(a: Candidate, b: Candidate, length: int) -> int:
    dist = int(int(a.init_value) != int(b.init_value))
    for t in range(length):
        dist += int(int(a.ops[t]) != int(b.ops[t]))
        dist += int(int(a.args[t]) != int(b.args[t]))
    return dist


def iterative_candidate_index(
    group: CandidateGroup,
    scores: torch.Tensor,
    k_steps: int,
    max_step_edits: int,
    accept_margin: float,
) -> Tuple[int, int, int]:
    current_idx = int(group.base_index)
    steps_used = 0
    total_step_edits = 0
    for _ in range(int(k_steps)):
        current = group.candidates[current_idx]
        allowed: List[int] = []
        for idx, cand in enumerate(group.candidates):
            if candidate_edit_distance(current, cand, group.length) <= int(max_step_edits):
                allowed.append(idx)
        if not allowed:
            break
        best_idx = max(allowed, key=lambda idx: float(scores[int(idx)]))
        if best_idx == current_idx:
            break
        if float(scores[best_idx]) <= float(scores[current_idx]) + float(accept_margin):
            break
        total_step_edits += candidate_edit_distance(current, group.candidates[best_idx], group.length)
        current_idx = int(best_idx)
        steps_used += 1
    return current_idx, steps_used, total_step_edits


@torch.no_grad()
def evaluate_iterative_candidate_groups(
    groups: Sequence[CandidateGroup],
    verifier: CandidateTraceVerifier,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    trace_mean: torch.Tensor,
    trace_std: torch.Tensor,
    device: torch.device,
    split_name: str,
    k_steps: int,
    max_step_edits: int,
    accept_margin: float,
    pair_top_m: int = 25,
    pair_bonus: float = 2.0,
) -> Dict[str, float]:
    base_indices = [group.base_index for group in groups]
    oracle_indices = [
        int(torch.where(group.state_labels)[0][group.priors[group.state_labels].argmax()].item()) if group.state_labels.any() else group.base_index
        for group in groups
    ]
    learned_indices: List[int] = []
    iter_indices: List[int] = []
    learned_scores: List[torch.Tensor] = []
    steps: List[float] = []
    step_edits: List[float] = []
    verifier.eval()
    for group in groups:
        idx, scores = select_index_with_model(verifier, group, feature_mean, feature_std, trace_mean, trace_std, device)
        learned_indices.append(idx)
        learned_scores.append(scores)
        iter_idx, used, edits = iterative_candidate_index(group, scores, k_steps, max_step_edits, accept_margin)
        iter_indices.append(iter_idx)
        steps.append(float(used))
        step_edits.append(float(edits))
    out: Dict[str, float] = {
        "n": float(len(groups)),
        "k_steps": float(k_steps),
        "candidate_max_step_edits": float(max_step_edits),
        "candidate_accept_margin": float(accept_margin),
        "oracle_found_fraction": sum(group.has_positive() for group in groups) / max(1, len(groups)),
        "avg_candidates": safe_mean([float(len(group.candidates)) for group in groups]),
        "avg_positive_candidates": safe_mean([float(group.state_labels.sum().item()) for group in groups]),
        "iter_steps_used_mean": safe_mean(steps),
        "iter_step_edits_mean": safe_mean(step_edits),
    }
    out.update(summarize_selected(groups, base_indices, "base"))
    out.update(summarize_selected(groups, learned_indices, "learned"))
    out.update(summarize_selected(groups, iter_indices, "iter"))
    out.update(summarize_selected(groups, oracle_indices, "oracle"))
    denom = out["oracle_executor_accuracy"] - out["base_executor_accuracy"]
    out["learned_oracle_gap_recovered"] = (
        (out["learned_executor_accuracy"] - out["base_executor_accuracy"]) / denom if abs(denom) > 1e-9 else math.nan
    )
    out["iter_oracle_gap_recovered"] = (
        (out["iter_executor_accuracy"] - out["base_executor_accuracy"]) / denom if abs(denom) > 1e-9 else math.nan
    )
    if "paired" in split_name:
        pair_indices = pair_consistency_indices(groups, learned_scores, pair_top_m, pair_bonus)
        out.update(summarize_selected(groups, pair_indices, "pair_rerank"))
        out.update(pair_metrics(groups, base_indices, "base"))
        out.update(pair_metrics(groups, learned_indices, "learned"))
        out.update(pair_metrics(groups, iter_indices, "iter"))
        out.update(pair_metrics(groups, pair_indices, "pair_rerank"))
        out.update(pair_metrics(groups, oracle_indices, "oracle"))
    return out


def rows_from_metrics(metrics_by_split: Dict[str, Dict[str, float]], run_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split, metrics in metrics_by_split.items():
        row = {"run": run_name, "split": split}
        row.update(metrics)
        rows.append(row)
    return rows


def write_summary(path: Path, rows: Sequence[Dict[str, Any]], best_epoch: int) -> None:
    def pct(v: Any) -> str:
        return "n/a" if not isinstance(v, (float, int)) or math.isnan(float(v)) else f"{100.0 * float(v):.1f}%"

    lines = [
        "# Iterative Repair Policy Analysis Summary",
        "",
        f"Best policy epoch: {best_epoch}",
        "",
        "| split | K | base | iterative policy | oracle | gap recovered | steps used | iter edits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {split} | {k} | {base} | {editor} | {oracle} | {gap} | {steps} | {edits} |".format(
                split=row["split"],
                k=int(row.get("k_steps", 0)) if isinstance(row.get("k_steps"), (float, int)) else "n/a",
                base=pct(row.get("base_executor_accuracy")),
                editor=pct(row.get("iter_executor_accuracy")),
                oracle=pct(row.get("oracle_executor_accuracy")),
                gap=pct(row.get("iter_oracle_gap_recovered")),
                steps=f"{float(row.get('iter_steps_used_mean', math.nan)):.2f}"
                if isinstance(row.get("iter_steps_used_mean"), (float, int))
                else "n/a",
                edits=f"{float(row.get('iter_total_iter_edits_mean', row.get('iter_step_edits_mean', math.nan))):.2f}"
                if isinstance(row.get("iter_total_iter_edits_mean", row.get("iter_step_edits_mean")), (float, int))
                else "n/a",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train an iterative sparse repair policy for Qwen-compiled numeric programs")
    p.add_argument("--run_name", type=str, default="main_iterative_repair_policy_s384")
    p.add_argument("--method", type=str, default="direct_policy", choices=["direct_policy", "candidate_scorer"])
    p.add_argument("--compiler_checkpoint", type=str, default=str(DEFAULT_COMPILER_CHECKPOINT))
    p.add_argument("--train_examples", type=int, default=512)
    p.add_argument("--val_examples", type=int, default=128)
    p.add_argument("--val_pairs", type=int, default=0)
    p.add_argument("--eval_examples", type=int, default=256)
    p.add_argument("--eval_pairs", type=int, default=256)
    p.add_argument("--train_length", type=int, default=24)
    p.add_argument("--eval_length", type=int, default=24)
    p.add_argument("--train_template_modes", type=str, default="standard,paraphrase")
    p.add_argument("--repair_topk", type=int, default=3)
    p.add_argument("--repair_max_edits", type=int, default=2)
    p.add_argument("--repair_max_pair_arg_slots", type=int, default=24)
    p.add_argument("--qwen_batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--editor_d_model", type=int, default=128)
    p.add_argument("--editor_layers", type=int, default=3)
    p.add_argument("--editor_heads", type=int, default=4)
    p.add_argument("--editor_ff_mult", type=int, default=4)
    p.add_argument("--editor_dropout", type=float, default=0.05)
    p.add_argument("--editor_epochs", type=int, default=18)
    p.add_argument("--editor_lr", type=float, default=1e-3)
    p.add_argument("--editor_weight_decay", type=float, default=1e-4)
    p.add_argument("--trace_d_model", type=int, default=128)
    p.add_argument("--trace_layers", type=int, default=2)
    p.add_argument("--trace_heads", type=int, default=4)
    p.add_argument("--trace_ff_mult", type=int, default=4)
    p.add_argument("--verifier_dropout", type=float, default=0.05)
    p.add_argument("--verifier_epochs", type=int, default=12)
    p.add_argument("--verifier_lr", type=float, default=1e-3)
    p.add_argument("--verifier_weight_decay", type=float, default=1e-4)
    p.add_argument("--base_positive_group_weight", type=float, default=0.25)
    p.add_argument("--repairable_group_weight", type=float, default=8.0)
    p.add_argument("--no_positive_group_weight", type=float, default=1.0)
    p.add_argument("--no_positive_base_weight", type=float, default=0.2)
    p.add_argument("--candidate_accept_margin", type=float, default=0.0)
    p.add_argument("--editor_target_mode", type=str, default="oracle_or_gold", choices=["oracle_or_gold", "oracle_or_base", "gold"])
    p.add_argument("--changed_slot_weight", type=float, default=4.0)
    p.add_argument("--base_copy_logit_bias", type=float, default=5.0)
    p.add_argument("--edit_gate_loss_weight", type=float, default=1.0)
    p.add_argument("--edit_value_loss_weight", type=float, default=1.0)
    p.add_argument("--edit_gate_pos_weight", type=float, default=8.0)
    p.add_argument("--unchanged_value_loss_weight", type=float, default=0.0)
    p.add_argument("--edit_threshold_grid", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    p.add_argument("--train_select_k_values", type=str, default="1,2,3")
    p.add_argument("--selection_split", type=str, default="val_len24")
    p.add_argument("--eval_k_values", type=str, default="0,1,2,3")
    p.add_argument("--max_edits_per_iteration", type=int, default=2)
    p.add_argument("--augment_cases_per_group", type=int, default=1)
    p.add_argument("--augment_top_pool", type=int, default=12)
    p.add_argument("--augment_repairable_only", action="store_true")
    p.add_argument("--include_target_noedit_case", action="store_true")
    p.add_argument("--stats_candidate_traces", type=int, default=4)
    p.add_argument("--init_loss_weight", type=float, default=1.0)
    p.add_argument("--op_loss_weight", type=float, default=1.0)
    p.add_argument("--arg_loss_weight", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--eval_seed", type=int, default=94001)
    return p


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    tokenizer, model, direct, compiler, run_args = load_fixed_compiler(args, device)
    executor = TransitionExecutor(run_args["modulus"], device)
    datasets = make_datasets(tokenizer, run_args, args)

    groups_by_split: Dict[str, List[CandidateGroup]] = {}
    for split, dataset in datasets.items():
        groups_by_split[split] = build_groups_for_dataset(split, dataset, tokenizer, model, compiler, executor, run_args, args, device)
    del model, direct, compiler, executor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.selection_split not in groups_by_split:
        raise KeyError(f"selection_split {args.selection_split!r} not in available splits {sorted(groups_by_split)}")
    metrics_by_split: Dict[str, Dict[str, float]] = {}
    eval_k_values = parse_int_grid(args.eval_k_values)
    best_epoch = 0
    best_threshold = 0.0
    best_k = 0
    artifact_name = ""
    artifact_path = ""
    train_log_name = "train_log.csv"
    if args.method == "candidate_scorer":
        verifier, feature_mean, feature_std, trace_mean, trace_std, train_rows = train_verifier(
            groups_by_split["train_len24"], groups_by_split[args.selection_split], args, run_dir, device
        )
        best_epoch = int(train_rows[-1].get("epoch", 0)) if train_rows else 0
        ckpt = torch.load(CHECKPOINT_ROOT / run_dir.name / "candidate_trace_verifier.pt", map_location="cpu")
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        artifact_name = "candidate_trace_verifier_checkpoint"
        artifact_path = str(CHECKPOINT_ROOT / run_dir.name / "candidate_trace_verifier.pt")
        train_log_name = "verifier_train_log.csv"
        for split, groups in groups_by_split.items():
            for k_steps in eval_k_values:
                key = f"{split}_k{k_steps}"
                metrics_by_split[key] = evaluate_iterative_candidate_groups(
                    groups,
                    verifier,
                    feature_mean,
                    feature_std,
                    trace_mean,
                    trace_std,
                    device,
                    split,
                    k_steps,
                    int(args.max_edits_per_iteration),
                    float(args.candidate_accept_margin),
                )
                print(
                    f"[metrics] {split} k={k_steps} base={100.0 * metrics_by_split[key]['base_executor_accuracy']:.1f}% "
                    f"iter={100.0 * metrics_by_split[key].get('iter_executor_accuracy', math.nan):.1f}% "
                    f"learned={100.0 * metrics_by_split[key].get('learned_executor_accuracy', math.nan):.1f}% "
                    f"oracle={100.0 * metrics_by_split[key]['oracle_executor_accuracy']:.1f}%",
                    flush=True,
                )
    else:
        policy, trace_mean, trace_std, train_rows, best_threshold, best_k, best_epoch = train_iterative_policy(
            groups_by_split["train_len24"], groups_by_split[args.selection_split], args, run_dir, run_args, device
        )
        artifact_name = "iterative_repair_policy_checkpoint"
        artifact_path = str(CHECKPOINT_ROOT / run_dir.name / "iterative_repair_policy.pt")
        train_log_name = "editor_train_log.csv"
        for split, groups in groups_by_split.items():
            for k_steps in eval_k_values:
                key = f"{split}_k{k_steps}"
                metrics_by_split[key] = evaluate_iterative_policy_groups(
                    groups,
                    policy,
                    trace_mean,
                    trace_std,
                    device,
                    split,
                    best_threshold,
                    k_steps,
                    int(args.max_edits_per_iteration),
                )
                print(
                    f"[metrics] {split} k={k_steps} base={100.0 * metrics_by_split[key]['base_executor_accuracy']:.1f}% "
                    f"iter={100.0 * metrics_by_split[key].get('iter_executor_accuracy', math.nan):.1f}% "
                    f"oracle={100.0 * metrics_by_split[key]['oracle_executor_accuracy']:.1f}%",
                    flush=True,
                )

    rows = rows_from_metrics(metrics_by_split, args.run_name)
    write_csv(run_dir / train_log_name, train_rows)
    write_csv(run_dir / "metrics.csv", rows)
    write_csv(ANALYSIS / "final_metrics.csv", rows)
    write_summary(ANALYSIS / "summary.md", rows, int(best_epoch))

    manifest = [
        {
            "run": args.run_name,
            "artifact": "fixed_compiler_checkpoint",
            "path": str(Path(args.compiler_checkpoint)),
        },
        {
            "run": args.run_name,
            "artifact": artifact_name,
            "path": artifact_path,
        },
    ]
    write_csv(ROOT / "checkpoint_manifest.csv", manifest)
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "args": vars(args),
        "compiler_args": {
            key: run_args.get(key)
            for key in [
                "model_id",
                "modulus",
                "max_steps",
                "head_width",
                "rank_temperature",
                "arg_reader_mode",
                "arg_window",
                "arg_distance_temperature",
            ]
        },
        "feature_names": FEATURE_NAMES,
        "trace_feature_names": TRACE_FEATURE_NAMES,
        "editor_type": str(args.method),
        "best_editor_epoch": int(best_epoch),
        "best_editor_threshold": float(best_threshold),
        "best_editor_k": int(best_k),
        "train_seconds": time.time() - t0,
    }
    json_dump(run_dir / "results.json", {"metadata": metadata, "metrics": metrics_by_split, "train_log": train_rows})
    print(f"[done] {run_dir} seconds={metadata['train_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
