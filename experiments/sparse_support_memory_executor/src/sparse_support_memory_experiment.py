#!/usr/bin/env python3
"""Sparse support-memory executor for modular filter programs.

The runtime state is an explicit bounded set of weighted `(A,B)` support slots.
Arithmetic updates move each active slot. Observation filters delete
inconsistent slots. The experiment measures how much slot capacity is required
for exact belief-state execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


OP_NAMES = [
    "A=A+c",
    "A=A-c",
    "B=B+c",
    "B=B-c",
    "A=A+B",
    "B=B+A",
    "A=A-B",
    "B=B-A",
    "OBS_A_BUCKET",
    "OBS_B_BUCKET",
]
CONST_A_OPS = {0, 1}
CONST_B_OPS = {2, 3}
OBS_A = 8
OBS_B = 9

QUERY_NAMES = ["A", "B", "A_PLUS_B", "A_MINUS_B"]


@dataclass
class ProgramBatch:
    ops: torch.Tensor
    args: torch.Tensor
    lengths: torch.Tensor
    init_delta: torch.Tensor
    query_type: torch.Tensor
    query_label: torch.Tensor
    query_probs: torch.Tensor
    target_probs: torch.Tensor
    support_size: torch.Tensor


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return vals


def parse_query_types(text: str) -> List[int]:
    if text == "all":
        return list(range(len(QUERY_NAMES)))
    aliases = {name: i for i, name in enumerate(QUERY_NAMES)}
    aliases.update({"a": 0, "b": 1, "sum": 2, "diff": 3})
    out: List[int] = []
    for part in text.split(","):
        key = part.strip()
        if not key:
            continue
        val = int(key) if key.isdigit() else aliases.get(key.upper(), aliases.get(key.lower(), -1))
        if val < 0 or val >= len(QUERY_NAMES):
            raise argparse.ArgumentTypeError(f"unknown query type: {key}")
        out.append(val)
    if not out:
        raise argparse.ArgumentTypeError("expected query type list")
    return out


def apply_arithmetic(a: int, b: int, op: int, arg: int, modulus: int) -> Tuple[int, int]:
    if op == 0:
        a = (a + arg) % modulus
    elif op == 1:
        a = (a - arg) % modulus
    elif op == 2:
        b = (b + arg) % modulus
    elif op == 3:
        b = (b - arg) % modulus
    elif op == 4:
        a = (a + b) % modulus
    elif op == 5:
        b = (b + a) % modulus
    elif op == 6:
        a = (a - b) % modulus
    elif op == 7:
        b = (b - a) % modulus
    else:
        raise ValueError(f"not an arithmetic op: {op}")
    return a, b


def states_to_probs(states: Sequence[Tuple[int, int]], modulus: int, device: torch.device) -> torch.Tensor:
    probs = torch.zeros(modulus * modulus, dtype=torch.float32, device=device)
    if not states:
        raise ValueError("empty support")
    weight = 1.0 / len(states)
    for a, b in states:
        probs[a * modulus + b] += weight
    return probs


def query_value(a: int, b: int, query_type: int, modulus: int) -> int:
    if query_type == 0:
        return a
    if query_type == 1:
        return b
    if query_type == 2:
        return (a + b) % modulus
    if query_type == 3:
        return (a - b) % modulus
    raise ValueError(query_type)


def states_to_query_probs(
    states: Sequence[Tuple[int, int]],
    query_type: int,
    modulus: int,
    device: torch.device,
) -> torch.Tensor:
    probs = torch.zeros(modulus, dtype=torch.float32, device=device)
    if not states:
        raise ValueError("empty support")
    weight = 1.0 / len(states)
    for a, b in states:
        probs[query_value(a, b, query_type, modulus)] += weight
    return probs


class FilterProgramGenerator:
    def __init__(
        self,
        modulus: int,
        observe_mod: int,
        observe_prob: float,
        seed: int,
        op_family: str,
        query_family: str,
    ) -> None:
        self.modulus = int(modulus)
        self.observe_mod = int(observe_mod)
        self.observe_prob = float(observe_prob)
        self.rng = random.Random(seed)
        if op_family == "full":
            self.arith_ops = list(range(8))
        elif op_family == "const":
            self.arith_ops = [0, 1, 2, 3]
        elif op_family == "cross":
            self.arith_ops = [4, 5, 6, 7]
        else:
            raise ValueError(f"unknown op_family {op_family}")
        if query_family == "all":
            self.query_types = list(range(len(QUERY_NAMES)))
        elif query_family == "marginal":
            self.query_types = [0, 1]
        elif query_family == "relational":
            self.query_types = [2, 3]
        else:
            raise ValueError(f"unknown query_family {query_family}")

    def initial_states(self, delta: int) -> List[Tuple[int, int]]:
        return [(a, (a + delta) % self.modulus) for a in range(self.modulus)]

    def choose_observation(self, states: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
        op = OBS_A if self.rng.random() < 0.5 else OBS_B
        idx = 0 if op == OBS_A else 1
        residues = sorted({state[idx] % self.observe_mod for state in states})
        return op, self.rng.choice(residues)

    def apply_step(self, states: Sequence[Tuple[int, int]], op: int, arg: int) -> List[Tuple[int, int]]:
        if op < 8:
            return [apply_arithmetic(a, b, op, arg, self.modulus) for a, b in states]
        if op == OBS_A:
            return [(a, b) for a, b in states if a % self.observe_mod == arg]
        if op == OBS_B:
            return [(a, b) for a, b in states if b % self.observe_mod == arg]
        raise ValueError(op)

    def batch(
        self,
        batch_size: int,
        min_len: int,
        max_len: int,
        trace_k: int,
        device: torch.device,
        fixed_len: Optional[int] = None,
        fixed_query_type: Optional[int] = None,
    ) -> ProgramBatch:
        if fixed_len is not None:
            min_len = max_len = int(fixed_len)
        lengths = [self.rng.randint(min_len, max_len) for _ in range(batch_size)]
        program_len = max(max_len, max(lengths), trace_k)
        pair_dim = self.modulus * self.modulus
        ops = torch.zeros(batch_size, program_len, dtype=torch.long)
        args = torch.zeros(batch_size, program_len, dtype=torch.long)
        init_delta = torch.empty(batch_size, dtype=torch.long)
        query_type = torch.empty(batch_size, dtype=torch.long)
        query_label = torch.empty(batch_size, dtype=torch.long)
        query_probs = torch.zeros(batch_size, self.modulus, dtype=torch.float32, device=device)
        target_probs = torch.zeros(batch_size, trace_k + 1, pair_dim, dtype=torch.float32, device=device)
        support_size = torch.zeros(batch_size, trace_k + 1, dtype=torch.float32, device=device)

        for i, length in enumerate(lengths):
            delta = self.rng.randrange(self.modulus)
            init_delta[i] = delta
            states = self.initial_states(delta)
            target_probs[i, 0] = states_to_probs(states, self.modulus, device)
            support_size[i, 0] = len(states)
            sampled_ops: List[int] = []
            sampled_args: List[int] = []
            cur_states = list(states)
            for _ in range(length):
                if self.rng.random() < self.observe_prob:
                    op, arg = self.choose_observation(cur_states)
                else:
                    op = self.rng.choice(self.arith_ops)
                    arg = self.rng.randint(1, self.modulus - 1) if op in CONST_A_OPS or op in CONST_B_OPS else 0
                sampled_ops.append(op)
                sampled_args.append(arg)
                cur_states = self.apply_step(cur_states, op, arg)
                if not cur_states:
                    raise RuntimeError("generator produced empty support")

            final_states = list(cur_states)
            q_type = self.rng.choice(self.query_types) if fixed_query_type is None else int(fixed_query_type)
            query_type[i] = q_type
            sampled_state = self.rng.choice(final_states)
            query_label[i] = query_value(sampled_state[0], sampled_state[1], q_type, self.modulus)
            query_probs[i] = states_to_query_probs(final_states, q_type, self.modulus, device)

            cur_states = list(states)
            for t in range(program_len):
                if t < length:
                    op = sampled_ops[t]
                    arg = sampled_args[t]
                    ops[i, t] = op
                    args[i, t] = arg
                    cur_states = self.apply_step(cur_states, op, arg)
                if t + 1 <= trace_k:
                    target_probs[i, t + 1] = states_to_probs(cur_states, self.modulus, device)
                    support_size[i, t + 1] = len(cur_states)

        return ProgramBatch(
            ops=ops.to(device),
            args=args.to(device),
            lengths=torch.tensor(lengths, dtype=torch.long, device=device),
            init_delta=init_delta.to(device),
            query_type=query_type.to(device),
            query_label=query_label.to(device),
            query_probs=query_probs,
            target_probs=target_probs,
            support_size=support_size,
        )


def query_value_index(modulus: int, device: torch.device) -> torch.Tensor:
    a = torch.arange(modulus, device=device).repeat_interleave(modulus)
    b = torch.arange(modulus, device=device).repeat(modulus)
    return torch.stack([a, b, (a + b) % modulus, (a - b) % modulus], dim=0)


def query_probs_from_pair_probs(pair_probs: torch.Tensor, query_type: torch.Tensor, modulus: int) -> torch.Tensor:
    pair_dim = modulus * modulus
    if pair_probs.shape[-1] != pair_dim:
        raise ValueError(f"expected pair dim {pair_dim}, got {pair_probs.shape[-1]}")
    idx = query_value_index(modulus, pair_probs.device)
    out = torch.zeros(pair_probs.shape[0], modulus, dtype=pair_probs.dtype, device=pair_probs.device)
    for q_type in range(len(QUERY_NAMES)):
        mask = query_type == q_type
        if bool(mask.any()):
            scatter_idx = idx[q_type].unsqueeze(0).expand(int(mask.sum()), pair_dim)
            projected = torch.zeros(int(mask.sum()), modulus, dtype=pair_probs.dtype, device=pair_probs.device)
            projected.scatter_add_(1, scatter_idx, pair_probs[mask])
            out[mask] = projected
    return out


def belief_metrics_from_pair_probs(pair_probs: torch.Tensor, target_probs: torch.Tensor) -> Dict[str, float]:
    mask = target_probs > 0
    target_mass = (pair_probs * mask.float()).sum(dim=-1)
    log_probs = pair_probs.clamp_min(1e-9).log()
    target_nll = -(target_probs * log_probs).sum(dim=-1)
    pred = pair_probs.argmax(dim=-1)
    top1 = mask.gather(1, pred.unsqueeze(1)).squeeze(1).float()
    return {
        "decoder_belief_target_mass": float(target_mass.mean().detach().cpu()),
        "decoder_belief_target_nll": float(target_nll.mean().detach().cpu()),
        "decoder_belief_top1_on_support": float(top1.mean().detach().cpu()),
        "mean_decoder_belief_support_size": float(mask.sum(dim=-1).float().mean().detach().cpu()),
    }


def query_metrics_from_pair_probs(
    pair_probs: torch.Tensor,
    query_type: torch.Tensor,
    target_query: torch.Tensor,
    modulus: int,
) -> Dict[str, float]:
    query_probs = query_probs_from_pair_probs(pair_probs, query_type, modulus)
    log_probs = query_probs.clamp_min(1e-9).log()
    mask = target_query > 0
    target_mass = (query_probs * mask.float()).sum(dim=-1)
    target_nll = -(target_query * log_probs).sum(dim=-1)
    pred = query_probs.argmax(dim=-1)
    top1 = mask.gather(1, pred.unsqueeze(1)).squeeze(1).float()
    return {
        "decoder_query_target_mass": float(target_mass.mean().detach().cpu()),
        "decoder_query_target_nll": float(target_nll.mean().detach().cpu()),
        "decoder_query_top1_on_support": float(top1.mean().detach().cpu()),
        "mean_decoder_query_support_size": float(mask.sum(dim=-1).float().mean().detach().cpu()),
    }


class SparseSupportExecutor:
    def __init__(self, modulus: int, slot_capacity: int, init_strategy: str = "stride") -> None:
        self.modulus = int(modulus)
        self.slot_capacity = int(slot_capacity)
        self.init_strategy = init_strategy
        if self.slot_capacity < 1:
            raise ValueError("slot_capacity must be positive")
        if init_strategy != "stride":
            raise ValueError(f"unsupported init_strategy {init_strategy}")

    def initial_slots(self, init_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.modulus
        s = self.slot_capacity
        device = init_delta.device
        slot_ids = torch.arange(s, device=device)
        if s >= p:
            a_slots = slot_ids % p
            active_row = slot_ids < p
            denom = float(p)
        else:
            a_slots = torch.div(slot_ids * p, s, rounding_mode="floor")
            active_row = torch.ones(s, dtype=torch.bool, device=device)
            denom = float(s)
        b_slots = (a_slots.unsqueeze(0) + init_delta.unsqueeze(1)) % p
        pair_slots = a_slots.unsqueeze(0) * p + b_slots
        pair_slots = pair_slots.expand(init_delta.shape[0], s).clone()
        active = active_row.unsqueeze(0).expand(init_delta.shape[0], s).clone()
        weights = active.float() / denom
        return pair_slots, weights, active

    def pair_distribution(
        self,
        pair_slots: torch.Tensor,
        weights: torch.Tensor,
        active: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = pair_slots.shape[0]
        pair_dim = self.modulus * self.modulus
        masked_weights = weights * active.float()
        probs = torch.zeros(batch_size, pair_dim, dtype=torch.float32, device=pair_slots.device)
        probs.scatter_add_(1, pair_slots.clamp(0, pair_dim - 1), masked_weights)
        denom = probs.sum(dim=-1, keepdim=True)
        empty = denom.squeeze(1) <= 0
        probs = probs / denom.clamp_min(1e-9)
        if bool(empty.any()):
            probs[empty] = 1.0 / pair_dim
        stats = {
            "empty": empty.float(),
            "active_slots": active.sum(dim=-1).float(),
            "total_slot_weight": masked_weights.sum(dim=-1),
        }
        return probs, stats

    def step(
        self,
        pair_slots: torch.Tensor,
        weights: torch.Tensor,
        active: torch.Tensor,
        op: torch.Tensor,
        arg: torch.Tensor,
        step_mask: torch.Tensor,
        observe_mod: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.modulus
        a = pair_slots // p
        b = pair_slots % p
        op2 = op.unsqueeze(1)
        arg2 = arg.unsqueeze(1)
        step2 = step_mask.unsqueeze(1)
        new_a = a
        new_b = b

        new_a = torch.where((op2 == 0) & step2, (a + arg2) % p, new_a)
        new_a = torch.where((op2 == 1) & step2, (a - arg2) % p, new_a)
        new_b = torch.where((op2 == 2) & step2, (b + arg2) % p, new_b)
        new_b = torch.where((op2 == 3) & step2, (b - arg2) % p, new_b)
        new_a = torch.where((op2 == 4) & step2, (a + b) % p, new_a)
        new_b = torch.where((op2 == 5) & step2, (b + a) % p, new_b)
        new_a = torch.where((op2 == 6) & step2, (a - b) % p, new_a)
        new_b = torch.where((op2 == 7) & step2, (b - a) % p, new_b)

        obs_a = (op2 == OBS_A) & step2
        obs_b = (op2 == OBS_B) & step2
        active = torch.where(obs_a, active & ((a % observe_mod) == arg2), active)
        active = torch.where(obs_b, active & ((b % observe_mod) == arg2), active)

        new_pair = new_a * p + new_b
        pair_slots = torch.where(step2, new_pair, pair_slots)
        return pair_slots, weights, active

    def execute(
        self,
        batch: ProgramBatch,
        k_values: Sequence[int],
        observe_mod: int,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict[str, torch.Tensor]]]:
        pair_slots, weights, active = self.initial_slots(batch.init_delta)
        collect = set(int(k) for k in k_values)
        max_k = max(collect)
        preds: Dict[int, torch.Tensor] = {}
        stats: Dict[int, Dict[str, torch.Tensor]] = {}
        if 0 in collect:
            preds[0], stats[0] = self.pair_distribution(pair_slots, weights, active)
        for t in range(max_k):
            step_mask = batch.lengths > t
            pair_slots, weights, active = self.step(
                pair_slots,
                weights,
                active,
                batch.ops[:, t],
                batch.args[:, t],
                step_mask,
                int(observe_mod),
            )
            k = t + 1
            if k in collect:
                preds[k], stats[k] = self.pair_distribution(pair_slots, weights, active)
        return preds, stats


def row_sums_template(k_values: Sequence[int]) -> Dict[int, Dict[str, float]]:
    keys = [
        "n",
        "dq_mass",
        "dq_nll",
        "dq_top1",
        "dq_support",
        "db_mass",
        "db_nll",
        "db_top1",
        "db_support",
        "empty_rate",
        "active_slots",
        "total_slot_weight",
        "target_support_size",
    ]
    return {int(k): {key: 0.0 for key in keys} for k in k_values}


@torch.no_grad()
def evaluate_sparse(
    executor: SparseSupportExecutor,
    generator: FilterProgramGenerator,
    args: argparse.Namespace,
    device: torch.device,
    lengths: Sequence[int],
    k_values: Sequence[int],
    examples_per_length: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    max_k = max(k_values)
    batch_size = min(args.eval_batch_size, examples_per_length)
    eval_query_types = parse_query_types(args.eval_query_types)
    for length in lengths:
        trace_k = max(length, max_k)
        for q_type in eval_query_types:
            sums = row_sums_template(k_values)
            while sums[k_values[0]]["n"] < examples_per_length:
                bsz = min(batch_size, examples_per_length - int(sums[k_values[0]]["n"]))
                batch = generator.batch(
                    bsz,
                    length,
                    length,
                    trace_k,
                    device=device,
                    fixed_len=length,
                    fixed_query_type=q_type,
                )
                preds, stats_by_k = executor.execute(batch, k_values, observe_mod=args.observe_mod)
                final_pair_target = batch.target_probs[:, length]
                final_support_size = batch.support_size[:, length]
                for k in k_values:
                    pred = preds[int(k)]
                    qm = query_metrics_from_pair_probs(pred, batch.query_type, batch.query_probs, args.modulus)
                    bm = belief_metrics_from_pair_probs(pred, final_pair_target)
                    st = stats_by_k[int(k)]
                    n = float(bsz)
                    sums[k]["n"] += n
                    sums[k]["dq_mass"] += qm["decoder_query_target_mass"] * n
                    sums[k]["dq_nll"] += qm["decoder_query_target_nll"] * n
                    sums[k]["dq_top1"] += qm["decoder_query_top1_on_support"] * n
                    sums[k]["dq_support"] += qm["mean_decoder_query_support_size"] * n
                    sums[k]["db_mass"] += bm["decoder_belief_target_mass"] * n
                    sums[k]["db_nll"] += bm["decoder_belief_target_nll"] * n
                    sums[k]["db_top1"] += bm["decoder_belief_top1_on_support"] * n
                    sums[k]["db_support"] += bm["mean_decoder_belief_support_size"] * n
                    sums[k]["empty_rate"] += float(st["empty"].mean().detach().cpu()) * n
                    sums[k]["active_slots"] += float(st["active_slots"].mean().detach().cpu()) * n
                    sums[k]["total_slot_weight"] += float(st["total_slot_weight"].mean().detach().cpu()) * n
                    sums[k]["target_support_size"] += float(final_support_size.float().mean().detach().cpu()) * n
            for k in k_values:
                n = max(1.0, sums[k]["n"])
                rows.append(
                    {
                        "model": "sparse_support",
                        "variant": args.variant_name,
                        "modulus": args.modulus,
                        "observe_mod": args.observe_mod,
                        "observe_prob": args.observe_prob,
                        "op_family": args.op_family,
                        "query_family": args.query_family,
                        "slot_capacity": args.slot_capacity,
                        "init_strategy": args.init_strategy,
                        "length": int(length),
                        "query_type": QUERY_NAMES[q_type],
                        "k": int(k),
                        "n": int(n),
                        "decoder_query_target_mass": sums[k]["dq_mass"] / n,
                        "decoder_query_target_nll": sums[k]["dq_nll"] / n,
                        "decoder_query_top1_on_support": sums[k]["dq_top1"] / n,
                        "mean_decoder_query_support_size": sums[k]["dq_support"] / n,
                        "decoder_belief_target_mass": sums[k]["db_mass"] / n,
                        "decoder_belief_target_nll": sums[k]["db_nll"] / n,
                        "decoder_belief_top1_on_support": sums[k]["db_top1"] / n,
                        "mean_decoder_belief_support_size": sums[k]["db_support"] / n,
                        "empty_slot_rate": sums[k]["empty_rate"] / n,
                        "mean_active_slots": sums[k]["active_slots"] / n,
                        "mean_total_slot_weight": sums[k]["total_slot_weight"] / n,
                        "mean_target_support_size": sums[k]["target_support_size"] / n,
                    }
                )
    return rows


def write_metrics_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_eval_summary(prefix: str, rows: Sequence[Dict[str, Any]]) -> None:
    by_len: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_len.setdefault(int(row["length"]), []).append(row)
    print(prefix, flush=True)
    for length in sorted(by_len):
        chosen = [r for r in by_len[length] if int(r["k"]) >= length]
        if chosen:
            first_k = min(int(r["k"]) for r in chosen)
            chosen = [r for r in chosen if int(r["k"]) == first_k]
        else:
            chosen = by_len[length]
        qmass = sum(float(r["decoder_query_target_mass"]) for r in chosen) / max(1, len(chosen))
        bmass = sum(float(r["decoder_belief_target_mass"]) for r in chosen) / max(1, len(chosen))
        empty = sum(float(r["empty_slot_rate"]) for r in chosen) / max(1, len(chosen))
        active = sum(float(r["mean_active_slots"]) for r in chosen) / max(1, len(chosen))
        k_label = min(int(r["k"]) for r in chosen)
        print(
            f"  L={length} K={k_label}: dqmass={qmass*100:.1f}% "
            f"dbmass={bmass*100:.1f}% empty={empty*100:.1f}% active={active:.1f}",
            flush=True,
        )


def write_results_json(path: Path, args: argparse.Namespace, rows: Sequence[Dict[str, Any]], elapsed: float) -> None:
    result = {
        "args": vars(args),
        "elapsed_seconds": elapsed,
        "eval": [{"step": 0, "rows": list(rows)}],
        "checkpoints": [],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    t0 = time.time()
    eval_lengths = parse_int_list(args.eval_lengths)
    eval_k = parse_int_list(args.eval_k)
    generator = FilterProgramGenerator(
        args.modulus,
        args.observe_mod,
        args.observe_prob,
        args.seed + 10_000,
        args.op_family,
        args.query_family,
    )
    executor = SparseSupportExecutor(args.modulus, args.slot_capacity, args.init_strategy)
    rows = evaluate_sparse(executor, generator, args, device, eval_lengths, eval_k, args.eval_examples)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(out / "metrics_final.csv", rows)
    write_results_json(out / "results.json", args, rows, time.time() - t0)
    print_eval_summary(f"[eval/{args.variant_name} final]", rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sparse support-memory executor")
    p.add_argument("--variant_name", type=str, default="sparse_support")
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--modulus", type=int, default=11)
    p.add_argument("--observe_mod", type=int, default=4)
    p.add_argument("--observe_prob", type=float, default=0.3)
    p.add_argument("--op_family", choices=["full", "const", "cross"], default="full")
    p.add_argument("--query_family", choices=["all", "marginal", "relational"], default="all")
    p.add_argument("--slot_capacity", type=int, default=11)
    p.add_argument("--init_strategy", choices=["stride"], default="stride")
    p.add_argument("--eval_lengths", type=str, default="3,6,9,12")
    p.add_argument("--eval_k", type=str, default="0,1,2,3,6,9,12")
    p.add_argument("--eval_query_types", type=str, default="all")
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--eval_examples", type=int, default=512)
    p.add_argument("--output_dir", type=str, default="experiments/sparse_support_memory_executor/runs/sparse_support")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
