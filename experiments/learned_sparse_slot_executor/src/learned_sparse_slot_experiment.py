#!/usr/bin/env python3
"""Learned sparse-slot executor for modular belief programs.

The model keeps a bounded set of weighted slots. Each slot represents a
distribution over one candidate A value and one candidate B value. The decoded
belief is the weighted mixture of slot-local pair distributions.
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
import torch.nn as nn
import torch.nn.functional as F


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
    flat = pair_probs.reshape(-1, pair_dim)
    if pair_probs.dim() == 2:
        flat_q = query_type.reshape(-1)
    elif pair_probs.dim() == 3:
        flat_q = query_type[:, None].expand(pair_probs.shape[0], pair_probs.shape[1]).reshape(-1)
    else:
        raise ValueError(f"unsupported pair_probs rank {pair_probs.dim()}")
    idx = query_value_index(modulus, pair_probs.device)
    out = torch.zeros(flat.shape[0], modulus, dtype=pair_probs.dtype, device=pair_probs.device)
    for q_type in range(len(QUERY_NAMES)):
        mask = flat_q == q_type
        if bool(mask.any()):
            scatter_idx = idx[q_type].unsqueeze(0).expand(int(mask.sum()), pair_dim)
            projected = torch.zeros(int(mask.sum()), modulus, dtype=pair_probs.dtype, device=pair_probs.device)
            projected.scatter_add_(1, scatter_idx, flat[mask])
            out[mask] = projected
    return out.reshape(*pair_probs.shape[:-1], modulus)


def belief_metrics(pair_probs: torch.Tensor, target_probs: torch.Tensor) -> Dict[str, float]:
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


def roll_probs(probs: torch.Tensor, shift: torch.Tensor, sign: int) -> torch.Tensor:
    m, s, p = probs.shape
    base = torch.arange(p, device=probs.device).view(1, 1, p)
    idx = (base + sign * shift.view(m, 1, 1)) % p
    out = torch.zeros_like(probs)
    out.scatter_add_(2, idx.expand(m, s, p), probs)
    return out


def combine_probs(a_probs: torch.Tensor, b_probs: torch.Tensor, mode: str) -> torch.Tensor:
    m, s, p = a_probs.shape
    pair = a_probs.unsqueeze(-1) * b_probs.unsqueeze(-2)
    ai = torch.arange(p, device=a_probs.device).view(1, 1, p, 1)
    bi = torch.arange(p, device=a_probs.device).view(1, 1, 1, p)
    if mode == "a_plus_b":
        idx = (ai + bi) % p
    elif mode == "a_minus_b":
        idx = (ai - bi) % p
    elif mode == "b_plus_a":
        idx = (bi + ai) % p
    elif mode == "b_minus_a":
        idx = (bi - ai) % p
    else:
        raise ValueError(mode)
    out = torch.zeros(m, s, p, dtype=a_probs.dtype, device=a_probs.device)
    out.scatter_add_(2, idx.expand(m, s, p, p).reshape(m, s, p * p), pair.reshape(m, s, p * p))
    return out


class LearnedSparseSlotExecutor(nn.Module):
    def __init__(
        self,
        modulus: int,
        observe_mod: int,
        slot_capacity: int,
        init_mode: str,
        transition_mode: str,
        slot_dim: int,
        hidden_dim: int,
        init_logit_scale: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.observe_mod = int(observe_mod)
        self.slot_capacity = int(slot_capacity)
        self.init_mode = init_mode
        self.transition_mode = transition_mode
        self.init_logit_scale = float(init_logit_scale)
        self.temperature = float(temperature)
        if init_mode not in {"oracle", "learned"}:
            raise ValueError(init_mode)
        if transition_mode not in {"exact", "neural"}:
            raise ValueError(transition_mode)

        if init_mode == "learned":
            self.delta_emb = nn.Embedding(self.modulus, slot_dim)
            self.init_slot_emb = nn.Embedding(self.slot_capacity, slot_dim)
            self.init_mlp = nn.Sequential(
                nn.Linear(slot_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.modulus * 2 + 1),
            )

        if transition_mode == "neural":
            self.a_value_emb = nn.Embedding(self.modulus, slot_dim)
            self.b_value_emb = nn.Embedding(self.modulus, slot_dim)
            self.op_emb = nn.Embedding(len(OP_NAMES), slot_dim)
            self.arg_emb = nn.Embedding(self.modulus, slot_dim)
            self.transition_mlp = nn.Sequential(
                nn.Linear(slot_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.modulus * 2 + 1),
            )

    def sharp_logits(self, indices: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (*indices.shape, self.modulus),
            -self.init_logit_scale,
            dtype=torch.float32,
            device=indices.device,
        )
        logits.scatter_(-1, indices.unsqueeze(-1), self.init_logit_scale)
        return logits

    def initial_state(self, init_delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = init_delta.shape[0]
        s = self.slot_capacity
        p = self.modulus
        device = init_delta.device
        if self.init_mode == "oracle":
            slot_ids = torch.arange(s, device=device)
            if s >= p:
                a_idx = slot_ids % p
                active = slot_ids < p
            else:
                a_idx = torch.div(slot_ids * p, s, rounding_mode="floor")
                active = torch.ones(s, dtype=torch.bool, device=device)
            b_idx = (a_idx.unsqueeze(0) + init_delta.unsqueeze(1)) % p
            a_idx = a_idx.unsqueeze(0).expand(bsz, s)
            a_logits = self.sharp_logits(a_idx)
            b_logits = self.sharp_logits(b_idx)
            weight_logits = torch.where(
                active.unsqueeze(0).expand(bsz, s),
                torch.zeros(bsz, s, dtype=torch.float32, device=device),
                torch.full((bsz, s), -20.0, dtype=torch.float32, device=device),
            )
            return a_logits, b_logits, weight_logits

        slot_ids = torch.arange(s, device=device)
        delta_context = self.delta_emb(init_delta).unsqueeze(1).expand(bsz, s, -1)
        slot_context = self.init_slot_emb(slot_ids).unsqueeze(0).expand(bsz, s, -1)
        raw = self.init_mlp(torch.cat([delta_context, slot_context], dim=-1))
        a_logits, b_logits, weight_logits = raw[..., :p], raw[..., p : 2 * p], raw[..., 2 * p]
        return a_logits, b_logits, weight_logits

    def slot_probs(
        self,
        a_logits: torch.Tensor,
        b_logits: torch.Tensor,
        weight_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_probs = F.softmax(a_logits / self.temperature, dim=-1)
        b_probs = F.softmax(b_logits / self.temperature, dim=-1)
        weights = F.softmax(weight_logits, dim=-1)
        return a_probs, b_probs, weights

    def decode_pair_probs(
        self,
        a_logits: torch.Tensor,
        b_logits: torch.Tensor,
        weight_logits: torch.Tensor,
    ) -> torch.Tensor:
        a_probs, b_probs, weights = self.slot_probs(a_logits, b_logits, weight_logits)
        slot_pair = a_probs.unsqueeze(-1) * b_probs.unsqueeze(-2)
        pair = (weights.unsqueeze(-1).unsqueeze(-1) * slot_pair).sum(dim=1)
        return pair.reshape(a_logits.shape[0], self.modulus * self.modulus)

    def diagnostics(
        self,
        a_logits: torch.Tensor,
        b_logits: torch.Tensor,
        weight_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        a_probs, b_probs, weights = self.slot_probs(a_logits, b_logits, weight_logits)
        eps = 1e-9
        a_entropy = -(a_probs * a_probs.clamp_min(eps).log()).sum(dim=-1)
        b_entropy = -(b_probs * b_probs.clamp_min(eps).log()).sum(dim=-1)
        weight_entropy = -(weights * weights.clamp_min(eps).log()).sum(dim=-1)
        purity = a_probs.max(dim=-1).values * b_probs.max(dim=-1).values
        return {
            "slot_entropy": (a_entropy + b_entropy).mean(dim=-1),
            "slot_purity": purity.mean(dim=-1),
            "weight_entropy": weight_entropy,
        }

    def exact_transition(
        self,
        a_logits: torch.Tensor,
        b_logits: torch.Tensor,
        weight_logits: torch.Tensor,
        op: torch.Tensor,
        arg: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_probs, b_probs, _ = self.slot_probs(a_logits, b_logits, weight_logits)
        new_a = a_probs.clone()
        new_b = b_probs.clone()
        new_w = weight_logits.clone()
        eps = 1e-9
        vals = torch.arange(self.modulus, device=a_logits.device)

        for code in range(len(OP_NAMES)):
            rows = (op == code) & step_mask
            if not bool(rows.any()):
                continue
            ap = a_probs[rows]
            bp = b_probs[rows]
            ar = arg[rows]
            if code == 0:
                new_a[rows] = roll_probs(ap, ar, +1)
            elif code == 1:
                new_a[rows] = roll_probs(ap, ar, -1)
            elif code == 2:
                new_b[rows] = roll_probs(bp, ar, +1)
            elif code == 3:
                new_b[rows] = roll_probs(bp, ar, -1)
            elif code == 4:
                new_a[rows] = combine_probs(ap, bp, "a_plus_b")
            elif code == 5:
                new_b[rows] = combine_probs(ap, bp, "b_plus_a")
            elif code == 6:
                new_a[rows] = combine_probs(ap, bp, "a_minus_b")
            elif code == 7:
                new_b[rows] = combine_probs(ap, bp, "b_minus_a")
            elif code == OBS_A:
                residue_mask = (vals.view(1, 1, -1) % self.observe_mod) == ar.view(-1, 1, 1)
                keep = (ap * residue_mask.float()).sum(dim=-1)
                conditioned = (ap * residue_mask.float()) / keep.unsqueeze(-1).clamp_min(eps)
                new_a[rows] = conditioned
                new_w[rows] = new_w[rows] + keep.clamp_min(eps).log()
            elif code == OBS_B:
                residue_mask = (vals.view(1, 1, -1) % self.observe_mod) == ar.view(-1, 1, 1)
                keep = (bp * residue_mask.float()).sum(dim=-1)
                conditioned = (bp * residue_mask.float()) / keep.unsqueeze(-1).clamp_min(eps)
                new_b[rows] = conditioned
                new_w[rows] = new_w[rows] + keep.clamp_min(eps).log()

        return new_a.clamp_min(eps).log(), new_b.clamp_min(eps).log(), new_w

    def neural_transition(
        self,
        a_logits: torch.Tensor,
        b_logits: torch.Tensor,
        weight_logits: torch.Tensor,
        op: torch.Tensor,
        arg: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_probs, b_probs, _ = self.slot_probs(a_logits, b_logits, weight_logits)
        a_ctx = torch.matmul(a_probs, self.a_value_emb.weight)
        b_ctx = torch.matmul(b_probs, self.b_value_emb.weight)
        op_ctx = self.op_emb(op).unsqueeze(1).expand(-1, self.slot_capacity, -1)
        arg_ctx = self.arg_emb(arg).unsqueeze(1).expand(-1, self.slot_capacity, -1)
        raw = self.transition_mlp(torch.cat([a_ctx, b_ctx, op_ctx, arg_ctx], dim=-1))
        p = self.modulus
        cand_a = raw[..., :p]
        cand_b = raw[..., p : 2 * p]
        keep_delta = raw[..., 2 * p]
        mask = step_mask.view(-1, 1, 1)
        new_a = torch.where(mask, cand_a, a_logits)
        new_b = torch.where(mask, cand_b, b_logits)
        new_w = torch.where(step_mask.view(-1, 1), weight_logits + keep_delta, weight_logits)
        return new_a, new_b, new_w

    def transition(
        self,
        a_logits: torch.Tensor,
        b_logits: torch.Tensor,
        weight_logits: torch.Tensor,
        op: torch.Tensor,
        arg: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.transition_mode == "exact":
            return self.exact_transition(a_logits, b_logits, weight_logits, op, arg, step_mask)
        return self.neural_transition(a_logits, b_logits, weight_logits, op, arg, step_mask)

    def forward(self, batch: ProgramBatch, k_max: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        a_logits, b_logits, weight_logits = self.initial_state(batch.init_delta)
        pair_trace = [self.decode_pair_probs(a_logits, b_logits, weight_logits)]
        diag_items = [self.diagnostics(a_logits, b_logits, weight_logits)]
        for t in range(k_max):
            step_mask = batch.lengths > t
            a_logits, b_logits, weight_logits = self.transition(
                a_logits,
                b_logits,
                weight_logits,
                batch.ops[:, t],
                batch.args[:, t],
                step_mask,
            )
            pair_trace.append(self.decode_pair_probs(a_logits, b_logits, weight_logits))
            diag_items.append(self.diagnostics(a_logits, b_logits, weight_logits))
        out_diag: Dict[str, torch.Tensor] = {}
        for key in diag_items[0]:
            out_diag[key] = torch.stack([item[key] for item in diag_items], dim=1)
        return torch.stack(pair_trace, dim=1), out_diag


def masked_belief_loss(pair_probs: torch.Tensor, target_probs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    steps = pair_probs.shape[1]
    losses = -(target_probs[:, :steps] * pair_probs.clamp_min(1e-9).log()).sum(dim=-1)
    active = torch.arange(steps, device=pair_probs.device).unsqueeze(0) <= lengths.unsqueeze(1)
    return losses[active].mean()


def final_query_loss(
    pair_probs_trace: torch.Tensor,
    query_type: torch.Tensor,
    target_query: torch.Tensor,
    lengths: torch.Tensor,
    modulus: int,
) -> torch.Tensor:
    idx = torch.arange(pair_probs_trace.shape[0], device=pair_probs_trace.device)
    final_pair_probs = pair_probs_trace[idx, lengths]
    pred_query = query_probs_from_pair_probs(final_pair_probs, query_type, modulus)
    return -(target_query * pred_query.clamp_min(1e-9).log()).sum(dim=-1).mean()


def entropy_loss(diag: Dict[str, torch.Tensor], lengths: torch.Tensor) -> torch.Tensor:
    steps = diag["slot_entropy"].shape[1]
    active = torch.arange(steps, device=lengths.device).unsqueeze(0) <= lengths.unsqueeze(1)
    return diag["slot_entropy"][active].mean()


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def build_model(args: argparse.Namespace, device: torch.device) -> LearnedSparseSlotExecutor:
    return LearnedSparseSlotExecutor(
        modulus=args.modulus,
        observe_mod=args.observe_mod,
        slot_capacity=args.slot_capacity,
        init_mode=args.init_mode,
        transition_mode=args.transition_mode,
        slot_dim=args.slot_dim,
        hidden_dim=args.hidden_dim,
        init_logit_scale=args.init_logit_scale,
        temperature=args.temperature,
    ).to(device)


def train_executor(args: argparse.Namespace, device: torch.device) -> Tuple[LearnedSparseSlotExecutor, Dict[str, Any]]:
    model = build_model(args, device)
    params = trainable_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay) if params else None
    gen = FilterProgramGenerator(args.modulus, args.observe_mod, args.observe_prob, args.seed, args.op_family, args.query_family)
    results: Dict[str, Any] = {
        "args": vars(args),
        "train": [],
        "eval": [],
        "checkpoints": [],
        "num_parameters": sum(p.numel() for p in params),
    }
    t0 = time.time()

    if optimizer is None or args.train_steps == 0:
        print(f"[train/{args.variant_name}] no trainable parameters or train_steps=0", flush=True)
        return model, results

    for step in range(1, args.train_steps + 1):
        batch = gen.batch(
            args.batch_size,
            args.train_min_len,
            args.train_max_len,
            args.train_max_len,
            device=device,
        )
        pair_trace, diag = model(batch, k_max=args.train_max_len)
        loss = torch.zeros((), dtype=torch.float32, device=device)
        loss_terms: Dict[str, float] = {}
        if args.supervision == "full_belief":
            belief = masked_belief_loss(pair_trace, batch.target_probs, batch.lengths)
            loss = loss + args.belief_loss_weight * belief
            loss_terms["belief_loss"] = float(belief.detach().cpu())
            if args.query_loss_weight > 0:
                qloss = final_query_loss(pair_trace, batch.query_type, batch.query_probs, batch.lengths, args.modulus)
                loss = loss + args.query_loss_weight * qloss
                loss_terms["query_loss"] = float(qloss.detach().cpu())
        elif args.supervision == "final_query":
            qloss = final_query_loss(pair_trace, batch.query_type, batch.query_probs, batch.lengths, args.modulus)
            loss = loss + qloss
            loss_terms["query_loss"] = float(qloss.detach().cpu())
        else:
            raise ValueError(args.supervision)
        if args.slot_entropy_weight > 0:
            ent = entropy_loss(diag, batch.lengths)
            loss = loss + args.slot_entropy_weight * ent
            loss_terms["slot_entropy_loss"] = float(ent.detach().cpu())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            idx = torch.arange(batch.lengths.shape[0], device=device)
            final_pair = pair_trace[idx, batch.lengths]
            bm = belief_metrics(final_pair, batch.target_probs[idx, batch.lengths])
            qm = query_metrics_from_pair_probs(final_pair, batch.query_type, batch.query_probs, args.modulus)
            purity = float(diag["slot_purity"][idx, batch.lengths].mean().detach().cpu())
            went = float(diag["weight_entropy"][idx, batch.lengths].mean().detach().cpu())
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad),
                "elapsed_seconds": time.time() - t0,
                "mean_slot_purity": purity,
                "mean_weight_entropy": went,
                **loss_terms,
                **bm,
                **qm,
            }
            results["train"].append(row)
            print(
                f"[train/{args.variant_name}] step={step:05d} loss={row['loss']:.4f} "
                f"dq={row['decoder_query_target_mass']*100:.1f}% "
                f"db={row['decoder_belief_target_mass']*100:.1f}% "
                f"purity={purity:.3f} went={went:.3f} grad={float(grad):.3f} "
                f"elapsed={(time.time()-t0)/60:.1f}m",
                flush=True,
            )
    return model, results


@torch.no_grad()
def evaluate(
    model: LearnedSparseSlotExecutor,
    generator: FilterProgramGenerator,
    args: argparse.Namespace,
    device: torch.device,
    lengths: Sequence[int],
    k_values: Sequence[int],
    examples_per_length: int,
) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    max_k = max(k_values)
    batch_size = min(args.eval_batch_size, examples_per_length)
    eval_query_types = parse_query_types(args.eval_query_types)
    for length in lengths:
        trace_k = max(length, max_k)
        for q_type in eval_query_types:
            sums: Dict[int, Dict[str, float]] = {
                k: {
                    "n": 0.0,
                    "dq_mass": 0.0,
                    "dq_nll": 0.0,
                    "dq_top1": 0.0,
                    "dq_support": 0.0,
                    "db_mass": 0.0,
                    "db_nll": 0.0,
                    "db_top1": 0.0,
                    "db_support": 0.0,
                    "slot_purity": 0.0,
                    "slot_entropy": 0.0,
                    "weight_entropy": 0.0,
                    "target_support_size": 0.0,
                }
                for k in k_values
            }
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
                pair_trace, diag = model(batch, k_max=max_k)
                final_pair_target = batch.target_probs[:, length]
                for k in k_values:
                    pred = pair_trace[:, k]
                    bm = belief_metrics(pred, final_pair_target)
                    qm = query_metrics_from_pair_probs(pred, batch.query_type, batch.query_probs, args.modulus)
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
                    sums[k]["slot_purity"] += float(diag["slot_purity"][:, k].mean().detach().cpu()) * n
                    sums[k]["slot_entropy"] += float(diag["slot_entropy"][:, k].mean().detach().cpu()) * n
                    sums[k]["weight_entropy"] += float(diag["weight_entropy"][:, k].mean().detach().cpu()) * n
                    sums[k]["target_support_size"] += float(batch.support_size[:, length].mean().detach().cpu()) * n
            for k in k_values:
                n = max(1.0, sums[k]["n"])
                rows.append(
                    {
                        "model": "learned_sparse_slot",
                        "variant": args.variant_name,
                        "modulus": args.modulus,
                        "observe_mod": args.observe_mod,
                        "observe_prob": args.observe_prob,
                        "op_family": args.op_family,
                        "query_family": args.query_family,
                        "slot_capacity": args.slot_capacity,
                        "init_mode": args.init_mode,
                        "transition_mode": args.transition_mode,
                        "supervision": args.supervision,
                        "slot_dim": args.slot_dim,
                        "hidden_dim": args.hidden_dim,
                        "train_steps": args.train_steps,
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
                        "mean_slot_purity": sums[k]["slot_purity"] / n,
                        "mean_slot_entropy": sums[k]["slot_entropy"] / n,
                        "mean_weight_entropy": sums[k]["weight_entropy"] / n,
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
            first_k = min(int(r["k"]) for r in chosen)
        qmass = sum(float(r["decoder_query_target_mass"]) for r in chosen) / max(1, len(chosen))
        bmass = sum(float(r["decoder_belief_target_mass"]) for r in chosen) / max(1, len(chosen))
        purity = sum(float(r["mean_slot_purity"]) for r in chosen) / max(1, len(chosen))
        went = sum(float(r["mean_weight_entropy"]) for r in chosen) / max(1, len(chosen))
        print(
            f"  L={length} K={first_k}: dqmass={qmass*100:.1f}% "
            f"dbmass={bmass*100:.1f}% purity={purity:.3f} went={went:.3f}",
            flush=True,
        )


def checkpoint_path(args: argparse.Namespace) -> Path:
    root = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path("large_artifacts/learned_sparse_slot_executor/checkpoints") / Path(args.output_dir).name
    )
    return root / "checkpoint_final.pt"


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, results: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "results": results,
        },
        path,
    )


def train_and_evaluate(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    model, results = train_executor(args, device)
    eval_lengths = parse_int_list(args.eval_lengths)
    eval_k = parse_int_list(args.eval_k)
    eval_gen = FilterProgramGenerator(
        args.modulus,
        args.observe_mod,
        args.observe_prob,
        args.seed + 10_000,
        args.op_family,
        args.query_family,
    )
    rows = evaluate(model, eval_gen, args, device, eval_lengths, eval_k, args.eval_examples)
    results["eval"].append({"step": args.train_steps, "rows": rows})
    print_eval_summary(f"[eval/{args.variant_name} final]", rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(out / "metrics_final.csv", rows)
    if results["num_parameters"] > 0:
        ckpt = checkpoint_path(args)
        save_checkpoint(ckpt, model, args, results)
        results["checkpoints"].append(str(ckpt))
    with (out / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Learned sparse-slot executor")
    p.add_argument("--variant_name", type=str, default="learned_sparse_slot")
    p.add_argument("--seed", type=int, default=83)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--modulus", type=int, default=11)
    p.add_argument("--observe_mod", type=int, default=4)
    p.add_argument("--observe_prob", type=float, default=0.3)
    p.add_argument("--op_family", choices=["full", "const", "cross"], default="full")
    p.add_argument("--query_family", choices=["all", "marginal", "relational"], default="all")
    p.add_argument("--slot_capacity", type=int, default=11)
    p.add_argument("--init_mode", choices=["oracle", "learned"], default="oracle")
    p.add_argument("--transition_mode", choices=["exact", "neural"], default="exact")
    p.add_argument("--supervision", choices=["full_belief", "final_query"], default="full_belief")
    p.add_argument("--slot_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--init_logit_scale", type=float, default=8.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--train_min_len", type=int, default=1)
    p.add_argument("--train_max_len", type=int, default=6)
    p.add_argument("--eval_lengths", type=str, default="3,6,9,12")
    p.add_argument("--eval_k", type=str, default="0,1,2,3,6,9,12")
    p.add_argument("--eval_query_types", type=str, default="all")
    p.add_argument("--train_steps", type=int, default=800)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--eval_examples", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--belief_loss_weight", type=float, default=1.0)
    p.add_argument("--query_loss_weight", type=float, default=0.0)
    p.add_argument("--slot_entropy_weight", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="experiments/learned_sparse_slot_executor/runs/learned_sparse_slot")
    p.add_argument("--checkpoint_dir", type=str, default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train_and_evaluate(args, device)


if __name__ == "__main__":
    main()
