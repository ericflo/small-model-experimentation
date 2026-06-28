#!/usr/bin/env python3
"""Dense teacher-distillation bottleneck test for recurrent belief execution.

Programs transform a correlated belief state over two modular registers using
arithmetic operations and observation filters. A teacher computes exact prefix
belief distributions over all `(A,B)` pairs. The student keeps only a dense
hidden vector and is trained to decode those exact prefix beliefs. The primary
question is whether dense recurrent state capacity and transition structure are
enough to support near-exact belief execution when supervision is no longer the
limiting factor.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
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


def safe_log_probs(probs: torch.Tensor) -> torch.Tensor:
    return probs.clamp_min(1e-9).log()


def select_query_logits(query_logits: torch.Tensor, query_type: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(query_logits.shape[0], device=query_logits.device)
    return query_logits[idx, query_type]


def sampled_query_loss(query_logits: torch.Tensor, query_type: torch.Tensor, query_label: torch.Tensor) -> torch.Tensor:
    chosen = select_query_logits(query_logits, query_type)
    return F.cross_entropy(chosen, query_label)


def soft_query_loss(query_logits: torch.Tensor, query_type: torch.Tensor, target_query: torch.Tensor) -> torch.Tensor:
    chosen = select_query_logits(query_logits, query_type)
    log_probs = F.log_softmax(chosen, dim=-1)
    return -(target_query * log_probs).sum(dim=-1).mean()


def query_value_index(modulus: int, device: torch.device) -> torch.Tensor:
    a = torch.arange(modulus, device=device).repeat_interleave(modulus)
    b = torch.arange(modulus, device=device).repeat(modulus)
    return torch.stack([a, b, (a + b) % modulus, (a - b) % modulus], dim=0)


def query_probs_from_pair_probs(pair_probs: torch.Tensor, query_type: torch.Tensor, modulus: int) -> torch.Tensor:
    """Project pair distributions to selected query distributions."""
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


def prefix_query_loss(
    query_logits: torch.Tensor,
    query_type: torch.Tensor,
    target_pair_probs: torch.Tensor,
    lengths: torch.Tensor,
    modulus: int,
) -> torch.Tensor:
    steps = query_logits.shape[1]
    target_query = query_probs_from_pair_probs(target_pair_probs[:, :steps], query_type, modulus)
    chosen = query_logits.gather(
        2,
        query_type[:, None, None, None].expand(query_logits.shape[0], steps, 1, modulus),
    ).squeeze(2)
    losses = -(target_query * F.log_softmax(chosen, dim=-1)).sum(dim=-1)
    mask = torch.arange(steps, device=query_logits.device).unsqueeze(0) <= lengths.unsqueeze(1)
    return (losses * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def query_metrics(query_logits: torch.Tensor, query_type: torch.Tensor, target_query: torch.Tensor) -> Dict[str, float]:
    chosen = select_query_logits(query_logits, query_type)
    log_probs = F.log_softmax(chosen, dim=-1)
    probs = log_probs.exp()
    mask = target_query > 0
    target_mass = (probs * mask.float()).sum(dim=-1)
    target_nll = -(target_query * log_probs).sum(dim=-1)
    pred = log_probs.argmax(dim=-1)
    top1 = mask.gather(1, pred.unsqueeze(1)).squeeze(1).float()
    return {
        "query_target_mass": float(target_mass.mean().detach().cpu()),
        "query_target_nll": float(target_nll.mean().detach().cpu()),
        "query_top1_on_support": float(top1.mean().detach().cpu()),
        "mean_query_support_size": float(mask.sum(dim=-1).float().mean().detach().cpu()),
    }


def query_metrics_from_pair_logits(
    pair_logits: torch.Tensor,
    query_type: torch.Tensor,
    target_query: torch.Tensor,
    modulus: int,
) -> Dict[str, float]:
    pair_probs = F.softmax(pair_logits, dim=-1)
    query_probs = query_probs_from_pair_probs(pair_probs, query_type, modulus)
    log_probs = query_probs.clamp_min(1e-9).log()
    mask = target_query > 0
    target_mass = (query_probs * mask.float()).sum(dim=-1)
    target_nll = -(target_query * log_probs).sum(dim=-1)
    pred = log_probs.argmax(dim=-1)
    top1 = mask.gather(1, pred.unsqueeze(1)).squeeze(1).float()
    return {
        "decoder_query_target_mass": float(target_mass.mean().detach().cpu()),
        "decoder_query_target_nll": float(target_nll.mean().detach().cpu()),
        "decoder_query_top1_on_support": float(top1.mean().detach().cpu()),
        "mean_decoder_query_support_size": float(mask.sum(dim=-1).float().mean().detach().cpu()),
    }


def belief_metrics(pair_logits: torch.Tensor, target_probs: torch.Tensor) -> Dict[str, float]:
    log_probs = F.log_softmax(pair_logits, dim=-1)
    probs = log_probs.exp()
    mask = target_probs > 0
    target_mass = (probs * mask.float()).sum(dim=-1)
    target_nll = -(target_probs * log_probs).sum(dim=-1)
    pred = log_probs.argmax(dim=-1)
    top1 = mask.gather(1, pred.unsqueeze(1)).squeeze(1).float()
    return {
        "probe_belief_target_mass": float(target_mass.mean().detach().cpu()),
        "probe_belief_target_nll": float(target_nll.mean().detach().cpu()),
        "probe_belief_top1_on_support": float(top1.mean().detach().cpu()),
        "mean_belief_support_size": float(mask.sum(dim=-1).float().mean().detach().cpu()),
    }


def belief_probe_loss(pair_logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(pair_logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def masked_belief_loss(
    pair_logits: torch.Tensor,
    target_probs: torch.Tensor,
    lengths: torch.Tensor,
    sample_prob: float,
) -> torch.Tensor:
    steps = pair_logits.shape[1]
    losses = -(target_probs[:, :steps] * F.log_softmax(pair_logits, dim=-1)).sum(dim=-1)
    active = torch.arange(steps, device=pair_logits.device).unsqueeze(0) <= lengths.unsqueeze(1)
    if sample_prob < 1.0:
        selected = active & (torch.rand(active.shape, device=pair_logits.device) < sample_prob)
        if not bool(selected.any()):
            selected = active
    else:
        selected = active
    return losses[selected].mean()


def decoder_prefix_query_loss(
    pair_logits: torch.Tensor,
    query_type: torch.Tensor,
    target_pair_probs: torch.Tensor,
    lengths: torch.Tensor,
    modulus: int,
) -> torch.Tensor:
    steps = pair_logits.shape[1]
    pair_probs = F.softmax(pair_logits, dim=-1)
    pred_query = query_probs_from_pair_probs(pair_probs, query_type, modulus)
    target_query = query_probs_from_pair_probs(target_pair_probs[:, :steps], query_type, modulus)
    losses = -(target_query * pred_query.clamp_min(1e-9).log()).sum(dim=-1)
    mask = torch.arange(steps, device=pair_logits.device).unsqueeze(0) <= lengths.unsqueeze(1)
    return (losses * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def rename_belief_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        f"{prefix}_belief_target_mass": metrics["probe_belief_target_mass"],
        f"{prefix}_belief_target_nll": metrics["probe_belief_target_nll"],
        f"{prefix}_belief_top1_on_support": metrics["probe_belief_top1_on_support"],
        f"mean_{prefix}_belief_support_size": metrics["mean_belief_support_size"],
    }


class BeliefDecoder(nn.Module):
    def __init__(self, state_dim: int, pair_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 4 * state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * state_dim, pair_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class LowRankBeliefDecoder(nn.Module):
    """Decode pair beliefs as a mixture of rank-1 distributions over A and B."""

    def __init__(self, state_dim: int, modulus: int, rank: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.rank = int(rank)
        out_dim = self.rank * (1 + 2 * self.modulus)
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 4 * state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * state_dim, out_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        raw = self.net(states)
        shape = raw.shape[:-1]
        raw = raw.view(*shape, self.rank, 1 + 2 * self.modulus)
        log_w = F.log_softmax(raw[..., 0], dim=-1)
        a_log = F.log_softmax(raw[..., 1 : 1 + self.modulus], dim=-1)
        b_log = F.log_softmax(raw[..., 1 + self.modulus :], dim=-1)
        pair_log = a_log.unsqueeze(-1) + b_log.unsqueeze(-2)
        pair_log = pair_log + log_w.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(pair_log, dim=-3).reshape(*shape, self.modulus * self.modulus)


class ResidualTransitionCell(nn.Module):
    def __init__(self, state_dim: int, instr_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim + instr_dim),
            nn.Linear(state_dim + instr_dim, 4 * state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * state_dim, state_dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(state_dim + instr_dim),
            nn.Linear(state_dim + instr_dim, state_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(state_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=-1)
        delta = self.net(hx)
        gate = self.gate(hx)
        return self.norm(h + gate * delta)


class DenseRecurrentExecutor(nn.Module):
    def __init__(
        self,
        modulus: int,
        observe_mod: int,
        state_dim: int,
        instr_dim: int,
        dropout: float,
        transition: str = "gru",
        decoder_type: str = "mlp",
        decoder_rank: int = 16,
        order_control: str = "none",
    ) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.observe_mod = int(observe_mod)
        self.arg_vocab = max(self.modulus, self.observe_mod)
        self.order_control = order_control
        self.transition = transition
        self.delta_embed = nn.Embedding(modulus, state_dim)
        self.op_embed = nn.Embedding(len(OP_NAMES), instr_dim)
        self.arg_embed = nn.Embedding(self.arg_vocab, instr_dim)
        if transition == "gru":
            self.cell = nn.GRUCell(instr_dim, state_dim)
        elif transition == "residual":
            self.cell = ResidualTransitionCell(state_dim, instr_dim, dropout)
        else:
            raise ValueError(f"unknown transition {transition}")
        self.state_norm = nn.LayerNorm(state_dim)
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 4 * state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * state_dim, state_dim),
        )
        self.query_head = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 2 * state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * state_dim, 4 * modulus),
        )
        if decoder_type == "mlp":
            self.belief_head = BeliefDecoder(state_dim, modulus * modulus, dropout)
        elif decoder_type == "low_rank":
            self.belief_head = LowRankBeliefDecoder(state_dim, modulus, decoder_rank, dropout)
        else:
            raise ValueError(f"unknown decoder_type {decoder_type}")

    def reorder(self, ops: torch.Tensor, args: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.order_control != "sorted":
            return ops, args
        ops2 = ops.clone()
        args2 = args.clone()
        for i in range(ops.shape[0]):
            length = int(lengths[i].detach().cpu())
            if length <= 1:
                continue
            keys = ops2[i, :length] * self.arg_vocab + args2[i, :length]
            perm = torch.argsort(keys)
            ops2[i, :length] = ops2[i, :length][perm]
            args2[i, :length] = args2[i, :length][perm]
        return ops2, args2

    def query_from_states(self, states: torch.Tensor) -> torch.Tensor:
        logits = self.query_head(states)
        return logits.view(*states.shape[:2], 4, self.modulus)

    def belief_from_states(self, states: torch.Tensor) -> torch.Tensor:
        return self.belief_head(states)

    def forward(
        self,
        ops: torch.Tensor,
        args: torch.Tensor,
        lengths: torch.Tensor,
        init_delta: torch.Tensor,
        k_max: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ops, args = self.reorder(ops, args, lengths)
        bsz = init_delta.shape[0]
        h = torch.tanh(self.delta_embed(init_delta))
        states = [h]
        for t in range(k_max):
            active = (lengths > t).unsqueeze(1)
            if t < ops.shape[1]:
                op = ops[:, t]
                arg = args[:, t].clamp_max(self.arg_vocab - 1)
            else:
                op = torch.zeros(bsz, dtype=torch.long, device=ops.device)
                arg = torch.zeros(bsz, dtype=torch.long, device=ops.device)
            x = self.op_embed(op) + self.arg_embed(arg)
            if self.transition == "gru":
                cand = self.cell(x, h)
                cand = self.state_norm(cand + 0.5 * self.state_mlp(cand))
            else:
                cand = self.cell(x, h)
            h = torch.where(active, cand, h)
            states.append(h)
        state_tensor = torch.stack(states, dim=1)
        return state_tensor, self.query_from_states(state_tensor)


class DenseStaticCompiler(nn.Module):
    def __init__(
        self,
        modulus: int,
        max_program_len: int,
        state_dim: int,
        instr_dim: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.arg_vocab = modulus
        self.op_embed = nn.Embedding(len(OP_NAMES), instr_dim)
        self.arg_embed = nn.Embedding(modulus, instr_dim)
        self.pos_embed = nn.Embedding(max_program_len, instr_dim)
        self.delta_embed = nn.Embedding(modulus, instr_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=instr_dim,
            nhead=heads,
            dim_feedforward=4 * instr_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.compiler = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.state_head = nn.Sequential(
            nn.LayerNorm(2 * instr_dim),
            nn.Linear(2 * instr_dim, 2 * state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * state_dim, state_dim),
            nn.LayerNorm(state_dim),
        )
        self.query_head = nn.Sequential(
            nn.Linear(state_dim, 2 * state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * state_dim, 4 * modulus),
        )
        self.belief_head = BeliefDecoder(state_dim, modulus * modulus, dropout)

    def forward(
        self,
        ops: torch.Tensor,
        args: torch.Tensor,
        lengths: torch.Tensor,
        init_delta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq = ops.shape
        pos = torch.arange(seq, device=ops.device).unsqueeze(0).expand(bsz, seq)
        x = self.op_embed(ops) + self.arg_embed(args.clamp_max(self.arg_vocab - 1)) + self.pos_embed(pos)
        h = self.compiler(x)
        mask = (torch.arange(seq, device=ops.device).unsqueeze(0) < lengths.unsqueeze(1)).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        state = self.state_head(torch.cat([pooled, self.delta_embed(init_delta)], dim=-1))
        logits = self.query_head(state).view(bsz, 4, self.modulus)
        return state, logits

    def belief_from_states(self, states: torch.Tensor) -> torch.Tensor:
        return self.belief_head(states)


class BeliefProbe(nn.Module):
    def __init__(self, state_dim: int, pair_dim: int, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = hidden_mult * state_dim
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pair_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


def build_model(args: argparse.Namespace, max_program_len: int) -> nn.Module:
    if args.mode == "dense":
        return DenseRecurrentExecutor(
            args.modulus,
            args.observe_mod,
            args.state_dim,
            args.instr_dim,
            args.dropout,
            transition=args.transition,
            decoder_type=args.decoder_type,
            decoder_rank=args.decoder_rank,
            order_control="none",
        )
    if args.mode == "shuffled":
        return DenseRecurrentExecutor(
            args.modulus,
            args.observe_mod,
            args.state_dim,
            args.instr_dim,
            args.dropout,
            transition=args.transition,
            decoder_type=args.decoder_type,
            decoder_rank=args.decoder_rank,
            order_control="sorted",
        )
    if args.mode == "static":
        return DenseStaticCompiler(
            args.modulus,
            max_program_len,
            args.state_dim,
            args.instr_dim,
            args.heads,
            args.compiler_layers,
            args.dropout,
        )
    raise ValueError(args.mode)


def recurrent_model(model: nn.Module) -> bool:
    return isinstance(model, DenseRecurrentExecutor)


@torch.no_grad()
def final_train_metrics(model: nn.Module, batch: ProgramBatch, args: argparse.Namespace) -> Dict[str, float]:
    if recurrent_model(model):
        states, logits = model(batch.ops, batch.args, batch.lengths, batch.init_delta, k_max=args.train_max_len)
        pair_logits = model.belief_from_states(states[:, -1])
        out = query_metrics(logits[:, -1], batch.query_type, batch.query_probs)
        out.update(rename_belief_metrics("decoder", belief_metrics(pair_logits, batch.target_probs[:, -1])))
        out.update(query_metrics_from_pair_logits(pair_logits, batch.query_type, batch.query_probs, args.modulus))
        return out
    state, logits = model(batch.ops, batch.args, batch.lengths, batch.init_delta)
    pair_logits = model.belief_from_states(state)
    out = query_metrics(logits, batch.query_type, batch.query_probs)
    out.update(rename_belief_metrics("decoder", belief_metrics(pair_logits, batch.target_probs[:, -1])))
    out.update(query_metrics_from_pair_logits(pair_logits, batch.query_type, batch.query_probs, args.modulus))
    return out


def recurrent_training_loss(
    model: DenseRecurrentExecutor,
    states: torch.Tensor,
    query_logits: torch.Tensor,
    batch: ProgramBatch,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if args.supervision != "teacher_belief":
        raise ValueError(args.supervision)
    zero = torch.zeros((), device=query_logits.device)
    query_loss = (
        prefix_query_loss(query_logits, batch.query_type, batch.target_probs, batch.lengths, args.modulus)
        if args.query_loss_weight > 0
        else zero
    )
    belief_logits = model.belief_from_states(states)
    belief_loss = masked_belief_loss(belief_logits, batch.target_probs, batch.lengths, args.belief_distill_prob)
    decoder_query_loss = (
        decoder_prefix_query_loss(belief_logits, batch.query_type, batch.target_probs, batch.lengths, args.modulus)
        if args.decoder_query_loss_weight > 0
        else zero
    )
    total = (
        args.query_loss_weight * query_loss
        + args.belief_loss_weight * belief_loss
        + args.decoder_query_loss_weight * decoder_query_loss
    )
    return total, {
        "query_loss": float(query_loss.detach().cpu()),
        "belief_loss": float(belief_loss.detach().cpu()),
        "decoder_query_loss": float(decoder_query_loss.detach().cpu()),
    }


def static_training_loss(
    model: DenseStaticCompiler,
    state: torch.Tensor,
    query_logits: torch.Tensor,
    batch: ProgramBatch,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if args.supervision != "teacher_belief":
        raise ValueError(args.supervision)
    zero = torch.zeros((), device=query_logits.device)
    query_loss = soft_query_loss(query_logits, batch.query_type, batch.query_probs) if args.query_loss_weight > 0 else zero
    pair_logits = model.belief_from_states(state)
    belief_loss = belief_probe_loss(pair_logits, batch.target_probs[:, -1])
    decoder_query_loss = zero
    if args.decoder_query_loss_weight > 0:
        pred_query = query_probs_from_pair_probs(F.softmax(pair_logits, dim=-1), batch.query_type, args.modulus)
        decoder_query_loss = -(batch.query_probs * pred_query.clamp_min(1e-9).log()).sum(dim=-1).mean()
    total = (
        args.query_loss_weight * query_loss
        + args.belief_loss_weight * belief_loss
        + args.decoder_query_loss_weight * decoder_query_loss
    )
    return total, {
        "query_loss": float(query_loss.detach().cpu()),
        "belief_loss": float(belief_loss.detach().cpu()),
        "decoder_query_loss": float(decoder_query_loss.detach().cpu()),
    }


def train_executor(args: argparse.Namespace, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    eval_lengths = parse_int_list(args.eval_lengths)
    eval_k = parse_int_list(args.eval_k)
    max_program_len = max(args.train_max_len, max(eval_lengths), max(eval_k))
    model = build_model(args, max_program_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_gen = FilterProgramGenerator(args.modulus, args.observe_mod, args.observe_prob, args.seed, args.op_family, args.query_family)
    results: Dict[str, Any] = {
        "mode": args.mode,
        "variant": args.variant_name,
        "transition": args.transition,
        "decoder_type": args.decoder_type,
        "args": vars(args),
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "train": [],
        "probe_train": [],
        "eval": [],
        "checkpoints": [],
    }
    t0 = time.time()
    model.train()
    for step in range(1, args.train_steps + 1):
        batch = train_gen.batch(args.batch_size, args.train_min_len, args.train_max_len, args.train_max_len, device=device)
        if recurrent_model(model):
            states, logits = model(batch.ops, batch.args, batch.lengths, batch.init_delta, k_max=args.train_max_len)
            loss, loss_parts = recurrent_training_loss(model, states, logits, batch, args)
        else:
            state, logits = model(batch.ops, batch.args, batch.lengths, batch.init_delta)
            loss, loss_parts = static_training_loss(model, state, logits, batch, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            metrics = final_train_metrics(model, batch, args)
            print(
                f"[train/{args.variant_name}] step={step:05d} loss={float(loss.detach().cpu()):.4f} "
                f"qtop={metrics['query_top1_on_support']*100:.1f}% "
                f"qmass={metrics['query_target_mass']*100:.1f}% "
                f"dqmass={metrics['decoder_query_target_mass']*100:.1f}% "
                f"dbmass={metrics['decoder_belief_target_mass']*100:.1f}% "
                f"qsupp={metrics['mean_query_support_size']:.1f} "
                f"qloss={loss_parts['query_loss']:.4f} bloss={loss_parts['belief_loss']:.4f} "
                f"dqloss={loss_parts['decoder_query_loss']:.4f} "
                f"grad={float(grad):.3f} elapsed={(time.time()-t0)/60:.1f}m",
                flush=True,
            )
            results["train"].append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad),
                    **loss_parts,
                    **metrics,
                }
            )
    return model, results


def train_probe(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    results: Dict[str, Any],
) -> BeliefProbe:
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    probe = BeliefProbe(args.state_dim, args.modulus * args.modulus, dropout=args.probe_dropout).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_weight_decay)
    gen = FilterProgramGenerator(args.modulus, args.observe_mod, args.observe_prob, args.seed + 20_000, args.op_family, args.query_family)
    t0 = time.time()
    for step in range(1, args.probe_steps + 1):
        batch = gen.batch(args.probe_batch_size, args.train_min_len, args.train_max_len, args.train_max_len, device=device)
        with torch.no_grad():
            if recurrent_model(model):
                states, _ = model(batch.ops, batch.args, batch.lengths, batch.init_delta, k_max=args.train_max_len)
                probe_in = states.reshape(-1, states.shape[-1])
                targets = batch.target_probs[:, : args.train_max_len + 1].reshape(-1, args.modulus * args.modulus)
            else:
                state, _ = model(batch.ops, batch.args, batch.lengths, batch.init_delta)
                probe_in = state
                targets = batch.target_probs[:, -1]
        logits = probe(probe_in)
        loss = belief_probe_loss(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(probe.parameters(), args.max_grad_norm)
        optimizer.step()

        if step == 1 or step % args.probe_log_every == 0 or step == args.probe_steps:
            with torch.no_grad():
                metrics = belief_metrics(logits, targets)
            print(
                f"[probe/{args.variant_name}] step={step:05d} loss={float(loss.detach().cpu()):.4f} "
                f"btop={metrics['probe_belief_top1_on_support']*100:.1f}% "
                f"bmass={metrics['probe_belief_target_mass']*100:.1f}% "
                f"bsupp={metrics['mean_belief_support_size']:.1f} "
                f"grad={float(grad):.3f} elapsed={(time.time()-t0)/60:.1f}m",
                flush=True,
            )
            results["probe_train"].append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad),
                    **metrics,
                }
            )
    return probe


@torch.no_grad()
def evaluate_recurrent(
    model: DenseRecurrentExecutor,
    probe: BeliefProbe,
    generator: FilterProgramGenerator,
    args: argparse.Namespace,
    device: torch.device,
    lengths: Sequence[int],
    k_values: Sequence[int],
    examples_per_length: int,
) -> List[Dict[str, Any]]:
    model.eval()
    probe.eval()
    rows: List[Dict[str, Any]] = []
    max_k = max(k_values)
    batch_size = min(args.eval_batch_size, examples_per_length)
    eval_query_types = parse_query_types(args.eval_query_types)
    for length in lengths:
        trace_k = max(length, max_k)
        for q_type in eval_query_types:
            sums: Dict[int, Dict[str, float]] = {
                k: {
                    "n": 0,
                    "q_mass": 0.0,
                    "q_nll": 0.0,
                    "q_top1": 0.0,
                    "q_support": 0.0,
                    "b_mass": 0.0,
                    "b_nll": 0.0,
                    "b_top1": 0.0,
                    "b_support": 0.0,
                    "d_mass": 0.0,
                    "d_nll": 0.0,
                    "d_top1": 0.0,
                    "d_support": 0.0,
                    "dq_mass": 0.0,
                    "dq_nll": 0.0,
                    "dq_top1": 0.0,
                    "dq_support": 0.0,
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
                states, q_logits = model(batch.ops, batch.args, batch.lengths, batch.init_delta, k_max=max_k)
                final_pair_target = batch.target_probs[:, length]
                for k in k_values:
                    qm = query_metrics(q_logits[:, k], batch.query_type, batch.query_probs)
                    bm = belief_metrics(probe(states[:, k]), final_pair_target)
                    pair_logits = model.belief_from_states(states[:, k])
                    dm = rename_belief_metrics("decoder", belief_metrics(pair_logits, final_pair_target))
                    dqm = query_metrics_from_pair_logits(pair_logits, batch.query_type, batch.query_probs, args.modulus)
                    sums[k]["n"] += bsz
                    sums[k]["q_mass"] += qm["query_target_mass"] * bsz
                    sums[k]["q_nll"] += qm["query_target_nll"] * bsz
                    sums[k]["q_top1"] += qm["query_top1_on_support"] * bsz
                    sums[k]["q_support"] += qm["mean_query_support_size"] * bsz
                    sums[k]["b_mass"] += bm["probe_belief_target_mass"] * bsz
                    sums[k]["b_nll"] += bm["probe_belief_target_nll"] * bsz
                    sums[k]["b_top1"] += bm["probe_belief_top1_on_support"] * bsz
                    sums[k]["b_support"] += bm["mean_belief_support_size"] * bsz
                    sums[k]["d_mass"] += dm["decoder_belief_target_mass"] * bsz
                    sums[k]["d_nll"] += dm["decoder_belief_target_nll"] * bsz
                    sums[k]["d_top1"] += dm["decoder_belief_top1_on_support"] * bsz
                    sums[k]["d_support"] += dm["mean_decoder_belief_support_size"] * bsz
                    sums[k]["dq_mass"] += dqm["decoder_query_target_mass"] * bsz
                    sums[k]["dq_nll"] += dqm["decoder_query_target_nll"] * bsz
                    sums[k]["dq_top1"] += dqm["decoder_query_top1_on_support"] * bsz
                    sums[k]["dq_support"] += dqm["mean_decoder_query_support_size"] * bsz
            for k in k_values:
                n = max(1, int(sums[k]["n"]))
                rows.append(
                    {
                        "model": args.mode,
                        "variant": args.variant_name,
                        "supervision": args.supervision,
                        "transition": args.transition,
                        "decoder_type": args.decoder_type,
                        "state_dim": args.state_dim,
                        "instr_dim": args.instr_dim,
                        "decoder_rank": args.decoder_rank,
                        "length": int(length),
                        "query_type": QUERY_NAMES[q_type],
                        "k": int(k),
                        "n": n,
                        "query_target_mass": sums[k]["q_mass"] / n,
                        "query_target_nll": sums[k]["q_nll"] / n,
                        "query_top1_on_support": sums[k]["q_top1"] / n,
                        "mean_query_support_size": sums[k]["q_support"] / n,
                        "probe_belief_target_mass": sums[k]["b_mass"] / n,
                        "probe_belief_target_nll": sums[k]["b_nll"] / n,
                        "probe_belief_top1_on_support": sums[k]["b_top1"] / n,
                        "mean_belief_support_size": sums[k]["b_support"] / n,
                        "decoder_belief_target_mass": sums[k]["d_mass"] / n,
                        "decoder_belief_target_nll": sums[k]["d_nll"] / n,
                        "decoder_belief_top1_on_support": sums[k]["d_top1"] / n,
                        "mean_decoder_belief_support_size": sums[k]["d_support"] / n,
                        "decoder_query_target_mass": sums[k]["dq_mass"] / n,
                        "decoder_query_target_nll": sums[k]["dq_nll"] / n,
                        "decoder_query_top1_on_support": sums[k]["dq_top1"] / n,
                        "mean_decoder_query_support_size": sums[k]["dq_support"] / n,
                    }
                )
    return rows


@torch.no_grad()
def evaluate_static(
    model: DenseStaticCompiler,
    probe: BeliefProbe,
    generator: FilterProgramGenerator,
    args: argparse.Namespace,
    device: torch.device,
    lengths: Sequence[int],
    examples_per_length: int,
) -> List[Dict[str, Any]]:
    model.eval()
    probe.eval()
    rows: List[Dict[str, Any]] = []
    batch_size = min(args.eval_batch_size, examples_per_length)
    eval_query_types = parse_query_types(args.eval_query_types)
    for length in lengths:
        for q_type in eval_query_types:
            sums = {
                "n": 0,
                "q_mass": 0.0,
                "q_nll": 0.0,
                "q_top1": 0.0,
                "q_support": 0.0,
                "b_mass": 0.0,
                "b_nll": 0.0,
                "b_top1": 0.0,
                "b_support": 0.0,
                "d_mass": 0.0,
                "d_nll": 0.0,
                "d_top1": 0.0,
                "d_support": 0.0,
                "dq_mass": 0.0,
                "dq_nll": 0.0,
                "dq_top1": 0.0,
                "dq_support": 0.0,
            }
            while sums["n"] < examples_per_length:
                bsz = min(batch_size, examples_per_length - int(sums["n"]))
                batch = generator.batch(
                    bsz,
                    length,
                    length,
                    length,
                    device=device,
                    fixed_len=length,
                    fixed_query_type=q_type,
                )
                state, q_logits = model(batch.ops, batch.args, batch.lengths, batch.init_delta)
                qm = query_metrics(q_logits, batch.query_type, batch.query_probs)
                bm = belief_metrics(probe(state), batch.target_probs[:, length])
                pair_logits = model.belief_from_states(state)
                dm = rename_belief_metrics("decoder", belief_metrics(pair_logits, batch.target_probs[:, length]))
                dqm = query_metrics_from_pair_logits(pair_logits, batch.query_type, batch.query_probs, args.modulus)
                sums["n"] += bsz
                sums["q_mass"] += qm["query_target_mass"] * bsz
                sums["q_nll"] += qm["query_target_nll"] * bsz
                sums["q_top1"] += qm["query_top1_on_support"] * bsz
                sums["q_support"] += qm["mean_query_support_size"] * bsz
                sums["b_mass"] += bm["probe_belief_target_mass"] * bsz
                sums["b_nll"] += bm["probe_belief_target_nll"] * bsz
                sums["b_top1"] += bm["probe_belief_top1_on_support"] * bsz
                sums["b_support"] += bm["mean_belief_support_size"] * bsz
                sums["d_mass"] += dm["decoder_belief_target_mass"] * bsz
                sums["d_nll"] += dm["decoder_belief_target_nll"] * bsz
                sums["d_top1"] += dm["decoder_belief_top1_on_support"] * bsz
                sums["d_support"] += dm["mean_decoder_belief_support_size"] * bsz
                sums["dq_mass"] += dqm["decoder_query_target_mass"] * bsz
                sums["dq_nll"] += dqm["decoder_query_target_nll"] * bsz
                sums["dq_top1"] += dqm["decoder_query_top1_on_support"] * bsz
                sums["dq_support"] += dqm["mean_decoder_query_support_size"] * bsz
            n = max(1, int(sums["n"]))
            rows.append(
                {
                    "model": args.mode,
                    "variant": args.variant_name,
                    "supervision": args.supervision,
                    "transition": args.transition,
                    "decoder_type": args.decoder_type,
                    "state_dim": args.state_dim,
                    "instr_dim": args.instr_dim,
                    "decoder_rank": args.decoder_rank,
                    "length": int(length),
                    "query_type": QUERY_NAMES[q_type],
                    "k": -1,
                    "n": n,
                    "query_target_mass": sums["q_mass"] / n,
                    "query_target_nll": sums["q_nll"] / n,
                    "query_top1_on_support": sums["q_top1"] / n,
                    "mean_query_support_size": sums["q_support"] / n,
                    "probe_belief_target_mass": sums["b_mass"] / n,
                    "probe_belief_target_nll": sums["b_nll"] / n,
                    "probe_belief_top1_on_support": sums["b_top1"] / n,
                    "mean_belief_support_size": sums["b_support"] / n,
                    "decoder_belief_target_mass": sums["d_mass"] / n,
                    "decoder_belief_target_nll": sums["d_nll"] / n,
                    "decoder_belief_top1_on_support": sums["d_top1"] / n,
                    "mean_decoder_belief_support_size": sums["d_support"] / n,
                    "decoder_query_target_mass": sums["dq_mass"] / n,
                    "decoder_query_target_nll": sums["dq_nll"] / n,
                    "decoder_query_top1_on_support": sums["dq_top1"] / n,
                    "mean_decoder_query_support_size": sums["dq_support"] / n,
                }
            )
    return rows


def write_metrics_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
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
        parts = []
        for row in sorted(by_len[length], key=lambda r: (str(r["query_type"]), int(r["k"]))):
            label = "static" if int(row["k"]) < 0 else f"K={int(row['k'])}"
            parts.append(
                f"{row['query_type']}/{label}:qtop={row['query_top1_on_support']*100:.1f}% "
                f"qmass={row['query_target_mass']*100:.1f}% "
                f"dqmass={row['decoder_query_target_mass']*100:.1f}% "
                f"pbmass={row['probe_belief_target_mass']*100:.1f}% "
                f"dbmass={row['decoder_belief_target_mass']*100:.1f}% "
                f"pnll={row['probe_belief_target_nll']:.3f}"
            )
        print(f"  L={length} " + " | ".join(parts), flush=True)


def checkpoint_path(args: argparse.Namespace) -> Path:
    root = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else Path("large_artifacts/dense_teacher_distillation/checkpoints") / Path(args.output_dir).name
    )
    return root / "checkpoint_final.pt"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    probe: BeliefProbe,
    optimizer_state: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    results: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "probe_state": probe.state_dict(),
            "optimizer_state": optimizer_state,
            "args": vars(args),
            "results": results,
        },
        path,
    )


def train_and_evaluate(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    model, results = train_executor(args, device)
    probe = train_probe(model, args, device, results)
    eval_lengths = parse_int_list(args.eval_lengths)
    eval_k = parse_int_list(args.eval_k)
    eval_gen = FilterProgramGenerator(args.modulus, args.observe_mod, args.observe_prob, args.seed + 10_000, args.op_family, args.query_family)
    if recurrent_model(model):
        rows = evaluate_recurrent(model, probe, eval_gen, args, device, eval_lengths, eval_k, args.eval_examples)
    else:
        rows = evaluate_static(model, probe, eval_gen, args, device, eval_lengths, args.eval_examples)
    results["eval"].append({"step": args.train_steps, "rows": rows})
    print_eval_summary(f"[eval/{args.variant_name} final]", rows)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(out / "metrics_final.csv", rows)
    ckpt = checkpoint_path(args)
    save_checkpoint(ckpt, model, probe, None, args, results)
    results["checkpoints"].append(str(ckpt))
    with (out / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dense teacher-distillation bottleneck executor")
    p.add_argument("--variant_name", type=str, default="dense_teacher")
    p.add_argument("--mode", choices=["dense", "static", "shuffled"], default="dense")
    p.add_argument(
        "--supervision",
        choices=["teacher_belief"],
        default="teacher_belief",
    )
    p.add_argument("--seed", type=int, default=53)
    p.add_argument("--modulus", type=int, default=11)
    p.add_argument("--observe_mod", type=int, default=4)
    p.add_argument("--observe_prob", type=float, default=0.3)
    p.add_argument("--op_family", choices=["full", "const", "cross"], default="full")
    p.add_argument("--query_family", choices=["all", "marginal", "relational"], default="all")
    p.add_argument("--train_min_len", type=int, default=1)
    p.add_argument("--train_max_len", type=int, default=6)
    p.add_argument("--eval_lengths", type=str, default="3,6,9,12")
    p.add_argument("--eval_k", type=str, default="0,1,2,3,6,9,12")
    p.add_argument("--eval_query_types", type=str, default="all")
    p.add_argument("--state_dim", type=int, default=256)
    p.add_argument("--instr_dim", type=int, default=128)
    p.add_argument("--transition", choices=["gru", "residual"], default="gru")
    p.add_argument("--decoder_type", choices=["mlp", "low_rank"], default="mlp")
    p.add_argument("--decoder_rank", type=int, default=16)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--compiler_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--eval_examples", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--query_loss_weight", type=float, default=0.0)
    p.add_argument("--belief_loss_weight", type=float, default=1.0)
    p.add_argument("--decoder_query_loss_weight", type=float, default=0.0)
    p.add_argument("--belief_distill_prob", type=float, default=1.0)
    p.add_argument("--probe_steps", type=int, default=1000)
    p.add_argument("--probe_batch_size", type=int, default=512)
    p.add_argument("--probe_lr", type=float, default=1e-3)
    p.add_argument("--probe_weight_decay", type=float, default=1e-4)
    p.add_argument("--probe_dropout", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--probe_log_every", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="experiments/dense_teacher_distillation/runs/dense_teacher")
    p.add_argument("--checkpoint_dir", type=str, default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(json.dumps(vars(args), indent=2), flush=True)
    print(f"[env] python={sys.version.split()[0]} torch={torch.__version__} device={device}", flush=True)
    results = train_and_evaluate(args, device)
    with (Path(args.output_dir) / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
