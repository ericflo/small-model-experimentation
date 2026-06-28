#!/usr/bin/env python3
"""Latent recurrent query-supervised belief filtering experiment.

Programs transform a correlated belief state over two modular registers using
arithmetic operations and observation filters. Training supervises only final
query distributions such as A, B, A+B, or A-B. Evaluation also audits the latent
belief state against the exact final support.
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
    query_probs: torch.Tensor  # [B, p], normalized final-query target
    target_probs: torch.Tensor  # [B, trace_k + 1, p*p], normalized target belief
    support_size: torch.Tensor  # [B, trace_k + 1]


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return vals


def parse_query_types(text: str) -> List[int]:
    if text == "all":
        return list(range(len(QUERY_NAMES)))
    out: List[int] = []
    aliases = {name: i for i, name in enumerate(QUERY_NAMES)}
    aliases.update({"sum": 2, "diff": 3, "a": 0, "b": 1})
    for part in text.split(","):
        key = part.strip()
        if not key:
            continue
        if key.isdigit():
            val = int(key)
        else:
            val = aliases.get(key.upper(), aliases.get(key.lower(), -1))
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


def query_index_table(modulus: int, device: torch.device) -> torch.Tensor:
    a_vals = torch.arange(modulus, device=device).repeat_interleave(modulus)
    b_vals = torch.arange(modulus, device=device).repeat(modulus)
    return torch.stack(
        [
            a_vals,
            b_vals,
            (a_vals + b_vals) % modulus,
            (a_vals - b_vals) % modulus,
        ],
        dim=0,
    )


def pair_probs_to_query_probs(pair_probs: torch.Tensor, query_type: torch.Tensor, modulus: int) -> torch.Tensor:
    table = query_index_table(modulus, pair_probs.device)
    idx = table[query_type]
    out = torch.zeros(pair_probs.shape[0], modulus, dtype=pair_probs.dtype, device=pair_probs.device)
    out.scatter_add_(1, idx, pair_probs)
    return out


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
        arg = self.rng.choice(residues)
        return op, arg

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
            query_probs=query_probs,
            target_probs=target_probs,
            support_size=support_size,
        )


def safe_log_probs(probs: torch.Tensor) -> torch.Tensor:
    return probs.clamp_min(1e-9).log()


def belief_loss(log_pair: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    steps = log_pair.shape[1]
    return -(target_probs[:, :steps] * log_pair).sum(dim=-1).mean()


def final_belief_loss(logits_pair: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    log_pair = F.log_softmax(logits_pair, dim=-1)
    return -(target_probs[:, -1] * log_pair).sum(dim=-1).mean()


def query_loss(log_pair: torch.Tensor, query_type: torch.Tensor, target_query: torch.Tensor, modulus: int) -> torch.Tensor:
    query_probs = pair_probs_to_query_probs(log_pair.exp(), query_type, modulus)
    return -(target_query * safe_log_probs(query_probs)).sum(dim=-1).mean()


def query_metrics(log_pair: torch.Tensor, query_type: torch.Tensor, target_query: torch.Tensor, modulus: int) -> Dict[str, float]:
    query_probs = pair_probs_to_query_probs(log_pair.exp(), query_type, modulus)
    query_log = safe_log_probs(query_probs)
    mask = target_query > 0
    target_mass = (query_probs * mask.float()).sum(dim=-1)
    target_nll = -(target_query * query_log).sum(dim=-1)
    pred = query_log.argmax(dim=-1)
    top1 = mask.gather(1, pred.unsqueeze(1)).squeeze(1).float()
    return {
        "query_target_mass": float(target_mass.mean().detach().cpu()),
        "query_target_nll": float(target_nll.mean().detach().cpu()),
        "query_top1_on_support": float(top1.mean().detach().cpu()),
        "mean_query_support_size": float(mask.sum(dim=-1).float().mean().detach().cpu()),
    }


def belief_metrics(log_pair: torch.Tensor, target_probs: torch.Tensor) -> Dict[str, float]:
    probs = log_pair.exp()
    mask = target_probs > 0
    target_mass = (probs * mask.float()).sum(dim=-1)
    target_nll = -(target_probs * log_pair).sum(dim=-1)
    pred = log_pair.argmax(dim=-1)
    top1 = mask.gather(1, pred.unsqueeze(1)).squeeze(1).float()
    return {
        "belief_target_mass": float(target_mass.mean().detach().cpu()),
        "belief_target_nll": float(target_nll.mean().detach().cpu()),
        "belief_top1_on_support": float(top1.mean().detach().cpu()),
        "mean_belief_support_size": float(mask.sum(dim=-1).float().mean().detach().cpu()),
    }


def all_metrics(
    log_pair: torch.Tensor,
    target_pair: torch.Tensor,
    query_type: torch.Tensor,
    target_query: torch.Tensor,
    modulus: int,
) -> Dict[str, float]:
    return {
        **query_metrics(log_pair, query_type, target_query, modulus),
        **belief_metrics(log_pair, target_pair),
    }


class JointFilterExecutor(nn.Module):
    def __init__(self, modulus: int, observe_mod: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.observe_mod = int(observe_mod)
        self.temperature = float(temperature)
        self.transition_logits = nn.Parameter(torch.randn(8, modulus, modulus, modulus) * 0.01)
        self.observe_logits = nn.Parameter(torch.zeros(2, observe_mod, modulus))
        self.last_aux: Dict[str, float] = {}

    def initial_state(self, init_delta: torch.Tensor) -> torch.Tensor:
        bsz = init_delta.shape[0]
        dist = torch.zeros(bsz, self.modulus, self.modulus, dtype=torch.float32, device=init_delta.device)
        a_vals = torch.arange(self.modulus, device=init_delta.device).view(1, self.modulus).expand(bsz, -1)
        b_vals = (a_vals + init_delta.view(bsz, 1)) % self.modulus
        dist[torch.arange(bsz, device=init_delta.device).view(bsz, 1), a_vals, b_vals] = 1.0 / self.modulus
        return dist

    def log_pair(self, dist: torch.Tensor) -> torch.Tensor:
        return safe_log_probs(dist.reshape(dist.shape[0], self.modulus * self.modulus))

    def transition_arg(self, op: torch.Tensor, arg: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.transition_logits[op, arg] / self.temperature, dim=-1)

    def transition_family(self, op: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.transition_logits[op] / self.temperature, dim=-1)

    def observe_likelihood(self, reg: int, arg: torch.Tensor) -> torch.Tensor:
        logits = self.observe_logits[reg, arg]
        return F.softmax(logits, dim=-1)

    def forward(
        self,
        ops: torch.Tensor,
        args: torch.Tensor,
        lengths: torch.Tensor,
        init_delta: torch.Tensor,
        k_max: int,
    ) -> torch.Tensor:
        bsz = init_delta.shape[0]
        dist = self.initial_state(init_delta)
        logs = [self.log_pair(dist)]
        entropy_accum = 0.0

        for t in range(k_max):
            active = (lengths > t).float().view(bsz, 1, 1)
            op = ops[:, t] if t < ops.shape[1] else torch.zeros(bsz, dtype=torch.long, device=ops.device)
            arg = args[:, t] if t < args.shape[1] else torch.zeros(bsz, dtype=torch.long, device=args.device)

            arith_op = op.clamp_max(7)
            arith_arg = arg.clamp_max(self.modulus - 1)
            trans_arg = self.transition_arg(arith_op, arith_arg)
            const_a = torch.einsum("nab,nac->ncb", dist, trans_arg)
            const_b = torch.einsum("nab,nbd->nad", dist, trans_arg)
            trans_family = self.transition_family(arith_op)
            cross_a = torch.einsum("nab,nbac->ncb", dist, trans_family)
            cross_b = torch.einsum("nab,nabd->nad", dist, trans_family)

            obs_arg = arg.clamp_max(self.observe_mod - 1)
            like_a = self.observe_likelihood(0, obs_arg).view(bsz, self.modulus, 1)
            like_b = self.observe_likelihood(1, obs_arg).view(bsz, 1, self.modulus)
            obs_a = dist * like_a
            obs_b = dist * like_b
            obs_a = obs_a / obs_a.sum(dim=(1, 2), keepdim=True).clamp_min(1e-9)
            obs_b = obs_b / obs_b.sum(dim=(1, 2), keepdim=True).clamp_min(1e-9)

            masks = {
                "const_a": ((op == 0) | (op == 1)).float().view(bsz, 1, 1),
                "const_b": ((op == 2) | (op == 3)).float().view(bsz, 1, 1),
                "cross_a": ((op == 4) | (op == 6)).float().view(bsz, 1, 1),
                "cross_b": ((op == 5) | (op == 7)).float().view(bsz, 1, 1),
                "obs_a": (op == OBS_A).float().view(bsz, 1, 1),
                "obs_b": (op == OBS_B).float().view(bsz, 1, 1),
            }
            updated = (
                masks["const_a"] * const_a
                + masks["const_b"] * const_b
                + masks["cross_a"] * cross_a
                + masks["cross_b"] * cross_b
                + masks["obs_a"] * obs_a
                + masks["obs_b"] * obs_b
            )
            dist = active * updated + (1.0 - active) * dist
            dist = dist / dist.sum(dim=(1, 2), keepdim=True).clamp_min(1e-9)
            logs.append(self.log_pair(dist))
            with torch.no_grad():
                ent = -(trans_arg.clamp_min(1e-9) * trans_arg.clamp_min(1e-9).log()).sum(dim=-1).mean()
                entropy_accum += float(ent.detach().cpu())

        self.last_aux = {"transition_entropy": entropy_accum / max(1, k_max)}
        return torch.stack(logs, dim=1)


class MarginalFilterExecutor(nn.Module):
    def __init__(self, modulus: int, observe_mod: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.observe_mod = int(observe_mod)
        self.temperature = float(temperature)
        self.transition_logits = nn.Parameter(torch.randn(8, modulus, modulus, modulus) * 0.01)
        self.observe_logits = nn.Parameter(torch.zeros(2, observe_mod, modulus))
        self.last_aux: Dict[str, float] = {}

    def log_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (safe_log_probs(a).unsqueeze(2) + safe_log_probs(b).unsqueeze(1)).reshape(a.shape[0], self.modulus * self.modulus)

    def transition_arg(self, op: torch.Tensor, arg: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.transition_logits[op, arg] / self.temperature, dim=-1)

    def transition_family(self, op: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.transition_logits[op] / self.temperature, dim=-1)

    def observe_likelihood(self, reg: int, arg: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.observe_logits[reg, arg], dim=-1)

    def forward(
        self,
        ops: torch.Tensor,
        args: torch.Tensor,
        lengths: torch.Tensor,
        init_delta: torch.Tensor,
        k_max: int,
    ) -> torch.Tensor:
        bsz = init_delta.shape[0]
        a = torch.full((bsz, self.modulus), 1.0 / self.modulus, dtype=torch.float32, device=init_delta.device)
        b = torch.full((bsz, self.modulus), 1.0 / self.modulus, dtype=torch.float32, device=init_delta.device)
        logs = [self.log_pair(a, b)]
        entropy_accum = 0.0

        for t in range(k_max):
            active = (lengths > t).float().view(bsz, 1)
            op = ops[:, t] if t < ops.shape[1] else torch.zeros(bsz, dtype=torch.long, device=ops.device)
            arg = args[:, t] if t < args.shape[1] else torch.zeros(bsz, dtype=torch.long, device=args.device)
            arith_op = op.clamp_max(7)
            arith_arg = arg.clamp_max(self.modulus - 1)
            trans_arg = self.transition_arg(arith_op, arith_arg)
            const_a = torch.einsum("na,nac->nc", a, trans_arg)
            const_b = torch.einsum("nb,nbd->nd", b, trans_arg)
            trans_family = self.transition_family(arith_op)
            cross_a = torch.einsum("nb,na,nbac->nc", b, a, trans_family)
            cross_b = torch.einsum("na,nb,nabd->nd", a, b, trans_family)

            obs_arg = arg.clamp_max(self.observe_mod - 1)
            like_a = self.observe_likelihood(0, obs_arg)
            like_b = self.observe_likelihood(1, obs_arg)
            obs_a = a * like_a
            obs_b = b * like_b
            obs_a = obs_a / obs_a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            obs_b = obs_b / obs_b.sum(dim=-1, keepdim=True).clamp_min(1e-9)

            m_const_a = ((op == 0) | (op == 1)).float().view(bsz, 1)
            m_const_b = ((op == 2) | (op == 3)).float().view(bsz, 1)
            m_cross_a = ((op == 4) | (op == 6)).float().view(bsz, 1)
            m_cross_b = ((op == 5) | (op == 7)).float().view(bsz, 1)
            m_obs_a = (op == OBS_A).float().view(bsz, 1)
            m_obs_b = (op == OBS_B).float().view(bsz, 1)
            new_a = m_const_a * const_a + m_cross_a * cross_a + m_obs_a * obs_a + (1.0 - m_const_a - m_cross_a - m_obs_a) * a
            new_b = m_const_b * const_b + m_cross_b * cross_b + m_obs_b * obs_b + (1.0 - m_const_b - m_cross_b - m_obs_b) * b
            a = active * new_a + (1.0 - active) * a
            b = active * new_b + (1.0 - active) * b
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            b = b / b.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            logs.append(self.log_pair(a, b))
            with torch.no_grad():
                ent = -(trans_arg.clamp_min(1e-9) * trans_arg.clamp_min(1e-9).log()).sum(dim=-1).mean()
                entropy_accum += float(ent.detach().cpu())

        self.last_aux = {"transition_entropy": entropy_accum / max(1, k_max)}
        return torch.stack(logs, dim=1)


class StaticFilterBaseline(nn.Module):
    def __init__(
        self,
        modulus: int,
        max_program_len: int,
        dim: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.op_embed = nn.Embedding(len(OP_NAMES), dim)
        self.arg_embed = nn.Embedding(modulus, dim)
        self.pos_embed = nn.Embedding(max_program_len, dim)
        self.delta_embed = nn.Embedding(modulus, dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.compiler = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, modulus * modulus),
        )

    def forward(self, ops: torch.Tensor, args: torch.Tensor, lengths: torch.Tensor, init_delta: torch.Tensor) -> torch.Tensor:
        bsz, seq = ops.shape
        pos = torch.arange(seq, device=ops.device).unsqueeze(0).expand(bsz, seq)
        x = self.op_embed(ops) + self.arg_embed(args) + self.pos_embed(pos)
        h = self.compiler(x)
        mask = (torch.arange(seq, device=ops.device).unsqueeze(0) < lengths.unsqueeze(1)).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.head(torch.cat([pooled, self.delta_embed(init_delta)], dim=-1))
        return logits


@torch.no_grad()
def evaluate_recurrent(
    model: nn.Module,
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
        for q_type in eval_query_types:
            sums = {
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
                }
                for k in k_values
            }
            trace_k = max(length, max_k)
            while sums[k_values[0]]["n"] < examples_per_length:
                bsz = min(batch_size, examples_per_length - sums[k_values[0]]["n"])
                batch = generator.batch(
                    bsz,
                    length,
                    length,
                    trace_k,
                    device=device,
                    fixed_len=length,
                    fixed_query_type=q_type,
                )
                log_pair = model(batch.ops, batch.args, batch.lengths, batch.init_delta, k_max=max_k)
                target = batch.target_probs[:, length]
                for k in k_values:
                    metrics = all_metrics(log_pair[:, k], target, batch.query_type, batch.query_probs, args.modulus)
                    sums[k]["n"] += bsz
                    sums[k]["q_mass"] += metrics["query_target_mass"] * bsz
                    sums[k]["q_nll"] += metrics["query_target_nll"] * bsz
                    sums[k]["q_top1"] += metrics["query_top1_on_support"] * bsz
                    sums[k]["q_support"] += metrics["mean_query_support_size"] * bsz
                    sums[k]["b_mass"] += metrics["belief_target_mass"] * bsz
                    sums[k]["b_nll"] += metrics["belief_target_nll"] * bsz
                    sums[k]["b_top1"] += metrics["belief_top1_on_support"] * bsz
                    sums[k]["b_support"] += metrics["mean_belief_support_size"] * bsz
            for k in k_values:
                n = max(1, int(sums[k]["n"]))
                rows.append(
                    {
                        "model": args.mode,
                        "length": int(length),
                        "query_type": QUERY_NAMES[q_type],
                        "k": int(k),
                        "n": n,
                        "query_target_mass": sums[k]["q_mass"] / n,
                        "query_target_nll": sums[k]["q_nll"] / n,
                        "query_top1_on_support": sums[k]["q_top1"] / n,
                        "mean_query_support_size": sums[k]["q_support"] / n,
                        "belief_target_mass": sums[k]["b_mass"] / n,
                        "belief_target_nll": sums[k]["b_nll"] / n,
                        "belief_top1_on_support": sums[k]["b_top1"] / n,
                        "mean_belief_support_size": sums[k]["b_support"] / n,
                    }
                )
    model.train()
    return rows


@torch.no_grad()
def evaluate_static(
    model: StaticFilterBaseline,
    generator: FilterProgramGenerator,
    args: argparse.Namespace,
    device: torch.device,
    lengths: Sequence[int],
    examples_per_length: int,
) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    batch_size = min(args.eval_batch_size, examples_per_length)
    eval_query_types = parse_query_types(args.eval_query_types)
    for length in lengths:
        for q_type in eval_query_types:
            total = 0
            q_mass = q_nll = q_top1 = q_support = 0.0
            b_mass = b_nll = b_top1 = b_support = 0.0
            while total < examples_per_length:
                bsz = min(batch_size, examples_per_length - total)
                batch = generator.batch(
                    bsz,
                    length,
                    length,
                    length,
                    device=device,
                    fixed_len=length,
                    fixed_query_type=q_type,
                )
                logits = model(batch.ops, batch.args, batch.lengths, batch.init_delta)
                metrics = all_metrics(F.log_softmax(logits, dim=-1), batch.target_probs[:, length], batch.query_type, batch.query_probs, args.modulus)
                q_mass += metrics["query_target_mass"] * bsz
                q_nll += metrics["query_target_nll"] * bsz
                q_top1 += metrics["query_top1_on_support"] * bsz
                q_support += metrics["mean_query_support_size"] * bsz
                b_mass += metrics["belief_target_mass"] * bsz
                b_nll += metrics["belief_target_nll"] * bsz
                b_top1 += metrics["belief_top1_on_support"] * bsz
                b_support += metrics["mean_belief_support_size"] * bsz
                total += bsz
            n = max(1, total)
            rows.append(
                {
                    "model": args.mode,
                    "length": int(length),
                    "query_type": QUERY_NAMES[q_type],
                    "k": -1,
                    "n": int(n),
                    "query_target_mass": q_mass / n,
                    "query_target_nll": q_nll / n,
                    "query_top1_on_support": q_top1 / n,
                    "mean_query_support_size": q_support / n,
                    "belief_target_mass": b_mass / n,
                    "belief_target_nll": b_nll / n,
                    "belief_top1_on_support": b_top1 / n,
                    "mean_belief_support_size": b_support / n,
                }
            )
    model.train()
    return rows


def write_metrics_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(args: argparse.Namespace, step: int) -> Path:
    root = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(args.output_dir)
    return root / f"checkpoint_step{step:05d}.pt"


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, args: argparse.Namespace, results: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "results": results,
        },
        path,
    )


def print_eval_summary(prefix: str, rows: Sequence[Dict[str, Any]]) -> None:
    by_len: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_len.setdefault(int(row["length"]), []).append(row)
    print(prefix, flush=True)
    for length in sorted(by_len):
        parts = []
        for row in sorted(by_len[length], key=lambda r: (str(r["query_type"]), int(r["k"]))):
            label = "static" if int(row["k"]) < 0 else f"K={int(row['k'])}"
            q_label = str(row["query_type"])
            parts.append(
                f"{q_label}/{label}:qtop={row['query_top1_on_support']*100:.1f}% "
                f"qmass={row['query_target_mass']*100:.1f}% bmass={row['belief_target_mass']*100:.1f}% "
                f"qnll={row['query_target_nll']:.3f}"
            )
        print(f"  L={length} " + " | ".join(parts), flush=True)


def build_model(args: argparse.Namespace, max_program_len: int) -> nn.Module:
    if args.mode == "joint":
        return JointFilterExecutor(args.modulus, args.observe_mod, args.transition_temperature)
    if args.mode == "marginal":
        return MarginalFilterExecutor(args.modulus, args.observe_mod, args.transition_temperature)
    if args.mode == "static":
        return StaticFilterBaseline(
            modulus=args.modulus,
            max_program_len=max_program_len,
            dim=args.dim,
            heads=args.heads,
            layers=args.compiler_layers,
            dropout=args.dropout,
        )
    raise ValueError(args.mode)


def train(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    eval_lengths = parse_int_list(args.eval_lengths)
    eval_k = parse_int_list(args.eval_k)
    max_eval_k = max(eval_k)
    max_program_len = max(args.train_max_len, max(eval_lengths), max_eval_k)
    model = build_model(args, max_program_len=max_program_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_gen = FilterProgramGenerator(args.modulus, args.observe_mod, args.observe_prob, args.seed, args.op_family, args.query_family)
    eval_gen = FilterProgramGenerator(args.modulus, args.observe_mod, args.observe_prob, args.seed + 10_000, args.op_family, args.query_family)
    results: Dict[str, Any] = {
        "mode": args.mode,
        "args": vars(args),
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "train": [],
        "eval": [],
        "checkpoints": [],
    }
    t0 = time.time()
    model.train()
    for step in range(1, args.train_steps + 1):
        batch = train_gen.batch(args.batch_size, args.train_min_len, args.train_max_len, args.train_max_len, device=device)
        if args.mode == "static":
            logits = model(batch.ops, batch.args, batch.lengths, batch.init_delta)
            log_pair = F.log_softmax(logits, dim=-1)
            loss = query_loss(log_pair, batch.query_type, batch.query_probs, args.modulus)
            with torch.no_grad():
                metrics = all_metrics(log_pair, batch.target_probs[:, -1], batch.query_type, batch.query_probs, args.modulus)
        else:
            log_pair = model(batch.ops, batch.args, batch.lengths, batch.init_delta, k_max=args.train_max_len)
            loss = query_loss(log_pair[:, -1], batch.query_type, batch.query_probs, args.modulus)
            with torch.no_grad():
                metrics = all_metrics(log_pair[:, -1], batch.target_probs[:, -1], batch.query_type, batch.query_probs, args.modulus)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            aux = getattr(model, "last_aux", {})
            aux_text = " ".join(f"{k}={v:.4g}" for k, v in aux.items())
            print(
                f"[train/{args.mode}] step={step:05d} loss={float(loss.detach().cpu()):.4f} "
                f"qtop={metrics['query_top1_on_support']*100:.1f}% qmass={metrics['query_target_mass']*100:.1f}% "
                f"bmass={metrics['belief_target_mass']*100:.1f}% "
                f"qsupp={metrics['mean_query_support_size']:.1f} bsupp={metrics['mean_belief_support_size']:.1f} "
                f"grad={float(grad):.3f} elapsed={(time.time()-t0)/60:.1f}m {aux_text}",
                flush=True,
            )
            results["train"].append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "grad_norm": float(grad),
                    **metrics,
                    **aux,
                }
            )

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.train_steps):
            if args.mode == "static":
                rows = evaluate_static(model, eval_gen, args, device, eval_lengths, args.eval_examples)
            else:
                rows = evaluate_recurrent(model, eval_gen, args, device, eval_lengths, eval_k, args.eval_examples)
            results["eval"].append({"step": step, "rows": rows})
            print_eval_summary(f"[eval/{args.mode} step={step}]", rows)
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            write_metrics_csv(out / f"metrics_step{step:05d}.csv", rows)
            ckpt = checkpoint_path(args, step)
            save_checkpoint(ckpt, model, optimizer, args, results)
            results["checkpoints"].append(str(ckpt))
            with (out / "results.json").open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Query-supervised latent recurrent belief filter executor")
    p.add_argument("--mode", choices=["joint", "marginal", "static"], default="joint")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--modulus", type=int, default=31)
    p.add_argument("--observe_mod", type=int, default=4)
    p.add_argument("--observe_prob", type=float, default=0.3)
    p.add_argument("--op_family", choices=["full", "const", "cross"], default="full")
    p.add_argument("--query_family", choices=["all", "marginal", "relational"], default="all")
    p.add_argument("--train_min_len", type=int, default=1)
    p.add_argument("--train_max_len", type=int, default=8)
    p.add_argument("--eval_lengths", type=str, default="4,8,12,16,24")
    p.add_argument("--eval_k", type=str, default="0,1,2,4,8,12,16,24")
    p.add_argument("--eval_query_types", type=str, default="all")
    p.add_argument("--transition_temperature", type=float, default=1.0)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--compiler_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--train_steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--eval_examples", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--output_dir", type=str, default="runs/joint")
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
    results = train(args, device)
    with (Path(args.output_dir) / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
