#!/usr/bin/env python3
"""Joint-state latent recurrent executor experiment.

This script tests whether a hidden recurrent runtime can execute modular
programs whose instructions couple two registers. The primary model keeps a
latent categorical distribution over joint states (A, B), applies one learned
transition per hidden step, and measures whether exact final-state accuracy
scales with the internal recurrent step budget K.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
]
CONST_A_OPS = {0, 1}
CONST_B_OPS = {2, 3}
CROSS_A_OPS = {4, 6}
CROSS_B_OPS = {5, 7}


@dataclass
class ProgramBatch:
    ops: torch.Tensor
    args: torch.Tensor
    lengths: torch.Tensor
    init_a: torch.Tensor
    init_b: torch.Tensor
    init_delta: torch.Tensor
    trace: torch.Tensor  # [B, trace_k + 1, 2], trace[:, t] is state after t ops
    support: torch.Tensor  # [B, trace_k + 1, S, 2], valid target support at each step


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return vals


def apply_op(a: int, b: int, op: int, arg: int, modulus: int) -> Tuple[int, int]:
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
        raise ValueError(f"unknown op {op}")
    return a, b


class RegisterProgramGenerator:
    def __init__(self, modulus: int, seed: int, op_family: str, task: str) -> None:
        self.modulus = int(modulus)
        self.rng = random.Random(seed)
        self.task = task
        if op_family == "full":
            self.op_choices = list(range(len(OP_NAMES)))
        elif op_family == "const":
            self.op_choices = [0, 1, 2, 3]
        elif op_family == "cross":
            self.op_choices = [4, 5, 6, 7]
        else:
            raise ValueError(f"unknown op_family {op_family}")

    def batch(
        self,
        batch_size: int,
        min_len: int,
        max_len: int,
        trace_k: int,
        device: torch.device,
        fixed_len: Optional[int] = None,
    ) -> ProgramBatch:
        if fixed_len is not None:
            min_len = max_len = int(fixed_len)
        lengths = [self.rng.randint(min_len, max_len) for _ in range(batch_size)]
        program_len = max(max_len, max(lengths), trace_k)
        support_size = self.modulus if self.task == "belief_line" else 1
        ops = torch.zeros(batch_size, program_len, dtype=torch.long)
        args = torch.zeros(batch_size, program_len, dtype=torch.long)
        init_a = torch.empty(batch_size, dtype=torch.long)
        init_b = torch.empty(batch_size, dtype=torch.long)
        init_delta = torch.zeros(batch_size, dtype=torch.long)
        trace = torch.empty(batch_size, trace_k + 1, 2, dtype=torch.long)
        support = torch.empty(batch_size, trace_k + 1, support_size, 2, dtype=torch.long)

        for i, length in enumerate(lengths):
            if self.task == "exact":
                a = self.rng.randrange(self.modulus)
                b = self.rng.randrange(self.modulus)
                states = [(a, b)]
                init_a[i] = a
                init_b[i] = b
            elif self.task == "belief_line":
                delta = self.rng.randrange(self.modulus)
                init_delta[i] = delta
                states = [(a, (a + delta) % self.modulus) for a in range(self.modulus)]
                # Static controls receive the observed relation parameter through init_a.
                init_a[i] = delta
                init_b[i] = 0
            else:
                raise ValueError(f"unknown task {self.task}")
            trace[i, 0, 0] = states[0][0]
            trace[i, 0, 1] = states[0][1]
            for s, (sa, sb) in enumerate(states):
                support[i, 0, s, 0] = sa
                support[i, 0, s, 1] = sb
            sampled_ops: List[int] = []
            sampled_args: List[int] = []
            for _ in range(length):
                op = self.rng.choice(self.op_choices)
                arg = self.rng.randint(1, self.modulus - 1) if op in CONST_A_OPS or op in CONST_B_OPS else 0
                sampled_ops.append(op)
                sampled_args.append(arg)
            cur_states = list(states)
            for t in range(program_len):
                if t < length:
                    op = sampled_ops[t]
                    arg = sampled_args[t]
                    ops[i, t] = op
                    args[i, t] = arg
                    cur_states = [apply_op(sa, sb, op, arg, self.modulus) for sa, sb in cur_states]
                if t + 1 <= trace_k:
                    trace[i, t + 1, 0] = cur_states[0][0]
                    trace[i, t + 1, 1] = cur_states[0][1]
                    for s, (sa, sb) in enumerate(cur_states):
                        support[i, t + 1, s, 0] = sa
                        support[i, t + 1, s, 1] = sb

        return ProgramBatch(
            ops=ops.to(device),
            args=args.to(device),
            lengths=torch.tensor(lengths, dtype=torch.long, device=device),
            init_a=init_a.to(device),
            init_b=init_b.to(device),
            init_delta=init_delta.to(device),
            trace=trace.to(device),
            support=support.to(device),
        )


def safe_log_probs(probs: torch.Tensor) -> torch.Tensor:
    return probs.clamp_min(1e-9).log()


def pair_targets(trace: torch.Tensor, modulus: int, steps: int) -> torch.Tensor:
    target = trace[:, :steps, 0] * modulus + trace[:, :steps, 1]
    return target.reshape(-1)


def support_indices(support: torch.Tensor, modulus: int) -> torch.Tensor:
    return support[..., 0] * modulus + support[..., 1]


def support_nll_loss(log_pair: torch.Tensor, support: torch.Tensor, modulus: int) -> torch.Tensor:
    bsz, steps, pair_dim = log_pair.shape
    if pair_dim != modulus * modulus:
        raise ValueError(f"expected pair_dim={modulus * modulus}, got {pair_dim}")
    idx = support_indices(support[:, :steps], modulus)
    selected = log_pair.gather(2, idx.reshape(bsz, steps, -1))
    return -selected.mean()


def final_support_ce_loss(logits_pair: torch.Tensor, support: torch.Tensor, modulus: int) -> torch.Tensor:
    log_pair = F.log_softmax(logits_pair, dim=-1)
    idx = support_indices(support[:, -1], modulus)
    selected = log_pair.gather(1, idx)
    return -selected.mean()


def support_metrics(log_pair: torch.Tensor, support: torch.Tensor, modulus: int) -> Dict[str, float]:
    idx = support_indices(support, modulus)
    probs = log_pair.exp()
    selected_log = log_pair.gather(1, idx)
    selected_prob = probs.gather(1, idx)
    pred = log_pair.argmax(dim=-1)
    top1 = (pred.unsqueeze(1) == idx).any(dim=1).float().mean().item()
    out: Dict[str, float] = {
        "target_nll": float((-selected_log.mean()).detach().cpu()),
        "target_mass": float(selected_prob.sum(dim=1).mean().detach().cpu()),
        "top1_on_support": float(top1),
        "pair_accuracy": float(top1),
    }
    if support.shape[1] == 1:
        target = idx[:, 0]
        pred_a = pred // modulus
        pred_b = pred % modulus
        target_a = target // modulus
        target_b = target % modulus
        out["register_accuracy"] = float(
            (0.5 * ((pred_a == target_a).float() + (pred_b == target_b).float())).mean().item()
        )
    else:
        out["register_accuracy"] = float("nan")
    return out


class JointCategoricalExecutor(nn.Module):
    """Latent recurrent executor with a joint categorical state over (A, B)."""

    def __init__(self, modulus: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.temperature = float(temperature)
        # [op, selector, current_value, next_value]
        # selector is arg for constant ops, current B for A<-A±B, current A for B<-B±A.
        self.transition_logits = nn.Parameter(torch.randn(len(OP_NAMES), modulus, modulus, modulus) * 0.01)
        self.last_aux: Dict[str, float] = {}

    def initial_state(
        self,
        init_a: torch.Tensor,
        init_b: torch.Tensor,
        init_delta: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        bsz = init_a.shape[0]
        dist = torch.zeros(bsz, self.modulus, self.modulus, dtype=torch.float32, device=init_a.device)
        if task == "exact":
            dist[torch.arange(bsz, device=init_a.device), init_a, init_b] = 1.0
        elif task == "belief_line":
            a_vals = torch.arange(self.modulus, device=init_a.device).view(1, self.modulus).expand(bsz, -1)
            b_vals = (a_vals + init_delta.view(bsz, 1)) % self.modulus
            dist[torch.arange(bsz, device=init_a.device).view(bsz, 1), a_vals, b_vals] = 1.0 / self.modulus
        else:
            raise ValueError(f"unknown task {task}")
        return dist

    def transition(self, op: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
        logits = self.transition_logits[op, selector] / self.temperature
        return F.softmax(logits, dim=-1)

    def transition_family(self, op: torch.Tensor) -> torch.Tensor:
        logits = self.transition_logits[op] / self.temperature
        return F.softmax(logits, dim=-1)

    def log_pair(self, dist: torch.Tensor) -> torch.Tensor:
        return safe_log_probs(dist.reshape(dist.shape[0], self.modulus * self.modulus))

    def forward(
        self,
        ops: torch.Tensor,
        args: torch.Tensor,
        lengths: torch.Tensor,
        init_a: torch.Tensor,
        init_b: torch.Tensor,
        init_delta: torch.Tensor,
        task: str,
        k_max: int,
    ) -> torch.Tensor:
        bsz = init_a.shape[0]
        dist = self.initial_state(init_a, init_b, init_delta, task)
        logs = [self.log_pair(dist)]
        entropy_accum = 0.0

        for t in range(k_max):
            active = (lengths > t).float().view(bsz, 1, 1)
            op = ops[:, t] if t < ops.shape[1] else torch.zeros(bsz, dtype=torch.long, device=ops.device)
            arg = args[:, t] if t < args.shape[1] else torch.zeros(bsz, dtype=torch.long, device=args.device)

            trans_arg = self.transition(op, arg)
            const_a = torch.einsum("nab,nac->ncb", dist, trans_arg)
            const_b = torch.einsum("nab,nbd->nad", dist, trans_arg)

            trans_family = self.transition_family(op)
            cross_a = torch.einsum("nab,nbac->ncb", dist, trans_family)
            cross_b = torch.einsum("nab,nabd->nad", dist, trans_family)

            m_const_a = ((op == 0) | (op == 1)).float().view(bsz, 1, 1)
            m_const_b = ((op == 2) | (op == 3)).float().view(bsz, 1, 1)
            m_cross_a = ((op == 4) | (op == 6)).float().view(bsz, 1, 1)
            m_cross_b = ((op == 5) | (op == 7)).float().view(bsz, 1, 1)
            updated = m_const_a * const_a + m_const_b * const_b + m_cross_a * cross_a + m_cross_b * cross_b
            dist = active * updated + (1.0 - active) * dist
            dist = dist / dist.sum(dim=(1, 2), keepdim=True).clamp_min(1e-9)
            logs.append(self.log_pair(dist))

            with torch.no_grad():
                chosen = trans_arg
                ent = -(chosen.clamp_min(1e-9) * chosen.clamp_min(1e-9).log()).sum(dim=-1).mean()
                entropy_accum += float(ent.detach().cpu())

        self.last_aux = {"transition_entropy": entropy_accum / max(1, k_max)}
        return torch.stack(logs, dim=1)


class MarginalCategoricalExecutor(nn.Module):
    """Control executor with separate A and B marginals and no joint state."""

    def __init__(self, modulus: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.temperature = float(temperature)
        self.transition_logits = nn.Parameter(torch.randn(len(OP_NAMES), modulus, modulus, modulus) * 0.01)
        self.last_aux: Dict[str, float] = {}

    def transition(self, op: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.transition_logits[op, selector] / self.temperature, dim=-1)

    def transition_family(self, op: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.transition_logits[op] / self.temperature, dim=-1)

    def log_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        log_joint = safe_log_probs(a).unsqueeze(2) + safe_log_probs(b).unsqueeze(1)
        return log_joint.reshape(a.shape[0], self.modulus * self.modulus)

    def forward(
        self,
        ops: torch.Tensor,
        args: torch.Tensor,
        lengths: torch.Tensor,
        init_a: torch.Tensor,
        init_b: torch.Tensor,
        init_delta: torch.Tensor,
        task: str,
        k_max: int,
    ) -> torch.Tensor:
        bsz = init_a.shape[0]
        if task == "exact":
            a = F.one_hot(init_a, num_classes=self.modulus).float()
            b = F.one_hot(init_b, num_classes=self.modulus).float()
        elif task == "belief_line":
            a = torch.full((bsz, self.modulus), 1.0 / self.modulus, dtype=torch.float32, device=init_a.device)
            b = torch.full((bsz, self.modulus), 1.0 / self.modulus, dtype=torch.float32, device=init_a.device)
        else:
            raise ValueError(f"unknown task {task}")
        logs = [self.log_pair(a, b)]
        entropy_accum = 0.0

        for t in range(k_max):
            active = (lengths > t).float().view(bsz, 1)
            op = ops[:, t] if t < ops.shape[1] else torch.zeros(bsz, dtype=torch.long, device=ops.device)
            arg = args[:, t] if t < args.shape[1] else torch.zeros(bsz, dtype=torch.long, device=args.device)
            trans_arg = self.transition(op, arg)
            const_a = torch.einsum("na,nac->nc", a, trans_arg)
            const_b = torch.einsum("nb,nbd->nd", b, trans_arg)

            trans_family = self.transition_family(op)
            cross_a = torch.einsum("nb,na,nbac->nc", b, a, trans_family)
            cross_b = torch.einsum("na,nb,nabd->nd", a, b, trans_family)

            m_const_a = ((op == 0) | (op == 1)).float().view(bsz, 1)
            m_const_b = ((op == 2) | (op == 3)).float().view(bsz, 1)
            m_cross_a = ((op == 4) | (op == 6)).float().view(bsz, 1)
            m_cross_b = ((op == 5) | (op == 7)).float().view(bsz, 1)
            new_a = m_const_a * const_a + m_cross_a * cross_a + (1.0 - m_const_a - m_cross_a) * a
            new_b = m_const_b * const_b + m_cross_b * cross_b + (1.0 - m_const_b - m_cross_b) * b
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


class StaticCompilerBaseline(nn.Module):
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
        self.value_embed = nn.Embedding(modulus, dim)
        self.reg_embed = nn.Embedding(2, dim)
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
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, modulus * modulus),
        )

    def forward(
        self,
        ops: torch.Tensor,
        args: torch.Tensor,
        lengths: torch.Tensor,
        init_a: torch.Tensor,
        init_b: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq = ops.shape
        pos = torch.arange(seq, device=ops.device).unsqueeze(0).expand(bsz, seq)
        x = self.op_embed(ops) + self.arg_embed(args) + self.pos_embed(pos)
        h = self.compiler(x)
        mask = (torch.arange(seq, device=ops.device).unsqueeze(0) < lengths.unsqueeze(1)).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        a = self.value_embed(init_a) + self.reg_embed(torch.zeros_like(init_a))
        b = self.value_embed(init_b) + self.reg_embed(torch.ones_like(init_b))
        return self.head(torch.cat([pooled, a, b], dim=-1))


@torch.no_grad()
def evaluate_recurrent(
    model: nn.Module,
    generator: RegisterProgramGenerator,
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
    for length in lengths:
        totals = {k: 0 for k in k_values}
        correct_pair = {k: 0 for k in k_values}
        correct_a = {k: 0 for k in k_values}
        correct_b = {k: 0 for k in k_values}
        trace_k = max(length, max_k)
        while totals[k_values[0]] < examples_per_length:
            bsz = min(batch_size, examples_per_length - totals[k_values[0]])
            batch = generator.batch(
                bsz,
                min_len=length,
                max_len=length,
                trace_k=trace_k,
                fixed_len=length,
                device=device,
            )
            log_pair = model(
                batch.ops,
                batch.args,
                batch.lengths,
                batch.init_a,
                batch.init_b,
                batch.init_delta,
                args.task,
                k_max=max_k,
            )
            target_support = batch.support[:, length]
            for k in k_values:
                metrics = support_metrics(log_pair[:, k, :], target_support, args.modulus)
                correct_pair[k] += int(round(metrics["top1_on_support"] * bsz))
                correct_a[k] += int(round(metrics["target_mass"] * bsz * 1_000_000))
                correct_b[k] += int(round(metrics["target_nll"] * bsz * 1_000_000))
                totals[k] += bsz
        for k in k_values:
            rows.append(
                {
                    "model": args.mode,
                    "length": int(length),
                    "k": int(k),
                    "n": int(totals[k]),
                    "pair_accuracy": correct_pair[k] / max(1, totals[k]),
                    "a_accuracy": float("nan"),
                    "b_accuracy": float("nan"),
                    "register_accuracy": float("nan"),
                    "target_mass": correct_a[k] / max(1, totals[k] * 1_000_000),
                    "target_nll": correct_b[k] / max(1, totals[k] * 1_000_000),
                    "top1_on_support": correct_pair[k] / max(1, totals[k]),
                }
            )
    model.train()
    return rows


@torch.no_grad()
def evaluate_static(
    model: StaticCompilerBaseline,
    generator: RegisterProgramGenerator,
    args: argparse.Namespace,
    device: torch.device,
    lengths: Sequence[int],
    examples_per_length: int,
) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    batch_size = min(args.eval_batch_size, examples_per_length)
    for length in lengths:
        total = 0
        correct_pair = 0
        correct_a = 0
        correct_b = 0
        while total < examples_per_length:
            bsz = min(batch_size, examples_per_length - total)
            batch = generator.batch(
                bsz,
                min_len=length,
                max_len=length,
                trace_k=length,
                fixed_len=length,
                device=device,
            )
            logits_pair = model(batch.ops, batch.args, batch.lengths, batch.init_a, batch.init_b)
            metrics = support_metrics(F.log_softmax(logits_pair, dim=-1), batch.support[:, length], args.modulus)
            correct_pair += int(round(metrics["top1_on_support"] * bsz))
            correct_a += int(round(metrics["target_mass"] * bsz * 1_000_000))
            correct_b += int(round(metrics["target_nll"] * bsz * 1_000_000))
            total += bsz
        rows.append(
            {
                "model": args.mode,
                "length": int(length),
                "k": -1,
                "n": int(total),
                "pair_accuracy": correct_pair / max(1, total),
                "a_accuracy": float("nan"),
                "b_accuracy": float("nan"),
                "register_accuracy": float("nan"),
                "target_mass": correct_a / max(1, total * 1_000_000),
                "target_nll": correct_b / max(1, total * 1_000_000),
                "top1_on_support": correct_pair / max(1, total),
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
        for row in sorted(by_len[length], key=lambda r: int(r["k"])):
            label = "static" if int(row["k"]) < 0 else f"K={int(row['k'])}"
            parts.append(
                f"{label}:top={row['top1_on_support']*100:.1f}% mass={row['target_mass']*100:.1f}% nll={row['target_nll']:.3f}"
            )
        print(f"  L={length} " + " | ".join(parts), flush=True)


def build_model(args: argparse.Namespace, max_program_len: int) -> nn.Module:
    if args.mode == "joint":
        return JointCategoricalExecutor(args.modulus, temperature=args.transition_temperature)
    if args.mode == "marginal":
        return MarginalCategoricalExecutor(args.modulus, temperature=args.transition_temperature)
    if args.mode == "static":
        return StaticCompilerBaseline(
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
    train_gen = RegisterProgramGenerator(args.modulus, seed=args.seed, op_family=args.op_family, task=args.task)
    eval_gen = RegisterProgramGenerator(args.modulus, seed=args.seed + 10_000, op_family=args.op_family, task=args.task)
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
        batch = train_gen.batch(
            args.batch_size,
            min_len=args.train_min_len,
            max_len=args.train_max_len,
            trace_k=args.train_max_len,
            device=device,
        )
        if args.mode == "static":
            logits_pair = model(batch.ops, batch.args, batch.lengths, batch.init_a, batch.init_b)
            loss = final_support_ce_loss(logits_pair, batch.support, args.modulus)
            with torch.no_grad():
                metrics = support_metrics(F.log_softmax(logits_pair, dim=-1), batch.support[:, -1], args.modulus)
                pair_acc = metrics["top1_on_support"]
        else:
            log_pair = model(
                batch.ops,
                batch.args,
                batch.lengths,
                batch.init_a,
                batch.init_b,
                batch.init_delta,
                args.task,
                k_max=args.train_max_len,
            )
            loss = support_nll_loss(log_pair, batch.support, args.modulus)
            with torch.no_grad():
                metrics = support_metrics(log_pair[:, -1, :], batch.support[:, -1], args.modulus)
                pair_acc = metrics["top1_on_support"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            aux = getattr(model, "last_aux", {})
            aux_text = " ".join(f"{k}={v:.4g}" for k, v in aux.items())
            print(
                f"[train/{args.mode}] step={step:05d} loss={float(loss.detach().cpu()):.4f} "
                f"pair={pair_acc*100:.1f}% grad={float(grad):.3f} elapsed={(time.time()-t0)/60:.1f}m {aux_text}",
                flush=True,
            )
            results["train"].append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "pair_acc": float(pair_acc),
                    "grad_norm": float(grad),
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
            with (out / "results.json").open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            write_metrics_csv(out / f"metrics_step{step:05d}.csv", rows)
            ckpt = checkpoint_path(args, step)
            save_checkpoint(ckpt, model, optimizer, args, results)
            results["checkpoints"].append(str(ckpt))
            with (out / "results.json").open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Joint-state latent recurrent executor")
    p.add_argument("--mode", choices=["joint", "marginal", "static"], default="joint")
    p.add_argument("--task", choices=["exact", "belief_line"], default="exact")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--modulus", type=int, default=31)
    p.add_argument("--op_family", choices=["full", "const", "cross"], default="full")
    p.add_argument("--train_min_len", type=int, default=1)
    p.add_argument("--train_max_len", type=int, default=8)
    p.add_argument("--eval_lengths", type=str, default="4,8,12,16,24")
    p.add_argument("--eval_k", type=str, default="0,1,2,4,8,12,16,24")
    p.add_argument("--transition_temperature", type=float, default=1.0)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--compiler_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--train_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--eval_examples", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-3)
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
