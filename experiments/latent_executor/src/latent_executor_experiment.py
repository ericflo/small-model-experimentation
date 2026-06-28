#!/usr/bin/env python3
"""Trace-supervised latent recurrent executor experiment.

This script tests whether a neural runtime can execute two-register modular
programs one hidden step at a time, then measures whether final accuracy scales
with the internal recurrent step budget K.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
CONST_OPS = {0, 1, 2, 3}


@dataclass
class ProgramBatch:
    ops: torch.Tensor
    args: torch.Tensor
    lengths: torch.Tensor
    init_a: torch.Tensor
    init_b: torch.Tensor
    trace: torch.Tensor  # [B, max_k + 1, 2], trace[:, t] is state after t ops


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return vals


class RegisterProgramGenerator:
    def __init__(self, modulus: int = 97, seed: int = 0, op_family: str = "full") -> None:
        self.modulus = int(modulus)
        self.rng = random.Random(seed)
        if op_family == "full":
            self.op_choices = list(range(len(OP_NAMES)))
        elif op_family == "const":
            self.op_choices = [0, 1, 2, 3]
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
        ops = torch.zeros(batch_size, program_len, dtype=torch.long)
        args = torch.zeros(batch_size, program_len, dtype=torch.long)
        init_a = torch.empty(batch_size, dtype=torch.long)
        init_b = torch.empty(batch_size, dtype=torch.long)
        trace = torch.empty(batch_size, trace_k + 1, 2, dtype=torch.long)

        for i, length in enumerate(lengths):
            a = self.rng.randrange(self.modulus)
            b = self.rng.randrange(self.modulus)
            init_a[i] = a
            init_b[i] = b
            trace[i, 0, 0] = a
            trace[i, 0, 1] = b
            cur_a, cur_b = a, b
            sampled_ops: List[int] = []
            sampled_args: List[int] = []
            for t in range(length):
                op = self.rng.choice(self.op_choices)
                if op in CONST_OPS:
                    arg = self.rng.randint(1, self.modulus - 1)
                else:
                    arg = 0
                sampled_ops.append(op)
                sampled_args.append(arg)
            for t in range(program_len):
                if t < length:
                    op = sampled_ops[t]
                    arg = sampled_args[t]
                    ops[i, t] = op
                    args[i, t] = arg
                    cur_a, cur_b = apply_op(cur_a, cur_b, op, arg, self.modulus)
                # Trace plateaus after the true program length.
                if t + 1 <= trace_k:
                    trace[i, t + 1, 0] = cur_a
                    trace[i, t + 1, 1] = cur_b
        return ProgramBatch(
            ops=ops.to(device),
            args=args.to(device),
            lengths=torch.tensor(lengths, dtype=torch.long, device=device),
            init_a=init_a.to(device),
            init_b=init_b.to(device),
            trace=trace.to(device),
        )


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
        raise ValueError(op)
    return a, b


class DynamicLowRank(nn.Module):
    def __init__(self, dim: int, num_bases: int, rank: int) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.randn(num_bases, dim, rank) / math.sqrt(dim))
        self.b = nn.Parameter(torch.randn(num_bases, rank, dim) * 0.02)

    def forward(self, x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        xa = torch.einsum("bd,ndr->bnr", x, self.a)
        xab = torch.einsum("bnr,nrd->bnd", xa, self.b)
        return torch.einsum("bn,bnd->bd", gates, xab)


class LatentExecutor(nn.Module):
    def __init__(
        self,
        modulus: int,
        max_program_len: int,
        dim: int = 128,
        heads: int = 4,
        compiler_layers: int = 1,
        num_bases: int = 16,
        rank: int = 16,
        mem_dim: int = 48,
        dropout: float = 0.05,
        disable_memory: bool = False,
        disable_dynamic: bool = False,
    ) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.max_program_len = int(max_program_len)
        self.dim = int(dim)
        self.mem_dim = int(mem_dim)
        self.disable_memory = disable_memory
        self.disable_dynamic = disable_dynamic

        self.value_embed = nn.Embedding(modulus, dim)
        self.register_embed = nn.Embedding(2, dim)
        self.op_embed = nn.Embedding(len(OP_NAMES), dim)
        self.arg_embed = nn.Embedding(modulus, dim)
        self.pos_embed = nn.Embedding(max_program_len, dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.compiler = nn.TransformerEncoder(enc_layer, num_layers=compiler_layers)
        self.init = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_bases))
        self.dynamic = DynamicLowRank(dim, num_bases=num_bases, rank=rank)
        self.mem_q = nn.Linear(dim, mem_dim, bias=False)
        self.mem_k = nn.Linear(dim, mem_dim, bias=False)
        self.mem_v = nn.Linear(dim, mem_dim, bias=False)
        self.mem_out = nn.Linear(mem_dim, dim, bias=False)
        self.mem_write = nn.Linear(2 * dim, 1)
        self.gru = nn.GRUCell(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.head_a = nn.Linear(dim, modulus)
        self.head_b = nn.Linear(dim, modulus)
        self.noop = nn.Parameter(torch.zeros(dim))
        self.last_aux: Dict[str, float] = {}

    def encode_program(self, ops: torch.Tensor, args: torch.Tensor) -> torch.Tensor:
        bsz, seq = ops.shape
        if seq > self.max_program_len:
            raise ValueError(f"program length {seq} exceeds model max_program_len={self.max_program_len}")
        pos = torch.arange(seq, device=ops.device).unsqueeze(0).expand(bsz, seq)
        x = self.op_embed(ops) + self.arg_embed(args) + self.pos_embed(pos)
        return self.compiler(x)

    def initial_state(self, init_a: torch.Tensor, init_b: torch.Tensor) -> torch.Tensor:
        a = self.value_embed(init_a) + self.register_embed(torch.zeros_like(init_a))
        b = self.value_embed(init_b) + self.register_embed(torch.ones_like(init_b))
        return self.init(torch.cat([a, b], dim=-1))

    def predict(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(state)
        return self.head_a(h), self.head_b(h)

    def forward(
        self,
        ops: torch.Tensor,
        args: torch.Tensor,
        lengths: torch.Tensor,
        init_a: torch.Tensor,
        init_b: torch.Tensor,
        k_max: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        program = self.encode_program(ops, args)
        state = self.initial_state(init_a, init_b)
        bsz = ops.shape[0]
        mem = torch.zeros(bsz, self.mem_dim, self.mem_dim, dtype=state.dtype, device=state.device)
        logits_a: List[torch.Tensor] = []
        logits_b: List[torch.Tensor] = []
        gate_entropy = 0.0
        mem_rms = 0.0

        la, lb = self.predict(state)
        logits_a.append(la)
        logits_b.append(lb)

        for t in range(k_max):
            active = (lengths > t).float().unsqueeze(-1)
            instr = program[:, t, :] if t < program.shape[1] else self.noop.unsqueeze(0).expand(bsz, -1)
            gates = torch.sigmoid(self.gate(instr))
            dyn = torch.zeros_like(state) if self.disable_dynamic else self.dynamic(state, gates)

            if self.disable_memory:
                read = torch.zeros_like(state)
            else:
                q = torch.tanh(self.mem_q(state))
                read_raw = torch.matmul(q.unsqueeze(1), mem).squeeze(1) / math.sqrt(max(1, self.mem_dim))
                read = self.mem_out(read_raw)
                k = torch.tanh(self.mem_k(state + instr))
                v = torch.tanh(self.mem_v(state + dyn))
                write = torch.matmul(k.unsqueeze(2), v.unsqueeze(1))
                write_gate = torch.sigmoid(self.mem_write(torch.cat([state, instr], dim=-1))).view(bsz, 1, 1)
                mem = mem * 0.92 + active.unsqueeze(-1) * write_gate * write

            inp = torch.cat([instr + dyn, read], dim=-1)
            new_state = self.gru(inp, state)
            state = active * new_state + (1.0 - active) * state
            la, lb = self.predict(state)
            logits_a.append(la)
            logits_b.append(lb)
            with torch.no_grad():
                ent = -(
                    gates.clamp_min(1e-7) * gates.clamp_min(1e-7).log()
                    + (1 - gates).clamp_min(1e-7) * (1 - gates).clamp_min(1e-7).log()
                ).sum(dim=-1).mean()
                gate_entropy = gate_entropy + float(ent.detach().cpu())
                mem_rms = float(mem.detach().pow(2).mean().sqrt().cpu()) if not self.disable_memory else 0.0

        self.last_aux = {
            "gate_entropy": gate_entropy / max(1, k_max),
            "mem_rms": mem_rms,
        }
        return torch.stack(logits_a, dim=1), torch.stack(logits_b, dim=1)


class StaticCompilerBaseline(nn.Module):
    def __init__(
        self,
        modulus: int,
        max_program_len: int,
        dim: int = 128,
        heads: int = 4,
        compiler_layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.max_program_len = int(max_program_len)
        self.value_embed = nn.Embedding(modulus, dim)
        self.register_embed = nn.Embedding(2, dim)
        self.op_embed = nn.Embedding(len(OP_NAMES), dim)
        self.arg_embed = nn.Embedding(modulus, dim)
        self.pos_embed = nn.Embedding(max_program_len, dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.compiler = nn.TransformerEncoder(enc_layer, num_layers=compiler_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, 2 * modulus),
        )

    def forward(
        self,
        ops: torch.Tensor,
        args: torch.Tensor,
        lengths: torch.Tensor,
        init_a: torch.Tensor,
        init_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq = ops.shape
        pos = torch.arange(seq, device=ops.device).unsqueeze(0).expand(bsz, seq)
        x = self.op_embed(ops) + self.arg_embed(args) + self.pos_embed(pos)
        h = self.compiler(x)
        mask = (torch.arange(seq, device=ops.device).unsqueeze(0) < lengths.unsqueeze(1)).float().unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        a = self.value_embed(init_a) + self.register_embed(torch.zeros_like(init_a))
        b = self.value_embed(init_b) + self.register_embed(torch.ones_like(init_b))
        logits = self.head(torch.cat([pooled, a, b], dim=-1))
        return logits[:, : self.modulus], logits[:, self.modulus :]


class CategoricalExecutor(nn.Module):
    """A structured latent executor over categorical register distributions.

    The hidden workspace is not text or a DSL: it is a pair of categorical
    distributions over register values. Each recurrent step selects a learned
    instruction-conditioned transition matrix and applies it to the relevant
    register distribution. This deliberately removes representation discovery
    from the first positive test and focuses on whether recurrent latent
    execution gives the expected K-scaling.
    """

    def __init__(self, modulus: int, max_program_len: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.modulus = int(modulus)
        self.max_program_len = int(max_program_len)
        self.temperature = float(temperature)
        # [op, arg, current_value, next_value]. Only ops 0-3 are used in the
        # const family, but keeping all op slots makes checkpoints simpler.
        self.transition_logits = nn.Parameter(torch.randn(len(OP_NAMES), modulus, modulus, modulus) * 0.01)
        self.last_aux: Dict[str, float] = {}

    def forward(
        self,
        ops: torch.Tensor,
        args: torch.Tensor,
        lengths: torch.Tensor,
        init_a: torch.Tensor,
        init_b: torch.Tensor,
        k_max: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = ops.shape[0]
        a = F.one_hot(init_a, num_classes=self.modulus).float()
        b = F.one_hot(init_b, num_classes=self.modulus).float()
        logits_a = [safe_log_probs(a)]
        logits_b = [safe_log_probs(b)]
        entropy_accum = 0.0

        for t in range(k_max):
            active = (lengths > t).float().view(bsz, 1)
            op = ops[:, t] if t < ops.shape[1] else torch.zeros(bsz, dtype=torch.long, device=ops.device)
            arg = args[:, t] if t < args.shape[1] else torch.zeros(bsz, dtype=torch.long, device=args.device)
            trans = F.softmax(self.transition_logits[op, arg] / self.temperature, dim=-1)
            next_a = torch.bmm(a.unsqueeze(1), trans).squeeze(1)
            next_b = torch.bmm(b.unsqueeze(1), trans).squeeze(1)

            update_a = ((op == 0) | (op == 1)).float().view(bsz, 1)
            update_b = ((op == 2) | (op == 3)).float().view(bsz, 1)
            a_new = update_a * next_a + (1.0 - update_a) * a
            b_new = update_b * next_b + (1.0 - update_b) * b
            a = active * a_new + (1.0 - active) * a
            b = active * b_new + (1.0 - active) * b
            logits_a.append(safe_log_probs(a))
            logits_b.append(safe_log_probs(b))
            with torch.no_grad():
                ent = -(trans.clamp_min(1e-8) * trans.clamp_min(1e-8).log()).sum(dim=-1).mean()
                entropy_accum += float(ent.detach().cpu())

        self.last_aux = {"transition_entropy": entropy_accum / max(1, k_max)}
        return torch.stack(logits_a, dim=1), torch.stack(logits_b, dim=1)


def safe_log_probs(probs: torch.Tensor) -> torch.Tensor:
    return probs.clamp_min(1e-8).log()


def trace_loss(logits_a: torch.Tensor, logits_b: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
    bsz, steps, mod = logits_a.shape
    loss_a = F.cross_entropy(logits_a.reshape(bsz * steps, mod), trace[:, :steps, 0].reshape(-1))
    loss_b = F.cross_entropy(logits_b.reshape(bsz * steps, mod), trace[:, :steps, 1].reshape(-1))
    return 0.5 * (loss_a + loss_b)


def final_loss_static(logits_a: torch.Tensor, logits_b: torch.Tensor, trace: torch.Tensor) -> torch.Tensor:
    target = trace[:, -1, :]
    return 0.5 * (
        F.cross_entropy(logits_a, target[:, 0])
        + F.cross_entropy(logits_b, target[:, 1])
    )


@torch.no_grad()
def evaluate_executor(
    model: LatentExecutor,
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
        correct_reg = {k: 0 for k in k_values}
        for _ in range(math.ceil(examples_per_length / batch_size)):
            bsz = min(batch_size, examples_per_length - totals[k_values[0]])
            if bsz <= 0:
                break
            batch = generator.batch(
                bsz,
                min_len=length,
                max_len=length,
                trace_k=max_k,
                fixed_len=length,
                device=device,
            )
            logits_a, logits_b = model(batch.ops, batch.args, batch.lengths, batch.init_a, batch.init_b, k_max=max_k)
            final = batch.trace[:, min(length, max_k), :]
            for k in k_values:
                pred_a = logits_a[:, k, :].argmax(dim=-1)
                pred_b = logits_b[:, k, :].argmax(dim=-1)
                pair = (pred_a == final[:, 0]) & (pred_b == final[:, 1])
                reg = torch.cat([(pred_a == final[:, 0]), (pred_b == final[:, 1])], dim=0)
                correct_pair[k] += int(pair.sum().item())
                correct_reg[k] += int(reg.sum().item())
                totals[k] += bsz
        for k in k_values:
            rows.append(
                {
                    "model": "executor",
                    "length": int(length),
                    "k": int(k),
                    "n": int(totals[k]),
                    "pair_accuracy": correct_pair[k] / max(1, totals[k]),
                    "register_accuracy": correct_reg[k] / max(1, 2 * totals[k]),
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
        correct_reg = 0
        for _ in range(math.ceil(examples_per_length / batch_size)):
            bsz = min(batch_size, examples_per_length - total)
            if bsz <= 0:
                break
            batch = generator.batch(
                bsz,
                min_len=length,
                max_len=length,
                trace_k=length,
                fixed_len=length,
                device=device,
            )
            logits_a, logits_b = model(batch.ops, batch.args, batch.lengths, batch.init_a, batch.init_b)
            final = batch.trace[:, length, :]
            pred_a = logits_a.argmax(dim=-1)
            pred_b = logits_b.argmax(dim=-1)
            pair = (pred_a == final[:, 0]) & (pred_b == final[:, 1])
            reg = torch.cat([(pred_a == final[:, 0]), (pred_b == final[:, 1])], dim=0)
            correct_pair += int(pair.sum().item())
            correct_reg += int(reg.sum().item())
            total += bsz
        rows.append(
            {
                "model": "static",
                "length": int(length),
                "k": -1,
                "n": int(total),
                "pair_accuracy": correct_pair / max(1, total),
                "register_accuracy": correct_reg / max(1, 2 * total),
            }
        )
    model.train()
    return rows


def write_metrics_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def train_executor(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    max_train_k = args.train_max_len
    max_eval_k = max(parse_int_list(args.eval_k))
    max_program_len = max(args.train_max_len, max(parse_int_list(args.eval_lengths)), max_eval_k)
    if args.mode == "categorical":
        model = CategoricalExecutor(
            modulus=args.modulus,
            max_program_len=max_program_len,
            temperature=args.transition_temperature,
        ).to(device)
    else:
        model = LatentExecutor(
            modulus=args.modulus,
            max_program_len=max_program_len,
            dim=args.dim,
            heads=args.heads,
            compiler_layers=args.compiler_layers,
            num_bases=args.num_bases,
            rank=args.rank,
            mem_dim=args.mem_dim,
            dropout=args.dropout,
            disable_memory=args.disable_memory,
            disable_dynamic=args.disable_dynamic,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_gen = RegisterProgramGenerator(args.modulus, seed=args.seed, op_family=args.op_family)
    eval_gen = RegisterProgramGenerator(args.modulus, seed=args.seed + 10_000, op_family=args.op_family)
    eval_lengths = parse_int_list(args.eval_lengths)
    eval_k = parse_int_list(args.eval_k)
    results: Dict[str, Any] = {"mode": args.mode, "args": vars(args), "train": [], "eval": []}
    t0 = time.time()
    model.train()
    for step in range(1, args.train_steps + 1):
        batch = train_gen.batch(
            args.batch_size,
            min_len=args.train_min_len,
            max_len=args.train_max_len,
            trace_k=max_train_k,
            device=device,
        )
        logits_a, logits_b = model(batch.ops, batch.args, batch.lengths, batch.init_a, batch.init_b, k_max=max_train_k)
        loss = trace_loss(logits_a, logits_b, batch.trace)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                final_pred_a = logits_a[:, -1, :].argmax(dim=-1)
                final_pred_b = logits_b[:, -1, :].argmax(dim=-1)
                final = batch.trace[:, -1, :]
                pair_acc = ((final_pred_a == final[:, 0]) & (final_pred_b == final[:, 1])).float().mean().item()
            aux = " ".join(f"{k}={v:.4g}" for k, v in model.last_aux.items())
            print(
                f"[train/{args.mode}] step={step:05d} loss={float(loss.detach().cpu()):.4f} "
                f"pair={pair_acc*100:.1f}% grad={float(grad):.3f} elapsed={(time.time()-t0)/60:.1f}m {aux}",
                flush=True,
            )
            results["train"].append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "pair_acc": pair_acc,
                    "grad_norm": float(grad),
                    **model.last_aux,
                }
            )

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.train_steps):
            rows = evaluate_executor(model, eval_gen, args, device, eval_lengths, eval_k, args.eval_examples)
            results["eval"].append({"step": step, "rows": rows})
            print_eval_summary(f"[eval/{args.mode} step={step}]", rows)
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            write_metrics_csv(out / f"metrics_step{step:05d}.csv", rows)
            save_checkpoint(out / f"checkpoint_step{step:05d}.pt", model, optimizer, args, results)
    return results


def train_static(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    max_program_len = max(args.train_max_len, max(parse_int_list(args.eval_lengths)))
    model = StaticCompilerBaseline(
        modulus=args.modulus,
        max_program_len=max_program_len,
        dim=args.dim,
        heads=args.heads,
        compiler_layers=max(1, args.compiler_layers),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_gen = RegisterProgramGenerator(args.modulus, seed=args.seed, op_family=args.op_family)
    eval_gen = RegisterProgramGenerator(args.modulus, seed=args.seed + 10_000, op_family=args.op_family)
    eval_lengths = parse_int_list(args.eval_lengths)
    results: Dict[str, Any] = {"mode": "static", "args": vars(args), "train": [], "eval": []}
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
        logits_a, logits_b = model(batch.ops, batch.args, batch.lengths, batch.init_a, batch.init_b)
        loss = final_loss_static(logits_a, logits_b, batch.trace)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                pred_a = logits_a.argmax(dim=-1)
                pred_b = logits_b.argmax(dim=-1)
                final = batch.trace[:, -1, :]
                pair_acc = ((pred_a == final[:, 0]) & (pred_b == final[:, 1])).float().mean().item()
            print(
                f"[train/static] step={step:05d} loss={float(loss.detach().cpu()):.4f} "
                f"pair={pair_acc*100:.1f}% grad={float(grad):.3f} elapsed={(time.time()-t0)/60:.1f}m",
                flush=True,
            )
            results["train"].append({"step": step, "loss": float(loss.detach().cpu()), "pair_acc": pair_acc, "grad_norm": float(grad)})

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.train_steps):
            rows = evaluate_static(model, eval_gen, args, device, eval_lengths, args.eval_examples)
            results["eval"].append({"step": step, "rows": rows})
            print_eval_summary(f"[eval/static step={step}]", rows)
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            write_metrics_csv(out / f"metrics_step{step:05d}.csv", rows)
            save_checkpoint(out / f"checkpoint_step{step:05d}.pt", model, optimizer, args, results)
    return results


def print_eval_summary(prefix: str, rows: Sequence[Dict[str, Any]]) -> None:
    by_len: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_len.setdefault(int(row["length"]), []).append(row)
    print(prefix, flush=True)
    for length in sorted(by_len):
        parts = []
        for row in sorted(by_len[length], key=lambda r: int(r["k"])):
            k = int(row["k"])
            label = "static" if k < 0 else f"K={k}"
            parts.append(f"{label}:{row['pair_accuracy']*100:.1f}%/{row['register_accuracy']*100:.1f}%")
        print(f"  L={length} " + " | ".join(parts), flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trace-supervised latent recurrent executor")
    p.add_argument("--mode", choices=["executor", "categorical", "static"], default="executor")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--modulus", type=int, default=97)
    p.add_argument("--op_family", choices=["full", "const"], default="full")
    p.add_argument("--train_min_len", type=int, default=1)
    p.add_argument("--train_max_len", type=int, default=8)
    p.add_argument("--eval_lengths", type=str, default="4,8,12,16,24")
    p.add_argument("--eval_k", type=str, default="0,1,2,4,8,12,16,24")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--compiler_layers", type=int, default=1)
    p.add_argument("--num_bases", type=int, default=16)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--mem_dim", type=int, default=48)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--disable_memory", action="store_true")
    p.add_argument("--disable_dynamic", action="store_true")
    p.add_argument("--transition_temperature", type=float, default=1.0)
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--eval_examples", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--output_dir", type=str, default="runs/executor")
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
    print(json.dumps(vars(args), indent=2), flush=True)
    print(f"[env] python={sys.version.split()[0]} torch={torch.__version__} device={device}", flush=True)
    if args.mode in {"executor", "categorical"}:
        results = train_executor(args, device)
    else:
        results = train_static(args, device)
    with open(Path(args.output_dir) / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
