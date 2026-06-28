#!/usr/bin/env python3
"""
latent_qwen_fastweight_experiment.py

Single-file experiment for a frozen Qwen 3.5 4B backbone plus an internal
latent recurrent fast-weight hyperadapter.

What this does
--------------
- Loads a frozen Qwen/Qwen3.5-4B model in 4-bit by default.
- Finds a late transformer layer and injects a trainable neural computation
  module by a forward hook.
- The module does NOT emit tools, code, or a DSL. It runs inside the PyTorch
  graph: workspace tokens + recurrent latent steps + dynamic low-rank program
  gates + differentiable fast-weight memory.
- Trains only the new module on generated, verifiable modular-arithmetic
  multiple-choice tasks.
- Evaluates by multiple-choice log-likelihood and reports accuracy as a
  function of the number of internal recurrent compute steps K.

Install notes
-------------
Qwen3.5 support may require very recent Transformers. A starting env:

  python -m venv .venv && source .venv/bin/activate
  pip install -U torch accelerate bitsandbytes sentencepiece safetensors einops
  pip install -U "transformers[torch] @ git+https://github.com/huggingface/transformers.git@main"

Example runs
------------
Faithful hook-in-LLM mode, conservative VRAM defaults:

  python latent_qwen_fastweight_experiment.py \
    --model_id Qwen/Qwen3.5-4B \
    --mode hook \
    --load_in_4bit \
    --batch_size 1 \
    --grad_accum 8 \
    --train_steps 300 \
    --hook_layer -4 \
    --train_k 1,2,4 \
    --eval_k 0,1,2,4,8

Cheaper diagnostic mode: freeze Qwen as encoder and train the latent module as
an answer classifier on top of Qwen hidden states. This is less faithful but
much more likely to run on tight hardware:

  python latent_qwen_fastweight_experiment.py \
    --model_id Qwen/Qwen3.5-4B \
    --mode head \
    --load_in_4bit \
    --batch_size 4 \
    --grad_accum 4 \
    --train_steps 300

For a fast plumbing check, use a tiny HF model:

  python latent_qwen_fastweight_experiment.py \
    --model_id hf-internal-testing/tiny-random-LlamaForCausalLM \
    --mode hook --no_load_in_4bit --train_steps 3 --eval_batches 2

Caveats
-------
This is an experiment scaffold, not a claim of solved intelligence. The main
thing to look for is whether validation accuracy improves with K and whether
that improvement survives harder/longer held-out operation chains.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    try:
        # Available in recent Transformers builds used by Qwen3.5.
        from transformers import AutoModelForMultimodalLM  # type: ignore
    except Exception:  # pragma: no cover - depends on installed transformers
        AutoModelForMultimodalLM = None  # type: ignore
    try:
        from transformers import BitsAndBytesConfig
    except Exception:  # pragma: no cover
        BitsAndBytesConfig = None  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires transformers. Install e.g.\n"
        "  pip install -U torch accelerate bitsandbytes sentencepiece safetensors\n"
        "  pip install -U 'transformers[torch] @ git+https://github.com/huggingface/transformers.git@main'\n"
        f"Original import error: {exc}"
    )


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# -----------------------------------------------------------------------------
# Synthetic verifiable task generator
# -----------------------------------------------------------------------------

@dataclass
class MCExample:
    prompt: str
    label: str              # e.g. " C"; intentionally includes a leading space
    correct_idx: int
    choices: List[int]
    answer_value: int
    family: str
    steps: int


class ModularArithmeticGenerator:
    """Generates exact multiple-choice modular arithmetic tasks.

    The tasks are deliberately synthetic so reward/eval is exact. Held-out eval
    uses a separate RNG and can use longer operation chains than training.
    """

    def __init__(
        self,
        modulus: int = 97,
        num_choices: int = 5,
        min_steps: int = 3,
        max_steps: int = 8,
        seed: int = 0,
    ) -> None:
        if num_choices > len(LETTERS):
            raise ValueError(f"num_choices must be <= {len(LETTERS)}")
        self.modulus = int(modulus)
        self.num_choices = int(num_choices)
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps)
        self.rng = random.Random(seed)

    def sample(self, harder: bool = False) -> MCExample:
        family = self.rng.choice(["chain", "tworeg", "reverse_chain"])
        extra = 3 if harder else 0
        steps = self.rng.randint(self.min_steps + extra, self.max_steps + extra)
        if family == "chain":
            question, ans = self._chain_task(steps)
        elif family == "tworeg":
            question, ans = self._two_register_task(steps)
        else:
            question, ans = self._reverse_chain_task(steps)

        choices, correct_idx = self._make_choices(ans)
        lines = [
            f"Solve this exactly. All arithmetic is modulo {self.modulus}.",
            question,
            "Choices:",
        ]
        for i, value in enumerate(choices):
            lines.append(f"{LETTERS[i]}) {value}")
        lines.extend([
            "Answer with only the letter of the correct choice.",
            "Answer:",
        ])
        prompt = "\n".join(lines)
        return MCExample(
            prompt=prompt,
            label=" " + LETTERS[correct_idx],
            correct_idx=correct_idx,
            choices=choices,
            answer_value=ans,
            family=family,
            steps=steps,
        )

    def batch(self, batch_size: int, harder: bool = False) -> List[MCExample]:
        return [self.sample(harder=harder) for _ in range(batch_size)]

    def _make_choices(self, ans: int) -> Tuple[List[int], int]:
        m = self.modulus
        values = {ans}
        # Distractors are close arithmetic confusions plus random values.
        offsets = [1, -1, 2, -2, 3, -3, 5, -5, 7, -7, 10, -10]
        self.rng.shuffle(offsets)
        for off in offsets:
            if len(values) >= self.num_choices:
                break
            values.add((ans + off) % m)
        while len(values) < self.num_choices:
            values.add(self.rng.randrange(m))
        choices = list(values)
        self.rng.shuffle(choices)
        return choices, choices.index(ans)

    def _op(self) -> Tuple[str, int]:
        op = self.rng.choice(["add", "subtract", "multiply by", "add", "subtract"])
        if op == "multiply by":
            # Avoid boring 0/1 multipliers.
            value = self.rng.randint(2, min(15, self.modulus - 1))
        else:
            value = self.rng.randint(1, min(40, self.modulus - 1))
        return op, value

    def _apply(self, x: int, op: str, value: int) -> int:
        if op == "add":
            return (x + value) % self.modulus
        if op == "subtract":
            return (x - value) % self.modulus
        if op == "multiply by":
            return (x * value) % self.modulus
        raise ValueError(op)

    def _chain_task(self, steps: int) -> Tuple[str, int]:
        x = self.rng.randrange(self.modulus)
        cur = x
        ops = []
        for _ in range(steps):
            op, value = self._op()
            cur = self._apply(cur, op, value)
            ops.append(f"{op} {value}")
        question = (
            f"Start with x = {x}. Apply these operations in order: "
            + "; ".join(ops)
            + ". What is the final value of x?"
        )
        return question, cur

    def _reverse_chain_task(self, steps: int) -> Tuple[str, int]:
        # Same arithmetic, different wording/order pressure.
        x = self.rng.randrange(self.modulus)
        cur = x
        ops = []
        for i in range(steps):
            op, value = self._op()
            cur = self._apply(cur, op, value)
            ops.append((i + 1, op, value))
        op_lines = " ".join(f"Step {i}: {op} {value}." for i, op, value in ops)
        question = f"Initial value x is {x}. {op_lines} Return x after the last step."
        return question, cur

    def _two_register_task(self, steps: int) -> Tuple[str, int]:
        a = self.rng.randrange(self.modulus)
        b = self.rng.randrange(self.modulus)
        cur_a, cur_b = a, b
        ops = []
        for _ in range(steps):
            kind = self.rng.choice(["A=A+B", "B=A+B", "A=A-B", "B=B-A", "A=kA+c", "B=kB+c"])
            if kind == "A=A+B":
                cur_a = (cur_a + cur_b) % self.modulus
                ops.append("set A to A plus B")
            elif kind == "B=A+B":
                cur_b = (cur_a + cur_b) % self.modulus
                ops.append("set B to A plus B")
            elif kind == "A=A-B":
                cur_a = (cur_a - cur_b) % self.modulus
                ops.append("set A to A minus B")
            elif kind == "B=B-A":
                cur_b = (cur_b - cur_a) % self.modulus
                ops.append("set B to B minus A")
            elif kind == "A=kA+c":
                k = self.rng.randint(2, min(9, self.modulus - 1))
                c = self.rng.randint(1, min(20, self.modulus - 1))
                cur_a = (k * cur_a + c) % self.modulus
                ops.append(f"set A to {k} times A plus {c}")
            else:
                k = self.rng.randint(2, min(9, self.modulus - 1))
                c = self.rng.randint(1, min(20, self.modulus - 1))
                cur_b = (k * cur_b + c) % self.modulus
                ops.append(f"set B to {k} times B plus {c}")
        ask_a = self.rng.random() < 0.5
        target = cur_a if ask_a else cur_b
        reg = "A" if ask_a else "B"
        question = (
            f"There are two registers. Initially A = {a} and B = {b}. "
            "Apply these updates in order: "
            + "; ".join(ops)
            + f". What is the final value of register {reg}?"
        )
        return question, target


# -----------------------------------------------------------------------------
# Tokenization / collation
# -----------------------------------------------------------------------------

@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_mask: torch.Tensor
    correct_idx: torch.Tensor
    answer_value: torch.Tensor
    examples: List[MCExample]


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        # Do not add a new token here: the frozen model's embedding matrix would
        # not be resized, and padded positions can still be looked up before the
        # attention mask removes them. Reuse an existing token instead.
        eos = getattr(tokenizer, "eos_token", None)
        unk = getattr(tokenizer, "unk_token", None)
        if eos is not None:
            tokenizer.pad_token = eos
        elif unk is not None:
            tokenizer.pad_token = unk
        else:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(0)


def _tokenize_no_special(tokenizer: Any, text: str) -> List[int]:
    out = tokenizer(text, add_special_tokens=False)
    ids = out["input_ids"] if isinstance(out, dict) else out.input_ids
    return list(ids)


def collate_train_examples(
    examples: List[MCExample],
    tokenizer: Any,
    max_length: int,
    device: torch.device,
) -> EncodedBatch:
    pad_id = int(tokenizer.pad_token_id)
    encoded: List[Tuple[List[int], List[int], List[int], int]] = []
    for ex in examples:
        prompt_ids = _tokenize_no_special(tokenizer, ex.prompt)
        label_ids = _tokenize_no_special(tokenizer, ex.label)
        if not label_ids:
            raise RuntimeError(f"Tokenizer produced no ids for label {ex.label!r}")
        # Keep the answer; truncate prompt from the left if needed.
        if len(prompt_ids) + len(label_ids) > max_length:
            keep_prompt = max(1, max_length - len(label_ids))
            prompt_ids = prompt_ids[-keep_prompt:]
        ids = prompt_ids + label_ids
        labels = [-100] * len(prompt_ids) + label_ids
        prompt_mask = [1] * len(prompt_ids) + [0] * len(label_ids)
        encoded.append((ids, labels, prompt_mask, ex.correct_idx))

    max_len = max(len(x[0]) for x in encoded)
    input_ids, attention_mask, labels, prompt_mask, correct_idx = [], [], [], [], []
    answer_value = []
    for ids, labs, pmask, cidx in encoded:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(labs + [-100] * pad)
        prompt_mask.append(pmask + [0] * pad)
        correct_idx.append(cidx)
    answer_value = [ex.answer_value for ex in examples]

    return EncodedBatch(
        input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long, device=device),
        labels=torch.tensor(labels, dtype=torch.long, device=device),
        prompt_mask=torch.tensor(prompt_mask, dtype=torch.bool, device=device),
        correct_idx=torch.tensor(correct_idx, dtype=torch.long, device=device),
        answer_value=torch.tensor(answer_value, dtype=torch.long, device=device),
        examples=examples,
    )


def make_candidate_examples(examples: List[MCExample]) -> Tuple[List[MCExample], List[int]]:
    """Expand each MC example into one prompt+candidate-letter example per choice."""
    expanded: List[MCExample] = []
    owners: List[int] = []
    for owner, ex in enumerate(examples):
        for i in range(len(ex.choices)):
            expanded.append(dataclasses.replace(ex, label=" " + LETTERS[i], correct_idx=i))
            owners.append(owner)
    return expanded, owners


# -----------------------------------------------------------------------------
# Latent recurrent fast-weight hyperadapter
# -----------------------------------------------------------------------------

class DynamicLowRank(nn.Module):
    """Activation-programmed low-rank transform bank.

    Given token/workspace states x and a program vector, compute gates over a
    bank of low-rank transforms. This is a neural analogue of selecting latent
    opcodes, but no discrete DSL is emitted.
    """

    def __init__(self, dim: int, num_bases: int = 12, rank: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.num_bases = num_bases
        self.rank = rank
        self.A = nn.Parameter(torch.randn(num_bases, dim, rank) / math.sqrt(dim))
        self.B = nn.Parameter(torch.randn(num_bases, rank, dim) * 0.02)

    def forward(self, x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], gates: [B, N]
        xa = torch.einsum("btd,ndr->btnr", x, self.A)
        xab = torch.einsum("btnr,nrd->btnd", xa, self.B)
        return torch.einsum("bn,btnd->btd", gates, xab)


class LatentFastWeightCore(nn.Module):
    """One recurrent latent compute block with fast-weight memory."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        num_bases: int = 12,
        rank: int = 16,
        mem_dim: int = 128,
        dropout: float = 0.05,
        disable_fast_memory: bool = False,
        disable_dynamic_lowrank: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mem_dim = mem_dim
        self.disable_fast_memory = disable_fast_memory
        self.disable_dynamic_lowrank = disable_dynamic_lowrank
        self.cross_norm = nn.LayerNorm(dim)
        self.self_norm = nn.LayerNorm(dim)
        self.mem_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.prog_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mem_q = nn.Linear(dim, mem_dim, bias=False)
        self.mem_k = nn.Linear(dim, mem_dim, bias=False)
        self.mem_v = nn.Linear(dim, mem_dim, bias=False)
        self.mem_out = nn.Linear(mem_dim, dim, bias=False)
        self.mem_update = nn.Linear(dim, 1)
        self.mem_decay_logit = nn.Parameter(torch.tensor(2.2))  # sigmoid ~= 0.90
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, num_bases),
        )
        self.base_ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )
        self.dynamic = DynamicLowRank(dim, num_bases=num_bases, rank=rank)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        ws: torch.Tensor,
        ctx: torch.Tensor,
        ctx_prompt_mask: torch.Tensor,
        mem: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ws: [B, W, D], ctx: [B, S, D], mask: [B, S] True for allowed prompt tokens
        key_padding_mask = ~ctx_prompt_mask.bool()

        x = self.cross_norm(ws)
        cross, _ = self.cross_attn(x, ctx, ctx, key_padding_mask=key_padding_mask, need_weights=False)
        ws = ws + self.dropout(cross)

        x = self.self_norm(ws)
        self_out, _ = self.self_attn(x, x, x, need_weights=False)
        ws = ws + self.dropout(self_out)

        # Differentiable fast-weight memory. M is temporary per forward pass.
        if not self.disable_fast_memory:
            x = self.mem_norm(ws)
            q = torch.tanh(self.mem_q(x))
            k = torch.tanh(self.mem_k(x))
            v = torch.tanh(self.mem_v(x))
            read = torch.matmul(q, mem) / math.sqrt(max(1, self.mem_dim))
            ws = ws + self.dropout(self.mem_out(read))
            write = torch.matmul(k.transpose(1, 2), v) / math.sqrt(max(1, ws.shape[1]))
            update = torch.sigmoid(self.mem_update(ws.mean(dim=1))).view(ws.shape[0], 1, 1)
            decay = torch.sigmoid(self.mem_decay_logit)
            mem = decay * mem + update * write

        program = self.prog_norm(ws.mean(dim=1))
        gates = F.softmax(self.gate_net(program), dim=-1)
        x = self.ff_norm(ws)
        dyn = torch.zeros_like(x) if self.disable_dynamic_lowrank else self.dynamic(x, gates)
        ws = ws + self.dropout(self.base_ff(x) + dyn)
        return ws, mem, gates


class RuntimeDeltaAdapter(nn.Module):
    """Internal recurrent computation module inserted into a Qwen layer.

    The adapter reads prompt hidden states only, updates latent workspace tokens
    for K recurrent steps, then writes a small residual delta back into every
    live token position. The future answer token is masked out of the runtime's
    context during teacher forcing, preventing answer leakage.
    """

    def __init__(
        self,
        hidden_dim: int,
        runtime_dim: int = 256,
        workspace_tokens: int = 8,
        heads: int = 4,
        num_bases: int = 12,
        rank: int = 16,
        mem_dim: int = 128,
        dropout: float = 0.05,
        init_scale: float = 0.05,
        disable_fast_memory: bool = False,
        disable_dynamic_lowrank: bool = False,
        value_modulus: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.runtime_dim = runtime_dim
        self.workspace_tokens = workspace_tokens
        self.value_modulus = int(value_modulus)
        self.ctx_proj = nn.Linear(hidden_dim, runtime_dim, bias=False)
        self.summary_to_ws = nn.Linear(runtime_dim, workspace_tokens * runtime_dim)
        self.ws_init = nn.Parameter(torch.randn(workspace_tokens, runtime_dim) * 0.02)
        self.core = LatentFastWeightCore(
            runtime_dim,
            heads=heads,
            num_bases=num_bases,
            rank=rank,
            mem_dim=mem_dim,
            dropout=dropout,
            disable_fast_memory=disable_fast_memory,
            disable_dynamic_lowrank=disable_dynamic_lowrank,
        )
        self.token_proj = nn.Linear(hidden_dim, runtime_dim, bias=False)
        self.read_norm = nn.LayerNorm(runtime_dim)
        self.read_attn = nn.MultiheadAttention(runtime_dim, heads, dropout=dropout, batch_first=True)
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(runtime_dim),
            nn.Linear(runtime_dim, 4 * runtime_dim),
            nn.SiLU(),
            nn.Linear(4 * runtime_dim, runtime_dim),
        )
        self.delta_out = nn.Linear(runtime_dim, hidden_dim, bias=False)
        nn.init.normal_(self.delta_out.weight, std=1e-4)
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))
        self.last_aux: Dict[str, float] = {}
        self.last_value_logits: Optional[torch.Tensor] = None
        self.value_head = (
            nn.Sequential(
                nn.LayerNorm(runtime_dim),
                nn.Linear(runtime_dim, 4 * runtime_dim),
                nn.SiLU(),
                nn.Linear(4 * runtime_dim, self.value_modulus),
            )
            if self.value_modulus > 0
            else None
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_mask: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        # Use fp32 inside the new module for stability, then cast delta back.
        orig_dtype = hidden.dtype
        h = hidden.float()
        attention_mask = attention_mask.bool().to(h.device)
        prompt_mask = prompt_mask.bool().to(h.device) & attention_mask
        bsz, seq_len, _ = h.shape

        ctx = self.ctx_proj(h)
        prompt_f = prompt_mask.float().unsqueeze(-1)
        denom = prompt_f.sum(dim=1).clamp_min(1.0)
        summary = (ctx * prompt_f).sum(dim=1) / denom
        ws = self.ws_init.unsqueeze(0).expand(bsz, -1, -1) + self.summary_to_ws(summary).view(
            bsz, self.workspace_tokens, self.runtime_dim
        )
        mem = torch.zeros(bsz, self.core.mem_dim, self.core.mem_dim, dtype=ws.dtype, device=ws.device)

        gate_entropy_accum = 0.0
        actual_steps = int(max(0, steps))
        for _ in range(actual_steps):
            ws, mem, gates = self.core(ws, ctx, prompt_mask, mem)
            ent = -(gates.clamp_min(1e-8).log() * gates).sum(dim=-1).mean()
            gate_entropy_accum = gate_entropy_accum + ent

        self.last_value_logits = self.value_head(ws.mean(dim=1)) if self.value_head is not None else None

        token_q = self.read_norm(self.token_proj(h))
        read, _ = self.read_attn(token_q, ws, ws, need_weights=False)
        delta_runtime = self.delta_mlp(read)
        delta = self.delta_out(delta_runtime)
        scale = self.log_scale.exp().clamp(max=1.0)
        delta = delta * scale * attention_mask.float().unsqueeze(-1)

        with torch.no_grad():
            if actual_steps > 0 and isinstance(gate_entropy_accum, torch.Tensor):
                gate_entropy = float((gate_entropy_accum / actual_steps).detach().cpu())
            else:
                gate_entropy = 0.0
            self.last_aux = {
                "delta_rms": float(delta.detach().float().pow(2).mean().sqrt().cpu()),
                "scale": float(scale.detach().cpu()),
                "gate_entropy": gate_entropy,
                "mem_rms": float(mem.detach().float().pow(2).mean().sqrt().cpu()),
            }
        return delta.to(dtype=orig_dtype)


class HeadClassifier(nn.Module):
    """Cheap diagnostic mode: Qwen hidden states -> latent runtime -> option logits."""

    def __init__(
        self,
        hidden_dim: int,
        num_choices: int,
        runtime_dim: int = 256,
        workspace_tokens: int = 8,
        heads: int = 4,
        num_bases: int = 12,
        rank: int = 16,
        mem_dim: int = 128,
        dropout: float = 0.05,
        disable_fast_memory: bool = False,
        disable_dynamic_lowrank: bool = False,
    ) -> None:
        super().__init__()
        self.adapter = RuntimeDeltaAdapter(
            hidden_dim=hidden_dim,
            runtime_dim=runtime_dim,
            workspace_tokens=workspace_tokens,
            heads=heads,
            num_bases=num_bases,
            rank=rank,
            mem_dim=mem_dim,
            dropout=dropout,
            init_scale=0.5,
            disable_fast_memory=disable_fast_memory,
            disable_dynamic_lowrank=disable_dynamic_lowrank,
        )
        self.pool_proj = nn.Linear(hidden_dim, runtime_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(runtime_dim),
            nn.Linear(runtime_dim, 4 * runtime_dim),
            nn.SiLU(),
            nn.Linear(4 * runtime_dim, num_choices),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_mask: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        # Reuse RuntimeDeltaAdapter as an invisible recurrent compute module, but
        # read out with a classifier instead of modifying LM logits.
        delta = self.adapter(hidden, attention_mask, prompt_mask, steps=steps).float()
        h = hidden.float() + delta
        mask = prompt_mask.float().unsqueeze(-1).to(h.device)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(self.pool_proj(pooled))


# -----------------------------------------------------------------------------
# Model loading and hook management
# -----------------------------------------------------------------------------

def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return vals


def dtype_from_string(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unknown dtype {name}")


def get_nested_attr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def maybe_get_nested_attr(obj: Any, path: str) -> Optional[Any]:
    try:
        return get_nested_attr(obj, path)
    except Exception:
        return None


def find_transformer_layers(model: nn.Module) -> Tuple[str, nn.ModuleList]:
    """Find the likely decoder layer ModuleList across model families."""
    preferred_paths = [
        "model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "model.language_model.layers",
        "base_model.model.layers",
        "base_model.model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ]
    for path in preferred_paths:
        module = maybe_get_nested_attr(model, path)
        if isinstance(module, nn.ModuleList) and len(module) >= 2:
            return path, module

    candidates: List[Tuple[int, str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 4:
            child_names = " ".join(module[i].__class__.__name__.lower() for i in range(min(3, len(module))))
            score = len(module)
            if "layer" in name.lower() or name.endswith("h"):
                score += 100
            if any(tok in child_names for tok in ["decoder", "layer", "qwen", "llama", "block"]):
                score += 50
            candidates.append((score, name, module))
    if not candidates:
        raise RuntimeError(
            "Could not locate a transformer layer ModuleList. Try --mode head, or add the model's layer path to find_transformer_layers()."
        )
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def infer_hidden_dim(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    for obj in [cfg, getattr(cfg, "text_config", None), getattr(cfg, "language_config", None), getattr(cfg, "llm_config", None)]:
        if obj is None:
            continue
        for name in ["hidden_size", "n_embd", "d_model"]:
            val = getattr(obj, name, None)
            if val is not None:
                return int(val)
    try:
        emb = model.get_input_embeddings()
        if hasattr(emb, "embedding_dim"):
            return int(emb.embedding_dim)
        if hasattr(emb, "weight"):
            return int(emb.weight.shape[1])
    except Exception:
        pass
    raise RuntimeError("Could not infer hidden dimension")


def get_input_device(model: nn.Module) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        return next(emb.parameters()).device
    except Exception:
        for p in model.parameters():
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_module_device(module: nn.Module) -> torch.device:
    for p in module.parameters(recurse=True):
        return p.device
    for b in module.buffers(recurse=True):
        return b.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HookController:
    def __init__(self, layer: nn.Module, adapter: RuntimeDeltaAdapter) -> None:
        self.layer = layer
        self.adapter = adapter
        self.attention_mask: Optional[torch.Tensor] = None
        self.prompt_mask: Optional[torch.Tensor] = None
        self.steps: int = 0
        self.handle = layer.register_forward_hook(self._hook)

    def close(self) -> None:
        self.handle.remove()

    def set_context(self, attention_mask: torch.Tensor, prompt_mask: torch.Tensor, steps: int) -> None:
        self.attention_mask = attention_mask
        self.prompt_mask = prompt_mask
        self.steps = int(steps)

    def clear_context(self) -> None:
        self.attention_mask = None
        self.prompt_mask = None
        self.steps = 0

    def _hook(self, module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
        if self.attention_mask is None or self.prompt_mask is None:
            return output
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return output

        transposed = False
        if hidden.shape[0] != self.attention_mask.shape[0] and hidden.shape[1] == self.attention_mask.shape[0]:
            hidden_bsh = hidden.transpose(0, 1)
            transposed = True
        else:
            hidden_bsh = hidden

        amask = self.attention_mask.to(hidden_bsh.device)
        pmask = self.prompt_mask.to(hidden_bsh.device)
        if amask.shape[1] != hidden_bsh.shape[1]:
            # Some models may internally alter sequence length. In that case, do not inject.
            return output
        delta = self.adapter(hidden_bsh, amask, pmask, steps=self.steps)
        hidden_mod = hidden_bsh + delta
        if transposed:
            hidden_mod = hidden_mod.transpose(0, 1)
        if rest is None:
            return hidden_mod
        return (hidden_mod,) + rest


def load_tokenizer_and_model(args: argparse.Namespace) -> Tuple[Any, nn.Module, str]:
    model_id = args.model_id
    tokenizer = None
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
    except Exception as exc:
        if args.verbose:
            print(f"[load] AutoProcessor failed, trying AutoTokenizer: {exc}")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"

    torch_dtype = dtype_from_string(args.torch_dtype)
    quantization_config = None
    if args.load_in_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes/transformers BitsAndBytesConfig unavailable; use --no_load_in_4bit")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype if torch_dtype != torch.float32 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    common_kwargs: Dict[str, Any] = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        common_kwargs["device_map"] = args.device_map
    if quantization_config is not None:
        common_kwargs["quantization_config"] = quantization_config

    # Qwen3.5's model card recommends AutoModelForMultimodalLM. Older/text-only
    # Qwen models generally use AutoModelForCausalLM. Try in the likely order.
    classes: List[Tuple[str, Any]] = []
    if args.model_class == "causal":
        classes = [("AutoModelForCausalLM", AutoModelForCausalLM)]
    elif args.model_class == "multimodal":
        if AutoModelForMultimodalLM is None:
            raise RuntimeError("AutoModelForMultimodalLM not available; install Transformers from main")
        classes = [("AutoModelForMultimodalLM", AutoModelForMultimodalLM)]
    else:
        if "qwen3.5" in model_id.lower() or "qwen3_5" in model_id.lower():
            if AutoModelForMultimodalLM is not None:
                classes.append(("AutoModelForMultimodalLM", AutoModelForMultimodalLM))
            classes.append(("AutoModelForCausalLM", AutoModelForCausalLM))
        else:
            classes.append(("AutoModelForCausalLM", AutoModelForCausalLM))
            if AutoModelForMultimodalLM is not None:
                classes.append(("AutoModelForMultimodalLM", AutoModelForMultimodalLM))

    last_exc: Optional[BaseException] = None
    for name, cls in classes:
        try:
            print(f"[load] Loading {model_id} with {name}...")
            model = cls.from_pretrained(model_id, **common_kwargs)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            return tokenizer, model, name
        except Exception as exc:
            last_exc = exc
            print(f"[load] {name} failed: {type(exc).__name__}: {exc}")

    if args.text_model_fallback and args.text_model_fallback != model_id:
        print(f"[load] Falling back to {args.text_model_fallback}")
        old_model_id = args.model_id
        args.model_id = args.text_model_fallback
        try:
            return load_tokenizer_and_model(args)
        finally:
            args.model_id = old_model_id

    raise RuntimeError(
        "Could not load model. For Qwen3.5, try installing Transformers from main, or use --model_class multimodal. "
        f"Last error: {last_exc}"
    )


def extract_hidden_states(outputs: Any) -> Optional[torch.Tensor]:
    # Try common ModelOutput fields recursively.
    if outputs is None:
        return None
    for attr in ["hidden_states", "decoder_hidden_states"]:
        val = getattr(outputs, attr, None)
        if val is not None:
            if isinstance(val, (tuple, list)):
                for item in reversed(val):
                    if torch.is_tensor(item) and item.ndim == 3:
                        return item
            if torch.is_tensor(val) and val.ndim == 3:
                return val
    if isinstance(outputs, dict):
        for key in ["hidden_states", "decoder_hidden_states", "language_model_outputs", "text_model_output"]:
            if key in outputs:
                got = extract_hidden_states(outputs[key])
                if got is not None:
                    return got
    for attr in ["language_model_outputs", "text_model_output", "model_output"]:
        got = extract_hidden_states(getattr(outputs, attr, None))
        if got is not None:
            return got
    val = getattr(outputs, "last_hidden_state", None)
    if torch.is_tensor(val) and val.ndim == 3:
        return val
    return None


# -----------------------------------------------------------------------------
# Losses, forward helpers, eval
# -----------------------------------------------------------------------------

def causal_lm_token_nll(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean token loss and per-sequence average NLL over non--100 labels."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    bsz, t, vocab = shift_logits.shape
    flat_loss = F.cross_entropy(
        shift_logits.view(bsz * t, vocab).float(),
        shift_labels.reshape(bsz * t),
        ignore_index=-100,
        reduction="none",
    ).view(bsz, t)
    valid = shift_labels.ne(-100)
    denom = valid.sum(dim=1).clamp_min(1)
    seq_nll = (flat_loss * valid.float()).sum(dim=1) / denom.float()
    mean_loss = (flat_loss * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
    return mean_loss, seq_nll


def forward_hook_loss(
    model: nn.Module,
    hook: HookController,
    batch: EncodedBatch,
    steps: int,
    aux_value_loss_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    hook.set_context(batch.attention_mask, batch.prompt_mask, steps)
    try:
        outputs = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError("Model output has no logits; hook mode requires a CausalLM-style logits field")
        loss, _ = causal_lm_token_nll(logits, batch.labels)
        aux_value_loss = None
        if aux_value_loss_weight > 0.0:
            value_logits = hook.adapter.last_value_logits
            if value_logits is None:
                raise RuntimeError("--aux_value_loss requires adapter value head; construct adapter with value_modulus > 0")
            aux_value_loss = F.cross_entropy(value_logits.float(), batch.answer_value.to(value_logits.device))
            loss = loss + aux_value_loss_weight * aux_value_loss
    finally:
        hook.clear_context()
    aux = dict(hook.adapter.last_aux)
    if aux_value_loss is not None:
        aux["aux_value_loss"] = float(aux_value_loss.detach().cpu())
    return loss, aux


@torch.no_grad()
def score_candidates_hook(
    model: nn.Module,
    hook: HookController,
    examples: List[MCExample],
    tokenizer: Any,
    max_length: int,
    input_device: torch.device,
    candidate_batch_size: int,
    steps: int,
) -> torch.Tensor:
    expanded, owners = make_candidate_examples(examples)
    scores = torch.empty(len(expanded), dtype=torch.float32)
    for start in range(0, len(expanded), candidate_batch_size):
        chunk = expanded[start : start + candidate_batch_size]
        batch = collate_train_examples(chunk, tokenizer, max_length=max_length, device=input_device)
        hook.set_context(batch.attention_mask, batch.prompt_mask, steps)
        try:
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise RuntimeError("Model output has no logits")
            _, seq_nll = causal_lm_token_nll(logits, batch.labels)
        finally:
            hook.clear_context()
        scores[start : start + len(chunk)] = seq_nll.detach().cpu()
    # Shape [num_examples, num_choices], lower is better.
    num_choices = len(examples[0].choices)
    return scores.view(len(examples), num_choices)


@torch.no_grad()
def score_candidates_base(
    model: nn.Module,
    examples: List[MCExample],
    tokenizer: Any,
    max_length: int,
    input_device: torch.device,
    candidate_batch_size: int,
) -> torch.Tensor:
    expanded, owners = make_candidate_examples(examples)
    scores = torch.empty(len(expanded), dtype=torch.float32)
    for start in range(0, len(expanded), candidate_batch_size):
        chunk = expanded[start : start + candidate_batch_size]
        batch = collate_train_examples(chunk, tokenizer, max_length=max_length, device=input_device)
        outputs = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError("Model output has no logits")
        _, seq_nll = causal_lm_token_nll(logits, batch.labels)
        scores[start : start + len(chunk)] = seq_nll.detach().cpu()
    num_choices = len(examples[0].choices)
    return scores.view(len(examples), num_choices)


@torch.no_grad()
def encode_hidden_for_head(
    model: nn.Module,
    batch: EncodedBatch,
) -> torch.Tensor:
    outputs = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = extract_hidden_states(outputs)
    if hidden is None:
        raise RuntimeError("Could not extract hidden states for --mode head")
    return hidden.detach()


def evaluate_base_lm(
    model: nn.Module,
    generator: ModularArithmeticGenerator,
    tokenizer: Any,
    input_device: torch.device,
    args: argparse.Namespace,
    harder: bool,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for _ in range(args.eval_batches):
        examples = generator.batch(args.eval_batch_size, harder=harder)
        total += len(examples)
        nll = score_candidates_base(
            model=model,
            examples=examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            input_device=input_device,
            candidate_batch_size=args.candidate_batch_size,
        )
        pred = nll.argmin(dim=1)
        gold = torch.tensor([ex.correct_idx for ex in examples], dtype=torch.long)
        correct += int((pred == gold).sum().item())
    return correct / max(1, total)


def evaluate_hook(
    model: nn.Module,
    hook: HookController,
    generator: ModularArithmeticGenerator,
    tokenizer: Any,
    input_device: torch.device,
    args: argparse.Namespace,
    k_values: Sequence[int],
    harder: bool,
) -> Dict[int, float]:
    model.eval()
    hook.adapter.eval()
    correct_by_k = {int(k): 0 for k in k_values}
    total = 0
    for _ in range(args.eval_batches):
        examples = generator.batch(args.eval_batch_size, harder=harder)
        total += len(examples)
        for k in k_values:
            nll = score_candidates_hook(
                model=model,
                hook=hook,
                examples=examples,
                tokenizer=tokenizer,
                max_length=args.max_length,
                input_device=input_device,
                candidate_batch_size=args.candidate_batch_size,
                steps=int(k),
            )
            pred = nll.argmin(dim=1)
            gold = torch.tensor([ex.correct_idx for ex in examples], dtype=torch.long)
            correct_by_k[int(k)] += int((pred == gold).sum().item())
    hook.adapter.train()
    return {k: correct_by_k[k] / max(1, total) for k in correct_by_k}


def evaluate_head(
    model: nn.Module,
    head: HeadClassifier,
    generator: ModularArithmeticGenerator,
    tokenizer: Any,
    input_device: torch.device,
    head_device: torch.device,
    args: argparse.Namespace,
    k_values: Sequence[int],
    harder: bool,
) -> Dict[int, float]:
    model.eval()
    head.eval()
    correct_by_k = {int(k): 0 for k in k_values}
    total = 0
    for _ in range(args.eval_batches):
        examples = generator.batch(args.eval_batch_size, harder=harder)
        batch = collate_train_examples(examples, tokenizer, max_length=args.max_length, device=input_device)
        hidden = encode_hidden_for_head(model, batch).to(head_device)
        amask = batch.attention_mask.to(head_device)
        pmask = batch.prompt_mask.to(head_device)
        gold = batch.correct_idx.to(head_device)
        total += len(examples)
        for k in k_values:
            logits = head(hidden, amask, pmask, steps=int(k))
            pred = logits.argmax(dim=-1)
            correct_by_k[int(k)] += int((pred == gold).sum().item())
    head.train()
    return {k: correct_by_k[k] / max(1, total) for k in correct_by_k}


def format_accs(prefix: str, accs: Dict[int, float]) -> str:
    parts = [f"K={k}: {v*100:.1f}%" for k, v in sorted(accs.items())]
    return f"{prefix} " + " | ".join(parts)


def collect_metadata(args: argparse.Namespace, loader_name: Optional[str] = None) -> Dict[str, Any]:
    try:
        import transformers  # type: ignore
        transformers_version = getattr(transformers, "__version__", "unknown")
    except Exception:
        transformers_version = "unavailable"
    try:
        import bitsandbytes as bnb  # type: ignore
        bnb_version = getattr(bnb, "__version__", "unknown")
    except Exception:
        bnb_version = "unavailable"
    source_path = Path(__file__)
    try:
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except Exception:
        source_sha256 = "unavailable"

    cuda: Dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        cuda["devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            cuda["devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "major": props.major,
                    "minor": props.minor,
                }
            )
    return {
        "created_unix": time.time(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "bitsandbytes_version": bnb_version,
        "cuda": cuda,
        "loader_name": loader_name,
        "script": str(source_path),
        "script_sha256": source_sha256,
        "args": vars(args),
    }


# -----------------------------------------------------------------------------
# Training loops
# -----------------------------------------------------------------------------

def train_hook_mode(
    args: argparse.Namespace,
    tokenizer: Any,
    model: nn.Module,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path, layers = find_transformer_layers(model)
    layer_idx = args.hook_layer if args.hook_layer >= 0 else len(layers) + args.hook_layer
    if not (0 <= layer_idx < len(layers)):
        raise ValueError(f"hook_layer {args.hook_layer} resolved to {layer_idx}, but model has {len(layers)} layers")
    hidden_dim = infer_hidden_dim(model)
    layer_device = get_module_device(layers[layer_idx])
    input_device = get_input_device(model)
    print(f"[hook] layer list: {path} ({len(layers)} layers); injecting at index {layer_idx} on {layer_device}")
    print(f"[hook] hidden_dim={hidden_dim}; input_device={input_device}")

    adapter = RuntimeDeltaAdapter(
        hidden_dim=hidden_dim,
        runtime_dim=args.runtime_dim,
        workspace_tokens=args.workspace_tokens,
        heads=args.runtime_heads,
        num_bases=args.num_bases,
        rank=args.adapter_rank,
        mem_dim=args.mem_dim,
        dropout=args.dropout,
        init_scale=args.init_scale,
        disable_fast_memory=args.disable_fast_memory,
        disable_dynamic_lowrank=args.disable_dynamic_lowrank,
        value_modulus=args.modulus if args.aux_value_loss > 0.0 else 0,
    ).to(layer_device)
    hook = HookController(layers[layer_idx], adapter)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_gen = ModularArithmeticGenerator(args.modulus, args.num_choices, args.min_steps, args.max_steps, seed=args.seed)
    val_gen = ModularArithmeticGenerator(args.modulus, args.num_choices, args.min_steps, args.max_steps, seed=args.seed + 10_000)
    hard_gen = ModularArithmeticGenerator(args.modulus, args.num_choices, args.min_steps, args.max_steps, seed=args.seed + 20_000)
    train_k = parse_int_list(args.train_k)
    eval_k = parse_int_list(args.eval_k)

    results: Dict[str, Any] = {
        "mode": "hook",
        "metadata": metadata or collect_metadata(args),
        "architecture": {
            "layer_path": path,
            "num_layers": len(layers),
            "hook_layer_index": layer_idx,
            "hidden_dim": hidden_dim,
            "runtime_dim": args.runtime_dim,
            "workspace_tokens": args.workspace_tokens,
            "runtime_heads": args.runtime_heads,
            "num_bases": args.num_bases,
            "adapter_rank": args.adapter_rank,
            "mem_dim": args.mem_dim,
            "disable_fast_memory": args.disable_fast_memory,
            "disable_dynamic_lowrank": args.disable_dynamic_lowrank,
            "aux_value_loss": args.aux_value_loss,
        },
        "train": [],
        "eval": [],
    }
    t0 = time.time()
    adapter.train()
    model.eval()
    print(f"[train] optimizer steps={args.train_steps}, grad_accum={args.grad_accum}, train_k={train_k}")

    try:
        if not args.skip_initial_eval and args.eval_every > 0:
            print("[eval/init] evaluating untrained adapter and unhooked frozen model", flush=True)
            base_val = evaluate_base_lm(model, val_gen, tokenizer, input_device, args, harder=False)
            base_hard = evaluate_base_lm(model, hard_gen, tokenizer, input_device, args, harder=True)
            init_val = evaluate_hook(model, hook, val_gen, tokenizer, input_device, args, eval_k, harder=False)
            init_hard = evaluate_hook(model, hook, hard_gen, tokenizer, input_device, args, eval_k, harder=True)
            print(f"[eval/base] val={base_val*100:.1f}% hard={base_hard*100:.1f}%", flush=True)
            print(format_accs("[eval/init/val] ", init_val), flush=True)
            print(format_accs("[eval/init/hard]", init_hard), flush=True)
            results["eval"].append(
                {
                    "step": 0,
                    "base_val": base_val,
                    "base_hard": base_hard,
                    "val": init_val,
                    "hard": init_hard,
                }
            )
            save_checkpoint(args, adapter, results)

        for step in range(1, args.train_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            aux_last: Dict[str, float] = {}
            for _ in range(args.grad_accum):
                examples = train_gen.batch(args.batch_size, harder=False)
                batch = collate_train_examples(examples, tokenizer, max_length=args.max_length, device=input_device)
                k = random.choice(train_k)
                loss, aux = forward_hook_loss(model, hook, batch, steps=k, aux_value_loss_weight=args.aux_value_loss)
                # Encourage non-collapsed program gates early; harmless if set 0.
                if args.gate_entropy_bonus != 0.0 and aux.get("gate_entropy", 0.0) > 0.0:
                    # aux is detached float; use adapter's last gate entropy only for logging in this minimal script.
                    pass
                (loss / args.grad_accum).backward()
                loss_accum += float(loss.detach().cpu())
                aux_last = aux
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
            optimizer.step()

            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - t0
                lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"[train] step={step:05d} loss={loss_accum/args.grad_accum:.4f} "
                    f"grad={float(grad_norm):.3f} lr={lr:.2e} elapsed={elapsed/60:.1f}m"
                )
                if aux_last:
                    msg += " " + " ".join(f"{k}={v:.4g}" for k, v in aux_last.items())
                print(msg, flush=True)
                results["train"].append({"step": step, "loss": loss_accum / args.grad_accum, **aux_last})

            if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.train_steps):
                val_acc = evaluate_hook(model, hook, val_gen, tokenizer, input_device, args, eval_k, harder=False)
                hard_acc = evaluate_hook(model, hook, hard_gen, tokenizer, input_device, args, eval_k, harder=True)
                print(format_accs("[eval/val] ", val_acc), flush=True)
                print(format_accs("[eval/hard]", hard_acc), flush=True)
                results["eval"].append({"step": step, "val": val_acc, "hard": hard_acc})
                save_checkpoint(args, adapter, results)
    finally:
        hook.close()
    save_checkpoint(args, adapter, results)
    return results


def train_head_mode(
    args: argparse.Namespace,
    tokenizer: Any,
    model: nn.Module,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hidden_dim = infer_hidden_dim(model)
    input_device = get_input_device(model)
    head_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[head] hidden_dim={hidden_dim}; input_device={input_device}; head_device={head_device}")
    head = HeadClassifier(
        hidden_dim=hidden_dim,
        num_choices=args.num_choices,
        runtime_dim=args.runtime_dim,
        workspace_tokens=args.workspace_tokens,
        heads=args.runtime_heads,
        num_bases=args.num_bases,
        rank=args.adapter_rank,
        mem_dim=args.mem_dim,
        dropout=args.dropout,
        disable_fast_memory=args.disable_fast_memory,
        disable_dynamic_lowrank=args.disable_dynamic_lowrank,
    ).to(head_device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_gen = ModularArithmeticGenerator(args.modulus, args.num_choices, args.min_steps, args.max_steps, seed=args.seed)
    val_gen = ModularArithmeticGenerator(args.modulus, args.num_choices, args.min_steps, args.max_steps, seed=args.seed + 10_000)
    hard_gen = ModularArithmeticGenerator(args.modulus, args.num_choices, args.min_steps, args.max_steps, seed=args.seed + 20_000)
    train_k = parse_int_list(args.train_k)
    eval_k = parse_int_list(args.eval_k)
    results: Dict[str, Any] = {
        "mode": "head",
        "metadata": metadata or collect_metadata(args),
        "architecture": {
            "hidden_dim": hidden_dim,
            "runtime_dim": args.runtime_dim,
            "workspace_tokens": args.workspace_tokens,
            "runtime_heads": args.runtime_heads,
            "num_bases": args.num_bases,
            "adapter_rank": args.adapter_rank,
            "mem_dim": args.mem_dim,
            "disable_fast_memory": args.disable_fast_memory,
            "disable_dynamic_lowrank": args.disable_dynamic_lowrank,
        },
        "train": [],
        "eval": [],
    }
    t0 = time.time()
    model.eval()
    head.train()

    for step in range(1, args.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        acc_accum = 0.0
        for _ in range(args.grad_accum):
            examples = train_gen.batch(args.batch_size, harder=False)
            batch = collate_train_examples(examples, tokenizer, max_length=args.max_length, device=input_device)
            with torch.no_grad():
                hidden = encode_hidden_for_head(model, batch).to(head_device)
            amask = batch.attention_mask.to(head_device)
            pmask = batch.prompt_mask.to(head_device)
            gold = batch.correct_idx.to(head_device)
            k = random.choice(train_k)
            logits = head(hidden, amask, pmask, steps=k)
            loss = F.cross_entropy(logits.float(), gold)
            (loss / args.grad_accum).backward()
            loss_accum += float(loss.detach().cpu())
            acc_accum += float((logits.argmax(dim=-1) == gold).float().mean().detach().cpu())
        grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            aux = head.adapter.last_aux
            msg = (
                f"[train/head] step={step:05d} loss={loss_accum/args.grad_accum:.4f} "
                f"acc={acc_accum/args.grad_accum*100:.1f}% grad={float(grad_norm):.3f} elapsed={elapsed/60:.1f}m"
            )
            if aux:
                msg += " " + " ".join(f"{k}={v:.4g}" for k, v in aux.items())
            print(msg, flush=True)
            results["train"].append({"step": step, "loss": loss_accum / args.grad_accum, "acc": acc_accum / args.grad_accum, **aux})

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.train_steps):
            val_acc = evaluate_head(model, head, val_gen, tokenizer, input_device, head_device, args, eval_k, harder=False)
            hard_acc = evaluate_head(model, head, hard_gen, tokenizer, input_device, head_device, args, eval_k, harder=True)
            print(format_accs("[eval/val] ", val_acc), flush=True)
            print(format_accs("[eval/hard]", hard_acc), flush=True)
            results["eval"].append({"step": step, "val": val_acc, "hard": hard_acc})
            save_checkpoint(args, head, results)
    save_checkpoint(args, head, results)
    return results


def _saved_arg(saved_args: Dict[str, Any], name: str, default: Any) -> Any:
    return saved_args[name] if name in saved_args else default


@torch.no_grad()
def eval_hook_checkpoint(args: argparse.Namespace, tokenizer: Any, model: nn.Module, metadata: Dict[str, Any]) -> Dict[str, Any]:
    ckpt_path = Path(args.eval_only_checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args: Dict[str, Any] = dict(ckpt.get("args", {}))
    path, layers = find_transformer_layers(model)
    saved_hook_layer = int(_saved_arg(saved_args, "hook_layer", args.hook_layer))
    layer_idx = saved_hook_layer if saved_hook_layer >= 0 else len(layers) + saved_hook_layer
    if not (0 <= layer_idx < len(layers)):
        raise ValueError(f"checkpoint hook_layer {saved_hook_layer} resolved to {layer_idx}, but model has {len(layers)} layers")

    hidden_dim = infer_hidden_dim(model)
    layer_device = get_module_device(layers[layer_idx])
    input_device = get_input_device(model)
    aux_value_loss = float(_saved_arg(saved_args, "aux_value_loss", 0.0))
    adapter = RuntimeDeltaAdapter(
        hidden_dim=hidden_dim,
        runtime_dim=int(_saved_arg(saved_args, "runtime_dim", args.runtime_dim)),
        workspace_tokens=int(_saved_arg(saved_args, "workspace_tokens", args.workspace_tokens)),
        heads=int(_saved_arg(saved_args, "runtime_heads", args.runtime_heads)),
        num_bases=int(_saved_arg(saved_args, "num_bases", args.num_bases)),
        rank=int(_saved_arg(saved_args, "adapter_rank", args.adapter_rank)),
        mem_dim=int(_saved_arg(saved_args, "mem_dim", args.mem_dim)),
        dropout=float(_saved_arg(saved_args, "dropout", args.dropout)),
        init_scale=float(_saved_arg(saved_args, "init_scale", args.init_scale)),
        disable_fast_memory=bool(_saved_arg(saved_args, "disable_fast_memory", False)),
        disable_dynamic_lowrank=bool(_saved_arg(saved_args, "disable_dynamic_lowrank", False)),
        value_modulus=int(_saved_arg(saved_args, "modulus", args.modulus)) if aux_value_loss > 0.0 else 0,
    ).to(layer_device)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    hook = HookController(layers[layer_idx], adapter)
    eval_k = parse_int_list(args.eval_k)
    val_gen = ModularArithmeticGenerator(args.modulus, args.num_choices, args.min_steps, args.max_steps, seed=args.seed + 10_000)
    hard_gen = ModularArithmeticGenerator(args.modulus, args.num_choices, args.min_steps, args.max_steps, seed=args.seed + 20_000)
    try:
        val_acc = evaluate_hook(model, hook, val_gen, tokenizer, input_device, args, eval_k, harder=False)
        hard_acc = evaluate_hook(model, hook, hard_gen, tokenizer, input_device, args, eval_k, harder=True)
    finally:
        hook.close()

    results = {
        "mode": "eval_hook_checkpoint",
        "metadata": metadata,
        "checkpoint": str(ckpt_path),
        "checkpoint_saved_args": saved_args,
        "architecture": {
            "layer_path": path,
            "num_layers": len(layers),
            "hook_layer_index": layer_idx,
            "hidden_dim": hidden_dim,
        },
        "eval": {
            "val": val_acc,
            "hard": hard_acc,
            "eval_batches": args.eval_batches,
            "eval_batch_size": args.eval_batch_size,
            "num_examples_per_split": args.eval_batches * args.eval_batch_size,
        },
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eval_only_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(format_accs("[eval-only/val] ", val_acc), flush=True)
    print(format_accs("[eval-only/hard]", hard_acc), flush=True)
    print(f"[eval-only] wrote {out_file}")
    return results


def save_checkpoint(args: argparse.Namespace, module: nn.Module, results: Dict[str, Any]) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "latent_adapter.pt"
    payload = {
        "state_dict": module.state_dict(),
        "args": vars(args),
        "results": results,
    }
    torch.save(payload, ckpt_path)
    latest_eval = results.get("eval", [])[-1] if results.get("eval") else None
    if latest_eval is not None and "step" in latest_eval:
        torch.save(payload, out_dir / f"latent_adapter_step{int(latest_eval['step']):05d}.pt")
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Frozen Qwen + latent recurrent fast-weight hyperadapter experiment")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-4B")
    p.add_argument("--text_model_fallback", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--model_class", type=str, default="auto", choices=["auto", "causal", "multimodal"])
    p.add_argument("--mode", type=str, default="hook", choices=["hook", "head"], help="hook = inject inside LM; head = cheap diagnostic classifier")
    p.add_argument("--eval_only_checkpoint", type=str, default="", help="Evaluate a saved hook-mode checkpoint without training")
    p.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--verbose", action="store_true")

    # Data/task controls
    p.add_argument("--modulus", type=int, default=97)
    p.add_argument("--num_choices", type=int, default=5)
    p.add_argument("--min_steps", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=8)
    p.add_argument("--max_length", type=int, default=384)

    # Adapter architecture
    p.add_argument("--hook_layer", type=int, default=-4, help="Layer index for hook mode; negative counts from end")
    p.add_argument("--runtime_dim", type=int, default=256)
    p.add_argument("--workspace_tokens", type=int, default=8)
    p.add_argument("--runtime_heads", type=int, default=4)
    p.add_argument("--num_bases", type=int, default=12)
    p.add_argument("--adapter_rank", type=int, default=16)
    p.add_argument("--mem_dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--init_scale", type=float, default=0.05)
    p.add_argument("--disable_fast_memory", action="store_true", help="Ablation: remove recurrent fast-weight memory reads/writes")
    p.add_argument("--disable_dynamic_lowrank", action="store_true", help="Ablation: remove activation-programmed low-rank transform bank")

    # Training/eval controls
    p.add_argument("--train_steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--train_k", type=str, default="1,2,4")
    p.add_argument("--eval_k", type=str, default="0,1,2,4,8")
    p.add_argument("--aux_value_loss", type=float, default=0.0, help="Hook mode only: weight for predicting the exact numeric answer from final workspace state")
    p.add_argument("--gate_entropy_bonus", type=float, default=0.0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--candidate_batch_size", type=int, default=5)
    p.add_argument("--skip_initial_eval", action="store_true")
    p.add_argument("--output_dir", type=str, default="runs/latent_qwen_fastweight")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    print(json.dumps(vars(args), indent=2))

    tokenizer, model, loader_name = load_tokenizer_and_model(args)
    print(f"[load] loaded with {loader_name}")
    print(f"[model] hidden_dim={infer_hidden_dim(model)} params_frozen={sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    metadata = collect_metadata(args, loader_name=loader_name)

    if args.eval_only_checkpoint:
        results = eval_hook_checkpoint(args, tokenizer, model, metadata=metadata)
    elif args.mode == "hook":
        results = train_hook_mode(args, tokenizer, model, metadata=metadata)
    else:
        results = train_head_mode(args, tokenizer, model, metadata=metadata)

    print("[done] final results:")
    eval_obj = results.get("eval") if isinstance(results, dict) else None
    if isinstance(eval_obj, list):
        printable = eval_obj[-1:] if eval_obj else results
    elif eval_obj:
        printable = eval_obj
    else:
        printable = results
    print(json.dumps(printable, indent=2))
    print(f"[done] wrote checkpoint/results to {args.output_dir}")


if __name__ == "__main__":
    main()
