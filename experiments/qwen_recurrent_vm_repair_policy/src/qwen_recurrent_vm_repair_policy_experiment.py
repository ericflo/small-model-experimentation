#!/usr/bin/env python3
"""Recurrent VM repair policy for a Qwen-attached bytecode compiler.

The compiler emits an initial typed VM program from frozen Qwen hidden states.
The repair policy then treats one forward pass as one transition in a loop:
given the prompt embedding, the current program, and the VM execution trace, it
predicts either one edit action or STOP. The edited program is executed and fed
back to the same policy for another step. A DAgger round exposes the policy to
states reached by its own imperfect actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from typed_bytecode_core import (
    CHECKPOINT_ROOT,
    MAX_PROGRAM_LEN,
    MAX_PROMPT_LEN,
    MODULUS,
    NO_ARG,
    OPCODES,
    OP_TO_ID,
    ROOT,
    RUNS,
    BytecodeProgram,
    TaskExample,
    TaskGenerator,
    choose_answer_verified_candidate,
    execute_program,
    generate_candidates,
    normalize_program,
    program_equal,
    program_from_logits,
    set_seed,
    stack_delta,
    allowed_ops_for_depth,
)


STOP_ID = 0
OP_OFFSET = 1
ARG_OFFSET = OP_OFFSET + MAX_PROGRAM_LEN * len(OPCODES)
ACTION_SIZE = ARG_OFFSET + MAX_PROGRAM_LEN * MODULUS


@dataclass
class RepairState:
    ex_idx: int
    ops: List[int]
    args: List[int]
    trace_top: List[int]
    trace_depth: List[int]
    trace_mask: List[float]
    valid: int
    final: int
    step: int
    action: int
    source: str


@dataclass
class TrajectoryStats:
    phase: str
    source_examples: int
    states: int
    stop_labels: int
    edit_labels: int
    start_correct_rate: float
    teacher_success_rate: float
    mean_states_per_example: float
    mean_edits_before_stop: float


@dataclass
class EvalResult:
    run: str
    phase: str
    split: str
    mode: str
    k: int
    n: int
    accuracy: float
    valid_rate: float
    program_exact: float
    stop_rate: float
    false_stop_rate: float
    mean_steps: float
    base_accuracy: float
    base_search_accuracy: float
    base_oracle_accuracy: float


class FeatureSet(Dataset):
    def __init__(self, examples: Sequence[TaskExample], hidden: torch.Tensor, mask: torch.Tensor) -> None:
        self.examples = list(examples)
        self.hidden = hidden
        self.mask = mask
        weights = mask.float().unsqueeze(-1)
        pooled = (hidden.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        self.prompt_features = pooled.to(torch.float16)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        program = normalize_program(ex.program)
        return {
            "hidden": self.hidden[idx],
            "attention_mask": self.mask[idx],
            "prompt_features": self.prompt_features[idx],
            "ops": torch.tensor(program.ops, dtype=torch.long),
            "args": torch.tensor(program.args, dtype=torch.long),
            "answer": torch.tensor(ex.answer, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class QwenCompiler(nn.Module):
    def __init__(self, hidden_size: int, d_model: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(hidden_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.slot_queries = nn.Parameter(torch.randn(MAX_PROGRAM_LEN, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=layers)
        self.op_head = nn.Linear(d_model, len(OPCODES))
        self.arg_head = nn.Linear(d_model, MODULUS)
        self.answer_head = nn.Linear(d_model, MODULUS)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        memory = self.input_norm(self.input_proj(hidden.float()))
        key_padding_mask = ~attention_mask.bool()
        bsz = hidden.shape[0]
        queries = self.slot_queries.unsqueeze(0).expand(bsz, -1, -1)
        decoded = self.decoder(queries, memory, memory_key_padding_mask=key_padding_mask)
        pooled = (memory * attention_mask.unsqueeze(-1).float()).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return {
            "op_logits": self.op_head(decoded),
            "arg_logits": self.arg_head(decoded),
            "answer_logits": self.answer_head(pooled),
        }


class RecurrentRepairPolicy(nn.Module):
    def __init__(self, hidden_size: int, d_model: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.prompt_proj = nn.Linear(hidden_size, d_model)
        self.prompt_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(d_model)
        self.op_embed = nn.Embedding(len(OPCODES), d_model)
        self.arg_embed = nn.Embedding(MODULUS, d_model)
        self.trace_top_embed = nn.Embedding(MODULUS, d_model)
        self.trace_depth_embed = nn.Embedding(MAX_PROGRAM_LEN + 1, d_model)
        self.valid_embed = nn.Embedding(2, d_model)
        self.final_embed = nn.Embedding(MODULUS, d_model)
        self.step_embed = nn.Embedding(64, d_model)
        self.kind_embed = nn.Embedding(2, d_model)
        self.pos_embed = nn.Parameter(torch.randn(MAX_PROGRAM_LEN + 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.kind_head = nn.Linear(d_model, 3)  # STOP, OP edit, ARG edit
        self.slot_head = nn.Linear(d_model, MAX_PROGRAM_LEN)
        self.op_value_head = nn.Linear(d_model, len(OPCODES))
        self.arg_value_head = nn.Linear(d_model, MODULUS)

    def forward(
        self,
        prompt_features: torch.Tensor,
        ops: torch.Tensor,
        args: torch.Tensor,
        trace_top: torch.Tensor,
        trace_depth: torch.Tensor,
        trace_mask: torch.Tensor,
        valid: torch.Tensor,
        final: torch.Tensor,
        step: torch.Tensor,
        prompt_hidden: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz = ops.shape[0]
        prompt = self.prompt_norm(self.prompt_proj(prompt_features.float())).unsqueeze(1)
        prompt = prompt + self.valid_embed(valid.long()).unsqueeze(1)
        prompt = prompt + self.final_embed(final.long().clamp(0, MODULUS - 1)).unsqueeze(1)
        prompt = prompt + self.step_embed(step.long().clamp(0, 63)).unsqueeze(1)
        prompt = prompt + self.kind_embed(torch.zeros(bsz, 1, dtype=torch.long, device=ops.device))
        slot_x = (
            self.op_embed(ops.long())
            + self.arg_embed((args % MODULUS).long())
            + self.trace_top_embed((trace_top % MODULUS).long())
            + self.trace_depth_embed(trace_depth.long().clamp(0, MAX_PROGRAM_LEN))
            + trace_mask.unsqueeze(-1).float()
        )
        slot_x = slot_x + self.kind_embed(torch.ones(bsz, MAX_PROGRAM_LEN, dtype=torch.long, device=ops.device))
        x = torch.cat([prompt, slot_x], dim=1) + self.pos_embed.unsqueeze(0)
        pooled = self.encoder(x)[:, 0]
        if prompt_hidden is not None and prompt_mask is not None:
            memory = self.prompt_norm(self.prompt_proj(prompt_hidden.float()))
            context, _ = self.cross_attn(
                pooled.unsqueeze(1),
                memory,
                memory,
                key_padding_mask=~prompt_mask.bool(),
                need_weights=False,
            )
            pooled = self.output_norm(pooled + context.squeeze(1))
        return {
            "kind_logits": self.kind_head(pooled),
            "slot_logits": self.slot_head(pooled),
            "op_logits": self.op_value_head(pooled),
            "arg_logits": self.arg_value_head(pooled),
        }


class RepairStateDataset(Dataset):
    def __init__(self, states: Sequence[RepairState]) -> None:
        self.states = list(states)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        state = self.states[idx]
        return {
            "ex_idx": torch.tensor(state.ex_idx, dtype=torch.long),
            "ops": torch.tensor(state.ops, dtype=torch.long),
            "args": torch.tensor(state.args, dtype=torch.long),
            "trace_top": torch.tensor(state.trace_top, dtype=torch.long),
            "trace_depth": torch.tensor(state.trace_depth, dtype=torch.long),
            "trace_mask": torch.tensor(state.trace_mask, dtype=torch.float32),
            "valid": torch.tensor(state.valid, dtype=torch.long),
            "final": torch.tensor(state.final, dtype=torch.long),
            "step": torch.tensor(state.step, dtype=torch.long),
            "action": torch.tensor(state.action, dtype=torch.long),
        }


def collate_repair_states(batch: Sequence[Dict[str, torch.Tensor]], features: FeatureSet) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in ["ex_idx", "ops", "args", "trace_top", "trace_depth", "trace_mask", "valid", "final", "step", "action"]:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    out["prompt_features"] = features.prompt_features[out["ex_idx"]]
    out["prompt_hidden"] = features.hidden[out["ex_idx"]]
    out["prompt_mask"] = features.mask[out["ex_idx"]]
    return out


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token


def load_qwen(model_name: str) -> Tuple[Any, Any, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=quant, device_map="auto")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model, int(model.config.hidden_size)


def extract_features(
    examples: Sequence[TaskExample],
    tokenizer: Any,
    qwen: Any,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_hidden: List[torch.Tensor] = []
    all_mask: List[torch.Tensor] = []
    prompts = [ex.prompt for ex in examples]
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            enc = tokenizer(batch, padding="max_length", truncation=True, max_length=MAX_PROMPT_LEN, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            out = qwen.model(input_ids=input_ids, attention_mask=mask, use_cache=False)
            all_hidden.append(out.last_hidden_state.detach().to(torch.float16).cpu())
            all_mask.append(mask.detach().bool().cpu())
            print(f"[features] {min(start + len(batch), len(prompts))}/{len(prompts)}", flush=True)
    return torch.cat(all_hidden, dim=0), torch.cat(all_mask, dim=0)


def make_splits(args: argparse.Namespace) -> Dict[str, List[TaskExample]]:
    gen = TaskGenerator(seed=args.seed, max_arith_steps=args.max_arith_steps)
    return {
        "seed_train": gen.make_set(args.seed_train_size, template="mixed", hard=False),
        "unlabeled_train": gen.make_set(args.unlabeled_train_size, template="mixed", hard=False),
        "full_supervised_train": gen.make_set(args.full_supervised_size, template="mixed", hard=False),
        "val_mixed": gen.make_set(args.val_size, template="mixed", hard=False),
        "fresh_standard": gen.make_set(args.fresh_size, template="standard", hard=False),
        "fresh_paraphrase": gen.make_set(args.fresh_size, template="paraphrase", hard=False),
        "fresh_paired": gen.make_paired_set(max(1, args.fresh_size // 2), hard=False),
        "hard_composition": gen.make_set(args.hard_size, template="mixed", hard=True),
    }


def append_csv(path: Path, row: Dict[str, Any], rewrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not rewrite:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            keys = list(reader.fieldnames or [])
        for key in row:
            if key not in keys:
                keys.append(key)
        rows.append(row)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_eval_results(path: Path, results: Sequence[EvalResult]) -> None:
    write_rows(path, [asdict(r) for r in results])


def action_op(slot: int, op: int) -> int:
    return OP_OFFSET + slot * len(OPCODES) + int(op)


def action_arg(slot: int, arg: int) -> int:
    return ARG_OFFSET + slot * MODULUS + int(arg % MODULUS)


def decode_action(action: int) -> Tuple[str, int, int]:
    if int(action) == STOP_ID:
        return "STOP", -1, 0
    if OP_OFFSET <= int(action) < ARG_OFFSET:
        x = int(action) - OP_OFFSET
        return "OP", x // len(OPCODES), x % len(OPCODES)
    x = int(action) - ARG_OFFSET
    return "ARG", x // MODULUS, x % MODULUS


def action_components_tensor(actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kinds: List[int] = []
    slots: List[int] = []
    ops: List[int] = []
    args: List[int] = []
    for action in actions.detach().cpu().tolist():
        kind, slot, value = decode_action(int(action))
        if kind == "STOP":
            kinds.append(0)
            slots.append(0)
            ops.append(0)
            args.append(0)
        elif kind == "OP":
            kinds.append(1)
            slots.append(int(slot))
            ops.append(int(value))
            args.append(0)
        else:
            kinds.append(2)
            slots.append(int(slot))
            ops.append(0)
            args.append(int(value))
    device = actions.device
    return (
        torch.tensor(kinds, dtype=torch.long, device=device),
        torch.tensor(slots, dtype=torch.long, device=device),
        torch.tensor(ops, dtype=torch.long, device=device),
        torch.tensor(args, dtype=torch.long, device=device),
    )


def compose_predicted_actions(outputs: Dict[str, torch.Tensor], forbid_stop: bool = False) -> torch.Tensor:
    kind_logits = outputs["kind_logits"]
    if forbid_stop:
        kind_logits = kind_logits.clone()
        kind_logits[:, 0] = -1e9
    kind = kind_logits.argmax(dim=-1)
    slot = outputs["slot_logits"].argmax(dim=-1)
    op = outputs["op_logits"].argmax(dim=-1)
    arg = outputs["arg_logits"].argmax(dim=-1)
    actions = torch.full_like(kind, STOP_ID)
    op_mask = kind.eq(1)
    arg_mask = kind.eq(2)
    actions[op_mask] = OP_OFFSET + slot[op_mask] * len(OPCODES) + op[op_mask]
    actions[arg_mask] = ARG_OFFSET + slot[arg_mask] * MODULUS + arg[arg_mask]
    return actions


def prefix_depth(program: BytecodeProgram, slot: int) -> Tuple[int, bool]:
    depth = 0
    ended = False
    for i, op in enumerate(normalize_program(program).ops[:slot]):
        name = OPCODES[int(op)] if 0 <= int(op) < len(OPCODES) else "BAD"
        if ended:
            if name != "PAD":
                return depth, False
            continue
        if name == "PAD":
            return depth, False
        if name == "END":
            ended = True
            continue
        depth += stack_delta(int(op))
        if depth < 0:
            return 0, False
    return depth, True


def masked_action_from_outputs(
    outputs: Dict[str, torch.Tensor],
    program: BytecodeProgram,
    forbid_stop: bool,
    stop_margin: float,
    typed_action_mask: bool,
) -> int:
    kind_logits = outputs["kind_logits"][0].detach().float().cpu()
    slot_logits = outputs["slot_logits"][0].detach().float().cpu()
    op_logits = outputs["op_logits"][0].detach().float().cpu()
    arg_logits = outputs["arg_logits"][0].detach().float().cpu()
    if forbid_stop:
        kind_logits[0] = -1e9
    elif float(stop_margin) > 0.0:
        edit_best = torch.max(kind_logits[1:]).item()
        if kind_logits[0].item() < edit_best + float(stop_margin):
            kind_logits[0] = -1e9
    kind = int(torch.argmax(kind_logits).item())
    if kind == 0:
        return STOP_ID

    prog = normalize_program(program)
    slot_scores = slot_logits.clone()
    if typed_action_mask:
        if kind == 1:
            allowed_slots: List[int] = []
            for slot in range(MAX_PROGRAM_LEN):
                _, prefix_ok = prefix_depth(prog, slot)
                if prefix_ok:
                    allowed_slots.append(slot)
            if allowed_slots:
                mask = torch.full_like(slot_scores, -1e9)
                mask[allowed_slots] = 0.0
                slot_scores = slot_scores + mask
        else:
            push_slots = [i for i, op in enumerate(prog.ops) if int(op) == OP_TO_ID["PUSH"]]
            if push_slots:
                mask = torch.full_like(slot_scores, -1e9)
                mask[push_slots] = 0.0
                slot_scores = slot_scores + mask
    slot = int(torch.argmax(slot_scores).item())
    if kind == 1:
        op_scores = op_logits.clone()
        if typed_action_mask:
            depth, prefix_ok = prefix_depth(prog, slot)
            allowed = allowed_ops_for_depth(depth, slot, MAX_PROGRAM_LEN) if prefix_ok else list(range(len(OPCODES)))
            if allowed:
                mask = torch.full_like(op_scores, -1e9)
                mask[allowed] = 0.0
                op_scores = op_scores + mask
        op = int(torch.argmax(op_scores).item())
        return action_op(slot, op)
    arg = int(torch.argmax(arg_logits).item())
    return action_arg(slot, arg)


def apply_action(program: BytecodeProgram, action: int) -> Tuple[BytecodeProgram, bool]:
    program = normalize_program(program)
    kind, slot, value = decode_action(int(action))
    ops = list(program.ops)
    args = list(program.args)
    if kind == "STOP":
        return program, True
    if slot < 0 or slot >= MAX_PROGRAM_LEN:
        return program, False
    if kind == "OP":
        ops[slot] = int(value)
        if int(value) != OP_TO_ID["PUSH"]:
            args[slot] = NO_ARG
    else:
        args[slot] = int(value) % MODULUS
    return normalize_program(BytecodeProgram(ops, args)), False


def observation(program: BytecodeProgram) -> Tuple[int, int, List[int], List[int], List[float]]:
    valid, final, trace = execute_program(normalize_program(program))
    top: List[int] = []
    depth: List[int] = []
    mask: List[float] = []
    for slot in range(MAX_PROGRAM_LEN):
        if slot < len(trace):
            stack = trace[slot]
            mask.append(1.0)
            depth.append(min(len(stack), MAX_PROGRAM_LEN))
            top.append(int(stack[-1] % MODULUS) if stack else 0)
        else:
            mask.append(0.0)
            depth.append(0)
            top.append(0)
    return int(valid), int(final % MODULUS if valid else 0), top, depth, mask


def is_answer_correct(program: BytecodeProgram, answer: int) -> bool:
    valid, final, _ = execute_program(normalize_program(program))
    return bool(valid and int(final) == int(answer))


def oracle_next_action(current: BytecodeProgram, target: BytecodeProgram, answer: int, stop_on_answer: bool) -> int:
    current = normalize_program(current)
    target = normalize_program(target)
    if program_equal(current, target):
        return STOP_ID
    if stop_on_answer and is_answer_correct(current, answer):
        return STOP_ID
    for slot, (cur_op, tgt_op) in enumerate(zip(current.ops, target.ops)):
        if int(cur_op) != int(tgt_op):
            return action_op(slot, int(tgt_op))
    for slot, (cur_arg, tgt_arg) in enumerate(zip(current.args, target.args)):
        if int(cur_arg) != int(tgt_arg):
            return action_arg(slot, int(tgt_arg))
    return STOP_ID


def compiler_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], answer_weight: float) -> torch.Tensor:
    op_loss = F.cross_entropy(outputs["op_logits"].reshape(-1, len(OPCODES)), batch["ops"].reshape(-1))
    arg_loss = F.cross_entropy(outputs["arg_logits"].reshape(-1, MODULUS), batch["args"].reshape(-1))
    ans_loss = F.cross_entropy(outputs["answer_logits"], batch["answer"])
    return op_loss + arg_loss + answer_weight * ans_loss


def train_compiler(
    model: QwenCompiler,
    dataset: FeatureSet,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    answer_weight: float,
    log_path: Path,
    phase: str,
    quick_val: Optional[FeatureSet],
    search_topk: int,
    max_two_arg_pairs: int,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            outputs = model(batch["hidden"], batch["attention_mask"])
            loss = compiler_loss(outputs, batch, answer_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = int(batch["hidden"].shape[0])
            total += float(loss.detach().cpu()) * n
            count += n
        row: Dict[str, Any] = {"phase": phase, "epoch": epoch, "loss": total / max(1, count), "train_examples": len(dataset)}
        if quick_val is not None:
            quick = evaluate_compiler(model, quick_val, device, "quick", phase, search_topk, max_two_arg_pairs, limit=min(96, len(quick_val)))
            row["quick_val_direct_accuracy"] = quick["direct_accuracy"]
            row["quick_val_search_accuracy"] = quick["search_accuracy"]
        append_csv(log_path, row, rewrite=not log_path.exists() and epoch == 1)
        print(f"[train:{phase}] epoch={epoch} loss={row['loss']:.4f}", flush=True)


def evaluate_compiler(
    model: QwenCompiler,
    dataset: FeatureSet,
    device: torch.device,
    run: str,
    phase: str,
    search_topk: int,
    max_two_arg_pairs: int,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    n_items = len(dataset) if limit is None else min(limit, len(dataset))
    direct_ok = search_ok = oracle_ok = exact = valid_count = total_candidates = valid_candidates = 0
    with torch.no_grad():
        for idx in range(n_items):
            item = dataset[idx]
            ex = dataset.examples[idx]
            hidden = item["hidden"].unsqueeze(0).to(device)
            mask = item["attention_mask"].unsqueeze(0).to(device)
            outputs = model(hidden, mask)
            op_logits = outputs["op_logits"][0]
            arg_logits = outputs["arg_logits"][0]
            base = program_from_logits(op_logits, arg_logits)
            valid, final, _ = execute_program(base)
            direct_ok += int(valid and final == ex.answer)
            valid_count += int(valid)
            exact += int(program_equal(base, ex.program))
            candidates = generate_candidates(base, op_logits, arg_logits, topk=search_topk, max_two_arg_pairs=max_two_arg_pairs)
            total_candidates += len(candidates)
            chosen, found, valid_cands = choose_answer_verified_candidate(candidates, ex.answer, op_logits, arg_logits)
            valid_candidates += valid_cands
            search_ok += int(chosen is not None)
            oracle_ok += int(found > 0)
    n = max(1, n_items)
    return {
        "run": run,
        "phase": phase,
        "n": n_items,
        "direct_accuracy": direct_ok / n,
        "valid_rate": valid_count / n,
        "program_exact": exact / n,
        "search_accuracy": search_ok / n,
        "oracle_accuracy": oracle_ok / n,
        "candidate_valid_rate": valid_candidates / max(1, total_candidates),
        "mean_candidates": total_candidates / n,
    }


@torch.no_grad()
def base_programs(model: QwenCompiler, dataset: FeatureSet, device: torch.device) -> Tuple[List[BytecodeProgram], List[Tuple[torch.Tensor, torch.Tensor]]]:
    model.eval()
    programs: List[BytecodeProgram] = []
    logits: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        hidden = item["hidden"].unsqueeze(0).to(device)
        mask = item["attention_mask"].unsqueeze(0).to(device)
        outputs = model(hidden, mask)
        op_logits = outputs["op_logits"][0].detach().cpu()
        arg_logits = outputs["arg_logits"][0].detach().cpu()
        programs.append(program_from_logits(op_logits, arg_logits))
        logits.append((op_logits, arg_logits))
    return programs, logits


def make_state(ex_idx: int, program: BytecodeProgram, step: int, action: int, source: str) -> RepairState:
    prog = normalize_program(program)
    valid, final, top, depth, mask = observation(prog)
    return RepairState(
        ex_idx=ex_idx,
        ops=list(prog.ops),
        args=list(prog.args),
        trace_top=top,
        trace_depth=depth,
        trace_mask=mask,
        valid=valid,
        final=final,
        step=step,
        action=int(action),
        source=source,
    )


def collect_teacher_states(
    dataset: FeatureSet,
    bases: Sequence[BytecodeProgram],
    max_steps: int,
    stop_on_answer: bool,
    phase: str,
) -> Tuple[List[RepairState], TrajectoryStats]:
    states: List[RepairState] = []
    start_correct = teacher_success = stop_labels = edit_labels = total_edits = 0
    for idx, (ex, base) in enumerate(zip(dataset.examples, bases)):
        current = normalize_program(base)
        start_correct += int(is_answer_correct(current, ex.answer))
        edits = 0
        stopped = False
        for step in range(max_steps + 1):
            action = oracle_next_action(current, ex.program, ex.answer, stop_on_answer)
            states.append(make_state(idx, current, step, action, phase))
            if action == STOP_ID:
                stop_labels += 1
                stopped = True
                break
            edit_labels += 1
            edits += 1
            current, _ = apply_action(current, action)
        if not stopped:
            states.append(make_state(idx, ex.program, max_steps + 1, STOP_ID, f"{phase}_gold_stop"))
            stop_labels += 1
        total_edits += edits
        teacher_success += int(is_answer_correct(current, ex.answer))
    return states, TrajectoryStats(
        phase=phase,
        source_examples=len(dataset),
        states=len(states),
        stop_labels=stop_labels,
        edit_labels=edit_labels,
        start_correct_rate=start_correct / max(1, len(dataset)),
        teacher_success_rate=teacher_success / max(1, len(dataset)),
        mean_states_per_example=len(states) / max(1, len(dataset)),
        mean_edits_before_stop=total_edits / max(1, len(dataset)),
    )


@torch.no_grad()
def predict_action(
    policy: RecurrentRepairPolicy,
    dataset: FeatureSet,
    ex_idx: int,
    program: BytecodeProgram,
    step: int,
    device: torch.device,
    forbid_stop: bool = False,
    stop_margin: float = 0.0,
    typed_action_mask: bool = False,
) -> int:
    valid, final, top, depth, mask = observation(program)
    outputs = policy(
        dataset.prompt_features[ex_idx].unsqueeze(0).to(device),
        torch.tensor([normalize_program(program).ops], dtype=torch.long, device=device),
        torch.tensor([normalize_program(program).args], dtype=torch.long, device=device),
        torch.tensor([top], dtype=torch.long, device=device),
        torch.tensor([depth], dtype=torch.long, device=device),
        torch.tensor([mask], dtype=torch.float32, device=device),
        torch.tensor([valid], dtype=torch.long, device=device),
        torch.tensor([final], dtype=torch.long, device=device),
        torch.tensor([step], dtype=torch.long, device=device),
        dataset.hidden[ex_idx].unsqueeze(0).to(device),
        dataset.mask[ex_idx].unsqueeze(0).to(device),
    )
    if typed_action_mask or float(stop_margin) > 0.0:
        return masked_action_from_outputs(outputs, program, forbid_stop=forbid_stop, stop_margin=stop_margin, typed_action_mask=typed_action_mask)
    return int(compose_predicted_actions(outputs, forbid_stop=forbid_stop)[0].item())


def collect_dagger_states(
    policy: RecurrentRepairPolicy,
    dataset: FeatureSet,
    bases: Sequence[BytecodeProgram],
    device: torch.device,
    rollout_steps: int,
    stop_on_answer: bool,
    phase: str,
    stop_margin: float,
    typed_action_mask: bool,
) -> Tuple[List[RepairState], TrajectoryStats]:
    policy.eval()
    states: List[RepairState] = []
    start_correct = teacher_success = stop_labels = edit_labels = total_edits = 0
    for idx, (ex, base) in enumerate(zip(dataset.examples, bases)):
        current = normalize_program(base)
        start_correct += int(is_answer_correct(current, ex.answer))
        edits = 0
        stopped = False
        for step in range(rollout_steps + 1):
            oracle_action = oracle_next_action(current, ex.program, ex.answer, stop_on_answer)
            states.append(make_state(idx, current, step, oracle_action, phase))
            if oracle_action == STOP_ID:
                stop_labels += 1
                stopped = True
                break
            edit_labels += 1
            pred = predict_action(
                policy,
                dataset,
                idx,
                current,
                step,
                device,
                stop_margin=stop_margin,
                typed_action_mask=typed_action_mask,
            )
            if pred == STOP_ID:
                break
            current, _ = apply_action(current, pred)
            edits += 1
        if not stopped:
            states.append(make_state(idx, ex.program, rollout_steps + 1, STOP_ID, f"{phase}_gold_stop"))
            stop_labels += 1
        total_edits += edits
        teacher_success += int(is_answer_correct(current, ex.answer))
    return states, TrajectoryStats(
        phase=phase,
        source_examples=len(dataset),
        states=len(states),
        stop_labels=stop_labels,
        edit_labels=edit_labels,
        start_correct_rate=start_correct / max(1, len(dataset)),
        teacher_success_rate=teacher_success / max(1, len(dataset)),
        mean_states_per_example=len(states) / max(1, len(dataset)),
        mean_edits_before_stop=total_edits / max(1, len(dataset)),
    )


def train_policy(
    policy: RecurrentRepairPolicy,
    states: Sequence[RepairState],
    features: FeatureSet,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    stop_loss_weight: float,
    log_path: Path,
    phase: str,
    quick_eval: Optional[Tuple[QwenCompiler, FeatureSet, List[int], int, int, Sequence[int]]] = None,
    quick_stop_margin: float = 0.0,
    quick_typed_action_mask: bool = False,
) -> None:
    dataset = RepairStateDataset(states)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda xs: collate_repair_states(xs, features),
    )
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    for epoch in range(1, epochs + 1):
        policy.train()
        total = correct = stop_correct = stop_total = count = 0
        kind_correct = slot_correct = op_correct = arg_correct = 0
        edit_total = op_total = arg_total = 0
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            outputs = policy(
                batch["prompt_features"],
                batch["ops"],
                batch["args"],
                batch["trace_top"],
                batch["trace_depth"],
                batch["trace_mask"],
                batch["valid"],
                batch["final"],
                batch["step"],
                batch["prompt_hidden"],
                batch["prompt_mask"],
            )
            kind_target, slot_target, op_target, arg_target = action_components_tensor(batch["action"])
            kind_loss_items = F.cross_entropy(outputs["kind_logits"], kind_target, reduction="none")
            weights = torch.ones_like(kind_loss_items)
            weights = torch.where(kind_target.eq(0), weights * float(stop_loss_weight), weights)
            kind_loss = (kind_loss_items * weights).mean()
            edit_mask = kind_target.ne(0)
            op_mask = kind_target.eq(1)
            arg_mask = kind_target.eq(2)
            slot_loss = F.cross_entropy(outputs["slot_logits"][edit_mask], slot_target[edit_mask]) if bool(edit_mask.any()) else kind_loss * 0.0
            op_loss = F.cross_entropy(outputs["op_logits"][op_mask], op_target[op_mask]) if bool(op_mask.any()) else kind_loss * 0.0
            arg_loss = F.cross_entropy(outputs["arg_logits"][arg_mask], arg_target[arg_mask]) if bool(arg_mask.any()) else kind_loss * 0.0
            loss = kind_loss + slot_loss + op_loss + arg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            pred = compose_predicted_actions(outputs)
            pred_kind, pred_slot, pred_op, pred_arg = action_components_tensor(pred)
            n = int(pred.numel())
            total += float(loss.detach().cpu()) * n
            correct += int(pred.eq(batch["action"]).sum().item())
            kind_correct += int(pred_kind.eq(kind_target).sum().item())
            edit_total += int(edit_mask.sum().item())
            if bool(edit_mask.any()):
                slot_correct += int(pred_slot[edit_mask].eq(slot_target[edit_mask]).sum().item())
            op_total += int(op_mask.sum().item())
            if bool(op_mask.any()):
                op_correct += int(pred_op[op_mask].eq(op_target[op_mask]).sum().item())
            arg_total += int(arg_mask.sum().item())
            if bool(arg_mask.any()):
                arg_correct += int(pred_arg[arg_mask].eq(arg_target[arg_mask]).sum().item())
            stop_mask = batch["action"].eq(STOP_ID)
            stop_total += int(stop_mask.sum().item())
            if bool(stop_mask.any()):
                stop_correct += int(pred[stop_mask].eq(STOP_ID).sum().item())
            count += n
        row: Dict[str, Any] = {
            "phase": phase,
            "epoch": epoch,
            "loss": total / max(1, count),
            "action_accuracy": correct / max(1, count),
            "kind_accuracy": kind_correct / max(1, count),
            "slot_accuracy": slot_correct / max(1, edit_total),
            "op_accuracy": op_correct / max(1, op_total),
            "arg_accuracy": arg_correct / max(1, arg_total),
            "stop_accuracy": stop_correct / max(1, stop_total),
            "train_states": len(states),
            "stop_labels": sum(1 for s in states if s.action == STOP_ID),
        }
        if quick_eval is not None:
            compiler, val_set, k_values, search_topk, max_two_arg_pairs, limit_splits = quick_eval
            # Quick validation uses only the largest requested K and a small prefix.
            k = max(k_values)
            bases, base_logits = base_programs(compiler, val_set, device)
            quick = evaluate_repair_policy(
                compiler,
                policy,
                val_set,
                bases,
                base_logits,
                device,
                "quick",
                phase,
                "quick",
                [k],
                search_topk,
                max_two_arg_pairs,
                stop_margin=quick_stop_margin,
                typed_action_mask=quick_typed_action_mask,
                limit=min(int(limit_splits[0]), len(val_set)) if limit_splits else min(96, len(val_set)),
            )
            learned_rows = [r for r in quick if r.mode == "learned_stop" and r.k == k]
            quick_row = learned_rows[0] if learned_rows else quick[0]
            row["quick_val_accuracy"] = quick_row.accuracy
            row["quick_val_mean_steps"] = quick_row.mean_steps
        append_csv(log_path, row, rewrite=not log_path.exists() and epoch == 1)
        print(f"[train:{phase}] epoch={epoch} loss={row['loss']:.4f} acc={100.0 * row['action_accuracy']:.1f}%", flush=True)


def run_oracle_steps(program: BytecodeProgram, ex: TaskExample, max_k: int, stop_on_answer: bool) -> Tuple[BytecodeProgram, bool, int]:
    current = normalize_program(program)
    stopped = False
    steps = 0
    for step in range(max_k):
        action = oracle_next_action(current, ex.program, ex.answer, stop_on_answer)
        if action == STOP_ID:
            stopped = True
            break
        current, _ = apply_action(current, action)
        steps += 1
    return current, stopped, steps


@torch.no_grad()
def run_policy_steps(
    policy: RecurrentRepairPolicy,
    dataset: FeatureSet,
    ex_idx: int,
    program: BytecodeProgram,
    max_k: int,
    device: torch.device,
    learned_stop: bool,
    stop_margin: float,
    typed_action_mask: bool,
) -> Tuple[BytecodeProgram, bool, int]:
    current = normalize_program(program)
    stopped = False
    steps = 0
    for step in range(max_k):
        action = predict_action(
            policy,
            dataset,
            ex_idx,
            current,
            step,
            device,
            forbid_stop=not learned_stop,
            stop_margin=stop_margin if learned_stop else 0.0,
            typed_action_mask=typed_action_mask,
        )
        if action == STOP_ID:
            stopped = True
            break
        current, _ = apply_action(current, action)
        steps += 1
    return current, stopped, steps


def evaluate_repair_policy(
    compiler: QwenCompiler,
    policy: Optional[RecurrentRepairPolicy],
    dataset: FeatureSet,
    bases: Sequence[BytecodeProgram],
    base_logits: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    run: str,
    phase: str,
    split: str,
    k_values: Sequence[int],
    search_topk: int,
    max_two_arg_pairs: int,
    stop_margin: float = 0.0,
    typed_action_mask: bool = False,
    limit: Optional[int] = None,
) -> List[EvalResult]:
    n_items = len(dataset) if limit is None else min(limit, len(dataset))
    base_ok = base_search = base_oracle = 0
    for idx in range(n_items):
        ex = dataset.examples[idx]
        base = bases[idx]
        valid, final, _ = execute_program(base)
        base_ok += int(valid and final == ex.answer)
        op_logits, arg_logits = base_logits[idx]
        candidates = generate_candidates(base, op_logits, arg_logits, topk=search_topk, max_two_arg_pairs=max_two_arg_pairs)
        chosen, found, _ = choose_answer_verified_candidate(candidates, ex.answer, op_logits, arg_logits)
        base_search += int(chosen is not None)
        base_oracle += int(found > 0)
    out: List[EvalResult] = []
    for mode in ["base", "oracle_teacher"] + ([] if policy is None else ["learned_stop", "learned_forced"]):
        for k in k_values:
            if mode == "base" and k != 0:
                continue
            correct = valid_count = exact = stop_count = false_stop = total_steps = 0
            for idx in range(n_items):
                ex = dataset.examples[idx]
                base = bases[idx]
                if mode == "base":
                    final_prog, stopped, steps = base, False, 0
                elif mode == "oracle_teacher":
                    final_prog, stopped, steps = run_oracle_steps(base, ex, k, stop_on_answer=True)
                elif mode == "learned_stop":
                    assert policy is not None
                    final_prog, stopped, steps = run_policy_steps(
                        policy,
                        dataset,
                        idx,
                        base,
                        k,
                        device,
                        learned_stop=True,
                        stop_margin=stop_margin,
                        typed_action_mask=typed_action_mask,
                    )
                else:
                    assert policy is not None
                    final_prog, stopped, steps = run_policy_steps(
                        policy,
                        dataset,
                        idx,
                        base,
                        k,
                        device,
                        learned_stop=False,
                        stop_margin=stop_margin,
                        typed_action_mask=typed_action_mask,
                    )
                valid, final, _ = execute_program(final_prog)
                ok = bool(valid and final == ex.answer)
                correct += int(ok)
                valid_count += int(valid)
                exact += int(program_equal(final_prog, ex.program))
                stop_count += int(stopped)
                false_stop += int(stopped and not ok)
                total_steps += int(steps)
            n = max(1, n_items)
            out.append(EvalResult(
                run=run,
                phase=phase,
                split=split,
                mode=mode,
                k=int(k),
                n=n_items,
                accuracy=correct / n,
                valid_rate=valid_count / n,
                program_exact=exact / n,
                stop_rate=stop_count / n,
                false_stop_rate=false_stop / n,
                mean_steps=total_steps / n,
                base_accuracy=base_ok / n,
                base_search_accuracy=base_search / n,
                base_oracle_accuracy=base_oracle / n,
            ))
    return out


def eval_all_splits(
    compiler: QwenCompiler,
    policy: Optional[RecurrentRepairPolicy],
    feature_cache: Dict[str, FeatureSet],
    base_cache: Dict[str, Tuple[List[BytecodeProgram], List[Tuple[torch.Tensor, torch.Tensor]]]],
    device: torch.device,
    run: str,
    phase: str,
    split_names: Sequence[str],
    k_values: Sequence[int],
    search_topk: int,
    max_two_arg_pairs: int,
    stop_margin: float,
    typed_action_mask: bool,
) -> List[EvalResult]:
    rows: List[EvalResult] = []
    for split in split_names:
        bases, logits = base_cache[split]
        rows.extend(evaluate_repair_policy(
            compiler,
            policy,
            feature_cache[split],
            bases,
            logits,
            device,
            run,
            phase,
            split,
            k_values,
            search_topk,
            max_two_arg_pairs,
            stop_margin=stop_margin,
            typed_action_mask=typed_action_mask,
        ))
    return rows


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def rebuild_gold_set_from_cache(seed_set: FeatureSet, source_set: FeatureSet) -> FeatureSet:
    examples = list(seed_set.examples) + list(source_set.examples)
    return FeatureSet(examples, torch.cat([seed_set.hidden, source_set.hidden], dim=0), torch.cat([seed_set.mask, source_set.mask], dim=0))


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    tokenizer, qwen, hidden_size = load_qwen(args.model_name)
    splits = make_splits(args)
    with (run_dir / "dataset_manifest.json").open("w") as f:
        json.dump(
            {
                "run": args.run_name,
                "model_name": args.model_name,
                "seed": args.seed,
                "sizes": {k: len(v) for k, v in splits.items()},
                "hidden_size": hidden_size,
                "backend": "frozen_qwen_hidden_states_recurrent_vm_repair_policy",
                "action_size": ACTION_SIZE,
                "k_values": args.k_values,
                "stop_margin": args.stop_margin,
                "typed_action_mask": bool(args.typed_action_mask),
                "dagger_rounds": args.dagger_rounds,
            },
            f,
            indent=2,
        )

    feature_cache: Dict[str, FeatureSet] = {}
    for name, examples in splits.items():
        print(f"[features] split={name} n={len(examples)}", flush=True)
        hidden, mask = extract_features(examples, tokenizer, qwen, args.qwen_batch_size, device)
        feature_cache[name] = FeatureSet(examples, hidden, mask)
    del qwen
    torch.cuda.empty_cache()

    train_log = run_dir / "train_log.csv"
    traj_log = run_dir / "trajectory_stats.csv"
    metrics_path = run_dir / "metrics.csv"
    compiler_metrics_path = run_dir / "compiler_metrics.csv"
    for path in [train_log, traj_log, metrics_path, compiler_metrics_path]:
        if path.exists():
            path.unlink()

    set_seed(args.seed + 1000)
    compiler = QwenCompiler(hidden_size, args.d_model, args.compiler_layers, args.heads, args.dropout).to(device)
    train_compiler(
        compiler,
        feature_cache["seed_train"],
        device,
        args.seed_epochs,
        args.batch_size,
        args.lr,
        args.answer_weight,
        train_log,
        "seed_compiler",
        feature_cache["val_mixed"],
        args.search_topk,
        args.max_two_arg_pairs,
    )
    save_checkpoint(CHECKPOINT_ROOT / args.run_name / "seed_compiler" / "compiler.pt", {"model": compiler.state_dict(), "args": vars(args)})

    eval_splits = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    base_cache: Dict[str, Tuple[List[BytecodeProgram], List[Tuple[torch.Tensor, torch.Tensor]]]] = {}
    for split in ["unlabeled_train", *eval_splits]:
        print(f"[base] split={split}", flush=True)
        base_cache[split] = base_programs(compiler, feature_cache[split], device)

    compiler_rows = []
    for split in eval_splits:
        row = evaluate_compiler(compiler, feature_cache[split], device, args.run_name, "seed_compiler", args.search_topk, args.max_two_arg_pairs)
        row["split"] = split
        compiler_rows.append(row)
    write_rows(compiler_metrics_path, compiler_rows)

    teacher_states, teacher_stats = collect_teacher_states(
        feature_cache["unlabeled_train"],
        base_cache["unlabeled_train"][0],
        args.teacher_max_steps,
        stop_on_answer=True,
        phase="teacher_trajectory",
    )
    append_csv(traj_log, asdict(teacher_stats), rewrite=True)
    print(f"[trajectory] teacher states={len(teacher_states)} success={teacher_stats.teacher_success_rate:.3f}", flush=True)

    set_seed(args.seed + 2000)
    policy = RecurrentRepairPolicy(hidden_size, args.policy_d_model, args.policy_layers, args.heads, args.dropout).to(device)
    train_policy(
        policy,
        teacher_states,
        feature_cache["unlabeled_train"],
        device,
        args.policy_epochs,
        args.policy_batch_size,
        args.policy_lr,
        args.stop_loss_weight,
        train_log,
        "teacher_policy",
        quick_eval=(compiler, feature_cache["val_mixed"], args.k_values, args.search_topk, args.max_two_arg_pairs, [96]),
        quick_stop_margin=args.stop_margin,
        quick_typed_action_mask=bool(args.typed_action_mask),
    )
    save_checkpoint(CHECKPOINT_ROOT / args.run_name / "teacher_policy" / "policy.pt", {"model": policy.state_dict(), "args": vars(args)})

    results: List[EvalResult] = []
    results.extend(eval_all_splits(
        compiler,
        policy,
        feature_cache,
        base_cache,
        device,
        args.run_name,
        "teacher_policy",
        eval_splits,
        args.k_values,
        args.search_topk,
        args.max_two_arg_pairs,
        args.stop_margin,
        bool(args.typed_action_mask),
    ))

    all_dagger_states: List[RepairState] = []
    for dagger_round in range(1, args.dagger_rounds + 1):
        traj_phase = "dagger_trajectory" if args.dagger_rounds == 1 else f"dagger_trajectory_r{dagger_round}"
        policy_phase = "dagger_policy" if args.dagger_rounds == 1 else f"dagger_policy_r{dagger_round}"
        dagger_states, dagger_stats = collect_dagger_states(
            policy,
            feature_cache["unlabeled_train"],
            base_cache["unlabeled_train"][0],
            device,
            args.dagger_rollout_steps,
            stop_on_answer=True,
            phase=traj_phase,
            stop_margin=args.stop_margin,
            typed_action_mask=bool(args.typed_action_mask),
        )
        append_csv(traj_log, asdict(dagger_stats))
        print(
            f"[trajectory] {traj_phase} states={len(dagger_states)} rollout_success={dagger_stats.teacher_success_rate:.3f}",
            flush=True,
        )
        all_dagger_states.extend(dagger_states)
        combined = list(teacher_states) + list(all_dagger_states)
        train_policy(
            policy,
            combined,
            feature_cache["unlabeled_train"],
            device,
            args.dagger_epochs,
            args.policy_batch_size,
            args.dagger_lr,
            args.stop_loss_weight,
            train_log,
            policy_phase,
            quick_eval=(compiler, feature_cache["val_mixed"], args.k_values, args.search_topk, args.max_two_arg_pairs, [96]),
            quick_stop_margin=args.stop_margin,
            quick_typed_action_mask=bool(args.typed_action_mask),
        )
        save_checkpoint(CHECKPOINT_ROOT / args.run_name / policy_phase / "policy.pt", {"model": policy.state_dict(), "args": vars(args)})
        results.extend(eval_all_splits(
            compiler,
            policy,
            feature_cache,
            base_cache,
            device,
            args.run_name,
            policy_phase,
            eval_splits,
            args.k_values,
            args.search_topk,
            args.max_two_arg_pairs,
            args.stop_margin,
            bool(args.typed_action_mask),
        ))

    if args.full_supervised_epochs > 0:
        full = QwenCompiler(hidden_size, args.d_model, args.compiler_layers, args.heads, args.dropout).to(device)
        full_set = rebuild_gold_set_from_cache(feature_cache["seed_train"], feature_cache["unlabeled_train"])
        train_compiler(
            full,
            full_set,
            device,
            args.full_supervised_epochs,
            args.batch_size,
            args.lr,
            args.answer_weight,
            train_log,
            "full_supervised_compiler",
            feature_cache["val_mixed"],
            args.search_topk,
            args.max_two_arg_pairs,
        )
        save_checkpoint(CHECKPOINT_ROOT / args.run_name / "full_supervised_compiler" / "compiler.pt", {"model": full.state_dict(), "args": vars(args)})
        for split in eval_splits:
            row = evaluate_compiler(full, feature_cache[split], device, args.run_name, "full_supervised_compiler", args.search_topk, args.max_two_arg_pairs)
            row["split"] = split
            append_csv(compiler_metrics_path, row, rewrite=False)

    write_eval_results(metrics_path, results)
    with (run_dir / "results.json").open("w") as f:
        json.dump({"run": args.run_name, "args": vars(args), "results": [asdict(r) for r in results]}, f, indent=2)
    manifest = ROOT / "checkpoint_manifest.csv"
    exists = manifest.exists()
    with manifest.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "checkpoint_dir", "created_unix", "notes"])
        if not exists:
            writer.writeheader()
        writer.writerow({"run": args.run_name, "checkpoint_dir": str(CHECKPOINT_ROOT / args.run_name), "created_unix": int(time.time()), "notes": f"recurrent VM repair policy; base={args.model_name}"})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_name", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--seed", type=int, default=173)
    p.add_argument("--device", default="")
    p.add_argument("--seed_train_size", type=int, default=192)
    p.add_argument("--unlabeled_train_size", type=int, default=1024)
    p.add_argument("--full_supervised_size", type=int, default=1024)
    p.add_argument("--val_size", type=int, default=128)
    p.add_argument("--fresh_size", type=int, default=128)
    p.add_argument("--hard_size", type=int, default=128)
    p.add_argument("--max_arith_steps", type=int, default=4)
    p.add_argument("--qwen_batch_size", type=int, default=16)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--policy_d_model", type=int, default=256)
    p.add_argument("--compiler_layers", type=int, default=3)
    p.add_argument("--policy_layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--policy_batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--policy_lr", type=float, default=4e-4)
    p.add_argument("--dagger_lr", type=float, default=2e-4)
    p.add_argument("--stop_loss_weight", type=float, default=4.0)
    p.add_argument("--stop_margin", type=float, default=0.0)
    p.add_argument("--typed_action_mask", type=int, default=0)
    p.add_argument("--answer_weight", type=float, default=0.2)
    p.add_argument("--seed_epochs", type=int, default=12)
    p.add_argument("--policy_epochs", type=int, default=8)
    p.add_argument("--dagger_epochs", type=int, default=6)
    p.add_argument("--dagger_rounds", type=int, default=1)
    p.add_argument("--full_supervised_epochs", type=int, default=18)
    p.add_argument("--teacher_max_steps", type=int, default=24)
    p.add_argument("--dagger_rollout_steps", type=int, default=8)
    p.add_argument("--search_topk", type=int, default=3)
    p.add_argument("--max_two_arg_pairs", type=int, default=8)
    p.add_argument("--k_values", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    return p


def main() -> None:
    run_experiment(parser().parse_args())


if __name__ == "__main__":
    main()
