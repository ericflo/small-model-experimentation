#!/usr/bin/env python3
"""Qwen VM-agent ECHO experiment.

The model is trained as a recurrent text agent over a typed VM. A seed compiler
produces an initial bytecode program from frozen Qwen hidden states. Qwen with
QLoRA adapters then receives a transcript:

    prompt -> current program/VM observation -> Action -> Observation -> ...

At each turn the model emits exactly one edit action or STOP. The VM executes
that action and appends a textual observation. The action-only control trains
loss only on action tokens. The ECHO treatment also trains on VM observation
tokens with a small weight, so the model learns consequences of its own action
language.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
)


STOP_ID = 0
OP_OFFSET = 1
ARG_OFFSET = OP_OFFSET + MAX_PROGRAM_LEN * len(OPCODES)
ACTION_SIZE = ARG_OFFSET + MAX_PROGRAM_LEN * MODULUS


def blank_program() -> BytecodeProgram:
    ops = [OP_TO_ID["PUSH"], OP_TO_ID["END"]] + [OP_TO_ID["PAD"]] * (MAX_PROGRAM_LEN - 2)
    args = [0, NO_ARG] + [NO_ARG] * (MAX_PROGRAM_LEN - 2)
    return normalize_program(BytecodeProgram(ops, args))


@dataclass
class EvalRow:
    run: str
    variant: str
    split: str
    k: int
    n: int
    accuracy: float
    valid_rate: float
    program_exact: float
    parse_rate: float
    stop_rate: float
    false_stop_rate: float
    mean_steps: float
    base_accuracy: float
    base_search_accuracy: float
    oracle_accuracy: float


@dataclass
class TrainRow:
    run: str
    variant: str
    epoch: int
    loss: float
    action_ce: float
    observation_ce: float
    train_examples: int
    action_tokens: int
    observation_tokens: int


@dataclass
class TraceSnapshot:
    program: BytecodeProgram
    stopped: bool
    steps: int
    parsed: int
    attempted: int
    actions: List[str]


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token


def load_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    return tokenizer


def load_base_qwen(model_name: str) -> Tuple[Any, Any, int]:
    tokenizer = load_tokenizer(model_name)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=quant, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hidden_size = int(model.config.hidden_size)
    return tokenizer, model, hidden_size


class FeatureSet(Dataset):
    def __init__(self, examples: Sequence[TaskExample], hidden: torch.Tensor, mask: torch.Tensor) -> None:
        self.examples = list(examples)
        self.hidden = hidden
        self.mask = mask

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prog = normalize_program(ex.program)
        return {
            "hidden": self.hidden[idx],
            "attention_mask": self.mask[idx],
            "ops": torch.tensor(prog.ops, dtype=torch.long),
            "args": torch.tensor(prog.args, dtype=torch.long),
            "answer": torch.tensor(ex.answer, dtype=torch.long),
        }


@torch.no_grad()
def extract_features(examples: Sequence[TaskExample], tokenizer: Any, qwen: Any, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    all_hidden: List[torch.Tensor] = []
    all_mask: List[torch.Tensor] = []
    prompts = [ex.prompt for ex in examples]
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
        queries = self.slot_queries.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        decoded = self.decoder(queries, memory, memory_key_padding_mask=key_padding_mask)
        pooled = (memory * attention_mask.unsqueeze(-1).float()).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return {
            "op_logits": self.op_head(decoded),
            "arg_logits": self.arg_head(decoded),
            "answer_logits": self.answer_head(pooled),
        }


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
    run: str,
) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            outputs = model(batch["hidden"], batch["attention_mask"])
            loss = compiler_loss(outputs, batch, answer_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = int(batch["hidden"].shape[0])
            total += float(loss.detach().cpu()) * n
            count += n
        append_csv(log_path, {"run": run, "phase": "seed_compiler", "epoch": epoch, "loss": total / max(1, count), "train_examples": len(dataset)}, rewrite=not log_path.exists() and epoch == 1)
        print(f"[seed_compiler] epoch={epoch} loss={total / max(1, count):.4f}", flush=True)


@torch.no_grad()
def base_programs(model: QwenCompiler, dataset: FeatureSet, device: torch.device) -> Tuple[List[BytecodeProgram], List[Tuple[torch.Tensor, torch.Tensor]]]:
    model.eval()
    programs: List[BytecodeProgram] = []
    logits: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        outputs = model(item["hidden"].unsqueeze(0).to(device), item["attention_mask"].unsqueeze(0).to(device))
        op_logits = outputs["op_logits"][0].detach().cpu()
        arg_logits = outputs["arg_logits"][0].detach().cpu()
        programs.append(program_from_logits(op_logits, arg_logits))
        logits.append((op_logits, arg_logits))
    return programs, logits


def evaluate_compiler(programs: Sequence[BytecodeProgram], logits: Sequence[Tuple[torch.Tensor, torch.Tensor]], dataset: FeatureSet, search_topk: int, max_two_arg_pairs: int) -> Dict[str, float]:
    direct = search = oracle = valid_count = exact = 0
    total_candidates = valid_candidates = 0
    for base, pair, ex in zip(programs, logits, dataset.examples):
        valid, final, _ = execute_program(base)
        direct += int(valid and final == ex.answer)
        valid_count += int(valid)
        exact += int(program_equal(base, ex.program))
        op_logits, arg_logits = pair
        candidates = generate_candidates(base, op_logits, arg_logits, topk=search_topk, max_two_arg_pairs=max_two_arg_pairs)
        total_candidates += len(candidates)
        chosen, found, valid_cands = choose_answer_verified_candidate(candidates, ex.answer, op_logits, arg_logits)
        valid_candidates += valid_cands
        search += int(chosen is not None)
        oracle += int(found > 0)
    n = max(1, len(dataset))
    return {
        "direct_accuracy": direct / n,
        "valid_rate": valid_count / n,
        "program_exact": exact / n,
        "search_accuracy": search / n,
        "oracle_accuracy": oracle / n,
        "candidate_valid_rate": valid_candidates / max(1, total_candidates),
        "mean_candidates": total_candidates / n,
    }


def evaluate_initial_programs(programs: Sequence[BytecodeProgram], examples: Sequence[TaskExample]) -> Dict[str, float]:
    direct = valid_count = exact = 0
    for base, ex in zip(programs, examples):
        valid, final, _ = execute_program(base)
        direct += int(valid and final == ex.answer)
        valid_count += int(valid)
        exact += int(program_equal(base, ex.program))
    n = max(1, len(examples))
    return {
        "direct_accuracy": direct / n,
        "valid_rate": valid_count / n,
        "program_exact": exact / n,
        "search_accuracy": 0.0,
        "oracle_accuracy": 0.0,
        "candidate_valid_rate": 0.0,
        "mean_candidates": 0.0,
    }


def action_op(slot: int, op: int) -> int:
    return OP_OFFSET + int(slot) * len(OPCODES) + int(op)


def action_arg(slot: int, arg: int) -> int:
    return ARG_OFFSET + int(slot) * MODULUS + int(arg % MODULUS)


def decode_action(action: int) -> Tuple[str, int, int]:
    if int(action) == STOP_ID:
        return "STOP", -1, 0
    if OP_OFFSET <= int(action) < ARG_OFFSET:
        x = int(action) - OP_OFFSET
        return "OP", x // len(OPCODES), x % len(OPCODES)
    x = int(action) - ARG_OFFSET
    return "ARG", x // MODULUS, x % MODULUS


def action_to_text(action: int) -> str:
    kind, slot, value = decode_action(action)
    if kind == "STOP":
        return "STOP"
    if kind == "OP":
        return f"OP {slot} {OPCODES[value]}"
    return f"ARG {slot} {value}"


ACTION_RE = re.compile(r"^\s*(STOP|OP|ARG)\b\s*(.*)$", re.IGNORECASE)


def parse_action(text: str) -> Optional[int]:
    line = text.strip().splitlines()[0] if text.strip() else ""
    line = line.replace("`", "").replace(":", " ").strip()
    m = ACTION_RE.match(line)
    if not m:
        return None
    kind = m.group(1).upper()
    rest = m.group(2).strip().split()
    if kind == "STOP":
        return STOP_ID
    if kind == "OP" and len(rest) >= 2:
        try:
            slot = int(rest[0])
        except ValueError:
            return None
        op = rest[1].upper().strip(",.;")
        if 0 <= slot < MAX_PROGRAM_LEN and op in OP_TO_ID:
            return action_op(slot, OP_TO_ID[op])
    if kind == "ARG" and len(rest) >= 2:
        try:
            slot = int(rest[0])
            arg = int(rest[1].strip(",.;"))
        except ValueError:
            return None
        if 0 <= slot < MAX_PROGRAM_LEN:
            return action_arg(slot, arg)
    return None


def apply_action(program: BytecodeProgram, action: int) -> Tuple[BytecodeProgram, bool]:
    prog = normalize_program(program)
    kind, slot, value = decode_action(action)
    if kind == "STOP":
        return prog, True
    ops = list(prog.ops)
    args = list(prog.args)
    if not (0 <= slot < MAX_PROGRAM_LEN):
        return prog, False
    if kind == "OP":
        ops[slot] = int(value)
        if int(value) != OP_TO_ID["PUSH"]:
            args[slot] = NO_ARG
    else:
        args[slot] = int(value) % MODULUS
    return normalize_program(BytecodeProgram(ops, args)), False


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


def program_text(program: BytecodeProgram) -> str:
    prog = normalize_program(program)
    parts = []
    for i, (op, arg) in enumerate(zip(prog.ops, prog.args)):
        name = OPCODES[int(op)]
        if name == "PUSH":
            parts.append(f"{i}:{name}({int(arg) % MODULUS})")
        else:
            parts.append(f"{i}:{name}")
    return "; ".join(parts)


def observation_text(program: BytecodeProgram, halted: bool = False) -> str:
    valid, final, trace = execute_program(normalize_program(program))
    tops: List[str] = []
    depths: List[str] = []
    for stack in trace[:MAX_PROGRAM_LEN]:
        tops.append(str(int(stack[-1] % MODULUS)) if stack else "_")
        depths.append(str(min(len(stack), MAX_PROGRAM_LEN)))
    if not tops:
        tops = ["_"]
        depths = ["0"]
    return (
        f"halted={1 if halted else 0}; valid={int(valid)}; final={int(final) if valid else 0}; "
        f"program={program_text(program)}; top={','.join(tops)}; depth={','.join(depths)}\n"
    )


def transcript_prefix(example: TaskExample, base: BytecodeProgram) -> List[Tuple[str, str]]:
    header = (
        "You are a bytecode VM repair agent. Emit exactly one action line.\n"
        "Allowed actions: OP <slot> <opcode>, ARG <slot> <0-96>, STOP.\n"
        "Use STOP only when the current VM state is a valid solution for the task; otherwise edit one slot.\n"
        f"Opcodes: {', '.join(OPCODES)}.\n"
        f"Task: {example.prompt}\n"
        "Initial observation:\n"
    )
    # The initial VM state is context. ECHO supervision is reserved for
    # observations caused by the agent's emitted actions.
    return [("context", header), ("context", observation_text(base))]


def build_segments(example: TaskExample, base: BytecodeProgram, max_steps: int, stop_on_answer: bool = True) -> Tuple[List[Tuple[str, str]], int]:
    segments = transcript_prefix(example, base)
    current = normalize_program(base)
    steps = 0
    for step in range(max_steps + 1):
        action = oracle_next_action(current, example.program, example.answer, stop_on_answer)
        segments.append(("context", f"Turn {step}\nAction: "))
        segments.append(("action", action_to_text(action) + "\n"))
        if action == STOP_ID:
            segments.append(("context", "Observation: "))
            segments.append(("observation", observation_text(current, halted=True)))
            break
        current, _ = apply_action(current, action)
        steps += 1
        segments.append(("context", "Observation: "))
        segments.append(("observation", observation_text(current)))
    return segments, steps


class TranscriptDataset(Dataset):
    def __init__(
        self,
        tokenizer: Any,
        examples: Sequence[TaskExample],
        bases: Sequence[BytecodeProgram],
        max_steps: int,
        max_seq_len: int,
        variant: str,
        echo_weight: float,
        stop_on_answer: bool,
    ) -> None:
        self.items: List[Dict[str, torch.Tensor]] = []
        self.variant = variant
        action_tokens = 0
        obs_tokens = 0
        for ex, base in zip(examples, bases):
            segments, _ = build_segments(ex, base, max_steps, stop_on_answer=stop_on_answer)
            ids: List[int] = []
            labels: List[int] = []
            weights: List[float] = []
            for kind, text in segments:
                toks = tokenizer.encode(text, add_special_tokens=False)
                if not toks:
                    continue
                train_action = kind == "action"
                train_obs = variant == "echo" and kind == "observation"
                weight = 1.0 if train_action else (float(echo_weight) if train_obs else 0.0)
                ids.extend(toks)
                labels.extend(toks if weight > 0 else [-100] * len(toks))
                weights.extend([weight] * len(toks))
                action_tokens += len(toks) if train_action else 0
                obs_tokens += len(toks) if kind == "observation" else 0
            if len(ids) > max_seq_len:
                ids = ids[-max_seq_len:]
                labels = labels[-max_seq_len:]
                weights = weights[-max_seq_len:]
            self.items.append(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "loss_weights": torch.tensor(weights, dtype=torch.float32),
                }
            )
        self.action_tokens = action_tokens
        self.observation_tokens = obs_tokens

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.items[idx]


def collate_transcripts(batch: Sequence[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(int(item["input_ids"].numel()) for item in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": [], "loss_weights": []}
    for item in batch:
        n = int(item["input_ids"].numel())
        pad = max_len - n
        out["input_ids"].append(F.pad(item["input_ids"], (0, pad), value=pad_id))
        out["attention_mask"].append(F.pad(torch.ones(n, dtype=torch.long), (0, pad), value=0))
        out["labels"].append(F.pad(item["labels"], (0, pad), value=-100))
        out["loss_weights"].append(F.pad(item["loss_weights"], (0, pad), value=0.0))
    return {k: torch.stack(v, dim=0) for k, v in out.items()}


def weighted_lm_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_weights = weights[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), reduction="none", ignore_index=-100)
    loss = loss.view_as(shift_labels) * shift_weights
    return loss.sum() / shift_weights.sum().clamp_min(1.0)


@torch.no_grad()
def eval_masked_ce(model: Any, dataset: TranscriptDataset, tokenizer: Any, device: torch.device, batch_size: int) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda xs: collate_transcripts(xs, tokenizer.pad_token_id))
    total = 0.0
    weight = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        shift_weights = batch["loss_weights"][:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), reduction="none", ignore_index=-100)
        loss = loss.view_as(shift_labels) * shift_weights
        total += float(loss.sum().detach().cpu())
        weight += float(shift_weights.sum().detach().cpu())
    return total / max(1.0, weight)


def load_lora_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float) -> Tuple[Any, Any]:
    tokenizer = load_tokenizer(model_name)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=quant, device_map="auto")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return tokenizer, model


def train_lora_variant(
    args: argparse.Namespace,
    run_dir: Path,
    variant: str,
    train_examples: Sequence[TaskExample],
    train_bases: Sequence[BytecodeProgram],
    val_examples: Sequence[TaskExample],
    val_bases: Sequence[BytecodeProgram],
    device: torch.device,
) -> Tuple[Any, Any, List[TrainRow]]:
    tokenizer, model = load_lora_model(args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout)
    stop_on_answer = bool(args.teacher_stop_on_answer)
    train_ds = TranscriptDataset(tokenizer, train_examples, train_bases, args.teacher_max_steps, args.max_seq_len, variant, args.echo_weight, stop_on_answer)
    action_val = TranscriptDataset(tokenizer, val_examples, val_bases, args.teacher_max_steps, args.max_seq_len, "action_only", args.echo_weight, stop_on_answer)
    obs_val = TranscriptDataset(tokenizer, val_examples, val_bases, args.teacher_max_steps, args.max_seq_len, "echo", 1.0, stop_on_answer)
    for item in obs_val.items:
        item["labels"] = torch.where(item["loss_weights"].gt(0), item["labels"], torch.full_like(item["labels"], -100))
        item["loss_weights"] = torch.where(item["loss_weights"].gt(0), torch.ones_like(item["loss_weights"]), torch.zeros_like(item["loss_weights"]))
    loader = DataLoader(train_ds, batch_size=args.lora_batch_size, shuffle=True, collate_fn=lambda xs: collate_transcripts(xs, tokenizer.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lora_lr, weight_decay=0.01)
    rows: List[TrainRow] = []
    step = 0
    accum = max(1, args.grad_accum)
    for epoch in range(1, args.lora_epochs + 1):
        model.train()
        total = 0.0
        denom = 0.0
        opt.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            loss = weighted_lm_loss(out.logits, batch["labels"], batch["loss_weights"])
            (loss / accum).backward()
            total += float(loss.detach().cpu()) * float(batch["loss_weights"][:, 1:].sum().detach().cpu())
            denom += float(batch["loss_weights"][:, 1:].sum().detach().cpu())
            if batch_idx % accum == 0 or batch_idx == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
        action_ce = eval_masked_ce(model, action_val, tokenizer, device, args.eval_lm_batch_size)
        obs_ce = eval_masked_ce(model, obs_val, tokenizer, device, args.eval_lm_batch_size)
        row = TrainRow(
            run=args.run_name,
            variant=variant,
            epoch=epoch,
            loss=total / max(1.0, denom),
            action_ce=action_ce,
            observation_ce=obs_ce,
            train_examples=len(train_ds),
            action_tokens=train_ds.action_tokens,
            observation_tokens=train_ds.observation_tokens,
        )
        rows.append(row)
        append_csv(run_dir / "lora_train_log.csv", asdict(row), rewrite=not (run_dir / "lora_train_log.csv").exists())
        print(f"[lora:{variant}] epoch={epoch} loss={row.loss:.4f} action_ce={action_ce:.4f} obs_ce={obs_ce:.4f}", flush=True)
    ckpt_dir = CHECKPOINT_ROOT / args.run_name / variant
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    return tokenizer, model, rows


def context_for_rollout(example: TaskExample, base: BytecodeProgram) -> str:
    return "".join(text for _, text in transcript_prefix(example, base))


@torch.no_grad()
def generate_action(model: Any, tokenizer: Any, context: str, device: torch.device, max_input_len: int, max_new_tokens: int) -> Tuple[Optional[int], str]:
    enc = tokenizer(context, return_tensors="pt", truncation=True, max_length=max_input_len)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0, input_ids.shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return parse_action(text), text.strip().splitlines()[0] if text.strip() else ""


@torch.no_grad()
def run_agent_steps(
    model: Any,
    tokenizer: Any,
    example: TaskExample,
    base: BytecodeProgram,
    device: torch.device,
    max_k: int,
    max_input_len: int,
    max_new_tokens: int,
) -> Tuple[BytecodeProgram, bool, int, int, int, List[str]]:
    current = normalize_program(base)
    context = context_for_rollout(example, current)
    stopped = False
    parsed = 0
    attempted = 0
    steps = 0
    actions: List[str] = []
    for step in range(max_k):
        attempted += 1
        context += f"Turn {step}\nAction: "
        action, raw = generate_action(model, tokenizer, context, device, max_input_len, max_new_tokens)
        actions.append(raw)
        if action is None:
            context += raw + "\nObservation: parse_error=1\n"
            break
        parsed += 1
        action_text = action_to_text(action)
        context += action_text + "\n"
        if action == STOP_ID:
            stopped = True
            context += "Observation: " + observation_text(current, halted=True)
            break
        current, _ = apply_action(current, action)
        steps += 1
        context += "Observation: " + observation_text(current)
    return current, stopped, steps, parsed, attempted, actions


@torch.no_grad()
def run_agent_trace(
    model: Any,
    tokenizer: Any,
    example: TaskExample,
    base: BytecodeProgram,
    device: torch.device,
    max_k: int,
    max_input_len: int,
    max_new_tokens: int,
) -> List[TraceSnapshot]:
    current = normalize_program(base)
    context = context_for_rollout(example, current)
    stopped = False
    parsed = 0
    attempted = 0
    steps = 0
    actions: List[str] = []
    snapshots = [TraceSnapshot(current, stopped, steps, parsed, attempted, list(actions))]
    for step in range(max_k):
        attempted += 1
        context += f"Turn {step}\nAction: "
        action, raw = generate_action(model, tokenizer, context, device, max_input_len, max_new_tokens)
        actions.append(raw)
        if action is None:
            context += raw + "\nObservation: parse_error=1\n"
            snapshots.append(TraceSnapshot(current, stopped, steps, parsed, attempted, list(actions)))
            break
        parsed += 1
        action_text = action_to_text(action)
        context += action_text + "\n"
        if action == STOP_ID:
            stopped = True
            context += "Observation: " + observation_text(current, halted=True)
            snapshots.append(TraceSnapshot(current, stopped, steps, parsed, attempted, list(actions)))
            break
        current, _ = apply_action(current, action)
        steps += 1
        context += "Observation: " + observation_text(current)
        snapshots.append(TraceSnapshot(current, stopped, steps, parsed, attempted, list(actions)))
    while len(snapshots) <= max_k:
        snapshots.append(TraceSnapshot(current, stopped, steps, parsed, attempted, list(actions)))
    return snapshots


def run_oracle_steps(program: BytecodeProgram, example: TaskExample, max_k: int) -> Tuple[BytecodeProgram, bool, int]:
    current = normalize_program(program)
    stopped = False
    steps = 0
    for _ in range(max_k):
        action = oracle_next_action(current, example.program, example.answer, stop_on_answer=True)
        if action == STOP_ID:
            stopped = True
            break
        current, _ = apply_action(current, action)
        steps += 1
    return current, stopped, steps


def run_oracle_trace(program: BytecodeProgram, example: TaskExample, max_k: int) -> List[TraceSnapshot]:
    current = normalize_program(program)
    stopped = False
    steps = 0
    snapshots = [TraceSnapshot(current, stopped, steps, 0, 0, [])]
    for step in range(max_k):
        action = oracle_next_action(current, example.program, example.answer, stop_on_answer=True)
        if action == STOP_ID:
            stopped = True
            snapshots.append(TraceSnapshot(current, stopped, steps, step + 1, step + 1, ["STOP"]))
            break
        current, _ = apply_action(current, action)
        steps += 1
        snapshots.append(TraceSnapshot(current, stopped, steps, step + 1, step + 1, [action_to_text(action)]))
    while len(snapshots) <= max_k:
        last = snapshots[-1]
        snapshots.append(TraceSnapshot(current, stopped, steps, last.parsed, last.attempted, list(last.actions)))
    return snapshots


def evaluate_agent(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    variant: str,
    split: str,
    examples: Sequence[TaskExample],
    bases: Sequence[BytecodeProgram],
    logits: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> List[EvalRow]:
    rows: List[EvalRow] = []
    base_ok = base_search = oracle_candidate = 0
    for ex, base, pair in zip(examples, bases, logits):
        valid, final, _ = execute_program(base)
        base_ok += int(valid and final == ex.answer)
        if pair[0].numel() > 0:
            candidates = generate_candidates(base, pair[0], pair[1], topk=args.search_topk, max_two_arg_pairs=args.max_two_arg_pairs)
            chosen, found, _ = choose_answer_verified_candidate(candidates, ex.answer, pair[0], pair[1])
            base_search += int(chosen is not None)
            oracle_candidate += int(found > 0)
    n = max(1, len(examples))
    base_acc = base_ok / n
    base_search_acc = base_search / n
    base_oracle_acc = oracle_candidate / n
    max_k = max(args.k_values) if args.k_values else 0
    agent_traces = [
        run_agent_trace(model, tokenizer, ex, base, device, max_k, args.max_input_len, args.max_new_tokens)
        for ex, base in zip(examples, bases)
    ]
    oracle_traces = [run_oracle_trace(base, ex, max_k) for ex, base in zip(examples, bases)]
    for k in args.k_values:
        for mode, traces in [("agent", agent_traces), ("oracle_teacher", oracle_traces)]:
            correct = valid_count = exact = parse_count = attempts = stop_count = false_stop = total_steps = 0
            for ex, trace in zip(examples, traces):
                snap = trace[int(k)]
                valid, final, _ = execute_program(snap.program)
                ok = bool(valid and final == ex.answer)
                correct += int(ok)
                valid_count += int(valid)
                exact += int(program_equal(snap.program, ex.program))
                parse_count += snap.parsed
                attempts += snap.attempted
                stop_count += int(snap.stopped)
                false_stop += int(snap.stopped and not ok)
                total_steps += snap.steps
            rows.append(
                EvalRow(
                    run=args.run_name,
                    variant=variant if mode == "agent" else "oracle_teacher",
                    split=split,
                    k=int(k),
                    n=len(examples),
                    accuracy=correct / n,
                    valid_rate=valid_count / n,
                    program_exact=exact / n,
                    parse_rate=(parse_count / max(1, attempts)) if k > 0 else 1.0,
                    stop_rate=stop_count / n,
                    false_stop_rate=false_stop / n,
                    mean_steps=total_steps / n,
                    base_accuracy=base_acc,
                    base_search_accuracy=base_search_acc,
                    oracle_accuracy=base_oracle_acc,
                )
            )
    return rows


@torch.no_grad()
def write_rollout_samples(
    path: Path,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    variant: str,
    split: str,
    examples: Sequence[TaskExample],
    bases: Sequence[BytecodeProgram],
    device: torch.device,
    count: int,
) -> None:
    rows: List[Dict[str, Any]] = []
    for idx, (ex, base) in enumerate(list(zip(examples, bases))[:count]):
        final_prog, stopped, steps, parsed, attempted, actions = run_agent_steps(
            model,
            tokenizer,
            ex,
            base,
            device,
            max(args.k_values),
            args.max_input_len,
            args.max_new_tokens,
        )
        valid, final, _ = execute_program(final_prog)
        rows.append(
            {
                "variant": variant,
                "split": split,
                "idx": idx,
                "prompt": ex.prompt,
                "answer": ex.answer,
                "base_program": program_text(base),
                "actions": " | ".join(actions),
                "parsed": parsed,
                "attempted": attempted,
                "stopped": int(stopped),
                "steps": steps,
                "valid": int(valid),
                "final": int(final) if valid else 0,
                "correct": int(valid and final == ex.answer),
                "final_program": program_text(final_prog),
            }
        )
    existing: List[Dict[str, Any]] = []
    if path.exists():
        with path.open(newline="") as f:
            existing = list(csv.DictReader(f))
    write_rows(path, existing + rows)


def make_splits(args: argparse.Namespace) -> Dict[str, List[TaskExample]]:
    gen = TaskGenerator(seed=args.seed, max_arith_steps=args.max_arith_steps)
    return {
        "seed_train": gen.make_set(args.seed_train_size, template="mixed", hard=False),
        "agent_train": gen.make_set(args.agent_train_size, template="mixed", hard=False),
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


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    for filename in ["compiler_metrics.csv", "lora_train_log.csv", "metrics.csv"]:
        path = run_dir / filename
        if path.exists():
            path.unlink()

    splits = make_splits(args)
    with (run_dir / "dataset_manifest.json").open("w") as f:
        json.dump({"run": args.run_name, "model_name": args.model_name, "sizes": {k: len(v) for k, v in splits.items()}, "args": vars(args)}, f, indent=2)

    eval_splits = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    base_cache: Dict[str, Tuple[List[BytecodeProgram], List[Tuple[torch.Tensor, torch.Tensor]]]] = {}
    feature_cache: Dict[str, FeatureSet] = {}
    if args.initial_program_source == "seed":
        print("[load] base qwen for feature extraction", flush=True)
        tokenizer, qwen, hidden_size = load_base_qwen(args.model_name)
        for name, examples in splits.items():
            print(f"[features] split={name} n={len(examples)}", flush=True)
            hidden, mask = extract_features(examples, tokenizer, qwen, args.qwen_batch_size, device)
            feature_cache[name] = FeatureSet(examples, hidden, mask)
        del qwen
        gc.collect()
        torch.cuda.empty_cache()

        compiler = QwenCompiler(hidden_size, args.compiler_d_model, args.compiler_layers, args.heads, args.dropout).to(device)
        train_compiler(compiler, feature_cache["seed_train"], device, args.seed_epochs, args.compiler_batch_size, args.compiler_lr, args.answer_weight, run_dir / "compiler_train_log.csv", args.run_name)
        compiler_ckpt = CHECKPOINT_ROOT / args.run_name / "seed_compiler" / "compiler.pt"
        compiler_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": compiler.state_dict(), "args": vars(args)}, compiler_ckpt)
        for split in ["agent_train", *eval_splits]:
            print(f"[base_programs] {split}", flush=True)
            base_cache[split] = base_programs(compiler, feature_cache[split], device)
        del compiler
        gc.collect()
        torch.cuda.empty_cache()
    else:
        empty_logits = (torch.empty(0), torch.empty(0))
        for split in ["agent_train", *eval_splits]:
            base_cache[split] = ([blank_program() for _ in splits[split]], [empty_logits for _ in splits[split]])
        for name, examples in splits.items():
            feature_cache[name] = FeatureSet(examples, torch.empty(len(examples), 1, 1, dtype=torch.float16), torch.ones(len(examples), 1, dtype=torch.bool))
    comp_rows: List[Dict[str, Any]] = []
    for split in eval_splits:
        if args.initial_program_source == "seed":
            row = evaluate_compiler(base_cache[split][0], base_cache[split][1], feature_cache[split], args.search_topk, args.max_two_arg_pairs)
        else:
            row = evaluate_initial_programs(base_cache[split][0], splits[split])
        row.update({"run": args.run_name, "split": split, "phase": "seed_compiler"})
        comp_rows.append(row)
    write_rows(run_dir / "compiler_metrics.csv", comp_rows)

    all_eval: List[EvalRow] = []
    variants = []
    if args.run_action_only:
        variants.append("action_only")
    if args.run_echo:
        variants.append("echo")
    for variant in variants:
        print(f"[variant] training {variant}", flush=True)
        tokenizer_lora, model, _ = train_lora_variant(
            args,
            run_dir,
            variant,
            feature_cache["agent_train"].examples,
            base_cache["agent_train"][0],
            feature_cache["val_mixed"].examples,
            base_cache["val_mixed"][0],
            device,
        )
        model.eval()
        for split in eval_splits:
            print(f"[eval] variant={variant} split={split}", flush=True)
            rows = evaluate_agent(args, model, tokenizer_lora, variant, split, feature_cache[split].examples, base_cache[split][0], base_cache[split][1], device)
            all_eval.extend(rows)
            write_rows(run_dir / "metrics.csv", [asdict(r) for r in all_eval])
            if split == "val_mixed":
                write_rollout_samples(run_dir / "rollout_samples.csv", args, model, tokenizer_lora, variant, split, feature_cache[split].examples, base_cache[split][0], device, min(3, len(feature_cache[split].examples)))
        del model, tokenizer_lora
        gc.collect()
        torch.cuda.empty_cache()

    with (run_dir / "results.json").open("w") as f:
        json.dump({"run": args.run_name, "args": vars(args), "metrics": [asdict(r) for r in all_eval]}, f, indent=2)
    with (ROOT / "checkpoint_manifest.csv").open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "checkpoint_dir", "created_unix", "notes"])
        writer.writerow({"run": args.run_name, "checkpoint_dir": str(CHECKPOINT_ROOT / args.run_name), "created_unix": int(time.time()), "notes": f"VM-agent ECHO QLoRA; base={args.model_name}"})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_name", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--seed", type=int, default=917)
    p.add_argument("--seed_train_size", type=int, default=128)
    p.add_argument("--agent_train_size", type=int, default=384)
    p.add_argument("--val_size", type=int, default=64)
    p.add_argument("--fresh_size", type=int, default=64)
    p.add_argument("--hard_size", type=int, default=64)
    p.add_argument("--max_arith_steps", type=int, default=4)
    p.add_argument("--qwen_batch_size", type=int, default=16)
    p.add_argument("--compiler_d_model", type=int, default=192)
    p.add_argument("--compiler_layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--compiler_batch_size", type=int, default=64)
    p.add_argument("--compiler_lr", type=float, default=5e-4)
    p.add_argument("--answer_weight", type=float, default=0.2)
    p.add_argument("--seed_epochs", type=int, default=8)
    p.add_argument("--initial_program_source", choices=["seed", "blank"], default="seed")
    p.add_argument("--teacher_max_steps", type=int, default=12)
    p.add_argument("--teacher_stop_on_answer", type=int, default=1)
    p.add_argument("--search_topk", type=int, default=3)
    p.add_argument("--max_two_arg_pairs", type=int, default=8)
    p.add_argument("--run_action_only", type=int, default=1)
    p.add_argument("--run_echo", type=int, default=1)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_lr", type=float, default=1e-4)
    p.add_argument("--lora_epochs", type=int, default=2)
    p.add_argument("--lora_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--eval_lm_batch_size", type=int, default=1)
    p.add_argument("--echo_weight", type=float, default=0.05)
    p.add_argument("--max_seq_len", type=int, default=1536)
    p.add_argument("--max_input_len", type=int, default=1536)
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--k_values", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    return p


def main() -> None:
    run_experiment(parser().parse_args())


if __name__ == "__main__":
    main()
