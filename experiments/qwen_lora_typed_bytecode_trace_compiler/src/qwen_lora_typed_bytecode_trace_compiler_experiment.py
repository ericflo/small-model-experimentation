#!/usr/bin/env python3
"""QLoRA Qwen typed-bytecode trace compiler experiment.

This file is intentionally self-contained. It defines a small typed stack
machine, generates natural-language tasks with executable gold bytecode,
trains a compiler head over Qwen hidden states, and optionally trains QLoRA
adapters through Qwen during bytecode trace supervision.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"transformers is required: {exc}")

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"peft is required: {exc}")


ROOT = Path("experiments/qwen_lora_typed_bytecode_trace_compiler")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_lora_typed_bytecode_trace_compiler/checkpoints")

MODULUS = 97
MAX_PROGRAM_LEN = 16
NO_ARG = 0

OPCODES = [
    "PAD",
    "PUSH",
    "ADD",
    "SUB",
    "MUL",
    "MOD",
    "MAX",
    "MIN",
    "GT",
    "EQ",
    "LOOKUP_A",
    "LOOKUP_B",
    "END",
]
OP_TO_ID = {name: i for i, name in enumerate(OPCODES)}
ID_TO_OP = {i: name for name, i in OP_TO_ID.items()}

LOOKUP_A = {0: 11, 1: 23, 2: 37, 3: 41, 4: 59, 5: 61, 6: 73, 7: 89}
LOOKUP_B = {0: 7, 1: 19, 2: 29, 3: 43, 4: 53, 5: 67, 6: 79, 7: 83}


@dataclass
class BytecodeProgram:
    ops: List[int]
    args: List[int]

    def padded(self, max_len: int = MAX_PROGRAM_LEN) -> "BytecodeProgram":
        ops = list(self.ops[:max_len])
        args = list(self.args[:max_len])
        while len(ops) < max_len:
            ops.append(OP_TO_ID["PAD"])
            args.append(NO_ARG)
        return BytecodeProgram(ops=ops, args=args)


@dataclass
class TaskExample:
    prompt: str
    domain: str
    answer: int
    program: BytecodeProgram
    template: str
    length: int


@dataclass
class EvalResult:
    run: str
    variant: str
    phase: str
    split: str
    n: int
    answer_head_accuracy: float
    bytecode_accuracy: float
    search_accuracy: float
    oracle_accuracy: float
    program_exact: float
    direct_valid_rate: float
    candidate_valid_rate: float
    mean_candidates: float
    found_rate: float
    gap_recovered: float


@dataclass
class FeatureSet:
    examples: List[TaskExample]
    hidden: torch.Tensor
    attention_mask: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.convert_ids_to_tokens(0)
    tokenizer.padding_side = "right"


def dtype_from_string(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(name)


def normalize_program(program: BytecodeProgram, max_len: int = MAX_PROGRAM_LEN) -> BytecodeProgram:
    padded = program.padded(max_len)
    ops = list(padded.ops)
    args = [int(x) % MODULUS for x in padded.args]
    seen_end = False
    for i, op in enumerate(ops):
        if seen_end:
            ops[i] = OP_TO_ID["PAD"]
            args[i] = NO_ARG
            continue
        if op != OP_TO_ID["PUSH"]:
            args[i] = NO_ARG
        if op == OP_TO_ID["END"]:
            seen_end = True
    if not seen_end:
        ops[-1] = OP_TO_ID["END"]
        args[-1] = NO_ARG
    return BytecodeProgram(ops=ops, args=args)


def make_prog(instrs: Sequence[Tuple[str, int]]) -> BytecodeProgram:
    return normalize_program(BytecodeProgram([OP_TO_ID[name] for name, _ in instrs], [int(arg) for _, arg in instrs]))


def program_equal(a: BytecodeProgram, b: BytecodeProgram) -> bool:
    aa = normalize_program(a)
    bb = normalize_program(b)
    return aa.ops == bb.ops and aa.args == bb.args


def execute_program(program: BytecodeProgram, max_len: int = MAX_PROGRAM_LEN) -> Tuple[bool, int, List[List[int]]]:
    stack: List[int] = []
    trace: List[List[int]] = []
    ended = False
    prog = normalize_program(program, max_len)
    for op, arg in zip(prog.ops[:max_len], prog.args[:max_len]):
        if ended:
            if op != OP_TO_ID["PAD"]:
                return False, 0, trace
            trace.append(list(stack))
            continue
        name = ID_TO_OP.get(int(op), "BAD")
        arg = int(arg) % MODULUS
        if name == "PAD":
            return False, 0, trace
        if name == "PUSH":
            stack.append(arg)
        elif name == "ADD":
            if len(stack) < 2:
                return False, 0, trace
            b, a = stack.pop(), stack.pop()
            stack.append((a + b) % MODULUS)
        elif name == "SUB":
            if len(stack) < 2:
                return False, 0, trace
            b, a = stack.pop(), stack.pop()
            stack.append((a - b) % MODULUS)
        elif name == "MUL":
            if len(stack) < 2:
                return False, 0, trace
            b, a = stack.pop(), stack.pop()
            stack.append((a * b) % MODULUS)
        elif name == "MOD":
            if len(stack) < 2:
                return False, 0, trace
            b, a = stack.pop(), stack.pop()
            stack.append(a % max(1, b))
        elif name == "MAX":
            if len(stack) < 2:
                return False, 0, trace
            b, a = stack.pop(), stack.pop()
            stack.append(max(a, b) % MODULUS)
        elif name == "MIN":
            if len(stack) < 2:
                return False, 0, trace
            b, a = stack.pop(), stack.pop()
            stack.append(min(a, b) % MODULUS)
        elif name == "GT":
            if len(stack) < 2:
                return False, 0, trace
            b, a = stack.pop(), stack.pop()
            stack.append(1 if a > b else 0)
        elif name == "EQ":
            if len(stack) < 2:
                return False, 0, trace
            b, a = stack.pop(), stack.pop()
            stack.append(1 if a == b else 0)
        elif name == "LOOKUP_A":
            if not stack:
                return False, 0, trace
            stack.append(LOOKUP_A[stack.pop() % 8] % MODULUS)
        elif name == "LOOKUP_B":
            if not stack:
                return False, 0, trace
            stack.append(LOOKUP_B[stack.pop() % 8] % MODULUS)
        elif name == "END":
            ended = True
            if len(stack) != 1:
                return False, 0, trace
        else:
            return False, 0, trace
        trace.append(list(stack))
    if not ended or len(stack) != 1:
        return False, 0, trace
    return True, int(stack[-1]) % MODULUS, trace


class TaskGenerator:
    def __init__(self, seed: int, max_arith_steps: int = 4) -> None:
        self.rng = random.Random(seed)
        self.max_arith_steps = max_arith_steps

    def _render(self, standard: str, paraphrases: Sequence[str], template: str) -> str:
        if template == "standard":
            return standard
        if template == "paraphrase":
            return self.rng.choice(list(paraphrases))
        if template == "mixed":
            return self.rng.choice([standard] + list(paraphrases))
        raise ValueError(template)

    def arithmetic(self, template: str, hard: bool = False) -> TaskExample:
        steps = self.rng.randint(2 if hard else 1, self.max_arith_steps + (2 if hard else 0))
        init = self.rng.randrange(MODULUS)
        instrs: List[Tuple[str, int]] = [("PUSH", init)]
        prompt_steps: List[str] = []
        x = init
        for _ in range(steps):
            name = self.rng.choice(["ADD", "SUB", "MUL"])
            if name == "MUL":
                arg = self.rng.randint(2, 12)
                x = (x * arg) % MODULUS
                prompt_steps.append(f"multiply by {arg}")
            elif name == "ADD":
                arg = self.rng.randint(1, 40)
                x = (x + arg) % MODULUS
                prompt_steps.append(f"add {arg}")
            else:
                arg = self.rng.randint(1, 40)
                x = (x - arg) % MODULUS
                prompt_steps.append(f"subtract {arg}")
            instrs.extend([("PUSH", arg), (name, NO_ARG)])
        instrs.append(("END", NO_ARG))
        standard = f"Modulo {MODULUS}. Start with x = {init}. Then " + "; then ".join(prompt_steps) + ". What is x?"
        paraphrases = [
            f"Use remainder arithmetic with base {MODULUS}. Initial value {init}; apply "
            + ", ".join(prompt_steps)
            + ". Return the final value.",
            f"Track a hidden integer mod {MODULUS}: begin at {init}, then "
            + " -> ".join(prompt_steps)
            + ". Final?",
        ]
        prog = make_prog(instrs)
        ok, answer, _ = execute_program(prog)
        assert ok and answer == x
        return TaskExample(self._render(standard, paraphrases, template), "arithmetic", answer, prog, template, steps)

    def calendar(self, template: str, hard: bool = False) -> TaskExample:
        day = self.rng.randint(0, 6)
        offset = self.rng.randint(1, 80 if hard else 30)
        instrs = [("PUSH", day), ("PUSH", offset), ("ADD", NO_ARG), ("PUSH", 7), ("MOD", NO_ARG), ("END", NO_ARG)]
        answer = (day + offset) % 7
        standard = f"Weekday numbers use 0 through 6. If today is {day}, what weekday number is {offset} days later?"
        paraphrases = [
            f"Calendar task: start on weekday {day}; advance by {offset} days; output the weekday index.",
            f"With days numbered 0..6, move {offset} days after day {day}. Which index results?",
        ]
        return TaskExample(self._render(standard, paraphrases, template), "calendar", answer, make_prog(instrs), template, 1)

    def unit(self, template: str, hard: bool = False) -> TaskExample:
        value = self.rng.randint(1, 30)
        factor = self.rng.choice([2, 3, 5, 10, 12] + ([25, 60] if hard else []))
        instrs = [("PUSH", value), ("PUSH", factor), ("MUL", NO_ARG), ("END", NO_ARG)]
        answer = (value * factor) % MODULUS
        standard = f"A conversion multiplies the input by {factor} modulo {MODULUS}. Convert {value}."
        paraphrases = [
            f"Unit conversion: take {value} and scale it by {factor}; keep the result mod {MODULUS}.",
            f"Apply the factor {factor} to quantity {value}, using modulo {MODULUS}.",
        ]
        return TaskExample(self._render(standard, paraphrases, template), "unit", answer, make_prog(instrs), template, 1)

    def list_task(self, template: str, hard: bool = False) -> TaskExample:
        n = 5 if hard else 4
        values = [self.rng.randint(0, 40) for _ in range(n)]
        mode = self.rng.choice(["sum", "max", "min"])
        instrs: List[Tuple[str, int]] = [("PUSH", values[0])]
        if mode == "sum":
            answer = sum(values) % MODULUS
            for v in values[1:]:
                instrs.extend([("PUSH", v), ("ADD", NO_ARG)])
        elif mode == "max":
            answer = max(values) % MODULUS
            for v in values[1:]:
                instrs.extend([("PUSH", v), ("MAX", NO_ARG)])
        else:
            answer = min(values) % MODULUS
            for v in values[1:]:
                instrs.extend([("PUSH", v), ("MIN", NO_ARG)])
        instrs.append(("END", NO_ARG))
        joined = ", ".join(str(v) for v in values)
        standard = f"Given the list [{joined}], compute the {mode} modulo {MODULUS}."
        paraphrases = [
            f"List operation: numbers are {joined}. Return their {mode}; use modulo {MODULUS} if needed.",
            f"For values {joined}, what is the {mode} result?",
        ]
        return TaskExample(self._render(standard, paraphrases, template), "list", answer, make_prog(instrs), template, n)

    def boolean(self, template: str, hard: bool = False) -> TaskExample:
        a = self.rng.randint(0, 60)
        b = self.rng.randint(0, 60)
        threshold = self.rng.randint(20, 90 if hard else 80)
        instrs = [("PUSH", a), ("PUSH", b), ("ADD", NO_ARG), ("PUSH", threshold), ("GT", NO_ARG), ("END", NO_ARG)]
        answer = 1 if (a + b) % MODULUS > threshold else 0
        standard = f"Return 1 if ({a} + {b}) modulo {MODULUS} is greater than {threshold}, else 0."
        paraphrases = [
            f"Boolean rule: add {a} and {b} under mod {MODULUS}; is it above {threshold}? Use 1 for yes and 0 for no.",
            f"Threshold check with a={a}, b={b}, threshold={threshold}, modulus={MODULUS}. Output 1 or 0.",
        ]
        return TaskExample(self._render(standard, paraphrases, template), "boolean", answer, make_prog(instrs), template, 1)

    def lookup(self, template: str, hard: bool = False) -> TaskExample:
        table = self.rng.choice(["A", "B"])
        key = self.rng.randint(0, 7)
        op = "LOOKUP_A" if table == "A" else "LOOKUP_B"
        answer = (LOOKUP_A if table == "A" else LOOKUP_B)[key] % MODULUS
        instrs = [("PUSH", key), (op, NO_ARG), ("END", NO_ARG)]
        table_text = (
            "A maps 0:11 1:23 2:37 3:41 4:59 5:61 6:73 7:89"
            if table == "A"
            else "B maps 0:7 1:19 2:29 3:43 4:53 5:67 6:79 7:83"
        )
        standard = f"Lookup table {table}. {table_text}. What value is stored for key {key}?"
        paraphrases = [
            f"Use table {table}: {table_text}. Retrieve key {key}.",
            f"Table lookup problem. In table {table}, find the entry for {key}. {table_text}.",
        ]
        return TaskExample(self._render(standard, paraphrases, template), "lookup", answer, make_prog(instrs), template, 1)

    def make_one(self, template: str = "mixed", hard: bool = False) -> TaskExample:
        fn = self.rng.choice([self.arithmetic, self.calendar, self.unit, self.list_task, self.boolean, self.lookup])
        return fn(template=template, hard=hard)

    def make_set(self, n: int, template: str, hard: bool = False) -> List[TaskExample]:
        return [self.make_one(template=template, hard=hard) for _ in range(n)]

    def make_paired_set(self, n: int, hard: bool = False) -> List[TaskExample]:
        out: List[TaskExample] = []
        for _ in range(n):
            state = self.rng.getstate()
            standard = self.make_one(template="standard", hard=hard)
            self.rng.setstate(state)
            paraphrase = self.make_one(template="paraphrase", hard=hard)
            out.extend([standard, paraphrase])
        return out


class QwenBytecodeHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
        max_program_len: int = MAX_PROGRAM_LEN,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(hidden_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.slot_queries = nn.Parameter(torch.randn(max_program_len, d_model) * 0.02)
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
        mask_f = attention_mask.float()
        memory = self.input_norm(self.input_proj(hidden.float()))
        key_padding_mask = ~attention_mask.bool()
        bsz = hidden.shape[0]
        queries = self.slot_queries.unsqueeze(0).expand(bsz, -1, -1)
        decoded = self.decoder(queries, memory, memory_key_padding_mask=key_padding_mask)
        pooled = (memory * mask_f.unsqueeze(-1)).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return {
            "op_logits": self.op_head(decoded),
            "arg_logits": self.arg_head(decoded),
            "answer_logits": self.answer_head(pooled),
        }


class DirectAnswerHead(nn.Module):
    def __init__(self, hidden_size: int, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, MODULUS),
        )

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_f = attention_mask.float()
        pooled = (hidden.float() * mask_f.unsqueeze(-1)).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return self.net(pooled)


def stack_delta(op: int) -> int:
    name = ID_TO_OP.get(int(op), "BAD")
    if name == "PUSH":
        return 1
    if name in {"ADD", "SUB", "MUL", "MOD", "MAX", "MIN", "GT", "EQ"}:
        return -1
    if name in {"LOOKUP_A", "LOOKUP_B"}:
        return 0
    return 0


def allowed_ops_for_depth(depth: int, slot: int, max_len: int = MAX_PROGRAM_LEN) -> List[int]:
    if slot >= max_len - 2 and depth == 1:
        return [OP_TO_ID["END"]]
    if slot == max_len - 1:
        return [OP_TO_ID["END"]] if depth == 1 else []
    allowed = [OP_TO_ID["PUSH"]]
    if depth >= 2:
        allowed.extend(OP_TO_ID[name] for name in ["ADD", "SUB", "MUL", "MOD", "MAX", "MIN", "GT", "EQ"])
    if depth >= 1:
        allowed.extend([OP_TO_ID["LOOKUP_A"], OP_TO_ID["LOOKUP_B"]])
    if depth == 1 and slot >= 2:
        allowed.append(OP_TO_ID["END"])
    return allowed


def program_from_logits(op_logits: torch.Tensor, arg_logits: torch.Tensor) -> BytecodeProgram:
    op_scores = op_logits.detach().cpu()
    args = arg_logits.argmax(dim=-1).detach().cpu().tolist()
    ops: List[int] = []
    depth = 0
    ended = False
    for slot in range(MAX_PROGRAM_LEN):
        if ended:
            ops.append(OP_TO_ID["PAD"])
            args[slot] = NO_ARG
            continue
        allowed = allowed_ops_for_depth(depth, slot, MAX_PROGRAM_LEN)
        if not allowed:
            ops.append(OP_TO_ID["END"])
            args[slot] = NO_ARG
            ended = True
            continue
        scores = op_scores[slot, allowed]
        chosen = int(allowed[int(torch.argmax(scores).item())])
        ops.append(chosen)
        if chosen != OP_TO_ID["PUSH"]:
            args[slot] = NO_ARG
        if chosen == OP_TO_ID["END"]:
            ended = True
        else:
            depth += stack_delta(chosen)
    return normalize_program(BytecodeProgram(ops=ops, args=args))


def program_logprob(program: BytecodeProgram, op_logits: torch.Tensor, arg_logits: torch.Tensor) -> float:
    prog = normalize_program(program)
    op_logp = F.log_softmax(op_logits, dim=-1).detach().cpu()
    arg_logp = F.log_softmax(arg_logits, dim=-1).detach().cpu()
    total = 0.0
    for i, (op, arg) in enumerate(zip(prog.ops, prog.args)):
        total += float(op_logp[i, int(op)])
        total += float(arg_logp[i, int(arg)])
    return total


def canonical_key(program: BytecodeProgram) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    prog = normalize_program(program)
    return tuple(prog.ops), tuple(prog.args)


def topk_indices(logits: torch.Tensor, k: int) -> List[List[int]]:
    return torch.topk(logits.detach().cpu(), k=min(k, logits.shape[-1]), dim=-1).indices.tolist()


def generate_candidates(
    base: BytecodeProgram,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    topk: int,
    max_two_arg_pairs: int,
) -> List[BytecodeProgram]:
    base = normalize_program(base)
    op_top = topk_indices(op_logits, topk)
    arg_top = topk_indices(arg_logits, topk)
    out: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], BytecodeProgram] = {canonical_key(base): base}
    for slot in range(MAX_PROGRAM_LEN):
        for new_op in op_top[slot]:
            if new_op != base.ops[slot]:
                p = BytecodeProgram(list(base.ops), list(base.args))
                p.ops[slot] = int(new_op)
                out[canonical_key(p)] = p
        for new_arg in arg_top[slot]:
            if new_arg != base.args[slot]:
                p = BytecodeProgram(list(base.ops), list(base.args))
                p.args[slot] = int(new_arg)
                out[canonical_key(p)] = p
        for new_op in op_top[slot]:
            for new_arg in arg_top[slot]:
                if new_op != base.ops[slot] or new_arg != base.args[slot]:
                    p = BytecodeProgram(list(base.ops), list(base.args))
                    p.ops[slot] = int(new_op)
                    p.args[slot] = int(new_arg)
                    out[canonical_key(p)] = p
    active_slots = list(range(min(MAX_PROGRAM_LEN, max_two_arg_pairs)))
    for i, slot_a in enumerate(active_slots):
        for slot_b in active_slots[i + 1 :]:
            for arg_a in arg_top[slot_a][1:topk]:
                for arg_b in arg_top[slot_b][1:topk]:
                    p = BytecodeProgram(list(base.ops), list(base.args))
                    p.args[slot_a] = int(arg_a)
                    p.args[slot_b] = int(arg_b)
                    out[canonical_key(p)] = p
    return list(out.values())


def choose_answer_verified_candidate(
    candidates: Sequence[BytecodeProgram],
    answer: int,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
) -> Tuple[Optional[BytecodeProgram], int, int]:
    valid_count = 0
    found = 0
    best: Optional[Tuple[float, BytecodeProgram]] = None
    for cand in candidates:
        valid, pred, _ = execute_program(cand)
        valid_count += int(valid)
        if valid and pred == int(answer):
            found += 1
            score = program_logprob(cand, op_logits, arg_logits)
            if best is None or score > best[0]:
                best = (score, cand)
    return (best[1] if best is not None else None), found, valid_count


def program_loss(
    outputs: Dict[str, torch.Tensor],
    ops: torch.Tensor,
    args: torch.Tensor,
    answers: torch.Tensor,
    op_weight: float,
    arg_weight: float,
    answer_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    op_loss = F.cross_entropy(outputs["op_logits"].reshape(-1, len(OPCODES)), ops.reshape(-1))
    arg_loss = F.cross_entropy(outputs["arg_logits"].reshape(-1, MODULUS), args.reshape(-1))
    ans_loss = F.cross_entropy(outputs["answer_logits"], answers)
    loss = op_weight * op_loss + arg_weight * arg_loss + answer_weight * ans_loss
    return loss, {
        "op_loss": float(op_loss.detach().cpu()),
        "arg_loss": float(arg_loss.detach().cpu()),
        "answer_loss": float(ans_loss.detach().cpu()),
    }


def append_csv(path: Path, row: Dict[str, Any], rewrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and not rewrite
    with path.open("a" if exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_eval_results(path: Path, rows: Sequence[EvalResult]) -> None:
    write_csv(path, [asdict(r) for r in rows])


def make_splits(args: argparse.Namespace) -> Dict[str, List[TaskExample]]:
    gen = TaskGenerator(seed=args.seed, max_arith_steps=args.max_arith_steps)
    return {
        "trace_train": gen.make_set(args.trace_train_size, template="mixed", hard=False),
        "seed_train": gen.make_set(args.seed_train_size, template="mixed", hard=False),
        "unlabeled_train": gen.make_set(args.unlabeled_train_size, template="mixed", hard=False),
        "answer_train": gen.make_set(args.answer_train_size, template="mixed", hard=False),
        "val_mixed": gen.make_set(args.val_size, template="mixed", hard=False),
        "fresh_standard": gen.make_set(args.fresh_size, template="standard", hard=False),
        "fresh_paraphrase": gen.make_set(args.fresh_size, template="paraphrase", hard=False),
        "fresh_paired": gen.make_paired_set(max(1, args.fresh_size // 2), hard=False),
        "hard_composition": gen.make_set(args.hard_size, template="mixed", hard=True),
    }


def prompt_lengths(examples: Sequence[TaskExample], tokenizer: Any) -> List[int]:
    return [len(tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]) for ex in examples]


def collate_examples(
    examples: Sequence[TaskExample],
    tokenizer: Any,
    max_prompt_len: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    prompts = [ex.prompt for ex in examples]
    enc = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_prompt_len,
        return_tensors="pt",
    )
    programs = [normalize_program(ex.program) for ex in examples]
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "ops": torch.tensor([p.ops for p in programs], dtype=torch.long, device=device),
        "args": torch.tensor([p.args for p in programs], dtype=torch.long, device=device),
        "answers": torch.tensor([ex.answer for ex in examples], dtype=torch.long, device=device),
    }


def sample_examples(examples: Sequence[TaskExample], batch_size: int) -> List[TaskExample]:
    if len(examples) >= batch_size:
        return random.sample(list(examples), batch_size)
    return [random.choice(list(examples)) for _ in range(batch_size)]


def sample_feature_batch(feature_set: FeatureSet, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    indices = [random.randrange(len(feature_set.examples)) for _ in range(batch_size)]
    programs = [normalize_program(feature_set.examples[i].program) for i in indices]
    return {
        "hidden": feature_set.hidden[indices].to(device),
        "attention_mask": feature_set.attention_mask[indices].to(device),
        "ops": torch.tensor([p.ops for p in programs], dtype=torch.long, device=device),
        "args": torch.tensor([p.args for p in programs], dtype=torch.long, device=device),
        "answers": torch.tensor([feature_set.examples[i].answer for i in indices], dtype=torch.long, device=device),
    }


def extract_hidden(outputs: Any) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]
    last = getattr(outputs, "last_hidden_state", None)
    if torch.is_tensor(last):
        return last
    raise RuntimeError("model did not return hidden states")


def forward_hidden(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    return extract_hidden(outputs)


def load_qwen(args: argparse.Namespace, train_lora: bool) -> Tuple[Any, nn.Module, int, str]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    ensure_pad_token(tokenizer)
    dtype = dtype_from_string(args.torch_dtype)
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": args.device_map if torch.cuda.is_available() else None,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    print(f"[load] Loading {args.model_id} train_lora={train_lora}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if train_lora:
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        target_modules: str | List[str]
        if args.lora_target_modules == "all-linear":
            target_modules = "all-linear"
        else:
            target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()
    hidden_size = int(model.config.hidden_size)
    return tokenizer, model, hidden_size, "AutoModelForCausalLM"


def optimizer_for(params: Sequence[nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer:
    trainable = [p for p in params if p.requires_grad]
    if args.optimizer == "paged_adamw_8bit":
        try:
            import bitsandbytes as bnb  # type: ignore

            return bnb.optim.PagedAdamW8bit(trainable, lr=args.lr, weight_decay=args.weight_decay)
        except Exception as exc:
            print(f"[optim] PagedAdamW8bit unavailable ({exc}); using AdamW", flush=True)
    return torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)


def extract_features(
    examples: Sequence[TaskExample],
    tokenizer: Any,
    model: nn.Module,
    batch_size: int,
    max_prompt_len: int,
    device: torch.device,
) -> FeatureSet:
    model.eval()
    all_hidden: List[torch.Tensor] = []
    all_mask: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            batch = collate_examples(batch_examples, tokenizer, max_prompt_len, device)
            hidden = forward_hidden(model, batch).detach().to(torch.float16).cpu()
            all_hidden.append(hidden)
            all_mask.append(batch["attention_mask"].detach().bool().cpu())
    return FeatureSet(list(examples), torch.cat(all_hidden, dim=0), torch.cat(all_mask, dim=0))


def train_trace_live(
    run_name: str,
    variant: str,
    phase: str,
    model: nn.Module,
    head: QwenBytecodeHead,
    examples: Sequence[TaskExample],
    val_examples: Sequence[TaskExample],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    steps: int,
    log_path: Path,
    offset: int = 0,
) -> int:
    params = list(model.parameters()) + list(head.parameters())
    opt = optimizer_for(params, args)
    for local_step in range(1, steps + 1):
        global_step = offset + local_step
        model.train()
        head.train()
        batch = collate_examples(sample_examples(examples, args.batch_size), tokenizer, args.max_prompt_len, device)
        hidden = forward_hidden(model, batch)
        outputs = head(hidden, batch["attention_mask"])
        loss, aux = program_loss(outputs, batch["ops"], batch["args"], batch["answers"], args.op_loss_weight, args.arg_loss_weight, args.answer_loss_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], args.grad_clip)
        opt.step()
        if local_step == 1 or local_step == steps or global_step % args.eval_every == 0:
            quick = evaluate_trace_live(
                model,
                head,
                val_examples,
                tokenizer,
                args,
                device,
                run_name,
                variant,
                phase,
                "quick_val",
                limit=min(args.quick_eval_size, len(val_examples)),
                search_topk=1,
                max_two_arg_pairs=0,
            )
            row = {
                "run": run_name,
                "variant": variant,
                "phase": phase,
                "local_step": local_step,
                "step": global_step,
                "loss": float(loss.detach().cpu()),
                "train_examples": len(examples),
                "quick_answer_head_accuracy": quick.answer_head_accuracy,
                "quick_bytecode_accuracy": quick.bytecode_accuracy,
                "quick_program_exact": quick.program_exact,
                **aux,
            }
            append_csv(log_path, row, rewrite=not log_path.exists())
            print(
                f"[{run_name}] {phase} step {global_step}/{offset + steps} "
                f"loss={row['loss']:.3f} quick_bytecode={100 * quick.bytecode_accuracy:.1f}%",
                flush=True,
            )
    return offset + steps


def train_trace_features(
    run_name: str,
    variant: str,
    phase: str,
    head: QwenBytecodeHead,
    train_set: FeatureSet,
    val_set: FeatureSet,
    args: argparse.Namespace,
    device: torch.device,
    steps: int,
    log_path: Path,
    offset: int = 0,
) -> int:
    opt = optimizer_for(list(head.parameters()), args)
    for local_step in range(1, steps + 1):
        global_step = offset + local_step
        head.train()
        batch = sample_feature_batch(train_set, args.batch_size, device)
        outputs = head(batch["hidden"], batch["attention_mask"])
        loss, aux = program_loss(outputs, batch["ops"], batch["args"], batch["answers"], args.op_loss_weight, args.arg_loss_weight, args.answer_loss_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        opt.step()
        if local_step == 1 or local_step == steps or global_step % args.eval_every == 0:
            quick = evaluate_trace_features(
                head,
                val_set,
                args,
                device,
                run_name,
                variant,
                phase,
                "quick_val",
                limit=min(args.quick_eval_size, len(val_set.examples)),
                search_topk=1,
                max_two_arg_pairs=0,
            )
            row = {
                "run": run_name,
                "variant": variant,
                "phase": phase,
                "local_step": local_step,
                "step": global_step,
                "loss": float(loss.detach().cpu()),
                "train_examples": len(train_set.examples),
                "quick_answer_head_accuracy": quick.answer_head_accuracy,
                "quick_bytecode_accuracy": quick.bytecode_accuracy,
                "quick_program_exact": quick.program_exact,
                **aux,
            }
            append_csv(log_path, row, rewrite=not log_path.exists())
            print(
                f"[{run_name}] {phase} step {global_step}/{offset + steps} "
                f"loss={row['loss']:.3f} quick_bytecode={100 * quick.bytecode_accuracy:.1f}%",
                flush=True,
            )
    return offset + steps


def train_answer_live(
    run_name: str,
    variant: str,
    phase: str,
    model: nn.Module,
    head: DirectAnswerHead,
    examples: Sequence[TaskExample],
    val_examples: Sequence[TaskExample],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    steps: int,
    log_path: Path,
) -> None:
    params = list(model.parameters()) + list(head.parameters())
    opt = optimizer_for(params, args)
    for step in range(1, steps + 1):
        model.train()
        head.train()
        batch = collate_examples(sample_examples(examples, args.batch_size), tokenizer, args.max_prompt_len, device)
        hidden = forward_hidden(model, batch)
        logits = head(hidden, batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["answers"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], args.grad_clip)
        opt.step()
        if step == 1 or step == steps or step % args.eval_every == 0:
            quick = evaluate_answer_live(model, head, val_examples, tokenizer, args, device, run_name, variant, phase, "quick_val", limit=args.quick_eval_size)
            row = {
                "run": run_name,
                "variant": variant,
                "phase": phase,
                "local_step": step,
                "step": step,
                "loss": float(loss.detach().cpu()),
                "train_examples": len(examples),
                "quick_answer_head_accuracy": quick.answer_head_accuracy,
            }
            append_csv(log_path, row, rewrite=not log_path.exists())
            print(f"[{run_name}] {phase} step {step}/{steps} loss={row['loss']:.3f} quick_answer={100 * quick.answer_head_accuracy:.1f}%", flush=True)


@torch.no_grad()
def evaluate_trace_outputs(
    outputs: Dict[str, torch.Tensor],
    examples: Sequence[TaskExample],
    run_name: str,
    variant: str,
    phase: str,
    split: str,
    search_topk: int,
    max_two_arg_pairs: int,
) -> Tuple[int, int, int, int, int, int, int, int, int, float]:
    answer_ok = 0
    bytecode_ok = 0
    search_ok = 0
    oracle_ok = 0
    prog_exact = 0
    direct_valid = 0
    total_candidates = 0
    valid_candidates = 0
    found_any = 0
    for idx, ex in enumerate(examples):
        op_logits = outputs["op_logits"][idx]
        arg_logits = outputs["arg_logits"][idx]
        answer_ok += int(int(outputs["answer_logits"][idx].argmax().item()) == ex.answer)
        base = program_from_logits(op_logits, arg_logits)
        base_valid, base_answer, _ = execute_program(base)
        direct_valid += int(base_valid)
        bytecode_ok += int(base_valid and base_answer == ex.answer)
        prog_exact += int(program_equal(base, ex.program))
        candidates = generate_candidates(base, op_logits, arg_logits, topk=search_topk, max_two_arg_pairs=max_two_arg_pairs)
        total_candidates += len(candidates)
        chosen, found, valid_count = choose_answer_verified_candidate(candidates, ex.answer, op_logits, arg_logits)
        valid_candidates += valid_count
        found_any += int(found > 0)
        oracle_ok += int(found > 0)
        search_ok += int(chosen is not None)
    return answer_ok, bytecode_ok, search_ok, oracle_ok, prog_exact, direct_valid, total_candidates, valid_candidates, found_any, 0.0


def build_eval_result(
    run_name: str,
    variant: str,
    phase: str,
    split: str,
    n: int,
    answer_ok: int,
    bytecode_ok: int,
    search_ok: int,
    oracle_ok: int,
    prog_exact: int,
    direct_valid: int,
    total_candidates: int,
    valid_candidates: int,
    found_any: int,
) -> EvalResult:
    denom = max(1, n)
    bytecode_acc = bytecode_ok / denom
    oracle_acc = oracle_ok / denom
    search_acc = search_ok / denom
    gap = 0.0 if oracle_acc <= bytecode_acc else (search_acc - bytecode_acc) / (oracle_acc - bytecode_acc)
    return EvalResult(
        run=run_name,
        variant=variant,
        phase=phase,
        split=split,
        n=n,
        answer_head_accuracy=answer_ok / denom,
        bytecode_accuracy=bytecode_acc,
        search_accuracy=search_acc,
        oracle_accuracy=oracle_acc,
        program_exact=prog_exact / denom,
        direct_valid_rate=direct_valid / denom,
        candidate_valid_rate=valid_candidates / max(1, total_candidates),
        mean_candidates=total_candidates / denom,
        found_rate=found_any / denom,
        gap_recovered=gap,
    )


@torch.no_grad()
def evaluate_trace_live(
    model: nn.Module,
    head: QwenBytecodeHead,
    examples: Sequence[TaskExample],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    run_name: str,
    variant: str,
    phase: str,
    split: str,
    limit: Optional[int] = None,
    search_topk: Optional[int] = None,
    max_two_arg_pairs: Optional[int] = None,
) -> EvalResult:
    model.eval()
    head.eval()
    selected = list(examples[:limit] if limit is not None else examples)
    totals = [0] * 9
    for start in range(0, len(selected), args.eval_batch_size):
        cur = selected[start : start + args.eval_batch_size]
        batch = collate_examples(cur, tokenizer, args.max_prompt_len, device)
        hidden = forward_hidden(model, batch)
        outputs = head(hidden, batch["attention_mask"])
        vals = evaluate_trace_outputs(
            outputs,
            cur,
            run_name,
            variant,
            phase,
            split,
            search_topk if search_topk is not None else args.search_topk,
            max_two_arg_pairs if max_two_arg_pairs is not None else args.max_two_arg_pairs,
        )[:9]
        totals = [a + int(b) for a, b in zip(totals, vals)]
    return build_eval_result(run_name, variant, phase, split, len(selected), *totals)


@torch.no_grad()
def evaluate_trace_features(
    head: QwenBytecodeHead,
    feature_set: FeatureSet,
    args: argparse.Namespace,
    device: torch.device,
    run_name: str,
    variant: str,
    phase: str,
    split: str,
    limit: Optional[int] = None,
    search_topk: Optional[int] = None,
    max_two_arg_pairs: Optional[int] = None,
) -> EvalResult:
    head.eval()
    n = min(limit if limit is not None else len(feature_set.examples), len(feature_set.examples))
    totals = [0] * 9
    for start in range(0, n, args.eval_batch_size):
        end = min(n, start + args.eval_batch_size)
        hidden = feature_set.hidden[start:end].to(device)
        mask = feature_set.attention_mask[start:end].to(device)
        outputs = head(hidden, mask)
        vals = evaluate_trace_outputs(
            outputs,
            feature_set.examples[start:end],
            run_name,
            variant,
            phase,
            split,
            search_topk if search_topk is not None else args.search_topk,
            max_two_arg_pairs if max_two_arg_pairs is not None else args.max_two_arg_pairs,
        )[:9]
        totals = [a + int(b) for a, b in zip(totals, vals)]
    return build_eval_result(run_name, variant, phase, split, n, *totals)


@torch.no_grad()
def evaluate_answer_live(
    model: nn.Module,
    head: DirectAnswerHead,
    examples: Sequence[TaskExample],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    run_name: str,
    variant: str,
    phase: str,
    split: str,
    limit: Optional[int] = None,
) -> EvalResult:
    model.eval()
    head.eval()
    selected = list(examples[:limit] if limit is not None else examples)
    correct = 0
    for start in range(0, len(selected), args.eval_batch_size):
        cur = selected[start : start + args.eval_batch_size]
        batch = collate_examples(cur, tokenizer, args.max_prompt_len, device)
        hidden = forward_hidden(model, batch)
        logits = head(hidden, batch["attention_mask"])
        correct += int((logits.argmax(dim=-1) == batch["answers"]).sum().item())
    nan = float("nan")
    return EvalResult(run_name, variant, phase, split, len(selected), correct / max(1, len(selected)), nan, nan, nan, nan, nan, nan, nan, nan, nan)


@torch.no_grad()
def collect_targets_live(
    model: nn.Module,
    head: QwenBytecodeHead,
    examples: Sequence[TaskExample],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    limit: Optional[int] = None,
) -> Tuple[List[TaskExample], Dict[str, Any]]:
    model.eval()
    head.eval()
    selected = list(examples[:limit] if limit is not None else examples)
    targets: List[TaskExample] = []
    total_candidates = 0
    valid_candidates = 0
    found = 0
    changed = 0
    for start in range(0, len(selected), args.eval_batch_size):
        cur = selected[start : start + args.eval_batch_size]
        batch = collate_examples(cur, tokenizer, args.max_prompt_len, device)
        outputs = head(forward_hidden(model, batch), batch["attention_mask"])
        for idx, ex in enumerate(cur):
            op_logits = outputs["op_logits"][idx]
            arg_logits = outputs["arg_logits"][idx]
            base = program_from_logits(op_logits, arg_logits)
            candidates = generate_candidates(base, op_logits, arg_logits, args.search_topk, args.max_two_arg_pairs)
            total_candidates += len(candidates)
            chosen, found_count, valid_count = choose_answer_verified_candidate(candidates, ex.answer, op_logits, arg_logits)
            valid_candidates += valid_count
            if chosen is not None:
                found += 1
                changed += int(not program_equal(chosen, base))
                targets.append(TaskExample(ex.prompt, ex.domain, ex.answer, chosen, ex.template, ex.length))
    return targets, {
        "source_examples": len(selected),
        "targets": len(targets),
        "found_rate": found / max(1, len(selected)),
        "changed_rate": changed / max(1, found),
        "mean_candidates": total_candidates / max(1, len(selected)),
        "candidate_valid_rate": valid_candidates / max(1, total_candidates),
    }


def save_trace_checkpoint(
    run_name: str,
    variant: str,
    phase: str,
    model: nn.Module,
    head: nn.Module,
    args: argparse.Namespace,
    train_lora: bool,
    hidden_size: int,
) -> Path:
    ckpt_dir = CHECKPOINT_ROOT / run_name / phase
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if train_lora and hasattr(model, "save_pretrained"):
        model.save_pretrained(ckpt_dir / "adapter")
    torch.save(
        {
            "variant": variant,
            "phase": phase,
            "head_state": head.state_dict(),
            "hidden_size": hidden_size,
            "args": vars(args),
            "opcodes": OPCODES,
        },
        ckpt_dir / "head.pt",
    )
    return ckpt_dir


def append_checkpoint_manifest(run_name: str, variant: str, ckpt_dir: Path, notes: str) -> None:
    append_csv(
        ROOT / "checkpoint_manifest.csv",
        {
            "run": run_name,
            "variant": variant,
            "checkpoint_dir": str(ckpt_dir),
            "created_unix": int(time.time()),
            "notes": notes,
        },
        rewrite=False,
    )


def collect_metadata(args: argparse.Namespace, loader: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "transformers_loader": loader,
        "peft_installed": importlib.util.find_spec("peft") is not None,
        "model_id": args.model_id,
        "load_in_4bit": args.load_in_4bit,
        "lora_r": args.lora_r,
    }
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
    return meta


def evaluate_all_trace_live(
    model: nn.Module,
    head: QwenBytecodeHead,
    splits: Dict[str, List[TaskExample]],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    run_name: str,
    variant: str,
    phase: str,
) -> List[EvalResult]:
    eval_names = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    return [evaluate_trace_live(model, head, splits[name], tokenizer, args, device, run_name, variant, phase, name) for name in eval_names]


def evaluate_all_trace_features(
    head: QwenBytecodeHead,
    feature_cache: Dict[str, FeatureSet],
    args: argparse.Namespace,
    device: torch.device,
    run_name: str,
    variant: str,
    phase: str,
) -> List[EvalResult]:
    eval_names = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    return [evaluate_trace_features(head, feature_cache[name], args, device, run_name, variant, phase, name) for name in eval_names]


def evaluate_all_answer_live(
    model: nn.Module,
    head: DirectAnswerHead,
    splits: Dict[str, List[TaskExample]],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
    run_name: str,
    variant: str,
    phase: str,
) -> List[EvalResult]:
    eval_names = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    return [evaluate_answer_live(model, head, splits[name], tokenizer, args, device, run_name, variant, phase, name) for name in eval_names]


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    RUNS.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["train_log.csv", "metrics.csv", "target_log.csv"]:
        path = run_dir / filename
        if path.exists():
            path.unlink()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_lora = args.variant in {"qlora_trace", "qlora_trace_ei", "qlora_answer"}
    tokenizer, model, hidden_size, loader = load_qwen(args, train_lora=train_lora)
    if not args.load_in_4bit:
        model.to(device)
    metadata = collect_metadata(args, loader)
    splits = make_splits(args)
    max_tokens = max(prompt_lengths([ex], tokenizer)[0] for examples in splits.values() for ex in examples)
    if max_tokens > args.max_prompt_len:
        raise RuntimeError(f"max_prompt_len={args.max_prompt_len} truncates a prompt of length {max_tokens}")
    with (run_dir / "dataset_manifest.json").open("w") as f:
        json.dump(
            {
                "run": args.run_name,
                "variant": args.variant,
                "sizes": {name: len(examples) for name, examples in splits.items()},
                "seed": args.seed,
                "opcodes": OPCODES,
                "modulus": MODULUS,
                "max_program_len": MAX_PROGRAM_LEN,
                "max_prompt_tokens": max_tokens,
                "metadata": metadata,
            },
            f,
            indent=2,
        )

    all_results: List[EvalResult] = []
    train_log = run_dir / "train_log.csv"

    if args.variant == "frozen_trace":
        feature_names = ["trace_train", "val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
        feature_cache: Dict[str, FeatureSet] = {}
        for name in feature_names:
            print(f"[features] extracting {name} ({len(splits[name])})", flush=True)
            feature_cache[name] = extract_features(splits[name], tokenizer, model, args.feature_batch_size, args.max_prompt_len, device)
        del model
        torch.cuda.empty_cache()
        head = QwenBytecodeHead(hidden_size, args.d_model, args.layers, args.heads, args.dropout).to(device)
        train_trace_features(args.run_name, args.variant, "trace_supervised", head, feature_cache["trace_train"], feature_cache["val_mixed"], args, device, args.trace_steps, train_log)
        all_results.extend(evaluate_all_trace_features(head, feature_cache, args, device, args.run_name, args.variant, "trace_supervised"))
        ckpt = save_trace_checkpoint(args.run_name, args.variant, "trace_supervised", nn.Module(), head, args, False, hidden_size)
        append_checkpoint_manifest(args.run_name, args.variant, ckpt, "frozen Qwen features; trained bytecode head")

    elif args.variant == "qlora_trace":
        head = QwenBytecodeHead(hidden_size, args.d_model, args.layers, args.heads, args.dropout).to(device)
        train_trace_live(args.run_name, args.variant, "trace_supervised", model, head, splits["trace_train"], splits["val_mixed"], tokenizer, args, device, args.trace_steps, train_log)
        all_results.extend(evaluate_all_trace_live(model, head, splits, tokenizer, args, device, args.run_name, args.variant, "trace_supervised"))
        ckpt = save_trace_checkpoint(args.run_name, args.variant, "trace_supervised", model, head, args, True, hidden_size)
        append_checkpoint_manifest(args.run_name, args.variant, ckpt, "QLoRA trace-supervised typed-bytecode compiler")

    elif args.variant == "qlora_trace_ei":
        head = QwenBytecodeHead(hidden_size, args.d_model, args.layers, args.heads, args.dropout).to(device)
        step = train_trace_live(args.run_name, args.variant, "seed_trace", model, head, splits["seed_train"], splits["val_mixed"], tokenizer, args, device, args.seed_steps, train_log)
        all_results.extend(evaluate_all_trace_live(model, head, splits, tokenizer, args, device, args.run_name, args.variant, "seed_trace"))
        ckpt = save_trace_checkpoint(args.run_name, args.variant, "seed_trace", model, head, args, True, hidden_size)
        append_checkpoint_manifest(args.run_name, args.variant, ckpt, "QLoRA seed trace phase")
        current_train = list(splits["seed_train"])
        for round_idx in range(1, args.expert_rounds + 1):
            targets, stats = collect_targets_live(model, head, splits["unlabeled_train"], tokenizer, args, device, limit=args.unlabeled_collect_limit)
            stats.update({"run": args.run_name, "variant": args.variant, "round": round_idx, "phase": f"expert_round_{round_idx}"})
            append_csv(run_dir / "target_log.csv", stats, rewrite=not (run_dir / "target_log.csv").exists())
            current_train = list(splits["seed_train"]) + targets
            print(
                f"[{args.run_name}] round {round_idx} targets={len(targets)} "
                f"found={100 * stats['found_rate']:.1f}% changed={100 * stats['changed_rate']:.1f}%",
                flush=True,
            )
            step = train_trace_live(
                args.run_name,
                args.variant,
                f"expert_round_{round_idx}",
                model,
                head,
                current_train,
                splits["val_mixed"],
                tokenizer,
                args,
                device,
                args.expert_steps,
                train_log,
                offset=step,
            )
            all_results.extend(evaluate_all_trace_live(model, head, splits, tokenizer, args, device, args.run_name, args.variant, f"expert_round_{round_idx}"))
            ckpt = save_trace_checkpoint(args.run_name, args.variant, f"expert_round_{round_idx}", model, head, args, True, hidden_size)
            append_checkpoint_manifest(args.run_name, args.variant, ckpt, f"QLoRA expert round {round_idx}")

    elif args.variant == "qlora_answer":
        head = DirectAnswerHead(hidden_size, args.answer_head_width, args.dropout).to(device)
        train_answer_live(args.run_name, args.variant, "answer_only", model, head, splits["answer_train"], splits["val_mixed"], tokenizer, args, device, args.answer_steps, train_log)
        all_results.extend(evaluate_all_answer_live(model, head, splits, tokenizer, args, device, args.run_name, args.variant, "answer_only"))
        ckpt = save_trace_checkpoint(args.run_name, args.variant, "answer_only", model, head, args, True, hidden_size)
        append_checkpoint_manifest(args.run_name, args.variant, ckpt, "QLoRA direct answer-only control")

    else:
        raise ValueError(args.variant)

    write_eval_results(run_dir / "metrics.csv", all_results)
    with (run_dir / "results.json").open("w") as f:
        json.dump(
            {
                "run": args.run_name,
                "variant": args.variant,
                "args": vars(args),
                "metadata": metadata,
                "final_results": [asdict(r) for r in all_results],
            },
            f,
            indent=2,
        )
    print(f"[done] wrote {run_dir / 'results.json'}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_name", required=True)
    p.add_argument("--variant", choices=["frozen_trace", "qlora_trace", "qlora_trace_ei", "qlora_answer"], required=True)
    p.add_argument("--model_id", default="Qwen/Qwen3-4B")
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch_dtype", default="bf16")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="all-linear")
    p.add_argument("--optimizer", choices=["adamw", "paged_adamw_8bit"], default="paged_adamw_8bit")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--trace_train_size", type=int, default=512)
    p.add_argument("--seed_train_size", type=int, default=256)
    p.add_argument("--unlabeled_train_size", type=int, default=1024)
    p.add_argument("--unlabeled_collect_limit", type=int, default=1024)
    p.add_argument("--answer_train_size", type=int, default=512)
    p.add_argument("--val_size", type=int, default=128)
    p.add_argument("--fresh_size", type=int, default=128)
    p.add_argument("--hard_size", type=int, default=128)
    p.add_argument("--max_arith_steps", type=int, default=4)
    p.add_argument("--max_prompt_len", type=int, default=160)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--answer_head_width", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--feature_batch_size", type=int, default=8)
    p.add_argument("--trace_steps", type=int, default=600)
    p.add_argument("--seed_steps", type=int, default=300)
    p.add_argument("--expert_steps", type=int, default=200)
    p.add_argument("--expert_rounds", type=int, default=2)
    p.add_argument("--answer_steps", type=int, default=600)
    p.add_argument("--eval_every", type=int, default=150)
    p.add_argument("--quick_eval_size", type=int, default=64)
    p.add_argument("--op_loss_weight", type=float, default=1.0)
    p.add_argument("--arg_loss_weight", type=float, default=1.0)
    p.add_argument("--answer_loss_weight", type=float, default=0.2)
    p.add_argument("--search_topk", type=int, default=3)
    p.add_argument("--max_two_arg_pairs", type=int, default=8)
    p.add_argument("--seed", type=int, default=41)
    return p


def main() -> None:
    run_experiment(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
