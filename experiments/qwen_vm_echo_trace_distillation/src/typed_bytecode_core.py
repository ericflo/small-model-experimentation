#!/usr/bin/env python3
"""Typed-bytecode VM core for VM-ECHO trace distillation.

This experiment is intentionally self-contained. It defines a small typed
stack-machine bytecode, generates natural-language tasks with gold bytecode,
trains a text-to-bytecode compiler, and performs answer-verified expert
iteration: the current compiler proposes programs, local typed search repairs
programs using only final-answer verification, and the repaired programs become
new supervision for the compiler.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path("experiments/qwen_vm_echo_trace_distillation")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_vm_echo_trace_distillation/checkpoints")

MODULUS = 97
MAX_PROGRAM_LEN = 16
MAX_PROMPT_LEN = 96

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
NO_ARG = 0

LOOKUP_A = {0: 11, 1: 23, 2: 37, 3: 41, 4: 59, 5: 61, 6: 73, 7: 89}
LOOKUP_B = {0: 7, 1: 19, 2: 29, 3: 43, 4: 53, 5: 67, 6: 79, 7: 83}

TOKEN_RE = re.compile(r"\d+|[A-Za-z_]+|[^\w\s]")


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
    phase: str
    split: str
    n: int
    direct_accuracy: float
    search_accuracy: float
    oracle_accuracy: float
    program_exact: float
    valid_rate: float
    direct_valid_rate: float
    mean_candidates: float
    found_rate: float
    gap_recovered: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_text(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class PromptVocab:
    def __init__(self) -> None:
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.itos = ["<pad>", "<unk>"]

    def add_texts(self, texts: Iterable[str]) -> None:
        for text in texts:
            for tok in tokenize_text(text):
                if tok not in self.stoi:
                    self.stoi[tok] = len(self.itos)
                    self.itos.append(tok)

    def encode(self, text: str, max_len: int = MAX_PROMPT_LEN) -> Tuple[List[int], List[int]]:
        ids = [self.stoi.get(tok, 1) for tok in tokenize_text(text)]
        ids = ids[:max_len]
        mask = [1] * len(ids)
        while len(ids) < max_len:
            ids.append(0)
            mask.append(0)
        return ids, mask

    def to_json(self) -> Dict[str, Any]:
        return {"itos": self.itos}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "PromptVocab":
        vocab = cls()
        vocab.itos = list(data["itos"])
        vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
        return vocab


def execute_program(program: BytecodeProgram, max_len: int = MAX_PROGRAM_LEN) -> Tuple[bool, int, List[List[int]]]:
    """Execute a bytecode program.

    Returns ``(valid, answer, trace)``. Values are i32-like but kept modulo 97
    for bounded targets. Boolean comparisons push 0/1.
    """
    stack: List[int] = []
    trace: List[List[int]] = []
    ended = False
    for op, arg in zip(program.ops[:max_len], program.args[:max_len]):
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
            b = max(1, b)
            stack.append(a % b)
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
            k = stack.pop() % 8
            stack.append(LOOKUP_A[k] % MODULUS)
        elif name == "LOOKUP_B":
            if not stack:
                return False, 0, trace
            k = stack.pop() % 8
            stack.append(LOOKUP_B[k] % MODULUS)
        elif name == "END":
            ended = True
        else:
            return False, 0, trace
        trace.append(list(stack))
    if not ended or len(stack) != 1:
        return False, 0, trace
    return True, int(stack[-1] % MODULUS), trace


def normalize_program(program: BytecodeProgram) -> BytecodeProgram:
    return program.padded(MAX_PROGRAM_LEN)


def program_equal(a: BytecodeProgram, b: BytecodeProgram) -> bool:
    ap = normalize_program(a)
    bp = normalize_program(b)
    return ap.ops == bp.ops and ap.args == bp.args


def make_prog(items: Sequence[Tuple[str, int]]) -> BytecodeProgram:
    ops = [OP_TO_ID[name] for name, _ in items]
    args = [arg % MODULUS for _, arg in items]
    return normalize_program(BytecodeProgram(ops=ops, args=args))


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
        threshold = self.rng.randint(20, 100 if hard else 80)
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
        table_text = "A maps 0:11 1:23 2:37 3:41 4:59 5:61 6:73 7:89" if table == "A" else "B maps 0:7 1:19 2:29 3:43 4:53 5:67 6:79 7:83"
        standard = f"Lookup table {table}. {table_text}. What value is stored for key {key}?"
        paraphrases = [
            f"Use table {table}: {table_text}. Retrieve key {key}.",
            f"Table lookup problem. In table {table}, find the entry for {key}. {table_text}.",
        ]
        return TaskExample(self._render(standard, paraphrases, template), "lookup", answer, make_prog(instrs), template, 1)

    def make_one(self, template: str = "mixed", hard: bool = False) -> TaskExample:
        domains = [self.arithmetic, self.calendar, self.unit, self.list_task, self.boolean, self.lookup]
        fn = self.rng.choice(domains)
        return fn(template=template, hard=hard)

    def make_set(self, n: int, template: str, hard: bool = False) -> List[TaskExample]:
        return [self.make_one(template=template, hard=hard) for _ in range(n)]

    def make_paired_set(self, n: int, hard: bool = False) -> List[TaskExample]:
        out: List[TaskExample] = []
        for _ in range(n):
            # Generate with a deterministic state snapshot so standard and
            # paraphrase share the same latent program.
            state = self.rng.getstate()
            standard = self.make_one(template="standard", hard=hard)
            self.rng.setstate(state)
            paraphrase = self.make_one(template="paraphrase", hard=hard)
            out.extend([standard, paraphrase])
        return out


class BytecodeDataset(Dataset):
    def __init__(self, examples: Sequence[TaskExample], vocab: PromptVocab) -> None:
        self.examples = list(examples)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        ids, mask = self.vocab.encode(ex.prompt)
        program = normalize_program(ex.program)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "ops": torch.tensor(program.ops, dtype=torch.long),
            "args": torch.tensor(program.args, dtype=torch.long),
            "answer": torch.tensor(ex.answer, dtype=torch.long),
            "domain": ex.domain,
        }


class BytecodeCompiler(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        max_prompt_len: int = MAX_PROMPT_LEN,
        max_program_len: int = MAX_PROGRAM_LEN,
    ) -> None:
        super().__init__()
        self.max_program_len = max_program_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_prompt_len, d_model)
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        key_padding_mask = ~attention_mask.bool()
        memory = self.encoder(x, src_key_padding_mask=key_padding_mask)
        queries = self.slot_queries.unsqueeze(0).expand(bsz, -1, -1)
        decoded = self.decoder(queries, memory, memory_key_padding_mask=key_padding_mask)
        pooled = (memory * attention_mask.unsqueeze(-1).float()).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return {
            "op_logits": self.op_head(decoded),
            "arg_logits": self.arg_head(decoded),
            "answer_logits": self.answer_head(pooled),
        }


def stack_delta(op: int) -> int:
    name = ID_TO_OP.get(int(op), "BAD")
    if name == "PUSH":
        return 1
    if name in {"ADD", "SUB", "MUL", "MOD", "MAX", "MIN", "GT", "EQ"}:
        return -1
    if name in {"LOOKUP_A", "LOOKUP_B"}:
        return 0
    return 0


def allowed_ops_for_depth(depth: int, slot: int, max_len: int) -> List[int]:
    """Return opcodes that preserve stack-machine well-formedness.

    This is the typed-bytecode ABI doing useful work: even a weak compiler is
    decoded through a validator instead of being allowed to emit arbitrary slot
    labels.
    """
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
    """Decode logits into a stack-valid bytecode program."""
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
    p = normalize_program(program)
    return tuple(p.ops), tuple(p.args)


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
                p = BytecodeProgram(ops=list(base.ops), args=list(base.args))
                p.ops[slot] = int(new_op)
                out[canonical_key(p)] = p
        for new_arg in arg_top[slot]:
            if new_arg != base.args[slot]:
                p = BytecodeProgram(ops=list(base.ops), args=list(base.args))
                p.args[slot] = int(new_arg)
                out[canonical_key(p)] = p
        for new_op in op_top[slot]:
            for new_arg in arg_top[slot]:
                if new_op != base.ops[slot] or new_arg != base.args[slot]:
                    p = BytecodeProgram(ops=list(base.ops), args=list(base.args))
                    p.ops[slot] = int(new_op)
                    p.args[slot] = int(new_arg)
                    out[canonical_key(p)] = p
    # A small second-order argument search is useful for copied constants.
    active_slots = list(range(min(MAX_PROGRAM_LEN, max_two_arg_pairs)))
    for i, slot_a in enumerate(active_slots):
        for slot_b in active_slots[i + 1 :]:
            for arg_a in arg_top[slot_a][1:topk]:
                for arg_b in arg_top[slot_b][1:topk]:
                    p = BytecodeProgram(ops=list(base.ops), args=list(base.args))
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
    best: Optional[Tuple[float, BytecodeProgram]] = None
    found = 0
    for cand in candidates:
        valid, pred, _ = execute_program(cand)
        if valid:
            valid_count += 1
        if valid and pred == int(answer):
            found += 1
            score = program_logprob(cand, op_logits, arg_logits)
            if best is None or score > best[0]:
                best = (score, cand)
    return (best[1] if best else None), found, valid_count


def program_loss(
    outputs: Dict[str, torch.Tensor],
    ops: torch.Tensor,
    args: torch.Tensor,
    answers: torch.Tensor,
    answer_weight: float,
) -> torch.Tensor:
    op_loss = F.cross_entropy(outputs["op_logits"].reshape(-1, len(OPCODES)), ops.reshape(-1))
    arg_loss = F.cross_entropy(outputs["arg_logits"].reshape(-1, MODULUS), args.reshape(-1))
    ans_loss = F.cross_entropy(outputs["answer_logits"], answers)
    return op_loss + arg_loss + answer_weight * ans_loss


def train_model(
    model: BytecodeCompiler,
    train_examples: Sequence[TaskExample],
    vocab: PromptVocab,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    answer_weight: float,
    log_path: Path,
    phase: str,
    val_examples: Optional[Sequence[TaskExample]] = None,
) -> None:
    ds = BytecodeDataset(train_examples, vocab)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rows: List[Dict[str, Any]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        count = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            ops = batch["ops"].to(device)
            args = batch["args"].to(device)
            answers = batch["answer"].to(device)
            opt.zero_grad(set_to_none=True)
            outputs = model(input_ids, mask)
            loss = program_loss(outputs, ops, args, answers, answer_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * input_ids.shape[0]
            count += input_ids.shape[0]
        row: Dict[str, Any] = {"phase": phase, "epoch": epoch, "loss": total / max(1, count), "train_examples": len(train_examples)}
        if val_examples is not None and len(val_examples) > 0:
            quick = evaluate_model(
                model,
                val_examples[: min(96, len(val_examples))],
                vocab,
                device,
                split="quick_val",
                run="quick",
                phase=phase,
                search_topk=1,
                max_two_arg_pairs=0,
            )
            row["quick_val_direct_accuracy"] = quick.direct_accuracy
            row["quick_val_valid_rate"] = quick.direct_valid_rate
        rows.append(row)
        append_csv(log_path, rows[-1], rewrite=not log_path.exists() and epoch == 1 and phase == rows[0]["phase"])


def append_csv(path: Path, row: Dict[str, Any], rewrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and not rewrite
    mode = "a" if exists else "w"
    with path.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_model(
    model: BytecodeCompiler,
    examples: Sequence[TaskExample],
    vocab: PromptVocab,
    device: torch.device,
    split: str,
    run: str,
    phase: str,
    search_topk: int,
    max_two_arg_pairs: int,
) -> EvalResult:
    model.eval()
    direct_ok = 0
    search_ok = 0
    oracle_ok = 0
    prog_exact = 0
    valid = 0
    direct_valid = 0
    total_candidates = 0
    found_any = 0
    with torch.no_grad():
        for ex in examples:
            ids, mask = vocab.encode(ex.prompt)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attn = torch.tensor([mask], dtype=torch.bool, device=device)
            outputs = model(input_ids, attn)
            op_logits = outputs["op_logits"][0]
            arg_logits = outputs["arg_logits"][0]
            base = program_from_logits(op_logits, arg_logits)
            base_valid, base_answer, _ = execute_program(base)
            direct_valid += int(base_valid)
            direct_ok += int(base_valid and base_answer == ex.answer)
            prog_exact += int(program_equal(base, ex.program))
            candidates = generate_candidates(base, op_logits, arg_logits, topk=search_topk, max_two_arg_pairs=max_two_arg_pairs)
            total_candidates += len(candidates)
            chosen, found, valid_count = choose_answer_verified_candidate(candidates, ex.answer, op_logits, arg_logits)
            found_any += int(found > 0)
            valid += valid_count
            if chosen is not None:
                search_ok += 1
            # For this experiment the local oracle uses the same answer
            # verifier but ignores model score when asking whether the
            # candidate set contains a correct program.
            oracle_ok += int(found > 0)
    n = max(1, len(examples))
    base = direct_ok / n
    oracle = oracle_ok / n
    search = search_ok / n
    gap = 0.0 if oracle <= base else (search - base) / (oracle - base)
    return EvalResult(
        run=run,
        phase=phase,
        split=split,
        n=len(examples),
        direct_accuracy=base,
        search_accuracy=search,
        oracle_accuracy=oracle,
        program_exact=prog_exact / n,
        valid_rate=valid / max(1, total_candidates),
        direct_valid_rate=direct_valid / n,
        mean_candidates=total_candidates / n,
        found_rate=found_any / n,
        gap_recovered=gap,
    )


def collect_expert_targets(
    model: BytecodeCompiler,
    examples: Sequence[TaskExample],
    vocab: PromptVocab,
    device: torch.device,
    search_topk: int,
    max_two_arg_pairs: int,
) -> Tuple[List[TaskExample], Dict[str, Any]]:
    model.eval()
    targets: List[TaskExample] = []
    total_candidates = 0
    found = 0
    changed = 0
    valid_total = 0
    with torch.no_grad():
        for ex in examples:
            ids, mask = vocab.encode(ex.prompt)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attn = torch.tensor([mask], dtype=torch.bool, device=device)
            outputs = model(input_ids, attn)
            op_logits = outputs["op_logits"][0]
            arg_logits = outputs["arg_logits"][0]
            base = program_from_logits(op_logits, arg_logits)
            candidates = generate_candidates(base, op_logits, arg_logits, topk=search_topk, max_two_arg_pairs=max_two_arg_pairs)
            total_candidates += len(candidates)
            chosen, found_count, valid_count = choose_answer_verified_candidate(candidates, ex.answer, op_logits, arg_logits)
            valid_total += valid_count
            if chosen is not None:
                found += 1
                changed += int(not program_equal(chosen, base))
                targets.append(TaskExample(prompt=ex.prompt, domain=ex.domain, answer=ex.answer, program=chosen, template=ex.template, length=ex.length))
    stats = {
        "source_examples": len(examples),
        "targets": len(targets),
        "found_rate": found / max(1, len(examples)),
        "changed_rate": changed / max(1, found),
        "mean_candidates": total_candidates / max(1, len(examples)),
        "candidate_valid_rate": valid_total / max(1, total_candidates),
    }
    return targets, stats


def write_eval_results(path: Path, results: Sequence[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for res in results:
            writer.writerow(asdict(res))


def save_checkpoint(model: BytecodeCompiler, vocab: PromptVocab, path: Path, extra: Dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": vocab.to_json(), "extra": extra}, path / "compiler.pt")


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


def build_vocab(splits: Dict[str, List[TaskExample]]) -> PromptVocab:
    vocab = PromptVocab()
    vocab.add_texts(ex.prompt for examples in splits.values() for ex in examples)
    return vocab


def new_model(vocab: PromptVocab, args: argparse.Namespace, device: torch.device) -> BytecodeCompiler:
    model = BytecodeCompiler(
        vocab_size=len(vocab.itos),
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    )
    return model.to(device)


def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    root_run = RUNS / args.run_name
    root_run.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "src").mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    splits = make_splits(args)
    vocab = build_vocab(splits)
    with (root_run / "dataset_manifest.json").open("w") as f:
        json.dump(
            {
                "run": args.run_name,
                "seed": args.seed,
                "sizes": {name: len(examples) for name, examples in splits.items()},
                "vocab_size": len(vocab.itos),
                "opcodes": OPCODES,
                "modulus": MODULUS,
                "max_program_len": MAX_PROGRAM_LEN,
            },
            f,
            indent=2,
        )

    eval_splits = {
        "val_mixed": splits["val_mixed"],
        "fresh_standard": splits["fresh_standard"],
        "fresh_paraphrase": splits["fresh_paraphrase"],
        "fresh_paired": splits["fresh_paired"],
        "hard_composition": splits["hard_composition"],
    }

    all_results: List[EvalResult] = []
    train_log = root_run / "train_log.csv"
    if train_log.exists():
        train_log.unlink()

    weak = new_model(vocab, args, device)
    train_model(
        weak,
        splits["seed_train"],
        vocab,
        device,
        epochs=args.seed_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        answer_weight=args.answer_weight,
        log_path=train_log,
        phase="seed_supervised",
        val_examples=splits["val_mixed"],
    )
    for split_name, examples in eval_splits.items():
        all_results.append(evaluate_model(weak, examples, vocab, device, split_name, args.run_name, "seed_supervised", args.search_topk, args.max_two_arg_pairs))
    save_checkpoint(weak, vocab, CHECKPOINT_ROOT / args.run_name / "seed_supervised", {"phase": "seed_supervised"})

    current = weak
    expert_pool: List[TaskExample] = list(splits["seed_train"])
    target_rows: List[Dict[str, Any]] = []
    for round_idx in range(1, args.expert_rounds + 1):
        targets, stats = collect_expert_targets(
            current,
            splits["unlabeled_train"],
            vocab,
            device,
            search_topk=args.search_topk,
            max_two_arg_pairs=args.max_two_arg_pairs,
        )
        stats["round"] = round_idx
        stats["phase"] = f"expert_round_{round_idx}"
        target_rows.append(stats)
        append_csv(root_run / "expert_targets.csv", stats, rewrite=round_idx == 1)
        expert_pool = list(splits["seed_train"]) + targets
        train_model(
            current,
            expert_pool,
            vocab,
            device,
            epochs=args.expert_epochs,
            batch_size=args.batch_size,
            lr=args.expert_lr,
            answer_weight=args.answer_weight,
            log_path=train_log,
            phase=f"expert_round_{round_idx}",
            val_examples=splits["val_mixed"],
        )
        for split_name, examples in eval_splits.items():
            all_results.append(evaluate_model(current, examples, vocab, device, split_name, args.run_name, f"expert_round_{round_idx}", args.search_topk, args.max_two_arg_pairs))
        save_checkpoint(current, vocab, CHECKPOINT_ROOT / args.run_name / f"expert_round_{round_idx}", {"phase": f"expert_round_{round_idx}", "targets": stats})

    if args.full_supervised_epochs > 0:
        full = new_model(vocab, args, device)
        train_model(
            full,
            splits["full_supervised_train"],
            vocab,
            device,
            epochs=args.full_supervised_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            answer_weight=args.answer_weight,
            log_path=train_log,
            phase="full_supervised",
            val_examples=splits["val_mixed"],
        )
        for split_name, examples in eval_splits.items():
            all_results.append(evaluate_model(full, examples, vocab, device, split_name, args.run_name, "full_supervised", args.search_topk, args.max_two_arg_pairs))
        save_checkpoint(full, vocab, CHECKPOINT_ROOT / args.run_name / "full_supervised", {"phase": "full_supervised"})

    write_eval_results(root_run / "metrics.csv", all_results)
    with (root_run / "results.json").open("w") as f:
        json.dump(
            {
                "run": args.run_name,
                "args": vars(args),
                "device": str(device),
                "sizes": {name: len(examples) for name, examples in splits.items()},
                "target_rows": target_rows,
                "final_results": [asdict(r) for r in all_results],
            },
            f,
            indent=2,
        )
    manifest = ROOT / "checkpoint_manifest.csv"
    manifest_exists = manifest.exists()
    with manifest.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "checkpoint_dir", "created_unix", "notes"])
        if not manifest_exists:
            writer.writeheader()
        writer.writerow({"run": args.run_name, "checkpoint_dir": str(CHECKPOINT_ROOT / args.run_name), "created_unix": int(time.time()), "notes": "typed bytecode compiler checkpoints"})


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_name", required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="")
    p.add_argument("--seed_train_size", type=int, default=96)
    p.add_argument("--unlabeled_train_size", type=int, default=256)
    p.add_argument("--full_supervised_size", type=int, default=512)
    p.add_argument("--val_size", type=int, default=128)
    p.add_argument("--fresh_size", type=int, default=128)
    p.add_argument("--hard_size", type=int, default=128)
    p.add_argument("--max_arith_steps", type=int, default=4)
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--expert_lr", type=float, default=2e-4)
    p.add_argument("--answer_weight", type=float, default=0.2)
    p.add_argument("--seed_epochs", type=int, default=8)
    p.add_argument("--expert_rounds", type=int, default=2)
    p.add_argument("--expert_epochs", type=int, default=5)
    p.add_argument("--full_supervised_epochs", type=int, default=10)
    p.add_argument("--search_topk", type=int, default=3)
    p.add_argument("--max_two_arg_pairs", type=int, default=10)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
