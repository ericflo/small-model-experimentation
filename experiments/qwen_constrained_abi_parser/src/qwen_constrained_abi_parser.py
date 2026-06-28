#!/usr/bin/env python3
"""Constrained ABI and parser stress test for explicit procedure compilation.

The primary question is whether validity constraints and a canonical parse
stage improve a small model's ability to compile natural-language tasks into
executable procedures over a fixed stack ABI.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


ROOT = Path("/workspace/experiments/qwen_constrained_abi_parser")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_constrained_abi_parser")
CHECKPOINTS = LARGE_ROOT / "checkpoints"
CACHE_DIR = Path("/workspace/.cache/huggingface")

MODEL_NAME = "Qwen/Qwen3-4B"
FAMILIES = ["string", "number", "table", "date", "list", "path"]
TRAIN_TARGETS = ["answer_only", "program_stack", "parse_emit"]
EVAL_ARMS = [
    "answer_only",
    "program_stack_free",
    "program_stack_constrained",
    "program_stack_resample_valid",
    "parse_then_emit_free",
    "parse_then_emit_constrained",
    "oracle_parse_constrained",
    "gold_abi_constrained",
]
DEPTH_SPLITS = [
    ("eval_indist_d1", 1, False),
    ("eval_comp_d2", 2, False),
    ("eval_comp_d3", 3, False),
    ("eval_comp_d4", 4, False),
    ("eval_comp_d6", 6, False),
    ("eval_template_d2", 2, True),
    ("eval_template_d4", 4, True),
    ("eval_template_d6", 6, True),
]
FINAL_RE = re.compile(r"\bFINAL\s*:?\s*([A-Za-z0-9_.:+/-]+)", re.IGNORECASE)


@dataclass
class Example:
    example_id: str
    split: str
    family: str
    depth: int
    prompt: str
    answer: str
    program: List[str]
    op_names: List[str]
    payload: Dict[str, Any]


@dataclass
class RunConfig:
    run_name: str
    suite: str
    model_name: str
    seeds: List[int]
    arms: List[str]
    train_n: int
    eval_n: int
    train_steps: int
    batch_size: int
    grad_accum: int
    lr: float
    max_length: int
    eval_batch_size: int
    lora_r: int
    lora_alpha: int
    max_new_tokens: int
    resample_attempts: int


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dirs() -> None:
    for path in [RUNS, ANALYSIS, FIGURES, REPORTS, CHECKPOINTS]:
        path.mkdir(parents=True, exist_ok=True)


def append_log(text: str) -> None:
    with (ROOT / "experiment_log.md").open("a") as f:
        f.write(text.rstrip() + "\n\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
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


def pct(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "n/a"
    if math.isnan(v):
        return "n/a"
    return f"{100.0 * v:.1f}%"


def clean_answer(x: Any) -> str:
    return str(x).strip().strip("`'\";,")


def parse_final(text: str) -> Optional[str]:
    m = FINAL_RE.search(text)
    return clean_answer(m.group(1)) if m else None


def has_final(text: str) -> bool:
    return bool(FINAL_RE.search(text))


def safe_int(token: str) -> int:
    cleaned = re.sub(r"[^0-9-]", "", token)
    if cleaned in {"", "-"}:
        raise ValueError(f"bad integer {token!r}")
    return int(cleaned)


def strip_step_prefix(line: str) -> str:
    line = line.strip().strip("`")
    line = re.sub(r"^(STEP\s*)?\d+\s*[\).:-]\s*", "", line, flags=re.IGNORECASE)
    return line.strip()


def canonical_program(text: str) -> List[Tuple[str, Tuple[str, ...]]]:
    out: List[Tuple[str, Tuple[str, ...]]] = []
    for raw in text.splitlines():
        line = strip_step_prefix(raw)
        if not line or line.startswith("#") or FINAL_RE.search(line):
            continue
        parts = line.split()
        if not parts:
            continue
        op = parts[0].upper().rstrip(":")
        out.append((op, tuple(parts[1:])))
    return out


def canon_op_names(text: str) -> List[str]:
    return [op for op, _ in canonical_program(text)]


def rand_code(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{rng.choice(letters)}{rng.choice(letters)}-{rng.randint(1000, 9999)}-{rng.choice(letters)}{rng.choice(letters)}"


def rand_path(rng: random.Random) -> str:
    roots = ["reports", "archive", "client", "ops", "ledger"]
    mids = ["north", "south", "alpha", "beta", "daily"]
    names = ["invoice", "summary", "errors", "sales", "index"]
    exts = ["csv", "json", "txt", "log"]
    return f"/{rng.choice(roots)}/{rng.choice(mids)}/{rng.choice(names)}_{rng.randint(10,99)}.{rng.choice(exts)}"


def describe_op(op: str, args: Sequence[Any], shifted: bool) -> str:
    if op == "TAKE_PREFIX":
        return f"retain its leading {args[0]} symbols" if shifted else f"keep the first {args[0]} characters"
    if op == "TAKE_SUFFIX":
        return f"retain its trailing {args[0]} symbols" if shifted else f"keep the last {args[0]} characters"
    if op == "SLICE":
        return f"clip the inclusive one-based span {args[0]} to {args[1]}" if shifted else f"keep one-indexed positions {args[0]} through {args[1]}"
    if op == "ADD":
        return f"increase it by {args[0]}" if shifted else f"add {args[0]}"
    if op == "SUB":
        return f"decrease it by {args[0]}" if shifted else f"subtract {args[0]}"
    if op == "MUL":
        return f"scale it by {args[0]}" if shifted else f"multiply by {args[0]}"
    if op == "LOOKUP":
        return f"fetch entry {args[0]}" if shifted else f"look up {args[0]}"
    if op == "ADD_LOOKUP":
        return f"include entry {args[0]} in the running total" if shifted else f"add the table value for {args[0]}"
    if op == "ADD_DAYS":
        return f"move forward {args[0]} days" if shifted else f"add {args[0]} days"
    if op == "ADD_WEEKS":
        return f"move forward {args[0]} weeks" if shifted else f"add {args[0]} weeks"
    phrases = {
        False: {
            "STRIP_HYPHENS": "remove hyphens",
            "REVERSE": "reverse the current text",
            "LOWER": "convert the text to lowercase",
            "UPPER": "convert the text to uppercase",
            "SUM": "sum the list",
            "MAX": "take the maximum value",
            "MIN": "take the minimum value",
            "BASENAME": "take the file name",
            "DIRNAME": "take the parent directory",
            "EXTENSION": "take the file extension",
            "STRIP_EXT": "remove the file extension",
        },
        True: {
            "STRIP_HYPHENS": "drop separator dashes",
            "REVERSE": "flip the current text left-to-right",
            "LOWER": "make all letters lowercase",
            "UPPER": "make all letters uppercase",
            "SUM": "total the numbers",
            "MAX": "keep the largest number",
            "MIN": "keep the smallest number",
            "BASENAME": "isolate the last path component",
            "DIRNAME": "isolate the containing folder",
            "EXTENSION": "isolate the suffix after the dot",
            "STRIP_EXT": "drop the suffix after the dot",
        },
    }
    return phrases[shifted][op]


def execute_gold(program: Sequence[str]) -> str:
    ok, value, err = execute_stack("\n".join(program))
    if not ok or value is None:
        raise RuntimeError(f"gold program failed: {err}\n{program}")
    return value


def make_string(seed_rng: random.Random, split: str, idx: int, depth: int, shifted: bool) -> Example:
    text = rand_code(seed_rng)
    ops: List[Tuple[str, Tuple[Any, ...]]] = []
    choices = ["STRIP_HYPHENS", "REVERSE", "LOWER", "UPPER"]
    for step in range(depth):
        if step == 0:
            op = seed_rng.choice(["STRIP_HYPHENS", "REVERSE", "LOWER", "UPPER"])
        elif step == depth - 1 and depth >= 2:
            op = seed_rng.choice(["TAKE_PREFIX", "TAKE_SUFFIX", "SLICE", "REVERSE", "LOWER", "UPPER"])
        else:
            op = seed_rng.choice(choices)
        if op in {"TAKE_PREFIX", "TAKE_SUFFIX"}:
            ops.append((op, (seed_rng.randint(3, 6),)))
        elif op == "SLICE":
            a = seed_rng.randint(1, 4)
            b = seed_rng.randint(a + 2, min(8, a + 5))
            ops.append((op, (a, b)))
        else:
            ops.append((op, ()))
    program = [f"PUSH_TEXT {text}"] + [f"{op} {' '.join(map(str, args))}".strip() for op, args in ops]
    steps = "; ".join(f"{i+1}) {describe_op(op, args, shifted)}" for i, (op, args) in enumerate(ops))
    if shifted:
        prompt = f"Transform the code-like text `{text}`. Work in this order: {steps}. Return the resulting text."
    else:
        prompt = f"Given text {text}, perform these steps in order: {steps}. Return the result."
    answer = execute_gold(program)
    return Example(f"{split}_string_{idx}", split, "string", depth, prompt, answer, program, [op for op, _ in ops], {"input": text})


def make_number(seed_rng: random.Random, split: str, idx: int, depth: int, shifted: bool) -> Example:
    start = seed_rng.randint(2, 30)
    ops: List[Tuple[str, Tuple[Any, ...]]] = []
    for _ in range(depth):
        op = seed_rng.choice(["ADD", "SUB", "MUL"])
        if op == "MUL":
            args = (seed_rng.randint(2, 7),)
        else:
            args = (seed_rng.randint(1, 25),)
        ops.append((op, args))
    program = [f"PUSH {start}"] + [f"{op} {args[0]}" for op, args in ops]
    steps = "; ".join(f"{i+1}) {describe_op(op, args, shifted)}" for i, (op, args) in enumerate(ops))
    prompt = (f"Start with the number {start}. Apply the following operations in order: {steps}. Return the integer."
              if not shifted else f"Beginning at {start}, update the running value like this: {steps}. What integer remains?")
    answer = execute_gold(program)
    return Example(f"{split}_number_{idx}", split, "number", depth, prompt, answer, program, [op for op, _ in ops], {"start": start})


def make_table(seed_rng: random.Random, split: str, idx: int, depth: int, shifted: bool) -> Example:
    keys = ["alpha", "beta", "gamma", "delta", "omega"]
    seed_rng.shuffle(keys)
    values = {k: seed_rng.randint(2, 40) for k in keys[:4]}
    first = seed_rng.choice(list(values))
    ops: List[Tuple[str, Tuple[Any, ...]]] = [("LOOKUP", (first,))]
    for _ in range(max(0, depth - 1)):
        op = seed_rng.choice(["ADD", "MUL", "ADD_LOOKUP"])
        if op == "ADD":
            args = (seed_rng.randint(1, 12),)
        elif op == "MUL":
            args = (seed_rng.randint(2, 5),)
        else:
            args = (seed_rng.choice(list(values)),)
        ops.append((op, args))
    table_line = "TABLE " + " ".join(f"{k}:{v}" for k, v in sorted(values.items()))
    program = [table_line] + [f"{op} {' '.join(map(str, args))}".strip() for op, args in ops]
    table_text = ", ".join(f"{k}={v}" for k, v in sorted(values.items()))
    steps = "; ".join(f"{i+1}) {describe_op(op, args, shifted)}" for i, (op, args) in enumerate(ops))
    prompt = (f"Table: {table_text}. Perform these table operations in order: {steps}. Return the integer."
              if not shifted else f"Using lookup entries [{table_text}], process the running value as follows: {steps}. Give the final integer.")
    answer = execute_gold(program)
    return Example(f"{split}_table_{idx}", split, "table", depth, prompt, answer, program, [op for op, _ in ops], {"table": values})


def make_date(seed_rng: random.Random, split: str, idx: int, depth: int, shifted: bool) -> Example:
    start = date(2026, seed_rng.randint(1, 12), seed_rng.randint(1, 23)).isoformat()
    ops: List[Tuple[str, Tuple[Any, ...]]] = []
    for _ in range(depth):
        if seed_rng.random() < 0.55:
            ops.append(("ADD_DAYS", (seed_rng.randint(1, 9),)))
        else:
            ops.append(("ADD_WEEKS", (seed_rng.randint(1, 4),)))
    program = [f"PUSH_DATE {start}"] + [f"{op} {args[0]}" for op, args in ops]
    steps = "; ".join(f"{i+1}) {describe_op(op, args, shifted)}" for i, (op, args) in enumerate(ops))
    prompt = (f"Starting from {start}, apply these date changes in order: {steps}. Return the ISO date."
              if not shifted else f"Begin on {start}; follow this calendar schedule: {steps}. Write the final YYYY-MM-DD date.")
    answer = execute_gold(program)
    return Example(f"{split}_date_{idx}", split, "date", depth, prompt, answer, program, [op for op, _ in ops], {"date": start})


def make_list(seed_rng: random.Random, split: str, idx: int, depth: int, shifted: bool) -> Example:
    values = [seed_rng.randint(1, 20) for _ in range(seed_rng.randint(4, 7))]
    first = seed_rng.choice(["SUM", "MAX", "MIN"])
    ops: List[Tuple[str, Tuple[Any, ...]]] = [(first, ())]
    for _ in range(max(0, depth - 1)):
        op = seed_rng.choice(["ADD", "SUB", "MUL"])
        args = (seed_rng.randint(2, 7) if op == "MUL" else seed_rng.randint(1, 15),)
        ops.append((op, args))
    program = ["PUSH_LIST " + " ".join(map(str, values))] + [f"{op} {' '.join(map(str, args))}".strip() for op, args in ops]
    steps = "; ".join(f"{i+1}) {describe_op(op, args, shifted)}" for i, (op, args) in enumerate(ops))
    prompt = (f"For the list {values}, perform these operations in order: {steps}. Return the integer."
              if not shifted else f"Given numbers {values}, carry out this ordered reduction: {steps}. What integer do you get?")
    answer = execute_gold(program)
    return Example(f"{split}_list_{idx}", split, "list", depth, prompt, answer, program, [op for op, _ in ops], {"values": values})


def make_path(seed_rng: random.Random, split: str, idx: int, depth: int, shifted: bool) -> Example:
    path = rand_path(seed_rng)
    ops: List[Tuple[str, Tuple[Any, ...]]] = []
    pool = ["BASENAME", "DIRNAME", "EXTENSION", "STRIP_EXT", "LOWER", "UPPER"]
    for i in range(depth):
        if i == 0:
            op = seed_rng.choice(["BASENAME", "DIRNAME", "EXTENSION"])
        else:
            op = seed_rng.choice(pool)
        ops.append((op, ()))
    program = [f"PUSH_TEXT {path}"] + [op for op, _ in ops]
    steps = "; ".join(f"{i+1}) {describe_op(op, args, shifted)}" for i, (op, args) in enumerate(ops))
    prompt = (f"Given path {path}, perform these path operations in order: {steps}. Return the resulting text."
              if not shifted else f"Work on filesystem path `{path}` using this sequence: {steps}. Give the final text.")
    answer = execute_gold(program)
    return Example(f"{split}_path_{idx}", split, "path", depth, prompt, answer, program, [op for op, _ in ops], {"path": path})


def make_example(rng: random.Random, split: str, family: str, idx: int, depth: int, shifted: bool) -> Example:
    if family == "string":
        return make_string(rng, split, idx, depth, shifted)
    if family == "number":
        return make_number(rng, split, idx, depth, shifted)
    if family == "table":
        return make_table(rng, split, idx, depth, shifted)
    if family == "date":
        return make_date(rng, split, idx, depth, shifted)
    if family == "list":
        return make_list(rng, split, idx, depth, shifted)
    if family == "path":
        return make_path(rng, split, idx, depth, shifted)
    raise ValueError(family)


def split_depth_shift(split: str) -> Tuple[int, bool]:
    if split == "train":
        return 1, False
    for name, depth, shifted in DEPTH_SPLITS:
        if split == name:
            return depth, shifted
    raise ValueError(split)


def make_split(seed: int, split: str, n: int) -> List[Example]:
    depth, shifted = split_depth_shift(split)
    offsets = {"train": 0}
    offsets.update({name: (i + 1) * 10_000 for i, (name, _, _) in enumerate(DEPTH_SPLITS)})
    rng = random.Random(seed + offsets[split])
    examples = [make_example(rng, split, FAMILIES[i % len(FAMILIES)], i, depth, shifted) for i in range(n)]
    rng.shuffle(examples)
    return examples


def numbered_program(lines: Sequence[str]) -> List[str]:
    return [f"STEP {i + 1}: {line}" for i, line in enumerate(lines)]


def render_target(ex: Example, arm: str) -> str:
    if arm == "answer_only":
        return f"FINAL: {ex.answer}\n"
    if arm == "program_stack":
        return "\n".join(ex.program) + "\n"
    if arm == "parse_emit":
        return "\n".join([
            "PARSE",
            f"FAMILY {ex.family}",
            f"INIT {ex.program[0]}",
            "OPS",
            *ex.program[1:],
            "END_PARSE",
            "PROGRAM",
            *ex.program,
        ]) + "\n"
    raise ValueError(arm)


def format_instruction(arm: str) -> str:
    if arm == "answer_only":
        return "Return one line exactly as: FINAL: <answer>."
    if arm == "program_stack":
        return "Return only raw stack instructions. Do not number steps and do not write a FINAL line or the answer."
    if arm == "parse_emit":
        return "Return PARSE, FAMILY, INIT, OPS, END_PARSE, then PROGRAM with raw stack instructions. Do not write a FINAL line or the answer."
    raise ValueError(arm)


def format_prompt(ex: Example, arm: str) -> str:
    return (
        "Compile the task into the requested executable stack format.\n"
        f"Output rule: {format_instruction(arm)}\n\n"
        f"Task:\n{ex.prompt}\n\n"
        "Output:\n"
    )


def program_section(text: str) -> str:
    if re.search(r"^PROGRAM\s*$", text, flags=re.IGNORECASE | re.MULTILINE):
        return re.split(r"^PROGRAM\s*$", text, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)[1].strip()
    return text.strip()


def parse_block_to_program(text: str) -> List[str]:
    lines = [strip_step_prefix(x) for x in text.splitlines()]
    init: Optional[str] = None
    ops: List[str] = []
    in_ops = False
    for line in lines:
        if not line:
            continue
        upper = line.upper()
        if upper == "PROGRAM":
            break
        if upper.startswith("INIT "):
            init = line[5:].strip()
            continue
        if upper == "OPS":
            in_ops = True
            continue
        if upper == "END_PARSE":
            in_ops = False
            continue
        if in_ops:
            if not upper.startswith("FAMILY ") and upper != "PARSE":
                ops.append(line)
    if init is None:
        return []
    return [init] + ops


def render_program_from_parse(text: str) -> str:
    return "\n".join(parse_block_to_program(text))


def parse_exact(ex: Example, text: str) -> bool:
    return canonical_program(render_program_from_parse(text)) == canonical_program("\n".join(ex.program))


def correct_given_valid(exec_accuracy: float, valid_rate: float) -> float:
    return exec_accuracy / valid_rate if valid_rate > 0 else float("nan")


def candidate_op_lines(ex: Example, step_idx: int) -> List[str]:
    """Return valid next instruction lines for a finite-state ABI grammar.

    step_idx is zero-based over executable operation lines after the initial
    initializer/TABLE line.
    """
    if ex.family == "string":
        lines = ["STRIP_HYPHENS", "REVERSE", "LOWER", "UPPER"]
        lines += [f"TAKE_PREFIX {n}" for n in range(1, 9)]
        lines += [f"TAKE_SUFFIX {n}" for n in range(1, 9)]
        lines += [f"SLICE {a} {b}" for a in range(1, 7) for b in range(a + 1, min(9, a + 6))]
        return lines
    if ex.family == "number":
        return [f"ADD {n}" for n in range(1, 26)] + [f"SUB {n}" for n in range(1, 26)] + [f"MUL {n}" for n in range(2, 8)]
    if ex.family == "table":
        keys = sorted(ex.payload["table"].keys())
        if step_idx == 0:
            return [f"LOOKUP {k}" for k in keys]
        return [f"ADD {n}" for n in range(1, 13)] + [f"MUL {n}" for n in range(2, 6)] + [f"ADD_LOOKUP {k}" for k in keys]
    if ex.family == "date":
        return [f"ADD_DAYS {n}" for n in range(1, 10)] + [f"ADD_WEEKS {n}" for n in range(1, 5)]
    if ex.family == "list":
        if step_idx == 0:
            return ["SUM", "MAX", "MIN"]
        return [f"ADD {n}" for n in range(1, 16)] + [f"SUB {n}" for n in range(1, 16)] + [f"MUL {n}" for n in range(2, 8)]
    if ex.family == "path":
        return ["BASENAME", "DIRNAME", "EXTENSION", "STRIP_EXT", "LOWER", "UPPER"]
    raise ValueError(ex.family)


def constrained_candidate_lines(ex: Example, line_idx: int) -> List[str]:
    if line_idx == 0:
        return [ex.program[0]]
    return candidate_op_lines(ex, line_idx - 1)


def path_basename(x: str) -> str:
    return str(x).rstrip("/").split("/")[-1]


def path_dirname(x: str) -> str:
    s = str(x).rstrip("/")
    if "/" not in s.strip("/"):
        return "/"
    return s.rsplit("/", 1)[0] or "/"


def path_ext(x: str) -> str:
    base = path_basename(x)
    return base.rsplit(".", 1)[1] if "." in base else ""


def path_strip_ext(x: str) -> str:
    s = str(x)
    base = path_basename(s)
    if "." not in base:
        return s
    prefix = s[: len(s) - len(base)]
    return prefix + base.rsplit(".", 1)[0]


def execute_stack(text: str) -> Tuple[bool, Optional[str], str]:
    stack: List[Any] = []
    table: Dict[str, int] = {}
    try:
        for raw in text.splitlines():
            line = strip_step_prefix(raw)
            if not line or line.startswith("#") or FINAL_RE.search(line):
                continue
            parts = line.split()
            if not parts:
                continue
            op = parts[0].upper().rstrip(":")
            if op == "PUSH_TEXT":
                stack.append(" ".join(parts[1:]).strip().strip('"'))
            elif op == "PUSH":
                stack.append(safe_int(parts[1]))
            elif op == "PUSH_DATE":
                stack.append(parts[1].strip())
            elif op == "PUSH_LIST":
                stack.append([safe_int(p) for p in parts[1:]])
            elif op == "TABLE":
                table = {}
                for item in parts[1:]:
                    if ":" in item:
                        k, v = item.split(":", 1)
                        table[k.strip()] = safe_int(v)
            elif op == "STRIP_HYPHENS":
                stack[-1] = str(stack[-1]).replace("-", "")
            elif op == "REVERSE":
                stack[-1] = str(stack[-1])[::-1]
            elif op == "LOWER":
                stack[-1] = str(stack[-1]).lower()
            elif op == "UPPER":
                stack[-1] = str(stack[-1]).upper()
            elif op == "TAKE_PREFIX":
                stack[-1] = str(stack[-1])[: safe_int(parts[1])]
            elif op == "TAKE_SUFFIX":
                stack[-1] = str(stack[-1])[-safe_int(parts[1]) :]
            elif op == "SLICE":
                a = safe_int(parts[1]) - 1
                b = safe_int(parts[2])
                stack[-1] = str(stack[-1])[a:b]
            elif op == "ADD":
                if len(parts) == 1:
                    right = int(stack.pop())
                    left = int(stack.pop())
                    stack.append(left + right)
                else:
                    stack[-1] = int(stack[-1]) + safe_int(parts[1])
            elif op == "SUB":
                stack[-1] = int(stack[-1]) - safe_int(parts[1])
            elif op == "MUL":
                if len(parts) == 1:
                    right = int(stack.pop())
                    left = int(stack.pop())
                    stack.append(left * right)
                else:
                    stack[-1] = int(stack[-1]) * safe_int(parts[1])
            elif op == "LOOKUP":
                stack.append(int(table[parts[1].strip()]))
            elif op == "ADD_LOOKUP":
                stack[-1] = int(stack[-1]) + int(table[parts[1].strip()])
            elif op == "ADD_DAYS":
                d = date.fromisoformat(str(stack[-1]))
                stack[-1] = (d + timedelta(days=safe_int(parts[1]))).isoformat()
            elif op == "ADD_WEEKS":
                d = date.fromisoformat(str(stack[-1]))
                stack[-1] = (d + timedelta(days=7 * safe_int(parts[1]))).isoformat()
            elif op == "SUM":
                stack[-1] = sum(int(x) for x in stack[-1])
            elif op == "MAX":
                stack[-1] = max(int(x) for x in stack[-1])
            elif op == "MIN":
                stack[-1] = min(int(x) for x in stack[-1])
            elif op == "BASENAME":
                stack[-1] = path_basename(str(stack[-1]))
            elif op == "DIRNAME":
                stack[-1] = path_dirname(str(stack[-1]))
            elif op == "EXTENSION":
                stack[-1] = path_ext(str(stack[-1]))
            elif op == "STRIP_EXT":
                stack[-1] = path_strip_ext(str(stack[-1]))
            else:
                return False, None, f"unknown_op:{op}"
        if not stack:
            return False, None, "empty_stack"
        return True, clean_answer(stack[-1]), ""
    except Exception as exc:
        return False, None, type(exc).__name__


def classify_failure(ex: Example, generated: str, valid_exec: bool, exec_correct: bool) -> str:
    if exec_correct:
        gold = canonical_program("\n".join(ex.program))
        got = canonical_program(generated)
        return "correct_exact" if got == gold else "correct_semantic_variant"
    if not valid_exec:
        return "invalid_or_unexecutable"
    got = canonical_program(generated)
    gold = canonical_program("\n".join(ex.program))
    if not got:
        return "empty_or_no_program"
    got_ops = [op for op, _ in got]
    gold_ops = [op for op, _ in gold]
    if got_ops != gold_ops:
        return "wrong_op_order_or_choice"
    if got != gold:
        return "wrong_constant_or_argument"
    return "same_program_wrong_result"


class TextDataset(Dataset):
    def __init__(self, examples: Sequence[Example], arm: str, tokenizer: Any, max_length: int):
        self.rows: List[Dict[str, torch.Tensor]] = []
        eos = tokenizer.eos_token or ""
        for ex in examples:
            prompt = format_prompt(ex, arm)
            target = render_target(ex, arm) + eos
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            ids = (prompt_ids + target_ids)[-max_length:]
            cut = max(0, len(prompt_ids) + len(target_ids) - max_length)
            prompt_kept = max(0, len(prompt_ids) - cut)
            labels = [-100] * prompt_kept + ids[prompt_kept:]
            self.rows.append({"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.rows[idx]


def collate(batch: Sequence[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, mask = [], [], []
    for row in batch:
        ids = row["input_ids"]
        labs = row["labels"]
        pad = max_len - len(ids)
        input_ids.append(torch.cat([torch.full((pad,), pad_id, dtype=torch.long), ids]))
        labels.append(torch.cat([torch.full((pad,), -100, dtype=torch.long), labs]))
        mask.append(torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels), "attention_mask": torch.stack(mask)}


def load_tokenizer(model_name: str) -> Any:
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=str(CACHE_DIR), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(model_name: str, lora_r: int, lora_alpha: int) -> Any:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, lora)


def train_adapter(config: RunConfig, arm: str, seed: int, tokenizer: Any, run_dir: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    set_seed(seed)
    model = load_model(config.model_name, config.lora_r, config.lora_alpha)
    model.train()
    train_examples = make_split(seed, "train", config.train_n)
    ds = TextDataset(train_examples, arm, tokenizer, config.max_length)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    logs: List[Dict[str, Any]] = []
    step = 0
    accum = 0
    opt.zero_grad(set_to_none=True)
    while step < config.train_steps:
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / config.grad_accum
            loss.backward()
            accum += 1
            if accum >= config.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                accum = 0
                if step == 1 or step % max(1, config.train_steps // 4) == 0 or step == config.train_steps:
                    logs.append({"suite": config.suite, "run": run_dir.name, "arm": arm, "seed": seed, "step": step, "loss": float(out.loss.detach().cpu())})
                    log(f"{run_dir.name}: step {step}/{config.train_steps} loss={float(out.loss.detach().cpu()):.4f}")
                if step >= config.train_steps:
                    break
    ckpt = CHECKPOINTS / run_dir.name
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    return model, logs


def generate_batch(model: Any, tokenizer: Any, prompts: List[str], max_new_tokens: int, do_sample: bool = False, temperature: float = 0.7) -> List[str]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    kwargs: Dict[str, Any] = {}
    if do_sample:
        kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.92})
    else:
        kwargs.update({"do_sample": False})
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
    texts: List[str] = []
    for i in range(out.shape[0]):
        gen = out[i, enc["input_ids"].shape[1] :]
        texts.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return texts


def token_prefix_allowed(tokenizer: Any, line_ids: List[List[List[int]]], total_lines: int, base_len: int):
    eos_id = tokenizer.eos_token_id
    newline = "\n"

    def allowed(_batch_id: int, input_ids: torch.Tensor) -> List[int]:
        gen_ids = input_ids.tolist()[base_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        completed = text.count(newline)
        if completed >= total_lines:
            return [eos_id]
        current_text = text.split(newline)[-1]
        current_ids = tokenizer(current_text, add_special_tokens=False).input_ids
        allowed_ids: List[int] = []
        for seq in line_ids[completed]:
            if len(current_ids) < len(seq) and seq[: len(current_ids)] == current_ids:
                allowed_ids.append(seq[len(current_ids)])
        if not allowed_ids:
            return [eos_id]
        return sorted(set(allowed_ids))

    return allowed


def constrained_generate_program(model: Any, tokenizer: Any, ex: Example, max_new_tokens: int) -> str:
    prompt = format_prompt(ex, "program_stack")
    line_texts = [[line + "\n" for line in constrained_candidate_lines(ex, i)] for i in range(len(ex.program))]
    line_ids = [[tokenizer(line, add_special_tokens=False).input_ids for line in choices] for choices in line_texts]
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    base_len = int(enc["input_ids"].shape[1])
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=token_prefix_allowed(tokenizer, line_ids, len(ex.program), base_len),
        )
    gen = out[0, base_len:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def generate_resample_valid(model: Any, tokenizer: Any, ex: Example, max_new_tokens: int, attempts: int) -> Tuple[str, int]:
    prompt = format_prompt(ex, "program_stack")
    first = ""
    for attempt in range(1, attempts + 1):
        do_sample = attempt > 1
        gen = generate_batch(model, tokenizer, [prompt], max_new_tokens, do_sample=do_sample, temperature=0.75)[0]
        if attempt == 1:
            first = gen
        ok, _, _ = execute_stack(gen)
        if ok:
            return gen, attempt
    return first, attempts


def eval_prompt_arm(eval_arm: str) -> str:
    if eval_arm == "answer_only":
        return "answer_only"
    if eval_arm.startswith("parse_then"):
        return "parse_emit"
    return "program_stack"


def target_for_eval_arm(eval_arm: str) -> str:
    if eval_arm == "answer_only":
        return "answer_only"
    if eval_arm.startswith("parse_then"):
        return "parse_emit"
    if eval_arm in {"oracle_parse_constrained", "gold_abi_constrained"}:
        return "oracle"
    return "program_stack"


def eval_arms_for_target(target: str) -> List[str]:
    if target == "answer_only":
        return ["answer_only"]
    if target == "program_stack":
        return ["program_stack_free", "program_stack_constrained", "program_stack_resample_valid"]
    if target == "parse_emit":
        return ["parse_then_emit_free", "parse_then_emit_constrained"]
    raise ValueError(target)


def generated_program_for_eval(ex: Example, generated: str, eval_arm: str) -> str:
    if eval_arm == "parse_then_emit_free":
        return program_section(generated)
    if eval_arm == "parse_then_emit_constrained":
        return render_program_from_parse(generated)
    if eval_arm == "oracle_parse_constrained":
        return render_program_from_parse(render_target(ex, "parse_emit"))
    if eval_arm == "gold_abi_constrained":
        return "\n".join(ex.program)
    return generated


def evaluate_model(
    config: RunConfig,
    model: Any,
    tokenizer: Any,
    examples: Sequence[Example],
    train_target: str,
    eval_arm: str,
    run_id: str,
    seed: int,
    trained: bool,
    free_reference: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if model is not None:
        model.eval()
    detail_rows: List[Dict[str, Any]] = []
    prompt_arm = eval_prompt_arm(eval_arm)
    prompts = [format_prompt(ex, prompt_arm) for ex in examples]
    generated: List[str] = []
    attempts_used: List[int] = []
    if eval_arm == "program_stack_constrained":
        for ex in examples:
            generated.append(constrained_generate_program(model, tokenizer, ex, config.max_new_tokens))
            attempts_used.append(1)
    elif eval_arm == "program_stack_resample_valid":
        for ex in examples:
            gen, attempts = generate_resample_valid(model, tokenizer, ex, config.max_new_tokens, config.resample_attempts)
            generated.append(gen)
            attempts_used.append(attempts)
    elif eval_arm == "oracle_parse_constrained":
        for ex in examples:
            generated.append(render_target(ex, "parse_emit"))
            attempts_used.append(0)
    elif eval_arm == "gold_abi_constrained":
        for ex in examples:
            generated.append("\n".join(ex.program))
            attempts_used.append(0)
    else:
        for i in range(0, len(prompts), config.eval_batch_size):
            generated.extend(generate_batch(model, tokenizer, prompts[i : i + config.eval_batch_size], config.max_new_tokens))
        attempts_used = [1] * len(generated)
    stats: Dict[str, List[float]] = {k: [] for k in ["primary", "exec", "valid", "final", "no_final", "exact_program", "parse_exact", "diverged", "constrained_only", "free_only", "both_correct", "neither_correct"]}
    by_family: Dict[str, List[float]] = {f: [] for f in FAMILIES}
    by_failure: Dict[str, int] = {}
    for ex, gen, attempts in zip(examples, generated, attempts_used):
        program_text = generated_program_for_eval(ex, gen, eval_arm)
        final = parse_final(gen)
        final_correct = clean_answer(final) == clean_answer(ex.answer) if final is not None else False
        valid_exec, exec_value, exec_err = (False, None, "")
        exec_correct = False
        exact_program = False
        if eval_arm != "answer_only":
            valid_exec, exec_value, exec_err = execute_stack(program_text)
            exec_correct = valid_exec and clean_answer(exec_value) == clean_answer(ex.answer)
            exact_program = canonical_program(program_text) == canonical_program("\n".join(ex.program))
        parse_is_exact = parse_exact(ex, gen) if eval_arm.startswith("parse_then") or eval_arm == "oracle_parse_constrained" else False
        primary = final_correct if eval_arm == "answer_only" else exec_correct
        failure = "n/a_answer_only" if eval_arm == "answer_only" else classify_failure(ex, program_text, valid_exec, exec_correct)
        by_failure[failure] = by_failure.get(failure, 0) + 1
        ref = free_reference.get(ex.example_id, {}) if free_reference else {}
        free_program = str(ref.get("program_text", ""))
        free_correct = bool(ref.get("exec_correct", False))
        diverged = bool(free_reference is not None and canonical_program(program_text) != canonical_program(free_program))
        stats["primary"].append(float(primary))
        stats["exec"].append(float(exec_correct))
        stats["valid"].append(float(valid_exec))
        stats["final"].append(float(final_correct))
        stats["no_final"].append(float(not has_final(gen)))
        stats["exact_program"].append(float(exact_program))
        stats["parse_exact"].append(float(parse_is_exact))
        stats["diverged"].append(float(diverged))
        stats["constrained_only"].append(float(diverged and exec_correct and not free_correct))
        stats["free_only"].append(float(diverged and free_correct and not exec_correct))
        stats["both_correct"].append(float(exec_correct and free_correct))
        stats["neither_correct"].append(float((not exec_correct) and (not free_correct)))
        by_family[ex.family].append(float(primary))
        detail_rows.append({
            "suite": config.suite,
            "run": run_id,
            "train_target": train_target,
            "arm": eval_arm,
            "seed": seed,
            "trained": int(trained),
            "split": ex.split,
            "family": ex.family,
            "depth": ex.depth,
            "example_id": ex.example_id,
            "answer": ex.answer,
            "generated": gen,
            "program_text": program_text,
            "target": render_target(ex, prompt_arm if prompt_arm in TRAIN_TARGETS else "program_stack"),
            "exec_value": exec_value,
            "exec_error": exec_err,
            "primary_correct": int(primary),
            "exec_correct": int(exec_correct),
            "final_correct": int(final_correct),
            "valid_exec": int(valid_exec),
            "exact_program": int(exact_program),
            "parse_exact": int(parse_is_exact),
            "has_final": int(has_final(gen)),
            "attempts_used": attempts,
            "diverged_from_free": int(diverged),
            "free_exec_correct": int(free_correct),
            "failure_type": failure,
            "gold_ops": " ".join(ex.op_names),
            "generated_ops": " ".join(canon_op_names(program_text)),
        })
    metric = {
        "suite": config.suite,
        "run": run_id,
        "train_target": train_target,
        "arm": eval_arm,
        "seed": seed,
        "trained": int(trained),
        "split": examples[0].split if examples else "",
        "depth": examples[0].depth if examples else 0,
        "template_shift": int(examples[0].split.startswith("eval_template")) if examples else 0,
        "n": len(examples),
        "primary_accuracy": float(np.mean(stats["primary"])) if examples else float("nan"),
        "exec_accuracy": float(np.mean(stats["exec"])) if examples else float("nan"),
        "valid_exec_rate": float(np.mean(stats["valid"])) if examples else float("nan"),
        "final_accuracy": float(np.mean(stats["final"])) if examples else float("nan"),
        "no_final_rate": float(np.mean(stats["no_final"])) if examples else float("nan"),
        "exact_program_rate": float(np.mean(stats["exact_program"])) if examples else float("nan"),
        "parse_exact_rate": float(np.mean(stats["parse_exact"])) if examples else float("nan"),
        "correct_given_valid": correct_given_valid(float(np.mean(stats["exec"])), float(np.mean(stats["valid"]))) if examples else float("nan"),
        "divergence_rate": float(np.mean(stats["diverged"])) if free_reference and examples else float("nan"),
        "constrained_only_rate": float(np.mean(stats["constrained_only"])) if free_reference and examples else float("nan"),
        "free_only_rate": float(np.mean(stats["free_only"])) if free_reference and examples else float("nan"),
        "both_correct_rate": float(np.mean(stats["both_correct"])) if free_reference and examples else float("nan"),
        "neither_correct_rate": float(np.mean(stats["neither_correct"])) if free_reference and examples else float("nan"),
        "mean_attempts": float(np.mean(attempts_used)) if attempts_used else float("nan"),
        "mean_new_tokens": float(np.mean([len(tokenizer(g, add_special_tokens=False).input_ids) for g in generated])) if generated else float("nan"),
    }
    for family, vals in by_family.items():
        metric[f"{family}_accuracy"] = float(np.mean(vals)) if vals else float("nan")
    for failure, count in by_failure.items():
        metric[f"failure_{failure}"] = count / max(1, len(examples))
    return metric, detail_rows


def cleanup_model(model: Any) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_config(args: argparse.Namespace) -> RunConfig:
    if args.suite == "smoke":
        train_n, eval_n, steps, seeds, arms = 36, 6, 2, [101], TRAIN_TARGETS
        batch, accum = 2, 1
    elif args.suite == "pilot":
        train_n, eval_n, steps, seeds, arms = 120, 8, 16, [101], TRAIN_TARGETS
        batch, accum = 2, 2
    else:
        train_n, eval_n, steps, seeds, arms = 180, 10, 32, [101, 202, 303, 404, 505], TRAIN_TARGETS
        batch, accum = 2, 2
    if args.train_n is not None:
        train_n = args.train_n
    if args.eval_n is not None:
        eval_n = args.eval_n
    if args.steps is not None:
        steps = args.steps
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.arms:
        arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    return RunConfig(
        run_name=args.run_name or f"{args.suite}_v1",
        suite=args.suite,
        model_name=args.model_name,
        seeds=seeds,
        arms=arms,
        train_n=train_n,
        eval_n=eval_n,
        train_steps=steps,
        batch_size=args.batch_size or batch,
        grad_accum=args.grad_accum or accum,
        lr=args.lr,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_new_tokens=args.max_new_tokens,
        resample_attempts=args.resample_attempts,
    )


def run_experiment(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = make_config(args)
    run_root = RUNS / config.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "config.json", config.__dict__)
    append_log(
        f"## Run `{config.run_name}`\n\n"
        f"- Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
        f"- Suite: `{config.suite}`\n"
        f"- Model: `{config.model_name}`\n"
        f"- Seeds: `{','.join(map(str, config.seeds))}`\n"
        f"- Training targets: `{','.join(config.arms)}`\n"
        f"- Training examples per seed: `{config.train_n}`\n"
        f"- Eval examples per split: `{config.eval_n}`\n"
        f"- Steps: `{config.train_steps}`\n"
        f"- Resample attempts: `{config.resample_attempts}`"
    )
    tokenizer = load_tokenizer(config.model_name)
    all_metrics: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []
    all_logs: List[Dict[str, Any]] = []
    start = time.time()
    eval_splits = [name for name, _, _ in DEPTH_SPLITS]
    for arm in config.arms:
        for seed in config.seeds:
            run_id = f"{config.run_name}_{arm}_s{seed}"
            model, logs = train_adapter(config, arm, seed, tokenizer, run_root / run_id)
            all_logs.extend(logs)
            for split in eval_splits:
                examples = make_split(seed + 777, split, config.eval_n)
                free_reference: Optional[Dict[str, Dict[str, Any]]] = None
                eval_arms = eval_arms_for_target(arm)
                ordered_eval_arms = sorted(eval_arms, key=lambda x: 0 if x.endswith("_free") or x == "answer_only" else 1)
                for eval_arm in ordered_eval_arms:
                    metric, details = evaluate_model(config, model, tokenizer, examples, arm, eval_arm, run_id, seed, True, free_reference)
                    all_metrics.append(metric)
                    all_details.extend(details)
                    log(f"{run_id} {split} {eval_arm}: primary={pct(metric['primary_accuracy'])} exec={pct(metric['exec_accuracy'])} valid={pct(metric['valid_exec_rate'])} cgv={pct(metric['correct_given_valid'])}")
                    if eval_arm in {"program_stack_free", "parse_then_emit_free"}:
                        free_reference = {
                            row["example_id"]: {
                                "program_text": row["program_text"],
                                "exec_correct": bool(row["exec_correct"]),
                            }
                            for row in details
                        }
            cleanup_model(model)
    for seed in config.seeds:
        for split in eval_splits:
            examples = make_split(seed + 777, split, config.eval_n)
            for eval_arm in ["oracle_parse_constrained", "gold_abi_constrained"]:
                run_id = f"{config.run_name}_{eval_arm}_s{seed}"
                metric, details = evaluate_model(config, None, tokenizer, examples, "oracle", eval_arm, run_id, seed, False)
                all_metrics.append(metric)
                all_details.extend(details)
                log(f"{run_id} {split}: exec={pct(metric['exec_accuracy'])} valid={pct(metric['valid_exec_rate'])}")
    write_csv(run_root / "metrics.csv", all_metrics)
    write_csv(run_root / "details.csv", all_details)
    write_csv(run_root / "train_log.csv", all_logs)
    append_log(
        f"Completed `{config.run_name}` in {time.time() - start:.1f}s.\n\n"
        f"- Metric rows: {len(all_metrics)}\n"
        f"- Detail rows: {len(all_details)}\n"
        f"- Training log rows: {len(all_logs)}"
    )
    analyze_all()


def read_all_csv(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(RUNS.glob(pattern)):
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    metric_cols = [
        c for c in metrics.columns
        if c.endswith("_accuracy")
        or c.endswith("_rate")
        or c.startswith("failure_")
        or c in {"valid_exec_rate", "mean_new_tokens", "mean_attempts", "exact_program_rate", "correct_given_valid"}
    ]
    rows = []
    for keys, sub in metrics.groupby(["suite", "arm", "split", "depth", "template_shift"], dropna=False):
        row = {"suite": keys[0], "arm": keys[1], "split": keys[2], "depth": keys[3], "template_shift": keys[4], "runs": len(sub), "n_total": int(sub["n"].sum())}
        for col in metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals):
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def primary_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    suites = list(summary["suite"].dropna().unique())
    suite = "main" if "main" in suites else suites[-1]
    return summary[summary["suite"].eq(suite)].copy()


def plot_depth(primary: pd.DataFrame) -> None:
    sub = primary[primary["split"].str.startswith("eval_comp") | primary["split"].eq("eval_indist_d1")].copy()
    if sub.empty:
        return
    plt.figure(figsize=(10.5, 5.8))
    for arm, g in sub.groupby("arm"):
        g = g.sort_values("depth")
        y = 100 * g["exec_accuracy_mean"].fillna(g["primary_accuracy_mean"])
        err = 100 * g.get("exec_accuracy_std", pd.Series([0] * len(g))).fillna(0)
        plt.errorbar(g["depth"], y, yerr=err, marker="o", capsize=4, label=arm)
    plt.xlabel("Procedure depth")
    plt.ylabel("Externally executed procedure accuracy (%)")
    plt.title("Procedure Execution by Composition Depth")
    plt.xticks(sorted(sub["depth"].unique()))
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "execution_by_depth.png", dpi=180)
    plt.close()


def plot_template(primary: pd.DataFrame) -> None:
    sub = primary[primary["split"].str.startswith("eval_template")].copy()
    if sub.empty:
        return
    plt.figure(figsize=(10.5, 5.8))
    for arm, g in sub.groupby("arm"):
        g = g.sort_values("depth")
        y = 100 * g["exec_accuracy_mean"].fillna(g["primary_accuracy_mean"])
        err = 100 * g.get("exec_accuracy_std", pd.Series([0] * len(g))).fillna(0)
        plt.errorbar(g["depth"], y, yerr=err, marker="o", capsize=4, label=arm)
    plt.xlabel("Procedure depth")
    plt.ylabel("Template-shift execution accuracy (%)")
    plt.title("Template Shift by Depth")
    plt.xticks(sorted(sub["depth"].unique()))
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "template_shift_by_depth.png", dpi=180)
    plt.close()


def plot_validity_execution(primary: pd.DataFrame) -> None:
    sub = primary[
        primary["split"].str.startswith("eval_comp")
        & primary["arm"].isin(["program_stack_free", "program_stack_constrained", "program_stack_resample_valid"])
    ].copy()
    if sub.empty:
        return
    plt.figure(figsize=(10.5, 5.8))
    for arm, g in sub.groupby("arm"):
        g = g.sort_values("depth")
        plt.plot(g["depth"], 100 * g["exec_accuracy_mean"], marker="o", label=f"{arm} execution")
        plt.plot(g["depth"], 100 * g["valid_exec_rate_mean"], marker="s", linestyle="--", label=f"{arm} valid")
    plt.xlabel("Procedure depth")
    plt.ylabel("Rate (%)")
    plt.title("Validity Is Not the Same as Correct Execution")
    plt.xticks(sorted(sub["depth"].unique()))
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "validity_vs_execution.png", dpi=180)
    plt.close()


def plot_divergence(primary: pd.DataFrame) -> None:
    sub = primary[
        primary["split"].str.startswith("eval_comp")
        & primary["arm"].isin(["program_stack_constrained", "program_stack_resample_valid", "parse_then_emit_constrained"])
    ].copy()
    if sub.empty or "divergence_rate_mean" not in sub:
        return
    plt.figure(figsize=(10.5, 5.8))
    for arm, g in sub.groupby("arm"):
        g = g.sort_values("depth")
        plt.plot(g["depth"], 100 * g["divergence_rate_mean"], marker="o", label=f"{arm} diverged")
        if "constrained_only_rate_mean" in g:
            plt.plot(g["depth"], 100 * g["constrained_only_rate_mean"], marker="s", linestyle="--", label=f"{arm} constrained-only correct")
        if "free_only_rate_mean" in g:
            plt.plot(g["depth"], 100 * g["free_only_rate_mean"], marker="x", linestyle=":", label=f"{arm} free-only correct")
    plt.xlabel("Procedure depth")
    plt.ylabel("Rate (%)")
    plt.title("Constrained/Resample Divergence From Free Decoding")
    plt.xticks(sorted(sub["depth"].unique()))
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "decoder_divergence.png", dpi=180)
    plt.close()


def plot_parse(primary: pd.DataFrame) -> None:
    sub = primary[primary["arm"].isin(["parse_then_emit_free", "parse_then_emit_constrained", "oracle_parse_constrained"])].copy()
    if sub.empty or "parse_exact_rate_mean" not in sub:
        return
    sub = sub[sub["split"].str.startswith("eval_comp") | sub["split"].eq("eval_indist_d1")]
    if sub.empty:
        return
    plt.figure(figsize=(10.5, 5.8))
    for arm, g in sub.groupby("arm"):
        g = g.sort_values("depth")
        y = g["parse_exact_rate_mean"] if arm != "oracle_parse_constrained" else g["exec_accuracy_mean"]
        plt.plot(g["depth"], 100 * y, marker="o", label=arm)
    plt.xlabel("Procedure depth")
    plt.ylabel("Exact parse / oracle execution (%)")
    plt.title("Parse Correctness by Depth")
    plt.xticks(sorted(sub["depth"].unique()))
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "parse_accuracy.png", dpi=180)
    plt.close()


def plot_failure(details: pd.DataFrame) -> None:
    if details.empty or "failure_type" not in details:
        return
    sub = details[(details["split"].str.startswith("eval_comp")) & (details["arm"].eq("program_stack_constrained"))].copy()
    if sub.empty:
        sub = details[(details["split"].str.startswith("eval_comp")) & (details["arm"].eq("program_stack_free"))].copy()
    if sub.empty:
        return
    counts = sub.groupby(["depth", "failure_type"]).size().reset_index(name="n")
    totals = counts.groupby("depth")["n"].transform("sum")
    counts["rate"] = counts["n"] / totals
    pivot = counts.pivot_table(index="depth", columns="failure_type", values="rate", fill_value=0).sort_index()
    ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 6))
    ax.set_ylabel("Fraction of generated procedures")
    ax.set_title("Failure Taxonomy on Composition Splits")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES / "failure_taxonomy.png", dpi=180)
    plt.close()


def plot_training(logs: pd.DataFrame) -> None:
    if logs.empty:
        return
    plt.figure(figsize=(11, 6))
    for run, g in logs.groupby("run"):
        label = run if len(run) < 42 else run[:39] + "..."
        plt.plot(g["step"], g["loss"], marker="o", linewidth=1.1, label=label)
    plt.xlabel("Training step")
    plt.ylabel("CE loss")
    plt.title("Training Curves")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "training_curves.png", dpi=180)
    plt.close()


def md_table(df: pd.DataFrame, cols: Sequence[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            v = row.get(col, "")
            if isinstance(v, float):
                if "accuracy" in col or "rate" in col or col.startswith("failure") or col == "correct_given_valid_mean":
                    vals.append(pct(v))
                else:
                    vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines)


def make_report(summary: pd.DataFrame, metrics: pd.DataFrame, details: pd.DataFrame, logs: pd.DataFrame) -> str:
    primary = primary_summary(summary)
    suite = primary["suite"].iloc[0] if not primary.empty else "none"

    def row_for(arm: str, split: str) -> Optional[pd.Series]:
        sub = primary[(primary["arm"].eq(arm)) & (primary["split"].eq(split))]
        return sub.iloc[0] if not sub.empty else None

    def gap(a: Optional[pd.Series], b: Optional[pd.Series], col: str = "exec_accuracy_mean") -> str:
        if a is None or b is None:
            return "n/a"
        return pct(float(a.get(col, float("nan"))) - float(b.get(col, float("nan"))))

    free_d6 = row_for("program_stack_free", "eval_comp_d6")
    constrained_d6 = row_for("program_stack_constrained", "eval_comp_d6")
    resample_d6 = row_for("program_stack_resample_valid", "eval_comp_d6")
    parse_d6 = row_for("parse_then_emit_constrained", "eval_comp_d6")
    answer_d6 = row_for("answer_only", "eval_comp_d6")
    constrained_t6 = row_for("program_stack_constrained", "eval_template_d6")
    free_t6 = row_for("program_stack_free", "eval_template_d6")
    oracle_d6 = row_for("oracle_parse_constrained", "eval_comp_d6")
    gold_d6 = row_for("gold_abi_constrained", "eval_comp_d6")

    lines: List[str] = []
    lines.append("# Qwen Constrained ABI Parser")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append("This standalone experiment tests whether a finite-state stack-ABI decoder and a canonical parse stage make a small model a more reliable compiler from natural language into executable procedures. The headline metric is external execution accuracy, not valid-program rate.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("Training examples contain one primitive operation. Evaluation uses held-out procedure depths 2, 3, 4, and 6, plus wording-shifted prompts at depths 2, 4, and 6. The task families are string, number, table, date, list, and path transformations.")
    lines.append("")
    lines.append("Three targets are trained: direct final answers, raw stack programs, and parse-plus-program outputs. The raw stack adapter is evaluated with free greedy decoding, finite-state constrained decoding, and a resample-to-valid baseline. The parse adapter is evaluated both by executing its free program section and by deterministically emitting a stack program from its parse block. Oracle parse and gold ABI sanity arms bound the decoder and interpreter.")
    lines.append("")
    lines.append("A valid-rate increase alone is pre-registered as insufficient. A useful constraint must improve execution accuracy while preserving correct-given-valid accuracy; otherwise the grammar merely forces wrong programs to become well formed.")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Primary suite: `{suite}`.")
    if not metrics.empty:
        main = metrics[metrics["suite"].eq(suite)]
        seeds = sorted(int(x) for x in main["seed"].dropna().unique()) if not main.empty else []
        lines.append(f"- Seeds: `{','.join(map(str, seeds))}`.")
        lines.append(f"- Evaluation rows: `{len(main)}` metric rows, `{int(main['n'].sum())}` scored examples across arms.")
    if not logs.empty and "suite" in logs:
        suite_logs = logs[logs["suite"].eq(suite)]
        if not suite_logs.empty:
            lines.append(f"- QLoRA update steps per adapter: `{int(suite_logs['step'].max())}`.")
    lines.append("- Large adapters are stored outside the experiment tree.")
    lines.append("")
    lines.append("## Primary Results")
    lines.append("")
    if free_d6 is not None and constrained_d6 is not None:
        lines.append(f"- Depth-6 standard execution: free raw stack {pct(free_d6['exec_accuracy_mean'])}; constrained raw stack {pct(constrained_d6['exec_accuracy_mean'])}; constraint delta {gap(constrained_d6, free_d6)}.")
        lines.append(f"- Depth-6 valid-rate/correct-given-valid: free valid {pct(free_d6['valid_exec_rate_mean'])}, cgv {pct(free_d6['correct_given_valid_mean'])}; constrained valid {pct(constrained_d6['valid_exec_rate_mean'])}, cgv {pct(constrained_d6['correct_given_valid_mean'])}.")
        if not metrics.empty:
            d6_rows = metrics[(metrics["suite"].eq(suite)) & (metrics["split"].eq("eval_comp_d6")) & (metrics["arm"].isin(["program_stack_free", "program_stack_constrained"]))]
            if not d6_rows.empty:
                piv = d6_rows.pivot_table(index="seed", columns="arm", values="exec_accuracy", aggfunc="mean")
                if {"program_stack_free", "program_stack_constrained"}.issubset(set(piv.columns)):
                    deltas = piv["program_stack_constrained"] - piv["program_stack_free"]
                    lines.append(f"- Depth-6 constrained raw stack beats free decoding on `{int((deltas > 0).sum())}/{len(deltas)}` seeds; mean per-seed delta {pct(deltas.mean())}.")
        if "constrained_only_rate_mean" in constrained_d6:
            lines.append(f"- Depth-6 divergence: constrained-only correct {pct(constrained_d6['constrained_only_rate_mean'])}; free-only correct {pct(constrained_d6['free_only_rate_mean'])}.")
    if resample_d6 is not None and free_d6 is not None:
        lines.append(f"- Depth-6 resample-to-valid execution: {pct(resample_d6['exec_accuracy_mean'])}; delta versus free {gap(resample_d6, free_d6)}; mean attempts {resample_d6.get('mean_attempts_mean', float('nan')):.2f}.")
    if parse_d6 is not None and free_d6 is not None:
        lines.append(f"- Depth-6 parse-then-emit execution: {pct(parse_d6['exec_accuracy_mean'])}; parse exactness {pct(parse_d6.get('parse_exact_rate_mean', float('nan')))}.")
    if answer_d6 is not None:
        lines.append(f"- Depth-6 direct-answer baseline: {pct(answer_d6['primary_accuracy_mean'])}.")
    if constrained_t6 is not None and constrained_d6 is not None:
        lines.append(f"- Template-shift depth-6 constrained execution: {pct(constrained_t6['exec_accuracy_mean'])}; drop from standard constrained depth-6 {pct(constrained_d6['exec_accuracy_mean'] - constrained_t6['exec_accuracy_mean'])}.")
    if free_t6 is not None and free_d6 is not None:
        lines.append(f"- Template-shift depth-6 free execution: {pct(free_t6['exec_accuracy_mean'])}; drop from standard free depth-6 {pct(free_d6['exec_accuracy_mean'] - free_t6['exec_accuracy_mean'])}.")
    if constrained_t6 is not None and free_t6 is not None:
        lines.append(f"- Template-shift depth-6 constraint delta over free: {gap(constrained_t6, free_t6)}.")
    if oracle_d6 is not None and gold_d6 is not None:
        lines.append(f"- Oracle parse and gold ABI depth-6 sanity: oracle parse {pct(oracle_d6['exec_accuracy_mean'])}; gold ABI {pct(gold_d6['exec_accuracy_mean'])}.")
    lines.append("")
    cols = [
        "arm", "split", "depth", "runs", "n_total",
        "exec_accuracy_mean", "exec_accuracy_std", "valid_exec_rate_mean",
        "correct_given_valid_mean", "parse_exact_rate_mean",
        "divergence_rate_mean", "constrained_only_rate_mean", "free_only_rate_mean",
        "mean_attempts_mean",
    ]
    available_cols = [c for c in cols if c in primary.columns]
    rows = primary.sort_values(["split", "arm"])[available_cols] if not primary.empty else pd.DataFrame()
    lines.append(md_table(rows, available_cols))
    lines.append("")
    lines.append("![Execution by depth](../analysis/figures/execution_by_depth.png)")
    lines.append("")
    lines.append("![Template shift by depth](../analysis/figures/template_shift_by_depth.png)")
    lines.append("")
    lines.append("![Validity versus execution](../analysis/figures/validity_vs_execution.png)")
    lines.append("")
    lines.append("![Decoder divergence](../analysis/figures/decoder_divergence.png)")
    lines.append("")
    lines.append("![Parse accuracy](../analysis/figures/parse_accuracy.png)")
    lines.append("")
    lines.append("![Failure taxonomy](../analysis/figures/failure_taxonomy.png)")
    lines.append("")
    lines.append("![Training curves](../analysis/figures/training_curves.png)")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The experiment separates two possible bottlenecks. If constrained decoding increases validity and execution together, malformed syntax was suppressing an otherwise useful compiler. If validity rises while execution stays flat or correct-given-valid falls, malformed syntax was mainly a symptom of unresolved semantic uncertainty. The divergence diagnostics show whether the grammar rescues examples free decoding missed or overrides examples free decoding already had right.")
    if free_d6 is not None and constrained_d6 is not None:
        lines.append(f"At depth 6, constrained decoding changes execution by {gap(constrained_d6, free_d6)} relative to free raw-stack decoding. The valid-rate change is {gap(constrained_d6, free_d6, 'valid_exec_rate_mean')}, and the correct-given-valid change is {gap(constrained_d6, free_d6, 'correct_given_valid_mean')}.")
    if resample_d6 is not None and constrained_d6 is not None:
        lines.append(f"The resample-to-valid baseline is the cheap alternative. At depth 6 it is {gap(resample_d6, constrained_d6)} versus the constrained decoder, so this comparison determines whether a full grammar adds value beyond simply rejecting invalid samples.")
    if parse_d6 is not None and constrained_t6 is not None:
        parse_free_d6 = row_for("parse_then_emit_free", "eval_comp_d6")
        parse_t6 = row_for("parse_then_emit_constrained", "eval_template_d6")
        if parse_free_d6 is not None:
            lines.append(f"The parse stage helps standard depth-6 execution relative to its own free program section: {pct(parse_d6['exec_accuracy_mean'])} versus {pct(parse_free_d6['exec_accuracy_mean'])}.")
        if parse_t6 is not None:
            lines.append(f"But the parse stage does not solve wording shift in this form: template depth-6 parse-then-emit is {pct(parse_t6['exec_accuracy_mean'])}, far below constrained raw stack at {pct(constrained_t6['exec_accuracy_mean'])}.")
    if not details.empty and "failure_type" in details:
        fail = details[(details["suite"].eq(suite)) & (details["split"].str.startswith("eval_comp")) & (details["arm"].eq("program_stack_constrained"))]
        if not fail.empty:
            counts = fail.groupby("failure_type").size().sort_values(ascending=False)
            bits = [f"{k} {pct(v / len(fail))}" for k, v in counts.items()]
            lines.append(f"For constrained raw-stack decoding on composition splits, generated procedures break down as: {', '.join(bits)}.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("This experiment tests robustness of compilation over a fixed known primitive library. It does not test invention of operations outside the ABI. The finite-state grammar is tied to the synthetic task schema and uses task-visible constants and type information.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Metrics: `analysis/summary_by_arm.csv` and `analysis/all_metrics.csv`")
    lines.append("- Details: `analysis/all_details.csv`")
    lines.append("- Training logs: `analysis/all_train_logs.csv`")
    lines.append("- Checkpoints: `/workspace/large_artifacts/qwen_constrained_abi_parser/checkpoints`")
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown_text: str) -> str:
    body: List[str] = []
    in_table = False
    in_ul = False
    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_ul:
                body.append("<ul>")
                in_ul = True
            body.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.startswith("![") and "](" in line:
            if in_ul:
                body.append("</ul>")
                in_ul = False
            alt = line[2 : line.index("]")]
            src = line[line.index("(") + 1 : line.rindex(")")]
            body.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.startswith("|") and line.endswith("|"):
            if in_ul:
                body.append("</ul>")
                in_ul = False
            cells = [html.escape(c.strip()) for c in line.strip("|").split("|")]
            if all(set(c) <= {"-"} for c in cells):
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            body.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
        else:
            if in_table:
                body.append("</table>")
                in_table = False
            if in_ul:
                body.append("</ul>")
                in_ul = False
            body.append(f"<p>{html.escape(line)}</p>" if line else "")
    if in_table:
        body.append("</table>")
    if in_ul:
        body.append("</ul>")
    css = "body{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1160px;margin:36px auto;line-height:1.5;color:#202124}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d4d7dc;padding:6px 8px;text-align:left}th{background:#f2f4f7}img{max-width:100%;border:1px solid #d4d7dc}figure{margin:24px 0}figcaption{color:#5f6368;font-size:13px}code{background:#f5f7fa;padding:1px 4px;border-radius:4px}"
    return "<!doctype html><html><head><meta charset='utf-8'><title>Qwen Constrained ABI Parser</title><style>" + css + "</style></head><body>" + "\n".join(body) + "</body></html>\n"


def write_checkpoint_manifest() -> None:
    rows = []
    for path in sorted(CHECKPOINTS.glob("*")):
        if path.is_dir():
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            rows.append({"checkpoint": path.name, "path": str(path), "bytes": total})
    write_csv(ROOT / "checkpoint_manifest.csv", rows)


def analyze_all() -> None:
    ensure_dirs()
    metrics = read_all_csv("*/metrics.csv")
    details = read_all_csv("*/details.csv")
    logs = read_all_csv("*/train_log.csv")
    summary = summarize(metrics)
    metrics.to_csv(ANALYSIS / "all_metrics.csv", index=False)
    details.to_csv(ANALYSIS / "all_details.csv", index=False)
    logs.to_csv(ANALYSIS / "all_train_logs.csv", index=False)
    summary.to_csv(ANALYSIS / "summary_by_arm.csv", index=False)
    primary = primary_summary(summary)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_depth(primary)
    plot_template(primary)
    plot_validity_execution(primary)
    plot_divergence(primary)
    plot_parse(primary)
    plot_failure(details[details["suite"].eq(primary["suite"].iloc[0])] if not details.empty and not primary.empty else details)
    plot_training(logs[logs["suite"].eq(primary["suite"].iloc[0])] if not logs.empty and not primary.empty else logs)
    report = make_report(summary, metrics, details, logs)
    (REPORTS / "qwen_constrained_abi_parser_report.md").write_text(report)
    (REPORTS / "qwen_constrained_abi_parser_report.html").write_text(markdown_to_html(report))
    write_checkpoint_manifest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["smoke", "pilot", "main", "analyze"], default="smoke")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--arms", default="")
    parser.add_argument("--train_n", type=int, default=None)
    parser.add_argument("--eval_n", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=448)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--resample_attempts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.suite == "analyze":
        analyze_all()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
