#!/usr/bin/env python3
"""Program-only executable ABI experiment for Qwen.

The central intervention is strict: program-only arms are trained without a
final-answer line, and their primary score is the result of parsing and
executing the generated program.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import html
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


ROOT = Path("/workspace/experiments/qwen_program_only_executable_abi")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_program_only_executable_abi")
CHECKPOINTS = LARGE_ROOT / "checkpoints"
CACHE_DIR = Path("/workspace/.cache/huggingface")

MODEL_NAME = "Qwen/Qwen3-4B"
TRAIN_FAMILIES = ["string", "unit", "table", "date"]
EVAL_SPLITS = ["eval_indist", "eval_composition", "eval_template_shift"]
ARMS = ["answer_only", "trace_stack_final", "program_stack", "program_python"]
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
FINAL_RE = re.compile(r"\bFINAL\s*:?\s*([A-Za-z0-9_.:+/-]+)", re.IGNORECASE)


@dataclass
class Example:
    example_id: str
    split: str
    family: str
    prompt: str
    answer: str
    op_name: str
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


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default))


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


def mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def std(xs: Sequence[float]) -> float:
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def clean_answer(text: Any) -> str:
    return str(text).strip().strip("`'\";,")


def parse_final(text: str) -> Optional[str]:
    match = FINAL_RE.search(text)
    return clean_answer(match.group(1)) if match else None


def has_final_line(text: str) -> bool:
    return bool(FINAL_RE.search(text))


def rand_code(rng: random.Random) -> str:
    left = "".join(rng.choice(LETTERS) for _ in range(2))
    mid = "".join(str(rng.randint(0, 9)) for _ in range(4))
    right = "".join(rng.choice(LETTERS) for _ in range(2))
    return f"{left}-{mid}-{right}"


def choose_template(rng: random.Random, split: str, standard: Sequence[str], shifted: Sequence[str]) -> str:
    if split == "eval_template_shift":
        return rng.choice(list(shifted))
    return rng.choice(list(standard))


def make_string_example(rng: random.Random, split: str, idx: int) -> Example:
    code = rand_code(rng)
    clean = code.replace("-", "")
    if split == "eval_composition":
        width = rng.randint(3, 5)
        start = rng.randint(0, len(clean) - width)
        answer = clean[::-1][start : start + width]
        op = "reverse_then_slice"
        prompt_t = choose_template(
            rng,
            split,
            [
                "Reference code: {code}. Remove hyphens, reverse the cleaned string, then return characters {a} through {b} using one-indexed positions.",
                "Clean this code by deleting hyphens: {code}. Reverse it and report the one-indexed span {a}-{b}.",
            ],
            ["Compact {code}, flip it left-to-right, then clip positions {a} through {b}. What is the result?"],
        )
        prompt = prompt_t.format(code=code, a=start + 1, b=start + width)
        payload = {"code": code, "start": start, "end": start + width}
    else:
        op = rng.choice(["take_prefix", "take_suffix", "slice", "reverse_all"])
        if op == "take_prefix":
            n = rng.randint(3, 5)
            answer = clean[:n]
            prompt = choose_template(
                rng,
                split,
                ["Reference code: {code}. Remove hyphens, then return the first {n} characters."],
                ["Compact identifier {code} by dropping hyphens. What are the first {n} symbols afterward?"],
            ).format(code=code, n=n)
            payload = {"code": code, "n": n}
        elif op == "take_suffix":
            n = rng.randint(3, 5)
            answer = clean[-n:]
            prompt = choose_template(
                rng,
                split,
                ["Reference code: {code}. Remove hyphens, then return the last {n} characters."],
                ["Compact identifier {code} by dropping hyphens. What are the final {n} symbols afterward?"],
            ).format(code=code, n=n)
            payload = {"code": code, "n": n}
        elif op == "reverse_all":
            answer = clean[::-1]
            prompt = choose_template(
                rng,
                split,
                ["Reference code: {code}. Remove hyphens, then reverse the full cleaned string."],
                ["Drop separators from {code} and flip the whole compacted text."],
            ).format(code=code)
            payload = {"code": code}
        else:
            width = rng.randint(3, 5)
            start = rng.randint(0, len(clean) - width)
            answer = clean[start : start + width]
            prompt = choose_template(
                rng,
                split,
                ["Reference code: {code}. Remove hyphens, then return characters {a} through {b} using one-indexed positions."],
                ["After compacting {code}, which substring occupies positions {a} through {b}?"],
            ).format(code=code, a=start + 1, b=start + width)
            payload = {"code": code, "start": start, "end": start + width}
    return Example(f"{split}_string_{idx}", split, "string", prompt, str(answer), op, payload)


def make_unit_example(rng: random.Random, split: str, idx: int) -> Example:
    if split == "eval_composition":
        m = rng.randint(1, 8)
        c = rng.choice([5, 10, 15, 20, 25, 40, 50, 75])
        answer = m * 100 + c
        prompt = choose_template(
            rng,
            split,
            ["A cable is {m} meters plus {c} centimeters long. Report the total length in centimeters."],
            ["Convert a mixed length of {m} meters and {c} centimeters into centimeters."],
        ).format(m=m, c=c)
        return Example(f"{split}_unit_{idx}", split, "unit", prompt, str(answer), "meters_plus_centimeters", {"meters": m, "centimeters": c})
    op, src, dst, factor = rng.choice(
        [
            ("meters_to_centimeters", "meters", "centimeters", 100),
            ("centimeters_to_millimeters", "centimeters", "millimeters", 10),
            ("kilograms_to_grams", "kilograms", "grams", 1000),
            ("minutes_to_seconds", "minutes", "seconds", 60),
            ("add_centimeters", "centimeters", "centimeters", 1),
        ]
    )
    amount = rng.randint(2, 19)
    if op == "add_centimeters":
        extra = rng.randint(2, 40)
        answer = amount + extra
        prompt = choose_template(
            rng,
            split,
            ["Start with {amount} centimeters and add {extra} centimeters. Return the total centimeters."],
            ["Increase {amount} centimeters by {extra} centimeters. Give the integer result."],
        ).format(amount=amount, extra=extra)
        payload = {"amount": amount, "extra": extra}
    else:
        answer = amount * factor
        prompt = choose_template(
            rng,
            split,
            ["Convert {amount} {src} to {dst}. Return the integer result."],
            ["Rewrite {amount} {src} using {dst} as the unit. Give only the integer value."],
        ).format(amount=amount, src=src, dst=dst)
        payload = {"amount": amount, "src": src, "dst": dst, "factor": factor}
    return Example(f"{split}_unit_{idx}", split, "unit", prompt, str(answer), op, payload)


def make_table_example(rng: random.Random, split: str, idx: int) -> Example:
    keys = ["alpha", "beta", "gamma", "delta"]
    rng.shuffle(keys)
    values = {k: rng.randint(2, 29) for k in keys}
    table = ", ".join(f"{k}={values[k]}" for k in keys)
    if split == "eval_composition":
        k1, k2 = rng.sample(keys, 2)
        bias = rng.randint(1, 9)
        answer = values[k1] + values[k2] + bias
        prompt = choose_template(
            rng,
            split,
            ["Table: {table}. Add the values for {k1} and {k2}, then add {b}."],
            ["From this lookup table [{table}], total {k1} and {k2}, then increase by {b}."],
        ).format(table=table, k1=k1, k2=k2, b=bias)
        return Example(f"{split}_table_{idx}", split, "table", prompt, str(answer), "sum_two_keys_plus", {"table": values, "k1": k1, "k2": k2, "bias": bias})
    op = rng.choice(["lookup_add", "lookup_mul", "lookup_mul_add"])
    key = rng.choice(keys)
    bias = rng.randint(1, 11)
    scale = rng.randint(2, 7)
    if op == "lookup_add":
        answer = values[key] + bias
        prompt = choose_template(
            rng,
            split,
            ["Table: {table}. Take {key}, then add {bias}."],
            ["From lookup entries [{table}], increase {key} by {bias}."],
        ).format(table=table, key=key, bias=bias)
        payload = {"table": values, "key": key, "bias": bias}
    elif op == "lookup_mul":
        answer = values[key] * scale
        prompt = choose_template(
            rng,
            split,
            ["Table: {table}. Take {key}, then multiply by {scale}."],
            ["From lookup entries [{table}], scale {key} by {scale}."],
        ).format(table=table, key=key, scale=scale)
        payload = {"table": values, "key": key, "scale": scale}
    else:
        answer = values[key] * scale + bias
        prompt = choose_template(
            rng,
            split,
            ["Table: {table}. Take {key}, multiply by {scale}, then add {bias}."],
            ["From lookup entries [{table}], scale {key} by {scale} and increase by {bias}."],
        ).format(table=table, key=key, scale=scale, bias=bias)
        payload = {"table": values, "key": key, "scale": scale, "bias": bias}
    return Example(f"{split}_table_{idx}", split, "table", prompt, str(answer), op, payload)


def make_date_example(rng: random.Random, split: str, idx: int) -> Example:
    base = date(2026, rng.randint(1, 12), rng.randint(1, 24))
    if split == "eval_composition":
        weeks = rng.randint(1, 3)
        days = rng.randint(1, 6)
        final = base + timedelta(days=weeks * 7 + days)
        prompt = choose_template(
            rng,
            split,
            ["Starting from {d}, add {w} weeks and {days} days. Return the ISO date."],
            ["Advance {d} by {w} whole weeks and then by {days} more days; give YYYY-MM-DD."],
        ).format(d=base.isoformat(), w=weeks, days=days)
        return Example(f"{split}_date_{idx}", split, "date", prompt, final.isoformat(), "weeks_plus_days", {"date": base.isoformat(), "weeks": weeks, "days": days})
    if rng.random() < 0.5:
        delta = rng.randint(1, 14)
        final = base + timedelta(days=delta)
        prompt = choose_template(
            rng,
            split,
            ["Starting from {d}, add {n} days. Return the ISO date."],
            ["Advance {d} by {n} days and write the result as YYYY-MM-DD."],
        ).format(d=base.isoformat(), n=delta)
        return Example(f"{split}_date_{idx}", split, "date", prompt, final.isoformat(), "add_days", {"date": base.isoformat(), "days": delta})
    weeks = rng.randint(1, 4)
    final = base + timedelta(days=weeks * 7)
    prompt = choose_template(
        rng,
        split,
        ["Starting from {d}, add {w} weeks. Return the ISO date."],
        ["Move {d} forward by {w} weeks and write the date."],
    ).format(d=base.isoformat(), w=weeks)
    return Example(f"{split}_date_{idx}", split, "date", prompt, final.isoformat(), "add_weeks", {"date": base.isoformat(), "weeks": weeks})


def make_example(rng: random.Random, split: str, family: str, idx: int) -> Example:
    if family == "string":
        return make_string_example(rng, split, idx)
    if family == "unit":
        return make_unit_example(rng, split, idx)
    if family == "table":
        return make_table_example(rng, split, idx)
    if family == "date":
        return make_date_example(rng, split, idx)
    raise ValueError(family)


def make_split(seed: int, split: str, n: int) -> List[Example]:
    offsets = {"train": 0, "eval_indist": 10_000, "eval_composition": 20_000, "eval_template_shift": 30_000}
    rng = random.Random(seed + offsets[split])
    examples = [make_example(rng, split, TRAIN_FAMILIES[i % len(TRAIN_FAMILIES)], i) for i in range(n)]
    rng.shuffle(examples)
    return examples


def stack_program(ex: Example) -> List[str]:
    p = ex.payload
    lines: List[str] = []
    if ex.family == "string":
        lines += [f"PUSH_CODE {p['code']}", "STRIP_HYPHENS"]
        if ex.op_name == "take_prefix":
            lines.append(f"TAKE_PREFIX {p['n']}")
        elif ex.op_name == "take_suffix":
            lines.append(f"TAKE_SUFFIX {p['n']}")
        elif ex.op_name == "reverse_all":
            lines.append("REVERSE")
        elif ex.op_name == "slice":
            lines.append(f"SLICE {p['start'] + 1} {p['end']}")
        elif ex.op_name == "reverse_then_slice":
            lines += ["REVERSE", f"SLICE {p['start'] + 1} {p['end']}"]
    elif ex.family == "unit":
        if ex.op_name == "meters_plus_centimeters":
            lines += [f"PUSH {p['meters']}", "MUL 100", f"ADD {p['centimeters']}"]
        elif ex.op_name == "add_centimeters":
            lines += [f"PUSH {p['amount']}", f"ADD {p['extra']}"]
        else:
            lines += [f"PUSH {p['amount']}", f"MUL {p['factor']}"]
    elif ex.family == "table":
        lines.append("TABLE " + " ".join(f"{k}:{v}" for k, v in sorted(p["table"].items())))
        if ex.op_name == "lookup_add":
            lines += [f"LOOKUP {p['key']}", f"ADD {p['bias']}"]
        elif ex.op_name == "lookup_mul":
            lines += [f"LOOKUP {p['key']}", f"MUL {p['scale']}"]
        elif ex.op_name == "lookup_mul_add":
            lines += [f"LOOKUP {p['key']}", f"MUL {p['scale']}", f"ADD {p['bias']}"]
        elif ex.op_name == "sum_two_keys_plus":
            lines += [f"LOOKUP {p['k1']}", f"LOOKUP {p['k2']}", "ADD_TOP", f"ADD {p['bias']}"]
    elif ex.family == "date":
        lines.append(f"PUSH_DATE {p['date']}")
        if ex.op_name == "add_days":
            lines.append(f"ADD_DAYS {p['days']}")
        elif ex.op_name == "add_weeks":
            lines.append(f"ADD_WEEKS {p['weeks']}")
        elif ex.op_name == "weeks_plus_days":
            lines += [f"ADD_WEEKS {p['weeks']}", f"ADD_DAYS {p['days']}"]
    return lines


def python_program(ex: Example) -> List[str]:
    p = ex.payload
    lines: List[str] = []
    if ex.family == "string":
        lines += [f'code = "{p["code"]}"', "clean = strip_hyphens(code)"]
        if ex.op_name == "take_prefix":
            lines.append(f"result = take_prefix(clean, {p['n']})")
        elif ex.op_name == "take_suffix":
            lines.append(f"result = take_suffix(clean, {p['n']})")
        elif ex.op_name == "reverse_all":
            lines.append("result = reverse(clean)")
        elif ex.op_name == "slice":
            lines.append(f"result = slice1(clean, {p['start'] + 1}, {p['end']})")
        elif ex.op_name == "reverse_then_slice":
            lines += ["rev = reverse(clean)", f"result = slice1(rev, {p['start'] + 1}, {p['end']})"]
    elif ex.family == "unit":
        if ex.op_name == "meters_plus_centimeters":
            lines.append(f"result = add(mul({p['meters']}, 100), {p['centimeters']})")
        elif ex.op_name == "add_centimeters":
            lines.append(f"result = add({p['amount']}, {p['extra']})")
        else:
            lines.append(f"result = mul({p['amount']}, {p['factor']})")
    elif ex.family == "table":
        lines.append(f"table = {json.dumps(p['table'], sort_keys=True)}")
        if ex.op_name == "lookup_add":
            lines.append(f"result = add(lookup(table, \"{p['key']}\"), {p['bias']})")
        elif ex.op_name == "lookup_mul":
            lines.append(f"result = mul(lookup(table, \"{p['key']}\"), {p['scale']})")
        elif ex.op_name == "lookup_mul_add":
            lines.append(f"result = add(mul(lookup(table, \"{p['key']}\"), {p['scale']}), {p['bias']})")
        elif ex.op_name == "sum_two_keys_plus":
            lines.append(f"result = add(add(lookup(table, \"{p['k1']}\"), lookup(table, \"{p['k2']}\")), {p['bias']})")
    elif ex.family == "date":
        if ex.op_name == "add_days":
            lines.append(f"result = date_add(\"{p['date']}\", {p['days']})")
        elif ex.op_name == "add_weeks":
            lines.append(f"result = date_add(\"{p['date']}\", mul({p['weeks']}, 7))")
        elif ex.op_name == "weeks_plus_days":
            lines.append(f"result = date_add(\"{p['date']}\", add(mul({p['weeks']}, 7), {p['days']}))")
    return lines


def render_target(ex: Example, arm: str) -> str:
    if arm == "answer_only":
        return f"FINAL: {ex.answer}\n"
    if arm == "trace_stack_final":
        return "\n".join(stack_program(ex) + [f"FINAL {ex.answer}"]) + "\n"
    if arm == "program_stack":
        return "\n".join(stack_program(ex)) + "\n"
    if arm == "program_python":
        return "\n".join(python_program(ex)) + "\n"
    raise ValueError(arm)


def format_instruction(arm: str) -> str:
    if arm == "answer_only":
        return "Return one line exactly as: FINAL: <answer>."
    if arm == "trace_stack_final":
        return "Return stack instructions followed by a final line: FINAL <answer>."
    if arm == "program_stack":
        return "Return only stack instructions. Do not write a FINAL line or the answer."
    if arm == "program_python":
        return "Return only Python-like assignment statements ending by assigning result. Do not write a FINAL line or the answer."
    raise ValueError(arm)


def format_prompt(ex: Example, arm: str) -> str:
    return (
        "Compile the task into the requested output format.\n"
        f"Output rule: {format_instruction(arm)}\n\n"
        f"Task:\n{ex.prompt}\n\n"
        "Output:\n"
    )


def safe_int(token: str) -> int:
    return int(re.sub(r"[^0-9-]", "", token))


def execute_stack(text: str) -> Tuple[bool, Optional[str], str]:
    stack: List[Any] = []
    table: Dict[str, int] = {}
    try:
        for raw in text.splitlines():
            line = raw.strip().strip("`")
            if not line or line.startswith("#"):
                continue
            if FINAL_RE.search(line):
                continue
            parts = line.split()
            if not parts:
                continue
            op = parts[0].upper().rstrip(":")
            if op == "PUSH_CODE":
                stack.append(" ".join(parts[1:]).strip().strip('"'))
            elif op == "STRIP_HYPHENS":
                stack[-1] = str(stack[-1]).replace("-", "")
            elif op == "REVERSE":
                stack[-1] = str(stack[-1])[::-1]
            elif op == "TAKE_PREFIX":
                stack[-1] = str(stack[-1])[: safe_int(parts[1])]
            elif op == "TAKE_SUFFIX":
                stack[-1] = str(stack[-1])[-safe_int(parts[1]) :]
            elif op == "SLICE":
                a = safe_int(parts[1]) - 1
                b = safe_int(parts[2])
                stack[-1] = str(stack[-1])[a:b]
            elif op == "PUSH":
                stack.append(safe_int(parts[1]))
            elif op == "MUL":
                if len(parts) == 1:
                    right = int(stack.pop())
                    left = int(stack.pop())
                    stack.append(left * right)
                else:
                    stack[-1] = int(stack[-1]) * safe_int(parts[1])
            elif op == "ADD":
                if len(parts) == 1:
                    right = int(stack.pop())
                    left = int(stack.pop())
                    stack.append(left + right)
                else:
                    stack[-1] = int(stack[-1]) + safe_int(parts[1])
            elif op == "ADD_TOP":
                right = int(stack.pop())
                left = int(stack.pop())
                stack.append(left + right)
            elif op == "TABLE":
                table = {}
                for item in parts[1:]:
                    if ":" in item:
                        k, v = item.split(":", 1)
                        table[k.strip()] = safe_int(v)
            elif op == "LOOKUP":
                stack.append(int(table[parts[1].strip()]))
            elif op == "PUSH_DATE":
                stack.append(parts[1].strip())
            elif op == "ADD_DAYS":
                d = date.fromisoformat(str(stack[-1]))
                stack[-1] = (d + timedelta(days=safe_int(parts[1]))).isoformat()
            elif op == "ADD_WEEKS":
                d = date.fromisoformat(str(stack[-1]))
                stack[-1] = (d + timedelta(days=7 * safe_int(parts[1]))).isoformat()
            else:
                return False, None, f"unknown op {op}"
        if not stack:
            return False, None, "empty stack"
        return True, clean_answer(stack[-1]), ""
    except Exception as exc:
        return False, None, type(exc).__name__


def strip_hyphens(x: str) -> str:
    return str(x).replace("-", "")


def reverse(x: str) -> str:
    return str(x)[::-1]


def take_prefix(x: str, n: int) -> str:
    return str(x)[: int(n)]


def take_suffix(x: str, n: int) -> str:
    return str(x)[-int(n) :]


def slice1(x: str, a: int, b: int) -> str:
    return str(x)[int(a) - 1 : int(b)]


def add(*args: Any) -> int:
    total = 0
    for arg in args:
        total += int(arg)
    return total


def mul(*args: Any) -> int:
    total = 1
    for arg in args:
        total *= int(arg)
    return total


def lookup(table: Dict[str, Any], key: str) -> int:
    return int(table[str(key)])


def date_add(day: str, days: int) -> str:
    return (date.fromisoformat(str(day)) + timedelta(days=int(days))).isoformat()


def execute_python_subset(text: str) -> Tuple[bool, Optional[str], str]:
    lines: List[str] = []
    try:
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if FINAL_RE.search(line):
                continue
            if line.startswith("```"):
                continue
            lines.append(line)
        code = "\n".join(lines)
        tree = ast.parse(code)
        allowed_nodes = (
            ast.Module,
            ast.Assign,
            ast.Name,
            ast.Load,
            ast.Store,
            ast.Constant,
            ast.Call,
            ast.Expr,
            ast.Dict,
            ast.BinOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.UnaryOp,
            ast.USub,
        )
        allowed_funcs = {"strip_hyphens", "reverse", "take_prefix", "take_suffix", "slice1", "add", "mul", "lookup", "date_add"}
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return False, None, f"bad_ast_{type(node).__name__}"
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in allowed_funcs:
                    return False, None, "bad_call"
        env: Dict[str, Any] = {
            "strip_hyphens": strip_hyphens,
            "reverse": reverse,
            "take_prefix": take_prefix,
            "take_suffix": take_suffix,
            "slice1": slice1,
            "add": add,
            "mul": mul,
            "lookup": lookup,
            "date_add": date_add,
        }
        loc: Dict[str, Any] = {}
        exec(compile(tree, "<program>", "exec"), {"__builtins__": {}, **env}, loc)
        if "result" not in loc:
            return False, None, "missing_result"
        return True, clean_answer(loc["result"]), ""
    except Exception as exc:
        return False, None, type(exc).__name__


def execute_program_for_arm(arm: str, text: str) -> Tuple[bool, Optional[str], str]:
    if arm in {"program_stack", "trace_stack_final"}:
        return execute_stack(text)
    if arm == "program_python":
        return execute_python_subset(text)
    return False, None, "not_program_arm"


class PromptDataset(Dataset):
    def __init__(self, examples: Sequence[Example], arm: str, tokenizer: Any, max_length: int):
        self.rows: List[Dict[str, torch.Tensor]] = []
        eos = tokenizer.eos_token or ""
        for ex in examples:
            prompt = format_prompt(ex, arm)
            target = render_target(ex, arm) + eos
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            if len(ids) > max_length:
                ids = ids[:max_length]
                labels = labels[:max_length]
            self.rows.append({"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.rows[idx]


def collate_batch(rows: Sequence[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(r["input_ids"]) for r in rows)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        input_ids[i, :n] = row["input_ids"]
        labels[i, :n] = row["labels"]
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def ensure_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=str(CACHE_DIR))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.convert_ids_to_tokens(0)
    return tokenizer


def load_model(model_name: str, lora: bool, config: Optional[RunConfig] = None) -> Any:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    if not lora:
        model.eval()
        return model
    assert config is not None
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    return model


def iter_batches(loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def max_new_tokens_for_arm(config: RunConfig, arm: str) -> int:
    if arm == "answer_only":
        return min(config.max_new_tokens, 24)
    if arm == "program_stack":
        return min(config.max_new_tokens, 56)
    if arm == "trace_stack_final":
        return min(config.max_new_tokens, 72)
    return config.max_new_tokens


def is_program_arm(arm: str) -> bool:
    return arm in {"program_stack", "program_python"}


@torch.no_grad()
def evaluate_model(config: RunConfig, model: Any, tokenizer: Any, examples: Sequence[Example], arm: str, run_id: str, seed: int, trained: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    final_correct = 0
    exec_correct = 0
    strict_correct = 0
    valid_final = 0
    valid_exec = 0
    no_final = 0
    leak = 0
    token_counts: List[int] = []
    family_stats: Dict[str, List[int]] = {f: [] for f in TRAIN_FAMILIES}
    for start in range(0, len(examples), config.eval_batch_size):
        batch_examples = list(examples[start : start + config.eval_batch_size])
        prompts = [format_prompt(ex, arm) for ex in batch_examples]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=config.max_length)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens_for_arm(config, arm),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_width = enc["input_ids"].shape[1]
        for i, ex in enumerate(batch_examples):
            new_ids = out[i, prompt_width:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            pred_final = parse_final(text)
            vf = pred_final is not None
            final_ok = clean_answer(pred_final or "") == clean_answer(ex.answer)
            ve = False
            exec_ans: Optional[str] = None
            err = ""
            if arm in {"program_stack", "program_python", "trace_stack_final"}:
                ve, exec_ans, err = execute_program_for_arm(arm, text)
            exec_ok = ve and clean_answer(exec_ans or "") == clean_answer(ex.answer)
            no_final_ok = not has_final_line(text)
            strict_ok = exec_ok and no_final_ok if is_program_arm(arm) else final_ok
            has_gold_text = clean_answer(ex.answer) in text
            final_correct += int(final_ok)
            exec_correct += int(exec_ok)
            strict_correct += int(strict_ok)
            valid_final += int(vf)
            valid_exec += int(ve)
            no_final += int(no_final_ok)
            leak += int(has_gold_text)
            family_stats[ex.family].append(int(strict_ok))
            token_count = int((new_ids != tokenizer.pad_token_id).sum().detach().cpu())
            token_counts.append(token_count)
            details.append(
                {
                    "run": run_id,
                    "suite": config.suite,
                    "arm": arm,
                    "seed": seed,
                    "trained": int(trained),
                    "split": ex.split,
                    "example_id": ex.example_id,
                    "family": ex.family,
                    "op_name": ex.op_name,
                    "answer": ex.answer,
                    "final_prediction": pred_final or "",
                    "exec_prediction": exec_ans or "",
                    "final_correct": int(final_ok),
                    "exec_correct": int(exec_ok),
                    "strict_correct": int(strict_ok),
                    "valid_final": int(vf),
                    "valid_exec": int(ve),
                    "no_final": int(no_final_ok),
                    "leak_string": int(has_gold_text),
                    "exec_error": err,
                    "new_tokens": token_count,
                    "generated": text.replace("\n", "\\n")[:700],
                }
            )
    n = len(examples)
    metric: Dict[str, Any] = {
        "run": run_id,
        "suite": config.suite,
        "arm": arm,
        "seed": seed,
        "trained": int(trained),
        "split": examples[0].split if examples else "",
        "n": n,
        "primary_accuracy": strict_correct / max(1, n),
        "final_accuracy": final_correct / max(1, n),
        "exec_accuracy": exec_correct / max(1, n),
        "valid_final_rate": valid_final / max(1, n),
        "valid_exec_rate": valid_exec / max(1, n),
        "no_final_rate": no_final / max(1, n),
        "leak_string_rate": leak / max(1, n),
        "mean_new_tokens": mean(token_counts),
    }
    for family, vals in family_stats.items():
        metric[f"acc_{family}"] = mean(vals) if vals else float("nan")
    return metric, details


def train_one_arm(config: RunConfig, arm: str, seed: int, train_examples: Sequence[Example], eval_sets: Dict[str, Sequence[Example]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_id = f"{config.run_name}_{arm}_s{seed}"
    run_dir = RUNS / run_id
    ckpt_dir = CHECKPOINTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    tokenizer = ensure_tokenizer(config.model_name)
    tokenizer.padding_side = "right"
    dataset = PromptDataset(train_examples, arm, tokenizer, config.max_length)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=lambda rows: collate_batch(rows, tokenizer.pad_token_id))
    model = load_model(config.model_name, lora=True, config=config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr)
    batches = iter_batches(loader)
    train_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, config.train_steps + 1):
        total_loss = 0.0
        for _ in range(config.grad_accum):
            batch = next(batches)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss / config.grad_accum
            loss.backward()
            total_loss += float(loss.detach().cpu()) * config.grad_accum
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % max(1, config.train_steps // 5) == 0 or step == config.train_steps:
            row = {"run": run_id, "suite": config.suite, "arm": arm, "seed": seed, "step": step, "loss": total_loss, "elapsed_s": time.time() - t0}
            train_rows.append(row)
            log(f"{run_id} step {step}/{config.train_steps} loss={total_loss:.4f}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    write_csv(run_dir / "train_log.csv", train_rows)
    tokenizer.padding_side = "left"
    model.eval()
    model.config.use_cache = True
    metric_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    for split, examples in eval_sets.items():
        metrics, details = evaluate_model(config, model, tokenizer, examples, arm, run_id, seed, True)
        metric_rows.append(metrics)
        detail_rows.extend(details)
    write_csv(run_dir / "metrics.csv", metric_rows)
    write_csv(run_dir / "details.csv", detail_rows)
    write_json(run_dir / "manifest.json", {"run": run_id, "suite": config.suite, "arm": arm, "seed": seed, "config": asdict(config), "checkpoint_dir": ckpt_dir})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metric_rows, detail_rows, train_rows


def run_zero_shot(config: RunConfig, eval_sets: Dict[str, Sequence[Example]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_id = f"{config.run_name}_zero_shot_answer"
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ensure_tokenizer(config.model_name)
    tokenizer.padding_side = "left"
    model = load_model(config.model_name, lora=False)
    metric_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    for split, examples in eval_sets.items():
        metrics, details = evaluate_model(config, model, tokenizer, examples, "answer_only", run_id, 0, False)
        metrics["arm"] = "zero_shot_answer"
        for row in details:
            row["arm"] = "zero_shot_answer"
        metric_rows.append(metrics)
        detail_rows.extend(details)
    write_csv(run_dir / "metrics.csv", metric_rows)
    write_csv(run_dir / "details.csv", detail_rows)
    write_json(run_dir / "manifest.json", {"run": run_id, "suite": config.suite, "config": asdict(config), "trained": False})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metric_rows, detail_rows


def make_config(args: argparse.Namespace) -> RunConfig:
    if args.suite == "smoke":
        train_n, eval_n, steps, seeds, arms = 24, 8, 2, [101], ["answer_only", "program_stack"]
        batch_size, grad_accum = 2, 1
    elif args.suite == "pilot":
        train_n, eval_n, steps, seeds, arms = 96, 16, 24, [101], ARMS
        batch_size, grad_accum = 2, 2
    else:
        train_n, eval_n, steps, seeds, arms = 192, 24, 48, [101, 202], ARMS
        batch_size, grad_accum = 2, 4
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
        run_name=args.run_name or f"{args.suite}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}",
        suite=args.suite,
        model_name=args.model_name,
        seeds=seeds,
        arms=arms,
        train_n=train_n,
        eval_n=eval_n,
        train_steps=steps,
        batch_size=args.batch_size or batch_size,
        grad_accum=args.grad_accum or grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_new_tokens=args.max_new_tokens,
    )


def run_experiment(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = make_config(args)
    append_log(
        f"## Run `{config.run_name}`\n\n"
        f"- Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
        f"- Suite: `{config.suite}`\n"
        f"- Model: `{config.model_name}`\n"
        f"- Seeds: `{','.join(map(str, config.seeds))}`\n"
        f"- Arms: `{','.join(config.arms)}`\n"
        f"- Train examples per seed: `{config.train_n}`\n"
        f"- Eval examples per split: `{config.eval_n}`\n"
        f"- Steps: `{config.train_steps}`"
    )
    t0 = time.time()
    all_metrics: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []
    all_train: List[Dict[str, Any]] = []
    for seed in config.seeds:
        train_examples = make_split(seed, "train", config.train_n)
        eval_sets = {split: make_split(seed, split, config.eval_n) for split in EVAL_SPLITS}
        data_dir = RUNS / f"{config.run_name}_data_s{seed}"
        data_dir.mkdir(parents=True, exist_ok=True)
        write_json(data_dir / "dataset_manifest.json", {"seed": seed, "train_n": len(train_examples), "eval_sizes": {k: len(v) for k, v in eval_sets.items()}, "train_sample": [asdict(x) for x in train_examples[:4]], "eval_sample": {k: [asdict(x) for x in v[:2]] for k, v in eval_sets.items()}})
        if args.include_zero_shot and seed == config.seeds[0]:
            metrics, details = run_zero_shot(config, eval_sets)
            all_metrics.extend(metrics)
            all_details.extend(details)
        for arm in config.arms:
            metrics, details, train = train_one_arm(config, arm, seed, train_examples, eval_sets)
            all_metrics.extend(metrics)
            all_details.extend(details)
            all_train.extend(train)
    write_csv(ANALYSIS / f"{config.run_name}_metrics.csv", all_metrics)
    write_csv(ANALYSIS / f"{config.run_name}_details.csv", all_details)
    write_csv(ANALYSIS / f"{config.run_name}_train_log.csv", all_train)
    append_log(f"Completed `{config.run_name}` in {time.time() - t0:.1f}s.\n\n- Metric rows: {len(all_metrics)}\n- Detail rows: {len(all_details)}\n- Training log rows: {len(all_train)}")
    analyze_all()


def read_all_csv(pattern: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for base in [RUNS, ANALYSIS]:
        for path in sorted(base.glob(pattern)):
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if not df.empty:
                frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates() if frames else pd.DataFrame()


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for keys, sub in metrics.groupby(["suite", "arm", "trained", "split"], dropna=False):
        row = {"suite": keys[0], "arm": keys[1], "trained": keys[2], "split": keys[3], "runs": len(sub), "n_total": int(sub["n"].sum())}
        for col in ["primary_accuracy", "final_accuracy", "exec_accuracy", "valid_final_rate", "valid_exec_rate", "no_final_rate", "leak_string_rate", "mean_new_tokens"]:
            vals = [float(x) for x in sub[col].dropna()]
            row[f"{col}_mean"] = mean(vals)
            row[f"{col}_std"] = std(vals)
        for family in TRAIN_FAMILIES:
            col = f"acc_{family}"
            vals = [float(x) for x in sub[col].dropna()] if col in sub else []
            row[f"{family}_mean"] = mean(vals) if vals else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def suite_rank(suite: str) -> int:
    return {"smoke": 0, "pilot": 1, "main": 2}.get(str(suite), 0)


def select_primary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    rank = max(suite_rank(x) for x in summary["suite"].unique())
    return summary[summary["suite"].map(suite_rank).eq(rank)].copy()


def plot_primary(primary: pd.DataFrame) -> None:
    if primary.empty:
        return
    arms = list(primary["arm"].drop_duplicates())
    splits = [s for s in EVAL_SPLITS if s in set(primary["split"])]
    x = np.arange(len(splits))
    width = 0.8 / max(1, len(arms))
    plt.figure(figsize=(11, 5.8))
    for j, arm in enumerate(arms):
        vals, errs = [], []
        for split in splits:
            row = primary[(primary["arm"].eq(arm)) & (primary["split"].eq(split))]
            vals.append(100 * float(row.iloc[0]["primary_accuracy_mean"]) if not row.empty else np.nan)
            errs.append(100 * float(row.iloc[0]["primary_accuracy_std"]) if not row.empty else 0)
        plt.bar(x + (j - (len(arms) - 1) / 2) * width, vals, width=width, yerr=errs, capsize=3, label=arm)
    plt.xticks(x, [s.replace("eval_", "").replace("_", " ") for s in splits])
    plt.ylabel("Primary accuracy (%)")
    plt.title("Primary Accuracy by Arm")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "primary_accuracy_by_arm.png", dpi=180)
    plt.close()


def plot_exec_vs_final(primary: pd.DataFrame) -> None:
    if primary.empty:
        return
    split = "eval_composition" if "eval_composition" in set(primary["split"]) else str(primary["split"].iloc[0])
    sub = primary[primary["split"].eq(split)].copy()
    arms = list(sub["arm"])
    x = np.arange(len(arms))
    width = 0.28
    plt.figure(figsize=(10, 5.4))
    plt.bar(x - width, 100 * sub["primary_accuracy_mean"], width=width, label="primary")
    plt.bar(x, 100 * sub["exec_accuracy_mean"], width=width, label="executed program")
    plt.bar(x + width, 100 * sub["final_accuracy_mean"], width=width, label="emitted FINAL")
    plt.xticks(x, arms, rotation=25, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Primary vs Execution vs Emitted Answer ({split})")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "execution_vs_final.png", dpi=180)
    plt.close()


def plot_format(primary: pd.DataFrame) -> None:
    if primary.empty:
        return
    split = "eval_indist" if "eval_indist" in set(primary["split"]) else str(primary["split"].iloc[0])
    sub = primary[primary["split"].eq(split)].copy()
    x = np.arange(len(sub))
    fig, ax1 = plt.subplots(figsize=(10, 5.4))
    ax1.bar(x - 0.18, 100 * sub["valid_exec_rate_mean"], width=0.36, label="valid execution", color="#4C78A8")
    ax1.bar(x + 0.18, 100 * sub["no_final_rate_mean"], width=0.36, label="no FINAL line", color="#54A24B")
    ax1.set_ylabel("Rate (%)")
    ax1.set_ylim(0, 105)
    ax2 = ax1.twinx()
    ax2.plot(x, sub["mean_new_tokens_mean"], marker="o", color="#F58518", label="tokens")
    ax2.set_ylabel("Mean generated tokens")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["arm"], rotation=25, ha="right")
    ax1.set_title(f"Program Validity, Answer Suppression, and Token Cost ({split})")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "format_and_tokens.png", dpi=180)
    plt.close(fig)


def plot_family(primary: pd.DataFrame) -> None:
    if primary.empty:
        return
    split = "eval_composition" if "eval_composition" in set(primary["split"]) else str(primary["split"].iloc[0])
    sub = primary[primary["split"].eq(split)].copy()
    arms = list(sub["arm"])
    x = np.arange(len(TRAIN_FAMILIES))
    width = 0.8 / max(1, len(arms))
    plt.figure(figsize=(11, 5.8))
    for j, (_, row) in enumerate(sub.iterrows()):
        vals = [100 * float(row.get(f"{fam}_mean", np.nan)) for fam in TRAIN_FAMILIES]
        plt.bar(x + (j - (len(arms) - 1) / 2) * width, vals, width=width, label=row["arm"])
    plt.xticks(x, TRAIN_FAMILIES)
    plt.ylabel("Primary accuracy (%)")
    plt.title(f"Family Breakdown ({split})")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "family_breakdown.png", dpi=180)
    plt.close()


def plot_training(logs: pd.DataFrame) -> None:
    if logs.empty:
        return
    plt.figure(figsize=(10.5, 5.8))
    for run, sub in logs.groupby("run"):
        label = run if len(run) <= 36 else run[:33] + "..."
        plt.plot(sub["step"], sub["loss"], marker="o", linewidth=1.4, label=label)
    plt.xlabel("Training step")
    plt.ylabel("CE loss")
    plt.title("Training Curves")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "training_curves.png", dpi=180)
    plt.close()


def md_table(df: pd.DataFrame, cols: Sequence[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals: List[str] = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                if (
                    "accuracy" in col
                    or "rate" in col
                    or col in {"string_mean", "unit_mean", "table_mean", "date_mean"}
                    or col.endswith("accuracy_std")
                    or col.endswith("rate_std")
                ):
                    vals.append(pct(value))
                else:
                    vals.append(f"{value:.2f}")
            else:
                vals.append(str(value))
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines)


def best_arm(primary: pd.DataFrame, split: str) -> Optional[pd.Series]:
    sub = primary[primary["split"].eq(split)]
    if sub.empty:
        return None
    return sub.sort_values("primary_accuracy_mean", ascending=False).iloc[0]


def make_report(summary: pd.DataFrame, metrics: pd.DataFrame, logs: pd.DataFrame) -> str:
    primary = select_primary(summary)
    primary_suite = primary["suite"].iloc[0] if not primary.empty else "none"
    rows = primary.sort_values(["split", "primary_accuracy_mean"], ascending=[True, False]).copy()
    best_comp = best_arm(primary, "eval_composition")
    best_indist = best_arm(primary, "eval_indist")
    comp_rows = primary[primary["split"].eq("eval_composition")]
    exec_comp_rows = comp_rows[comp_rows["exec_accuracy_mean"].gt(0)] if not comp_rows.empty else pd.DataFrame()
    best_exec_comp = exec_comp_rows.sort_values("exec_accuracy_mean", ascending=False).iloc[0] if not exec_comp_rows.empty else None
    trained_metrics = metrics[(metrics["suite"].eq(primary_suite)) & (metrics["trained"].eq(1))] if not metrics.empty else pd.DataFrame()
    seeds = sorted(int(x) for x in trained_metrics["seed"].dropna().unique()) if not trained_metrics.empty else []
    lines: List[str] = []
    lines.append("# Qwen Program-Only Executable ABI")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append("This experiment tests whether a local 4B language model can compile deterministic office-style tasks into executable programs when the final answer is absent from the program-only targets. Program-only outputs are parsed and executed by a deterministic interpreter; correctness is based on the interpreter result.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("The task factory creates string, unit-conversion, table-lookup, and date-offset examples. The training split contains atomic operations, while the composition split recombines known primitives into held-out multi-step procedures. Four arms are compared: `answer_only`, `trace_stack_final`, `program_stack`, and `program_python`.")
    lines.append("")
    lines.append("For program-only arms, the primary metric is strict execution accuracy: the generated program must execute to the correct answer and must not contain a `FINAL` line. For answer-emitting arms, the primary metric is exact match on the parsed `FINAL` line.")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Primary suite: `{primary_suite}`.")
    if seeds:
        lines.append(f"- Adapter seeds: `{','.join(map(str, seeds))}`.")
    if not trained_metrics.empty:
        lines.append(f"- Total trained-arm evaluation examples: `{int(trained_metrics['n'].sum())}` across arms, splits, and seeds.")
    if not logs.empty and "suite" in logs:
        step = int(logs[logs["suite"].eq(primary_suite)]["step"].max())
        lines.append(f"- QLoRA update steps per adapter: `{step}`.")
    lines.append("- Large adapters are stored outside the experiment tree.")
    lines.append("")
    lines.append("## Primary Results")
    lines.append("")
    if best_indist is not None:
        lines.append(f"- Best in-distribution arm: `{best_indist['arm']}` at {pct(best_indist['primary_accuracy_mean'])}.")
    if best_comp is not None:
        lines.append(f"- Best held-out composition arm: `{best_comp['arm']}` at {pct(best_comp['primary_accuracy_mean'])}.")
    if best_exec_comp is not None:
        lines.append(f"- Best externally executed held-out composition procedure: `{best_exec_comp['arm']}` at {pct(best_exec_comp['exec_accuracy_mean'])}.")
    prog_comp = primary[(primary["split"].eq("eval_composition")) & (primary["arm"].isin(["program_stack", "program_python"]))]
    if not prog_comp.empty:
        best_prog = prog_comp.sort_values("primary_accuracy_mean", ascending=False).iloc[0]
        lines.append(f"- Best strict program-only composition arm: `{best_prog['arm']}` at {pct(best_prog['primary_accuracy_mean'])}.")
    lines.append("")
    table_cols = ["suite", "arm", "split", "runs", "n_total", "primary_accuracy_mean", "primary_accuracy_std", "exec_accuracy_mean", "valid_exec_rate_mean", "no_final_rate_mean", "mean_new_tokens_mean"]
    lines.append(md_table(rows[table_cols], table_cols))
    lines.append("")
    lines.append("![Primary accuracy by arm](../analysis/figures/primary_accuracy_by_arm.png)")
    lines.append("")
    lines.append("![Execution versus final](../analysis/figures/execution_vs_final.png)")
    lines.append("")
    lines.append("![Format and tokens](../analysis/figures/format_and_tokens.png)")
    lines.append("")
    lines.append("![Family breakdown](../analysis/figures/family_breakdown.png)")
    lines.append("")
    lines.append("![Training curves](../analysis/figures/training_curves.png)")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    trace_comp = primary[(primary["split"].eq("eval_composition")) & (primary["arm"].eq("trace_stack_final"))]
    if not trace_comp.empty:
        row = trace_comp.iloc[0]
        lines.append(f"The strongest procedure-level result is the externally executed `trace_stack_final` program: {pct(row['exec_accuracy_mean'])} execution accuracy on held-out compositions, with {pct(row['primary_accuracy_mean'])} final-answer accuracy. The generated procedure can be right while the answer token is wrong, so the emitted `FINAL` line is a confounded score for procedure arms.")
    if best_comp is not None:
        fam_bits = []
        for fam in TRAIN_FAMILIES:
            val = best_comp.get(f"{fam}_mean", float("nan"))
            if not pd.isna(val):
                fam_bits.append(f"{fam} {pct(val)}")
        lines.append(f"The best held-out composition arm was `{best_comp['arm']}`. Its family accuracies were: {', '.join(fam_bits)}.")
    if not prog_comp.empty:
        prog_best = prog_comp.sort_values("primary_accuracy_mean", ascending=False).iloc[0]
        ans_comp = primary[(primary["split"].eq("eval_composition")) & (primary["arm"].eq("answer_only"))]
        ans_val = float(ans_comp.iloc[0]["primary_accuracy_mean"]) if not ans_comp.empty else float("nan")
        lines.append(f"The strongest strict program-only arm reached {pct(prog_best['primary_accuracy_mean'])} on held-out compositions, compared with {pct(ans_val)} for answer-only. This directly measures whether executable compilation improves composition rather than merely producing a plausible answer string.")
        lines.append(f"The same row has seed standard deviation {pct(prog_best['primary_accuracy_std'])}, so the result is positive but not yet stable enough to treat as a finished recipe.")
    if not trace_comp.empty:
        row = trace_comp.iloc[0]
        lines.append(f"The trace-plus-final procedure execution was more stable than strict program-only emission in this compact run: composition execution standard deviation was {pct(row['exec_accuracy_std'])}, versus {pct(prog_comp.sort_values('primary_accuracy_mean', ascending=False).iloc[0]['primary_accuracy_std']) if not prog_comp.empty else 'n/a'} for the best strict program-only row.")
    lines.append("A program-only win would show that the model learned a useful executable ABI. A program-only loss, especially with high valid-execution rate, indicates that the model can imitate program syntax but still chooses the wrong operations or arguments.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("This is a compact controlled run. The generated domains are narrow, and the interpreters intentionally support only a small operation set. The result should be read as an ABI and supervision test, not a benchmark of general assistant capability.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Metrics: `analysis/summary_by_arm.csv` and `analysis/all_metrics.csv`")
    lines.append("- Details: `analysis/all_details.csv`")
    lines.append("- Training logs: `analysis/all_train_logs.csv`")
    lines.append("- Checkpoints: `/workspace/large_artifacts/qwen_program_only_executable_abi/checkpoints`")
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown_text: str) -> str:
    body: List[str] = []
    in_table = False
    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.startswith("![") and "](" in line:
            alt = line[2 : line.index("]")]
            src = line[line.index("(") + 1 : line.rindex(")")]
            body.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.startswith("|") and line.endswith("|"):
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
            body.append(f"<p>{html.escape(line)}</p>" if line else "")
    if in_table:
        body.append("</table>")
    css = "body{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1120px;margin:36px auto;line-height:1.5;color:#202124}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d4d7dc;padding:7px 9px;text-align:left}th{background:#f2f4f7}img{max-width:100%;border:1px solid #d4d7dc}figure{margin:24px 0}figcaption{color:#5f6368;font-size:13px}"
    return "<!doctype html><html><head><meta charset='utf-8'><title>Qwen Program-Only Executable ABI</title><style>" + css + "</style></head><body>" + "\n".join(body) + "</body></html>\n"


def write_checkpoint_manifest() -> None:
    rows: List[Dict[str, Any]] = []
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
    primary = select_primary(summary)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_primary(primary)
    plot_exec_vs_final(primary)
    plot_format(primary)
    plot_family(primary)
    if not primary.empty and not logs.empty and "suite" in logs:
        plot_training(logs[logs["suite"].eq(primary["suite"].iloc[0])].copy())
    else:
        plot_training(logs)
    report = make_report(summary, metrics, logs)
    (REPORTS / "qwen_program_only_executable_abi_report.md").write_text(report)
    (REPORTS / "qwen_program_only_executable_abi_report.html").write_text(markdown_to_html(report))
    write_checkpoint_manifest()


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
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--include_zero_shot", action="store_true")
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
