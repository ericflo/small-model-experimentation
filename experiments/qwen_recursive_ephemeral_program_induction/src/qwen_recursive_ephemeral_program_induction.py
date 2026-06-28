#!/usr/bin/env python3
"""Recursive ephemeral program induction for text transformations.

The experiment asks a frozen language model to induce a task-local executable
Python function from examples. Candidate functions are selected only with
visible train examples, then evaluated on held-out rows. Repair prompts receive
train failures but never held-out rows.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import os
import random
import re
import shutil
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path("/workspace/experiments/qwen_recursive_ephemeral_program_induction")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_recursive_ephemeral_program_induction")
SOURCE_BENCH_ROOT = Path("/workspace/large_artifacts/qwen_batched_transduction_consistency/prose-benchmarks")
BENCH_ROOT = LARGE_ROOT / "prose-benchmarks"
TRANSFORM_ROOT = BENCH_ROOT / "Transformation.Text"
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CACHE_DIR = Path("/workspace/.cache/huggingface")
MODEL_NAME = "Qwen/Qwen3-4B"


@dataclass(frozen=True)
class Example:
    inputs: Tuple[str, ...]
    output: str


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    synthetic: bool
    features: Tuple[str, ...]
    examples: Tuple[Example, ...]
    source_path: str


@dataclass
class Candidate:
    task_id: str
    source: str
    variant: str
    repair_round: int
    raw_text: str
    code: str
    status: str
    train_pass: bool
    train_row_exact: float
    heldout_row_exact: float
    heldout_full_exact: bool
    error: str = ""


def ensure_dirs() -> None:
    for d in [ROOT, LARGE_ROOT, RUNS, ANALYSIS, FIGURES, REPORTS]:
        d.mkdir(parents=True, exist_ok=True)


def mirror_benchmark() -> None:
    if BENCH_ROOT.exists():
        return
    if not SOURCE_BENCH_ROOT.exists():
        raise FileNotFoundError(f"Missing benchmark source: {SOURCE_BENCH_ROOT}")
    LARGE_ROOT.mkdir(parents=True, exist_ok=True)
    BENCH_ROOT.symlink_to(SOURCE_BENCH_ROOT, target_is_directory=True)


def clean(s: Any) -> str:
    return re.sub(r"\s+", " ", "" if s is None else str(s)).strip()


def norm_eq(a: Any, b: Any) -> bool:
    return clean(a) == clean(b)


def render_inputs(vals: Sequence[str]) -> str:
    if len(vals) == 1:
        return vals[0]
    return " | ".join(f"col{i}={v}" for i, v in enumerate(vals))


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def load_tasks(min_examples: int) -> List[Task]:
    tasks: List[Task] = []
    for d in sorted(TRANSFORM_ROOT.iterdir()):
        if not d.is_dir() or not (d / "spec.json").exists():
            continue
        spec = json.loads((d / "spec.json").read_text())
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        examples = [Example(tuple(str(x) for x in ex.get("Input", [])), str(ex.get("Output", ""))) for ex in spec.get("Examples", [])]
        if len(examples) < min_examples:
            continue
        tasks.append(
            Task(
                task_id=d.name,
                family=d.name.split(".", 1)[0],
                synthetic=bool(meta.get("Synthetic", False)),
                features=tuple(str(x) for x in meta.get("Features", [])),
                examples=tuple(examples),
                source_path=str(d),
            )
        )
    return tasks


def choose_tasks(tasks: Sequence[Task], limit: int, seed: int) -> List[Task]:
    rng = random.Random(seed)
    if limit and limit < len(tasks):
        chosen = rng.sample(list(tasks), limit)
    else:
        chosen = list(tasks)
    return sorted(chosen, key=lambda t: t.task_id)


def split_examples(task: Task, train_n: int, heldout_cap: int, min_heldout: int) -> Tuple[List[Example], List[Example]]:
    train = list(task.examples[:train_n])
    heldout = list(task.examples[train_n:])
    if heldout_cap:
        heldout = heldout[:heldout_cap]
    if len(train) < train_n or len(heldout) < min_heldout:
        raise ValueError(f"Task {task.task_id} lacks required examples")
    return train, heldout


def load_qwen() -> Tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()
    return tok, model


@torch.inference_mode()
def generate_text(tok: Any, model: Any, system: str, user: str, max_new_tokens: int) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()


class GenerationCache:
    def __init__(self, path: Path):
        self.path = path
        self.rows: Dict[str, Dict[str, str]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open(newline="") as f:
                for row in csv.DictReader(f):
                    self.rows[row["key"]] = row

    def key(self, system: str, user: str, max_new_tokens: int) -> str:
        payload = json.dumps(
            {"model": MODEL_NAME, "system": system, "user": user, "max_new_tokens": max_new_tokens},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_or_generate(self, tok: Any, model: Any, system: str, user: str, max_new_tokens: int) -> str:
        key = self.key(system, user, max_new_tokens)
        if key in self.rows:
            return self.rows[key]["text"]
        text = generate_text(tok, model, system, user, max_new_tokens)
        row = {
            "key": key,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "system": system,
            "user": user,
            "max_new_tokens": str(max_new_tokens),
            "text": text,
        }
        new_file = not self.path.exists()
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        self.rows[key] = row
        return text


def clean_prediction(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    text = re.sub(r"(?i)^output\s*:\s*", "", text).strip()
    first = text.splitlines()[0].strip() if text else ""
    if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
        first = first[1:-1].strip()
    return first


def words(s: Any) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", clean(s))


def alpha(s: Any) -> str:
    return re.sub(r"[^A-Za-z]+", "", clean(s))


def alnum(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", clean(s))


def digits(s: Any) -> str:
    return re.sub(r"\D+", "", clean(s))


def first_word(s: Any) -> str:
    ws = words(s)
    return ws[0] if ws else ""


def last_word(s: Any) -> str:
    ws = words(s)
    return ws[-1] if ws else ""


def word(s: Any, idx: int) -> str:
    ws = words(s)
    j = idx if idx >= 0 else len(ws) + idx
    return ws[j] if 0 <= j < len(ws) else ""


def field(s: Any, sep: str, idx: int) -> str:
    parts = [p.strip() for p in str(s).split(sep)]
    j = idx if idx >= 0 else len(parts) + idx
    return parts[j] if 0 <= j < len(parts) else ""


def before(s: Any, sep: str) -> str:
    return str(s).split(sep, 1)[0].strip()


def after(s: Any, sep: str) -> str:
    parts = str(s).split(sep, 1)
    return parts[1].strip() if len(parts) > 1 else ""


def between(s: Any, left: str, right: str) -> str:
    text = str(s)
    if left in text:
        text = text.split(left, 1)[1]
    if right in text:
        text = text.split(right, 1)[0]
    return text.strip()


def regex_group(s: Any, pattern: str, group: int = 1) -> str:
    m = re.search(pattern, str(s))
    if not m:
        return ""
    return m.group(group)


def number(s: Any) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m.group(0)) if m else 0.0


def number_int(s: Any) -> str:
    return str(int(round(number(s))))


def number_1dp(s: Any) -> str:
    return f"{number(s):.1f}"


def number_2dp(s: Any) -> str:
    return f"{number(s):.2f}"


def round_to(s: Any, base: int) -> str:
    if base == 0:
        return number_int(s)
    return str(int(round(number(s) / base) * base))


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
MONTH_ABBR = {v: k.title() for k, v in MONTHS.items() if len(k) == 3}
MONTH_NAME = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def date_parts(s: Any) -> Tuple[int, int, int]:
    text = clean(s)
    patterns = [
        (r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", ("Y", "M", "D")),
        (r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", ("M", "D", "Y")),
        (r"\b(\d{1,2})-(\d{1,2})-(\d{2,4})\b", ("M", "D", "Y")),
        (r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b", ("D", "M", "Y")),
    ]
    for pattern, order in patterns:
        m = re.search(pattern, text)
        if m:
            vals = [int(x) for x in m.groups()]
            parts = dict(zip(order, vals))
            year = parts["Y"] + (2000 if parts["Y"] < 100 else 0)
            return year, parts["M"], parts["D"]
    m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b", text)
    if m and m.group(1).lower() in MONTHS:
        return int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2))
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b", text)
    if m and m.group(2).lower() in MONTHS:
        return int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1))
    return 0, 0, 0


def year(s: Any) -> str:
    return str(date_parts(s)[0])


def month_num(s: Any) -> str:
    return str(date_parts(s)[1])


def month_2(s: Any) -> str:
    return f"{date_parts(s)[1]:02d}"


def month_name(s: Any) -> str:
    return MONTH_NAME.get(date_parts(s)[1], "")


def month_abbr(s: Any) -> str:
    return MONTH_ABBR.get(date_parts(s)[1], "")


def day(s: Any) -> str:
    return str(date_parts(s)[2])


def day_2(s: Any) -> str:
    return f"{date_parts(s)[2]:02d}"


def time_parts(s: Any) -> Tuple[int, int, int]:
    m = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AP]M)?\b", str(s), flags=re.I)
    if not m:
        return 0, 0, 0
    h = int(m.group(1))
    minute = int(m.group(2))
    sec = int(m.group(3) or 0)
    ampm = (m.group(4) or "").upper()
    if ampm == "PM" and h < 12:
        h += 12
    if ampm == "AM" and h == 12:
        h = 0
    return h, minute, sec


def hour(s: Any) -> str:
    return str(time_parts(s)[0])


def hour_2(s: Any) -> str:
    return f"{time_parts(s)[0]:02d}"


def minute_2(s: Any) -> str:
    return f"{time_parts(s)[1]:02d}"


HELPER_DOC = """Available helper functions:
- clean(x), words(x), alpha(x), alnum(x), digits(x)
- first_word(x), last_word(x), word(x, i)
- field(x, sep, i), before(x, sep), after(x, sep), between(x, left, right)
- regex_group(x, pattern, group=1)
- number(x), number_int(x), number_1dp(x), number_2dp(x), round_to(x, base)
- year(x), month_num(x), month_2(x), month_name(x), month_abbr(x), day(x), day_2(x)
- hour(x), hour_2(x), minute_2(x)
"""


def direct_prompt(train: Sequence[Example], query: Example) -> str:
    lines = [
        "Infer the exact text transformation from the examples.",
        "Return only the transformed output for the query. Do not explain.",
        "",
        "Examples:",
    ]
    for ex in train:
        lines.append(f"Input: {render_inputs(ex.inputs)}")
        lines.append(f"Output: {ex.output}")
    lines.extend(["", "Query:", f"Input: {render_inputs(query.inputs)}", "Output:"])
    return "\n".join(lines)


def batch_prompt(train: Sequence[Example], heldout: Sequence[Example]) -> str:
    lines = [
        "Infer one deterministic text transformation from the examples.",
        "Apply it to every query.",
        "Return only a JSON array of output strings, in query order.",
        "",
        "Examples:",
    ]
    for ex in train:
        lines.append(f"Input: {render_inputs(ex.inputs)}")
        lines.append(f"Output: {ex.output}")
    lines.append("")
    lines.append("Queries:")
    for i, ex in enumerate(heldout):
        lines.append(f"{i}. {render_inputs(ex.inputs)}")
    lines.append("")
    lines.append("JSON outputs:")
    return "\n".join(lines)


def parse_json_outputs(text: str, n: int) -> Tuple[List[str], bool]:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    m = re.search(r"\[[\s\S]*\]", text)
    blob = m.group(0) if m else text
    try:
        vals = json.loads(blob)
        if not isinstance(vals, list):
            return [""] * n, False
        outs = [clean(str(x)) for x in vals[:n]]
        if len(outs) < n:
            outs += [""] * (n - len(outs))
        return outs, len(vals) == n
    except Exception:
        lines = [clean_prediction(x) for x in text.splitlines() if clean_prediction(x)]
        outs = lines[:n] + [""] * max(0, n - len(lines))
        return outs, False


def examples_block(examples: Sequence[Example]) -> str:
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"{i}. transform({list(ex.inputs)!r}) -> {ex.output!r}")
    return "\n".join(lines)


def code_prompt(train: Sequence[Example], variant: str) -> str:
    helper = "row is a list of input columns. Use row[0] for a single-column task."
    if variant == "monolithic":
        style = "Write one concise function. Keep the logic direct."
    elif variant == "helpers":
        style = "Decompose the task into small local helper functions, then call them from transform."
    else:
        style = "Prefer robust parsing, normalization, and explicit conditionals over memorizing examples."
    return f"""Write Python code for a deterministic text transformation.

Requirements:
- Define exactly one public function named transform(row).
- {helper}
- Return a string.
- Use only Python built-ins, re, math, datetime, calendar, and the helper functions listed below.
- Do not read files, import packages other than the allowed modules, print, or include tests.
- The function must match every example exactly.

{HELPER_DOC}

Examples:
{examples_block(train)}

Style: {style}

Return only a Python code block."""


def repair_prompt(train: Sequence[Example], previous_code: str, failures: Sequence[Tuple[Example, str]], variant: str) -> str:
    fail_lines = []
    for ex, pred in failures:
        fail_lines.append(f"row = {list(ex.inputs)!r}")
        fail_lines.append(f"expected = {ex.output!r}")
        fail_lines.append(f"actual = {pred!r}")
    return f"""Repair this Python text-transformation function.

It must define transform(row), where row is a list of input columns, and return a string.
Use only Python built-ins, re, math, datetime, calendar, and these helper functions:
{HELPER_DOC}
Return only a Python code block.

All examples:
{examples_block(train)}

Failing cases:
{chr(10).join(fail_lines)}

Previous code:
```python
{previous_code}
```

Repair style: {variant}. Preserve correct behavior and fix the failures."""


def shuffled_train(train: Sequence[Example], seed_key: str) -> List[Example]:
    outs = [ex.output for ex in train]
    if len(outs) > 1:
        shift = 1 + (stable_int(seed_key) % (len(outs) - 1))
        outs = outs[shift:] + outs[:shift]
    return [Example(ex.inputs, out) for ex, out in zip(train, outs)]


ALLOWED_IMPORTS = {"re", "math", "datetime", "calendar"}
BLOCKED_NAMES = {
    "__import__",
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "help",
    "breakpoint",
    "print",
}
ALLOWED_BUILTINS = {
    "str": str,
    "int": int,
    "float": float,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "reversed": reversed,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "any": any,
    "all": all,
    "bool": bool,
    "isinstance": isinstance,
    "map": map,
    "filter": filter,
    "ord": ord,
    "chr": chr,
    "divmod": divmod,
    "pow": pow,
}


def safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    root = name.split(".", 1)[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError(f"blocked import {name}")
    return __import__(name, globals, locals, fromlist, level)


class SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".", 1)[0] not in ALLOWED_IMPORTS:
                self.errors.append(f"blocked import {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = (node.module or "").split(".", 1)[0]
        if mod not in ALLOWED_IMPORTS:
            self.errors.append(f"blocked import from {node.module}")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
            self.errors.append(f"blocked call {node.func.id}")
        if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("__"):
            self.errors.append(f"blocked dunder attr {node.func.attr}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.errors.append(f"blocked dunder attr {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BLOCKED_NAMES or node.id.startswith("__"):
            self.errors.append(f"blocked name {node.id}")


def extract_code(raw: str) -> str:
    raw = re.sub(r"(?is)<think>.*?</think>", "", raw).strip()
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw, flags=re.I)
    code = m.group(1).strip() if m else raw
    if "def transform" in code:
        code = code[code.find("def transform") :]
    return code.strip()


def compile_transform(code: str) -> Tuple[Optional[Callable[[Tuple[str, ...]], str]], str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return None, f"syntax_error: {e}"
    visitor = SafetyVisitor()
    visitor.visit(tree)
    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if "transform" not in funcs:
        visitor.errors.append("missing transform(row)")
    if visitor.errors:
        return None, "; ".join(visitor.errors[:5])
    env: Dict[str, Any] = {
        "__builtins__": {**ALLOWED_BUILTINS, "__import__": safe_import},
        "re": __import__("re"),
        "math": __import__("math"),
        "datetime": __import__("datetime"),
        "calendar": __import__("calendar"),
        "clean": clean,
        "words": words,
        "alpha": alpha,
        "alnum": alnum,
        "digits": digits,
        "first_word": first_word,
        "last_word": last_word,
        "word": word,
        "field": field,
        "before": before,
        "after": after,
        "between": between,
        "regex_group": regex_group,
        "number": number,
        "number_int": number_int,
        "number_1dp": number_1dp,
        "number_2dp": number_2dp,
        "round_to": round_to,
        "date_parts": date_parts,
        "year": year,
        "month_num": month_num,
        "month_2": month_2,
        "month_name": month_name,
        "month_abbr": month_abbr,
        "day": day,
        "day_2": day_2,
        "time_parts": time_parts,
        "hour": hour,
        "hour_2": hour_2,
        "minute_2": minute_2,
    }
    try:
        exec(compile(tree, "<generated_transform>", "exec"), env, env)
        fn = env.get("transform")
        if not callable(fn):
            return None, "transform is not callable"
        return fn, "ok"
    except Exception as e:
        return None, f"compile_runtime_error: {type(e).__name__}: {e}"


def run_transform(fn: Callable[[Tuple[str, ...]], Any], ex: Example) -> Tuple[str, str]:
    try:
        out = fn(list(ex.inputs))
        return clean(out), "ok"
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def score_outputs(preds: Sequence[str], examples: Sequence[Example]) -> Tuple[float, bool]:
    if not examples:
        return 0.0, False
    ok = [norm_eq(p, ex.output) for p, ex in zip(preds, examples)]
    return sum(ok) / len(ok), all(ok)


def evaluate_code(task_id: str, source: str, variant: str, repair_round: int, raw: str, train: Sequence[Example], heldout: Sequence[Example]) -> Candidate:
    code = extract_code(raw)
    fn, status = compile_transform(code)
    if fn is None:
        return Candidate(task_id, source, variant, repair_round, raw, code, status, False, 0.0, 0.0, False, status)
    train_preds = [run_transform(fn, ex)[0] for ex in train]
    held_preds = [run_transform(fn, ex)[0] for ex in heldout]
    train_row, train_full = score_outputs(train_preds, train)
    held_row, held_full = score_outputs(held_preds, heldout)
    return Candidate(task_id, source, variant, repair_round, raw, code, "ok", train_full, train_row, held_row, held_full)


def failing_train_cases(candidate: Candidate, train: Sequence[Example]) -> List[Tuple[Example, str]]:
    fn, _ = compile_transform(candidate.code)
    if fn is None:
        return [(ex, "<compile failure>") for ex in train[:3]]
    fails = []
    for ex in train:
        pred, _ = run_transform(fn, ex)
        if not norm_eq(pred, ex.output):
            fails.append((ex, pred))
    return fails[:4]


def selected_candidate(cands: Sequence[Candidate]) -> Optional[Candidate]:
    train_pass = [c for c in cands if c.train_pass]
    if not train_pass:
        return None
    return sorted(train_pass, key=lambda c: (len(c.code), c.repair_round, c.variant))[0]


def suspicious_memorizer(code: str, train: Sequence[Example]) -> bool:
    output_hits = 0
    input_hits = 0
    for ex in train:
        if len(clean(ex.output)) >= 2 and (repr(ex.output) in code or json.dumps(ex.output) in code):
            output_hits += 1
        for val in ex.inputs:
            if len(clean(val)) >= 3 and (repr(val) in code or json.dumps(val) in code):
                input_hits += 1
    literal_branches = len(re.findall(r"\bif\b[^\n]{0,120}==\s*['\"]", code))
    return output_hits >= 2 or input_hits >= 3 or literal_branches >= 3


def selected_nonmemorizing_candidate(cands: Sequence[Candidate], train: Sequence[Example]) -> Optional[Candidate]:
    train_pass = [c for c in cands if c.train_pass and not suspicious_memorizer(c.code, train)]
    if not train_pass:
        return None
    return sorted(train_pass, key=lambda c: (len(c.code), c.repair_round, c.variant))[0]


def oracle_candidate(cands: Sequence[Candidate]) -> Optional[Candidate]:
    train_pass = [c for c in cands if c.train_pass]
    if not train_pass:
        return None
    return sorted(train_pass, key=lambda c: (not c.heldout_full_exact, -c.heldout_row_exact, len(c.code)))[0]


def append_log(text: str) -> None:
    log = ROOT / "experiment_log.md"
    with log.open("a") as f:
        f.write(text.rstrip() + "\n")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def run_experiment(args: argparse.Namespace) -> None:
    ensure_dirs()
    mirror_benchmark()
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    tasks = choose_tasks(load_tasks(args.train_n + args.min_heldout), args.task_limit, args.sample_seed)
    tasks = [t for t in tasks if len(t.examples) >= args.train_n + args.min_heldout]
    if args.task_limit:
        tasks = tasks[: args.task_limit]

    tok = model = None
    if not args.no_qwen:
        tok, model = load_qwen()
    cache = GenerationCache(run_dir / "generations.csv")

    system = "You are a precise programming assistant. Follow the user's required output format exactly."
    direct_system = "You are a precise text-transformation engine. Return only the requested output."

    direct_rows: List[Dict[str, Any]] = []
    batch_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    repair_variants = [v.strip() for v in args.repair_variants.split(",") if v.strip()]

    start = time.time()
    for ti, task in enumerate(tasks, 1):
        train, heldout = split_examples(task, args.train_n, args.heldout_cap, args.min_heldout)
        print(f"[{ti}/{len(tasks)}] {task.task_id} ({task.family})", flush=True)

        direct_preds = []
        for hi, ex in enumerate(heldout):
            user = direct_prompt(train, ex)
            raw = "" if args.no_qwen else cache.get_or_generate(tok, model, direct_system, user, args.answer_max_new_tokens)
            pred = clean_prediction(raw)
            direct_preds.append(pred)
            direct_rows.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "method": "direct_row",
                    "row_index": hi,
                    "input": render_inputs(ex.inputs),
                    "target": ex.output,
                    "prediction": pred,
                    "exact": norm_eq(pred, ex.output),
                    "raw": raw,
                }
            )
        direct_row_exact, direct_full_exact = score_outputs(direct_preds, heldout)

        batch_raw = "" if args.no_qwen else cache.get_or_generate(tok, model, direct_system, batch_prompt(train, heldout), args.batch_max_new_tokens)
        batch_preds, batch_parse_ok = parse_json_outputs(batch_raw, len(heldout))
        batch_row_exact, batch_full_exact = score_outputs(batch_preds, heldout)
        batch_rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "method": "direct_batch",
                "row_exact": batch_row_exact,
                "full_task_exact": batch_full_exact,
                "parse_ok": batch_parse_ok,
                "raw": batch_raw,
            }
        )

        def induce_candidates(source: str, source_train: Sequence[Example]) -> List[Candidate]:
            out: List[Candidate] = []
            for variant in variants:
                raw = "" if args.no_qwen else cache.get_or_generate(tok, model, system, code_prompt(source_train, variant), args.code_max_new_tokens)
                cand = evaluate_code(task.task_id, source, variant, 0, raw, source_train, heldout)
                out.append(cand)
                if source == "recursive" and args.repair_rounds > 0:
                    current = cand
                    for rr in range(1, args.repair_rounds + 1):
                        if current.train_pass:
                            break
                        failures = failing_train_cases(current, source_train)
                        repair_variant = repair_variants[(rr - 1) % len(repair_variants)] if repair_variants else "minimal"
                        prompt = repair_prompt(source_train, current.code, failures, repair_variant)
                        raw_repair = "" if args.no_qwen else cache.get_or_generate(tok, model, system, prompt, args.repair_max_new_tokens)
                        current = evaluate_code(task.task_id, source, variant, rr, raw_repair, source_train, heldout)
                        out.append(current)
            return out

        mono_cands = induce_candidates("monolithic", train)
        recursive_cands = induce_candidates("recursive", train)
        shuffled_cands: List[Candidate] = []
        if args.run_shuffled:
            shuffled_cands = induce_candidates("recursive_shuffled", shuffled_train(train, task.task_id))

        for cand in mono_cands + recursive_cands + shuffled_cands:
            candidate_rows.append(cand.__dict__)

        selected_mono = selected_candidate(mono_cands)
        selected_recursive = selected_candidate(recursive_cands)
        selected_recursive_gated = selected_nonmemorizing_candidate(recursive_cands, train)
        selected_shuffled = selected_candidate(shuffled_cands)
        oracle_recursive = oracle_candidate(recursive_cands)

        def method_metrics(name: str, cand: Optional[Candidate]) -> Tuple[float, bool, bool, str]:
            if cand is None:
                return 0.0, False, False, ""
            return cand.heldout_row_exact, cand.heldout_full_exact, cand.train_pass, cand.code

        mono_row, mono_full, mono_train, mono_code = method_metrics("monolithic_program", selected_mono)
        rec_row, rec_full, rec_train, rec_code = method_metrics("recursive_program", selected_recursive)
        gated_used_program = selected_recursive_gated is not None
        gated_row, gated_full, gated_train, gated_code = method_metrics("recursive_gated_direct", selected_recursive_gated)
        if not gated_used_program:
            gated_row, gated_full, gated_train, gated_code = direct_row_exact, direct_full_exact, False, ""
        shuf_row, shuf_full, shuf_train, shuf_code = method_metrics("recursive_shuffled", selected_shuffled)
        ora_row, ora_full, ora_train, ora_code = method_metrics("recursive_oracle", oracle_recursive)

        task_rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "features": ",".join(task.features),
                "synthetic": task.synthetic,
                "heldout_rows": len(heldout),
                "direct_row_exact": direct_row_exact,
                "direct_full_exact": direct_full_exact,
                "batch_row_exact": batch_row_exact,
                "batch_full_exact": batch_full_exact,
                "monolithic_row_exact": mono_row,
                "monolithic_full_exact": mono_full,
                "monolithic_train_pass": mono_train,
                "recursive_row_exact": rec_row,
                "recursive_full_exact": rec_full,
                "recursive_train_pass": rec_train,
                "recursive_gated_row_exact": gated_row,
                "recursive_gated_full_exact": gated_full,
                "recursive_gated_train_pass": gated_train,
                "recursive_gated_used_program": gated_used_program,
                "recursive_oracle_row_exact": ora_row,
                "recursive_oracle_full_exact": ora_full,
                "recursive_oracle_train_pass": ora_train,
                "recursive_shuffled_row_exact": shuf_row,
                "recursive_shuffled_full_exact": shuf_full,
                "recursive_shuffled_train_pass": shuf_train,
                "recursive_code_chars": len(rec_code),
                "recursive_candidate_count": len(recursive_cands),
                "recursive_train_pass_count": sum(1 for c in recursive_cands if c.train_pass),
                "recursive_any_train_pass": any(c.train_pass for c in recursive_cands),
                "recursive_any_heldout_pass": any(c.train_pass and c.heldout_full_exact for c in recursive_cands),
                "recursive_helped_vs_direct": bool(rec_full and not direct_full_exact),
                "recursive_hurt_vs_direct": bool(direct_full_exact and not rec_full),
                "recursive_gated_helped_vs_direct": bool(gated_full and not direct_full_exact),
                "recursive_gated_hurt_vs_direct": bool(direct_full_exact and not gated_full),
            }
        )

    pd.DataFrame(direct_rows).to_csv(run_dir / "direct_rows.csv", index=False)
    pd.DataFrame(batch_rows).to_csv(run_dir / "batch_rows.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(run_dir / "candidates.csv", index=False)
    task_df = pd.DataFrame(task_rows)
    task_df.to_csv(run_dir / "task_summary.csv", index=False)

    summary = summarize(task_df)
    summary.to_csv(run_dir / "summary.csv", index=False)
    family_summary = summarize_by_family(task_df)
    family_summary.to_csv(run_dir / "family_summary.csv", index=False)

    for p in [run_dir / "summary.csv", run_dir / "family_summary.csv", run_dir / "task_summary.csv", run_dir / "candidates.csv"]:
        shutil.copy2(p, ANALYSIS / p.name)
    make_figures(task_df, summary)
    write_report(args, task_df, summary, family_summary, run_dir, elapsed=time.time() - start)
    append_log(
        f"""
## Run `{args.run_name}`

- Time UTC: `{datetime.now(timezone.utc).isoformat()}`
- Elapsed seconds: `{time.time() - start:.1f}`
- Config: `{json.dumps(config, sort_keys=True)}`
- Tasks: `{len(task_df)}`
- Direct row-by-row full-task exact: `{pct(float(task_df['direct_full_exact'].mean()) if len(task_df) else 0.0)}`
- Recursive selected-program full-task exact: `{pct(float(task_df['recursive_full_exact'].mean()) if len(task_df) else 0.0)}`
- Recursive gated-direct full-task exact: `{pct(float(task_df['recursive_gated_full_exact'].mean()) if len(task_df) else 0.0)}`
- Recursive train-pass rate: `{pct(float(task_df['recursive_any_train_pass'].mean()) if len(task_df) else 0.0)}`
- Recursive oracle among train-passing candidates: `{pct(float(task_df['recursive_oracle_full_exact'].mean()) if len(task_df) else 0.0)}`
"""
    )


def summarize(task_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = [
        ("direct_row", "direct_row_exact", "direct_full_exact", None),
        ("direct_batch", "batch_row_exact", "batch_full_exact", None),
        ("monolithic_program", "monolithic_row_exact", "monolithic_full_exact", "monolithic_train_pass"),
        ("recursive_program", "recursive_row_exact", "recursive_full_exact", "recursive_train_pass"),
        ("recursive_gated_direct", "recursive_gated_row_exact", "recursive_gated_full_exact", "recursive_gated_train_pass"),
        ("recursive_oracle", "recursive_oracle_row_exact", "recursive_oracle_full_exact", "recursive_oracle_train_pass"),
        ("recursive_shuffled", "recursive_shuffled_row_exact", "recursive_shuffled_full_exact", "recursive_shuffled_train_pass"),
    ]
    for method, row_col, full_col, train_col in methods:
        if row_col not in task_df:
            continue
        rows.append(
            {
                "method": method,
                "tasks": len(task_df),
                "row_exact": float(task_df[row_col].mean()) if len(task_df) else 0.0,
                "full_task_exact": float(task_df[full_col].mean()) if len(task_df) else 0.0,
                "train_pass_rate": float(task_df[train_col].mean()) if train_col and train_col in task_df else math.nan,
                "tasks_helped_vs_direct": int(((task_df[full_col] == True) & (task_df["direct_full_exact"] == False)).sum()) if full_col != "direct_full_exact" else 0,
                "tasks_hurt_vs_direct": int(((task_df[full_col] == False) & (task_df["direct_full_exact"] == True)).sum()) if full_col != "direct_full_exact" else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["full_task_exact", "row_exact"], ascending=False)


def summarize_by_family(task_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, g in task_df.groupby("family"):
        for method, row_col, full_col in [
            ("direct_row", "direct_row_exact", "direct_full_exact"),
            ("recursive_gated_direct", "recursive_gated_row_exact", "recursive_gated_full_exact"),
            ("recursive_program", "recursive_row_exact", "recursive_full_exact"),
            ("recursive_oracle", "recursive_oracle_row_exact", "recursive_oracle_full_exact"),
            ("monolithic_program", "monolithic_row_exact", "monolithic_full_exact"),
        ]:
            rows.append(
                {
                    "family": family,
                    "method": method,
                    "tasks": len(g),
                    "row_exact": float(g[row_col].mean()),
                    "full_task_exact": float(g[full_col].mean()),
                }
            )
    return pd.DataFrame(rows)


def make_figures(task_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    s = summary[summary["method"].isin(["direct_row", "direct_batch", "monolithic_program", "recursive_program", "recursive_gated_direct", "recursive_oracle", "recursive_shuffled"])].copy()
    ax.bar(s["method"], s["full_task_exact"] * 100, color=["#4c78a8" if m != "recursive_program" else "#f58518" for m in s["method"]])
    ax.set_ylabel("Full-task exact (%)")
    ax.set_title("Strict Held-Out Task Accuracy")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / "method_full_task_exact.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(task_df["direct_row_exact"] * 100, task_df["recursive_row_exact"] * 100, s=55, alpha=0.8)
    ax.plot([0, 100], [0, 100], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Direct row exact (%)")
    ax.set_ylabel("Recursive program row exact (%)")
    ax.set_title("Row Accuracy: Direct vs Executed Program")
    fig.tight_layout()
    fig.savefig(FIGURES / "row_scatter_direct_vs_recursive.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts = [
        int(task_df["recursive_gated_helped_vs_direct"].sum()),
        int(task_df["recursive_gated_hurt_vs_direct"].sum()),
        int((~task_df["recursive_gated_helped_vs_direct"] & ~task_df["recursive_gated_hurt_vs_direct"]).sum()),
    ]
    ax.bar(["helped", "hurt", "tied"], counts, color=["#54a24b", "#e45756", "#9ecae9"])
    ax.set_ylabel("Tasks")
    ax.set_title("Gated Recursive Program Flips Versus Direct")
    fig.tight_layout()
    fig.savefig(FIGURES / "wins_losses_vs_direct.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    fam = task_df.groupby("family").agg(
        tasks=("task_id", "count"),
        direct=("direct_full_exact", "mean"),
        recursive=("recursive_full_exact", "mean"),
        oracle=("recursive_oracle_full_exact", "mean"),
    )
    fam = fam[fam["tasks"] >= 1].sort_values("tasks", ascending=False).head(12)
    x = range(len(fam))
    w = 0.25
    ax.bar([i - w for i in x], fam["direct"] * 100, width=w, label="direct")
    ax.bar(x, fam["recursive"] * 100, width=w, label="recursive")
    ax.bar([i + w for i in x], fam["oracle"] * 100, width=w, label="oracle")
    ax.set_xticks(list(x))
    ax.set_xticklabels(fam.index, rotation=35, ha="right")
    ax.set_ylabel("Full-task exact (%)")
    ax.set_title("Family Breakdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "family_breakdown.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(task_df["recursive_train_pass_count"], bins=range(0, int(task_df["recursive_train_pass_count"].max()) + 3), color="#72b7b2", edgecolor="white")
    ax.set_xlabel("Train-passing recursive candidates")
    ax.set_ylabel("Tasks")
    ax.set_title("Generated Program Availability")
    fig.tight_layout()
    fig.savefig(FIGURES / "train_passing_candidates.png", dpi=180)
    plt.close(fig)


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            if col.endswith("exact") or col.endswith("rate") or col in {"row_exact", "full_task_exact", "train_pass_rate"}:
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}%")
            else:
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return view.to_markdown(index=False)


def write_report(args: argparse.Namespace, task_df: pd.DataFrame, summary: pd.DataFrame, family_summary: pd.DataFrame, run_dir: Path, elapsed: float) -> None:
    report = REPORTS / "qwen_recursive_ephemeral_program_induction_report.md"
    html_report = REPORTS / "qwen_recursive_ephemeral_program_induction_report.html"

    helped = int(task_df["recursive_gated_helped_vs_direct"].sum())
    hurt = int(task_df["recursive_gated_hurt_vs_direct"].sum())
    direct = float(task_df["direct_full_exact"].mean()) if len(task_df) else 0.0
    rec = float(task_df["recursive_full_exact"].mean()) if len(task_df) else 0.0
    gated = float(task_df["recursive_gated_full_exact"].mean()) if len(task_df) else 0.0
    oracle = float(task_df["recursive_oracle_full_exact"].mean()) if len(task_df) else 0.0
    train_pass = float(task_df["recursive_any_train_pass"].mean()) if len(task_df) else 0.0

    task_view = task_df[
        [
            "task_id",
            "family",
            "heldout_rows",
            "direct_full_exact",
            "recursive_full_exact",
            "recursive_gated_full_exact",
            "recursive_oracle_full_exact",
            "recursive_train_pass_count",
            "recursive_gated_used_program",
            "recursive_gated_helped_vs_direct",
            "recursive_gated_hurt_vs_direct",
        ]
    ].sort_values(["recursive_gated_helped_vs_direct", "recursive_gated_hurt_vs_direct", "task_id"], ascending=[False, False, True])

    fam_view = family_summary.pivot(index="family", columns="method", values="full_task_exact").reset_index().fillna(0.0)

    md = f"""# Recursive Ephemeral Program Induction

## Question

Can a frozen language model convert sparse input-output examples into a task-local executable program that is more consistent than direct row-by-row answering?

The method asks the model to write a Python `transform(row)` function. Candidate programs are executed on visible examples. Only visible examples are used for selection and repair. Held-out rows are used only for final scoring.

## Setup

- Run: `{args.run_name}`
- Dataset: public text-transformation tasks.
- Tasks: `{len(task_df)}`
- Visible examples per task: `{args.train_n}`
- Held-out cap per task: `{args.heldout_cap}`
- Program variants: `{args.variants}`
- Repair rounds: `{args.repair_rounds}`
- Elapsed seconds: `{elapsed:.1f}`

## Main Result

{md_table(summary)}

## Interpretation

Direct row-by-row answering solves `{pct(direct)}` of tasks under strict full-task exactness. The selected recursive executable program solves `{pct(rec)}`. The gated recursive method, which falls back to direct answering when no non-memorizing train-passing program is available, solves `{pct(gated)}`. The hidden diagnostic oracle over train-passing generated programs solves `{pct(oracle)}`, and at least one recursive candidate passes visible examples on `{pct(train_pass)}` of tasks.

The gated recursive method helps `{helped}` tasks and hurts `{hurt}` tasks relative to direct row-by-row answering.

The shuffled-label control is included to check whether executable programs can be induced from mismatched examples. A useful executable-program result should beat both direct answering and this shuffled control.

## Charts

![Method full-task exact](../analysis/figures/method_full_task_exact.png)

![Row scatter](../analysis/figures/row_scatter_direct_vs_recursive.png)

![Wins and losses](../analysis/figures/wins_losses_vs_direct.png)

![Family breakdown](../analysis/figures/family_breakdown.png)

![Train-passing candidates](../analysis/figures/train_passing_candidates.png)

## Task Details

{md_table(task_view, max_rows=80)}

## Family Summary

{md_table(fam_view, max_rows=80)}

## Limitations

Generated code is sandboxed by a conservative AST pass, so some potentially valid programs may be rejected. The benchmark tasks are public text transformations and do not cover arbitrary software engineering problems. Full-task exact is intentionally strict and can be much lower than row accuracy.

## Artifacts

- Run directory: `{run_dir}`
- Summary: `analysis/summary.csv`
- Task details: `analysis/task_summary.csv`
- Candidate programs: `analysis/candidates.csv`
- Figures: `analysis/figures/`
"""
    report.write_text(md)
    html_body = markdown_to_html(md)
    html_report.write_text(html_body)


def markdown_to_html(md: str) -> str:
    try:
        import markdown  # type: ignore

        body = markdown.markdown(md, extensions=["tables"])
    except Exception:
        body = "<pre>" + html.escape(md) + "</pre>"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Recursive Ephemeral Program Induction</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1120px; margin: 36px auto; padding: 0 24px; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 8px 0 22px; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
    pre {{ background: #f6f8fa; padding: 16px; overflow: auto; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="main_v1")
    parser.add_argument("--task_limit", type=int, default=30)
    parser.add_argument("--sample_seed", type=int, default=20260627)
    parser.add_argument("--train_n", type=int, default=4)
    parser.add_argument("--heldout_cap", type=int, default=6)
    parser.add_argument("--min_heldout", type=int, default=3)
    parser.add_argument("--variants", default="monolithic,helpers,robust")
    parser.add_argument("--repair_variants", default="minimal,broaden,conditional")
    parser.add_argument("--repair_rounds", type=int, default=2)
    parser.add_argument("--answer_max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_max_new_tokens", type=int, default=320)
    parser.add_argument("--code_max_new_tokens", type=int, default=520)
    parser.add_argument("--repair_max_new_tokens", type=int, default=560)
    parser.add_argument("--run_shuffled", action="store_true")
    parser.add_argument("--no_qwen", action="store_true")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
