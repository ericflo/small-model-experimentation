#!/usr/bin/env python3
"""Counterexample-guided task-local program induction.

The experiment asks a local model to generate executable Python transformation
programs from a few visible examples. Visible examples alone can select brittle
programs, so this runner creates synthetic rows where train-passing candidate
programs disagree, labels those probes with the model, and uses the resulting
counterexamples to select or route programs. Held-out benchmark outputs are used
only for final scoring and oracle diagnostics.
"""

from __future__ import annotations

import argparse
import ast
import calendar
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
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path("/workspace/experiments/qwen_counterexample_guided_ephemeral_program")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_counterexample_guided_ephemeral_program")
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
    output: str = ""


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    synthetic: bool
    features: Tuple[str, ...]
    examples: Tuple[Example, ...]
    source_path: str


@dataclass
class ProgramCandidate:
    task_id: str
    variant: str
    raw_text: str
    code: str
    status: str
    train_pass: bool
    train_row_exact: float
    heldout_row_exact: float
    heldout_full_exact: bool
    code_chars: int
    suspicious: bool
    error: str = ""


@dataclass
class SelectedProgram:
    name: str
    row_exact: float
    full_exact: bool
    predictions: List[str]
    description: str
    train_pass: bool = False
    probe_score: float = 0.0
    used_program: bool = False


def ensure_dirs() -> None:
    for d in [ROOT, LARGE_ROOT, RUNS, ANALYSIS, FIGURES, REPORTS, ROOT / "src"]:
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


def compact(s: Any) -> str:
    return re.sub(r"\s+", "", "" if s is None else str(s)).strip()


def norm_eq(a: Any, b: Any) -> bool:
    return compact(a) == compact(b)


def clean_prediction(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text or "").strip()
    text = re.sub(r"^```(?:text|json|python)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    if text.lower().startswith("output:"):
        text = text.split(":", 1)[1].strip()
    first = text.splitlines()[0].strip() if text else ""
    if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
        first = first[1:-1].strip()
    return first


def render_inputs(vals: Sequence[str]) -> str:
    if len(vals) == 1:
        return vals[0]
    return " | ".join(f"col{i}={v}" for i, v in enumerate(vals))


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def pct(x: float) -> str:
    if x != x:
        return ""
    return f"{100 * x:.1f}%"


def load_tasks(min_examples: int) -> List[Task]:
    tasks: List[Task] = []
    for d in sorted(TRANSFORM_ROOT.iterdir()):
        if not d.is_dir() or not (d / "spec.json").exists():
            continue
        spec = json.loads((d / "spec.json").read_text())
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        examples = [
            Example(tuple(str(x) for x in ex.get("Input", [])), str(ex.get("Output", "")))
            for ex in spec.get("Examples", [])
        ]
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
    chosen = list(tasks)
    if limit and limit < len(chosen):
        chosen = rng.sample(chosen, limit)
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
    if model is None:
        return ""
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: Dict[str, Dict[str, str]] = {}
        if self.path.exists():
            with self.path.open(newline="") as f:
                for row in csv.DictReader(f):
                    self.rows[row["key"]] = row
        self.file = self.path.open("a", newline="")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=["key", "task_id", "phase", "method", "row_id", "prediction", "raw"],
        )
        if self.path.stat().st_size == 0:
            self.writer.writeheader()
            self.file.flush()

    def close(self) -> None:
        self.file.close()

    def key(self, task_id: str, phase: str, method: str, row_id: str, system: str, user: str, max_new_tokens: int) -> str:
        payload = json.dumps(
            {
                "model": MODEL_NAME,
                "task_id": task_id,
                "phase": phase,
                "method": method,
                "row_id": row_id,
                "system": system,
                "user": user,
                "max_new_tokens": max_new_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self,
        tok: Any,
        model: Any,
        task_id: str,
        phase: str,
        method: str,
        row_id: str,
        system: str,
        user: str,
        max_new_tokens: int,
        prediction_parser: Callable[[str], str] = clean_prediction,
    ) -> Tuple[str, str]:
        key = self.key(task_id, phase, method, row_id, system, user, max_new_tokens)
        if key in self.rows:
            row = self.rows[key]
            return row["prediction"], row["raw"]
        raw = generate_text(tok, model, system, user, max_new_tokens)
        pred = prediction_parser(raw)
        row = {
            "key": key,
            "task_id": task_id,
            "phase": phase,
            "method": method,
            "row_id": row_id,
            "prediction": pred,
            "raw": raw,
        }
        self.rows[key] = row
        self.writer.writerow(row)
        self.file.flush()
        return pred, raw


def direct_prompt(train: Sequence[Example], query: Example, style: str = "plain") -> str:
    opener = "Infer the exact text transformation from the examples."
    if style == "rule":
        opener = "Infer the single deterministic rule that maps each input to its output."
    elif style == "strict":
        opener = "Use all examples consistently. Do not copy an example unless the query is identical."
    lines = [opener, "Return only the transformed output for the query. Do not explain.", "", "Examples:"]
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
    text = re.sub(r"(?is)<think>.*?</think>", "", text or "").strip()
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
        return (lines[:n] + [""] * max(0, n - len(lines))), False


def examples_block(examples: Sequence[Example]) -> str:
    return "\n".join(f"{i}. transform({list(ex.inputs)!r}) -> {ex.output!r}" for i, ex in enumerate(examples, 1))


HELPER_DOC = """Available helper functions:
- clean(x), words(x), alpha(x), alnum(x), digits(x)
- first_word(x), last_word(x), word(x, i)
- field(x, sep, i), before(x, sep), after(x, sep), between(x, left, right)
- regex_group(x, pattern, group=1)
- number(x), number_int(x), number_1dp(x), number_2dp(x), round_to(x, base)
- year(x), month_num(x), month_2(x), month_name(x), month_abbr(x), day(x), day_2(x)
- hour(x), hour_2(x), minute_2(x)
"""


def code_prompt(train: Sequence[Example], variant: str) -> str:
    if variant == "regex":
        style = "Prefer regular expressions and delimiter parsing when appropriate."
    elif variant == "conditional":
        style = "Use explicit conditionals only when the examples imply a real case distinction."
    elif variant == "helpers":
        style = "Use helper functions for parsing and formatting, then keep transform(row) short."
    else:
        style = "Write one concise robust function. Avoid memorizing individual examples."
    return f"""Write Python code for a deterministic text transformation.

Requirements:
- Define exactly one public function named transform(row).
- row is a list of input columns. Use row[0] for a single-column task.
- Return a string.
- Use only Python built-ins, re, math, datetime, calendar, and the helper functions listed below.
- Do not read files, import packages other than allowed modules, print, or include tests.
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

All visible examples:
{examples_block(train)}

Failing visible cases:
{chr(10).join(fail_lines)}

Previous code:
```python
{previous_code}
```

Repair style: {variant}. Prefer a general rule over a literal lookup table.
Return only a Python code block."""


def clean_helper(x: Any) -> str:
    return clean(x)


def words(x: Any) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", str(x))


def alpha(x: Any) -> str:
    return "".join(re.findall(r"[A-Za-z]+", str(x)))


def alnum(x: Any) -> str:
    return "".join(re.findall(r"[A-Za-z0-9]+", str(x)))


def digits(x: Any) -> str:
    return "".join(re.findall(r"\d+", str(x)))


def first_word(x: Any) -> str:
    ws = words(x)
    return ws[0] if ws else ""


def last_word(x: Any) -> str:
    ws = words(x)
    return ws[-1] if ws else ""


def word(x: Any, i: int) -> str:
    ws = words(x)
    try:
        return ws[i]
    except Exception:
        return ""


def field(x: Any, sep: str, i: int) -> str:
    parts = str(x).split(sep)
    try:
        return parts[i].strip()
    except Exception:
        return ""


def before(x: Any, sep: str) -> str:
    return str(x).split(sep, 1)[0].strip()


def after(x: Any, sep: str) -> str:
    s = str(x)
    return s.split(sep, 1)[1].strip() if sep in s else ""


def between(x: Any, left: str, right: str) -> str:
    s = str(x)
    if left in s:
        s = s.split(left, 1)[1]
    if right in s:
        s = s.split(right, 1)[0]
    return s.strip()


def regex_group(x: Any, pattern: str, group: int = 1) -> str:
    m = re.search(pattern, str(x))
    if not m:
        return ""
    try:
        return m.group(group)
    except Exception:
        return ""


def number(x: Any) -> float:
    m = re.search(r"-?\d+(?:\.\d+)?", str(x).replace(",", ""))
    return float(m.group(0)) if m else 0.0


def number_int(x: Any) -> str:
    return str(int(round(number(x))))


def number_1dp(x: Any) -> str:
    return f"{number(x):.1f}"


def number_2dp(x: Any) -> str:
    return f"{number(x):.2f}"


def round_to(x: Any, base: int) -> str:
    b = max(1, int(base))
    return str(int(round(number(x) / b) * b))


MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})


def date_parts(x: Any) -> Tuple[int, int, int]:
    s = str(x)
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})", s)
    if m:
        y = int(m.group(3))
        if y < 100:
            y += 2000
        return y, int(m.group(1)), int(m.group(2))
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", s)
    if m:
        return int(m.group(3)), MONTHS.get(m.group(1).lower(), 0), int(m.group(2))
    nums = [int(v) for v in re.findall(r"\d+", s)]
    if len(nums) >= 3:
        y = nums[0] if nums[0] > 31 else nums[2]
        if y < 100:
            y += 2000
        return y, nums[1], nums[2] if nums[0] > 31 else nums[0]
    return 0, 0, 0


def year(x: Any) -> str:
    return str(date_parts(x)[0])


def month_num(x: Any) -> str:
    return str(date_parts(x)[1])


def month_2(x: Any) -> str:
    return f"{date_parts(x)[1]:02d}"


def month_name(x: Any) -> str:
    m = date_parts(x)[1]
    return calendar.month_name[m] if 1 <= m <= 12 else ""


def month_abbr(x: Any) -> str:
    m = date_parts(x)[1]
    return calendar.month_abbr[m] if 1 <= m <= 12 else ""


def day(x: Any) -> str:
    return str(date_parts(x)[2])


def day_2(x: Any) -> str:
    return f"{date_parts(x)[2]:02d}"


def time_parts(x: Any) -> Tuple[int, int, int]:
    s = str(x)
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AP]M)?", s, flags=re.I)
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


def hour(x: Any) -> str:
    return str(time_parts(x)[0])


def hour_2(x: Any) -> str:
    return f"{time_parts(x)[0]:02d}"


def minute_2(x: Any) -> str:
    return f"{time_parts(x)[1]:02d}"


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
    raw = re.sub(r"(?is)<think>.*?</think>", "", raw or "").strip()
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw, flags=re.I)
    code = m.group(1).strip() if m else raw
    if "def transform" in code:
        code = code[code.find("def transform") :]
    return code.strip()


def compile_transform(code: str) -> Tuple[Optional[Callable[[Sequence[str]], str]], str]:
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
        "re": re,
        "math": math,
        "datetime": __import__("datetime"),
        "calendar": calendar,
        "clean": clean_helper,
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


def run_transform(fn: Callable[[Sequence[str]], Any], ex: Example) -> Tuple[str, str]:
    try:
        return clean(fn(list(ex.inputs))), "ok"
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def score_outputs(preds: Sequence[str], examples: Sequence[Example]) -> Tuple[float, bool]:
    if not examples:
        return 0.0, False
    exact = [norm_eq(p, ex.output) for p, ex in zip(preds, examples)]
    return sum(exact) / len(exact), all(exact)


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


def evaluate_candidate(raw: str, task_id: str, variant: str, train: Sequence[Example], heldout: Sequence[Example]) -> ProgramCandidate:
    code = extract_code(raw)
    fn, status = compile_transform(code)
    if fn is None:
        return ProgramCandidate(task_id, variant, raw, code, status, False, 0.0, 0.0, False, len(code), False, status)
    train_preds = [run_transform(fn, ex)[0] for ex in train]
    held_preds = [run_transform(fn, ex)[0] for ex in heldout]
    train_row, train_full = score_outputs(train_preds, train)
    held_row, held_full = score_outputs(held_preds, heldout)
    return ProgramCandidate(
        task_id=task_id,
        variant=variant,
        raw_text=raw,
        code=code,
        status=status,
        train_pass=train_full,
        train_row_exact=train_row,
        heldout_row_exact=held_row,
        heldout_full_exact=held_full,
        code_chars=len(code),
        suspicious=suspicious_memorizer(code, train),
    )


def failing_train_cases(cand: ProgramCandidate, train: Sequence[Example]) -> List[Tuple[Example, str]]:
    fn = candidate_fn(cand)
    if fn is None:
        return [(ex, "<compile failure>") for ex in train[:4]]
    fails = []
    for ex in train:
        pred, _ = run_transform(fn, ex)
        if not norm_eq(pred, ex.output):
            fails.append((ex, pred))
    return fails[:4]


def candidate_fn(cand: ProgramCandidate) -> Optional[Callable[[Sequence[str]], str]]:
    fn, _ = compile_transform(cand.code)
    return fn


def eval_candidate_on(cand: ProgramCandidate, rows: Sequence[Example]) -> List[str]:
    fn = candidate_fn(cand)
    if fn is None:
        return [""] * len(rows)
    return [run_transform(fn, ex)[0] for ex in rows]


def mutate_digits(s: str, rng: random.Random) -> str:
    def repl(m: re.Match[str]) -> str:
        old = m.group(0)
        width = len(old)
        val = int(old) if old.isdigit() else rng.randint(1, 999)
        new = (val + rng.choice([1, 2, 3, 7, 11, 13, 17])) % (10**min(width, 6))
        if width > 1:
            return f"{new:0{width}d}"[-width:]
        return str(new)
    return re.sub(r"\d+", repl, s)


def mutate_text_value(s: str, rng: random.Random) -> List[str]:
    outs = {s}
    if re.search(r"\d", s):
        outs.add(mutate_digits(s, rng))
    if any(ch.isalpha() for ch in s):
        outs.add(s.upper())
        outs.add(s.lower())
        outs.add(s.title())
    replacements = [("-", "/"), ("/", "-"), (":", "."), (".", "-"), ("_", "-"), (" ", "_"), (",", "")]
    for a, b in replacements:
        if a in s:
            outs.add(s.replace(a, b))
    tokens = re.findall(r"[A-Za-z0-9]+", s)
    if len(tokens) >= 2:
        outs.add(" ".join(reversed(tokens)))
        outs.add(tokens[0] + " " + tokens[-1])
    if "@" in s:
        name, _, domain = s.partition("@")
        outs.add(f"{name}{rng.randint(10,99)}@{domain}")
    if len(s) > 6:
        outs.add(s[: max(1, len(s) // 2)])
    return [x for x in outs if clean(x) and x != s]


def synthetic_rows(train: Sequence[Example], count: int, seed: str) -> List[Example]:
    rng = random.Random(stable_int(seed))
    rows: List[Example] = []
    for ex in train:
        for col_i, value in enumerate(ex.inputs):
            for mutated in mutate_text_value(value, rng):
                vals = list(ex.inputs)
                vals[col_i] = mutated
                rows.append(Example(tuple(vals), ""))
    for ex in train:
        vals = list(ex.inputs)
        if vals:
            vals[0] = f"{vals[0]} {rng.randint(10, 999)}"
            rows.append(Example(tuple(vals), ""))
    dedup: Dict[Tuple[str, ...], Example] = {}
    for row in rows:
        dedup[row.inputs] = row
    rows = list(dedup.values())
    rng.shuffle(rows)
    return rows[: max(count * 4, count)]


def distinct_outputs(cands: Sequence[ProgramCandidate], row: Example) -> Tuple[int, Dict[str, int]]:
    counts: Counter[str] = Counter()
    for cand in cands:
        fn = candidate_fn(cand)
        if fn is None:
            continue
        pred, _ = run_transform(fn, row)
        counts[pred] += 1
    return len(counts), dict(counts)


def select_probe_rows(cands: Sequence[ProgramCandidate], train: Sequence[Example], count: int, seed: str, mode: str) -> List[Example]:
    pool = synthetic_rows(train, count=max(20, count * 8), seed=seed)
    scored = []
    for row in pool:
        n, counts = distinct_outputs(cands, row)
        balance = 0.0
        if counts:
            total = sum(counts.values())
            balance = 1.0 - max(counts.values()) / total
        scored.append((n, balance, len(render_inputs(row.inputs)), row))
    rng = random.Random(stable_int(seed + mode))
    if mode == "random":
        rng.shuffle(scored)
        return [x[-1] for x in scored[:count]]
    scored.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
    return [x[-1] for x in scored if x[0] > 1][:count]


def label_probe(
    tok: Any,
    model: Any,
    cache: GenerationCache,
    task_id: str,
    train: Sequence[Example],
    probe: Example,
    styles: Sequence[str],
    max_new_tokens: int,
) -> Tuple[str, float, List[str]]:
    labels = []
    system = "You are a precise text-transformation engine. Return only the requested output."
    for style in styles:
        pred, _ = cache.get(
            tok,
            model,
            task_id,
            "probe_label",
            f"label_{style}",
            hashlib.sha256(render_inputs(probe.inputs).encode("utf-8")).hexdigest()[:12],
            system,
            direct_prompt(train, probe, style=style),
            max_new_tokens,
        )
        labels.append(clean(pred))
    if not labels:
        return "", 0.0, []
    counts = Counter(compact(x) for x in labels)
    winner_key, winner_count = counts.most_common(1)[0]
    winner = next(x for x in labels if compact(x) == winner_key)
    return winner, winner_count / len(labels), labels


def candidate_probe_score(cand: ProgramCandidate, probes: Sequence[Example]) -> float:
    if not probes:
        return 1.0
    preds = eval_candidate_on(cand, probes)
    return sum(norm_eq(p, ex.output) for p, ex in zip(preds, probes)) / len(probes)


def select_visible(cands: Sequence[ProgramCandidate]) -> Optional[ProgramCandidate]:
    passed = [c for c in cands if c.train_pass]
    if not passed:
        return None
    return sorted(passed, key=lambda c: (c.suspicious, c.code_chars, c.variant))[0]


def select_by_probes(cands: Sequence[ProgramCandidate], probes: Sequence[Example]) -> Tuple[Optional[ProgramCandidate], float]:
    passed = [c for c in cands if c.train_pass]
    if not passed:
        return None, 0.0
    scored = [(candidate_probe_score(c, probes), c) for c in passed]
    scored.sort(key=lambda x: (-x[0], x[1].suspicious, x[1].code_chars, x[1].variant))
    return scored[0][1], scored[0][0]


Predicate = Tuple[str, Callable[[Example], bool]]


def build_predicates(labeled_rows: Sequence[Example]) -> List[Predicate]:
    texts = [render_inputs(ex.inputs) for ex in labeled_rows]
    lens = sorted(len(t) for t in texts)
    med = lens[len(lens) // 2] if lens else 0
    chars = ["@", "/", "-", ":", ".", ",", "$", "(", ")", "_", "#"]
    preds: List[Predicate] = []
    for ch in chars:
        preds.append((f"contains {ch!r}", lambda ex, ch=ch: ch in render_inputs(ex.inputs)))
    preds.extend(
        [
            ("has_digit", lambda ex: any(c.isdigit() for c in render_inputs(ex.inputs))),
            ("has_alpha", lambda ex: any(c.isalpha() for c in render_inputs(ex.inputs))),
            ("starts_digit", lambda ex: bool(render_inputs(ex.inputs)) and render_inputs(ex.inputs)[0].isdigit()),
            ("len_gt_median", lambda ex, med=med: len(render_inputs(ex.inputs)) > med),
            ("multi_column", lambda ex: len(ex.inputs) > 1),
        ]
    )
    return preds


def fit_router(cands: Sequence[ProgramCandidate], train: Sequence[Example], probes: Sequence[Example]) -> Tuple[Any, float, str]:
    passed = [c for c in cands if c.train_pass]
    labels = list(train) + list(probes)
    if len(passed) < 2 or len(labels) < 3:
        return None, 0.0, ""
    preds_by_cand = {id(c): eval_candidate_on(c, labels) for c in passed}
    best: Tuple[float, int, Any, str] = (-1.0, 0, None, "")
    for pred_name, pred_fn in build_predicates(labels):
        sides = [pred_fn(ex) for ex in labels]
        if not any(sides) or all(sides):
            continue
        for left in passed:
            for right in passed:
                combined = []
                for i, side in enumerate(sides):
                    combined.append(preds_by_cand[id(left)][i] if side else preds_by_cand[id(right)][i])
                score = sum(norm_eq(p, ex.output) for p, ex in zip(combined, labels)) / len(labels)
                complexity = left.code_chars + right.code_chars
                desc = f"if {pred_name}: {left.variant} else {right.variant}"
                if (score, -complexity) > (best[0], best[1]):
                    best = (score, -complexity, (pred_fn, left, right), desc)
    return best[2], best[0], best[3]


def eval_router(router: Any, rows: Sequence[Example]) -> List[str]:
    if router is None:
        return [""] * len(rows)
    pred_fn, left, right = router
    left_fn = candidate_fn(left)
    right_fn = candidate_fn(right)
    outs = []
    for ex in rows:
        fn = left_fn if pred_fn(ex) else right_fn
        if fn is None:
            outs.append("")
        else:
            outs.append(run_transform(fn, ex)[0])
    return outs


def eval_program_selection(
    name: str,
    cand: Optional[ProgramCandidate],
    heldout: Sequence[Example],
    probe_score: float = 0.0,
) -> SelectedProgram:
    if cand is None:
        return SelectedProgram(name, 0.0, False, [""] * len(heldout), "", False, probe_score, False)
    preds = eval_candidate_on(cand, heldout)
    row, full = score_outputs(preds, heldout)
    return SelectedProgram(name, row, full, preds, cand.variant, cand.train_pass, probe_score, True)


def oracle_candidate(cands: Sequence[ProgramCandidate]) -> Optional[ProgramCandidate]:
    passed = [c for c in cands if c.train_pass]
    if not passed:
        return None
    return sorted(passed, key=lambda c: (not c.heldout_full_exact, -c.heldout_row_exact, c.code_chars))[0]


def direct_answers(
    tok: Any,
    model: Any,
    cache: GenerationCache,
    task_id: str,
    train: Sequence[Example],
    heldout: Sequence[Example],
    max_new_tokens: int,
) -> Tuple[List[str], float, bool]:
    preds = []
    system = "You are a precise text-transformation engine. Return only the requested output."
    for i, ex in enumerate(heldout):
        pred, _ = cache.get(tok, model, task_id, "answer", "direct_row", str(i), system, direct_prompt(train, ex), max_new_tokens)
        preds.append(pred)
    row, full = score_outputs(preds, heldout)
    return preds, row, full


def batch_answers(
    tok: Any,
    model: Any,
    cache: GenerationCache,
    task_id: str,
    train: Sequence[Example],
    heldout: Sequence[Example],
    max_new_tokens: int,
) -> Tuple[List[str], float, bool, bool]:
    system = "You are a precise text-transformation engine. Return only the requested output."
    _, raw = cache.get(
        tok,
        model,
        task_id,
        "answer",
        "direct_batch",
        "all",
        system,
        batch_prompt(train, heldout),
        max_new_tokens,
        prediction_parser=lambda x: x,
    )
    preds, parse_ok = parse_json_outputs(raw, len(heldout))
    row, full = score_outputs(preds, heldout)
    return preds, row, full, parse_ok


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows).copy().fillna("")
    for col in shown.columns:
        if any(key in str(col) for key in ["exact", "rate", "score", "consensus"]):
            def fmt(v: Any) -> Any:
                if v == "":
                    return v
                try:
                    fv = float(v)
                except Exception:
                    return v
                return pct(fv) if -1.0 <= fv <= 1.0 else f"{fv:.3f}"
            shown[col] = shown[col].map(fmt)
    return shown.to_markdown(index=False)


def html_report(md: str) -> str:
    chunks: List[str] = []
    table_buf: List[str] = []

    def flush_table() -> None:
        if table_buf:
            chunks.append("<pre class='table'>" + html.escape("\n".join(table_buf)) + "</pre>")
            table_buf.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            table_buf.append(line)
            continue
        flush_table()
        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if image_match:
            alt = html.escape(image_match.group(1))
            src = html.escape(image_match.group(2))
            chunks.append(f"<figure><img src='{src}' alt='{alt}'><figcaption>{alt}</figcaption></figure>")
        elif line.startswith("# "):
            chunks.append(f"<h1>{html.escape(line[2:].strip())}</h1>")
        elif line.startswith("## "):
            chunks.append(f"<h2>{html.escape(line[3:].strip())}</h2>")
        elif line.startswith("### "):
            chunks.append(f"<h3>{html.escape(line[4:].strip())}</h3>")
        elif not line.strip():
            chunks.append("")
        elif line.startswith("- "):
            chunks.append(f"<p>{html.escape(line)}</p>")
        else:
            escaped = html.escape(line)
            escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
            chunks.append(f"<p>{escaped}</p>")
    flush_table()
    body = "\n".join(chunks)
    css = """
    body { font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; margin: 40px; color: #111827; line-height: 1.5; }
    h1 { font-size: 32px; }
    h2 { margin-top: 30px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
    code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }
    img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 4px; }
    figure { margin: 20px 0 30px; }
    figcaption { color: #4b5563; font-size: 13px; margin-top: 6px; }
    pre.table { overflow-x: auto; background: #f9fafb; border: 1px solid #e5e7eb; padding: 12px; border-radius: 4px; font-size: 12px; }
    """
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Counterexample-Guided Ephemeral Program</title><style>{css}</style></head><body>{body}</body></html>"


def summarize(task_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("direct_qwen_row", "direct_row_exact", "direct_full_exact", None),
        ("direct_qwen_batch", "batch_row_exact", "batch_full_exact", None),
        ("program_visible", "visible_row_exact", "visible_full_exact", "visible_used_program"),
        ("program_ceg", "ceg_row_exact", "ceg_full_exact", "ceg_used_program"),
        ("program_ceg_gated", "ceg_gated_row_exact", "ceg_gated_full_exact", "ceg_gated_used_program"),
        ("program_ceg_router", "router_row_exact", "router_full_exact", "router_used"),
        ("program_random_probe", "random_probe_row_exact", "random_probe_full_exact", "random_probe_used_program"),
        ("program_shuffled_probe_labels", "shuffled_probe_row_exact", "shuffled_probe_full_exact", "shuffled_probe_used_program"),
        ("hidden_candidate_oracle", "oracle_row_exact", "oracle_full_exact", "oracle_used_program"),
    ]
    for method, row_col, full_col, used_col in specs:
        if row_col not in task_df:
            continue
        rows.append(
            {
                "method": method,
                "tasks": len(task_df),
                "row_exact": float(task_df[row_col].mean()) if len(task_df) else 0.0,
                "full_task_exact": float(task_df[full_col].mean()) if len(task_df) else 0.0,
                "used_program_rate": float(task_df[used_col].mean()) if used_col and used_col in task_df else math.nan,
                "helped_vs_direct": int(((task_df[full_col] == True) & (task_df["direct_full_exact"] == False)).sum())
                if full_col != "direct_full_exact"
                else 0,
                "hurt_vs_direct": int(((task_df[full_col] == False) & (task_df["direct_full_exact"] == True)).sum())
                if full_col != "direct_full_exact"
                else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["full_task_exact", "row_exact"], ascending=False)


def family_summary(task_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, row_col, full_col in [
        ("direct_qwen_row", "direct_row_exact", "direct_full_exact"),
        ("program_visible", "visible_row_exact", "visible_full_exact"),
        ("program_ceg", "ceg_row_exact", "ceg_full_exact"),
        ("program_ceg_gated", "ceg_gated_row_exact", "ceg_gated_full_exact"),
        ("program_ceg_router", "router_row_exact", "router_full_exact"),
    ]:
        for fam, g in task_df.groupby("family"):
            rows.append(
                {
                    "method": method,
                    "family": fam,
                    "tasks": len(g),
                    "row_exact": float(g[row_col].mean()),
                    "full_task_exact": float(g[full_col].mean()),
                }
            )
    return pd.DataFrame(rows)


def make_figures(task_df: pd.DataFrame, summary: pd.DataFrame, family_df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    s = summary.sort_values("full_task_exact", ascending=True)
    plt.barh(s["method"], s["full_task_exact"], color="#3b82f6")
    plt.xlabel("Full-task exact")
    plt.xlim(0, 1)
    plt.title("Strict Full-Task Exact by Method")
    plt.tight_layout()
    plt.savefig(FIGURES / "method_full_task_exact.png", dpi=170)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(task_df["direct_row_exact"], task_df["ceg_gated_row_exact"], s=60, alpha=0.8, color="#0f766e")
    plt.plot([0, 1], [0, 1], "--", color="#6b7280")
    plt.xlabel("Direct row exact")
    plt.ylabel("CEG gated row exact")
    plt.title("Row Accuracy Movement")
    plt.tight_layout()
    plt.savefig(FIGURES / "row_scatter_direct_vs_ceg.png", dpi=170)
    plt.close()

    methods = ["program_visible", "program_ceg", "program_ceg_gated", "program_ceg_router", "program_shuffled_probe_labels"]
    helps = []
    hurts = []
    for m in methods:
        row = summary[summary.method.eq(m)]
        helps.append(int(row["helped_vs_direct"].iloc[0]) if not row.empty else 0)
        hurts.append(int(row["hurt_vs_direct"].iloc[0]) if not row.empty else 0)
    x = range(len(methods))
    plt.figure(figsize=(12, 5))
    plt.bar(x, helps, label="helped", color="#16a34a")
    plt.bar(x, [-h for h in hurts], label="hurt", color="#dc2626")
    plt.xticks(list(x), methods, rotation=30, ha="right")
    plt.ylabel("Tasks versus direct")
    plt.title("Wins and Losses Versus Direct Qwen")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "wins_losses_vs_direct.png", dpi=170)
    plt.close()

    diag_cols = ["train_pass_count", "confident_probe_count", "ceg_probe_score", "router_probe_score"]
    plt.figure(figsize=(10, 6))
    vals = [float(task_df[c].mean()) if c in task_df and len(task_df) else 0.0 for c in diag_cols]
    plt.bar(diag_cols, vals, color=["#6366f1", "#0891b2", "#ea580c", "#9333ea"])
    plt.xticks(rotation=20, ha="right")
    plt.title("Program and Probe Diagnostics")
    plt.tight_layout()
    plt.savefig(FIGURES / "probe_diagnostics.png", dpi=170)
    plt.close()

    pivot = family_df.pivot_table(index="family", columns="method", values="full_task_exact", aggfunc="mean")
    if not pivot.empty:
        plt.figure(figsize=(12, max(5, 0.35 * len(pivot))))
        plt.imshow(pivot.fillna(0).values, aspect="auto", vmin=0, vmax=1, cmap="Blues")
        plt.colorbar(label="Full-task exact")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
        plt.title("Family Breakdown")
        plt.tight_layout()
        plt.savefig(FIGURES / "family_heatmap.png", dpi=170)
        plt.close()


def write_report(args: argparse.Namespace, summary: pd.DataFrame, task_df: pd.DataFrame, family_df: pd.DataFrame, candidate_df: pd.DataFrame, probe_df: pd.DataFrame, elapsed: float) -> None:
    def method_metric(method: str, col: str = "full_task_exact") -> float:
        row = summary[summary.method.eq(method)]
        return float(row[col].iloc[0]) if not row.empty else float("nan")

    direct = method_metric("direct_qwen_row")
    visible = method_metric("program_visible")
    ceg = method_metric("program_ceg")
    gated = method_metric("program_ceg_gated")
    router = method_metric("program_ceg_router")
    shuffled = method_metric("program_shuffled_probe_labels")
    oracle = method_metric("hidden_candidate_oracle")
    gated_used = method_metric("program_ceg_gated", "used_program_rate")
    raw_best = max(ceg, router)
    if oracle <= direct:
        verdict = (
            f"Negative for executable-program improvement: the hidden candidate oracle is {pct(oracle)}, below direct Qwen at {pct(direct)}. "
            f"This means the generated train-passing program set usually lacks a better candidate to select. "
            f"The gated method ties direct at {pct(gated)} by falling back on most tasks; it uses a program on {pct(gated_used)} of tasks and produces no net task wins."
        )
        headroom_text = (
            f"The hidden candidate oracle reaches `{pct(oracle)}` full-task exact, below direct Qwen. "
            "That makes candidate reachability the binding failure: even a perfect selector over these generated programs would not improve the task set."
        )
    elif gated > direct and gated > visible and gated > shuffled:
        verdict = (
            f"Positive: counterexample-guided gated programs beat direct Qwen ({pct(direct)} -> {pct(gated)}), "
            f"visible-only program selection ({pct(visible)}), and shuffled probe labels ({pct(shuffled)})."
        )
        headroom_text = (
            f"The hidden candidate oracle reaches `{pct(oracle)}` full-task exact. "
            "This measures candidate reachability if a selector could choose with held-out knowledge."
        )
    elif raw_best > visible and raw_best > shuffled:
        verdict = (
            f"Mixed: disagreement probes carry real signal, but the deployable best does not clearly beat direct Qwen. "
            f"Direct Qwen is {pct(direct)}, visible-only programs are {pct(visible)}, best raw counterexample-guided arm is {pct(raw_best)}, "
            f"and shuffled probe labels are {pct(shuffled)}."
        )
        headroom_text = (
            f"The hidden candidate oracle reaches `{pct(oracle)}` full-task exact. "
            "This measures candidate reachability if a selector could choose with held-out knowledge."
        )
    else:
        verdict = (
            f"Negative: counterexample-guided selection does not improve over direct Qwen or visible-only selection. "
            f"Direct Qwen is {pct(direct)}, visible-only programs are {pct(visible)}, gated CEG is {pct(gated)}, "
            f"router CEG is {pct(router)}, and shuffled probe labels are {pct(shuffled)}."
        )
        headroom_text = (
            f"The hidden candidate oracle reaches `{pct(oracle)}` full-task exact. "
            "This measures candidate reachability if a selector could choose with held-out knowledge."
        )

    task_cols = [
        "task_id",
        "family",
        "direct_full_exact",
        "visible_full_exact",
        "ceg_full_exact",
        "ceg_gated_full_exact",
        "router_full_exact",
        "oracle_full_exact",
        "train_pass_count",
        "confident_probe_count",
        "ceg_probe_score",
        "router_probe_score",
        "ceg_description",
        "router_description",
    ]
    cand_cols = ["task_id", "variant", "status", "train_pass", "heldout_full_exact", "heldout_row_exact", "code_chars", "suspicious"]
    probe_cols = ["task_id", "probe_index", "selected_mode", "input", "label", "consensus", "candidate_disagreement", "candidate_outputs_json"]
    md = f"""# Counterexample-Guided Ephemeral Program

## Question

Can synthetic disagreement probes convert model-generated train-fitting programs into a more reliable task-local executable rule than direct row-by-row inference?

The method generates candidate `transform(row)` programs from visible examples, finds synthetic inputs where those programs disagree, asks the model to label those probes, and selects or routes programs against the expanded label set. Held-out benchmark outputs are used only for final evaluation and hidden oracle diagnostics.

## Setup

- Run: `{args.run_name}`
- Dataset: public text-transformation tasks.
- Tasks: `{len(task_df)}`
- Visible examples per task: `{args.train_n}`
- Held-out rows per task cap: `{args.heldout_cap}`
- Program variants per task: `{args.variants}`
- Repair rounds for non-passing programs: `{args.repair_rounds}`
- Disagreement probes per task: `{args.probe_count}`
- Probe label styles: `{args.probe_label_styles}`
- Elapsed seconds: `{elapsed:.1f}`

## Main Result

{md_table(summary, max_rows=30)}

## Interpretation

{verdict}

{headroom_text}

## Charts

![Full-task exact by method](../analysis/figures/method_full_task_exact.png)

![Row movement](../analysis/figures/row_scatter_direct_vs_ceg.png)

![Wins and losses](../analysis/figures/wins_losses_vs_direct.png)

![Probe diagnostics](../analysis/figures/probe_diagnostics.png)

![Family heatmap](../analysis/figures/family_heatmap.png)

## Task Details

{md_table(task_df[task_cols].sort_values(["ceg_gated_full_exact", "direct_full_exact", "task_id"], ascending=[False, True, True]), max_rows=120)}

## Probe Labels

{md_table(probe_df[probe_cols], max_rows=120) if not probe_df.empty else "_No probes were labeled._"}

## Candidate Programs

{md_table(candidate_df[cand_cols], max_rows=160) if not candidate_df.empty else "_No candidate rows._"}

## Family Breakdown

{md_table(family_df, max_rows=180)}

## Files

- `runs/{args.run_name}/task_summary.csv`
- `runs/{args.run_name}/candidate_programs.csv`
- `runs/{args.run_name}/probe_labels.csv`
- `runs/{args.run_name}/generations.csv`
- `analysis/*.csv`
- `analysis/figures/*.png`
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "qwen_counterexample_guided_ephemeral_program_report.md").write_text(md)
    (REPORTS / "qwen_counterexample_guided_ephemeral_program_report.html").write_text(html_report(md))


def append_log(text: str) -> None:
    with (ROOT / "experiment_log.md").open("a") as f:
        f.write(text.rstrip() + "\n")


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    mirror_benchmark()
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    tasks = choose_tasks(load_tasks(args.train_n + args.min_heldout), args.task_limit, args.seed)
    tok = model = None
    if not args.no_qwen:
        tok, model = load_qwen()
    cache = GenerationCache(run_dir / "generations.csv")
    start = time.time()

    direct_system = "You are a precise text-transformation engine. Return only the requested output."
    code_system = "You are a precise programming assistant. Follow the requested output format exactly."
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    label_styles = [v.strip() for v in args.probe_label_styles.split(",") if v.strip()]

    task_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []

    for ti, task in enumerate(tasks, 1):
        train, heldout = split_examples(task, args.train_n, args.heldout_cap, args.min_heldout)
        print(f"[{ti}/{len(tasks)}] {task.task_id} ({task.family})", flush=True)

        direct_preds, direct_row, direct_full = direct_answers(tok, model, cache, task.task_id, train, heldout, args.answer_max_new_tokens)
        batch_preds, batch_row, batch_full, batch_parse_ok = batch_answers(tok, model, cache, task.task_id, train, heldout, args.batch_max_new_tokens)

        cands: List[ProgramCandidate] = []
        for variant in variants:
            _, raw = cache.get(
                tok,
                model,
                task.task_id,
                "program",
                variant,
                "0",
                code_system,
                code_prompt(train, variant),
                args.code_max_new_tokens,
                prediction_parser=lambda x: x,
            )
            cand = evaluate_candidate(raw, task.task_id, variant, train, heldout)
            cands.append(cand)
            current = cand
            for repair_round in range(1, args.repair_rounds + 1):
                if current.train_pass:
                    break
                failures = failing_train_cases(current, train)
                _, repair_raw = cache.get(
                    tok,
                    model,
                    task.task_id,
                    "program_repair",
                    f"{variant}_repair{repair_round}",
                    str(repair_round),
                    code_system,
                    repair_prompt(train, current.code, failures, variant),
                    args.code_max_new_tokens,
                    prediction_parser=lambda x: x,
                )
                current = evaluate_candidate(repair_raw, task.task_id, f"{variant}_repair{repair_round}", train, heldout)
                cands.append(current)
        train_pass = [c for c in cands if c.train_pass]

        for ci, cand in enumerate(cands):
            row = cand.__dict__.copy()
            row["candidate_index"] = ci
            candidate_rows.append(row)

        visible_cand = select_visible(cands)
        visible_sel = eval_program_selection("program_visible", visible_cand, heldout)
        oracle_sel = eval_program_selection("hidden_candidate_oracle", oracle_candidate(cands), heldout)

        selected_probes = select_probe_rows(train_pass, train, args.probe_count, task.task_id, "disagreement") if train_pass else []
        random_probes = select_probe_rows(train_pass, train, args.probe_count, task.task_id, "random") if train_pass else []

        def label_probe_set(rows: Sequence[Example], mode: str) -> List[Example]:
            labeled = []
            for pi, probe in enumerate(rows):
                label, consensus, all_labels = label_probe(
                    tok,
                    model,
                    cache,
                    task.task_id,
                    train,
                    probe,
                    label_styles,
                    args.answer_max_new_tokens,
                )
                disagreement, counts = distinct_outputs(train_pass, probe)
                probe_rows.append(
                    {
                        "task_id": task.task_id,
                        "family": task.family,
                        "probe_index": pi,
                        "selected_mode": mode,
                        "input": render_inputs(probe.inputs),
                        "label": label,
                        "consensus": consensus,
                        "all_labels_json": json.dumps(all_labels, ensure_ascii=False),
                        "candidate_disagreement": disagreement,
                        "candidate_outputs_json": json.dumps(counts, ensure_ascii=False),
                    }
                )
                if consensus >= args.min_probe_consensus and label:
                    labeled.append(Example(probe.inputs, label))
            return labeled

        labeled_probes = label_probe_set(selected_probes, "disagreement")
        random_labeled_probes = label_probe_set(random_probes, "random")
        shuffled_labeled = list(labeled_probes)
        if len(shuffled_labeled) > 1:
            labels = [ex.output for ex in shuffled_labeled]
            labels = labels[1:] + labels[:1]
            shuffled_labeled = [Example(ex.inputs, label) for ex, label in zip(shuffled_labeled, labels)]

        ceg_cand, ceg_score = select_by_probes(cands, labeled_probes)
        random_cand, random_score = select_by_probes(cands, random_labeled_probes)
        shuffled_cand, shuffled_score = select_by_probes(cands, shuffled_labeled)
        ceg_sel = eval_program_selection("program_ceg", ceg_cand, heldout, ceg_score)
        random_sel = eval_program_selection("program_random_probe", random_cand, heldout, random_score)
        shuffled_sel = eval_program_selection("program_shuffled_probe_labels", shuffled_cand, heldout, shuffled_score)

        use_gated = (
            ceg_cand is not None
            and not ceg_cand.suspicious
            and len(labeled_probes) >= args.min_confident_probes
            and ceg_score >= args.gate_probe_score
        )
        if use_gated:
            gated_preds = ceg_sel.predictions
            gated_row, gated_full = ceg_sel.row_exact, ceg_sel.full_exact
            gated_desc = ceg_sel.description
        else:
            gated_preds = direct_preds
            gated_row, gated_full = direct_row, direct_full
            gated_desc = "fallback_direct"

        router, router_score, router_desc = fit_router(cands, train, labeled_probes)
        router_preds = eval_router(router, heldout)
        router_row, router_full = score_outputs(router_preds, heldout)
        use_router = router is not None and router_score >= max(ceg_score, args.gate_probe_score)
        if not use_router:
            router_preds = gated_preds
            router_row, router_full = gated_row, gated_full
            router_desc = f"fallback:{gated_desc}"

        task_rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "features": ",".join(task.features),
                "synthetic": task.synthetic,
                "heldout_rows": len(heldout),
                "direct_row_exact": direct_row,
                "direct_full_exact": direct_full,
                "batch_row_exact": batch_row,
                "batch_full_exact": batch_full,
                "batch_parse_ok": batch_parse_ok,
                "visible_row_exact": visible_sel.row_exact,
                "visible_full_exact": visible_sel.full_exact,
                "visible_used_program": visible_sel.used_program,
                "ceg_row_exact": ceg_sel.row_exact,
                "ceg_full_exact": ceg_sel.full_exact,
                "ceg_used_program": ceg_sel.used_program,
                "ceg_gated_row_exact": gated_row,
                "ceg_gated_full_exact": gated_full,
                "ceg_gated_used_program": use_gated,
                "router_row_exact": router_row,
                "router_full_exact": router_full,
                "router_used": use_router,
                "random_probe_row_exact": random_sel.row_exact,
                "random_probe_full_exact": random_sel.full_exact,
                "random_probe_used_program": random_sel.used_program,
                "shuffled_probe_row_exact": shuffled_sel.row_exact,
                "shuffled_probe_full_exact": shuffled_sel.full_exact,
                "shuffled_probe_used_program": shuffled_sel.used_program,
                "oracle_row_exact": oracle_sel.row_exact,
                "oracle_full_exact": oracle_sel.full_exact,
                "oracle_used_program": oracle_sel.used_program,
                "train_pass_count": len(train_pass),
                "candidate_count": len(cands),
                "confident_probe_count": len(labeled_probes),
                "random_confident_probe_count": len(random_labeled_probes),
                "ceg_probe_score": ceg_score,
                "random_probe_score": random_score,
                "shuffled_probe_score": shuffled_score,
                "router_probe_score": router_score,
                "visible_description": visible_sel.description,
                "ceg_description": ceg_sel.description,
                "ceg_gated_description": gated_desc,
                "router_description": router_desc,
                "direct_predictions_json": json.dumps(direct_preds, ensure_ascii=False),
                "ceg_predictions_json": json.dumps(ceg_sel.predictions, ensure_ascii=False),
                "gated_predictions_json": json.dumps(gated_preds, ensure_ascii=False),
                "router_predictions_json": json.dumps(router_preds, ensure_ascii=False),
                "targets_json": json.dumps([ex.output for ex in heldout], ensure_ascii=False),
            }
        )

    cache.close()

    task_df = pd.DataFrame(task_rows)
    candidate_df = pd.DataFrame(
        candidate_rows,
        columns=[
            "task_id",
            "variant",
            "raw_text",
            "code",
            "status",
            "train_pass",
            "train_row_exact",
            "heldout_row_exact",
            "heldout_full_exact",
            "code_chars",
            "suspicious",
            "error",
            "candidate_index",
        ],
    )
    probe_df = pd.DataFrame(
        probe_rows,
        columns=[
            "task_id",
            "family",
            "probe_index",
            "selected_mode",
            "input",
            "label",
            "consensus",
            "all_labels_json",
            "candidate_disagreement",
            "candidate_outputs_json",
        ],
    )
    summary = summarize(task_df)
    fam = family_summary(task_df)

    task_df.to_csv(run_dir / "task_summary.csv", index=False)
    candidate_df.to_csv(run_dir / "candidate_programs.csv", index=False)
    probe_df.to_csv(run_dir / "probe_labels.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    fam.to_csv(run_dir / "family_summary.csv", index=False)
    for p in [run_dir / "task_summary.csv", run_dir / "candidate_programs.csv", run_dir / "probe_labels.csv", run_dir / "summary.csv", run_dir / "family_summary.csv"]:
        shutil.copy2(p, ANALYSIS / p.name)
    make_figures(task_df, summary, fam)
    elapsed = time.time() - start
    write_report(args, summary, task_df, fam, candidate_df, probe_df, elapsed)
    append_log(
        f"""
## Run `{args.run_name}`

- Time UTC: `{datetime.now(timezone.utc).isoformat()}`
- Elapsed seconds: `{elapsed:.1f}`
- Config: `{json.dumps(vars(args), sort_keys=True)}`
- Tasks: `{len(task_df)}`
- `direct_qwen_row`: `{pct(float(task_df['direct_full_exact'].mean()) if len(task_df) else 0.0)}` full-task exact.
- `program_visible`: `{pct(float(task_df['visible_full_exact'].mean()) if len(task_df) else 0.0)}` full-task exact.
- `program_ceg`: `{pct(float(task_df['ceg_full_exact'].mean()) if len(task_df) else 0.0)}` full-task exact.
- `program_ceg_gated`: `{pct(float(task_df['ceg_gated_full_exact'].mean()) if len(task_df) else 0.0)}` full-task exact.
- `program_ceg_router`: `{pct(float(task_df['router_full_exact'].mean()) if len(task_df) else 0.0)}` full-task exact.
- `program_shuffled_probe_labels`: `{pct(float(task_df['shuffled_probe_full_exact'].mean()) if len(task_df) else 0.0)}` full-task exact.
- Hidden candidate oracle: `{pct(float(task_df['oracle_full_exact'].mean()) if len(task_df) else 0.0)}` full-task exact.
"""
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="main_v1")
    p.add_argument("--task_limit", type=int, default=30)
    p.add_argument("--train_n", type=int, default=4)
    p.add_argument("--heldout_cap", type=int, default=4)
    p.add_argument("--min_heldout", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260628)
    p.add_argument("--variants", default="direct,helpers,regex,conditional")
    p.add_argument("--repair_rounds", type=int, default=1)
    p.add_argument("--probe_count", type=int, default=4)
    p.add_argument("--probe_label_styles", default="plain,rule,strict")
    p.add_argument("--min_probe_consensus", type=float, default=2 / 3)
    p.add_argument("--min_confident_probes", type=int, default=2)
    p.add_argument("--gate_probe_score", type=float, default=0.75)
    p.add_argument("--answer_max_new_tokens", type=int, default=64)
    p.add_argument("--batch_max_new_tokens", type=int, default=220)
    p.add_argument("--code_max_new_tokens", type=int, default=620)
    p.add_argument("--no_qwen", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
