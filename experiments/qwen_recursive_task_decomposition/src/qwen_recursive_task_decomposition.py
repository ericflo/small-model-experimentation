#!/usr/bin/env python3
"""Recursive task decomposition on public text transformations.

This standalone experiment tests whether committing to an explicit task
decomposition improves full-task consistency. It evaluates both executable
static decompositions and frozen-Qwen textual decomposition rules.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import math
import random
import re
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


ROOT = Path("/workspace/experiments/qwen_recursive_task_decomposition")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_recursive_task_decomposition")
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


@dataclass(frozen=True)
class Program:
    code: str
    func: Callable[[Tuple[str, ...]], Optional[str]]
    depth: int
    kind: str

    def eval_row(self, row: Tuple[str, ...]) -> Optional[str]:
        try:
            val = self.func(row)
            if val is None:
                return None
            return str(val)
        except Exception:
            return None

    def eval_many(self, rows: Sequence[Tuple[str, ...]]) -> Optional[Tuple[str, ...]]:
        out: List[str] = []
        for row in rows:
            val = self.eval_row(row)
            if val is None:
                return None
            out.append(val)
        return tuple(out)


def clean(s: Any) -> str:
    return re.sub(r"\s+", " ", "" if s is None else str(s)).strip()


def title_words(s: Any) -> str:
    return " ".join(w.capitalize() for w in re.findall(r"[A-Za-z0-9]+", clean(s)))


def only_digits(s: Any) -> str:
    return re.sub(r"\D+", "", "" if s is None else str(s))


def only_alpha_space(s: Any) -> str:
    return clean(re.sub(r"[^A-Za-z]+", " ", "" if s is None else str(s)))


def only_alnum(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", "" if s is None else str(s))


def initials(s: Any, sep: str = "") -> str:
    return sep.join(w[0].upper() for w in re.findall(r"[A-Za-z]+", str(s)))


def first_last_initials(s: Any, sep: str = " ") -> Optional[str]:
    words = re.findall(r"[A-Za-z]+", str(s))
    if len(words) < 2:
        return None
    return words[0][0].upper() + sep + words[-1][0].upper()


def parse_float(s: Any) -> Optional[float]:
    text = clean(s).replace("$", "").replace(",", "").replace("%", "")
    text = text.replace("₹", "").replace("€", "").replace("£", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        return None if not m else float(m.group(0))


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
MONTH_NAMES = {v: k.title() for k, v in MONTHS.items() if len(k) > 3}
MONTH_ABBR = {v: k.title() for k, v in MONTHS.items() if len(k) == 3}


def date_parts(s: Any) -> Optional[Tuple[int, int, int]]:
    text = clean(s)
    patterns = [
        (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), ("Y", "M", "D")),
        (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b"), ("M", "D", "Y")),
        (re.compile(r"\b(\d{1,2})-(\d{1,2})-(\d{2,4})\b"), ("M", "D", "Y")),
        (re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b"), ("D", "M", "Y")),
    ]
    for rgx, order in patterns:
        m = rgx.search(text)
        if not m:
            continue
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
    return None


def time_parts(s: Any) -> Optional[Tuple[int, int, int]]:
    m = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", str(s))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)


def fmt_int(v: Optional[float]) -> Optional[str]:
    if v is None:
        return None
    if abs(v - int(v)) < 1e-9:
        return str(int(v))
    return str(v)


def fmt_money(v: Optional[float]) -> Optional[str]:
    if v is None:
        return None
    return f"${v:,.2f}"


def nth_field(text: str, sep: str, idx: int) -> Optional[str]:
    parts = [p.strip() for p in text.split(sep)]
    if not parts:
        return None
    j = idx if idx >= 0 else len(parts) + idx
    if j < 0 or j >= len(parts):
        return None
    return parts[j]


def word_at(text: str, idx: int) -> Optional[str]:
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words:
        return None
    j = idx if idx >= 0 else len(words) + idx
    if j < 0 or j >= len(words):
        return None
    return words[j]


def regex_group(text: str, pattern: str, idx: int = 1) -> Optional[str]:
    m = re.search(pattern, text)
    if not m:
        return None
    return m.group(idx)


def char_slice(text: str, a: Optional[int], b: Optional[int]) -> Optional[str]:
    s = str(text)
    out = s[a:b]
    return out if out != "" else None


def render_inputs(vals: Sequence[str]) -> str:
    if len(vals) == 1:
        return vals[0]
    return " | ".join(f"col{i}={v}" for i, v in enumerate(vals))


def load_tasks(limit: Optional[int] = None, min_examples: int = 5) -> List[Task]:
    tasks: List[Task] = []
    for d in sorted(TRANSFORM_ROOT.iterdir()):
        if not d.is_dir() or not (d / "spec.json").exists():
            continue
        spec = json.loads((d / "spec.json").read_text())
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        examples: List[Example] = []
        for ex in spec.get("Examples", []):
            examples.append(Example(tuple(str(x) for x in ex.get("Input", [])), str(ex.get("Output", ""))))
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
        if limit and len(tasks) >= limit:
            break
    return tasks


def split_examples(task: Task, train_n: int, heldout_cap: int) -> Tuple[List[Example], List[Example]]:
    n = len(task.examples)
    k = min(train_n, max(2, n // 2))
    train = list(task.examples[:k])
    test = list(task.examples[k:])
    if heldout_cap > 0:
        test = test[:heldout_cap]
    return train, test


def source_programs(num_cols: int) -> List[Program]:
    exprs: List[Program] = []
    seps = [" ", ",", "-", "/", "\\", "_", ":", ";", "|", ".", "\n", "\t", "(", ")", "[", "]"]
    regexes = [
        ("first_digits", r"(\d+)", 1),
        ("last_digits", r".*(\d+)", 1),
        ("first_word", r"([A-Za-z]+)", 1),
        ("last_word", r".*\b([A-Za-z]+)\b", 1),
        ("email_user", r"([A-Za-z0-9._%+\-]+)@[A-Za-z0-9.\-]+", 1),
        ("email_domain", r"[A-Za-z0-9._%+\-]+@([A-Za-z0-9.\-]+)", 1),
        ("paren", r"\(([^()]*)\)", 1),
        ("bracket", r"\[([^\[\]]*)\]", 1),
        ("zip5", r"\b(\d{5})(?:-\d{4})?\b", 1),
        ("state_zip", r"\b([A-Z]{2})\s+\d{5}(?:-\d{4})?\b", 1),
        ("url_domain", r"^(?:[A-Za-z]+://)?([^/]+)", 1),
        ("file_ext", r"\.([A-Za-z0-9]+)$", 1),
        ("file_stem", r"([^/\\]+?)(?:\.[A-Za-z0-9]+)?$", 1),
    ]
    for i in range(num_cols):
        prefix = f"COL{i}"
        exprs.append(Program(prefix, lambda row, j=i: row[j] if j < len(row) else "", 0, "source"))
        exprs.extend(
            [
                Program(f"strip({prefix})", lambda row, j=i: clean(row[j]), 1, "surface"),
                Program(f"lower({prefix})", lambda row, j=i: clean(row[j]).lower(), 1, "surface"),
                Program(f"upper({prefix})", lambda row, j=i: clean(row[j]).upper(), 1, "surface"),
                Program(f"title({prefix})", lambda row, j=i: title_words(row[j]), 1, "surface"),
                Program(f"digits({prefix})", lambda row, j=i: only_digits(row[j]), 1, "surface"),
                Program(f"alpha({prefix})", lambda row, j=i: only_alpha_space(row[j]), 1, "surface"),
                Program(f"alnum({prefix})", lambda row, j=i: only_alnum(row[j]), 1, "surface"),
                Program(f"initials({prefix})", lambda row, j=i: initials(row[j]), 1, "surface"),
                Program(f"initials_sp({prefix})", lambda row, j=i: initials(row[j], " "), 1, "surface"),
                Program(f"first_last_initials({prefix})", lambda row, j=i: first_last_initials(row[j]), 1, "surface"),
            ]
        )
        for idx in [0, 1, 2, 3, 4, 5, -1, -2, -3]:
            exprs.append(Program(f"word[{idx}]({prefix})", lambda row, j=i, k=idx: word_at(row[j], k), 1, "substring"))
        for sep in seps:
            label = {" ": "space", "\n": "nl", "\t": "tab", "\\": "bslash"}.get(sep, sep)
            for idx in [0, 1, 2, 3, 4, 5, -1, -2, -3]:
                exprs.append(Program(f"field[{label},{idx}]({prefix})", lambda row, j=i, s=sep, k=idx: nth_field(row[j], s, k), 1, "substring"))
        for n in range(1, 13):
            exprs.append(Program(f"first_chars[{n}]({prefix})", lambda row, j=i, k=n: char_slice(row[j], 0, k), 1, "substring"))
            exprs.append(Program(f"last_chars[{n}]({prefix})", lambda row, j=i, k=n: char_slice(row[j], -k, None), 1, "substring"))
            exprs.append(Program(f"drop_first[{n}]({prefix})", lambda row, j=i, k=n: char_slice(row[j], k, None), 1, "substring"))
            exprs.append(Program(f"drop_last[{n}]({prefix})", lambda row, j=i, k=n: char_slice(row[j], 0, -k), 1, "substring"))
        for a in range(0, 8):
            for b in range(a + 1, min(14, a + 8)):
                exprs.append(Program(f"slice[{a}:{b}]({prefix})", lambda row, j=i, x=a, y=b: char_slice(row[j], x, y), 1, "substring"))
        for name, pattern, group in regexes:
            exprs.append(Program(f"{name}({prefix})", lambda row, j=i, p=pattern, g=group: regex_group(row[j], p, g), 1, "regex"))
        exprs.extend(
            [
                Program(f"date_year({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else str(date_parts(row[j])[0]), 1, "date"),
                Program(f"date_month({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else str(date_parts(row[j])[1]), 1, "date"),
                Program(f"date_month2({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else f"{date_parts(row[j])[1]:02d}", 1, "date"),
                Program(f"date_month_name({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else MONTH_NAMES.get(date_parts(row[j])[1]), 1, "date"),
                Program(f"date_month_abbr({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else MONTH_ABBR.get(date_parts(row[j])[1]), 1, "date"),
                Program(f"date_day({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else str(date_parts(row[j])[2]), 1, "date"),
                Program(f"date_day2({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else f"{date_parts(row[j])[2]:02d}", 1, "date"),
                Program(f"date_iso({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else f"{date_parts(row[j])[0]:04d}-{date_parts(row[j])[1]:02d}-{date_parts(row[j])[2]:02d}", 1, "date"),
                Program(f"time_hour({prefix})", lambda row, j=i: None if time_parts(row[j]) is None else str(time_parts(row[j])[0]), 1, "time"),
                Program(f"time_hour2({prefix})", lambda row, j=i: None if time_parts(row[j]) is None else f"{time_parts(row[j])[0]:02d}", 1, "time"),
                Program(f"time_minute({prefix})", lambda row, j=i: None if time_parts(row[j]) is None else str(time_parts(row[j])[1]), 1, "time"),
                Program(f"number_int({prefix})", lambda row, j=i: fmt_int(parse_float(row[j])), 1, "number"),
                Program(f"number_round10({prefix})", lambda row, j=i: None if parse_float(row[j]) is None else str(int(round(parse_float(row[j]) / 10) * 10)), 1, "number"),
                Program(f"number_round100({prefix})", lambda row, j=i: None if parse_float(row[j]) is None else str(int(round(parse_float(row[j]) / 100) * 100)), 1, "number"),
                Program(f"number_1dp({prefix})", lambda row, j=i: None if parse_float(row[j]) is None else f"{parse_float(row[j]):.1f}", 1, "number"),
                Program(f"number_2dp({prefix})", lambda row, j=i: None if parse_float(row[j]) is None else f"{parse_float(row[j]):.2f}", 1, "number"),
                Program(f"money_us({prefix})", lambda row, j=i: fmt_money(parse_float(row[j])), 1, "number"),
            ]
        )
    return exprs


def add_wrappers(exprs: List[Program]) -> List[Program]:
    transforms: List[Tuple[str, Callable[[str], Optional[str]]]] = [
        ("strip", lambda x: clean(x)),
        ("lower", lambda x: clean(x).lower()),
        ("upper", lambda x: clean(x).upper()),
        ("title", title_words),
        ("digits", only_digits),
        ("alnum", only_alnum),
        ("alpha", only_alpha_space),
    ]
    out = list(exprs)
    for e in exprs:
        if e.depth > 1:
            continue
        for name, fn in transforms:
            out.append(Program(f"{name}({e.code})", lambda row, ex=e, f=fn: None if ex.eval_row(row) is None else f(ex.eval_row(row)), e.depth + 1, f"wrap_{e.kind}"))
    return out


def add_affixes(exprs: List[Program], train_rows: Sequence[Tuple[str, ...]], train_y: Tuple[str, ...]) -> List[Program]:
    out: List[Program] = []
    for e in exprs:
        vals = e.eval_many(train_rows)
        if vals is None:
            continue
        pairs: List[Tuple[str, str]] = []
        ok = True
        for val, y in zip(vals, train_y):
            if val == "" or val not in y:
                ok = False
                break
            idx = y.find(val)
            pairs.append((y[:idx], y[idx + len(val) :]))
        if not ok:
            continue
        prefixes = {p for p, _ in pairs}
        suffixes = {s for _, s in pairs}
        if len(prefixes) == 1 and len(suffixes) == 1:
            pre = next(iter(prefixes))
            suf = next(iter(suffixes))
            if pre or suf:
                out.append(Program(f"affix[{pre!r},{suf!r}]({e.code})", lambda row, ex=e, p=pre, s=suf: None if ex.eval_row(row) is None else f"{p}{ex.eval_row(row)}{s}", e.depth + 1, "affix"))
    return out


def add_concat(exprs: List[Program], train_rows: Sequence[Tuple[str, ...]], max_source: int = 36) -> List[Program]:
    ranked = sorted(exprs, key=lambda e: (e.depth, len(e.code), e.code))
    usable: List[Program] = []
    for e in ranked:
        vals = e.eval_many(train_rows)
        if vals is None or any(v == "" for v in vals):
            continue
        if len(set(vals)) <= max(1, len(vals) // 3):
            continue
        usable.append(e)
        if len(usable) >= max_source:
            break
    seps = ["", " ", ", ", "-", "/", "_", ":", ".", " - ", " / "]
    out: List[Program] = []
    for a in usable:
        for b in usable:
            if a.code == b.code:
                continue
            for sep in seps:
                out.append(Program(f"concat[{sep!r}]({a.code},{b.code})", lambda row, x=a, y=b, s=sep: None if x.eval_row(row) is None or y.eval_row(row) is None else f"{x.eval_row(row)}{s}{y.eval_row(row)}", max(a.depth, b.depth) + 1, "concat"))
    return out


def add_maps(exprs: List[Program], train_rows: Sequence[Tuple[str, ...]], train_y: Tuple[str, ...]) -> List[Program]:
    out: List[Program] = []
    for e in exprs:
        vals = e.eval_many(train_rows)
        if vals is None:
            continue
        mapping: Dict[str, str] = {}
        ok = True
        for x, y in zip(vals, train_y):
            if x in mapping and mapping[x] != y:
                ok = False
                break
            mapping[x] = y
        if not ok or len(mapping) < 2 or len(mapping) > 8:
            continue
        short = ",".join(f"{k}->{v}" for k, v in list(mapping.items())[:4])
        out.append(Program(f"map[{short}]({e.code})", lambda row, ex=e, m=mapping: m.get(ex.eval_row(row), None), e.depth + 1, "map"))
    return out


def dedup_programs(programs: Iterable[Program], max_programs: int) -> List[Program]:
    seen: set[str] = set()
    out: List[Program] = []
    for p in sorted(programs, key=lambda x: (x.depth, len(x.code), x.code)):
        if p.code in seen:
            continue
        seen.add(p.code)
        out.append(p)
        if len(out) >= max_programs:
            break
    return out


def monolithic_candidates(num_cols: int, train_rows: Sequence[Tuple[str, ...]], train_y: Tuple[str, ...], max_candidates: int) -> List[Program]:
    base = add_wrappers(source_programs(num_cols))
    programs = list(base)
    programs += add_affixes(base, train_rows, train_y)
    programs += add_concat(base, train_rows)
    programs += add_maps(base, train_rows, train_y)
    return dedup_programs(programs, max_candidates)


def split_by_delimiter(outputs: Sequence[str], delimiter: str) -> Optional[List[List[str]]]:
    parts = [str(y).split(delimiter) for y in outputs]
    if not parts:
        return None
    n = len(parts[0])
    if n < 2 or n > 5:
        return None
    if any(len(p) != n or any(x == "" for x in p) for p in parts):
        return None
    return [[p[i] for p in parts] for i in range(n)]


def tokenize_template(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]+", s)


def template_splits(outputs: Sequence[str]) -> List[Tuple[str, List[List[str]], List[str]]]:
    toks = [tokenize_template(y) for y in outputs]
    if not toks or any(len(t) != len(toks[0]) for t in toks):
        return []
    n = len(toks[0])
    if n < 3 or n > 9:
        return []
    pieces: List[List[str]] = []
    template: List[str] = []
    variable_positions: List[int] = []
    for i in range(n):
        col = [t[i] for t in toks]
        if len(set(col)) == 1 and re.fullmatch(r"[^A-Za-z0-9]+", col[0]):
            template.append(col[0])
        else:
            template.append("{}")
            pieces.append(col)
            variable_positions.append(i)
    if len(pieces) < 2 or len(pieces) > 5:
        return []
    if not any(x != "{}" for x in template):
        return []
    label = "token_template[" + "".join(template).replace("\n", "\\n") + "]"
    return [(label, pieces, template)]


def recursive_candidates(
    num_cols: int,
    train_rows: Sequence[Tuple[str, ...]],
    train_y: Tuple[str, ...],
    max_candidates: int,
    max_depth: int,
    child_limit: int,
) -> List[Program]:
    mono = monolithic_candidates(num_cols, train_rows, train_y, max_candidates)
    if max_depth <= 0:
        return mono
    composite: List[Program] = []

    def child_programs(child_y: Sequence[str], depth_left: int) -> List[Program]:
        candidates = monolithic_candidates(num_cols, train_rows, tuple(child_y), max(4000, max_candidates // 4))
        fits = [p for p in candidates if p.eval_many(train_rows) == tuple(child_y)]
        if depth_left > 0:
            candidates2 = recursive_candidates(num_cols, train_rows, tuple(child_y), max(4000, max_candidates // 4), depth_left - 1, child_limit)
            fits += [p for p in candidates2 if p.eval_many(train_rows) == tuple(child_y)]
        return dedup_programs(fits, child_limit)

    split_specs: List[Tuple[str, List[List[str]], Optional[str], Optional[List[str]]]] = []
    for delim in [" ", ", ", ",", "-", "/", "_", ":", ".", " - ", " / ", "@", "#", "|"]:
        parts = split_by_delimiter(train_y, delim)
        if parts:
            split_specs.append((f"split[{delim!r}]", parts, delim, None))
    for label, pieces, template in template_splits(train_y):
        split_specs.append((label, pieces, None, template))

    for label, pieces, delim, template in split_specs:
        child_lists = [child_programs(piece, max_depth - 1) for piece in pieces]
        if any(not c for c in child_lists):
            continue
        for combo in itertools.product(*child_lists):
            if len(combo) > 4 and len(composite) > max_candidates // 2:
                break
            if delim is not None:
                code = f"{label}(" + ",".join(c.code for c in combo) + ")"

                def fn(row: Tuple[str, ...], children=combo, d=delim) -> Optional[str]:
                    vals = [c.eval_row(row) for c in children]
                    if any(v is None for v in vals):
                        return None
                    return d.join(str(v) for v in vals)

            else:
                code = f"{label}(" + ",".join(c.code for c in combo) + ")"

                def fn(row: Tuple[str, ...], children=combo, tmpl=template) -> Optional[str]:
                    vals = [c.eval_row(row) for c in children]
                    if any(v is None for v in vals):
                        return None
                    it = iter(str(v) for v in vals)
                    out: List[str] = []
                    for part in tmpl or []:
                        out.append(next(it) if part == "{}" else part)
                    return "".join(out)

            composite.append(Program(code, fn, max(c.depth for c in combo) + 1, "recursive_template"))
            if len(composite) >= max_candidates:
                break
        if len(composite) >= max_candidates:
            break
    return dedup_programs(list(mono) + composite, max_candidates)


def select_shortest(programs: Sequence[Program], train_rows: Sequence[Tuple[str, ...]], train_y: Tuple[str, ...]) -> Optional[Program]:
    fits = [p for p in programs if p.eval_many(train_rows) == train_y]
    return min(fits, key=lambda p: (p.depth, len(p.code), p.code), default=None)


def best_oracle(programs: Sequence[Program], train_rows: Sequence[Tuple[str, ...]], train_y: Tuple[str, ...], test_rows: Sequence[Tuple[str, ...]], test_y: Tuple[str, ...]) -> Optional[Program]:
    fits = [p for p in programs if p.eval_many(train_rows) == train_y and p.eval_many(test_rows) == test_y]
    return min(fits, key=lambda p: (p.depth, len(p.code), p.code), default=None)


def eval_full(program: Optional[Program], rows: Sequence[Tuple[str, ...]], y: Tuple[str, ...]) -> bool:
    return bool(program) and program.eval_many(rows) == y


def eval_row_accuracy(program: Optional[Program], rows: Sequence[Tuple[str, ...]], y: Tuple[str, ...]) -> float:
    if program is None or not rows:
        return 0.0
    ok = 0
    for row, target in zip(rows, y):
        ok += int(program.eval_row(row) == target)
    return ok / len(rows)


def evaluate_static_task(task: Task, cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_ex, test_ex = split_examples(task, cfg["train_n"], cfg["heldout_cap"])
    train_rows = [e.inputs for e in train_ex]
    test_rows = [e.inputs for e in test_ex]
    train_y = tuple(e.output for e in train_ex)
    test_y = tuple(e.output for e in test_ex)
    num_cols = max(len(e.inputs) for e in task.examples)

    mono = monolithic_candidates(num_cols, train_rows, train_y, cfg["max_candidates"])
    rec = recursive_candidates(num_cols, train_rows, train_y, cfg["max_candidates"], cfg["recursive_depth"], cfg["child_limit"])
    shifted_y = tuple(list(train_y)[1:] + list(train_y)[:1]) if len(train_y) > 1 else train_y
    rec_shuf = recursive_candidates(num_cols, train_rows, shifted_y, cfg["max_candidates"], cfg["recursive_depth"], cfg["child_limit"])

    mono_pick = select_shortest(mono, train_rows, train_y)
    mono_oracle = best_oracle(mono, train_rows, train_y, test_rows, test_y)
    rec_pick = select_shortest(rec, train_rows, train_y)
    rec_oracle = best_oracle(rec, train_rows, train_y, test_rows, test_y)
    rec_shuf_pick = select_shortest(rec_shuf, train_rows, shifted_y)

    return {
        "task_id": task.task_id,
        "family": task.family,
        "synthetic": task.synthetic,
        "features": ",".join(task.features),
        "num_examples": len(task.examples),
        "heldout_rows": len(test_rows),
        "mono_candidates": len(mono),
        "recursive_candidates": len(rec),
        "mono_train_match": mono_pick is not None,
        "recursive_train_match": rec_pick is not None,
        "mono_examples_full_exact": eval_full(mono_pick, test_rows, test_y),
        "recursive_examples_full_exact": eval_full(rec_pick, test_rows, test_y),
        "recursive_shuffled_full_exact": eval_full(rec_shuf_pick, test_rows, test_y),
        "mono_oracle_full_exact": mono_oracle is not None,
        "recursive_oracle_full_exact": rec_oracle is not None,
        "mono_examples_row_exact": eval_row_accuracy(mono_pick, test_rows, test_y),
        "recursive_examples_row_exact": eval_row_accuracy(rec_pick, test_rows, test_y),
        "recursive_shuffled_row_exact": eval_row_accuracy(rec_shuf_pick, test_rows, test_y),
        "mono_program": mono_pick.code if mono_pick else "",
        "recursive_program": rec_pick.code if rec_pick else "",
        "recursive_oracle_program": rec_oracle.code if rec_oracle else "",
        "recursive_kind": rec_pick.kind if rec_pick else "",
        "recursive_oracle_kind": rec_oracle.kind if rec_oracle else "",
    }


def run_static(tasks: List[Task], cfg: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks, start=1):
        if i == 1 or i % 25 == 0 or i == len(tasks):
            print(f"static {i}/{len(tasks)} {task.task_id}", flush=True)
        rows.append(evaluate_static_task(task, cfg))
    return pd.DataFrame(rows)


def clean_prediction(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    if text.lower().startswith("output:"):
        text = text.split(":", 1)[1].strip()
    first = text.splitlines()[0].strip() if text else ""
    if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
        first = first[1:-1].strip()
    return first


def load_qwen() -> Tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR), trust_remote_code=True, quantization_config=bnb, device_map="auto")
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
    out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def direct_prompt(train_pairs: Sequence[Tuple[Tuple[str, ...], str]], query: Tuple[str, ...]) -> str:
    lines = [
        "Infer the text transformation from the examples.",
        "Return only the transformed output for the query. Do not explain.",
        "",
        "Examples:",
    ]
    for inp, out in train_pairs:
        lines.append(f"Input: {render_inputs(inp)}")
        lines.append(f"Output: {out}")
    lines.append("")
    lines.append("Query:")
    lines.append(f"Input: {render_inputs(query)}")
    lines.append("Output:")
    return "\n".join(lines)


def rule_prompt(train_pairs: Sequence[Tuple[Tuple[str, ...], str]], style: str = "decompose") -> str:
    if style == "terse":
        opener = [
            "Infer the shortest precise transformation algorithm from these examples.",
            "Prefer explicit conditions and exact formatting. Do not include examples in the rule.",
            "Return only the algorithm.",
        ]
    elif style == "conditional":
        opener = [
            "Infer one reusable text transformation rule from these examples.",
            "Write it as ordered if/then steps, including exceptions such as empty inputs, null-like inputs, abbreviations, rounding, and formatting.",
            "Return only the rule.",
        ]
    else:
        opener = [
            "Infer one reusable text transformation rule from these examples.",
            "Write a concise recursive decomposition: name the substeps in order and describe how their outputs are combined.",
            "The rule must be reusable for new rows. Do not solve only the shown examples.",
            "Return only the rule, with no preamble.",
        ]
    lines = [
        *opener,
        "",
        "Examples:",
    ]
    for inp, out in train_pairs:
        lines.append(f"Input: {render_inputs(inp)}")
        lines.append(f"Output: {out}")
    return "\n".join(lines)


def locked_prompt(train_pairs: Sequence[Tuple[Tuple[str, ...], str]], rule: str, query: Tuple[str, ...]) -> str:
    lines = [
        "Use the fixed transformation rule below. Do not infer a new rule.",
        "",
        "Fixed rule:",
        rule.strip() or "(empty rule)",
        "",
        "Training examples for reference:",
    ]
    for inp, out in train_pairs:
        lines.append(f"Input: {render_inputs(inp)}")
        lines.append(f"Output: {out}")
    lines.extend(["", "Query:", f"Input: {render_inputs(query)}", "Output only:"])
    return "\n".join(lines)


def train_verify_rule(tok: Any, model: Any, train_pairs: Sequence[Tuple[Tuple[str, ...], str]], rule: str, max_new_tokens: int) -> Tuple[int, List[str]]:
    preds: List[str] = []
    ok = 0
    for inp, out in train_pairs:
        pred = clean_prediction(
            generate_text(
                tok,
                model,
                "You are a precise text transformation function that must follow a provided rule.",
                locked_prompt(train_pairs, rule, inp),
                max_new_tokens,
            )
        )
        preds.append(pred)
        ok += int(pred == out)
    return ok, preds


def choose_qwen_tasks(tasks: List[Task], qwen_task_limit: int, seed: int, min_heldout: int) -> List[Task]:
    eligible = [t for t in tasks if len(split_examples(t, 4, 9999)[1]) >= min_heldout]
    rng = random.Random(seed)
    if qwen_task_limit and qwen_task_limit < len(eligible):
        chosen = rng.sample(eligible, qwen_task_limit)
    else:
        chosen = eligible
    return sorted(chosen, key=lambda t: t.task_id)


def run_qwen(tasks: List[Task], cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selected = choose_qwen_tasks(tasks, cfg["qwen_task_limit"], cfg["sample_seed"], cfg["qwen_min_heldout"])
    tok, model = load_qwen()
    rule_rows: List[Dict[str, Any]] = []
    rules: Dict[str, str] = {}
    train_cache: Dict[str, List[Tuple[Tuple[str, ...], str]]] = {}
    split_cache: Dict[str, Tuple[List[Example], List[Example]]] = {}

    styles = [s.strip() for s in str(cfg.get("rule_styles", "decompose")).split(",") if s.strip()]
    for i, task in enumerate(selected, start=1):
        train_ex, test_ex = split_examples(task, cfg["train_n"], cfg["qwen_heldout_cap"])
        train_pairs = [(e.inputs, e.output) for e in train_ex]
        candidates: List[Dict[str, Any]] = []
        for style in styles:
            rule = generate_text(
                tok,
                model,
                "You write precise reusable data transformation rules.",
                rule_prompt(train_pairs, style),
                cfg["rule_max_new_tokens"],
            )
            train_ok = -1
            train_preds: List[str] = []
            if cfg.get("train_verify_rules"):
                train_ok, train_preds = train_verify_rule(tok, model, train_pairs, rule, cfg["answer_max_new_tokens"])
            candidates.append({"style": style, "rule": rule, "train_ok": train_ok, "train_preds": train_preds})
        if cfg.get("train_verify_rules"):
            best = max(candidates, key=lambda c: (c["train_ok"], -len(c["rule"]), c["style"]))
        else:
            best = candidates[0]
        rule = best["rule"]
        rules[task.task_id] = rule
        train_cache[task.task_id] = train_pairs
        split_cache[task.task_id] = (train_ex, test_ex)
        rule_rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "features": ",".join(task.features),
                "selected_style": best["style"],
                "rule_candidates": len(candidates),
                "train_verify_exact": best["train_ok"] / len(train_pairs) if best["train_ok"] >= 0 and train_pairs else None,
                "candidate_train_exact": json.dumps({c["style"]: (c["train_ok"] / len(train_pairs) if c["train_ok"] >= 0 and train_pairs else None) for c in candidates}, ensure_ascii=False),
                "rule": rule,
                "rule_chars": len(rule),
                "rule_lines": len([x for x in rule.splitlines() if x.strip()]),
            }
        )
        if i == 1 or i % 10 == 0 or i == len(selected):
            print(f"qwen rules {i}/{len(selected)}", flush=True)

    ids = [t.task_id for t in selected]
    shuffled_rule_by_id = {tid: rules[ids[(i + 1) % len(ids)]] for i, tid in enumerate(ids)} if ids else {}
    row_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []

    for i, task in enumerate(selected, start=1):
        train_ex, test_ex = split_cache[task.task_id]
        train_pairs = train_cache[task.task_id]
        rule = rules[task.task_id]
        shuffled_rule = shuffled_rule_by_id.get(task.task_id, "")
        counters = Counter()
        for ri, ex in enumerate(test_ex, start=1):
            direct = clean_prediction(generate_text(tok, model, "You are a precise text transformation function.", direct_prompt(train_pairs, ex.inputs), cfg["answer_max_new_tokens"]))
            locked = clean_prediction(generate_text(tok, model, "You are a precise text transformation function that must follow a provided rule.", locked_prompt(train_pairs, rule, ex.inputs), cfg["answer_max_new_tokens"]))
            shuffled = clean_prediction(generate_text(tok, model, "You are a precise text transformation function that must follow a provided rule.", locked_prompt(train_pairs, shuffled_rule, ex.inputs), cfg["answer_max_new_tokens"]))
            direct_ok = direct == ex.output
            locked_ok = locked == ex.output
            shuffled_ok = shuffled == ex.output
            counters["direct"] += int(direct_ok)
            counters["locked"] += int(locked_ok)
            counters["shuffled"] += int(shuffled_ok)
            row_rows.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "features": ",".join(task.features),
                    "row_index": ri,
                    "input": render_inputs(ex.inputs),
                    "target": ex.output,
                    "direct_prediction": direct,
                    "locked_prediction": locked,
                    "shuffled_prediction": shuffled,
                    "direct_exact": direct_ok,
                    "locked_exact": locked_ok,
                    "shuffled_exact": shuffled_ok,
                }
            )
        denom = len(test_ex)
        task_rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "features": ",".join(task.features),
                "heldout_rows": denom,
                "direct_row_exact": counters["direct"] / denom if denom else 0.0,
                "locked_row_exact": counters["locked"] / denom if denom else 0.0,
                "shuffled_row_exact": counters["shuffled"] / denom if denom else 0.0,
                "direct_full_exact": counters["direct"] == denom and denom > 0,
                "locked_full_exact": counters["locked"] == denom and denom > 0,
                "shuffled_full_exact": counters["shuffled"] == denom and denom > 0,
                "locked_minus_direct_rows": (counters["locked"] - counters["direct"]) / denom if denom else 0.0,
            }
        )
        if i == 1 or i % 5 == 0 or i == len(selected):
            d_full = sum(r["direct_full_exact"] for r in task_rows)
            l_full = sum(r["locked_full_exact"] for r in task_rows)
            s_full = sum(r["shuffled_full_exact"] for r in task_rows)
            print(f"qwen apply {i}/{len(selected)} full direct={d_full}/{len(task_rows)} locked={l_full}/{len(task_rows)} shuffled={s_full}/{len(task_rows)}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rule_rows), pd.DataFrame(task_rows).merge(pd.DataFrame(row_rows).groupby("task_id", as_index=False).size().rename(columns={"size": "row_records"}), on="task_id", how="left"), pd.DataFrame(row_rows)


def pct(x: Any) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{100 * float(x):.1f}%"


def md_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    view = df[list(cols)].head(max_rows).copy()
    for c in view.columns:
        if view[c].dtype.kind in "fc":
            if c in {"score", "rate", "row_exact", "full_exact"} or c.endswith("_exact") or c.endswith("_rate") or c.endswith("_coverage") or "row_exact" in c:
                view[c] = view[c].map(pct)
            else:
                view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    header = "|" + "|".join(cols) + "|"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for _, row in view.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            cells.append("" if pd.isna(val) else html.escape(str(val)))
        body.append("|" + "|".join(cells) + "|")
    return "\n".join([header, sep] + body)


def summarize_static(df: pd.DataFrame) -> pd.DataFrame:
    methods = [
        ("static_mono_examples", "mono_examples_full_exact", "mono_examples_row_exact"),
        ("static_mono_oracle", "mono_oracle_full_exact", None),
        ("static_recursive_examples", "recursive_examples_full_exact", "recursive_examples_row_exact"),
        ("static_recursive_oracle", "recursive_oracle_full_exact", None),
        ("static_recursive_shuffled", "recursive_shuffled_full_exact", "recursive_shuffled_row_exact"),
    ]
    rows = []
    for method, full_col, row_col in methods:
        rows.append(
            {
                "method": method,
                "tasks": len(df),
                "full_exact": float(df[full_col].mean()) if len(df) else 0.0,
                "row_exact": float(df[row_col].mean()) if row_col and len(df) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_qwen(task_df: pd.DataFrame) -> pd.DataFrame:
    if task_df.empty:
        return pd.DataFrame(columns=["method", "tasks", "full_exact", "row_exact"])
    rows = []
    for method, full_col, row_col in [
        ("direct_qwen", "direct_full_exact", "direct_row_exact"),
        ("locked_rule_qwen", "locked_full_exact", "locked_row_exact"),
        ("shuffled_rule_qwen", "shuffled_full_exact", "shuffled_row_exact"),
    ]:
        rows.append(
            {
                "method": method,
                "tasks": len(task_df),
                "full_exact": float(task_df[full_col].mean()),
                "row_exact": float(task_df[row_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_scores(static_summary: pd.DataFrame, qwen_summary: pd.DataFrame) -> None:
    combined = pd.concat([static_summary.assign(group="static"), qwen_summary.assign(group="qwen")], ignore_index=True)
    if combined.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = ["#6b7280" if g == "static" else "#2563eb" for g in combined["group"]]
    ax.bar(combined["method"], combined["full_exact"] * 100, color=colors)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Full-task exact (%)")
    ax.set_title("Recursive decomposition methods")
    ax.tick_params(axis="x", rotation=28)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "method_full_task_scores.png", dpi=160)
    plt.close(fig)


def plot_row_vs_full(qwen_summary: pd.DataFrame) -> None:
    if qwen_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = range(len(qwen_summary))
    ax.bar([i - 0.18 for i in x], qwen_summary["row_exact"] * 100, width=0.36, label="row exact", color="#10b981")
    ax.bar([i + 0.18 for i in x], qwen_summary["full_exact"] * 100, width=0.36, label="full-task exact", color="#f59e0b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(qwen_summary["method"], rotation=20)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact (%)")
    ax.set_title("Qwen row competence vs task-level consistency")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "qwen_row_vs_full.png", dpi=160)
    plt.close(fig)


def plot_static_gain(static_df: pd.DataFrame) -> None:
    if static_df.empty:
        return
    vals = pd.Series(
        {
            "mono examples": static_df["mono_examples_full_exact"].mean(),
            "recursive examples": static_df["recursive_examples_full_exact"].mean(),
            "mono oracle": static_df["mono_oracle_full_exact"].mean(),
            "recursive oracle": static_df["recursive_oracle_full_exact"].mean(),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(vals.index, vals.values * 100, color=["#64748b", "#2563eb", "#94a3b8", "#38bdf8"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Full-task exact (%)")
    ax.set_title("Executable recursive tree contribution")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "static_recursive_gain.png", dpi=160)
    plt.close(fig)


def plot_family_qwen(qwen_df: pd.DataFrame) -> None:
    if qwen_df.empty:
        return
    fam = qwen_df.groupby("family", as_index=False).agg(tasks=("task_id", "count"), direct=("direct_full_exact", "mean"), locked=("locked_full_exact", "mean"))
    fam = fam[fam["tasks"] >= 2].sort_values("locked")
    if fam.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(fam))))
    y = range(len(fam))
    ax.barh([i - 0.18 for i in y], fam["direct"] * 100, height=0.36, label="direct", color="#94a3b8")
    ax.barh([i + 0.18 for i in y], fam["locked"] * 100, height=0.36, label="locked rule", color="#2563eb")
    ax.set_yticks(list(y))
    ax.set_yticklabels(fam["family"])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Full-task exact (%)")
    ax.set_title("Qwen locked-rule effect by family")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "qwen_family_locked_effect.png", dpi=160)
    plt.close(fig)


def write_reports(static_df: pd.DataFrame, qwen_rules: pd.DataFrame, qwen_tasks: pd.DataFrame, qwen_rows: pd.DataFrame, cfg: Dict[str, Any], suite: str) -> None:
    static_summary = summarize_static(static_df)
    qwen_summary = summarize_qwen(qwen_tasks)
    plot_scores(static_summary, qwen_summary)
    plot_row_vs_full(qwen_summary)
    plot_static_gain(static_df)
    plot_family_qwen(qwen_tasks)

    family_static = (
        static_df.groupby("family", as_index=False)
        .agg(
            tasks=("task_id", "count"),
            mono_oracle=("mono_oracle_full_exact", "mean"),
            recursive_oracle=("recursive_oracle_full_exact", "mean"),
            mono_examples=("mono_examples_full_exact", "mean"),
            recursive_examples=("recursive_examples_full_exact", "mean"),
        )
        .sort_values(["recursive_oracle", "tasks"], ascending=[True, False])
    )

    lines: List[str] = []
    lines.append("# Recursive Task Decomposition on Public Text Transformations")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append("This standalone experiment tests whether recursive task decomposition improves task-level consistency on public text-transformation tasks. It compares executable recursive decomposition trees with a frozen language-model rule-locking procedure: first infer one reusable rule, then apply that same rule to every held-out row.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Dataset: public Microsoft PROSE `Transformation.Text` tasks.")
    lines.append(f"- Split: first `{cfg['train_n']}` examples are training examples; up to `{cfg['heldout_cap']}` held-out examples are scored for static methods.")
    lines.append("- Primary metric: full-task exact. A task is correct only if every held-out row is exact.")
    lines.append("- Static recursive decomposition: derive output templates from delimiters or token/literal structure, synthesize child transformations for each slot, and compose the children into a tree.")
    lines.append("- Qwen locked-rule decomposition: frozen Qwen writes one reusable rule/decomposition from the examples, then answers each held-out row while conditioned on that same rule.")
    lines.append("- Shuffled controls: static recursive synthesis on rotated labels, and Qwen application with a rule from another task.")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Suite: `{suite}`.")
    lines.append(f"- Static tasks: `{len(static_df)}`.")
    lines.append(f"- Qwen tasks: `{len(qwen_tasks)}`.")
    lines.append(f"- Qwen model: `{MODEL_NAME}`.")
    lines.append(f"- Static max candidates: `{cfg['max_candidates']}`.")
    lines.append(f"- Recursive depth: `{cfg['recursive_depth']}`; child limit: `{cfg['child_limit']}`.")
    lines.append(f"- Qwen held-out cap: `{cfg['qwen_heldout_cap']}` rows per task.")
    lines.append("")
    lines.append("## Primary Results")
    lines.append("")
    lines.append("### Static Executable Decomposition")
    lines.append("")
    lines.append(md_table(static_summary, ["method", "tasks", "full_exact", "row_exact"]))
    lines.append("")
    if not qwen_summary.empty:
        lines.append("### Frozen Qwen Decomposition")
        lines.append("")
        lines.append(md_table(qwen_summary, ["method", "tasks", "full_exact", "row_exact"]))
        lines.append("")
    lines.append("### Static Coverage By Family")
    lines.append("")
    lines.append(md_table(family_static, ["family", "tasks", "mono_oracle", "recursive_oracle", "mono_examples", "recursive_examples"], max_rows=40))
    lines.append("")
    if not qwen_tasks.empty:
        lines.append("### Qwen Task-Level Details")
        lines.append("")
        qcols = ["task_id", "family", "heldout_rows", "direct_row_exact", "locked_row_exact", "shuffled_row_exact", "direct_full_exact", "locked_full_exact", "shuffled_full_exact", "locked_minus_direct_rows"]
        lines.append(md_table(qwen_tasks.sort_values(["locked_full_exact", "locked_minus_direct_rows", "task_id"], ascending=[False, False, True]), qcols, max_rows=80))
        lines.append("")
        lines.append("### Example Locked Rules")
        lines.append("")
        rcols = ["task_id", "family", "rule_chars", "rule"]
        lines.append(md_table(qwen_rules.sort_values("task_id"), rcols, max_rows=16))
        lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fig in ["method_full_task_scores.png", "qwen_row_vs_full.png", "static_recursive_gain.png", "qwen_family_locked_effect.png"]:
        if (FIGURES / fig).exists():
            lines.append(f"![{fig}](../analysis/figures/{fig})")
            lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    static_map = dict(zip(static_summary["method"], static_summary["full_exact"]))
    lines.append(f"Executable recursion changes the static oracle from {pct(static_map.get('static_mono_oracle', 0.0))} to {pct(static_map.get('static_recursive_oracle', 0.0))}. The examples-selected recursive tree reaches {pct(static_map.get('static_recursive_examples', 0.0))}, compared with {pct(static_map.get('static_mono_examples', 0.0))} for monolithic expressions and {pct(static_map.get('static_recursive_shuffled', 0.0))} for the shuffled-label control.")
    if not qwen_summary.empty:
        qmap = {row["method"]: row for _, row in qwen_summary.iterrows()}
        direct = qmap.get("direct_qwen")
        locked = qmap.get("locked_rule_qwen")
        shuffled = qmap.get("shuffled_rule_qwen")
        if direct is not None and locked is not None and shuffled is not None:
            lines.append(f"For frozen Qwen, direct answering reaches {pct(direct['row_exact'])} row exact and {pct(direct['full_exact'])} full-task exact. The locked-rule decomposition reaches {pct(locked['row_exact'])} row exact and {pct(locked['full_exact'])} full-task exact. The shuffled-rule control reaches {pct(shuffled['row_exact'])} row exact and {pct(shuffled['full_exact'])} full-task exact.")
            delta = float(locked["full_exact"] - direct["full_exact"])
            lines.append(f"The locked-rule full-task delta over direct answering is {pct(delta)}. Positive values mean decomposition improved consistency; negative values mean rule commitment damaged useful row-level inference.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("The Qwen arm is capped for runtime and uses deterministic decoding. The static recursive tree only explores delimiter and token-template decompositions, not arbitrary semantic subgoals. Full-task exact is intentionally strict and can be much lower than row exact when a method is inconsistent across rows.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Static details: `analysis/static_details.csv`")
    lines.append("- Qwen rules: `analysis/qwen_rules.csv`")
    lines.append("- Qwen task summary: `analysis/qwen_task_details.csv`")
    lines.append("- Qwen row details: `analysis/qwen_row_details.csv`")
    lines.append("- Figures: `analysis/figures/`")

    REPORTS.mkdir(parents=True, exist_ok=True)
    md = "\n".join(lines) + "\n"
    (REPORTS / "qwen_recursive_task_decomposition_report.md").write_text(md)

    html_body = []
    for line in lines:
        if line.startswith("# "):
            html_body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html_body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            html_body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("![") and "](" in line:
            alt = line[2 : line.find("]")]
            src = line[line.find("(") + 1 : line.rfind(")")]
            html_body.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.startswith("|"):
            html_body.append(f"<pre>{html.escape(line)}</pre>")
        elif line.startswith("- "):
            html_body.append(f"<p>{html.escape(line)}</p>")
        elif line.strip():
            html_body.append(f"<p>{html.escape(line)}</p>")
    html_doc = """<!doctype html>
<html><head><meta charset="utf-8"><title>Recursive Task Decomposition</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:32px;line-height:1.5;color:#111827;max-width:1120px}
h1,h2,h3{line-height:1.2} img{max-width:100%;border:1px solid #e5e7eb;border-radius:6px} figure{margin:24px 0}
pre{background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;padding:8px;overflow:auto;margin:0}
p{max-width:920px}
</style></head><body>
""" + "\n".join(html_body) + "\n</body></html>\n"
    (REPORTS / "qwen_recursive_task_decomposition_report.html").write_text(html_doc)


def write_log(suite: str, cfg: Dict[str, Any], static_df: pd.DataFrame, qwen_tasks: pd.DataFrame) -> None:
    static_summary = summarize_static(static_df)
    qwen_summary = summarize_qwen(qwen_tasks)
    with (ROOT / "experiment_log.md").open("a") as f:
        f.write(f"\n## Run `{suite}`\n\n")
        f.write(f"- Time UTC: `{datetime.now(timezone.utc).isoformat()}`\n")
        f.write(f"- Static tasks: `{len(static_df)}`; Qwen tasks: `{len(qwen_tasks)}`\n")
        f.write(f"- Config: `{json.dumps(cfg, sort_keys=True)}`\n")
        f.write("- Static summary:\n")
        for _, row in static_summary.iterrows():
            f.write(f"  - `{row['method']}` full-task exact: {pct(row['full_exact'])}\n")
        if not qwen_summary.empty:
            f.write("- Qwen summary:\n")
            for _, row in qwen_summary.iterrows():
                f.write(f"  - `{row['method']}` row exact: {pct(row['row_exact'])}; full-task exact: {pct(row['full_exact'])}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="smoke")
    p.add_argument("--task_limit", type=int, default=40)
    p.add_argument("--train_n", type=int, default=4)
    p.add_argument("--heldout_cap", type=int, default=20)
    p.add_argument("--max_candidates", type=int, default=20000)
    p.add_argument("--recursive_depth", type=int, default=2)
    p.add_argument("--child_limit", type=int, default=6)
    p.add_argument("--run_qwen", action="store_true")
    p.add_argument("--qwen_task_limit", type=int, default=12)
    p.add_argument("--qwen_heldout_cap", type=int, default=6)
    p.add_argument("--qwen_min_heldout", type=int, default=3)
    p.add_argument("--sample_seed", type=int, default=20260627)
    p.add_argument("--rule_max_new_tokens", type=int, default=180)
    p.add_argument("--answer_max_new_tokens", type=int, default=64)
    p.add_argument("--rule_styles", default="decompose")
    p.add_argument("--train_verify_rules", action="store_true")
    p.add_argument("--reuse_static_from", default="")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    started = time.time()
    RUNS.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    LARGE_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    tasks = load_tasks(limit=args.task_limit if args.task_limit > 0 else None, min_examples=5)
    run_dir = RUNS / args.suite
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(), **cfg}, indent=2) + "\n")

    if args.reuse_static_from:
        static_path = RUNS / args.reuse_static_from / "static_details.csv"
        static_df = pd.read_csv(static_path)
        print(f"reused static details from {static_path}", flush=True)
    else:
        static_df = run_static(tasks, cfg)
    static_df.to_csv(run_dir / "static_details.csv", index=False)
    static_df.to_csv(ANALYSIS / "static_details.csv", index=False)

    qwen_rules = pd.DataFrame()
    qwen_tasks = pd.DataFrame()
    qwen_rows = pd.DataFrame()
    if args.run_qwen:
        qwen_rules, qwen_tasks, qwen_rows = run_qwen(tasks, cfg)
        qwen_rules.to_csv(run_dir / "qwen_rules.csv", index=False)
        qwen_tasks.to_csv(run_dir / "qwen_task_details.csv", index=False)
        qwen_rows.to_csv(run_dir / "qwen_row_details.csv", index=False)
        qwen_rules.to_csv(ANALYSIS / "qwen_rules.csv", index=False)
        qwen_tasks.to_csv(ANALYSIS / "qwen_task_details.csv", index=False)
        qwen_rows.to_csv(ANALYSIS / "qwen_row_details.csv", index=False)
    else:
        pd.DataFrame().to_csv(ANALYSIS / "qwen_rules.csv", index=False)
        pd.DataFrame().to_csv(ANALYSIS / "qwen_task_details.csv", index=False)
        pd.DataFrame().to_csv(ANALYSIS / "qwen_row_details.csv", index=False)

    summarize_static(static_df).to_csv(run_dir / "static_summary.csv", index=False)
    summarize_static(static_df).to_csv(ANALYSIS / "static_summary.csv", index=False)
    summarize_qwen(qwen_tasks).to_csv(run_dir / "qwen_summary.csv", index=False)
    summarize_qwen(qwen_tasks).to_csv(ANALYSIS / "qwen_summary.csv", index=False)

    write_reports(static_df, qwen_rules, qwen_tasks, qwen_rows, cfg, args.suite)
    write_log(args.suite, cfg, static_df, qwen_tasks)
    elapsed = time.time() - started
    (run_dir / "done.json").write_text(json.dumps({"elapsed_sec": round(elapsed, 2), "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    print(f"finished suite={args.suite} elapsed={elapsed:.1f}s")
    print((REPORTS / "qwen_recursive_task_decomposition_report.md").as_posix())


if __name__ == "__main__":
    run(parse_args())
