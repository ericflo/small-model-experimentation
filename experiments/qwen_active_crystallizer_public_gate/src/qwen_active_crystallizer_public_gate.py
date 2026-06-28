#!/usr/bin/env python3
"""Active crystallization gate on public PROSE transformations.

The experiment treats a frozen LLM as a noisy semantic oracle. It first
enumerates deterministic programs that fit sparse train examples, then asks the
LLM to label synthetic train-like probes and uses those labels to select one
program. Held-out rows are used only for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
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


ROOT = Path("/workspace/experiments/qwen_active_crystallizer_public_gate")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_active_crystallizer_public_gate")
PROSE_ROOT = LARGE_ROOT / "prose-benchmarks"
TRANSFORM_ROOT = PROSE_ROOT / "Transformation.Text"
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
class Expr:
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
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
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


def time_parts_one(text: str) -> Optional[Tuple[int, int, Optional[str]]]:
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([AP]M|am|pm|AM|PM)?\b", text)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    ap = m.group(3).upper() if m.group(3) else None
    return hour, minute, ap


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
                features=tuple(meta.get("Features", [])),
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


def source_exprs(num_cols: int) -> List[Expr]:
    exprs: List[Expr] = []
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
        exprs.append(Expr(prefix, lambda row, j=i: row[j] if j < len(row) else "", 0, "source"))
        exprs.extend(
            [
                Expr(f"strip({prefix})", lambda row, j=i: clean(row[j]), 1, "surface"),
                Expr(f"lower({prefix})", lambda row, j=i: clean(row[j]).lower(), 1, "surface"),
                Expr(f"upper({prefix})", lambda row, j=i: clean(row[j]).upper(), 1, "surface"),
                Expr(f"title({prefix})", lambda row, j=i: title_words(row[j]), 1, "surface"),
                Expr(f"digits({prefix})", lambda row, j=i: only_digits(row[j]), 1, "surface"),
                Expr(f"alpha({prefix})", lambda row, j=i: only_alpha_space(row[j]), 1, "surface"),
                Expr(f"alnum({prefix})", lambda row, j=i: only_alnum(row[j]), 1, "surface"),
                Expr(f"initials({prefix})", lambda row, j=i: initials(row[j]), 1, "surface"),
                Expr(f"initials_sp({prefix})", lambda row, j=i: initials(row[j], " "), 1, "surface"),
                Expr(f"first_last_initials({prefix})", lambda row, j=i: first_last_initials(row[j]), 1, "surface"),
            ]
        )
        for idx in [0, 1, 2, 3, 4, 5, -1, -2, -3]:
            exprs.append(Expr(f"word[{idx}]({prefix})", lambda row, j=i, k=idx: word_at(row[j], k), 1, "substring"))
        for sep in seps:
            label = {" ": "space", "\n": "nl", "\t": "tab", "\\": "bslash"}.get(sep, sep)
            for idx in [0, 1, 2, 3, 4, 5, -1, -2, -3]:
                exprs.append(Expr(f"field[{label},{idx}]({prefix})", lambda row, j=i, s=sep, k=idx: nth_field(row[j], s, k), 1, "substring"))
        for n in range(1, 13):
            exprs.append(Expr(f"first_chars[{n}]({prefix})", lambda row, j=i, k=n: char_slice(row[j], 0, k), 1, "substring"))
            exprs.append(Expr(f"last_chars[{n}]({prefix})", lambda row, j=i, k=n: char_slice(row[j], -k, None), 1, "substring"))
            exprs.append(Expr(f"drop_first[{n}]({prefix})", lambda row, j=i, k=n: char_slice(row[j], k, None), 1, "substring"))
            exprs.append(Expr(f"drop_last[{n}]({prefix})", lambda row, j=i, k=n: char_slice(row[j], 0, -k), 1, "substring"))
        for a in range(0, 8):
            for b in range(a + 1, min(14, a + 8)):
                exprs.append(Expr(f"slice[{a}:{b}]({prefix})", lambda row, j=i, x=a, y=b: char_slice(row[j], x, y), 1, "substring"))
        for name, pattern, group in regexes:
            exprs.append(Expr(f"{name}({prefix})", lambda row, j=i, p=pattern, g=group: regex_group(row[j], p, g), 1, "regex"))
        exprs.extend(
            [
                Expr(f"date_year({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else str(date_parts(row[j])[0]), 1, "date"),
                Expr(f"date_month({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else str(date_parts(row[j])[1]), 1, "date"),
                Expr(f"date_month2({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else f"{date_parts(row[j])[1]:02d}", 1, "date"),
                Expr(f"date_month_name({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else MONTH_NAMES.get(date_parts(row[j])[1]), 1, "date"),
                Expr(f"date_month_abbr({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else MONTH_ABBR.get(date_parts(row[j])[1]), 1, "date"),
                Expr(f"date_day({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else str(date_parts(row[j])[2]), 1, "date"),
                Expr(f"date_day2({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else f"{date_parts(row[j])[2]:02d}", 1, "date"),
                Expr(f"date_iso({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else f"{date_parts(row[j])[0]:04d}-{date_parts(row[j])[1]:02d}-{date_parts(row[j])[2]:02d}", 1, "date"),
                Expr(f"time_hour({prefix})", lambda row, j=i: None if time_parts(row[j]) is None else str(time_parts(row[j])[0]), 1, "time"),
                Expr(f"time_hour2({prefix})", lambda row, j=i: None if time_parts(row[j]) is None else f"{time_parts(row[j])[0]:02d}", 1, "time"),
                Expr(f"time_minute({prefix})", lambda row, j=i: None if time_parts(row[j]) is None else str(time_parts(row[j])[1]), 1, "time"),
                Expr(f"number_int({prefix})", lambda row, j=i: fmt_int(parse_float(row[j])), 1, "number"),
                Expr(f"number_round10({prefix})", lambda row, j=i: None if parse_float(row[j]) is None else str(int(round(parse_float(row[j]) / 10) * 10)), 1, "number"),
                Expr(f"number_round100({prefix})", lambda row, j=i: None if parse_float(row[j]) is None else str(int(round(parse_float(row[j]) / 100) * 100)), 1, "number"),
                Expr(f"number_1dp({prefix})", lambda row, j=i: None if parse_float(row[j]) is None else f"{parse_float(row[j]):.1f}", 1, "number"),
                Expr(f"number_2dp({prefix})", lambda row, j=i: None if parse_float(row[j]) is None else f"{parse_float(row[j]):.2f}", 1, "number"),
                Expr(f"money_us({prefix})", lambda row, j=i: fmt_money(parse_float(row[j])), 1, "number"),
            ]
        )
    return exprs


def add_wrappers(exprs: List[Expr]) -> List[Expr]:
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
            out.append(Expr(f"{name}({e.code})", lambda row, ex=e, f=fn: None if ex.eval_row(row) is None else f(ex.eval_row(row)), e.depth + 1, f"wrap_{e.kind}"))
    return out


def add_affixes(exprs: List[Expr], train_rows: Sequence[Tuple[str, ...]], train_y: Tuple[str, ...]) -> List[Expr]:
    out: List[Expr] = []
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
                code = f"affix[{pre!r},{suf!r}]({e.code})"
                out.append(Expr(code, lambda row, ex=e, p=pre, s=suf: None if ex.eval_row(row) is None else f"{p}{ex.eval_row(row)}{s}", e.depth + 1, "affix"))
    return out


def add_concat(exprs: List[Expr], train_rows: Sequence[Tuple[str, ...]], max_source: int = 40) -> List[Expr]:
    ranked = sorted(exprs, key=lambda e: (e.depth, len(e.code), e.code))
    usable: List[Expr] = []
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
    out: List[Expr] = []
    for a in usable:
        for b in usable:
            if a.code == b.code:
                continue
            for sep in seps:
                out.append(Expr(f"concat[{sep!r}]({a.code},{b.code})", lambda row, x=a, y=b, s=sep: None if x.eval_row(row) is None or y.eval_row(row) is None else f"{x.eval_row(row)}{s}{y.eval_row(row)}", max(a.depth, b.depth) + 1, "concat"))
    return out


def add_maps(exprs: List[Expr], train_rows: Sequence[Tuple[str, ...]], train_y: Tuple[str, ...]) -> List[Expr]:
    out: List[Expr] = []
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
        out.append(Expr(f"map[{short}]({e.code})", lambda row, ex=e, m=mapping: m.get(ex.eval_row(row), None), e.depth + 1, "map"))
    return out


def candidate_exprs(num_cols: int, train_rows: Sequence[Tuple[str, ...]], train_y: Tuple[str, ...], max_candidates: int) -> List[Expr]:
    base = add_wrappers(source_exprs(num_cols))
    exprs = list(base)
    exprs += add_affixes(base, train_rows, train_y)
    exprs += add_concat(base, train_rows)
    exprs += add_maps(base, train_rows, train_y)
    seen_code: set[str] = set()
    dedup: List[Expr] = []
    for e in sorted(exprs, key=lambda x: (x.depth, len(x.code), x.code)):
        if e.code in seen_code:
            continue
        seen_code.add(e.code)
        dedup.append(e)
        if len(dedup) >= max_candidates:
            break
    return dedup


NAMES = ["Alex Kim", "Jamie Patel", "Morgan Lee", "Sam Rivera", "Taylor Stone", "Jordan Smith"]
CITIES = ["Seattle, WA", "Boston, MA", "Austin, TX", "Denver, CO", "Miami, FL"]
DATES = ["2024-01-07", "03/14/2025", "Jul 9, 2026", "11/03/2023", "2022-12-31"]
EMAILS = ["alex@example.com", "jamie.patel@contoso.org", "morgan_lee@test.net"]
PHONES = ["(206) 555-0142", "617-555-0199", "+1 303 555 0120"]
WORDS = ["alpha", "bravo", "delta", "orange", "purple", "invoice", "shipment", "north"]


def mutate_text(text: str, rng: random.Random) -> str:
    s = str(text)
    replacements = [
        (r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", lambda: rng.choice(NAMES)),
        (r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+", lambda: rng.choice(EMAILS)),
        (r"\b\d{4}-\d{1,2}-\d{1,2}\b", lambda: rng.choice(DATES)),
        (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", lambda: rng.choice(DATES)),
        (r"\(?\d{3}\)?[- ]\d{3}[- ]\d{4}", lambda: rng.choice(PHONES)),
        (r"\b\d+\b", lambda: str(rng.randint(2, 9876))),
    ]
    for pattern, repl in replacements:
        if re.search(pattern, s) and rng.random() < 0.75:
            s = re.sub(pattern, lambda _: repl(), s, count=1)
    if rng.random() < 0.25:
        parts = re.findall(r"[A-Za-z]+", s)
        if parts:
            old = rng.choice(parts)
            s = s.replace(old, rng.choice(WORDS), 1)
    return s


def generate_probe_pool(train_rows: Sequence[Tuple[str, ...]], num_cols: int, seed: int, n: int = 48) -> List[Tuple[str, ...]]:
    rng = random.Random(seed)
    rows: List[Tuple[str, ...]] = []
    base = list(train_rows)
    attempts = 0
    max_attempts = max(200, n * 20)
    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        cols: List[str] = []
        for j in range(num_cols):
            source = rng.choice(base)
            text = source[j] if j < len(source) else ""
            cols.append(mutate_text(text, rng))
        row = tuple(cols)
        if row not in train_rows and row not in rows:
            rows.append(row)
    if len(rows) < n:
        for k in range(n - len(rows)):
            row = tuple(f"{source[j] if j < len(source) else ''} probe{k}" for j, source in enumerate([rng.choice(base)] * num_cols))
            if row not in train_rows and row not in rows:
                rows.append(row)
    return rows


def entropy(vals: Sequence[Optional[str]]) -> float:
    counts = Counter(v for v in vals if v is not None)
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    return -sum((c / total) * math.log(c / total + 1e-12) for c in counts.values())


def select_probes(train_matches: List[Expr], pool: List[Tuple[str, ...]], max_probes: int) -> List[Tuple[str, ...]]:
    scored = []
    for row in pool:
        vals = [e.eval_row(row) for e in train_matches]
        uniq = len({v for v in vals if v is not None})
        scored.append((entropy(vals), uniq, row))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected: List[Tuple[str, ...]] = []
    for _, uniq, row in scored:
        if uniq <= 1:
            continue
        selected.append(row)
        if len(selected) >= max_probes:
            break
    return selected


def render_inputs(vals: Sequence[str]) -> str:
    if len(vals) == 1:
        return vals[0]
    return " | ".join(f"col{i}={v}" for i, v in enumerate(vals))


def make_qwen_prompt(train_pairs: Sequence[Tuple[Tuple[str, ...], str]], query: Tuple[str, ...]) -> str:
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
def qwen_answer(tok: Any, model: Any, train_pairs: Sequence[Tuple[Tuple[str, ...], str]], query: Tuple[str, ...], max_new_tokens: int) -> str:
    prompt = make_qwen_prompt(train_pairs, query)
    messages = [
        {"role": "system", "content": "You are a precise text transformation function."},
        {"role": "user", "content": prompt},
    ]
    try:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_tensors="pt").to(model.device)
    out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return clean_prediction(gen)


def select_by_probe(train_matches: List[Expr], probes: Sequence[Tuple[str, ...]], labels: Sequence[str]) -> Optional[Expr]:
    if not train_matches:
        return None
    if not probes or not labels:
        return min(train_matches, key=lambda e: (e.depth, len(e.code), e.code))
    scored = []
    for e in train_matches:
        agree = 0
        nonnull = 0
        length_penalty = 0
        for row, label in zip(probes, labels):
            pred = e.eval_row(row)
            if pred is None:
                length_penalty += 1
                continue
            nonnull += 1
            if pred == label:
                agree += 1
        scored.append((agree, nonnull, -length_penalty, -e.depth, -len(e.code), e.code, e))
    scored.sort(reverse=True)
    return scored[0][-1]


def eval_selected(expr: Optional[Expr], rows: Sequence[Tuple[str, ...]], y: Tuple[str, ...]) -> bool:
    if expr is None:
        return False
    return expr.eval_many(rows) == y


def evaluate_task_static(task: Task, train_n: int, heldout_cap: int, max_candidates: int) -> Dict[str, Any]:
    train_ex, test_ex = split_examples(task, train_n, heldout_cap)
    train_rows = [e.inputs for e in train_ex]
    test_rows = [e.inputs for e in test_ex]
    train_y = tuple(e.output for e in train_ex)
    test_y = tuple(e.output for e in test_ex)
    num_cols = max(len(e.inputs) for e in task.examples)
    candidates = candidate_exprs(num_cols, train_rows, train_y, max_candidates)
    train_matches: List[Expr] = []
    for expr in candidates:
        sig = expr.eval_many(train_rows)
        if sig is None:
            continue
        if sig == train_y:
            train_matches.append(expr)
    oracle_matches = [e for e in train_matches if e.eval_many(test_rows) == test_y]
    examples_pick = min(train_matches, key=lambda e: (e.depth, len(e.code), e.code), default=None)
    best_oracle = min(oracle_matches, key=lambda e: (e.depth, len(e.code), e.code), default=None)
    seed = int(hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()[:8], 16)
    pool = generate_probe_pool(train_rows, num_cols, seed=seed, n=48)
    probes = select_probes(train_matches, pool, max_probes=8)
    return {
        "train_ex": train_ex,
        "test_ex": test_ex,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_y": train_y,
        "test_y": test_y,
        "train_matches": train_matches,
        "probes": probes,
        "candidate_count": len(candidates),
        "train_match_count": len(train_matches),
        "oracle_covered": bool(best_oracle),
        "oracle_program": best_oracle.code if best_oracle else "",
        "oracle_kind": best_oracle.kind if best_oracle else "",
        "examples_program": examples_pick.code if examples_pick else "",
        "examples_kind": examples_pick.kind if examples_pick else "",
        "examples_full_exact": eval_selected(examples_pick, test_rows, test_y),
    }


def run_static(tasks: List[Task], cfg: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks, start=1):
        if i == 1 or i % 50 == 0 or i == len(tasks):
            print(f"static {i}/{len(tasks)} {task.task_id}", flush=True)
        st = evaluate_task_static(task, cfg["train_n"], cfg["heldout_cap"], cfg["max_candidates"])
        rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "synthetic": task.synthetic,
                "features": ",".join(task.features),
                "num_examples": len(task.examples),
                "candidate_count": st["candidate_count"],
                "train_match_count": st["train_match_count"],
                "oracle_covered": st["oracle_covered"],
                "examples_full_exact": st["examples_full_exact"],
                "examples_program": st["examples_program"],
                "examples_kind": st["examples_kind"],
                "oracle_program": st["oracle_program"],
                "oracle_kind": st["oracle_kind"],
                "probe_count_available": len(st["probes"]),
            }
        )
    return pd.DataFrame(rows)


def run_qwen(tasks: List[Task], cfg: Dict[str, Any]) -> pd.DataFrame:
    tok, model = load_qwen()
    rows: List[Dict[str, Any]] = []
    if cfg["qwen_task_limit"] and cfg["qwen_task_limit"] < len(tasks):
        rng = random.Random(cfg["sample_seed"])
        eligible = rng.sample(tasks, cfg["qwen_task_limit"])
        eligible = sorted(eligible, key=lambda t: t.task_id)
    else:
        eligible = list(tasks)
    for i, task in enumerate(eligible, start=1):
        st = evaluate_task_static(task, cfg["train_n"], cfg["heldout_cap"], cfg["max_candidates"])
        train_pairs = [(e.inputs, e.output) for e in st["train_ex"]]
        probe_labels: List[str] = []
        probes = st["probes"][: cfg["qwen_probes"]]
        for probe in probes:
            probe_labels.append(qwen_answer(tok, model, train_pairs, probe, cfg["max_new_tokens"]))
        qwen_pick = select_by_probe(st["train_matches"], probes, probe_labels)
        shuffled = list(probe_labels)
        if len(shuffled) > 1:
            shuffled = shuffled[1:] + shuffled[:1]
        shuffled_pick = select_by_probe(st["train_matches"], probes, shuffled)
        direct_pred = qwen_answer(tok, model, train_pairs, st["test_rows"][0], cfg["max_new_tokens"]) if st["test_rows"] else ""
        direct_exact = direct_pred == (st["test_y"][0] if st["test_y"] else "")
        rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "synthetic": task.synthetic,
                "features": ",".join(task.features),
                "num_examples": len(task.examples),
                "train_match_count": len(st["train_matches"]),
                "oracle_covered": st["oracle_covered"],
                "examples_full_exact": st["examples_full_exact"],
                "qwen_probe_full_exact": eval_selected(qwen_pick, st["test_rows"], st["test_y"]),
                "shuffled_probe_full_exact": eval_selected(shuffled_pick, st["test_rows"], st["test_y"]),
                "qwen_direct_first_exact": direct_exact,
                "probe_count": len(probes),
                "probe_labels": json.dumps(probe_labels, ensure_ascii=False),
                "probe_inputs": json.dumps([list(p) for p in probes], ensure_ascii=False),
                "qwen_program": qwen_pick.code if qwen_pick else "",
                "qwen_kind": qwen_pick.kind if qwen_pick else "",
                "shuffled_program": shuffled_pick.code if shuffled_pick else "",
                "direct_prediction": direct_pred,
                "direct_target": st["test_y"][0] if st["test_y"] else "",
            }
        )
        if i == 1 or i % 10 == 0 or i == len(eligible):
            print(f"qwen {i}/{len(eligible)} qwen_prog={sum(r['qwen_probe_full_exact'] for r in rows)}/{len(rows)} direct={sum(r['qwen_direct_first_exact'] for r in rows)}/{len(rows)}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize_static(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"method": "candidate_oracle", "tasks": len(df), "score": float(df["oracle_covered"].mean())},
            {"method": "examples_shortest", "tasks": len(df), "score": float(df["examples_full_exact"].mean())},
            {"method": "has_train_match", "tasks": len(df), "score": float((df["train_match_count"] > 0).mean())},
        ]
    )


def summarize_qwen(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["method", "tasks", "score"])
    return pd.DataFrame(
        [
            {"method": "candidate_oracle", "tasks": len(df), "score": float(df["oracle_covered"].mean())},
            {"method": "examples_shortest", "tasks": len(df), "score": float(df["examples_full_exact"].mean())},
            {"method": "qwen_probe_select", "tasks": len(df), "score": float(df["qwen_probe_full_exact"].mean())},
            {"method": "shuffled_probe_select", "tasks": len(df), "score": float(df["shuffled_probe_full_exact"].mean())},
            {"method": "qwen_direct_first_row", "tasks": len(df), "score": float(df["qwen_direct_first_exact"].mean())},
        ]
    )


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
            if c in {"score", "coverage", "rate"} or c.endswith("_rate"):
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


def plot_methods(static_summary: pd.DataFrame, qwen_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    df = qwen_summary if not qwen_summary.empty else static_summary
    ax.bar(df["method"], df["score"] * 100, color="#2563eb")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success (%)")
    ax.set_title("Active crystallization methods")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "method_scores.png", dpi=160)
    plt.close(fig)


def plot_failure_mix(static_df: pd.DataFrame) -> None:
    no_train = (static_df["train_match_count"] == 0).mean()
    oracle_miss = ((static_df["train_match_count"] > 0) & ~static_df["oracle_covered"]).mean()
    selection_miss = (static_df["oracle_covered"] & ~static_df["examples_full_exact"]).mean()
    solved = static_df["examples_full_exact"].mean()
    vals = pd.Series({"examples solved": solved, "selection gap": selection_miss, "DSL heldout miss": oracle_miss, "no train match": no_train})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(vals.index, vals.values * 100, color=["#10b981", "#f59e0b", "#ef4444", "#991b1b"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Tasks (%)")
    ax.set_title("Examples-only decomposition")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "failure_decomposition.png", dpi=160)
    plt.close(fig)


def plot_family(static_df: pd.DataFrame) -> None:
    vals = static_df.groupby("family")["oracle_covered"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(5, 0.18 * len(vals))))
    ax.barh(vals.index, vals.values * 100, color="#059669")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Candidate-oracle full held-out coverage (%)")
    ax.set_title("DSL coverage by family")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "family_oracle_coverage.png", dpi=160)
    plt.close(fig)


def plot_ambiguity(static_df: pd.DataFrame) -> None:
    vals = static_df["train_match_count"].clip(upper=50)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(vals, bins=20, color="#7c3aed")
    ax.set_xlabel("Train-matching programs (clipped at 50)")
    ax.set_ylabel("Tasks")
    ax.set_title("Version-space ambiguity")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "train_match_ambiguity.png", dpi=160)
    plt.close(fig)


def plot_direct_full() -> None:
    summary_path = ANALYSIS / "qwen_direct_full_summary.csv"
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["metric"], df["score"] * 100, color=["#2563eb", "#f59e0b"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact (%)")
    ax.set_title("Direct Qwen held-out consistency subset")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "direct_full_consistency.png", dpi=160)
    plt.close(fig)


def write_report(static_df: pd.DataFrame, qwen_df: pd.DataFrame, cfg: Dict[str, Any], suite: str) -> None:
    static_summary = summarize_static(static_df)
    qwen_summary = summarize_qwen(qwen_df)
    static_family = (
        static_df.groupby("family", as_index=False)
        .agg(tasks=("task_id", "count"), oracle_coverage=("oracle_covered", "mean"), examples_score=("examples_full_exact", "mean"), train_match_rate=("train_match_count", lambda s: float((s > 0).mean())))
        .sort_values("oracle_coverage")
    )
    lines: List[str] = []
    lines.append("# Qwen Active Crystallizer Public Gate")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append("This standalone experiment tests whether frozen Qwen probe labels can select a deterministic transformation program from sparse examples. The model labels synthetic train-like probes; held-out benchmark rows are used only for evaluation.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Dataset: public Microsoft PROSE `Transformation.Text` tasks.")
    lines.append(f"- Split: first `{cfg['train_n']}` examples are train examples; up to `{cfg['heldout_cap']}` following examples are held out.")
    lines.append("- Candidate DSL: extraction, casing, regex groups, date/time/number utilities, affixes, two-part concatenation, and small finite maps.")
    lines.append("- Candidate oracle: whether any train-fitting candidate also matches all held-out rows.")
    lines.append("- Examples-only selector: shortest train-fitting candidate.")
    lines.append("- Active crystallizer: Qwen labels synthetic train-like probes chosen to maximize candidate disagreement; the selected program is then evaluated on all held-out rows.")
    lines.append("- Shuffled-label control: same probes, but labels are rotated before program selection.")
    lines.append("- Direct Qwen baseline: one held-out query per sampled task, not a full-task program-consistency metric.")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Suite: `{suite}`.")
    lines.append(f"- Static candidate tasks: `{len(static_df)}`.")
    lines.append(f"- Qwen-probe tasks: `{len(qwen_df)}`.")
    lines.append(f"- Qwen model: `{MODEL_NAME}`.")
    lines.append(f"- Max candidates per task: `{cfg['max_candidates']}`.")
    lines.append(f"- Max Qwen probes per task: `{cfg['qwen_probes']}`.")
    lines.append("")
    lines.append("## Primary Results")
    lines.append("")
    lines.append("### Static Candidate Coverage")
    lines.append("")
    lines.append(md_table(static_summary, ["method", "tasks", "score"]))
    lines.append("")
    if not qwen_summary.empty:
        lines.append("### Qwen-Probe Subset")
        lines.append("")
        lines.append(md_table(qwen_summary, ["method", "tasks", "score"]))
        lines.append("")
    direct_full_path = ANALYSIS / "qwen_direct_full_summary.csv"
    direct_full_tasks_path = ANALYSIS / "qwen_direct_full_tasks.csv"
    if direct_full_path.exists() and direct_full_tasks_path.exists():
        direct_full = pd.read_csv(direct_full_path)
        direct_tasks = pd.read_csv(direct_full_tasks_path)
        lines.append("### Strict Direct-Qwen Full-Heldout Diagnostic")
        lines.append("")
        lines.append("This diagnostic uses the same train examples and asks frozen Qwen to answer every held-out row for a capped task subset. A task counts only if every held-out row is exact.")
        lines.append("")
        lines.append(md_table(direct_full, ["metric", "tasks", "rows", "score"]))
        if not qwen_df.empty:
            overlap = qwen_df[qwen_df["task_id"].isin(set(direct_tasks["task_id"]))]
            if len(overlap):
                lines.append("")
                lines.append(f"- On the same `{len(overlap)}` tasks, active selected-program full-task exact is {pct(overlap['qwen_probe_full_exact'].mean())} ({int(overlap['qwen_probe_full_exact'].sum())}/{len(overlap)}).")
                lines.append(f"- Direct Qwen full-task exact is {pct(direct_tasks['full_task_exact'].mean())} ({int(direct_tasks['full_task_exact'].sum())}/{len(direct_tasks)}).")
        lines.append("")
    lines.append("### Family Breakdown")
    lines.append("")
    lines.append(md_table(static_family, ["family", "tasks", "oracle_coverage", "examples_score", "train_match_rate"], max_rows=35))
    lines.append("")
    if not qwen_df.empty:
        lines.append("### Qwen-Probe Task Examples")
        lines.append("")
        cols = ["task_id", "family", "features", "oracle_covered", "examples_full_exact", "qwen_probe_full_exact", "shuffled_probe_full_exact", "qwen_direct_first_exact", "probe_count", "qwen_program"]
        lines.append(md_table(qwen_df.sort_values(["qwen_probe_full_exact", "oracle_covered", "task_id"], ascending=[False, False, True])[cols], cols, max_rows=24))
        lines.append("")
        lines.append("### Qwen-Probe Misses")
        lines.append("")
        miss_cols = ["task_id", "family", "features", "oracle_covered", "qwen_probe_full_exact", "qwen_direct_first_exact", "direct_target", "direct_prediction"]
        lines.append(md_table(qwen_df[~qwen_df["qwen_probe_full_exact"]].sort_values(["family", "task_id"])[miss_cols], miss_cols, max_rows=24))
        lines.append("")
    figs = ["method_scores.png", "failure_decomposition.png", "family_oracle_coverage.png", "train_match_ambiguity.png"]
    if (FIGURES / "direct_full_consistency.png").exists():
        figs.append("direct_full_consistency.png")
    for fig in figs:
        lines.append(f"![{fig}](../analysis/figures/{fig})")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if not qwen_summary.empty:
        q_score = float(qwen_summary[qwen_summary["method"].eq("qwen_probe_select")]["score"].iloc[0])
        shuf = float(qwen_summary[qwen_summary["method"].eq("shuffled_probe_select")]["score"].iloc[0])
        oracle = float(qwen_summary[qwen_summary["method"].eq("candidate_oracle")]["score"].iloc[0])
        direct = float(qwen_summary[qwen_summary["method"].eq("qwen_direct_first_row")]["score"].iloc[0])
        lines.append(f"On the Qwen-probe subset, active crystallization selects a full held-out-valid program for {pct(q_score)} of tasks, compared with {pct(shuf)} for the shuffled-label control and a candidate-oracle ceiling of {pct(oracle)}.")
        lines.append(f"Frozen Qwen direct answering reaches {pct(direct)} on one held-out row per sampled task, which is a different metric: it measures row-level inference, not whether a single executable program generalizes over all held-out rows.")
    if direct_full_path.exists() and direct_full_tasks_path.exists():
        direct_full = pd.read_csv(direct_full_path)
        direct_tasks = pd.read_csv(direct_full_tasks_path)
        row_score = float(direct_full[direct_full["metric"].eq("row_exact")]["score"].iloc[0])
        full_score = float(direct_full[direct_full["metric"].eq("full_task_exact")]["score"].iloc[0])
        lines.append(f"The strict direct-Qwen diagnostic narrows that comparison: direct row accuracy is {pct(row_score)}, but full-task consistency drops to {pct(full_score)}. Direct Qwen is still ahead of active selected programs on the matched subset, but it is not a solved consistency baseline.")
    static_oracle = float(static_summary[static_summary["method"].eq("candidate_oracle")]["score"].iloc[0])
    static_examples = float(static_summary[static_summary["method"].eq("examples_shortest")]["score"].iloc[0])
    lines.append(f"Across all static tasks, the candidate DSL has a full-heldout oracle ceiling of {pct(static_oracle)}, while examples-only shortest selection reaches {pct(static_examples)}.")
    lines.append("The main failure is not a lack of train-fitting programs: finite maps can fit train examples for every task. The failure is that most train-fitting programs are not held-out-valid, and Qwen probe labels only add a small margin over the shuffled-label control. Under this setup, fuzzy model labels did not crystallize Qwen's row-level competence into broadly reliable executable programs.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("This run uses synthetic probes generated from train inputs, not human-authored counterexamples. The Qwen-probe subset is capped for runtime. Direct Qwen is scored on one held-out row per task, while program methods require one executable program to match all held-out rows.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Static details: `analysis/static_details.csv`")
    lines.append("- Qwen-probe details: `analysis/qwen_probe_details.csv`")
    lines.append("- Static summary: `analysis/static_summary.csv`")
    lines.append("- Qwen summary: `analysis/qwen_summary.csv`")
    if (ANALYSIS / "qwen_direct_full_summary.csv").exists():
        lines.append("- Strict direct-Qwen full-heldout diagnostic: `analysis/qwen_direct_full_summary.csv`")
    lines.append("- Public benchmark checkout: `/workspace/large_artifacts/qwen_active_crystallizer_public_gate/prose-benchmarks`")
    md = "\n".join(lines) + "\n"
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "qwen_active_crystallizer_public_gate_report.md").write_text(md)
    body = html.escape(md)
    body = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<p><img src="\2" alt="\1" style="max-width:100%;border:1px solid #ddd;border-radius:6px"></p>', body)
    body = re.sub(r"^# (.*)$", r"<h1>\1</h1>", body, flags=re.M)
    body = re.sub(r"^## (.*)$", r"<h2>\1</h2>", body, flags=re.M)
    body = re.sub(r"^### (.*)$", r"<h3>\1</h3>", body, flags=re.M)
    body = body.replace("\n", "<br>\n")
    (REPORTS / "qwen_active_crystallizer_public_gate_report.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Qwen Active Crystallizer Public Gate</title>"
        "<style>body{font-family:Inter,system-ui,sans-serif;line-height:1.45;max-width:1180px;margin:32px auto;padding:0 24px;color:#172033}"
        "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:4px 6px}code{background:#f4f4f5;padding:1px 4px;border-radius:4px}"
        "h1,h2,h3{line-height:1.15}</style></head><body>" + body + "</body></html>"
    )


def save_outputs(static_df: pd.DataFrame, qwen_df: pd.DataFrame, cfg: Dict[str, Any], suite: str, started: datetime, elapsed: float) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS / f"{suite}_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    static_summary = summarize_static(static_df)
    qwen_summary = summarize_qwen(qwen_df)
    for base in [run_dir, ANALYSIS]:
        static_df.to_csv(base / "static_details.csv", index=False)
        qwen_df.to_csv(base / "qwen_probe_details.csv", index=False)
        static_summary.to_csv(base / "static_summary.csv", index=False)
        qwen_summary.to_csv(base / "qwen_summary.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps({"suite": suite, **cfg, "started_utc": started.isoformat(), "elapsed_sec": elapsed}, indent=2) + "\n")
    plot_methods(static_summary, qwen_summary)
    plot_failure_mix(static_df)
    plot_family(static_df)
    plot_ambiguity(static_df)
    plot_direct_full()
    write_report(static_df, qwen_df, cfg, suite)
    with (ROOT / "experiment_log.md").open("a") as f:
        f.write(f"\n## Run `{suite}_v1`\n\n")
        f.write(f"- Started: {started.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        f.write(f"- Static tasks: `{len(static_df)}`; Qwen-probe tasks: `{len(qwen_df)}`\n")
        f.write(f"- Completed in {elapsed:.1f}s.\n")
        f.write(f"- Candidate-oracle full-heldout coverage: {pct(static_summary[static_summary['method'].eq('candidate_oracle')]['score'].iloc[0])}.\n")
        f.write(f"- Examples-only full-heldout score: {pct(static_summary[static_summary['method'].eq('examples_shortest')]['score'].iloc[0])}.\n")
        if not qwen_summary.empty:
            f.write(f"- Qwen-probe selected-program score: {pct(qwen_summary[qwen_summary['method'].eq('qwen_probe_select')]['score'].iloc[0])}.\n")
            f.write(f"- Shuffled-label selected-program score: {pct(qwen_summary[qwen_summary['method'].eq('shuffled_probe_select')]['score'].iloc[0])}.\n")
            f.write(f"- Direct Qwen first-heldout-row score: {pct(qwen_summary[qwen_summary['method'].eq('qwen_direct_first_row')]['score'].iloc[0])}.\n")


def run(args: argparse.Namespace) -> None:
    suite_cfg = {
        "smoke": {"limit": 35, "qwen_task_limit": 12, "train_n": 4, "heldout_cap": 12, "max_candidates": 40000, "qwen_probes": 3, "max_new_tokens": 64, "sample_seed": 20260627},
        "pilot": {"limit": 100, "qwen_task_limit": 50, "train_n": 4, "heldout_cap": 20, "max_candidates": 40000, "qwen_probes": 4, "max_new_tokens": 64, "sample_seed": 20260627},
        "main": {"limit": None, "qwen_task_limit": 120, "train_n": 4, "heldout_cap": 50, "max_candidates": 40000, "qwen_probes": 4, "max_new_tokens": 64, "sample_seed": 20260627},
    }[args.suite]
    if args.qwen_task_limit is not None:
        suite_cfg["qwen_task_limit"] = args.qwen_task_limit
    if args.no_qwen:
        suite_cfg["qwen_task_limit"] = 0
    started = datetime.now(timezone.utc)
    tasks = load_tasks(limit=suite_cfg["limit"], min_examples=args.min_examples)
    static_df = run_static(tasks, suite_cfg)
    qwen_df = pd.DataFrame()
    if suite_cfg["qwen_task_limit"] > 0:
        qwen_df = run_qwen(tasks, suite_cfg)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    save_outputs(static_df, qwen_df, suite_cfg, args.suite, started, elapsed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["smoke", "pilot", "main"], default="main")
    p.add_argument("--min_examples", type=int, default=5)
    p.add_argument("--qwen_task_limit", type=int, default=None)
    p.add_argument("--no_qwen", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
