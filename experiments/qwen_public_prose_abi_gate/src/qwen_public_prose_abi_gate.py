#!/usr/bin/env python3
"""Public PROSE Transformation.Text ABI coverage gate.

The experiment freezes a reusable transformation-template ABI and evaluates it
on public Microsoft PROSE Transformation.Text benchmarks. Programs are selected
using only train examples and counted as coverage only when they also match
held-out examples from the same benchmark.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/workspace/experiments/qwen_public_prose_abi_gate")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_public_prose_abi_gate")
PROSE_ROOT = LARGE_ROOT / "prose-benchmarks"
TRANSFORM_ROOT = PROSE_ROOT / "Transformation.Text"
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"


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
    tier: str

    def eval_many(self, rows: Sequence[Tuple[str, ...]]) -> Optional[Tuple[str, ...]]:
        vals: List[str] = []
        try:
            for row in rows:
                v = self.func(row)
                if v is None:
                    return None
                vals.append(str(v))
        except Exception:
            return None
        return tuple(vals)


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
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        return None if not m else float(m.group(0))


DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), ("Y", "M", "D")),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b"), ("M", "D", "Y")),
    (re.compile(r"\b(\d{1,2})-(\d{1,2})-(\d{2,4})\b"), ("M", "D", "Y")),
]
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def date_parts(s: Any) -> Optional[Tuple[int, int, int]]:
    text = clean(s)
    for rgx, order in DATE_PATTERNS:
        m = rgx.search(text)
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


def load_tasks(limit: Optional[int] = None, min_examples: int = 5) -> List[Task]:
    tasks: List[Task] = []
    for d in sorted(TRANSFORM_ROOT.iterdir()):
        if not d.is_dir() or not (d / "spec.json").exists():
            continue
        spec = json.loads((d / "spec.json").read_text())
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        examples = []
        for ex in spec.get("Examples", []):
            inp = tuple(str(x) for x in ex.get("Input", []))
            out = str(ex.get("Output", ""))
            examples.append(Example(inp, out))
        if len(examples) < min_examples:
            continue
        family = d.name.split(".", 1)[0]
        tasks.append(
            Task(
                task_id=d.name,
                family=family,
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


def nth_field(text: str, sep: str, idx: int) -> Optional[str]:
    parts = [p.strip() for p in text.split(sep)]
    if not parts:
        return None
    j = idx if idx >= 0 else len(parts) + idx
    if j < 0 or j >= len(parts):
        return None
    return parts[j]


def regex_group(text: str, pattern: str, idx: int = 1) -> Optional[str]:
    m = re.search(pattern, text)
    if not m:
        return None
    return m.group(idx)


def source_exprs(num_cols: int, tier: str) -> List[Expr]:
    exprs: List[Expr] = []
    for i in range(num_cols):
        base = lambda row, j=i: row[j] if j < len(row) else ""
        prefix = f"COL{i}"
        exprs.append(Expr(prefix, base, 0, "core"))
        exprs.extend(
            [
                Expr(f"strip({prefix})", lambda row, j=i: clean(row[j]), 1, "core"),
                Expr(f"lower({prefix})", lambda row, j=i: clean(row[j]).lower(), 1, "core"),
                Expr(f"upper({prefix})", lambda row, j=i: clean(row[j]).upper(), 1, "core"),
                Expr(f"title({prefix})", lambda row, j=i: title_words(row[j]), 1, "core"),
                Expr(f"digits({prefix})", lambda row, j=i: only_digits(row[j]), 1, "core"),
                Expr(f"alpha({prefix})", lambda row, j=i: only_alpha_space(row[j]), 1, "core"),
                Expr(f"alnum({prefix})", lambda row, j=i: only_alnum(row[j]), 1, "core"),
                Expr(f"initials({prefix})", lambda row, j=i: initials(row[j]), 1, "core"),
                Expr(f"initials_sp({prefix})", lambda row, j=i: initials(row[j], " "), 1, "core"),
                Expr(f"first_last_initials({prefix})", lambda row, j=i: first_last_initials(row[j]), 1, "core"),
            ]
        )
        seps = [" ", ",", "-", "/", "\\", "_", ":", ";", "|", ".", "\n", "\t", "(", ")"]
        for sep in seps:
            label = {" ": "space", "\n": "nl", "\t": "tab", "\\": "bslash"}.get(sep, sep)
            for idx in [0, 1, 2, 3, 4, -1, -2]:
                exprs.append(
                    Expr(
                        f"field[{label},{idx}]({prefix})",
                        lambda row, j=i, s=sep, k=idx: nth_field(row[j], s, k),
                        1,
                        "core",
                    )
                )
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
        for name, pattern, group in regexes:
            exprs.append(Expr(f"{name}({prefix})", lambda row, j=i, p=pattern, g=group: regex_group(row[j], p, g), 1, "office"))
        exprs.extend(
            [
                Expr(f"date_year({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else str(date_parts(row[j])[0]), 1, "office"),
                Expr(f"date_month({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else str(date_parts(row[j])[1]), 1, "office"),
                Expr(f"date_month2({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else f"{date_parts(row[j])[1]:02d}", 1, "office"),
                Expr(f"date_day({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else str(date_parts(row[j])[2]), 1, "office"),
                Expr(f"date_day2({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else f"{date_parts(row[j])[2]:02d}", 1, "office"),
                Expr(f"date_iso({prefix})", lambda row, j=i: None if date_parts(row[j]) is None else f"{date_parts(row[j])[0]:04d}-{date_parts(row[j])[1]:02d}-{date_parts(row[j])[2]:02d}", 1, "office"),
                Expr(f"time_hour({prefix})", lambda row, j=i: None if time_parts(row[j]) is None else str(time_parts(row[j])[0]), 1, "office"),
                Expr(f"time_hour2({prefix})", lambda row, j=i: None if time_parts(row[j]) is None else f"{time_parts(row[j])[0]:02d}", 1, "office"),
                Expr(f"time_minute({prefix})", lambda row, j=i: None if time_parts(row[j]) is None else str(time_parts(row[j])[1]), 1, "office"),
                Expr(f"number_int({prefix})", lambda row, j=i: fmt_int(parse_float(row[j])), 1, "office"),
                Expr(f"number_2dp({prefix})", lambda row, j=i: None if parse_float(row[j]) is None else f"{parse_float(row[j]):.2f}", 1, "office"),
            ]
        )
    return [e for e in exprs if e.tier == "core" or tier in {"office", "concat"}]


def wrappers(exprs: List[Expr], tier: str) -> List[Expr]:
    if tier == "core":
        return exprs
    out = list(exprs)
    transforms = [
        ("strip", lambda x: clean(x)),
        ("lower", lambda x: clean(x).lower()),
        ("upper", lambda x: clean(x).upper()),
        ("title", title_words),
        ("digits", only_digits),
        ("alnum", only_alnum),
        ("alpha", only_alpha_space),
    ]
    for e in exprs:
        if e.depth > 1:
            continue
        for name, fn in transforms:
            out.append(Expr(f"{name}({e.code})", lambda row, ex=e, f=fn: None if ex.func(row) is None else f(ex.func(row)), e.depth + 1, "office"))
    return out


def concat_exprs(exprs: List[Expr], train_rows: Sequence[Tuple[str, ...]], max_source: int = 90) -> List[Expr]:
    ranked = sorted(exprs, key=lambda e: (e.depth, len(e.code), e.code))
    usable = []
    for e in ranked:
        vals = e.eval_many(train_rows)
        if vals is None:
            continue
        if any(v == "" for v in vals):
            continue
        if len(set(vals)) <= max(1, len(vals) // 3):
            continue
        usable.append(e)
        if len(usable) >= max_source:
            break
    seps = ["", " ", ", ", "-", "/", "_", ":", "."]
    out: List[Expr] = []
    for i, a in enumerate(usable):
        for b in usable:
            if a.code == b.code:
                continue
            for sep in seps:
                out.append(
                    Expr(
                        f"concat[{sep!r}]({a.code},{b.code})",
                        lambda row, x=a, y=b, s=sep: None if x.func(row) is None or y.func(row) is None else f"{x.func(row)}{s}{y.func(row)}",
                        max(a.depth, b.depth) + 1,
                        "concat",
                    )
                )
    return out


def candidate_exprs(num_cols: int, tier: str, train_rows: Sequence[Tuple[str, ...]]) -> List[Expr]:
    exprs = wrappers(source_exprs(num_cols, "office" if tier in {"office", "concat"} else "core"), tier)
    if tier == "concat":
        exprs += concat_exprs(exprs, train_rows)
    seen = set()
    dedup: List[Expr] = []
    for e in sorted(exprs, key=lambda x: (x.depth, len(x.code), x.code)):
        key = e.code
        if key in seen:
            continue
        seen.add(key)
        dedup.append(e)
    return dedup


def evaluate_task(task: Task, tier: str, train_n: int, heldout_cap: int) -> Dict[str, Any]:
    train_ex, test_ex = split_examples(task, train_n, heldout_cap)
    train_rows = [e.inputs for e in train_ex]
    test_rows = [e.inputs for e in test_ex]
    train_y = tuple(e.output for e in train_ex)
    test_y = tuple(e.output for e in test_ex)
    num_cols = max(len(e.inputs) for e in task.examples)
    candidates = candidate_exprs(num_cols, tier, train_rows)
    train_matches: List[Expr] = []
    test_matches: List[Expr] = []
    seen_sig: Dict[Tuple[str, ...], Expr] = {}
    for expr in candidates:
        sig = expr.eval_many(train_rows)
        if sig is None:
            continue
        if sig in seen_sig:
            continue
        seen_sig[sig] = expr
        if sig == train_y:
            train_matches.append(expr)
            if expr.eval_many(test_rows) == test_y:
                test_matches.append(expr)
    best_train = min(train_matches, key=lambda e: (e.depth, len(e.code), e.code), default=None)
    best_test = min(test_matches, key=lambda e: (e.depth, len(e.code), e.code), default=None)
    return {
        "covered": best_test is not None,
        "train_match": best_train is not None,
        "failure_reason": "covered" if best_test else ("train_match_only" if best_train else "no_train_match"),
        "program": best_test.code if best_test else (best_train.code if best_train else ""),
        "program_depth": best_test.depth if best_test else (best_train.depth if best_train else None),
        "candidate_count": len(candidates),
        "train_match_count": len(train_matches),
        "train_examples": len(train_ex),
        "heldout_examples": len(test_ex),
    }


def evaluate(tasks: List[Task], tiers: List[str], train_n: int, heldout_cap: int) -> pd.DataFrame:
    rows = []
    for tier in tiers:
        print(f"tier={tier} tasks={len(tasks)}", flush=True)
        for i, task in enumerate(tasks, start=1):
            if i == 1 or i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} {task.task_id}", flush=True)
            result = evaluate_task(task, tier, train_n, heldout_cap)
            rows.append(
                {
                    "tier": tier,
                    "task_id": task.task_id,
                    "family": task.family,
                    "synthetic": task.synthetic,
                    "features": ",".join(task.features),
                    "num_features": len(task.features),
                    "num_examples": len(task.examples),
                    "num_inputs": max(len(e.inputs) for e in task.examples),
                    **result,
                }
            )
    return pd.DataFrame(rows)


def summarize(details: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = (
        details.groupby("tier", as_index=False)
        .agg(
            tasks=("task_id", "count"),
            coverage=("covered", "mean"),
            train_match_rate=("train_match", "mean"),
            train_only_rate=("failure_reason", lambda s: float((s == "train_match_only").mean())),
            no_train_match_rate=("failure_reason", lambda s: float((s == "no_train_match").mean())),
            median_candidates=("candidate_count", "median"),
        )
        .sort_values("tier")
    )
    family = (
        details.groupby(["tier", "family"], as_index=False)
        .agg(tasks=("task_id", "count"), coverage=("covered", "mean"), train_only_rate=("failure_reason", lambda s: float((s == "train_match_only").mean())))
        .sort_values(["tier", "coverage"])
    )
    feat_rows = []
    for tier, sub in details.groupby("tier"):
        features = sorted({f for val in sub["features"] for f in str(val).split(",") if f})
        for feat in features:
            fs = sub[sub["features"].str.contains(rf"(?:^|,){re.escape(feat)}(?:,|$)", regex=True)]
            feat_rows.append({"tier": tier, "feature": feat, "tasks": len(fs), "coverage": float(fs["covered"].mean()), "train_only_rate": float((fs["failure_reason"] == "train_match_only").mean())})
    feature = pd.DataFrame(feat_rows).sort_values(["tier", "coverage", "feature"])
    return overall, family, feature


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
            if "coverage" in c or "rate" in c:
                view[c] = view[c].map(pct)
            else:
                view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    header = "|" + "|".join(cols) + "|"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["|" + "|".join(html.escape(str(row[c])) for c in cols) + "|" for _, row in view.iterrows()]
    return "\n".join([header, sep] + body)


def plot_overall(overall: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(overall["tier"], overall["coverage"] * 100, color="#2563eb")
    ax.set_ylabel("Held-out coverage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Coverage by frozen ABI tier")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage_by_tier.png", dpi=160)
    plt.close(fig)


def plot_family(details: pd.DataFrame, tier: str) -> None:
    sub = details[details["tier"].eq(tier)]
    vals = sub.groupby("family")["covered"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(5, 0.18 * len(vals))))
    ax.barh(vals.index, vals.values * 100, color="#059669")
    ax.set_xlabel("Held-out coverage (%)")
    ax.set_xlim(0, 105)
    ax.set_title(f"Coverage by family ({tier})")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage_by_family.png", dpi=160)
    plt.close(fig)


def plot_feature(feature: pd.DataFrame, tier: str) -> None:
    sub = feature[feature["tier"].eq(tier)].sort_values("coverage")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(sub["feature"], sub["coverage"] * 100, color="#7c3aed")
    ax.set_xlabel("Held-out coverage (%)")
    ax.set_xlim(0, 105)
    ax.set_title(f"Coverage by PROSE feature ({tier})")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage_by_feature.png", dpi=160)
    plt.close(fig)


def plot_failures(details: pd.DataFrame, tier: str) -> None:
    sub = details[details["tier"].eq(tier)]
    vals = sub["failure_reason"].value_counts(normalize=True).reindex(["covered", "train_match_only", "no_train_match"]).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(vals.index, vals.values * 100, color=["#10b981", "#f59e0b", "#ef4444"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Tasks (%)")
    ax.set_title(f"Outcome mix ({tier})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "failure_modes.png", dpi=160)
    plt.close(fig)


def plot_synthetic(details: pd.DataFrame, tier: str) -> None:
    sub = details[details["tier"].eq(tier)]
    vals = sub.groupby("synthetic")["covered"].mean()
    labels = ["non-synthetic" if not k else "synthetic" for k in vals.index]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(labels, vals.values * 100, color="#ea580c")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Held-out coverage (%)")
    ax.set_title(f"Synthetic metadata split ({tier})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "synthetic_split.png", dpi=160)
    plt.close(fig)


def plot_qwen_direct() -> None:
    sample_path = ANALYSIS / "qwen_direct_sample.csv"
    if not sample_path.exists():
        return
    sample = pd.read_csv(sample_path)
    if sample.empty:
        return
    vals = [
        ("overall", sample["exact"].mean()),
        ("ABI covered", sample[sample["abi_covered"].eq(True)]["exact"].mean()),
        ("ABI missed", sample[sample["abi_covered"].eq(False)]["exact"].mean()),
    ]
    labels = [x[0] for x in vals]
    scores = [0.0 if pd.isna(x[1]) else x[1] * 100 for x in vals]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, scores, color=["#2563eb", "#10b981", "#f59e0b"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact match on one held-out query (%)")
    ax.set_title("Frozen Qwen direct-answer sample")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "qwen_direct_sample.png", dpi=160)
    plt.close(fig)


def make_report(details: pd.DataFrame, overall: pd.DataFrame, family: pd.DataFrame, feature: pd.DataFrame, suite: str, train_n: int, heldout_cap: int) -> None:
    tier = "concat"
    primary = details[details["tier"].eq(tier)]
    covered = primary[primary["covered"]]
    train_only = primary[primary["failure_reason"].eq("train_match_only")]
    no_train = primary[primary["failure_reason"].eq("no_train_match")]
    synthetic = primary[primary["synthetic"]]
    nonsynth = primary[~primary["synthetic"]]

    lines = []
    lines.append("# Qwen Public PROSE ABI Gate")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append("This standalone experiment evaluates a frozen deterministic transformation ABI on the public Microsoft PROSE `Transformation.Text` benchmark. A program is selected from train examples and counted only if it also matches held-out examples from the same task.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Dataset: Microsoft PROSE public benchmark suite, `Transformation.Text`.")
    lines.append("- Tasks with too few examples are excluded so every scored task has held-out examples.")
    lines.append(f"- Split: first `{train_n}` examples, capped by task size, are train examples; up to `{heldout_cap}` following examples are held out.")
    lines.append("- Frozen ABI tiers: `core` surface extraction/casing, `office` adds regex/date/number/domain primitives, `concat` adds two-part concatenation over ABI expressions.")
    lines.append("- Train-only fits are counted as failures because they do not validate task semantics on held-out rows.")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Suite: `{suite}`.")
    lines.append(f"- Scored tasks: `{len(primary)}`.")
    lines.append(f"- Public benchmark checkout: `{PROSE_ROOT}`.")
    lines.append("")
    lines.append("## Primary Results")
    lines.append("")
    lines.append(f"- Best frozen tier coverage: {pct(primary['covered'].mean())} ({int(primary['covered'].sum())}/{len(primary)} tasks).")
    lines.append(f"- Train-only/coincidence failures: {pct((primary['failure_reason'] == 'train_match_only').mean())} ({len(train_only)} tasks).")
    lines.append(f"- No-train-match failures: {pct((primary['failure_reason'] == 'no_train_match').mean())} ({len(no_train)} tasks).")
    if len(synthetic):
        lines.append(f"- Synthetic-metadata coverage: {pct(synthetic['covered'].mean())} ({int(synthetic['covered'].sum())}/{len(synthetic)} tasks).")
    if len(nonsynth):
        lines.append(f"- Non-synthetic-metadata coverage: {pct(nonsynth['covered'].mean())} ({int(nonsynth['covered'].sum())}/{len(nonsynth)} tasks).")
    lines.append("")
    qwen_sample_path = ANALYSIS / "qwen_direct_sample.csv"
    qwen_summary_path = ANALYSIS / "qwen_direct_summary.csv"
    qwen_meta_path = RUNS / "qwen_direct_sample_meta.json"
    if qwen_sample_path.exists() and qwen_summary_path.exists():
        qwen_sample = pd.read_csv(qwen_sample_path)
        qwen_summary = pd.read_csv(qwen_summary_path)
        qwen_meta = json.loads(qwen_meta_path.read_text()) if qwen_meta_path.exists() else {}
        q_overall = qwen_summary[qwen_summary["slice"].eq("overall")]
        q_score = float(q_overall.iloc[0]["exact"]) if len(q_overall) else float(qwen_sample["exact"].mean())
        lines.append("### Frozen Qwen Direct-Answer Sample")
        lines.append("")
        lines.append(f"- Model: `{qwen_meta.get('model', 'Qwen/Qwen3-4B')}`.")
        lines.append(f"- Sample: `{len(qwen_sample)}` tasks, seed `{qwen_meta.get('seed', 'n/a')}`.")
        lines.append(f"- Prompt: first `{qwen_meta.get('train_n', train_n)}` examples, then one held-out query.")
        lines.append(f"- Exact match on that one held-out query: {pct(q_score)} ({int(qwen_sample['exact'].sum())}/{len(qwen_sample)} tasks).")
        lines.append("- This is a diagnostic baseline, not the same metric as ABI coverage: it scores one held-out query, while ABI coverage requires one program to match all held-out rows.")
        lines.append("")
        lines.append(md_table(qwen_summary, ["slice", "tasks", "exact"], max_rows=30))
        lines.append("")
        q_cols = ["task_id", "family", "features", "abi_covered", "exact", "target", "prediction"]
        lines.append("#### Qwen Sample Misses")
        lines.append("")
        lines.append(md_table(qwen_sample[~qwen_sample["exact"]].sort_values(["family", "task_id"])[q_cols], q_cols, max_rows=18))
        lines.append("")
    lines.append("### Overall By Tier")
    lines.append("")
    lines.append(md_table(overall, ["tier", "tasks", "coverage", "train_match_rate", "train_only_rate", "no_train_match_rate", "median_candidates"]))
    lines.append("")
    lines.append("### Feature Coverage")
    lines.append("")
    lines.append(md_table(feature[feature["tier"].eq(tier)].sort_values("coverage"), ["feature", "tasks", "coverage", "train_only_rate"], max_rows=40))
    lines.append("")
    lines.append("### Lowest-Coverage Families")
    lines.append("")
    lines.append(md_table(family[family["tier"].eq(tier)].sort_values("coverage"), ["family", "tasks", "coverage", "train_only_rate"], max_rows=35))
    lines.append("")
    lines.append("### Covered Program Examples")
    lines.append("")
    ex_cols = ["task_id", "family", "features", "num_examples", "program_depth", "program"]
    lines.append(md_table(covered.sort_values(["family", "task_id"])[ex_cols], ex_cols, max_rows=22))
    lines.append("")
    lines.append("### Train-Only Failures")
    lines.append("")
    fail_cols = ["task_id", "family", "features", "num_examples", "program"]
    lines.append(md_table(train_only.sort_values(["family", "task_id"])[fail_cols], fail_cols, max_rows=22))
    lines.append("")
    lines.append("### No-Train-Match Failures")
    lines.append("")
    lines.append(md_table(no_train.sort_values(["family", "task_id"])[["task_id", "family", "features", "num_examples"]], ["task_id", "family", "features", "num_examples"], max_rows=28))
    lines.append("")
    figs = ["coverage_by_tier.png", "coverage_by_family.png", "coverage_by_feature.png", "failure_modes.png", "synthetic_split.png"]
    if (FIGURES / "qwen_direct_sample.png").exists():
        figs.append("qwen_direct_sample.png")
    for fig in figs:
        lines.append(f"![{fig}](../analysis/figures/{fig})")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This public benchmark is a hard gate because task definitions come from an independent repository rather than this experiment's generator. It directly tests whether a frozen transformation ABI covers independent string-transformation tasks under held-out validation.")
    lines.append(f"The result is negative for this frozen ABI: the best tier covers only {pct(primary['covered'].mean())}, and most misses are no-train-match failures rather than held-out coincidence failures. The ABI is therefore too narrow for the public `Transformation.Text` suite without substantial primitive expansion or a retrieval step that selects a domain-specific ABI.")
    if qwen_sample_path.exists() and qwen_summary_path.exists():
        lines.append("The frozen-Qwen direct-answer diagnostic points in a different direction: the model often infers these public transformations from examples even when the frozen ABI has no matching program. That suggests the immediate bottleneck is not that the transformations are impossible for the model; it is that the fixed ABI does not expose the right operations for this benchmark.")
    lines.append("The train-only column is load-bearing. Those tasks are exactly the cases where a plausible expression fits examples but fails held-out rows, so they are excluded from coverage.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("This run covers `Transformation.Text`, not `Split.Text` or `Extraction.Text`. The ABI is a fixed template library rather than a complete PROSE-style DSL, so misses can reflect missing primitives or bounded search. The benchmark itself contains mostly synthetic data according to its metadata, though it is independent public data rather than generated by this experiment.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Details: `analysis/details.csv`")
    lines.append("- Overall summary: `analysis/overall_summary.csv`")
    lines.append("- Family summary: `analysis/family_summary.csv`")
    lines.append("- Feature summary: `analysis/feature_summary.csv`")
    if qwen_sample_path.exists():
        lines.append("- Frozen Qwen direct-answer sample: `analysis/qwen_direct_sample.csv`")
    lines.append("- Public benchmark checkout: `/workspace/large_artifacts/qwen_public_prose_abi_gate/prose-benchmarks`")

    md = "\n".join(lines) + "\n"
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "qwen_public_prose_abi_gate_report.md").write_text(md)
    body = html.escape(md)
    body = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<p><img src="\2" alt="\1" style="max-width:100%;border:1px solid #ddd;border-radius:6px"></p>', body)
    body = re.sub(r"^# (.*)$", r"<h1>\1</h1>", body, flags=re.M)
    body = re.sub(r"^## (.*)$", r"<h2>\1</h2>", body, flags=re.M)
    body = re.sub(r"^### (.*)$", r"<h3>\1</h3>", body, flags=re.M)
    body = body.replace("\n", "<br>\n")
    (REPORTS / "qwen_public_prose_abi_gate_report.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Qwen Public PROSE ABI Gate</title>"
        "<style>body{font-family:Inter,system-ui,sans-serif;line-height:1.45;max-width:1180px;margin:32px auto;padding:0 24px;color:#172033}"
        "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:4px 6px}code{background:#f4f4f5;padding:1px 4px;border-radius:4px}"
        "h1,h2,h3{line-height:1.15}</style></head><body>" + body + "</body></html>"
    )


def run(args: argparse.Namespace) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LARGE_ROOT.mkdir(parents=True, exist_ok=True)

    suite_cfg = {
        "smoke": {"limit": 40, "train_n": 3, "heldout_cap": 12},
        "pilot": {"limit": 120, "train_n": 4, "heldout_cap": 20},
        "main": {"limit": None, "train_n": 4, "heldout_cap": 50},
    }[args.suite]
    started = datetime.now(timezone.utc)
    tasks = load_tasks(limit=suite_cfg["limit"], min_examples=args.min_examples)
    tiers = ["core", "office", "concat"]
    details = evaluate(tasks, tiers, suite_cfg["train_n"], suite_cfg["heldout_cap"])
    overall, family, feature = summarize(details)

    run_dir = RUNS / f"{args.suite}_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(run_dir / "details.csv", index=False)
    overall.to_csv(run_dir / "overall_summary.csv", index=False)
    family.to_csv(run_dir / "family_summary.csv", index=False)
    feature.to_csv(run_dir / "feature_summary.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps({"suite": args.suite, **suite_cfg, "min_examples": args.min_examples, "started_utc": started.isoformat()}, indent=2))

    details.to_csv(ANALYSIS / "details.csv", index=False)
    overall.to_csv(ANALYSIS / "overall_summary.csv", index=False)
    family.to_csv(ANALYSIS / "family_summary.csv", index=False)
    feature.to_csv(ANALYSIS / "feature_summary.csv", index=False)

    plot_overall(overall)
    plot_family(details, "concat")
    plot_feature(feature, "concat")
    plot_failures(details, "concat")
    plot_synthetic(details, "concat")
    plot_qwen_direct()
    make_report(details, overall, family, feature, args.suite, suite_cfg["train_n"], suite_cfg["heldout_cap"])

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    primary = details[details["tier"].eq("concat")]
    with (ROOT / "experiment_log.md").open("a") as f:
        f.write(f"\n## Run `{args.suite}_v1`\n\n")
        f.write(f"- Started: {started.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        f.write(f"- Tasks: `{len(tasks)}`\n")
        f.write(f"- Train examples per task: `{suite_cfg['train_n']}`; held-out cap: `{suite_cfg['heldout_cap']}`\n")
        f.write(f"- Completed in {elapsed:.1f}s.\n")
        f.write(f"- Primary coverage: {pct(primary['covered'].mean())} ({int(primary['covered'].sum())}/{len(primary)} tasks).\n")
        f.write(f"- Train-only: {pct((primary['failure_reason'] == 'train_match_only').mean())}; no-train-match: {pct((primary['failure_reason'] == 'no_train_match').mean())}.\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["smoke", "pilot", "main"], default="main")
    p.add_argument("--min_examples", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
