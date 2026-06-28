#!/usr/bin/env python3
"""Active clarifying-example acquisition for text transformations.

The experiment starts each task with four visible input/output examples. A
selector may reveal one or three additional examples from a withheld acquisition
pool before the model answers a separate held-out evaluation set. The primary
metric is strict full-task exactness over the held-out rows.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path("/workspace/experiments/qwen_active_example_acquisition")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_active_example_acquisition")
SOURCE_BENCH_ROOT = Path("/workspace/large_artifacts/qwen_batched_transduction_consistency/prose-benchmarks")
BENCH_ROOT = LARGE_ROOT / "prose-benchmarks"
TRANSFORM_ROOT = BENCH_ROOT / "Transformation.Text"
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CACHE_DIR = Path("/workspace/.cache/huggingface")
MODEL_NAME = "Qwen/Qwen3-4B"
VARIANTS = ("plain", "format", "consistency")


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


def ensure_dirs() -> None:
    for d in [RUNS, ANALYSIS, FIGURES, REPORTS, LARGE_ROOT]:
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


def split_task(task: Task, base_n: int, pool_n: int, eval_n: int) -> Tuple[List[Example], List[Example], List[Example]]:
    base = list(task.examples[:base_n])
    pool = list(task.examples[base_n : base_n + pool_n])
    eval_rows = list(task.examples[base_n + pool_n : base_n + pool_n + eval_n])
    if len(base) < base_n or len(pool) < pool_n or len(eval_rows) < eval_n:
        raise ValueError(f"Task {task.task_id} does not have enough examples")
    return base, pool, eval_rows


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


def clean_prediction(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    if text.lower().startswith("output:"):
        text = text.split(":", 1)[1].strip()
    first = text.splitlines()[0].strip() if text else ""
    if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
        first = first[1:-1].strip()
    return first


def row_prompt(train_pairs: Sequence[Tuple[Tuple[str, ...], str]], query: Tuple[str, ...], variant: str) -> str:
    if variant == "format":
        intro = [
            "Infer the exact text transformation and exact output formatting from the examples.",
            "Return only the transformed output for the query. Do not explain.",
        ]
    elif variant == "consistency":
        intro = [
            "The examples define one deterministic text transformation.",
            "Apply the same transformation to the query. Return only the output string.",
        ]
    else:
        intro = [
            "Infer the text transformation from the examples.",
            "Return only the transformed output for the query. Do not explain.",
        ]
    lines = intro + ["", "Examples:"]
    for inp, out in train_pairs:
        lines.append(f"Input: {render_inputs(inp)}")
        lines.append(f"Output: {out}")
    lines.extend(["", "Query:", f"Input: {render_inputs(query)}", "Output:"])
    return "\n".join(lines)


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
        self.file = self.path.open("a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=["key", "task_id", "phase", "method", "row_id", "prediction", "raw"])
        if self.path.stat().st_size == 0:
            self.writer.writeheader()
            self.file.flush()

    def close(self) -> None:
        self.file.close()

    def get_or_generate(
        self,
        tok: Any,
        model: Any,
        key: str,
        task_id: str,
        phase: str,
        method: str,
        row_id: str,
        train_pairs: Sequence[Tuple[Tuple[str, ...], str]],
        query: Tuple[str, ...],
        max_new_tokens: int,
    ) -> str:
        if key in self.rows:
            return self.rows[key]["prediction"]
        raw = generate_text(tok, model, "You transform text exactly from examples.", row_prompt(train_pairs, query, method), max_new_tokens)
        pred = clean_prediction(raw)
        row = {"key": key, "task_id": task_id, "phase": phase, "method": method, "row_id": row_id, "prediction": pred, "raw": raw}
        self.rows[key] = row
        self.writer.writerow(row)
        self.file.flush()
        return pred


def entropy(values: Sequence[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    return -sum((c / total) * math.log(c / total + 1e-12) for c in counts.values())


def evaluate_outputs(preds: Sequence[str], targets: Sequence[str]) -> Tuple[float, bool]:
    if not targets:
        return 0.0, False
    row_acc = sum(norm_eq(p, y) for p, y in zip(preds, targets)) / len(targets)
    return row_acc, row_acc == 1.0


def select_active(pool_probe_preds: Dict[int, List[str]], budget: int) -> List[int]:
    scored = []
    for idx, vals in pool_probe_preds.items():
        scored.append((entropy(vals), len(set(vals)), idx))
    scored.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
    return [idx for _, _, idx in scored[:budget]]


def select_random(task_id: str, pool_n: int, budget: int, seed: int) -> List[int]:
    rng = random.Random(seed + stable_int(task_id))
    return sorted(rng.sample(list(range(pool_n)), budget))


def row_numeric_features(row: Tuple[str, ...]) -> np.ndarray:
    text = " | ".join(row)
    length = len(text)
    alpha = sum(ch.isalpha() for ch in text)
    digit = sum(ch.isdigit() for ch in text)
    space = sum(ch.isspace() for ch in text)
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
    tokens = len(re.findall(r"[A-Za-z0-9]+", text))
    return np.array([length, alpha, digit, space, punct, tokens], dtype=float)


def row_distance(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    fa = row_numeric_features(a)
    fb = row_numeric_features(b)
    denom = np.maximum(np.maximum(np.abs(fa), np.abs(fb)), 1.0)
    return float(np.mean(np.abs(fa - fb) / denom))


def select_diverse(base: Sequence[Example], pool: Sequence[Example], budget: int) -> List[int]:
    selected: List[int] = []
    available = set(range(len(pool)))
    anchors = [ex.inputs for ex in base]
    while available and len(selected) < budget:
        best_idx = max(
            available,
            key=lambda i: (
                min(row_distance(pool[i].inputs, anchor) for anchor in anchors),
                -i,
            ),
        )
        selected.append(best_idx)
        available.remove(best_idx)
        anchors.append(pool[best_idx].inputs)
    return selected


def rotate_labels(examples: Sequence[Example]) -> List[str]:
    labels = [ex.output for ex in examples]
    if len(labels) <= 1:
        return labels
    return labels[1:] + labels[:1]


def train_pairs_with_extra(base: Sequence[Example], pool: Sequence[Example], selected: Sequence[int], shuffled: bool = False) -> List[Tuple[Tuple[str, ...], str]]:
    pairs = [(ex.inputs, ex.output) for ex in base]
    selected_examples = [pool[i] for i in selected]
    labels = rotate_labels(selected_examples) if shuffled else [ex.output for ex in selected_examples]
    for ex, label in zip(selected_examples, labels):
        pairs.append((ex.inputs, label))
    return pairs


def loo_select_variant(
    tok: Any,
    model: Any,
    cache: GenerationCache,
    task_id: str,
    examples: Sequence[Example],
    max_new_tokens: int,
) -> Tuple[str, Dict[str, float]]:
    scores: Dict[str, float] = {}
    for variant in VARIANTS:
        correct = 0
        for i, heldout in enumerate(examples):
            train = [(ex.inputs, ex.output) for j, ex in enumerate(examples) if j != i]
            key = f"{task_id}|loo|{variant}|{len(examples)}|{i}"
            pred = cache.get_or_generate(tok, model, key, task_id, "loo", variant, f"visible{i}", train, heldout.inputs, max_new_tokens)
            correct += int(norm_eq(pred, heldout.output))
        scores[variant] = correct / max(1, len(examples))
    best = max(VARIANTS, key=lambda v: (scores[v], v == "plain", v == "format"))
    return best, scores


def evaluate_strategy(
    tok: Any,
    model: Any,
    cache: GenerationCache,
    task: Task,
    strategy: str,
    train_pairs: Sequence[Tuple[Tuple[str, ...], str]],
    eval_rows: Sequence[Example],
    variant: str,
    max_new_tokens: int,
) -> Tuple[List[str], float, bool]:
    preds: List[str] = []
    for i, ex in enumerate(eval_rows):
        train_sig = hashlib.sha256(json.dumps([(list(a), b) for a, b in train_pairs], ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        key = f"{task.task_id}|eval|{strategy}|{variant}|{train_sig}|{i}"
        preds.append(cache.get_or_generate(tok, model, key, task.task_id, "eval", variant, f"eval{i}", train_pairs, ex.inputs, max_new_tokens))
    targets = [ex.output for ex in eval_rows]
    row_acc, full = evaluate_outputs(preds, targets)
    return preds, row_acc, full


def run_task(
    tok: Any,
    model: Any,
    cache: GenerationCache,
    task: Task,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base, pool, eval_rows = split_task(task, args.base_examples, args.pool_examples, args.eval_examples)
    base_pairs = [(ex.inputs, ex.output) for ex in base]

    pool_probe_preds: Dict[int, List[str]] = {}
    for i, ex in enumerate(pool):
        vals: List[str] = []
        for variant in VARIANTS:
            key = f"{task.task_id}|probe|{variant}|base{args.base_examples}|pool{i}"
            vals.append(cache.get_or_generate(tok, model, key, task.task_id, "probe", variant, f"pool{i}", base_pairs, ex.inputs, args.max_new_tokens))
        pool_probe_preds[i] = vals

    active1 = select_active(pool_probe_preds, 1)
    active3 = select_active(pool_probe_preds, 3)
    random1 = select_random(task.task_id, len(pool), 1, args.seed)
    random3 = select_random(task.task_id, len(pool), 3, args.seed)
    diverse1 = select_diverse(base, pool, 1)
    diverse3 = select_diverse(base, pool, 3)
    order1 = [0]
    order3 = [0, 1, 2]

    methods: List[Tuple[str, List[int], bool, str]] = [
        ("base_plain", [], False, "plain"),
        ("order1_plain", order1, False, "plain"),
        ("random1_plain", random1, False, "plain"),
        ("active1_plain", active1, False, "plain"),
        ("diverse1_plain", diverse1, False, "plain"),
        ("order3_plain", order3, False, "plain"),
        ("random3_plain", random3, False, "plain"),
        ("active3_plain", active3, False, "plain"),
        ("diverse3_plain", diverse3, False, "plain"),
        ("active3_shuffled_labels", active3, True, "plain"),
    ]

    variant, loo_scores = loo_select_variant(tok, model, cache, task.task_id, base, args.max_new_tokens)
    methods.append(("loo_portfolio_base", [], False, variant))

    rows: List[Dict[str, Any]] = []
    predictions_by_method: Dict[str, List[str]] = {}
    for method, selected, shuffled, variant_name in methods:
        train_pairs = train_pairs_with_extra(base, pool, selected, shuffled=shuffled)
        preds, row_acc, full = evaluate_strategy(tok, model, cache, task, method, train_pairs, eval_rows, variant_name, args.max_new_tokens)
        predictions_by_method[method] = preds
        rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "features": ",".join(task.features),
                "method": method,
                "budget": len(selected),
                "variant": variant_name,
                "selected_pool_indices": json.dumps(selected),
                "shuffled_labels": shuffled,
                "row_exact": row_acc,
                "full_task_exact": full,
                "predictions_json": json.dumps(preds, ensure_ascii=False),
                "targets_json": json.dumps([ex.output for ex in eval_rows], ensure_ascii=False),
            }
        )

    # A hidden diagnostic: did any tested acquisition strategy contain the right
    # move? This is not deployable, but it separates strategy coverage from
    # strategy selection.
    candidates = [r for r in rows if r["method"] in {"order1_plain", "random1_plain", "active1_plain", "diverse1_plain", "order3_plain", "random3_plain", "active3_plain", "diverse3_plain"}]
    best = max(candidates, key=lambda r: (float(r["full_task_exact"]), float(r["row_exact"])))
    rows.append({**best, "method": "oracle_among_tested_acquisitions", "variant": "hidden_diagnostic"})

    acq = {
        "task_id": task.task_id,
        "family": task.family,
        "features": ",".join(task.features),
        "active1": json.dumps(active1),
        "active3": json.dumps(active3),
        "diverse1": json.dumps(diverse1),
        "diverse3": json.dumps(diverse3),
        "random1": json.dumps(random1),
        "random3": json.dumps(random3),
        "order1": json.dumps(order1),
        "order3": json.dumps(order3),
        "loo_variant": variant,
        "loo_plain": loo_scores["plain"],
        "loo_format": loo_scores["format"],
        "loo_consistency": loo_scores["consistency"],
        "pool_probe_entropy_json": json.dumps({i: entropy(v) for i, v in pool_probe_preds.items()}),
        "pool_probe_unique_json": json.dumps({i: len(set(v)) for i, v in pool_probe_preds.items()}),
    }
    return rows, acq


def summarize(details: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        details.groupby("method", as_index=False)
        .agg(tasks=("task_id", "count"), row_exact=("row_exact", "mean"), full_task_exact=("full_task_exact", "mean"), mean_budget=("budget", "mean"))
        .sort_values("full_task_exact", ascending=False)
    )
    family = (
        details.groupby(["method", "family"], as_index=False)
        .agg(tasks=("task_id", "count"), row_exact=("row_exact", "mean"), full_task_exact=("full_task_exact", "mean"))
        .sort_values(["method", "family"])
    )
    return summary, family


def pct(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{100 * float(x):.1f}%"


def md_table(df: pd.DataFrame, cols: Sequence[str], max_rows: int = 100) -> str:
    if df.empty:
        return "_No rows._"
    view = df[list(cols)].head(max_rows).copy()
    for c in view.columns:
        if view[c].dtype.kind in "fc":
            if "exact" in c or "rate" in c or c.startswith("loo_"):
                view[c] = view[c].map(pct)
            else:
                view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    header = "|" + "|".join(cols) + "|"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["|" + "|".join(html.escape(str(row[c])) for c in cols) + "|" for _, row in view.iterrows()]
    return "\n".join([header, sep] + body)


def plot_method(summary: pd.DataFrame) -> None:
    order = list(summary.sort_values("full_task_exact")["method"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(order, [float(summary[summary.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in order], color="#2563eb")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Strict full-task exact (%)")
    ax.set_title("Active example acquisition methods")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "method_full_task_exact.png", dpi=170)
    plt.close(fig)


def plot_row_vs_full(summary: pd.DataFrame) -> None:
    order = list(summary.sort_values("row_exact")["method"])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(x, [float(summary[summary.method.eq(m)]["row_exact"].iloc[0]) * 100 for m in order], marker="o", label="row exact", color="#059669")
    ax.plot(x, [float(summary[summary.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in order], marker="o", label="full-task exact", color="#dc2626")
    ax.set_xticks(x, order, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Row accuracy versus strict task consistency")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "row_vs_full_task.png", dpi=170)
    plt.close(fig)


def plot_budget(summary: pd.DataFrame) -> None:
    methods = ["base_plain", "order1_plain", "random1_plain", "active1_plain", "diverse1_plain", "order3_plain", "random3_plain", "active3_plain", "diverse3_plain"]
    data = summary[summary["method"].isin(methods)].copy()
    if data.empty:
        return
    labels = ["base", "order1", "random1", "active1", "diverse1", "order3", "random3", "active3", "diverse3"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    vals = [float(data[data.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in methods]
    ax.bar(labels, vals, color=["#6b7280", "#94a3b8", "#f59e0b", "#2563eb", "#7c3aed", "#94a3b8", "#f59e0b", "#2563eb", "#7c3aed"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Strict full-task exact (%)")
    ax.set_title("Effect of acquiring 1 or 3 extra examples")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "budget_comparison.png", dpi=170)
    plt.close(fig)


def plot_win_loss(details: pd.DataFrame) -> None:
    base = details[details["method"].eq("base_plain")][["task_id", "full_task_exact"]].rename(columns={"full_task_exact": "base"})
    rows = []
    for method in ["order1_plain", "random1_plain", "active1_plain", "diverse1_plain", "order3_plain", "random3_plain", "active3_plain", "diverse3_plain", "loo_portfolio_base"]:
        sub = details[details["method"].eq(method)][["task_id", "full_task_exact"]].rename(columns={"full_task_exact": "method"})
        if sub.empty:
            continue
        m = base.merge(sub, on="task_id")
        rows.append({"method": method, "helped": int((~m["base"].astype(bool) & m["method"].astype(bool)).sum()), "hurt": int((m["base"].astype(bool) & ~m["method"].astype(bool)).sum())})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["helped"], width=0.36, label="helped", color="#059669")
    ax.bar(x + 0.18, df["hurt"], width=0.36, label="hurt", color="#dc2626")
    ax.set_xticks(x, df["method"], rotation=30, ha="right")
    ax.set_ylabel("Tasks versus base")
    ax.set_title("Task-level wins and losses relative to 4 examples")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "wins_losses_vs_base.png", dpi=170)
    plt.close(fig)


def plot_family_heatmap(family: pd.DataFrame) -> None:
    keep = ["base_plain", "random3_plain", "active3_plain", "diverse3_plain", "loo_portfolio_base", "oracle_among_tested_acquisitions"]
    data = family[family["method"].isin(keep)].copy()
    if data.empty:
        return
    pivot = data.pivot_table(index="family", columns="method", values="full_task_exact", aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(pivot))))
    im = ax.imshow(pivot.values * 100, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j] * 100:.0f}", ha="center", va="center", fontsize=7)
    ax.set_title("Full-task exact by family")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(FIGURES / "family_heatmap.png", dpi=170)
    plt.close(fig)


def plot_acquisition_entropy(acq: pd.DataFrame) -> None:
    vals: List[float] = []
    for raw in acq["pool_probe_entropy_json"]:
        try:
            vals.extend(float(x) for x in json.loads(raw).values())
        except Exception:
            pass
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(vals, bins=14, color="#7c3aed")
    ax.set_xlabel("Probe-prediction entropy")
    ax.set_ylabel("Pool rows")
    ax.set_title("Disagreement signal used for active acquisition")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "probe_entropy_distribution.png", dpi=170)
    plt.close(fig)


def markdown_to_html(md: str) -> str:
    lines = md.splitlines()
    out: List[str] = []
    in_table = False
    for line in lines:
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt = line[2 : line.index("]")]
            src = line[line.index("(") + 1 : -1]
            out.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set(cells) == {"---"}:
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        else:
            if in_table:
                out.append("</table>")
                in_table = False
            if not line.strip():
                out.append("")
            else:
                escaped = html.escape(line)
                escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
                out.append(f"<p>{escaped}</p>")
    if in_table:
        out.append("</table>")
    css = """
    body { font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; margin: 40px; color: #111827; line-height: 1.5; }
    h1 { font-size: 32px; margin-bottom: 4px; }
    h2 { margin-top: 30px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
    table { border-collapse: collapse; width: 100%; margin: 14px 0 24px; font-size: 13px; }
    th, td { border: 1px solid #d1d5db; padding: 6px 8px; vertical-align: top; }
    th { background: #f3f4f6; text-align: left; }
    img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 4px; }
    figure { margin: 18px 0 26px; }
    figcaption { color: #4b5563; font-size: 13px; margin-top: 6px; }
    code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }
    """
    return "<!doctype html><html><head><meta charset='utf-8'><title>Active Example Acquisition</title><style>" + css + "</style></head><body>" + "\n".join(out) + "</body></html>"


def write_reports(run_name: str, config: Dict[str, Any], summary: pd.DataFrame, family: pd.DataFrame, details: pd.DataFrame, acq: pd.DataFrame) -> None:
    def m(method: str, col: str = "full_task_exact") -> float:
        hit = summary[summary["method"].eq(method)]
        return float(hit[col].iloc[0]) if not hit.empty else float("nan")

    base = m("base_plain")
    active1 = m("active1_plain")
    active3 = m("active3_plain")
    diverse3 = m("diverse3_plain")
    random3 = m("random3_plain")
    order3 = m("order3_plain")
    oracle = m("oracle_among_tested_acquisitions")
    shuffled = m("active3_shuffled_labels")
    portfolio = m("loo_portfolio_base")

    base_df = details[details["method"].eq("base_plain")][["task_id", "full_task_exact"]].rename(columns={"full_task_exact": "base"})
    active_df = details[details["method"].eq("active3_plain")][["task_id", "full_task_exact", "selected_pool_indices"]].rename(columns={"full_task_exact": "active3_exact", "selected_pool_indices": "active3_indices"})
    random_df = details[details["method"].eq("random3_plain")][["task_id", "full_task_exact", "selected_pool_indices"]].rename(columns={"full_task_exact": "random3_exact", "selected_pool_indices": "random3_indices"})
    comp = base_df.merge(active_df, on="task_id", how="left").merge(random_df, on="task_id", how="left")
    comp = comp.merge(acq[["task_id", "family", "features", "loo_variant"]], on="task_id", how="left")
    comp["active_helped"] = (~comp["base"].astype(bool)) & comp["active3_exact"].astype(bool)
    comp["active_hurt"] = comp["base"].astype(bool) & (~comp["active3_exact"].astype(bool))

    report = f"""# Active Example Acquisition

## Question

Can a small number of actively selected clarifying examples improve strict held-out text-transformation accuracy?

Each task starts with four visible examples. The selector may reveal one or three additional examples from a separate acquisition pool before the model answers three held-out rows. The held-out rows are never used for acquisition.

## Setup

- Run: `{run_name}`
- Dataset: public text-transformation tasks.
- Tasks: `{config.get('tasks')}`
- Visible examples per task: `{config.get('base_examples')}`
- Acquisition pool examples per task: `{config.get('pool_examples')}`
- Held-out evaluation rows per task: `{config.get('eval_examples')}`
- Probe variants for active disagreement: `{', '.join(VARIANTS)}`
- Generation records: `{config.get('generation_records')}`

## Main Result

{md_table(summary, ['method', 'tasks', 'mean_budget', 'row_exact', 'full_task_exact'])}

## Interpretation

The baseline with four examples solves `{base * 100:.1f}%` of tasks. Active acquisition solves `{active1 * 100:.1f}%` with one extra example and `{active3 * 100:.1f}%` with three extra examples. Input-diversity acquisition solves `{diverse3 * 100:.1f}%` with three extra examples. Random three-example acquisition solves `{random3 * 100:.1f}%`, order-three solves `{order3 * 100:.1f}%`, and the shuffled-label control solves `{shuffled * 100:.1f}%`.

The hidden diagnostic oracle over the tested acquisition policies reaches `{oracle * 100:.1f}%`, which measures whether any tested extra-example choice contained a better move. Visible-example portfolio selection reaches `{portfolio * 100:.1f}%`.

## Charts

![Full-task exact by method](../analysis/figures/method_full_task_exact.png)

![Row versus full-task exact](../analysis/figures/row_vs_full_task.png)

![Budget comparison](../analysis/figures/budget_comparison.png)

![Wins and losses versus base](../analysis/figures/wins_losses_vs_base.png)

![Probe entropy distribution](../analysis/figures/probe_entropy_distribution.png)

![Family heatmap](../analysis/figures/family_heatmap.png)

## Task-Level Active Versus Random

{md_table(comp.sort_values(['active_helped', 'active_hurt', 'task_id'], ascending=[False, True, True]), ['task_id', 'family', 'features', 'base', 'active3_exact', 'random3_exact', 'active_helped', 'active_hurt', 'active3_indices', 'random3_indices', 'loo_variant'], max_rows=120)}

## Acquisition Diagnostics

{md_table(acq.sort_values('task_id'), ['task_id', 'family', 'features', 'active1', 'active3', 'diverse1', 'diverse3', 'random1', 'random3', 'loo_variant', 'loo_plain', 'loo_format', 'loo_consistency'], max_rows=120)}

## Family Breakdown

{md_table(family, ['method', 'family', 'tasks', 'row_exact', 'full_task_exact'], max_rows=180)}

## Files

- `runs/{run_name}/generations.csv`
- `runs/{run_name}/method_details.csv`
- `runs/{run_name}/acquisition_details.csv`
- `runs/{run_name}/summary.csv`
- `analysis/summary.csv`
- `analysis/method_details.csv`
- `analysis/acquisition_details.csv`
- `analysis/family_summary.csv`
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "qwen_active_example_acquisition_report.md").write_text(report)
    (REPORTS / "qwen_active_example_acquisition_report.html").write_text(markdown_to_html(report))


def append_log(text: str) -> None:
    with (ROOT / "experiment_log.md").open("a") as f:
        f.write(text.rstrip() + "\n")


def write_artifacts(run_dir: Path, run_name: str, config: Dict[str, Any], details: pd.DataFrame, acq: pd.DataFrame, summary: pd.DataFrame, family: pd.DataFrame) -> None:
    details.to_csv(run_dir / "method_details.csv", index=False)
    acq.to_csv(run_dir / "acquisition_details.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    family.to_csv(run_dir / "family_summary.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    for name, df in [("method_details.csv", details), ("acquisition_details.csv", acq), ("summary.csv", summary), ("family_summary.csv", family)]:
        df.to_csv(ANALYSIS / name, index=False)
    plot_method(summary)
    plot_row_vs_full(summary)
    plot_budget(summary)
    plot_win_loss(details)
    plot_family_heatmap(family)
    plot_acquisition_entropy(acq)
    write_reports(run_name, config, summary, family, details, acq)


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    mirror_benchmark()
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    min_examples = args.base_examples + args.pool_examples + args.eval_examples
    tasks = choose_tasks(load_tasks(min_examples), args.task_limit, args.seed)
    tok, model = load_qwen()
    cache = GenerationCache(run_dir / "generations.csv")
    started = time.time()
    all_rows: List[Dict[str, Any]] = []
    acq_rows: List[Dict[str, Any]] = []
    try:
        for i, task in enumerate(tasks, start=1):
            rows, acq = run_task(tok, model, cache, task, args)
            all_rows.extend(rows)
            acq_rows.append(acq)
            if i == 1 or i % 5 == 0 or i == len(tasks):
                done_df = pd.DataFrame(all_rows)
                base = done_df[done_df.method.eq("base_plain")]["full_task_exact"].mean()
                active3 = done_df[done_df.method.eq("active3_plain")]["full_task_exact"].mean()
                print(f"task {i}/{len(tasks)} base={base:.3f} active3={active3:.3f}", flush=True)
    finally:
        cache.close()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    details = pd.DataFrame(all_rows)
    acq = pd.DataFrame(acq_rows)
    summary, family = summarize(details)
    config = {
        **vars(args),
        "tasks": len(tasks),
        "generation_records": len(pd.read_csv(run_dir / "generations.csv")) if (run_dir / "generations.csv").exists() else 0,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - started,
    }
    write_artifacts(run_dir, args.run_name, config, details, acq, summary, family)
    append_log(
        f"\n### Run `{args.run_name}`\n"
        f"- Tasks: {len(tasks)}\n"
        f"- Generation records: {config['generation_records']}\n"
        f"- `base_plain`: {float(summary[summary.method.eq('base_plain')]['full_task_exact'].iloc[0]) * 100:.1f}% full-task exact.\n"
        f"- `active1_plain`: {float(summary[summary.method.eq('active1_plain')]['full_task_exact'].iloc[0]) * 100:.1f}% full-task exact.\n"
        f"- `active3_plain`: {float(summary[summary.method.eq('active3_plain')]['full_task_exact'].iloc[0]) * 100:.1f}% full-task exact.\n"
        f"- `diverse3_plain`: {float(summary[summary.method.eq('diverse3_plain')]['full_task_exact'].iloc[0]) * 100:.1f}% full-task exact.\n"
        f"- `random3_plain`: {float(summary[summary.method.eq('random3_plain')]['full_task_exact'].iloc[0]) * 100:.1f}% full-task exact.\n"
        f"- `active3_shuffled_labels`: {float(summary[summary.method.eq('active3_shuffled_labels')]['full_task_exact'].iloc[0]) * 100:.1f}% full-task exact.\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="main_v1")
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--task_limit", type=int, default=30)
    p.add_argument("--base_examples", type=int, default=4)
    p.add_argument("--pool_examples", type=int, default=5)
    p.add_argument("--eval_examples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
