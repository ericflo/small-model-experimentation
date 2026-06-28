#!/usr/bin/env python3
"""Oracle-distilled active acquisition policy for text transformations.

For each task, the runner measures the downstream utility of revealing each
candidate acquisition row. It then trains a cross-validated scorer to predict
that utility from task-local row features and uses the scorer to choose rows on
held-out tasks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path("/workspace/experiments/qwen_oracle_distilled_acquisition_policy")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_oracle_distilled_acquisition_policy")
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
    chosen = rng.sample(list(tasks), limit) if limit and limit < len(tasks) else list(tasks)
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


def clean_prediction(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    if text.lower().startswith("output:"):
        text = text.split(":", 1)[1].strip()
    first = text.splitlines()[0].strip() if text else ""
    if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
        first = first[1:-1].strip()
    return first


def row_prompt(train_pairs: Sequence[Tuple[Tuple[str, ...], str]], query: Tuple[str, ...]) -> str:
    lines = [
        "Infer the exact text transformation from the examples.",
        "Return only the transformed output for the query. Do not explain.",
        "",
        "Examples:",
    ]
    for inp, out in train_pairs:
        lines.append(f"Input: {render_inputs(inp)}")
        lines.append(f"Output: {out}")
    lines.extend(["", "Query:", f"Input: {render_inputs(query)}", "Output:"])
    return "\n".join(lines)


def selection_prompt(visible: Sequence[Example], pool: Sequence[Example], remaining: Sequence[int], budget_left: int) -> str:
    lines = [
        "A hidden deterministic text transformation is defined by the visible examples.",
        "You may reveal the output for one unlabeled candidate input.",
        "Choose the candidate whose label would be most useful for inferring the transformation.",
        "Return only the integer index. Do not explain.",
        "",
        "Visible examples:",
    ]
    for ex in visible:
        lines.append(f"Input: {render_inputs(ex.inputs)}")
        lines.append(f"Output: {ex.output}")
    lines.extend(["", f"Selections remaining after this one: {max(0, budget_left - 1)}", "Unlabeled candidates:"])
    for idx in remaining:
        lines.append(f"[{idx}] {render_inputs(pool[idx].inputs)}")
    lines.extend(["", "Chosen index:"])
    return "\n".join(lines)


def parse_choice(text: str, remaining: Sequence[int]) -> Optional[int]:
    allowed = set(remaining)
    for match in re.findall(r"-?\d+", text):
        val = int(match)
        if val in allowed:
            return val
    return None


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

    def get_text(
        self,
        tok: Any,
        model: Any,
        key: str,
        task_id: str,
        phase: str,
        method: str,
        row_id: str,
        system: str,
        user: str,
        max_new_tokens: int,
    ) -> Tuple[str, str]:
        if key in self.rows:
            return self.rows[key]["prediction"], self.rows[key]["raw"]
        raw = generate_text(tok, model, system, user, max_new_tokens)
        pred = clean_prediction(raw)
        row = {"key": key, "task_id": task_id, "phase": phase, "method": method, "row_id": row_id, "prediction": pred, "raw": raw}
        self.rows[key] = row
        self.writer.writerow(row)
        self.file.flush()
        return pred, raw

    def answer(
        self,
        tok: Any,
        model: Any,
        key: str,
        task_id: str,
        method: str,
        row_id: str,
        train_pairs: Sequence[Tuple[Tuple[str, ...], str]],
        query: Tuple[str, ...],
        max_new_tokens: int,
    ) -> str:
        pred, _ = self.get_text(
            tok,
            model,
            key,
            task_id,
            "answer",
            method,
            row_id,
            "You transform text exactly from examples.",
            row_prompt(train_pairs, query),
            max_new_tokens,
        )
        return pred


def eval_outputs(preds: Sequence[str], targets: Sequence[str]) -> Tuple[float, bool]:
    if not targets:
        return 0.0, False
    row = sum(norm_eq(a, b) for a, b in zip(preds, targets)) / len(targets)
    return row, row == 1.0


def train_pairs(base: Sequence[Example], pool: Sequence[Example], selected: Sequence[int], shuffled: bool = False) -> List[Tuple[Tuple[str, ...], str]]:
    pairs = [(ex.inputs, ex.output) for ex in base]
    labels = [pool[i].output for i in selected]
    if shuffled and len(labels) > 1:
        labels = labels[1:] + labels[:1]
    for idx, label in zip(selected, labels):
        pairs.append((pool[idx].inputs, label))
    return pairs


def train_sig(pairs: Sequence[Tuple[Tuple[str, ...], str]]) -> str:
    payload = json.dumps([(list(a), b) for a, b in pairs], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:14]


def evaluate_method(
    tok: Any,
    model: Any,
    cache: GenerationCache,
    task: Task,
    method: str,
    pairs: Sequence[Tuple[Tuple[str, ...], str]],
    eval_rows: Sequence[Example],
    max_new_tokens: int,
) -> Tuple[List[str], float, bool]:
    preds = []
    sig = train_sig(pairs)
    for i, ex in enumerate(eval_rows):
        key = f"{task.task_id}|{method}|{sig}|eval{i}"
        preds.append(cache.answer(tok, model, key, task.task_id, method, f"eval{i}", pairs, ex.inputs, max_new_tokens))
    targets = [ex.output for ex in eval_rows]
    row, full = eval_outputs(preds, targets)
    return preds, row, full


def numeric_stats(text: str) -> Dict[str, float]:
    length = max(1, len(text))
    digits = sum(ch.isdigit() for ch in text)
    alpha = sum(ch.isalpha() for ch in text)
    spaces = sum(ch.isspace() for ch in text)
    punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return {
        "len": float(len(text)),
        "digit_frac": digits / length,
        "alpha_frac": alpha / length,
        "space_frac": spaces / length,
        "punct_frac": punct / length,
        "token_count": float(len(re.findall(r"[A-Za-z0-9]+", text))),
        "num_count": float(len(nums)),
        "has_date_sep": float(any(x in text for x in ["/", "-", ":"])),
        "has_currency": float(any(x in text for x in ["$", "€", "£", "₹"])),
    }


def vector_stats(row: Tuple[str, ...]) -> np.ndarray:
    s = numeric_stats(render_inputs(row))
    return np.array([s["len"], s["digit_frac"], s["alpha_frac"], s["space_frac"], s["punct_frac"], s["token_count"], s["num_count"]], dtype=float)


def row_distance(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    va, vb = vector_stats(a), vector_stats(b)
    denom = np.maximum(np.maximum(np.abs(va), np.abs(vb)), 1.0)
    return float(np.mean(np.abs(va - vb) / denom))


def candidate_features(task: Task, base: Sequence[Example], pool: Sequence[Example], idx: int) -> Dict[str, Any]:
    text = render_inputs(pool[idx].inputs)
    stats = numeric_stats(text)
    d_base = [row_distance(pool[idx].inputs, ex.inputs) for ex in base]
    d_pool = [row_distance(pool[idx].inputs, ex.inputs) for j, ex in enumerate(pool) if j != idx]
    feats: Dict[str, Any] = {
        "task_id": task.task_id,
        "family": task.family,
        "candidate_idx": idx,
        "candidate_idx_norm": idx / max(1, len(pool) - 1),
        "base_min_dist": min(d_base) if d_base else 0.0,
        "base_mean_dist": float(np.mean(d_base)) if d_base else 0.0,
        "pool_mean_dist": float(np.mean(d_pool)) if d_pool else 0.0,
    }
    for k, v in stats.items():
        feats[f"row_{k}"] = v
    for feature in task.features:
        feats[f"feature={feature}"] = 1.0
    return feats


def select_random(task_id: str, pool_size: int, budget: int, seed: int) -> List[int]:
    rng = random.Random(seed + stable_int(task_id))
    return sorted(rng.sample(list(range(pool_size)), min(pool_size, budget)))


def select_diverse(base: Sequence[Example], pool: Sequence[Example], budget: int) -> List[int]:
    selected: List[int] = []
    anchors = [ex.inputs for ex in base]
    remaining = set(range(len(pool)))
    while remaining and len(selected) < budget:
        best = max(remaining, key=lambda i: (min(row_distance(pool[i].inputs, a) for a in anchors), -i))
        selected.append(best)
        remaining.remove(best)
        anchors.append(pool[best].inputs)
    return selected


def select_qwen(
    tok: Any,
    model: Any,
    cache: GenerationCache,
    task: Task,
    base: Sequence[Example],
    pool: Sequence[Example],
    budget: int,
    max_new_tokens: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    visible = list(base)
    remaining = list(range(len(pool)))
    selected: List[int] = []
    trace: List[Dict[str, Any]] = []
    for step in range(min(budget, len(pool))):
        prompt = selection_prompt(visible, pool, remaining, budget - step)
        sig = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:14]
        key = f"{task.task_id}|qwen_select|budget{budget}|step{step}|{sig}"
        pred, raw = cache.get_text(
            tok,
            model,
            key,
            task.task_id,
            "select",
            f"qwen_choose{budget}",
            f"step{step}",
            "You choose informative examples for text-transformation tasks.",
            prompt,
            max_new_tokens,
        )
        choice = parse_choice(raw or pred, remaining)
        fallback = False
        if choice is None:
            choice = remaining[0]
            fallback = True
        selected.append(choice)
        remaining.remove(choice)
        visible.append(pool[choice])
        trace.append({"step": step, "choice": choice, "fallback": fallback, "raw": raw})
    return selected, trace


def feature_matrix(candidate_df: pd.DataFrame) -> pd.DataFrame:
    drop = {
        "task_id",
        "family",
        "candidate_idx",
        "utility",
        "utility_row_exact",
        "utility_full_exact",
        "selected_by_learned1",
        "predicted_utility",
    }
    cols = [c for c in candidate_df.columns if c not in drop and not c.startswith("pred_")]
    numeric = candidate_df[cols].copy()
    family = pd.get_dummies(candidate_df["family"], prefix="family", dtype=float)
    return pd.concat([numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0), family], axis=1)


def cross_validated_predictions(candidate_df: pd.DataFrame, folds: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = candidate_df.copy()
    out["predicted_utility"] = np.nan
    tasks = np.array(sorted(out["task_id"].unique()))
    n_splits = max(2, min(folds, len(tasks)))
    fold_rows = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_all = feature_matrix(out)
    y_all = out["utility"].astype(float).to_numpy()
    for fold, (train_idx, test_idx) in enumerate(kf.split(tasks)):
        train_tasks = set(tasks[train_idx])
        test_tasks = set(tasks[test_idx])
        train_mask = out["task_id"].isin(train_tasks).to_numpy()
        test_mask = out["task_id"].isin(test_tasks).to_numpy()
        model = RandomForestRegressor(n_estimators=240, min_samples_leaf=2, random_state=seed + fold, n_jobs=-1)
        model.fit(X_all.loc[train_mask], y_all[train_mask])
        pred = model.predict(X_all.loc[test_mask])
        out.loc[test_mask, "predicted_utility"] = pred
        y_test = y_all[test_mask]
        try:
            auc = roc_auc_score((y_test > 0).astype(int), pred) if len(set((y_test > 0).astype(int))) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        fold_rows.append(
            {
                "fold": fold,
                "train_tasks": len(train_tasks),
                "test_tasks": len(test_tasks),
                "candidate_auc": auc,
                "mean_true_utility": float(np.mean(y_test)),
                "mean_predicted_utility": float(np.mean(pred)),
            }
        )
    return out, pd.DataFrame(fold_rows)


def summarize(details: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        details.groupby("method", as_index=False)
        .agg(tasks=("task_id", "count"), mean_budget=("budget", "mean"), row_exact=("row_exact", "mean"), full_task_exact=("full_task_exact", "mean"))
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


def md_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None, max_rows: int = 100) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    if cols is not None:
        view = view[list(cols)]
    view = view.head(max_rows).copy()
    for c in view.columns:
        if view[c].dtype.kind in "fc":
            if "exact" in c or "auc" in c or "utility" in c:
                view[c] = view[c].map(pct)
            else:
                view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
    header = "|" + "|".join(view.columns) + "|"
    sep = "|" + "|".join(["---"] * len(view.columns)) + "|"
    rows = ["|" + "|".join(html.escape(str(row[c])) for c in view.columns) + "|" for _, row in view.iterrows()]
    return "\n".join([header, sep] + rows)


def plot_summary(summary: pd.DataFrame) -> None:
    order = list(summary.sort_values("full_task_exact")["method"])
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#2563eb" if "learned" in m else "#6b7280" for m in order]
    ax.barh(order, [float(summary[summary.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in order], color=colors)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Strict full-task exact (%)")
    ax.set_title("Acquisition policy comparison")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "method_full_task_exact.png", dpi=170)
    plt.close(fig)


def plot_budget(summary: pd.DataFrame) -> None:
    methods = [m for m in summary["method"] if re.search(r"(order|random|diverse|qwen_choose|learned)\d+_plain", str(m)) or m == "base_plain"]
    methods = sorted(methods, key=lambda m: (0 if m == "base_plain" else int(re.findall(r"\d+", str(m))[0]), str(m)))
    fig, ax = plt.subplots(figsize=(12, 5))
    vals = [float(summary[summary.method.eq(m)]["full_task_exact"].iloc[0]) * 100 for m in methods]
    colors = []
    for m in methods:
        if "learned" in m:
            colors.append("#2563eb")
        elif "qwen" in m:
            colors.append("#9333ea")
        elif "random" in m:
            colors.append("#f59e0b")
        elif "order" in m:
            colors.append("#64748b")
        elif "diverse" in m:
            colors.append("#0d9488")
        else:
            colors.append("#111827")
    ax.bar([m.replace("_plain", "") for m in methods], vals, color=colors)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Strict full-task exact (%)")
    ax.set_title("Accuracy by acquisition budget")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "budget_comparison.png", dpi=170)
    plt.close(fig)


def plot_utility(candidate_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(candidate_df["utility"], candidate_df["predicted_utility"], alpha=0.65, s=30, color="#2563eb")
    ax.set_xlabel("Measured utility")
    ax.set_ylabel("Cross-validated predicted utility")
    ax.set_title("Candidate utility prediction")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "utility_prediction_scatter.png", dpi=170)
    plt.close(fig)


def plot_wins(details: pd.DataFrame, policy_budget: int) -> None:
    base = details[details.method.eq("base_plain")][["task_id", "full_task_exact"]].rename(columns={"full_task_exact": "base"})
    rows = []
    for method in [
        "order1_plain",
        "random1_plain",
        "qwen_choose1_plain",
        "learned1_plain",
        f"order{policy_budget}_plain",
        f"random{policy_budget}_plain",
        f"qwen_choose{policy_budget}_plain",
        f"learned{policy_budget}_plain",
    ]:
        sub = details[details.method.eq(method)][["task_id", "full_task_exact"]].rename(columns={"full_task_exact": "method"})
        if sub.empty:
            continue
        merged = base.merge(sub, on="task_id")
        rows.append({"method": method, "helped": int((~merged.base.astype(bool) & merged.method.astype(bool)).sum()), "hurt": int((merged.base.astype(bool) & ~merged.method.astype(bool)).sum())})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["helped"], width=0.36, color="#059669", label="helped")
    ax.bar(x + 0.18, df["hurt"], width=0.36, color="#dc2626", label="hurt")
    ax.set_xticks(x, df["method"], rotation=35, ha="right")
    ax.set_ylabel("Tasks versus base")
    ax.set_title("Task-level wins and losses")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "wins_losses_vs_base.png", dpi=170)
    plt.close(fig)


def plot_family(family: pd.DataFrame) -> None:
    learned = [m for m in family.method.unique() if str(m).startswith("learned") and str(m).endswith("_plain")]
    budget = max([int(re.findall(r"\d+", str(m))[0]) for m in learned if re.findall(r"\d+", str(m))] or [4])
    keep = ["base_plain", f"order{budget}_plain", f"random{budget}_plain", f"qwen_choose{budget}_plain", f"learned{budget}_plain", "oracle_single"]
    data = family[family.method.isin(keep)]
    pivot = data.pivot_table(index="family", columns="method", values="full_task_exact", aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(pivot))))
    im = ax.imshow(pivot.values * 100, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j] * 100:.0f}", ha="center", va="center", fontsize=7)
    ax.set_title("Full-task exact by family")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(FIGURES / "family_heatmap.png", dpi=170)
    plt.close(fig)


def markdown_to_html(md: str) -> str:
    try:
        import markdown  # type: ignore

        body = markdown.markdown(md, extensions=["tables"])
    except Exception:
        body = "<pre>" + html.escape(md) + "</pre>"
    css = """
    body { font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; margin: 40px; color: #111827; line-height: 1.5; }
    h1 { font-size: 32px; }
    h2 { margin-top: 30px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }
    table { border-collapse: collapse; width: 100%; margin: 14px 0 24px; font-size: 13px; }
    th, td { border: 1px solid #d1d5db; padding: 6px 8px; vertical-align: top; }
    th { background: #f3f4f6; text-align: left; }
    img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 4px; }
    code { background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }
    """
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Oracle-Distilled Acquisition Policy</title><style>{css}</style></head><body>{body}</body></html>"


def write_report(
    run_name: str,
    config: Dict[str, Any],
    summary: pd.DataFrame,
    family: pd.DataFrame,
    details: pd.DataFrame,
    candidate_df: pd.DataFrame,
    fold_df: pd.DataFrame,
) -> None:
    def metric(method: str) -> float:
        row = summary[summary.method.eq(method)]
        return float(row["full_task_exact"].iloc[0]) if not row.empty else float("nan")

    base = metric("base_plain")
    learned1 = metric("learned1_plain")
    learned_budget_methods = [m for m in summary.method if str(m).startswith("learned") and str(m).endswith("_plain")]
    policy_budget = max([int(re.findall(r"\d+", str(m))[0]) for m in learned_budget_methods if re.findall(r"\d+", str(m))] or [4])
    learnedN = metric(f"learned{policy_budget}_plain")
    qwenN = metric(f"qwen_choose{policy_budget}_plain")
    orderN = metric(f"order{policy_budget}_plain")
    randomN = metric(f"random{policy_budget}_plain")
    diverseN = metric(f"diverse{policy_budget}_plain")
    oracle = metric("oracle_single")
    shuffled = metric(f"learned{policy_budget}_shuffled_labels")
    base_df = details[details.method.eq("base_plain")][["task_id", "full_task_exact"]].rename(columns={"full_task_exact": "base"})
    learned_df = details[details.method.eq(f"learned{policy_budget}_plain")][["task_id", "full_task_exact", "selected_indices"]].rename(columns={"full_task_exact": "learned_budget", "selected_indices": "learned_indices"})
    order_df = details[details.method.eq(f"order{policy_budget}_plain")][["task_id", "full_task_exact"]].rename(columns={"full_task_exact": "order_budget"})
    comp = base_df.merge(learned_df, on="task_id", how="left").merge(order_df, on="task_id", how="left")
    comp["learned_helped"] = (~comp.base.astype(bool)) & comp.learned_budget.astype(bool)
    comp["learned_hurt"] = comp.base.astype(bool) & (~comp.learned_budget.astype(bool))
    task_candidates = candidate_df.sort_values(["task_id", "predicted_utility"], ascending=[True, False]).groupby("task_id").head(6)
    learned_wins = int(comp["learned_helped"].sum())
    learned_losses = int(comp["learned_hurt"].sum())
    competitor_scores = {
        f"fixed-order {policy_budget}-label": orderN,
        f"random {policy_budget}-label": randomN,
        f"diverse {policy_budget}-label": diverseN,
        f"Qwen-chosen {policy_budget}-label": qwenN,
    }
    best_competitor, best_competitor_score = max(
        competitor_scores.items(),
        key=lambda kv: -1.0 if math.isnan(kv[1]) else kv[1],
    )
    if learnedN > best_competitor_score and learnedN > base and learnedN > shuffled:
        verdict_text = (
            f"Positive: learned acquisition is the best deployable policy in this run. "
            f"It beats the strongest non-oracle acquisition baseline ({best_competitor}, {pct(best_competitor_score)}) "
            f"and separates from the shuffled-label control ({pct(shuffled)})."
        )
    elif learnedN > base and learnedN > shuffled:
        verdict_text = (
            f"Mixed negative: learned acquisition uses real signal but is not the best policy. "
            f"It improves over no acquisition ({pct(base)} -> {pct(learnedN)}) and separates from shuffled labels ({pct(shuffled)}), "
            f"but loses to the strongest non-oracle baseline ({best_competitor}, {pct(best_competitor_score)}). "
            f"It helps {learned_wins} tasks and hurts {learned_losses} tasks versus the base prompt."
        )
    else:
        verdict_text = (
            f"Negative: learned acquisition does not beat the no-acquisition baseline ({pct(base)} -> {pct(learnedN)}). "
            f"The best non-oracle acquisition baseline is {best_competitor} at {pct(best_competitor_score)}, "
            f"while the shuffled-label control lands at {pct(shuffled)}. "
            f"Learned acquisition helps {learned_wins} tasks and hurts {learned_losses} tasks versus the base prompt."
        )
    report = f"""# Oracle-Distilled Acquisition Policy

## Question

Can a supervised acquisition-row scorer choose more useful clarifying examples than fixed, random, or model-chosen acquisition policies?

Each task starts with a small visible set. For every candidate acquisition row, the experiment measures downstream utility by revealing that row's true output and then scoring held-out answers. A cross-validated scorer is trained on other tasks to predict candidate utility for a held-out task.

## Setup

- Run: `{run_name}`
- Dataset: public text-transformation tasks.
- Tasks: `{config.get('tasks')}`
- Visible examples per task: `{config.get('base_examples')}`
- Acquisition pool examples per task: `{config.get('pool_examples')}`
- Held-out evaluation rows per task: `{config.get('eval_examples')}`
- Cross-validation folds: `{config.get('folds')}`
- Generation records: `{config.get('generation_records')}`

## Main Result

{md_table(summary, max_rows=80)}

## Interpretation

The no-acquisition baseline solves `{pct(base)}` of tasks. The learned scorer solves `{pct(learned1)}` with one acquired label and `{pct(learnedN)}` with `{policy_budget}` acquired labels. At the same `{policy_budget}`-label budget, fixed-order acquisition solves `{pct(orderN)}`, random acquisition solves `{pct(randomN)}`, and Qwen-chosen acquisition solves `{pct(qwenN)}`.

The shuffled-label control for learned `{policy_budget}`-label acquisition solves `{pct(shuffled)}`. The hidden single-acquisition oracle solves `{pct(oracle)}`, measuring how much headroom exists if the best single row is known.

## Verdict

{verdict_text}

The strict win condition is learned acquisition beating fixed-order, random, diverse, and Qwen-chosen acquisition at the same label budget while separating from the shuffled-label control. This run should therefore be read by comparing learned acquisition to the best non-oracle baseline, not only to the no-acquisition prompt.

## Charts

![Full-task exact by method](../analysis/figures/method_full_task_exact.png)

![Budget comparison](../analysis/figures/budget_comparison.png)

![Utility prediction scatter](../analysis/figures/utility_prediction_scatter.png)

![Wins and losses](../analysis/figures/wins_losses_vs_base.png)

![Family heatmap](../analysis/figures/family_heatmap.png)

## Fold Diagnostics

{md_table(fold_df, max_rows=20)}

## Candidate Utility Examples

{md_table(task_candidates[['task_id', 'family', 'candidate_idx', 'utility', 'utility_row_exact', 'utility_full_exact', 'predicted_utility']], max_rows=120)}

## Task-Level Learned Versus Base

{md_table(comp.sort_values(['learned_helped', 'learned_hurt', 'task_id'], ascending=[False, True, True]), max_rows=120)}

## Family Breakdown

{md_table(family, max_rows=220)}

## Files

- `runs/{run_name}/generations.csv`
- `runs/{run_name}/candidate_utilities.csv`
- `runs/{run_name}/method_details.csv`
- `runs/{run_name}/fold_diagnostics.csv`
- `analysis/*.csv`
- `analysis/figures/*.png`
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "qwen_oracle_distilled_acquisition_policy_report.md").write_text(report)
    (REPORTS / "qwen_oracle_distilled_acquisition_policy_report.html").write_text(markdown_to_html(report))


def append_log(text: str) -> None:
    with (ROOT / "experiment_log.md").open("a") as f:
        f.write(text.rstrip() + "\n")


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    mirror_benchmark()
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    min_examples = args.base_examples + args.pool_examples + args.eval_examples
    tasks = choose_tasks(load_tasks(min_examples), args.task_limit, args.seed)
    tok, model = (None, None) if args.no_qwen else load_qwen()
    cache = GenerationCache(run_dir / "generations.csv")
    started = time.time()
    task_info: Dict[str, Dict[str, Any]] = {}
    candidate_rows: List[Dict[str, Any]] = []
    method_rows: List[Dict[str, Any]] = []
    qwen_choices: Dict[Tuple[str, int], List[int]] = {}
    try:
        for n, task in enumerate(tasks, start=1):
            base, pool, eval_rows = split_task(task, args.base_examples, args.pool_examples, args.eval_examples)
            base_pairs = train_pairs(base, pool, [])
            base_preds, base_row, base_full = evaluate_method(tok, model, cache, task, "base_plain", base_pairs, eval_rows, args.max_new_tokens)
            task_info[task.task_id] = {"task": task, "base": base, "pool": pool, "eval": eval_rows, "base_row": base_row, "base_full": base_full}
            method_rows.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "method": "base_plain",
                    "budget": 0,
                    "selected_indices": "[]",
                    "row_exact": base_row,
                    "full_task_exact": base_full,
                    "predictions_json": json.dumps(base_preds, ensure_ascii=False),
                }
            )
            for idx in range(len(pool)):
                pairs = train_pairs(base, pool, [idx])
                preds, row, full = evaluate_method(tok, model, cache, task, f"single_{idx}", pairs, eval_rows, args.max_new_tokens)
                feats = candidate_features(task, base, pool, idx)
                utility = (row - base_row) + (1.0 if full and not base_full else 0.0) - (1.0 if base_full and not full else 0.0)
                candidate_rows.append(
                    {
                        **feats,
                        "utility": utility,
                        "utility_row_exact": row,
                        "utility_full_exact": full,
                        "base_row_exact": base_row,
                        "base_full_exact": base_full,
                        "predictions_json": json.dumps(preds, ensure_ascii=False),
                    }
                )
            for budget in [1, args.policy_budget]:
                chosen, trace = select_qwen(tok, model, cache, task, base, pool, budget, args.select_max_new_tokens)
                qwen_choices[(task.task_id, budget)] = chosen
                method_rows.append({"task_id": task.task_id, "family": task.family, "method": f"qwen_choose{budget}_trace", "budget": budget, "selected_indices": json.dumps(chosen), "row_exact": np.nan, "full_task_exact": False, "predictions_json": json.dumps(trace, ensure_ascii=False)})
            if n == 1 or n % 5 == 0 or n == len(tasks):
                print(f"labeled utilities {n}/{len(tasks)}", flush=True)
    finally:
        cache.file.flush()

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df, fold_df = cross_validated_predictions(candidate_df, args.folds, args.seed)

    def candidate_lookup(task_id: str, idx: int) -> Dict[str, Any]:
        row = candidate_df[(candidate_df.task_id.eq(task_id)) & (candidate_df.candidate_idx.eq(idx))]
        return row.iloc[0].to_dict()

    # Convert utility predictions into deployable selections and evaluate the multi-label policies.
    for task_id, info in task_info.items():
        task: Task = info["task"]
        base: List[Example] = info["base"]
        pool: List[Example] = info["pool"]
        eval_rows: List[Example] = info["eval"]
        sub = candidate_df[candidate_df.task_id.eq(task_id)].sort_values("predicted_utility", ascending=False)
        learned1 = [int(sub.iloc[0]["candidate_idx"])]
        learned4 = [int(x) for x in sub.head(min(args.policy_budget, len(pool)))["candidate_idx"].tolist()]
        order1 = [0]
        order4 = list(range(min(args.policy_budget, len(pool))))
        random1 = select_random(task_id, len(pool), 1, args.seed)
        random4 = select_random(task_id, len(pool), args.policy_budget, args.seed)
        diverse1 = select_diverse(base, pool, 1)
        diverse4 = select_diverse(base, pool, args.policy_budget)
        qwen1 = qwen_choices.get((task_id, 1), [])
        qwen4 = qwen_choices.get((task_id, args.policy_budget), [])
        oracle_single_idx = int(candidate_df[candidate_df.task_id.eq(task_id)].sort_values(["utility", "utility_row_exact"], ascending=False).iloc[0]["candidate_idx"])

        single_methods = {
            "learned1_plain": learned1,
            "order1_plain": order1,
            "random1_plain": random1,
            "diverse1_plain": diverse1,
            "qwen_choose1_plain": qwen1,
            "oracle_single": [oracle_single_idx],
        }
        for method, selected in single_methods.items():
            if selected:
                rec = candidate_lookup(task_id, selected[0])
                method_rows.append(
                    {
                        "task_id": task_id,
                        "family": task.family,
                        "method": method,
                        "budget": len(selected),
                        "selected_indices": json.dumps(selected),
                        "row_exact": rec["utility_row_exact"],
                        "full_task_exact": bool(rec["utility_full_exact"]),
                        "predictions_json": rec["predictions_json"],
                    }
                )

        multi_methods = {
            f"learned{args.policy_budget}_plain": (learned4, False),
            f"learned{args.policy_budget}_shuffled_labels": (learned4, True),
            f"order{args.policy_budget}_plain": (order4, False),
            f"random{args.policy_budget}_plain": (random4, False),
            f"diverse{args.policy_budget}_plain": (diverse4, False),
            f"qwen_choose{args.policy_budget}_plain": (qwen4, False),
        }
        for method, (selected, shuffled) in multi_methods.items():
            pairs = train_pairs(base, pool, selected, shuffled=shuffled)
            preds, row, full = evaluate_method(tok, model, cache, task, method, pairs, eval_rows, args.max_new_tokens)
            method_rows.append(
                {
                    "task_id": task_id,
                    "family": task.family,
                    "method": method,
                    "budget": len(selected),
                    "selected_indices": json.dumps(selected),
                    "row_exact": row,
                    "full_task_exact": full,
                    "predictions_json": json.dumps(preds, ensure_ascii=False),
                }
            )

    cache.close()
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    method_df = pd.DataFrame([r for r in method_rows if not str(r["method"]).endswith("_trace")])
    trace_df = pd.DataFrame([r for r in method_rows if str(r["method"]).endswith("_trace")])
    summary, family = summarize(method_df)
    config = {
        **vars(args),
        "tasks": len(tasks),
        "generation_records": len(pd.read_csv(run_dir / "generations.csv")) if (run_dir / "generations.csv").exists() else 0,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - started,
    }
    for name, df in [
        ("candidate_utilities.csv", candidate_df),
        ("method_details.csv", method_df),
        ("qwen_selection_traces.csv", trace_df),
        ("summary.csv", summary),
        ("family_summary.csv", family),
        ("fold_diagnostics.csv", fold_df),
    ]:
        df.to_csv(run_dir / name, index=False)
        df.to_csv(ANALYSIS / name, index=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    plot_summary(summary)
    plot_budget(summary)
    plot_utility(candidate_df)
    plot_wins(method_df, args.policy_budget)
    plot_family(family)
    write_report(args.run_name, config, summary, family, method_df, candidate_df, fold_df)

    def m(method: str) -> float:
        row = summary[summary.method.eq(method)]
        return float(row.full_task_exact.iloc[0]) if not row.empty else float("nan")

    append_log(
        f"\n### Run `{args.run_name}`\n"
        f"- Tasks: {len(tasks)}\n"
        f"- Generation records: {config['generation_records']}\n"
        f"- `base_plain`: {m('base_plain') * 100:.1f}% full-task exact.\n"
        f"- `learned1_plain`: {m('learned1_plain') * 100:.1f}% full-task exact.\n"
        f"- `learned{args.policy_budget}_plain`: {m(f'learned{args.policy_budget}_plain') * 100:.1f}% full-task exact.\n"
        f"- `order{args.policy_budget}_plain`: {m(f'order{args.policy_budget}_plain') * 100:.1f}% full-task exact.\n"
        f"- `random{args.policy_budget}_plain`: {m(f'random{args.policy_budget}_plain') * 100:.1f}% full-task exact.\n"
        f"- `qwen_choose{args.policy_budget}_plain`: {m(f'qwen_choose{args.policy_budget}_plain') * 100:.1f}% full-task exact.\n"
        f"- `learned{args.policy_budget}_shuffled_labels`: {m(f'learned{args.policy_budget}_shuffled_labels') * 100:.1f}% full-task exact.\n"
        f"- `oracle_single`: {m('oracle_single') * 100:.1f}% full-task exact.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="main_v1")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--task_limit", type=int, default=30)
    parser.add_argument("--base_examples", type=int, default=2)
    parser.add_argument("--pool_examples", type=int, default=6)
    parser.add_argument("--eval_examples", type=int, default=3)
    parser.add_argument("--policy_budget", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=56)
    parser.add_argument("--select_max_new_tokens", type=int, default=16)
    parser.add_argument("--no_qwen", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
