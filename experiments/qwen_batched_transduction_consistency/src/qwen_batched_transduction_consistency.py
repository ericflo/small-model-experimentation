#!/usr/bin/env python3
"""Batched transduction consistency experiment.

This standalone experiment compares row-by-row inference with batched
transduction on public text-transformation tasks. The primary metric is strict
full-task exact: every held-out query row for a task must be exact.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import html
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path("/workspace/experiments/qwen_batched_transduction_consistency")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_batched_transduction_consistency")
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


def choose_tasks(tasks: List[Task], limit: int, seed: int, train_n: int, heldout_cap: int, min_heldout: int) -> List[Task]:
    eligible = []
    for t in tasks:
        _, test = split_examples(t, train_n, heldout_cap)
        if len(test) >= min_heldout:
            eligible.append(t)
    rng = random.Random(seed)
    if limit and limit < len(eligible):
        chosen = rng.sample(eligible, limit)
    else:
        chosen = eligible
    return sorted(chosen, key=lambda t: t.task_id)


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
        "Infer the text transformation from the examples.",
        "Return only the transformed output for the query. Do not explain.",
        "",
        "Examples:",
    ]
    for inp, out in train_pairs:
        lines.append(f"Input: {render_inputs(inp)}")
        lines.append(f"Output: {out}")
    lines.extend(["", "Query:", f"Input: {render_inputs(query)}", "Output:"])
    return "\n".join(lines)


def batch_prompt(
    train_pairs: Sequence[Tuple[Tuple[str, ...], str]],
    queries: Sequence[Tuple[str, ...]],
    rule_hint: bool = False,
    verify_hint: bool = False,
    structured: bool = False,
) -> str:
    if structured:
        payload = {
            "examples": [{"input": render_inputs(inp), "output": out} for inp, out in train_pairs],
            "queries": [{"index": i, "input": render_inputs(query)} for i, query in enumerate(queries)],
        }
        lines = [
            "Infer the text transformation from the examples in this JSON object.",
            "Apply the same transformation to every query.",
            "Return only a valid JSON array of strings in query index order.",
            f"The array must contain exactly {len(queries)} strings.",
            "",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "",
            "JSON array only:",
        ]
        return "\n".join(lines)
    lines = [
        "Infer the text transformation from the examples.",
    ]
    if rule_hint:
        lines.append("Use one consistent internal rule for every query row.")
    if verify_hint:
        lines.append("Before returning, silently verify every output against the same inferred rule and the exact formatting shown in the examples.")
    lines.extend(
        [
            "Return a valid JSON array of strings, in the same order as the queries.",
            "Return only the JSON array. Do not include explanations, markdown, keys, or numbering.",
            "",
            "Examples:",
        ]
    )
    for inp, out in train_pairs:
        lines.append(f"Input: {render_inputs(inp)}")
        lines.append(f"Output: {out}")
    lines.extend(["", "Queries:"])
    for i, query in enumerate(queries):
        lines.append(f"{i}: {render_inputs(query)}")
    lines.append("")
    lines.append(f"JSON array with exactly {len(queries)} strings:")
    return "\n".join(lines)


def parse_json_list(text: str, expected: int) -> Tuple[List[str], bool, str]:
    raw = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    candidates = [raw]
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        candidates.insert(0, m.group(0))
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(candidate)
            except Exception:
                continue
            if isinstance(obj, list):
                vals = ["" if x is None else str(x) for x in obj]
                return vals, len(vals) == expected, "ok" if len(vals) == expected else "wrong_count"
    # Last-ditch line parser is marked as not parse-clean.
    lines = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", x).strip().strip('"').strip("'") for x in raw.splitlines() if x.strip()]
    if len(lines) == expected:
        return lines, False, "line_fallback"
    return [], False, "parse_fail"


def chunks(seq: Sequence[Any], n: int) -> List[List[Any]]:
    return [list(seq[i : i + n]) for i in range(0, len(seq), n)]


def deterministic_shuffle(seq: Sequence[Any], seed_text: str) -> Tuple[List[Any], List[int]]:
    rng = random.Random(int.from_bytes(seed_text.encode("utf-8"), "little", signed=False) % (2**32))
    idx = list(range(len(seq)))
    rng.shuffle(idx)
    return [seq[i] for i in idx], idx


def unshuffle(vals: Sequence[str], idx: Sequence[int], n: int) -> List[str]:
    out = [""] * n
    for val, original_i in zip(vals, idx):
        if 0 <= original_i < n:
            out[original_i] = val
    return out


def run_row_by_row(tok: Any, model: Any, train_pairs: Sequence[Tuple[Tuple[str, ...], str]], test: Sequence[Example], max_new_tokens: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    preds: List[str] = []
    meta: List[Dict[str, Any]] = []
    for ex in test:
        raw = generate_text(tok, model, "You are a precise text transformation function.", row_prompt(train_pairs, ex.inputs), max_new_tokens)
        pred = clean_prediction(raw)
        preds.append(pred)
        meta.append({"raw": raw, "parse_ok": True, "parse_status": "single"})
    return preds, meta


def run_batched(
    tok: Any,
    model: Any,
    train_pairs: Sequence[Tuple[Tuple[str, ...], str]],
    test: Sequence[Example],
    batch_size: int,
    max_new_tokens: int,
    rule_hint: bool = False,
    verify_hint: bool = False,
    structured: bool = False,
    shuffled: bool = False,
    task_id: str = "",
) -> Tuple[List[str], List[Dict[str, Any]]]:
    queries = [ex.inputs for ex in test]
    order_idx: Optional[List[int]] = None
    if shuffled:
        queries, order_idx = deterministic_shuffle(queries, f"{task_id}|batch_all_shuffled")
    preds_ordered: List[str] = []
    meta: List[Dict[str, Any]] = []
    size = len(queries) if batch_size <= 0 else batch_size
    offset = 0
    for group in chunks(queries, size):
        raw = generate_text(
            tok,
            model,
            "You are a precise batch text transformation function.",
            batch_prompt(train_pairs, group, rule_hint=rule_hint, verify_hint=verify_hint, structured=structured),
            max_new_tokens,
        )
        vals, parse_ok, status = parse_json_list(raw, len(group))
        if len(vals) < len(group):
            vals = vals + [""] * (len(group) - len(vals))
        vals = vals[: len(group)]
        preds_ordered.extend(vals)
        for local_i, val in enumerate(vals):
            meta.append(
                {
                    "raw": raw,
                    "parse_ok": parse_ok,
                    "parse_status": status,
                    "batch_offset": offset,
                    "batch_local_index": local_i,
                    "batch_size": len(group),
                    "rule_hint": rule_hint,
                    "verify_hint": verify_hint,
                    "structured": structured,
                    "shuffled": shuffled,
                }
            )
        offset += len(group)
    if shuffled and order_idx is not None:
        preds = unshuffle(preds_ordered, order_idx, len(test))
        # Keep parse diagnostics attached to original row positions.
        meta_by_orig: List[Dict[str, Any]] = [{} for _ in range(len(test))]
        for m, original_i in zip(meta, order_idx):
            meta_by_orig[original_i] = m
        meta = meta_by_orig
    else:
        preds = preds_ordered
    return preds[: len(test)], meta[: len(test)]


def score_predictions(preds: Sequence[str], test: Sequence[Example]) -> Tuple[float, bool, List[bool]]:
    exacts = [p == ex.output for p, ex in zip(preds, test)]
    return (sum(exacts) / len(test) if test else 0.0), bool(test) and all(exacts), exacts


def run_experiment(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tasks = load_tasks(limit=args.task_limit if args.task_limit > 0 else None, min_examples=5)
    selected = choose_tasks(tasks, args.qwen_task_limit, args.sample_seed, args.train_n, args.heldout_cap, args.min_heldout)
    tok, model = load_qwen()
    task_rows: List[Dict[str, Any]] = []
    row_rows: List[Dict[str, Any]] = []
    methods: List[Tuple[str, Dict[str, Any]]] = [
        ("row_by_row", {"mode": "row"}),
        ("batch_2", {"mode": "batch", "batch_size": 2}),
        ("batch_4", {"mode": "batch", "batch_size": 4}),
        ("batch_all", {"mode": "batch", "batch_size": 0}),
        ("batch_all_shuffled", {"mode": "batch", "batch_size": 0, "shuffled": True}),
        ("batch_all_rule_hint", {"mode": "batch", "batch_size": 0, "rule_hint": True}),
        ("batch_all_verify_hint", {"mode": "batch", "batch_size": 0, "rule_hint": True, "verify_hint": True}),
        ("batch_all_structured", {"mode": "batch", "batch_size": 0, "structured": True}),
    ]
    if args.methods:
        keep = set(x.strip() for x in args.methods.split(",") if x.strip())
        methods = [(m, c) for m, c in methods if m in keep]
    for ti, task in enumerate(selected, start=1):
        train, test = split_examples(task, args.train_n, args.heldout_cap)
        train_pairs = [(e.inputs, e.output) for e in train]
        for method, spec in methods:
            if spec["mode"] == "row":
                preds, meta = run_row_by_row(tok, model, train_pairs, test, args.row_max_new_tokens)
            else:
                preds, meta = run_batched(
                    tok,
                    model,
                    train_pairs,
                    test,
                    batch_size=int(spec.get("batch_size", 0)),
                    max_new_tokens=args.batch_max_new_tokens,
                    rule_hint=bool(spec.get("rule_hint", False)),
                    verify_hint=bool(spec.get("verify_hint", False)),
                    structured=bool(spec.get("structured", False)),
                    shuffled=bool(spec.get("shuffled", False)),
                    task_id=task.task_id,
                )
            row_exact, full_exact, exacts = score_predictions(preds, test)
            parse_ok_rate = sum(bool(m.get("parse_ok", False)) for m in meta) / len(meta) if meta else 0.0
            task_rows.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "features": ",".join(task.features),
                    "method": method,
                    "heldout_rows": len(test),
                    "row_exact": row_exact,
                    "full_task_exact": full_exact,
                    "parse_ok_rate": parse_ok_rate,
                    "parse_statuses": json.dumps(
                        {str(k): int(v) for k, v in pd.Series([m.get("parse_status", "") for m in meta]).value_counts().items()},
                        ensure_ascii=False,
                    ),
                }
            )
            for ri, (ex, pred, exact, m) in enumerate(zip(test, preds, exacts, meta), start=1):
                row_rows.append(
                    {
                        "task_id": task.task_id,
                        "family": task.family,
                        "features": ",".join(task.features),
                        "method": method,
                        "row_index": ri,
                        "input": render_inputs(ex.inputs),
                        "target": ex.output,
                        "prediction": pred,
                        "exact": exact,
                        "parse_ok": bool(m.get("parse_ok", False)),
                        "parse_status": m.get("parse_status", ""),
                        "raw": m.get("raw", ""),
                    }
                )
        if ti == 1 or ti % 5 == 0 or ti == len(selected):
            sofar = pd.DataFrame(task_rows)
            bits = []
            for method, _ in methods:
                sub = sofar[sofar["method"].eq(method)]
                bits.append(f"{method}={int(sub['full_task_exact'].sum())}/{len(sub)}")
            print(f"task {ti}/{len(selected)} " + " ".join(bits), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(task_rows), pd.DataFrame(row_rows)


def summarize(task_df: pd.DataFrame) -> pd.DataFrame:
    if task_df.empty:
        return pd.DataFrame(columns=["method", "tasks", "full_task_exact", "row_exact", "parse_ok_rate"])
    return (
        task_df.groupby("method", as_index=False)
        .agg(
            tasks=("task_id", "count"),
            full_task_exact=("full_task_exact", "mean"),
            row_exact=("row_exact", "mean"),
            parse_ok_rate=("parse_ok_rate", "mean"),
        )
        .sort_values("method")
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
            if c.endswith("_exact") or c.endswith("_rate") or c in {"row_exact", "full_task_exact"}:
                view[c] = view[c].map(pct)
            else:
                view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    header = "|" + "|".join(cols) + "|"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for _, row in view.iterrows():
        body.append("|" + "|".join("" if pd.isna(row[c]) else html.escape(str(row[c])) for c in cols) + "|")
    return "\n".join([header, sep] + body)


def plot_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    order = ["row_by_row", "batch_2", "batch_4", "batch_all", "batch_all_shuffled", "batch_all_rule_hint", "batch_all_verify_hint", "batch_all_structured"]
    df = summary.set_index("method").reindex([m for m in order if m in set(summary["method"])]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    ax.bar([i - 0.18 for i in x], df["row_exact"] * 100, width=0.36, label="row exact", color="#10b981")
    ax.bar([i + 0.18 for i in x], df["full_task_exact"] * 100, width=0.36, label="full-task exact", color="#2563eb")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["method"], rotation=24)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact (%)")
    ax.set_title("Batched transduction accuracy")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "method_accuracy.png", dpi=160)
    plt.close(fig)


def plot_parse(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    df = summary[summary["method"].ne("row_by_row")].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(df["method"], df["parse_ok_rate"] * 100, color="#f59e0b")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Parse clean rows (%)")
    ax.set_title("JSON parse cleanliness by batch method")
    ax.tick_params(axis="x", rotation=24)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "parse_cleanliness.png", dpi=160)
    plt.close(fig)


def plot_family(task_df: pd.DataFrame) -> None:
    if task_df.empty:
        return
    pivot = task_df.pivot_table(index="family", columns="method", values="full_task_exact", aggfunc="mean")
    if "row_by_row" not in pivot.columns or "batch_all" not in pivot.columns:
        return
    fam = task_df.groupby("family").size().rename("method_rows").reset_index()
    counts = task_df[task_df["method"].eq("row_by_row")].groupby("family").size()
    diff = (pivot["batch_all"] - pivot["row_by_row"]).to_frame("batch_all_delta")
    diff["tasks"] = counts
    diff = diff[diff["tasks"] >= 2].sort_values("batch_all_delta")
    if diff.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(diff))))
    colors = ["#ef4444" if v < 0 else "#10b981" for v in diff["batch_all_delta"]]
    ax.barh(diff.index, diff["batch_all_delta"] * 100, color=colors)
    ax.axvline(0, color="#111827", lw=1)
    ax.set_xlabel("Batch-all full-task delta vs row-by-row (pp)")
    ax.set_title("Batch effect by family")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "family_batch_delta.png", dpi=160)
    plt.close(fig)


def plot_task_scatter(task_df: pd.DataFrame) -> None:
    if task_df.empty:
        return
    pivot = task_df.pivot(index="task_id", columns="method", values="row_exact")
    if "row_by_row" not in pivot.columns or "batch_all" not in pivot.columns:
        return
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(pivot["row_by_row"] * 100, pivot["batch_all"] * 100, alpha=0.75, color="#2563eb")
    ax.plot([0, 100], [0, 100], color="#111827", lw=1, ls="--")
    ax.set_xlabel("Row-by-row row exact (%)")
    ax.set_ylabel("Batch-all row exact (%)")
    ax.set_title("Per-task row accuracy")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "task_row_scatter.png", dpi=160)
    plt.close(fig)


def write_report(task_df: pd.DataFrame, row_df: pd.DataFrame, cfg: Dict[str, Any], suite: str) -> None:
    summary = summarize(task_df)
    plot_summary(summary)
    plot_parse(summary)
    plot_family(task_df)
    plot_task_scatter(task_df)
    deltas = pd.DataFrame()
    if not task_df.empty and {"row_by_row", "batch_all"}.issubset(set(task_df["method"])):
        wide = task_df.pivot(index="task_id", columns="method", values=["full_task_exact", "row_exact"])
        delta_rows = []
        for task_id in wide.index:
            delta_rows.append(
                {
                    "task_id": task_id,
                    "row_by_row_full": bool(wide.loc[task_id, ("full_task_exact", "row_by_row")]),
                    "batch_all_full": bool(wide.loc[task_id, ("full_task_exact", "batch_all")]),
                    "batch_all_row_delta": float(wide.loc[task_id, ("row_exact", "batch_all")] - wide.loc[task_id, ("row_exact", "row_by_row")]),
                }
            )
        deltas = pd.DataFrame(delta_rows)

    lines: List[str] = []
    lines.append("# Qwen Batched Transduction Consistency")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append("This standalone experiment tests whether answering multiple query rows in one shared generation context improves task-level consistency on public text-transformation tasks. The strict primary metric is full-task exact: all held-out rows for a task must be answered exactly.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Dataset: public `Transformation.Text` tasks.")
    lines.append(f"- Split: first `{cfg['train_n']}` examples are train examples; up to `{cfg['heldout_cap']}` held-out examples are scored.")
    lines.append("- Row-by-row baseline: one prompt per held-out row.")
    lines.append("- Batched transduction: one JSON-array output per query batch, with batch sizes 2, 4, and all held-out rows.")
    lines.append("- Shuffled-order control: query rows are shuffled inside the batch and then unshuffled for scoring.")
    lines.append("- Rule-hint arm: one-batch output with an instruction to use one consistent internal rule.")
    lines.append("- Parse failures are counted directly; malformed or wrong-length JSON arrays receive empty predictions for missing rows.")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Suite: `{suite}`.")
    lines.append(f"- Qwen model: `{MODEL_NAME}`.")
    lines.append(f"- Tasks: `{summary['tasks'].max() if not summary.empty else 0}`.")
    lines.append(f"- Held-out cap: `{cfg['heldout_cap']}`.")
    lines.append(f"- Sample seed: `{cfg['sample_seed']}`.")
    lines.append("")
    lines.append("## Primary Results")
    lines.append("")
    lines.append(md_table(summary, ["method", "tasks", "full_task_exact", "row_exact", "parse_ok_rate"]))
    lines.append("")
    if not deltas.empty:
        wins = int((~deltas["row_by_row_full"] & deltas["batch_all_full"]).sum())
        losses = int((deltas["row_by_row_full"] & ~deltas["batch_all_full"]).sum())
        ties = int((deltas["row_by_row_full"] == deltas["batch_all_full"]).sum())
        lines.append("### Batch-All Task Flips")
        lines.append("")
        lines.append(f"- Batch-all wins over row-by-row on `{wins}` tasks.")
        lines.append(f"- Batch-all loses to row-by-row on `{losses}` tasks.")
        lines.append(f"- Batch-all ties row-by-row on `{ties}` tasks.")
        lines.append("")
    family = (
        task_df.groupby(["family", "method"], as_index=False)
        .agg(tasks=("task_id", "count"), full_task_exact=("full_task_exact", "mean"), row_exact=("row_exact", "mean"))
        .sort_values(["family", "method"])
    )
    lines.append("### Family Summary")
    lines.append("")
    lines.append(md_table(family, ["family", "method", "tasks", "full_task_exact", "row_exact"], max_rows=80))
    lines.append("")
    task_view = task_df[task_df["method"].isin(["row_by_row", "batch_all", "batch_all_rule_hint"])].copy()
    lines.append("### Task Details")
    lines.append("")
    lines.append(md_table(task_view.sort_values(["task_id", "method"]), ["task_id", "family", "method", "heldout_rows", "full_task_exact", "row_exact", "parse_ok_rate"], max_rows=120))
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fig in ["method_accuracy.png", "parse_cleanliness.png", "family_batch_delta.png", "task_row_scatter.png"]:
        if (FIGURES / fig).exists():
            lines.append(f"![{fig}](../analysis/figures/{fig})")
            lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if not summary.empty:
        smap = {row["method"]: row for _, row in summary.iterrows()}
        row = smap.get("row_by_row")
        batch_all = smap.get("batch_all")
        if row is not None and batch_all is not None:
            delta_full = float(batch_all["full_task_exact"] - row["full_task_exact"])
            delta_row = float(batch_all["row_exact"] - row["row_exact"])
            lines.append(f"Row-by-row inference reaches {pct(row['row_exact'])} row exact and {pct(row['full_task_exact'])} full-task exact. Batch-all reaches {pct(batch_all['row_exact'])} row exact and {pct(batch_all['full_task_exact'])} full-task exact, a full-task delta of {pct(delta_full)} and a row-exact delta of {pct(delta_row)}.")
        shuffled = smap.get("batch_all_shuffled")
        if batch_all is not None and shuffled is not None:
            lines.append(f"The shuffled-order batch control reaches {pct(shuffled['full_task_exact'])} full-task exact, compared with {pct(batch_all['full_task_exact'])} for normal batch-all.")
        hint = smap.get("batch_all_rule_hint")
        if batch_all is not None and hint is not None:
            lines.append(f"The rule-hint batch arm reaches {pct(hint['full_task_exact'])} full-task exact, compared with {pct(batch_all['full_task_exact'])} for plain batch-all.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("This run uses deterministic decoding and a capped task sample. Batched JSON output is stricter than ordinary text output, so parse cleanliness is reported separately. Full-task exact is intentionally harsh; a method can have high row exact while failing full-task exact because one row is wrong.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Task-level details: `analysis/task_details.csv`")
    lines.append("- Row-level details: `analysis/row_details.csv`")
    lines.append("- Summary: `analysis/summary.csv`")
    lines.append("- Figures: `analysis/figures/`")
    lines.append("- Benchmark mirror: `/workspace/large_artifacts/qwen_batched_transduction_consistency/prose-benchmarks`")

    REPORTS.mkdir(parents=True, exist_ok=True)
    md = "\n".join(lines) + "\n"
    (REPORTS / "qwen_batched_transduction_consistency_report.md").write_text(md)
    write_html(lines, REPORTS / "qwen_batched_transduction_consistency_report.html")


def write_html(lines: Sequence[str], path: Path) -> None:
    body: List[str] = []
    in_ul = False
    in_table = False
    table_lines: List[str] = []

    def flush_ul() -> None:
        nonlocal in_ul
        if in_ul:
            body.append("</ul>")
            in_ul = False

    def flush_table() -> None:
        nonlocal in_table, table_lines
        if not in_table:
            return
        rows: List[List[str]] = []
        for line in table_lines:
            if set(line.replace("|", "").strip()) <= {"-", ":"}:
                continue
            rows.append([c.strip() for c in line.strip("|").split("|")])
        if rows:
            body.append("<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in rows[0]) + "</tr></thead><tbody>")
            for row in rows[1:]:
                body.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>")
            body.append("</tbody></table>")
        in_table = False
        table_lines = []

    for line in lines:
        if line.startswith("|"):
            flush_ul()
            in_table = True
            table_lines.append(line)
            continue
        flush_table()
        if line.startswith("# "):
            flush_ul()
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            flush_ul()
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            flush_ul()
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("![") and "](" in line:
            flush_ul()
            alt = line[2 : line.find("]")]
            src = line[line.find("(") + 1 : line.rfind(")")]
            body.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.startswith("- "):
            if not in_ul:
                body.append("<ul>")
                in_ul = True
            body.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.strip():
            flush_ul()
            body.append(f"<p>{html.escape(line)}</p>")
        else:
            flush_ul()
    flush_table()
    flush_ul()
    doc = (
        '<!doctype html><html><head><meta charset="utf-8"><title>Qwen Batched Transduction Consistency</title>'
        "<style>body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:32px;line-height:1.5;color:#111827;max-width:1160px}"
        "h1,h2,h3{line-height:1.2}table{border-collapse:collapse;margin:14px 0 24px;width:100%;font-size:14px}"
        "th,td{border:1px solid #e5e7eb;padding:6px 8px;text-align:left;vertical-align:top}th{background:#f3f4f6}"
        "img{max-width:100%;border:1px solid #e5e7eb;border-radius:6px}figure{margin:24px 0}p,li{max-width:940px}</style>"
        "</head><body>"
        + "\n".join(body)
        + "\n</body></html>\n"
    )
    path.write_text(doc)


def write_log(suite: str, cfg: Dict[str, Any], summary: pd.DataFrame, elapsed: float) -> None:
    with (ROOT / "experiment_log.md").open("a") as f:
        f.write(f"\n## Run `{suite}`\n\n")
        f.write(f"- Time UTC: `{datetime.now(timezone.utc).isoformat()}`\n")
        f.write(f"- Elapsed seconds: `{elapsed:.1f}`\n")
        f.write(f"- Config: `{json.dumps(cfg, sort_keys=True)}`\n")
        for _, row in summary.iterrows():
            f.write(f"- `{row['method']}`: row exact {pct(row['row_exact'])}; full-task exact {pct(row['full_task_exact'])}; parse ok {pct(row['parse_ok_rate'])}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="smoke")
    p.add_argument("--task_limit", type=int, default=0)
    p.add_argument("--qwen_task_limit", type=int, default=12)
    p.add_argument("--train_n", type=int, default=4)
    p.add_argument("--heldout_cap", type=int, default=6)
    p.add_argument("--min_heldout", type=int, default=3)
    p.add_argument("--sample_seed", type=int, default=20260627)
    p.add_argument("--row_max_new_tokens", type=int, default=64)
    p.add_argument("--batch_max_new_tokens", type=int, default=256)
    p.add_argument("--methods", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    RUNS.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS / args.suite
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    (run_dir / "config.json").write_text(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(), **cfg}, indent=2) + "\n")
    task_df, row_df = run_experiment(args)
    summary = summarize(task_df)
    task_df.to_csv(run_dir / "task_details.csv", index=False)
    row_df.to_csv(run_dir / "row_details.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    task_df.to_csv(ANALYSIS / "task_details.csv", index=False)
    row_df.to_csv(ANALYSIS / "row_details.csv", index=False)
    summary.to_csv(ANALYSIS / "summary.csv", index=False)
    write_report(task_df, row_df, cfg, args.suite)
    elapsed = time.time() - started
    (run_dir / "done.json").write_text(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_sec": round(elapsed, 2)}, indent=2) + "\n")
    write_log(args.suite, cfg, summary, elapsed)
    print(f"finished suite={args.suite} elapsed={elapsed:.1f}s")
    print((REPORTS / "qwen_batched_transduction_consistency_report.md").as_posix())


if __name__ == "__main__":
    main()
