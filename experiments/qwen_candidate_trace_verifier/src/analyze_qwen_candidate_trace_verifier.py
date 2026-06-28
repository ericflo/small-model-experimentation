#!/usr/bin/env python3
"""Aggregate candidate-trace verifier runs and regenerate summary artifacts."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path("experiments/qwen_candidate_trace_verifier")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def pct(value: Any) -> str:
    x = as_float(value)
    if math.isnan(x):
        return "n/a"
    return f"{100.0 * x:.1f}%"


def choose_primary(rows: List[Dict[str, Any]]) -> str:
    preferred = "main_trace_verifier_s512"
    if any(row.get("run") == preferred for row in rows):
        return preferred
    runs = sorted({row.get("run", "") for row in rows if row.get("run")})
    return runs[-1] if runs else ""


def write_summary(rows: List[Dict[str, Any]], primary_run: str) -> None:
    primary = [row for row in rows if row.get("run") == primary_run]
    lines = [
        "# Candidate-Trace Verifier Analysis Summary",
        "",
        f"Primary run: `{primary_run}`",
        "",
        "| split | base | trace verifier | pair rerank | oracle | gap recovered |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            "| {split} | {base} | {learned} | {pair} | {oracle} | {gap} |".format(
                split=row.get("split", ""),
                base=pct(row.get("base_executor_accuracy")),
                learned=pct(row.get("learned_executor_accuracy")),
                pair=pct(row.get("pair_rerank_executor_accuracy")),
                oracle=pct(row.get("oracle_executor_accuracy")),
                gap=pct(row.get("learned_oracle_gap_recovered")),
            )
        )
    lines.extend(
        [
            "",
            "## Fresh Paired Details",
            "",
            "| metric | base | trace verifier | pair rerank | oracle |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    paired = next((row for row in primary if row.get("split") == "fresh_paired_len24"), {})
    for metric in ["executor_accuracy", "program_exact", "state_prefix_fraction", "pair_both_correct", "pair_state_consistency"]:
        lines.append(
            "| {metric} | {base} | {learned} | {pair} | {oracle} |".format(
                metric=metric,
                base=pct(paired.get(f"base_{metric}")),
                learned=pct(paired.get(f"learned_{metric}")),
                pair=pct(paired.get(f"pair_rerank_{metric}")),
                oracle=pct(paired.get(f"oracle_{metric}")),
            )
        )
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")


def write_figure(rows: List[Dict[str, Any]], primary_run: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[figures] matplotlib unavailable: {exc}")
        return
    primary = [row for row in rows if row.get("run") == primary_run and row.get("split", "").startswith("fresh_")]
    if not primary:
        return
    labels = [row["split"].replace("fresh_", "").replace("_len24", "") for row in primary]
    base = [as_float(row.get("base_executor_accuracy")) for row in primary]
    learned = [as_float(row.get("learned_executor_accuracy")) for row in primary]
    pair = [as_float(row.get("pair_rerank_executor_accuracy")) for row in primary]
    oracle = [as_float(row.get("oracle_executor_accuracy")) for row in primary]
    x = list(range(len(labels)))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - 1.5 * width for i in x], base, width, label="base")
    ax.bar([i - 0.5 * width for i in x], learned, width, label="trace verifier")
    ax.bar([i + 0.5 * width for i in x], pair, width, label="pair rerank")
    ax.bar([i + 1.5 * width for i in x], oracle, width, label="oracle")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("executor accuracy")
    ax.legend(loc="upper left", ncols=4, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = ANALYSIS / "figures" / "executor_accuracy.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/metrics.csv")):
        rows.extend(read_csv(path))
    primary_run = choose_primary(rows)
    primary_rows = [row for row in rows if row.get("run") == primary_run]
    write_csv(ANALYSIS / "all_final_metrics.csv", rows)
    write_csv(ANALYSIS / "final_metrics.csv", primary_rows)
    write_summary(rows, primary_run)
    write_figure(rows, primary_run)
    print(ANALYSIS / "summary.md")


if __name__ == "__main__":
    main()
