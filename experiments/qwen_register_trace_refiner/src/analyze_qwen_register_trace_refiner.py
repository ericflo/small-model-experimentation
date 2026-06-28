#!/usr/bin/env python3
"""Aggregate Qwen register trace refiner runs and regenerate analysis artifacts."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path("experiments/qwen_register_trace_refiner")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
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
    return "n/a" if math.isnan(x) else f"{100.0 * x:.1f}%"


def num(value: Any) -> str:
    x = as_float(value)
    return "n/a" if math.isnan(x) else f"{x:.2f}"


def choose_primary(rows: Sequence[Dict[str, Any]], metric_paths: Sequence[Path]) -> str:
    preferred = "main_register_trace_refiner_s512"
    if any(row.get("run") == preferred for row in rows):
        return preferred
    by_run = {path.parent.name: path.stat().st_mtime for path in metric_paths}
    candidates = sorted(
        {row.get("run", "") for row in rows if row.get("run") and not row.get("run", "").startswith("smoke")},
        key=lambda run: by_run.get(run, 0.0),
    )
    if candidates:
        return candidates[-1]
    runs = sorted({row.get("run", "") for row in rows if row.get("run")}, key=lambda run: by_run.get(run, 0.0))
    return runs[-1] if runs else ""


def fresh_rows(rows: Sequence[Dict[str, Any]], run: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("run") == run and row.get("split", "").startswith("fresh_")]


def split_label(split: str) -> str:
    return split.replace("fresh_", "").replace("_len24", "").replace("_", " ")


def write_summary(rows: Sequence[Dict[str, Any]], primary_run: str) -> None:
    primary = [row for row in rows if row.get("run") == primary_run]
    lines = [
        "# Qwen Register Trace Refiner Analysis Summary",
        "",
        f"Primary run: `{primary_run}`",
        "",
        "| split | base | prior | soft-trace | learned | guarded | pair-rerank | oracle | learned gap | guarded gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            "| {split} | {base} | {prior} | {soft} | {learned} | {guarded} | {pair} | {oracle} | {gap} | {guard_gap} |".format(
                split=row.get("split", ""),
                base=pct(row.get("base_executor_accuracy")),
                prior=pct(row.get("prior_executor_accuracy")),
                soft=pct(row.get("soft_trace_executor_accuracy")),
                learned=pct(row.get("learned_executor_accuracy")),
                guarded=pct(row.get("guarded_executor_accuracy")),
                pair=pct(row.get("pair_rerank_executor_accuracy")),
                oracle=pct(row.get("oracle_executor_accuracy")),
                gap=pct(row.get("learned_oracle_gap_recovered")),
                guard_gap=pct(row.get("guarded_oracle_gap_recovered")),
            )
        )
    paired = next((row for row in primary if row.get("split") == "fresh_paired_len24"), {})
    lines.extend(
        [
            "",
            "## Fresh Paired Details",
            "",
            "| metric | base | learned | guarded | pair-rerank | oracle |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in [
        "executor_accuracy",
        "program_exact",
        "state_prefix_fraction",
        "pair_both_correct",
        "pair_state_consistency",
        "changed_fraction",
        "avg_edits",
    ]:
        fmt = num if metric == "avg_edits" else pct
        lines.append(
            "| {metric} | {base} | {learned} | {guarded} | {pair} | {oracle} |".format(
                metric=metric,
                base=fmt(paired.get(f"base_{metric}")),
                learned=fmt(paired.get(f"learned_{metric}")),
                guarded=fmt(paired.get(f"guarded_{metric}")),
                pair=fmt(paired.get(f"pair_rerank_{metric}")),
                oracle=fmt(paired.get(f"oracle_{metric}")),
            )
        )
    lines.extend(
        [
            "",
            "## Candidate Set",
            "",
            "| split | candidates/example | positive candidates/example | oracle found |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in primary:
        lines.append(
            "| {split} | {cand:.1f} | {pos:.2f} | {found} |".format(
                split=row.get("split", ""),
                cand=as_float(row.get("avg_candidates")),
                pos=as_float(row.get("avg_positive_candidates")),
                found=pct(row.get("oracle_found_fraction")),
            )
        )
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")


def write_figures(rows: Sequence[Dict[str, Any]], primary_run: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[figures] matplotlib unavailable: {exc}")
        return
    primary = fresh_rows(rows, primary_run)
    if not primary:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)

    labels = [split_label(row["split"]) for row in primary]
    x = list(range(len(labels)))
    width = 0.16
    series = [
        ("base", "base_executor_accuracy"),
        ("soft-trace", "soft_trace_executor_accuracy"),
        ("learned", "learned_executor_accuracy"),
        ("guarded", "guarded_executor_accuracy"),
        ("pair-rerank", "pair_rerank_executor_accuracy"),
        ("oracle", "oracle_executor_accuracy"),
    ]
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    width = 0.13
    for offset, (name, key) in enumerate(series):
        values = [as_float(row.get(key)) * 100.0 for row in primary]
        ax.bar([i + (offset - 2.5) * width for i in x], values, width, label=name)
    ax.set_title("Register-program repair accuracy on fresh length-24 tasks")
    ax.set_ylabel("executor accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=5, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "executor_accuracy_by_split.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    learned_gaps = [as_float(row.get("learned_oracle_gap_recovered")) * 100.0 for row in primary]
    guarded_gaps = [as_float(row.get("guarded_oracle_gap_recovered")) * 100.0 for row in primary]
    gap_width = 0.34
    ax.bar([i - gap_width / 2 for i in x], learned_gaps, gap_width, label="learned", color="#4c78a8")
    ax.bar([i + gap_width / 2 for i in x], guarded_gaps, gap_width, label="guarded", color="#59a14f")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Fraction of available oracle repair gap recovered")
    ax.set_ylabel("gap recovered (%)")
    finite = [x for x in learned_gaps + guarded_gaps if not math.isnan(x)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(min(-5, min(finite or [0]) - 5), max(100, max(finite or [0]) + 5))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "oracle_gap_recovered.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    cand = [as_float(row.get("avg_candidates")) for row in primary]
    pos = [as_float(row.get("avg_positive_candidates")) for row in primary]
    ax.plot(labels, cand, marker="o", linewidth=2, label="candidates/example")
    ax.plot(labels, pos, marker="o", linewidth=2, label="positive candidates/example")
    ax.set_title("Repair search space and exact candidates")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "candidate_set_profile.png", dpi=180)
    plt.close(fig)


def main() -> None:
    rows: List[Dict[str, Any]] = []
    metric_paths = sorted(RUNS.glob("*/metrics.csv"))
    for path in metric_paths:
        rows.extend(read_csv(path))
    primary_run = choose_primary(rows, metric_paths)
    primary_rows = [row for row in rows if row.get("run") == primary_run]
    write_csv(ANALYSIS / "all_final_metrics.csv", rows)
    write_csv(ANALYSIS / "final_metrics.csv", primary_rows)
    write_summary(rows, primary_run)
    write_figures(rows, primary_run)
    print(ANALYSIS / "summary.md")


if __name__ == "__main__":
    main()
