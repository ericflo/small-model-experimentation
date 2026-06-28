#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_json, write_json  # noqa: E402


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def rows_from(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload.get("results", [])


def metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(row.get("records", {}).get(key, default))


def tokens(row: dict[str, Any]) -> int:
    return int(metric(row, "forward_tokens", 0.0))


def plot_coverage(rows: list[dict[str, Any]], title: str, out: Path, limit: int | None = None) -> None:
    data = rows[:limit] if limit else rows
    labels = [row["arm_name"] for row in data]
    coverage = [metric(row, "coverage") for row in data]
    pass1 = [metric(row, "pass1_proxy") for row in data]
    x = list(range(len(data)))
    fig, ax = plt.subplots(figsize=(max(9.0, len(data) * 0.85), 4.8))
    ax.bar([i - 0.18 for i in x], coverage, width=0.36, label="coverage", color="#2563eb")
    ax.bar([i + 0.18 for i in x], pass1, width=0.36, label="pass@1 proxy", color="#16a34a")
    ax.set_xticks(x, [label.replace("_", "\n") for label in labels], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_ylabel("rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_pareto(rows: list[dict[str, Any]], title: str, out: Path, skip_full: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for row in rows:
        if skip_full and "full_union" in row["arm_name"]:
            continue
        ax.scatter(tokens(row), metric(row, "coverage"), s=70)
        ax.text(tokens(row), metric(row, "coverage") + 0.01, row["arm_name"], fontsize=7, ha="center")
    ax.set_xlabel("forward tokens")
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_scheduler(source_payload: dict[str, Any], loo_payload: dict[str, Any], out: Path) -> None:
    labels = []
    counts = []
    for key, value in source_payload.get("scheduler", {}).get("label_counts", {}).items():
        labels.append(f"source label:\n{key}")
        counts.append(value)
    for key, value in source_payload.get("learned_action_counts", {}).items():
        labels.append(f"source pred:\n{key}")
        counts.append(value)
    for key, value in loo_payload.get("summary", {}).get("action_counts", {}).items():
        labels.append(f"subset LOO:\n{key}")
        counts.append(value)
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.7), 4.5))
    ax.bar(range(len(labels)), counts, color="#7c3aed")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("count")
    ax.set_title("Scheduler Label and Action Counts")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| arm | n | coverage | pass@1 | candidates/task | parse/task | functional diversity | forward tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rec = row["records"]
        lines.append(
            f"| {row['arm_name']} | {rec.get('records', 0)} | {pct(rec.get('coverage', 0.0))} | "
            f"{pct(rec.get('pass1_proxy', 0.0))} | {rec.get('candidate_count_mean', 0.0):.2f} | "
            f"{rec.get('parse_success_mean', 0.0):.2f} | {pct(rec.get('distinct_functional_rate_mean', 0.0))} | "
            f"{int(rec.get('forward_tokens', 0))} |"
        )
    return "\n".join(lines)


def compact_rows(rows: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    by_name = {row["arm_name"]: row for row in rows}
    return [by_name[name] for name in names if name in by_name]


def gate_text(source_rows: list[dict[str, Any]], subset_rows: list[dict[str, Any]], loo_payload: dict[str, Any]) -> str:
    src = {row["arm_name"]: row for row in source_rows}
    sub = {row["arm_name"]: row for row in subset_rows}
    lines = []
    if "base_prefix_k4" in src and "learned_scheduler_after_prefix2" in src and "oracle_best_block_after_prefix2" in src:
        lines.append(
            "Broad source-policy run: base prefix K4 reached "
            f"{pct(metric(src['base_prefix_k4'], 'coverage'))}; learned scheduling reached "
            f"{pct(metric(src['learned_scheduler_after_prefix2'], 'coverage'))}; oracle after the same prefix reached "
            f"{pct(metric(src['oracle_best_block_after_prefix2'], 'coverage'))}."
        )
    if "full_union_all_candidates" in src:
        lines.append(
            "The full source-policy union reached "
            f"{pct(metric(src['full_union_all_candidates'], 'coverage'))}, but at "
            f"{tokens(src['full_union_all_candidates'])} forward tokens."
        )
    if "subset_base_hot_k8" in sub and "subset_oracle_choose_arm" in sub and "subset_hot4_plus_constrained4" in sub:
        lines.append(
            "Constrained-arm subset: base hot K8 reached "
            f"{pct(metric(sub['subset_base_hot_k8'], 'coverage'))}; hot4+constrained4 tied it at "
            f"{pct(metric(sub['subset_hot4_plus_constrained4'], 'coverage'))}; oracle arm choice reached "
            f"{pct(metric(sub['subset_oracle_choose_arm'], 'coverage'))} at {tokens(sub['subset_oracle_choose_arm'])} forward tokens."
        )
    lines.append(
        "Leave-one-task-out constrained scheduler reached "
        f"{pct(loo_payload.get('summary', {}).get('coverage', 0.0))} with action counts "
        f"{loo_payload.get('summary', {}).get('action_counts', {})}."
    )
    lines.append(
        "Gate readout: no deployable scheduler in this pilot beat the single-policy sample-more reference. The positive signal is oracle headroom for choosing among policy arms, not an already solved scheduler."
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--subset-results", type=Path, required=True)
    parser.add_argument("--loo-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/final_report.md")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "reports/report_summary.json")
    args = parser.parse_args()

    source = load_json(args.source_results)
    subset = load_json(args.subset_results)
    loo = load_json(args.loo_results)
    source_rows = rows_from(source)
    subset_rows = rows_from(subset)
    fig_dir = ROOT / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    source_compact = compact_rows(
        source_rows,
        [
            "base_prefix_k4",
            "prefix2_mid4",
            "learned_scheduler_after_prefix2",
            "oracle_best_block_after_prefix2",
            "full_union_all_candidates",
        ],
    )
    subset_compact = compact_rows(
        subset_rows,
        [
            "subset_base_hot_k4",
            "subset_base_hot_k8",
            "subset_constrained_k4",
            "subset_hot4_plus_constrained4",
            "subset_visible_gate_hot4_then_constrained",
            "subset_oracle_choose_arm",
        ],
    )
    plot_coverage(source_compact, "Broad Source-Policy Portfolio", fig_dir / "source_coverage_pass1.png")
    plot_pareto(source_compact, "Broad Source-Policy Pareto", fig_dir / "source_pareto.png")
    plot_coverage(subset_compact, "Constrained-Arm Subset Portfolio", fig_dir / "subset_coverage_pass1.png")
    plot_pareto(subset_compact, "Constrained-Arm Subset Pareto", fig_dir / "subset_pareto.png")
    plot_scheduler(source, loo, fig_dir / "scheduler_actions.png")

    summary = {"source": source, "subset": subset, "loo": loo}
    write_json(args.summary_out, summary)
    report = f"""# qwen35_4b_sampler_portfolio_scheduler

## Question

Can a portfolio of generation policies, selected by static schedules or a small deployable scheduler, improve the coverage/pass@1/forward-token Pareto frontier over simply sampling more from one policy?

This package evaluates two views:

- a broad 80-task source-policy pool with low/mid/high/diverse candidate blocks;
- a 24-task constrained-arm subset comparing base-hot sampling with a constrained preference policy.

Oracle rows are reported only as headroom. Deployable rows use fixed schedules, visible-test gates, or schedulers trained without hidden eval labels.

## Broad Source-Policy Results

{table(source_compact)}

![source coverage](figures/source_coverage_pass1.png)

![source pareto](figures/source_pareto.png)

## Constrained-Arm Subset Results

{table(subset_compact)}

![subset coverage](figures/subset_coverage_pass1.png)

![subset pareto](figures/subset_pareto.png)

## Scheduler Diagnostics

Source scheduler train labels: `{source.get('scheduler', {}).get('label_counts', {})}`

Source scheduler eval actions: `{source.get('learned_action_counts', {})}`

Constrained leave-one-task-out labels: `{loo.get('oracle_label_counts', {})}`

Constrained leave-one-task-out summary: `{loo.get('summary', {})}`

![scheduler actions](figures/scheduler_actions.png)

## Gate Readout

{gate_text(source_rows, subset_rows, loo)}

## Interpretation

The portfolio idea is not dead, but the naive scheduler is not enough. The broad pool shows that large unions contain much more coverage, while small static or learned schedules do not extract it. The constrained subset is sharper: the oracle arm chooser reaches 75.0% at roughly the same token cost as K4, while ordinary hot K8 reaches 66.7% at about double the tokens. That means the next valuable target is a better policy-value estimator, not more blind sampling and not a one-arm LoRA scale-up.

The concrete next direction is to collect a larger multi-policy training set where every task has matched candidates from each policy arm, then train a policy-value model to predict which arm is likely to add new hidden coverage from prompt plus cheap prefix evidence. This pilot says there is arm-choice headroom, but it is sparse and not exposed cleanly enough to the simple visible-feature schedulers used here.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
