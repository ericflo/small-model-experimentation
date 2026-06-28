#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from src.jsonl import load_json, load_jsonl, write_json


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def arm_rows(summary: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return list(summary["arms"].items())


def plot_coverage(summary: dict[str, Any], out: Path) -> None:
    rows = arm_rows(summary)
    labels = [name.replace("_", "\n") for name, _ in rows]
    coverage = [row.get("coverage", 0.0) for _, row in rows]
    pass1 = [row.get("pass1_proxy", 0.0) for _, row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(rows))
    ax.bar([i - 0.18 for i in x], coverage, width=0.36, label="coverage", color="#2563eb")
    ax.bar([i + 0.18 for i in x], pass1, width=0.36, label="pass@1 proxy", color="#16a34a")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Prefix Search Gate: Coverage")
    ax.legend()
    for i, value in enumerate(coverage):
        ax.text(i - 0.18, value + 0.02, pct(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_diversity(summary: dict[str, Any], out: Path) -> None:
    rows = arm_rows(summary)
    labels = [name.replace("_", "\n") for name, _ in rows]
    func = [row.get("mean_distinct_functional_rate", 0.0) for _, row in rows]
    prog = [row.get("mean_distinct_program_rate", 0.0) for _, row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(rows))
    ax.bar([i - 0.18 for i in x], func, width=0.36, label="functional", color="#9333ea")
    ax.bar([i + 0.18 for i in x], prog, width=0.36, label="program", color="#f97316")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean distinct rate")
    ax.set_title("Prefix Search Gate: Diversity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_prefix_validity(records: list[dict[str, Any]], out: Path) -> None:
    task_ids = [str(row["task_id"]) for row in records]
    valid = [row["prefix_count_valid"] for row in records]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(task_ids, valid, color="#475569")
    ax.set_ylabel("Valid proposed prefixes")
    ax.set_xlabel("Task id")
    ax.set_title("Prefix Proposal Validity")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def results_table(summary: dict[str, Any]) -> str:
    lines = [
        "| arm | coverage | pass@1 proxy | visible coverage | functional diversity | program diversity | forward tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in arm_rows(summary):
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    pct(row.get("coverage", 0.0)),
                    pct(row.get("pass1_proxy", 0.0)),
                    pct(row.get("visible_coverage", 0.0)),
                    pct(row.get("mean_distinct_functional_rate", 0.0)),
                    pct(row.get("mean_distinct_program_rate", 0.0)),
                    str(row.get("forward_tokens", 0)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def gate_text(summary: dict[str, Any]) -> str:
    full = summary["arms"]["full_sample_base"]["coverage"]
    oracle = summary["arms"]["prefix_oracle_selected"]["coverage"]
    union = summary["arms"]["prefix_union"]["coverage"]
    lexical = summary["arms"]["prefix_lexical_selected"]["coverage"]
    if oracle <= full and union <= full:
        verdict = "failed"
        action = "Do not train a prefix value model from this result."
    else:
        verdict = "cleared"
        action = "A value-model pilot is justified because the hidden-test oracle prefix ceiling beats ordinary full sampling."
    return (
        f"Oracle-prefix gate: **{verdict}**. Full sampling coverage was {pct(full)}; "
        f"prefix union coverage was {pct(union)}; oracle-selected prefix coverage was {pct(oracle)}; "
        f"lexical-selected prefix coverage was {pct(lexical)}. {action}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/final_report.md")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "reports/report_summary.json")
    args = parser.parse_args()

    records = load_jsonl(args.records)
    summary = load_json(args.summary)
    fig_dir = ROOT / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_coverage(summary, fig_dir / "coverage.png")
    plot_diversity(summary, fig_dir / "diversity.png")
    plot_prefix_validity(records, fig_dir / "prefix_validity.png")
    write_json(args.summary_out, {"summary": summary})

    report = f"""# qwen35_4b_prefix_value_guided_search

## Question

Can hidden-test-valued code prefixes expose a search state that is better than ordinary full-code sampling at matched completion budget?

## Setup

- Dataset split: `{summary['split']}`.
- Task count: {summary['count']}.
- Full-code samples per task: {summary['full_samples']}.
- Prefix proposals per task: {summary['prefix_count']}.
- Completions per prefix: {summary['completions_per_prefix']}.
- Matched completion budget: {summary['completion_budget_matched']}.
- Mean valid prefixes per task: {summary['mean_valid_prefixes']:.2f}.

## Results

{results_table(summary)}

![coverage](figures/coverage.png)

![diversity](figures/diversity.png)

![prefix validity](figures/prefix_validity.png)

## Gate Decision

{gate_text(summary)}

## Interpretation

This is an oracle-ceiling experiment. A positive result would mean the prefix state space contains selectable states whose completions solve tasks more efficiently than ordinary full-code sampling.

There is an efficiency hint: the oracle-selected prefix arm matches full-sampling coverage with only one selected prefix's completion set per task. But that is not enough to justify training here, because discovering that prefix still required the full prefix-completion sweep, and the prefix union did not improve coverage over ordinary full-code sampling at matched completion count. The strict gate therefore fails: this proposed prefix action space is not yet a useful enough MDP state representation for a learned value model.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
