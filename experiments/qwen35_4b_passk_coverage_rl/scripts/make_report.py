#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from src.jsonl import load_json, write_json


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def read_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if path.exists():
            rows.append(load_json(path))
    return rows


def plot_metrics(rows: list[dict[str, Any]], out: Path) -> None:
    arms = [row["arm_name"] for row in rows]
    coverage = [row["records"].get("coverage", 0.0) for row in rows]
    pass1 = [row["records"].get("pass1_proxy", 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = range(len(arms))
    ax.bar([i - 0.18 for i in x], coverage, width=0.36, label="coverage@K", color="#2563eb")
    ax.bar([i + 0.18 for i in x], pass1, width=0.36, label="pass@1 proxy", color="#16a34a")
    ax.set_xticks(list(x), [arm.replace("_", "\n") for arm in arms])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Held-out Coverage and First-Sample Guardrail")
    ax.legend()
    for i, value in enumerate(coverage):
        ax.text(i - 0.18, value + 0.02, pct(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_diversity(rows: list[dict[str, Any]], out: Path) -> None:
    arms = [row["arm_name"] for row in rows]
    functional = [row["records"].get("mean_distinct_functional_rate", 0.0) for row in rows]
    program = [row["records"].get("mean_distinct_program_rate", 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = range(len(arms))
    ax.bar([i - 0.18 for i in x], functional, width=0.36, label="functional diversity", color="#9333ea")
    ax.bar([i + 0.18 for i in x], program, width=0.36, label="program diversity", color="#f97316")
    ax.set_xticks(list(x), [arm.replace("_", "\n") for arm in arms])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean distinct rate")
    ax.set_title("Diversity Diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_training(metrics: dict[str, Any] | None, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    if metrics and metrics.get("metrics"):
        steps = [row["step"] for row in metrics["metrics"]]
        utilities = [row["passk_utility"] for row in metrics["metrics"]]
        positives = [row["positive_count"] for row in metrics["metrics"]]
        ax.plot(steps, utilities, marker="o", label="Pass@K utility", color="#2563eb")
        ax.plot(steps, positives, marker="s", label="full-pass rollouts", color="#16a34a")
        ax.legend()
    ax.set_xlabel("Training step")
    ax.set_title("Online Training Signals")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def result_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| arm | split | K | coverage@K | pass@1 proxy | visible coverage | functional diversity | forward tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rec = row["records"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["arm_name"],
                    row["split"],
                    str(row["samples_per_task"]),
                    pct(rec.get("coverage", 0.0)),
                    pct(rec.get("pass1_proxy", 0.0)),
                    pct(rec.get("visible_coverage", 0.0)),
                    pct(rec.get("mean_distinct_functional_rate", 0.0)),
                    str(rec.get("forward_tokens", 0)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def training_readout(training: dict[str, Any] | None) -> str:
    if not training or not training.get("metrics"):
        return "No training metrics were provided."
    metrics = training["metrics"]
    updates = [row for row in metrics if not row.get("skipped")]
    skipped_zero = sum(1 for row in metrics if row.get("skip_reason") == "zero_positive")
    skipped_sat = sum(1 for row in metrics if row.get("skip_reason") == "saturated_positive")
    positives = [row.get("positive_count", 0) for row in updates]
    utilities = [row.get("passk_utility", 0.0) for row in updates]
    return (
        f"Training attempted {training.get('attempts', len(metrics))} rollout groups and took "
        f"{training.get('updates', len(updates))} updates. It skipped {skipped_zero} zero-positive "
        f"groups and {skipped_sat} saturated-positive groups. Among update groups, positive rollouts "
        f"ranged from {min(positives) if positives else 0} to {max(positives) if positives else 0}; "
        f"mean Pass@K utility was {sum(utilities) / len(utilities):.3f}." if updates else
        f"Training attempted {training.get('attempts', len(metrics))} rollout groups but took no updates."
    )


def comparison_readout(rows: list[dict[str, Any]]) -> str:
    by_arm = {row["arm_name"]: row for row in rows}
    adapter = by_arm.get("passk_rl_unsat_k4") or by_arm.get("passk_rl_v2_k4") or by_arm.get("passk_rl_adapter_k4")
    hot = by_arm.get("base_hot_k4")
    matched = by_arm.get("base_t09_k4")
    sample_more = by_arm.get("base_t09_k8_sample_more")
    lines = []
    if adapter and hot:
        delta = adapter["records"]["coverage"] - hot["records"]["coverage"]
        lines.append(f"- Adapter vs tuned-hot K=4: {pct(adapter['records']['coverage'])} vs {pct(hot['records']['coverage'])}, delta {pct(delta)}.")
    if adapter and matched:
        delta = adapter["records"]["coverage"] - matched["records"]["coverage"]
        lines.append(f"- Adapter vs same-temperature base K=4: {pct(adapter['records']['coverage'])} vs {pct(matched['records']['coverage'])}, delta {pct(delta)}.")
    if sample_more:
        lines.append(f"- Sample-more reference K=8: {pct(sample_more['records']['coverage'])} at {sample_more['records'].get('forward_tokens', 0)} forward tokens.")
    if adapter:
        lines.append(
            f"- Adapter guardrails: pass@1 proxy {pct(adapter['records'].get('pass1_proxy', 0.0))}, "
            f"functional diversity {pct(adapter['records'].get('mean_distinct_functional_rate', 0.0))}."
        )
    if not lines:
        return "No pilot comparison arms were found."
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", type=Path, default=[])
    parser.add_argument("--training-metrics", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/final_report.md")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "reports/report_summary.json")
    args = parser.parse_args()

    rows = read_summaries(args.summary)
    training = load_json(args.training_metrics) if args.training_metrics and args.training_metrics.exists() else None
    fig_dir = ROOT / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        plot_metrics(rows, fig_dir / "coverage_pass1.png")
        plot_diversity(rows, fig_dir / "diversity.png")
    plot_training(training, fig_dir / "training_signals.png")
    write_json(args.summary_out, {"arms": rows, "training": training})

    best = max(rows, key=lambda row: row["records"].get("coverage", 0.0)) if rows else None
    report = f"""# qwen35_4b_passk_coverage_rl

## Question

Can a small online QLoRA update aimed at set-level coverage make cheap K-sample generation cover more held-out MBPP tasks than tuned inference-only sampling?

## Results

{result_table(rows) if rows else 'No evaluation summaries were found.'}

![coverage and pass1](figures/coverage_pass1.png)

![diversity](figures/diversity.png)

![training signals](figures/training_signals.png)

## Readout

Best observed held-out coverage arm: {best['arm_name'] if best else 'none'}.

{comparison_readout(rows)}

## Training Diagnostics

{training_readout(training)}

## Interpretation

The decisive pilot comparison is the Pass@K adapter versus tuned-hot base sampling at matched K. A positive result requires the adapter to improve coverage without collapsing first-sample quality or functional diversity. In this pilot, the adapter did not pass that gate if tuned-hot K=4 is available and higher. The useful finding is diagnostic: online Pass@K reward can be obtained, but sparse or saturated rollout groups dominate, and the small adapter did not convert that reward into held-out coverage.

## Gate Decision

The pilot gate failed, so the experiment stops here rather than scaling a larger training run. Scaling this exact online RL variant would spend compute on a configuration that is already dominated by inference-only hot sampling at matched K in the pilot.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
