#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from src.jsonl import load_json, write_json  # noqa: E402


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def load_many(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if path and path.exists():
            rows.append(load_json(path))
    return rows


def arm_label(row: dict[str, Any]) -> str:
    return str(row.get("arm_name") or row.get("run_name") or Path(row.get("path", "arm")).stem)


def plot_eval(rows: list[dict[str, Any]], out: Path) -> None:
    arms = [arm_label(row) for row in rows]
    coverage = [row.get("records", {}).get("coverage", 0.0) for row in rows]
    pass1 = [row.get("records", {}).get("pass1_proxy", 0.0) for row in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.1), 4.6))
    ax.bar([i - 0.18 for i in x], coverage, width=0.36, label="coverage@K", color="#2563eb")
    ax.bar([i + 0.18 for i in x], pass1, width=0.36, label="pass@1 proxy", color="#16a34a")
    ax.set_xticks(x, [item.replace("_", "\n") for item in arms], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Held-out Coverage and Pass@1 Guardrail")
    ax.legend()
    for idx, value in enumerate(coverage):
        ax.text(idx - 0.18, value + 0.02, pct(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_diversity(rows: list[dict[str, Any]], out: Path) -> None:
    arms = [arm_label(row) for row in rows]
    functional = [row.get("records", {}).get("distinct_functional_rate_mean", 0.0) for row in rows]
    behavior = [row.get("records", {}).get("distinct_behavior_rate_mean", 0.0) for row in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.1), 4.6))
    ax.bar([i - 0.18 for i in x], functional, width=0.36, label="functional diversity", color="#9333ea")
    ax.bar([i + 0.18 for i in x], behavior, width=0.36, label="behavior diversity", color="#f97316")
    ax.set_xticks(x, [item.replace("_", "\n") for item in arms], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean distinct rate")
    ax.set_title("Diversity Diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_pareto(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for row in rows:
        rec = row.get("records", {})
        tokens = row.get("token_usage", {}).get("forward_tokens") or rec.get("forward_tokens") or 0
        coverage = rec.get("coverage", 0.0)
        ax.scatter(tokens, coverage, s=70)
        ax.text(tokens, coverage + 0.015, arm_label(row), fontsize=8, ha="center")
    ax.set_xlabel("Forward tokens")
    ax.set_ylabel("coverage@K")
    ax.set_ylim(0, 1.05)
    ax.set_title("Coverage / Token Pareto")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_pairs(pair_summaries: list[dict[str, Any]], out: Path) -> None:
    labels = ["real" if not row.get("shuffle_labels") else "shuffled" for row in pair_summaries]
    pairs = [row.get("pairs", 0) for row in pair_summaries]
    tasks = [row.get("tasks_with_pairs", 0) for row in pair_summaries]
    x = list(range(len(pair_summaries)))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar([i - 0.18 for i in x], pairs, width=0.36, label="pairs", color="#2563eb")
    ax.bar([i + 0.18 for i in x], tasks, width=0.36, label="tasks with pairs", color="#16a34a")
    ax.set_xticks(x, labels)
    ax.set_title("Preference Pair Mining")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_training(train_logs: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for log in train_logs:
        label = log.get("run_name", "train")
        if log.get("metrics"):
            x = [row["step"] for row in log["metrics"]]
            y = [row["loss"] for row in log["metrics"]]
            ax.plot(x, y, label=label)
        elif log.get("losses"):
            x = [row["step"] for row in log["losses"]]
            y = [row["loss"] for row in log["losses"]]
            ax.plot(x, y, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training Loss")
    if train_logs:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def result_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| arm | K | coverage@K | pass@1 proxy | parse / task | visible coverage | functional diversity | forward tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rec = row.get("records", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    arm_label(row),
                    str(row.get("samples_per_task", "")),
                    pct(rec.get("coverage", 0.0)),
                    pct(rec.get("pass1_proxy", 0.0)),
                    f"{rec.get('parse_success_mean', 0.0):.2f}",
                    pct(rec.get("visible_coverage", 0.0)),
                    pct(rec.get("distinct_functional_rate_mean", 0.0)),
                    str(row.get("token_usage", {}).get("forward_tokens") or rec.get("forward_tokens") or 0),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def pair_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No pair-mining summaries were provided."
    lines = [
        "| pair set | pairs | tasks with pairs | visible-wrong pair rate |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        name = "shuffled" if row.get("shuffle_labels") else "real"
        lines.append(
            f"| {name} | {row.get('pairs', 0)} | {row.get('tasks_with_pairs', 0)} | {pct(row.get('visible_wrong_pair_rate', 0.0))} |"
        )
    return "\n".join(lines)


def gate_readout(rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> str:
    by_arm = {arm_label(row): row for row in rows}
    full_dpo = by_arm.get("hard_negative_dpo_k4")
    conservative = by_arm.get("conservative_dpo_k4")
    hot = next((row for name, row in by_arm.items() if "hot" in name and "base" in name), None)
    sample_more = by_arm.get("base_hot_k8_sample_more")
    shuffled = by_arm.get("conservative_shuffled_dpo_k4") or by_arm.get("shuffled_dpo_k4")
    sft = by_arm.get("positive_sft_k4")
    if pairs and pairs[0].get("pairs", 0) < 32:
        return "Pair mining produced fewer than 32 real pairs, so the training gate is not considered informative enough for a scaled run."
    if not hot:
        return "The tuned-hot base comparison was not available."
    hot_rec = hot.get("records", {})
    lines: list[str] = []
    if full_dpo:
        rec = full_dpo.get("records", {})
        lines.append(
            f"Aggressive 60-step DPO failed the parse guardrail: coverage {pct(rec.get('coverage', 0.0))}, "
            f"parse successes/task {rec.get('parse_success_mean', 0.0):.2f}."
        )
    if not conservative:
        return "\n".join(lines + ["No conservative DPO rescue arm was available."])
    dpo_rec = conservative.get("records", {})
    hot_rec = hot.get("records", {})
    delta = dpo_rec.get("coverage", 0.0) - hot_rec.get("coverage", 0.0)
    div_delta = dpo_rec.get("distinct_functional_rate_mean", 0.0) - hot_rec.get("distinct_functional_rate_mean", 0.0)
    pass_delta = dpo_rec.get("pass1_proxy", 0.0) - hot_rec.get("pass1_proxy", 0.0)
    lines.extend(
        [
            f"Conservative 10-step DPO coverage {pct(dpo_rec.get('coverage', 0.0))} vs tuned-hot K=4 {pct(hot_rec.get('coverage', 0.0))}, delta {pct(delta)}.",
            f"Guardrails for conservative DPO: pass@1 delta {pct(pass_delta)}, functional-diversity delta {pct(div_delta)}, parse successes/task {dpo_rec.get('parse_success_mean', 0.0):.2f}.",
        ]
    )
    if sample_more:
        sm_rec = sample_more.get("records", {})
        lines.append(f"Sample-more K=8 reference: {pct(sm_rec.get('coverage', 0.0))}.")
    if shuffled:
        sh_rec = shuffled.get("records", {})
        lines.append(f"Matched conservative shuffled-pair control coverage: {pct(sh_rec.get('coverage', 0.0))}.")
    if sft:
        sft_rec = sft.get("records", {})
        lines.append(f"Positive-only SFT coverage: {pct(sft_rec.get('coverage', 0.0))}.")
    if delta > 0 and pass_delta >= -0.02 and div_delta >= -0.05 and (not shuffled or dpo_rec.get("coverage", 0.0) > shuffled.get("records", {}).get("coverage", 0.0)):
        lines.append("Gate readout: weak pilot pass for the conservative DPO variant; scale only as a multi-seed replication because the lift is one task on n=24.")
    else:
        lines.append("Gate readout: fail for the pilot; this configuration should not be scaled without changing the mechanism.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-summary", action="append", type=Path, default=[])
    parser.add_argument("--pair-summary", action="append", type=Path, default=[])
    parser.add_argument("--train-log", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, default=ROOT / "reports/final_report.md")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "reports/report_summary.json")
    args = parser.parse_args()

    eval_rows = load_many(args.eval_summary)
    pair_rows = load_many(args.pair_summary)
    train_logs = load_many(args.train_log)
    fig_dir = ROOT / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if eval_rows:
        plot_eval(eval_rows, fig_dir / "coverage_pass1.png")
        plot_diversity(eval_rows, fig_dir / "diversity.png")
        plot_pareto(eval_rows, fig_dir / "coverage_token_pareto.png")
    if pair_rows:
        plot_pairs(pair_rows, fig_dir / "pair_mining.png")
    plot_training(train_logs, fig_dir / "training_loss.png")
    write_json(args.summary_out, {"eval": eval_rows, "pairs": pair_rows, "training": train_logs})

    report = f"""# qwen35_4b_offline_hard_negative_coverage_dpo

## Question

Can a small offline preference update, trained to prefer hidden-correct code over hard hidden-wrong candidates from the same task, improve held-out coverage from a fixed Qwen3.5-4B generator without collapsing useful sampling diversity?

## Pair Mining

{pair_table(pair_rows)}

![pair mining](figures/pair_mining.png)

## Held-Out Results

{result_table(eval_rows) if eval_rows else 'No held-out evaluation summaries were provided.'}

![coverage and pass1](figures/coverage_pass1.png)

![diversity](figures/diversity.png)

![pareto](figures/coverage_token_pareto.png)

## Training

![training loss](figures/training_loss.png)

## Gate Readout

{gate_readout(eval_rows, pair_rows)}

## Interpretation

This pilot is intentionally a gate, not a final benchmark. A positive result requires the real hard-negative DPO adapter to beat tuned-hot sampling at the same sample budget while preserving first-sample quality and functional diversity, and to beat a shuffled-pair control. A negative result is still informative: it means the available offline hidden-correct-vs-hard-negative signal did not convert into better held-out coverage under this small local adapter.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
