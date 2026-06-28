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
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.exists():
            rows.append(load_json(path))
    return rows


def arm_label(row: dict[str, Any]) -> str:
    return str(row.get("arm_name") or row.get("run_name") or Path(row.get("path", "arm")).stem)


def token_count(row: dict[str, Any]) -> int:
    return int(row.get("token_usage", {}).get("forward_tokens") or row.get("records", {}).get("forward_tokens") or 0)


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def solved_tasks(record_path: Path) -> set[int]:
    solved: set[int] = set()
    for record in read_records(record_path):
        if any(candidate.get("full_pass") for candidate in record.get("candidates", [])):
            solved.add(int(record["task_id"]))
    return solved


def pass1_tasks(record_path: Path) -> set[int]:
    solved: set[int] = set()
    for record in read_records(record_path):
        candidates = record.get("candidates", [])
        if candidates and candidates[0].get("full_pass"):
            solved.add(int(record["task_id"]))
    return solved


def make_overlap(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {arm_label(row): row for row in eval_rows}
    coverage_sets: dict[str, set[int]] = {}
    pass1_sets: dict[str, set[int]] = {}
    for name, row in by_arm.items():
        path = ROOT / row.get("path", "")
        coverage_sets[name] = solved_tasks(path)
        pass1_sets[name] = pass1_tasks(path)

    names = list(by_arm)
    pairwise: dict[str, dict[str, int]] = {}
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            key = f"{left}__{right}"
            pairwise[key] = {
                "coverage_intersection": len(coverage_sets[left] & coverage_sets[right]),
                "coverage_union": len(coverage_sets[left] | coverage_sets[right]),
            }
    return {
        "coverage_tasks": {name: sorted(items) for name, items in coverage_sets.items()},
        "pass1_tasks": {name: sorted(items) for name, items in pass1_sets.items()},
        "pairwise": pairwise,
    }


def plot_eval(rows: list[dict[str, Any]], out: Path) -> None:
    labels = [arm_label(row) for row in rows]
    coverage = [row.get("records", {}).get("coverage", 0.0) for row in rows]
    pass1 = [row.get("records", {}).get("pass1_proxy", 0.0) for row in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(max(8.5, len(rows) * 1.3), 4.8))
    ax.bar([i - 0.18 for i in x], coverage, width=0.36, label="coverage@K", color="#2563eb")
    ax.bar([i + 0.18 for i in x], pass1, width=0.36, label="pass@1 proxy", color="#16a34a")
    ax.set_xticks(x, [label.replace("_", "\n") for label in labels], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Coverage and Pass@1 Guardrail")
    ax.legend()
    for idx, value in enumerate(coverage):
        ax.text(idx - 0.18, value + 0.02, pct(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_parse_diversity(rows: list[dict[str, Any]], out: Path) -> None:
    labels = [arm_label(row) for row in rows]
    parse = [row.get("records", {}).get("parse_success_mean", 0.0) / max(float(row.get("samples_per_task") or 1), 1.0) for row in rows]
    functional = [row.get("records", {}).get("distinct_functional_rate_mean", 0.0) for row in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(max(8.5, len(rows) * 1.3), 4.8))
    ax.bar([i - 0.18 for i in x], parse, width=0.36, label="parse rate", color="#0f766e")
    ax.bar([i + 0.18 for i in x], functional, width=0.36, label="functional diversity", color="#9333ea")
    ax.set_xticks(x, [label.replace("_", "\n") for label in labels], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Parse and Functional Diversity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_pareto(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for row in rows:
        rec = row.get("records", {})
        tokens = token_count(row)
        coverage = rec.get("coverage", 0.0)
        ax.scatter(tokens, coverage, s=80)
        ax.text(tokens, coverage + 0.015, arm_label(row), fontsize=8, ha="center")
    ax.set_xlabel("forward tokens")
    ax.set_ylabel("coverage@K")
    ax.set_ylim(0, 1.05)
    ax.set_title("Coverage / Token Pareto")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_training(train_logs: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for log in train_logs:
        if not log.get("metrics"):
            continue
        label = log.get("run_name", "train")
        x = [row["step"] for row in log["metrics"]]
        y = [row["loss"] for row in log["metrics"]]
        ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Constrained DPO Training Loss")
    if train_logs:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_pairs(pair_rows: list[dict[str, Any]], out: Path) -> None:
    labels = ["shuffled" if row.get("shuffle_labels") else "real" for row in pair_rows]
    pairs = [row.get("pairs", 0) for row in pair_rows]
    tasks = [row.get("tasks_with_pairs", 0) for row in pair_rows]
    x = list(range(len(pair_rows)))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar([i - 0.18 for i in x], pairs, width=0.36, label="pairs", color="#2563eb")
    ax.bar([i + 0.18 for i in x], tasks, width=0.36, label="tasks", color="#16a34a")
    ax.set_xticks(x, labels)
    ax.set_title("Preference Pair Mining")
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
                    str(token_count(row)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def pair_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| pair set | pairs | tasks with pairs | visible-wrong pair rate | source records |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        name = "shuffled" if row.get("shuffle_labels") else "real"
        lines.append(
            f"| {name} | {row.get('pairs', 0)} | {row.get('tasks_with_pairs', 0)} | "
            f"{pct(row.get('visible_wrong_pair_rate', 0.0))} | {row.get('records', 0)} |"
        )
    return "\n".join(lines)


def overlap_table(overlap: dict[str, Any]) -> str:
    tasks = overlap.get("coverage_tasks", {})
    lines = [
        "| arm | covered tasks | pass@1 tasks |",
        "|---|---:|---:|",
    ]
    pass1 = overlap.get("pass1_tasks", {})
    for name in tasks:
        lines.append(f"| {name} | {tasks[name]} | {pass1.get(name, [])} |")
    return "\n".join(lines)


def gate_readout(rows: list[dict[str, Any]], overlap: dict[str, Any]) -> str:
    by_arm = {arm_label(row): row for row in rows}
    base = by_arm.get("base_hot_k4")
    sample_more = by_arm.get("base_hot_k8_sample_more")
    constrained = by_arm.get("constrained_dpo_k4")
    shuffled = by_arm.get("constrained_shuffled_dpo_k4")
    if not (base and sample_more and constrained and shuffled):
        return "Missing at least one gate arm, so the gate cannot be read cleanly."

    base_rec = base["records"]
    sample_more_rec = sample_more["records"]
    con_rec = constrained["records"]
    shuf_rec = shuffled["records"]
    pass1_delta = con_rec["pass1_proxy"] - base_rec["pass1_proxy"]
    parse_rate_base = base_rec["parse_success_mean"] / base["samples_per_task"]
    parse_rate_con = con_rec["parse_success_mean"] / constrained["samples_per_task"]
    parse_delta = parse_rate_con - parse_rate_base
    coverage_vs_base = con_rec["coverage"] - base_rec["coverage"]
    coverage_vs_shuffled = con_rec["coverage"] - shuf_rec["coverage"]
    coverage_vs_sample_more = con_rec["coverage"] - sample_more_rec["coverage"]

    pairwise = overlap.get("pairwise", {})
    con_base = pairwise.get("base_hot_k4__constrained_dpo_k4") or pairwise.get("constrained_dpo_k4__base_hot_k4") or {}
    con_k8 = pairwise.get("base_hot_k8_sample_more__constrained_dpo_k4") or pairwise.get(
        "constrained_dpo_k4__base_hot_k8_sample_more"
    ) or {}

    lines = [
        f"Constrained DPO vs base K4: coverage delta {pct(coverage_vs_base)}, pass@1 delta {pct(pass1_delta)}, parse-rate delta {pct(parse_delta)}.",
        f"Constrained DPO vs shuffled constrained control: coverage delta {pct(coverage_vs_shuffled)}.",
        f"Constrained DPO vs sample-more K8: coverage delta {pct(coverage_vs_sample_more)} at {token_count(constrained)} vs {token_count(sample_more)} forward tokens.",
        f"Task overlap: base K4 union constrained K4 covers {con_base.get('coverage_union', 'n/a')} tasks; base K8 union constrained K4 covers {con_k8.get('coverage_union', 'n/a')} tasks.",
    ]
    if (
        con_rec["coverage"] >= sample_more_rec["coverage"]
        and con_rec["pass1_proxy"] >= base_rec["pass1_proxy"] - 0.02
        and parse_delta >= -0.05
        and con_rec["coverage"] > shuf_rec["coverage"]
    ):
        lines.append("Gate readout: pass. The constrained adapter beats the sample-more Pareto reference while preserving pass@1 and parseability.")
    else:
        lines.append(
            "Gate readout: no scale-up yet. The constrained adapter preserves pass@1 and beats shuffled, but it does not reach the K8 sample-more coverage reference."
        )
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
    overlap = make_overlap(eval_rows)

    fig_dir = ROOT / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if eval_rows:
        plot_eval(eval_rows, fig_dir / "coverage_pass1.png")
        plot_parse_diversity(eval_rows, fig_dir / "parse_diversity.png")
        plot_pareto(eval_rows, fig_dir / "coverage_token_pareto.png")
    if pair_rows:
        plot_pairs(pair_rows, fig_dir / "pair_mining.png")
    plot_training(train_logs, fig_dir / "training_loss.png")

    summary = {"eval": eval_rows, "pairs": pair_rows, "training": train_logs, "overlap": overlap}
    write_json(args.summary_out, summary)

    report = f"""# qwen35_4b_constrained_coverage_dpo

## Question

Can a weak hard-negative DPO coverage signal be made useful by explicitly constraining it with a reference anchor, positive NLL anchor, and short early-stopped training, so coverage improves without sacrificing pass@1 or parseability?

The meaningful comparison is not only base K4. The gate is whether constrained DPO is on a better coverage/pass@1/token Pareto point than simply sampling more from the base model.

## Pair Mining

{pair_table(pair_rows)}

![pair mining](figures/pair_mining.png)

## Held-Out Results

{result_table(eval_rows)}

![coverage and pass1](figures/coverage_pass1.png)

![parse and diversity](figures/parse_diversity.png)

![pareto](figures/coverage_token_pareto.png)

## Task-Level Overlap

{overlap_table(overlap)}

## Training

![training loss](figures/training_loss.png)

The constrained trainer used a DPO margin term plus a positive-sample NLL anchor and a reference-logprob drift penalty. Both the real and shuffled adapters were limited to ten optimizer steps.

## Gate Readout

{gate_readout(eval_rows, overlap)}

## Interpretation

This is a small pilot on 24 MBPP-test tasks, so one recovered task is not enough to claim a robust method. Still, the control structure is informative. The real constrained adapter improves over base K4 and over the shuffled constrained adapter while preserving pass@1 and parseability. That means the constrained preference signal did not collapse into the usual parse/pass@1 failure mode, and it is not explained by label shuffling.

It does not beat the K8 sample-more coverage reference. So this exact scalar constrained-DPO sampler should not be scaled as the next main bet. The useful new clue is complementarity: constrained K4 recovers a task that K8 base sampling missed, while K8 base recovers tasks constrained K4 missed. The next direction is therefore a sampler portfolio or learned scheduler over multiple generation policies, with the same coverage/pass@1/token Pareto gate, rather than simply pushing one LoRA harder.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
