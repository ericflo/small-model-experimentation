#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DATA = ROOT / "data"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, color: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=color)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, min(v + 0.025, 0.97), pct(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    summary = load_json(REPORTS / "coverage_gate_summary.json")
    calibration = summary["calibration"]["overall"]
    heldout = summary["heldout"]["overall"]
    train = summary["train"]["overall"]
    sweep = summary["sweep"]
    heldout_rows = load_jsonl(DATA / "heldout_records.jsonl")

    bar(
        FIGURES / "coverage_main_splits.png",
        ["calibration", "heldout", "train", "sweep mean"],
        [
            calibration["oracle_coverage"],
            heldout["oracle_coverage"],
            train["oracle_coverage"],
            sweep["mean_coverage"],
        ],
        "Frozen Independent ABI Oracle Coverage",
        "Coverage",
        "#4c78a8",
    )

    bar(
        FIGURES / "coverage_sweep_seeds.png",
        [f"seed {item['seed']}" for item in sweep["results"]],
        [item["overall"]["oracle_coverage"] for item in sweep["results"]],
        "Held-Out Test-Suffix Coverage Sweep",
        "Coverage",
        "#f58518",
    )

    slices = sorted(summary["heldout"]["by_slice"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(slices))
    ax.bar(x, [summary["heldout"]["by_slice"][s]["oracle_coverage"] for s in slices], color="#54a24b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(slices, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Coverage")
    ax.set_title("Held-Out Coverage by Task Slice")
    for idx, s in enumerate(slices):
        value = summary["heldout"]["by_slice"][s]["oracle_coverage"]
        ax.text(idx, value + 0.025, pct(value), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "heldout_coverage_by_slice.png", dpi=160)
    plt.close(fig)

    depth_counts = Counter(str(row["winning_depth"]) for row in heldout_rows if row["oracle_covered"])
    depth_labels = sorted(depth_counts, key=int)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(depth_labels, [depth_counts[d] for d in depth_labels], color="#72b7b2")
    ax.set_xlabel("Winning ABI depth")
    ax.set_ylabel("Held-out covered tasks")
    ax.set_title("Held-Out Covered Tasks by Depth")
    for idx, d in enumerate(depth_labels):
        ax.text(idx, depth_counts[d] + 0.2, str(depth_counts[d]), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "heldout_depth_counts.png", dpi=160)
    plt.close(fig)

    bar(
        FIGURES / "false_pass_rates.png",
        ["calibration", "heldout", "train"],
        [
            calibration["visible_false_pass_rate_among_visible"],
            heldout["visible_false_pass_rate_among_visible"],
            train["visible_false_pass_rate_among_visible"],
        ],
        "Task-Level Visible-Pass Without Full Winner Rate",
        "Rate among visible-any tasks",
        "#e45756",
    )

    main_table = f"""| split | n | oracle-covered | oracle coverage | first-visible correct | visible-any tasks | task false-pass rate | candidate hidden-wrong rate | mean candidates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| calibration | {calibration['n']} | {calibration['oracle_covered']} | {pct(calibration['oracle_coverage'])} | {calibration['first_visible_full_pass']} | {calibration['visible_any']} | {pct(calibration['visible_false_pass_rate_among_visible'])} | {pct(calibration['visible_hidden_wrong_rate_among_candidates'])} | {calibration['candidate_count_mean']:.1f} |
| heldout | {heldout['n']} | {heldout['oracle_covered']} | {pct(heldout['oracle_coverage'])} | {heldout['first_visible_full_pass']} | {heldout['visible_any']} | {pct(heldout['visible_false_pass_rate_among_visible'])} | {pct(heldout['visible_hidden_wrong_rate_among_candidates'])} | {heldout['candidate_count_mean']:.1f} |
| train | {train['n']} | {train['oracle_covered']} | {pct(train['oracle_coverage'])} | {train['first_visible_full_pass']} | {train['visible_any']} | {pct(train['visible_false_pass_rate_among_visible'])} | {pct(train['visible_hidden_wrong_rate_among_candidates'])} | {train['candidate_count_mean']:.1f} |"""

    sweep_rows = "\n".join(
        f"| {item['seed']} | {item['overall']['n']} | {item['overall']['oracle_covered']} | {pct(item['overall']['oracle_coverage'])} | {item['overall']['first_visible_full_pass']} | {pct(item['overall']['visible_false_pass_rate_among_visible'])} |"
        for item in sweep["results"]
    )

    slice_rows = "\n".join(
        f"| {s} | {summary['heldout']['by_slice'][s]['n']} | {summary['heldout']['by_slice'][s]['oracle_covered']} | {pct(summary['heldout']['by_slice'][s]['oracle_coverage'])} | {summary['heldout']['by_slice'][s]['first_visible_full_pass']} | {pct(summary['heldout']['by_slice'][s]['visible_false_pass_rate_among_visible'])} |"
        for s in slices
    )

    covered_examples = [
        row
        for row in heldout_rows
        if row["oracle_covered"]
    ][:10]
    covered_rows = "\n".join(
        f"| {row['task_id']} | {row['slice']} | {row['winning_depth']} | `{json.dumps(row['winning_program'], sort_keys=True)}` | {row['task_text'][:90]} |"
        for row in covered_examples
    )

    report = f"""# Independent Code ABI Coverage Gate

## Purpose

This standalone no-training experiment tests whether an independently specified Python/stdlib-style ABI covers held-out MBPP tasks at a useful rate. The ABI was frozen before evaluation. No kernels were added after looking at held-out misses.

## Frozen ABI

The inventory contains generic Python operations: argument routing, list/tuple transforms, dictionary/counter utilities, regex/string transforms, predicates and label adapters, bounded arithmetic/combinatorics/bit utilities, and small generic compositions. At max arity 4 the inventory enumerates {summary['inventory']['candidate_count']} candidates: {summary['inventory']['by_category']}.

This is an oracle coverage gate, not a learned compiler and not a deployable solver. Coverage means at least one ABI candidate passes all available tests for a task.

## Headline Result

Held-out coverage is low. The frozen independent ABI covers {heldout['oracle_covered']}/{heldout['n']} held-out tasks ({pct(heldout['oracle_coverage'])}). The three-seed held-out sweep averages {pct(sweep['mean_coverage'])}, range {pct(sweep['min_coverage'])}-{pct(sweep['max_coverage'])}.

Calibration coverage is higher at {pct(calibration['oracle_coverage'])}, but the held-out slice and random sweep are the primary readout.

![Coverage main splits](figures/coverage_main_splits.png)

{main_table}

## Held-Out Sweep

| seed | n | oracle-covered | oracle coverage | first-visible correct | task false-pass rate |
|---:|---:|---:|---:|---:|---:|
{sweep_rows}

![Coverage sweep](figures/coverage_sweep_seeds.png)

## Slice Diagnostics

| slice | n | oracle-covered | oracle coverage | first-visible correct | task false-pass rate |
|---|---:|---:|---:|---:|---:|
{slice_rows}

![Held-out coverage by slice](figures/heldout_coverage_by_slice.png)

## Depth Diagnostics

Held-out covered tasks by winning depth: {dict(depth_counts)}. Most covered tasks are depth-1 single-primitive hits; held-out composition coverage is very small.

![Held-out depth counts](figures/heldout_depth_counts.png)

## False-Pass Pressure

Visible-test filtering is not reliable. On held-out tasks, {heldout['visible_any']} tasks have at least one visible-consistent candidate, but the task-level visible-pass/no-full-winner rate is {pct(heldout['visible_false_pass_rate_among_visible'])}, and the candidate-level hidden-wrong rate among visible-consistent candidates is {pct(heldout['visible_hidden_wrong_rate_among_candidates'])}.

![False pass rates](figures/false_pass_rates.png)

## Covered Held-Out Examples

| task | slice | depth | winning program | task |
|---:|---|---:|---|---|
{covered_rows}

Some covered tasks are likely test-suite coincidences rather than semantic equivalence. That makes the low held-out coverage an optimistic upper bound, not a pessimistic one.

## Decision

This gate does not support a code-ABI compiler-training run. A frozen independent ABI covers only about 14-18% of held-out MBPP tasks, mostly as depth-1 single-primitive matches. Training a compiler on this ABI would have too little held-out expressivity to test broad reusable code compilation.

The useful next step is not to add task-specific kernels. A better test would use a real independently curated task domain whose tasks are known to be deterministic transformations, then repeat this same held-out coverage gate.

## Files

- `scripts/run_independent_abi_gate.py`: frozen ABI evaluator and gate runner.
- `reports/coverage_gate_summary.json`: all summary metrics.
- `data/calibration_records.jsonl`, `data/heldout_records.jsonl`, `data/train_records.jsonl`: per-task records.
- `data/sweep_records.jsonl`: per-task sweep records.
- `reports/figures/`: generated charts.
"""
    (REPORTS / "final_report.md").write_text(report, encoding="utf-8")
    print(REPORTS / "final_report.md")


if __name__ == "__main__":
    main()
