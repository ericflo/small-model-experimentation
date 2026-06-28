#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def load_summary(name: str) -> dict:
    return json.loads((REPORTS / f"coverage_summary_{name}.json").read_text())


def load_records(name: str) -> list[dict]:
    path = ROOT / "data" / f"abi_coverage_records_{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, ylim: tuple[float, float] = (0, 1)) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, pct(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    initial = load_summary("initial")
    expanded = load_summary("expanded")
    final = load_summary("final")
    final_rows = load_records("final")
    summaries = [("core", initial), ("expanded", expanded), ("final", final)]

    save_bar(
        FIGURES / "oracle_coverage_ladder.png",
        [name for name, _ in summaries],
        [summary["overall"]["oracle_coverage"] for _, summary in summaries],
        "Oracle Test-Suite Coverage by ABI Rung",
        "Coverage",
    )

    save_bar(
        FIGURES / "first_visible_ladder.png",
        [name for name, _ in summaries],
        [summary["overall"]["first_visible_full_pass"] / summary["overall"]["n"] for _, summary in summaries],
        "First Visible-Consistent Candidate Accuracy",
        "Accuracy",
    )

    slices = sorted(final["by_slice"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(slices))
    ax.bar(x, [final["by_slice"][s]["oracle_coverage"] for s in slices], color="#54a24b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(slices, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Coverage")
    ax.set_title("Final ABI Oracle Coverage by Task Slice")
    for idx, s in enumerate(slices):
        value = final["by_slice"][s]["oracle_coverage"]
        ax.text(idx, value + 0.02, pct(value), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_coverage_by_slice.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, [final["by_slice"][s]["visible_false_pass_rate_among_visible"] for s in slices], color="#e45756")
    ax.set_xticks(list(x))
    ax.set_xticklabels(slices, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Task-Level Rate")
    ax.set_title("Final Visible-Pass / No-Full-Winner Rate by Slice")
    for idx, s in enumerate(slices):
        value = final["by_slice"][s]["visible_false_pass_rate_among_visible"]
        ax.text(idx, value + 0.02, pct(value), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "final_visible_false_pass_by_slice.png", dpi=160)
    plt.close(fig)

    depth_counts = Counter(row["winning_depth"] for row in final_rows if row["oracle_covered"])
    labels = [str(k) for k in sorted(depth_counts)]
    values = [depth_counts[k] for k in sorted(depth_counts)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#72b7b2")
    ax.set_title("Winning ABI Program Depths")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Covered Tasks")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.5, str(value), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "winning_depths_final.png", dpi=160)
    plt.close(fig)

    table_rows = []
    for name, summary in summaries:
        overall = summary["overall"]
        table_rows.append(
            "| {name} | {covered}/{n} ({cov}) | {first}/{n} ({first_pct}) | {vis_any} | {task_false} ({task_false_pct}) | {cand_wrong} ({cand_wrong_pct}) | {cand_mean:.1f} |".format(
                name=name,
                covered=overall["oracle_covered"],
                n=overall["n"],
                cov=pct(overall["oracle_coverage"]),
                first=overall["first_visible_full_pass"],
                first_pct=pct(overall["first_visible_full_pass"] / overall["n"]),
                vis_any=overall["visible_any"],
                task_false=overall["visible_false_pass"],
                task_false_pct=pct(overall["visible_false_pass_rate_among_visible"]),
                cand_wrong=overall["visible_hidden_wrong_candidates"],
                cand_wrong_pct=pct(overall["visible_hidden_wrong_rate_among_candidates"]),
                cand_mean=overall["candidate_count_mean"],
            )
        )

    slice_rows = []
    for s in slices:
        m = final["by_slice"][s]
        slice_rows.append(
            f"| {s} | {m['oracle_covered']}/{m['n']} ({pct(m['oracle_coverage'])}) | "
            f"{m['first_visible_full_pass']}/{m['n']} ({pct(m['first_visible_full_pass'] / m['n'])}) | "
            f"{m['visible_false_pass']} ({pct(m['visible_false_pass_rate_among_visible'])}) | "
            f"{m['candidate_count_mean']:.1f} |"
        )

    uncovered = ", ".join(str(x) for x in final["uncovered_task_ids"])
    covered_by_depth = ", ".join(f"{depth}: {count}" for depth, count in sorted(depth_counts.items()))

    report = f"""# Code ABI Oracle Coverage Ladder

## Purpose

This standalone no-training experiment asks whether a finite verified code-primitive ABI can express a meaningful slice of MBPP-style Python tasks before spending any compute on a compiler model.

The run uses the first 160 records from the MBPP test split. Each task gets one visible test for visible-consistency accounting, while oracle coverage is measured against every available test in the task record. The experiment does not train Qwen, does not save checkpoints, and does not use reference source code as a template.

## ABI Rungs

- `core`: small generic string/list/dict/numeric kernels.
- `expanded`: broad reusable kernels for row-sum sorting, counters, simple geometry, bit operations, list transforms, and regex/string transforms.
- `final`: expanded plus reusable algorithmic utility kernels such as sequence recurrences, tuple/list conversions, range sums, run-length encoding, divisor/bit counts, and small dynamic reducers.

## Headline Results

| rung | oracle coverage | first visible candidate correct | visible-any tasks | task-level false-visible-pass | candidate-level hidden-wrong among visible | mean candidates/task |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

![Oracle coverage ladder](figures/oracle_coverage_ladder.png)

![First-visible ladder](figures/first_visible_ladder.png)

## Final Rung by Slice

| slice | oracle coverage | first visible candidate correct | task-level false-visible-pass | mean candidates/task |
|---|---:|---:|---:|---:|
{chr(10).join(slice_rows)}

![Final coverage by slice](figures/final_coverage_by_slice.png)

![Final false pass by slice](figures/final_visible_false_pass_by_slice.png)

## Winning Program Depths

Covered task count by winning ABI depth: {covered_by_depth}.

![Winning depths](figures/winning_depths_final.png)

## Gate Read

The oracle coverage gate clears for the intended decomposable-code direction: the final reusable ABI covers 134/160 tasks (83.75%) under the available test suites, with coverage above 79% in every automatically classified slice except `other`.

The result does not make the ABI a deployable solver by itself. First-visible selection is only 95/160 (59.4%), and candidate-level hidden-wrong pressure among visible-consistent candidates remains high at 628 wrong visible-consistent candidates out of 793 total visible-consistent candidates. This means compiler training should use constrained decoding plus verification and should not rely on one public test or first visible-consistent selection.

The strongest caution is that this is test-suite oracle coverage, not a proof of semantic equivalence. Some candidates can satisfy all available tests while being semantically too broad, too narrow, or coincidentally correct. The package keeps these candidates visible in `data/abi_coverage_records_final.jsonl` so those cases can be audited.

## Remaining Uncovered Tasks

Uncovered task IDs after the final rung:

{uncovered}

The residual is a mix of specialized number theory, bespoke formulas, tasks with ambiguous or thin test suites, and algorithmic problems outside the current ABI. The next step should not be to keep adding one-off kernels indefinitely. The useful next gate is a compiler-training pilot on this frozen final ABI, paired with stronger generated counterexample tests for covered tasks and a held-out slice that excludes primitives added after inspection.

## Files

- `configs/experiment.json`: run configuration.
- `scripts/run_coverage_ladder.py`: ABI oracle evaluator.
- `scripts/make_report.py`: report and figure generator.
- `data/abi_coverage_records_initial.jsonl`: core rung records.
- `data/abi_coverage_records_expanded.jsonl`: expanded rung records.
- `data/abi_coverage_records_final.jsonl`: final rung records.
- `reports/coverage_summary_initial.json`: core summary.
- `reports/coverage_summary_expanded.json`: expanded summary.
- `reports/coverage_summary_final.json`: final summary.
- `reports/figures/`: generated charts.
"""

    (REPORTS / "final_report.md").write_text(report)
    print(REPORTS / "final_report.md")


if __name__ == "__main__":
    main()
