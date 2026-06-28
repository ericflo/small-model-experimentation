#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from src.jsonl import load_jsonl
from src.jsonl import load_json


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def plot_coverage(summary: dict[str, Any], out: Path) -> None:
    rungs = list(summary["rungs"].keys())
    values = [summary["rungs"][r]["coverage"] for r in rungs]
    labels = [r.replace("_", "\n") for r in rungs]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, values, color=["#4d7c0f", "#0f766e", "#2563eb", "#7c3aed"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Hidden-test oracle coverage")
    ax.set_title("Substrate Ladder Coverage on K128-Residual Tasks")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, pct(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_false_pass(summary: dict[str, Any], out: Path) -> None:
    rungs = list(summary["rungs"].keys())
    visible = [summary["rungs"][r]["visible_pass_candidates"] for r in rungs]
    hidden = [summary["rungs"][r]["hidden_pass_candidates"] for r in rungs]
    false = [summary["rungs"][r]["visible_hidden_fail_candidates"] for r in rungs]
    labels = [r.replace("_", "\n") for r in rungs]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = range(len(rungs))
    ax.bar(x, hidden, label="hidden-pass", color="#15803d")
    ax.bar(x, false, bottom=hidden, label="visible-pass hidden-fail", color="#dc2626")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Candidate count")
    ax.set_title("Visible-Passing Candidates: True vs Spurious")
    ax.legend()
    for i, total in enumerate(visible):
        ax.text(i, total + max(1, total * 0.03), str(total), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_candidate_cost(summary: dict[str, Any], out: Path) -> None:
    rungs = list(summary["rungs"].keys())
    means = [summary["rungs"][r]["mean_candidates_per_task"] for r in rungs]
    labels = [r.replace("_", "\n") for r in rungs]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, means, color="#475569")
    ax.set_ylabel("Mean candidates per task")
    ax.set_title("Search Cost by Ladder Rung")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_task_heatmap(summary: dict[str, Any], out: Path) -> None:
    tasks = sorted(summary["per_task"].keys(), key=int)
    rungs = list(summary["rungs"].keys())
    data = [[1 if summary["per_task"][task][rung]["solved"] else 0 for rung in rungs] for task in tasks]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.imshow(data, cmap="Greens", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(rungs)), [r.replace("_", "\n") for r in rungs])
    ax.set_yticks(range(len(tasks)), tasks)
    ax.set_ylabel("MBPP task id")
    ax.set_title("Per-Task Hidden-Test Solve Matrix")
    for y, row in enumerate(data):
        for x, value in enumerate(row):
            ax.text(x, y, "1" if value else "0", ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def table_rows(summary: dict[str, Any]) -> str:
    lines = [
        "| rung | hidden coverage | solved | candidates/task | visible-pass | hidden-pass | visible-pass hidden-fail | false-pass rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rung, row in summary["rungs"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    rung,
                    pct(row["coverage"]),
                    str(row["solved_count"]),
                    f"{row['mean_candidates_per_task']:.1f}",
                    str(row["visible_pass_candidates"]),
                    str(row["hidden_pass_candidates"]),
                    str(row["visible_hidden_fail_candidates"]),
                    pct(row["visible_pass_hidden_fail_rate"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def per_task_rows(summary: dict[str, Any]) -> str:
    rungs = list(summary["rungs"].keys())
    lines = ["| task_id | " + " | ".join(rungs) + " | winning templates |", "|---:|" + "|".join(["---:" for _ in rungs]) + "|---|"]
    for task in sorted(summary["per_task"].keys(), key=int):
        cells = []
        winners: list[str] = []
        for rung in rungs:
            cell = summary["per_task"][task][rung]
            cells.append("yes" if cell["solved"] else "no")
            winners.extend(cell["winning_templates"])
        unique_winners = []
        for item in winners:
            if item not in unique_winners:
                unique_winners.append(item)
        lines.append(f"| {task} | " + " | ".join(cells) + " | " + ", ".join(unique_winners[:6]) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "data/main_substrate_results.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "reports/main_summary.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/dataset_manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/final_report.md")
    args = parser.parse_args()

    _ = load_jsonl(args.results)
    summary = load_json(args.summary)
    manifest = load_json(args.manifest)
    fig_dir = ROOT / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    coverage_fig = fig_dir / "coverage_by_rung.png"
    false_fig = fig_dir / "visible_false_passes.png"
    cost_fig = fig_dir / "candidate_cost.png"
    heatmap_fig = fig_dir / "task_solve_heatmap.png"

    plot_coverage(summary, coverage_fig)
    plot_false_pass(summary, false_fig)
    plot_candidate_cost(summary, cost_fig)
    plot_task_heatmap(summary, heatmap_fig)

    combined = summary["rungs"]["combined"]
    manual_expanded = summary["rungs"]["manual_expanded"]
    retrieved = summary["rungs"]["retrieved_transplant"]
    false_rate = combined["visible_pass_hidden_fail_rate"]
    residual_ids = manifest["k128_residual_task_ids"]

    weak_gate_status = "cleared" if combined["solved_count"] == summary["task_count"] else ("cleared partially" if combined["solved_count"] else "failed")
    strict_gate_status = "failed" if retrieved["solved_count"] == 0 else ("cleared partially" if retrieved["solved_count"] < summary["task_count"] else "cleared")

    report = f"""# qwen35_4b_substrate_coverage_ladder

## Question

Can an executable kernel/template substrate express MBPP held-out tasks that remained unsolved by a large direct-sampling candidate pool? This is an oracle-ceiling experiment: no model is trained, and hidden tests are used only to measure whether the substrate contains a correct graph/program and how often public-test filtering would produce spurious targets.

## Dataset

- Residual tasks: {len(residual_ids)} MBPP held-out tasks.
- Residual task ids: `{residual_ids}`.
- Train reference library used for retrieval/transplant controls: {manifest["train_library_count"]} train-split entries.
- Primary metric: hidden-test oracle coverage on the residual set.

## Results

{table_rows(summary)}

![coverage by rung](figures/coverage_by_rung.png)

![visible false passes](figures/visible_false_passes.png)

![candidate cost](figures/candidate_cost.png)

![task solve heatmap](figures/task_solve_heatmap.png)

## Per-Task Solve Matrix

{per_task_rows(summary)}

## Interpretation

The combined substrate solved {combined["solved_count"]}/{summary["task_count"]} residual tasks ({pct(combined["coverage"])} hidden-test oracle coverage), but this is only a weak expressivity ceiling. The coverage came from the `manual_expanded` rung ({manual_expanded["solved_count"]}/{summary["task_count"]}), whose winning templates are task-specific kernels such as top-k frequency, odd-bit masking, no-adjacent string rearrangement, and longest subsequence with adjacent difference. This should not be read as evidence that a reusable substrate generalizes to new residual tasks.

The stricter reusable-substrate readout failed. The `retrieved_transplant` rung, which retrieves train-split reference solutions and adapts their function signature, solved {retrieved["solved_count"]}/{summary["task_count"]} residual tasks. It produced {retrieved["visible_pass_candidates"]} public-test-passing candidates and all {retrieved["visible_hidden_fail_candidates"]} were hidden-wrong. That is evidence against the specific reuse hypothesis tested here.

The public-test trap is also visible. The combined arm produced {combined["visible_pass_candidates"]} public-test-passing candidates, but {combined["visible_hidden_fail_candidates"]} of those failed hidden tests, for a false-pass rate of {pct(false_rate)} among public-pass candidates. This is the load-bearing warning for any follow-up configurator: visible tests alone are not a trustworthy source of perfect latent programs.

## Gate Readout

- Weak Gate 1, any readable substrate expression: {weak_gate_status}.
- Strict Gate 1, reusable substrate expression: {strict_gate_status}.
- Gate 2, target trust: {'requires filtering beyond public tests' if combined["visible_hidden_fail_candidates"] else 'public-pass candidates were clean in this run'}.

## Next Step Implied By This Run

Do not train a graph configurator from this result. The meaningful reusable-substrate precondition did not clear, and training on these residual tasks would risk learning a lookup over task-specific hand-authored templates. A follow-up should first test transfer: author substrate templates on one residual slice, then measure hidden-test coverage on a disjoint slice for which no task-specific templates were authored. If that transfer coverage looks like the retrieved-transplant rung, the substrate is not a reusable frontier-expansion mechanism.

If any configurator is later tested, it should not train on arbitrary public-pass substrate candidates. It should either train only from hidden-verified train tasks or include a verifier strong enough to reject the public-pass hidden-fail cases.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
