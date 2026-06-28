#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl, write_json  # noqa: E402


EXP = "qwen35_4b_opsd_pressure_locality_audit"
ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS = ROOT_DIR / "reports"
FIGS = REPORTS / "figures"
DATA = ROOT_DIR / "data"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float) -> str:
    return f"{value:.3f}"


def make_figures(summary: dict[str, Any], fork_rows: list[dict[str, Any]]) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    contexts = ["weak_retrieved", "shuffled_retrieved", "full_reference"]
    labels = ["weak retrieved", "shuffled", "full ref"]
    strata = ["task_specific", "hint_overlap"]

    x = range(len(contexts))
    width = 0.35
    plt.figure(figsize=(8, 4.5))
    for offset, stratum in [(-width / 2, "task_specific"), (width / 2, "hint_overlap")]:
        values = [
            summary["fork_summary"][f"{context}/{stratum}"]["mean_delta_over_student"]
            for context in contexts
        ]
        plt.bar([i + offset for i in x], values, width=width, label=stratum.replace("_", " "))
    plt.axhline(0, color="#333333", linewidth=1)
    plt.xticks(list(x), labels)
    plt.ylabel("Teacher preference delta over no-hint student")
    plt.title("Hint adds signal mostly on hint-overlap forks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "fork_delta_over_student.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    for offset, stratum in [(-width / 2, "task_specific"), (width / 2, "hint_overlap")]:
        values = [
            summary["fork_summary"][f"{context}/{stratum}"]["mean_preference"]
            for context in contexts
        ]
        student_value = summary["fork_summary"][f"weak_retrieved/{stratum}"]["mean_student_preference"]
        plt.bar([i + offset for i in x], values, width=width, label=stratum.replace("_", " "))
        plt.axhline(student_value, color="#777777", linestyle=":", linewidth=1)
    plt.xticks(list(x), labels)
    plt.ylabel("log p(correct branch) - log p(wrong branch)")
    plt.title("Absolute teacher preference is confounded by student preference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "fork_absolute_preference.png", dpi=160)
    plt.close()

    token_keys = [
        ("weak_retrieved/correct/discriminating_correct", "weak correct discrim"),
        ("weak_retrieved/wrong/discriminating_wrong", "weak wrong discrim"),
        ("weak_retrieved/correct/shared_parse_format", "weak correct parse"),
        ("weak_retrieved/wrong/shared_parse_format", "weak wrong parse"),
        ("shuffled_retrieved/correct/discriminating_correct", "shuffled correct discrim"),
        ("full_reference/correct/discriminating_correct", "full-ref correct discrim"),
    ]
    values = [summary["token_summary"].get(key, {}).get("mean_positive_gap", 0.0) for key, _ in token_keys]
    labels = [label for _, label in token_keys]
    plt.figure(figsize=(9, 4.8))
    plt.bar(range(len(labels)), values, color=["#2f6f9f", "#c75f46", "#5a9e6f", "#b2772c", "#8b5fbf", "#444444"])
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("Mean positive gap")
    plt.title("Positive pressure by token bucket")
    plt.tight_layout()
    plt.savefig(FIGS / "token_positive_pressure.png", dpi=160)
    plt.close()

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in fork_rows:
        if row["context"] == "weak_retrieved":
            grouped[row["stratum"]].append(row["preference_mean"] - row["student_preference_mean"])
    plt.figure(figsize=(7, 4.5))
    data = [grouped["task_specific"], grouped["hint_overlap"]]
    plt.boxplot(data, showmeans=True)
    plt.xticks([1, 2], ["task specific", "hint overlap"])
    plt.axhline(0, color="#333333", linewidth=1)
    plt.ylabel("Weak-hint delta over student")
    plt.title("Fork-level distribution")
    plt.tight_layout()
    plt.savefig(FIGS / "weak_delta_distribution.png", dpi=160)
    plt.close()


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pair_summary = load_json(REPORTS / "pair_summary.json")
    pressure = load_json(REPORTS / "pressure_summary.json")
    fork_rows = load_jsonl(DATA / "fork_pressure_scores.jsonl")

    weak_task = pressure["fork_summary"]["weak_retrieved/task_specific"]
    weak_hint = pressure["fork_summary"]["weak_retrieved/hint_overlap"]
    shuffled_task = pressure["fork_summary"]["shuffled_retrieved/task_specific"]
    full_task = pressure["fork_summary"]["full_reference/task_specific"]
    gate = pressure["gate"]

    top_task_rows = [
        row
        for row in fork_rows
        if row["context"] == "weak_retrieved" and row["stratum"] == "task_specific"
    ]
    top_task_rows.sort(key=lambda row: row["preference_mean"] - row["student_preference_mean"])
    worst_examples = top_task_rows[:3]
    best_examples = top_task_rows[-3:]

    result = {
        "experiment": EXP,
        "pair_summary": pair_summary,
        "gate": gate,
        "weak_task_specific": weak_task,
        "weak_hint_overlap": weak_hint,
        "shuffled_task_specific": shuffled_task,
        "full_reference_task_specific": full_task,
        "usage_estimate": pressure["usage_estimate"],
        "primary_read": "static OPSD audit fails: weak hints do not add task-specific fork signal beyond the no-hint student",
    }
    make_figures(pressure, fork_rows)
    write_json(REPORTS / "report_summary.json", result)

    def rows_to_md(rows: list[dict[str, Any]]) -> str:
        lines = ["| task | correct branch | wrong branch | weak delta over student | weak preference | student preference |", "|---:|---|---|---:|---:|---:|"]
        for row in rows:
            delta = row["preference_mean"] - row["student_preference_mean"]
            lines.append(
                "| "
                + str(row["task_id"])
                + " | `"
                + row["correct_branch_preview"].replace("|", "\\|").replace("`", "'")
                + "` | `"
                + row["wrong_branch_preview"].replace("|", "\\|").replace("`", "'")
                + "` | "
                + fmt(delta)
                + " | "
                + fmt(row["preference_mean"])
                + " | "
                + fmt(row["student_preference_mean"])
                + " |"
            )
        return "\n".join(lines)

    report = f"""# {EXP}

## Motivation

This no-training audit tests whether positive-only on-policy self-distillation has the right token-localized signal before any adapter training is attempted. The target case is hidden-correct code versus visible-pass hidden-wrong near-misses for the same task.

The primary gate is same-prefix counterfactual branch preference: at executable code forks, does a weak hinted teacher prefer the hidden-correct branch over the hidden-wrong branch, and does that hint add preference beyond the no-hint student and shuffled-hint control?

## Data

- Matched correct/wrong pairs: {pair_summary['matched_pairs']}
- Tasks represented: {pair_summary['tasks']}
- Executable code forks scored: {pair_summary['forks']}
- Task-specific forks: {pair_summary['fork_strata'].get('task_specific', 0)}
- Hint-overlap forks: {pair_summary['fork_strata'].get('hint_overlap', 0)}
- Estimated scoring cost: {pressure['usage_estimate']['forward_tokens_estimate']} forward tokens across {pressure['usage_estimate']['scored_sequences']} scored sequences.

## Gate Result

**Gate: {'PASS' if gate['passed'] else 'FAIL'}**

{gate['reason']}.

| statistic | value |
|---|---:|
| weak task-specific absolute preference | {fmt(gate['weak_task_specific_mean'])} |
| weak task-specific delta over student | {fmt(gate['weak_task_specific_delta_over_student'])} |
| weak task-specific fraction prefers correct | {fmt(gate['weak_task_specific_frac_prefers_correct'])} |
| shuffled task-specific absolute preference | {fmt(gate['shuffled_task_specific_mean'])} |
| shuffled task-specific delta over student | {fmt(gate['shuffled_task_specific_delta_over_student'])} |
| full-reference task-specific absolute preference | {fmt(gate['full_reference_task_specific_mean'])} |
| full-reference task-specific delta over student | {fmt(gate['full_reference_task_specific_delta_over_student'])} |

The important distinction is absolute preference versus incremental signal. The no-hint student already prefers the correct task-specific branches by {fmt(weak_task['mean_student_preference'])} nats/token on average. The weak retrieved hint scores those branches at {fmt(weak_task['mean_preference'])}, which is a slight decrease ({fmt(weak_task['mean_delta_over_student'])}) rather than an added signal.

![Fork delta over student](figures/fork_delta_over_student.png)

![Absolute fork preference](figures/fork_absolute_preference.png)

## Fork Summary

| context / stratum | n | mean preference | mean student preference | delta over student | frac delta positive |
|---|---:|---:|---:|---:|---:|
| weak / task-specific | {weak_task['n']} | {fmt(weak_task['mean_preference'])} | {fmt(weak_task['mean_student_preference'])} | {fmt(weak_task['mean_delta_over_student'])} | {fmt(weak_task['frac_delta_over_student_positive'])} |
| weak / hint-overlap | {weak_hint['n']} | {fmt(weak_hint['mean_preference'])} | {fmt(weak_hint['mean_student_preference'])} | {fmt(weak_hint['mean_delta_over_student'])} | {fmt(weak_hint['frac_delta_over_student_positive'])} |
| shuffled / task-specific | {shuffled_task['n']} | {fmt(shuffled_task['mean_preference'])} | {fmt(shuffled_task['mean_student_preference'])} | {fmt(shuffled_task['mean_delta_over_student'])} | {fmt(shuffled_task['frac_delta_over_student_positive'])} |
| full-reference / task-specific | {full_task['n']} | {fmt(full_task['mean_preference'])} | {fmt(full_task['mean_student_preference'])} | {fmt(full_task['mean_delta_over_student'])} | {fmt(full_task['frac_delta_over_student_positive'])} |

The weak hint does add large signal on hint-overlap forks: {fmt(weak_hint['mean_delta_over_student'])}. That is exactly the retrieval-surface effect the audit was designed to catch. The effect does not transfer to task-specific forks.

![Weak delta distribution](figures/weak_delta_distribution.png)

## Token Pressure Buckets

| bucket | mean positive gap | positive rate | n |
|---|---:|---:|---:|
| weak correct discriminating | {fmt(pressure['token_summary']['weak_retrieved/correct/discriminating_correct']['mean_positive_gap'])} | {fmt(pressure['token_summary']['weak_retrieved/correct/discriminating_correct']['positive_rate'])} | {pressure['token_summary']['weak_retrieved/correct/discriminating_correct']['n']} |
| weak wrong discriminating | {fmt(pressure['token_summary']['weak_retrieved/wrong/discriminating_wrong']['mean_positive_gap'])} | {fmt(pressure['token_summary']['weak_retrieved/wrong/discriminating_wrong']['positive_rate'])} | {pressure['token_summary']['weak_retrieved/wrong/discriminating_wrong']['n']} |
| weak correct parse/format | {fmt(pressure['token_summary']['weak_retrieved/correct/shared_parse_format']['mean_positive_gap'])} | {fmt(pressure['token_summary']['weak_retrieved/correct/shared_parse_format']['positive_rate'])} | {pressure['token_summary']['weak_retrieved/correct/shared_parse_format']['n']} |
| weak wrong parse/format | {fmt(pressure['token_summary']['weak_retrieved/wrong/shared_parse_format']['mean_positive_gap'])} | {fmt(pressure['token_summary']['weak_retrieved/wrong/shared_parse_format']['positive_rate'])} | {pressure['token_summary']['weak_retrieved/wrong/shared_parse_format']['n']} |
| shuffled correct discriminating | {fmt(pressure['token_summary']['shuffled_retrieved/correct/discriminating_correct']['mean_positive_gap'])} | {fmt(pressure['token_summary']['shuffled_retrieved/correct/discriminating_correct']['positive_rate'])} | {pressure['token_summary']['shuffled_retrieved/correct/discriminating_correct']['n']} |
| full-reference correct discriminating | {fmt(pressure['token_summary']['full_reference/correct/discriminating_correct']['mean_positive_gap'])} | {fmt(pressure['token_summary']['full_reference/correct/discriminating_correct']['positive_rate'])} | {pressure['token_summary']['full_reference/correct/discriminating_correct']['n']} |

The rollout-level bucket view is more optimistic than the fork gate: weak hints give positive pressure to correct discriminating tokens overall. But the fork gate shows the crucial caveat: at task-specific same-prefix branches, the weak hint does not improve the student's preference. This means the broad token pressure is likely dominated by trajectory or retrieval-surface effects, not the local correctness bit needed for training.

![Token positive pressure](figures/token_positive_pressure.png)

## Example Forks

Worst weak-hint task-specific deltas:

{rows_to_md(worst_examples)}

Best weak-hint task-specific deltas:

{rows_to_md(best_examples)}

## Interpretation

This audit kills the immediate positive-only OPSD training run under the weak retrieved-hint setup.

The hinted teacher is not useless: it strongly moves probability on hint-overlap forks and broad correct-rollout discriminating tokens. But it does not add incremental task-specific branch knowledge beyond what the base student already assigns. That is the near-fatal failure mode for OPSD here: dense credit exists, but it is not localized at the hidden-correct bits that distinguish correct code from visible-pass hidden-wrong near-misses.

Full-reference hints do add task-specific signal, but that is a leakage ceiling. It does not justify deployable OPSD because the hint contains the answer and resembles gold/reference distillation rather than weak privileged guidance.

## Decision

Do not proceed to Stage-2 OPSD training on this weak-hint formulation.

The next experiment should either:

1. create stronger deployable evidence before distillation, such as independent retrieval-consensus or mined counterexample observations, then rerun this locality audit; or
2. change the teacher hint so it contains task-specific discriminating evidence without leaking the reference solution.

Until the static locality gate passes, training would likely amplify retrieved surface form and shared structure rather than teach the missing correctness bits.

## Artifacts

- `data/matched_pairs.jsonl`
- `data/fork_pressure_scores.jsonl`
- `data/token_pressure_scores.jsonl`
- `reports/pair_summary.json`
- `reports/pressure_summary.json`
- `reports/report_summary.json`
- `reports/figures/`
"""
    (REPORTS / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
