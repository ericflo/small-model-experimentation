#!/usr/bin/env python3
"""Build final report and figures for the reliability/execution OPSD audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_records(path: Path) -> dict:
    rows = read_jsonl(path)
    records = len(rows)
    coverage = sum(1 for row in rows if row.get("coverage"))
    pass1 = sum(1 for row in rows if row.get("pass1_proxy"))
    visible = sum(1 for row in rows if row.get("visible_coverage"))
    visible_pass_candidates = 0
    visible_hidden_wrong = 0
    hidden_pass_candidates = 0
    functional_rates = []
    behavior_rates = []
    forward_tokens = 0
    candidate_count = 0
    for row in rows:
        functional_rates.append(row.get("distinct_functional_rate", 0.0))
        behavior_rates.append(row.get("distinct_behavior_rate", 0.0))
        token_usage = row.get("token_usage") or {}
        forward_tokens += int(token_usage.get("forward_tokens", 0))
        candidates = row.get("candidates") or []
        candidate_count += len(candidates)
        for candidate in candidates:
            if candidate.get("visible_all_pass"):
                visible_pass_candidates += 1
                if not candidate.get("full_pass"):
                    visible_hidden_wrong += 1
            if candidate.get("full_pass"):
                hidden_pass_candidates += 1
    return {
        "path": str(path.relative_to(ROOT)),
        "records": records,
        "coverage": coverage,
        "coverage_rate": coverage / records if records else 0.0,
        "pass1": pass1,
        "pass1_rate": pass1 / records if records else 0.0,
        "visible": visible,
        "visible_rate": visible / records if records else 0.0,
        "visible_pass_candidates": visible_pass_candidates,
        "visible_hidden_wrong": visible_hidden_wrong,
        "selected_false_pass_rate": (
            visible_hidden_wrong / visible_pass_candidates if visible_pass_candidates else 0.0
        ),
        "hidden_pass_candidates": hidden_pass_candidates,
        "candidate_count": candidate_count,
        "functional_rate_mean": (
            sum(functional_rates) / len(functional_rates) if functional_rates else 0.0
        ),
        "behavior_rate_mean": (
            sum(behavior_rates) / len(behavior_rates) if behavior_rates else 0.0
        ),
        "forward_tokens": forward_tokens,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt(value: float) -> str:
    return f"{value:.3f}"


def selector_table(summary: dict) -> str:
    lines = [
        "| selector | hidden-correct commits | hidden-wrong commits | commit rate | false-pass rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ["first_visible", "map_mean", "oracle_hidden"]:
        item = summary["selectors"][key]
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(item["selected_hidden_correct"]),
                    str(item["selected_visible_hidden_wrong"]),
                    pct(item["commit_rate"]),
                    pct(item["selected_false_pass_rate"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def plot_temperature(arms: list[tuple[str, dict]]) -> None:
    labels = [label for label, _ in arms]
    coverage = [arm["coverage"] for _, arm in arms]
    pass1 = [arm["pass1"] for _, arm in arms]
    x = range(len(labels))
    width = 0.36
    plt.figure(figsize=(7, 4))
    plt.bar([i - width / 2 for i in x], coverage, width=width, label="pool coverage")
    plt.bar([i + width / 2 for i in x], pass1, width=width, label="pass1 proxy")
    plt.xticks(list(x), labels)
    plt.ylim(0, 10)
    plt.ylabel("tasks out of 24")
    plt.title("Low-temperature semantic retrieval-adapt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "temperature_coverage.png", dpi=180)
    plt.close()


def plot_map(summary: dict, path: Path, title: str) -> None:
    labels = ["first_visible", "map_mean", "oracle_hidden"]
    correct = [summary["selectors"][label]["selected_hidden_correct"] for label in labels]
    wrong = [summary["selectors"][label]["selected_visible_hidden_wrong"] for label in labels]
    x = range(len(labels))
    plt.figure(figsize=(7, 4))
    plt.bar(x, correct, label="hidden-correct")
    plt.bar(x, wrong, bottom=correct, label="visible-pass hidden-wrong")
    plt.xticks(list(x), labels, rotation=15)
    plt.ylim(0, 16)
    plt.ylabel("selected candidates")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_fork_deltas(pressure: dict) -> None:
    contexts = ["exec_input_only", "exec_observation", "shuffled_exec", "full_reference"]
    labels = ["input only", "exec obs", "shuffled", "full ref"]
    task = [
        pressure["fork_summary"][f"{context}/task_specific"]["mean_delta_over_student"]
        for context in contexts
    ]
    hint = [
        pressure["fork_summary"][f"{context}/hint_overlap"]["mean_delta_over_student"]
        for context in contexts
    ]
    x = range(len(labels))
    width = 0.36
    plt.figure(figsize=(8, 4))
    plt.axhline(0, color="#555555", linewidth=0.8)
    plt.bar([i - width / 2 for i in x], task, width=width, label="task-specific forks")
    plt.bar([i + width / 2 for i in x], hint, width=width, label="hint-overlap forks")
    plt.xticks(list(x), labels)
    plt.ylabel("delta over no-hint student (nats)")
    plt.title("Teacher preference at position-matched forks")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fork_delta_over_student.png", dpi=180)
    plt.close()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    arms = [
        ("T=0.0", summarize_records(DATA / "retrieval_adapt_semantic_t0p0_top3_records.jsonl")),
        ("T=0.1", summarize_records(DATA / "retrieval_adapt_semantic_t0p1_top3_records.jsonl")),
        ("T=0.2", summarize_records(DATA / "retrieval_adapt_semantic_t0p2_top3_records.jsonl")),
    ]
    map_semantic = read_json(REPORTS / "map_selector_semantic_temps_summary.json")
    map_copy = read_json(REPORTS / "map_selector_copy_semantic_t0p2_summary.json")
    pair_summary = read_json(REPORTS / "exec_pair_summary.json")
    pressure = read_json(REPORTS / "exec_pressure_summary.json")
    gate = pressure["gate"]

    plot_temperature(arms)
    plot_map(
        map_semantic,
        FIGURES / "map_selector_semantic_temps.png",
        "MAP selector on semantic temperature union",
    )
    plot_map(
        map_copy,
        FIGURES / "map_selector_copy_semantic_t0p2.png",
        "MAP selector on copy plus semantic T=0.2",
    )
    plot_fork_deltas(pressure)

    temp_lines = [
        "| arm | coverage | pass1 proxy | visible coverage | visible-pass hidden-wrong candidates | functional diversity | forward tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in arms:
        temp_lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{summary['coverage']}/24 ({pct(summary['coverage_rate'])})",
                    f"{summary['pass1']}/24 ({pct(summary['pass1_rate'])})",
                    f"{summary['visible']}/24 ({pct(summary['visible_rate'])})",
                    f"{summary['visible_hidden_wrong']}/{summary['visible_pass_candidates']}",
                    pct(summary["functional_rate_mean"]),
                    f"{summary['forward_tokens']:,}",
                ]
            )
            + " |"
        )

    fork_lines = [
        "| context | task-specific preference | delta over student | fraction prefers correct |",
        "|---|---:|---:|---:|",
    ]
    for context, label in [
        ("exec_input_only", "failing input only"),
        ("exec_observation", "failing input + correct output"),
        ("shuffled_exec", "shuffled execution observation"),
        ("full_reference", "full-reference leakage ceiling"),
    ]:
        row = pressure["fork_summary"][f"{context}/task_specific"]
        fork_lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    fmt(row["mean_preference"]),
                    fmt(row["mean_delta_over_student"]),
                    pct(row["frac_prefers_correct"]),
                ]
            )
            + " |"
        )

    report = f"""# qwen35_4b_reliability_exec_opsd_audit

Date: 2026-06-26

## Decision

Do **not** train the execution-grounded OPSD variant from this audit. The position-matched fork gate failed: execution evidence with the correct output added +0.063 nats over the no-hint student on task-specific forks, while shuffled execution evidence added +0.079 nats. The full-reference leakage ceiling added +0.237 nats, so the audit is capable of seeing signal when the answer is leaked.

The reliability probes also failed to produce a deployable selector: raw MAP likelihood selected fewer hidden-correct candidates than first-visible in both candidate pools, and increased hidden-wrong visible-pass selections.

## Question

This no-training experiment tested two cheap hypotheses before any OPSD run:

1. If the base model already weakly prefers correct fork tokens, can lower-temperature decoding or raw model likelihood turn that into reliable retrieval-adapt selection?
2. If retrieval hints are only surface-level, can an execution-grounded teacher with counterexample input and correct output localize positive pressure onto task-specific discriminating tokens?

## Inputs

- Residual retrieval-adapt slice: 24 MBPP held-out tasks.
- Model: Qwen3.5-4B used as generator/scorer.
- Generated new semantic retrieval-adapt pools at T=0.0 and T=0.1, top-3 retrieved algorithms per task.
- Used existing semantic T=0.2, random, shuffled, and copy/rename pools as controls and pair sources.
- No model training was performed.

## Low-Temperature Probe

![Low-temperature coverage](figures/temperature_coverage.png)

{chr(10).join(temp_lines)}

Lower temperature did not dominate the existing semantic T=0.2 pool. Greedy decoding produced slightly better pass1 proxy but lower pool coverage than T=0.1/T=0.2. T=0.1 matched the best 8/24 coverage but lower pass1 proxy.

## MAP Likelihood Selector

MAP scoring used raw average token log-probability of each candidate code under the task prompt, then selected among visible-passing candidates.

Semantic temperature union:

![MAP semantic temps](figures/map_selector_semantic_temps.png)

{selector_table(map_semantic)}

Copy plus semantic T=0.2:

![MAP copy semantic](figures/map_selector_copy_semantic_t0p2.png)

{selector_table(map_copy)}

Result: MAP likelihood is not a reliable selector for this near-miss pool. It selected 6/24 hidden-correct candidates in both views, below first-visible's 8/24 and 7/24, and increased visible-pass hidden-wrong selections.

## Execution-Grounded OPSD Audit

Matched-pair builder found {pair_summary['matched_pairs']} correct-vs-hidden-wrong adaptation pairs across tasks {pair_summary['tasks']}. It produced {pair_summary['forks']} position-matched fork rows: {pair_summary['fork_strata']['task_specific']} task-specific and {pair_summary['fork_strata']['hint_overlap']} hint-overlap.

Gate:

- Passed: `{gate['passed']}`
- Reason: {gate['reason']}
- Task-specific forks: {gate['task_specific_forks']}

![Fork delta over student](figures/fork_delta_over_student.png)

{chr(10).join(fork_lines)}

Interpretation: the execution observation moves the model in the right direction a little, but not beyond shuffled execution evidence. The full-reference ceiling moves substantially more, confirming that the audit can detect a teacher that truly contains task-specific information.

## Readout

The task-specific fork result repeats the important reliability pattern from the prior audit: the no-hint student already strongly prefers the correct branch at almost every task-specific fork ({pct(gate['exec_task_specific_frac_prefers_correct'])} under execution-observation rows, same fork set), but the margin is not converted into reliable whole-program assembly. The missing ingredient is not a weak retrieval or execution hint that tells the teacher where the task-specific token is; these hints do not beat shuffled controls at the exact fork.

## Next Best Direction

The evidence points away from another token-credit training run and back toward adding independent behavioral evidence before selection. The most promising next experiment is independent-retrieval consensus: retrieve and adapt from several semantically distinct source algorithms, execute them on generated disagreement inputs, and commit only when independently sourced adaptations converge on outputs. That directly attacks the current selection wall with new evidence rather than another learned judge over thin public tests.

## Artifacts

- Records: `data/`
- Run logs: `run_logs/`
- Summaries and figures: `reports/`
- Experiment log: `logs/experiment_log.md`
- Large-artifact policy: large model/checkpoint/cache files are outside this directory; see `large_artifacts_manifest.md`.
"""

    (REPORTS / "final_report.md").write_text(report)


if __name__ == "__main__":
    main()
