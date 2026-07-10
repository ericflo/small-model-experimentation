#!/usr/bin/env python3
"""Build compact terminal G0 tables and lessons from saved artifacts."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from io_utils import read_json, read_jsonl, write_json  # noqa: E402
from stats import mean, roc_auc  # noqa: E402

CAL = EXP / "runs" / "calibration"
OUT = EXP / "analysis"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def task_auc(rows: list[dict[str, Any]]) -> tuple[float | None, int]:
    values = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    for task_rows in by_task.values():
        labels = []
        scores = []
        for row in task_rows:
            labels.extend(bool(outcome["correct"]) for outcome in row["outcomes"])
            scores.extend([float(row["gain_sum"])] * len(row["outcomes"]))
        value = roc_auc(labels, scores)
        if value is not None:
            values.append(value)
    return (mean(values) if values else None, len(values))


def main() -> int:
    gate = read_json(CAL / "g0.json")
    thoughts = read_jsonl(CAL / "thoughts.jsonl")
    potential = read_jsonl(CAL / "potential.jsonl")
    rollouts = read_jsonl(CAL / "rollouts.jsonl")
    diagnostics = read_json(CAL / "premember_diagnostics.json")
    score_by_trace = {row["trace_id"]: row for row in potential}
    rollout_by_trace = {row["trace_id"]: row for row in rollouts}
    joined = [
        {**row, **score_by_trace[row["trace_id"]], **rollout_by_trace[row["trace_id"]]}
        for row in thoughts
        if row["trace_id"] in score_by_trace
    ]
    outcomes = [outcome for row in rollouts for outcome in row["outcomes"]]
    parsed = [outcome for outcome in outcomes if outcome["answer_value"] is not None]

    top = gate["metrics"]["top_one"]
    auroc = gate["metrics"]["within_task_auroc"]
    controls = gate["metrics"]["controls"]
    format_metric = gate["metrics"]["format_kendall_tau"]
    premember = gate["metrics"]["premember"]
    metric_rows = [
        {
            "metric": "within_task_answer_gain_auroc",
            "observed": auroc["gain_task_macro"],
            "threshold": gate["thresholds"]["within_task_auroc_min"],
            "passed": gate["criteria"]["auroc"],
            "detail": f"{auroc['mixed_tasks_gain']} mixed tasks",
        },
        {
            "metric": "top_gain_minus_seeded_random",
            "observed": top["gain_minus_random"]["mean_delta"],
            "threshold": gate["thresholds"]["top1_uplift_min"],
            "passed": gate["criteria"]["top1_random_uplift"],
            "detail": f"95% CI [{top['gain_minus_random']['ci95_low']:.6f}, {top['gain_minus_random']['ci95_high']:.6f}]",
        },
        {
            "metric": "top_gain_minus_shortest",
            "observed": top["gain_minus_shortest"]["mean_delta"],
            "threshold": gate["thresholds"]["top1_uplift_min"],
            "passed": gate["criteria"]["top1_shortest_uplift"],
            "detail": f"95% CI [{top['gain_minus_shortest']['ci95_low']:.6f}, {top['gain_minus_shortest']['ci95_high']:.6f}]",
        },
        {
            "metric": "beats_length_and_trace_prior",
            "observed": auroc["gain_task_macro"] - auroc["negative_length_task_macro"],
            "threshold": "strictly beats both",
            "passed": gate["criteria"]["beats_length_and_prior_auroc"],
            "detail": "gain beat length; trace-prior logprob was not captured, so this criterion failed closed",
        },
        {
            "metric": "real_minus_token_shuffled_gain_nats",
            "observed": controls["token_shuffled"]["bootstrap"]["mean_delta"],
            "threshold": "95% lower > 0",
            "passed": gate["criteria"]["beats_token_shuffled"],
            "detail": f"95% CI [{controls['token_shuffled']['bootstrap']['ci95_low']:.6f}, {controls['token_shuffled']['bootstrap']['ci95_high']:.6f}]",
        },
        {
            "metric": "real_minus_foreign_gain_nats",
            "observed": controls["foreign"]["bootstrap"]["mean_delta"],
            "threshold": "95% lower > 0",
            "passed": gate["criteria"]["beats_foreign"],
            "detail": f"95% CI [{controls['foreign']['bootstrap']['ci95_low']:.6f}, {controls['foreign']['bootstrap']['ci95_high']:.6f}]",
        },
        {
            "metric": "format_rank_kendall_tau",
            "observed": format_metric["task_macro"],
            "threshold": gate["thresholds"]["kendall_tau_min"],
            "passed": gate["criteria"]["format_rank_stability"],
            "detail": f"{format_metric['n_tasks']} tasks",
        },
        {
            "metric": "positive_before_answer_mention_fraction",
            "observed": premember["fraction"],
            "threshold": gate["thresholds"]["premember_fraction_min"],
            "passed": gate["criteria"]["premember"],
            "detail": f"{sum(d['no_answer_mention'] for d in diagnostics.values())} selected traces never mentioned the answer",
        },
    ]
    write_csv(
        OUT / "g0_metrics.csv",
        ["metric", "observed", "threshold", "passed", "detail"],
        metric_rows,
    )

    family_rows = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_family[row["family"]].append(row)
    for family in sorted(by_family):
        rows = by_family[family]
        family_outcomes = [outcome for row in rows for outcome in row["outcomes"]]
        family_parsed = [
            outcome for outcome in family_outcomes if outcome["answer_value"] is not None
        ]
        auc, mixed = task_auc(rows)
        family_rows.append(
            {
                "family": family,
                "tasks": len({row["task_id"] for row in rows}),
                "mixed_tasks": mixed,
                "task_macro_auroc": "" if auc is None else auc,
                "mean_answer_gain": mean([float(row["gain_sum"]) for row in rows]),
                "rollout_accuracy": mean([float(outcome["correct"]) for outcome in family_outcomes]),
                "parse_rate": mean([float(outcome["answer_value"] is not None) for outcome in family_outcomes]),
                "parsed_conditional_accuracy": (
                    mean([float(outcome["correct"]) for outcome in family_parsed])
                    if family_parsed
                    else ""
                ),
            }
        )
    write_csv(
        OUT / "family_summary.csv",
        [
            "family",
            "tasks",
            "mixed_tasks",
            "task_macro_auroc",
            "mean_answer_gain",
            "rollout_accuracy",
            "parse_rate",
            "parsed_conditional_accuracy",
        ],
        family_rows,
    )

    selected_rows = []
    joined_by_trace = {row["trace_id"]: row for row in joined}
    for task_id, trace_id in sorted(gate["selected_trace_ids"].items()):
        row = joined_by_trace[trace_id]
        selected_rows.append(
            {
                "task_id": task_id,
                "family": row["family"],
                "level": row["level"],
                "trace_id": trace_id,
                "gain_sum": row["gain_sum"],
                "rollout_success_fraction": row["success_fraction"],
                "n_tokens": row["n_tokens"],
                "natural_close": row["natural_close"],
                "no_answer_mention": diagnostics[task_id]["no_answer_mention"],
                "premember_pass": premember["passed_by_task"][task_id],
            }
        )
    write_csv(
        OUT / "selected_trace_summary.csv",
        [
            "task_id",
            "family",
            "level",
            "trace_id",
            "gain_sum",
            "rollout_success_fraction",
            "n_tokens",
            "natural_close",
            "no_answer_mention",
            "premember_pass",
        ],
        selected_rows,
    )

    stage_names = [
        "thoughts",
        "potential",
        "potential_format_variant",
        "potential_token_shuffled",
        "potential_foreign",
        "rollouts",
        "potential_premember",
    ]
    stage_meta = {name: read_json(CAL / f"{name}.meta.json") for name in stage_names}
    final_counts = stage_meta["potential_premember"]["logical_counts"]
    compute = {
        "stage_elapsed_seconds": {
            name: stage_meta[name]["elapsed_seconds"] for name in stage_names
        },
        "gpu_operation_elapsed_seconds_sum": sum(
            stage_meta[name]["elapsed_seconds"] for name in stage_names
        ),
        "final_cumulative_logical_counts": final_counts,
        "total_counted_logical_tokens": sum(final_counts.values()),
    }
    write_json(OUT / "compute_summary.json", compute)

    summary = {
        "schema_version": 1,
        "verdict": "SCORER_NEGATIVE",
        "g0_passed": False,
        "criteria_passed": sum(bool(value) for value in gate["criteria"].values()),
        "criteria_total": len(gate["criteria"]),
        "n_calibration_prompts": 64,
        "n_scorable_prompts": gate["metrics"]["n_tasks"],
        "n_thoughts": len(thoughts),
        "natural_close_count": sum(bool(row["natural_close"]) for row in thoughts),
        "natural_close_rate": mean([float(row["natural_close"]) for row in thoughts]),
        "cap_contact_count": sum(int(row["n_tokens"]) == 512 for row in thoughts),
        "cap_contact_rate": mean([float(int(row["n_tokens"]) == 512) for row in thoughts]),
        "mean_thought_tokens": statistics.mean(int(row["n_tokens"]) for row in thoughts),
        "median_thought_tokens": statistics.median(int(row["n_tokens"]) for row in thoughts),
        "rollout_outcomes": len(outcomes),
        "rollout_accuracy": mean([float(outcome["correct"]) for outcome in outcomes]),
        "rollout_parse_rate": mean([float(outcome["answer_value"] is not None) for outcome in outcomes]),
        "parsed_conditional_accuracy": mean([float(outcome["correct"]) for outcome in parsed]),
        "canonical_decoy_positive_margin_rate": mean(
            [float(row["canonical_decoy_margin"] > 0.0) for row in potential]
        ),
        "trace_prior_logprob_available": auroc["prior_mean_logprob_available"],
        "full_harvest_run": False,
        "sft_run": False,
        "reason_stopped": "preregistered G0 failed",
    }
    write_json(OUT / "g0_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
