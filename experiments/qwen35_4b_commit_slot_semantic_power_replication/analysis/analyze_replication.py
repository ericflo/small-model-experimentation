#!/usr/bin/env python3
"""Deterministic audit of the two independently gated seam stages.

Pooled quantities are descriptive only.  They never replace either frozen
stage decision.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


EXPERIMENT = Path(__file__).resolve().parents[1]
RUNS = EXPERIMENT / "runs"
OUTPUT = EXPERIMENT / "analysis" / "replication_audit.json"
STAGES = ("seam_selection", "seam_confirmation")
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260712


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_interval(
    values: list[float], rng: random.Random
) -> dict[str, float | int]:
    n = len(values)
    draws = [
        mean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    return {
        "mean": mean(values),
        "two_sided_95_lower": quantile(draws, 0.025),
        "two_sided_95_upper": quantile(draws, 0.975),
        "resamples": BOOTSTRAP_RESAMPLES,
        "unit": "task",
    }


def task_means(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(float(row["correct"]))
    return {task_id: mean(values) for task_id, values in grouped.items()}


def stage_audit(stage: str) -> tuple[dict, list[float], list[float]]:
    summary = load_json(RUNS / f"{stage}.json")
    real = load_jsonl(RUNS / f"{stage}_slot_rows.jsonl")
    shuffled = load_jsonl(RUNS / f"{stage}_shuffled_slot_rows.jsonl")
    no_thought = load_jsonl(RUNS / f"{stage}_no_thought_rows.jsonl")

    real_by_key = {(row["task_id"], row["trace_index"]): row for row in real}
    shuffled_by_key = {
        (row["task_id"], row["trace_index"]): row for row in shuffled
    }
    if real_by_key.keys() != shuffled_by_key.keys():
        raise RuntimeError(f"{stage}: real/shuffled path keys differ")

    real_task = task_means(real)
    shuffled_task = task_means(shuffled)
    no_thought_task = task_means(no_thought)
    if not (real_task.keys() == shuffled_task.keys() == no_thought_task.keys()):
        raise RuntimeError(f"{stage}: task keys differ across controls")

    shuffled_differences = [
        real_task[task_id] - shuffled_task[task_id] for task_id in sorted(real_task)
    ]
    no_thought_differences = [
        real_task[task_id] - no_thought_task[task_id] for task_id in sorted(real_task)
    ]

    paired_counts = {"both_correct": 0, "real_only": 0, "shuffled_only": 0, "neither": 0}
    for key, real_row in real_by_key.items():
        pair = (bool(real_row["correct"]), bool(shuffled_by_key[key]["correct"]))
        label = {
            (True, True): "both_correct",
            (True, False): "real_only",
            (False, True): "shuffled_only",
            (False, False): "neither",
        }[pair]
        paired_counts[label] += 1

    sign_counts = {
        "positive": sum(value > 0 for value in shuffled_differences),
        "zero": sum(value == 0 for value in shuffled_differences),
        "negative": sum(value < 0 for value in shuffled_differences),
    }

    alias_rows: dict[str, list[dict]] = defaultdict(list)
    alias_shuffled: dict[str, list[dict]] = defaultdict(list)
    for row in real:
        alias_rows[row["correct_alias"]].append(row)
    for row in shuffled:
        alias_shuffled[row["correct_alias"]].append(row)
    alias_results = {}
    for alias in sorted(alias_rows):
        alias_results[alias] = {
            "rows": len(alias_rows[alias]),
            "real_successes": sum(bool(row["correct"]) for row in alias_rows[alias]),
            "real_accuracy": mean(bool(row["correct"]) for row in alias_rows[alias]),
            "shuffled_successes": sum(
                bool(row["correct"]) for row in alias_shuffled[alias]
            ),
            "shuffled_accuracy": mean(
                bool(row["correct"]) for row in alias_shuffled[alias]
            ),
        }

    mention_results = {}
    for mentioned in (False, True):
        subset = [
            row for row in real if bool(row["thought_contains_correct_alias"]) == mentioned
        ]
        mention_results[str(mentioned).lower()] = {
            "rows": len(subset),
            "successes": sum(bool(row["correct"]) for row in subset),
            "accuracy": mean(bool(row["correct"]) for row in subset),
        }

    rng = random.Random(BOOTSTRAP_SEED + STAGES.index(stage))
    if "metrics" in summary:
        metrics = summary["metrics"]
    elif (
        isinstance(summary.get("metrics_by_cap"), list)
        and len(summary["metrics_by_cap"]) == 1
        and summary["metrics_by_cap"][0].get("cap") == 1024
    ):
        metrics = summary["metrics_by_cap"][0]
    else:
        raise RuntimeError(f"{stage}: unknown summary metrics schema")
    audit = {
        "decision": summary["decision"],
        "passed": summary["passed"],
        "tasks": summary["items"],
        "paths": len(real),
        "real": {
            "successes": sum(bool(row["correct"]) for row in real),
            "accuracy": mean(bool(row["correct"]) for row in real),
        },
        "shuffled": {
            "successes": sum(bool(row["correct"]) for row in shuffled),
            "accuracy": mean(bool(row["correct"]) for row in shuffled),
        },
        "no_thought": {
            "successes": sum(bool(row["correct"]) for row in no_thought),
            "accuracy": mean(bool(row["correct"]) for row in no_thought),
        },
        "real_minus_shuffled_task_bootstrap": bootstrap_mean_interval(
            shuffled_differences, rng
        ),
        "real_minus_no_thought_task_bootstrap": bootstrap_mean_interval(
            no_thought_differences, rng
        ),
        "registered_one_sided_95_lower_real_minus_shuffled": metrics[
            "task_real_minus_shuffled_one_sided_95_lower"
        ],
        "paired_path_outcomes": paired_counts,
        "task_effect_signs_real_minus_shuffled": sign_counts,
        "alias_results": alias_results,
        "correct_alias_mention_strata": mention_results,
        "interface": {
            "full_vocab_top_is_alias_rate": metrics["full_vocab_top_is_alias_rate"],
            "mean_alias_probability_mass": metrics[
                "mean_full_vocab_alias_probability_mass"
            ],
            "close_only_parse_rate": metrics["close_only_parse_rate"],
            "close_only_success_rate": metrics["close_only_success_rate"],
        },
    }
    return audit, shuffled_differences, no_thought_differences


def main() -> None:
    stage_results = {}
    shuffled_differences = {}
    no_thought_differences = {}
    for stage in STAGES:
        stage_results[stage], shuffled_differences[stage], no_thought_differences[stage] = (
            stage_audit(stage)
        )

    rng = random.Random(BOOTSTRAP_SEED + 10)
    selection = shuffled_differences["seam_selection"]
    confirmation = shuffled_differences["seam_confirmation"]
    effect_difference_draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        selection_draw = mean(
            selection[rng.randrange(len(selection))] for _ in range(len(selection))
        )
        confirmation_draw = mean(
            confirmation[rng.randrange(len(confirmation))]
            for _ in range(len(confirmation))
        )
        effect_difference_draws.append(confirmation_draw - selection_draw)

    pooled_shuffled = selection + confirmation
    pooled_no_thought = (
        no_thought_differences["seam_selection"]
        + no_thought_differences["seam_confirmation"]
    )
    pooled_rng = random.Random(BOOTSTRAP_SEED + 20)
    output = {
        "schema_version": 1,
        "scientific_result": False,
        "purpose": "deterministic post-decision audit; stage decisions remain independent",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "stages": stage_results,
        "cross_stage": {
            "both_independent_registered_gates_passed": all(
                stage_results[stage]["passed"] for stage in STAGES
            ),
            "confirmation_minus_selection_real_minus_shuffled": {
                "mean": mean(confirmation) - mean(selection),
                "two_sided_95_lower": quantile(effect_difference_draws, 0.025),
                "two_sided_95_upper": quantile(effect_difference_draws, 0.975),
                "resamples": BOOTSTRAP_RESAMPLES,
                "unit": "independently resampled task",
            },
        },
        "pooled_diagnostic_not_a_rescue": {
            "tasks": len(pooled_shuffled),
            "real_minus_shuffled_task_bootstrap": bootstrap_mean_interval(
                pooled_shuffled, pooled_rng
            ),
            "real_minus_no_thought_task_bootstrap": bootstrap_mean_interval(
                pooled_no_thought, pooled_rng
            ),
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
