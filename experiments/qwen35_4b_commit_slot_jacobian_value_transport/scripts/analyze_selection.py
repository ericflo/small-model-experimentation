#!/usr/bin/env python3
"""Derive aggregate commit-slot selection diagnostics without reading trace text."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


EXP = Path(__file__).resolve().parents[1]
RUNS = EXP / "runs"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_interval(values: list[float], *, seed: int, draws: int = 10_000) -> list[float]:
    generator = random.Random(seed)
    samples = sorted(
        mean([values[generator.randrange(len(values))] for _ in values])
        for _ in range(draws)
    )
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]]


def rate_slice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "successes": sum(bool(row["correct"]) for row in rows),
        "success_rate": mean([float(bool(row["correct"])) for row in rows]),
        "mean_correct_alias_probability": mean(
            [float(row["correct_alias_probability"]) for row in rows]
        ),
    }


def residual_choice(
    real_probabilities: dict[str, float], control_probabilities: dict[str, float]
) -> str:
    aliases = sorted(real_probabilities)
    return max(
        aliases,
        key=lambda alias: (
            math.log(max(float(real_probabilities[alias]), 1e-30))
            - math.log(max(float(control_probabilities[alias]), 1e-30)),
            alias,
        ),
    )


def fixed_bias_choice(real_probabilities: dict[str, float], bias: dict[str, float]) -> str:
    aliases = sorted(real_probabilities)
    return max(
        aliases,
        key=lambda alias: (
            math.log(max(float(real_probabilities[alias]), 1e-30)) - bias[alias],
            alias,
        ),
    )


def policy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    return {
        "rows": len(rows),
        "successes": sum(bool(row["correct"]) for row in rows),
        "success_rate": mean([float(bool(row["correct"])) for row in rows]),
        "mixed_tasks": sum(
            any(item["correct"] for item in items)
            and any(not item["correct"] for item in items)
            for items in by_task.values()
        ),
        "chosen_alias_histogram": dict(
            sorted(Counter(row["chosen_alias"] for row in rows).items())
        ),
    }


def main() -> int:
    summary = read_json(RUNS / "seam_selection.json")
    if summary.get("decision") != "COMMIT_SLOT_SEAM_FAIL":
        raise RuntimeError("analysis is frozen to the terminal failed selection")
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text(encoding="utf-8"))
    cap = 1024
    real = [
        row for row in read_jsonl(RUNS / "seam_selection_slot_rows.jsonl")
        if int(row["cap"]) == cap
    ]
    shuffled = [
        row for row in read_jsonl(RUNS / "seam_selection_shuffled_slot_rows.jsonl")
        if int(row["cap"]) == cap
    ]
    no_thought = read_jsonl(RUNS / "seam_selection_no_thought_rows.jsonl")
    all_real = read_jsonl(RUNS / "seam_selection_slot_rows.jsonl")
    real_by_key = {(row["task_id"], int(row["trace_index"])): row for row in real}
    shuffled_by_key = {
        (row["task_id"], int(row["trace_index"])): row for row in shuffled
    }
    no_thought_by_task = {row["task_id"]: row for row in no_thought}
    global_no_thought_log_bias = {
        alias: mean([
            math.log(max(float(row["alias_probabilities"][alias]), 1e-30))
            for row in no_thought
        ])
        for alias in sorted(no_thought[0]["alias_probabilities"])
    }
    by_task_real: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task_shuffled: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in real:
        by_task_real[row["task_id"]].append(row)
    for row in shuffled:
        by_task_shuffled[row["task_id"]].append(row)

    task_rows = []
    task_real_minus_shuffled = []
    task_real_minus_no_thought = []
    for task_id in sorted(by_task_real):
        real_rate = mean([float(bool(row["correct"])) for row in by_task_real[task_id]])
        shuffled_rate = mean(
            [float(bool(row["correct"])) for row in by_task_shuffled[task_id]]
        )
        no_thought_rate = float(bool(no_thought_by_task[task_id]["correct"]))
        task_real_minus_shuffled.append(real_rate - shuffled_rate)
        task_real_minus_no_thought.append(real_rate - no_thought_rate)
        task_rows.append({
            "task_id": task_id,
            "correct_alias": by_task_real[task_id][0]["correct_alias"],
            "real_successes": int(round(real_rate * 3)),
            "shuffled_successes": int(round(shuffled_rate * 3)),
            "no_thought_success": int(no_thought_rate),
            "real_minus_shuffled": real_rate - shuffled_rate,
            "real_minus_no_thought": real_rate - no_thought_rate,
        })

    real_wins = sum(
        bool(real_by_key[key]["correct"]) and not bool(shuffled_by_key[key]["correct"])
        for key in real_by_key
    )
    shuffled_wins = sum(
        not bool(real_by_key[key]["correct"]) and bool(shuffled_by_key[key]["correct"])
        for key in real_by_key
    )
    cap_256 = {
        (row["task_id"], int(row["trace_index"])): row
        for row in all_real if int(row["cap"]) == 256
    }
    cap_improvements = sum(
        bool(real_by_key[key]["correct"]) and not bool(cap_256[key]["correct"])
        for key in real_by_key
    )
    cap_regressions = sum(
        not bool(real_by_key[key]["correct"]) and bool(cap_256[key]["correct"])
        for key in real_by_key
    )

    correct_mention = [row for row in real if row["thought_contains_correct_alias"]]
    no_correct_mention = [row for row in real if not row["thought_contains_correct_alias"]]
    any_mention = [row for row in real if row["thought_contains_any_alias"]]
    no_mention = [row for row in real if not row["thought_contains_any_alias"]]
    unmasked_alias_top = [row for row in real if row["full_vocab_top_is_alias"]]
    unmasked_non_alias_top = [row for row in real if not row["full_vocab_top_is_alias"]]

    post_hoc_residual_policies = {}
    for policy_cap in (256, 512, 1024):
        cap_real = [row for row in all_real if int(row["cap"]) == policy_cap]
        cap_shuffled = {
            (row["task_id"], int(row["trace_index"])): row
            for row in read_jsonl(RUNS / "seam_selection_shuffled_slot_rows.jsonl")
            if int(row["cap"]) == policy_cap
        }
        residual_no_thought = []
        residual_shuffled = []
        global_calibrated = []
        for row in cap_real:
            key = (row["task_id"], int(row["trace_index"]))
            choice_no_thought = residual_choice(
                row["alias_probabilities"],
                no_thought_by_task[row["task_id"]]["alias_probabilities"],
            )
            choice_shuffled = residual_choice(
                row["alias_probabilities"], cap_shuffled[key]["alias_probabilities"]
            )
            residual_no_thought.append({
                "task_id": row["task_id"],
                "chosen_alias": choice_no_thought,
                "correct": choice_no_thought == row["correct_alias"],
            })
            residual_shuffled.append({
                "task_id": row["task_id"],
                "chosen_alias": choice_shuffled,
                "correct": choice_shuffled == row["correct_alias"],
            })
            choice_global = fixed_bias_choice(
                row["alias_probabilities"], global_no_thought_log_bias
            )
            global_calibrated.append({
                "task_id": row["task_id"],
                "chosen_alias": choice_global,
                "correct": choice_global == row["correct_alias"],
            })
        post_hoc_residual_policies[str(policy_cap)] = {
            "real_minus_no_thought_logits": policy_metrics(residual_no_thought),
            "real_minus_shuffled_logits": policy_metrics(residual_shuffled),
            "real_minus_global_no_thought_log_bias": policy_metrics(global_calibrated),
        }

    aliases = sorted(set(row["correct_alias"] for row in real))
    by_alias = {}
    for alias in aliases:
        alias_real = [row for row in real if row["correct_alias"] == alias]
        alias_shuffled = [row for row in shuffled if row["correct_alias"] == alias]
        alias_no_thought = [row for row in no_thought if row["correct_alias"] == alias]
        by_alias[alias] = {
            "real": rate_slice(alias_real),
            "shuffled": rate_slice(alias_shuffled),
            "no_thought": rate_slice(alias_no_thought),
        }

    bootstrap_seed = int(config["seeds"]["bootstrap"])
    result = {
        "schema_version": 1,
        "scientific_result": False,
        "diagnostic_only_cannot_rescue_gate": True,
        "terminal_decision": summary["decision"],
        "focus_cap": cap,
        "failed_gate": {
            "name": "mixed_slot_tasks_min",
            "observed": 5,
            "required": 6,
            "all_other_selection_gates_passed": True,
        },
        "paired_trace_real_vs_shuffled": {
            "real_only_correct": real_wins,
            "shuffled_only_correct": shuffled_wins,
            "net_successes": real_wins - shuffled_wins,
            "ties": len(real) - real_wins - shuffled_wins,
        },
        "paired_trace_1024_vs_256": {
            "1024_only_correct": cap_improvements,
            "256_only_correct": cap_regressions,
            "net_successes": cap_improvements - cap_regressions,
            "ties": len(real) - cap_improvements - cap_regressions,
        },
        "task_macro_differences": {
            "real_minus_shuffled_mean": mean(task_real_minus_shuffled),
            "real_minus_shuffled_bootstrap_95": bootstrap_interval(
                task_real_minus_shuffled, seed=bootstrap_seed
            ),
            "real_minus_no_thought_mean": mean(task_real_minus_no_thought),
            "real_minus_no_thought_bootstrap_95": bootstrap_interval(
                task_real_minus_no_thought, seed=bootstrap_seed + 1
            ),
            "bootstrap_unit": "task",
            "bootstrap_draws": 10_000,
        },
        "task_success_count_histograms": {
            "real": dict(sorted(Counter(row["real_successes"] for row in task_rows).items())),
            "shuffled": dict(
                sorted(Counter(row["shuffled_successes"] for row in task_rows).items())
            ),
            "no_thought": dict(
                sorted(Counter(row["no_thought_success"] for row in task_rows).items())
            ),
        },
        "verbalization_slices": {
            "correct_alias_mentioned": rate_slice(correct_mention),
            "correct_alias_not_mentioned": rate_slice(no_correct_mention),
            "any_alias_mentioned": rate_slice(any_mention),
            "no_alias_mentioned": rate_slice(no_mention),
        },
        "unmasked_top_slices": {
            "alias_is_unmasked_top": rate_slice(unmasked_alias_top),
            "alias_is_not_unmasked_top": rate_slice(unmasked_non_alias_top),
            "mask_rescued_correct_rows": sum(
                bool(row["correct"]) and not bool(row["full_vocab_top_is_alias"])
                for row in real
            ),
        },
        "post_hoc_label_free_residual_policies": {
            "warning": "exploratory only; cannot rescue selection or tune this experiment",
            "definition": "argmax_alias(log p_real(alias) - log p_control(alias))",
            "by_cap": post_hoc_residual_policies,
        },
        "chosen_alias_histograms": {
            "real": dict(sorted(Counter(row["chosen_alias"] for row in real).items())),
            "shuffled": dict(sorted(Counter(row["chosen_alias"] for row in shuffled).items())),
            "no_thought": dict(
                sorted(Counter(row["chosen_alias"] for row in no_thought).items())
            ),
        },
        "by_correct_alias": by_alias,
        "shuffle_contract": {
            "all_token_multisets_match": all(row["token_multiset_match"] for row in shuffled),
            "minimum_moved_position_rate": min(
                float(row["shuffle_moved_position_rate"]) for row in shuffled
            ),
            "mean_moved_position_rate": mean(
                [float(row["shuffle_moved_position_rate"]) for row in shuffled]
            ),
        },
        "task_rows": task_rows,
    }
    output = EXP / "analysis" / "selection_diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
