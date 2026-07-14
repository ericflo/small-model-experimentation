"""Executable paired decision gates for the reflection-transfer experiment."""

from __future__ import annotations

import random
from statistics import mean
from typing import Any


def _index(rows: list[dict[str, Any]], arm: str) -> dict[str, dict[str, Any]]:
    selected = [row for row in rows if row["arm"] == arm]
    result = {row["task_id"]: row for row in selected}
    if len(result) != len(selected):
        raise ValueError(f"duplicate task rows for arm {arm}")
    return result


def paired_values(
    rows: list[dict[str, Any]], left: str, right: str, metric: str
) -> list[tuple[str, str, float]]:
    left_rows = _index(rows, left)
    right_rows = _index(rows, right)
    if set(left_rows) != set(right_rows):
        raise ValueError(f"task pairing differs for {left} versus {right}")
    return [
        (
            task_id,
            str(left_rows[task_id]["family"]),
            float(left_rows[task_id][metric]) - float(right_rows[task_id][metric]),
        )
        for task_id in sorted(left_rows)
    ]


def paired_bootstrap_interval(
    deltas: list[float], resamples: int, seed: int
) -> tuple[float, float]:
    if not deltas:
        raise ValueError("cannot bootstrap an empty paired sample")
    if resamples < 100:
        raise ValueError("bootstrap requires at least 100 resamples")
    rng = random.Random(seed)
    size = len(deltas)
    draws = sorted(
        mean(deltas[rng.randrange(size)] for _ in range(size))
        for _ in range(resamples)
    )
    lower = draws[int(0.025 * resamples)]
    upper = draws[min(resamples - 1, int(0.975 * resamples))]
    return lower, upper


def _comparison(
    rows: list[dict[str, Any]],
    left: str,
    right: str,
    metric: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    paired = paired_values(rows, left, right, metric)
    deltas = [value for _, _, value in paired]
    families = sorted({family for _, family, _ in paired})
    lower, upper = paired_bootstrap_interval(deltas, resamples, seed)
    return {
        "left": left,
        "right": right,
        "n": len(paired),
        "delta": mean(deltas),
        "lower_95": lower,
        "upper_95": upper,
        "family_delta": {
            family: mean(value for _, row_family, value in paired if row_family == family)
            for family in families
        },
    }


def evaluate_seed_block(
    rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    """Apply the exact per-seed qualification/confirmation capability gates."""
    required = {
        "frozen_action",
        "reflection_correct_action",
        "reflection_shuffled_action",
        "auxiliary_plan_label_correct_action",
    }
    observed = {row["arm"] for row in rows}
    missing = required - observed
    if missing:
        raise ValueError(f"missing required evaluation arms: {sorted(missing)}")
    resamples = int(bootstrap["paired_task_resamples"])
    seed = int(bootstrap["seed"])
    shuffle = _comparison(
        rows,
        "reflection_correct_action",
        "reflection_shuffled_action",
        "coverage_at_16",
        resamples,
        seed,
    )
    frozen = _comparison(
        rows,
        "reflection_correct_action",
        "frozen_action",
        "coverage_at_16",
        resamples,
        seed + 1,
    )
    auxiliary = _comparison(
        rows,
        "reflection_correct_action",
        "auxiliary_plan_label_correct_action",
        "coverage_at_16",
        resamples,
        seed + 2,
    )
    family_shuffle_ok = all(
        value >= float(thresholds["each_family_correct_minus_shuffled_min"])
        for value in shuffle["family_delta"].values()
    )
    family_frozen_ok = all(
        value >= float(thresholds["each_family_correct_minus_frozen_min"])
        for value in frozen["family_delta"].values()
    )
    correct_rows = list(_index(rows, "reflection_correct_action").values())
    parse_rate = mean(float(row["strict_parse_rate"]) for row in correct_rows)
    answer_contacts = mean(float(row["answer_limit_contact"]) for row in correct_rows)
    loop_contacts = mean(float(row["periodic_loop_contact"]) for row in correct_rows)
    capability_checks = {
        "correct_minus_shuffled_effect": shuffle["delta"]
        >= float(thresholds["correct_minus_shuffled_min"]),
        "correct_minus_frozen_effect": frozen["delta"]
        >= float(thresholds["correct_minus_frozen_min"]),
        "correct_minus_shuffled_lower": shuffle["lower_95"]
        > float(thresholds["paired_delta_lower_95_gt"]),
        "correct_minus_frozen_lower": frozen["lower_95"]
        > float(thresholds["paired_delta_lower_95_gt"]),
        "each_family_correct_minus_shuffled": family_shuffle_ok,
        "each_family_correct_minus_frozen": family_frozen_ok,
        "strict_parse_rate": parse_rate >= float(thresholds["strict_parse_rate_min"]),
        "answer_limit_contact": answer_contacts
        <= float(thresholds["answer_limit_contact_max"]),
        "periodic_loop_contact": loop_contacts
        <= float(thresholds["periodic_loop_contact_max"]),
    }
    mechanism_checks = {
        "correct_minus_auxiliary_effect": auxiliary["delta"]
        >= float(thresholds["reflection_specific_correct_minus_auxiliary_min"]),
        "correct_minus_auxiliary_lower": auxiliary["lower_95"]
        > float(thresholds["paired_delta_lower_95_gt"]),
    }
    return {
        "capability_pass": all(capability_checks.values()),
        "reflection_specific_pass": all(mechanism_checks.values()),
        "capability_checks": capability_checks,
        "mechanism_checks": mechanism_checks,
        "comparisons": {
            "correct_minus_shuffled": shuffle,
            "correct_minus_frozen": frozen,
            "correct_minus_auxiliary": auxiliary,
        },
        "rates": {
            "strict_parse": parse_rate,
            "answer_limit_contact": answer_contacts,
            "periodic_loop_contact": loop_contacts,
        },
    }


def evaluate_positive_control(
    rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    comparison = _comparison(
        rows,
        "direct_plan_answer_positive_control_action",
        "frozen_action",
        "coverage_at_16",
        int(bootstrap["paired_task_resamples"]),
        int(bootstrap["seed"]) + 3,
    )
    direct = _index(rows, "direct_plan_answer_positive_control_action")
    coverage = mean(float(row["coverage_at_16"]) for row in direct.values())
    checks = {
        "coverage": coverage >= float(thresholds["coverage_at_16_min"]),
        "delta": comparison["delta"] >= float(thresholds["delta_over_frozen_min"]),
        "lower": comparison["lower_95"]
        > float(thresholds["paired_delta_lower_95_gt"]),
    }
    return {"pass": all(checks.values()), "checks": checks, "comparison": comparison}


def evaluate_calibration(
    rows: list[dict[str, Any]], thresholds: dict[str, Any]
) -> dict[str, Any]:
    frozen = list(_index(rows, "frozen_action").values())
    if not frozen:
        raise ValueError("calibration has no frozen rows")
    rates = {
        "coverage_at_16": mean(float(row["coverage_at_16"]) for row in frozen),
        "strict_parse_rate": mean(float(row["strict_parse_rate"]) for row in frozen),
        "answer_limit_contact": mean(float(row["answer_limit_contact"]) for row in frozen),
        "periodic_loop_contact": mean(float(row["periodic_loop_contact"]) for row in frozen),
    }
    checks = {
        "coverage_min": rates["coverage_at_16"]
        >= float(thresholds["base_coverage_at_16_min"]),
        "coverage_max": rates["coverage_at_16"]
        <= float(thresholds["base_coverage_at_16_max"]),
        "strict_parse": rates["strict_parse_rate"]
        >= float(thresholds["strict_parse_rate_min"]),
        "answer_limit": rates["answer_limit_contact"]
        <= float(thresholds["answer_limit_contact_max"]),
        "periodic_loop": rates["periodic_loop_contact"]
        <= float(thresholds["periodic_loop_contact_max"]),
    }
    return {"pass": all(checks.values()), "checks": checks, "rates": rates}


def evaluate_retention(
    rows: list[dict[str, Any]], arm: str, depth_min: float, family_min: float
) -> dict[str, Any]:
    paired = paired_values(rows, arm, "frozen_action", "coverage_at_16")
    indexed = _index(rows, arm)
    by_depth: dict[int, list[float]] = {}
    by_family: dict[str, list[float]] = {}
    for task_id, family, delta in paired:
        depth = int(indexed[task_id]["depth"])
        by_depth.setdefault(depth, []).append(delta)
        by_family.setdefault(family, []).append(delta)
    depth_delta = {depth: mean(values) for depth, values in sorted(by_depth.items())}
    family_delta = {family: mean(values) for family, values in sorted(by_family.items())}
    checks = {
        "each_depth": all(value >= depth_min for value in depth_delta.values()),
        "each_family": all(value >= family_min for value in family_delta.values()),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "depth_delta": depth_delta,
        "family_delta": family_delta,
    }
