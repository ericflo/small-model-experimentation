#!/usr/bin/env python3
"""Post-decision phase diagnostics for terminal NO_PREFIX_J_VALUE.

This script reads only committed prefix_value rows.  It cannot change the
registered decision or authorize causal data.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from value_probe import (  # noqa: E402
    FEATURE_KEYS,
    _center_within_task_fraction,
    fit_final_ridge_model,
    oof_ridge_scores,
    ordered_rows,
    pairwise_task_scores,
)


ROWS_PATH = EXP / "runs" / "prefix_value_rows.jsonl"
SUMMARY_PATH = EXP / "runs" / "prefix_value.json"
CONFIG_PATH = EXP / "configs" / "prefix_value.yaml"
OUTPUT_PATH = EXP / "analysis" / "prefix_phase_diagnostics.json"


def stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (
        2**31
    )


def macro(values: dict[str, float]) -> float:
    return float(np.mean(list(values.values())))


def bootstrap_lower(
    values: list[float], *, seed: int, resamples: int = 10_000
) -> float:
    generator = random.Random(seed)
    draws = sorted(
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(resamples)
    )
    return float(draws[min(int(0.05 * resamples), resamples - 1)])


def paired_lower(
    primary: dict[str, float], baseline: dict[str, float], *, seed: int
) -> float:
    if set(primary) != set(baseline):
        raise RuntimeError("phase baseline task sets differ")
    values = [primary[key] - baseline[key] for key in sorted(primary)]
    return bootstrap_lower(values, seed=seed)


def fit_phase(
    rows: list[dict[str, Any]],
    *,
    task_folds: dict[str, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    cv = config["cross_validation"]
    gap = float(config["outcome"]["minimum_pair_label_gap"])
    scores = {}
    tasks = {}
    for key in FEATURE_KEYS:
        scores[key] = oof_ridge_scores(
            rows,
            feature_key=key,
            task_folds=task_folds,
            folds=int(cv["folds"]),
            l2=float(cv["l2"]),
            standardize=bool(
                config["features"]["standardize_from_training_tasks_only"]
            ),
        )
        tasks[key] = pairwise_task_scores(
            rows, scores[key], minimum_label_gap=gap
        )
    primary = tasks["j_features"]
    base_seed = int(config["uncertainty"]["seed"])
    result = {
        "eligible_tasks": len(primary),
        "task_macro_pairwise_auc": macro(primary),
        "one_sided_95_task_bootstrap_lower": bootstrap_lower(
            list(primary.values()), seed=stable_seed(base_seed, "posthoc-phase-primary")
        ),
        "baselines": {},
        "per_task_j": primary,
    }
    for key in FEATURE_KEYS:
        if key == "j_features":
            continue
        difference = macro(primary) - macro(tasks[key])
        result["baselines"][key] = {
            "task_macro_pairwise_auc": macro(tasks[key]),
            "j_minus_baseline": difference,
            "j_minus_baseline_one_sided_95_task_lower": paired_lower(
                primary,
                tasks[key],
                seed=stable_seed(base_seed, "posthoc-phase-difference", key),
            ),
        }
    result["oof_j_scores"] = scores["j_features"]
    return result


def cross_phase_scores(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    *,
    feature_key: str,
    task_folds: dict[str, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    source_rows = ordered_rows(source_rows)
    target_rows = ordered_rows(target_rows)
    source_x, source_y = _center_within_task_fraction(source_rows, feature_key)
    target_x, _target_y = _center_within_task_fraction(target_rows, feature_key)
    cv = config["cross_validation"]
    folds = int(cv["folds"])
    l2 = float(cv["l2"])
    predictions = np.full(len(target_rows), np.nan)
    identity = np.eye(source_x.shape[1], dtype=np.float64)
    for fold in range(folds):
        train = np.asarray(
            [task_folds[str(row["task_id"])] != fold for row in source_rows],
            dtype=bool,
        )
        test = np.asarray(
            [task_folds[str(row["task_id"])] == fold for row in target_rows],
            dtype=bool,
        )
        train_x = source_x[train]
        test_x = target_x[test]
        if bool(config["features"]["standardize_from_training_tasks_only"]):
            location = train_x.mean(axis=0)
            scale = train_x.std(axis=0)
            scale[scale < 1e-12] = 1.0
            train_x = (train_x - location) / scale
            test_x = (test_x - location) / scale
        gram = train_x.T @ train_x + l2 * identity
        rhs = train_x.T @ source_y[train]
        coefficients = np.linalg.solve(gram, rhs)
        predictions[test] = test_x @ coefficients
    if not bool(np.isfinite(predictions).all()):
        raise RuntimeError("cross-phase predictions are incomplete")
    per_task = pairwise_task_scores(
        target_rows,
        [float(value) for value in predictions],
        minimum_label_gap=float(config["outcome"]["minimum_pair_label_gap"]),
    )
    return {
        "task_macro_pairwise_auc": macro(per_task),
        "one_sided_95_task_bootstrap_lower": bootstrap_lower(
            list(per_task.values()),
            seed=stable_seed(
                int(config["uncertainty"]["seed"]), "cross-phase", feature_key
            ),
        ),
        "per_task": per_task,
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator else 0.0


def feature_stability(
    half_rows: list[dict[str, Any]], full_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    half_rows = ordered_rows(half_rows)
    full_rows = ordered_rows(full_rows)
    half_keys = [(row["task_id"], row["trace_index"]) for row in half_rows]
    full_keys = [(row["task_id"], row["trace_index"]) for row in full_rows]
    if half_keys != full_keys:
        raise RuntimeError("phase rows do not align by task/path")
    half_x, _ = _center_within_task_fraction(half_rows, "j_features")
    full_x, _ = _center_within_task_fraction(full_rows, "j_features")
    row_cosines = [cosine(left, right) for left, right in zip(half_x, full_x)]
    coordinate_correlations = []
    for column in range(half_x.shape[1]):
        if half_x[:, column].std() < 1e-12 or full_x[:, column].std() < 1e-12:
            continue
        coordinate_correlations.append(
            float(np.corrcoef(half_x[:, column], full_x[:, column])[0, 1])
        )
    return {
        "paired_paths": len(half_rows),
        "mean_centered_row_cosine": float(np.mean(row_cosines)),
        "median_centered_row_cosine": float(np.median(row_cosines)),
        "coordinate_correlations": {
            "count": len(coordinate_correlations),
            "mean": float(np.mean(coordinate_correlations)),
            "median": float(np.median(coordinate_correlations)),
            "minimum": min(coordinate_correlations),
            "maximum": max(coordinate_correlations),
            "negative_count": sum(value < 0 for value in coordinate_correlations),
        },
    }


def coefficient_alignment(
    half_rows: list[dict[str, Any]],
    full_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, float]:
    kwargs = {
        "feature_key": "j_features",
        "l2": float(config["cross_validation"]["l2"]),
        "standardize": bool(
            config["features"]["standardize_from_training_tasks_only"]
        ),
    }
    half = fit_final_ridge_model(half_rows, **kwargs)
    full = fit_final_ridge_model(full_rows, **kwargs)
    half_standard = np.asarray(half["coefficients"], dtype=np.float64)
    full_standard = np.asarray(full["coefficients"], dtype=np.float64)
    half_raw = half_standard / np.asarray(half["scale"], dtype=np.float64)
    full_raw = full_standard / np.asarray(full["scale"], dtype=np.float64)
    return {
        "standardized_coefficient_cosine": cosine(half_standard, full_standard),
        "raw_coordinate_coefficient_cosine": cosine(half_raw, full_raw),
    }


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    if summary.get("decision") != "NO_PREFIX_J_VALUE" or summary.get("passed") is not False:
        raise RuntimeError("post-decision diagnostic requires terminal negative")
    if summary.get("causal_split_opened") is not False:
        raise RuntimeError("causal firewall changed")
    rows = ordered_rows(
        [json.loads(line) for line in ROWS_PATH.read_text().splitlines() if line]
    )
    config = yaml.safe_load(CONFIG_PATH.read_text())
    task_folds = {
        str(key): int(value)
        for key, value in summary["analysis"]["task_folds"].items()
    }
    half_fraction = float(config["outcome"]["prospective_fraction"])
    full_fraction = float(config["outcome"]["endpoint_fraction"])
    half_rows = [row for row in rows if float(row["fraction"]) == half_fraction]
    full_rows = [row for row in rows if float(row["fraction"]) == full_fraction]
    half = fit_phase(half_rows, task_folds=task_folds, config=config)
    full = fit_phase(full_rows, task_folds=task_folds, config=config)
    cross = {}
    for key in ("j_features", "non_j_random_features", "slot_margin_features"):
        cross[f"half_train_to_full_{key}"] = cross_phase_scores(
            half_rows,
            full_rows,
            feature_key=key,
            task_folds=task_folds,
            config=config,
        )
        cross[f"full_train_to_half_{key}"] = cross_phase_scores(
            full_rows,
            half_rows,
            feature_key=key,
            task_folds=task_folds,
            config=config,
        )
    output = {
        "schema_version": 1,
        "scientific_result": False,
        "post_decision": True,
        "cannot_rescue_registered_decision": True,
        "registered_decision": "NO_PREFIX_J_VALUE",
        "causal_split_opened": False,
        "source_summary_sha256": hashlib.sha256(SUMMARY_PATH.read_bytes()).hexdigest(),
        "source_rows_sha256": hashlib.sha256(ROWS_PATH.read_bytes()).hexdigest(),
        "fraction_specific_oof": {
            "half": half,
            "full": full,
        },
        "cross_phase_transfer": cross,
        "phase_feature_stability": feature_stability(half_rows, full_rows),
        "phase_coefficient_alignment": coefficient_alignment(
            half_rows, full_rows, config
        ),
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
