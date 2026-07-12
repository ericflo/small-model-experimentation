"""Task-held-out prospective prefix-value evaluation.

All scientific choices live in configs/prefix_value.yaml and its preregistration.
This module is pure CPU analysis so contracts can be tested without loading or
opening model data.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from collections import defaultdict
from typing import Any

import numpy as np


FEATURE_KEYS = (
    "j_features",
    "non_j_random_features",
    "correct_alias_activity_features",
    "slot_margin_features",
    "alias_identity_features",
)


def _stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (
        2**31
    )


def ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["task_id"]),
            float(row["fraction"]),
            int(row["trace_index"]),
        ),
    )


def validate_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("prefix-value rows are empty")
    layers = tuple(int(value) for value in config["features"]["layers"])
    coordinates = int(config["features"]["coordinates_per_layer"])
    expected_width = len(layers) * coordinates
    expected_fractions = {
        float(config["outcome"]["prospective_fraction"]),
        float(config["outcome"]["endpoint_fraction"]),
    }
    by_path: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[tuple[str, float], list[int]] = defaultdict(list)
    aliases_by_task: dict[str, set[str]] = defaultdict(set)
    finite = 0
    for row in rows:
        by_path[(str(row["task_id"]), int(row["trace_index"]))].append(row)
        by_group[(str(row["task_id"]), float(row["fraction"]))].append(
            int(row["trace_index"])
        )
        aliases_by_task[str(row["task_id"])].add(str(row["correct_alias"]))
        if len(row["j_features"]) != expected_width:
            raise ValueError("J feature width changed")
        if len(row["non_j_random_features"]) != expected_width:
            raise ValueError("non-J random feature width changed")
        if len(row["correct_alias_activity_features"]) != len(layers):
            raise ValueError("correct-alias activity width changed")
        if len(row["slot_margin_features"]) != 1:
            raise ValueError("slot-margin width changed")
        identity = row["alias_identity_features"]
        if len(identity) != int(row["alias_count"]) or sum(identity) != 1.0:
            raise ValueError("alias identity is not one-hot")
        numeric = [
            *row["j_features"],
            *row["non_j_random_features"],
            *row["correct_alias_activity_features"],
            *row["slot_margin_features"],
            *identity,
            float(row["terminal_value"]),
        ]
        row_finite = bool(np.isfinite(np.asarray(numeric, dtype=np.float64)).all())
        if row_finite != bool(row["finite"]):
            raise ValueError("finite flag disagrees with numeric payload")
        finite += int(row_finite)
    if any(len(values) != 1 for values in aliases_by_task.values()):
        raise ValueError("a task has multiple correct aliases")
    for path_rows in by_path.values():
        if {float(row["fraction"]) for row in path_rows} != expected_fractions:
            raise ValueError("a path does not contain both frozen fractions")
        terminal_values = {float(row["terminal_value"]) for row in path_rows}
        if len(terminal_values) != 1:
            raise ValueError("fractions of one path have different terminal labels")
    for trace_indices in by_group.values():
        if sorted(trace_indices) != [0, 1, 2]:
            raise ValueError("a task/fraction group does not contain three paths")
    return {
        "rows": len(rows),
        "tasks": len(aliases_by_task),
        "paths": len(by_path),
        "expected_j_width": expected_width,
        "finite_rows": finite,
        "finite_rate": finite / len(rows),
    }


def assign_alias_stratified_folds(
    rows: list[dict[str, Any]], *, folds: int, seed: int
) -> dict[str, int]:
    if folds < 2:
        raise ValueError("at least two folds are required")
    task_alias: dict[str, str] = {}
    for row in rows:
        task_id = str(row["task_id"])
        alias = str(row["correct_alias"])
        if task_id in task_alias and task_alias[task_id] != alias:
            raise ValueError("task alias changed across rows")
        task_alias[task_id] = alias
    by_alias: dict[str, list[str]] = defaultdict(list)
    for task_id, alias in task_alias.items():
        by_alias[alias].append(task_id)
    result = {}
    for alias in sorted(by_alias):
        tasks = sorted(by_alias[alias])
        random.Random(_stable_seed(seed, alias)).shuffle(tasks)
        offset = _stable_seed(seed, alias, "offset") % folds
        for index, task_id in enumerate(tasks):
            result[task_id] = int((index + offset) % folds)
    if set(result) != set(task_alias):
        raise RuntimeError("fold assignment dropped tasks")
    if set(result.values()) != set(range(folds)):
        raise RuntimeError("one or more task folds are empty")
    return result


def _center_within_task_fraction(
    rows: list[dict[str, Any]], feature_key: str
) -> tuple[np.ndarray, np.ndarray]:
    if feature_key not in FEATURE_KEYS:
        raise ValueError(f"unknown feature key: {feature_key}")
    features = np.asarray([row[feature_key] for row in rows], dtype=np.float64)
    labels = np.asarray([float(row["terminal_value"]) for row in rows], dtype=np.float64)
    centered_features = np.empty_like(features)
    centered_labels = np.empty_like(labels)
    groups: dict[tuple[str, float], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(str(row["task_id"]), float(row["fraction"]))].append(index)
    for indices in groups.values():
        group_features = features[indices]
        group_labels = labels[indices]
        centered_features[indices] = group_features - group_features.mean(axis=0)
        centered_labels[indices] = group_labels - group_labels.mean()
    return centered_features, centered_labels


def oof_ridge_scores(
    rows: list[dict[str, Any]],
    *,
    feature_key: str,
    task_folds: dict[str, int],
    folds: int,
    l2: float,
    standardize: bool,
) -> list[float]:
    if l2 <= 0:
        raise ValueError("ridge L2 must be positive")
    rows = ordered_rows(rows)
    features, labels = _center_within_task_fraction(rows, feature_key)
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    identity = np.eye(features.shape[1], dtype=np.float64)
    for fold in range(folds):
        test = np.asarray(
            [task_folds[str(row["task_id"])] == fold for row in rows], dtype=bool
        )
        train = ~test
        if not bool(train.any()) or not bool(test.any()):
            raise RuntimeError("empty train or test fold")
        train_features = features[train]
        test_features = features[test]
        if standardize:
            location = train_features.mean(axis=0)
            scale = train_features.std(axis=0)
            scale[scale < 1e-12] = 1.0
            train_features = (train_features - location) / scale
            test_features = (test_features - location) / scale
        gram = train_features.T @ train_features + float(l2) * identity
        rhs = train_features.T @ labels[train]
        try:
            coefficients = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        scores[test] = test_features @ coefficients
    if not bool(np.isfinite(scores).all()):
        raise RuntimeError("out-of-fold scores are nonfinite or incomplete")
    return [float(value) for value in scores]


def fit_final_ridge_model(
    rows: list[dict[str, Any]],
    *,
    feature_key: str,
    l2: float,
    standardize: bool,
) -> dict[str, Any]:
    """Fit the frozen post-gate model for later untouched-task scoring."""
    rows = ordered_rows(rows)
    features, labels = _center_within_task_fraction(rows, feature_key)
    location = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    fitted_features = (features - location) / scale if standardize else features
    identity = np.eye(fitted_features.shape[1], dtype=np.float64)
    gram = fitted_features.T @ fitted_features + float(l2) * identity
    rhs = fitted_features.T @ labels
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return {
        "schema_version": 1,
        "feature_key": feature_key,
        "feature_width": int(features.shape[1]),
        "within_task_fraction_centering_required": True,
        "standardized": bool(standardize),
        "location": [float(value) for value in location],
        "scale": [float(value) for value in scale],
        "coefficients": [float(value) for value in coefficients],
        "l2": float(l2),
        "training_rows": len(rows),
        "training_tasks": len({str(row["task_id"]) for row in rows}),
    }


def pairwise_task_scores(
    rows: list[dict[str, Any]],
    scores: list[float],
    *,
    minimum_label_gap: float,
    fraction: float | None = None,
) -> dict[str, float]:
    rows = ordered_rows(rows)
    if len(rows) != len(scores):
        raise ValueError("row and score counts differ")
    groups: dict[tuple[str, float], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        row_fraction = float(row["fraction"])
        if fraction is None or row_fraction == float(fraction):
            groups[(str(row["task_id"]), row_fraction)].append(index)
    outcomes: dict[str, list[float]] = defaultdict(list)
    for (task_id, _row_fraction), indices in groups.items():
        for left, right in itertools.combinations(indices, 2):
            label_delta = float(rows[left]["terminal_value"]) - float(
                rows[right]["terminal_value"]
            )
            if abs(label_delta) < minimum_label_gap:
                continue
            score_delta = float(scores[left]) - float(scores[right])
            if abs(score_delta) <= 1e-12:
                outcome = 0.5
            else:
                outcome = float((score_delta > 0) == (label_delta > 0))
            outcomes[task_id].append(outcome)
    return {
        task_id: float(np.mean(values))
        for task_id, values in outcomes.items()
        if values
    }


def _macro(task_scores: dict[str, float]) -> float:
    if not task_scores:
        raise ValueError("no eligible task pairwise scores")
    return float(np.mean(list(task_scores.values())))


def _bootstrap_lower(
    values: list[float], *, seed: int, resamples: int, tail_probability: float
) -> float:
    if not values or resamples <= 0 or not 0 < tail_probability < 0.5:
        raise ValueError("invalid task bootstrap")
    generator = random.Random(seed)
    draws = sorted(
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(resamples)
    )
    return float(draws[min(int(tail_probability * resamples), resamples - 1)])


def _paired_difference(
    primary: dict[str, float], baseline: dict[str, float]
) -> dict[str, float]:
    if set(primary) != set(baseline):
        raise RuntimeError("primary and baseline eligible task sets differ")
    return {task_id: primary[task_id] - baseline[task_id] for task_id in primary}


def _shuffle_j_within_groups(
    rows: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in ordered_rows(rows)]
    groups: dict[tuple[str, float], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(str(row["task_id"]), float(row["fraction"]))].append(index)
    for key in sorted(groups):
        indices = groups[key]
        permutation = list(indices)
        random.Random(_stable_seed(seed, key[0], str(key[1]))).shuffle(permutation)
        original = [list(rows[index]["j_features"]) for index in indices]
        for target, source_index in zip(indices, permutation, strict=True):
            source_offset = indices.index(source_index)
            rows[target]["j_features"] = original[source_offset]
    return rows


def evaluate_prefix_value(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    rows = ordered_rows(rows)
    contract = validate_rows(rows, config)
    cv = config["cross_validation"]
    task_folds = assign_alias_stratified_folds(
        rows, folds=int(cv["folds"]), seed=int(cv["seed"])
    )
    minimum_gap = float(config["outcome"]["minimum_pair_label_gap"])

    all_scores = {}
    all_task_scores = {}
    for feature_key in FEATURE_KEYS:
        scores = oof_ridge_scores(
            rows,
            feature_key=feature_key,
            task_folds=task_folds,
            folds=int(cv["folds"]),
            l2=float(cv["l2"]),
            standardize=bool(config["features"]["standardize_from_training_tasks_only"]),
        )
        all_scores[feature_key] = scores
        all_task_scores[feature_key] = pairwise_task_scores(
            rows, scores, minimum_label_gap=minimum_gap
        )

    primary = all_task_scores["j_features"]
    correct = all_task_scores["correct_alias_activity_features"]
    margin = all_task_scores["slot_margin_features"]
    identity = all_task_scores["alias_identity_features"]
    non_j = all_task_scores["non_j_random_features"]
    primary_auc = _macro(primary)
    correct_auc = _macro(correct)
    margin_auc = _macro(margin)
    identity_auc = _macro(identity)
    prospective_fraction = float(config["outcome"]["prospective_fraction"])
    prospective_task_scores = pairwise_task_scores(
        rows,
        all_scores["j_features"],
        minimum_label_gap=minimum_gap,
        fraction=prospective_fraction,
    )
    endpoint_task_scores = pairwise_task_scores(
        rows,
        all_scores["j_features"],
        minimum_label_gap=minimum_gap,
        fraction=float(config["outcome"]["endpoint_fraction"]),
    )

    uncertainty = config["uncertainty"]
    resamples = int(uncertainty["bootstrap_resamples"])
    tail = float(uncertainty["one_sided_tail_probability"])
    base_seed = int(uncertainty["seed"])
    correct_difference = _paired_difference(primary, correct)
    margin_difference = _paired_difference(primary, margin)
    non_j_difference = _paired_difference(primary, non_j)
    bootstrap = {
        "primary_auc_one_sided_95_lower": _bootstrap_lower(
            list(primary.values()),
            seed=_stable_seed(base_seed, "primary"),
            resamples=resamples,
            tail_probability=tail,
        ),
        "j_minus_correct_alias_activity_one_sided_95_lower": _bootstrap_lower(
            list(correct_difference.values()),
            seed=_stable_seed(base_seed, "correct_alias_activity"),
            resamples=resamples,
            tail_probability=tail,
        ),
        "j_minus_slot_margin_one_sided_95_lower": _bootstrap_lower(
            list(margin_difference.values()),
            seed=_stable_seed(base_seed, "slot_margin"),
            resamples=resamples,
            tail_probability=tail,
        ),
        "j_minus_non_j_random_one_sided_95_lower": _bootstrap_lower(
            list(non_j_difference.values()),
            seed=_stable_seed(base_seed, "non_j_random"),
            resamples=resamples,
            tail_probability=tail,
        ),
        "resamples": resamples,
        "unit": "task",
    }

    null_config = config["null"]
    null_aucs = []
    for repeat in range(int(null_config["repeats"])):
        shuffled_rows = _shuffle_j_within_groups(
            rows, seed=_stable_seed(int(null_config["seed"]), str(repeat))
        )
        shuffled_scores = oof_ridge_scores(
            shuffled_rows,
            feature_key="j_features",
            task_folds=task_folds,
            folds=int(cv["folds"]),
            l2=float(cv["l2"]),
            standardize=bool(config["features"]["standardize_from_training_tasks_only"]),
        )
        null_aucs.append(
            _macro(
                pairwise_task_scores(
                    shuffled_rows,
                    shuffled_scores,
                    minimum_label_gap=minimum_gap,
                )
            )
        )
    mean_null_auc = float(np.mean(null_aucs))

    fold_metrics = {}
    for fold in range(int(cv["folds"])):
        task_ids = {task_id for task_id, value in task_folds.items() if value == fold}
        values = [primary[task_id] for task_id in sorted(task_ids) if task_id in primary]
        fold_metrics[str(fold)] = {
            "tasks": len(task_ids),
            "eligible_tasks": len(values),
            "task_macro_pairwise_auc": float(np.mean(values)) if values else None,
        }

    task_alias = {}
    for row in rows:
        task_alias[str(row["task_id"])] = str(row["correct_alias"])
    alias_metrics = {}
    for alias in sorted(set(task_alias.values())):
        values = [
            primary[task_id]
            for task_id in sorted(primary)
            if task_alias[task_id] == alias
        ]
        alias_metrics[alias] = {
            "eligible_tasks": len(values),
            "task_macro_pairwise_auc": float(np.mean(values)) if values else None,
        }

    gates = config["gates"]
    metrics = {
        "mixed_tasks": len(primary),
        "scored_prefixes": contract["finite_rows"],
        "finite_feature_rows_rate": contract["finite_rate"],
        "task_macro_pairwise_auc": primary_auc,
        "prospective_half_pairwise_auc": _macro(prospective_task_scores),
        "endpoint_full_pairwise_auc": _macro(endpoint_task_scores),
        "correct_alias_activity_pairwise_auc": correct_auc,
        "slot_margin_pairwise_auc": margin_auc,
        "alias_identity_pairwise_auc": identity_auc,
        "non_j_random_pairwise_auc": _macro(non_j),
        "j_minus_correct_alias_activity": primary_auc - correct_auc,
        "j_minus_slot_margin": primary_auc - margin_auc,
        "j_minus_alias_identity": primary_auc - identity_auc,
        "j_minus_non_j_random": primary_auc - _macro(non_j),
        "shuffled_null_mean_pairwise_auc": mean_null_auc,
        "shuffled_null_min_pairwise_auc": min(null_aucs),
        "shuffled_null_max_pairwise_auc": max(null_aucs),
        "bootstrap": bootstrap,
    }
    gate_checks = {
        "mixed_tasks": metrics["mixed_tasks"] >= int(gates["mixed_tasks_min"]),
        "scored_prefixes": metrics["scored_prefixes"]
        >= int(gates["scored_prefixes_min"]),
        "task_macro_pairwise_auc": primary_auc
        >= float(gates["task_macro_pairwise_auc_min"]),
        "prospective_half_pairwise_auc": metrics["prospective_half_pairwise_auc"]
        >= float(gates["prospective_half_pairwise_auc_min"]),
        "j_minus_correct_alias_activity": metrics["j_minus_correct_alias_activity"]
        >= float(gates["j_minus_correct_alias_activity_min"]),
        "j_minus_slot_margin": metrics["j_minus_slot_margin"]
        >= float(gates["j_minus_slot_margin_min"]),
        "j_minus_alias_identity": metrics["j_minus_alias_identity"]
        >= float(gates["j_minus_alias_identity_min"]),
        "j_minus_non_j_random": metrics["j_minus_non_j_random"]
        >= float(gates["j_minus_non_j_random_min"]),
        "task_bootstrap_auc_lower": bootstrap["primary_auc_one_sided_95_lower"]
        > float(gates["task_bootstrap_auc_lower_min"]),
        "task_bootstrap_j_minus_correct_alias_lower": bootstrap[
            "j_minus_correct_alias_activity_one_sided_95_lower"
        ]
        > float(gates["task_bootstrap_j_minus_correct_alias_lower_min"]),
        "task_bootstrap_j_minus_slot_margin_lower": bootstrap[
            "j_minus_slot_margin_one_sided_95_lower"
        ]
        > float(gates["task_bootstrap_j_minus_slot_margin_lower_min"]),
        "task_bootstrap_j_minus_non_j_random_lower": bootstrap[
            "j_minus_non_j_random_one_sided_95_lower"
        ]
        > float(gates["task_bootstrap_j_minus_non_j_random_lower_min"]),
        "shuffled_null": abs(mean_null_auc - 0.5)
        <= float(gates["shuffled_auc_abs_from_chance_max"]),
        "finite_feature_rows": metrics["finite_feature_rows_rate"]
        >= float(gates["finite_feature_rows_rate_min"]),
    }
    passed = all(gate_checks.values())
    final_model = fit_final_ridge_model(
        rows,
        feature_key="j_features",
        l2=float(cv["l2"]),
        standardize=bool(config["features"]["standardize_from_training_tasks_only"]),
    )
    return {
        "passed": passed,
        "decision": config["decision_labels"]["pass"]
        if passed
        else config["decision_labels"]["no_value"],
        "contract": contract,
        "metrics": metrics,
        "gate_checks": gate_checks,
        "task_folds": task_folds,
        "fold_metrics": fold_metrics,
        "alias_metrics": alias_metrics,
        "null_repeat_aucs": null_aucs,
        "oof_scores": {
            feature_key: all_scores[feature_key] for feature_key in FEATURE_KEYS
        },
        "per_task_pairwise": {
            "j": primary,
            "correct_alias_activity": correct,
            "slot_margin": margin,
            "alias_identity": identity,
            "non_j_random": non_j,
            "prospective_half_j": prospective_task_scores,
            "endpoint_full_j": endpoint_task_scores,
        },
        "final_model": final_model,
    }
