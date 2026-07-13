"""Pure mechanics scoring, controls, and frozen gate arithmetic.

This module has no model-loading path.  It operates only on public mechanics
tasks, public-live audit labels, and already-authenticated runner outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from typing import Any

import numpy as np

from protocol import parse_program, score_candidate
from task_data import (
    ALIASES,
    CONCRETE_OPERATIONS,
    INVALID,
    BoundOperation,
    alias_program,
    apply_pipeline,
    canonical_operation,
    canonical_program,
    operation_from_record,
)


SURFACE_NUMERIC_FEATURES = (
    "mean_abs_length_difference",
    "max_abs_length_difference",
    "equal_length_fraction",
    "mean_abs_sum_difference",
    "max_abs_sum_difference",
    "mean_abs_min_difference",
    "max_abs_min_difference",
    "mean_abs_max_difference",
    "max_abs_max_difference",
    "mean_aligned_l1_equal_length_zero_otherwise",
    "exact_state_equality_fraction",
)
SURFACE_FEATURE_NAMES = tuple(f"candidate_{alias}" for alias in ALIASES) + (
    SURFACE_NUMERIC_FEATURES
)
SURFACE_SOLVER = {
    "name": "float64_damped_newton_v1",
    "ridge_lambda": 1.0,
    "objective": "balanced_weighted_mean_log_loss_plus_lambda_half_l2",
    "intercept_penalized": False,
    "all_features_standardized": True,
    "zero_variance_scale": 1.0,
    "class_weight": "n_train/(2*n_class)",
    "gradient_infinity_tolerance": 1e-10,
    "newton_decrement_half_tolerance": 1e-10,
    "maximum_iterations": 10000,
    "armijo_constant": 1e-4,
    "maximum_line_search_halvings": 60,
    "decision_score": "standardized_linear_predictor",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def operation_alias(operation: BoundOperation) -> str:
    try:
        return ALIASES[CONCRETE_OPERATIONS.index(operation)]
    except ValueError as exc:
        raise ValueError("operation is outside the frozen bank") from exc


def record_id(domain: str, task_id: str, candidate: BoundOperation | None = None) -> str:
    parts = [domain, task_id]
    if candidate is not None:
        parts.append(canonical_operation(candidate))
    return hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()


def public_live_map(audit_rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for row in audit_rows:
        task_id = row.get("task_id")
        live_rows = row.get("public_live")
        if not isinstance(task_id, str) or not isinstance(live_rows, list):
            raise ValueError("invalid mechanics audit row")
        aliases: set[str] = set()
        for live in live_rows:
            operation = operation_from_record(live["operation"])
            alias = operation_alias(operation)
            if alias in aliases:
                raise ValueError("duplicate public-live operation")
            aliases.add(alias)
        if not 1 <= len(aliases) <= 4 or task_id in result:
            raise ValueError("invalid public-live task geometry")
        result[task_id] = aliases
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def surface_features(task: Mapping[str, Any], candidate: BoundOperation) -> np.ndarray:
    """Return the frozen 24 one-hot + 11 relation-feature vector."""

    if candidate not in CONCRETE_OPERATIONS:
        raise ValueError("candidate is outside the frozen bank")
    visible = task.get("visible")
    if not isinstance(visible, list) or len(visible) != 8:
        raise ValueError("surface features require eight visible rows")
    one_hot = [0.0] * len(CONCRETE_OPERATIONS)
    one_hot[CONCRETE_OPERATIONS.index(candidate)] = 1.0
    length_differences: list[float] = []
    equal_lengths: list[float] = []
    sum_differences: list[float] = []
    min_differences: list[float] = []
    max_differences: list[float] = []
    aligned_l1: list[float] = []
    equal_states: list[float] = []
    for row in visible:
        state = apply_pipeline(row["input"], (candidate,))
        target = row["output"]
        if state is INVALID or not isinstance(state, list) or not state or not target:
            raise ValueError("surface relation contains invalid or empty state")
        length_differences.append(float(abs(len(state) - len(target))))
        same_length = len(state) == len(target)
        equal_lengths.append(float(same_length))
        sum_differences.append(float(abs(sum(state) - sum(target))))
        min_differences.append(float(abs(min(state) - min(target))))
        max_differences.append(float(abs(max(state) - max(target))))
        aligned_l1.append(
            math.fsum(float(abs(left - right)) for left, right in zip(state, target))
            if same_length
            else 0.0
        )
        equal_states.append(float(state == target))
    numeric = [
        _mean(length_differences),
        max(length_differences),
        _mean(equal_lengths),
        _mean(sum_differences),
        max(sum_differences),
        _mean(min_differences),
        max(min_differences),
        _mean(max_differences),
        max(max_differences),
        _mean(aligned_l1),
        _mean(equal_states),
    ]
    values = np.asarray([*one_hot, *numeric], dtype=np.float64)
    if values.shape != (35,) or not np.isfinite(values).all():
        raise RuntimeError("surface feature geometry is invalid")
    return values


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def fit_balanced_ridge(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_lambda: float = 1.0,
    tolerance: float = 1e-10,
    maximum_iterations: int = 10000,
) -> dict[str, Any]:
    """Fit the frozen deterministic balanced ridge-logistic estimator.

    The objective is the class-balanced weighted *mean* logistic loss plus
    ``lambda/2 * ||beta||^2``.  Every feature, including candidate one-hots, is
    standardized using the training fold only.  The intercept is unpenalized.
    """

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[1] != 35:
        raise ValueError("ridge input geometry is invalid")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not set(y) <= {0.0, 1.0}:
        raise ValueError("ridge inputs must be finite binary data")
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("balanced ridge requires both classes")
    if ridge_lambda <= 0 or tolerance <= 0 or maximum_iterations < 1:
        raise ValueError("invalid ridge solver setting")

    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64)
    scale[scale == 0.0] = 1.0
    standardized = (x - mean) / scale
    weights = np.where(
        y == 1.0,
        len(y) / (2.0 * positives),
        len(y) / (2.0 * negatives),
    ).astype(np.float64)
    weight_total = math.fsum(float(value) for value in weights)
    beta = np.zeros(x.shape[1], dtype=np.float64)
    intercept = 0.0
    identity = np.eye(x.shape[1], dtype=np.float64)

    def objective(current_beta: np.ndarray, current_intercept: float) -> float:
        linear = standardized @ current_beta + current_intercept
        losses = np.logaddexp(0.0, linear) - y * linear
        return float(
            np.dot(weights, losses) / weight_total
            + 0.5 * ridge_lambda * np.dot(current_beta, current_beta)
        )

    converged = False
    final_gradient = math.inf
    final_decrement = math.inf
    iterations = 0
    for iteration in range(maximum_iterations):
        linear = standardized @ beta + intercept
        probability = _sigmoid(linear)
        residual = weights * (probability - y) / weight_total
        gradient_beta = standardized.T @ residual + ridge_lambda * beta
        gradient_intercept = float(residual.sum())
        gradient = np.concatenate((gradient_beta, [gradient_intercept]))
        final_gradient = float(np.max(np.abs(gradient)))
        iterations = iteration
        if final_gradient <= tolerance:
            converged = True
            break

        curvature = weights * probability * (1.0 - probability) / weight_total
        hessian_beta = (
            standardized.T @ (standardized * curvature[:, None])
            + ridge_lambda * identity
        )
        hessian_cross = standardized.T @ curvature
        hessian = np.empty((x.shape[1] + 1, x.shape[1] + 1), dtype=np.float64)
        hessian[:-1, :-1] = hessian_beta
        hessian[:-1, -1] = hessian_cross
        hessian[-1, :-1] = hessian_cross
        hessian[-1, -1] = float(curvature.sum())
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("ridge Newton Hessian is singular") from exc
        directional_derivative = float(np.dot(gradient, direction))
        if not math.isfinite(directional_derivative) or directional_derivative <= 0:
            raise RuntimeError("ridge Newton direction is not descending")
        final_decrement = 0.5 * directional_derivative
        if final_decrement <= tolerance:
            converged = True
            break
        current_objective = objective(beta, intercept)
        step = 1.0
        accepted = False
        for _ in range(60):
            proposal_beta = beta - step * direction[:-1]
            proposal_intercept = intercept - step * float(direction[-1])
            proposal_objective = objective(proposal_beta, proposal_intercept)
            if proposal_objective <= current_objective - 1e-4 * step * directional_derivative:
                beta = proposal_beta
                intercept = proposal_intercept
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise RuntimeError("ridge Newton line search failed")
    if not converged:
        raise RuntimeError(
            "ridge solver did not converge within "
            f"{maximum_iterations} iterations; gradient={final_gradient}"
        )
    return {
        "mean": mean,
        "scale": scale,
        "coefficient": beta,
        "intercept": intercept,
        "iterations": iterations,
        "gradient_infinity_norm": final_gradient,
        "half_newton_decrement": final_decrement,
        "objective": objective(beta, intercept),
        "positive_rows": positives,
        "negative_rows": negatives,
        "weight_total": weight_total,
        "converged": True,
    }


def build_surface_control(
    public_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run 24 grouped leave-one-task-out surface fits and return scores/folds."""

    tasks = {row["task_id"]: row for row in public_rows}
    live = public_live_map(audit_rows)
    if set(tasks) != set(live) or len(tasks) != 24:
        raise ValueError("surface control requires the complete mechanics split")
    ordered_task_ids = [row["task_id"] for row in public_rows]
    feature_by_key: dict[tuple[str, str], np.ndarray] = {}
    label_by_key: dict[tuple[str, str], int] = {}
    for task_id in ordered_task_ids:
        for alias, candidate in zip(ALIASES, CONCRETE_OPERATIONS, strict=True):
            key = (task_id, alias)
            feature_by_key[key] = surface_features(tasks[task_id], candidate)
            label_by_key[key] = int(alias in live[task_id])

    score_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for held_task_id in ordered_task_ids:
        train_keys = [key for key in feature_by_key if key[0] != held_task_id]
        held_keys = [(held_task_id, alias) for alias in ALIASES]
        train_x = np.stack([feature_by_key[key] for key in train_keys])
        train_y = np.asarray([label_by_key[key] for key in train_keys], dtype=np.float64)
        fit = fit_balanced_ridge(train_x, train_y)
        held_x = np.stack([feature_by_key[key] for key in held_keys])
        standardized = (held_x - fit["mean"]) / fit["scale"]
        scores = standardized @ fit["coefficient"] + fit["intercept"]
        if not np.isfinite(scores).all():
            raise RuntimeError("surface control produced nonfinite scores")
        held_rows = []
        for key, score in zip(held_keys, scores, strict=True):
            row = {
                "task_id": key[0],
                "candidate_alias": key[1],
                "score": float(score),
                "public_live": bool(label_by_key[key]),
            }
            held_rows.append(row)
            score_rows.append(row)
        fold_rows.append(
            {
                "held_task_id": held_task_id,
                "training_rows": len(train_keys),
                "held_rows": len(held_keys),
                "training_task_ids": [
                    task_id for task_id in ordered_task_ids if task_id != held_task_id
                ],
                "feature_names": list(SURFACE_FEATURE_NAMES),
                "solver": SURFACE_SOLVER,
                "mean": fit["mean"].tolist(),
                "scale": fit["scale"].tolist(),
                "coefficient": fit["coefficient"].tolist(),
                "intercept": fit["intercept"],
                "iterations": fit["iterations"],
                "gradient_infinity_norm": fit["gradient_infinity_norm"],
                "half_newton_decrement": fit["half_newton_decrement"],
                "objective": fit["objective"],
                "positive_rows": fit["positive_rows"],
                "negative_rows": fit["negative_rows"],
                "converged": fit["converged"],
                "held_scores_sha256": hashlib.sha256(
                    canonical_json(held_rows).encode("utf-8")
                ).hexdigest(),
            }
        )
    return score_rows, fold_rows


def build_random_control(
    public_rows: Sequence[Mapping[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in public_rows:
        task_id = task["task_id"]
        order = sorted(
            ALIASES,
            key=lambda alias: stable_digest(
                str(seed), task_id, alias, "frozen-random-rank-v1"
            ),
        )
        scores = {alias: float(len(ALIASES) - index) for index, alias in enumerate(order)}
        rows.extend(
            {
                "task_id": task_id,
                "candidate_alias": alias,
                "score": scores[alias],
            }
            for alias in ALIASES
        )
    return rows


def _logprob_value(value: Any) -> float:
    if isinstance(value, Mapping):
        value = value.get("logprob")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("requested logprob has invalid type")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("requested logprob is not finite")
    return result


def extract_requested_logprobs(
    output: Mapping[str, Any], requested: Mapping[str, int]
) -> dict[str, float]:
    """Extract requested IDs from position-zero raw runner log probabilities."""

    if int(output.get("n_sampled_tokens", -1)) != 1:
        raise ValueError("ranking request did not sample exactly one token")
    if output.get("stage2_logprobs") is not None or output.get("seed_stage2") is not None:
        raise ValueError("ranking request unexpectedly used a second stage")
    steps = output.get("stage1_logprobs")
    if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], dict):
        raise ValueError("ranking request lacks one position-zero logprob map")
    step = steps[0]
    result: dict[str, float] = {}
    for label, token_id in requested.items():
        matches = [key for key in step if str(key) == str(token_id)]
        if len(matches) != 1:
            raise ValueError(f"requested token {label}/{token_id} is not present exactly once")
        result[label] = _logprob_value(step[matches[0]])
    return result


def binary_rank_score(
    output: Mapping[str, Any], *, live_alias: str, token_ids: Mapping[str, int]
) -> tuple[float, dict[str, float]]:
    if live_alias not in {"A", "B"} or set(token_ids) != {"A", "B"}:
        raise ValueError("binary orientation is invalid")
    values = extract_requested_logprobs(output, token_ids)
    dead_alias = "B" if live_alias == "A" else "A"
    score = float(np.float32(values[live_alias]) - np.float32(values[dead_alias]))
    if not math.isfinite(score):
        raise ValueError("binary oriented score is not finite")
    return score, values


def listwise_rank_scores(
    output: Mapping[str, Any], *, token_ids: Mapping[str, int]
) -> tuple[dict[str, float], dict[str, float]]:
    if tuple(token_ids) != ALIASES:
        raise ValueError("listwise aliases changed")
    values = extract_requested_logprobs(output, token_ids)
    scores = {alias: float(np.float32(value)) for alias, value in values.items()}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("listwise score is not finite")
    return scores, values


def rank_candidates(task_id: str, scores: Mapping[str, float]) -> list[str]:
    if set(scores) != set(ALIASES):
        raise ValueError("ranker must score every fixed candidate")
    if not all(math.isfinite(float(value)) for value in scores.values()):
        raise ValueError("ranker scores must be finite")
    return sorted(
        ALIASES,
        key=lambda alias: (
            -float(scores[alias]),
            stable_digest(task_id, alias, "rank-tie-v1"),
        ),
    )


def ranking_metrics(
    score_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    live = public_live_map(audit_rows)
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in score_rows:
        task_id = row.get("task_id")
        alias = row.get("candidate_alias")
        score = row.get("score")
        if task_id not in live or alias not in ALIASES:
            raise ValueError("score row references an unknown task or candidate")
        if alias in grouped[task_id]:
            raise ValueError("duplicate task/candidate score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("score row has invalid score")
        grouped[task_id][alias] = float(score)
    if set(grouped) != set(live):
        raise ValueError("ranker task coverage is incomplete")

    task_rows: list[dict[str, Any]] = []
    retrieved_operations: set[str] = set()
    for task_id in [row["task_id"] for row in audit_rows]:
        order = rank_candidates(task_id, grouped[task_id])
        live_aliases = live[task_id]
        top4_live = [alias for alias in order[:4] if alias in live_aliases]
        top8_live = [alias for alias in order[:8] if alias in live_aliases]
        first_rank = min(order.index(alias) + 1 for alias in live_aliases)
        retrieved_operations.update(top4_live)
        task_rows.append(
            {
                "task_id": task_id,
                "live_aliases": sorted(live_aliases),
                "ranked_aliases": order,
                "top4_aliases": order[:4],
                "recall_at_4": len(top4_live) / len(live_aliases),
                "hit_at_4": bool(top4_live),
                "recall_at_8": len(top8_live) / len(live_aliases),
                "reciprocal_rank": 1.0 / first_rank,
            }
        )
    return {
        "score_rows": len(score_rows),
        "finite_score_rows": len(score_rows),
        "mean_recall_at_4": _mean([row["recall_at_4"] for row in task_rows]),
        "mean_hit_at_4": _mean([float(row["hit_at_4"]) for row in task_rows]),
        "mean_recall_at_8": _mean([row["recall_at_8"] for row in task_rows]),
        "mean_reciprocal_rank": _mean(
            [row["reciprocal_rank"] for row in task_rows]
        ),
        "hit_tasks": sum(bool(row["hit_at_4"]) for row in task_rows),
        "retrieved_live_operation_count": len(retrieved_operations),
        "retrieved_live_aliases": sorted(retrieved_operations),
        "task_rows": task_rows,
    }


def answer_cap_contact(output: Mapping[str, Any], answer_cap: int) -> bool:
    """Conservatively include exact-cap answers and final length finishes."""

    if answer_cap < 1:
        raise ValueError("answer cap must be positive")
    return bool(
        int(output.get("n_answer_tokens", -1)) >= answer_cap
        or output.get("stage2_finish_reason") == "length"
        or output.get("finish_reason") == "length"
    )


def score_generation_arm(
    public_by_id: Mapping[str, Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    answer_cap: int,
    direct: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in raw_rows:
        row_id = row.get("id")
        meta = row.get("meta")
        outputs = row.get("outputs")
        if not isinstance(row_id, str) or row_id in seen_ids:
            raise ValueError("generation row ID is missing or duplicated")
        seen_ids.add(row_id)
        if not isinstance(meta, dict) or not isinstance(outputs, list) or len(outputs) != 1:
            raise ValueError("generation row schema is invalid")
        task_id = meta.get("task_id")
        if task_id not in public_by_id:
            raise ValueError("generation row references an unknown task")
        output = outputs[0]
        if not isinstance(output, dict):
            raise ValueError("generation output is invalid")
        candidate: BoundOperation | None
        if direct:
            candidate = None
        else:
            candidate = operation_from_record(meta.get("candidate"))
        scored = score_candidate(
            dict(public_by_id[task_id]),
            text=str(output.get("text", "")),
            candidate=candidate,
        )
        parsed = parse_program(str(output.get("text", "")), arity=3 if direct else 2)
        # score_candidate's selector eligibility also checks every public probe.
        # Mechanics A's registered capability gate is raw visible execution, so
        # recompute it independently and report probe eligibility separately.
        raw_visible_pass = False
        full_program = None
        if parsed["parsed"]:
            full_program = (
                tuple(parsed["program"])
                if direct
                else (candidate, *tuple(parsed["program"]))
            )
            raw_visible_pass = all(
                (value := apply_pipeline(visible["input"], full_program)) is not INVALID
                and value == visible["output"]
                for visible in public_by_id[task_id]["visible"]
            )
        exact_echo = False
        if not direct and meta.get("condition") == "suffix_echo" and parsed["parsed"]:
            supplied = tuple(
                operation_from_record(value) for value in meta["supplied_suffix"]
            )
            exact_echo = tuple(parsed["program"]) == supplied
        suffix_parameterized = bool(
            not direct
            and parsed["parsed"]
            and any(parameter is not None for _name, parameter in parsed["program"])
        )
        suffix_parameter_free = bool(
            not direct
            and parsed["parsed"]
            and any(parameter is None for _name, parameter in parsed["program"])
        )
        scored_rows.append(
            {
                "id": row_id,
                "task_id": task_id,
                "candidate_alias": None if direct else operation_alias(candidate),
                "parsed": bool(parsed["parsed"]),
                "parse_error": parsed["error"],
                "canonical_suffix": parsed["canonical"] if not direct else None,
                "alias_program": (
                    alias_program(full_program) if full_program is not None else None
                ),
                "canonical_program": (
                    canonical_program(full_program)
                    if full_program is not None
                    else None
                ),
                "raw_visible_pass": raw_visible_pass,
                "probe_eligible": bool(scored.get("visible_pass", False)),
                "probe_vector": scored.get("probe_vector"),
                "answer_cap_contact": answer_cap_contact(output, answer_cap),
                "exact_echo": exact_echo,
                "successful_suffix_has_parameterized": bool(
                    raw_visible_pass and suffix_parameterized
                ),
                "successful_suffix_has_parameter_free": bool(
                    raw_visible_pass and suffix_parameter_free
                ),
                "seed_stage1": output.get("seed_stage1"),
                "seed_stage2": output.get("seed_stage2"),
                "n_sampled_tokens": output.get("n_sampled_tokens"),
                "n_stage1_prompt_tokens": output.get("n_stage1_prompt_tokens"),
                "n_stage2_prompt_tokens": output.get("n_stage2_prompt_tokens"),
            }
        )
    task_success: dict[str, bool] = {}
    for task_id in public_by_id:
        relevant = [row for row in scored_rows if row["task_id"] == task_id]
        if relevant:
            task_success[task_id] = any(row["raw_visible_pass"] for row in relevant)
    count = len(scored_rows)
    if count == 0:
        raise ValueError("generation arm is empty")
    return scored_rows, {
        "rows": count,
        "parse_successes": sum(row["parsed"] for row in scored_rows),
        "parse_rate": _mean([float(row["parsed"]) for row in scored_rows]),
        "answer_cap_contacts": sum(row["answer_cap_contact"] for row in scored_rows),
        "answer_cap_contact_rate": _mean(
            [float(row["answer_cap_contact"]) for row in scored_rows]
        ),
        "raw_visible_successes": sum(row["raw_visible_pass"] for row in scored_rows),
        "raw_visible_success_rate": _mean(
            [float(row["raw_visible_pass"]) for row in scored_rows]
        ),
        "probe_eligible_successes": sum(row["probe_eligible"] for row in scored_rows),
        "exact_echoes": sum(row["exact_echo"] for row in scored_rows),
        "task_any_live_successes": sum(task_success.values()),
        "task_any_live_success_rate": (
            _mean([float(value) for value in task_success.values()])
            if task_success
            else None
        ),
        "task_success": task_success,
        "successful_parameterized_rows": sum(
            row["successful_suffix_has_parameterized"] for row in scored_rows
        ),
        "successful_parameter_free_rows": sum(
            row["successful_suffix_has_parameter_free"] for row in scored_rows
        ),
    }


def _minimum_successes(rate: float, rows: int) -> int:
    if not 0.0 <= float(rate) <= 1.0 or rows < 1:
        raise ValueError("invalid minimum-rate gate")
    return int((Decimal(str(rate)) * rows).to_integral_value(rounding=ROUND_CEILING))


def _maximum_failures(rate: float, rows: int) -> int:
    if not 0.0 <= float(rate) <= 1.0 or rows < 1:
        raise ValueError("invalid maximum-rate gate")
    return int((Decimal(str(rate)) * rows).to_integral_value(rounding=ROUND_FLOOR))


def _at_least(value: float, threshold: float) -> bool:
    return float(value) >= float(threshold) or math.isclose(
        float(value), float(threshold), rel_tol=0.0, abs_tol=1e-12
    )


def decide_mechanics_a(
    generation_metrics: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the registered Mechanics A gates using exact integer boundaries."""

    suffix_arms = (
        "suffix_materialized",
        "suffix_name_only",
        "suffix_shuffled",
        "suffix_echo",
    )
    required = {*suffix_arms, "direct"}
    if set(generation_metrics) != required:
        raise ValueError("Mechanics A arm inventory changed")
    if any(int(generation_metrics[name].get("rows", -1)) != 52 for name in suffix_arms):
        raise ValueError("Mechanics A suffix row count changed")
    if int(generation_metrics["direct"].get("rows", -1)) != 24:
        raise ValueError("Mechanics A direct row count changed")

    suffix_parse_min = _minimum_successes(
        float(thresholds["suffix_parse_rate_min"]), 52
    )
    suffix_cap_max = _maximum_failures(
        float(thresholds["suffix_answer_limit_contact_max"]), 52
    )
    direct_parse_min = _minimum_successes(
        float(thresholds["direct_parse_rate_min"]), 24
    )
    direct_cap_max = _maximum_failures(
        float(thresholds["direct_answer_limit_contact_max"]), 24
    )
    echo_visible_min = _minimum_successes(
        float(thresholds["supplied_suffix_visible_execution_min"]), 52
    )
    materialized_task_min = _minimum_successes(
        float(thresholds["materialized_live_suffix_visible_pass_min"]), 24
    )
    gain_name_min = _minimum_successes(
        float(thresholds["live_suffix_gain_vs_name_min"]), 24
    )
    gain_shuffled_min = _minimum_successes(
        float(thresholds["live_suffix_gain_vs_shuffled_min"]), 24
    )

    suffix_interface_by_arm = {
        name: bool(
            int(generation_metrics[name]["parse_successes"]) >= suffix_parse_min
            and int(generation_metrics[name]["answer_cap_contacts"])
            <= suffix_cap_max
        )
        for name in suffix_arms
    }
    direct_interface = bool(
        int(generation_metrics["direct"]["parse_successes"]) >= direct_parse_min
        and int(generation_metrics["direct"]["answer_cap_contacts"])
        <= direct_cap_max
    )
    echo_interface = bool(
        int(generation_metrics["suffix_echo"]["raw_visible_successes"])
        >= echo_visible_min
    )
    abi_valid = all(suffix_interface_by_arm.values()) and direct_interface and echo_interface

    materialized = generation_metrics["suffix_materialized"]
    materialized_tasks = int(materialized["task_any_live_successes"])
    name_tasks = int(generation_metrics["suffix_name_only"]["task_any_live_successes"])
    shuffled_tasks = int(
        generation_metrics["suffix_shuffled"]["task_any_live_successes"]
    )
    scientific_checks = {
        "materialized_task_floor": materialized_tasks >= materialized_task_min,
        "gain_over_name": materialized_tasks - name_tasks >= gain_name_min,
        "gain_over_shuffled": (
            materialized_tasks - shuffled_tasks >= gain_shuffled_min
        ),
        "parameterized_success": int(materialized["successful_parameterized_rows"])
        >= 1,
        "parameter_free_success": int(
            materialized["successful_parameter_free_rows"]
        )
        >= 1,
    }
    scientific_pass = all(scientific_checks.values())
    if not abi_valid:
        decision = "MECHANICS_INTERFACE_INVALID"
    elif not scientific_pass:
        decision = "NO_ACTIONABLE_MATERIALIZED_RESIDUAL"
    else:
        decision = "MATERIALIZED_SUFFIX_INTERFACE_PASS"
    return {
        "decision": decision,
        "abi_valid": abi_valid,
        "scientific_pass": scientific_pass,
        "suffix_interface_by_arm": suffix_interface_by_arm,
        "direct_interface": direct_interface,
        "echo_interface": echo_interface,
        "scientific_checks": scientific_checks,
        "integer_boundaries": {
            "suffix_parse_successes_min": suffix_parse_min,
            "suffix_answer_cap_contacts_max": suffix_cap_max,
            "direct_parse_successes_min": direct_parse_min,
            "direct_answer_cap_contacts_max": direct_cap_max,
            "echo_visible_successes_min": echo_visible_min,
            "materialized_task_successes_min": materialized_task_min,
            "task_gain_over_name_min": gain_name_min,
            "task_gain_over_shuffled_min": gain_shuffled_min,
        },
    }


def decide_mechanics_b(
    ranking_results: Mapping[str, Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    *,
    registered_top4_static_context_fit: bool,
) -> dict[str, Any]:
    """Apply the registered Mechanics B gates to authenticated rank metrics."""

    required = {
        "viability_materialized",
        "viability_name_only",
        "viability_shuffled",
        "listwise",
        "surface",
        "random",
    }
    if set(ranking_results) != required:
        raise ValueError("Mechanics B ranker inventory changed")
    if any(
        int(ranking_results[name].get("score_rows", -1)) != 576
        or int(ranking_results[name].get("finite_score_rows", -1)) != 576
        for name in required
    ):
        raise ValueError("Mechanics B score coverage changed")
    materialized = ranking_results["viability_materialized"]
    comparisons = {
        "name_only": float(materialized["mean_recall_at_4"])
        - float(ranking_results["viability_name_only"]["mean_recall_at_4"]),
        "shuffled": float(materialized["mean_recall_at_4"])
        - float(ranking_results["viability_shuffled"]["mean_recall_at_4"]),
        "listwise": float(materialized["mean_recall_at_4"])
        - float(ranking_results["listwise"]["mean_recall_at_4"]),
        "surface": float(materialized["mean_recall_at_4"])
        - float(ranking_results["surface"]["mean_recall_at_4"]),
        "random": float(materialized["mean_recall_at_4"])
        - float(ranking_results["random"]["mean_recall_at_4"]),
    }
    checks = {
        "finite_scores": True,
        "recall_floor": _at_least(
            float(materialized["mean_recall_at_4"]),
            float(thresholds["materialized_live_recall_at_4_min"]),
        ),
        "hit_floor": _at_least(
            float(materialized["mean_hit_at_4"]),
            float(thresholds["materialized_live_hit_at_4_min"]),
        ),
        "gain_over_name": _at_least(
            comparisons["name_only"],
            float(thresholds["recall_at_4_gain_vs_name_min"]),
        ),
        "gain_over_shuffled": _at_least(
            comparisons["shuffled"],
            float(thresholds["recall_at_4_gain_vs_shuffled_min"]),
        ),
        "gain_over_listwise": _at_least(
            comparisons["listwise"],
            float(thresholds["recall_at_4_gain_vs_listwise_min"]),
        ),
        "gain_over_surface": _at_least(
            comparisons["surface"],
            float(thresholds["recall_at_4_gain_vs_surface_min"]),
        ),
        "gain_over_random": _at_least(
            comparisons["random"],
            float(thresholds["recall_at_4_gain_vs_random_min"]),
        ),
        "task_support": int(materialized["hit_tasks"])
        >= int(thresholds["successful_task_support_min"]),
        "operation_support": int(materialized["retrieved_live_operation_count"])
        >= int(thresholds["live_operation_support_min"]),
        "static_context_fit": registered_top4_static_context_fit is True,
    }
    passed = all(checks.values())
    return {
        "decision": (
            "CHEAP_SIBLING_RANKING_PASS"
            if passed
            else "CHEAP_SIBLING_RANKING_FAIL"
        ),
        "pass": passed,
        "checks": checks,
        "recall_at_4_gains": comparisons,
        "authenticated_model_score_rows": 4 * 576,
        "authenticated_requested_raw_logprob_values": 4032,
    }


def mechanics_authorization(
    mechanics_a_decision: str, mechanics_b_decision: str
) -> dict[str, bool]:
    """Keep the descriptive top-four branch unable to veto the all-24 primary."""

    valid_a = {
        "MECHANICS_INTERFACE_INVALID",
        "NO_ACTIONABLE_MATERIALIZED_RESIDUAL",
        "MATERIALIZED_SUFFIX_INTERFACE_PASS",
    }
    valid_b = {"CHEAP_SIBLING_RANKING_PASS", "CHEAP_SIBLING_RANKING_FAIL"}
    if mechanics_a_decision not in valid_a or mechanics_b_decision not in valid_b:
        raise ValueError("unknown mechanics decision")
    primary = mechanics_a_decision == "MATERIALIZED_SUFFIX_INTERFACE_PASS"
    return {
        "qualification_authorized": primary,
        "top4_secondary_authorized": primary
        and mechanics_b_decision == "CHEAP_SIBLING_RANKING_PASS",
    }
