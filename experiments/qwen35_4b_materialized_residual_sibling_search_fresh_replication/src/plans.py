"""Outcome-blind taskwise resource accounting and conservative match points."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


def completion_cost(output: dict[str, Any]) -> dict[str, int]:
    required = (
        "n_sampled_tokens",
        "n_stage1_prompt_tokens",
        "n_stage2_prompt_tokens",
    )
    values: dict[str, int] = {}
    for field in required:
        value = output.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid completion cost field {field}")
        values[field] = value
    return {
        "sampled_tokens": values["n_sampled_tokens"],
        "logical_model_tokens": (
            values["n_stage1_prompt_tokens"]
            + values["n_stage2_prompt_tokens"]
            + values["n_sampled_tokens"]
        ),
    }


def pool_cost(outputs: Sequence[dict[str, Any]]) -> dict[str, int]:
    costs = [completion_cost(output) for output in outputs]
    return {
        key: sum(value[key] for value in costs)
        for key in ("sampled_tokens", "logical_model_tokens")
    }


def conservative_first_over(
    outputs: Sequence[dict[str, Any]], *, target: int, metric: str
) -> dict[str, Any]:
    if metric not in {"sampled_tokens", "logical_model_tokens"}:
        raise ValueError("unknown resource metric")
    if not isinstance(target, int) or isinstance(target, bool) or target < 1:
        raise ValueError("target must be a positive integer")
    cumulative = 0
    for index, output in enumerate(outputs):
        cumulative += completion_cost(output)[metric]
        if cumulative >= target:
            return {
                "metric": metric,
                "target": target,
                "first_over_k": index + 1,
                "first_over_cost": cumulative,
                "under_k": index,
                "under_cost": cumulative - completion_cost(output)[metric],
                "pool_exhausted": False,
            }
    return {
        "metric": metric,
        "target": target,
        "first_over_k": None,
        "first_over_cost": cumulative,
        "under_k": len(outputs),
        "under_cost": cumulative,
        "pool_exhausted": True,
    }


def freeze_taskwise_matches(
    *,
    task_id: str,
    treatment_outputs: Sequence[dict[str, Any]],
    direct_outputs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    treatment = pool_cost(treatment_outputs)
    sampled = conservative_first_over(
        direct_outputs, target=treatment["sampled_tokens"], metric="sampled_tokens"
    )
    logical = conservative_first_over(
        direct_outputs,
        target=treatment["logical_model_tokens"],
        metric="logical_model_tokens",
    )
    payload = {
        "task_id": task_id,
        "treatment": treatment,
        "sampled": sampled,
        "logical": logical,
        "direct_pool_rows": len(direct_outputs),
    }
    payload["resource_plan_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def top4_cost(
    *, ranking_outputs: Sequence[dict[str, Any]], suffix_outputs: Sequence[dict[str, Any]]
) -> dict[str, int]:
    if len(ranking_outputs) != 24 or len(suffix_outputs) != 4:
        raise ValueError("top-four policy requires 24 ranks and four suffixes")
    return pool_cost([*ranking_outputs, *suffix_outputs])


def strict_taskwise_dominance(
    top4: dict[str, int], all24: dict[str, int]
) -> bool:
    return all(
        top4[metric] < all24[metric]
        for metric in ("sampled_tokens", "logical_model_tokens")
    )
