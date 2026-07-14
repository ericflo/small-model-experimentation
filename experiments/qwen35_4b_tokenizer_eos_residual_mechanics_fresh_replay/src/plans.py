"""Outcome-blind sampled- and logical-token matched-compute prefixes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


def completion_cost(output: dict[str, Any]) -> dict[str, int]:
    values: dict[str, int] = {}
    for field in (
        "n_sampled_tokens",
        "n_stage1_prompt_tokens",
        "n_stage2_prompt_tokens",
    ):
        value = output.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid completion cost field: {field}")
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
        metric: sum(row[metric] for row in costs)
        for metric in ("sampled_tokens", "logical_model_tokens")
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
        cost = completion_cost(output)[metric]
        cumulative += cost
        if cumulative >= target:
            return {
                "metric": metric,
                "target": target,
                "first_over_k": index + 1,
                "first_over_cost": cumulative,
                "under_k": index,
                "under_cost": cumulative - cost,
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
    direct_row_ids: Sequence[str],
) -> dict[str, Any]:
    if (
        len(direct_row_ids) != len(direct_outputs)
        or len(set(direct_row_ids)) != len(direct_row_ids)
        or any(not isinstance(row_id, str) or not row_id for row_id in direct_row_ids)
    ):
        raise ValueError("direct row-ID inventory changed")
    treatment = pool_cost(treatment_outputs)
    sampled = conservative_first_over(
        direct_outputs, target=treatment["sampled_tokens"], metric="sampled_tokens"
    )
    logical = conservative_first_over(
        direct_outputs,
        target=treatment["logical_model_tokens"],
        metric="logical_model_tokens",
    )
    for receipt in (sampled, logical):
        selected_k = (
            len(direct_row_ids)
            if receipt["pool_exhausted"]
            else int(receipt["first_over_k"])
        )
        receipt["selected_direct_row_ids"] = list(direct_row_ids[:selected_k])
        receipt["overshoot"] = (
            None
            if receipt["pool_exhausted"]
            else int(receipt["first_over_cost"]) - int(receipt["target"])
        )
    payload = {
        "task_id": task_id,
        "treatment": treatment,
        "sampled": sampled,
        "logical": logical,
        "direct_pool_rows": len(direct_outputs),
        "direct_pool_row_ids": list(direct_row_ids),
    }
    payload["resource_plan_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload
