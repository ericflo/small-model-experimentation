"""Outcome-blind branch and resource plans for early hypothesis forking."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence
from typing import Any

from task_data import (
    CONCRETE_OPERATIONS,
    BoundOperation,
    apply_pipeline,
    canonical_operation,
    operation_record,
    scheduled_first_operation,
)


TASK_ID_RE = re.compile(r"^(qualification|confirmation)-(\d{5})$")


def _stable_seed(base_seed: int, *, domain: str, key: str) -> int:
    payload = f"{base_seed}\0{domain}\0{key}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _task_coordinates(task_id: str) -> tuple[str, int]:
    match = TASK_ID_RE.fullmatch(task_id)
    if match is None:
        raise ValueError(f"unsupported task id for branch planning: {task_id!r}")
    return match.group(1), int(match.group(2))


def expected_first_operation(task_id: str) -> BoundOperation:
    """Return the task-index schedule without opening any gold artifact."""

    _split, index = _task_coordinates(task_id)
    return scheduled_first_operation(index)


def _balanced_gold_slot(task_id: str, branch_seed: int) -> int:
    """Assign every slot once per 24-task block using only public task identity."""

    split, index = _task_coordinates(task_id)
    block, within_block = divmod(index, len(CONCRETE_OPERATIONS))
    positions = list(range(len(CONCRETE_OPERATIONS)))
    random.Random(
        _stable_seed(branch_seed, domain="gold-slot-block", key=f"{split}:{block}")
    ).shuffle(positions)
    return positions[within_block]


def branch_operations(task_id: str, branch_seed: int) -> tuple[BoundOperation, ...]:
    """Create a non-cyclic task permutation without reading task outputs or gold."""

    operations = list(CONCRETE_OPERATIONS)
    random.Random(
        _stable_seed(branch_seed, domain="operation-permutation", key=task_id)
    ).shuffle(operations)
    expected = expected_first_operation(task_id)
    desired_slot = _balanced_gold_slot(task_id, branch_seed)
    current_slot = operations.index(expected)
    operations[current_slot], operations[desired_slot] = (
        operations[desired_slot],
        operations[current_slot],
    )
    return tuple(operations)


def branch_plan(
    task_id: str,
    branch_seed: int,
    *,
    behavior_panel: Sequence[list[int]],
) -> dict[str, Any]:
    """Serialize the complete slot→operation→behavior composition."""

    operations = branch_operations(task_id, branch_seed)
    rows = []
    for slot, operation in enumerate(operations):
        signature = [apply_pipeline(list(values), [operation]) for values in behavior_panel]
        rows.append(
            {
                "slot": slot,
                "operation": operation_record(operation),
                "canonical_operation": canonical_operation(operation),
                "behavior_signature": signature,
            }
        )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "task_id": task_id,
        "gold_slot_from_public_schedule": operations.index(
            expected_first_operation(task_id)
        ),
        "rows": rows,
        "composed_map_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _validated_cost_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, int | str]]:
    validated: list[dict[str, int | str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "sampled_tokens",
            "logical_model_tokens",
        }:
            raise ValueError("resource rows require exact outcome-free schema")
        record_id = row["id"]
        sampled = row["sampled_tokens"]
        logical = row["logical_model_tokens"]
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError("resource row ids must be unique nonempty strings")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (sampled, logical)
        ):
            raise ValueError("resource token counts must be nonnegative integers")
        if logical < sampled:
            raise ValueError("logical model tokens cannot be below sampled tokens")
        seen.add(record_id)
        validated.append(
            {
                "id": record_id,
                "sampled_tokens": sampled,
                "logical_model_tokens": logical,
            }
        )
    if not validated:
        raise ValueError("resource master pool is empty")
    return validated


def _prefix_match(
    rows: Sequence[dict[str, int | str]], target: int, field: str
) -> dict[str, Any]:
    cumulative = 0
    under_count = 0
    over_count: int | None = None
    cumulative_values: list[int] = []
    for index, row in enumerate(rows, start=1):
        cumulative += int(row[field])
        cumulative_values.append(cumulative)
        if cumulative <= target:
            under_count = index
        if over_count is None and cumulative >= target:
            over_count = index
    exhausted = over_count is None
    return {
        "target": target,
        "under_count": under_count,
        "under_ids": [str(row["id"]) for row in rows[:under_count]],
        "under_cost": cumulative_values[under_count - 1] if under_count else 0,
        "over_count": over_count,
        "over_ids": (
            [str(row["id"]) for row in rows[:over_count]]
            if over_count is not None
            else []
        ),
        "over_cost": (
            cumulative_values[over_count - 1] if over_count is not None else None
        ),
        "pool_exhausted": exhausted,
    }


def freeze_resource_plan(
    master_rows: Sequence[dict[str, Any]],
    *,
    target_sampled_tokens: int,
    target_logical_model_tokens: int,
    full_k: int = 24,
) -> dict[str, Any]:
    """Freeze matched prefixes from ordered receipts without any outcome field."""

    for value in (target_sampled_tokens, target_logical_model_tokens, full_k):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("resource targets and full_k must be positive integers")
    rows = _validated_cost_rows(master_rows)
    if len(rows) < full_k:
        raise ValueError("master pool is smaller than the registered full-K reference")
    plan = {
        "sampled": _prefix_match(rows, target_sampled_tokens, "sampled_tokens"),
        "logical": _prefix_match(
            rows, target_logical_model_tokens, "logical_model_tokens"
        ),
        "full_k_ids": [str(row["id"]) for row in rows[:full_k]],
        "master_ids": [str(row["id"]) for row in rows],
    }
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return {
        **plan,
        "resource_plan_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
