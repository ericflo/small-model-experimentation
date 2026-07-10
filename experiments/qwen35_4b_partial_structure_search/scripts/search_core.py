#!/usr/bin/env python3
"""Backend-independent type-prefix search and visible-only deployment helpers."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import experiment_common as C  # noqa: E402
import families as F  # noqa: E402


Prefix = tuple[str, ...]


def stable_unit_score(*parts: object) -> float:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / 2**64


def _stable_uint(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "big")


def _skeleton_from_index(index: int, depth: int) -> Prefix:
    """Decode one base-|TYPES| leaf index in a frozen operation order."""

    if depth < 0:
        raise ValueError("depth must be non-negative")
    total = len(F.TYPES) ** depth
    if index < 0 or index >= total:
        raise ValueError(f"skeleton index {index} is outside [0, {total})")
    digits = [0] * depth
    for position in range(depth - 1, -1, -1):
        index, digit = divmod(index, len(F.TYPES))
        digits[position] = digit
    return tuple(F.TYPES[digit] for digit in digits)


def iter_hash_seeded_skeletons(task_id: str, depth: int) -> Iterator[Prefix]:
    """Visit every complete type skeleton once in a task-seeded permutation.

    An affine permutation avoids materializing and sorting ``16 ** depth``
    hashes.  The offset and coprime stride are hash-derived, so the order is
    deterministic, task-specific, and independent of hidden or oracle data.
    """

    if depth < 1:
        raise ValueError("budget-truncated brute requires positive depth")
    total = len(F.TYPES) ** depth
    offset = _stable_uint("budget_truncated_brute", task_id, depth, "offset") % total
    step = _stable_uint("budget_truncated_brute", task_id, depth, "step") % total
    if step == 0:
        step = 1
    while math.gcd(step, total) != 1:
        step = (step + 1) % total or 1
    for ordinal in range(total):
        yield _skeleton_from_index((offset + ordinal * step) % total, depth)


def expand(frontier: Sequence[Prefix]) -> list[Prefix]:
    return [prefix + (operation,) for prefix in frontier for operation in F.TYPES]


def select_beam(
    candidates: Sequence[Prefix], scores: Mapping[Prefix, float], beam: int
) -> list[Prefix]:
    return sorted(candidates, key=lambda prefix: (-float(scores[prefix]), prefix))[:beam]


def oracle_maps(
    oracle: Mapping[str, Any], depth: int
) -> tuple[dict[Prefix, int], dict[Prefix, int]]:
    return C.live_prefix_maps(C.success_rows(oracle), depth)


def evaluate_leaves(
    task: Mapping[str, Any],
    leaves: Sequence[Prefix],
    *,
    fill_cap: int,
) -> dict[str, Any]:
    """Fill leaves on visible cases, select without labels, then hidden-grade."""
    started = time.perf_counter()
    unique_leaves = list(dict.fromkeys(tuple(leaf) for leaf in leaves))
    remaining = int(fill_cap)
    candidates: list[F.Pipeline] = []
    receipts = []
    for skeleton in unique_leaves:
        if remaining <= 0:
            break
        passers, receipt = C.fills_passing(
            skeleton, task, splits=("visible",), fill_cap=remaining
        )
        remaining -= receipt["parameter_fills_attempted"]
        receipts.append(receipt)
        candidates.extend(passers)
    candidates = list(dict.fromkeys(candidates))
    chosen, selector = C.consensus_select(candidates, task)
    hidden_hits = [C.hidden_grade(pipeline, task) for pipeline in candidates]
    return {
        "leaves": [list(leaf) for leaf in unique_leaves],
        "completed_skeletons": len(unique_leaves),
        "planned_complete_skeletons": len(unique_leaves),
        "attempted_complete_skeletons": len(receipts),
        "fill_cap": fill_cap,
        "fill_cap_used": fill_cap - remaining,
        "visible_passing_concrete_candidates": len(candidates),
        "pool_hidden_coverage": any(hidden_hits),
        "selected_hidden_success": C.hidden_grade(chosen, task),
        "selected_pipeline": [F.format_op(operation) for operation in chosen] if chosen else None,
        "visible_overfit_candidates": sum(not hit for hit in hidden_hits),
        "selector": selector,
        "interpreter_accounting": C.aggregate_counts(receipts),
        "evaluation_wall_seconds": time.perf_counter() - started,
    }


def evaluate_budget_truncated_brute(
    task: Mapping[str, Any], *, fill_cap: int
) -> dict[str, Any]:
    """Evaluate a deterministic permutation of full leaves until the fill cap.

    This is an unguided complete-skeleton enumeration baseline, not a narrow
    beam.  Its logical plan contains every ``16 ** depth`` type skeleton.  The
    shared parameter-fill cap determines how many leading leaves in the frozen
    task-specific permutation are physically attempted.
    """

    started = time.perf_counter()
    if isinstance(fill_cap, bool) or not isinstance(fill_cap, int) or fill_cap < 0:
        raise ValueError("fill_cap must be a non-negative integer")
    task_id = str(task["task_id"])
    depth = int(task["depth"])
    planned = len(F.TYPES) ** depth
    remaining = fill_cap
    attempted_leaves: list[Prefix] = []
    candidates: list[F.Pipeline] = []
    receipts: list[dict[str, int]] = []
    for skeleton in iter_hash_seeded_skeletons(task_id, depth):
        if remaining <= 0:
            break
        passers, receipt = C.fills_passing(
            skeleton, task, splits=("visible",), fill_cap=remaining
        )
        attempted = int(receipt["parameter_fills_attempted"])
        if attempted <= 0:
            raise AssertionError("every complete DSL skeleton must have a parameter fill")
        attempted_leaves.append(skeleton)
        receipts.append(receipt)
        candidates.extend(passers)
        remaining -= attempted

    candidates = list(dict.fromkeys(candidates))
    chosen, selector = C.consensus_select(candidates, task)
    hidden_hits = [C.hidden_grade(pipeline, task) for pipeline in candidates]
    attempted_payload = _json_dumps_prefixes(attempted_leaves)
    return {
        "leaves": [list(leaf) for leaf in attempted_leaves],
        "completed_skeletons": len(attempted_leaves),
        "planned_complete_skeletons": planned,
        "attempted_complete_skeletons": len(attempted_leaves),
        "planned_concrete_parameterized_leaves": len(F.CONCRETE_OPS) ** depth,
        "leaf_order": "task_hash_seeded_affine_permutation",
        "attempted_leaf_order_sha256": hashlib.sha256(attempted_payload).hexdigest(),
        "enumeration_exhausted": len(attempted_leaves) == planned,
        "decoded_type_positions": len(attempted_leaves) * depth,
        "fill_cap": fill_cap,
        "fill_cap_used": fill_cap - remaining,
        "visible_passing_concrete_candidates": len(candidates),
        "pool_hidden_coverage": any(hidden_hits),
        "selected_hidden_success": C.hidden_grade(chosen, task),
        "selected_pipeline": [F.format_op(operation) for operation in chosen] if chosen else None,
        "visible_overfit_candidates": sum(not hit for hit in hidden_hits),
        "selector": selector,
        "interpreter_accounting": C.aggregate_counts(receipts),
        "evaluation_wall_seconds": time.perf_counter() - started,
    }


def _json_dumps_prefixes(prefixes: Sequence[Prefix]) -> bytes:
    """Canonical compact bytes for an attempted-order audit hash."""

    return json.dumps(prefixes, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "task_count": len(rows),
        "expanded_prefix_nodes": sum(
            int(row.get("expanded_prefix_nodes", 0)) for row in rows
        ),
        "planned_complete_skeletons": sum(
            int(row.get("planned_complete_skeletons", row["completed_skeletons"]))
            for row in rows
        ),
        "attempted_complete_skeletons": sum(
            int(row.get("attempted_complete_skeletons", row["completed_skeletons"]))
            for row in rows
        ),
        "parameter_fills_attempted": sum(
            int(row["interpreter_accounting"].get("parameter_fills_attempted", 0))
            for row in rows
        ),
        "sum_task_wall_seconds": sum(float(row["task_wall_seconds"]) for row in rows),
        "sum_evaluation_wall_seconds": sum(
            float(row["evaluation_wall_seconds"]) for row in rows
        ),
    }


def run_budget_truncated_brute(
    tasks: Sequence[Mapping[str, Any]], *, fill_cap: int
) -> dict[str, Any]:
    """Run the unguided complete-leaf enumeration baseline on every task."""

    started = time.perf_counter()
    rows = []
    for task_index, task in enumerate(tasks):
        task_started = time.perf_counter()
        evaluated = evaluate_budget_truncated_brute(task, fill_cap=fill_cap)
        rows.append(
            {
                "task_id": str(task["task_id"]),
                "task_index": task_index,
                "shard": task_index % 2,
                "method": "budget_truncated_brute",
                "layers": [],
                # Complete leaves are depth-D prefix nodes. The affine iterator
                # reaches them directly without expanding their ancestors, so
                # charge exactly the physically decoded leaves.
                "expanded_prefix_nodes": evaluated[
                    "attempted_complete_skeletons"
                ],
                **evaluated,
                "task_wall_seconds": time.perf_counter() - task_started,
            }
        )
    return {
        "schema_version": 1,
        "method": "budget_truncated_brute",
        "beam_width": None,
        "wall_seconds": time.perf_counter() - started,
        "summary": _arm_summary(rows),
        "rows": rows,
    }


def run_static_arm(
    tasks: Sequence[Mapping[str, Any]],
    oracles: Mapping[str, Mapping[str, Any]],
    *,
    method: str,
    beam: int,
    fill_cap: int,
    surface_score: Callable[[Prefix], float] | None = None,
) -> dict[str, Any]:
    if method == "budget_truncated_brute":
        return run_budget_truncated_brute(tasks, fill_cap=fill_cap)
    started = time.perf_counter()
    task_rows = []
    for task_index, task in enumerate(tasks):
        task_started = time.perf_counter()
        task_id = str(task["task_id"])
        depth = int(task["depth"])
        if method == "oracle_live":
            live, fill_counts = oracle_maps(oracles[task_id], depth)
        else:
            # Non-oracle controls must not even consult semantic labels while
            # constructing their deployment frontier. Path-retention diagnostics
            # belong in post-hoc analysis.
            live, fill_counts = {}, {}
        frontier: list[Prefix] = [()]
        layers = []
        for layer in range(1, depth + 1):
            candidates = expand(frontier)
            if method == "oracle_live":
                scores = {
                    prefix: float(live.get(prefix, 0))
                    + 1e-9 * float(fill_counts.get(prefix, 0))
                    for prefix in candidates
                }
            elif method == "uniform_seeded":
                scores = {
                    prefix: stable_unit_score(method, task_id, layer, prefix)
                    for prefix in candidates
                }
            elif method == "lexicographic":
                scores = {prefix: -float(index) for index, prefix in enumerate(candidates)}
            elif method == "surface":
                if surface_score is None:
                    raise ValueError("surface arm requires a fitted surface_score")
                scores = {prefix: float(surface_score(prefix)) for prefix in candidates}
            else:
                raise ValueError(f"unknown static search method: {method}")
            frontier = select_beam(candidates, scores, beam)
            layer_row = {
                "layer": layer,
                "expanded": len(candidates),
                "retained": len(frontier),
            }
            if method == "oracle_live":
                layer_row["retained_live"] = sum(
                    live.get(prefix, 0) > 0 for prefix in frontier
                )
            layers.append(layer_row)
        evaluated = evaluate_leaves(task, frontier, fill_cap=fill_cap)
        task_rows.append(
            {
                "task_id": task_id,
                "task_index": task_index,
                "shard": task_index % 2,
                "method": method,
                "layers": layers,
                "expanded_prefix_nodes": sum(row["expanded"] for row in layers),
                **evaluated,
                "task_wall_seconds": time.perf_counter() - task_started,
            }
        )
    return {
        "schema_version": 1,
        "method": method,
        "beam_width": beam,
        "wall_seconds": time.perf_counter() - started,
        "summary": _arm_summary(task_rows),
        "rows": task_rows,
    }


def fit_surface_prior(calibration_rows: Sequence[Mapping[str, Any]]) -> Callable[[Prefix], float]:
    """Fit a smoothed task-independent op-transition prior on calibration labels."""
    exact: dict[tuple[Any, ...], list[int]] = defaultdict(lambda: [0, 0])
    backoff: dict[tuple[Any, ...], list[int]] = defaultdict(lambda: [0, 0])
    global_by_len: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    pooled_parent_child: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    pooled_child: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    pooled_global = [0, 0]
    for row in calibration_rows:
        prefix = tuple(row["candidate_prefix"])
        parent_last = prefix[-2] if len(prefix) >= 2 else "ROOT"
        child = prefix[-1]
        label = int(bool(row["live"]))
        for table, key in (
            (exact, (len(prefix), parent_last, child)),
            (backoff, (len(prefix), child)),
            (global_by_len, len(prefix)),
        ):
            table[key][0] += label
            table[key][1] += 1
        for table, key in (
            (pooled_parent_child, (parent_last, child)),
            (pooled_child, child),
        ):
            table[key][0] += label
            table[key][1] += 1
        pooled_global[0] += label
        pooled_global[1] += 1

    if pooled_global[1] == 0:
        raise ValueError("surface prior requires at least one calibration row")

    def score(prefix: Prefix) -> float:
        if not prefix:
            raise ValueError("surface prior requires a non-empty prefix")
        parent_last = prefix[-2] if len(prefix) >= 2 else "ROOT"
        child = prefix[-1]
        keys = (
            exact.get((len(prefix), parent_last, child)),
            backoff.get((len(prefix), child)),
            global_by_len.get(len(prefix)),
            pooled_parent_child.get((parent_last, child)),
            pooled_child.get(child),
            pooled_global,
        )
        pos, total = next(value for value in keys if value is not None and value[1] > 0)
        return (pos + 1.0) / (total + 2.0)

    return score
