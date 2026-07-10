#!/usr/bin/env python3
"""Exact CPU full-brute deployment baseline for the depth-5 primary split.

The implementation uses the experiment-local exact behavioral state quotient.
That quotient is only an execution optimization: it exhaustively represents all
``16 ** depth`` operation-type skeletons (and all concrete parameter fills),
backtracks every concrete pipeline that passes the first visible case, and then
checks each survivor on every remaining visible case.  Label-probe and hidden
cases are never consulted during search or selection.

Every visible passer is sent to the shared visible-only consensus selector.
Hidden examples are read only afterwards to report pool coverage and selected
success.  Per-task receipts distinguish the logical tree size from the exact
physical transitions and primitive applications performed by the quotient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import experiment_common as C  # noqa: E402
import families as F  # noqa: E402
import oracle_data as O  # noqa: E402


BASELINE_NAME = "full_brute"
DEPLOYMENT_LABEL_SPLITS = ("visible",)


def _serialize_pipeline(pipeline: F.Pipeline | None) -> list[list[Any]] | None:
    if pipeline is None:
        return None
    return [[name, parameter] for name, parameter in pipeline]


def _logical_search_space(depth: int) -> dict[str, int]:
    """Return the unquotiented tree size represented by the exact search."""

    return {
        "operation_types": len(F.TYPES),
        "concrete_operations": len(F.CONCRETE_OPS),
        "type_skeleton_leaves": len(F.TYPES) ** depth,
        "concrete_parameterized_leaves": len(F.CONCRETE_OPS) ** depth,
        "type_prefix_nodes_including_root": sum(
            len(F.TYPES) ** level for level in range(depth + 1)
        ),
    }


def run_full_brute_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Exhaust all visible-solving exact-length pipelines for one task.

    This function is intentionally depth-generic so a tiny task can verify it in
    unit tests.  The command-line entry point separately enforces depth 5.
    """

    started = time.perf_counter()
    depth = int(task["depth"])
    if depth < 1:
        raise ValueError("full brute requires a positive task depth")
    if not task.get("visible"):
        raise ValueError("full brute requires at least one visible example")

    oracle = O.ExactSemanticOracle(
        task,
        depth=depth,
        label_source_splits=DEPLOYMENT_LABEL_SPLITS,
    ).build()

    # The graph enumerator yields every successful concrete operation path, not
    # merely one representative per behavioral state or per type skeleton.
    candidates = list(oracle.iter_successful_pipelines())
    successful_skeletons = {
        tuple(name for name, _parameter in pipeline) for pipeline in candidates
    }
    oracle.accounting.successful_type_skeletons = len(successful_skeletons)

    selected, selector = C.consensus_select(candidates, task)

    # Hidden data enter for grading only, after both enumeration and selection
    # are complete.  Grade the whole visible-passer pool to measure coverage.
    hidden_successes = [C.hidden_grade(pipeline, task) for pipeline in candidates]
    selected_hidden_success = C.hidden_grade(selected, task)

    accounting = oracle.accounting.to_dict()
    if accounting["transition_requests"] != accounting["vector_transitions_computed"]:
        raise AssertionError("every requested quotient transition must be computed exactly once")
    if accounting["successful_concrete_pipelines"] != len(candidates):
        raise AssertionError("oracle receipt disagrees with exhaustive visible-passer list")
    if selector["candidate_count"] != len(candidates):
        raise AssertionError("shared selector did not receive the complete visible-passer pool")

    return {
        "schema_version": 1,
        "baseline": BASELINE_NAME,
        "task_id": str(task["task_id"]),
        "depth": depth,
        "exact": True,
        "search_algorithm": "exact_first_visible_case_behavioral_state_quotient",
        "quotient_rule": (
            "identical output on the frozen first visible input; deterministic "
            "suffix congruence; all successful concrete paths backtracked"
        ),
        "label_source_splits": list(DEPLOYMENT_LABEL_SPLITS),
        "label_probe_used_for_search_or_selection": False,
        "hidden_used_for_search_or_selection": False,
        "logical_search_space": _logical_search_space(depth),
        "visible_successful_pipeline_count": len(candidates),
        "visible_successful_type_skeleton_count": len(successful_skeletons),
        "visible_successful_pipelines": [
            _serialize_pipeline(pipeline) for pipeline in candidates
        ],
        "path_coverage_hidden": any(hidden_successes),
        "hidden_successful_pipeline_count": sum(hidden_successes),
        "selected_hidden_success": selected_hidden_success,
        "selected_pipeline": _serialize_pipeline(selected),
        "selected_ops": (
            [F.format_op(operation) for operation in selected]
            if selected is not None
            else None
        ),
        "selector": selector,
        "accounting": accounting,
        "wall_seconds": time.perf_counter() - started,
    }


def run_tasks(
    tasks: Sequence[Mapping[str, Any]], *, workers: int
) -> list[dict[str, Any]]:
    """Run independent tasks in deterministic input order, in parallel."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if not tasks:
        return []
    if workers == 1:
        return [run_full_brute_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        # executor.map preserves task order, keeping output byte-stable apart
        # from measured wall times.
        return list(executor.map(run_full_brute_task, tasks, chunksize=1))


def build_result(
    tasks: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    smoke: bool,
    task_source: Path | None = None,
) -> dict[str, Any]:
    """Run the baseline and assemble its aggregate machine-readable receipt."""

    started = time.perf_counter()
    rows = run_tasks(tasks, workers=workers)
    aggregate = C.aggregate_counts(row["accounting"] for row in rows)
    logical_type_leaves = sum(
        int(row["logical_search_space"]["type_skeleton_leaves"]) for row in rows
    )
    logical_concrete_leaves = sum(
        int(row["logical_search_space"]["concrete_parameterized_leaves"])
        for row in rows
    )
    source_sha256 = (
        hashlib.sha256(task_source.read_bytes()).hexdigest()
        if task_source is not None
        else None
    )
    return {
        "schema_version": 1,
        "baseline": BASELINE_NAME,
        "exact": True,
        "split": "primary",
        "smoke": bool(smoke),
        "workers": int(workers),
        "task_source": str(task_source) if task_source is not None else None,
        "task_source_sha256": source_sha256,
        "task_count": len(rows),
        "depths": sorted({int(row["depth"]) for row in rows}),
        "label_source_splits": list(DEPLOYMENT_LABEL_SPLITS),
        "label_probe_used_for_search_or_selection": False,
        "hidden_used_for_search_or_selection": False,
        "selector": "experiment_common.consensus_select",
        "path_coverage_rate": (
            sum(bool(row["path_coverage_hidden"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "selected_hidden_success_rate": (
            sum(bool(row["selected_hidden_success"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "logical_accounting": {
            "type_skeleton_leaves": logical_type_leaves,
            "concrete_parameterized_leaves": logical_concrete_leaves,
        },
        "aggregate_accounting": aggregate,
        "sum_task_wall_seconds": sum(float(row["wall_seconds"]) for row in rows),
        "parallel_wall_seconds": time.perf_counter() - started,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="independent task processes (default: up to 4)",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing cached receipt"
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    suffix = "_smoke" if args.smoke else ""
    task_path = EXP / "data" / f"primary_tasks{suffix}.jsonl"
    output_path = EXP / "runs" / f"full_brute{suffix}.json"
    if output_path.exists() and not args.force:
        print(f"[full-brute] cached: {output_path}", flush=True)
        return 0
    tasks = C.load_jsonl(task_path)
    if not tasks:
        raise RuntimeError(f"no primary tasks found at {task_path}")
    wrong_depth = [str(task.get("task_id")) for task in tasks if int(task["depth"]) != 5]
    if wrong_depth:
        raise ValueError(
            "the full-brute deployment command is pre-registered for depth 5; "
            f"wrong-depth tasks: {wrong_depth}"
        )

    result = build_result(
        tasks,
        workers=args.workers,
        smoke=args.smoke,
        task_source=task_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "tasks": result["task_count"],
                "path_coverage_rate": result["path_coverage_rate"],
                "selected_hidden_success_rate": result[
                    "selected_hidden_success_rate"
                ],
                "logical_type_skeleton_leaves": result["logical_accounting"][
                    "type_skeleton_leaves"
                ],
                "parallel_wall_seconds": result["parallel_wall_seconds"],
                "output": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
