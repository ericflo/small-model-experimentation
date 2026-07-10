#!/usr/bin/env python3
"""Exact semantic skeleton oracle and live-prefix dataset builder (CPU only).

The deployable search must never consult these labels.  By default the oracle
uses ``visible + label_probe`` cases and never touches ``hidden``.  It exhausts
an exact-length behavioral state graph on a frozen first pruning case, backtracks
every concrete pipeline that passes that case, validates every survivor on the
remaining label cases, collapses only parameter fills (not operation-type
skeletons), and then labels every partial type prefix with both:

* the number of semantically successful full skeleton completions; and
* the number of successful concrete parameter-fill completions.

The state quotient is exact on the selected case bank because DSL suffixes are
deterministic.  No sample, state, path, or wall-clock cap is accepted here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import families as F  # noqa: E402


@dataclasses.dataclass
class CpuAccounting:
    """Exact integer work receipt; elapsed time is intentionally not a budget unit."""

    # Direct enumeration APIs.
    full_skeletons_yielded: int = 0
    parameter_fills_yielded: int = 0

    # Behavioral graph construction/backtracking.
    transition_requests: int = 0
    vector_transitions_computed: int = 0
    case_operation_applications: int = 0
    invalid_vector_transitions: int = 0
    behavior_states_inserted: int = 0
    pruning_case_candidate_pipelines: int = 0
    validation_pipelines_checked: int = 0
    validation_case_operation_applications: int = 0
    successful_concrete_pipelines: int = 0
    successful_type_skeletons: int = 0
    prefix_rows_emitted: int = 0

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)

    def copy(self) -> "CpuAccounting":
        return dataclasses.replace(self)

    def delta(self, before: "CpuAccounting") -> dict[str, int]:
        return {
            field.name: getattr(self, field.name) - getattr(before, field.name)
            for field in dataclasses.fields(self)
        }


@dataclasses.dataclass(frozen=True)
class SuccessfulSkeleton:
    """One successful full type skeleton, with all fills counted exactly."""

    skeleton: tuple[str, ...]
    parameter_fill_count: int
    representative_pipeline: F.Pipeline

    def to_dict(self) -> dict[str, Any]:
        return {
            "skeleton": list(self.skeleton),
            "parameter_fill_count": self.parameter_fill_count,
            "representative_pipeline": [
                [name, parameter] for name, parameter in self.representative_pipeline
            ],
        }


@dataclasses.dataclass(frozen=True)
class SemanticOracleResult:
    task_id: str
    depth: int
    label_source_splits: tuple[str, ...]
    successful_skeletons: tuple[SuccessfulSkeleton, ...]
    prefix_rows: tuple[dict[str, Any], ...]
    pruning_case_index: int
    behavior_states_by_layer: tuple[int, ...]
    live_behavior_states_by_layer: tuple[int, ...]
    accounting: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_id": self.task_id,
            "depth": self.depth,
            "label_source_splits": list(self.label_source_splits),
            "hidden_cases_used_for_labels": "hidden" in self.label_source_splits,
            "logical_search_space": {
                "full_type_skeletons": len(F.TYPES) ** self.depth,
                "concrete_parameter_fills": len(F.CONCRETE_OPS) ** self.depth,
            },
            "successful_skeleton_count": len(self.successful_skeletons),
            "successful_parameter_fill_count": sum(
                row.parameter_fill_count for row in self.successful_skeletons
            ),
            "successful_skeletons": [row.to_dict() for row in self.successful_skeletons],
            "prefix_rows": list(self.prefix_rows),
            "pruning_case_index": self.pruning_case_index,
            "behavior_graph_case_count": 1,
            "behavior_states_by_layer": list(self.behavior_states_by_layer),
            "live_behavior_states_by_layer": list(self.live_behavior_states_by_layer),
            "accounting": dict(self.accounting),
        }


def enumerate_full_skeletons(
    depth: int, accounting: CpuAccounting | None = None
) -> Iterator[tuple[str, ...]]:
    """Yield all full type skeletons while charging the explicit enumeration."""

    receipt = accounting if accounting is not None else CpuAccounting()
    for skeleton in F.enumerate_skeletons(depth):
        receipt.full_skeletons_yielded += 1
        yield skeleton


def enumerate_parameter_fills(
    skeleton: Sequence[str], accounting: CpuAccounting | None = None
) -> Iterator[F.Pipeline]:
    """Yield every concrete fill of ``skeleton`` while charging the enumeration."""

    receipt = accounting if accounting is not None else CpuAccounting()
    for pipeline in F.enumerate_parameter_fills(skeleton):
        receipt.parameter_fills_yielded += 1
        yield pipeline


def _case_bank(
    task: Mapping[str, Any], split_names: Sequence[str]
) -> tuple[F.Vector, F.Vector]:
    if not split_names:
        raise ValueError("at least one label-source split is required")
    if len(set(split_names)) != len(split_names):
        raise ValueError("label-source split names must not repeat")
    inputs, outputs = F.task_cases(task, split_names)
    if not inputs:
        raise ValueError("the selected label-source splits contain no cases")
    return (
        tuple(tuple(value) for value in inputs),
        tuple(tuple(value) for value in outputs),
    )


def _transition(
    vector: F.Vector, operation: F.ConcreteOp, accounting: CpuAccounting
) -> F.Vector | None:
    accounting.transition_requests += 1
    accounting.vector_transitions_computed += 1
    result: list[tuple[int, ...]] = []
    for row in vector:
        accounting.case_operation_applications += 1
        output = F.apply_operation(operation, row)
        if output is None:
            accounting.invalid_vector_transitions += 1
            return None
        result.append(tuple(output))
    return tuple(result)


class ExactSemanticOracle:
    """Exhaustive exact-length oracle over the selected task cases.

    Reachable states are kept separately at every exact depth.  Therefore cycles
    and semantic no-ops remain available as genuine length-consuming paths.  The
    backward pass records every concrete operation edge that can still reach the
    target at exactly ``depth``; walking those edges enumerates all successful
    parameter fills without treating the serialized target as privileged.
    """

    def __init__(
        self,
        task: Mapping[str, Any],
        *,
        depth: int | None = None,
        label_source_splits: Sequence[str] = ("visible", "label_probe"),
        accounting: CpuAccounting | None = None,
    ) -> None:
        resolved_depth = int(task.get("depth", 0) if depth is None else depth)
        if resolved_depth < 1:
            raise ValueError("oracle depth must be positive")
        self.task = task
        self.task_id = str(task.get("task_id", "unknown"))
        self.depth = resolved_depth
        self.label_source_splits = tuple(label_source_splits)
        self.accounting = accounting if accounting is not None else CpuAccounting()
        self.full_start, self.full_target = _case_bank(task, self.label_source_splits)
        # Exhausting the graph on one case is a lossless first-case filter: every
        # full-bank solution must pass it.  Enumerating all paths through the live
        # graph and validating each survivor on the remaining cases is therefore
        # exact, while avoiding a depth-5 graph whose state keys duplicate a dozen
        # complete case vectors.  The first case is frozen by serialized split order.
        self.pruning_case_index = 0
        self.start: F.Vector = (self.full_start[self.pruning_case_index],)
        self.target: F.Vector = (self.full_target[self.pruning_case_index],)
        self.layers: list[dict[F.Vector, None]] = [{self.start: None}]
        self.live_states: list[set[F.Vector]] = [set() for _ in range(self.depth + 1)]
        self.live_edges: list[
            dict[F.Vector, tuple[tuple[F.ConcreteOp, F.Vector], ...]]
        ] = [dict() for _ in range(self.depth)]
        self._built = False
        self._successful_cache: tuple[SuccessfulSkeleton, ...] | None = None

    def build(self) -> "ExactSemanticOracle":
        if self._built:
            return self

        self.accounting.behavior_states_inserted += 1
        for _level in range(1, self.depth + 1):
            next_layer: dict[F.Vector, None] = {}
            for vector in self.layers[-1]:
                for operation in F.CONCRETE_OPS:
                    new_vector = _transition(vector, operation, self.accounting)
                    if new_vector is not None:
                        next_layer.setdefault(new_vector, None)
            self.accounting.behavior_states_inserted += len(next_layer)
            self.layers.append(next_layer)

        if self.target not in self.layers[self.depth]:
            # A task's serialized target should make this impossible, but keeping
            # the empty result exact makes the oracle useful for adversarial tests.
            self._built = True
            return self

        self.live_states[self.depth] = {self.target}
        for level in range(self.depth - 1, -1, -1):
            next_live = self.live_states[level + 1]
            current_live: set[F.Vector] = set()
            current_edges: dict[F.Vector, tuple[tuple[F.ConcreteOp, F.Vector], ...]] = {}
            for vector in self.layers[level]:
                edges: list[tuple[F.ConcreteOp, F.Vector]] = []
                for operation in F.CONCRETE_OPS:
                    new_vector = _transition(vector, operation, self.accounting)
                    if new_vector is not None and new_vector in next_live:
                        edges.append((operation, new_vector))
                if edges:
                    current_live.add(vector)
                    current_edges[vector] = tuple(edges)
            self.live_states[level] = current_live
            self.live_edges[level] = current_edges
        self._built = True
        return self

    def iter_successful_pipelines(self) -> Iterator[F.Pipeline]:
        self.build()
        if self.start not in self.live_states[0]:
            return

        prefix: list[F.ConcreteOp] = []

        def walk(level: int, vector: F.Vector) -> Iterator[F.Pipeline]:
            if level == self.depth:
                self.accounting.pruning_case_candidate_pipelines += 1
                candidate = tuple(prefix)
                if self._matches_remaining_cases(candidate):
                    self.accounting.successful_concrete_pipelines += 1
                    yield candidate
                return
            for operation, new_vector in self.live_edges[level].get(vector, ()):
                prefix.append(operation)
                yield from walk(level + 1, new_vector)
                prefix.pop()

        yield from walk(0, self.start)

    def _matches_remaining_cases(self, pipeline: F.Pipeline) -> bool:
        """Validate one first-case survivor on every other label-source case."""

        self.accounting.validation_pipelines_checked += 1
        for start, target in zip(
            self.full_start[1:], self.full_target[1:], strict=True
        ):
            state = list(start)
            for operation in pipeline:
                self.accounting.case_operation_applications += 1
                self.accounting.validation_case_operation_applications += 1
                output = F.apply_operation(operation, state)
                if output is None:
                    return False
                state = output
            if tuple(state) != target:
                return False
        return True

    def successful_skeletons(self) -> tuple[SuccessfulSkeleton, ...]:
        if self._successful_cache is not None:
            return self._successful_cache
        counts: dict[tuple[str, ...], int] = defaultdict(int)
        representatives: dict[tuple[str, ...], F.Pipeline] = {}
        for pipeline in self.iter_successful_pipelines():
            skeleton = tuple(name for name, _parameter in pipeline)
            counts[skeleton] += 1
            representatives.setdefault(skeleton, pipeline)
        ordered = tuple(
            SuccessfulSkeleton(
                skeleton=skeleton,
                parameter_fill_count=counts[skeleton],
                representative_pipeline=representatives[skeleton],
            )
            for skeleton in sorted(counts)
        )
        self.accounting.successful_type_skeletons = len(ordered)
        self._successful_cache = ordered
        return ordered


def all_semantically_successful_skeletons(
    task: Mapping[str, Any],
    *,
    depth: int | None = None,
    label_source_splits: Sequence[str] = ("visible", "label_probe"),
    accounting: CpuAccounting | None = None,
) -> tuple[SuccessfulSkeleton, ...]:
    """Return every skeleton with at least one successful parameter fill."""

    return ExactSemanticOracle(
        task,
        depth=depth,
        label_source_splits=label_source_splits,
        accounting=accounting,
    ).successful_skeletons()


def live_prefix_rows(
    *,
    task_id: str,
    depth: int,
    successful_skeletons: Sequence[SuccessfulSkeleton],
    label_source_splits: Sequence[str] = ("visible", "label_probe"),
    include_full: bool = False,
    accounting: CpuAccounting | None = None,
) -> tuple[dict[str, Any], ...]:
    """Label every type prefix, including dead alternatives, with exact counts."""

    if depth < 1:
        raise ValueError("depth must be positive")
    max_length = depth if include_full else depth - 1
    skeleton_counts: dict[tuple[str, ...], int] = defaultdict(int)
    fill_counts: dict[tuple[str, ...], int] = defaultdict(int)
    for success in successful_skeletons:
        if len(success.skeleton) != depth:
            raise ValueError("successful skeleton length does not match oracle depth")
        for length in range(max_length + 1):
            prefix = success.skeleton[:length]
            skeleton_counts[prefix] += 1
            fill_counts[prefix] += success.parameter_fill_count

    rows: list[dict[str, Any]] = []
    for length in range(max_length + 1):
        for prefix in F.enumerate_skeletons(length):
            rows.append(
                {
                    "task_id": str(task_id),
                    "task_depth": depth,
                    "prefix": list(prefix),
                    "prefix_length": length,
                    "remaining_depth": depth - length,
                    "live": skeleton_counts[prefix] > 0,
                    "completion_skeleton_count": skeleton_counts[prefix],
                    "completion_parameter_fill_count": fill_counts[prefix],
                    "label_source_splits": list(label_source_splits),
                    "hidden_cases_used_for_label": "hidden" in label_source_splits,
                }
            )
    if accounting is not None:
        accounting.prefix_rows_emitted += len(rows)
    return tuple(rows)


def build_oracle_result(
    task: Mapping[str, Any],
    *,
    depth: int | None = None,
    label_source_splits: Sequence[str] = ("visible", "label_probe"),
    include_full_prefixes: bool = False,
    accounting: CpuAccounting | None = None,
) -> SemanticOracleResult:
    receipt = accounting if accounting is not None else CpuAccounting()
    oracle = ExactSemanticOracle(
        task,
        depth=depth,
        label_source_splits=label_source_splits,
        accounting=receipt,
    ).build()
    successes = oracle.successful_skeletons()
    prefixes = live_prefix_rows(
        task_id=oracle.task_id,
        depth=oracle.depth,
        successful_skeletons=successes,
        label_source_splits=oracle.label_source_splits,
        include_full=include_full_prefixes,
        accounting=receipt,
    )
    return SemanticOracleResult(
        task_id=oracle.task_id,
        depth=oracle.depth,
        label_source_splits=oracle.label_source_splits,
        successful_skeletons=successes,
        prefix_rows=prefixes,
        pruning_case_index=oracle.pruning_case_index,
        behavior_states_by_layer=tuple(len(layer) for layer in oracle.layers),
        live_behavior_states_by_layer=tuple(len(layer) for layer in oracle.live_states),
        accounting=receipt.to_dict(),
    )


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    text = path.read_text().strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("tasks"), list):
        return list(payload["tasks"])
    raise ValueError("task input must be a JSON list, {'tasks': [...]}, or JSONL")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="fresh task JSON/JSONL")
    parser.add_argument("--output", type=Path, required=True, help="oracle JSONL output")
    parser.add_argument(
        "--include-full-prefixes",
        action="store_true",
        help="also emit length-D terminal skeleton rows",
    )
    args = parser.parse_args(argv)

    tasks = _read_tasks(args.tasks)
    results = [
        build_oracle_result(task, include_full_prefixes=args.include_full_prefixes).to_dict()
        for task in tasks
    ]
    _write_jsonl(args.output, results)
    aggregate = CpuAccounting()
    for result in results:
        for name, value in result["accounting"].items():
            setattr(aggregate, name, getattr(aggregate, name) + int(value))
    print(
        json.dumps(
            {
                "tasks": len(results),
                "output": str(args.output),
                "label_source_splits": ["visible", "label_probe"],
                "hidden_cases_used_for_labels": False,
                "accounting": aggregate.to_dict(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
