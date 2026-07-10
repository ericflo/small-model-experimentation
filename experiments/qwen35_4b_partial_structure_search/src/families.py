"""Deterministic 16-type list DSL and contamination-free task generation.

This is an experiment-local implementation of the list pipeline substrate used by
the preceding structure-search experiments.  Operations are represented as
``(type_name, parameter)`` pairs.  A parameterless operation always has ``None``
as its parameter.

The minimum-depth audit below is exact *for the complete case bank supplied to
it*.  It performs an uncapped breadth-first exhaustion of the behavioral
quotient.  Merging two programs with identical outputs on every audit input is
lossless: every DSL suffix is deterministic, so the two programs have identical
continuations on that case bank.  There is deliberately no ``seen_cap`` escape
hatch.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import product
from typing import Any, TypeAlias


MAX_ABS_VALUE = 10**7
MAX_LIST_LENGTH = 64

ConcreteOp: TypeAlias = tuple[str, int | None]
Pipeline: TypeAlias = tuple[ConcreteOp, ...]
Vector: TypeAlias = tuple[tuple[int, ...], ...]


@dataclasses.dataclass(frozen=True)
class PrimitiveSpec:
    """One operation type and its complete finite parameter domain."""

    name: str
    params: tuple[int | None, ...]
    description: str


# Insertion order is part of the search protocol.  These are the same 16
# operation TYPES and parameter domains used by the current list substrate.
PRIMITIVES: dict[str, PrimitiveSpec] = {
    "reverse": PrimitiveSpec("reverse", (None,), "reverse the list"),
    "sort_asc": PrimitiveSpec("sort_asc", (None,), "sort ascending"),
    "sort_desc": PrimitiveSpec("sort_desc", (None,), "sort descending"),
    "unique_stable": PrimitiveSpec(
        "unique_stable", (None,), "remove later duplicates while preserving order"
    ),
    "dedup_adjacent": PrimitiveSpec(
        "dedup_adjacent", (None,), "collapse adjacent equal values"
    ),
    "abs_all": PrimitiveSpec("abs_all", (None,), "take absolute values"),
    "square": PrimitiveSpec("square", (None,), "square every value"),
    "negate": PrimitiveSpec("negate", (None,), "negate every value"),
    "running_sum": PrimitiveSpec(
        "running_sum", (None,), "replace values by inclusive running sums"
    ),
    "adjacent_diff": PrimitiveSpec(
        "adjacent_diff", (None,), "replace by successive right-minus-left differences"
    ),
    "add_k": PrimitiveSpec("add_k", (-3, -2, -1, 1, 2, 3), "add k to every value"),
    "mul_k": PrimitiveSpec("mul_k", (-2, 2, 3), "multiply every value by k"),
    "mod_k": PrimitiveSpec("mod_k", (2, 3, 4), "reduce every value modulo k"),
    "take_k": PrimitiveSpec("take_k", (1, 2, 3, 4), "keep the first k values"),
    "drop_k": PrimitiveSpec("drop_k", (1, 2, 3), "drop the first k values"),
    "rotate_k": PrimitiveSpec(
        "rotate_k", (1, 2, 3), "rotate left by k modulo the current length"
    ),
}

TYPES: tuple[str, ...] = tuple(PRIMITIVES)
PARAMS: dict[str, tuple[int | None, ...]] = {
    name: spec.params for name, spec in PRIMITIVES.items()
}
CONCRETE_OPS: tuple[ConcreteOp, ...] = tuple(
    (name, parameter) for name in TYPES for parameter in PARAMS[name]
)

if len(TYPES) != 16:  # Fail loudly if a later edit silently changes 16**depth.
    raise RuntimeError(f"the protocol requires exactly 16 operation types, got {len(TYPES)}")


def _normalize_list(value: Sequence[int], *, where: str = "DSL value") -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{where} must be a sequence of integers")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise TypeError(f"{where} must contain only non-boolean integers")
        result.append(int(item))
    return result


def normalize_op(operation: Any) -> ConcreteOp:
    """Normalize a tuple/list or ``{'type', 'param'}`` JSON representation."""

    if isinstance(operation, Mapping):
        name = operation.get("type")
        parameter = operation.get("param")
    elif (
        isinstance(operation, Sequence)
        and not isinstance(operation, (str, bytes))
        and len(operation) == 2
    ):
        name, parameter = operation
    else:
        raise TypeError("an operation must be a (type, parameter) pair or mapping")
    if not isinstance(name, str) or name not in PRIMITIVES:
        raise ValueError(f"unknown operation type: {name!r}")
    if parameter not in PARAMS[name]:
        raise ValueError(
            f"invalid parameter {parameter!r} for {name}; allowed={PARAMS[name]!r}"
        )
    return name, parameter


def normalize_pipeline(pipeline: Iterable[Any]) -> Pipeline:
    if isinstance(pipeline, (str, bytes)):
        raise TypeError("pipeline must be an iterable of structured operations")
    return tuple(normalize_op(operation) for operation in pipeline)


def format_op(operation: Any) -> str:
    name, parameter = normalize_op(operation)
    return name if parameter is None else f"{name}({parameter})"


def _bounded(result: list[int]) -> list[int] | None:
    if len(result) > MAX_LIST_LENGTH:
        return None
    if any(abs(value) >= MAX_ABS_VALUE for value in result):
        return None
    return result


def apply_operation(
    operation: ConcreteOp | Sequence[Any] | Mapping[str, Any], value: Sequence[int]
) -> list[int] | None:
    """Apply one validated DSL operation, returning ``None`` on numeric explosion."""

    name, parameter = normalize_op(operation)
    xs = _normalize_list(value)

    if name == "reverse":
        result = xs[::-1]
    elif name == "sort_asc":
        result = sorted(xs)
    elif name == "sort_desc":
        result = sorted(xs, reverse=True)
    elif name == "unique_stable":
        result = list(dict.fromkeys(xs))
    elif name == "dedup_adjacent":
        result = [x for index, x in enumerate(xs) if index == 0 or x != xs[index - 1]]
    elif name == "abs_all":
        result = [abs(x) for x in xs]
    elif name == "square":
        result = [x * x for x in xs]
    elif name == "negate":
        result = [-x for x in xs]
    elif name == "running_sum":
        result = []
        running = 0
        for x in xs:
            running += x
            result.append(running)
    elif name == "adjacent_diff":
        result = [xs[index + 1] - xs[index] for index in range(len(xs) - 1)]
    elif name == "add_k":
        assert parameter is not None
        result = [x + parameter for x in xs]
    elif name == "mul_k":
        assert parameter is not None
        result = [x * parameter for x in xs]
    elif name == "mod_k":
        assert parameter is not None
        result = [x % parameter for x in xs]
    elif name == "take_k":
        assert parameter is not None
        result = xs[:parameter]
    elif name == "drop_k":
        assert parameter is not None
        result = xs[parameter:]
    elif name == "rotate_k":
        assert parameter is not None
        offset = parameter % len(xs) if xs else 0
        result = xs[offset:] + xs[:offset]
    else:  # pragma: no cover - normalize_op makes this unreachable.
        raise AssertionError(name)
    return _bounded(result)


def execute_pipeline(pipeline: Iterable[Any], value: Sequence[int]) -> list[int] | None:
    state = _normalize_list(value, where="pipeline input")
    for operation in normalize_pipeline(pipeline):
        next_state = apply_operation(operation, state)
        if next_state is None:
            return None
        state = next_state
    return state


def enumerate_parameter_fills(skeleton: Sequence[str]) -> Iterator[Pipeline]:
    """Enumerate every concrete parameter fill for one full or partial skeleton."""

    normalized = tuple(skeleton)
    if any(not isinstance(name, str) or name not in PRIMITIVES for name in normalized):
        raise ValueError(f"skeleton contains an unknown operation type: {skeleton!r}")
    for parameters in product(*(PARAMS[name] for name in normalized)):
        yield tuple(zip(normalized, parameters, strict=True))


def enumerate_skeletons(depth: int) -> Iterator[tuple[str, ...]]:
    """Enumerate all ``16**depth`` type skeletons in frozen protocol order."""

    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        raise ValueError("depth must be a non-negative integer")
    yield from product(TYPES, repeat=depth)


def pipeline_behavior(
    pipeline: Iterable[Any], inputs: Sequence[Sequence[int]]
) -> Vector | None:
    normalized = normalize_pipeline(pipeline)
    rows: list[tuple[int, ...]] = []
    for value in inputs:
        output = execute_pipeline(normalized, value)
        if output is None:
            return None
        rows.append(tuple(output))
    return tuple(rows)


def pipeline_solves(
    pipeline: Iterable[Any], inputs: Sequence[Sequence[int]], outputs: Sequence[Sequence[int]]
) -> bool:
    if len(inputs) != len(outputs):
        raise ValueError("inputs and outputs must have equal length")
    target = tuple(tuple(_normalize_list(row, where="target output")) for row in outputs)
    return pipeline_behavior(pipeline, inputs) == target


@dataclasses.dataclass(frozen=True)
class MinimumDepthAudit:
    """Machine-readable receipt for one exact behavioral BFS decision."""

    max_depth: int
    found_depth: int | None
    representative_pipeline: Pipeline | None
    frontier_unique_behaviors: tuple[int, ...]
    levels_fully_exhausted: tuple[int, ...]
    transitions_considered: int
    case_operation_applications: int
    invalid_transitions: int
    duplicate_behaviors: int
    unique_behaviors_seen: int

    @property
    def within_limit(self) -> bool:
        return self.found_depth is not None

    @property
    def exhaustive_decision(self) -> bool:
        # A witness settles YES; completing every requested level settles NO.
        return self.within_limit or self.levels_fully_exhausted == tuple(
            range(1, self.max_depth + 1)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "uncapped_behavioral_bfs",
            "quotient_rule": (
                "identical outputs on every audit input; deterministic suffix congruence"
            ),
            "seen_cap": None,
            "max_depth": self.max_depth,
            "found_depth": self.found_depth,
            "representative_pipeline": (
                [[name, parameter] for name, parameter in self.representative_pipeline]
                if self.representative_pipeline is not None
                else None
            ),
            "frontier_unique_behaviors": list(self.frontier_unique_behaviors),
            "levels_fully_exhausted": list(self.levels_fully_exhausted),
            "transitions_considered": self.transitions_considered,
            "case_operation_applications": self.case_operation_applications,
            "invalid_transitions": self.invalid_transitions,
            "duplicate_behaviors": self.duplicate_behaviors,
            "unique_behaviors_seen": self.unique_behaviors_seen,
            "within_limit": self.within_limit,
            "exhaustive_decision": self.exhaustive_decision,
        }


def _normalize_case_bank(
    inputs: Sequence[Sequence[int]], outputs: Sequence[Sequence[int]]
) -> tuple[Vector, Vector]:
    if not inputs:
        raise ValueError("minimum-depth auditing requires at least one input")
    if len(inputs) != len(outputs):
        raise ValueError("inputs and outputs must have equal length")
    start = tuple(tuple(_normalize_list(row, where="audit input")) for row in inputs)
    target = tuple(tuple(_normalize_list(row, where="audit output")) for row in outputs)
    return start, target


def _apply_vector(vector: Vector, operation: ConcreteOp) -> tuple[Vector | None, int]:
    result: list[tuple[int, ...]] = []
    applications = 0
    for row in vector:
        applications += 1
        output = apply_operation(operation, row)
        if output is None:
            return None, applications
        result.append(tuple(output))
    return tuple(result), applications


def exhaustive_min_depth_leq(
    inputs: Sequence[Sequence[int]],
    outputs: Sequence[Sequence[int]],
    max_depth: int,
) -> MinimumDepthAudit:
    """Exactly decide whether any concrete pipeline of depth ``<= max_depth`` solves.

    The search has no state-count or wall-clock cutoff.  The returned receipt records
    quotient-frontier sizes and every concrete transition that was evaluated.
    """

    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer")
    start, target = _normalize_case_bank(inputs, outputs)
    if start == target:
        return MinimumDepthAudit(
            max_depth=max_depth,
            found_depth=0,
            representative_pipeline=(),
            frontier_unique_behaviors=(1,),
            levels_fully_exhausted=(),
            transitions_considered=0,
            case_operation_applications=0,
            invalid_transitions=0,
            duplicate_behaviors=0,
            unique_behaviors_seen=1,
        )

    representative: dict[Vector, Pipeline] = {start: ()}
    frontier: tuple[Vector, ...] = (start,)
    frontier_sizes = [1]
    exhausted: list[int] = []
    transitions = applications = invalid = duplicates = 0

    for depth in range(1, max_depth + 1):
        next_frontier: list[Vector] = []
        for vector in frontier:
            prefix = representative[vector]
            for operation in CONCRETE_OPS:
                transitions += 1
                new_vector, used = _apply_vector(vector, operation)
                applications += used
                if new_vector is None:
                    invalid += 1
                    continue
                candidate = prefix + (operation,)
                if new_vector == target:
                    # The YES decision is exact even though the remainder of this
                    # level need not be traversed after a shortest witness appears.
                    frontier_sizes.append(len(next_frontier) + 1)
                    return MinimumDepthAudit(
                        max_depth=max_depth,
                        found_depth=depth,
                        representative_pipeline=candidate,
                        frontier_unique_behaviors=tuple(frontier_sizes),
                        levels_fully_exhausted=tuple(exhausted),
                        transitions_considered=transitions,
                        case_operation_applications=applications,
                        invalid_transitions=invalid,
                        duplicate_behaviors=duplicates,
                        unique_behaviors_seen=len(representative) + 1,
                    )
                if new_vector in representative:
                    duplicates += 1
                    continue
                representative[new_vector] = candidate
                next_frontier.append(new_vector)
        exhausted.append(depth)
        frontier = tuple(next_frontier)
        frontier_sizes.append(len(frontier))
        if not frontier:
            # Later frontiers are also empty; count them as exhaustively resolved.
            exhausted.extend(range(depth + 1, max_depth + 1))
            frontier_sizes.extend(0 for _ in range(depth + 1, max_depth + 1))
            break

    return MinimumDepthAudit(
        max_depth=max_depth,
        found_depth=None,
        representative_pipeline=None,
        frontier_unique_behaviors=tuple(frontier_sizes),
        levels_fully_exhausted=tuple(exhausted),
        transitions_considered=transitions,
        case_operation_applications=applications,
        invalid_transitions=invalid,
        duplicate_behaviors=duplicates,
        unique_behaviors_seen=len(representative),
    )


def min_depth_leq(
    inputs: Sequence[Sequence[int]], outputs: Sequence[Sequence[int]], max_depth: int
) -> bool:
    """Boolean compatibility wrapper around :func:`exhaustive_min_depth_leq`."""

    return exhaustive_min_depth_leq(inputs, outputs, max_depth).within_limit


def _draw_input(rng: random.Random) -> list[int]:
    return [rng.randint(-9, 9) for _ in range(rng.randint(5, 8))]


def _draw_pipeline(rng: random.Random, depth: int) -> Pipeline:
    return tuple(
        (name := rng.choice(TYPES), rng.choice(PARAMS[name])) for _ in range(depth)
    )


def task_cases(
    task: Mapping[str, Any], split_names: Sequence[str] = ("visible", "label_probe", "hidden")
) -> tuple[list[list[int]], list[list[int]]]:
    inputs: list[list[int]] = []
    outputs: list[list[int]] = []
    for split_name in split_names:
        if split_name not in {"visible", "label_probe", "hidden"}:
            raise ValueError(f"unknown task split: {split_name}")
        for case in task.get(split_name, []):
            inputs.append(_normalize_list(case["input"], where=f"{split_name} input"))
            outputs.append(_normalize_list(case["output"], where=f"{split_name} output"))
    return inputs, outputs


def build_task_from_pipeline(
    *,
    task_id: str,
    seed: int,
    pipeline: Iterable[Any],
    visible_inputs: Sequence[Sequence[int]],
    label_probe_inputs: Sequence[Sequence[int]],
    hidden_inputs: Sequence[Sequence[int]],
    require_exact_depth: bool = True,
) -> dict[str, Any]:
    """Build and audit one task from explicit, mutually disjoint input splits."""

    concrete = normalize_pipeline(pipeline)
    if not concrete:
        raise ValueError("task pipeline must have positive depth")
    split_inputs = {
        "visible": visible_inputs,
        "label_probe": label_probe_inputs,
        "hidden": hidden_inputs,
    }
    all_seen: set[tuple[int, ...]] = set()
    split_rows: dict[str, list[dict[str, list[int]]]] = {}
    for split_name, rows in split_inputs.items():
        built: list[dict[str, list[int]]] = []
        for raw in rows:
            value = _normalize_list(raw, where=f"{split_name} input")
            key = tuple(value)
            if key in all_seen:
                raise ValueError("task inputs must be distinct across all three splits")
            all_seen.add(key)
            output = execute_pipeline(concrete, value)
            if output is None:
                raise ValueError("target pipeline explodes on a supplied task input")
            built.append({"input": value, "output": output})
        if not built:
            raise ValueError(f"task split {split_name!r} must be non-empty")
        split_rows[split_name] = built

    inputs = [case["input"] for name in split_rows for case in split_rows[name]]
    outputs = [case["output"] for name in split_rows for case in split_rows[name]]
    audit = exhaustive_min_depth_leq(inputs, outputs, len(concrete) - 1)
    if require_exact_depth and audit.within_limit:
        witness = [format_op(operation) for operation in audit.representative_pipeline or ()]
        raise ValueError(
            f"pipeline is not behaviorally exact depth {len(concrete)}; "
            f"shorter witness={witness!r}"
        )
    if require_exact_depth and not audit.exhaustive_decision:
        raise AssertionError("minimum-depth rejection did not exhaust the requested search")

    return {
        "schema_version": 1,
        "task_id": str(task_id),
        "seed": int(seed),
        "depth": len(concrete),
        "target_pipeline": [[name, parameter] for name, parameter in concrete],
        "target_skeleton": [name for name, _ in concrete],
        "target_ops": [format_op(operation) for operation in concrete],
        "visible": split_rows["visible"],
        "label_probe": split_rows["label_probe"],
        "hidden": split_rows["hidden"],
        "min_depth_audit": audit.to_dict(),
    }


def generate_task(
    *,
    task_id: str,
    depth: int,
    seed: int,
    n_visible: int = 6,
    n_label_probe: int = 6,
    n_hidden: int = 6,
    require_exact_depth: bool = True,
    max_attempts: int = 500,
) -> dict[str, Any]:
    """Generate one deterministic fresh procedural task and its exact-depth receipt."""

    if depth < 1:
        raise ValueError("depth must be positive")
    if min(n_visible, n_label_probe, n_hidden) < 1:
        raise ValueError("all three task splits must be non-empty")
    rng = random.Random(seed)
    total = n_visible + n_label_probe + n_hidden
    for _attempt in range(max_attempts):
        pipeline = _draw_pipeline(rng, depth)
        raw_inputs: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for _ in range(total * 20):
            candidate = _draw_input(rng)
            key = tuple(candidate)
            if key in seen:
                continue
            output = execute_pipeline(pipeline, candidate)
            if output is None:
                break
            seen.add(key)
            raw_inputs.append(candidate)
            if len(raw_inputs) == total:
                break
        if len(raw_inputs) != total:
            continue
        outputs = [execute_pipeline(pipeline, value) for value in raw_inputs]
        if any(output is None for output in outputs):
            continue
        if all(output == value for output, value in zip(outputs, raw_inputs, strict=True)):
            continue
        if len({tuple(output or ()) for output in outputs}) < max(3, total // 3):
            continue
        try:
            return build_task_from_pipeline(
                task_id=task_id,
                seed=seed,
                pipeline=pipeline,
                visible_inputs=raw_inputs[:n_visible],
                label_probe_inputs=raw_inputs[n_visible : n_visible + n_label_probe],
                hidden_inputs=raw_inputs[n_visible + n_label_probe :],
                require_exact_depth=require_exact_depth,
            )
        except ValueError as error:
            if "not behaviorally exact depth" not in str(error):
                raise
    raise RuntimeError(
        f"could not generate an exact depth-{depth} task after {max_attempts} attempts "
        f"(seed={seed})"
    )


def generate_tasks(
    *,
    count: int,
    depth: int,
    seed: int,
    **task_kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate independently seeded tasks; no state or examples are shared."""

    if count < 0:
        raise ValueError("count must be non-negative")
    seed_rng = random.Random(seed)
    tasks = []
    for index in range(count):
        task_seed = seed_rng.randrange(2**63)
        tasks.append(
            generate_task(
                task_id=f"list-d{depth}-{index:04d}",
                depth=depth,
                seed=task_seed,
                **task_kwargs,
            )
        )
    return tasks
