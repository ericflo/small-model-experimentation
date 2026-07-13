"""Fresh exact-depth-two list programs for early hypothesis forking."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from itertools import product
from typing import Any


def _reverse(xs: list[int], _: int | None) -> list[int]:
    return xs[::-1]


def _sort_asc(xs: list[int], _: int | None) -> list[int]:
    return sorted(xs)


def _sort_desc(xs: list[int], _: int | None) -> list[int]:
    return sorted(xs, reverse=True)


def _abs_all(xs: list[int], _: int | None) -> list[int]:
    return [abs(value) for value in xs]


def _square(xs: list[int], _: int | None) -> list[int]:
    return [value * value for value in xs]


def _negate(xs: list[int], _: int | None) -> list[int]:
    return [-value for value in xs]


def _running_sum(xs: list[int], _: int | None) -> list[int]:
    return [sum(xs[: index + 1]) for index in range(len(xs))]


def _adjacent_diff(xs: list[int], _: int | None) -> list[int]:
    return [xs[index + 1] - xs[index] for index in range(len(xs) - 1)]


def _add_k(xs: list[int], parameter: int | None) -> list[int]:
    if parameter is None:
        raise ValueError("add_k requires a parameter")
    return [value + parameter for value in xs]


def _mul_k(xs: list[int], parameter: int | None) -> list[int]:
    if parameter is None:
        raise ValueError("mul_k requires a parameter")
    return [value * parameter for value in xs]


def _take_k(xs: list[int], parameter: int | None) -> list[int]:
    if parameter is None or parameter < 1:
        raise ValueError("take_k requires a positive parameter")
    return xs[:parameter]


def _rotate_k(xs: list[int], parameter: int | None) -> list[int]:
    if parameter is None or not xs:
        raise ValueError("rotate_k requires a parameter and nonempty input")
    shift = parameter % len(xs)
    return xs[shift:] + xs[:shift]


Operation = tuple[Callable[[list[int], int | None], list[int]], tuple[int | None, ...]]
BoundOperation = tuple[str, int | None]
Program = tuple[BoundOperation, BoundOperation]


OPERATIONS: dict[str, Operation] = {
    "reverse": (_reverse, (None,)),
    "sort_asc": (_sort_asc, (None,)),
    "sort_desc": (_sort_desc, (None,)),
    "abs_all": (_abs_all, (None,)),
    "square": (_square, (None,)),
    "negate": (_negate, (None,)),
    "running_sum": (_running_sum, (None,)),
    "adjacent_diff": (_adjacent_diff, (None,)),
    "add_k": (_add_k, (-3, -2, -1, 1, 2, 3)),
    "mul_k": (_mul_k, (-2, 2, 3)),
    "take_k": (_take_k, (1, 2, 3, 4)),
    "rotate_k": (_rotate_k, (1, 2, 3)),
}
EXPECTED_PARAMETERS: dict[str, tuple[int | None, ...]] = {
    "reverse": (None,),
    "sort_asc": (None,),
    "sort_desc": (None,),
    "abs_all": (None,),
    "square": (None,),
    "negate": (None,),
    "running_sum": (None,),
    "adjacent_diff": (None,),
    "add_k": (-3, -2, -1, 1, 2, 3),
    "mul_k": (-2, 2, 3),
    "take_k": (1, 2, 3, 4),
    "rotate_k": (1, 2, 3),
}
if {name: parameters for name, (_function, parameters) in OPERATIONS.items()} != (
    EXPECTED_PARAMETERS
):
    raise RuntimeError("the frozen bound-operation inventory changed")

# Negation frequently participates in behaviorally equivalent two-step forms;
# retain it in the public candidate menu but do not make it the unique target.
CONCRETE_OPERATIONS: tuple[BoundOperation, ...] = tuple(
    (name, parameter)
    for name, (_function, parameters) in OPERATIONS.items()
    for parameter in parameters
)
if len(CONCRETE_OPERATIONS) != 24:
    raise RuntimeError("the frozen DSL must contain exactly 24 bound operations")

IDENTIFIABLE_FIRST_OPERATIONS: tuple[BoundOperation, ...] = tuple(
    operation for operation in CONCRETE_OPERATIONS if operation[0] != "negate"
)
if len(IDENTIFIABLE_FIRST_OPERATIONS) != 23:
    raise RuntimeError("negate must be the sole distractor-only first operation")

DEPTH_TWO_PROGRAMS: tuple[Program, ...] = tuple(
    (first, second) for first, second in product(CONCRETE_OPERATIONS, repeat=2)
)
if len(DEPTH_TWO_PROGRAMS) != 24**2:
    raise RuntimeError("the exhaustive depth-two grammar is incomplete")


def concrete_operations() -> tuple[BoundOperation, ...]:
    """Return the frozen candidate bank in canonical iteration order."""

    return CONCRETE_OPERATIONS


def depth_two_programs() -> tuple[Program, ...]:
    """Return all 24^2 programs; no model-facing search may use a subset."""

    return DEPTH_TWO_PROGRAMS


def scheduled_first_operation(task_index: int) -> BoundOperation:
    """Canonical 23-operation cycle, independent of every random seed."""

    if (
        not isinstance(task_index, int)
        or isinstance(task_index, bool)
        or task_index < 0
    ):
        raise ValueError("task index must be a nonnegative integer")
    return IDENTIFIABLE_FIRST_OPERATIONS[
        task_index % len(IDENTIFIABLE_FIRST_OPERATIONS)
    ]


def _validate_bound_operation(operation: BoundOperation) -> None:
    if not isinstance(operation, tuple) or len(operation) != 2:
        raise ValueError("bound operation must be a (name, parameter) tuple")
    name, parameter = operation
    if not isinstance(name, str) or name not in OPERATIONS:
        raise ValueError(f"unknown operation {name!r}")
    if isinstance(parameter, bool) or parameter not in OPERATIONS[name][1]:
        raise ValueError(f"invalid parameter for {name}: {parameter!r}")


def operation_record(operation: BoundOperation) -> dict[str, int | str | None]:
    """Encode a bound operation without dropping its parameter."""

    _validate_bound_operation(operation)
    return {"name": operation[0], "parameter": operation[1]}


def operation_from_record(value: dict[str, Any]) -> BoundOperation:
    """Decode the strict JSON representation used in gold artifacts."""

    if not isinstance(value, dict) or set(value) != {"name", "parameter"}:
        raise ValueError("operation record must have exactly name and parameter")
    operation = (value["name"], value["parameter"])
    _validate_bound_operation(operation)
    return operation


def canonical_operation(operation: BoundOperation) -> str:
    """Return the unique prompt/parser spelling of one bound operation."""

    _validate_bound_operation(operation)
    name, parameter = operation
    return name if parameter is None else f"{name}({parameter})"


def canonical_program(pipeline: Sequence[BoundOperation]) -> str:
    """Return the canonical parser payload (without the ``PROGRAM:`` prefix)."""

    values = tuple(pipeline)
    if len(values) != 2:
        raise ValueError("a canonical program must have exactly two operations")
    return " | ".join(canonical_operation(operation) for operation in values)


def apply_pipeline(
    values: list[int], pipeline: Sequence[BoundOperation]
) -> list[int]:
    if not isinstance(values, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in values
    ):
        raise ValueError("pipeline input must be a list of integers")
    state = list(values)
    for operation in pipeline:
        _validate_bound_operation(operation)
        name, parameter = operation
        state = OPERATIONS[name][0](state, parameter)
        if len(state) > 64 or any(abs(value) > 10**7 for value in state):
            raise ValueError("pipeline state exceeded safety bound")
    return state


def diagnostic_apply(operation: BoundOperation, values: list[int]) -> list[int]:
    """Apply one supplied bound operation for concrete-operation mechanics."""

    return apply_pipeline(values, [operation])


def _random_input(rng: random.Random) -> list[int]:
    return [rng.randint(-9, 9) for _ in range(rng.randint(4, 8))]


def _outputs(
    pipeline: Sequence[BoundOperation], inputs: Sequence[list[int]]
) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(apply_pipeline(values, pipeline)) for values in inputs)


def matching_programs(
    inputs: Sequence[list[int]], outputs: Sequence[Sequence[int]]
) -> tuple[Program, ...]:
    """Exhaustively enumerate every one of the 24^2 visible-consistent programs."""

    expected = tuple(tuple(value) for value in outputs)
    if len(inputs) != len(expected):
        raise ValueError("matching inputs and outputs have different lengths")
    matches: list[Program] = []
    for pipeline in DEPTH_TWO_PROGRAMS:
        try:
            candidate = _outputs(pipeline, inputs)
        except ValueError:
            continue
        if candidate == expected:
            matches.append(pipeline)
    return tuple(matches)


def matching_first_operations(
    inputs: Sequence[list[int]], outputs: Sequence[Sequence[int]]
) -> set[BoundOperation]:
    return {program[0] for program in matching_programs(inputs, outputs)}


def matching_first_types(
    inputs: Sequence[list[int]], outputs: Sequence[Sequence[int]]
) -> set[str]:
    """Compatibility diagnostic; task acceptance uses concrete operations."""

    return {operation[0] for operation in matching_first_operations(inputs, outputs)}


def matching_depth_one(
    inputs: Sequence[list[int]], outputs: Sequence[Sequence[int]]
) -> tuple[BoundOperation, ...]:
    expected = tuple(tuple(value) for value in outputs)
    if len(inputs) != len(expected):
        raise ValueError("matching inputs and outputs have different lengths")
    matches: list[BoundOperation] = []
    for operation in CONCRETE_OPERATIONS:
        try:
            if _outputs([operation], inputs) == expected:
                matches.append(operation)
        except ValueError:
            continue
    return tuple(matches)


def make_task(
    rng: random.Random,
    *,
    task_id: str,
    first_operation: BoundOperation,
    visible_count: int,
    hidden_count: int,
    probe_count: int,
) -> dict[str, Any]:
    _validate_bound_operation(first_operation)
    if first_operation not in IDENTIFIABLE_FIRST_OPERATIONS:
        raise ValueError("negate is distractor-only and cannot be a gold first operation")
    if min(visible_count, hidden_count, probe_count) < 1:
        raise ValueError("visible, hidden, and probe partitions must all be nonempty")
    for _ in range(5000):
        pipeline: Program = (first_operation, rng.choice(CONCRETE_OPERATIONS))
        inputs: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        while len(inputs) < visible_count + hidden_count + probe_count:
            value = _random_input(rng)
            if tuple(value) not in seen:
                seen.add(tuple(value))
                inputs.append(value)
        labeled = inputs[: visible_count + hidden_count]
        try:
            outputs = _outputs(pipeline, labeled)
        except ValueError:
            continue
        visible_inputs = labeled[:visible_count]
        visible_outputs = outputs[:visible_count]
        if outputs == tuple(tuple(value) for value in labeled):
            continue
        visible_programs = matching_programs(visible_inputs, visible_outputs)
        if not visible_programs:
            continue
        visible_first_operations = {program[0] for program in visible_programs}
        if visible_first_operations != {first_operation}:
            continue
        depth_one_matches = matching_depth_one(visible_inputs, visible_outputs)
        if depth_one_matches:
            continue
        hidden_inputs = labeled[visible_count:]
        probe_inputs = inputs[visible_count + hidden_count :]
        try:
            hidden_behaviors = {
                _outputs(candidate, hidden_inputs) for candidate in visible_programs
            }
            probe_behaviors = {
                _outputs(candidate, probe_inputs) for candidate in visible_programs
            }
        except ValueError:
            continue
        if len(hidden_behaviors) != 1 or len(probe_behaviors) != 1:
            # A visible-only tie would make held-out accuracy depend on an
            # arbitrary syntactic tie-break, or let the selector's probe panel
            # manufacture a preference among otherwise valid programs.
            continue
        examples = [
            {"input": value, "output": list(output)}
            for value, output in zip(labeled, outputs, strict=True)
        ]
        task = {
            "task_id": task_id,
            "depth": 2,
            "first_op": operation_record(first_operation),
            "target_pipeline": [
                operation_record(operation) for operation in pipeline
            ],
            "visible": examples[:visible_count],
            "hidden": examples[visible_count:],
            "unlabeled_probe_inputs": inputs[visible_count + hidden_count :],
            "visible_consistent_programs": [
                canonical_program(candidate) for candidate in visible_programs
            ],
            "visible_consistent_program_count": len(visible_programs),
            "visible_consistent_hidden_behavior_count": len(hidden_behaviors),
            "visible_consistent_probe_behavior_count": len(probe_behaviors),
            "visible_matching_first_operations": [
                canonical_operation(first_operation)
            ],
            "visible_matching_depth_one": [],
        }
        validate_task(task)
        return task
    raise RuntimeError(f"failed to construct exact-depth task {task_id}")


def build_splits(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    data = config["data"]
    if tuple(data["operations"]) != tuple(OPERATIONS):
        raise ValueError("configured operation order changed")
    expected_parameter_values = {
        name: list(parameters)
        for name, parameters in EXPECTED_PARAMETERS.items()
        if parameters != (None,)
    }
    if data.get("parameter_values") != expected_parameter_values:
        raise ValueError("configured bound-operation parameters changed")
    if data.get("depth") != 2 or isinstance(data.get("depth"), bool):
        raise ValueError("configured task depth must remain exactly two")
    specs = (
        ("qualification", int(data["qualification_tasks"]), 23),
        ("confirmation", int(data["confirmation_tasks"]), 37),
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for split, count, salt in specs:
        rng = random.Random(int(config["seeds"]["data"]) + salt)
        rows = [
            make_task(
                rng,
                task_id=f"{split}-{index:05d}",
                first_operation=scheduled_first_operation(index),
                visible_count=int(data["visible_examples"]),
                hidden_count=int(data["hidden_examples"]),
                probe_count=int(data["unlabeled_probe_inputs"]),
            )
            for index in range(count)
        ]
        result[split] = rows
    validate_splits(result)
    return result


def behavior_fingerprint(task: dict[str, Any]) -> str:
    # Deliberately exclude gold metadata. This makes collision checks invariant
    # to the old type-only versus new bound-operation representation and catches
    # an exact repeated input/output behavior regardless of its program label.
    payload = {
        "depth": int(task["depth"]),
        "visible": task["visible"],
        "hidden": task["hidden"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def task_fingerprint(task: dict[str, Any]) -> str:
    payload = {
        "behavior": behavior_fingerprint(task),
        "target_pipeline": task["target_pipeline"],
        "unlabeled_probe_inputs": task["unlabeled_probe_inputs"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def public_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "depth": task["depth"],
        "visible": task["visible"],
        "unlabeled_probe_inputs": task["unlabeled_probe_inputs"],
    }


def gold_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "first_op": task["first_op"],
        "target_pipeline": task["target_pipeline"],
        "hidden": task["hidden"],
        "visible_consistent_programs": task["visible_consistent_programs"],
        "visible_consistent_program_count": task[
            "visible_consistent_program_count"
        ],
        "visible_consistent_hidden_behavior_count": task[
            "visible_consistent_hidden_behavior_count"
        ],
        "visible_consistent_probe_behavior_count": task[
            "visible_consistent_probe_behavior_count"
        ],
        "visible_matching_first_operations": task[
            "visible_matching_first_operations"
        ],
        "visible_matching_depth_one": task["visible_matching_depth_one"],
        "behavior_fingerprint": behavior_fingerprint(task),
        "task_fingerprint": task_fingerprint(task),
    }


def validate_task(task: dict[str, Any]) -> None:
    expected_fields = {
        "task_id",
        "depth",
        "first_op",
        "target_pipeline",
        "visible",
        "hidden",
        "unlabeled_probe_inputs",
        "visible_consistent_programs",
        "visible_consistent_program_count",
        "visible_consistent_hidden_behavior_count",
        "visible_consistent_probe_behavior_count",
        "visible_matching_first_operations",
        "visible_matching_depth_one",
    }
    if not isinstance(task, dict) or set(task) != expected_fields:
        raise ValueError("task does not have the strict frozen schema")
    if not isinstance(task["task_id"], str) or not task["task_id"]:
        raise ValueError("task_id must be a nonempty string")
    if (
        task["depth"] != 2
        or isinstance(task["depth"], bool)
        or not isinstance(task["target_pipeline"], list)
        or len(task["target_pipeline"]) != 2
    ):
        raise ValueError("task is not stored at depth two")
    pipeline = tuple(operation_from_record(value) for value in task["target_pipeline"])
    first_operation = operation_from_record(task["first_op"])
    if first_operation not in IDENTIFIABLE_FIRST_OPERATIONS:
        raise ValueError("gold first operation is not in the 23-operation support")
    if pipeline[0] != first_operation:
        raise ValueError("stored first operation disagrees with pipeline")

    partitions = [task["visible"], task["hidden"]]
    if any(not isinstance(rows, list) or not rows for rows in partitions):
        raise ValueError("visible and hidden partitions must be nonempty lists")
    for rows in partitions:
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"input", "output"}:
                raise ValueError("example rows require exact input/output schema")
            for key in ("input", "output"):
                if not isinstance(row[key], list) or any(
                    not isinstance(item, int) or isinstance(item, bool)
                    for item in row[key]
                ):
                    raise ValueError("examples must contain integer lists")
            if not row["input"]:
                raise ValueError("example inputs must be nonempty")
    probes = task["unlabeled_probe_inputs"]
    if not isinstance(probes, list) or not probes:
        raise ValueError("unlabeled probes must be a nonempty list")
    if any(
        not isinstance(values, list)
        or not values
        or any(not isinstance(item, int) or isinstance(item, bool) for item in values)
        for values in probes
    ):
        raise ValueError("unlabeled probes must contain nonempty integer lists")
    all_inputs = [
        *(row["input"] for row in task["visible"]),
        *(row["input"] for row in task["hidden"]),
        *probes,
    ]
    if len({tuple(values) for values in all_inputs}) != len(all_inputs):
        raise ValueError("task input partitions overlap")

    visible_inputs = [row["input"] for row in task["visible"]]
    visible_outputs = tuple(tuple(row["output"]) for row in task["visible"])
    visible_programs = matching_programs(visible_inputs, visible_outputs)
    if not visible_programs:
        raise ValueError("no depth-two program fits the visible examples")
    if {program[0] for program in visible_programs} != {first_operation}:
        raise ValueError(
            "visible examples do not uniquely identify the bound first operation"
        )
    frozen_programs = [canonical_program(program) for program in visible_programs]
    if task["visible_consistent_programs"] != frozen_programs:
        raise ValueError("frozen visible-consistent program report is incomplete")
    if task["visible_consistent_program_count"] != len(visible_programs):
        raise ValueError("visible-consistent program count disagrees with enumeration")
    expected_first = [canonical_operation(first_operation)]
    if task["visible_matching_first_operations"] != expected_first:
        raise ValueError("frozen concrete first-operation support disagrees")
    depth_one_matches = matching_depth_one(visible_inputs, visible_outputs)
    frozen_depth_one = [canonical_operation(value) for value in depth_one_matches]
    if task["visible_matching_depth_one"] != frozen_depth_one:
        raise ValueError("frozen depth-one report disagrees with enumeration")
    if depth_one_matches:
        raise ValueError("visible behavior is depth one")

    for row in [*task["visible"], *task["hidden"]]:
        if apply_pipeline(row["input"], pipeline) != row["output"]:
            raise ValueError("stored example disagrees with pipeline")
    hidden_inputs = [row["input"] for row in task["hidden"]]
    hidden_behaviors = {_outputs(program, hidden_inputs) for program in visible_programs}
    if task["visible_consistent_hidden_behavior_count"] != len(hidden_behaviors):
        raise ValueError("frozen hidden-behavior equivalence report disagrees")
    if len(hidden_behaviors) != 1:
        raise ValueError("visible-consistent programs are not hidden equivalent")
    probe_behaviors = {
        _outputs(program, probes) for program in visible_programs
    }
    if task["visible_consistent_probe_behavior_count"] != len(probe_behaviors):
        raise ValueError("frozen probe-behavior equivalence report disagrees")
    if len(probe_behaviors) != 1:
        raise ValueError("visible-consistent programs are not probe equivalent")


def validate_splits(splits: dict[str, Sequence[dict[str, Any]]]) -> None:
    ids: set[str] = set()
    behaviors: set[str] = set()
    tasks: set[str] = set()
    for split, rows in splits.items():
        counts: Counter[BoundOperation] = Counter()
        for task in rows:
            validate_task(task)
            counts[operation_from_record(task["first_op"])] += 1
            values = (
                task["task_id"], behavior_fingerprint(task), task_fingerprint(task)
            )
            if values[0] in ids or values[1] in behaviors or values[2] in tasks:
                raise ValueError("duplicate task or behavior across splits")
            ids.add(values[0])
            behaviors.add(values[1])
            tasks.add(values[2])
        support = [counts[operation] for operation in IDENTIFIABLE_FIRST_OPERATIONS]
        if support and max(support) - min(support) > 1:
            raise ValueError(f"{split} is not balanced over 23 bound first operations")
