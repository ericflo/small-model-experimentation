"""Fresh exact-depth-three list programs for answer-seam calibration and mechanics."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import product
from typing import Any, Callable

from identity import TASK_NAMESPACE, namespaced_task_id, public_instance_fingerprint


BoundOperation = tuple[str, int | None]
Program = tuple[BoundOperation, ...]


class _Invalid:
    pass


INVALID = _Invalid()
State = list[int] | _Invalid


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
    total = 0
    result: list[int] = []
    for value in xs:
        total += value
        result.append(total)
    return result


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
CONCRETE_OPERATIONS: tuple[BoundOperation, ...] = tuple(
    (name, parameter)
    for name, (_function, parameters) in OPERATIONS.items()
    for parameter in parameters
)
ALIASES: tuple[str, ...] = tuple(chr(ord("A") + index) for index in range(24))
OPERATION_TO_ALIAS = dict(zip(CONCRETE_OPERATIONS, ALIASES, strict=True))
ALIAS_TO_OPERATION = dict(zip(ALIASES, CONCRETE_OPERATIONS, strict=True))
DEPTH_TWO: tuple[Program, ...] = tuple(
    tuple(values) for values in product(CONCRETE_OPERATIONS, repeat=2)
)
DEPTH_THREE: tuple[Program, ...] = tuple(
    tuple(values) for values in product(CONCRETE_OPERATIONS, repeat=3)
)
if len(CONCRETE_OPERATIONS) != 24 or len(DEPTH_THREE) != 13_824:
    raise RuntimeError("the frozen operation bank changed")


def operation_record(operation: BoundOperation) -> dict[str, int | str | None]:
    if operation not in CONCRETE_OPERATIONS:
        raise ValueError("operation is outside the frozen bank")
    return {"name": operation[0], "parameter": operation[1]}


def operation_from_record(value: dict[str, Any]) -> BoundOperation:
    if not isinstance(value, dict) or set(value) != {"name", "parameter"}:
        raise ValueError("operation record has the wrong schema")
    operation = (value["name"], value["parameter"])
    if operation not in CONCRETE_OPERATIONS:
        raise ValueError("operation record is outside the frozen bank")
    return operation


def canonical_operation(operation: BoundOperation) -> str:
    name, parameter = operation
    if operation not in CONCRETE_OPERATIONS:
        raise ValueError("operation is outside the frozen bank")
    return name if parameter is None else f"{name}({parameter})"


def canonical_program(program: Sequence[BoundOperation]) -> str:
    values = tuple(program)
    if len(values) not in {2, 3}:
        raise ValueError("program arity must be two or three")
    return " | ".join(canonical_operation(operation) for operation in values)


def alias_program(program: Sequence[BoundOperation]) -> str:
    return " | ".join(OPERATION_TO_ALIAS[operation] for operation in program)


def apply_pipeline(values: list[int], program: Sequence[BoundOperation]) -> State:
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, int) or isinstance(value, bool) for value in values)
    ):
        return INVALID
    state = list(values)
    for operation in program:
        if operation not in CONCRETE_OPERATIONS:
            return INVALID
        try:
            state = OPERATIONS[operation[0]][0](state, operation[1])
        except (TypeError, ValueError, OverflowError):
            return INVALID
        if not state or len(state) > 64 or any(abs(value) > 10**7 for value in state):
            return INVALID
    return state


def _vector(program: Sequence[BoundOperation], inputs: Sequence[list[int]]) -> tuple[tuple[int, ...], ...] | None:
    result: list[tuple[int, ...]] = []
    for values in inputs:
        state = apply_pipeline(values, program)
        if state is INVALID:
            return None
        result.append(tuple(state))
    return tuple(result)


def _fingerprint(outputs: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(
        json.dumps(outputs, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _draw_inputs(
    *,
    seed: int,
    count: int,
    config: dict[str, Any],
    forbidden: set[tuple[int, ...]],
    target: Program,
) -> list[list[int]]:
    data = config["data"]
    rng = random.Random(seed)
    result: list[list[int]] = []
    for _ in range(int(data["max_input_draws"])):
        if len(result) == count:
            break
        length = rng.randint(int(data["input_min_length"]), int(data["input_max_length"]))
        values = [
            rng.randint(int(data["input_value_min"]), int(data["input_value_max"]))
            for _ in range(length)
        ]
        key = tuple(values)
        if key in forbidden or apply_pipeline(values, target) is INVALID:
            continue
        forbidden.add(key)
        result.append(values)
    if len(result) != count:
        raise RuntimeError("fresh input draw limit exhausted")
    return result


def build_common_panel(config: dict[str, Any]) -> list[list[int]]:
    data = config["data"]
    rng = random.Random(int(config["seeds"]["construction"]))
    result: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    per_length = int(data["common_panel_per_length"])
    for length in range(int(data["input_min_length"]), int(data["input_max_length"]) + 1):
        while sum(len(row) == length for row in result) < per_length:
            values = [
                rng.randint(int(data["input_value_min"]), int(data["input_value_max"]))
                for _ in range(length)
            ]
            if tuple(values) not in seen:
                seen.add(tuple(values))
                result.append(values)
    return result


def fitting_suffixes(
    visible_inputs: Sequence[list[int]], visible_outputs: Sequence[Sequence[int]]
) -> dict[BoundOperation, tuple[Program, ...]]:
    expected = tuple(tuple(row) for row in visible_outputs)
    result: dict[BoundOperation, tuple[Program, ...]] = {}
    for candidate in CONCRETE_OPERATIONS:
        states = _vector((candidate,), visible_inputs)
        if states is None:
            continue
        matches: list[Program] = []
        for suffix in DEPTH_TWO:
            outputs: list[tuple[int, ...]] = []
            for state in states:
                value = apply_pipeline(list(state), suffix)
                if value is INVALID:
                    break
                outputs.append(tuple(value))
            if tuple(outputs) == expected:
                matches.append(suffix)
        if matches:
            result[candidate] = tuple(matches)
    return result


def _shallow_fingerprints(common: Sequence[list[int]]) -> set[str]:
    result = {_fingerprint(common)}
    for operation in CONCRETE_OPERATIONS:
        outputs = _vector((operation,), common)
        if outputs is not None:
            result.add(_fingerprint(outputs))
    for program in DEPTH_TWO:
        outputs = _vector(program, common)
        if outputs is not None:
            result.add(_fingerprint(outputs))
    return result


def _visible_has_shallow_match(inputs: Sequence[list[int]], outputs: Sequence[Sequence[int]]) -> bool:
    expected = tuple(tuple(row) for row in outputs)
    if tuple(tuple(row) for row in inputs) == expected:
        return True
    return any(
        _vector(program, inputs) == expected
        for program in (
            *((operation,) for operation in CONCRETE_OPERATIONS),
            *DEPTH_TWO,
        )
    )


def _stratum(live_count: int) -> str | None:
    return {1: "single", 2: "double", 3: "triple", 4: "quad"}.get(live_count)


def _candidate(
    *,
    program: Program,
    desired: str,
    config: dict[str, Any],
    global_forbidden: set[tuple[int, ...]],
) -> dict[str, Any] | None:
    base = int(config["seeds"]["construction"])
    data = config["data"]
    canonical = canonical_program(program)
    for attempt in range(int(data["max_visible_attempts"])):
        local_forbidden = set(global_forbidden)
        visible = _draw_inputs(
            seed=_stable_seed(base, "visible", canonical, str(attempt)),
            count=int(data["visible_examples"]),
            config=config,
            forbidden=local_forbidden,
            target=program,
        )
        outputs = _vector(program, visible)
        if (
            outputs is None
            or len(set(outputs)) != len(outputs)
            or _visible_has_shallow_match(visible, outputs)
        ):
            continue
        candidate_states = [_vector((operation,), visible) for operation in CONCRETE_OPERATIONS]
        if any(value is None for value in candidate_states) or len(set(candidate_states)) != 24:
            continue
        live = fitting_suffixes(visible, outputs)
        if _stratum(len(live)) != desired:
            continue
        hidden = _draw_inputs(
            seed=_stable_seed(base, "hidden", canonical, str(attempt)),
            count=int(data["hidden_examples"]),
            config=config,
            forbidden=local_forbidden,
            target=program,
        )
        probes = _draw_inputs(
            seed=_stable_seed(base, "probe", canonical, str(attempt)),
            count=int(data["unlabeled_probe_inputs"]),
            config=config,
            forbidden=local_forbidden,
            target=program,
        )
        hidden_outputs = _vector(program, hidden)
        if hidden_outputs is None:
            raise RuntimeError("accepted target became invalid on hidden inputs")
        return {
            "target": program,
            "visible": visible,
            "visible_outputs": outputs,
            "hidden": hidden,
            "hidden_outputs": hidden_outputs,
            "probes": probes,
            "live": live,
            "stratum": desired,
            "attempt": attempt,
            "reserved_inputs": local_forbidden - global_forbidden,
        }
    return None


def _schedule(split: str, config: dict[str, Any]) -> list[str]:
    values = ["single"] * 8 + ["double"] * 8 + ["triple"] * 4 + ["quad"] * 4
    random.Random(
        _stable_seed(int(config["seeds"]["construction"]), "schedule", split)
    ).shuffle(values)
    return values


def build_tasks(
    config: dict[str, Any], *,
    excluded_public_fingerprints: set[str],
    excluded_function_fingerprints: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    common = build_common_panel(config)
    shallow = _shallow_fingerprints(common)
    groups: dict[str, list[Program]] = defaultdict(list)
    for program in DEPTH_THREE:
        common_outputs = _vector(program, common)
        if common_outputs is None:
            continue
        fingerprint = _fingerprint(common_outputs)
        if fingerprint not in shallow:
            groups[fingerprint].append(program)
    function_order = sorted(set(groups) - excluded_function_fingerprints)
    random.Random(int(config["seeds"]["construction"])).shuffle(function_order)
    order_index = {
        fingerprint: index for index, fingerprint in enumerate(function_order)
    }
    required = {"single": 24, "double": 24, "triple": 12, "quad": 12}
    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_forbidden: set[tuple[int, ...]] = {tuple(row) for row in common}
    used_common: set[str] = set()
    used_programs: set[Program] = set()
    used_suffixes: set[Program] = set()
    scanned = 0
    for desired in ("quad", "triple", "double", "single"):
        desired_count = {"single": 1, "double": 2, "triple": 3, "quad": 4}[desired]
        prioritized = sorted(
            function_order,
            key=lambda fingerprint: (
                len({program[0] for program in groups[fingerprint]}) != desired_count,
                abs(
                    len({program[0] for program in groups[fingerprint]})
                    - desired_count
                ),
                order_index[fingerprint],
            ),
        )
        for fingerprint in prioritized:
            if len(queues[desired]) >= required[desired]:
                break
            if fingerprint in used_common:
                continue
            value: dict[str, Any] | None = None
            chosen_program: Program | None = None
            for program in sorted(groups[fingerprint], key=canonical_program):
                scanned += 1
                suffix = tuple(program[1:])
                if program in used_programs or suffix in used_suffixes:
                    continue
                value = _candidate(
                    program=program,
                    desired=desired,
                    config=config,
                    global_forbidden=global_forbidden,
                )
                if value is not None:
                    chosen_program = program
                    break
            if value is None:
                continue
            if chosen_program is None:
                raise RuntimeError("accepted construction candidate lacks a program")
            program = chosen_program
            suffix = tuple(program[1:])
            provisional = {
                "depth": 3,
                "visible": [
                    {"input": row, "output": list(output)}
                    for row, output in zip(
                        value["visible"], value["visible_outputs"], strict=True
                    )
                ],
                "unlabeled_probe_inputs": value["probes"],
            }
            if public_instance_fingerprint(provisional) in excluded_public_fingerprints:
                continue
            global_forbidden.update(value.pop("reserved_inputs"))
            used_common.add(fingerprint)
            used_programs.add(program)
            used_suffixes.add(suffix)
            value["common_fingerprint"] = fingerprint
            queues[desired].append(value)
        if len(queues[desired]) != required[desired]:
            raise RuntimeError(f"construction exhausted before {desired} quota")

    split_tasks: dict[str, list[dict[str, Any]]] = {}
    offsets = Counter()
    for split, count in (("calibration", 48), ("mechanics", 24)):
        rows: list[dict[str, Any]] = []
        schedule = _schedule(split, config)
        if split == "calibration":
            schedule = schedule + _schedule(split + "-second", config)
        if len(schedule) != count:
            raise RuntimeError("split schedule count changed")
        for index, desired in enumerate(schedule):
            value = queues[desired][offsets[desired]]
            offsets[desired] += 1
            task_id = namespaced_task_id(TASK_NAMESPACE, split, index)
            target: Program = value["target"]
            live: dict[BoundOperation, tuple[Program, ...]] = value["live"]
            rows.append(
                {
                    "task_id": task_id,
                    "depth": 3,
                    "stratum": desired,
                    "common_fingerprint": value["common_fingerprint"],
                    "target_pipeline": [operation_record(operation) for operation in target],
                    "visible": [
                        {"input": row, "output": list(output)}
                        for row, output in zip(
                            value["visible"], value["visible_outputs"], strict=True
                        )
                    ],
                    "hidden": [
                        {"input": row, "output": list(output)}
                        for row, output in zip(
                            value["hidden"], value["hidden_outputs"], strict=True
                        )
                    ],
                    "unlabeled_probe_inputs": value["probes"],
                    "public_live": [
                        {
                            "operation": operation_record(operation),
                            "canonical": canonical_operation(operation),
                            "first_fitting_suffix": [
                                operation_record(item) for item in live[operation][0]
                            ],
                            "visible_fitting_suffix_count": len(live[operation]),
                        }
                        for operation in CONCRETE_OPERATIONS
                        if operation in live
                    ],
                    "visible_attempt": value["attempt"],
                }
            )
        split_tasks[split] = rows
    validate_tasks(
        split_tasks,
        config,
        excluded_public_fingerprints,
        excluded_function_fingerprints,
    )
    common_fingerprints = sorted(
        task["common_fingerprint"]
        for rows in split_tasks.values()
        for task in rows
    )
    return split_tasks, {
        "schema_version": 1,
        "namespace": TASK_NAMESPACE,
        "common_panel_sha256": hashlib.sha256(
            json.dumps(common, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "common_panel_rows": len(common),
        "shallow_function_count": len(shallow),
        "eligible_exact_depth_three_functions": len(groups),
        "programs_scanned_with_repeats": scanned,
        "split_rows": {split: len(rows) for split, rows in split_tasks.items()},
        "strata": {
            split: dict(sorted(Counter(row["stratum"] for row in rows).items()))
            for split, rows in split_tasks.items()
        },
        "public_instance_fingerprints": sorted(
            public_instance_fingerprint(public_task(task))
            for rows in split_tasks.values()
            for task in rows
        ),
        "excluded_parent_public_fingerprints": len(excluded_public_fingerprints),
        "parent_public_overlap": 0,
        "common_function_fingerprints": common_fingerprints,
        "excluded_parent_function_fingerprints": len(
            excluded_function_fingerprints
        ),
        "parent_function_fingerprint_overlap": 0,
    }


def public_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "depth": task["depth"],
        "visible": task["visible"],
        "unlabeled_probe_inputs": task["unlabeled_probe_inputs"],
    }


def audit_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "stratum": task["stratum"],
        "public_live": task["public_live"],
        "visible_attempt": task["visible_attempt"],
    }


def gold_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "common_fingerprint": task["common_fingerprint"],
        "target_pipeline": task["target_pipeline"],
        "hidden": task["hidden"],
    }


def validate_tasks(
    splits: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    excluded_public_fingerprints: set[str],
    excluded_function_fingerprints: set[str],
) -> None:
    if set(splits) != {"calibration", "mechanics"}:
        raise ValueError("construction split inventory changed")
    if len(splits["calibration"]) != 48 or len(splits["mechanics"]) != 24:
        raise ValueError("construction split counts changed")
    ids = [task["task_id"] for rows in splits.values() for task in rows]
    expected = [
        namespaced_task_id(TASK_NAMESPACE, split, index)
        for split, count in (("calibration", 48), ("mechanics", 24))
        for index in range(count)
    ]
    if ids != expected or len(set(ids)) != 72:
        raise ValueError("task IDs or order changed")
    fingerprints = [
        public_instance_fingerprint(public_task(task))
        for rows in splits.values()
        for task in rows
    ]
    if len(set(fingerprints)) != 72 or set(fingerprints) & excluded_public_fingerprints:
        raise ValueError("public-instance freshness failed")
    function_fingerprints = [
        task["common_fingerprint"] for rows in splits.values() for task in rows
    ]
    if (
        len(set(function_fingerprints)) != 72
        or set(function_fingerprints) & excluded_function_fingerprints
    ):
        raise ValueError("common-function freshness failed")
    all_inputs: list[tuple[int, ...]] = []
    for rows in splits.values():
        for task in rows:
            target = tuple(
                operation_from_record(operation) for operation in task["target_pipeline"]
            )
            for section in ("visible", "hidden"):
                for row in task[section]:
                    if apply_pipeline(row["input"], target) != row["output"]:
                        raise ValueError("stored task output differs from interpreter")
                    all_inputs.append(tuple(row["input"]))
            all_inputs.extend(tuple(row) for row in task["unlabeled_probe_inputs"])
            live = fitting_suffixes(
                [row["input"] for row in task["visible"]],
                [row["output"] for row in task["visible"]],
            )
            stored = [row["canonical"] for row in task["public_live"]]
            observed = [
                canonical_operation(operation)
                for operation in CONCRETE_OPERATIONS
                if operation in live
            ]
            if stored != observed or _stratum(len(stored)) != task["stratum"]:
                raise ValueError("stored public viability differs from exhaustive result")
    if len(set(all_inputs)) != len(all_inputs):
        raise ValueError("fresh input partitions overlap across tasks")
