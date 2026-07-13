"""Fresh exact-depth-two tasks with task-local alias semantics."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
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
    assert parameter is not None
    return [value + parameter for value in xs]


def _mul_k(xs: list[int], parameter: int | None) -> list[int]:
    assert parameter is not None
    return [value * parameter for value in xs]


def _take_k(xs: list[int], parameter: int | None) -> list[int]:
    assert parameter is not None
    return xs[:parameter]


def _rotate_k(xs: list[int], parameter: int | None) -> list[int]:
    assert parameter is not None
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
IDENTIFIABLE_FIRST_OPERATIONS = tuple(name for name in OPERATIONS if name != "negate")


def concrete_operations() -> tuple[tuple[str, int | None], ...]:
    return tuple(
        (name, parameter)
        for name, (_function, parameters) in OPERATIONS.items()
        for parameter in parameters
    )


def apply_pipeline(values: list[int], pipeline: Sequence[tuple[str, int | None]]) -> list[int]:
    state = list(values)
    for name, parameter in pipeline:
        state = OPERATIONS[name][0](state, parameter)
        if len(state) > 64 or any(abs(value) > 10**7 for value in state):
            raise ValueError("pipeline state exceeded safety bound")
    return state


def _random_input(rng: random.Random) -> list[int]:
    return [rng.randint(-9, 9) for _ in range(rng.randint(4, 8))]


def _outputs(
    pipeline: Sequence[tuple[str, int | None]], inputs: Sequence[list[int]]
) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(apply_pipeline(values, pipeline)) for values in inputs)


def matching_first_types(
    inputs: list[list[int]], outputs: tuple[tuple[int, ...], ...]
) -> set[str]:
    matches: set[str] = set()
    concrete = concrete_operations()
    for first in concrete:
        try:
            intermediate = [apply_pipeline(values, [first]) for values in inputs]
        except ValueError:
            continue
        for second in concrete:
            try:
                candidate = tuple(
                    tuple(apply_pipeline(values, [second])) for values in intermediate
                )
            except ValueError:
                continue
            if candidate == outputs:
                matches.add(first[0])
    return matches


def matching_depth_one(
    inputs: list[list[int]], outputs: tuple[tuple[int, ...], ...]
) -> list[tuple[str, int | None]]:
    matches = []
    for operation in concrete_operations():
        try:
            candidate = _outputs([operation], inputs)
        except ValueError:
            continue
        if candidate == outputs:
            matches.append(operation)
    return matches


def _alias_to_operation(
    operations: Sequence[str], aliases: Sequence[str], *, shift: int
) -> dict[str, str]:
    """Latin-cycle mapping: each pair balances on 12-task multiples."""

    return {
        aliases[(index + shift) % len(aliases)]: operation
        for index, operation in enumerate(operations)
    }


def _result_labels(
    operations: Sequence[str], labels: Sequence[str], *, shift: int
) -> dict[str, str]:
    return {
        operation: labels[(index + shift) % len(labels)]
        for index, operation in enumerate(operations)
    }


def make_task(
    rng: random.Random,
    *,
    task_id: str,
    first_name: str,
    alias_to_operation: dict[str, str],
    result_label_by_operation: dict[str, str],
    source_alias: str,
    visible: int,
    hidden: int,
) -> dict[str, Any]:
    concrete = concrete_operations()
    first_choices = tuple(value for value in concrete if value[0] == first_name)
    for _ in range(3000):
        pipeline = [rng.choice(first_choices), rng.choice(concrete)]
        inputs: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        while len(inputs) < visible + hidden:
            value = _random_input(rng)
            if tuple(value) not in seen:
                seen.add(tuple(value))
                inputs.append(value)
        try:
            outputs = _outputs(pipeline, inputs)
        except ValueError:
            continue
        visible_inputs = inputs[:visible]
        visible_outputs = outputs[:visible]
        if outputs == tuple(tuple(value) for value in inputs):
            continue
        if matching_first_types(visible_inputs, visible_outputs) != {first_name}:
            continue
        if matching_depth_one(visible_inputs, visible_outputs):
            continue
        operation_to_alias = {operation: alias for alias, operation in alias_to_operation.items()}
        examples = [
            {"input": value, "output": list(output)}
            for value, output in zip(inputs, outputs, strict=True)
        ]
        task = {
            "task_id": task_id,
            "depth": 2,
            "alias_to_operation": alias_to_operation,
            "source_alias": source_alias,
            "result_label_by_operation": result_label_by_operation,
            "first_op": first_name,
            "correct_alias": operation_to_alias[first_name],
            "target_pipeline": [
                {"name": name, "parameter": parameter} for name, parameter in pipeline
            ],
            "visible": examples[:visible],
            "hidden": examples[visible:],
            "visible_matching_first_types": [first_name],
            "visible_matching_depth_one": [],
        }
        validate_task(task)
        return task
    raise RuntimeError(f"failed to construct exact-depth task {task_id}")


def build_splits(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    data = config["data"]
    seed = int(config["seeds"]["split"])
    label_seed = int(config["seeds"]["label_map"])
    operations = tuple(data["operation_names"])
    aliases = tuple(data["alias_tokens"])
    labels = tuple(config["anchor"]["result_labels"])
    if operations != tuple(OPERATIONS):
        raise ValueError("configured operation order changed")
    specs = (
        ("mechanics", int(data["mechanics_tasks"]), seed + 11, 11),
        ("qualification", int(data["qualification_tasks"]), seed + 23, 23),
        ("confirmation", int(data["confirmation_tasks"]), seed + 37, 37),
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for split, count, split_seed, label_salt in specs:
        rng = random.Random(split_seed)
        first_offset = rng.randrange(len(IDENTIFIABLE_FIRST_OPERATIONS))
        alias_offset = rng.randrange(len(aliases))
        # Retain the split RNG draw so task behaviors stay invariant to this
        # independent diagnostic-label stream, then use the preregistered seed.
        rng.randrange(len(labels))
        label_offset = random.Random(label_seed + label_salt).randrange(len(labels))
        source_offset = rng.randrange(len(aliases))
        rows = []
        for index in range(count):
            rows.append(make_task(
                rng,
                task_id=f"{split}-{index:05d}",
                first_name=IDENTIFIABLE_FIRST_OPERATIONS[
                    (index + first_offset) % len(IDENTIFIABLE_FIRST_OPERATIONS)
                ],
                alias_to_operation=_alias_to_operation(
                    operations, aliases, shift=(index + alias_offset) % len(aliases)
                ),
                result_label_by_operation=_result_labels(
                    operations, labels, shift=(index + label_offset) % len(labels)
                ),
                source_alias=aliases[(index * 5 + source_offset) % len(aliases)],
                visible=int(data["visible_examples"]),
                hidden=int(data["hidden_examples"]),
            ))
        result[split] = rows
    validate_splits(result)
    return result


def behavior_fingerprint(task: dict[str, Any]) -> str:
    payload = {
        "depth": task["depth"],
        "visible": task["visible"],
        "hidden": task["hidden"],
        "first_op": task["first_op"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def task_fingerprint(task: dict[str, Any]) -> str:
    payload = {
        "behavior": behavior_fingerprint(task),
        "alias_to_operation": task["alias_to_operation"],
        "source_alias": task["source_alias"],
        "result_label_by_operation": task["result_label_by_operation"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def public_mechanics(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "visible": task["visible"],
        "alias_to_operation": task["alias_to_operation"],
        "source_alias": task["source_alias"],
        "result_label_by_operation": task["result_label_by_operation"],
    }


def task_prompt(task: dict[str, Any]) -> str:
    required = {"visible", "alias_to_operation"}
    if not required.issubset(task):
        raise ValueError("task prompt lacks public fields")
    examples = "\n".join(
        f"transform({row['input']!r}) == {row['output']!r}" for row in task["visible"]
    )
    menu = ", ".join(
        f"{alias}={operation}" for alias, operation in task["alias_to_operation"].items()
    )
    return (
        "Infer the hidden sequence of exactly two list operations from these examples:\n"
        f"{examples}\n\nOperation aliases for this task: {menu}.\n"
        "Reason naturally in the private reasoning section. Then answer with exactly "
        "`First: <alias>` and no other final text."
    )


def validate_task(task: dict[str, Any]) -> None:
    alias_to_operation = task["alias_to_operation"]
    if len(alias_to_operation) != 12 or set(alias_to_operation.values()) != set(OPERATIONS):
        raise ValueError("task alias mapping is not a complete bijection")
    if task["source_alias"] not in alias_to_operation:
        raise ValueError("source alias is outside the task mapping")
    labels = task["result_label_by_operation"]
    if set(labels) != set(OPERATIONS) or len(set(labels.values())) != 12:
        raise ValueError("diagnostic result labels are not a complete bijection")
    if task["correct_alias"] not in alias_to_operation:
        raise ValueError("correct alias is outside the task mapping")
    if alias_to_operation[task["correct_alias"]] != task["first_op"]:
        raise ValueError("stored correct alias disagrees with first operation")
    if task["visible_matching_first_types"] != [task["first_op"]]:
        raise ValueError("visible examples do not uniquely identify first operation")
    if task["visible_matching_depth_one"]:
        raise ValueError("task is behaviorally depth one")


def validate_splits(splits: dict[str, Sequence[dict[str, Any]]]) -> None:
    ids: set[str] = set()
    behaviors: set[str] = set()
    full: set[str] = set()
    for rows in splits.values():
        for task in rows:
            validate_task(task)
            if task["task_id"] in ids:
                raise ValueError("duplicate task id across splits")
            ids.add(task["task_id"])
            behavior = behavior_fingerprint(task)
            fingerprint = task_fingerprint(task)
            if behavior in behaviors or fingerprint in full:
                raise ValueError("duplicate task across splits")
            behaviors.add(behavior)
            full.add(fingerprint)
