"""Fresh first-operation-identifiable procedural list tasks."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
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
# Negate-first depth-two pipelines are algebraically reorderable in this DSL.
IDENTIFIABLE_FIRST_OPERATIONS = tuple(name for name in OPERATIONS if name != "negate")


def concrete_operations() -> tuple[tuple[str, int | None], ...]:
    return tuple(
        (name, parameter)
        for name, (_function, parameters) in OPERATIONS.items()
        for parameter in parameters
    )


def apply_pipeline(values: list[int], pipeline: list[tuple[str, int | None]]) -> list[int]:
    state = list(values)
    for name, parameter in pipeline:
        state = OPERATIONS[name][0](state, parameter)
        if len(state) > 64 or any(abs(value) > 10**7 for value in state):
            raise ValueError("pipeline state exceeded safety bound")
    return state


def _random_input(rng: random.Random) -> list[int]:
    return [rng.randint(-9, 9) for _ in range(rng.randint(4, 8))]


def _matching_first_types(
    inputs: list[list[int]], outputs: tuple[tuple[int, ...], ...], *, depth: int
) -> set[str]:
    if depth != 2:
        raise ValueError("this experiment freezes depth two")
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


def make_task(
    rng: random.Random,
    *,
    task_id: str,
    first_name: str,
    depth: int,
    visible: int,
    hidden: int,
) -> dict[str, Any]:
    concrete = concrete_operations()
    first_choices = tuple(value for value in concrete if value[0] == first_name)
    for _ in range(2000):
        pipeline = [rng.choice(first_choices), rng.choice(concrete)]
        inputs: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        while len(inputs) < visible + hidden:
            value = _random_input(rng)
            if tuple(value) not in seen:
                seen.add(tuple(value))
                inputs.append(value)
        try:
            outputs = tuple(tuple(apply_pipeline(value, pipeline)) for value in inputs)
        except ValueError:
            continue
        if outputs == tuple(tuple(value) for value in inputs):
            continue
        if _matching_first_types(inputs[:visible], outputs[:visible], depth=depth) != {first_name}:
            continue
        examples = [
            {"input": value, "output": list(output)}
            for value, output in zip(inputs, outputs, strict=True)
        ]
        return {
            "task_id": task_id,
            "depth": depth,
            "first_op": first_name,
            "target_pipeline": [
                {"name": name, "parameter": parameter} for name, parameter in pipeline
            ],
            "visible": examples[:visible],
            "hidden": examples[visible:],
            "visible_matching_first_types": [first_name],
        }
    raise RuntimeError(f"failed to construct identifiable task {task_id}")


def build_splits(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    data = config["data"]
    seed = int(config["seeds"]["split"])
    specifications = (
        ("budget_selection", int(data["budget_selection_tasks"]), seed + 11),
        ("seam_confirmation", int(data["seam_confirmation_tasks"]), seed + 23),
    )
    splits: dict[str, list[dict[str, Any]]] = {}
    for split, count, split_seed in specifications:
        rng = random.Random(split_seed)
        offset = rng.randrange(len(IDENTIFIABLE_FIRST_OPERATIONS))
        splits[split] = [
            make_task(
                rng,
                task_id=f"{split}-{index:05d}",
                first_name=IDENTIFIABLE_FIRST_OPERATIONS[
                    (index + offset) % len(IDENTIFIABLE_FIRST_OPERATIONS)
                ],
                depth=int(data["depth"]),
                visible=int(data["visible_examples"]),
                hidden=int(data["hidden_examples"]),
            )
            for index in range(count)
        ]
    return splits


def task_fingerprint(task: dict[str, Any]) -> str:
    payload = {
        "depth": task["depth"],
        "visible": task["visible"],
        "hidden": task["hidden"],
        "first_op": task["first_op"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def task_prompt(task: dict[str, Any], aliases: dict[str, str]) -> str:
    if set(aliases) != set(OPERATIONS) or len(set(aliases.values())) != len(aliases):
        raise ValueError("operation aliases must be a one-to-one complete mapping")
    examples = "\n".join(
        f"transform({row['input']!r}) == {row['output']!r}" for row in task["visible"]
    )
    menu = ", ".join(f"{alias}={name}" for name, alias in aliases.items())
    # Byte-identical wording to the immediate parent's prompt contract.
    return (
        "Infer the hidden sequence of exactly two list operations from these examples:\n"
        f"{examples}\n\nOperation aliases: {menu}.\n"
        "Reason naturally in the private reasoning section. Then answer with exactly "
        "`First: <alias>` and no other final text."
    )


def parse_alias(text: str, aliases: dict[str, str]) -> str | None:
    if "First:" not in text:
        return None
    tail = text.rsplit("First:", 1)[1].strip().split()
    if not tail:
        return None
    candidate = tail[0].strip("`.,:;\n")
    return candidate if candidate in set(aliases.values()) else None


def verify_answer(task: dict[str, Any], text: str, aliases: dict[str, str]) -> bool:
    return parse_alias(text, aliases) == aliases[task["first_op"]]
