from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .operator_library import OperatorSpec, build_operator_library


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    hole_count: int
    output_kind: str


TEMPLATES = [
    TemplateSpec("single_mod", hole_count=1, output_kind="int"),
    TemplateSpec("single_offset", hole_count=1, output_kind="int"),
    TemplateSpec("pair_affine_mod", hole_count=2, output_kind="int"),
    TemplateSpec("pair_compare_gate", hole_count=2, output_kind="bool"),
]


def _values(rng: random.Random, *, min_len: int = 3, max_len: int = 8) -> list[int]:
    return [rng.randint(0, 20) for _ in range(rng.randint(min_len, max_len))]


def template_eval(template: str, op_values: tuple[int, ...], env: dict[str, Any]) -> int:
    if template == "single_mod":
        return op_values[0] % int(env["m"])
    if template == "single_offset":
        return op_values[0] + int(env["offset"])
    if template == "pair_affine_mod":
        return (3 * op_values[0] + op_values[1]) % int(env["m"])
    if template == "pair_compare_gate":
        return int((op_values[0] - op_values[1]) > int(env["threshold"]))
    raise KeyError(template)


def program_for(template: str, operators: tuple[str, ...]) -> str:
    if template == "single_mod":
        return f"(mod ({operators[0]} xs) m)"
    if template == "single_offset":
        return f"(add ({operators[0]} xs) offset)"
    if template == "pair_affine_mod":
        return f"(mod (add (mul 3 ({operators[0]} xs)) ({operators[1]} xs)) m)"
    if template == "pair_compare_gate":
        return f"(gt (sub ({operators[0]} xs) ({operators[1]} xs)) threshold)"
    raise KeyError(template)


def _choose_targets(library_size: int, template: TemplateSpec, rng: random.Random) -> tuple[int, ...]:
    if template.hole_count == 1:
        return (rng.randrange(library_size),)
    left = rng.randrange(library_size)
    right = rng.randrange(library_size)
    if right == left:
        right = (right + 1) % library_size
    return (left, right)


def _make_env(template: str, target_values: tuple[int, ...], rng: random.Random) -> dict[str, Any]:
    env: dict[str, Any] = {"xs": _values(rng)}
    if template in {"single_mod", "pair_affine_mod"}:
        env["m"] = rng.choice([5, 7, 8, 9, 11, 13, 16, 17, 19, 23])
    elif template == "single_offset":
        env["offset"] = rng.randint(-25, 25)
    elif template == "pair_compare_gate":
        delta = target_values[0] - target_values[1]
        env["threshold"] = delta + rng.choice([-8, -5, -3, -1, 1, 3, 5, 8])
    else:
        raise KeyError(template)
    return env


def _case(template: str, target_indexes: tuple[int, ...], operators: list[OperatorSpec], rng: random.Random) -> dict[str, Any]:
    while True:
        xs = _values(rng)
        target_values = tuple(operators[index].eval(xs) for index in target_indexes)
        env = {"xs": xs}
        if template in {"single_mod", "pair_affine_mod"}:
            env["m"] = rng.choice([5, 7, 8, 9, 11, 13, 16, 17, 19, 23])
        elif template == "single_offset":
            env["offset"] = rng.randint(-25, 25)
        elif template == "pair_compare_gate":
            delta = target_values[0] - target_values[1]
            env["threshold"] = delta + rng.choice([-8, -5, -3, -1, 1, 3, 5, 8])
        else:
            raise KeyError(template)
        expected = template_eval(template, target_values, env)
        return {"input": env, "expected": expected}


def _target_bucket(indexes: tuple[int, ...], library_size: int) -> str:
    highest = max(indexes)
    fraction = highest / max(library_size - 1, 1)
    if fraction < 0.25:
        return "early"
    if fraction < 0.75:
        return "middle"
    return "late"


def build_records(
    *,
    library_sizes: list[int],
    records_per_template: int,
    visible_count: int,
    hidden_count: int,
    query_count: int,
    seed: int,
    max_library_size: int = 512,
) -> tuple[list[dict[str, Any]], list[OperatorSpec]]:
    rng = random.Random(seed)
    full_library = build_operator_library(max_library_size)
    records: list[dict[str, Any]] = []
    total_cases = visible_count + hidden_count + query_count

    for library_size in library_sizes:
        operators = full_library[:library_size]
        for template in TEMPLATES:
            for index in range(records_per_template):
                target_indexes = _choose_targets(library_size, template, rng)
                target_names = tuple(operators[target_index].name for target_index in target_indexes)
                cases = [_case(template.name, target_indexes, operators, rng) for _ in range(total_cases)]
                record = {
                    "id": f"lib{library_size:04d}_{template.name}_{index:04d}",
                    "library_size": library_size,
                    "template": template.name,
                    "hole_count": template.hole_count,
                    "output_kind": template.output_kind,
                    "target_operator_indexes": list(target_indexes),
                    "target_operators": list(target_names),
                    "target_bucket": _target_bucket(target_indexes, library_size),
                    "target_program": program_for(template.name, target_names),
                    "visible": cases[:visible_count],
                    "hidden": cases[visible_count : visible_count + hidden_count],
                    "query_pool": cases[visible_count + hidden_count :],
                }
                records.append(record)
    rng.shuffle(records)
    return records, full_library

