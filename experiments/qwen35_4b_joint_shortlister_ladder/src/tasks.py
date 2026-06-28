from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .operator_library import OperatorSpec, build_operator_library


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    output_kind: str


TEMPLATES = [
    TemplateSpec("pair_affine_mod", output_kind="int"),
    TemplateSpec("pair_compare_gate", output_kind="bool"),
]


def template_eval(template: str, left_value: int, right_value: int, env: dict[str, Any]) -> int:
    if template == "pair_affine_mod":
        return (3 * left_value + right_value) % int(env["m"])
    if template == "pair_compare_gate":
        return int((left_value - right_value) > int(env["threshold"]))
    raise KeyError(template)


def template_text(template: str) -> str:
    if template == "pair_affine_mod":
        return "output = (3 * LEFT(xs) + RIGHT(xs)) modulo m"
    if template == "pair_compare_gate":
        return "output = 1 if LEFT(xs) - RIGHT(xs) > threshold else 0"
    raise KeyError(template)


def _values(rng: random.Random, *, min_len: int = 3, max_len: int = 8) -> list[int]:
    return [rng.randint(0, 20) for _ in range(rng.randint(min_len, max_len))]


def _case(template: str, left_index: int, right_index: int, operators: list[OperatorSpec], rng: random.Random) -> dict[str, Any]:
    xs = _values(rng)
    left_value = operators[left_index].eval(xs)
    right_value = operators[right_index].eval(xs)
    env: dict[str, Any] = {"xs": xs}
    if template == "pair_affine_mod":
        env["m"] = rng.choice([5, 7, 8, 9, 11, 13, 16, 17, 19, 23])
    elif template == "pair_compare_gate":
        delta = left_value - right_value
        env["threshold"] = delta + rng.choice([-8, -5, -3, -1, 1, 3, 5, 8])
    else:
        raise KeyError(template)
    expected = template_eval(template, left_value, right_value, env)
    return {"input": env, "expected": expected}


def _target_indexes(library_size: int, rng: random.Random) -> tuple[int, int]:
    left = rng.randrange(library_size)
    right = rng.randrange(library_size)
    if right == left:
        right = (right + 1) % library_size
    return left, right


def _inventory_permutation(library_size: int, rng: random.Random) -> list[int]:
    values = list(range(library_size))
    rng.shuffle(values)
    return values


def _code_for_operator(permutation: list[int], operator_index: int) -> int:
    return permutation.index(operator_index)


def build_records_for_cell(
    *,
    library_size: int,
    template: str,
    count: int,
    visible_count: int,
    hidden_count: int,
    query_count: int,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    operators = build_operator_library(library_size)
    records: list[dict[str, Any]] = []
    total_cases = visible_count + hidden_count + query_count
    for index in range(count):
        left, right = _target_indexes(library_size, rng)
        inventory_permutation = _inventory_permutation(library_size, rng)
        left_code = _code_for_operator(inventory_permutation, left)
        right_code = _code_for_operator(inventory_permutation, right)
        cases = [_case(template, left, right, operators, rng) for _ in range(total_cases)]
        records.append(
            {
                "id": f"{split}_lib{library_size}_{template}_{index:05d}",
                "split": split,
                "library_size": library_size,
                "template": template,
                "template_text": template_text(template),
                "target_left_index": left,
                "target_right_index": right,
                "target_left_code": left_code,
                "target_right_code": right_code,
                "target_pair_text": f"{left_code:03d},{right_code:03d}",
                "inventory_permutation": inventory_permutation,
                "visible": cases[:visible_count],
                "hidden": cases[visible_count : visible_count + hidden_count],
                "query_pool": cases[visible_count + hidden_count :],
            }
        )
    rng.shuffle(records)
    return records


def build_ladder_records(
    *,
    library_sizes: list[int],
    count_per_cell: int,
    visible_count: int,
    hidden_count: int,
    query_count: int,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for size_index, library_size in enumerate(library_sizes):
        for template_index, template in enumerate(template.name for template in TEMPLATES):
            cell_seed = seed + 10_003 * size_index + 331 * template_index
            rows.extend(
                build_records_for_cell(
                    library_size=library_size,
                    template=template,
                    count=count_per_cell,
                    visible_count=visible_count,
                    hidden_count=hidden_count,
                    query_count=query_count,
                    seed=cell_seed,
                    split=split,
                )
            )
    rng = random.Random(seed + 91_919)
    rng.shuffle(rows)
    return rows


def pair_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"] + "_pair",
            "record_id": row["id"],
            "answer_pair": row["target_pair_text"],
            "record": row,
        }
        for row in records
    ]

