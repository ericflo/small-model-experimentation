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


def build_records(
    *,
    library_size: int,
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
        template = TEMPLATES[index % len(TEMPLATES)].name
        left, right = _target_indexes(library_size, rng)
        cases = [_case(template, left, right, operators, rng) for _ in range(total_cases)]
        records.append(
            {
                "id": f"{split}_{template}_{index:05d}",
                "split": split,
                "library_size": library_size,
                "template": template,
                "template_text": template_text(template),
                "target_left_index": left,
                "target_right_index": right,
                "target_left_alias": operators[left].alias,
                "target_right_alias": operators[right].alias,
                "target_pair_rank": left * library_size + right + 1,
                "visible": cases[:visible_count],
                "hidden": cases[visible_count : visible_count + hidden_count],
                "query_pool": cases[visible_count + hidden_count :],
            }
        )
    rng.shuffle(records)
    return records


def slot_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "id": record["id"] + "_left",
                "record_id": record["id"],
                "slot": "LEFT",
                "answer_alias": record["target_left_alias"],
                "answer_index": record["target_left_index"],
                "record": record,
            }
        )
        rows.append(
            {
                "id": record["id"] + "_right",
                "record_id": record["id"],
                "slot": "RIGHT",
                "answer_alias": record["target_right_alias"],
                "answer_index": record["target_right_index"],
                "record": record,
            }
        )
    return rows

