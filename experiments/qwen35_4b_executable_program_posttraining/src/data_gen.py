from __future__ import annotations

import json
import random
import string
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dsl import execute


@dataclass(frozen=True)
class FamilySpec:
    name: str
    schema: str
    program: str
    wrong_programs: tuple[str, ...]
    make_input: Callable[[random.Random], dict[str, Any]]


def _values(rng: random.Random, *, min_len: int = 3, max_len: int = 7) -> list[int]:
    return [rng.randint(-8, 18) for _ in range(rng.randint(min_len, max_len))]


def _word(rng: random.Random, min_len: int = 3, max_len: int = 9) -> str:
    alphabet = "abcde"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(min_len, max_len)))


def _text_with_needle(rng: random.Random) -> dict[str, Any]:
    needle = rng.choice(["aa", "bb", "cc", "de", "ed"])
    text = _word(rng, 4, 12)
    if rng.random() < 0.55:
        pos = rng.randint(0, len(text))
        text = text[:pos] + needle + text[pos:]
    return {"text": text, "needle": needle}


def make_family_specs() -> dict[str, FamilySpec]:
    return {
        "sum_label": FamilySpec(
            name="sum_label",
            schema="values: list[int]",
            program='(format "S{}" (sum values))',
            wrong_programs=('(format "S{}" (len values))', '(format "S{}" (first values))'),
            make_input=lambda rng: {"values": _values(rng)},
        ),
        "mod_scalar_label": FamilySpec(
            name="mod_scalar_label",
            schema="n: int; modulus: int",
            program='(format "R{}" (mod n modulus))',
            wrong_programs=('(format "R{}" n)', '(format "R{}" modulus)'),
            make_input=lambda rng: {"n": rng.randint(-50, 80), "modulus": rng.randint(2, 9)},
        ),
        "length_label": FamilySpec(
            name="length_label",
            schema="text: str",
            program='(format "L{}" (len text))',
            wrong_programs=('(format "L{}" text)', '(format "L{}" 0)'),
            make_input=lambda rng: {"text": _word(rng, 2, 14)},
        ),
        "contains_code": FamilySpec(
            name="contains_code",
            schema="text: str; needle: str",
            program='(if (contains text needle) "HAS" "MISS")',
            wrong_programs=('(if (contains needle text) "HAS" "MISS")', '"MISS"'),
            make_input=_text_with_needle,
        ),
        "tuple_get_label": FamilySpec(
            name="tuple_get_label",
            schema="item: list[int]; index: int",
            program='(format "T{}" (tuple_get item index))',
            wrong_programs=('(format "T{}" (sum item))', '(format "T{}" index)'),
            make_input=lambda rng: {
                "item": [rng.randint(-6, 12), rng.randint(-6, 12), rng.randint(-6, 12)],
                "index": rng.randint(0, 2),
            },
        ),
        "scalar_branch_label": FamilySpec(
            name="scalar_branch_label",
            schema="score: int; threshold: int; high_label: str; low_label: str",
            program='(if (gt score threshold) high_label low_label)',
            wrong_programs=('(if (lt score threshold) high_label low_label)', 'low_label'),
            make_input=lambda rng: {
                "score": rng.randint(-15, 30),
                "threshold": rng.randint(-5, 18),
                "high_label": rng.choice(["HIGH", "UP", "YES"]),
                "low_label": rng.choice(["LOW", "DOWN", "NO"]),
            },
        ),
        "sum_threshold_label": FamilySpec(
            name="sum_threshold_label",
            schema="values: list[int]; threshold: int; high_label: str; low_label: str",
            program='(if (gt (sum values) threshold) high_label low_label)',
            wrong_programs=('(if (gt (len values) threshold) high_label low_label)', 'low_label'),
            make_input=lambda rng: {
                "values": _values(rng),
                "threshold": rng.randint(-8, 28),
                "high_label": rng.choice(["BIG", "ABOVE", "PASS"]),
                "low_label": rng.choice(["SMALL", "BELOW", "FAIL"]),
            },
        ),
        "length_mod_label": FamilySpec(
            name="length_mod_label",
            schema="text: str; modulus: int",
            program='(format "LM{}" (mod (len text) modulus))',
            wrong_programs=('(format "LM{}" (len text))', '(format "LM{}" modulus)'),
            make_input=lambda rng: {"text": _word(rng, 2, 16), "modulus": rng.randint(2, 7)},
        ),
        "tuple_sum_label": FamilySpec(
            name="tuple_sum_label",
            schema="item: list[int]",
            program='(format "TS{}" (sum item))',
            wrong_programs=('(format "TS{}" (first item))', '(format "TS{}" (len item))'),
            make_input=lambda rng: {"item": [rng.randint(-5, 10), rng.randint(-5, 10), rng.randint(-5, 10)]},
        ),
        "contains_count_label": FamilySpec(
            name="contains_count_label",
            schema="tokens: list[str]; needle: str",
            program='(format "C{}" (count_eq tokens needle))',
            wrong_programs=('(if (contains tokens needle) "C1" "C0")', '(format "C{}" (len tokens))'),
            make_input=lambda rng: {
                "tokens": [rng.choice(list("abcd")) for _ in range(rng.randint(3, 8))],
                "needle": rng.choice(list("abcd")),
            },
        ),
        "sorted_first_label": FamilySpec(
            name="sorted_first_label",
            schema="values: list[int]",
            program='(format "F{}" (first (sort values)))',
            wrong_programs=('(format "F{}" (first values))', '(format "F{}" (last values))'),
            make_input=lambda rng: {"values": _values(rng)},
        ),
        "sum_add_label": FamilySpec(
            name="sum_add_label",
            schema="values: list[int]; offset: int",
            program='(format "A{}" (add (sum values) offset))',
            wrong_programs=('(format "A{}" (sum values))', '(format "A{}" offset)'),
            make_input=lambda rng: {"values": _values(rng), "offset": rng.randint(-10, 10)},
        ),
        "contains_and_count_code": FamilySpec(
            name="contains_and_count_code",
            schema="tokens: list[str]; needle: str; threshold: int",
            program='(if (and (contains tokens needle) (gt (count_eq tokens needle) threshold)) "MANY" "FEW")',
            wrong_programs=('(if (contains tokens needle) "MANY" "FEW")', '(if (gt (count_eq tokens needle) threshold) "MANY" "FEW")'),
            make_input=lambda rng: {
                "tokens": [rng.choice(list("abcd")) for _ in range(rng.randint(3, 9))],
                "needle": rng.choice(list("abcd")),
                "threshold": rng.randint(0, 4),
            },
        ),
        "length_and_mod_code": FamilySpec(
            name="length_and_mod_code",
            schema="text: str; threshold: int; modulus: int; target: int",
            program='(if (and (gt (len text) threshold) (eq (mod (len text) modulus) target)) "LEN_OK" "MISS")',
            wrong_programs=('(if (gt (len text) threshold) "LEN_OK" "MISS")', '(if (eq (mod (len text) modulus) target) "LEN_OK" "MISS")'),
            make_input=lambda rng: {
                "text": _word(rng, 2, 18),
                "threshold": rng.randint(3, 12),
                "modulus": (modulus := rng.randint(2, 7)),
                "target": rng.randint(0, modulus - 1),
            },
        ),
        "sum_and_scalar_code": FamilySpec(
            name="sum_and_scalar_code",
            schema="values: list[int]; threshold: int; n: int",
            program='(if (and (gt (sum values) threshold) (gt n 0)) "PASS" "FAIL")',
            wrong_programs=('(if (gt (sum values) threshold) "PASS" "FAIL")', '(if (gt n 0) "PASS" "FAIL")'),
            make_input=lambda rng: {
                "values": _values(rng),
                "threshold": rng.randint(-5, 25),
                "n": rng.randint(-8, 8),
            },
        ),
        "modulo_sum_label": FamilySpec(
            name="modulo_sum_label",
            schema="values: list[int]; modulus: int",
            program='(format "M{}" (mod (sum values) modulus))',
            wrong_programs=('(format "M{}" (sum values))', '(format "M{}" (mod (len values) modulus))'),
            make_input=lambda rng: {"values": _values(rng), "modulus": rng.randint(2, 9)},
        ),
        "length_contains_code": FamilySpec(
            name="length_contains_code",
            schema="text: str; needle: str; threshold: int",
            program='(if (and (contains text needle) (gt (len text) threshold)) "MATCH_LONG" "MISS")',
            wrong_programs=('(if (contains text needle) "MATCH_LONG" "MISS")', '(if (gt (len text) threshold) "MATCH_LONG" "MISS")'),
            make_input=lambda rng: {
                **_text_with_needle(rng),
                "threshold": rng.randint(4, 12),
            },
        ),
        "tuple_branch_label": FamilySpec(
            name="tuple_branch_label",
            schema="item: list[int]; index: int; threshold: int; high_label: str; low_label: str",
            program='(if (gt (tuple_get item index) threshold) high_label low_label)',
            wrong_programs=('(if (gt (sum item) threshold) high_label low_label)', '(format "T{}" (tuple_get item index))'),
            make_input=lambda rng: {
                "item": [rng.randint(-8, 20), rng.randint(-8, 20), rng.randint(-8, 20)],
                "index": rng.randint(0, 2),
                "threshold": rng.randint(-4, 14),
                "high_label": rng.choice(["TAKE", "RIGHT", "HOT"]),
                "low_label": rng.choice(["SKIP", "LEFT", "COLD"]),
            },
        ),
    }


TRAIN_FAMILIES = [
    "sum_label",
    "mod_scalar_label",
    "length_label",
    "contains_code",
    "tuple_get_label",
    "scalar_branch_label",
    "sum_threshold_label",
    "length_mod_label",
    "tuple_sum_label",
    "contains_count_label",
    "sorted_first_label",
    "sum_add_label",
    "contains_and_count_code",
    "length_and_mod_code",
    "sum_and_scalar_code",
]

HOLDOUT_FAMILIES = [
    "modulo_sum_label",
    "length_contains_code",
    "tuple_branch_label",
]


def make_cases(spec: FamilySpec, rng: random.Random, count: int) -> list[dict[str, Any]]:
    cases = []
    seen_inputs = set()
    attempts = 0
    while len(cases) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError(f"could not create distinct cases for {spec.name}")
        env = spec.make_input(rng)
        key = json.dumps(env, sort_keys=True)
        if key in seen_inputs:
            continue
        seen_inputs.add(key)
        cases.append({"input": env, "expected": execute(spec.program, env)})
    return cases


def wrong_outputs(wrong_program: str, cases: list[dict[str, Any]]) -> list[Any]:
    outputs = []
    for case in cases:
        try:
            outputs.append(execute(wrong_program, case["input"]))
        except Exception as exc:
            outputs.append(f"<error: {exc}>")
    return outputs


def make_record(
    *,
    rng: random.Random,
    split: str,
    family_name: str,
    index: int,
    visible_count: int,
    hidden_count: int,
) -> dict[str, Any]:
    spec = make_family_specs()[family_name]
    wrong_program = rng.choice(spec.wrong_programs)
    visible = make_cases(spec, rng, visible_count)
    hidden = make_cases(spec, rng, hidden_count)
    got = wrong_outputs(wrong_program, visible)
    visible_with_got = [
        {**case, "got": got_value}
        for case, got_value in zip(visible, got, strict=True)
    ]
    return {
        "id": f"{split}_{family_name}_{index:03d}",
        "split": split,
        "family": family_name,
        "schema": spec.schema,
        "wrong_program": wrong_program,
        "target_program": spec.program,
        "visible": visible_with_got,
        "hidden": hidden,
    }


def build_dataset(
    *,
    seed: int,
    train_records: int,
    iid_eval_records: int,
    holdout_records_per_family: int,
    visible_cases: int,
    hidden_cases: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    data: dict[str, list[dict[str, Any]]] = {"train": [], "iid": [], "holdout": []}
    for i in range(train_records):
        family = TRAIN_FAMILIES[i % len(TRAIN_FAMILIES)]
        data["train"].append(
            make_record(
                rng=rng,
                split="train",
                family_name=family,
                index=i,
                visible_count=visible_cases,
                hidden_count=hidden_cases,
            )
        )
    for i in range(iid_eval_records):
        family = TRAIN_FAMILIES[(i * 5 + 3) % len(TRAIN_FAMILIES)]
        data["iid"].append(
            make_record(
                rng=rng,
                split="iid",
                family_name=family,
                index=i,
                visible_count=visible_cases,
                hidden_count=hidden_cases,
            )
        )
    for family in HOLDOUT_FAMILIES:
        for i in range(holdout_records_per_family):
            data["holdout"].append(
                make_record(
                    rng=rng,
                    split="holdout",
                    family_name=family,
                    index=i,
                    visible_count=visible_cases,
                    hidden_count=hidden_cases,
                )
            )
    return data


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
