from __future__ import annotations

import json
import random
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
    static_distractors: tuple[str, ...]
    make_input: Callable[[random.Random], dict[str, Any]]


def _values(rng: random.Random, min_len: int = 3, max_len: int = 7) -> list[int]:
    return [rng.randint(-8, 18) for _ in range(rng.randint(min_len, max_len))]


def _word(rng: random.Random, min_len: int = 3, max_len: int = 12) -> str:
    alphabet = "abcde"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(min_len, max_len)))


def _text_with_needle(rng: random.Random) -> dict[str, Any]:
    needle = rng.choice(["aa", "bb", "cc", "de", "ed"])
    text = _word(rng, 4, 12)
    if rng.random() < 0.55:
        pos = rng.randint(0, len(text))
        text = text[:pos] + needle + text[pos:]
    return {"text": text, "needle": needle}


def _token_case(rng: random.Random) -> dict[str, Any]:
    return {
        "tokens": [rng.choice(list("abcd")) for _ in range(rng.randint(3, 9))],
        "needle": rng.choice(list("abcd")),
    }


def make_specs() -> dict[str, FamilySpec]:
    return {
        "sum_label": FamilySpec(
            "sum_label",
            "values: list[int]",
            '(format "S{}" (sum values))',
            ('(format "S{}" (len values))', '(format "S{}" (first values))'),
            ('(format "S{}" (len values))', '(format "S{}" (first values))', '(format "S{}" (last values))'),
            lambda rng: {"values": _values(rng)},
        ),
        "mod_scalar_label": FamilySpec(
            "mod_scalar_label",
            "n: int; modulus: int",
            '(format "R{}" (mod n modulus))',
            ('(format "R{}" n)', '(format "R{}" modulus)'),
            ('(format "R{}" n)', '(format "R{}" modulus)', '(format "R{}" (add n modulus))'),
            lambda rng: {"n": rng.randint(-50, 80), "modulus": rng.randint(2, 9)},
        ),
        "length_label": FamilySpec(
            "length_label",
            "text: str",
            '(format "L{}" (len text))',
            ('(format "L{}" text)', '(format "L{}" 0)'),
            ('(format "L{}" text)', '(format "L{}" 0)', '(format "L{}" (first text))'),
            lambda rng: {"text": _word(rng, 2, 16)},
        ),
        "contains_code": FamilySpec(
            "contains_code",
            "text: str; needle: str",
            '(if (contains text needle) "HAS" "MISS")',
            ('(if (contains needle text) "HAS" "MISS")', '"MISS"'),
            ('(if (contains needle text) "HAS" "MISS")', '"MISS"', '"HAS"'),
            _text_with_needle,
        ),
        "tuple_get_label": FamilySpec(
            "tuple_get_label",
            "item: list[int]; index: int",
            '(format "T{}" (tuple_get item index))',
            ('(format "T{}" (sum item))', '(format "T{}" index)'),
            ('(format "T{}" (sum item))', '(format "T{}" index)', '(format "T{}" (first item))'),
            lambda rng: {"item": [rng.randint(-6, 12), rng.randint(-6, 12), rng.randint(-6, 12)], "index": rng.randint(0, 2)},
        ),
        "scalar_branch_label": FamilySpec(
            "scalar_branch_label",
            "score: int; threshold: int; high_label: str; low_label: str",
            "(if (gt score threshold) high_label low_label)",
            ("(if (lt score threshold) high_label low_label)", "low_label"),
            ("(if (lt score threshold) high_label low_label)", "low_label", "high_label"),
            lambda rng: {
                "score": rng.randint(-15, 30),
                "threshold": rng.randint(-5, 18),
                "high_label": rng.choice(["HIGH", "UP", "YES"]),
                "low_label": rng.choice(["LOW", "DOWN", "NO"]),
            },
        ),
        "sum_threshold_label": FamilySpec(
            "sum_threshold_label",
            "values: list[int]; threshold: int; high_label: str; low_label: str",
            "(if (gt (sum values) threshold) high_label low_label)",
            ("(if (gt (len values) threshold) high_label low_label)", "low_label"),
            ("(if (gt (len values) threshold) high_label low_label)", "low_label", "high_label"),
            lambda rng: {
                "values": _values(rng),
                "threshold": rng.randint(-8, 28),
                "high_label": rng.choice(["BIG", "ABOVE", "PASS"]),
                "low_label": rng.choice(["SMALL", "BELOW", "FAIL"]),
            },
        ),
        "length_mod_label": FamilySpec(
            "length_mod_label",
            "text: str; modulus: int",
            '(format "LM{}" (mod (len text) modulus))',
            ('(format "LM{}" (len text))', '(format "LM{}" modulus)'),
            ('(format "LM{}" (len text))', '(format "LM{}" modulus)', '(format "LM{}" (mod (len text) 2))'),
            lambda rng: {"text": _word(rng, 2, 18), "modulus": rng.randint(2, 7)},
        ),
        "tuple_sum_label": FamilySpec(
            "tuple_sum_label",
            "item: list[int]",
            '(format "TS{}" (sum item))',
            ('(format "TS{}" (first item))', '(format "TS{}" (len item))'),
            ('(format "TS{}" (first item))', '(format "TS{}" (last item))', '(format "TS{}" (len item))'),
            lambda rng: {"item": [rng.randint(-5, 10), rng.randint(-5, 10), rng.randint(-5, 10)]},
        ),
        "contains_count_label": FamilySpec(
            "contains_count_label",
            "tokens: list[str]; needle: str",
            '(format "C{}" (count_eq tokens needle))',
            ('(if (contains tokens needle) "C1" "C0")', '(format "C{}" (len tokens))'),
            ('(if (contains tokens needle) "C1" "C0")', '(format "C{}" (len tokens))', '(format "C{}" 0)'),
            lambda rng: {**_token_case(rng)},
        ),
        "sorted_first_label": FamilySpec(
            "sorted_first_label",
            "values: list[int]",
            '(format "F{}" (first (sort values)))',
            ('(format "F{}" (first values))', '(format "F{}" (last values))'),
            ('(format "F{}" (first values))', '(format "F{}" (last values))', '(format "F{}" (last (sort values)))'),
            lambda rng: {"values": _values(rng)},
        ),
        "sum_add_label": FamilySpec(
            "sum_add_label",
            "values: list[int]; offset: int",
            '(format "A{}" (add (sum values) offset))',
            ('(format "A{}" (sum values))', '(format "A{}" offset)'),
            ('(format "A{}" (sum values))', '(format "A{}" offset)', '(format "A{}" (sub (sum values) offset))'),
            lambda rng: {"values": _values(rng), "offset": rng.randint(-10, 10)},
        ),
        "contains_and_count_code": FamilySpec(
            "contains_and_count_code",
            "tokens: list[str]; needle: str; threshold: int",
            '(if (and (contains tokens needle) (gt (count_eq tokens needle) threshold)) "MANY" "FEW")',
            ('(if (contains tokens needle) "MANY" "FEW")', '(if (gt (count_eq tokens needle) threshold) "MANY" "FEW")'),
            (
                '(if (contains tokens needle) "MANY" "FEW")',
                '(if (gt (count_eq tokens needle) threshold) "MANY" "FEW")',
                '(if (or (contains tokens needle) (gt (count_eq tokens needle) threshold)) "MANY" "FEW")',
            ),
            lambda rng: {**_token_case(rng), "threshold": rng.randint(0, 4)},
        ),
        "length_and_mod_code": FamilySpec(
            "length_and_mod_code",
            "text: str; threshold: int; modulus: int; target: int",
            '(if (and (gt (len text) threshold) (eq (mod (len text) modulus) target)) "LEN_OK" "MISS")',
            ('(if (gt (len text) threshold) "LEN_OK" "MISS")', '(if (eq (mod (len text) modulus) target) "LEN_OK" "MISS")'),
            (
                '(if (gt (len text) threshold) "LEN_OK" "MISS")',
                '(if (eq (mod (len text) modulus) target) "LEN_OK" "MISS")',
                '(if (or (gt (len text) threshold) (eq (mod (len text) modulus) target)) "LEN_OK" "MISS")',
            ),
            lambda rng: {
                "text": _word(rng, 2, 18),
                "threshold": rng.randint(3, 12),
                "modulus": (modulus := rng.randint(2, 7)),
                "target": rng.randint(0, modulus - 1),
            },
        ),
        "sum_and_scalar_code": FamilySpec(
            "sum_and_scalar_code",
            "values: list[int]; threshold: int; n: int",
            '(if (and (gt (sum values) threshold) (gt n 0)) "PASS" "FAIL")',
            ('(if (gt (sum values) threshold) "PASS" "FAIL")', '(if (gt n 0) "PASS" "FAIL")'),
            (
                '(if (gt (sum values) threshold) "PASS" "FAIL")',
                '(if (gt n 0) "PASS" "FAIL")',
                '(if (or (gt (sum values) threshold) (gt n 0)) "PASS" "FAIL")',
            ),
            lambda rng: {"values": _values(rng), "threshold": rng.randint(-5, 25), "n": rng.randint(-8, 8)},
        ),
        "modulo_sum_label": FamilySpec(
            "modulo_sum_label",
            "values: list[int]; modulus: int",
            '(format "M{}" (mod (sum values) modulus))',
            ('(format "M{}" (sum values))', '(format "M{}" (mod (len values) modulus))'),
            (
                '(format "M{}" (sum values))',
                '(format "M{}" (mod (len values) modulus))',
                '(format "M{}" (mod (first values) modulus))',
                '(format "M{}" modulus)',
            ),
            lambda rng: {"values": _values(rng), "modulus": rng.randint(2, 9)},
        ),
        "length_contains_code": FamilySpec(
            "length_contains_code",
            "text: str; needle: str; threshold: int",
            '(if (and (contains text needle) (gt (len text) threshold)) "MATCH_LONG" "MISS")',
            ('(if (contains text needle) "MATCH_LONG" "MISS")', '(if (gt (len text) threshold) "MATCH_LONG" "MISS")'),
            (
                '(if (contains text needle) "MATCH_LONG" "MISS")',
                '(if (gt (len text) threshold) "MATCH_LONG" "MISS")',
                '(if (gt (count_eq text needle) threshold) "MATCH_LONG" "MISS")',
                '(if (and (contains text needle) (gt (len needle) threshold)) "MATCH_LONG" "MISS")',
                '(if (or (contains text needle) (gt (len text) threshold)) "MATCH_LONG" "MISS")',
            ),
            lambda rng: {**_text_with_needle(rng), "threshold": rng.randint(4, 12)},
        ),
        "tuple_branch_label": FamilySpec(
            "tuple_branch_label",
            "item: list[int]; index: int; threshold: int; high_label: str; low_label: str",
            "(if (gt (tuple_get item index) threshold) high_label low_label)",
            ('(if (gt (sum item) threshold) high_label low_label)', '(format "T{}" (tuple_get item index))'),
            (
                '(if (gt (sum item) threshold) high_label low_label)',
                '(if (gt (first item) threshold) high_label low_label)',
                '(if (gt index threshold) high_label low_label)',
                '(format "T{}" (tuple_get item index))',
            ),
            lambda rng: {
                "item": [rng.randint(-8, 20), rng.randint(-8, 20), rng.randint(-8, 20)],
                "index": rng.randint(0, 2),
                "threshold": rng.randint(-4, 14),
                "high_label": rng.choice(["TAKE", "RIGHT", "HOT"]),
                "low_label": rng.choice(["SKIP", "LEFT", "COLD"]),
            },
        ),
    }


BASE_FAMILIES = [
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

CHALLENGE_FAMILIES = ["modulo_sum_label", "length_contains_code", "tuple_branch_label"]

BRIDGE_ALLOCATION = {
    "length_contains_code": 40,
    "modulo_sum_label": 10,
    "tuple_branch_label": 10,
}


def safe_execute(program: str, env: dict[str, Any]) -> Any:
    try:
        return execute(program, env)
    except Exception as exc:
        return f"<error:{exc}>"


def make_case(spec: FamilySpec, rng: random.Random) -> dict[str, Any]:
    env = spec.make_input(rng)
    return {"input": env, "expected": execute(spec.program, env)}


def random_cases(spec: FamilySpec, rng: random.Random, count: int) -> list[dict[str, Any]]:
    cases = []
    seen = set()
    attempts = 0
    while len(cases) < count:
        attempts += 1
        if attempts > count * 300:
            raise RuntimeError(f"could not create cases for {spec.name}")
        case = make_case(spec, rng)
        key = json.dumps(case["input"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        cases.append(case)
    return cases


def attach_got(wrong_program: str, visible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**case, "got": safe_execute(wrong_program, case["input"])} for case in visible]


def select_counterexample_cases(
    spec: FamilySpec,
    rng: random.Random,
    wrong_programs: list[str],
    *,
    count: int,
    pool_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = random_cases(spec, rng, pool_size)
    remaining = {
        program
        for program in wrong_programs
        if program and program.strip() != spec.program.strip()
    }
    selected: list[dict[str, Any]] = []
    seen = set()
    eliminated_total: set[str] = set()
    while len(selected) < count and remaining:
        scored = []
        for case in pool:
            key = json.dumps(case["input"], sort_keys=True)
            if key in seen:
                continue
            eliminated = {
                program
                for program in remaining
                if safe_execute(program, case["input"]) != case["expected"]
            }
            scored.append((len(eliminated), rng.random(), case, eliminated))
        if not scored:
            break
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _, _, case, eliminated = scored[0]
        selected.append(case)
        seen.add(json.dumps(case["input"], sort_keys=True))
        eliminated_total |= eliminated
        remaining -= eliminated
    if len(selected) < count:
        for case in pool:
            if len(selected) >= count:
                break
            key = json.dumps(case["input"], sort_keys=True)
            if key not in seen:
                selected.append(case)
                seen.add(key)
    stats = {
        "wrong_program_count": len({program for program in wrong_programs if program}),
        "eliminated_wrong_programs": len(eliminated_total),
        "remaining_wrong_programs": len(remaining),
        "pool_size": pool_size,
    }
    return selected, stats


def make_record(
    *,
    rng: random.Random,
    split: str,
    family_name: str,
    index: int,
    trace_strategy: str,
    visible_count: int,
    hidden_count: int,
    pool_size: int,
    wrong_program: str | None = None,
    selector_programs: list[str] | None = None,
    case_pool_count: int = 0,
) -> dict[str, Any]:
    spec = make_specs()[family_name]
    chosen_wrong = wrong_program or rng.choice(spec.wrong_programs)
    selection_stats: dict[str, Any] | None = None
    if trace_strategy in {"static_bridge", "static_counterexample"}:
        visible, selection_stats = select_counterexample_cases(
            spec,
            rng,
            list(spec.static_distractors),
            count=visible_count,
            pool_size=pool_size,
        )
    elif trace_strategy == "model_mined":
        visible, selection_stats = select_counterexample_cases(
            spec,
            rng,
            selector_programs or [chosen_wrong],
            count=visible_count,
            pool_size=pool_size,
        )
    elif trace_strategy in {"seed_random", "mining_prompt", "eval_random"}:
        visible = random_cases(spec, rng, visible_count)
    else:
        raise ValueError(trace_strategy)
    hidden = random_cases(spec, rng, hidden_count)
    record = {
        "id": f"{split}_{trace_strategy}_{family_name}_{index:03d}",
        "split": split,
        "trace_strategy": trace_strategy,
        "family": family_name,
        "schema": spec.schema,
        "wrong_program": chosen_wrong,
        "target_program": spec.program,
        "visible": attach_got(chosen_wrong, visible),
        "hidden": hidden,
        "static_distractors": list(spec.static_distractors),
    }
    if selector_programs is not None:
        record["selector_programs"] = selector_programs
    if selection_stats is not None:
        record["selection_stats"] = selection_stats
    if case_pool_count:
        record["case_pool"] = random_cases(spec, rng, case_pool_count)
    return record


def base_records(
    *,
    rng: random.Random,
    count: int,
    split: str,
    visible_count: int,
    hidden_count: int,
    pool_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for i in range(count):
        family = BASE_FAMILIES[i % len(BASE_FAMILIES)]
        rows.append(
            make_record(
                rng=rng,
                split=split,
                family_name=family,
                index=i,
                trace_strategy="seed_random",
                visible_count=visible_count,
                hidden_count=hidden_count,
                pool_size=pool_size,
            )
        )
    return rows


def challenge_records(
    *,
    rng: random.Random,
    split: str,
    records_per_family: int,
    trace_strategy: str,
    visible_count: int,
    hidden_count: int,
    pool_size: int,
    case_pool_count: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    for family in CHALLENGE_FAMILIES:
        for i in range(records_per_family):
            rows.append(
                make_record(
                    rng=rng,
                    split=split,
                    family_name=family,
                    index=i,
                    trace_strategy=trace_strategy,
                    visible_count=visible_count,
                    hidden_count=hidden_count,
                    pool_size=pool_size,
                    case_pool_count=case_pool_count,
                )
            )
    return rows


def static_bridge_records(
    *,
    rng: random.Random,
    visible_count: int,
    hidden_count: int,
    pool_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for family, count in BRIDGE_ALLOCATION.items():
        for i in range(count):
            rows.append(
                make_record(
                    rng=rng,
                    split="train_bridge",
                    family_name=family,
                    index=i,
                    trace_strategy="static_bridge",
                    visible_count=visible_count,
                    hidden_count=hidden_count,
                    pool_size=pool_size,
                )
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

