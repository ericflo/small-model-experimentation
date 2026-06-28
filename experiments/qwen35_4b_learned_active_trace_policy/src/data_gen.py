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
    return [rng.randint(-9, 21) for _ in range(rng.randint(min_len, max_len))]


def _triple(rng: random.Random) -> list[int]:
    return [rng.randint(-9, 21), rng.randint(-9, 21), rng.randint(-9, 21)]


def _word(rng: random.Random, min_len: int = 3, max_len: int = 14) -> str:
    alphabet = "abcde"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(min_len, max_len)))


def _text_with_needle(rng: random.Random) -> dict[str, Any]:
    needle = rng.choice(["aa", "bb", "cc", "de", "ed"])
    text = _word(rng, 4, 14)
    if rng.random() < 0.55:
        pos = rng.randint(0, len(text))
        text = text[:pos] + needle + text[pos:]
    return {"text": text, "needle": needle}


def _text_maybe_without_needle(rng: random.Random) -> dict[str, Any]:
    needle = rng.choice(["aa", "bb", "cc", "de", "ed"])
    text = _word(rng, 5, 16)
    if rng.random() < 0.45:
        pos = rng.randint(0, len(text))
        text = text[:pos] + needle + text[pos:]
    else:
        while needle in text:
            text = _word(rng, 5, 16)
    return {"text": text, "needle": needle}


def _token_case(rng: random.Random, min_len: int = 3, max_len: int = 10) -> dict[str, Any]:
    return {
        "tokens": [rng.choice(list("abcd")) for _ in range(rng.randint(min_len, max_len))],
        "needle": rng.choice(list("abcd")),
    }


def _hard_values(rng: random.Random, min_len: int = 5, max_len: int = 12) -> list[int]:
    values = [rng.randint(-40, 65) for _ in range(rng.randint(min_len, max_len))]
    if len(values) >= 4 and rng.random() < 0.7:
        values[rng.randrange(len(values))] = values[rng.randrange(len(values))]
    return values


def _fixed_word(rng: random.Random, length: int, alphabet: str = "abcde") -> str:
    return "".join(rng.choice(alphabet) for _ in range(max(0, length)))


def _text_with_final_length(rng: random.Random, final_len: int, *, present: bool) -> dict[str, Any]:
    needle = rng.choice(["aa", "bb", "cc", "de", "ed"])
    final_len = max(final_len, len(needle) if present else 1)
    if present:
        base = _fixed_word(rng, final_len - len(needle))
        pos = rng.randint(0, len(base))
        text = base[:pos] + needle + base[pos:]
    else:
        text = _fixed_word(rng, final_len)
        attempts = 0
        while needle in text and attempts < 200:
            text = _fixed_word(rng, final_len)
            attempts += 1
    return {"text": text, "needle": needle}


def _token_case_controlled(rng: random.Random, length: int, count: int | None = None) -> dict[str, Any]:
    alphabet = list("abcd")
    needle = rng.choice(alphabet)
    length = max(1, length)
    if count is None:
        count = rng.randint(0, length)
    count = max(0, min(count, length))
    other = [token for token in alphabet if token != needle]
    tokens = [needle] * count + [rng.choice(other) for _ in range(length - count)]
    rng.shuffle(tokens)
    return {"tokens": tokens, "needle": needle}


def _near(value: int, rng: random.Random, radius: int = 2) -> int:
    return value + rng.choice([delta for delta in range(-radius, radius + 1)])


def hard_input_for(family: str, rng: random.Random) -> dict[str, Any] | None:
    if family == "modulo_sum_label":
        values = _hard_values(rng, 6, 13)
        return {"values": values, "modulus": rng.randint(2, 11)}
    if family == "length_contains_code":
        threshold = rng.randint(6, 24)
        final_len = max(2, _near(threshold, rng, 3))
        return {**_text_with_final_length(rng, final_len, present=rng.random() < 0.65), "threshold": threshold}
    if family == "tuple_branch_label":
        item = [rng.randint(-20, 35) for _ in range(3)]
        index = rng.randint(0, 2)
        return {
            "item": item,
            "index": index,
            "threshold": _near(item[index], rng, 2),
            "high_label": rng.choice(["TAKE", "RIGHT", "HOT"]),
            "low_label": rng.choice(["SKIP", "LEFT", "COLD"]),
        }
    if family == "sum_offset_mod_label":
        values = _hard_values(rng, 6, 12)
        return {"values": values, "offset": rng.randint(-30, 30), "modulus": rng.randint(2, 11)}
    if family == "length_mod_contains_code":
        modulus = rng.randint(2, 9)
        target = rng.randint(0, modulus - 1)
        candidates = [length for length in range(4, 32) if length % modulus == target]
        if rng.random() < 0.55 and candidates:
            final_len = rng.choice(candidates)
        else:
            final_len = rng.randint(4, 31)
        return {**_text_with_final_length(rng, final_len, present=rng.random() < 0.7), "modulus": modulus, "target": target}
    if family == "sum_length_branch_label":
        values = _hard_values(rng, 5, 11)
        text_len = rng.randint(4, 24)
        return {
            "values": values,
            "text": _fixed_word(rng, text_len),
            "threshold": _near(sum(values), rng, 5),
            "min_len": _near(text_len, rng, 3),
            "high_label": rng.choice(["READY", "ALLOW", "OPEN"]),
            "low_label": rng.choice(["WAIT", "DENY", "SHUT"]),
        }
    if family == "sorted_index_offset_label":
        values = _hard_values(rng, 6, 12)
        return {"values": values, "index": rng.randint(0, min(4, len(values) - 1)), "offset": rng.randint(-25, 25)}
    if family == "contains_count_length_code":
        length = rng.randint(5, 16)
        count = rng.randint(0, length)
        return {
            **_token_case_controlled(rng, length, count),
            "threshold": max(0, _near(count, rng, 2)),
            "min_len": max(1, _near(length, rng, 3)),
        }
    if family == "tuple_sum_gate_label":
        item = [rng.randint(-20, 35) for _ in range(3)]
        index = rng.randint(0, 2)
        return {
            "item": item,
            "index": index,
            "threshold": _near(item[index], rng, 2),
            "sum_threshold": _near(sum(item), rng, 5),
            "high_label": rng.choice(["KEEP", "RAISE", "SEND"]),
            "low_label": rng.choice(["DROP", "LOWER", "HOLD"]),
        }
    if family == "not_contains_length_code":
        threshold = rng.randint(6, 24)
        final_len = max(2, _near(threshold, rng, 3))
        return {**_text_with_final_length(rng, final_len, present=rng.random() < 0.45), "threshold": threshold}
    return None


def make_specs() -> dict[str, FamilySpec]:
    specs = {
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
            lambda rng: {"item": _triple(rng), "index": rng.randint(0, 2)},
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
            lambda rng: {"item": _triple(rng)},
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
    }

    specs.update(
        {
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
                    "item": _triple(rng),
                    "index": rng.randint(0, 2),
                    "threshold": rng.randint(-4, 14),
                    "high_label": rng.choice(["TAKE", "RIGHT", "HOT"]),
                    "low_label": rng.choice(["SKIP", "LEFT", "COLD"]),
                },
            ),
            "sum_offset_mod_label": FamilySpec(
                "sum_offset_mod_label",
                "values: list[int]; offset: int; modulus: int",
                '(format "OM{}" (mod (add (sum values) offset) modulus))',
                ('(format "OM{}" (mod (sum values) modulus))', '(format "OM{}" (mod offset modulus))'),
                (
                    '(format "OM{}" (mod (sum values) modulus))',
                    '(format "OM{}" (mod offset modulus))',
                    '(format "OM{}" (mod (sub (sum values) offset) modulus))',
                    '(format "OM{}" (sum values))',
                ),
                lambda rng: {"values": _values(rng), "offset": rng.randint(-12, 12), "modulus": rng.randint(2, 9)},
            ),
            "length_mod_contains_code": FamilySpec(
                "length_mod_contains_code",
                "text: str; needle: str; modulus: int; target: int",
                '(if (and (contains text needle) (eq (mod (len text) modulus) target)) "HIT_MOD" "MISS")',
                ('(if (contains text needle) "HIT_MOD" "MISS")', '(if (eq (mod (len text) modulus) target) "HIT_MOD" "MISS")'),
                (
                    '(if (contains text needle) "HIT_MOD" "MISS")',
                    '(if (eq (mod (len text) modulus) target) "HIT_MOD" "MISS")',
                    '(if (and (contains text needle) (eq (mod (count_eq text needle) modulus) target)) "HIT_MOD" "MISS")',
                    '(if (or (contains text needle) (eq (mod (len text) modulus) target)) "HIT_MOD" "MISS")',
                ),
                lambda rng: {
                    **_text_with_needle(rng),
                    "modulus": (modulus := rng.randint(2, 7)),
                    "target": rng.randint(0, modulus - 1),
                },
            ),
            "sum_length_branch_label": FamilySpec(
                "sum_length_branch_label",
                "values: list[int]; text: str; threshold: int; min_len: int; high_label: str; low_label: str",
                "(if (and (gt (sum values) threshold) (gt (len text) min_len)) high_label low_label)",
                ("(if (gt (sum values) threshold) high_label low_label)", "(if (gt (len text) min_len) high_label low_label)"),
                (
                    "(if (gt (sum values) threshold) high_label low_label)",
                    "(if (gt (len text) min_len) high_label low_label)",
                    "(if (and (gt (len values) threshold) (gt (len text) min_len)) high_label low_label)",
                    "(if (or (gt (sum values) threshold) (gt (len text) min_len)) high_label low_label)",
                ),
                lambda rng: {
                    "values": _values(rng),
                    "text": _word(rng, 3, 18),
                    "threshold": rng.randint(-5, 28),
                    "min_len": rng.randint(4, 12),
                    "high_label": rng.choice(["READY", "ALLOW", "OPEN"]),
                    "low_label": rng.choice(["WAIT", "DENY", "SHUT"]),
                },
            ),
            "sorted_index_offset_label": FamilySpec(
                "sorted_index_offset_label",
                "values: list[int]; index: int; offset: int",
                '(format "SI{}" (add (tuple_get (sort values) index) offset))',
                ('(format "SI{}" (add (tuple_get values index) offset))', '(format "SI{}" (tuple_get (sort values) index))'),
                (
                    '(format "SI{}" (add (tuple_get values index) offset))',
                    '(format "SI{}" (tuple_get (sort values) index))',
                    '(format "SI{}" (add (first (sort values)) offset))',
                    '(format "SI{}" (add (last (sort values)) offset))',
                ),
                lambda rng: {"values": _values(rng, 4, 8), "index": rng.randint(0, 2), "offset": rng.randint(-8, 8)},
            ),
            "contains_count_length_code": FamilySpec(
                "contains_count_length_code",
                "tokens: list[str]; needle: str; threshold: int; min_len: int",
                '(if (and (contains tokens needle) (gt (count_eq tokens needle) threshold) (gt (len tokens) min_len)) "MANY_LONG" "MISS")',
                ('(if (and (contains tokens needle) (gt (count_eq tokens needle) threshold)) "MANY_LONG" "MISS")', '(if (gt (len tokens) min_len) "MANY_LONG" "MISS")'),
                (
                    '(if (and (contains tokens needle) (gt (count_eq tokens needle) threshold)) "MANY_LONG" "MISS")',
                    '(if (gt (len tokens) min_len) "MANY_LONG" "MISS")',
                    '(if (and (contains tokens needle) (gt (len tokens) threshold)) "MANY_LONG" "MISS")',
                    '(if (or (contains tokens needle) (gt (count_eq tokens needle) threshold) (gt (len tokens) min_len)) "MANY_LONG" "MISS")',
                ),
                lambda rng: {**_token_case(rng, 3, 11), "threshold": rng.randint(0, 4), "min_len": rng.randint(3, 8)},
            ),
            "tuple_sum_gate_label": FamilySpec(
                "tuple_sum_gate_label",
                "item: list[int]; index: int; threshold: int; sum_threshold: int; high_label: str; low_label: str",
                "(if (and (gt (tuple_get item index) threshold) (gt (sum item) sum_threshold)) high_label low_label)",
                ("(if (gt (tuple_get item index) threshold) high_label low_label)", "(if (gt (sum item) sum_threshold) high_label low_label)"),
                (
                    "(if (gt (tuple_get item index) threshold) high_label low_label)",
                    "(if (gt (sum item) sum_threshold) high_label low_label)",
                    "(if (and (gt (sum item) threshold) (gt (tuple_get item index) sum_threshold)) high_label low_label)",
                    "(if (or (gt (tuple_get item index) threshold) (gt (sum item) sum_threshold)) high_label low_label)",
                ),
                lambda rng: {
                    "item": _triple(rng),
                    "index": rng.randint(0, 2),
                    "threshold": rng.randint(-5, 16),
                    "sum_threshold": rng.randint(-8, 28),
                    "high_label": rng.choice(["KEEP", "RAISE", "SEND"]),
                    "low_label": rng.choice(["DROP", "LOWER", "HOLD"]),
                },
            ),
            "not_contains_length_code": FamilySpec(
                "not_contains_length_code",
                "text: str; needle: str; threshold: int",
                '(if (and (not (contains text needle)) (gt (len text) threshold)) "ABSENT_LONG" "OTHER")',
                ('(if (contains text needle) "ABSENT_LONG" "OTHER")', '(if (gt (len text) threshold) "ABSENT_LONG" "OTHER")'),
                (
                    '(if (contains text needle) "ABSENT_LONG" "OTHER")',
                    '(if (gt (len text) threshold) "ABSENT_LONG" "OTHER")',
                    '(if (and (not (contains text needle)) (gt (count_eq text needle) threshold)) "ABSENT_LONG" "OTHER")',
                    '(if (or (not (contains text needle)) (gt (len text) threshold)) "ABSENT_LONG" "OTHER")',
                ),
                lambda rng: {**_text_maybe_without_needle(rng), "threshold": rng.randint(4, 13)},
            ),
        }
    )
    specs.update(
        {
            "sum_length_mod_gate_label": FamilySpec(
                "sum_length_mod_gate_label",
                "values: list[int]; text: str; modulus: int; target: int; high_label: str; low_label: str",
                "(if (eq (mod (add (sum values) (len text)) modulus) target) high_label low_label)",
                (
                    "(if (eq (mod (sum values) modulus) target) high_label low_label)",
                    "(if (eq (mod (len text) modulus) target) high_label low_label)",
                ),
                (
                    "(if (eq (mod (sum values) modulus) target) high_label low_label)",
                    "(if (eq (mod (len text) modulus) target) high_label low_label)",
                    "(if (eq (mod (sub (sum values) (len text)) modulus) target) high_label low_label)",
                    "(if (gt (add (sum values) (len text)) target) high_label low_label)",
                ),
                lambda rng: {
                    "values": _hard_values(rng, 5, 12),
                    "text": _word(rng, 5, 24),
                    "modulus": (modulus := rng.randint(2, 11)),
                    "target": rng.randint(0, modulus - 1),
                    "high_label": rng.choice(["SYNC", "OPEN", "PASS"]),
                    "low_label": rng.choice(["DRIFT", "SHUT", "FAIL"]),
                },
            ),
            "sorted_index_sum_branch_label": FamilySpec(
                "sorted_index_sum_branch_label",
                "values: list[int]; index: int; threshold: int; high_label: str; low_label: str",
                "(if (gt (add (tuple_get (sort values) index) (sum values)) threshold) high_label low_label)",
                (
                    "(if (gt (add (tuple_get values index) (sum values)) threshold) high_label low_label)",
                    "(if (gt (tuple_get (sort values) index) threshold) high_label low_label)",
                ),
                (
                    "(if (gt (add (tuple_get values index) (sum values)) threshold) high_label low_label)",
                    "(if (gt (tuple_get (sort values) index) threshold) high_label low_label)",
                    "(if (gt (sum values) threshold) high_label low_label)",
                    "(if (gt (add (first (sort values)) (sum values)) threshold) high_label low_label)",
                ),
                lambda rng: {
                    "values": (values := _hard_values(rng, 6, 13)),
                    "index": rng.randint(0, min(5, len(values) - 1)),
                    "threshold": rng.randint(-45, 115),
                    "high_label": rng.choice(["RISE", "KEEP", "SEND"]),
                    "low_label": rng.choice(["FALL", "DROP", "HOLD"]),
                },
            ),
            "token_absent_length_code": FamilySpec(
                "token_absent_length_code",
                "tokens: list[str]; needle: str; min_len: int",
                '(if (and (not (contains tokens needle)) (gt (len tokens) min_len)) "ABSENT_LONG" "OTHER")',
                (
                    '(if (and (contains tokens needle) (gt (len tokens) min_len)) "ABSENT_LONG" "OTHER")',
                    '(if (gt (len tokens) min_len) "ABSENT_LONG" "OTHER")',
                ),
                (
                    '(if (and (contains tokens needle) (gt (len tokens) min_len)) "ABSENT_LONG" "OTHER")',
                    '(if (gt (len tokens) min_len) "ABSENT_LONG" "OTHER")',
                    '(if (not (contains tokens needle)) "ABSENT_LONG" "OTHER")',
                    '(if (or (not (contains tokens needle)) (gt (len tokens) min_len)) "ABSENT_LONG" "OTHER")',
                ),
                lambda rng: {
                    **_token_case_controlled(
                        rng,
                        length := rng.randint(6, 18),
                        0 if rng.random() < 0.5 else rng.randint(1, max(1, min(5, length))),
                    ),
                    "min_len": max(1, _near(length, rng, 3)),
                },
            ),
            "token_count_mod_length_code": FamilySpec(
                "token_count_mod_length_code",
                "tokens: list[str]; needle: str; modulus: int; target: int; min_len: int",
                '(if (and (contains tokens needle) (eq (mod (count_eq tokens needle) modulus) target) (gt (len tokens) min_len)) "COUNT_MOD_LONG" "MISS")',
                (
                    '(if (and (contains tokens needle) (eq (mod (count_eq tokens needle) modulus) target)) "COUNT_MOD_LONG" "MISS")',
                    '(if (and (contains tokens needle) (gt (len tokens) min_len)) "COUNT_MOD_LONG" "MISS")',
                ),
                (
                    '(if (and (contains tokens needle) (eq (mod (count_eq tokens needle) modulus) target)) "COUNT_MOD_LONG" "MISS")',
                    '(if (and (contains tokens needle) (gt (len tokens) min_len)) "COUNT_MOD_LONG" "MISS")',
                    '(if (and (contains tokens needle) (eq (mod (len tokens) modulus) target) (gt (len tokens) min_len)) "COUNT_MOD_LONG" "MISS")',
                    '(if (and (contains tokens needle) (gt (count_eq tokens needle) target) (gt (len tokens) min_len)) "COUNT_MOD_LONG" "MISS")',
                ),
                lambda rng: {
                    **_token_case_controlled(rng, length := rng.randint(7, 20), count := rng.randint(0, length)),
                    "modulus": (modulus := rng.randint(2, 8)),
                    "target": rng.randint(0, modulus - 1) if rng.random() < 0.5 else count % modulus,
                    "min_len": max(1, _near(length, rng, 3)),
                },
            ),
            "text_value_gate_label": FamilySpec(
                "text_value_gate_label",
                "text: str; needle: str; values: list[int]; min_len: int; threshold: int; high_label: str; low_label: str",
                "(if (and (contains text needle) (gt (len text) min_len) (gt (sum values) threshold)) high_label low_label)",
                (
                    "(if (and (contains text needle) (gt (len text) min_len)) high_label low_label)",
                    "(if (gt (sum values) threshold) high_label low_label)",
                ),
                (
                    "(if (and (contains text needle) (gt (len text) min_len)) high_label low_label)",
                    "(if (gt (sum values) threshold) high_label low_label)",
                    "(if (and (contains text needle) (gt (count_eq text needle) threshold)) high_label low_label)",
                    "(if (or (contains text needle) (gt (len text) min_len) (gt (sum values) threshold)) high_label low_label)",
                ),
                lambda rng: {
                    **_text_with_final_length(rng, text_len := rng.randint(6, 28), present=rng.random() < 0.65),
                    "values": (values := _hard_values(rng, 5, 12)),
                    "min_len": max(1, _near(text_len, rng, 3)),
                    "threshold": _near(sum(values), rng, 8),
                    "high_label": rng.choice(["TRIPLE", "ALLOW", "READY"]),
                    "low_label": rng.choice(["MISS", "DENY", "WAIT"]),
                },
            ),
            "tuple_value_mod_label": FamilySpec(
                "tuple_value_mod_label",
                "item: list[int]; index: int; values: list[int]; modulus: int",
                '(format "TV{}" (add (tuple_get item index) (mod (sum values) modulus)))',
                (
                    '(format "TV{}" (add (sum item) (mod (sum values) modulus)))',
                    '(format "TV{}" (add (tuple_get item index) (sum values)))',
                ),
                (
                    '(format "TV{}" (add (sum item) (mod (sum values) modulus)))',
                    '(format "TV{}" (add (tuple_get item index) (sum values)))',
                    '(format "TV{}" (mod (add (tuple_get item index) (sum values)) modulus))',
                    '(format "TV{}" (tuple_get item index))',
                ),
                lambda rng: {
                    "item": _triple(rng),
                    "index": rng.randint(0, 2),
                    "values": _hard_values(rng, 5, 12),
                    "modulus": rng.randint(2, 11),
                },
            ),
            "sorted_join_contains_code": FamilySpec(
                "sorted_join_contains_code",
                "tokens: list[str]; needle: str",
                '(if (contains (join "" (sort tokens)) needle) "JOIN_HAS" "JOIN_MISS")',
                (
                    '(if (contains (join "" tokens) needle) "JOIN_HAS" "JOIN_MISS")',
                    '(if (contains tokens needle) "JOIN_HAS" "JOIN_MISS")',
                ),
                (
                    '(if (contains (join "" tokens) needle) "JOIN_HAS" "JOIN_MISS")',
                    '(if (contains tokens needle) "JOIN_HAS" "JOIN_MISS")',
                    '(if (contains (join "" (sort tokens)) (first tokens)) "JOIN_HAS" "JOIN_MISS")',
                    '"JOIN_MISS"',
                ),
                lambda rng: {
                    "tokens": [rng.choice(list("abcd")) for _ in range(rng.randint(6, 16))],
                    "needle": rng.choice(["aa", "bb", "cc", "dd", "ab", "bc", "cd"]),
                },
            ),
            "text_absent_mod_code": FamilySpec(
                "text_absent_mod_code",
                "text: str; needle: str; modulus: int; target: int",
                '(if (and (not (contains text needle)) (eq (mod (len text) modulus) target)) "ABSENT_MOD" "OTHER")',
                (
                    '(if (and (contains text needle) (eq (mod (len text) modulus) target)) "ABSENT_MOD" "OTHER")',
                    '(if (eq (mod (len text) modulus) target) "ABSENT_MOD" "OTHER")',
                ),
                (
                    '(if (and (contains text needle) (eq (mod (len text) modulus) target)) "ABSENT_MOD" "OTHER")',
                    '(if (eq (mod (len text) modulus) target) "ABSENT_MOD" "OTHER")',
                    '(if (not (contains text needle)) "ABSENT_MOD" "OTHER")',
                    '(if (and (not (contains text needle)) (eq (mod (count_eq text needle) modulus) target)) "ABSENT_MOD" "OTHER")',
                ),
                lambda rng: {
                    **_text_with_final_length(
                        rng,
                        final_len := rng.randint(5, 31),
                        present=rng.random() < 0.45,
                    ),
                    "modulus": (modulus := rng.randint(2, 9)),
                    "target": rng.randint(0, modulus - 1) if rng.random() < 0.5 else final_len % modulus,
                },
            ),
            "sum_len_mod_label": FamilySpec(
                "sum_len_mod_label",
                "values: list[int]; text: str; modulus: int",
                '(format "SL{}" (mod (add (sum values) (len text)) modulus))',
                (
                    '(format "SL{}" (mod (sum values) modulus))',
                    '(format "SL{}" (mod (len text) modulus))',
                ),
                (
                    '(format "SL{}" (mod (sum values) modulus))',
                    '(format "SL{}" (mod (len text) modulus))',
                    '(format "SL{}" (add (mod (sum values) modulus) (mod (len text) modulus)))',
                    '(format "SL{}" (add (sum values) (len text)))',
                ),
                lambda rng: {"values": _hard_values(rng, 5, 13), "text": _word(rng, 5, 26), "modulus": rng.randint(2, 11)},
            ),
            "tuple_sum_mod_gate_label": FamilySpec(
                "tuple_sum_mod_gate_label",
                "item: list[int]; index: int; threshold: int; modulus: int; target: int; high_label: str; low_label: str",
                "(if (and (gt (tuple_get item index) threshold) (eq (mod (sum item) modulus) target)) high_label low_label)",
                (
                    "(if (gt (tuple_get item index) threshold) high_label low_label)",
                    "(if (eq (mod (sum item) modulus) target) high_label low_label)",
                ),
                (
                    "(if (gt (tuple_get item index) threshold) high_label low_label)",
                    "(if (eq (mod (sum item) modulus) target) high_label low_label)",
                    "(if (and (gt (sum item) threshold) (eq (mod (tuple_get item index) modulus) target)) high_label low_label)",
                    "(if (or (gt (tuple_get item index) threshold) (eq (mod (sum item) modulus) target)) high_label low_label)",
                ),
                lambda rng: {
                    "item": (item := _triple(rng)),
                    "index": (index := rng.randint(0, 2)),
                    "threshold": _near(item[index], rng, 3),
                    "modulus": (modulus := rng.randint(2, 9)),
                    "target": rng.randint(0, modulus - 1) if rng.random() < 0.5 else sum(item) % modulus,
                    "high_label": rng.choice(["GATE", "SEND", "KEEP"]),
                    "low_label": rng.choice(["STOP", "HOLD", "DROP"]),
                },
            ),
        }
    )
    return specs


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

FRONTIER_FAMILIES = [
    "modulo_sum_label",
    "length_contains_code",
    "tuple_branch_label",
    "sum_offset_mod_label",
    "length_mod_contains_code",
    "sum_length_branch_label",
    "sorted_index_offset_label",
    "contains_count_length_code",
    "tuple_sum_gate_label",
    "not_contains_length_code",
]

CEILING_FAMILIES = [
    "sum_length_mod_gate_label",
    "sorted_index_sum_branch_label",
    "token_absent_length_code",
    "token_count_mod_length_code",
    "text_value_gate_label",
    "tuple_value_mod_label",
    "sorted_join_contains_code",
    "text_absent_mod_code",
    "sum_len_mod_label",
    "tuple_sum_mod_gate_label",
]

STATIC_BRIDGE_ALLOCATION = {family: 6 for family in FRONTIER_FAMILIES}
STATIC_BRIDGE_80_ALLOCATION = {family: 8 for family in FRONTIER_FAMILIES}
BRIDGE_TOTAL = sum(STATIC_BRIDGE_ALLOCATION.values())
CHALLENGE_FAMILIES = FRONTIER_FAMILIES
BRIDGE_ALLOCATION = STATIC_BRIDGE_ALLOCATION


def alias_selector_programs(spec: FamilySpec) -> list[str]:
    extras = {
        "length_contains_code": [
            '(if (and (contains text needle) (gt (count_eq text needle) threshold)) "MATCH_LONG" "MISS")',
            '(if (and (gt (len text) threshold) (gt (len needle) 0)) "MATCH_LONG" "MISS")',
        ],
        "not_contains_length_code": [
            '(if (and (contains text needle) (gt (len text) threshold)) "ABSENT_LONG" "OTHER")',
            '(if (and (not (contains text needle)) (gt (count_eq text needle) threshold)) "ABSENT_LONG" "OTHER")',
        ],
        "sorted_index_offset_label": [
            '(format "SI{}" (add (sum values) offset))',
            '(format "SI{}" (add (tuple_get values index) offset))',
            '(format "SI{}" (add (tuple_get (sort values) 0) offset))',
        ],
        "sum_offset_mod_label": [
            '(format "OM{}" (add (mod (sum values) modulus) offset))',
            '(format "OM{}" (add (sum values) offset))',
            '(format "OM{}" (mod (sum values) offset))',
        ],
        "contains_count_length_code": [
            '(if (and (contains tokens needle) (gt (count_eq tokens needle) threshold)) "MANY_LONG" "MISS")',
            '(if (and (gt (len tokens) min_len) (gt (count_eq tokens needle) threshold)) "MANY_LONG" "MISS")',
        ],
        "length_mod_contains_code": [
            '(if (and (contains text needle) (eq (mod (count_eq text needle) modulus) target)) "HIT_MOD" "MISS")',
            '(if (and (contains text needle) (gt (count_eq text needle) target)) "HIT_MOD" "MISS")',
        ],
        "tuple_branch_label": [
            "(if (gt (tuple_get item index) 0) high_label low_label)",
            "(if (gt (sum item) threshold) high_label low_label)",
        ],
        "tuple_sum_gate_label": [
            "(if (or (gt (tuple_get item index) threshold) (gt (sum item) sum_threshold)) high_label low_label)",
            "(if (and (gt (sum item) threshold) (gt (tuple_get item index) sum_threshold)) high_label low_label)",
        ],
        "sum_length_branch_label": [
            "(if (and (gt (len values) threshold) (gt (len text) min_len)) high_label low_label)",
            "(if (or (gt (sum values) threshold) (gt (len text) min_len)) high_label low_label)",
        ],
        "modulo_sum_label": [
            '(format "M{}" (mod (first values) modulus))',
            '(format "M{}" (mod (len values) modulus))',
        ],
    }.get(spec.name, [])
    ordered = list(spec.static_distractors) + list(spec.wrong_programs) + extras
    return list(dict.fromkeys(program for program in ordered if program and program.strip() != spec.program.strip()))


def safe_execute(program: str, env: dict[str, Any]) -> Any:
    try:
        return execute(program, env)
    except Exception as exc:
        return f"<error:{exc}>"


def make_case(spec: FamilySpec, rng: random.Random, *, case_mode: str = "normal") -> dict[str, Any]:
    env = None
    if case_mode == "hard":
        env = hard_input_for(spec.name, rng)
    elif case_mode == "mixed" and rng.random() < 0.5:
        env = hard_input_for(spec.name, rng)
    if env is None:
        env = spec.make_input(rng)
    return {"input": env, "expected": execute(spec.program, env)}


def random_cases(spec: FamilySpec, rng: random.Random, count: int, *, case_mode: str = "normal") -> list[dict[str, Any]]:
    cases = []
    seen = set()
    attempts = 0
    while len(cases) < count:
        attempts += 1
        if attempts > count * 400:
            raise RuntimeError(f"could not create cases for {spec.name}")
        case = make_case(spec, rng, case_mode=case_mode)
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
    case_mode: str = "normal",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = random_cases(spec, rng, pool_size, case_mode=case_mode)
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
        "case_mode": case_mode,
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
    case_mode: str = "normal",
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
            case_mode=case_mode,
        )
    elif trace_strategy == "alias_discriminative":
        selector_programs = selector_programs or alias_selector_programs(spec)
        visible, selection_stats = select_counterexample_cases(
            spec,
            rng,
            selector_programs,
            count=visible_count,
            pool_size=pool_size,
            case_mode=case_mode,
        )
    elif trace_strategy == "model_discriminative":
        visible, selection_stats = select_counterexample_cases(
            spec,
            rng,
            selector_programs or [chosen_wrong],
            count=visible_count,
            pool_size=pool_size,
            case_mode=case_mode,
        )
    elif trace_strategy in {"seed_random", "mining_prompt", "eval_random"}:
        visible = random_cases(spec, rng, visible_count, case_mode=case_mode)
    else:
        raise ValueError(trace_strategy)
    hidden = random_cases(spec, rng, hidden_count, case_mode=case_mode)
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
        record["case_pool"] = random_cases(spec, rng, case_pool_count, case_mode=case_mode)
    return record


def base_records(
    *,
    rng: random.Random,
    count: int,
    split: str,
    visible_count: int,
    hidden_count: int,
    pool_size: int,
    case_pool_count: int = 0,
    case_mode: str = "normal",
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
                case_pool_count=case_pool_count,
                case_mode=case_mode,
            )
        )
    return rows


def frontier_records(
    *,
    rng: random.Random,
    split: str,
    records_per_family: int,
    trace_strategy: str,
    visible_count: int,
    hidden_count: int,
    pool_size: int,
    case_pool_count: int = 0,
    case_mode: str = "normal",
) -> list[dict[str, Any]]:
    rows = []
    for family in FRONTIER_FAMILIES:
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
                    case_mode=case_mode,
                )
            )
    return rows


def challenge_records(**kwargs) -> list[dict[str, Any]]:
    return frontier_records(**kwargs)


def ceiling_records(
    *,
    rng: random.Random,
    split: str,
    records_per_family: int,
    trace_strategy: str,
    visible_count: int,
    hidden_count: int,
    pool_size: int,
    case_pool_count: int = 0,
    case_mode: str = "normal",
) -> list[dict[str, Any]]:
    rows = []
    for family in CEILING_FAMILIES:
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
                    case_mode=case_mode,
                )
            )
    return rows


def static_bridge_records(
    *,
    rng: random.Random,
    visible_count: int,
    hidden_count: int,
    pool_size: int,
    allocation: dict[str, int] | None = None,
    case_mode: str = "normal",
    case_pool_count: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    for family, count in (allocation or STATIC_BRIDGE_ALLOCATION).items():
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
                    case_mode=case_mode,
                    case_pool_count=case_pool_count,
                )
            )
    return rows


def alias_discriminative_bridge_records(
    *,
    rng: random.Random,
    visible_count: int,
    hidden_count: int,
    pool_size: int,
    case_mode: str = "hard",
) -> list[dict[str, Any]]:
    rows = []
    for family, count in STATIC_BRIDGE_ALLOCATION.items():
        spec = make_specs()[family]
        selector_programs = alias_selector_programs(spec)
        for i in range(count):
            rows.append(
                make_record(
                    rng=rng,
                    split="train_bridge",
                    family_name=family,
                    index=i,
                    trace_strategy="alias_discriminative",
                    visible_count=visible_count,
                    hidden_count=hidden_count,
                    pool_size=pool_size,
                    wrong_program=selector_programs[0],
                    selector_programs=selector_programs,
                    case_mode=case_mode,
                )
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
