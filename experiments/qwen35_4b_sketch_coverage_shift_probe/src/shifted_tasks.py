from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .dsl import execute
from .sketch import make_target_sketch


@dataclass(frozen=True)
class FamilySpec:
    name: str
    shift_type: str
    schema: str
    program: str
    manual_sketch: str
    erased_sketch: str
    make_input: Callable[[random.Random], dict[str, Any]]


def _ints(rng: random.Random, min_len: int = 4, max_len: int = 8) -> list[int]:
    return [rng.randint(-9, 21) for _ in range(rng.randint(min_len, max_len))]


def _small_positive_ints(rng: random.Random, min_len: int = 3, max_len: int = 6) -> list[int]:
    return [rng.randint(1, 7) for _ in range(rng.randint(min_len, max_len))]


def _triple(rng: random.Random) -> list[int]:
    return [rng.randint(-12, 24), rng.randint(-12, 24), rng.randint(-12, 24)]


def _word(rng: random.Random, min_len: int = 4, max_len: int = 16) -> str:
    alphabet = "abcde"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(min_len, max_len)))


def _text_with_piece(rng: random.Random, *, piece_name: str = "needle", text_name: str = "text") -> dict[str, Any]:
    piece = rng.choice(["aa", "bb", "cc", "de", "ed"])
    text = _word(rng, 4, 16)
    if rng.random() < 0.58:
        pos = rng.randint(0, len(text))
        text = text[:pos] + piece + text[pos:]
    return {text_name: text, piece_name: piece}


def _text_with_prefix(rng: random.Random) -> dict[str, Any]:
    prefix = rng.choice(["ab", "bc", "cd", "de", "aa"])
    if rng.random() < 0.58:
        suffix = _word(rng, 2, 14)
        text = prefix + suffix
    else:
        text = _word(rng, 4, 16)
        attempts = 0
        while text.startswith(prefix) and attempts < 100:
            text = _word(rng, 4, 16)
            attempts += 1
    return {"s": text, "prefix": prefix}


def make_specs() -> list[FamilySpec]:
    return [
        FamilySpec(
            name="control_sum_mod",
            shift_type="control_in_bank",
            schema="values: list[int]; modulus: int",
            program='(format "M{}" (mod (sum values) modulus))',
            manual_sketch='(format "M{}" (mod (sum ?SEQ0) ?NUM0))',
            erased_sketch='(format "M{}" ?NUM0)',
            make_input=lambda rng: {"values": _ints(rng), "modulus": rng.randint(2, 9)},
        ),
        FamilySpec(
            name="control_length_contains",
            shift_type="control_in_bank",
            schema="text: str; needle: str; threshold: int",
            program='(if (and (contains text needle) (gt (len text) threshold)) "MATCH_LONG" "MISS")',
            manual_sketch='(if (and (contains ?TEXT0 ?TEXT1) (gt (len ?TEXT0) ?NUM0)) "MATCH_LONG" "MISS")',
            erased_sketch='(if ?PRED0 "MATCH_LONG" "MISS")',
            make_input=lambda rng: {**_text_with_piece(rng), "threshold": rng.randint(4, 12)},
        ),
        FamilySpec(
            name="control_tuple_mod_gate",
            shift_type="control_in_bank",
            schema="item: list[int]; index: int; threshold: int; modulus: int; target: int; high_label: str; low_label: str",
            program="(if (and (gt (tuple_get item index) threshold) (eq (mod (sum item) modulus) target)) high_label low_label)",
            manual_sketch="(if (and (gt (tuple_get ?SEQ0 ?NUM0) ?NUM1) (eq (mod (sum ?SEQ0) ?NUM2) ?NUM3)) high_label low_label)",
            erased_sketch="(if ?PRED0 high_label low_label)",
            make_input=lambda rng: {
                "item": _triple(rng),
                "index": rng.randint(0, 2),
                "threshold": rng.randint(-5, 18),
                "modulus": (modulus := rng.randint(2, 9)),
                "target": rng.randint(0, modulus - 1),
                "high_label": rng.choice(["KEEP", "RAISE", "SEND"]),
                "low_label": rng.choice(["DROP", "LOWER", "HOLD"]),
            },
        ),
        FamilySpec(
            name="alias_sum_mod",
            shift_type="name_shift",
            schema="xs: list[int]; m: int",
            program='(format "M{}" (mod (sum xs) m))',
            manual_sketch='(format "M{}" (mod (sum ?SEQ0) ?NUM0))',
            erased_sketch='(format "M{}" ?NUM0)',
            make_input=lambda rng: {"xs": _ints(rng), "m": rng.randint(2, 9)},
        ),
        FamilySpec(
            name="alias_length_contains",
            shift_type="name_shift",
            schema="s: str; pat: str; cutoff: int",
            program='(if (and (contains s pat) (gt (len s) cutoff)) "MATCH_LONG" "MISS")',
            manual_sketch='(if (and (contains ?TEXT0 ?TEXT1) (gt (len ?TEXT0) ?NUM0)) "MATCH_LONG" "MISS")',
            erased_sketch='(if ?PRED0 "MATCH_LONG" "MISS")',
            make_input=lambda rng: {**_text_with_piece(rng, piece_name="pat", text_name="s"), "cutoff": rng.randint(4, 12)},
        ),
        FamilySpec(
            name="alias_tuple_mod_gate",
            shift_type="name_shift",
            schema="triple: list[int]; pos: int; cut: int; m: int; want: int; yes: str; no: str",
            program="(if (and (gt (tuple_get triple pos) cut) (eq (mod (sum triple) m) want)) yes no)",
            manual_sketch="(if (and (gt (tuple_get ?SEQ0 ?NUM0) ?NUM1) (eq (mod (sum ?SEQ0) ?NUM2) ?NUM3)) yes no)",
            erased_sketch="(if ?PRED0 yes no)",
            make_input=lambda rng: {
                "triple": _triple(rng),
                "pos": rng.randint(0, 2),
                "cut": rng.randint(-5, 18),
                "m": (modulus := rng.randint(2, 9)),
                "want": rng.randint(0, modulus - 1),
                "yes": rng.choice(["KEEP", "RAISE", "SEND"]),
                "no": rng.choice(["DROP", "LOWER", "HOLD"]),
            },
        ),
        FamilySpec(
            name="primitive_max_mod",
            shift_type="primitive_shift",
            schema="xs: list[int]; m: int",
            program='(format "MX{}" (mod (max xs) m))',
            manual_sketch='(format "MX{}" (mod (max ?SEQ0) ?NUM0))',
            erased_sketch='(format "MX{}" ?NUM0)',
            make_input=lambda rng: {"xs": _ints(rng), "m": rng.randint(2, 9)},
        ),
        FamilySpec(
            name="primitive_prod_mod",
            shift_type="primitive_shift",
            schema="xs: list[int]; m: int",
            program='(format "PR{}" (mod (prod xs) m))',
            manual_sketch='(format "PR{}" (mod (prod ?SEQ0) ?NUM0))',
            erased_sketch='(format "PR{}" ?NUM0)',
            make_input=lambda rng: {"xs": _small_positive_ints(rng), "m": rng.randint(2, 9)},
        ),
        FamilySpec(
            name="primitive_abs_min_delta",
            shift_type="primitive_shift",
            schema="xs: list[int]; pivot: int",
            program='(format "D{}" (abs (sub (min xs) pivot)))',
            manual_sketch='(format "D{}" (abs (sub (min ?SEQ0) ?NUM0)))',
            erased_sketch='(format "D{}" ?NUM0)',
            make_input=lambda rng: {"xs": _ints(rng), "pivot": rng.randint(-8, 16)},
        ),
        FamilySpec(
            name="primitive_mul_sum",
            shift_type="primitive_shift",
            schema="xs: list[int]; scale: int",
            program='(format "W{}" (mul (sum xs) scale))',
            manual_sketch='(format "W{}" (mul (sum ?SEQ0) ?NUM0))',
            erased_sketch='(format "W{}" ?NUM0)',
            make_input=lambda rng: {"xs": _ints(rng), "scale": rng.randint(-4, 7)},
        ),
        FamilySpec(
            name="primitive_prefix_gate",
            shift_type="primitive_shift",
            schema="s: str; prefix: str; cutoff: int",
            program='(if (and (startswith s prefix) (gt (len s) cutoff)) "PREFIX_LONG" "MISS")',
            manual_sketch='(if (and (startswith ?TEXT0 ?TEXT1) (gt (len ?TEXT0) ?NUM0)) "PREFIX_LONG" "MISS")',
            erased_sketch='(if ?PRED0 "PREFIX_LONG" "MISS")',
            make_input=lambda rng: {**_text_with_prefix(rng), "cutoff": rng.randint(4, 12)},
        ),
        FamilySpec(
            name="primitive_max_gate",
            shift_type="primitive_shift",
            schema="xs: list[int]; cut: int; hi: str; lo: str",
            program="(if (gt (max xs) cut) hi lo)",
            manual_sketch="(if (gt (max ?SEQ0) ?NUM0) hi lo)",
            erased_sketch="(if ?PRED0 hi lo)",
            make_input=lambda rng: {
                "xs": _ints(rng),
                "cut": rng.randint(-2, 20),
                "hi": rng.choice(["HIGH", "UP", "YES"]),
                "lo": rng.choice(["LOW", "DOWN", "NO"]),
            },
        ),
    ]


def _cases_for(spec: FamilySpec, rng: random.Random, count: int) -> list[dict[str, Any]]:
    cases = []
    attempts = 0
    while len(cases) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError(f"could not generate cases for {spec.name}")
        env = spec.make_input(rng)
        try:
            expected = execute(spec.program, env)
        except Exception:
            continue
        cases.append({"input": env, "expected": expected})
    return cases


def make_record(
    spec: FamilySpec,
    *,
    index: int,
    rng: random.Random,
    visible_count: int,
    hidden_count: int,
    query_count: int,
) -> dict[str, Any]:
    cases = _cases_for(spec, rng, visible_count + hidden_count + query_count)
    visible = cases[:visible_count]
    hidden = cases[visible_count : visible_count + hidden_count]
    query_pool = cases[visible_count + hidden_count :]
    auto_target_sketch = make_target_sketch(spec.program, visible[0]["input"])
    return {
        "id": f"{spec.shift_type}_{spec.name}_{index:04d}",
        "family": spec.name,
        "shift_type": spec.shift_type,
        "schema": spec.schema,
        "target_program": spec.program,
        "target_sketch_auto": auto_target_sketch,
        "target_sketch_manual": spec.manual_sketch,
        "target_sketch_erased": spec.erased_sketch,
        "visible": visible,
        "hidden": hidden,
        "query_pool": query_pool,
    }


def build_records(
    *,
    records_per_family: int,
    visible_count: int = 6,
    hidden_count: int = 18,
    query_count: int = 48,
    seed: int = 20260624,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for spec in make_specs():
        for index in range(records_per_family):
            records.append(
                make_record(
                    spec,
                    index=index,
                    rng=rng,
                    visible_count=visible_count,
                    hidden_count=hidden_count,
                    query_count=query_count,
                )
            )
    rng.shuffle(records)
    return records

