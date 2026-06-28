#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from repair_experiment.patching import apply_patch_to_files, unified_diff_for_files  # noqa: E402
from repair_experiment.runner import run_pytest, syntax_valid  # noqa: E402


DATASET_NAME = "bridge_dose_recombination_curriculum"

BUGGY_FILE = '''"""Synthetic repair target."""


def apply_rule(value):
    raise NotImplementedError("apply_rule still needs the validator rule")
'''

ISSUE = (
    "The apply_rule function was patched with the wrong validator rule. "
    "Use the repository context and failing visible test output to infer the "
    "minimal corrective diff."
)

BRIDGE_DOSES = [0, 1, 2, 4, 8]
TOTAL_TRAIN_RECORDS = 240


@dataclass(frozen=True)
class CaseBundle:
    task_id: str
    split: str
    family: str
    current_source: str
    target_source: str
    visible_cases: list[tuple[Any, Any]]
    hidden_cases: list[tuple[Any, Any]]
    markers: list[str]
    params: dict[str, Any]
    factors: list[str]
    heldout_pair: str | None = None


def module(body: str) -> str:
    return '"""Generated repair target."""\n\n' + body.strip() + "\n"


def token(rng: random.Random, prefix: str, digits: int = 3) -> str:
    return f"{prefix}{rng.randrange(10 ** digits):0{digits}d}"


def choose_other(rng: random.Random, value: Any, choices: list[Any]) -> Any:
    candidates = [choice for choice in choices if choice != value]
    return rng.choice(candidates)


def choose_changed(rng: random.Random, value: int, deltas: list[int]) -> int:
    candidates = [value + delta for delta in deltas if value + delta != value]
    return rng.choice(candidates)


def test_file(cases: list[tuple[Any, Any]], *, visible: bool) -> str:
    lines = [
        "import pytest",
        "",
        "from repair_target import apply_rule",
        "",
        "CASES = [",
    ]
    for value, expected in cases:
        lines.append(f"    ({value!r}, {expected!r}),")
    lines.extend(
        [
            "]",
            "",
            '@pytest.mark.parametrize("value, expected", CASES)',
            f"def test_{'visible' if visible else 'hidden'}_cases(value, expected):",
            "    assert apply_rule(value) == expected",
            "",
        ]
    )
    return "\n".join(lines)


def source_scalar_affine(scale: int, bias: int) -> str:
    return module(
        f'''def apply_rule(value):
    return value * {scale} + {bias}
'''
    )


def make_scalar_affine(rng: random.Random, split: str, index: int) -> CaseBundle:
    scale = rng.choice([-4, -2, 2, 3, 5])
    bias = rng.choice([-11, -5, 1, 7, 13])
    wrong_scale = choose_changed(rng, scale, [-2, -1, 1, 2])
    wrong_bias = choose_changed(rng, bias, [-4, -2, 2, 4])
    rule = lambda value: value * scale + bias
    visible = [-3, 0, 4, 9]
    hidden = [-8, 2, 7, 12]
    return CaseBundle(
        task_id=f"{split}_scalar_affine_{index:04d}",
        split=split,
        family="scalar_affine",
        current_source=source_scalar_affine(wrong_scale, wrong_bias),
        target_source=source_scalar_affine(scale, bias),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[str(scale), str(bias)],
        params={"scale": scale, "bias": bias, "wrong_scale": wrong_scale, "wrong_bias": wrong_bias},
        factors=["arithmetic"],
    )


def source_threshold_offset(threshold: int, low_offset: int, high_offset: int) -> str:
    return module(
        f'''def apply_rule(value):
    if value >= {threshold}:
        return value + {high_offset}
    return value + {low_offset}
'''
    )


def make_threshold_offset(rng: random.Random, split: str, index: int) -> CaseBundle:
    threshold = rng.choice([-4, 0, 5, 9])
    low_offset = rng.choice([-10, -3, 2])
    high_offset = rng.choice([4, 8, 15])
    wrong_threshold = choose_changed(rng, threshold, [-2, 2, 3])
    wrong_low = choose_changed(rng, low_offset, [-4, 4])
    wrong_high = choose_changed(rng, high_offset, [-5, 5])
    rule = lambda value: value + (high_offset if value >= threshold else low_offset)
    visible = [threshold - 2, threshold - 1, threshold, threshold + 3]
    hidden = [threshold - 6, threshold + 1, threshold + 5, threshold + 9]
    return CaseBundle(
        task_id=f"{split}_threshold_offset_{index:04d}",
        split=split,
        family="threshold_offset",
        current_source=source_threshold_offset(wrong_threshold, wrong_low, wrong_high),
        target_source=source_threshold_offset(threshold, low_offset, high_offset),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[str(threshold), str(low_offset), str(high_offset)],
        params={"threshold": threshold, "low_offset": low_offset, "high_offset": high_offset},
        factors=["branching", "arithmetic"],
    )


def source_modulo_label(modulus: int, labels: list[str]) -> str:
    mapping = ", ".join(f"{idx}: {label!r}" for idx, label in enumerate(labels))
    return module(
        f'''def apply_rule(value):
    labels = {{{mapping}}}
    return labels[value % {modulus}]
'''
    )


def make_modulo_label(rng: random.Random, split: str, index: int) -> CaseBundle:
    modulus = rng.choice([2, 3, 4])
    labels = [token(rng, f"M{idx}_") for idx in range(modulus)]
    wrong_labels = labels[1:] + labels[:1]
    rule = lambda value: labels[value % modulus]
    visible = list(range(-1, modulus + 3))
    hidden = [modulus + 4, modulus + 7, -2, -5]
    return CaseBundle(
        task_id=f"{split}_modulo_label_{index:04d}",
        split=split,
        family="modulo_label",
        current_source=source_modulo_label(modulus, wrong_labels),
        target_source=source_modulo_label(modulus, labels),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[str(modulus), *labels],
        params={"modulus": modulus, "labels": labels},
        factors=["modulo", "branching", "string_format"],
    )


def source_tuple_affine(left_scale: int, right_scale: int, bias: int) -> str:
    return module(
        f'''def apply_rule(value):
    left, right = value
    return left * {left_scale} + right * {right_scale} + {bias}
'''
    )


def make_tuple_affine(rng: random.Random, split: str, index: int) -> CaseBundle:
    left_scale = rng.choice([-3, -1, 2, 4])
    right_scale = rng.choice([-2, 1, 3, 5])
    bias = rng.choice([-8, -3, 2, 9])
    wrong_left = choose_changed(rng, left_scale, [-2, 2])
    wrong_right = choose_changed(rng, right_scale, [-2, 2])
    wrong_bias = choose_changed(rng, bias, [-3, 3])
    rule = lambda pair: pair[0] * left_scale + pair[1] * right_scale + bias
    visible = [(1, 2), (-3, 4), (0, 5), (6, -1)]
    hidden = [(-2, -2), (4, 7), (9, 0), (3, -5)]
    return CaseBundle(
        task_id=f"{split}_tuple_affine_{index:04d}",
        split=split,
        family="tuple_affine",
        current_source=source_tuple_affine(wrong_left, wrong_right, wrong_bias),
        target_source=source_tuple_affine(left_scale, right_scale, bias),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[str(left_scale), str(right_scale), str(bias)],
        params={"left_scale": left_scale, "right_scale": right_scale, "bias": bias},
        factors=["tuple_access", "arithmetic"],
    )


def source_tuple_sum_wrap(prefix: str, suffix: str, offset: int) -> str:
    return module(
        f'''def apply_rule(value):
    total = sum(value)
    return {prefix!r} + str(total + {offset}) + {suffix!r}
'''
    )


def make_tuple_sum_wrap(rng: random.Random, split: str, index: int) -> CaseBundle:
    prefix = token(rng, "TS_")
    suffix = token(rng, "_ST")
    offset = rng.choice([-9, -4, 3, 8])
    wrong_prefix = token(rng, "BAD_TS_")
    wrong_suffix = token(rng, "_BADST")
    wrong_offset = choose_changed(rng, offset, [-3, 3])
    rule = lambda value: prefix + str(sum(value) + offset) + suffix
    visible = [(1, 2, 3), (-3, 4, 1), (0, 0, 5), (9, -2, -1)]
    hidden = [(-5, -1, 2), (7, 8, -3), (4, 4, 4), (10, -3, 2)]
    return CaseBundle(
        task_id=f"{split}_tuple_sum_wrap_{index:04d}",
        split=split,
        family="tuple_sum_wrap",
        current_source=source_tuple_sum_wrap(wrong_prefix, wrong_suffix, wrong_offset),
        target_source=source_tuple_sum_wrap(prefix, suffix, offset),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[prefix, suffix, str(offset)],
        params={"prefix": prefix, "suffix": suffix, "offset": offset},
        factors=["tuple_access", "aggregation", "string_format"],
    )


def source_length_label(threshold: int, short_label: str, long_label: str) -> str:
    return module(
        f'''def apply_rule(value):
    text = str(value).strip()
    if len(text) >= {threshold}:
        return {long_label!r}
    return {short_label!r}
'''
    )


def make_length_label(rng: random.Random, split: str, index: int) -> CaseBundle:
    threshold = rng.choice([4, 5, 7, 9])
    short_label = token(rng, "SHORT_")
    long_label = token(rng, "LONG_")
    wrong_threshold = choose_changed(rng, threshold, [-2, -1, 1, 2])
    wrong_short = token(rng, "BAD_SHORT_")
    wrong_long = token(rng, "BAD_LONG_")
    rule = lambda value: long_label if len(str(value).strip()) >= threshold else short_label
    visible = ["x" * (threshold - 1), "x" * threshold, "  abc  ", "long-value"]
    hidden = ["q" * max(1, threshold - 2), "q" * (threshold + 2), " mid ", "another-long-value"]
    return CaseBundle(
        task_id=f"{split}_length_label_{index:04d}",
        split=split,
        family="length_label",
        current_source=source_length_label(wrong_threshold, wrong_short, wrong_long),
        target_source=source_length_label(threshold, short_label, long_label),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[str(threshold), short_label, long_label],
        params={"threshold": threshold},
        factors=["length", "branching", "string_format"],
    )


def source_contains_label(needle: str, hit_label: str, miss_label: str) -> str:
    return module(
        f'''def apply_rule(value):
    text = str(value).lower()
    if {needle!r} in text:
        return {hit_label!r}
    return {miss_label!r}
'''
    )


def make_contains_label(rng: random.Random, split: str, index: int) -> CaseBundle:
    needle = rng.choice(["ax", "mid", "zz", "core"])
    hit_label = token(rng, "HIT_")
    miss_label = token(rng, "MISS_")
    wrong_needle = choose_other(rng, needle, ["ax", "mid", "zz", "core"])
    wrong_hit = token(rng, "BAD_HIT_")
    wrong_miss = token(rng, "BAD_MISS_")
    rule = lambda value: hit_label if needle in str(value).lower() else miss_label
    visible = [f"pre-{needle}-post", "outside", f"{needle.upper()}Z", "plain"]
    hidden = [f"left{needle}", "neutral", f"{needle}-tail", "empty"]
    return CaseBundle(
        task_id=f"{split}_contains_label_{index:04d}",
        split=split,
        family="contains_label",
        current_source=source_contains_label(wrong_needle, wrong_hit, wrong_miss),
        target_source=source_contains_label(needle, hit_label, miss_label),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[needle, hit_label, miss_label],
        params={"needle": needle},
        factors=["string_match", "branching", "string_format"],
    )


def source_sort_join(prefix: str, suffix: str, separator: str) -> str:
    return module(
        f'''def apply_rule(value):
    pieces = sorted(str(part).strip().lower() for part in value)
    return {prefix!r} + {separator!r}.join(pieces) + {suffix!r}
'''
    )


def make_sort_join(rng: random.Random, split: str, index: int) -> CaseBundle:
    prefix = token(rng, "SJ_")
    suffix = token(rng, "_JS")
    separator = rng.choice([".", "+", "~"])
    wrong_prefix = token(rng, "BAD_SJ_")
    wrong_suffix = token(rng, "_BJS")
    wrong_separator = choose_other(rng, separator, [".", "+", "~", "/"])
    rule = lambda value: prefix + separator.join(sorted(str(part).strip().lower() for part in value)) + suffix
    visible = [("Beta", " alpha "), ("two", "One"), ("Z", "a", "m")]
    hidden = [("gamma", "beta"), ("Root", "leaf"), ("same", "Same", "alpha")]
    return CaseBundle(
        task_id=f"{split}_sort_join_{index:04d}",
        split=split,
        family="sort_join",
        current_source=source_sort_join(wrong_prefix, wrong_suffix, wrong_separator),
        target_source=source_sort_join(prefix, suffix, separator),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[prefix, suffix, separator],
        params={"prefix": prefix, "suffix": suffix, "separator": separator},
        factors=["ordering", "sequence_iteration", "string_normalization", "string_format"],
    )


def source_sum_offset(offset: int) -> str:
    return module(
        f'''def apply_rule(value):
    return sum(value) + {offset}
'''
    )


def make_sum_offset(rng: random.Random, split: str, index: int) -> CaseBundle:
    offset = rng.choice([-10, -4, 3, 8, 13])
    wrong_offset = choose_changed(rng, offset, [-5, -2, 2, 5])
    rule = lambda values: sum(values) + offset
    visible = [[1, 2, 3], [-3, 4], [0, 0, 5], [9, -2]]
    hidden = [[-5, -1], [7, 8], [4, 4, 4], [10, -3, 2]]
    return CaseBundle(
        task_id=f"{split}_sum_offset_{index:04d}",
        split=split,
        family="sum_offset",
        current_source=source_sum_offset(wrong_offset),
        target_source=source_sum_offset(offset),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[str(offset)],
        params={"offset": offset},
        factors=["aggregation", "arithmetic"],
    )


def source_sum_threshold(threshold: int, low_offset: int, high_offset: int) -> str:
    return module(
        f'''def apply_rule(value):
    total = sum(value)
    if total >= {threshold}:
        return total + {high_offset}
    return total + {low_offset}
'''
    )


def make_sum_threshold(rng: random.Random, split: str, index: int) -> CaseBundle:
    threshold = rng.choice([3, 5, 8])
    low_offset = rng.choice([-9, -4, 1])
    high_offset = rng.choice([4, 8, 12])
    wrong_low = choose_changed(rng, low_offset, [-3, 3])
    wrong_high = choose_changed(rng, high_offset, [-4, 4])
    rule = lambda values: sum(values) + (high_offset if sum(values) >= threshold else low_offset)
    visible = [[1, 1], [3, 4], [-2, 1, 2], [8, 2]]
    hidden = [[threshold, 0], [threshold - 5, 2], [10, -1], [-5, 2]]
    return CaseBundle(
        task_id=f"{split}_sum_threshold_{index:04d}",
        split=split,
        family="sum_threshold",
        current_source=source_sum_threshold(threshold, wrong_low, wrong_high),
        target_source=source_sum_threshold(threshold, low_offset, high_offset),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[str(threshold), str(low_offset), str(high_offset)],
        params={"threshold": threshold, "low_offset": low_offset, "high_offset": high_offset},
        factors=["aggregation", "branching", "arithmetic"],
    )


def source_modulo_shift(modulus: int, residue: int, hit_offset: int, miss_offset: int) -> str:
    return module(
        f'''def apply_rule(value):
    if value % {modulus} == {residue}:
        return value + {hit_offset}
    return value + {miss_offset}
'''
    )


def make_modulo_shift(rng: random.Random, split: str, index: int) -> CaseBundle:
    modulus = rng.choice([2, 3, 4, 5])
    residue = rng.randrange(modulus)
    hit_offset = rng.choice([-8, -2, 5, 11])
    miss_offset = rng.choice([-5, 1, 7, 13])
    wrong_hit = choose_changed(rng, hit_offset, [-4, 4])
    wrong_miss = choose_changed(rng, miss_offset, [-3, 3])
    rule = lambda value: value + (hit_offset if value % modulus == residue else miss_offset)
    visible = [-2, -1, 0, 1, modulus + residue]
    candidates = [modulus * 2 + residue, modulus * 3 + residue, modulus * 4 + residue + 1, 7, 8, -5, -7]
    hidden = []
    for candidate in candidates:
        if candidate not in visible and candidate not in hidden:
            hidden.append(candidate)
        if len(hidden) == 4:
            break
    return CaseBundle(
        task_id=f"{split}_modulo_shift_{index:04d}",
        split=split,
        family="modulo_shift",
        current_source=source_modulo_shift(modulus, residue, wrong_hit, wrong_miss),
        target_source=source_modulo_shift(modulus, residue, hit_offset, miss_offset),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[str(modulus), str(residue), str(hit_offset), str(miss_offset)],
        params={"modulus": modulus, "residue": residue, "hit_offset": hit_offset, "miss_offset": miss_offset},
        factors=["modulo", "branching", "arithmetic"],
    )


def source_length_affine(scale: int, bias: int) -> str:
    return module(
        f'''def apply_rule(value):
    text = str(value).strip()
    return len(text) * {scale} + {bias}
'''
    )


def make_length_affine(rng: random.Random, split: str, index: int) -> CaseBundle:
    scale = rng.choice([-3, -1, 2, 4])
    bias = rng.choice([-7, 0, 5, 10])
    wrong_scale = choose_changed(rng, scale, [-2, 2])
    wrong_bias = choose_changed(rng, bias, [-3, 3])
    rule = lambda value: len(str(value).strip()) * scale + bias
    visible = ["a", "abcd", "  trim  ", "long-value"]
    hidden = ["xy", " hidden ", "1234567", ""]
    return CaseBundle(
        task_id=f"{split}_length_affine_{index:04d}",
        split=split,
        family="length_affine",
        current_source=source_length_affine(wrong_scale, wrong_bias),
        target_source=source_length_affine(scale, bias),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[str(scale), str(bias)],
        params={"scale": scale, "bias": bias},
        factors=["length", "arithmetic"],
    )


def source_tuple_branch_label(threshold: int, low_label: str, high_label: str, low_offset: int, high_offset: int) -> str:
    return module(
        f'''def apply_rule(value):
    left, right = value
    selected = left if left >= right else right
    if selected >= {threshold}:
        return {high_label!r} + str(selected + {high_offset})
    return {low_label!r} + str(selected + {low_offset})
'''
    )


def make_tuple_branch_label(rng: random.Random, split: str, index: int) -> CaseBundle:
    threshold = rng.choice([3, 5, 8])
    low_label = token(rng, "TBL_")
    high_label = token(rng, "TBH_")
    low_offset = rng.choice([-6, -2, 1])
    high_offset = rng.choice([4, 7, 11])
    wrong_low = choose_changed(rng, low_offset, [-3, 3])
    wrong_high = choose_changed(rng, high_offset, [-4, 4])
    wrong_low_label = token(rng, "BAD_TBL_")
    wrong_high_label = token(rng, "BAD_TBH_")

    def rule(pair: tuple[int, int]) -> str:
        selected = pair[0] if pair[0] >= pair[1] else pair[1]
        return (high_label + str(selected + high_offset)) if selected >= threshold else (low_label + str(selected + low_offset))

    visible = [(4, 1), (2, 5), (threshold - 2, threshold - 1), (9, 3)]
    hidden = [(-1, 7), (8, 2), (0, 0), (threshold + 4, threshold + 5)]
    return CaseBundle(
        task_id=f"{split}_tuple_branch_label_{index:04d}",
        split=split,
        family="tuple_branch_label",
        current_source=source_tuple_branch_label(threshold, wrong_low_label, wrong_high_label, wrong_low, wrong_high),
        target_source=source_tuple_branch_label(threshold, low_label, high_label, low_offset, high_offset),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[str(threshold), low_label, high_label, str(low_offset), str(high_offset)],
        params={"threshold": threshold, "low_offset": low_offset, "high_offset": high_offset},
        factors=["tuple_access", "branching", "arithmetic", "string_format"],
        heldout_pair="branching+tuple_access",
    )


def source_length_contains_code(needle: str, threshold: int, prefix: str, fallback: str, bonus: int, penalty: int) -> str:
    return module(
        f'''def apply_rule(value):
    text = str(value).strip().lower()
    if {needle!r} in text and len(text) >= {threshold}:
        return {prefix!r} + str(len(text) + {bonus})
    return {fallback!r} + str(len(text) - {penalty})
'''
    )


def make_length_contains_code(rng: random.Random, split: str, index: int) -> CaseBundle:
    needle = rng.choice(["key", "ax", "mid"])
    threshold = rng.choice([6, 8, 10])
    prefix = token(rng, "LC_")
    fallback = token(rng, "NL_")
    bonus = rng.choice([2, 5, 9])
    penalty = rng.choice([1, 3, 6])
    wrong_threshold = choose_changed(rng, threshold, [-2, 2])
    wrong_prefix = token(rng, "BAD_LC_")
    wrong_fallback = token(rng, "BAD_NL_")

    def rule(value: str) -> str:
        text = str(value).strip().lower()
        return (prefix + str(len(text) + bonus)) if needle in text and len(text) >= threshold else (fallback + str(len(text) - penalty))

    visible = [f"{needle}-long", needle, "plain-value", f"pre-{needle.upper()}"]
    hidden = [f"left-{needle}-right", "short", f"{needle}xx", "ordinary"]
    return CaseBundle(
        task_id=f"{split}_length_contains_code_{index:04d}",
        split=split,
        family="length_contains_code",
        current_source=source_length_contains_code(needle, wrong_threshold, wrong_prefix, wrong_fallback, bonus + 1, penalty + 1),
        target_source=source_length_contains_code(needle, threshold, prefix, fallback, bonus, penalty),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[needle, str(threshold), prefix, fallback, str(bonus), str(penalty)],
        params={"needle": needle, "threshold": threshold, "bonus": bonus, "penalty": penalty},
        factors=["length", "string_match", "branching", "arithmetic", "string_format"],
        heldout_pair="length+string_match",
    )


def source_modulo_sum_label(modulus: int, residue: int, hit_label: str, miss_label: str, bonus: int, penalty: int) -> str:
    return module(
        f'''def apply_rule(value):
    total = sum(value)
    if total % {modulus} == {residue}:
        return {hit_label!r} + str(total + {bonus})
    return {miss_label!r} + str(total - {penalty})
'''
    )


def make_modulo_sum_label(rng: random.Random, split: str, index: int) -> CaseBundle:
    modulus = rng.choice([2, 3, 4])
    residue = rng.randrange(modulus)
    hit_label = token(rng, "MSH_")
    miss_label = token(rng, "MSM_")
    bonus = rng.choice([3, 7, 11])
    penalty = rng.choice([2, 5, 9])
    wrong_hit = token(rng, "BAD_MSH_")
    wrong_miss = token(rng, "BAD_MSM_")
    wrong_bonus = choose_changed(rng, bonus, [-2, 2])
    wrong_penalty = choose_changed(rng, penalty, [-2, 2])

    def rule(values: list[int]) -> str:
        total = sum(values)
        return (hit_label + str(total + bonus)) if total % modulus == residue else (miss_label + str(total - penalty))

    visible = [[1, 2, 3], [modulus + residue, 1], [-2, 5], [8, -3, 4]]
    hidden = [[residue, modulus], [7, 8, -1], [-5, -1, 2], [10, -3, 2]]
    return CaseBundle(
        task_id=f"{split}_modulo_sum_label_{index:04d}",
        split=split,
        family="modulo_sum_label",
        current_source=source_modulo_sum_label(modulus, residue, wrong_hit, wrong_miss, wrong_bonus, wrong_penalty),
        target_source=source_modulo_sum_label(modulus, residue, hit_label, miss_label, bonus, penalty),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[str(modulus), str(residue), hit_label, miss_label, str(bonus), str(penalty)],
        params={"modulus": modulus, "residue": residue, "bonus": bonus, "penalty": penalty},
        factors=["modulo", "aggregation", "branching", "arithmetic", "string_format"],
        heldout_pair="aggregation+modulo",
    )


def source_sorted_tuple_affine(scale: int, bias: int) -> str:
    return module(
        f'''def apply_rule(value):
    left, right = sorted(value)
    return left * {scale} + right + {bias}
'''
    )


def make_sorted_tuple_affine(rng: random.Random, split: str, index: int) -> CaseBundle:
    scale = rng.choice([-3, -1, 2, 4])
    bias = rng.choice([-8, -2, 3, 9])
    wrong_scale = choose_changed(rng, scale, [-2, 2])
    wrong_bias = choose_changed(rng, bias, [-3, 3])
    rule = lambda pair: sorted(pair)[0] * scale + sorted(pair)[1] + bias
    visible = [(5, 1), (2, 7), (-3, 4), (6, -1)]
    hidden = [(-2, -5), (8, 0), (9, 3), (4, 4)]
    return CaseBundle(
        task_id=f"{split}_sorted_tuple_affine_{index:04d}",
        split=split,
        family="sorted_tuple_affine",
        current_source=source_sorted_tuple_affine(wrong_scale, wrong_bias),
        target_source=source_sorted_tuple_affine(scale, bias),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[str(scale), str(bias)],
        params={"scale": scale, "bias": bias},
        factors=["ordering", "tuple_access", "arithmetic"],
        heldout_pair="ordering+tuple_access",
    )


def source_sorted_contains_count(needle: str, prefix: str, separator: str) -> str:
    return module(
        f'''def apply_rule(value):
    pieces = sorted(str(part).strip().lower() for part in value)
    count = sum(1 for piece in pieces if {needle!r} in piece)
    return {prefix!r} + str(count) + ":" + {separator!r}.join(pieces)
'''
    )


def make_sorted_contains_count(rng: random.Random, split: str, index: int) -> CaseBundle:
    needle = rng.choice(["ax", "mid", "key"])
    prefix = token(rng, "SC_")
    separator = rng.choice([".", "+", "~"])
    wrong_needle = choose_other(rng, needle, ["ax", "mid", "key"])
    wrong_prefix = token(rng, "BAD_SC_")
    wrong_separator = choose_other(rng, separator, [".", "+", "~", "/"])

    def rule(value: tuple[str, ...]) -> str:
        pieces = sorted(str(part).strip().lower() for part in value)
        count = sum(1 for piece in pieces if needle in piece)
        return prefix + str(count) + ":" + separator.join(pieces)

    visible = [(f"{needle}z", "beta", f"pre{needle}"), ("plain", "none"), ("Z", f"{needle}x", "a")]
    hidden = [("gamma", f"{needle}-beta"), ("Root", "leaf"), (f"{needle}", f"{needle}tail", "alpha")]
    return CaseBundle(
        task_id=f"{split}_sorted_contains_count_{index:04d}",
        split=split,
        family="sorted_contains_count",
        current_source=source_sorted_contains_count(wrong_needle, wrong_prefix, wrong_separator),
        target_source=source_sorted_contains_count(needle, prefix, separator),
        visible_cases=[(value, rule(value)) for value in visible],
        hidden_cases=[(value, rule(value)) for value in hidden],
        markers=[needle, prefix, separator],
        params={"needle": needle, "prefix": prefix, "separator": separator},
        factors=["ordering", "string_match", "aggregation", "sequence_iteration", "string_normalization", "string_format"],
        heldout_pair="ordering+string_match",
    )


TRAIN_FAMILIES = [
    make_scalar_affine,
    make_threshold_offset,
    make_modulo_label,
    make_tuple_affine,
    make_tuple_sum_wrap,
    make_length_label,
    make_contains_label,
    make_sort_join,
    make_sum_offset,
    make_sum_threshold,
    make_modulo_shift,
    make_length_affine,
]

HOLDOUT_FAMILIES = [
    make_tuple_branch_label,
    make_length_contains_code,
    make_modulo_sum_label,
    make_sorted_tuple_affine,
    make_sorted_contains_count,
]


def augment_trace(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    pair = metadata.get("heldout_pair") or "seen_combination"
    block = "\n".join(
        [
            "<FACTOR_TRACE>",
            f"family={metadata['bug_family']}",
            f"factors={','.join(metadata['factor_tags'])}",
            f"recombination_cell={pair}",
            "</FACTOR_TRACE>",
            "",
        ]
    )
    labelled = dict(record)
    labelled["test_output_after_wrong_patch"] = block + record["test_output_after_wrong_patch"]
    labelled["metadata"] = {**metadata, "trace_has_factor_labels": True}
    return labelled


def build_record(bundle: CaseBundle, *, labelled_trace: bool) -> dict[str, Any]:
    buggy_files = {"repair_target.py": BUGGY_FILE}
    current_files = {"repair_target.py": bundle.current_source}
    target_files = {"repair_target.py": bundle.target_source}
    visible_tests = {"test_visible.py": test_file(bundle.visible_cases, visible=True)}
    hidden_tests = {"test_hidden.py": test_file(bundle.hidden_cases, visible=False)}
    wrong_patch = unified_diff_for_files(buggy_files, current_files)
    target_next_diff = unified_diff_for_files(current_files, target_files)
    base_buggy_diff = unified_diff_for_files(buggy_files, target_files)

    current_visible = run_pytest(current_files, visible_tests, hidden_tests, which="visible")
    if current_visible["passed"]:
        raise AssertionError(f"wrong patch unexpectedly passes visible tests: {bundle.task_id}")
    applied, patched_files, apply_output = apply_patch_to_files(current_files, target_next_diff)
    if not applied:
        raise AssertionError(f"target_next_diff did not apply for {bundle.task_id}: {apply_output}")
    syntax_ok, syntax_error = syntax_valid(patched_files)
    if not syntax_ok:
        raise AssertionError(f"target source has syntax error for {bundle.task_id}: {syntax_error}")
    target_visible = run_pytest(patched_files, visible_tests, hidden_tests, which="visible")
    target_hidden = run_pytest(patched_files, visible_tests, hidden_tests, which="hidden")
    if not target_visible["passed"] or not target_hidden["passed"]:
        raise AssertionError(f"target failed tests for {bundle.task_id}: {target_visible['output']} {target_hidden['output']}")

    visible_inputs = {repr(value) for value, _ in bundle.visible_cases}
    hidden_inputs = {repr(value) for value, _ in bundle.hidden_cases}
    if visible_inputs & hidden_inputs:
        raise AssertionError(f"visible/hidden overlap for {bundle.task_id}: {visible_inputs & hidden_inputs}")

    record = {
        "episode_id": bundle.task_id,
        "task_id": bundle.task_id,
        "split": bundle.split,
        "issue": ISSUE,
        "buggy_files": buggy_files,
        "current_files": current_files,
        "wrong_patch": wrong_patch,
        "target_next_diff": target_next_diff,
        "base_buggy_diff": base_buggy_diff,
        "visible_tests": visible_tests,
        "hidden_tests": hidden_tests,
        "test_output_after_wrong_patch": current_visible["output"],
        "metadata": {
            "dataset": DATASET_NAME,
            "bug_family": bundle.family,
            "factor_tags": bundle.factors,
            "heldout_pair": bundle.heldout_pair,
            "target_markers": bundle.markers,
            "params": bundle.params,
            "visible_cases": bundle.visible_cases,
            "hidden_cases": bundle.hidden_cases,
            "trace_has_factor_labels": False,
        },
    }
    return augment_trace(record) if labelled_trace else record


def make_split(
    *,
    split: str,
    families,
    records_per_family: int,
    seed: int,
    labelled_trace: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family_index, maker in enumerate(families):
        family_name = maker(random.Random(seed + family_index), split, 0).family
        print(f"[build] {split} {family_name} x{records_per_family}", file=sys.stderr, flush=True)
        for index in range(records_per_family):
            rng = random.Random(seed + family_index * 10000 + index)
            bundle = maker(rng, split, index)
            if split == "format_shift":
                bundle = replace(bundle, task_id=bundle.task_id.replace("format_shift", "format_shifted"))
            records.append(build_record(bundle, labelled_trace=labelled_trace))
    return records


def family_name(maker) -> str:
    return maker(random.Random(10_000_003), "name_probe", 0).family


def records_from_counts(
    *,
    split: str,
    counts: dict[Any, int],
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family_index, maker in enumerate(counts):
        records_per_family = counts[maker]
        if records_per_family <= 0:
            continue
        name = family_name(maker)
        print(f"[build] {split} {name} x{records_per_family}", file=sys.stderr, flush=True)
        for index in range(records_per_family):
            rng = random.Random(seed + family_index * 10000 + index)
            bundle = maker(rng, split, index)
            records.append(build_record(bundle, labelled_trace=False))
    return records


def distribute_seen_counts(total_seen: int, *, rotation: int = 0) -> dict[Any, int]:
    if total_seen < 0:
        raise ValueError(f"total_seen must be non-negative, got {total_seen}")
    base = total_seen // len(TRAIN_FAMILIES)
    remainder = total_seen % len(TRAIN_FAMILIES)
    ordered = TRAIN_FAMILIES[rotation % len(TRAIN_FAMILIES) :] + TRAIN_FAMILIES[: rotation % len(TRAIN_FAMILIES)]
    counts = {maker: base for maker in TRAIN_FAMILIES}
    for maker in ordered[:remainder]:
        counts[maker] += 1
    return counts


def dose_train_records(*, dose: int, seed: int) -> list[dict[str, Any]]:
    bridge_total = dose * len(HOLDOUT_FAMILIES)
    seen_total = TOTAL_TRAIN_RECORDS - bridge_total
    seen_counts = distribute_seen_counts(seen_total, rotation=dose)
    bridge_counts = {maker: dose for maker in HOLDOUT_FAMILIES}
    records = records_from_counts(split=f"train_dose{dose}_seen", counts=seen_counts, seed=seed + dose * 100000)
    records.extend(
        records_from_counts(
            split=f"train_dose{dose}_bridge",
            counts=bridge_counts,
            seed=seed + 500000 + dose * 100000,
        )
    )
    if len(records) != TOTAL_TRAIN_RECORDS:
        raise AssertionError(f"dose {dose} produced {len(records)} records, expected {TOTAL_TRAIN_RECORDS}")
    return records


def near_miss_focus_records(*, seed: int) -> list[dict[str, Any]]:
    priority_counts = {
        make_sum_offset: 26,
        make_modulo_shift: 26,
        make_threshold_offset: 26,
        make_tuple_affine: 26,
        make_length_affine: 26,
        make_contains_label: 25,
        make_sort_join: 25,
        make_scalar_affine: 12,
        make_modulo_label: 12,
        make_tuple_sum_wrap: 12,
        make_length_label: 12,
        make_sum_threshold: 12,
    }
    total = sum(priority_counts.values())
    if total != TOTAL_TRAIN_RECORDS:
        raise AssertionError(f"near-miss focus count is {total}, expected {TOTAL_TRAIN_RECORDS}")
    return records_from_counts(split="train_near_miss_focus", counts=priority_counts, seed=seed)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def split_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        family = record["metadata"]["bug_family"]
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def factor_pair_table(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        factors = sorted(record["metadata"]["factor_tags"])
        for i, left in enumerate(factors):
            for right in factors[i + 1 :]:
                key = f"{left}+{right}"
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def heldout_pair_table(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        pair = record["metadata"].get("heldout_pair")
        if pair:
            counts[pair] = counts.get(pair, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()

    seen_iid = make_split(
        split="seen_iid",
        families=TRAIN_FAMILIES,
        records_per_family=3,
        seed=args.seed + 100000,
        labelled_trace=False,
    )
    format_shift = make_split(
        split="format_shift",
        families=TRAIN_FAMILIES,
        records_per_family=3,
        seed=args.seed + 200000,
        labelled_trace=False,
    )
    recombination = make_split(
        split="recombination_holdout",
        families=HOLDOUT_FAMILIES,
        records_per_family=12,
        seed=args.seed + 300000,
        labelled_trace=False,
    )
    dose_trains = {f"repair_train_dose{dose}": dose_train_records(dose=dose, seed=args.seed + 600000) for dose in BRIDGE_DOSES}
    near_miss_train = near_miss_focus_records(seed=args.seed + 900000)

    outputs = {
        **dose_trains,
        "repair_train_near_miss_focus": near_miss_train,
        "repair_val_seen_iid": seen_iid,
        "repair_val_format_shift": format_shift,
        "repair_val_recombination_holdout": recombination,
    }
    for name, records in outputs.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", records)

    heldout_pairs = sorted({record["metadata"]["heldout_pair"] for record in recombination})
    train_pair_counts = {name: factor_pair_table(records) for name, records in dose_trains.items()}
    train_pair_counts["repair_train_near_miss_focus"] = factor_pair_table(near_miss_train)
    bridge_pair_counts = {name: heldout_pair_table(records) for name, records in dose_trains.items()}
    bridge_pair_counts["repair_train_near_miss_focus"] = heldout_pair_table(near_miss_train)
    unexpected_bridge = {
        name: counts
        for name, counts in bridge_pair_counts.items()
        if (name == "repair_train_dose0" or name == "repair_train_near_miss_focus") and counts
    }
    if unexpected_bridge:
        raise AssertionError(f"zero-dose or near-miss control contains bridge examples: {unexpected_bridge}")
    for dose in BRIDGE_DOSES:
        name = f"repair_train_dose{dose}"
        counts = bridge_pair_counts[name]
        for pair in heldout_pairs:
            observed = counts.get(pair, 0)
            if observed != dose:
                raise AssertionError(f"{name} has {observed} bridge examples for {pair}, expected {dose}")

    manifest = {
        "dataset": DATASET_NAME,
        "seed": args.seed,
        "question": "How many exact bridge examples are needed before trace-conditioned repair generalizes across withheld factor-pair cells?",
        "total_train_records_per_condition": TOTAL_TRAIN_RECORDS,
        "bridge_doses_per_heldout_pair": BRIDGE_DOSES,
        "records": {name: len(records) for name, records in outputs.items()},
        "families": {name: split_counts(records) for name, records in outputs.items()},
        "train_factor_pair_counts": train_pair_counts,
        "bridge_pair_counts": bridge_pair_counts,
        "heldout_factor_pairs": heldout_pairs,
        "train_files": {name: str(args.output_dir / f"{name}.jsonl") for name in dose_trains},
        "near_miss_control_file": str(args.output_dir / "repair_train_near_miss_focus.jsonl"),
        "evaluation_files": {
            "seen_iid": str(args.output_dir / "repair_val_seen_iid.jsonl"),
            "format_shift": str(args.output_dir / "repair_val_format_shift.jsonl"),
            "recombination_holdout": str(args.output_dir / "repair_val_recombination_holdout.jsonl"),
        },
    }
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
