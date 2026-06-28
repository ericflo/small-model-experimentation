#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from repair_experiment.patching import apply_patch_to_files, unified_diff_for_files  # noqa: E402
from repair_experiment.runner import run_pytest, syntax_valid  # noqa: E402


DATASET_NAME = "feature_factorized_rule_diversity"

BUGGY_FILE = '''"""Synthetic repair target."""


def apply_rule(value):
    raise NotImplementedError("apply_rule still needs the validator rule")
'''

ISSUE = (
    "The apply_rule function was patched with the wrong transformation. "
    "Use the repository context and the failing visible test output to infer the "
    "minimal corrective diff."
)


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


def module(body: str) -> str:
    return '"""Generated repair target."""\n\n' + body.strip() + "\n"


def token(rng: random.Random, prefix: str, digits: int = 3) -> str:
    return f"{prefix}{rng.randrange(10 ** digits):0{digits}d}"


def choose_changed(rng: random.Random, value: int, deltas: list[int]) -> int:
    candidates = [value + delta for delta in deltas if value + delta != value]
    return rng.choice(candidates)


def choose_other(rng: random.Random, value: str, choices: list[str]) -> str:
    candidates = [choice for choice in choices if choice != value]
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


def source_affine_add(offset: int) -> str:
    return module(
        f'''def apply_rule(value):
    return value + {offset}
'''
    )


def make_affine_add(rng: random.Random, split: str, index: int) -> CaseBundle:
    offset = rng.choice([-11, -6, -2, 3, 7, 12, 15])
    wrong_offset = choose_changed(rng, offset, [-5, -3, 3, 5])
    rule = lambda value: value + offset
    visible_inputs = [-3, 0, 5, 9]
    hidden_inputs = [-8, 2, 11, 17]
    return CaseBundle(
        task_id=f"{split}_affine_add_{index:04d}",
        split=split,
        family="affine_add",
        current_source=source_affine_add(wrong_offset),
        target_source=source_affine_add(offset),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(offset)],
        params={"offset": offset, "wrong_offset": wrong_offset},
        factors=["arithmetic"],
    )


def source_scale_shift(scale: int, bias: int) -> str:
    return module(
        f'''def apply_rule(value):
    return value * {scale} + {bias}
'''
    )


def make_scale_shift(rng: random.Random, split: str, index: int) -> CaseBundle:
    scale = rng.choice([-3, -2, 2, 3, 4])
    bias = rng.choice([-9, -4, 1, 6, 10])
    wrong_scale = choose_changed(rng, scale, [-2, -1, 1, 2])
    wrong_bias = choose_changed(rng, bias, [-4, -2, 2, 4])
    rule = lambda value: value * scale + bias
    visible_inputs = [-2, 0, 3, 6]
    hidden_inputs = [-5, 1, 4, 9]
    return CaseBundle(
        task_id=f"{split}_scale_shift_{index:04d}",
        split=split,
        family="scale_shift",
        current_source=source_scale_shift(wrong_scale, wrong_bias),
        target_source=source_scale_shift(scale, bias),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(scale), str(bias)],
        params={"scale": scale, "bias": bias, "wrong_scale": wrong_scale, "wrong_bias": wrong_bias},
        factors=["arithmetic"],
    )


def source_threshold_label(threshold: int, low_label: str, high_label: str) -> str:
    return module(
        f'''def apply_rule(value):
    if value <= {threshold}:
        return {low_label!r}
    return {high_label!r}
'''
    )


def make_threshold_label(rng: random.Random, split: str, index: int) -> CaseBundle:
    threshold = rng.choice([-3, 0, 4, 8])
    low_label = token(rng, "LOW_")
    high_label = token(rng, "HIGH_")
    wrong_threshold = choose_changed(rng, threshold, [-2, 2, 3])
    wrong_low = token(rng, "BAD_LOW_")
    wrong_high = token(rng, "BAD_HIGH_")
    rule = lambda value: low_label if value <= threshold else high_label
    visible_inputs = [threshold - 2, threshold, threshold + 1, threshold + 4]
    hidden_inputs = [threshold - 5, threshold - 1, threshold + 2, threshold + 7]
    return CaseBundle(
        task_id=f"{split}_threshold_label_{index:04d}",
        split=split,
        family="threshold_label",
        current_source=source_threshold_label(wrong_threshold, wrong_low, wrong_high),
        target_source=source_threshold_label(threshold, low_label, high_label),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(threshold), low_label, high_label],
        params={"threshold": threshold, "wrong_threshold": wrong_threshold},
        factors=["branching"],
    )


def source_modulo_label(modulus: int, labels: list[str]) -> str:
    label_map = ", ".join(f"{idx}: {label!r}" for idx, label in enumerate(labels))
    return module(
        f'''def apply_rule(value):
    labels = {{{label_map}}}
    return labels[value % {modulus}]
'''
    )


def make_modulo_label(rng: random.Random, split: str, index: int) -> CaseBundle:
    modulus = rng.choice([2, 3, 4])
    labels = [token(rng, f"R{idx}_") for idx in range(modulus)]
    wrong_labels = labels[1:] + labels[:1]
    rule = lambda value: labels[value % modulus]
    visible_inputs = list(range(modulus + 2))
    hidden_inputs = [modulus + 4, modulus + 7, -1, -2]
    return CaseBundle(
        task_id=f"{split}_modulo_label_{index:04d}",
        split=split,
        family="modulo_label",
        current_source=source_modulo_label(modulus, wrong_labels),
        target_source=source_modulo_label(modulus, labels),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(modulus), *labels],
        params={"modulus": modulus, "labels": labels, "wrong_labels": wrong_labels},
        factors=["modulo", "branching"],
    )


def source_normalize_join(separator: str, prefix: str, suffix: str) -> str:
    return module(
        f'''def apply_rule(value):
    pieces = [str(part).strip().lower() for part in value]
    return {prefix!r} + {separator!r}.join(pieces) + {suffix!r}
'''
    )


def make_normalize_join(rng: random.Random, split: str, index: int) -> CaseBundle:
    separator = rng.choice([".", ":", "|"])
    prefix = token(rng, "NJ_")
    suffix = token(rng, "_JN")
    wrong_separator = choose_other(rng, separator, [".", ":", "|", "/"])
    wrong_prefix = token(rng, "BAD_NJ_")
    wrong_suffix = token(rng, "_BADJ")

    def rule(value: tuple[str, ...]) -> str:
        return prefix + separator.join(str(part).strip().lower() for part in value) + suffix

    visible_inputs = [(" Alpha ", "Beta"), ("TWO", " one "), ("Z", "a", "M")]
    hidden_inputs = [("  root", "Leaf"), ("same", "Same"), ("UP", "down", "Mid")]
    return CaseBundle(
        task_id=f"{split}_normalize_join_{index:04d}",
        split=split,
        family="normalize_join",
        current_source=source_normalize_join(wrong_separator, wrong_prefix, wrong_suffix),
        target_source=source_normalize_join(separator, prefix, suffix),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[separator, prefix, suffix],
        params={"separator": separator, "prefix": prefix, "suffix": suffix},
        factors=["string_normalization", "sequence_iteration"],
    )


def source_replace_wrap(old: str, new: str, prefix: str, suffix: str) -> str:
    return module(
        f'''def apply_rule(value):
    return {prefix!r} + str(value).strip().replace({old!r}, {new!r}) + {suffix!r}
'''
    )


def make_replace_wrap(rng: random.Random, split: str, index: int) -> CaseBundle:
    old = rng.choice(["-", "_", " "])
    new = rng.choice([".", ":", "~"])
    prefix = token(rng, "RW_")
    suffix = token(rng, "_WR")
    wrong_old = choose_other(rng, old, ["-", "_", " "])
    wrong_new = choose_other(rng, new, [".", ":", "~", "+"])
    wrong_prefix = token(rng, "BAD_RW_")
    wrong_suffix = token(rng, "_BADR")

    def rule(value: str) -> str:
        return prefix + str(value).strip().replace(old, new) + suffix

    visible_inputs = [f"alpha{old}beta", f" one{old}two ", f"x{old}y{old}z"]
    hidden_inputs = [f"left{old}right", f"edge{old}case ", f"m{old}n"]
    return CaseBundle(
        task_id=f"{split}_replace_wrap_{index:04d}",
        split=split,
        family="replace_wrap",
        current_source=source_replace_wrap(wrong_old, wrong_new, wrong_prefix, wrong_suffix),
        target_source=source_replace_wrap(old, new, prefix, suffix),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[old, new, prefix, suffix],
        params={"old": old, "new": new, "prefix": prefix, "suffix": suffix},
        factors=["string_rewrite"],
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
    visible_inputs = [f"pre-{needle}-post", "outside", f"{needle.upper()}Z", "plain"]
    hidden_inputs = [f"left{needle}", "neutral", f"{needle}-tail", "empty"]
    return CaseBundle(
        task_id=f"{split}_contains_label_{index:04d}",
        split=split,
        family="contains_label",
        current_source=source_contains_label(wrong_needle, wrong_hit, wrong_miss),
        target_source=source_contains_label(needle, hit_label, miss_label),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[needle, hit_label, miss_label],
        params={"needle": needle},
        factors=["string_match", "branching"],
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
    threshold = rng.choice([4, 5, 6, 7])
    short_label = token(rng, "SHORT_")
    long_label = token(rng, "LONG_")
    wrong_threshold = choose_changed(rng, threshold, [-2, -1, 1, 2])
    wrong_short = token(rng, "BAD_SHORT_")
    wrong_long = token(rng, "BAD_LONG_")
    rule = lambda value: long_label if len(str(value).strip()) >= threshold else short_label
    visible_inputs = ["x" * (threshold - 1), "x" * threshold, "  abc  ", "long-value"]
    hidden_inputs = ["q" * max(1, threshold - 2), "q" * (threshold + 2), " mid ", "another-long-value"]
    return CaseBundle(
        task_id=f"{split}_length_label_{index:04d}",
        split=split,
        family="length_label",
        current_source=source_length_label(wrong_threshold, wrong_short, wrong_long),
        target_source=source_length_label(threshold, short_label, long_label),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(threshold), short_label, long_label],
        params={"threshold": threshold, "wrong_threshold": wrong_threshold},
        factors=["length", "branching"],
    )


def source_tuple_linear(left_scale: int, right_scale: int, bias: int) -> str:
    return module(
        f'''def apply_rule(value):
    left, right = value
    return left * {left_scale} + right * {right_scale} + {bias}
'''
    )


def make_tuple_linear(rng: random.Random, split: str, index: int) -> CaseBundle:
    left_scale = rng.choice([-3, -1, 2, 4])
    right_scale = rng.choice([-2, 1, 3, 5])
    bias = rng.choice([-8, -3, 2, 9])
    wrong_left = choose_changed(rng, left_scale, [-2, 2])
    wrong_right = choose_changed(rng, right_scale, [-2, 2])
    wrong_bias = choose_changed(rng, bias, [-3, 3])
    rule = lambda pair: pair[0] * left_scale + pair[1] * right_scale + bias
    visible_inputs = [(1, 2), (-3, 4), (0, 5), (6, -1)]
    hidden_inputs = [(-2, -2), (4, 7), (9, 0), (3, -5)]
    return CaseBundle(
        task_id=f"{split}_tuple_linear_{index:04d}",
        split=split,
        family="tuple_linear",
        current_source=source_tuple_linear(wrong_left, wrong_right, wrong_bias),
        target_source=source_tuple_linear(left_scale, right_scale, bias),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(left_scale), str(right_scale), str(bias)],
        params={"left_scale": left_scale, "right_scale": right_scale, "bias": bias},
        factors=["tuple_access", "arithmetic"],
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
    visible_inputs = [[1, 2, 3], [-3, 4], [0, 0, 5], [9, -2]]
    hidden_inputs = [[-5, -1], [7, 8], [4, 4, 4], [10, -3, 2]]
    return CaseBundle(
        task_id=f"{split}_sum_offset_{index:04d}",
        split=split,
        family="sum_offset",
        current_source=source_sum_offset(wrong_offset),
        target_source=source_sum_offset(offset),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(offset)],
        params={"offset": offset, "wrong_offset": wrong_offset},
        factors=["aggregation", "arithmetic"],
    )


def source_parity_labelled_shift(even_offset: int, odd_offset: int, even_prefix: str, odd_prefix: str) -> str:
    return module(
        f'''def apply_rule(value):
    if value % 2 == 0:
        return {even_prefix!r} + str(value + {even_offset})
    return {odd_prefix!r} + str(value + {odd_offset})
'''
    )


def make_parity_labelled_shift(rng: random.Random, split: str, index: int) -> CaseBundle:
    even_offset = rng.choice([-8, -2, 4, 10])
    odd_offset = rng.choice([-7, -1, 5, 11])
    even_prefix = token(rng, "EV_")
    odd_prefix = token(rng, "OD_")
    wrong_even = choose_changed(rng, even_offset, [-4, 4])
    wrong_odd = choose_changed(rng, odd_offset, [-3, 3])
    wrong_even_prefix = token(rng, "BAD_EV_")
    wrong_odd_prefix = token(rng, "BAD_OD_")
    rule = lambda value: (even_prefix + str(value + even_offset)) if value % 2 == 0 else (odd_prefix + str(value + odd_offset))
    visible_inputs = [-2, -1, 0, 3, 6]
    hidden_inputs = [-5, 2, 7, 8, 11]
    return CaseBundle(
        task_id=f"{split}_parity_labelled_shift_{index:04d}",
        split=split,
        family="parity_labelled_shift",
        current_source=source_parity_labelled_shift(wrong_even, wrong_odd, wrong_even_prefix, wrong_odd_prefix),
        target_source=source_parity_labelled_shift(even_offset, odd_offset, even_prefix, odd_prefix),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(even_offset), str(odd_offset), even_prefix, odd_prefix],
        params={"even_offset": even_offset, "odd_offset": odd_offset},
        factors=["modulo", "branching", "arithmetic", "string_format"],
    )


def source_dedupe_join_wrap(prefix: str, suffix: str, separator: str) -> str:
    return module(
        f'''def apply_rule(value):
    pieces = []
    for part in value:
        piece = str(part).strip().lower()
        if piece not in pieces:
            pieces.append(piece)
    return {prefix!r} + {separator!r}.join(pieces) + {suffix!r}
'''
    )


def make_dedupe_join_wrap(rng: random.Random, split: str, index: int) -> CaseBundle:
    prefix = token(rng, "DU_")
    suffix = token(rng, "_UP")
    separator = rng.choice(["/", "+", "~"])
    wrong_prefix = token(rng, "BAD_DU_")
    wrong_suffix = token(rng, "_BUP")
    wrong_separator = choose_other(rng, separator, ["/", "+", "~", "."])

    def rule(value: tuple[str, ...]) -> str:
        pieces: list[str] = []
        for part in value:
            piece = str(part).strip().lower()
            if piece not in pieces:
                pieces.append(piece)
        return prefix + separator.join(pieces) + suffix

    visible_inputs = [("A", "b", "a"), (" x ", "Y", "x"), ("Solo", "solo")]
    hidden_inputs = [("Root", "leaf", "root"), ("One", "two", "Three"), ("M", "m", "N")]
    return CaseBundle(
        task_id=f"{split}_dedupe_join_wrap_{index:04d}",
        split=split,
        family="dedupe_join_wrap",
        current_source=source_dedupe_join_wrap(wrong_prefix, wrong_suffix, wrong_separator),
        target_source=source_dedupe_join_wrap(prefix, suffix, separator),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[prefix, suffix, separator],
        params={"prefix": prefix, "suffix": suffix, "separator": separator},
        factors=["string_normalization", "sequence_iteration", "aggregation"],
    )


def source_tuple_min_offset(offset: int, label: str) -> str:
    return module(
        f'''def apply_rule(value):
    left, right = value
    selected = left if left <= right else right
    return {label!r} + str(selected + {offset})
'''
    )


def make_tuple_min_offset(rng: random.Random, split: str, index: int) -> CaseBundle:
    offset = rng.choice([-6, -1, 4, 9])
    label = token(rng, "MIN_")
    wrong_offset = choose_changed(rng, offset, [-3, 3])
    wrong_label = token(rng, "BAD_MIN_")
    rule = lambda pair: label + str((pair[0] if pair[0] <= pair[1] else pair[1]) + offset)
    visible_inputs = [(4, 1), (2, 5), (3, 3), (-2, 7)]
    hidden_inputs = [(-1, -4), (8, 2), (0, 9), (5, 5)]
    return CaseBundle(
        task_id=f"{split}_tuple_min_offset_{index:04d}",
        split=split,
        family="tuple_min_offset",
        current_source=source_tuple_min_offset(wrong_offset, wrong_label),
        target_source=source_tuple_min_offset(offset, label),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(offset), label],
        params={"offset": offset, "label": label},
        factors=["tuple_access", "aggregation", "branching", "string_format"],
    )


def source_clamp_affine(low: int, high: int, scale: int, bias: int) -> str:
    return module(
        f'''def apply_rule(value):
    clipped = min(max(value, {low}), {high})
    return clipped * {scale} + {bias}
'''
    )


def make_clamp_affine(rng: random.Random, split: str, index: int) -> CaseBundle:
    low = rng.choice([-6, -3, 0])
    high = rng.choice([5, 8, 11])
    scale = rng.choice([2, 3, -2])
    bias = rng.choice([-5, 1, 7])
    wrong_scale = choose_changed(rng, scale, [-1, 1, 2])
    wrong_bias = choose_changed(rng, bias, [-3, 3])
    rule = lambda value: min(max(value, low), high) * scale + bias
    visible_inputs = [low - 3, low, (low + high) // 2, high + 3]
    hidden_inputs = [low - 8, low + 1, high - 1, high + 9]
    return CaseBundle(
        task_id=f"{split}_clamp_affine_{index:04d}",
        split=split,
        family="clamp_affine",
        current_source=source_clamp_affine(low, high, wrong_scale, wrong_bias),
        target_source=source_clamp_affine(low, high, scale, bias),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(low), str(high), str(scale), str(bias)],
        params={"low": low, "high": high, "scale": scale, "bias": bias},
        factors=["branching", "arithmetic"],
    )


def source_contains_length_prefix(needle: str, threshold: int, prefix: str, suffix: str) -> str:
    return module(
        f'''def apply_rule(value):
    text = str(value).strip().lower()
    if {needle!r} in text and len(text) >= {threshold}:
        return {prefix!r} + text
    return {suffix!r} + text
'''
    )


def make_contains_length_prefix(rng: random.Random, split: str, index: int) -> CaseBundle:
    needle = rng.choice(["key", "ax", "mid"])
    threshold = rng.choice([5, 7, 9])
    prefix = token(rng, "CL_")
    suffix = token(rng, "NC_")
    wrong_threshold = choose_changed(rng, threshold, [-2, 2])
    wrong_prefix = token(rng, "BAD_CL_")
    wrong_suffix = token(rng, "BAD_NC_")
    rule = lambda value: (prefix + str(value).strip().lower()) if needle in str(value).strip().lower() and len(str(value).strip().lower()) >= threshold else (suffix + str(value).strip().lower())
    visible_inputs = [f"{needle}-long", needle, "plain-value", f"pre-{needle.upper()}"]
    hidden_inputs = [f"left-{needle}-right", "short", f"{needle}xx", "ordinary"]
    return CaseBundle(
        task_id=f"{split}_contains_length_prefix_{index:04d}",
        split=split,
        family="contains_length_prefix",
        current_source=source_contains_length_prefix(needle, wrong_threshold, wrong_prefix, wrong_suffix),
        target_source=source_contains_length_prefix(needle, threshold, prefix, suffix),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[needle, str(threshold), prefix, suffix],
        params={"needle": needle, "threshold": threshold},
        factors=["string_match", "length", "branching", "string_format"],
    )


def source_sum_threshold_label(threshold: int, low_offset: int, high_offset: int) -> str:
    return module(
        f'''def apply_rule(value):
    total = sum(value)
    if total >= {threshold}:
        return total + {high_offset}
    return total + {low_offset}
'''
    )


def make_sum_threshold_label(rng: random.Random, split: str, index: int) -> CaseBundle:
    threshold = rng.choice([3, 5, 8])
    low_offset = rng.choice([-9, -4, 1])
    high_offset = rng.choice([4, 8, 12])
    wrong_low = choose_changed(rng, low_offset, [-3, 3])
    wrong_high = choose_changed(rng, high_offset, [-4, 4])
    rule = lambda values: sum(values) + (high_offset if sum(values) >= threshold else low_offset)
    visible_inputs = [[1, 1], [3, 4], [-2, 1, 2], [8, 2]]
    hidden_inputs = [[threshold, 0], [threshold - 4, 1], [10, -1], [-5, 2]]
    return CaseBundle(
        task_id=f"{split}_sum_threshold_label_{index:04d}",
        split=split,
        family="sum_threshold_label",
        current_source=source_sum_threshold_label(threshold, wrong_low, wrong_high),
        target_source=source_sum_threshold_label(threshold, low_offset, high_offset),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(threshold), str(low_offset), str(high_offset)],
        params={"threshold": threshold, "low_offset": low_offset, "high_offset": high_offset},
        factors=["aggregation", "branching", "arithmetic"],
    )


def source_sorted_join_holdout(prefix: str, suffix: str, separator: str) -> str:
    return module(
        f'''def apply_rule(value):
    pieces = sorted(str(part).strip().lower() for part in value)
    return {prefix!r} + {separator!r}.join(pieces) + {suffix!r}
'''
    )


def make_sorted_join_holdout(rng: random.Random, split: str, index: int) -> CaseBundle:
    prefix = token(rng, "SJ_")
    suffix = token(rng, "_JS")
    separator = rng.choice([".", "+", "~"])
    wrong_prefix = token(rng, "BAD_SJ_")
    wrong_suffix = token(rng, "_BJS")
    wrong_separator = choose_other(rng, separator, [".", "+", "~", "/"])

    def rule(value: tuple[str, ...]) -> str:
        return prefix + separator.join(sorted(str(part).strip().lower() for part in value)) + suffix

    visible_inputs = [("Beta", " alpha "), ("two", "One"), ("Z", "a", "m")]
    hidden_inputs = [("gamma", "beta"), ("Root", "leaf"), ("same", "Same", "alpha")]
    return CaseBundle(
        task_id=f"{split}_sorted_join_holdout_{index:04d}",
        split=split,
        family="sorted_join_holdout",
        current_source=source_sorted_join_holdout(wrong_prefix, wrong_suffix, wrong_separator),
        target_source=source_sorted_join_holdout(prefix, suffix, separator),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[prefix, suffix, separator],
        params={"prefix": prefix, "suffix": suffix, "separator": separator},
        factors=["string_normalization", "sequence_iteration", "sorting", "string_format"],
    )


def source_tuple_max_label_holdout(threshold: int, low_label: str, high_label: str, low_offset: int, high_offset: int) -> str:
    return module(
        f'''def apply_rule(value):
    left, right = value
    selected = left if left >= right else right
    if selected >= {threshold}:
        return {high_label!r} + str(selected + {high_offset})
    return {low_label!r} + str(selected + {low_offset})
'''
    )


def make_tuple_max_label_holdout(rng: random.Random, split: str, index: int) -> CaseBundle:
    threshold = rng.choice([3, 5, 8])
    low_label = token(rng, "MXL_")
    high_label = token(rng, "MXH_")
    low_offset = rng.choice([-6, -2, 1])
    high_offset = rng.choice([4, 7, 11])
    wrong_low_offset = choose_changed(rng, low_offset, [-3, 3])
    wrong_high_offset = choose_changed(rng, high_offset, [-4, 4])
    wrong_low_label = token(rng, "BAD_MXL_")
    wrong_high_label = token(rng, "BAD_MXH_")

    def rule(pair: tuple[int, int]) -> str:
        selected = pair[0] if pair[0] >= pair[1] else pair[1]
        return (high_label + str(selected + high_offset)) if selected >= threshold else (low_label + str(selected + low_offset))

    visible_inputs = [(4, 1), (2, 5), (threshold - 2, threshold - 1), (9, 3)]
    hidden_inputs = [(-1, 7), (8, 2), (0, 0), (threshold + 4, threshold + 5)]
    return CaseBundle(
        task_id=f"{split}_tuple_max_label_holdout_{index:04d}",
        split=split,
        family="tuple_max_label_holdout",
        current_source=source_tuple_max_label_holdout(threshold, wrong_low_label, wrong_high_label, wrong_low_offset, wrong_high_offset),
        target_source=source_tuple_max_label_holdout(threshold, low_label, high_label, low_offset, high_offset),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(threshold), low_label, high_label, str(low_offset), str(high_offset)],
        params={"threshold": threshold, "low_offset": low_offset, "high_offset": high_offset},
        factors=["tuple_access", "aggregation", "branching", "arithmetic", "string_format"],
    )


def source_parity_offset_holdout(even_offset: int, odd_offset: int) -> str:
    return module(
        f'''def apply_rule(value):
    if value % 2 == 0:
        return value + {even_offset}
    return value + {odd_offset}
'''
    )


def make_parity_offset_holdout(rng: random.Random, split: str, index: int) -> CaseBundle:
    even_offset = rng.choice([-12, -6, 4, 10, 16])
    odd_offset = rng.choice([-9, -3, 5, 11, 15])
    wrong_even = choose_changed(rng, even_offset, [-5, 5])
    wrong_odd = choose_changed(rng, odd_offset, [-4, 4])
    rule = lambda value: value + even_offset if value % 2 == 0 else value + odd_offset
    visible_inputs = [-2, -1, 0, 3, 6]
    hidden_inputs = [-5, 2, 7, 8, 11]
    return CaseBundle(
        task_id=f"{split}_parity_offset_holdout_{index:04d}",
        split=split,
        family="parity_offset_holdout",
        current_source=source_parity_offset_holdout(wrong_even, wrong_odd),
        target_source=source_parity_offset_holdout(even_offset, odd_offset),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(even_offset), str(odd_offset)],
        params={"even_offset": even_offset, "odd_offset": odd_offset},
        factors=["modulo", "branching", "arithmetic"],
    )


def source_contains_length_code_holdout(needle: str, threshold: int, prefix: str, fallback: str, bonus: int) -> str:
    return module(
        f'''def apply_rule(value):
    text = str(value).strip().lower()
    if {needle!r} in text and len(text) >= {threshold}:
        return {prefix!r} + str(len(text) + {bonus})
    return {fallback!r} + text[-2:]
'''
    )


def make_contains_length_code_holdout(rng: random.Random, split: str, index: int) -> CaseBundle:
    needle = rng.choice(["key", "ax", "mid"])
    threshold = rng.choice([6, 8, 10])
    prefix = token(rng, "OK_")
    fallback = token(rng, "NO_")
    bonus = rng.choice([2, 5, 9])
    wrong_threshold = choose_changed(rng, threshold, [-2, 2])
    wrong_prefix = token(rng, "BAD_OK_")
    wrong_fallback = token(rng, "BAD_NO_")
    wrong_bonus = choose_changed(rng, bonus, [-2, 2])

    def rule(value: str) -> str:
        text = str(value).strip().lower()
        if needle in text and len(text) >= threshold:
            return prefix + str(len(text) + bonus)
        return fallback + text[-2:]

    visible_inputs = [f"{needle}-long", needle, "plain-value", f"pre-{needle.upper()}-tail"]
    hidden_inputs = [f"left-{needle}-right", "short", f"{needle}xxxx", "ordinary"]
    return CaseBundle(
        task_id=f"{split}_contains_length_code_holdout_{index:04d}",
        split=split,
        family="contains_length_code_holdout",
        current_source=source_contains_length_code_holdout(needle, wrong_threshold, wrong_prefix, wrong_fallback, wrong_bonus),
        target_source=source_contains_length_code_holdout(needle, threshold, prefix, fallback, bonus),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[needle, str(threshold), prefix, fallback, str(bonus)],
        params={"needle": needle, "threshold": threshold, "bonus": bonus},
        factors=["string_match", "length", "branching", "arithmetic", "string_format"],
    )


def source_sum_parity_shift_holdout(even_offset: int, odd_offset: int) -> str:
    return module(
        f'''def apply_rule(value):
    total = sum(value)
    if total % 2 == 0:
        return total + {even_offset}
    return total + {odd_offset}
'''
    )


def make_sum_parity_shift_holdout(rng: random.Random, split: str, index: int) -> CaseBundle:
    even_offset = rng.choice([-8, -2, 6, 12])
    odd_offset = rng.choice([-7, -1, 5, 13])
    wrong_even = choose_changed(rng, even_offset, [-4, 4])
    wrong_odd = choose_changed(rng, odd_offset, [-3, 3])
    rule = lambda values: sum(values) + (even_offset if sum(values) % 2 == 0 else odd_offset)
    visible_inputs = [[1, 2], [2, 2], [-3, 4, 1], [9, -2]]
    hidden_inputs = [[-5, -1], [7, 8], [4, 4, 4], [10, -3, 2]]
    return CaseBundle(
        task_id=f"{split}_sum_parity_shift_holdout_{index:04d}",
        split=split,
        family="sum_parity_shift_holdout",
        current_source=source_sum_parity_shift_holdout(wrong_even, wrong_odd),
        target_source=source_sum_parity_shift_holdout(even_offset, odd_offset),
        visible_cases=[(value, rule(value)) for value in visible_inputs],
        hidden_cases=[(value, rule(value)) for value in hidden_inputs],
        markers=[str(even_offset), str(odd_offset)],
        params={"even_offset": even_offset, "odd_offset": odd_offset},
        factors=["aggregation", "modulo", "branching", "arithmetic"],
    )


SINGLETON_FAMILIES = [
    "affine_add",
    "scale_shift",
    "threshold_label",
    "modulo_label",
    "normalize_join",
    "replace_wrap",
    "contains_label",
    "length_label",
    "tuple_linear",
    "sum_offset",
]

COMPOSITE_FAMILIES = [
    "parity_labelled_shift",
    "dedupe_join_wrap",
    "tuple_min_offset",
    "clamp_affine",
    "contains_length_prefix",
    "sum_threshold_label",
]

MIXED_FAMILIES = [
    "affine_add",
    "contains_label",
    "length_label",
    "tuple_linear",
    "parity_labelled_shift",
    "dedupe_join_wrap",
    "tuple_min_offset",
    "sum_threshold_label",
]

HOLDOUT_FAMILIES = [
    "sorted_join_holdout",
    "tuple_max_label_holdout",
    "parity_offset_holdout",
    "contains_length_code_holdout",
    "sum_parity_shift_holdout",
]

BUILDERS: dict[str, Callable[[random.Random, str, int], CaseBundle]] = {
    "affine_add": make_affine_add,
    "scale_shift": make_scale_shift,
    "threshold_label": make_threshold_label,
    "modulo_label": make_modulo_label,
    "normalize_join": make_normalize_join,
    "replace_wrap": make_replace_wrap,
    "contains_label": make_contains_label,
    "length_label": make_length_label,
    "tuple_linear": make_tuple_linear,
    "sum_offset": make_sum_offset,
    "parity_labelled_shift": make_parity_labelled_shift,
    "dedupe_join_wrap": make_dedupe_join_wrap,
    "tuple_min_offset": make_tuple_min_offset,
    "clamp_affine": make_clamp_affine,
    "contains_length_prefix": make_contains_length_prefix,
    "sum_threshold_label": make_sum_threshold_label,
    "sorted_join_holdout": make_sorted_join_holdout,
    "tuple_max_label_holdout": make_tuple_max_label_holdout,
    "parity_offset_holdout": make_parity_offset_holdout,
    "contains_length_code_holdout": make_contains_length_code_holdout,
    "sum_parity_shift_holdout": make_sum_parity_shift_holdout,
}


def record_from_bundle(bundle: CaseBundle) -> dict[str, Any]:
    buggy_files = {"src/repair_target.py": BUGGY_FILE}
    current_files = {"src/repair_target.py": bundle.current_source}
    target_files = {"src/repair_target.py": bundle.target_source}
    visible_tests = {"tests/test_visible.py": test_file(bundle.visible_cases, visible=True)}
    hidden_tests = {"tests/test_hidden.py": test_file(bundle.hidden_cases, visible=False)}
    wrong_patch = unified_diff_for_files(buggy_files, current_files)
    base_buggy_diff = unified_diff_for_files(buggy_files, target_files)
    target_next_diff = unified_diff_for_files(current_files, target_files)

    wrong_visible = run_pytest(current_files, visible_tests, hidden_tests, which="visible")
    applied, repaired_files, apply_output = apply_patch_to_files(current_files, target_next_diff)
    target_visible = run_pytest(repaired_files, visible_tests, hidden_tests, which="visible") if applied else {"passed": False, "output": apply_output}
    target_hidden = run_pytest(repaired_files, visible_tests, hidden_tests, which="hidden") if applied else {"passed": False, "output": apply_output}
    syntax_ok, syntax_error = syntax_valid(repaired_files) if applied else (False, apply_output)

    if wrong_visible["passed"]:
        raise AssertionError(f"{bundle.task_id}: wrong implementation unexpectedly passed visible tests")
    if not applied:
        raise AssertionError(f"{bundle.task_id}: target diff did not apply: {apply_output}")
    if not target_visible["passed"] or not target_hidden["passed"]:
        raise AssertionError(f"{bundle.task_id}: target implementation did not pass tests")
    if not syntax_ok:
        raise AssertionError(f"{bundle.task_id}: target source has syntax error: {syntax_error}")

    visible_inputs = {repr(value) for value, _ in bundle.visible_cases}
    hidden_inputs = {repr(value) for value, _ in bundle.hidden_cases}
    if visible_inputs & hidden_inputs:
        raise AssertionError(f"{bundle.task_id}: visible and hidden inputs overlap")

    trace = wrong_visible["output"]
    missing_outputs = [
        repr(expected)
        for _, expected in bundle.visible_cases
        if repr(expected) not in trace
    ]
    if missing_outputs:
        raise AssertionError(f"{bundle.task_id}: expected outputs missing from trace: {missing_outputs}")

    return {
        "task_id": bundle.task_id,
        "episode_id": f"{bundle.task_id}::{DATASET_NAME}",
        "split": bundle.split,
        "issue": ISSUE,
        "buggy_files": buggy_files,
        "current_files": current_files,
        "visible_tests": visible_tests,
        "hidden_tests": hidden_tests,
        "wrong_patch": wrong_patch,
        "base_buggy_diff": base_buggy_diff,
        "target_next_diff": target_next_diff,
        "test_output_after_wrong_patch": trace,
        "metadata": {
            "bug_family": bundle.family,
            "failure_class": "counterexample_assertion",
            "factor_tags": bundle.factors,
            "visible_cases": bundle.visible_cases,
            "hidden_cases": bundle.hidden_cases,
            "target_markers": bundle.markers,
            "rule_params": bundle.params,
            "wrong_visible_passed": wrong_visible["passed"],
            "target_patch_applied": applied,
            "target_visible_passed": target_visible["passed"],
            "target_hidden_passed": target_hidden["passed"],
            "target_syntax_valid": syntax_ok,
            "target_syntax_error": syntax_error,
        },
    }


def make_records_for_families(
    rng: random.Random,
    *,
    split: str,
    families: list[str],
    per_family: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family in families:
        builder = BUILDERS[family]
        accepted = 0
        attempts = 0
        max_attempts = per_family * 100
        print(f"[build] {split}/{family}: target {per_family}", file=sys.stderr, flush=True)
        while accepted < per_family and attempts < max_attempts:
            attempts += 1
            try:
                records.append(record_from_bundle(builder(rng, split, attempts - 1)))
            except AssertionError:
                continue
            accepted += 1
            if accepted % 10 == 0 or accepted == per_family:
                print(
                    f"[build] {split}/{family}: accepted {accepted}/{per_family} after {attempts} attempts",
                    file=sys.stderr,
                    flush=True,
                )
        if accepted != per_family:
            raise AssertionError(f"{split}/{family}: accepted {accepted}/{per_family} after {attempts} attempts")
    rng.shuffle(records)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def family_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        family = row.get("metadata", {}).get("bug_family", "missing")
        counts[family] = counts.get(family, 0) + 1
    return dict(sorted(counts.items()))


def factor_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        for factor in row.get("metadata", {}).get("factor_tags", []):
            counts[factor] = counts.get(factor, 0) + 1
    return dict(sorted(counts.items()))


def build_split(
    *,
    key: str,
    output_dir: Path,
    filename: str,
    seed: int,
    families: list[str],
    per_family: int,
) -> tuple[Path, list[dict[str, Any]]]:
    rng = random.Random(seed)
    records = make_records_for_families(rng, split=key, families=families, per_family=per_family)
    path = output_dir / filename
    write_jsonl(path, records)
    return path, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--total-train-records", type=int, default=240)
    parser.add_argument("--singleton-val-per-family", type=int, default=4)
    parser.add_argument("--composite-val-per-family", type=int, default=6)
    parser.add_argument("--holdout-per-family", type=int, default=12)
    args = parser.parse_args()

    if args.total_train_records % len(SINGLETON_FAMILIES) != 0:
        raise ValueError("total train records must divide singleton family count")
    if args.total_train_records % len(COMPOSITE_FAMILIES) != 0:
        raise ValueError("total train records must divide composite family count")
    if args.total_train_records % len(MIXED_FAMILIES) != 0:
        raise ValueError("total train records must divide mixed family count")

    split_specs = [
        (
            "train_singletons",
            "repair_train_singletons.jsonl",
            args.seed + 101,
            SINGLETON_FAMILIES,
            args.total_train_records // len(SINGLETON_FAMILIES),
        ),
        (
            "train_composites",
            "repair_train_composites.jsonl",
            args.seed + 202,
            COMPOSITE_FAMILIES,
            args.total_train_records // len(COMPOSITE_FAMILIES),
        ),
        (
            "train_mixed",
            "repair_train_mixed.jsonl",
            args.seed + 303,
            MIXED_FAMILIES,
            args.total_train_records // len(MIXED_FAMILIES),
        ),
        (
            "val_singleton_iid",
            "repair_val_singleton_iid.jsonl",
            args.seed + 404,
            SINGLETON_FAMILIES,
            args.singleton_val_per_family,
        ),
        (
            "val_composite_iid",
            "repair_val_composite_iid.jsonl",
            args.seed + 505,
            COMPOSITE_FAMILIES,
            args.composite_val_per_family,
        ),
        (
            "val_recombination_holdout",
            "repair_val_recombination_holdout.jsonl",
            args.seed + 606,
            HOLDOUT_FAMILIES,
            args.holdout_per_family,
        ),
    ]

    paths: dict[str, Path] = {}
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    for key, filename, seed, families, per_family in split_specs:
        path, records = build_split(
            key=key,
            output_dir=args.output_dir,
            filename=filename,
            seed=seed,
            families=families,
            per_family=per_family,
        )
        paths[key] = path
        records_by_split[key] = records
        all_records.extend(records)

    paths["all"] = args.output_dir / "repair_all.jsonl"
    write_jsonl(paths["all"], all_records)

    manifest = {
        "dataset": DATASET_NAME,
        "seed": args.seed,
        "total_train_records_per_condition": args.total_train_records,
        "singleton_families": SINGLETON_FAMILIES,
        "composite_families": COMPOSITE_FAMILIES,
        "mixed_families": MIXED_FAMILIES,
        "holdout_families": HOLDOUT_FAMILIES,
        "records": {key: len(records) for key, records in records_by_split.items()} | {"all": len(all_records)},
        "family_counts": {key: family_counts(records) for key, records in records_by_split.items()} | {"all": family_counts(all_records)},
        "factor_counts": {key: factor_counts(records) for key, records in records_by_split.items()} | {"all": factor_counts(all_records)},
        "invariants": [
            "wrong-patched implementation fails visible counterexamples",
            "target corrective diff applies to the wrong-patched implementation",
            "target implementation passes visible and hidden tests",
            "hidden test inputs do not overlap visible trace inputs",
            "visible expected outputs appear in the failed execution trace",
            "all training conditions use the same record budget",
        ],
        "paths": {key: str(path) for key, path in paths.items()},
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
