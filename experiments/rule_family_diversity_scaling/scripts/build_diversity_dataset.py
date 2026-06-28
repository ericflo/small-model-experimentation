#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from repair_experiment.patching import apply_patch_to_files, unified_diff_for_files  # noqa: E402
from repair_experiment.runner import run_pytest, syntax_valid  # noqa: E402


BUGGY_FILE = '''"""Repair target for rule-family diversity scaling experiments."""


def apply_rule(value):
    raise NotImplementedError("validator rule has not been implemented")
'''

ISSUE = (
    "The apply_rule function must match the validator's transformation. "
    "The repository does not state the exact rule. Use the failed-test "
    "counterexamples to infer a compact general rule, then patch the implementation "
    "so it also passes unseen inputs."
)

TRAIN_FAMILIES = [
    "affine_int",
    "threshold_label",
    "slug_affix",
    "abs_shift",
    "clamp_offset",
    "tuple_linear",
    "length_label",
    "contains_label",
    "prefix_switch",
    "modulo_label",
    "sign_piece",
    "replace_wrap",
]

SCALES = {
    "scale3": TRAIN_FAMILIES[:3],
    "scale6": TRAIN_FAMILIES[:6],
    "scale12": TRAIN_FAMILIES[:12],
}

HOLDOUT_FAMILIES = [
    "parity_offset_holdout",
    "quadratic_shift_holdout",
    "tuple_max_holdout",
    "sorted_join_holdout",
]


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


def module(body: str) -> str:
    return f'"""Repair target for rule-family diversity scaling experiments."""\n\n\n{body.rstrip()}\n'


def token(rng: random.Random, prefix: str, length: int = 4, *, lower: bool = False) -> str:
    alphabet = string.ascii_lowercase if lower else string.ascii_uppercase + string.digits
    return prefix + "".join(rng.choice(alphabet) for _ in range(length))


def choose_changed(rng: random.Random, value: int, deltas: list[int]) -> int:
    return value + rng.choice([delta for delta in deltas if delta != 0])


def label_source(label: str) -> str:
    return repr(label)


def test_file(cases: list[tuple[Any, Any]], *, visible: bool) -> str:
    name = "visible" if visible else "hidden"
    return f'''from src.repair_target import apply_rule


CASES = {cases!r}


def test_{name}_counterexamples():
    failures = []
    for value, expected in CASES:
        actual = apply_rule(value)
        if actual != expected:
            failures.append(
                f"COUNTEREXAMPLE input={{value!r}} expected={{expected!r}} actual={{actual!r}}"
            )
    assert not failures, "\\n".join(failures)
'''


def source_affine(slope: int, intercept: int) -> str:
    return module(f'''def apply_rule(value):
    return {slope} * value + {intercept}
''')


def make_affine(rng: random.Random, split: str, index: int) -> CaseBundle:
    if split == "val_format_holdout":
        slope = rng.choice([-11, -8, 9, 13])
        intercept = rng.choice([-29, -17, 23, 41])
    else:
        slope = rng.choice([2, 3, 4, 5, 6, 7])
        intercept = rng.choice([-9, -5, -2, 3, 6, 9])
    wrong_slope = choose_changed(rng, slope, [-3, -2, 2, 3])
    wrong_intercept = choose_changed(rng, intercept, [-7, -4, 4, 7])
    visible_inputs = [0, 1, 3]
    hidden_inputs = [-2, 2, 5, 8]
    visible = [(x, slope * x + intercept) for x in visible_inputs]
    hidden = [(x, slope * x + intercept) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_affine_{index:04d}",
        split=split,
        family="affine_int",
        current_source=source_affine(wrong_slope, wrong_intercept),
        target_source=source_affine(slope, intercept),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[str(slope), str(intercept)],
        params={"slope": slope, "intercept": intercept, "wrong_slope": wrong_slope, "wrong_intercept": wrong_intercept},
    )


def source_threshold(threshold: int, low_label: str, high_label: str) -> str:
    return module(f'''def apply_rule(value):
    if value < {threshold}:
        return {label_source(low_label)}
    return {label_source(high_label)}
''')


def make_threshold(rng: random.Random, split: str, index: int) -> CaseBundle:
    if split == "val_format_holdout":
        threshold = rng.choice([-8, -5, 11, 14])
        low_label = token(rng, "low:", 5, lower=True)
        high_label = token(rng, "hi:", 5, lower=True)
    else:
        threshold = rng.choice([-3, -1, 2, 4, 6, 9])
        low_label = token(rng, "LOW_", 4)
        high_label = token(rng, "HIGH_", 4)
    wrong_threshold = choose_changed(rng, threshold, [-3, -2, 2, 3])
    wrong_low = token(rng, "BADL_", 4)
    wrong_high = token(rng, "BADH_", 4)
    rule = lambda x: low_label if x < threshold else high_label
    visible_inputs = [threshold - 2, threshold - 1, threshold, threshold + 2]
    hidden_inputs = [threshold - 5, threshold + 1, threshold + 4, threshold + 7]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_threshold_{index:04d}",
        split=split,
        family="threshold_label",
        current_source=source_threshold(wrong_threshold, wrong_low, wrong_high),
        target_source=source_threshold(threshold, low_label, high_label),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[str(threshold), low_label, high_label],
        params={"threshold": threshold, "low_label": low_label, "high_label": high_label, "wrong_threshold": wrong_threshold},
    )


def slug_value(value: str, prefix: str, suffix: str, separator: str) -> str:
    pieces = [piece for piece in value.strip().lower().replace("_", " ").split() if piece]
    return prefix + separator.join(pieces) + suffix


def source_slug(prefix: str, suffix: str, separator: str) -> str:
    return module(f'''def apply_rule(value):
    text = str(value).strip().lower().replace("_", " ")
    pieces = [piece for piece in text.split() if piece]
    body = {separator!r}.join(pieces)
    return {prefix!r} + body + {suffix!r}
''')


def make_slug(rng: random.Random, split: str, index: int) -> CaseBundle:
    if split == "val_format_holdout":
        prefix = rng.choice(["pre:", "[[", "tag."]) + token(rng, "", 3, lower=True)
        suffix = rng.choice([":done", "]]", ".ok"]) + token(rng, "", 2, lower=True)
        separator = rng.choice(["::", ".", "~"])
    else:
        prefix = token(rng, "P_", 4)
        suffix = token(rng, "_S", 4)
        separator = rng.choice(["-", "_", "+"])
    wrong_prefix = token(rng, "WP_", 4)
    wrong_suffix = token(rng, "_WS", 4)
    wrong_separator = rng.choice(["/", "|", "#"])
    visible_inputs = ["  Alpha Beta  ", "MIXED_case Word", "two   spaces"]
    hidden_inputs = ["New Input Value", "already_clean", "  Edge   CASE_test  "]
    visible = [(text, slug_value(text, prefix, suffix, separator)) for text in visible_inputs]
    hidden = [(text, slug_value(text, prefix, suffix, separator)) for text in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_slug_{index:04d}",
        split=split,
        family="slug_affix",
        current_source=source_slug(wrong_prefix, wrong_suffix, wrong_separator),
        target_source=source_slug(prefix, suffix, separator),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[prefix, suffix, separator],
        params={"prefix": prefix, "suffix": suffix, "separator": separator, "wrong_separator": wrong_separator},
    )


def source_abs_shift(factor: int, offset: int) -> str:
    return module(f'''def apply_rule(value):
    return {factor} * abs(value) + {offset}
''')


def make_abs_shift(rng: random.Random, split: str, index: int) -> CaseBundle:
    factor = rng.choice([2, 3, 5, 7, 9])
    offset = rng.choice([-8, -3, 4, 10, 15])
    wrong_factor = choose_changed(rng, factor, [-2, -1, 1, 2])
    wrong_offset = choose_changed(rng, offset, [-6, -3, 3, 6])
    visible_inputs = [-3, 0, 4]
    hidden_inputs = [-7, 2, 5, 9]
    visible = [(x, factor * abs(x) + offset) for x in visible_inputs]
    hidden = [(x, factor * abs(x) + offset) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_abs_{index:04d}",
        split=split,
        family="abs_shift",
        current_source=source_abs_shift(wrong_factor, wrong_offset),
        target_source=source_abs_shift(factor, offset),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[str(factor), str(offset)],
        params={"factor": factor, "offset": offset, "wrong_factor": wrong_factor, "wrong_offset": wrong_offset},
    )


def source_clamp(lo: int, hi: int, offset: int) -> str:
    return module(f'''def apply_rule(value):
    if value < {lo}:
        core = {lo}
    elif value > {hi}:
        core = {hi}
    else:
        core = value
    return core + {offset}
''')


def make_clamp(rng: random.Random, split: str, index: int) -> CaseBundle:
    lo = rng.choice([-8, -5, -2, 0])
    width = rng.choice([5, 7, 9])
    hi = lo + width
    offset = rng.choice([-4, 3, 6, 11])
    wrong_lo = choose_changed(rng, lo, [-2, 2])
    wrong_hi = choose_changed(rng, hi, [-2, 2])
    wrong_offset = choose_changed(rng, offset, [-3, 3])

    def rule(x: int) -> int:
        return min(max(x, lo), hi) + offset

    visible_inputs = [lo - 3, lo, hi, hi + 4]
    hidden_inputs = [lo - 1, lo + 2, hi - 1, hi + 7]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_clamp_{index:04d}",
        split=split,
        family="clamp_offset",
        current_source=source_clamp(wrong_lo, wrong_hi, wrong_offset),
        target_source=source_clamp(lo, hi, offset),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[str(lo), str(hi), str(offset)],
        params={"lo": lo, "hi": hi, "offset": offset, "wrong_lo": wrong_lo, "wrong_hi": wrong_hi},
    )


def source_tuple_linear(a: int, b: int, c: int) -> str:
    return module(f'''def apply_rule(value):
    left, right = value
    return {a} * left + {b} * right + {c}
''')


def make_tuple_linear(rng: random.Random, split: str, index: int) -> CaseBundle:
    a = rng.choice([2, 3, 4, 5])
    b = rng.choice([-3, -2, 2, 4])
    c = rng.choice([-7, -1, 5, 9])
    wrong_a = choose_changed(rng, a, [-1, 1, 2])
    wrong_b = choose_changed(rng, b, [-2, -1, 1, 2])
    wrong_c = choose_changed(rng, c, [-4, 4])
    rule = lambda pair: a * pair[0] + b * pair[1] + c
    visible_inputs = [(0, 0), (1, 2), (3, -1)]
    hidden_inputs = [(-2, 4), (5, 1), (2, 2), (4, -3)]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_tuple_linear_{index:04d}",
        split=split,
        family="tuple_linear",
        current_source=source_tuple_linear(wrong_a, wrong_b, wrong_c),
        target_source=source_tuple_linear(a, b, c),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[str(a), str(b), str(c)],
        params={"a": a, "b": b, "c": c, "wrong_a": wrong_a, "wrong_b": wrong_b, "wrong_c": wrong_c},
    )


def source_length_label(cutoff: int, short_label: str, long_label: str) -> str:
    return module(f'''def apply_rule(value):
    n = len(str(value).strip())
    if n <= {cutoff}:
        return {short_label!r}
    return {long_label!r}
''')


def make_length_label(rng: random.Random, split: str, index: int) -> CaseBundle:
    cutoff = rng.choice([3, 4, 5, 7])
    short_label = token(rng, "SHORT_", 4)
    long_label = token(rng, "LONG_", 4)
    wrong_cutoff = choose_changed(rng, cutoff, [-2, -1, 1, 2])
    wrong_short = token(rng, "BADSHORT_", 3)
    wrong_long = token(rng, "BADLONG_", 3)
    rule = lambda text: short_label if len(str(text).strip()) <= cutoff else long_label
    visible_inputs = ["a", "abcd", "alphabet", "  xy  "]
    hidden_inputs = ["tool", "longer value", "", "seven77"]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs if repr(x) not in {repr(v) for v in visible_inputs}]
    return CaseBundle(
        task_id=f"{split}_length_{index:04d}",
        split=split,
        family="length_label",
        current_source=source_length_label(wrong_cutoff, wrong_short, wrong_long),
        target_source=source_length_label(cutoff, short_label, long_label),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[str(cutoff), short_label, long_label],
        params={"cutoff": cutoff, "short_label": short_label, "long_label": long_label, "wrong_cutoff": wrong_cutoff},
    )


def source_contains(marker: str, hit_label: str, miss_label: str) -> str:
    return module(f'''def apply_rule(value):
    text = str(value).lower()
    if {marker!r} in text:
        return {hit_label!r}
    return {miss_label!r}
''')


def make_contains(rng: random.Random, split: str, index: int) -> CaseBundle:
    marker = rng.choice(["ax", "zen", "core", "mix"])
    hit_label = token(rng, "HIT_", 4)
    miss_label = token(rng, "MISS_", 4)
    wrong_marker = rng.choice([item for item in ["bad", "none", "zz", "old"] if item != marker])
    wrong_hit = token(rng, "BADH_", 4)
    wrong_miss = token(rng, "BADM_", 4)
    rule = lambda text: hit_label if marker in str(text).lower() else miss_label
    visible_inputs = [f"{marker} start", "plain text", f"mid-{marker}-word"]
    hidden_inputs = ["nothing here", f"caps {marker.upper()} token", "separate"]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_contains_{index:04d}",
        split=split,
        family="contains_label",
        current_source=source_contains(wrong_marker, wrong_hit, wrong_miss),
        target_source=source_contains(marker, hit_label, miss_label),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[marker, hit_label, miss_label],
        params={"marker": marker, "hit_label": hit_label, "miss_label": miss_label, "wrong_marker": wrong_marker},
    )


def source_prefix(marker: str, yes: str, no: str) -> str:
    return module(f'''def apply_rule(value):
    text = str(value).strip().lower()
    if text.startswith({marker!r}):
        return {yes!r} + text
    return {no!r} + text
''')


def make_prefix(rng: random.Random, split: str, index: int) -> CaseBundle:
    marker = rng.choice(["pre", "go", "id", "tag"])
    yes = token(rng, "YES_", 3)
    no = token(rng, "NO_", 3)
    wrong_marker = rng.choice(["old", "xx", "start"])
    wrong_yes = token(rng, "WY_", 3)
    wrong_no = token(rng, "WN_", 3)
    rule = lambda text: (yes if str(text).strip().lower().startswith(marker) else no) + str(text).strip().lower()
    visible_inputs = [f"{marker}Alpha", " other", f"{marker}-case"]
    hidden_inputs = ["misc", f" {marker} hidden ", "nothing"]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_prefix_{index:04d}",
        split=split,
        family="prefix_switch",
        current_source=source_prefix(wrong_marker, wrong_yes, wrong_no),
        target_source=source_prefix(marker, yes, no),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[marker, yes, no],
        params={"marker": marker, "yes": yes, "no": no, "wrong_marker": wrong_marker},
    )


def source_modulo(labels: list[str]) -> str:
    return module(f'''def apply_rule(value):
    remainder = value % 3
    if remainder == 0:
        return {labels[0]!r}
    if remainder == 1:
        return {labels[1]!r}
    return {labels[2]!r}
''')


def make_modulo(rng: random.Random, split: str, index: int) -> CaseBundle:
    labels = [token(rng, f"R{idx}_", 4) for idx in range(3)]
    wrong_labels = [token(rng, f"WR{idx}_", 4) for idx in range(3)]
    rule = lambda x: labels[x % 3]
    visible_inputs = [0, 1, 2, 3]
    hidden_inputs = [4, 5, 6, -1]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_modulo_{index:04d}",
        split=split,
        family="modulo_label",
        current_source=source_modulo(wrong_labels),
        target_source=source_modulo(labels),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=labels,
        params={"labels": labels, "wrong_labels": wrong_labels},
    )


def source_sign(neg_delta: int, zero_label: str, pos_delta: int) -> str:
    return module(f'''def apply_rule(value):
    if value < 0:
        return value - {neg_delta}
    if value == 0:
        return {zero_label!r}
    return value + {pos_delta}
''')


def make_sign(rng: random.Random, split: str, index: int) -> CaseBundle:
    neg_delta = rng.choice([2, 4, 7, 9])
    zero_label = token(rng, "ZERO_", 4)
    pos_delta = rng.choice([3, 5, 8, 11])
    wrong_neg = choose_changed(rng, neg_delta, [-2, 2])
    wrong_zero = token(rng, "BADZ_", 4)
    wrong_pos = choose_changed(rng, pos_delta, [-2, 2])

    def rule(x: int) -> int | str:
        if x < 0:
            return x - neg_delta
        if x == 0:
            return zero_label
        return x + pos_delta

    visible_inputs = [-3, 0, 4]
    hidden_inputs = [-8, -1, 2, 9]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_sign_{index:04d}",
        split=split,
        family="sign_piece",
        current_source=source_sign(wrong_neg, wrong_zero, wrong_pos),
        target_source=source_sign(neg_delta, zero_label, pos_delta),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[str(neg_delta), zero_label, str(pos_delta)],
        params={"neg_delta": neg_delta, "zero_label": zero_label, "pos_delta": pos_delta},
    )


def source_replace(old: str, new: str, prefix: str, suffix: str) -> str:
    return module(f'''def apply_rule(value):
    text = str(value).strip().lower().replace({old!r}, {new!r})
    return {prefix!r} + text + {suffix!r}
''')


def make_replace(rng: random.Random, split: str, index: int) -> CaseBundle:
    old = rng.choice([" ", "_", "-"])
    new = rng.choice([".", "+", "~"])
    prefix = token(rng, "RW_", 3)
    suffix = token(rng, "_OK", 3)
    wrong_old = rng.choice(["x", "/", ":"])
    wrong_new = rng.choice(["/", "|", "#"])
    wrong_prefix = token(rng, "BAD_", 3)
    wrong_suffix = token(rng, "_BAD", 3)
    rule = lambda text: prefix + str(text).strip().lower().replace(old, new) + suffix
    visible_inputs = ["Alpha Beta", "MIXED_case", "two-parts"]
    hidden_inputs = ["New Value", "already_clean", "edge-case value"]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_replace_{index:04d}",
        split=split,
        family="replace_wrap",
        current_source=source_replace(wrong_old, wrong_new, wrong_prefix, wrong_suffix),
        target_source=source_replace(old, new, prefix, suffix),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[old, new, prefix, suffix],
        params={"old": old, "new": new, "prefix": prefix, "suffix": suffix},
    )


def source_parity(even_offset: int, odd_offset: int) -> str:
    return module(f'''def apply_rule(value):
    if value % 2 == 0:
        return value + {even_offset}
    return value + {odd_offset}
''')


def make_parity(rng: random.Random, split: str, index: int) -> CaseBundle:
    even_offset = rng.choice([-12, -6, 4, 10, 14])
    odd_offset = rng.choice([-9, -3, 5, 11, 17])
    wrong_even = choose_changed(rng, even_offset, [-5, -2, 2, 5])
    wrong_odd = choose_changed(rng, odd_offset, [-6, -3, 3, 6])
    rule = lambda x: x + even_offset if x % 2 == 0 else x + odd_offset
    visible_inputs = [0, 1, 4, 7]
    hidden_inputs = [-3, 2, 8, 11]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_parity_{index:04d}",
        split=split,
        family="parity_offset_holdout",
        current_source=source_parity(wrong_even, wrong_odd),
        target_source=source_parity(even_offset, odd_offset),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[str(even_offset), str(odd_offset)],
        params={"even_offset": even_offset, "odd_offset": odd_offset, "wrong_even": wrong_even, "wrong_odd": wrong_odd},
    )


def source_quadratic(a: int, b: int, c: int) -> str:
    return module(f'''def apply_rule(value):
    return {a} * value * value + {b} * value + {c}
''')


def make_quadratic(rng: random.Random, split: str, index: int) -> CaseBundle:
    a = rng.choice([1, 2, 3])
    b = rng.choice([-4, -2, 3, 5])
    c = rng.choice([-7, -1, 4, 8])
    wrong_a = choose_changed(rng, a, [-1, 1])
    wrong_b = choose_changed(rng, b, [-2, 2])
    wrong_c = choose_changed(rng, c, [-3, 3])
    rule = lambda x: a * x * x + b * x + c
    visible_inputs = [-1, 0, 1, 2]
    hidden_inputs = [-3, 3, 4, 5]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_quadratic_{index:04d}",
        split=split,
        family="quadratic_shift_holdout",
        current_source=source_quadratic(wrong_a, wrong_b, wrong_c),
        target_source=source_quadratic(a, b, c),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[str(a), str(b), str(c)],
        params={"a": a, "b": b, "c": c, "wrong_a": wrong_a, "wrong_b": wrong_b, "wrong_c": wrong_c},
    )


def source_tuple_max(left_offset: int, right_offset: int) -> str:
    return module(f'''def apply_rule(value):
    left, right = value
    if left >= right:
        return left + {left_offset}
    return right + {right_offset}
''')


def make_tuple_max(rng: random.Random, split: str, index: int) -> CaseBundle:
    left_offset = rng.choice([-5, 2, 6, 10])
    right_offset = rng.choice([-3, 4, 8, 12])
    wrong_left = choose_changed(rng, left_offset, [-2, 2])
    wrong_right = choose_changed(rng, right_offset, [-3, 3])
    rule = lambda pair: (pair[0] + left_offset) if pair[0] >= pair[1] else (pair[1] + right_offset)
    visible_inputs = [(4, 1), (2, 5), (3, 3)]
    hidden_inputs = [(-1, 7), (8, 2), (0, 0), (5, 9)]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_tuple_max_{index:04d}",
        split=split,
        family="tuple_max_holdout",
        current_source=source_tuple_max(wrong_left, wrong_right),
        target_source=source_tuple_max(left_offset, right_offset),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[str(left_offset), str(right_offset)],
        params={"left_offset": left_offset, "right_offset": right_offset, "wrong_left": wrong_left, "wrong_right": wrong_right},
    )


def source_sorted_join(prefix: str, suffix: str, separator: str) -> str:
    return module(f'''def apply_rule(value):
    pieces = sorted(str(part).strip().lower() for part in value)
    body = {separator!r}.join(pieces)
    return {prefix!r} + body + {suffix!r}
''')


def make_sorted_join(rng: random.Random, split: str, index: int) -> CaseBundle:
    prefix = token(rng, "SJ_", 3)
    suffix = token(rng, "_END", 2)
    separator = rng.choice([".", "+", "~"])
    wrong_prefix = token(rng, "BADJ_", 3)
    wrong_suffix = token(rng, "_WR", 2)
    wrong_separator = rng.choice(["/", "|", "#"])

    def rule(parts: tuple[str, str]) -> str:
        return prefix + separator.join(sorted(str(part).strip().lower() for part in parts)) + suffix

    visible_inputs = [("Beta", "alpha"), ("two", "One"), ("Z", "a")]
    hidden_inputs = [("gamma", "beta"), ("Root", "leaf"), ("same", "Same")]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    return CaseBundle(
        task_id=f"{split}_sorted_join_{index:04d}",
        split=split,
        family="sorted_join_holdout",
        current_source=source_sorted_join(wrong_prefix, wrong_suffix, wrong_separator),
        target_source=source_sorted_join(prefix, suffix, separator),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[prefix, suffix, separator],
        params={"prefix": prefix, "suffix": suffix, "separator": separator, "wrong_separator": wrong_separator},
    )


BUILDERS: dict[str, Callable[[random.Random, str, int], CaseBundle]] = {
    "affine_int": make_affine,
    "threshold_label": make_threshold,
    "slug_affix": make_slug,
    "abs_shift": make_abs_shift,
    "clamp_offset": make_clamp,
    "tuple_linear": make_tuple_linear,
    "length_label": make_length_label,
    "contains_label": make_contains,
    "prefix_switch": make_prefix,
    "modulo_label": make_modulo,
    "sign_piece": make_sign,
    "replace_wrap": make_replace,
    "parity_offset_holdout": make_parity,
    "quadratic_shift_holdout": make_quadratic,
    "tuple_max_holdout": make_tuple_max,
    "sorted_join_holdout": make_sorted_join,
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
        raise AssertionError(f"{bundle.task_id}: wrong patch unexpectedly passed visible tests")
    if not applied:
        raise AssertionError(f"{bundle.task_id}: target diff did not apply: {apply_output}")
    if not target_visible["passed"] or not target_hidden["passed"]:
        raise AssertionError(f"{bundle.task_id}: target did not pass tests")
    if not syntax_ok:
        raise AssertionError(f"{bundle.task_id}: repaired source has syntax error: {syntax_error}")
    visible_inputs = {repr(value) for value, _ in bundle.visible_cases}
    hidden_inputs = {repr(value) for value, _ in bundle.hidden_cases}
    if visible_inputs & hidden_inputs:
        raise AssertionError(f"{bundle.task_id}: hidden inputs overlap visible inputs")

    trace = wrong_visible["output"]
    missing_outputs = [
        repr(expected)
        for _, expected in bundle.visible_cases
        if repr(expected) not in trace
    ]
    if missing_outputs:
        raise AssertionError(f"{bundle.task_id}: visible expected outputs missing from trace: {missing_outputs}")

    return {
        "task_id": bundle.task_id,
        "episode_id": f"{bundle.task_id}::rule_family_diversity",
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
        max_attempts = per_family * 80
        while accepted < per_family and attempts < max_attempts:
            attempts += 1
            try:
                records.append(record_from_bundle(builder(rng, split, attempts - 1)))
            except AssertionError:
                continue
            accepted += 1
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-train-records", type=int, default=240)
    parser.add_argument("--base-iid-per-family", type=int, default=12)
    parser.add_argument("--format-per-family", type=int, default=12)
    parser.add_argument("--holdout-per-family", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()

    paths: dict[str, Path] = {}
    all_records: list[dict[str, Any]] = []
    manifest_records: dict[str, int] = {}
    manifest_counts: dict[str, dict[str, int]] = {}

    for scale, families in SCALES.items():
        if args.total_train_records % len(families) != 0:
            raise ValueError(f"total train records must divide family count for {scale}")
        rng = random.Random(args.seed + len(families) * 101)
        per_family = args.total_train_records // len(families)
        records = make_records_for_families(
            rng,
            split=f"train_{scale}",
            families=families,
            per_family=per_family,
        )
        key = f"train_{scale}"
        paths[key] = args.output_dir / f"repair_train_{scale}.jsonl"
        write_jsonl(paths[key], records)
        all_records.extend(records)
        manifest_records[key] = len(records)
        manifest_counts[key] = family_counts(records)

    rng = random.Random(args.seed + 7001)
    val_base_iid = make_records_for_families(
        rng,
        split="val_base_iid",
        families=SCALES["scale3"],
        per_family=args.base_iid_per_family,
    )
    paths["val_base_iid"] = args.output_dir / "repair_val_base_iid.jsonl"
    write_jsonl(paths["val_base_iid"], val_base_iid)
    all_records.extend(val_base_iid)
    manifest_records["val_base_iid"] = len(val_base_iid)
    manifest_counts["val_base_iid"] = family_counts(val_base_iid)

    rng = random.Random(args.seed + 8001)
    val_base_format = make_records_for_families(
        rng,
        split="val_format_holdout",
        families=SCALES["scale3"],
        per_family=args.format_per_family,
    )
    paths["val_format_holdout"] = args.output_dir / "repair_val_format_holdout.jsonl"
    write_jsonl(paths["val_format_holdout"], val_base_format)
    all_records.extend(val_base_format)
    manifest_records["val_format_holdout"] = len(val_base_format)
    manifest_counts["val_format_holdout"] = family_counts(val_base_format)

    rng = random.Random(args.seed + 9001)
    val_rule_holdout = make_records_for_families(
        rng,
        split="val_rule_holdout",
        families=HOLDOUT_FAMILIES,
        per_family=args.holdout_per_family,
    )
    paths["val_rule_holdout"] = args.output_dir / "repair_val_rule_holdout.jsonl"
    write_jsonl(paths["val_rule_holdout"], val_rule_holdout)
    all_records.extend(val_rule_holdout)
    manifest_records["val_rule_holdout"] = len(val_rule_holdout)
    manifest_counts["val_rule_holdout"] = family_counts(val_rule_holdout)

    paths["all"] = args.output_dir / "repair_all.jsonl"
    write_jsonl(paths["all"], all_records)
    manifest_records["all"] = len(all_records)
    manifest_counts["all"] = family_counts(all_records)

    manifest = {
        "dataset": "rule_family_diversity_scaling",
        "seed": args.seed,
        "total_train_records_per_scale": args.total_train_records,
        "scales": SCALES,
        "train_families": TRAIN_FAMILIES,
        "holdout_families": HOLDOUT_FAMILIES,
        "records": manifest_records,
        "family_counts": manifest_counts,
        "invariants": [
            "wrong-patched implementation fails visible counterexamples",
            "target corrective diff applies to the wrong-patched implementation",
            "target implementation passes visible and hidden tests",
            "hidden test inputs do not overlap visible trace inputs",
            "visible expected outputs appear in the failed execution trace",
            "each diversity scale has the same total training record count",
        ],
        "paths": {key: str(value) for key, value in paths.items()},
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
