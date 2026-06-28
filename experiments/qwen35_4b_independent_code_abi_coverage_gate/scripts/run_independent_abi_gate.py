#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset


EXPERIMENT = "qwen35_4b_independent_code_abi_coverage_gate"


@dataclass
class Example:
    assert_src: str
    call_expr: str
    args: tuple[Any, ...]
    expected: Any


@dataclass
class Candidate:
    name: str
    category: str
    depth: int
    func: Callable[..., Any]
    program: dict[str, Any]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def parse_assert(assertion: str) -> Example | None:
    try:
        tree = ast.parse(assertion)
    except SyntaxError:
        return None
    if not tree.body or not isinstance(tree.body[0], ast.Assert):
        return None
    test = tree.body[0].test
    if not isinstance(test, ast.Compare) or not isinstance(test.left, ast.Call) or not test.comparators:
        return None
    call = test.left
    if call.keywords:
        return None
    try:
        args = tuple(ast.literal_eval(arg) for arg in call.args)
        expected = ast.literal_eval(test.comparators[0])
    except Exception:
        return None
    return Example(assertion, ast.unparse(call), args, expected)


def parse_entry(assertion: str) -> str | None:
    parsed = parse_assert(assertion)
    if parsed is None:
        return None
    try:
        call = ast.parse(parsed.call_expr).body[0].value
    except Exception:
        return None
    func = call.func
    return func.id if isinstance(func, ast.Name) else None


def approx_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) <= 1e-6
        except Exception:
            return False
    return left == right


def safe_call(func: Callable[..., Any], args: tuple[Any, ...]) -> tuple[bool, Any]:
    try:
        return True, func(*args)
    except Exception as exc:
        return False, type(exc).__name__


def bounded_int(value: Any, limit: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or abs(value) > limit:
        raise ValueError("outside_bounded_abi_domain")
    return value


def as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return list(value)
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def add_candidate(rows: list[Candidate], name: str, category: str, depth: int, func: Callable[..., Any], **program: Any) -> None:
    rows.append(Candidate(name, category, depth, func, {"op": name, **program}))


def product(values: Any) -> Any:
    out = 1
    for item in as_list(values):
        out *= item
    return out


def unique_preserve(values: Any) -> list[Any]:
    out = []
    for item in as_list(values):
        if item not in out:
            out.append(item)
    return out


def flatten_one(values: Any) -> list[Any]:
    out = []
    for item in as_list(values):
        if isinstance(item, (list, tuple, set)):
            out.extend(list(item))
        else:
            out.append(item)
    return out


def flatten_deep(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        out: list[Any] = []
        for item in value:
            out.extend(flatten_deep(item))
        return out
    return [value]


def transpose(values: Any) -> list[list[Any]]:
    return [list(row) for row in zip(*values)]


def counter_dict(values: Any) -> dict[Any, int]:
    return dict(Counter(as_list(values)))


def counter_flat(values: Any) -> dict[Any, int]:
    return dict(Counter(flatten_deep(values)))


def most_common(values: Any, k: int | None = None) -> list[tuple[Any, int]]:
    return Counter(as_list(values)).most_common(k)


def all_distinct(values: Any) -> bool:
    seq = as_list(values)
    return len(seq) == len(set(map(repr, seq)))


def has_duplicate(values: Any) -> bool:
    return not all_distinct(values)


def count_pred(values: Any, pred: Callable[[Any], bool]) -> int:
    return sum(1 for item in as_list(values) if pred(item))


def filter_pred(values: Any, pred: Callable[[Any], bool]) -> list[Any]:
    return [item for item in as_list(values) if pred(item)]


def remove_present(left: Any, right: Any) -> list[Any]:
    banned = set(as_list(right))
    return [item for item in as_list(left) if item not in banned]


def contains_subsequence(values: Any, sub: Any) -> bool:
    seq = as_list(values)
    target = as_list(sub)
    return any(seq[i : i + len(target)] == target for i in range(max(0, len(seq) - len(target) + 1)))


def common_all(values: Any) -> list[Any]:
    rows = [as_list(row) for row in values]
    if not rows:
        return []
    common = set(rows[0])
    for row in rows[1:]:
        common &= set(row)
    return [item for item in rows[0] if item in common]


def sort_by_sum(values: Any) -> list[Any]:
    return sorted(values, key=sum)


def sort_by_len(values: Any) -> list[Any]:
    return sorted(values, key=len)


def sort_by_second(values: Any) -> list[Any]:
    return sorted(values, key=lambda item: item[1])


def column(values: Any, idx: int) -> list[Any]:
    return [row[idx] for row in values]


def zip_cycle(left: Any, right: Any) -> list[tuple[Any, Any]]:
    right_list = as_list(right)
    return [(item, right_list[i % len(right_list)]) for i, item in enumerate(as_list(left))]


def run_length_encode(values: Any) -> list[list[Any]]:
    seq = list(values) if isinstance(values, str) else as_list(values)
    if not seq:
        return []
    out = []
    prev = seq[0]
    count = 1
    for item in seq[1:]:
        if item == prev:
            count += 1
        else:
            out.append([count, prev])
            prev = item
            count = 1
    out.append([count, prev])
    return out


def recursive_sum(values: Any) -> Any:
    return sum(flatten_deep(values))


def sum_abs_pair_diffs(values: Any) -> Any:
    seq = as_list(values)
    if len(seq) > 500:
        raise ValueError("outside_bounded_abi_domain")
    return sum(abs(a - b) for i, a in enumerate(seq) for b in seq[i + 1 :])


def max_abs_diff(values: Any) -> Any:
    seq = as_list(values)
    return max(seq) - min(seq)


def lower(text: Any) -> str:
    return str(text).lower()


def upper(text: Any) -> str:
    return str(text).upper()


def title(text: Any) -> str:
    return str(text).title()


def reverse(text: Any) -> Any:
    if isinstance(text, str):
        return text[::-1]
    return list(reversed(text))


def words(text: Any) -> list[str]:
    return str(text).split()


def regex_findall(text: Any, pattern: str) -> list[str]:
    return re.findall(pattern, str(text))


def regex_split(text: Any, pattern: str) -> list[str]:
    return re.split(pattern, str(text))


def remove_regex(text: Any, pattern: str) -> str:
    return re.sub(pattern, "", str(text))


def compress_spaces(text: Any) -> str:
    return re.sub(r" +", " ", str(text))


def reverse_vowels(text: Any) -> str:
    chars = list(str(text))
    vowels = set("aeiouAEIOU")
    positions = [idx for idx, ch in enumerate(chars) if ch in vowels]
    values = [chars[idx] for idx in positions][::-1]
    for idx, value in zip(positions, values):
        chars[idx] = value
    return "".join(chars)


def tuple_join(value: Any) -> str:
    return "".join(str(item) for item in as_list(value))


def safe_pow(a: Any, b: Any) -> Any:
    left = bounded_int(a, 10_000)
    right = bounded_int(b, 12)
    return left**right


def safe_comb(n: Any, k: Any) -> int:
    n_int = bounded_int(n, 1_000)
    k_int = bounded_int(k, 1_000)
    return math.comb(n_int, k_int)


def factorial(n: Any) -> int:
    return math.factorial(bounded_int(n, 1_000))


def divisor_count(n: Any) -> int:
    n_int = bounded_int(n, 100_000)
    return sum(1 for d in range(1, n_int + 1) if n_int % d == 0)


def sum_common_divisors(a: Any, b: Any) -> int:
    left = bounded_int(a, 100_000)
    right = bounded_int(b, 100_000)
    g = math.gcd(left, right)
    return sum(d for d in range(1, g + 1) if left % d == 0 and right % d == 0)


def bit_count(n: Any) -> int:
    return bounded_int(n, 1_000_000).bit_count()


def next_power_two(n: Any) -> int:
    n_int = bounded_int(n, 1_000_000)
    out = 1
    while out < n_int:
        out *= 2
    return out


def int_to_binary_string(n: Any) -> str:
    return bin(bounded_int(n, 1_000_000))[2:]


def int_to_binary_int(n: Any) -> int:
    return int(int_to_binary_string(n))


def binary_to_int(n: Any) -> int:
    return int(str(n), 2)


def numeric_range_sum(n: Any, step: int) -> int:
    n_int = bounded_int(n, 1_000_000)
    return sum(range(n_int, 0, -step))


def triangle_number(n: Any) -> int:
    n_int = bounded_int(n, 1_000_000)
    return n_int * (n_int + 1) // 2


def polygonal_number(n: Any, sides: int) -> int:
    n_int = bounded_int(n, 1_000_000)
    return ((sides - 2) * n_int * n_int - (sides - 4) * n_int) // 2


def sphere_volume(r: Any) -> float:
    return 4 / 3 * math.pi * float(r) ** 3


def sphere_surface(r: Any) -> float:
    return 4 * math.pi * float(r) ** 2


def circle_circumference(r: Any) -> float:
    return 2 * math.pi * float(r)


def regular_polygon_area(n: Any, side: Any) -> float:
    n_int = bounded_int(n, 10_000)
    return n_int * float(side) ** 2 / (4 * math.tan(math.pi / n_int))


def label_bool(value: bool, yes: str, no: str) -> str:
    return yes if value else no


COMMON_LABELS = [
    ("Found a match!", "Not matched!"),
    ("Equal", "Not Equal"),
    ("All tuples have same length", "All tuples do not have same length"),
    ("Valid", "Invalid"),
    ("Yes", "No"),
    ("true", "false"),
]


def classify_task(text: str) -> str:
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|\d+", text.lower()))
    if tokens & {"regex", "string", "substring", "uppercase", "lowercase", "spaces", "split", "match", "characters", "vowels"}:
        return "string_regex"
    if tokens & {"dictionary", "dictionaries", "counter", "frequency", "frequencies", "occurrences"}:
        return "dict_counter"
    if tokens & {"list", "array", "tuple", "tuples", "sequence"}:
        return "list_tuple"
    if tokens & {"number", "integer", "sum", "product", "area", "perimeter", "volume", "gcd", "lcm", "factorial"}:
        return "numeric"
    if tokens & {"tree", "graph", "path", "prime", "palindrome", "permutation", "subsequence", "dynamic"}:
        return "algorithmic_control"
    return "other"


def bool_candidates_for_arg(rows: list[Candidate], i: int) -> None:
    bool_ops: list[tuple[str, Callable[..., bool]]] = [
        ("is_empty", lambda *args, i=i: len(args[i]) == 0),
        ("is_nonempty", lambda *args, i=i: len(args[i]) > 0),
        ("all_distinct", lambda *args, i=i: all_distinct(args[i])),
        ("has_duplicate", lambda *args, i=i: has_duplicate(args[i])),
        ("is_sorted_asc", lambda *args, i=i: as_list(args[i]) == sorted(as_list(args[i]))),
        ("is_sorted_desc", lambda *args, i=i: as_list(args[i]) == sorted(as_list(args[i]), reverse=True)),
        ("is_even", lambda *args, i=i: args[i] % 2 == 0),
        ("is_odd", lambda *args, i=i: args[i] % 2 == 1),
    ]
    for name, func in bool_ops:
        add_candidate(rows, f"{name}_arg{i}", "predicate", 1, func, arg=i)
        for yes, no in COMMON_LABELS:
            add_candidate(
                rows,
                f"{name}_arg{i}_label_{yes[:3]}_{no[:3]}",
                "predicate_label",
                2,
                lambda *args, func=func, yes=yes, no=no: label_bool(bool(func(*args)), yes, no),
                arg=i,
                yes=yes,
                no=no,
            )


def generate_candidates(max_arity: int) -> list[Candidate]:
    rows: list[Candidate] = []
    regex_patterns = {
        "digits": r"\d+",
        "letters": r"[A-Za-z]+",
        "words": r"\w+",
        "uppercase_runs": r"[A-Z]+",
        "lowercase_runs": r"[a-z]+",
    }

    for i in range(max_arity):
        add_candidate(rows, f"arg{i}", "generic", 0, lambda *args, i=i: args[i], arg=i)
        add_candidate(rows, f"len_arg{i}", "generic", 1, lambda *args, i=i: len(args[i]), arg=i)
        bool_candidates_for_arg(rows, i)

        unary_ops: list[tuple[str, str, int, Callable[..., Any]]] = [
            ("sum", "numeric", 1, lambda *args, i=i: sum(args[i])),
            ("product", "numeric", 1, lambda *args, i=i: product(args[i])),
            ("min", "numeric", 1, lambda *args, i=i: min(args[i])),
            ("max", "numeric", 1, lambda *args, i=i: max(args[i])),
            ("mean", "numeric", 1, lambda *args, i=i: sum(args[i]) / len(args[i])),
            ("sorted", "list_tuple", 1, lambda *args, i=i: sorted(args[i])),
            ("sorted_desc", "list_tuple", 1, lambda *args, i=i: sorted(args[i], reverse=True)),
            ("reverse", "list_tuple", 1, lambda *args, i=i: reverse(args[i])),
            ("unique", "list_tuple", 1, lambda *args, i=i: unique_preserve(args[i])),
            ("flatten_one", "list_tuple", 1, lambda *args, i=i: flatten_one(args[i])),
            ("flatten_deep", "list_tuple", 2, lambda *args, i=i: flatten_deep(args[i])),
            ("transpose", "list_tuple", 2, lambda *args, i=i: transpose(args[i])),
            ("counter_dict", "dict_counter", 1, lambda *args, i=i: counter_dict(args[i])),
            ("counter_flat", "dict_counter", 2, lambda *args, i=i: counter_flat(args[i])),
            ("most_common_all", "dict_counter", 1, lambda *args, i=i: most_common(args[i])),
            ("count_positive", "numeric", 1, lambda *args, i=i: count_pred(args[i], lambda x: x > 0)),
            ("count_negative", "numeric", 1, lambda *args, i=i: count_pred(args[i], lambda x: x < 0)),
            ("count_even", "numeric", 1, lambda *args, i=i: count_pred(args[i], lambda x: x % 2 == 0)),
            ("count_odd", "numeric", 1, lambda *args, i=i: count_pred(args[i], lambda x: x % 2 == 1)),
            ("filter_even", "list_tuple", 1, lambda *args, i=i: filter_pred(args[i], lambda x: x % 2 == 0)),
            ("filter_odd", "list_tuple", 1, lambda *args, i=i: filter_pred(args[i], lambda x: x % 2 == 1)),
            ("recursive_sum", "list_tuple", 2, lambda *args, i=i: recursive_sum(args[i])),
            ("sum_abs_pair_diffs", "list_tuple", 2, lambda *args, i=i: sum_abs_pair_diffs(args[i])),
            ("max_abs_diff", "list_tuple", 1, lambda *args, i=i: max_abs_diff(args[i])),
            ("sort_by_sum", "list_tuple", 2, lambda *args, i=i: sort_by_sum(args[i])),
            ("sort_by_len", "list_tuple", 1, lambda *args, i=i: sort_by_len(args[i])),
            ("sort_by_second", "list_tuple", 1, lambda *args, i=i: sort_by_second(args[i])),
            ("common_all", "list_tuple", 2, lambda *args, i=i: common_all(args[i])),
            ("run_length_encode", "list_tuple", 1, lambda *args, i=i: run_length_encode(args[i])),
            ("lower", "string_regex", 1, lambda *args, i=i: lower(args[i])),
            ("upper", "string_regex", 1, lambda *args, i=i: upper(args[i])),
            ("title", "string_regex", 1, lambda *args, i=i: title(args[i])),
            ("split_words", "string_regex", 1, lambda *args, i=i: words(args[i])),
            ("compress_spaces", "string_regex", 1, lambda *args, i=i: compress_spaces(args[i])),
            ("reverse_vowels", "string_regex", 1, lambda *args, i=i: reverse_vowels(args[i])),
            ("tuple_join", "string_regex", 1, lambda *args, i=i: tuple_join(args[i])),
            ("remove_digits", "string_regex", 1, lambda *args, i=i: remove_regex(args[i], r"\d+")),
            ("remove_letters", "string_regex", 1, lambda *args, i=i: remove_regex(args[i], r"[A-Za-z]+")),
            ("factorial", "numeric", 1, lambda *args, i=i: factorial(args[i])),
            ("divisor_count", "numeric", 2, lambda *args, i=i: divisor_count(args[i])),
            ("bit_count", "numeric", 1, lambda *args, i=i: bit_count(args[i])),
            ("next_power_two", "numeric", 1, lambda *args, i=i: next_power_two(args[i])),
            ("int_to_binary_string", "numeric", 1, lambda *args, i=i: int_to_binary_string(args[i])),
            ("int_to_binary_int", "numeric", 1, lambda *args, i=i: int_to_binary_int(args[i])),
            ("binary_to_int", "numeric", 1, lambda *args, i=i: binary_to_int(args[i])),
            ("triangle_number", "numeric", 1, lambda *args, i=i: triangle_number(args[i])),
            ("square_number", "numeric", 1, lambda *args, i=i: args[i] * args[i]),
            ("pentagonal_number", "numeric", 1, lambda *args, i=i: polygonal_number(args[i], 5)),
            ("hexagonal_number", "numeric", 1, lambda *args, i=i: polygonal_number(args[i], 6)),
            ("sphere_volume", "numeric", 1, lambda *args, i=i: sphere_volume(args[i])),
            ("sphere_surface", "numeric", 1, lambda *args, i=i: sphere_surface(args[i])),
            ("circle_circumference", "numeric", 1, lambda *args, i=i: circle_circumference(args[i])),
        ]
        for op, category, depth, func in unary_ops:
            add_candidate(rows, f"{op}_arg{i}", category, depth, func, arg=i)
        for label, pattern in regex_patterns.items():
            add_candidate(rows, f"regex_findall_{label}_arg{i}", "string_regex", 1, lambda *args, i=i, pattern=pattern: regex_findall(args[i], pattern), arg=i, pattern=pattern)
            add_candidate(rows, f"regex_split_{label}_arg{i}", "string_regex", 1, lambda *args, i=i, pattern=pattern: regex_split(args[i], pattern), arg=i, pattern=pattern)

    if max_arity >= 2:
        for i, j in itertools.permutations(range(max_arity), 2):
            pair_ops: list[tuple[str, str, int, Callable[..., Any]]] = [
                ("add", "numeric", 1, lambda *args, i=i, j=j: args[i] + args[j]),
                ("sub", "numeric", 1, lambda *args, i=i, j=j: args[i] - args[j]),
                ("mul", "numeric", 1, lambda *args, i=i, j=j: args[i] * args[j]),
                ("floordiv", "numeric", 1, lambda *args, i=i, j=j: args[i] // args[j]),
                ("truediv", "numeric", 1, lambda *args, i=i, j=j: args[i] / args[j]),
                ("mod", "numeric", 1, lambda *args, i=i, j=j: args[i] % args[j]),
                ("pow", "numeric", 1, lambda *args, i=i, j=j: safe_pow(args[i], args[j])),
                ("gcd", "numeric", 1, lambda *args, i=i, j=j: math.gcd(args[i], args[j])),
                ("lcm", "numeric", 1, lambda *args, i=i, j=j: math.lcm(args[i], args[j])),
                ("absdiff", "numeric", 1, lambda *args, i=i, j=j: abs(args[i] - args[j])),
                ("comb", "numeric", 2, lambda *args, i=i, j=j: safe_comb(args[i], args[j])),
                ("sum_common_divisors", "numeric", 2, lambda *args, i=i, j=j: sum_common_divisors(args[i], args[j])),
                ("count", "list_tuple", 1, lambda *args, i=i, j=j: args[i].count(args[j])),
                ("contains", "list_tuple", 1, lambda *args, i=i, j=j: args[j] in args[i]),
                ("contains_subsequence", "list_tuple", 2, lambda *args, i=i, j=j: contains_subsequence(args[i], args[j])),
                ("remove_present", "list_tuple", 1, lambda *args, i=i, j=j: remove_present(args[i], args[j])),
                ("column", "list_tuple", 1, lambda *args, i=i, j=j: column(args[i], args[j])),
                ("zip_cycle", "list_tuple", 1, lambda *args, i=i, j=j: zip_cycle(args[i], args[j])),
                ("str_count", "string_regex", 1, lambda *args, i=i, j=j: str(args[i]).count(str(args[j]))),
                ("str_remove", "string_regex", 1, lambda *args, i=i, j=j: str(args[i]).replace(str(args[j]), "")),
                ("regular_polygon_area", "numeric", 2, lambda *args, i=i, j=j: regular_polygon_area(args[i], args[j])),
            ]
            for op, category, depth, func in pair_ops:
                add_candidate(rows, f"{op}_{i}_{j}", category, depth, func, args=[i, j])
            for yes, no in COMMON_LABELS:
                add_candidate(rows, f"contains_{i}_{j}_label_{yes[:3]}_{no[:3]}", "predicate_label", 2, lambda *args, i=i, j=j, yes=yes, no=no: label_bool(args[j] in args[i], yes, no), args=[i, j], yes=yes, no=no)

    if max_arity >= 3:
        for i, j, k in itertools.permutations(range(max_arity), 3):
            ternary_ops: list[tuple[str, str, int, Callable[..., Any]]] = [
                ("slice_sum_inclusive", "list_tuple", 2, lambda *args, i=i, j=j, k=k: sum(as_list(args[i])[args[j] : args[k] + 1])),
                ("index", "list_tuple", 1, lambda *args, i=i, j=j, k=k: args[i][args[k] - 1]),
                ("linear_combo", "numeric", 2, lambda *args, i=i, j=j, k=k: args[i] * args[j] + args[k]),
                ("box_volume", "numeric", 1, lambda *args, i=i, j=j, k=k: args[i] * args[j] * args[k]),
            ]
            for op, category, depth, func in ternary_ops:
                add_candidate(rows, f"{op}_{i}_{j}_{k}", category, depth, func, args=[i, j, k])

    if max_arity >= 2:
        add_candidate(rows, "merge_dicts_left_to_right", "dict_counter", 2, lambda *args: {k: v for d in args for k, v in d.items()}, args="all")
        add_candidate(rows, "merge_dicts_right_to_left", "dict_counter", 2, lambda *args: {k: v for d in reversed(args) for k, v in d.items()}, args="all")
        add_candidate(rows, "merge_sorted_flatten_args", "list_tuple", 2, lambda *args: sorted(flatten_deep(args)), args="all")

    return rows


def candidate_results(candidate: Candidate, examples: list[Example]) -> list[dict[str, Any]]:
    rows = []
    for ex in examples:
        ok, value = safe_call(candidate.func, ex.args)
        rows.append({"ok": ok, "value_repr": repr(value), "passed": ok and approx_equal(value, ex.expected)})
    return rows


def load_records(split: str, count: int, offset: int, visible_tests: int, task_ids: list[int] | None = None) -> list[dict[str, Any]]:
    dataset = list(load_dataset("google-research-datasets/mbpp")[split])
    if task_ids is not None:
        id_set = set(task_ids)
        selected = [raw for raw in dataset if raw["task_id"] in id_set]
    else:
        selected = dataset[offset : offset + count]
    rows = []
    for raw in selected:
        tests = list(raw.get("test_list") or [])
        if not tests:
            continue
        parsed = [parse_assert(test) for test in tests + list(raw.get("challenge_test_list") or [])]
        examples = [ex for ex in parsed if ex is not None]
        visible_examples = [ex for ex in [parse_assert(test) for test in tests[:visible_tests]] if ex is not None]
        rows.append(
            {
                "record_id": f"mbpp_{split}_{raw['task_id']}",
                "task_id": raw["task_id"],
                "task_text": raw["text"],
                "entry_point": parse_entry(tests[0]) or "unknown",
                "examples": examples,
                "visible_examples": visible_examples,
                "parseable": len(examples) == len(tests) + len(raw.get("challenge_test_list") or []),
            }
        )
    return rows


def run_task(record: dict[str, Any]) -> dict[str, Any]:
    examples: list[Example] = record["examples"]
    visible_count = len(record["visible_examples"])
    max_arity = max((len(ex.args) for ex in examples), default=0)
    candidates = generate_candidates(max_arity)
    full_winners: list[Candidate] = []
    visible_winners: list[Candidate] = []
    visible_hidden_wrong = 0
    candidate_rows = []
    for cand in candidates:
        results = candidate_results(cand, examples)
        visible_pass = bool(visible_count) and all(row["passed"] for row in results[:visible_count])
        full_pass = bool(results) and all(row["passed"] for row in results)
        if visible_pass:
            visible_winners.append(cand)
            if not full_pass:
                visible_hidden_wrong += 1
        if full_pass:
            full_winners.append(cand)
        if visible_pass or full_pass:
            candidate_rows.append(
                {
                    "name": cand.name,
                    "category": cand.category,
                    "depth": cand.depth,
                    "program": cand.program,
                    "visible_pass": visible_pass,
                    "full_pass": full_pass,
                    "outputs": [row["value_repr"] for row in results],
                }
            )
    first_visible = visible_winners[0] if visible_winners else None
    return {
        "record_id": record["record_id"],
        "task_id": record["task_id"],
        "task_text": record["task_text"],
        "entry_point": record["entry_point"],
        "slice": classify_task(record["task_text"]),
        "parseable": record["parseable"],
        "example_count": len(examples),
        "visible_count": visible_count,
        "candidate_count": len(candidates),
        "visible_consistent_count": len(visible_winners),
        "visible_hidden_wrong_count": visible_hidden_wrong,
        "full_winner_count": len(full_winners),
        "oracle_covered": bool(full_winners),
        "visible_any": bool(visible_winners),
        "visible_false_pass": bool(visible_winners) and not bool(full_winners),
        "first_visible_full_pass": first_visible is not None and any(first_visible.name == win.name for win in full_winners),
        "winning_program": full_winners[0].program if full_winners else None,
        "winning_category": full_winners[0].category if full_winners else None,
        "winning_depth": full_winners[0].depth if full_winners else None,
        "visible_or_winning_candidates": candidate_rows[:12],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        return {
            "n": len(group),
            "parseable": sum(row["parseable"] for row in group),
            "oracle_covered": sum(row["oracle_covered"] for row in group),
            "oracle_coverage": sum(row["oracle_covered"] for row in group) / len(group),
            "visible_any": sum(row["visible_any"] for row in group),
            "visible_false_pass": sum(row["visible_false_pass"] for row in group),
            "visible_false_pass_rate_among_visible": sum(row["visible_false_pass"] for row in group) / max(1, sum(row["visible_any"] for row in group)),
            "visible_hidden_wrong_candidates": sum(row["visible_hidden_wrong_count"] for row in group),
            "visible_hidden_wrong_rate_among_candidates": sum(row["visible_hidden_wrong_count"] for row in group) / max(1, sum(row["visible_consistent_count"] for row in group)),
            "first_visible_full_pass": sum(row["first_visible_full_pass"] for row in group),
            "candidate_count_mean": sum(row["candidate_count"] for row in group) / len(group),
            "visible_consistent_mean": sum(row["visible_consistent_count"] for row in group) / len(group),
        }

    slices = sorted(set(row["slice"] for row in rows))
    depth_counts = {str(k): v for k, v in sorted(Counter(row["winning_depth"] for row in rows if row["oracle_covered"]).items())}
    return {
        "experiment": EXPERIMENT,
        "overall": metrics(rows),
        "by_slice": {slice_name: metrics([row for row in rows if row["slice"] == slice_name]) for slice_name in slices},
        "winning_depth_counts": depth_counts,
        "covered_task_ids": [row["task_id"] for row in rows if row["oracle_covered"]],
        "uncovered_task_ids": [row["task_id"] for row in rows if not row["oracle_covered"]],
    }


def run_named_slice(split: str, count: int, offset: int, visible_tests: int, out_prefix: str, out_dir: Path) -> dict[str, Any]:
    records = load_records(split, count, offset, visible_tests)
    rows = [run_task(record) for record in records]
    summary = summarize(rows)
    summary.update({"split": split, "count": count, "offset": offset, "visible_tests": visible_tests, "name": out_prefix})
    write_jsonl(out_dir / f"{out_prefix}_records.jsonl", rows)
    write_json(out_dir / f"{out_prefix}_summary.json", summary)
    return summary


def run_sweep(exclude_first: int, sample_count: int, seeds: list[int], visible_tests: int, out_dir: Path) -> dict[str, Any]:
    test_rows = list(load_dataset("google-research-datasets/mbpp")["test"])[exclude_first:]
    results = []
    all_rows = []
    for seed in seeds:
        sample = random.Random(seed).sample(test_rows, min(sample_count, len(test_rows)))
        records = load_records("test", len(sample), 0, visible_tests, task_ids=[row["task_id"] for row in sample])
        rows = [run_task(record) for record in records]
        summary = summarize(rows)
        summary.update({"seed": seed, "sample_count": len(rows), "exclude_first": exclude_first, "visible_tests": visible_tests})
        results.append(summary)
        for row in rows:
            row = dict(row)
            row["sweep_seed"] = seed
            all_rows.append(row)
    sweep = {
        "results": results,
        "coverage_values": [item["overall"]["oracle_coverage"] for item in results],
        "covered_counts": [item["overall"]["oracle_covered"] for item in results],
        "mean_coverage": sum(item["overall"]["oracle_coverage"] for item in results) / len(results),
        "min_coverage": min(item["overall"]["oracle_coverage"] for item in results),
        "max_coverage": max(item["overall"]["oracle_coverage"] for item in results),
    }
    write_json(out_dir / "sweep_summary.json", sweep)
    write_jsonl(out_dir / "sweep_records.jsonl", all_rows)
    return sweep


def inventory_summary(max_arity: int) -> dict[str, Any]:
    candidates = generate_candidates(max_arity)
    return {
        "max_arity": max_arity,
        "candidate_count": len(candidates),
        "by_category": dict(sorted(Counter(c.category for c in candidates).items())),
        "by_depth": {str(k): v for k, v in sorted(Counter(c.depth for c in candidates).items())},
        "sample_programs": [c.program for c in candidates[:40]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-offset", type=int, default=0)
    parser.add_argument("--calibration-count", type=int, default=160)
    parser.add_argument("--heldout-offset", type=int, default=160)
    parser.add_argument("--heldout-count", type=int, default=160)
    parser.add_argument("--visible-tests", type=int, default=1)
    parser.add_argument("--sweep-exclude-first", type=int, default=160)
    parser.add_argument("--sweep-sample-count", type=int, default=160)
    parser.add_argument("--sweep-seeds", type=int, nargs="+", default=[11, 17, 23])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    data_dir = args.data_dir
    reports_dir = args.reports_dir
    calibration = run_named_slice("test", args.calibration_count, args.calibration_offset, args.visible_tests, "calibration", data_dir)
    heldout = run_named_slice("test", args.heldout_count, args.heldout_offset, args.visible_tests, "heldout", data_dir)
    train = run_named_slice("train", 374, 0, args.visible_tests, "train", data_dir)
    sweep = run_sweep(args.sweep_exclude_first, args.sweep_sample_count, args.sweep_seeds, args.visible_tests, data_dir)
    combined = {
        "calibration": calibration,
        "heldout": heldout,
        "train": train,
        "sweep": sweep,
        "coverage_drop": calibration["overall"]["oracle_coverage"] - heldout["overall"]["oracle_coverage"],
        "inventory": inventory_summary(4),
    }
    write_json(reports_dir / "coverage_gate_summary.json", combined)
    print(json.dumps(combined, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
