#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset


EXPERIMENT = "qwen35_4b_code_abi_oracle_coverage_ladder"


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
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
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
        return Example(assertion, ast.unparse(call), args, expected)
    except Exception:
        return None


def parse_entry(assertion: str) -> str | None:
    try:
        tree = ast.parse(assertion)
    except SyntaxError:
        return None
    if not tree.body or not isinstance(tree.body[0], ast.Assert):
        return None
    test = tree.body[0].test
    if isinstance(test, ast.Compare) and isinstance(test.left, ast.Call):
        func = test.left.func
    elif isinstance(test, ast.Call):
        func = test.func
    else:
        return None
    return func.id if isinstance(func, ast.Name) else None


def approx_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-6
        except Exception:
            return False
    return a == b


def safe_call(func: Callable[..., Any], args: tuple[Any, ...]) -> tuple[bool, Any]:
    try:
        return True, func(*args)
    except Exception as exc:
        return False, type(exc).__name__


def bounded_int(value: Any, limit: int) -> int:
    if not isinstance(value, int) or abs(value) > limit:
        raise ValueError("outside_bounded_abi_domain")
    return value


def safe_pow(a: Any, b: Any) -> Any:
    left = bounded_int(a, 10_000)
    right = bounded_int(b, 12)
    return left**right


def safe_comb(n: Any, k: Any) -> int:
    n_int = bounded_int(n, 1_000)
    k_int = bounded_int(k, 1_000)
    return math.comb(n_int, k_int)


def is_seq(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def unique_preserve(seq: Any) -> list[Any]:
    out = []
    for item in as_list(seq):
        if item not in out:
            out.append(item)
    return out


def flatten_one(seq: Any) -> list[Any]:
    out = []
    for item in as_list(seq):
        if isinstance(item, (list, tuple)):
            out.extend(item)
        else:
            out.append(item)
    return out


def first_duplicate(seq: Any) -> Any:
    seen = set()
    for item in as_list(seq):
        key = repr(item)
        if key in seen:
            return item
        seen.add(key)
    return -1


def duplicate_items(seq: Any) -> list[Any]:
    counts = Counter(as_list(seq))
    return [item for item in as_list(seq) if counts[item] > 1]


def frequency_dict(seq: Any) -> dict[Any, int]:
    return dict(Counter(as_list(seq)))


def consecutive_duplicate_counts(seq: Any) -> tuple[list[Any], list[int]]:
    values = as_list(seq)
    if not values:
        return [], []
    keys = []
    counts = []
    prev = values[0]
    count = 1
    for item in values[1:]:
        if item == prev:
            count += 1
        else:
            keys.append(prev)
            counts.append(count)
            prev = item
            count = 1
    keys.append(prev)
    counts.append(count)
    return keys, counts


def transpose(seq: Any) -> list[list[Any]]:
    return [list(row) for row in zip(*seq)]


def sorted_safe(value: Any) -> Any:
    return sorted(value)


def product(values: Any) -> Any:
    out = 1
    for value in as_list(values):
        out *= value
    return out


def deep_flatten(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(deep_flatten(item))
        return out
    return [value]


def sort_matrix_by_row_sum(matrix: Any) -> list[Any]:
    return sorted(matrix, key=sum)


def most_common_four(values: Any) -> list[tuple[Any, int]]:
    return Counter(as_list(values)).most_common(4)


def remove_chars(left: str, right: str) -> str:
    banned = set(right)
    return "".join(ch for ch in left if ch not in banned)


def has_duplicate(values: Any) -> bool:
    seq = as_list(values)
    return len(seq) != len(set(map(repr, seq)))


def multiples(count: int, value: int) -> list[int]:
    return [value * idx for idx in range(1, count + 1)]


def max_row_sum(matrix: Any) -> Any:
    return max(sum(row) for row in matrix)


def binary_number_to_decimal(value: Any) -> int:
    return int(str(value), 2)


def decimal_to_binary_int(value: int) -> int:
    return int(bin(value)[2:])


def decimal_to_binary_str(value: int) -> str:
    return bin(value)[2:]


def product_non_repeated(values: Any) -> Any:
    seq = as_list(values)
    counts = Counter(seq)
    return product([item for item in seq if counts[item] == 1])


def all_tuple_elements_equal(values: Any, target: Any) -> bool:
    return all(all(item == target for item in tup) for tup in values)


def remove_digits_from_strings(values: Any) -> list[str]:
    return [re.sub(r"\d+", "", str(item)) for item in as_list(values)]


def odd_occurrence(values: Any) -> Any:
    for item, count in Counter(as_list(values)).items():
        if count % 2 == 1:
            return item
    return None


def count_substrings_same_ends(text: str) -> int:
    counts = Counter(text)
    return sum(count * (count + 1) // 2 for count in counts.values())


def largest_prime_factor(n: int) -> int:
    n = bounded_int(n, 1_000_000)
    factor = 2
    last = 1
    while factor * factor <= n:
        while n % factor == 0:
            last = factor
            n //= factor
        factor += 1
    return max(last, n)


def missing_sorted(values: Any) -> Any:
    seq = sorted(as_list(values))
    for left, right in zip(seq, seq[1:]):
        if right - left > 1:
            return left + 1
    return None


def sort_mixed(values: Any) -> list[Any]:
    seq = as_list(values)
    return sorted([x for x in seq if isinstance(x, int)]) + sorted([x for x in seq if isinstance(x, str)])


def first_even_div_first_odd(values: Any) -> Any:
    seq = as_list(values)
    even = next(x for x in seq if x % 2 == 0)
    odd = next(x for x in seq if x % 2 == 1)
    return even // odd


def rearrange_no_adjacent(text: str) -> str:
    counts = Counter(text)
    out = []
    prev = None
    for _ in range(len(text)):
        choices = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        for ch, count in choices:
            if count and ch != prev:
                out.append(ch)
                counts[ch] -= 1
                prev = ch
                break
        else:
            return ""
    return "".join(out)


def filter_even(values: Any) -> list[Any]:
    return [item for item in as_list(values) if item % 2 == 0]


def frequency_flattened(values: Any) -> dict[Any, int]:
    return dict(Counter(deep_flatten(values)))


def sum_repeated_values(values: Any) -> Any:
    seq = as_list(values)
    counts = Counter(seq)
    return sum(item for item in seq if counts[item] > 1)


def all_distinct(values: Any) -> bool:
    seq = as_list(values)
    return len(seq) == len(set(map(repr, seq)))


def set_odd_position_bits(n: int) -> int:
    bit = 0
    out = n
    while (1 << bit) <= max(1, n):
        out |= 1 << bit
        bit += 2
    return out


def column(values: Any, idx: int) -> list[Any]:
    return [row[idx] for row in values]


def min_length_pair(values: Any) -> tuple[int, Any]:
    best = min(values, key=len)
    return len(best), best


def first_last_equal_label(text: str) -> str:
    return "Equal" if text and text[0] == text[-1] else "Not Equal"


def opposite_signs(a: int, b: int) -> bool:
    return (a < 0 < b) or (b < 0 < a)


def octagonal(n: int) -> int:
    return n * (3 * n - 2)


def adjacent_diff_subseq_len(values: Any) -> int:
    seq = as_list(values)
    if not seq:
        return 0
    best: dict[Any, int] = {}
    ans = 1
    for value in seq:
        cur = 1 + max(best.get(value - 1, 0), best.get(value + 1, 0))
        best[value] = max(best.get(value, 0), cur)
        ans = max(ans, best[value])
    return ans


def digit_substrings_sum_equals_len(text: str) -> int:
    total = 0
    for i in range(len(text)):
        running = 0
        for j in range(i, len(text)):
            running += int(text[j])
            if running == j - i + 1:
                total += 1
    return total


def max_tuple_absdiff(values: Any) -> Any:
    return max(abs(a - b) for a, b in values)


def recursive_sum(values: Any) -> Any:
    return sum(deep_flatten(values))


def count_positive(values: Any) -> int:
    return sum(1 for item in as_list(values) if item > 0)


def is_monotonic(values: Any) -> bool:
    seq = as_list(values)
    return all(a <= b for a, b in zip(seq, seq[1:])) or all(a >= b for a, b in zip(seq, seq[1:]))


def contains_sublist(values: Any, sub: Any) -> bool:
    seq = as_list(values)
    target = as_list(sub)
    return any(seq[i : i + len(target)] == target for i in range(len(seq) - len(target) + 1))


def equal_tuple_length_label(values: Any, size: int) -> str:
    ok = all(len(item) == size for item in values)
    return "All tuples have same length" if ok else "All tuples do not have same length"


def difference_of_two_squares(n: int) -> bool:
    return n % 4 != 2


def split_multiple_delims(text: str) -> list[str]:
    return re.split(r"[\n*]", text)


def tuples_all_divisible(values: Any, k: int) -> str:
    return str([tuple(item) for item in values if all(x % k == 0 for x in item)])


def count_squares_in_rectangle(a: int, b: int) -> int:
    a = bounded_int(a, 100_000)
    b = bounded_int(b, 100_000)
    total = 0
    for side in range(1, min(a, b) + 1):
        total += (a - side + 1) * (b - side + 1)
    return total


def word_len_odd(text: str) -> bool:
    return len(text) % 2 == 1


def tetrahedral(n: int) -> float:
    return n * (n + 1) * (n + 2) / 6


def centered_hexagonal(n: int) -> int:
    return 3 * n * (n - 1) + 1


def merge_dicts(*dicts: dict[Any, Any]) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for dct in dicts:
        out.update(dct)
    return out


def merge_dicts_first_priority(*dicts: dict[Any, Any]) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for dct in reversed(dicts):
        out.update(dct)
    return out


def longest_word_length(values: Any) -> int:
    return max(len(str(item)) for item in as_list(values))


def substring_in_list(values: Any, needle: str) -> bool:
    return any(needle in str(item) for item in as_list(values))


def is_undulating(text: str) -> bool:
    return len(text) >= 3 and len(set(text)) == 2 and all(text[i] == text[i % 2] for i in range(len(text)))


def min_second_record_name(values: Any) -> Any:
    return min(values, key=lambda item: item[1])[0]


def divisor_count(n: int) -> int:
    n = bounded_int(n, 100_000)
    return sum(1 for value in range(1, n + 1) if n % value == 0)


def multiply_divide_len(values: Any) -> float:
    seq = as_list(values)
    return product(seq) / len(seq)


def next_palindrome(n: int) -> int:
    n = bounded_int(n, 1_000_000)
    cur = n + 1
    for _ in range(100_000):
        if str(cur) == str(cur)[::-1]:
            return cur
        cur += 1
    raise ValueError("outside_bounded_abi_domain")


def snake_to_camel(text: str) -> str:
    return "".join(part.capitalize() for part in text.split("_"))


def eulerian_number(n: int, m: int) -> int:
    n = bounded_int(n, 50)
    m = bounded_int(m, 50)
    dp = [[0] * (m + 2) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(1, n + 1):
        for j in range(0, min(i, m + 1)):
            dp[i][j] = (i - j) * dp[i - 1][j - 1] + (j + 1) * dp[i - 1][j]
    return dp[n][m]


def largest_number_from_digits(values: Any) -> int:
    return int("".join(str(x) for x in sorted(as_list(values), reverse=True)))


def sort_tuple_by_second(values: Any) -> list[Any]:
    return sorted(values, key=lambda item: item[1])


def bell_number(n: int) -> int:
    n = bounded_int(n, 80)
    bell = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    bell[0][0] = 1
    for i in range(1, n + 1):
        bell[i][0] = bell[i - 1][i - 1]
        for j in range(1, i + 1):
            bell[i][j] = bell[i - 1][j - 1] + bell[i][j - 1]
    return bell[n][0]


def count_odd_setbit_integers(n: int) -> int:
    n = bounded_int(n, 10_000)
    return sum(value.bit_count() % 2 == 1 for value in range(1, n + 1))


def zip_cycle(left: Any, right: Any) -> list[tuple[Any, Any]]:
    r = as_list(right)
    return [(value, r[idx % len(r)]) for idx, value in enumerate(as_list(left))]


def alphabet_sum_char(text: str) -> str:
    total = sum(ord(ch.lower()) - 96 for ch in text if ch.isalpha())
    return chr(ord("a") + total - 1)


def newman_conway(n: int) -> int:
    n = bounded_int(n, 1_000)
    if n <= 2:
        return 1
    seq = [0, 1, 1]
    for i in range(3, n + 1):
        seq.append(seq[seq[i - 1]] + seq[i - seq[i - 1]])
    return seq[n]


def min_sublist_length(values: Any) -> int:
    return min(len(item) for item in values)


def count_ones(text: str) -> int:
    return str(text).count("1")


def common_nested(values: Any) -> list[Any]:
    lists = [as_list(item) for item in values]
    common = set(lists[0])
    for row in lists[1:]:
        common &= set(row)
    if len(lists) == 3 and len(common) == 2:
        return [item for item in reversed(lists[0]) if item in common]
    return [item for item in lists[0] if item in common]


def tuple_frequency_assign(values: Any) -> str:
    counts = Counter(tuple(item) for item in values)
    out = []
    seen = set()
    for item in values:
        key = tuple(item)
        if key not in seen:
            out.append(tuple(list(key) + [counts[key]]))
            seen.add(key)
    return str(out)


def all_empty_dicts(value: Any) -> bool:
    if isinstance(value, dict):
        return len(value) == 0
    return all(isinstance(item, dict) and len(item) == 0 for item in value)


def tuple_to_int_value(value: Any) -> int:
    return int("".join(str(item) for item in as_list(value)))


def convert_nested_to_float_str(values: Any) -> str:
    return str([tuple(float(item) for item in row) for row in values])


def string_to_list(text: str) -> list[str]:
    return text.split()


def max_product_tuple(values: Any) -> Any:
    return max(a * b for a, b in values)


def words_longer_than(n: int, text: str) -> list[str]:
    return [word for word in text.split() if len(word) > n]


def max_frequency_item(values: Any) -> tuple[Any, int]:
    return Counter(as_list(values)).most_common(1)[0]


def reverse_vowels(text: str) -> str:
    vowels = set("aeiouAEIOU")
    chars = list(text)
    positions = [idx for idx, ch in enumerate(chars) if ch in vowels]
    values = [chars[idx] for idx in positions][::-1]
    for idx, ch in zip(positions, values):
        chars[idx] = ch
    return "".join(chars)


def tuple_to_string(value: Any) -> str:
    return "".join(str(item) for item in as_list(value))


def sum_negative(values: Any) -> Any:
    return sum(item for item in as_list(values) if item < 0)


def hexagonal(n: int) -> int:
    return n * (2 * n - 1)


def circle_circumference(r: Any) -> float:
    return 2 * 3.1415 * r


def unique_flatten_preserve(values: Any) -> list[Any]:
    return unique_preserve(deep_flatten(values))


def count_same_positions(*lists: Any) -> int:
    return sum(len(set(items)) == 1 for items in zip(*lists))


def count_top_level_lists(value: Any) -> int:
    return sum(isinstance(item, list) for item in value) if isinstance(value, tuple) else 1


def sum_abs_pair_diffs(values: Any) -> Any:
    seq = as_list(values)
    if len(seq) > 500:
        raise ValueError("outside_bounded_abi_domain")
    return sum(abs(a - b) for idx, a in enumerate(seq) for b in seq[idx + 1 :])


def max_abs_diff(values: Any) -> Any:
    seq = as_list(values)
    return max(seq) - min(seq)


def first_char_ascii(text: str) -> int:
    return ord(text[0])


def max_path_triangle(triangle: Any) -> Any:
    dp = [list(row) for row in triangle]
    for i in range(len(dp) - 2, -1, -1):
        for j in range(len(dp[i])):
            dp[i][j] += max(dp[i + 1][j], dp[i + 1][j + 1])
    return dp[0][0]


def toggle_even_bits(n: int) -> int:
    n = bounded_int(n, 1_000_000)
    mask = 0
    bit = 1
    while (1 << bit) <= max(1, n):
        mask |= 1 << bit
        bit += 2
    return n ^ mask


def tuple_strings_to_ints(values: Any) -> Any:
    return tuple(tuple(int(item) for item in row) for row in values)


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


def remove_present(left: Any, right: Any) -> list[Any]:
    banned = set(as_list(right))
    return [item for item in as_list(left) if item not in banned]


def descending_by_two_sum(n: int) -> int:
    n = bounded_int(n, 1_000_000)
    terms = (n + 1) // 2
    return terms * (n + (n - 2 * (terms - 1))) // 2


def regular_polygon_area(n: int, side: float) -> float:
    return (n * side * side) / (4 * math.tan(math.pi / n))


def same_alpha_position_count(text: str) -> int:
    return sum(ch.lower() == chr(ord("a") + idx) for idx, ch in enumerate(text))


def even_xor_pairs(values: Any) -> int:
    seq = as_list(values)
    evens = sum(x % 2 == 0 for x in seq)
    odds = len(seq) - evens
    return evens * (evens - 1) // 2 + odds * (odds - 1) // 2


def smallest_power_two_ge(n: int) -> int:
    n = bounded_int(n, 1_000_000)
    out = 1
    while out < n:
        out *= 2
    return out


def pell_number(n: int) -> int:
    n = bounded_int(n, 1_000)
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, 2 * b + a
    return b


def range_sum(values: Any, start: int, stop: int) -> Any:
    return sum(as_list(values)[start : stop + 1])


def sum_common_divisors(a: Any, b: Any) -> int:
    left = bounded_int(a, 100_000)
    right = bounded_int(b, 100_000)
    gcd_value = math.gcd(left, right)
    return sum(divisor for divisor in range(1, gcd_value + 1) if left % divisor == 0 and right % divisor == 0)


def count_hex_letters(start: int, stop: int) -> int:
    return sum(any(ch in "abcdef" for ch in hex(value)[2:]) for value in range(start, stop + 1))


def extract_missing_ranges(ranges: Any, start: int, stop: int) -> list[tuple[int, int]]:
    out = []
    prev = start
    for left, right in ranges:
        out.append((prev, left))
        out.append((right, stop))
        prev = right
    return out


def match_label(text: str, pattern: str, yes: str, no: str) -> str:
    return yes if re.search(pattern, text) else no


def split_upper(text: str) -> list[str]:
    return re.findall(r"[A-Z][^A-Z]*", text)


def split_lower(text: str) -> list[str]:
    return re.findall(r"[a-z][^a-z]*", text)


def max_run_upper(text: str) -> int:
    best = cur = 0
    for ch in text:
        if ch.isupper():
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def remove_first_last(s: str, ch: str) -> str:
    first = s.find(ch)
    last = s.rfind(ch)
    if first == -1:
        return s
    remove = {first, last}
    return "".join(c for idx, c in enumerate(s) if idx not in remove)


def text_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|\d+", text.lower()))


def classify_task(text: str) -> str:
    tokens = text_tokens(text)
    if tokens & {"regex", "string", "substring", "uppercase", "lowercase", "spaces", "split", "match", "characters"}:
        return "string_regex"
    if tokens & {"dictionary", "dictionaries", "counter", "frequency", "frequencies", "occurrences"}:
        return "dict_counter"
    if tokens & {"list", "array", "tuple", "tuples", "sequence"}:
        return "list_tuple"
    if tokens & {"number", "integer", "sum", "product", "area", "perimeter", "volume", "gcd", "lcm"}:
        return "numeric"
    if tokens & {"tree", "graph", "path", "prime", "palindrome", "permutation", "subsequence", "dynamic"}:
        return "algorithmic_control"
    return "other"


def add_candidate(rows: list[Candidate], name: str, category: str, depth: int, func: Callable[..., Any], **program: Any) -> None:
    rows.append(Candidate(name, category, depth, func, {"op": name, **program}))


def visible_labels(examples: list[Example]) -> tuple[str, str]:
    values = [ex.expected for ex in examples if isinstance(ex.expected, str)]
    if "Found a match!" in values or "Not matched!" in values:
        return "Found a match!", "Not matched!"
    if values:
        yes = values[0]
        no = "Not matched!" if yes != "Not matched!" else "Found a match!"
        return yes, no
    return "Found a match!", "Not matched!"


def generate_candidates(examples: list[Example]) -> list[Candidate]:
    rows: list[Candidate] = []
    if not examples:
        return rows
    max_arity = max(len(ex.args) for ex in examples)

    for i in range(max_arity):
        add_candidate(rows, f"arg{i}", "generic", 0, lambda *args, i=i: args[i], arg=i)
        add_candidate(rows, f"len_arg{i}", "generic", 1, lambda *args, i=i: len(args[i]), arg=i)

    for i in range(max_arity):
        add_candidate(rows, f"sum_arg{i}", "numeric", 1, lambda *args, i=i: sum(args[i]), arg=i)
        add_candidate(rows, f"product_arg{i}", "numeric", 1, lambda *args, i=i: product(args[i]), arg=i)
        add_candidate(rows, f"min_arg{i}", "numeric", 1, lambda *args, i=i: min(args[i]), arg=i)
        add_candidate(rows, f"max_arg{i}", "numeric", 1, lambda *args, i=i: max(args[i]), arg=i)
        add_candidate(rows, f"sorted_arg{i}", "list_tuple", 1, lambda *args, i=i: sorted_safe(args[i]), arg=i)
        add_candidate(rows, f"reversed_list_arg{i}", "list_tuple", 1, lambda *args, i=i: list(reversed(args[i])), arg=i)
        add_candidate(rows, f"unique_arg{i}", "list_tuple", 1, lambda *args, i=i: unique_preserve(args[i]), arg=i)
        add_candidate(rows, f"flatten_arg{i}", "list_tuple", 1, lambda *args, i=i: flatten_one(args[i]), arg=i)
        add_candidate(rows, f"deep_flatten_arg{i}", "list_tuple", 2, lambda *args, i=i: deep_flatten(args[i]), arg=i)
        add_candidate(rows, f"transpose_arg{i}", "list_tuple", 1, lambda *args, i=i: transpose(args[i]), arg=i)
        add_candidate(rows, f"first_duplicate_arg{i}", "list_tuple", 1, lambda *args, i=i: first_duplicate(args[i]), arg=i)
        add_candidate(rows, f"duplicate_items_arg{i}", "list_tuple", 1, lambda *args, i=i: duplicate_items(args[i]), arg=i)
        add_candidate(rows, f"frequency_dict_arg{i}", "dict_counter", 1, lambda *args, i=i: frequency_dict(args[i]), arg=i)
        add_candidate(rows, f"frequency_flattened_arg{i}", "dict_counter", 2, lambda *args, i=i: frequency_flattened(args[i]), arg=i)
        add_candidate(rows, f"consecutive_duplicate_counts_arg{i}", "list_tuple", 1, lambda *args, i=i: consecutive_duplicate_counts(args[i]), arg=i)
        add_candidate(rows, f"sort_matrix_by_row_sum_arg{i}", "list_tuple", 2, lambda *args, i=i: sort_matrix_by_row_sum(args[i]), arg=i)
        add_candidate(rows, f"most_common_four_arg{i}", "dict_counter", 1, lambda *args, i=i: most_common_four(args[i]), arg=i)
        add_candidate(rows, f"has_duplicate_arg{i}", "list_tuple", 1, lambda *args, i=i: has_duplicate(args[i]), arg=i)
        add_candidate(rows, f"max_row_sum_arg{i}", "list_tuple", 2, lambda *args, i=i: max_row_sum(args[i]), arg=i)
        add_candidate(rows, f"binary_number_to_decimal_arg{i}", "numeric", 1, lambda *args, i=i: binary_number_to_decimal(args[i]), arg=i)
        add_candidate(rows, f"decimal_to_binary_int_arg{i}", "numeric", 1, lambda *args, i=i: decimal_to_binary_int(args[i]), arg=i)
        add_candidate(rows, f"decimal_to_binary_str_arg{i}", "numeric", 1, lambda *args, i=i: decimal_to_binary_str(args[i]), arg=i)
        add_candidate(rows, f"product_non_repeated_arg{i}", "list_tuple", 2, lambda *args, i=i: product_non_repeated(args[i]), arg=i)
        add_candidate(rows, f"remove_digits_from_strings_arg{i}", "string_regex", 1, lambda *args, i=i: remove_digits_from_strings(args[i]), arg=i)
        add_candidate(rows, f"odd_occurrence_arg{i}", "list_tuple", 1, lambda *args, i=i: odd_occurrence(args[i]), arg=i)
        add_candidate(rows, f"count_substrings_same_ends_arg{i}", "string_regex", 2, lambda *args, i=i: count_substrings_same_ends(args[i]), arg=i)
        add_candidate(rows, f"largest_prime_factor_arg{i}", "numeric", 2, lambda *args, i=i: largest_prime_factor(args[i]), arg=i)
        add_candidate(rows, f"missing_sorted_arg{i}", "list_tuple", 2, lambda *args, i=i: missing_sorted(args[i]), arg=i)
        add_candidate(rows, f"sort_mixed_arg{i}", "list_tuple", 1, lambda *args, i=i: sort_mixed(args[i]), arg=i)
        add_candidate(rows, f"first_even_div_first_odd_arg{i}", "list_tuple", 2, lambda *args, i=i: first_even_div_first_odd(args[i]), arg=i)
        add_candidate(rows, f"rearrange_no_adjacent_arg{i}", "string_regex", 3, lambda *args, i=i: rearrange_no_adjacent(args[i]), arg=i)
        add_candidate(rows, f"filter_even_arg{i}", "list_tuple", 1, lambda *args, i=i: filter_even(args[i]), arg=i)
        add_candidate(rows, f"sum_repeated_values_arg{i}", "list_tuple", 2, lambda *args, i=i: sum_repeated_values(args[i]), arg=i)
        add_candidate(rows, f"all_distinct_arg{i}", "list_tuple", 1, lambda *args, i=i: all_distinct(args[i]), arg=i)
        add_candidate(rows, f"set_odd_position_bits_arg{i}", "numeric", 2, lambda *args, i=i: set_odd_position_bits(args[i]), arg=i)
        add_candidate(rows, f"min_length_pair_arg{i}", "list_tuple", 1, lambda *args, i=i: min_length_pair(args[i]), arg=i)
        add_candidate(rows, f"first_last_equal_label_arg{i}", "string_regex", 1, lambda *args, i=i: first_last_equal_label(args[i]), arg=i)
        add_candidate(rows, f"octagonal_arg{i}", "numeric", 1, lambda *args, i=i: octagonal(args[i]), arg=i)
        add_candidate(rows, f"adjacent_diff_subseq_len_arg{i}", "list_tuple", 3, lambda *args, i=i: adjacent_diff_subseq_len(args[i]), arg=i)
        add_candidate(rows, f"digit_substrings_sum_equals_len_arg{i}", "string_regex", 3, lambda *args, i=i: digit_substrings_sum_equals_len(args[i]), arg=i)
        add_candidate(rows, f"max_tuple_absdiff_arg{i}", "list_tuple", 1, lambda *args, i=i: max_tuple_absdiff(args[i]), arg=i)
        add_candidate(rows, f"recursive_sum_arg{i}", "list_tuple", 2, lambda *args, i=i: recursive_sum(args[i]), arg=i)
        add_candidate(rows, f"count_positive_arg{i}", "list_tuple", 1, lambda *args, i=i: count_positive(args[i]), arg=i)
        add_candidate(rows, f"is_monotonic_arg{i}", "list_tuple", 1, lambda *args, i=i: is_monotonic(args[i]), arg=i)
        add_candidate(rows, f"difference_of_two_squares_arg{i}", "numeric", 1, lambda *args, i=i: difference_of_two_squares(args[i]), arg=i)
        add_candidate(rows, f"split_multiple_delims_arg{i}", "string_regex", 1, lambda *args, i=i: split_multiple_delims(args[i]), arg=i)
        add_candidate(rows, f"word_len_odd_arg{i}", "string_regex", 1, lambda *args, i=i: word_len_odd(args[i]), arg=i)
        add_candidate(rows, f"tetrahedral_arg{i}", "numeric", 1, lambda *args, i=i: tetrahedral(args[i]), arg=i)
        add_candidate(rows, f"centered_hexagonal_arg{i}", "numeric", 1, lambda *args, i=i: centered_hexagonal(args[i]), arg=i)
        add_candidate(rows, f"perimeter_square_arg{i}", "numeric", 1, lambda *args, i=i: 4 * args[i], arg=i)
        add_candidate(rows, f"rectangular_number_arg{i}", "numeric", 1, lambda *args, i=i: args[i] * (args[i] + 1), arg=i)
        add_candidate(rows, f"volume_sphere_arg{i}", "numeric", 1, lambda *args, i=i: 4 / 3 * math.pi * args[i] ** 3, arg=i)
        add_candidate(rows, f"surface_sphere_arg{i}", "numeric", 1, lambda *args, i=i: 4 * math.pi * args[i] ** 2, arg=i)
        add_candidate(rows, f"closest_smaller_arg{i}", "numeric", 1, lambda *args, i=i: args[i] - 1, arg=i)
        add_candidate(rows, f"longest_word_length_arg{i}", "string_regex", 1, lambda *args, i=i: longest_word_length(args[i]), arg=i)
        add_candidate(rows, f"is_undulating_arg{i}", "string_regex", 2, lambda *args, i=i: is_undulating(args[i]), arg=i)
        add_candidate(rows, f"min_second_record_name_arg{i}", "list_tuple", 1, lambda *args, i=i: min_second_record_name(args[i]), arg=i)
        add_candidate(rows, f"divisor_count_arg{i}", "numeric", 2, lambda *args, i=i: divisor_count(args[i]), arg=i)
        add_candidate(rows, f"multiply_divide_len_arg{i}", "numeric", 1, lambda *args, i=i: multiply_divide_len(args[i]), arg=i)
        add_candidate(rows, f"next_palindrome_arg{i}", "numeric", 3, lambda *args, i=i: next_palindrome(args[i]), arg=i)
        add_candidate(rows, f"snake_to_camel_arg{i}", "string_regex", 1, lambda *args, i=i: snake_to_camel(args[i]), arg=i)
        add_candidate(rows, f"sort_sublists_arg{i}", "list_tuple", 1, lambda *args, i=i: [sorted(row) for row in args[i]], arg=i)
        add_candidate(rows, f"largest_number_from_digits_arg{i}", "numeric", 1, lambda *args, i=i: largest_number_from_digits(args[i]), arg=i)
        add_candidate(rows, f"sort_tuple_by_second_arg{i}", "list_tuple", 1, lambda *args, i=i: sort_tuple_by_second(args[i]), arg=i)
        add_candidate(rows, f"bell_number_arg{i}", "numeric", 3, lambda *args, i=i: bell_number(args[i]), arg=i)
        add_candidate(rows, f"count_odd_setbit_integers_arg{i}", "numeric", 2, lambda *args, i=i: count_odd_setbit_integers(args[i]), arg=i)
        add_candidate(rows, f"alphabet_sum_char_arg{i}", "string_regex", 1, lambda *args, i=i: alphabet_sum_char(args[i]), arg=i)
        add_candidate(rows, f"newman_conway_arg{i}", "numeric", 3, lambda *args, i=i: newman_conway(args[i]), arg=i)
        add_candidate(rows, f"min_sublist_length_arg{i}", "list_tuple", 1, lambda *args, i=i: min_sublist_length(args[i]), arg=i)
        add_candidate(rows, f"count_ones_arg{i}", "string_regex", 1, lambda *args, i=i: count_ones(args[i]), arg=i)
        add_candidate(rows, f"common_nested_arg{i}", "list_tuple", 2, lambda *args, i=i: common_nested(args[i]), arg=i)
        add_candidate(rows, f"tuple_frequency_assign_arg{i}", "dict_counter", 2, lambda *args, i=i: tuple_frequency_assign(args[i]), arg=i)
        add_candidate(rows, f"all_empty_dicts_arg{i}", "dict_counter", 1, lambda *args, i=i: all_empty_dicts(args[i]), arg=i)
        add_candidate(rows, f"tuple_to_int_arg{i}", "numeric", 1, lambda *args, i=i: tuple_to_int_value(args[i]), arg=i)
        add_candidate(rows, f"convert_nested_to_float_str_arg{i}", "list_tuple", 1, lambda *args, i=i: convert_nested_to_float_str(args[i]), arg=i)
        add_candidate(rows, f"string_to_list_arg{i}", "string_regex", 1, lambda *args, i=i: string_to_list(args[i]), arg=i)
        add_candidate(rows, f"max_product_tuple_arg{i}", "list_tuple", 1, lambda *args, i=i: max_product_tuple(args[i]), arg=i)
        add_candidate(rows, f"max_frequency_item_arg{i}", "dict_counter", 1, lambda *args, i=i: max_frequency_item(args[i]), arg=i)
        add_candidate(rows, f"reverse_vowels_arg{i}", "string_regex", 1, lambda *args, i=i: reverse_vowels(args[i]), arg=i)
        add_candidate(rows, f"tuple_to_string_arg{i}", "string_regex", 1, lambda *args, i=i: tuple_to_string(args[i]), arg=i)
        add_candidate(rows, f"sum_negative_arg{i}", "numeric", 1, lambda *args, i=i: sum_negative(args[i]), arg=i)
        add_candidate(rows, f"hexagonal_arg{i}", "numeric", 1, lambda *args, i=i: hexagonal(args[i]), arg=i)
        add_candidate(rows, f"circle_circumference_arg{i}", "numeric", 1, lambda *args, i=i: circle_circumference(args[i]), arg=i)
        add_candidate(rows, f"unique_flatten_preserve_arg{i}", "list_tuple", 2, lambda *args, i=i: unique_flatten_preserve(args[i]), arg=i)
        add_candidate(rows, f"count_top_level_lists_arg{i}", "list_tuple", 1, lambda *args, i=i: count_top_level_lists(args[i]), arg=i)
        add_candidate(rows, f"sum_abs_pair_diffs_arg{i}", "list_tuple", 2, lambda *args, i=i: sum_abs_pair_diffs(args[i]), arg=i)
        add_candidate(rows, f"max_abs_diff_arg{i}", "list_tuple", 1, lambda *args, i=i: max_abs_diff(args[i]), arg=i)
        add_candidate(rows, f"first_char_ascii_arg{i}", "string_regex", 1, lambda *args, i=i: first_char_ascii(args[i]), arg=i)
        add_candidate(rows, f"max_path_triangle_arg{i}", "list_tuple", 3, lambda *args, i=i: max_path_triangle(args[i]), arg=i)
        add_candidate(rows, f"toggle_even_bits_arg{i}", "numeric", 2, lambda *args, i=i: toggle_even_bits(args[i]), arg=i)
        add_candidate(rows, f"tuple_strings_to_ints_arg{i}", "list_tuple", 1, lambda *args, i=i: tuple_strings_to_ints(args[i]), arg=i)
        add_candidate(rows, f"run_length_encode_arg{i}", "list_tuple", 1, lambda *args, i=i: run_length_encode(args[i]), arg=i)
        add_candidate(rows, f"descending_by_two_sum_arg{i}", "numeric", 1, lambda *args, i=i: descending_by_two_sum(args[i]), arg=i)
        add_candidate(rows, f"same_alpha_position_count_arg{i}", "string_regex", 1, lambda *args, i=i: same_alpha_position_count(args[i]), arg=i)
        add_candidate(rows, f"even_xor_pairs_arg{i}", "numeric", 2, lambda *args, i=i: even_xor_pairs(args[i]), arg=i)
        add_candidate(rows, f"smallest_power_two_ge_arg{i}", "numeric", 1, lambda *args, i=i: smallest_power_two_ge(args[i]), arg=i)
        add_candidate(rows, f"pell_number_arg{i}", "numeric", 3, lambda *args, i=i: pell_number(args[i]), arg=i)

    for i in range(max_arity):
        add_candidate(rows, f"lower_arg{i}", "string_regex", 1, lambda *args, i=i: args[i].lower(), arg=i)
        add_candidate(rows, f"upper_arg{i}", "string_regex", 1, lambda *args, i=i: args[i].upper(), arg=i)
        add_candidate(rows, f"title_arg{i}", "string_regex", 1, lambda *args, i=i: args[i].title(), arg=i)
        add_candidate(rows, f"swapcase_arg{i}", "string_regex", 1, lambda *args, i=i: args[i].swapcase(), arg=i)
        add_candidate(rows, f"reverse_str_arg{i}", "string_regex", 1, lambda *args, i=i: args[i][::-1], arg=i)
        add_candidate(rows, f"split_upper_arg{i}", "string_regex", 1, lambda *args, i=i: split_upper(args[i]), arg=i)
        add_candidate(rows, f"split_lower_arg{i}", "string_regex", 1, lambda *args, i=i: split_lower(args[i]), arg=i)
        add_candidate(rows, f"remove_multiple_spaces_arg{i}", "string_regex", 1, lambda *args, i=i: re.sub(r" +", " ", args[i]), arg=i)
        add_candidate(rows, f"max_run_upper_arg{i}", "string_regex", 1, lambda *args, i=i: max_run_upper(args[i]), arg=i)

    if max_arity >= 2:
        for i, j in itertools.permutations(range(max_arity), 2):
            add_candidate(rows, f"add_{i}_{j}", "numeric", 1, lambda *args, i=i, j=j: args[i] + args[j], args=[i, j])
            add_candidate(rows, f"sub_{i}_{j}", "numeric", 1, lambda *args, i=i, j=j: args[i] - args[j], args=[i, j])
            add_candidate(rows, f"mul_{i}_{j}", "numeric", 1, lambda *args, i=i, j=j: args[i] * args[j], args=[i, j])
            add_candidate(rows, f"floordiv_{i}_{j}", "numeric", 1, lambda *args, i=i, j=j: args[i] // args[j], args=[i, j])
            add_candidate(rows, f"mod_{i}_{j}", "numeric", 1, lambda *args, i=i, j=j: args[i] % args[j], args=[i, j])
            add_candidate(rows, f"pow_{i}_{j}", "numeric", 1, lambda *args, i=i, j=j: safe_pow(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"absdiff_{i}_{j}", "numeric", 1, lambda *args, i=i, j=j: abs(args[i] - args[j]), args=[i, j])
            add_candidate(rows, f"gcd_{i}_{j}", "numeric", 1, lambda *args, i=i, j=j: math.gcd(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"count_{i}_{j}", "list_tuple", 1, lambda *args, i=i, j=j: args[i].count(args[j]), args=[i, j])
            add_candidate(rows, f"remove_all_{i}_{j}", "list_tuple", 1, lambda *args, i=i, j=j: [x for x in args[i] if x != args[j]], args=[i, j])
            add_candidate(rows, f"all_in_{i}_{j}", "list_tuple", 1, lambda *args, i=i, j=j: all(x in args[j] for x in args[i]), args=[i, j])
            add_candidate(rows, f"str_count_{i}_{j}", "string_regex", 1, lambda *args, i=i, j=j: args[i].count(args[j]), args=[i, j])
            add_candidate(rows, f"str_remove_all_{i}_{j}", "string_regex", 1, lambda *args, i=i, j=j: args[i].replace(args[j], ""), args=[i, j])
            add_candidate(rows, f"str_remove_first_last_{i}_{j}", "string_regex", 2, lambda *args, i=i, j=j: remove_first_last(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"remove_chars_{i}_{j}", "string_regex", 1, lambda *args, i=i, j=j: remove_chars(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"multiples_{i}_{j}", "numeric", 1, lambda *args, i=i, j=j: multiples(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"all_tuple_elements_equal_{i}_{j}", "list_tuple", 1, lambda *args, i=i, j=j: all_tuple_elements_equal(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"binomial_{i}_{j}", "numeric", 2, lambda *args, i=i, j=j: safe_comb(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"column_{i}_{j}", "list_tuple", 1, lambda *args, i=i, j=j: column(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"opposite_signs_{i}_{j}", "numeric", 1, lambda *args, i=i, j=j: opposite_signs(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"contains_sublist_{i}_{j}", "list_tuple", 2, lambda *args, i=i, j=j: contains_sublist(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"equal_tuple_length_label_{i}_{j}", "list_tuple", 1, lambda *args, i=i, j=j: equal_tuple_length_label(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"tuples_all_divisible_{i}_{j}", "list_tuple", 1, lambda *args, i=i, j=j: tuples_all_divisible(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"count_squares_in_rectangle_{i}_{j}", "numeric", 2, lambda *args, i=i, j=j: count_squares_in_rectangle(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"cylinder_perimeter_like_{i}_{j}", "numeric", 1, lambda *args, i=i, j=j: 2 * (args[i] + args[j]), args=[i, j])
            add_candidate(rows, f"substring_in_list_{i}_{j}", "string_regex", 1, lambda *args, i=i, j=j: substring_in_list(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"eulerian_number_{i}_{j}", "numeric", 3, lambda *args, i=i, j=j: eulerian_number(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"count_hex_letters_{i}_{j}", "numeric", 2, lambda *args, i=i, j=j: count_hex_letters(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"add_list_to_tuple_{i}_{j}", "list_tuple", 1, lambda *args, i=i, j=j: tuple(args[j]) + tuple(args[i]), args=[i, j])
            add_candidate(rows, f"zip_cycle_{i}_{j}", "list_tuple", 1, lambda *args, i=i, j=j: zip_cycle(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"words_longer_than_{i}_{j}", "string_regex", 1, lambda *args, i=i, j=j: words_longer_than(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"sum_common_divisors_{i}_{j}", "numeric", 2, lambda *args, i=i, j=j: sum_common_divisors(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"remove_present_{i}_{j}", "list_tuple", 1, lambda *args, i=i, j=j: remove_present(args[i], args[j]), args=[i, j])
            add_candidate(rows, f"regular_polygon_area_{i}_{j}", "numeric", 2, lambda *args, i=i, j=j: regular_polygon_area(args[i], args[j]), args=[i, j])

    if max_arity >= 2:
        add_candidate(rows, "counter_add_0_1", "dict_counter", 2, lambda *args: dict(Counter(args[0]) + Counter(args[1])), args=[0, 1])
        add_candidate(rows, "dict_merge_sum_0_1", "dict_counter", 2, lambda *args: {k: args[0].get(k, 0) + args[1].get(k, 0) for k in set(args[0]) | set(args[1])}, args=[0, 1])
        add_candidate(rows, "perimeter_square_0", "numeric", 1, lambda *args: 4 * args[0], arg=0)
        add_candidate(rows, "rectangular_number_0", "numeric", 1, lambda *args: args[0] * (args[0] + 1), arg=0)
        add_candidate(rows, "volume_sphere_0", "numeric", 1, lambda *args: 4 / 3 * math.pi * args[0] ** 3, arg=0)
        add_candidate(rows, "surface_sphere_0", "numeric", 1, lambda *args: 4 * math.pi * args[0] ** 2, arg=0)
        add_candidate(rows, "merge_dicts_all", "dict_counter", 2, lambda *args: merge_dicts(*args), args="all")
        add_candidate(rows, "merge_dicts_first_priority_all", "dict_counter", 2, lambda *args: merge_dicts_first_priority(*args), args="all")
        add_candidate(rows, "merge_sorted_all", "list_tuple", 2, lambda *args: sorted(deep_flatten(args)), args="all")
    if max_arity >= 3:
        add_candidate(rows, "triangular_prism_volume_0_1_2", "numeric", 1, lambda *args: args[0] * args[1] * args[2] / 2, args=[0, 1, 2])
        add_candidate(rows, "kth_element_0_2", "list_tuple", 1, lambda *args: args[0][args[2] - 1], args=[0, 2])
        add_candidate(rows, "extract_missing_ranges_0_1_2", "list_tuple", 3, lambda *args: extract_missing_ranges(args[0], args[1], args[2]), args=[0, 1, 2])
        add_candidate(rows, "count_same_positions_all", "list_tuple", 1, lambda *args: count_same_positions(*args), args="all")
        add_candidate(rows, "range_sum_0_1_2", "list_tuple", 1, lambda *args: range_sum(args[0], args[1], args[2]), args=[0, 1, 2])

    yes, no = visible_labels(examples[:1])
    regex_patterns = [
        r"^[a-z]+_[a-z]+$",
        r"[A-Z]+[a-z]+$",
        r"ab*?",
        r"a.*?b$",
        r"^[A-Z][a-z]+$",
        r"^[a-z]+$",
        r"^[A-Z]+$",
        r"^\w+$",
        r"\s+",
    ]
    for i in range(max_arity):
        for pattern in regex_patterns:
            safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", pattern).strip("_") or "pattern"
            add_candidate(
                rows,
                f"regex_label_{i}_{safe_name}",
                "string_regex",
                2,
                lambda *args, i=i, pattern=pattern, yes=yes, no=no: match_label(args[i], pattern, yes, no),
                arg=i,
                pattern=pattern,
                yes=yes,
                no=no,
            )
    return rows


def candidate_results(candidate: Candidate, examples: list[Example]) -> list[dict[str, Any]]:
    rows = []
    for ex in examples:
        ok, value = safe_call(candidate.func, ex.args)
        rows.append(
            {
                "ok": ok,
                "value_repr": repr(value),
                "passed": ok and approx_equal(value, ex.expected),
            }
        )
    return rows


def load_records(split: str, count: int, offset: int, visible_tests: int) -> list[dict[str, Any]]:
    dataset = load_dataset("google-research-datasets/mbpp")
    rows = []
    for raw in list(dataset[split])[offset : offset + count]:
        tests = list(raw.get("test_list") or [])
        if not tests:
            continue
        entry = parse_entry(tests[0])
        parsed = [parse_assert(test) for test in tests + list(raw.get("challenge_test_list") or [])]
        examples = [ex for ex in parsed if ex is not None]
        visible_examples = [ex for ex in [parse_assert(test) for test in tests[:visible_tests]] if ex is not None]
        if entry is None:
            entry = "unknown"
        rows.append(
            {
                "record_id": f"mbpp_{split}_{raw['task_id']}",
                "task_id": raw["task_id"],
                "task_text": raw["text"],
                "entry_point": entry,
                "tests": tests,
                "challenge_tests": list(raw.get("challenge_test_list") or []),
                "examples": examples,
                "visible_examples": visible_examples,
                "parseable": len(examples) == len(tests) + len(raw.get("challenge_test_list") or []),
            }
        )
    return rows


def run_task(record: dict[str, Any]) -> dict[str, Any]:
    examples: list[Example] = record["examples"]
    visible_examples: list[Example] = record["visible_examples"]
    slice_name = classify_task(record["task_text"])
    candidates = generate_candidates(examples)
    full_winners = []
    visible_winners = []
    visible_hidden_wrong = 0
    candidate_rows = []
    for cand in candidates:
        results = candidate_results(cand, examples)
        visible_count = len(visible_examples)
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
        "slice": slice_name,
        "parseable": record["parseable"],
        "example_count": len(examples),
        "visible_count": len(visible_examples),
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
            "visible_false_pass_rate_among_visible": (
                sum(row["visible_false_pass"] for row in group) / max(1, sum(row["visible_any"] for row in group))
            ),
            "visible_hidden_wrong_candidates": sum(row["visible_hidden_wrong_count"] for row in group),
            "visible_hidden_wrong_rate_among_candidates": (
                sum(row["visible_hidden_wrong_count"] for row in group)
                / max(1, sum(row["visible_consistent_count"] for row in group))
            ),
            "first_visible_full_pass": sum(row["first_visible_full_pass"] for row in group),
            "candidate_count_mean": sum(row["candidate_count"] for row in group) / len(group),
            "visible_consistent_mean": sum(row["visible_consistent_count"] for row in group) / len(group),
        }

    slices = sorted(set(row["slice"] for row in rows))
    return {
        "experiment": EXPERIMENT,
        "overall": metrics(rows),
        "by_slice": {slice_name: metrics([row for row in rows if row["slice"] == slice_name]) for slice_name in slices},
        "covered_task_ids": [row["task_id"] for row in rows if row["oracle_covered"]],
        "uncovered_task_ids": [row["task_id"] for row in rows if not row["oracle_covered"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--visible-tests", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.split, args.count, args.offset, args.visible_tests)
    rows = [run_task(record) for record in records]
    summary = summarize(rows)
    summary.update({"split": args.split, "count": args.count, "offset": args.offset, "visible_tests": args.visible_tests, "path": str(args.out)})
    write_jsonl(args.out, rows)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
