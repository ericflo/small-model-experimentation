from __future__ import annotations

from dataclasses import dataclass


MODULE_PATH = "src/repair_target.py"
VISIBLE_TEST_PATH = "tests/test_visible.py"
HIDDEN_TEST_PATH = "tests/test_hidden.py"


@dataclass(frozen=True)
class WrongVariant:
    name: str
    source: str
    touched_gold_file: bool = True
    touched_gold_function: bool = True


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    split: str
    bug_family: str
    issue: str
    clean_source: str
    buggy_source: str
    visible_tests: str
    hidden_tests: str
    near_miss_source: str
    visible_overfit_source: str

    def wrong_variants(self) -> list[WrongVariant]:
        return [
            WrongVariant("near_miss", self.near_miss_source, True, True),
            WrongVariant("wrong_localization", "UNRELATED_FLAG = True\n" + self.buggy_source, True, False),
            WrongVariant("syntax_error", self.buggy_source + "\ndef broken(:\n    pass\n", True, False),
            WrongVariant("import_error", "import definitely_missing_dependency\n" + self.buggy_source, True, False),
            WrongVariant("visible_overfit", self.visible_overfit_source, True, True),
        ]


def _tests(body: str) -> str:
    return "from repair_target import *\n\n" + body.strip() + "\n"


TASKS: list[TaskSpec] = [
    TaskSpec(
        task_id="clamp_bounds",
        split="train",
        bug_family="comparison_flip",
        issue="`clamp(value, lower, upper)` returns the wrong bound for values below or above the allowed interval.",
        clean_source="""def clamp(value, lower, upper):
    if lower > upper:
        raise ValueError("lower bound cannot exceed upper bound")
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value
""",
        buggy_source="""def clamp(value, lower, upper):
    if lower > upper:
        raise ValueError("lower bound cannot exceed upper bound")
    if value < lower:
        return upper
    if value > upper:
        return lower
    return value
""",
        visible_tests=_tests("""
def test_lower_bound():
    assert clamp(-1, 0, 10) == 0

def test_inside_interval():
    assert clamp(5, 0, 10) == 5
"""),
        hidden_tests=_tests("""
import pytest

def test_upper_bound():
    assert clamp(15, 0, 10) == 10

def test_invalid_bounds():
    with pytest.raises(ValueError):
        clamp(1, 5, 2)
"""),
        near_miss_source="""def clamp(value, lower, upper):
    if lower > upper:
        raise ValueError("lower bound cannot exceed upper bound")
    if value < lower:
        return lower
    if value > upper:
        return value
    return value
""",
        visible_overfit_source="""def clamp(value, lower, upper):
    if value == -1 and lower == 0 and upper == 10:
        return 0
    if lower > upper:
        raise ValueError("lower bound cannot exceed upper bound")
    if value < lower:
        return upper
    if value > upper:
        return lower
    return value
""",
    ),
    TaskSpec(
        task_id="parse_bool_case",
        split="train",
        bug_family="case_normalization",
        issue="`parse_bool` should accept common boolean strings case-insensitively and reject unknown values.",
        clean_source="""def parse_bool(text):
    value = str(text).strip().lower()
    if value in {"true", "yes", "1", "on"}:
        return True
    if value in {"false", "no", "0", "off"}:
        return False
    raise ValueError(f"not a boolean: {text!r}")
""",
        buggy_source="""def parse_bool(text):
    value = str(text).strip()
    if value in {"true", "yes", "1", "on"}:
        return True
    if value in {"false", "no", "0", "off"}:
        return False
    return bool(value)
""",
        visible_tests=_tests("""
def test_true_words():
    assert parse_bool("YES") is True

def test_false_words():
    assert parse_bool("off") is False
"""),
        hidden_tests=_tests("""
import pytest

def test_false_case_insensitive():
    assert parse_bool("No") is False

def test_unknown_raises():
    with pytest.raises(ValueError):
        parse_bool("maybe")
"""),
        near_miss_source="""def parse_bool(text):
    value = str(text).strip().lower()
    if value in {"true", "yes", "1", "on"}:
        return True
    if value in {"false", "no", "0", "off"}:
        return False
    return bool(value)
""",
        visible_overfit_source="""def parse_bool(text):
    if text == "YES":
        return True
    value = str(text).strip()
    if value in {"true", "yes", "1", "on"}:
        return True
    if value in {"false", "no", "0", "off"}:
        return False
    return bool(value)
""",
    ),
    TaskSpec(
        task_id="median_even",
        split="train",
        bug_family="off_by_one",
        issue="`median` should sort the inputs, reject empty input, and average the two middle values for even-length inputs.",
        clean_source="""def median(values):
    values = sorted(values)
    if not values:
        raise ValueError("median of empty data")
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2
""",
        buggy_source="""def median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid]
""",
        visible_tests=_tests("""
def test_odd_values():
    assert median([3, 1, 2]) == 2

def test_even_values():
    assert median([10, 2, 4, 8]) == 6
"""),
        hidden_tests=_tests("""
import pytest

def test_empty_rejected():
    with pytest.raises(ValueError):
        median([])

def test_even_negative_values():
    assert median([-4, -1, 1, 10]) == 0
"""),
        near_miss_source="""def median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2
""",
        visible_overfit_source="""def median(values):
    if values == [10, 2, 4, 8]:
        return 6
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid]
""",
    ),
    TaskSpec(
        task_id="merge_intervals_adjacent",
        split="train",
        bug_family="edge_case",
        issue="`merge_intervals` should merge overlapping and directly adjacent inclusive intervals.",
        clean_source="""def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if start > end:
            raise ValueError("interval start exceeds end")
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [tuple(item) for item in merged]
""",
        buggy_source="""def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [tuple(item) for item in merged]
""",
        visible_tests=_tests("""
def test_overlapping_intervals():
    assert merge_intervals([(1, 3), (2, 5)]) == [(1, 5)]

def test_adjacent_intervals():
    assert merge_intervals([(1, 2), (3, 4)]) == [(1, 4)]
"""),
        hidden_tests=_tests("""
import pytest

def test_invalid_interval_rejected():
    with pytest.raises(ValueError):
        merge_intervals([(5, 1)])

def test_separate_intervals_stay_separate():
    assert merge_intervals([(1, 2), (4, 6)]) == [(1, 2), (4, 6)]
"""),
        near_miss_source="""def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [tuple(item) for item in merged]
""",
        visible_overfit_source="""def merge_intervals(intervals):
    if intervals == [(1, 2), (3, 4)]:
        return [(1, 4)]
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [tuple(item) for item in merged]
""",
    ),
    TaskSpec(
        task_id="dedupe_order",
        split="train",
        bug_family="ordering",
        issue="`dedupe_preserve_order` should remove later duplicates without sorting or changing the first occurrence order.",
        clean_source="""def dedupe_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
""",
        buggy_source="""def dedupe_preserve_order(items):
    return sorted(set(items))
""",
        visible_tests=_tests("""
def test_keeps_first_occurrence_order():
    assert dedupe_preserve_order(["b", "a", "b"]) == ["b", "a"]
"""),
        hidden_tests=_tests("""
def test_numbers_keep_order():
    assert dedupe_preserve_order([3, 1, 3, 2, 1]) == [3, 1, 2]

def test_empty():
    assert dedupe_preserve_order([]) == []
"""),
        near_miss_source="""def dedupe_preserve_order(items):
    return list(dict.fromkeys(sorted(items)))
""",
        visible_overfit_source="""def dedupe_preserve_order(items):
    if items == ["b", "a", "b"]:
        return ["b", "a"]
    return sorted(set(items))
""",
    ),
    TaskSpec(
        task_id="slugify_collapse",
        split="train",
        bug_family="normalization",
        issue="`slugify` should lowercase text, replace runs of non-alphanumeric characters with one dash, and trim leading/trailing dashes.",
        clean_source="""import re


def slugify(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return slug.strip("-")
""",
        buggy_source="""import re


def slugify(text):
    return re.sub(r"[^a-z0-9]", "-", text.lower())
""",
        visible_tests=_tests("""
def test_spaces_collapse():
    assert slugify("Hello   World") == "hello-world"
"""),
        hidden_tests=_tests("""
def test_trim_punctuation():
    assert slugify(" **Ready, Set, Go!** ") == "ready-set-go"

def test_numbers_kept():
    assert slugify("Qwen 3 4B") == "qwen-3-4b"
"""),
        near_miss_source="""import re


def slugify(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return slug
""",
        visible_overfit_source="""import re


def slugify(text):
    if text == "Hello   World":
        return "hello-world"
    return re.sub(r"[^a-z0-9]", "-", text.lower())
""",
    ),
    TaskSpec(
        task_id="roman_subtractive",
        split="train",
        bug_family="algorithmic_edge",
        issue="`roman_to_int` should handle subtractive Roman numeral pairs such as IV, IX, XL, and CM.",
        clean_source="""VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(text):
    total = 0
    prev = 0
    for char in reversed(text):
        value = VALUES[char]
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total
""",
        buggy_source="""VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(text):
    total = 0
    for char in text:
        total += VALUES[char]
    return total
""",
        visible_tests=_tests("""
def test_simple_additive():
    assert roman_to_int("VIII") == 8

def test_subtractive_four():
    assert roman_to_int("IV") == 4
"""),
        hidden_tests=_tests("""
def test_complex_value():
    assert roman_to_int("MCMXCIV") == 1994

def test_nine():
    assert roman_to_int("IX") == 9
"""),
        near_miss_source="""VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(text):
    if text == "IV":
        return 4
    total = 0
    for char in text:
        total += VALUES[char]
    return total
""",
        visible_overfit_source="""VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(text):
    if text == "IV":
        return 4
    total = 0
    for char in text:
        total += VALUES[char]
    return total
""",
    ),
    TaskSpec(
        task_id="duration_multiunit",
        split="train",
        bug_family="parser_partial",
        issue="`parse_duration` should parse compact duration strings containing days, hours, minutes, and seconds in combination.",
        clean_source="""import re

UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}


def parse_duration(text):
    total = 0
    pos = 0
    for match in re.finditer(r"(\\d+)([dhms])", text.strip()):
        if match.start() != pos:
            raise ValueError(f"invalid duration: {text!r}")
        total += int(match.group(1)) * UNITS[match.group(2)]
        pos = match.end()
    if pos != len(text.strip()) or pos == 0:
        raise ValueError(f"invalid duration: {text!r}")
    return total
""",
        buggy_source="""import re

UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}


def parse_duration(text):
    match = re.match(r"(\\d+)([dhms])", text.strip())
    if not match:
        raise ValueError(f"invalid duration: {text!r}")
    return int(match.group(1)) * UNITS[match.group(2)]
""",
        visible_tests=_tests("""
def test_single_unit():
    assert parse_duration("45s") == 45

def test_two_units():
    assert parse_duration("2h30m") == 9000
"""),
        hidden_tests=_tests("""
import pytest

def test_three_units():
    assert parse_duration("1d2h3s") == 93603

def test_rejects_trailing_text():
    with pytest.raises(ValueError):
        parse_duration("5mx")
"""),
        near_miss_source="""import re

UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}


def parse_duration(text):
    total = 0
    for match in re.finditer(r"(\\d+)([dhms])", text.strip()):
        total += int(match.group(1)) * UNITS[match.group(2)]
    return total
""",
        visible_overfit_source="""import re

UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}


def parse_duration(text):
    if text == "2h30m":
        return 9000
    match = re.match(r"(\\d+)([dhms])", text.strip())
    if not match:
        raise ValueError(f"invalid duration: {text!r}")
    return int(match.group(1)) * UNITS[match.group(2)]
""",
    ),
    TaskSpec(
        task_id="moving_average_window",
        split="train",
        bug_family="off_by_one",
        issue="`moving_average(values, window)` should return each full sliding-window average and reject invalid window sizes.",
        clean_source="""def moving_average(values, window):
    if window <= 0:
        raise ValueError("window must be positive")
    if window > len(values):
        return []
    return [
        sum(values[index:index + window]) / window
        for index in range(len(values) - window + 1)
    ]
""",
        buggy_source="""def moving_average(values, window):
    if window <= 0:
        return []
    return [
        sum(values[index:index + window]) / window
        for index in range(len(values) - window)
    ]
""",
        visible_tests=_tests("""
def test_basic_windows():
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
"""),
        hidden_tests=_tests("""
import pytest

def test_window_equal_length():
    assert moving_average([2, 4, 6], 3) == [4]

def test_invalid_window():
    with pytest.raises(ValueError):
        moving_average([1, 2], 0)
"""),
        near_miss_source="""def moving_average(values, window):
    if window <= 0:
        return []
    if window > len(values):
        return []
    return [
        sum(values[index:index + window]) / window
        for index in range(len(values) - window + 1)
    ]
""",
        visible_overfit_source="""def moving_average(values, window):
    if values == [1, 2, 3, 4] and window == 2:
        return [1.5, 2.5, 3.5]
    if window <= 0:
        return []
    return [
        sum(values[index:index + window]) / window
        for index in range(len(values) - window)
    ]
""",
    ),
    TaskSpec(
        task_id="safe_get_default",
        split="train",
        bug_family="exception_case",
        issue="`safe_get(mapping, path, default)` should traverse dotted dictionary paths and return the default when any segment is missing.",
        clean_source="""def safe_get(mapping, path, default=None):
    current = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
""",
        buggy_source="""def safe_get(mapping, path, default=None):
    current = mapping
    for part in path.split("."):
        current = current[part]
    return current
""",
        visible_tests=_tests("""
def test_present_path():
    assert safe_get({"a": {"b": 3}}, "a.b") == 3

def test_missing_path():
    assert safe_get({"a": {}}, "a.b", default="x") == "x"
"""),
        hidden_tests=_tests("""
def test_non_dict_midpoint():
    assert safe_get({"a": 1}, "a.b", default=None) is None

def test_top_level_missing():
    assert safe_get({}, "missing", default=42) == 42
"""),
        near_miss_source="""def safe_get(mapping, path, default=None):
    current = mapping
    for part in path.split("."):
        if part not in current:
            return default
        current = current[part]
    return current
""",
        visible_overfit_source="""def safe_get(mapping, path, default=None):
    if mapping == {"a": {}} and path == "a.b":
        return default
    current = mapping
    for part in path.split("."):
        current = current[part]
    return current
""",
    ),
    TaskSpec(
        task_id="normalize_path_parent",
        split="val_synth",
        bug_family="path_norm",
        issue="`normalize_path` should collapse duplicate slashes, ignore '.', and resolve '..' without escaping above root.",
        clean_source="""def normalize_path(path):
    parts = []
    for part in path.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/" + "/".join(parts)
""",
        buggy_source="""def normalize_path(path):
    parts = [part for part in path.split("/") if part]
    return "/" + "/".join(parts)
""",
        visible_tests=_tests("""
def test_duplicate_slashes():
    assert normalize_path("/a//b") == "/a/b"

def test_parent_directory():
    assert normalize_path("/a/b/../c") == "/a/c"
"""),
        hidden_tests=_tests("""
def test_current_directory_ignored():
    assert normalize_path("/a/./b") == "/a/b"

def test_parent_at_root():
    assert normalize_path("/../a") == "/a"
"""),
        near_miss_source="""def normalize_path(path):
    parts = []
    for part in path.split("/"):
        if part in {"", "."}:
            continue
        if part == ".." and parts:
            parts.pop()
            continue
        parts.append(part)
    return "/" + "/".join(parts)
""",
        visible_overfit_source="""def normalize_path(path):
    if path == "/a/b/../c":
        return "/a/c"
    parts = [part for part in path.split("/") if part]
    return "/" + "/".join(parts)
""",
    ),
    TaskSpec(
        task_id="top_k_tiebreak",
        split="val_synth",
        bug_family="ordering",
        issue="`top_k_words` should rank words by frequency descending and alphabetically ascending to break ties.",
        clean_source="""from collections import Counter


def top_k_words(words, k):
    counts = Counter(words)
    return [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:k]]
""",
        buggy_source="""from collections import Counter


def top_k_words(words, k):
    counts = Counter(words)
    return [word for word, _ in counts.most_common(k)]
""",
        visible_tests=_tests("""
def test_frequency_order():
    assert top_k_words(["b", "a", "b"], 1) == ["b"]

def test_tie_break_alphabetical():
    assert top_k_words(["b", "a"], 2) == ["a", "b"]
"""),
        hidden_tests=_tests("""
def test_limit_after_tie_sort():
    assert top_k_words(["c", "b", "a"], 2) == ["a", "b"]
"""),
        near_miss_source="""from collections import Counter


def top_k_words(words, k):
    counts = Counter(words)
    return [word for word, _ in sorted(counts.items(), key=lambda item: item[0])[:k]]
""",
        visible_overfit_source="""from collections import Counter


def top_k_words(words, k):
    if words == ["b", "a"] and k == 2:
        return ["a", "b"]
    counts = Counter(words)
    return [word for word, _ in counts.most_common(k)]
""",
    ),
    TaskSpec(
        task_id="format_bytes_units",
        split="val_synth",
        bug_family="boundary_condition",
        issue="`format_bytes` should format bytes using binary units and one decimal place after the first unit conversion.",
        clean_source="""UNITS = ["B", "KiB", "MiB", "GiB"]


def format_bytes(size):
    value = float(size)
    unit = 0
    while value >= 1024 and unit < len(UNITS) - 1:
        value /= 1024
        unit += 1
    if unit == 0:
        return f"{int(value)} B"
    return f"{value:.1f} {UNITS[unit]}"
""",
        buggy_source="""UNITS = ["B", "KB", "MB", "GB"]


def format_bytes(size):
    value = float(size)
    unit = 0
    while value > 1024 and unit < len(UNITS) - 1:
        value /= 1000
        unit += 1
    return f"{value:.1f} {UNITS[unit]}"
""",
        visible_tests=_tests("""
def test_exact_kib():
    assert format_bytes(1024) == "1.0 KiB"
"""),
        hidden_tests=_tests("""
def test_bytes_no_decimal():
    assert format_bytes(512) == "512 B"

def test_mib():
    assert format_bytes(1536) == "1.5 KiB"
"""),
        near_miss_source="""UNITS = ["B", "KiB", "MiB", "GiB"]


def format_bytes(size):
    value = float(size)
    unit = 0
    while value >= 1024 and unit < len(UNITS) - 1:
        value /= 1024
        unit += 1
    return f"{value:.1f} {UNITS[unit]}"
""",
        visible_overfit_source="""UNITS = ["B", "KB", "MB", "GB"]


def format_bytes(size):
    if size == 1024:
        return "1.0 KiB"
    value = float(size)
    unit = 0
    while value > 1024 and unit < len(UNITS) - 1:
        value /= 1000
        unit += 1
    return f"{value:.1f} {UNITS[unit]}"
""",
    ),
]
