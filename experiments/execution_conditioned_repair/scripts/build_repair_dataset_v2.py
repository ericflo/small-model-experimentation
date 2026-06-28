#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repair_experiment.patching import patch_stats, unified_diff_for_files
from repair_experiment.runner import classify_failure, run_pytest, syntax_valid
from repair_experiment.tasks import (
    HIDDEN_TEST_PATH,
    MODULE_PATH,
    VISIBLE_TEST_PATH,
    TaskSpec,
)


def tests(body: str) -> str:
    return "from repair_target import *\n\n" + body.strip() + "\n"


def quote(value) -> str:
    return repr(value)


def clamp_task(index: int, split: str) -> TaskSpec:
    fn = f"clamp_value_{index}"
    low = index % 4 - 2
    high = low + 8 + (index % 5)
    below = low - 3
    above = high + 4
    inside = low + 2
    return TaskSpec(
        task_id=f"{split}_clamp_{index}",
        split=split,
        bug_family="comparison_flip",
        issue=f"`{fn}` should clamp a value into an inclusive range and reject inverted bounds.",
        clean_source=f"""def {fn}(value, lower, upper):
    if lower > upper:
        raise ValueError("lower bound cannot exceed upper bound")
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value
""",
        buggy_source=f"""def {fn}(value, lower, upper):
    if lower > upper:
        raise ValueError("lower bound cannot exceed upper bound")
    if value < lower:
        return upper
    if value > upper:
        return lower
    return value
""",
        visible_tests=tests(f"""
def test_lower_bound():
    assert {fn}({below}, {low}, {high}) == {low}

def test_inside_interval():
    assert {fn}({inside}, {low}, {high}) == {inside}
"""),
        hidden_tests=tests(f"""
import pytest

def test_upper_bound():
    assert {fn}({above}, {low}, {high}) == {high}

def test_invalid_bounds():
    with pytest.raises(ValueError):
        {fn}(1, {high}, {low})
"""),
        near_miss_source=f"""def {fn}(value, lower, upper):
    if lower > upper:
        raise ValueError("lower bound cannot exceed upper bound")
    if value < lower:
        return lower
    if value > upper:
        return value
    return value
""",
        visible_overfit_source=f"""def {fn}(value, lower, upper):
    if value == {below} and lower == {low} and upper == {high}:
        return {low}
    if lower > upper:
        raise ValueError("lower bound cannot exceed upper bound")
    if value < lower:
        return upper
    if value > upper:
        return lower
    return value
""",
    )


def parse_bool_task(index: int, split: str) -> TaskSpec:
    fn = f"parse_bool_{index}"
    true_visible = "YES" if index % 2 else "ON"
    false_hidden = "No" if index % 2 else "OFF"
    unknown = f"maybe_{index}"
    return TaskSpec(
        task_id=f"{split}_parse_bool_{index}",
        split=split,
        bug_family="case_normalization",
        issue=f"`{fn}` should parse common boolean words case-insensitively and reject unknown values.",
        clean_source=f"""def {fn}(text):
    value = str(text).strip().lower()
    if value in {{"true", "yes", "1", "on"}}:
        return True
    if value in {{"false", "no", "0", "off"}}:
        return False
    raise ValueError("not a boolean")
""",
        buggy_source=f"""def {fn}(text):
    value = str(text).strip()
    if value in {{"true", "yes", "1", "on"}}:
        return True
    if value in {{"false", "no", "0", "off"}}:
        return False
    return bool(value)
""",
        visible_tests=tests(f"""
def test_true_word_case_insensitive():
    assert {fn}({quote(true_visible)}) is True

def test_false_word_lowercase():
    assert {fn}("off") is False
"""),
        hidden_tests=tests(f"""
import pytest

def test_false_word_case_insensitive():
    assert {fn}({quote(false_hidden)}) is False

def test_unknown_rejected():
    with pytest.raises(ValueError):
        {fn}({quote(unknown)})
"""),
        near_miss_source=f"""def {fn}(text):
    value = str(text).strip().lower()
    if value in {{"true", "yes", "1", "on"}}:
        return True
    if value in {{"false", "no", "0", "off"}}:
        return False
    return bool(value)
""",
        visible_overfit_source=f"""def {fn}(text):
    if text == {quote(true_visible)}:
        return True
    value = str(text).strip()
    if value in {{"true", "yes", "1", "on"}}:
        return True
    if value in {{"false", "no", "0", "off"}}:
        return False
    return bool(value)
""",
    )


def moving_average_task(index: int, split: str) -> TaskSpec:
    fn = f"moving_average_{index}"
    start = index % 7
    values = [start + 1, start + 3, start + 5, start + 7]
    visible_expected = [(values[0] + values[1]) / 2, (values[1] + values[2]) / 2, (values[2] + values[3]) / 2]
    hidden_values = [start + 2, start + 4, start + 8]
    hidden_expected = [sum(hidden_values) / 3]
    return TaskSpec(
        task_id=f"{split}_moving_average_{index}",
        split=split,
        bug_family="off_by_one",
        issue=f"`{fn}` should emit every full sliding-window average and reject non-positive window sizes.",
        clean_source=f"""def {fn}(values, window):
    if window <= 0:
        raise ValueError("window must be positive")
    if window > len(values):
        return []
    return [
        sum(values[index:index + window]) / window
        for index in range(len(values) - window + 1)
    ]
""",
        buggy_source=f"""def {fn}(values, window):
    if window <= 0:
        return []
    return [
        sum(values[index:index + window]) / window
        for index in range(len(values) - window)
    ]
""",
        visible_tests=tests(f"""
def test_basic_windows():
    assert {fn}({values!r}, 2) == {visible_expected!r}
"""),
        hidden_tests=tests(f"""
import pytest

def test_window_equal_length():
    assert {fn}({hidden_values!r}, 3) == {hidden_expected!r}

def test_invalid_window():
    with pytest.raises(ValueError):
        {fn}([1, 2], 0)
"""),
        near_miss_source=f"""def {fn}(values, window):
    if window <= 0:
        return []
    if window > len(values):
        return []
    return [
        sum(values[index:index + window]) / window
        for index in range(len(values) - window + 1)
    ]
""",
        visible_overfit_source=f"""def {fn}(values, window):
    if values == {values!r} and window == 2:
        return {visible_expected!r}
    if window <= 0:
        return []
    return [
        sum(values[index:index + window]) / window
        for index in range(len(values) - window)
    ]
""",
    )


def dedupe_task(index: int, split: str) -> TaskSpec:
    fn = f"dedupe_preserve_order_{index}"
    visible = ["b", f"a{index}", "b"]
    hidden = [index + 3, index + 1, index + 3, index + 2, index + 1]
    hidden_expected = [index + 3, index + 1, index + 2]
    return TaskSpec(
        task_id=f"{split}_dedupe_{index}",
        split=split,
        bug_family="ordering",
        issue=f"`{fn}` should remove later duplicates without sorting or changing first-occurrence order.",
        clean_source=f"""def {fn}(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
""",
        buggy_source=f"""def {fn}(items):
    return sorted(set(items))
""",
        visible_tests=tests(f"""
def test_keeps_first_occurrence_order():
    assert {fn}({visible!r}) == {visible[:2]!r}
"""),
        hidden_tests=tests(f"""
def test_numbers_keep_order():
    assert {fn}({hidden!r}) == {hidden_expected!r}

def test_empty():
    assert {fn}([]) == []
"""),
        near_miss_source=f"""def {fn}(items):
    return list(dict.fromkeys(sorted(items)))
""",
        visible_overfit_source=f"""def {fn}(items):
    if items == {visible!r}:
        return {visible[:2]!r}
    return sorted(set(items))
""",
    )


def safe_get_task(index: int, split: str) -> TaskSpec:
    fn = f"safe_get_{index}"
    default = f"missing_{index}"
    return TaskSpec(
        task_id=f"{split}_safe_get_{index}",
        split=split,
        bug_family="exception_case",
        issue=f"`{fn}` should traverse dotted dictionary paths and return the default when any segment is missing.",
        clean_source=f"""def {fn}(mapping, path, default=None):
    current = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
""",
        buggy_source=f"""def {fn}(mapping, path, default=None):
    current = mapping
    for part in path.split("."):
        current = current[part]
    return current
""",
        visible_tests=tests(f"""
def test_present_path():
    assert {fn}({{"a": {{"b": {index}}}}}, "a.b") == {index}

def test_missing_path():
    assert {fn}({{"a": {{}}}}, "a.b", default={quote(default)}) == {quote(default)}
"""),
        hidden_tests=tests(f"""
def test_non_dict_midpoint():
    assert {fn}({{"a": 1}}, "a.b", default=None) is None

def test_top_level_missing():
    assert {fn}({{}}, "missing", default=42) == 42
"""),
        near_miss_source=f"""def {fn}(mapping, path, default=None):
    current = mapping
    for part in path.split("."):
        if part not in current:
            return default
        current = current[part]
    return current
""",
        visible_overfit_source=f"""def {fn}(mapping, path, default=None):
    if mapping == {{"a": {{}}}} and path == "a.b":
        return default
    current = mapping
    for part in path.split("."):
        current = current[part]
    return current
""",
    )


def slugify_task(index: int, split: str) -> TaskSpec:
    fn = f"slugify_{index}"
    visible_text = f"Hello   World {index}"
    visible_slug = f"hello-world-{index}"
    hidden_text = f" **Ready, Set, Go {index}!** "
    hidden_slug = f"ready-set-go-{index}"
    return TaskSpec(
        task_id=f"{split}_slugify_{index}",
        split=split,
        bug_family="normalization",
        issue=f"`{fn}` should lowercase text, replace runs of non-alphanumeric characters with one dash, and trim dashes.",
        clean_source=f"""import re


def {fn}(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return slug.strip("-")
""",
        buggy_source=f"""import re


def {fn}(text):
    return re.sub(r"[^a-z0-9]", "-", text.lower())
""",
        visible_tests=tests(f"""
def test_spaces_collapse():
    assert {fn}({quote(visible_text)}) == {quote(visible_slug)}
"""),
        hidden_tests=tests(f"""
def test_trim_punctuation():
    assert {fn}({quote(hidden_text)}) == {quote(hidden_slug)}

def test_numbers_kept():
    assert {fn}("Qwen 3 4B") == "qwen-3-4b"
"""),
        near_miss_source=f"""import re


def {fn}(text):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return slug
""",
        visible_overfit_source=f"""import re


def {fn}(text):
    if text == {quote(visible_text)}:
        return {quote(visible_slug)}
    return re.sub(r"[^a-z0-9]", "-", text.lower())
""",
    )


def normalize_path_task(index: int, split: str) -> TaskSpec:
    fn = f"normalize_path_{index}"
    return TaskSpec(
        task_id=f"{split}_normalize_path_{index}",
        split=split,
        bug_family="path_norm",
        issue=f"`{fn}` should collapse duplicate slashes, ignore '.', and resolve '..' without escaping above root.",
        clean_source=f"""def {fn}(path):
    parts = []
    for part in path.split("/"):
        if part in {{"", "."}}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/" + "/".join(parts)
""",
        buggy_source=f"""def {fn}(path):
    parts = [part for part in path.split("/") if part]
    return "/" + "/".join(parts)
""",
        visible_tests=tests(f"""
def test_duplicate_slashes():
    assert {fn}("/a//b{index}") == "/a/b{index}"

def test_parent_directory():
    assert {fn}("/a/b/../c{index}") == "/a/c{index}"
"""),
        hidden_tests=tests(f"""
def test_current_directory_ignored():
    assert {fn}("/a/./b{index}") == "/a/b{index}"

def test_parent_at_root():
    assert {fn}("/../a{index}") == "/a{index}"
"""),
        near_miss_source=f"""def {fn}(path):
    parts = []
    for part in path.split("/"):
        if part in {{"", "."}}:
            continue
        if part == ".." and parts:
            parts.pop()
            continue
        parts.append(part)
    return "/" + "/".join(parts)
""",
        visible_overfit_source=f"""def {fn}(path):
    if path == "/a/b/../c{index}":
        return "/a/c{index}"
    parts = [part for part in path.split("/") if part]
    return "/" + "/".join(parts)
""",
    )


def top_k_task(index: int, split: str) -> TaskSpec:
    fn = f"top_k_words_{index}"
    return TaskSpec(
        task_id=f"{split}_top_k_{index}",
        split=split,
        bug_family="tie_breaking",
        issue=f"`{fn}` should rank words by frequency descending and alphabetically ascending to break ties.",
        clean_source=f"""from collections import Counter


def {fn}(words, k):
    counts = Counter(words)
    return [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:k]]
""",
        buggy_source=f"""from collections import Counter


def {fn}(words, k):
    counts = Counter(words)
    return [word for word, _ in counts.most_common(k)]
""",
        visible_tests=tests(f"""
def test_frequency_order():
    assert {fn}(["b{index}", "a{index}", "b{index}"], 1) == ["b{index}"]

def test_tie_break_alphabetical():
    assert {fn}(["b{index}", "a{index}"], 2) == ["a{index}", "b{index}"]
"""),
        hidden_tests=tests(f"""
def test_limit_after_tie_sort():
    assert {fn}(["c{index}", "b{index}", "a{index}"], 2) == ["a{index}", "b{index}"]
"""),
        near_miss_source=f"""from collections import Counter


def {fn}(words, k):
    counts = Counter(words)
    return [word for word, _ in sorted(counts.items(), key=lambda item: item[0])[:k]]
""",
        visible_overfit_source=f"""from collections import Counter


def {fn}(words, k):
    if words == ["b{index}", "a{index}"] and k == 2:
        return ["a{index}", "b{index}"]
    counts = Counter(words)
    return [word for word, _ in counts.most_common(k)]
""",
    )


TRAIN_FAMILIES = [
    clamp_task,
    parse_bool_task,
    moving_average_task,
    dedupe_task,
    safe_get_task,
    slugify_task,
]

HELDOUT_FAMILIES = [
    normalize_path_task,
    top_k_task,
]


def build_episode(task: TaskSpec, variant) -> dict:
    clean_files = {MODULE_PATH: task.clean_source}
    buggy_files = {MODULE_PATH: task.buggy_source}
    current_files = {MODULE_PATH: variant.source}
    visible_tests = {VISIBLE_TEST_PATH: task.visible_tests}
    hidden_tests = {HIDDEN_TEST_PATH: task.hidden_tests}

    base_buggy_diff = unified_diff_for_files(buggy_files, clean_files)
    wrong_patch = unified_diff_for_files(buggy_files, current_files)
    target_next_diff = unified_diff_for_files(current_files, clean_files)

    visible = run_pytest(current_files, visible_tests, hidden_tests, which="visible")
    hidden = run_pytest(current_files, visible_tests, hidden_tests, which="hidden")
    syntax_ok, syntax_error = syntax_valid(current_files)
    failure_class = classify_failure(
        visible["output"] if visible["output"].strip() else syntax_error,
        visible["passed"],
        hidden["passed"],
        "applied",
    )
    stats = patch_stats(target_next_diff)
    return {
        "task_id": task.task_id,
        "episode_id": f"{task.task_id}::{variant.name}",
        "repo": "local_synthetic_python_v2",
        "split": task.split,
        "issue": task.issue,
        "opened_files": [{"path": MODULE_PATH, "content": variant.source}],
        "clean_files": clean_files,
        "buggy_files": buggy_files,
        "current_files": current_files,
        "visible_tests": visible_tests,
        "hidden_tests": hidden_tests,
        "base_buggy_diff": base_buggy_diff,
        "wrong_patch": wrong_patch,
        "wrong_patch_apply_status": "applied",
        "test_output_after_wrong_patch": visible["output"],
        "target_next_diff": target_next_diff,
        "metadata": {
            "generator": "scripted_wrong_patch_v2",
            "wrong_patch_variant": variant.name,
            "wrong_patch_touched_gold_file": variant.touched_gold_file,
            "wrong_patch_touched_gold_function": variant.touched_gold_function,
            "failure_class": failure_class,
            "bug_family": task.bug_family,
            "visible_tests_passed": visible["passed"],
            "hidden_tests_passed": hidden["passed"],
            "syntax_valid": syntax_ok,
            "target_files_touched": stats["files_touched"],
            "target_added_lines": stats["added_lines"],
            "target_removed_lines": stats["removed_lines"],
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def make_tasks(train_per_family: int, val_iid_per_family: int, val_holdout_per_family: int) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for family in TRAIN_FAMILIES:
        for index in range(train_per_family):
            tasks.append(family(index, "train"))
        for index in range(train_per_family, train_per_family + val_iid_per_family):
            tasks.append(family(index, "val_synth_iid"))
    for family in HELDOUT_FAMILIES:
        for index in range(val_holdout_per_family):
            tasks.append(family(index, "val_synth_family_holdout"))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/v2"))
    parser.add_argument("--train-per-family", type=int, default=8)
    parser.add_argument("--val-iid-per-family", type=int, default=2)
    parser.add_argument("--val-holdout-per-family", type=int, default=3)
    parser.add_argument("--include-passing-wrong-patches", action="store_true")
    args = parser.parse_args()

    tasks = make_tasks(
        args.train_per_family,
        args.val_iid_per_family,
        args.val_holdout_per_family,
    )
    episodes: list[dict] = []
    skipped: list[dict] = []
    for task in tasks:
        for variant in task.wrong_variants():
            episode = build_episode(task, variant)
            if episode["metadata"]["hidden_tests_passed"] and not args.include_passing_wrong_patches:
                skipped.append(
                    {
                        "episode_id": episode["episode_id"],
                        "reason": "wrong patch already passes hidden tests",
                    }
                )
                continue
            episodes.append(episode)

    train = [row for row in episodes if row["split"] == "train"]
    val_iid = [row for row in episodes if row["split"] == "val_synth_iid"]
    val_holdout = [row for row in episodes if row["split"] == "val_synth_family_holdout"]
    val_all = val_iid + val_holdout

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "repair_train.jsonl", train)
    write_jsonl(args.output_dir / "repair_val_synth_iid.jsonl", val_iid)
    write_jsonl(args.output_dir / "repair_val_synth_family_holdout.jsonl", val_holdout)
    write_jsonl(args.output_dir / "repair_val_synth.jsonl", val_all)
    write_jsonl(args.output_dir / "repair_all_synth.jsonl", episodes)

    manifest = {
        "dataset": "local_synthetic_python_v2",
        "notes": [
            "Container-free executable synthetic expansion.",
            "Train split uses six parameterized bug families.",
            "Validation includes both same-family IID tasks and held-out-family tasks.",
            "Every target_next_diff is computed from the wrong-patched source state to the clean source state.",
        ],
        "num_tasks": len(tasks),
        "num_episodes": len(episodes),
        "num_skipped": len(skipped),
        "splits": Counter(row["split"] for row in episodes),
        "failure_classes": Counter(row["metadata"]["failure_class"] for row in episodes),
        "wrong_patch_variants": Counter(row["metadata"]["wrong_patch_variant"] for row in episodes),
        "bug_families": Counter(row["metadata"]["bug_family"] for row in episodes),
        "task_ids": sorted({row["task_id"] for row in episodes}),
        "train_families": [family(0, "train").bug_family for family in TRAIN_FAMILIES],
        "heldout_families": [family(0, "val_synth_family_holdout").bug_family for family in HELDOUT_FAMILIES],
        "skipped": skipped,
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
