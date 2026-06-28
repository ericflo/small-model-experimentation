#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from repair_experiment.patching import apply_patch_to_files, unified_diff_for_files  # noqa: E402
from repair_experiment.runner import run_pytest, syntax_valid  # noqa: E402


MODULE_PATH = "src/repair_target.py"
VISIBLE_TEST_PATH = "tests/test_visible.py"
HIDDEN_TEST_PATH = "tests/test_hidden.py"


@dataclass(frozen=True)
class TokenStyle:
    name: str
    prefix: str
    alphabet: str
    groups: int
    group_len: int
    separator: str


TRAIN_STYLES = [
    TokenStyle("dash_upper", "CANON", string.ascii_uppercase + string.digits, 2, 4, "-"),
    TokenStyle("underscore_upper", "SIG", string.ascii_uppercase + string.digits, 3, 3, "_"),
    TokenStyle("mixed_hex", "key", string.ascii_lowercase + string.digits, 2, 5, "-"),
]

HOLDOUT_STYLES = [
    TokenStyle("colon_upper", "TRACE", string.ascii_uppercase + string.digits, 3, 2, ":"),
    TokenStyle("dot_lower", "audit", string.ascii_lowercase + string.digits, 3, 3, "."),
]


def token_for(rng: random.Random, style: TokenStyle) -> str:
    parts = [
        "".join(rng.choice(style.alphabet) for _ in range(style.group_len))
        for _ in range(style.groups)
    ]
    return style.prefix + style.separator + style.separator.join(parts)


def source_for(token: str) -> str:
    return f'''"""Tiny target module for trace-keyed repair."""

CANONICAL_TOKEN = "{token}"


def canonical_token(case_id):
    """Return the externally validated canonical token for a case."""
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a non-empty string")
    return CANONICAL_TOKEN
'''


def visible_test_for(case_id: str, expected: str) -> str:
    return f'''from repair_target import canonical_token


def test_trace_reveals_expected_token():
    actual = canonical_token("{case_id}")
    assert actual == "{expected}", (
        "TRACE_KEY expected_token={expected} actual_token=" + actual
    )
'''


def hidden_test_for(case_id: str, expected: str) -> str:
    return f'''import pytest
from repair_target import canonical_token


def test_hidden_expected_token_exact():
    assert canonical_token("{case_id}") == "{expected}"


def test_hidden_rejects_empty_case_id():
    with pytest.raises(ValueError):
        canonical_token("")
'''


def make_record(
    *,
    split: str,
    index: int,
    style: TokenStyle,
    rng: random.Random,
) -> dict[str, Any]:
    case_id = f"{split}_{style.name}_{index:04d}"
    expected = token_for(rng, style)
    wrong = token_for(rng, style)
    while wrong == expected:
        wrong = token_for(rng, style)
    buggy = "__TRACE_WILL_REVEAL_EXPECTED_TOKEN__"

    buggy_files = {MODULE_PATH: source_for(buggy)}
    wrong_files = {MODULE_PATH: source_for(wrong)}
    clean_files = {MODULE_PATH: source_for(expected)}
    visible_tests = {VISIBLE_TEST_PATH: visible_test_for(case_id, expected)}
    hidden_tests = {HIDDEN_TEST_PATH: hidden_test_for(case_id, expected)}

    wrong_patch = unified_diff_for_files(buggy_files, wrong_files)
    target_next_diff = unified_diff_for_files(wrong_files, clean_files)
    base_buggy_diff = unified_diff_for_files(buggy_files, clean_files)
    wrong_visible = run_pytest(wrong_files, visible_tests, hidden_tests, which="visible")
    target_applied, repaired_files, target_apply_output = apply_patch_to_files(wrong_files, target_next_diff)
    repaired_visible = (
        run_pytest(repaired_files, visible_tests, hidden_tests, which="visible")
        if target_applied
        else {"passed": False, "output": target_apply_output}
    )
    repaired_hidden = (
        run_pytest(repaired_files, visible_tests, hidden_tests, which="hidden")
        if target_applied
        else {"passed": False, "output": target_apply_output}
    )
    syntax_ok, syntax_error = syntax_valid(repaired_files) if target_applied else (False, target_apply_output)

    return {
        "task_id": f"{split}_{style.name}_{index:04d}",
        "episode_id": f"{split}_{style.name}_{index:04d}::wrong_token",
        "split": split,
        "issue": (
            "The canonical_token function must return the validator-approved token for the requested case. "
            "The repository does not contain the approved token; use the failing test output to identify it."
        ),
        "buggy_files": buggy_files,
        "current_files": wrong_files,
        "wrong_patch": wrong_patch,
        "test_output_after_wrong_patch": wrong_visible["output"],
        "target_next_diff": target_next_diff,
        "base_buggy_diff": base_buggy_diff,
        "visible_tests": visible_tests,
        "hidden_tests": hidden_tests,
        "metadata": {
            "bug_family": "trace_keyed_literal",
            "failure_class": "wrong_constant",
            "token_style": style.name,
            "expected_token": expected,
            "wrong_token": wrong,
            "case_id": case_id,
            "wrong_visible_passed": wrong_visible["passed"],
            "target_patch_applied": target_applied,
            "target_visible_passed": repaired_visible["passed"],
            "target_hidden_passed": repaired_hidden["passed"],
            "target_syntax_valid": syntax_ok,
            "target_syntax_error": syntax_error,
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in records) + "\n",
        encoding="utf-8",
    )


def assert_records(records: list[dict[str, Any]]) -> None:
    for row in records:
        meta = row["metadata"]
        if meta["wrong_visible_passed"]:
            raise AssertionError(f"wrong patch unexpectedly passed visible tests: {row['episode_id']}")
        for key in ["target_patch_applied", "target_visible_passed", "target_hidden_passed", "target_syntax_valid"]:
            if not meta[key]:
                raise AssertionError(f"target check failed for {row['episode_id']}: {key}")
        expected = meta["expected_token"]
        wrong = meta["wrong_token"]
        context = json.dumps(row["current_files"], sort_keys=True)
        if expected in context:
            raise AssertionError(f"expected token leaked into current files: {row['episode_id']}")
        if expected not in row["test_output_after_wrong_patch"]:
            raise AssertionError(f"expected token missing from trace: {row['episode_id']}")
        if wrong not in context:
            raise AssertionError(f"wrong token missing from current files: {row['episode_id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/trace_keyed_symbol_repair/data"))
    parser.add_argument("--train", type=int, default=240)
    parser.add_argument("--iid", type=int, default=60)
    parser.add_argument("--holdout", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train = [
        make_record(split="train", index=i, style=TRAIN_STYLES[i % len(TRAIN_STYLES)], rng=rng)
        for i in range(args.train)
    ]
    iid = [
        make_record(split="val_iid", index=i, style=TRAIN_STYLES[i % len(TRAIN_STYLES)], rng=rng)
        for i in range(args.iid)
    ]
    holdout = [
        make_record(split="val_format_holdout", index=i, style=HOLDOUT_STYLES[i % len(HOLDOUT_STYLES)], rng=rng)
        for i in range(args.holdout)
    ]
    all_records = train + iid + holdout
    assert_records(all_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "repair_train.jsonl", train)
    write_jsonl(args.output_dir / "repair_val_iid.jsonl", iid)
    write_jsonl(args.output_dir / "repair_val_format_holdout.jsonl", holdout)
    write_jsonl(args.output_dir / "repair_all.jsonl", all_records)

    manifest = {
        "seed": args.seed,
        "records": {
            "train": len(train),
            "val_iid": len(iid),
            "val_format_holdout": len(holdout),
            "all": len(all_records),
        },
        "train_token_styles": [style.name for style in TRAIN_STYLES],
        "holdout_token_styles": [style.name for style in HOLDOUT_STYLES],
        "paths": {
            "train": str(args.output_dir / "repair_train.jsonl"),
            "val_iid": str(args.output_dir / "repair_val_iid.jsonl"),
            "val_format_holdout": str(args.output_dir / "repair_val_format_holdout.jsonl"),
            "all": str(args.output_dir / "repair_all.jsonl"),
        },
        "invariant": "expected token is absent from current files and present in failing trace",
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
