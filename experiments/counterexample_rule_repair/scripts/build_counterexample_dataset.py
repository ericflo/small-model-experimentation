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


BUGGY_FILE = '''"""Repair target for counterexample-to-rule experiments."""


def apply_rule(value):
    raise NotImplementedError("validator rule has not been implemented")
'''

ISSUE = (
    "The apply_rule function must match the validator's hidden transformation. "
    "The repository does not state the exact rule. Use the failed-test counterexamples "
    "to infer a compact general rule, then patch the implementation so it also passes "
    "unseen inputs."
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


def py_repr(value: Any) -> str:
    return repr(value)


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
    return f'''"""Repair target for counterexample-to-rule experiments."""


def apply_rule(value):
    return {slope} * value + {intercept}
'''


def make_affine(rng: random.Random, split: str, index: int) -> CaseBundle:
    if split == "val_format_holdout":
        slope = rng.choice([-11, -9, -7, 8, 10, 12])
        intercept = rng.choice([-31, -23, 19, 29, 37])
    else:
        slope = rng.choice([2, 3, 4, 5, 6, 7])
        intercept = rng.choice([-9, -5, -2, 3, 6, 9])
    wrong_slope = slope + rng.choice([2, 3, -2, -3])
    wrong_intercept = intercept + rng.choice([4, -4, 7, -7])
    visible_inputs = [0, 1, 3]
    hidden_inputs = [-2, 2, 5, 8]
    visible = [(x, slope * x + intercept) for x in visible_inputs]
    hidden = [(x, slope * x + intercept) for x in hidden_inputs]
    task_id = f"{split}_affine_{index:04d}"
    return CaseBundle(
        task_id=task_id,
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
    return f'''"""Repair target for counterexample-to-rule experiments."""


def apply_rule(value):
    if value < {threshold}:
        return "{low_label}"
    return "{high_label}"
'''


def token(rng: random.Random, prefix: str, length: int = 4, *, lower: bool = False) -> str:
    alphabet = string.ascii_lowercase if lower else string.ascii_uppercase + string.digits
    body = "".join(rng.choice(alphabet) for _ in range(length))
    return f"{prefix}{body}"


def make_threshold(rng: random.Random, split: str, index: int) -> CaseBundle:
    if split == "val_format_holdout":
        threshold = rng.choice([-8, -5, 11, 14])
        low_label = token(rng, "low:", 5, lower=True)
        high_label = token(rng, "hi:", 5, lower=True)
    else:
        threshold = rng.choice([-3, -1, 2, 4, 6, 9])
        low_label = token(rng, "LOW_", 4)
        high_label = token(rng, "HIGH_", 4)
    wrong_threshold = threshold + rng.choice([-3, -2, 2, 3])
    wrong_low = token(rng, "BADL_", 4)
    wrong_high = token(rng, "BADH_", 4)

    def rule(x: int) -> str:
        return low_label if x < threshold else high_label

    visible_inputs = [threshold - 2, threshold - 1, threshold, threshold + 2]
    hidden_inputs = [threshold - 5, threshold + 1, threshold + 4, threshold + 7]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    task_id = f"{split}_threshold_{index:04d}"
    return CaseBundle(
        task_id=task_id,
        split=split,
        family="threshold_label",
        current_source=source_threshold(wrong_threshold, wrong_low, wrong_high),
        target_source=source_threshold(threshold, low_label, high_label),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[str(threshold), low_label, high_label],
        params={
            "threshold": threshold,
            "low_label": low_label,
            "high_label": high_label,
            "wrong_threshold": wrong_threshold,
            "wrong_low_label": wrong_low,
            "wrong_high_label": wrong_high,
        },
    )


def source_slug(prefix: str, suffix: str, separator: str) -> str:
    return f'''"""Repair target for counterexample-to-rule experiments."""


def apply_rule(value):
    text = str(value).strip().lower().replace("_", " ")
    pieces = [piece for piece in text.split() if piece]
    body = "{separator}".join(pieces)
    return "{prefix}" + body + "{suffix}"
'''


def slug_expected(value: str, prefix: str, suffix: str, separator: str) -> str:
    pieces = [piece for piece in value.strip().lower().replace("_", " ").split() if piece]
    return prefix + separator.join(pieces) + suffix


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
    visible = [(text, slug_expected(text, prefix, suffix, separator)) for text in visible_inputs]
    hidden = [(text, slug_expected(text, prefix, suffix, separator)) for text in hidden_inputs]
    task_id = f"{split}_slug_{index:04d}"
    return CaseBundle(
        task_id=task_id,
        split=split,
        family="slug_affix",
        current_source=source_slug(wrong_prefix, wrong_suffix, wrong_separator),
        target_source=source_slug(prefix, suffix, separator),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[prefix, suffix, separator],
        params={
            "prefix": prefix,
            "suffix": suffix,
            "separator": separator,
            "wrong_prefix": wrong_prefix,
            "wrong_suffix": wrong_suffix,
            "wrong_separator": wrong_separator,
        },
    )


def source_parity(even_offset: int, odd_offset: int) -> str:
    return f'''"""Repair target for counterexample-to-rule experiments."""


def apply_rule(value):
    if value % 2 == 0:
        return value + {even_offset}
    return value + {odd_offset}
'''


def make_parity_holdout(rng: random.Random, split: str, index: int) -> CaseBundle:
    even_offset = rng.choice([-12, -6, 4, 10, 14])
    odd_offset = rng.choice([-9, -3, 5, 11, 17])
    if even_offset == odd_offset:
        odd_offset += 3
    wrong_even = even_offset + rng.choice([2, -2, 5, -5])
    wrong_odd = odd_offset + rng.choice([3, -3, 6, -6])

    def rule(x: int) -> int:
        return x + even_offset if x % 2 == 0 else x + odd_offset

    visible_inputs = [0, 1, 4, 7]
    hidden_inputs = [-3, 2, 8, 11]
    visible = [(x, rule(x)) for x in visible_inputs]
    hidden = [(x, rule(x)) for x in hidden_inputs]
    task_id = f"{split}_parity_{index:04d}"
    return CaseBundle(
        task_id=task_id,
        split=split,
        family="parity_offset_holdout",
        current_source=source_parity(wrong_even, wrong_odd),
        target_source=source_parity(even_offset, odd_offset),
        visible_cases=visible,
        hidden_cases=hidden,
        markers=[str(even_offset), str(odd_offset)],
        params={
            "even_offset": even_offset,
            "odd_offset": odd_offset,
            "wrong_even_offset": wrong_even,
            "wrong_odd_offset": wrong_odd,
        },
    )


BUILDERS: dict[str, Callable[[random.Random, str, int], CaseBundle]] = {
    "affine_int": make_affine,
    "threshold_label": make_threshold,
    "slug_affix": make_slug,
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
        "episode_id": f"{bundle.task_id}::counterexample_rule",
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


def make_split(rng: random.Random, split: str, per_family: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family, builder in BUILDERS.items():
        for index in range(per_family):
            records.append(record_from_bundle(builder(rng, split, index)))
    rng.shuffle(records)
    return records


def make_family_holdout(rng: random.Random, count: int) -> list[dict[str, Any]]:
    records = [
        record_from_bundle(make_parity_holdout(rng, "val_rule_holdout", index))
        for index in range(count)
    ]
    rng.shuffle(records)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-family", type=int, default=80)
    parser.add_argument("--iid-per-family", type=int, default=15)
    parser.add_argument("--format-per-family", type=int, default=15)
    parser.add_argument("--rule-holdout", type=int, default=45)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train = make_split(rng, "train", args.train_per_family)
    val_iid = make_split(rng, "val_iid", args.iid_per_family)
    val_format = make_split(rng, "val_format_holdout", args.format_per_family)
    val_rule = make_family_holdout(rng, args.rule_holdout)
    all_records = train + val_iid + val_format + val_rule

    paths = {
        "train": args.output_dir / "repair_train.jsonl",
        "val_iid": args.output_dir / "repair_val_iid.jsonl",
        "val_format_holdout": args.output_dir / "repair_val_format_holdout.jsonl",
        "val_rule_holdout": args.output_dir / "repair_val_rule_holdout.jsonl",
        "all": args.output_dir / "repair_all.jsonl",
    }
    write_jsonl(paths["train"], train)
    write_jsonl(paths["val_iid"], val_iid)
    write_jsonl(paths["val_format_holdout"], val_format)
    write_jsonl(paths["val_rule_holdout"], val_rule)
    write_jsonl(paths["all"], all_records)
    manifest = {
        "dataset": "counterexample_rule_repair",
        "seed": args.seed,
        "records": {
            "train": len(train),
            "val_iid": len(val_iid),
            "val_format_holdout": len(val_format),
            "val_rule_holdout": len(val_rule),
            "all": len(all_records),
        },
        "train_families": sorted(BUILDERS),
        "rule_holdout_families": ["parity_offset_holdout"],
        "invariants": [
            "wrong-patched implementation fails visible counterexamples",
            "target corrective diff applies to the wrong-patched implementation",
            "target implementation passes visible and hidden tests",
            "hidden test inputs do not overlap visible trace inputs",
            "visible expected outputs appear in the failed execution trace",
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
