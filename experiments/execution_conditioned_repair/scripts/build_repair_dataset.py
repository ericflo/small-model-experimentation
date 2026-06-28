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
from repair_experiment.tasks import HIDDEN_TEST_PATH, MODULE_PATH, TASKS, VISIBLE_TEST_PATH


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_episode(task, variant) -> dict:
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
        "repo": "local_synthetic_python",
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
            "generator": "scripted_wrong_patch_v1",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--include-passing-wrong-patches", action="store_true")
    args = parser.parse_args()

    episodes: list[dict] = []
    skipped: list[dict] = []
    for task in TASKS:
        for variant in task.wrong_variants():
            episode = build_episode(task, variant)
            if (
                episode["metadata"]["hidden_tests_passed"]
                and not args.include_passing_wrong_patches
            ):
                skipped.append(
                    {
                        "episode_id": episode["episode_id"],
                        "reason": "wrong patch already passes hidden tests",
                    }
                )
                continue
            episodes.append(episode)

    train = [row for row in episodes if row["split"] == "train"]
    val_synth = [row for row in episodes if row["split"] == "val_synth"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "repair_train.jsonl", train)
    write_jsonl(args.output_dir / "repair_val_synth.jsonl", val_synth)
    write_jsonl(args.output_dir / "repair_all_synth.jsonl", episodes)

    manifest = {
        "dataset": "local_synthetic_python_v1",
        "notes": [
            "This is a container-free executable synthetic pilot.",
            "SWE-smith/SWE-bench Docker execution must be run separately once Docker preflight passes.",
            "Every target_next_diff is computed from the wrong-patched source state to the clean source state.",
        ],
        "num_tasks": len(TASKS),
        "num_episodes": len(episodes),
        "num_skipped": len(skipped),
        "splits": Counter(row["split"] for row in episodes),
        "failure_classes": Counter(row["metadata"]["failure_class"] for row in episodes),
        "wrong_patch_variants": Counter(row["metadata"]["wrong_patch_variant"] for row in episodes),
        "bug_families": Counter(row["metadata"]["bug_family"] for row in episodes),
        "task_ids": sorted({row["task_id"] for row in episodes}),
        "skipped": skipped,
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "repair_val_real_manifest.json").write_text(
        json.dumps(
            {
                "status": "not_generated",
                "reason": "Official SWE-bench/SWE-smith execution requires Docker preflight to pass.",
                "required_script": "scripts/eval_repair_swebench.py",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
