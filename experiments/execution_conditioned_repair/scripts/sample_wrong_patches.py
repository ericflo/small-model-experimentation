#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repair_experiment.modeling import load_model_for_generation, load_tokenizer, render_generation_prompt  # noqa: E402
from repair_experiment.patching import apply_patch_to_files, extract_unified_diff, unified_diff_for_files  # noqa: E402
from repair_experiment.runner import classify_failure, run_pytest, syntax_valid  # noqa: E402
from repair_experiment.tasks import HIDDEN_TEST_PATH, MODULE_PATH, TASKS, VISIBLE_TEST_PATH  # noqa: E402


def base_record_for_task(task) -> dict:
    return {
        "task_id": task.task_id,
        "episode_id": f"{task.task_id}::frozen_sample_seed",
        "repo": "local_synthetic_python",
        "split": task.split,
        "issue": task.issue,
        "clean_files": {MODULE_PATH: task.clean_source},
        "buggy_files": {MODULE_PATH: task.buggy_source},
        "current_files": {MODULE_PATH: task.buggy_source},
        "visible_tests": {VISIBLE_TEST_PATH: task.visible_tests},
        "hidden_tests": {HIDDEN_TEST_PATH: task.hidden_tests},
        "base_buggy_diff": unified_diff_for_files({MODULE_PATH: task.buggy_source}, {MODULE_PATH: task.clean_source}),
        "wrong_patch": "",
        "target_next_diff": "",
        "test_output_after_wrong_patch": "",
        "metadata": {"bug_family": task.bug_family},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="cdbee75f17c01a7cc42f958dc650907174af0554")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_id, args.revision)
    model = load_model_for_generation(args.model_id, args.revision, load_in_4bit=True)
    tasks = TASKS[: args.max_tasks] if args.max_tasks else TASKS
    rows: list[dict] = []
    for task in tqdm(tasks, desc="tasks"):
        base_record = base_record_for_task(task)
        prompt = render_generation_prompt(tokenizer, base_record, "final_patch")
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.samples_per_task,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for sample_idx, output in enumerate(output_ids):
            completion = tokenizer.decode(output[encoded["input_ids"].shape[1] :], skip_special_tokens=True)
            wrong_patch = extract_unified_diff(completion)
            applied, current_files, apply_output = apply_patch_to_files(base_record["buggy_files"], wrong_patch)
            if not applied:
                current_files = base_record["buggy_files"]
            visible = run_pytest(current_files, base_record["visible_tests"], base_record["hidden_tests"], which="visible")
            hidden = run_pytest(current_files, base_record["visible_tests"], base_record["hidden_tests"], which="hidden")
            syntax_ok, syntax_error = syntax_valid(current_files)
            target = unified_diff_for_files(current_files, base_record["clean_files"])
            failure_class = classify_failure(
                visible["output"] if visible["output"].strip() else syntax_error,
                visible["passed"],
                hidden["passed"],
                "applied" if applied else "rejected",
            )
            row = dict(base_record)
            row.update(
                {
                    "episode_id": f"{task.task_id}::frozen_sample_{sample_idx}",
                    "opened_files": [{"path": MODULE_PATH, "content": current_files[MODULE_PATH]}],
                    "current_files": current_files,
                    "wrong_patch": wrong_patch,
                    "wrong_patch_apply_status": "applied" if applied else "rejected",
                    "wrong_patch_apply_output": apply_output,
                    "test_output_after_wrong_patch": visible["output"],
                    "target_next_diff": target,
                    "metadata": {
                        "generator": args.model_id,
                        "generator_revision": args.revision,
                        "wrong_patch_variant": "frozen_model_sample",
                        "wrong_patch_touched_gold_file": "src/repair_target.py" in wrong_patch,
                        "wrong_patch_touched_gold_function": "unknown",
                        "failure_class": failure_class,
                        "bug_family": task.bug_family,
                        "visible_tests_passed": visible["passed"],
                        "hidden_tests_passed": hidden["passed"],
                        "syntax_valid": syntax_ok,
                    },
                    "raw_completion": completion,
                }
            )
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"records": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
