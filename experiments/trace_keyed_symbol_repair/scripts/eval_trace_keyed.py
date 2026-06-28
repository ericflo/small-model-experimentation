#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from repair_experiment.modeling import (  # noqa: E402
    load_jsonl,
    load_model_for_generation,
    load_tokenizer,
    render_generation_prompt,
)
from repair_experiment.patching import apply_patch_to_files, extract_unified_diff, patch_stats  # noqa: E402
from repair_experiment.runner import run_pytest, syntax_valid  # noqa: E402


def shuffled_traces(records: list[dict[str, Any]], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    traces = [row.get("test_output_after_wrong_patch", "") for row in records]
    ids = [row["episode_id"] for row in records]
    shuffled = traces[:]
    for _ in range(10):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(traces, shuffled)):
            break
    return dict(zip(ids, shuffled))


def expected_token_copied(patch: str, expected: str) -> bool:
    return expected in patch


def wrong_token_removed(patch: str, wrong: str) -> bool:
    return wrong in patch and f'+CANONICAL_TOKEN = "{wrong}"' not in patch


def evaluate_record(
    *,
    model,
    tokenizer,
    record: dict[str, Any],
    prompt_mode: str,
    trace_override: str | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = render_generation_prompt(tokenizer, record, prompt_mode, trace_override=trace_override)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
    elapsed = time.time() - start
    completion_ids = output[encoded["input_ids"].shape[1] :]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    patch = extract_unified_diff(completion)
    base_files = record["buggy_files"] if prompt_mode == "final_patch" else record["current_files"]
    applied, patched_files, apply_output = apply_patch_to_files(base_files, patch)
    visible = (
        run_pytest(patched_files, record["visible_tests"], record["hidden_tests"], which="visible")
        if applied
        else {"passed": False, "output": apply_output}
    )
    hidden = (
        run_pytest(patched_files, record["visible_tests"], record["hidden_tests"], which="hidden")
        if applied
        else {"passed": False, "output": apply_output}
    )
    syntax_ok, syntax_error = syntax_valid(patched_files) if applied else (False, apply_output)
    expected = record["metadata"]["expected_token"]
    wrong = record["metadata"]["wrong_token"]
    return {
        "episode_id": record["episode_id"],
        "task_id": record["task_id"],
        "split": record["split"],
        "token_style": record["metadata"]["token_style"],
        "expected_token": expected,
        "wrong_token": wrong,
        "completion": completion,
        "extracted_patch": patch,
        "patch_applied": applied,
        "patch_apply_output": apply_output,
        "syntax_valid": syntax_ok,
        "syntax_error": syntax_error,
        "visible_passed": visible["passed"],
        "hidden_passed": hidden["passed"],
        "visible_output": visible["output"][-3000:],
        "hidden_output": hidden["output"][-3000:],
        "expected_token_copied": expected_token_copied(patch, expected),
        "wrong_token_removed": wrong_token_removed(patch, wrong),
        "generation_seconds": elapsed,
        **patch_stats(patch),
    }


def summarize(results: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    n = len(results)
    if not n:
        return {**metadata, "records": 0}
    return {
        **metadata,
        "records": n,
        "repair@1": sum(row["hidden_passed"] for row in results) / n,
        "visible_pass_rate": sum(row["visible_passed"] for row in results) / n,
        "patch_apply_rate": sum(row["patch_applied"] for row in results) / n,
        "syntax_valid_rate": sum(row["syntax_valid"] for row in results) / n,
        "expected_token_copy_rate": sum(row["expected_token_copied"] for row in results) / n,
        "wrong_token_removed_rate": sum(row["wrong_token_removed"] for row in results) / n,
        "successes": sum(row["hidden_passed"] for row in results),
        "failures": n - sum(row["hidden_passed"] for row in results),
        "median_generation_seconds": sorted(row["generation_seconds"] for row in results)[n // 2],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=["trace", "no_trace", "shuffled_trace", "final_patch"], required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--revision", default="488639f1ff808d1d3d0ba301aef8c11461451ec5")
    parser.add_argument("--adapter")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--shuffle-seed", type=int, default=9173)
    args = parser.parse_args()

    records = load_jsonl(args.data)
    if args.max_records:
        records = records[: args.max_records]
    trace_overrides = shuffled_traces(records, args.shuffle_seed) if args.condition == "shuffled_trace" else {}
    prompt_mode = {
        "trace": "trace",
        "no_trace": "no_trace",
        "shuffled_trace": "trace",
        "final_patch": "final_patch",
    }[args.condition]

    tokenizer = load_tokenizer(args.model_id, args.revision)
    model = load_model_for_generation(args.model_id, args.revision, args.adapter, load_in_4bit=True)
    results = []
    for record in tqdm(records, desc=f"eval {args.condition}"):
        results.append(
            evaluate_record(
                model=model,
                tokenizer=tokenizer,
                record=record,
                prompt_mode=prompt_mode,
                trace_override=trace_overrides.get(record["episode_id"]),
                max_new_tokens=args.max_new_tokens,
            )
        )

    metadata = {
        "data": str(args.data),
        "condition": args.condition,
        "prompt_mode": prompt_mode,
        "model_id": args.model_id,
        "revision": args.revision,
        "adapter": args.adapter,
        "max_new_tokens": args.max_new_tokens,
        "shuffle_seed": args.shuffle_seed if args.condition == "shuffled_trace" else None,
    }
    payload = {"summary": summarize(results, metadata), "records": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
