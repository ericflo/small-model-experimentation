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
    for _ in range(20):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(traces, shuffled)):
            break
    return dict(zip(ids, shuffled))


def target_added_lines(record: dict[str, Any]) -> list[str]:
    lines = []
    for line in record.get("target_next_diff", "").splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            text = line[1:].rstrip()
            if text:
                lines.append(text)
    return lines


def target_added_lines_present(record: dict[str, Any], patch: str) -> bool:
    lines = target_added_lines(record)
    return bool(lines) and all(line in patch for line in lines)


def marker_presence(record: dict[str, Any], patch: str) -> bool:
    markers = [str(item) for item in record.get("metadata", {}).get("target_markers", [])]
    return bool(markers) and all(marker in patch for marker in markers)


def visible_input_literal_present(record: dict[str, Any], patch: str) -> bool:
    inputs = [repr(value) for value, _ in record.get("metadata", {}).get("visible_cases", [])]
    return any(value in patch for value in inputs)


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
    metadata = record.get("metadata", {})
    return {
        "episode_id": record["episode_id"],
        "task_id": record["task_id"],
        "split": record["split"],
        "bug_family": metadata.get("bug_family"),
        "completion": completion,
        "extracted_patch": patch,
        "patch_applied": applied,
        "patch_apply_output": apply_output,
        "syntax_valid": syntax_ok,
        "syntax_error": syntax_error,
        "visible_passed": visible["passed"],
        "hidden_passed": hidden["passed"],
        "all_tests_passed": bool(visible["passed"] and hidden["passed"]),
        "visible_output": visible["output"][-3000:],
        "hidden_output": hidden["output"][-3000:],
        "target_added_lines_present": target_added_lines_present(record, patch),
        "target_marker_presence": marker_presence(record, patch),
        "visible_input_literal_present": visible_input_literal_present(record, patch),
        "generation_seconds": elapsed,
        **patch_stats(patch),
    }


def rate(results: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row.get(key)) for row in results) / len(results) if results else 0.0


def summarize_subset(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if not n:
        return {"records": 0}
    return {
        "records": n,
        "repair@1": rate(results, "all_tests_passed"),
        "visible_pass_rate": rate(results, "visible_passed"),
        "hidden_pass_rate": rate(results, "hidden_passed"),
        "patch_apply_rate": rate(results, "patch_applied"),
        "syntax_valid_rate": rate(results, "syntax_valid"),
        "target_added_line_match_rate": rate(results, "target_added_lines_present"),
        "target_marker_presence_rate": rate(results, "target_marker_presence"),
        "visible_input_literal_rate": rate(results, "visible_input_literal_present"),
        "successes": sum(row["all_tests_passed"] for row in results),
        "failures": n - sum(row["all_tests_passed"] for row in results),
        "median_generation_seconds": sorted(row["generation_seconds"] for row in results)[n // 2],
    }


def summarize(results: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    summary = {**metadata, **summarize_subset(results)}
    by_family = {}
    for family in sorted({row.get("bug_family") for row in results}):
        family_rows = [row for row in results if row.get("bug_family") == family]
        by_family[str(family)] = summarize_subset(family_rows)
    summary["by_family"] = by_family
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=["trace", "no_trace", "shuffled_trace", "final_patch"], required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--revision", default="488639f1ff808d1d3d0ba301aef8c11461451ec5")
    parser.add_argument("--adapter")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
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
