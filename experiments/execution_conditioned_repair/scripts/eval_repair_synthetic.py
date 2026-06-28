#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repair_experiment.modeling import (  # noqa: E402
    load_jsonl,
    load_model_for_generation,
    load_tokenizer,
    render_generation_prompt,
)
from repair_experiment.patching import apply_patch_to_files, extract_unified_diff, patch_stats  # noqa: E402
from repair_experiment.runner import run_pytest, syntax_valid  # noqa: E402


def evaluate_record(model, tokenizer, record: dict, args, trace_override: str | None = None) -> dict:
    prompt = render_generation_prompt(tokenizer, record, args.prompt_mode, trace_override=trace_override)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            num_return_sequences=args.num_samples,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start
    sample_results = []
    for sample_index, output in enumerate(output_ids):
        completion_ids = output[encoded["input_ids"].shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        patch = extract_unified_diff(completion)
        applied, patched_files, apply_output = apply_patch_to_files(record["current_files"], patch)
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
        stats = patch_stats(patch)
        sample_results.append(
            {
                "sample_index": sample_index,
                "completion": completion,
                "extracted_patch": patch,
                "patch_applied": applied,
                "patch_apply_output": apply_output,
                "syntax_valid": syntax_ok,
                "syntax_error": syntax_error,
                "visible_passed": visible["passed"],
                "hidden_passed": hidden["passed"],
                "visible_output": visible["output"][-4000:],
                "hidden_output": hidden["output"][-4000:],
                **stats,
            }
        )
    first = sample_results[0]
    return {
        "episode_id": record["episode_id"],
        "task_id": record["task_id"],
        "split": record["split"],
        "bug_family": record["metadata"]["bug_family"],
        "failure_class": record["metadata"]["failure_class"],
        "prompt_mode": args.prompt_mode,
        "num_samples": args.num_samples,
        "completion": first["completion"],
        "extracted_patch": first["extracted_patch"],
        "patch_applied": first["patch_applied"],
        "patch_apply_output": first["patch_apply_output"],
        "syntax_valid": first["syntax_valid"],
        "syntax_error": first["syntax_error"],
        "visible_passed": first["visible_passed"],
        "hidden_passed": first["hidden_passed"],
        "visible_output": first["visible_output"],
        "hidden_output": first["hidden_output"],
        "sample_results": sample_results,
        "generation_seconds": elapsed,
        **{key: first[key] for key in ["files_touched", "added_lines", "removed_lines"]},
    }


def summarize(results: list[dict], metadata: dict) -> dict:
    n = len(results)
    if n == 0:
        return {"records": 0, **metadata}
    def samples(row: dict) -> list[dict]:
        return row.get("sample_results") or [row]

    repaired_at1 = sum(row["hidden_passed"] for row in results)
    repaired_atk = sum(any(sample["hidden_passed"] for sample in samples(row)) for row in results)
    applied = sum(any(sample["patch_applied"] for sample in samples(row)) for row in results)
    syntax = sum(any(sample["syntax_valid"] for sample in samples(row)) for row in results)
    visible_pass_hidden_fail = sum(
        any(sample["visible_passed"] and not sample["hidden_passed"] for sample in samples(row))
        and not any(sample["hidden_passed"] for sample in samples(row))
        for row in results
    )
    num_samples = max(row.get("num_samples", 1) for row in results)
    return {
        **metadata,
        "records": n,
        "repair_after_first_failure@1": repaired_at1 / n,
        f"repair_after_first_failure@{num_samples}": repaired_atk / n,
        "patch_apply_rate": applied / n,
        "syntax_valid_rate": syntax / n,
        "visible_pass_hidden_fail_rate": visible_pass_hidden_fail / n,
        "median_files_touched": sorted(row["files_touched"] for row in results)[n // 2],
        "median_edit_size": sorted(row["added_lines"] + row["removed_lines"] for row in results)[n // 2],
        "successes": repaired_atk,
        "failures": n - repaired_atk,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="cdbee75f17c01a7cc42f958dc650907174af0554")
    parser.add_argument("--adapter")
    parser.add_argument("--prompt-mode", default="trace", choices=["trace", "no_trace", "wrong_patch_only", "trace_only", "gold_file_removed"])
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be >= 1")
    if args.num_samples > 1 and args.temperature <= 0:
        raise SystemExit("--temperature must be > 0 when --num-samples > 1")

    records = load_jsonl(args.data)
    if args.max_records:
        records = records[: args.max_records]
    tokenizer = load_tokenizer(args.model_id, args.revision)
    model = load_model_for_generation(args.model_id, args.revision, args.adapter, load_in_4bit=True)

    results = []
    for record in tqdm(records, desc="eval"):
        results.append(evaluate_record(model, tokenizer, record, args))

    metadata = {
        "data": str(args.data),
        "model_id": args.model_id,
        "revision": args.revision,
        "adapter": args.adapter,
        "prompt_mode": args.prompt_mode,
    }
    payload = {"summary": summarize(results, metadata), "records": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
