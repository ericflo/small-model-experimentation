#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coverage_utils import (  # noqa: E402
    EXPERIMENT,
    add_usage,
    candidate_from_completion,
    dedupe_candidates,
    empty_usage,
    estimate_text_tokens,
    recompute_record_metrics,
    sample_prompt_with_usage,
    summarize_records,
    write_manifest,
)
from src.jsonl import load_jsonl, write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, code_chat_prompt, load_quant_model, load_tokenizer  # noqa: E402


def public_tests(record: dict[str, Any]) -> list[str]:
    return [case["assert_src"] for case in record.get("public_cases", [])]


def adaptation_user_prompt(record: dict[str, Any], retrieved: dict[str, Any], mode: str) -> str:
    alg = retrieved["algorithm"]
    if mode == "copy_rename":
        raise ValueError("copy_rename does not use a prompt")
    prefix = "Use the retrieved verified algorithm only if it is relevant." if mode == "adapt" else "The retrieved code may be unrelated; adapt only useful ideas."
    return f"""Return only Python code. Do not use markdown.

Target task:
{record['task_text']}

Define a function named `{record['entry_point']}` and any helpers needed.

Public tests:
{chr(10).join(public_tests(record))}

Retrieved verified algorithm:
Task: {alg['task_text']}
Function: {alg['entry_point']}
Code:
{alg['code']}

{prefix}
Write a correct implementation for the target task.
"""


def copy_rename_code(record: dict[str, Any], retrieved: dict[str, Any]) -> str:
    code = retrieved["algorithm"]["code"]
    target = record["entry_point"]
    source = retrieved["algorithm"]["entry_point"]
    return re.sub(rf"def\s+{re.escape(source)}\s*\(", f"def {target}(", code, count=1)


def make_record(record: dict[str, Any], arm_name: str, candidates: list[dict[str, Any]], usage: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in record.items() if key != "prompt"}
    out["round_name"] = "retrieval_adaptation"
    out["arm_name"] = arm_name
    out["samples_per_task"] = len(candidates)
    out["temperatures"] = []
    out["top_p"] = None
    out["token_usage"] = usage
    out["candidates"] = dedupe_candidates(candidates)
    recompute_record_metrics(out)
    return out


def run_generated_arm(
    plan_rows: list[dict[str, Any]],
    arm_name: str,
    retrieval_key: str,
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    for rec_index, item in enumerate(tqdm(plan_rows, desc=arm_name)):
        record = item["record"]
        candidates: list[dict[str, Any]] = []
        usage = empty_usage()
        order = 0
        for rank, retrieved in enumerate(item[retrieval_key][: args.top_k]):
            user_prompt = adaptation_user_prompt(record, retrieved, mode="adapt")
            prompt = code_chat_prompt(tokenizer, user_prompt)
            prompt_tokens = estimate_text_tokens(tokenizer, prompt)
            completions, batch_usage = sample_prompt_with_usage(
                model,
                tokenizer,
                prompt,
                count=args.samples_per_retrieval,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.generation_batch_size,
                seed=args.seed + rec_index * 100003 + rank * 1009,
            )
            usage = add_usage(usage, batch_usage)
            for sample_index, completion in enumerate(completions):
                candidates.append(
                    candidate_from_completion(
                        completion,
                        record,
                        source=f"{arm_name}_{retrieval_key}_r{rank}_s{sample_index}",
                        order=order,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=estimate_text_tokens(tokenizer, completion),
                    )
                )
                order += 1
        rows.append(make_record(record, arm_name, candidates, usage))
    return rows


def run_copy_arm(plan_rows: list[dict[str, Any]], arm_name: str, retrieval_key: str, top_k: int) -> list[dict[str, Any]]:
    rows = []
    for item in tqdm(plan_rows, desc=arm_name):
        record = item["record"]
        candidates = []
        for rank, retrieved in enumerate(item[retrieval_key][:top_k]):
            code = copy_rename_code(record, retrieved)
            candidates.append(
                candidate_from_completion(
                    code,
                    record,
                    source=f"{arm_name}_{retrieval_key}_r{rank}",
                    order=rank,
                    prompt_tokens=0,
                    completion_tokens=0,
                )
            )
        rows.append(make_record(record, arm_name, candidates, empty_usage()))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--samples-per-retrieval", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--arms", type=str, default="copy_semantic,adapt_semantic,adapt_random,adapt_shuffled")
    args = parser.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_rows = load_jsonl(args.plan)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = None
    all_outputs: list[Path] = []
    for arm in arms:
        if arm == "copy_semantic":
            rows = run_copy_arm(plan_rows, "retrieval_copy_rename_top3", "semantic", args.top_k)
        else:
            if model is None:
                model = load_quant_model(args.model_path, for_training=False)
                model.eval()
            if arm == "adapt_semantic":
                rows = run_generated_arm(plan_rows, "retrieval_adapt_semantic_top3", "semantic", model, tokenizer, args)
            elif arm == "adapt_random":
                rows = run_generated_arm(plan_rows, "retrieval_adapt_random_top3", "random", model, tokenizer, args)
            elif arm == "adapt_shuffled":
                rows = run_generated_arm(plan_rows, "retrieval_adapt_shuffled_top3", "shuffled", model, tokenizer, args)
            else:
                raise ValueError(f"unknown arm: {arm}")
        out = args.out_dir / f"{rows[0]['arm_name']}_records.jsonl"
        write_jsonl(out, rows)
        usage = empty_usage()
        for row in rows:
            usage = add_usage(usage, row.get("token_usage", {}))
        manifest = {
            "experiment": EXPERIMENT,
            "arm_name": rows[0]["arm_name"],
            "plan": str(args.plan),
            "top_k": args.top_k,
            "samples_per_retrieval": args.samples_per_retrieval,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "token_usage": usage,
            "records": summarize_records(rows),
            "path": str(out),
        }
        write_manifest(out.with_suffix(".manifest.json"), manifest)
        all_outputs.append(out)
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
