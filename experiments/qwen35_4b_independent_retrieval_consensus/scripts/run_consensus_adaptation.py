#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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


def adaptation_user_prompt(record: dict[str, Any], retrieved: dict[str, Any]) -> str:
    alg = retrieved["algorithm"]
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

Use the retrieved verified algorithm only if it is relevant. Adapt useful algorithmic structure, but write a correct implementation for the target task.
"""


def make_record(record: dict[str, Any], arm_name: str, candidates: list[dict[str, Any]], usage: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in record.items() if key != "prompt"}
    out["round_name"] = "independent_retrieval_consensus"
    out["arm_name"] = arm_name
    out["samples_per_task"] = len(candidates)
    out["temperatures"] = []
    out["top_p"] = None
    out["token_usage"] = usage
    out["candidates"] = dedupe_candidates(candidates)
    recompute_record_metrics(out)
    return out


def run_arm(plan_rows: list[dict[str, Any]], retrieval_key: str, arm_name: str, model: Any, tokenizer: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for rec_index, item in enumerate(tqdm(plan_rows, desc=arm_name)):
        record = item["record"]
        candidates: list[dict[str, Any]] = []
        usage = empty_usage()
        order = 0
        for rank, retrieved in enumerate(item[retrieval_key][: args.top_k]):
            user_prompt = adaptation_user_prompt(record, retrieved)
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
            alg = retrieved["algorithm"]
            for sample_index, completion in enumerate(completions):
                candidate = candidate_from_completion(
                    completion,
                    record,
                    source=f"{arm_name}_{retrieval_key}_r{rank}_s{sample_index}",
                    order=order,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=estimate_text_tokens(tokenizer, completion),
                )
                candidate.update(
                    {
                        "retrieval_key": retrieval_key,
                        "retrieved_rank": rank,
                        "retrieved_score": retrieved.get("score", 0.0),
                        "retrieved_library_id": alg.get("library_id"),
                        "retrieved_task_id": alg.get("task_id"),
                        "retrieved_entry_point": alg.get("entry_point"),
                        "retrieved_code_chars": len(alg.get("code", "")),
                    }
                )
                candidates.append(candidate)
                order += 1
        rows.append(make_record(record, arm_name, candidates, usage))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--samples-per-retrieval", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--arms", type=str, default="independent,same_neighborhood")
    args = parser.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_rows = load_jsonl(args.plan)
    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = load_quant_model(args.model_path, for_training=False)
    model.eval()

    for arm in [item.strip() for item in args.arms.split(",") if item.strip()]:
        if arm not in {"independent", "same_neighborhood"}:
            raise ValueError(f"unknown arm: {arm}")
        arm_name = f"retrieval_adapt_{arm}_top{args.top_k}"
        rows = run_arm(plan_rows, arm, arm_name, model, tokenizer, args)
        out = args.out_dir / f"{arm_name}_records.jsonl"
        write_jsonl(out, rows)
        usage = empty_usage()
        for row in rows:
            usage = add_usage(usage, row.get("token_usage", {}))
        manifest = {
            "experiment": EXPERIMENT,
            "arm_name": arm_name,
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
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
