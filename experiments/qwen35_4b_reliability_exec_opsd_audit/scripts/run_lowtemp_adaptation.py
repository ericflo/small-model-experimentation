#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
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
    summarize_records,
    write_manifest,
)
from src.jsonl import load_jsonl, write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, code_chat_prompt, load_quant_model, load_tokenizer  # noqa: E402


def public_tests(record: dict[str, Any]) -> list[str]:
    return [case["assert_src"] for case in record.get("public_cases", [])]


def temp_tag(value: float) -> str:
    return str(value).replace(".", "p")


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

Use the retrieved verified algorithm only if it is relevant.
Write a correct implementation for the target task.
"""


@torch.no_grad()
def generate_with_usage(
    model: Any,
    tokenizer: Any,
    prompt: str,
    count: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
) -> tuple[list[str], dict[str, int]]:
    rows: list[str] = []
    prompt_tokens = estimate_text_tokens(tokenizer, prompt)
    completion_tokens = 0
    device = model.device
    offset = 0
    while len(rows) < count:
        current = min(batch_size, count - len(rows))
        batch = tokenizer([prompt for _ in range(current)], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        torch.manual_seed(seed + offset)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + offset)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature <= 0.0:
            kwargs["do_sample"] = False
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        output = model.generate(**batch, **kwargs)
        prompt_len = batch["input_ids"].shape[1]
        completions = tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
        for completion in completions:
            completion_tokens += estimate_text_tokens(tokenizer, completion)
            rows.append(completion)
        offset += current
    return rows, {
        "calls": count,
        "prompt_tokens": prompt_tokens * count,
        "completion_tokens": completion_tokens,
        "forward_tokens": prompt_tokens * count + completion_tokens,
    }


def make_record(record: dict[str, Any], arm_name: str, candidates: list[dict[str, Any]], usage: dict[str, Any], temperature: float) -> dict[str, Any]:
    out = {key: value for key, value in record.items() if key != "prompt"}
    out["round_name"] = "lowtemp_retrieval_adaptation"
    out["arm_name"] = arm_name
    out["samples_per_task"] = len(candidates)
    out["temperatures"] = [temperature]
    out["top_p"] = None
    out["token_usage"] = usage
    out["candidates"] = dedupe_candidates(candidates)
    recompute_record_metrics(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--samples-per-retrieval", type=int, default=1)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    plan_rows = load_jsonl(args.plan)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = load_quant_model(args.model_path, for_training=False)
    model.eval()

    arm_name = f"retrieval_adapt_semantic_t{temp_tag(args.temperature)}_top3"
    rows: list[dict[str, Any]] = []
    for rec_index, item in enumerate(tqdm(plan_rows, desc=arm_name)):
        record = item["record"]
        candidates: list[dict[str, Any]] = []
        usage = empty_usage()
        order = 0
        for rank, retrieved in enumerate(item["semantic"][: args.top_k]):
            prompt = code_chat_prompt(tokenizer, adaptation_user_prompt(record, retrieved))
            prompt_tokens = estimate_text_tokens(tokenizer, prompt)
            completions, batch_usage = generate_with_usage(
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
                        source=f"{arm_name}_semantic_r{rank}_s{sample_index}",
                        order=order,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=estimate_text_tokens(tokenizer, completion),
                    )
                )
                order += 1
        rows.append(make_record(record, arm_name, candidates, usage, args.temperature))

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
