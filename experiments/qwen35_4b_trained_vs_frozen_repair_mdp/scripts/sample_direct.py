#!/usr/bin/env python
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

from src.jsonl import write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, load_generation_model, load_tokenizer  # noqa: E402
from src.repair_utils import (  # noqa: E402
    EXPERIMENT,
    add_usage,
    candidate_from_completion,
    dedupe_candidates,
    empty_usage,
    estimate_text_tokens,
    load_task_records,
    recompute_record_metrics,
    sample_prompt_with_usage,
    sampling_prompt,
    split_counts,
    summarize_records,
    write_manifest,
)


def sample_record(record: dict[str, Any], model: Any, tokenizer: Any, args: argparse.Namespace, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt, continuation_prompt = sampling_prompt(record, tokenizer)
    temperatures = [float(item) for item in args.temperatures.split(",")]
    counts = split_counts(args.samples_per_task, temperatures)
    candidates: list[dict[str, Any]] = []
    usage = empty_usage()
    order = 0
    for temp, count in zip(temperatures, counts):
        completions, batch_usage = sample_prompt_with_usage(
            model,
            tokenizer,
            prompt,
            count=count,
            temperature=temp,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.generation_batch_size,
            seed=seed + int(temp * 1000),
        )
        usage = add_usage(usage, batch_usage)
        prompt_each = estimate_text_tokens(tokenizer, prompt)
        for i, completion in enumerate(completions):
            completion_tokens = estimate_text_tokens(tokenizer, completion)
            candidates.append(
                candidate_from_completion(
                    completion,
                    record,
                    source=f"direct_t{temp:g}_{i}",
                    order=order,
                    continuation_prompt=continuation_prompt,
                    prompt_tokens=prompt_each,
                    completion_tokens=completion_tokens,
                )
            )
            order += 1
    record = dict(record)
    record["round_name"] = args.round_name
    record["model_adapter"] = "base"
    record["generation_usage"] = usage
    record["candidates"] = dedupe_candidates(candidates)
    recompute_record_metrics(record)
    return record, usage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["mbpp_train", "mbpp_heldout", "humaneval_transfer"], required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--round-name", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--visible-tests", type=int, default=1)
    parser.add_argument("--humaneval-public-limit", type=int, default=3)
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--temperatures", type=str, default="0.2,0.7,1.0")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = load_generation_model(args.model_path)
    task_records = load_task_records(args.split, args.count, args.offset, args.visible_tests, args.humaneval_public_limit, args.timeout_s)

    records: list[dict[str, Any]] = []
    total_usage = empty_usage()
    for index, record in enumerate(tqdm(task_records, desc=f"direct-{args.split}")):
        sampled, usage = sample_record(record, model, tokenizer, args, seed=args.seed + args.offset * 100003 + index * 1009)
        total_usage = add_usage(total_usage, usage)
        records.append(sampled)

    write_jsonl(args.out, records)
    manifest = {
        "experiment": EXPERIMENT,
        "round_name": args.round_name,
        "split": args.split,
        "count": args.count,
        "offset": args.offset,
        "seed": args.seed,
        "samples_per_task": args.samples_per_task,
        "temperatures": args.temperatures,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "token_usage": total_usage,
        "records": summarize_records(records),
        "path": str(args.out),
    }
    write_manifest(args.out.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
