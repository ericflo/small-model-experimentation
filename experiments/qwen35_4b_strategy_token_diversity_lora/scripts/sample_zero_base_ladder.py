#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diversity_utils import (  # noqa: E402
    EXPERIMENT,
    add_usage,
    candidate_from_completion,
    dedupe_candidates,
    empty_usage,
    estimate_text_tokens,
    recompute_record_metrics,
    sample_prompt_with_usage,
    sampling_prompt,
    split_counts,
    summarize_records,
    write_manifest,
)
from src.jsonl import load_jsonl, write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, load_generation_model, load_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round-name", type=str, required=True)
    parser.add_argument("--arm-name", type=str, required=True)
    parser.add_argument("--extra-per-zero-task", type=int, default=28)
    parser.add_argument("--temperatures", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_records = load_jsonl(args.records)
    records = [deepcopy(record) for record in base_records]
    zero_indices = [idx for idx, record in enumerate(base_records) if not record.get("coverage")]
    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = load_generation_model(args.model_path)
    temperatures = [float(item) for item in args.temperatures.split(",")]
    counts = split_counts(args.extra_per_zero_task, temperatures)
    usage = empty_usage()

    for local_index, record_index in enumerate(tqdm(zero_indices, desc=f"zero-ladder-{args.arm_name}")):
        record = records[record_index]
        prompt = sampling_prompt(record, tokenizer)
        order = len(record.get("candidates", []))
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
                seed=args.seed + local_index * 100003 + int(temp * 1000),
            )
            usage = add_usage(usage, batch_usage)
            prompt_tokens = estimate_text_tokens(tokenizer, prompt)
            for index, completion in enumerate(completions):
                record.setdefault("candidates", []).append(
                    candidate_from_completion(
                        completion,
                        record,
                        source=f"{args.arm_name}_t{temp:g}_{index}",
                        order=order,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=estimate_text_tokens(tokenizer, completion),
                    )
                )
                order += 1
        record["extra_sampling_arm"] = args.arm_name
        record["extra_sampling_usage"] = usage
        record["candidates"] = dedupe_candidates(record.get("candidates", []))
        recompute_record_metrics(record)

    for record in records:
        record["round_name"] = args.round_name
        recompute_record_metrics(record)

    write_jsonl(args.out, records)
    manifest = {
        "experiment": EXPERIMENT,
        "round_name": args.round_name,
        "arm_name": args.arm_name,
        "records_in": str(args.records),
        "seed": args.seed,
        "zero_base_records": len(zero_indices),
        "extra_per_zero_task": args.extra_per_zero_task,
        "temperatures": args.temperatures,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "token_usage": usage,
        "records": summarize_records(records, base_records=base_records),
        "path": str(args.out),
    }
    write_manifest(args.out.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
