#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl, write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, load_generation_model, load_tokenizer  # noqa: E402
from src.repair_utils import (  # noqa: E402
    EXPERIMENT,
    add_usage,
    candidate_from_completion,
    dedupe_candidates,
    empty_usage,
    estimate_text_tokens,
    recompute_record_metrics,
    sample_prompt_with_usage,
    sampling_prompt,
    summarize_records,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round-name", type=str, default="sample_more_token_matched")
    parser.add_argument("--target-forward-tokens", type=int, required=True)
    parser.add_argument("--max-extra-per-task", type=int, default=12)
    parser.add_argument("--temperatures", type=str, default="0.2,0.7,1.0")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_records = load_jsonl(args.records)
    records = [deepcopy(record) for record in base_records]
    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = load_generation_model(args.model_path)
    temperatures = [float(item) for item in args.temperatures.split(",")]
    prompts: dict[str, tuple[str, str | None]] = {record["record_id"]: sampling_prompt(record, tokenizer) for record in records}

    usage = empty_usage()
    extra_counts = {record["record_id"]: 0 for record in records}
    progress = tqdm(total=args.target_forward_tokens, desc="sample-more-token-budget")
    pass_index = 0
    while usage["forward_tokens"] < args.target_forward_tokens:
        made_progress = False
        for index, record in enumerate(records):
            if usage["forward_tokens"] >= args.target_forward_tokens:
                break
            if extra_counts[record["record_id"]] >= args.max_extra_per_task:
                continue
            temp = temperatures[(pass_index + index) % len(temperatures)]
            prompt, continuation_prompt = prompts[record["record_id"]]
            completions, batch_usage = sample_prompt_with_usage(
                model,
                tokenizer,
                prompt,
                count=1,
                temperature=temp,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                batch_size=1,
                seed=args.seed + pass_index * 100000 + index * 997,
            )
            usage = add_usage(usage, batch_usage)
            progress.n = min(usage["forward_tokens"], args.target_forward_tokens)
            progress.refresh()
            completion = completions[0]
            candidate = candidate_from_completion(
                completion,
                record,
                source=f"sample_more_t{temp:g}_{extra_counts[record['record_id']]}",
                order=len(record.get("candidates", [])),
                continuation_prompt=continuation_prompt,
                prompt_tokens=estimate_text_tokens(tokenizer, prompt),
                completion_tokens=estimate_text_tokens(tokenizer, completion),
            )
            record.setdefault("candidates", []).append(candidate)
            extra_counts[record["record_id"]] += 1
            made_progress = True
        pass_index += 1
        if not made_progress:
            break
    progress.close()

    for record in records:
        record["round_name"] = args.round_name
        record["model_adapter"] = "base"
        record["sample_more_extra_count"] = extra_counts[record["record_id"]]
        record["sample_more_usage"] = usage
        record["candidates"] = dedupe_candidates(record.get("candidates", []))
        recompute_record_metrics(record)

    write_jsonl(args.out, records)
    manifest = {
        "experiment": EXPERIMENT,
        "round_name": args.round_name,
        "records_in": str(args.records),
        "target_forward_tokens": args.target_forward_tokens,
        "max_extra_per_task": args.max_extra_per_task,
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
