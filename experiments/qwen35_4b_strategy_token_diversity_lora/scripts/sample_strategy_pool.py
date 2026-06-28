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
    summarize_records,
    write_manifest,
)
from src.jsonl import load_jsonl, write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, attach_existing_lora, load_quant_model, load_tokenizer  # noqa: E402
from src.strategy_utils import STRATEGY_KEYS, strategy_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--base-records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round-name", type=str, required=True)
    parser.add_argument("--arm-name", type=str, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--samples-per-strategy", type=int, default=4)
    parser.add_argument("--strategy-keys", type=str, default=",".join(STRATEGY_KEYS))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source_records = load_jsonl(args.records)
    base_records = load_jsonl(args.base_records)
    records = []
    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = load_quant_model(args.model_path, for_training=False)
    if args.adapter_dir:
        model = attach_existing_lora(model, args.adapter_dir)
    model.eval()

    strategy_keys = [item.strip() for item in args.strategy_keys.split(",") if item.strip()]
    usage = empty_usage()
    for record_index, source_record in enumerate(tqdm(source_records, desc=f"strategy-{args.arm_name}")):
        record = deepcopy(source_record)
        record["candidates"] = []
        order = 0
        for key_index, strategy_key in enumerate(strategy_keys):
            prompt = strategy_prompt(record, strategy_key, tokenizer)
            completions, batch_usage = sample_prompt_with_usage(
                model,
                tokenizer,
                prompt,
                count=args.samples_per_strategy,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.generation_batch_size,
                seed=args.seed + record_index * 100003 + key_index * 1000,
            )
            usage = add_usage(usage, batch_usage)
            prompt_tokens = estimate_text_tokens(tokenizer, prompt)
            for attempt, completion in enumerate(completions):
                candidate = candidate_from_completion(
                    completion,
                    record,
                    source=f"strategy_{strategy_key}_{attempt}",
                    order=order,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=estimate_text_tokens(tokenizer, completion),
                )
                record["candidates"].append(candidate)
                order += 1
        record["round_name"] = args.round_name
        record["candidates"] = dedupe_candidates(record["candidates"])
        recompute_record_metrics(record)
        records.append(record)

    write_jsonl(args.out, records)
    manifest = {
        "experiment": EXPERIMENT,
        "round_name": args.round_name,
        "arm_name": args.arm_name,
        "records_in": str(args.records),
        "base_records": str(args.base_records),
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "strategy_keys": strategy_keys,
        "samples_per_strategy": args.samples_per_strategy,
        "temperature": args.temperature,
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
