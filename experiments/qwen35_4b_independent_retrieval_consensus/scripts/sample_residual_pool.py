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
    sampling_prompt,
    split_counts,
    summarize_records,
    write_manifest,
)
from src.jsonl import load_jsonl, write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, load_quant_model, load_tokenizer  # noqa: E402


def parse_temps(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def sample_record(record: dict[str, Any], model: Any, tokenizer: Any, args: argparse.Namespace, record_index: int) -> dict[str, Any]:
    temperatures = parse_temps(args.temperatures)
    counts = split_counts(args.samples_per_task, temperatures)
    prompt = sampling_prompt(record, tokenizer)
    prompt_tokens = estimate_text_tokens(tokenizer, prompt)
    candidates: list[dict[str, Any]] = []
    usage = empty_usage()
    order = 0
    for temp_index, (temperature, count) in enumerate(zip(temperatures, counts)):
        completions, batch_usage = sample_prompt_with_usage(
            model,
            tokenizer,
            prompt,
            count=count,
            temperature=temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.generation_batch_size,
            seed=args.seed + record_index * 100003 + temp_index * 1000,
        )
        usage = add_usage(usage, batch_usage)
        for attempt, completion in enumerate(completions):
            candidates.append(
                candidate_from_completion(
                    completion,
                    record,
                    source=f"{args.arm_name}_t{temperature:g}_{attempt}",
                    order=order,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=estimate_text_tokens(tokenizer, completion),
                )
            )
            order += 1
    out = {key: value for key, value in record.items() if key != "prompt"}
    out["prompt"] = prompt
    out["round_name"] = args.round_name
    out["arm_name"] = args.arm_name
    out["samples_per_task"] = args.samples_per_task
    out["temperatures"] = temperatures
    out["top_p"] = args.top_p
    out["token_usage"] = usage
    out["candidates"] = dedupe_candidates(candidates)
    recompute_record_metrics(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--round-name", type=str, default="direct_sample_more")
    parser.add_argument("--arm-name", type=str, default="direct_sample_more_k12")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=12)
    parser.add_argument("--temperatures", type=str, default="0.7,1.0,1.2")
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026062627)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plan_rows = load_jsonl(args.plan)
    records = [row["record"] for row in plan_rows]
    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = load_quant_model(args.model_path, for_training=False)
    model.eval()
    rows = [sample_record(record, model, tokenizer, args, idx) for idx, record in enumerate(tqdm(records, desc=args.arm_name))]
    write_jsonl(args.out, rows)
    usage = empty_usage()
    for row in rows:
        usage = add_usage(usage, row.get("token_usage", {}))
    manifest = {
        "experiment": EXPERIMENT,
        "round_name": args.round_name,
        "arm_name": args.arm_name,
        "records": len(rows),
        "samples_per_task": args.samples_per_task,
        "temperatures": parse_temps(args.temperatures),
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "token_usage": usage,
        "summary": summarize_records(rows),
        "path": str(args.out),
    }
    write_manifest(args.out.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
