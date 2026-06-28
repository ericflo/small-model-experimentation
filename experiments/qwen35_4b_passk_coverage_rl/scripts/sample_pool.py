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

from src.jsonl import write_json, write_jsonl
from src.mbpp_env import candidate_from_completion, load_mbpp_records, recompute_metrics, summarize_records
from src.model_utils import DEFAULT_MODEL_PATH, attach_existing_lora, load_generation_model, load_quant_model, load_tokenizer, sample_prompt


def split_counts(total: int, temperatures: list[float]) -> list[int]:
    base = total // len(temperatures)
    counts = [base] * len(temperatures)
    for i in range(total - sum(counts)):
        counts[i] += 1
    return counts


def load_model(args: argparse.Namespace) -> Any:
    if args.adapter_dir:
        model = load_quant_model(args.model_path, for_training=False)
        model = attach_existing_lora(model, args.adapter_dir, is_trainable=False)
        model.eval()
        return model
    return load_generation_model(args.model_path)


def sample_record(record: dict[str, Any], model: Any, tokenizer: Any, args: argparse.Namespace, task_index: int) -> dict[str, Any]:
    temperatures = [float(item) for item in args.temperatures.split(",")]
    counts = split_counts(args.samples_per_task, temperatures)
    candidates = []
    order = 0
    for temp, count in zip(temperatures, counts):
        completions = sample_prompt(
            model,
            tokenizer,
            record["prompt"],
            count=count,
            temperature=temp,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.generation_batch_size,
            seed=args.seed + task_index * 1009 + int(temp * 1000),
        )
        for index, completion in enumerate(completions):
            candidates.append(
                candidate_from_completion(
                    completion,
                    record,
                    source=f"{args.arm_name}_t{temp:g}_{index}",
                    order=order,
                    tokenizer=tokenizer,
                    prompt=record["prompt"],
                )
            )
            order += 1
    out = {key: value for key, value in record.items() if key != "prompt"}
    out["round_name"] = args.round_name
    out["arm_name"] = args.arm_name
    out["samples_per_task"] = args.samples_per_task
    out["temperatures"] = temperatures
    out["candidates"] = candidates
    recompute_metrics(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "validation", "test", "prompt"], default="test")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--visible-tests", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--round-name", type=str, required=True)
    parser.add_argument("--arm-name", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--temperatures", type=str, default="0.2,0.7,1.0")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-dir", type=Path)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    records = load_mbpp_records(args.split, args.count, args.offset, args.visible_tests, args.timeout_s, tokenizer=tokenizer)
    model = load_model(args)

    rows = []
    for idx, record in enumerate(tqdm(records, desc=args.arm_name)):
        rows.append(sample_record(record, model, tokenizer, args, idx))

    summary = {
        "experiment": "qwen35_4b_passk_coverage_rl",
        "round_name": args.round_name,
        "arm_name": args.arm_name,
        "split": args.split,
        "count": args.count,
        "offset": args.offset,
        "samples_per_task": args.samples_per_task,
        "temperatures": args.temperatures,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "records": summarize_records(rows),
    }
    write_jsonl(args.out, rows)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
