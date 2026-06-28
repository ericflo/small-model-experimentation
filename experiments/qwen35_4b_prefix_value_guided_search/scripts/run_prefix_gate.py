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
from src.mbpp_env import (
    candidate_from_completion,
    extract_prefix_code,
    load_mbpp_records,
    make_continue_prompt,
    make_prefix_prompt,
    summarize_candidates,
)
from src.model_utils import DEFAULT_MODEL_PATH, load_generation_model, load_tokenizer, sample_prompt


def dedupe_prefixes(prefixes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for item in prefixes:
        key = item["prefix_code"].strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def lexical_score(prefix_code: str, record: dict[str, Any]) -> float:
    text = (record["task_text"] + " " + " ".join(record["public_tests"])).lower()
    toks = {tok for tok in text.replace("_", " ").split() if len(tok) > 2}
    code = prefix_code.lower().replace("_", " ")
    return sum(1 for tok in toks if tok in code) / max(1, len(toks))


def sample_full(record: dict[str, Any], model: Any, tokenizer: Any, args: argparse.Namespace, task_index: int) -> list[dict[str, Any]]:
    completions = sample_prompt(
        model,
        tokenizer,
        record["prompt"],
        count=args.full_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed + task_index * 1009,
    )
    return [
        candidate_from_completion(comp, record, source=f"full_t{args.temperature:g}_{idx}", order=idx, tokenizer=tokenizer, prompt=record["prompt"])
        for idx, comp in enumerate(completions)
    ]


def sample_prefixes(record: dict[str, Any], model: Any, tokenizer: Any, args: argparse.Namespace, task_index: int) -> list[dict[str, Any]]:
    prompt = make_prefix_prompt(record, tokenizer, args.prefix_lines)
    raw = sample_prompt(
        model,
        tokenizer,
        prompt,
        count=args.prefix_count,
        temperature=args.prefix_temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_prefix_tokens,
        batch_size=args.batch_size,
        seed=args.seed + task_index * 2003 + 17,
    )
    prefixes = []
    for idx, completion in enumerate(raw):
        prefix_code, status = extract_prefix_code(completion, record["entry_point"], args.prefix_lines)
        if prefix_code is None:
            continue
        prefixes.append(
            {
                "prefix_id": f"prefix_{idx:03d}",
                "prefix_code": prefix_code,
                "parse_status": status,
                "raw_completion": completion,
                "lexical_score": lexical_score(prefix_code, record),
                "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
                "completion_tokens": len(tokenizer.encode(completion, add_special_tokens=False)),
            }
        )
    return dedupe_prefixes(prefixes)


def complete_prefix(
    record: dict[str, Any],
    prefix: dict[str, Any],
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    task_index: int,
    prefix_index: int,
) -> list[dict[str, Any]]:
    prompt = make_continue_prompt(record, tokenizer, prefix["prefix_code"])
    completions = sample_prompt(
        model,
        tokenizer,
        prompt,
        count=args.completions_per_prefix,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed + task_index * 3001 + prefix_index * 53,
    )
    out = []
    for idx, comp in enumerate(completions):
        cand = candidate_from_completion(
            comp,
            record,
            source=f"{prefix['prefix_id']}_cont_{idx}",
            order=idx,
            tokenizer=tokenizer,
            prompt=prompt,
        )
        cand["prefix_id"] = prefix["prefix_id"]
        cand["prefix_code"] = prefix["prefix_code"]
        cand["prefix_lexical_score"] = prefix["lexical_score"]
        out.append(cand)
    return out


def run_record(record: dict[str, Any], model: Any, tokenizer: Any, args: argparse.Namespace, task_index: int) -> dict[str, Any]:
    full_candidates = sample_full(record, model, tokenizer, args, task_index)
    prefixes = sample_prefixes(record, model, tokenizer, args, task_index)

    prefix_groups = []
    prefix_union_candidates = []
    for pidx, prefix in enumerate(prefixes):
        candidates = complete_prefix(record, prefix, model, tokenizer, args, task_index, pidx)
        prefix_summary = summarize_candidates(candidates)
        prefix_groups.append({**prefix, "candidates": candidates, "summary": prefix_summary})
        prefix_union_candidates.extend(candidates)

    full_summary = summarize_candidates(full_candidates)
    union_summary = summarize_candidates(prefix_union_candidates)
    oracle_group = None
    if prefix_groups:
        oracle_group = max(prefix_groups, key=lambda group: (group["summary"]["coverage"], group["summary"]["hidden_pass_candidates"], group["summary"]["visible_candidates"]))
    lexical_group = None
    if prefix_groups:
        lexical_group = max(prefix_groups, key=lambda group: group["lexical_score"])
    random_group = random.choice(prefix_groups) if prefix_groups else None

    return {
        "task_id": record["task_id"],
        "record_id": record["record_id"],
        "entry_point": record["entry_point"],
        "task_text": record["task_text"],
        "full_samples": {"summary": full_summary, "candidates": full_candidates},
        "prefix_count_requested": args.prefix_count,
        "prefix_count_valid": len(prefixes),
        "prefix_groups": prefix_groups,
        "prefix_union": {"summary": union_summary},
        "prefix_oracle_selected": {"summary": oracle_group["summary"], "prefix_id": oracle_group["prefix_id"], "prefix_code": oracle_group["prefix_code"]} if oracle_group else {"summary": summarize_candidates([])},
        "prefix_lexical_selected": {"summary": lexical_group["summary"], "prefix_id": lexical_group["prefix_id"], "prefix_code": lexical_group["prefix_code"]} if lexical_group else {"summary": summarize_candidates([])},
        "prefix_random_selected": {"summary": random_group["summary"], "prefix_id": random_group["prefix_id"], "prefix_code": random_group["prefix_code"]} if random_group else {"summary": summarize_candidates([])},
    }


def summarize_arm(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {}
    summaries = [row[key]["summary"] for row in rows]
    return {
        "task_count": n,
        "coverage": sum(1 for item in summaries if item.get("coverage")) / n,
        "visible_coverage": sum(1 for item in summaries if item.get("visible_coverage")) / n,
        "pass1_proxy": sum(1 for item in summaries if item.get("pass1_proxy")) / n,
        "mean_distinct_functional_rate": sum(item.get("distinct_functional_rate", 0.0) for item in summaries) / n,
        "mean_distinct_program_rate": sum(item.get("distinct_program_rate", 0.0) for item in summaries) / n,
        "forward_tokens": sum(item.get("forward_tokens", 0) for item in summaries),
        "hidden_pass_candidates": sum(item.get("hidden_pass_candidates", 0) for item in summaries),
        "visible_candidates": sum(item.get("visible_candidates", 0) for item in summaries),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "validation", "test", "prompt"], default="test")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--visible-tests", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--full-samples", type=int, default=8)
    parser.add_argument("--prefix-count", type=int, default=4)
    parser.add_argument("--completions-per-prefix", type=int, default=2)
    parser.add_argument("--prefix-lines", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--prefix-temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--max-prefix-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    random.seed(args.seed)
    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    records = load_mbpp_records(args.split, args.count, args.offset, args.visible_tests, args.timeout_s, tokenizer=tokenizer)
    model = load_generation_model(args.model_path)
    rows = []
    for idx, record in enumerate(tqdm(records, desc="prefix-gate")):
        rows.append(run_record(record, model, tokenizer, args, idx))

    summary = {
        "experiment": "qwen35_4b_prefix_value_guided_search",
        "split": args.split,
        "count": args.count,
        "offset": args.offset,
        "full_samples": args.full_samples,
        "prefix_count": args.prefix_count,
        "completions_per_prefix": args.completions_per_prefix,
        "completion_budget_matched": args.full_samples == args.prefix_count * args.completions_per_prefix,
        "prefix_lines": args.prefix_lines,
        "temperature": args.temperature,
        "seed": args.seed,
        "arms": {
            "full_sample_base": summarize_arm(rows, key="full_samples"),
            "prefix_union": summarize_arm(rows, key="prefix_union"),
            "prefix_oracle_selected": summarize_arm(rows, key="prefix_oracle_selected"),
            "prefix_lexical_selected": summarize_arm(rows, key="prefix_lexical_selected"),
            "prefix_random_selected": summarize_arm(rows, key="prefix_random_selected"),
        },
        "mean_valid_prefixes": sum(row["prefix_count_valid"] for row in rows) / max(1, len(rows)),
    }
    write_jsonl(args.out, rows)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
