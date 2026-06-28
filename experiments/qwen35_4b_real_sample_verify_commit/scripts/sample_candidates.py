#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.code_env import (  # noqa: E402
    execute_humaneval,
    execute_public_and_asserts,
    extract_candidate_code,
    extract_doctest_public_tests,
    humaneval_sampling_prompt,
    mbpp_sampling_prompt,
    parse_assert_case,
    parse_entry_from_assert,
    public_signature,
)
from src.jsonl import write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, code_chat_prompt, load_generation_model, load_tokenizer, sample_one_prompt  # noqa: E402


def split_counts(total: int, temperatures: list[float]) -> list[int]:
    base = total // len(temperatures)
    rem = total % len(temperatures)
    return [base + (1 if i < rem else 0) for i in range(len(temperatures))]


def build_repair_prompt(record: dict[str, Any], candidate: dict[str, Any]) -> str:
    public_tests = "\n".join(case["assert_src"] for case in record["public_cases"]) or "(none)"
    return (
        "Return only corrected Python code. Do not use markdown.\n"
        "The candidate below failed at least one public test. Write a complete corrected solution.\n\n"
        f"Task:\n{record['task_text']}\n\n"
        f"Public tests:\n{public_tests}\n\n"
        f"Candidate code:\n{candidate.get('code') or candidate.get('raw_completion', '')}\n"
    )


def candidate_from_completion(
    raw_completion: str,
    record: dict[str, Any],
    source: str,
    order: int,
    continuation_prompt: str | None = None,
) -> dict[str, Any]:
    code, parse_reason = extract_candidate_code(raw_completion, record["entry_point"], continuation_prompt=continuation_prompt)
    candidate: dict[str, Any] = {
        "candidate_id": f"cand_{order:03d}",
        "order": order,
        "source": source,
        "raw_completion": raw_completion,
        "code": code or "",
        "parse_status": parse_reason,
    }
    if code is None:
        candidate.update(
            {
                "safe": False,
                "safety_reason": "parse_failed",
                "visible_all_pass": False,
                "full_pass": False,
                "public_passed": [],
                "public_outputs": [],
                "public_signature": "parse_failed",
            }
        )
        return candidate
    if record["dataset"] == "mbpp":
        result = execute_public_and_asserts(
            code,
            record["public_cases"],
            record["hidden_asserts"],
            setup_code=record.get("setup_code", ""),
            timeout_s=record["timeout_s"],
        )
    else:
        result = execute_humaneval(
            code,
            record["public_cases"],
            record["entry_point"],
            record["official_test"],
            timeout_s=record["timeout_s"],
        )
    candidate.update(result)
    candidate["public_signature"] = public_signature(candidate)
    return candidate


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        key = candidate.get("code") or candidate.get("raw_completion", "")
        if key in seen:
            continue
        seen.add(key)
        candidate = dict(candidate)
        candidate["candidate_id"] = f"cand_{len(rows):03d}"
        candidate["order"] = len(rows)
        rows.append(candidate)
    return rows


def mbpp_record(raw: dict[str, Any], split: str, visible_tests: int, timeout_s: float) -> dict[str, Any] | None:
    tests = list(raw.get("test_list") or [])
    if len(tests) < 1:
        return None
    entry = parse_entry_from_assert(tests[0])
    if entry is None:
        return None
    public_asserts = tests[: min(visible_tests, len(tests))]
    public_cases = [case for test in public_asserts if (case := parse_assert_case(test)) is not None]
    hidden_asserts = tests[min(visible_tests, len(tests)) :] + list(raw.get("challenge_test_list") or [])
    return {
        "record_id": f"mbpp_{split}_{raw['task_id']}",
        "dataset": "mbpp",
        "split": split,
        "task_id": raw["task_id"],
        "task_text": raw["text"],
        "entry_point": entry,
        "public_cases": public_cases,
        "hidden_asserts": hidden_asserts,
        "setup_code": raw.get("test_setup_code") or "",
        "timeout_s": timeout_s,
    }


def humaneval_record(raw: dict[str, Any], split: str, public_limit: int, timeout_s: float) -> dict[str, Any]:
    public_cases = extract_doctest_public_tests(raw["prompt"], raw["entry_point"], public_limit)
    return {
        "record_id": f"humaneval_{split}_{raw['task_id'].replace('/', '_')}",
        "dataset": "humaneval",
        "split": split,
        "task_id": raw["task_id"],
        "task_text": raw["prompt"],
        "entry_point": raw["entry_point"],
        "public_cases": public_cases,
        "official_test": raw["test"],
        "timeout_s": timeout_s,
        "continuation_prompt": raw["prompt"],
    }


def sample_for_record(record: dict[str, Any], model: Any, tokenizer: Any, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    if record["dataset"] == "mbpp":
        prompt = code_chat_prompt(tokenizer, mbpp_sampling_prompt(record, record["entry_point"], [case["assert_src"] for case in record["public_cases"]]))
        continuation_prompt = None
    else:
        prompt = code_chat_prompt(tokenizer, humaneval_sampling_prompt(record, record["public_cases"]))
        continuation_prompt = record["continuation_prompt"]

    temperatures = [float(item) for item in args.temperatures.split(",")]
    counts = split_counts(args.samples_per_task, temperatures)
    candidates: list[dict[str, Any]] = []
    order = 0
    for temp, count in zip(temperatures, counts):
        completions = sample_one_prompt(
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
        for i, completion in enumerate(completions):
            candidates.append(
                candidate_from_completion(
                    completion,
                    record,
                    source=f"sample_t{temp:g}_{i}",
                    order=order,
                    continuation_prompt=continuation_prompt,
                )
            )
            order += 1

    repair_pool = [candidate for candidate in candidates if candidate.get("parse_status") == "parsed" and not candidate.get("visible_all_pass")]
    repair_pool = repair_pool[: args.repair_per_task]
    for i, failed in enumerate(repair_pool):
        repair_prompt = code_chat_prompt(tokenizer, build_repair_prompt(record, failed))
        completion = sample_one_prompt(
            model,
            tokenizer,
            repair_prompt,
            count=1,
            temperature=args.repair_temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            batch_size=1,
            seed=seed + 90000 + i,
        )[0]
        candidates.append(
            candidate_from_completion(
                completion,
                record,
                source=f"repair_of_{failed['candidate_id']}",
                order=order,
                continuation_prompt=continuation_prompt,
            )
        )
        order += 1

    candidates = dedupe_candidates(candidates)
    record = {key: value for key, value in record.items() if key not in {"timeout_s", "official_test", "continuation_prompt"}}
    record["candidates"] = candidates
    record["candidate_count"] = len(candidates)
    record["visible_candidate_count"] = sum(1 for candidate in candidates if candidate.get("visible_all_pass"))
    record["coverage"] = any(candidate.get("full_pass") for candidate in candidates)
    record["visible_coverage"] = any(candidate.get("visible_all_pass") and candidate.get("full_pass") for candidate in candidates)
    record["parse_success_count"] = sum(1 for candidate in candidates if candidate.get("parse_status") == "parsed")
    record["safe_count"] = sum(1 for candidate in candidates if candidate.get("safe"))
    return record


def build_split(
    rows: list[dict[str, Any]],
    split: str,
    dataset_name: str,
    target: int,
    builder: Any,
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    seed_offset: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(tqdm(rows[:target], desc=f"sample-{dataset_name}-{split}")):
        base = builder(raw, split)
        if base is None:
            continue
        sampled = sample_for_record(base, model, tokenizer, args, seed=args.seed + seed_offset + index * 1009)
        records.append(sampled)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    return {
        "records": len(records),
        "candidate_count_mean": sum(record["candidate_count"] for record in records) / len(records),
        "parse_success_mean": sum(record["parse_success_count"] for record in records) / len(records),
        "visible_candidates_mean": sum(record["visible_candidate_count"] for record in records) / len(records),
        "coverage": sum(1 for record in records if record["coverage"]) / len(records),
        "visible_coverage": sum(1 for record in records if record["visible_coverage"]) / len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbpp-train", type=int, default=80)
    parser.add_argument("--mbpp-eval", type=int, default=40)
    parser.add_argument("--humaneval-eval", type=int, default=50)
    parser.add_argument("--visible-tests", type=int, default=1)
    parser.add_argument("--humaneval-public-limit", type=int, default=3)
    parser.add_argument("--samples-per-task", type=int, default=16)
    parser.add_argument("--repair-per-task", type=int, default=2)
    parser.add_argument("--temperatures", type=str, default="0.2,0.7,1.0")
    parser.add_argument("--repair-temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = load_generation_model(args.model_path)

    mbpp = load_dataset("google-research-datasets/mbpp")
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    mbpp_train = build_split(
        list(mbpp["train"]),
        "train",
        "mbpp",
        args.mbpp_train,
        lambda raw, split: mbpp_record(raw, split, args.visible_tests, args.timeout_s),
        model,
        tokenizer,
        args,
        0,
    )
    mbpp_eval = build_split(
        list(mbpp["validation"]),
        "eval",
        "mbpp",
        args.mbpp_eval,
        lambda raw, split: mbpp_record(raw, split, args.visible_tests, args.timeout_s),
        model,
        tokenizer,
        args,
        100000,
    )
    humaneval_eval = build_split(
        list(humaneval),
        "eval",
        "humaneval",
        args.humaneval_eval,
        lambda raw, split: humaneval_record(raw, split, args.humaneval_public_limit, args.timeout_s),
        model,
        tokenizer,
        args,
        200000,
    )

    write_jsonl(data_dir / "mbpp_train_records.jsonl", mbpp_train)
    write_jsonl(data_dir / "mbpp_eval_records.jsonl", mbpp_eval)
    write_jsonl(data_dir / "humaneval_eval_records.jsonl", humaneval_eval)

    manifest = {
        "experiment": "qwen35_4b_real_sample_verify_commit",
        "seed": args.seed,
        "samples_per_task": args.samples_per_task,
        "repair_per_task": args.repair_per_task,
        "temperatures": args.temperatures,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "visible_tests": args.visible_tests,
        "humaneval_public_limit": args.humaneval_public_limit,
        "records": {
            "mbpp_train": summarize(mbpp_train),
            "mbpp_eval": summarize(mbpp_eval),
            "humaneval_eval": summarize(humaneval_eval),
        },
        "paths": {
            "mbpp_train": "data/mbpp_train_records.jsonl",
            "mbpp_eval": "data/mbpp_eval_records.jsonl",
            "humaneval_eval": "data/humaneval_eval_records.jsonl",
        },
    }
    (data_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
