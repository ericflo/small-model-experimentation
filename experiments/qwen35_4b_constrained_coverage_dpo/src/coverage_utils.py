from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from src.code_env import (
    execute_public_and_asserts,
    extract_candidate_code,
    mbpp_sampling_prompt,
    parse_assert_case,
    parse_entry_from_assert,
    public_signature,
    run_python_script,
    static_safety_check,
)
from src.model_utils import code_chat_prompt


EXPERIMENT = "qwen35_4b_constrained_coverage_dpo"


def split_counts(total: int, temperatures: list[float]) -> list[int]:
    base = total // len(temperatures)
    rem = total % len(temperatures)
    return [base + (1 if index < rem else 0) for index in range(len(temperatures))]


def estimate_text_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


@torch.no_grad()
def sample_prompt_with_usage(
    model: Any,
    tokenizer: Any,
    prompt: str,
    count: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
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
        output = model.generate(
            **batch,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
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


def empty_usage() -> dict[str, int]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "forward_tokens": 0}


def add_usage(a: dict[str, Any], b: dict[str, Any]) -> dict[str, int]:
    return {
        "calls": int(a.get("calls", 0)) + int(b.get("calls", 0)),
        "prompt_tokens": int(a.get("prompt_tokens", 0)) + int(b.get("prompt_tokens", 0)),
        "completion_tokens": int(a.get("completion_tokens", 0)) + int(b.get("completion_tokens", 0)),
        "forward_tokens": int(a.get("forward_tokens", 0)) + int(b.get("forward_tokens", 0)),
    }


def mbpp_record(raw: dict[str, Any], split: str, visible_tests: int, timeout_s: float) -> dict[str, Any] | None:
    tests = list(raw.get("test_list") or [])
    if not tests:
        return None
    entry = parse_entry_from_assert(tests[0])
    if entry is None:
        return None
    public_asserts = tests[: min(visible_tests, len(tests))]
    public_cases = [case for test in public_asserts if (case := parse_assert_case(test)) is not None]
    hidden_asserts = tests[min(visible_tests, len(tests)) :] + list(raw.get("challenge_test_list") or [])
    all_asserts = [case["assert_src"] for case in public_cases] + hidden_asserts
    return {
        "record_id": f"mbpp_{split}_{raw['task_id']}",
        "dataset": "mbpp",
        "split": split,
        "task_id": raw["task_id"],
        "task_text": raw["text"],
        "entry_point": entry,
        "public_cases": public_cases,
        "hidden_asserts": hidden_asserts,
        "all_asserts": all_asserts,
        "setup_code": raw.get("test_setup_code") or "",
        "reference_code": raw.get("code", ""),
        "timeout_s": timeout_s,
    }


def load_mbpp_records(split: str, count: int, offset: int, visible_tests: int, timeout_s: float) -> list[dict[str, Any]]:
    dataset = load_dataset("google-research-datasets/mbpp")
    source_split = "train" if split == "mbpp_train" else "test"
    name = "train" if split == "mbpp_train" else "heldout"
    records = []
    for raw in list(dataset[source_split])[offset : offset + count]:
        record = mbpp_record(raw, name, visible_tests, timeout_s)
        if record is not None:
            records.append(record)
    return records


def sampling_prompt(record: dict[str, Any], tokenizer: Any) -> str:
    prompt = mbpp_sampling_prompt(record, record["entry_point"], [case["assert_src"] for case in record["public_cases"]])
    return code_chat_prompt(tokenizer, prompt)


def evaluate_failure_bits(code: str, record: dict[str, Any]) -> dict[str, Any]:
    assertions = record.get("all_asserts", [])
    safe, reason = static_safety_check(code)
    if not safe:
        return {"all_passed": False, "failure_bits": "U" * len(assertions), "test_errors": [reason for _ in assertions]}
    script = f"""
import collections
import functools
import heapq
import itertools
import math
import re
import string
from typing import *

{record.get("setup_code", "")}

{code}

assertions = {assertions!r}
rows = []
for assertion in assertions:
    try:
        exec(assertion, globals())
        rows.append({{"passed": True, "error": ""}})
    except BaseException as exc:
        rows.append({{"passed": False, "error": type(exc).__name__}})
print(json.dumps(rows))
"""
    result = run_python_script("import json\n" + script, timeout_s=float(record.get("timeout_s", 5.0)))
    if not result["ok"]:
        return {
            "all_passed": False,
            "failure_bits": "U" * len(assertions),
            "test_errors": [result.get("stderr", "runtime_error")[-80:] for _ in assertions],
        }
    try:
        rows = json.loads(result["stdout"].strip().splitlines()[-1])
    except Exception:
        return {"all_passed": False, "failure_bits": "U" * len(assertions), "test_errors": ["bad_json" for _ in assertions]}
    bits = "".join("1" if row.get("passed") else "0" for row in rows)
    errors = [str(row.get("error", "")) for row in rows]
    return {"all_passed": bool(bits) and set(bits) == {"1"}, "failure_bits": bits, "test_errors": errors}


def candidate_from_completion(
    raw_completion: str,
    record: dict[str, Any],
    source: str,
    order: int,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    parent_id: str | None = None,
    repair_round: int = 0,
) -> dict[str, Any]:
    code, parse_reason = extract_candidate_code(raw_completion, record["entry_point"])
    candidate: dict[str, Any] = {
        "candidate_id": f"cand_{order:04d}",
        "order": order,
        "source": source,
        "parent_id": parent_id,
        "repair_round": repair_round,
        "raw_completion": raw_completion,
        "code": code or "",
        "parse_status": parse_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "forward_tokens": prompt_tokens + completion_tokens,
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
                "failure_bits": "parse_failed",
                "functional_signature": "parse_failed",
            }
        )
        return candidate
    result = execute_public_and_asserts(
        code,
        record["public_cases"],
        record.get("hidden_asserts", []),
        setup_code=record.get("setup_code", ""),
        timeout_s=float(record.get("timeout_s", 5.0)),
    )
    candidate.update(result)
    candidate["public_signature"] = public_signature(candidate)
    failure = evaluate_failure_bits(code, record)
    candidate["failure_bits"] = failure["failure_bits"]
    candidate["test_errors"] = failure["test_errors"]
    candidate["functional_signature"] = failure["failure_bits"]
    candidate["behavior_signature"] = behavior_signature(candidate)
    return candidate


def behavior_signature(candidate: dict[str, Any]) -> str:
    if candidate.get("parse_status") != "parsed":
        return "parse_failed"
    status = "V1" if candidate.get("visible_all_pass") else "V0"
    full = "H1" if candidate.get("full_pass") else "H0"
    code_len_bucket = len(candidate.get("code", "")) // 80
    return f"{status}:{full}:{candidate.get('public_signature', '')}:F{candidate.get('functional_signature', '')}:L{code_len_bucket}"


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        key = candidate.get("code") or candidate.get("raw_completion", "")
        if key in seen:
            continue
        seen.add(key)
        item = dict(candidate)
        item["candidate_id"] = f"cand_{len(rows):04d}"
        item["order"] = len(rows)
        rows.append(item)
    return rows


def recompute_record_metrics(record: dict[str, Any]) -> dict[str, Any]:
    candidates = record.get("candidates", [])
    ordered = sorted(candidates, key=lambda c: c.get("order", 0))
    record["candidate_count"] = len(candidates)
    record["visible_candidate_count"] = sum(1 for c in candidates if c.get("visible_all_pass"))
    record["coverage"] = any(c.get("full_pass") for c in candidates)
    record["visible_coverage"] = any(c.get("visible_all_pass") and c.get("full_pass") for c in candidates)
    record["pass1_proxy"] = bool(ordered and ordered[0].get("full_pass"))
    record["parse_success_count"] = sum(1 for c in candidates if c.get("parse_status") == "parsed")
    record["hidden_pass_candidate_count"] = sum(1 for c in candidates if c.get("full_pass"))
    behavior = {c.get("behavior_signature") or behavior_signature(c) for c in candidates}
    functional = {c.get("functional_signature") or c.get("failure_bits", "") for c in candidates}
    record["distinct_behavior_count"] = len(behavior)
    record["distinct_behavior_rate"] = len(behavior) / len(candidates) if candidates else 0.0
    record["distinct_functional_count"] = len(functional)
    record["distinct_functional_rate"] = len(functional) / len(candidates) if candidates else 0.0
    return record


def summarize_records(records: list[dict[str, Any]], base_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not records:
        return {}
    base_by_id = {r["record_id"]: r for r in base_records or []}
    zero_base = [r for r in records if r["record_id"] in base_by_id and not base_by_id[r["record_id"]].get("coverage")]
    zero_to_one = [r for r in zero_base if r.get("coverage")]
    false_repairs = 0
    visible_repair_pass = 0
    for record in records:
        for candidate in record.get("candidates", []):
            if str(candidate.get("source", "")).startswith("repair") and candidate.get("visible_all_pass"):
                visible_repair_pass += 1
                if not candidate.get("full_pass"):
                    false_repairs += 1
    return {
        "records": len(records),
        "candidate_count_mean": sum(r.get("candidate_count", 0) for r in records) / len(records),
        "parse_success_mean": sum(r.get("parse_success_count", 0) for r in records) / len(records),
        "visible_candidates_mean": sum(r.get("visible_candidate_count", 0) for r in records) / len(records),
        "hidden_pass_candidates_mean": sum(r.get("hidden_pass_candidate_count", 0) for r in records) / len(records),
        "coverage": sum(1 for r in records if r.get("coverage")) / len(records),
        "visible_coverage": sum(1 for r in records if r.get("visible_coverage")) / len(records),
        "pass1_proxy": sum(1 for r in records if r.get("pass1_proxy")) / len(records),
        "distinct_behavior_rate_mean": sum(r.get("distinct_behavior_rate", 0.0) for r in records) / len(records),
        "distinct_functional_rate_mean": sum(r.get("distinct_functional_rate", 0.0) for r in records) / len(records),
        "forward_tokens": sum(sum(c.get("forward_tokens", 0) for c in r.get("candidates", [])) for r in records),
        "zero_base_records": len(zero_base),
        "zero_to_one": len(zero_to_one),
        "zero_to_one_rate": len(zero_to_one) / len(zero_base) if zero_base else 0.0,
        "visible_repair_pass_count": visible_repair_pass,
        "false_repair_count": false_repairs,
        "false_repair_rate": false_repairs / visible_repair_pass if visible_repair_pass else 0.0,
    }


def coverage_at_prefix(records: list[dict[str, Any]], prefix: int, source_prefixes: list[str] | None = None) -> dict[str, Any]:
    rows = []
    for record in records:
        candidates = record.get("candidates", [])
        if source_prefixes is not None:
            candidates = [c for c in candidates if any(str(c.get("source", "")).startswith(prefix) for prefix in source_prefixes)]
        candidates = sorted(candidates, key=lambda c: c.get("order", 0))[:prefix]
        tmp = dict(record, candidates=candidates)
        recompute_record_metrics(tmp)
        rows.append(tmp)
    return summarize_records(rows)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def repair_prompt(record: dict[str, Any], candidate: dict[str, Any], prior_summaries: list[str] | None = None) -> str:
    public_tests = "\n".join(case["assert_src"] for case in record.get("public_cases", [])) or "(none)"
    prior_text = "\n".join(prior_summaries or []) or "(none)"
    return (
        "Return only corrected Python code. Do not use markdown, tests, or explanations.\n"
        "Repair the candidate solution using only the task, public tests, and visible execution trace.\n"
        "Keep the requested function name and provide a complete runnable Python solution.\n\n"
        f"Task:\n{record['task_text']}\n\n"
        f"Required entry point: {record['entry_point']}\n\n"
        f"Public tests:\n{public_tests}\n\n"
        f"Visible execution trace:\n{visible_trace(record, candidate)}\n\n"
        f"Prior repair summaries:\n{prior_text}\n\n"
        f"Candidate code:\n{candidate.get('code') or candidate.get('raw_completion', '')}\n"
    )


def visible_trace(record: dict[str, Any], candidate: dict[str, Any]) -> str:
    lines: list[str] = []
    passed = candidate.get("public_passed") or []
    outputs = candidate.get("public_outputs") or []
    for idx, case in enumerate(record.get("public_cases", [])):
        status = "PASS" if idx < len(passed) and passed[idx] else "FAIL"
        observed = outputs[idx] if idx < len(outputs) else ""
        lines.append(f"{status}: {case['assert_src']}")
        if observed:
            lines.append(f"Observed output: {observed}")
    if candidate.get("runtime_error"):
        lines.append("Runtime error:")
        lines.append(str(candidate["runtime_error"])[-1200:])
    if candidate.get("parse_status") != "parsed":
        lines.append("The candidate could not be parsed as a complete Python solution.")
    return "\n".join(lines).strip() or "No public examples are available."


def choose_repair_sources(record: dict[str, Any], max_sources: int, visible_fail_only: bool = True) -> list[dict[str, Any]]:
    candidates = [c for c in record.get("candidates", []) if c.get("parse_status") == "parsed"]
    if visible_fail_only:
        candidates = [c for c in candidates if not c.get("visible_all_pass")]
    return sorted(candidates, key=lambda c: (c.get("visible_all_pass", False), c.get("order", 0)))[:max_sources]


def stable_sample(rows: list[Any], count: int, seed: int) -> list[Any]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:count]
