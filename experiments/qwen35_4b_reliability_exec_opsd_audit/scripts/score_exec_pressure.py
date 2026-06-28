#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coverage_utils import EXPERIMENT, estimate_text_tokens  # noqa: E402
from src.jsonl import load_jsonl, write_json, write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, code_chat_prompt, load_quant_model, load_tokenizer  # noqa: E402


CONTEXTS = ["student_no_hint", "exec_observation", "exec_input_only", "shuffled_exec", "full_reference"]
TEACHER_CONTEXTS = ["exec_observation", "exec_input_only", "shuffled_exec", "full_reference"]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def softmax_logprob_rows(
    model: Any,
    tokenizer: Any,
    context: str,
    completion: str,
    max_length: int,
) -> dict[str, Any]:
    text = context + completion
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"][0]
    if input_ids.shape[0] > max_length:
        return {
            "ok": False,
            "reason": f"too_long:{int(input_ids.shape[0])}",
            "token_count": 0,
            "logprob_sum": 0.0,
            "logprob_mean": 0.0,
            "tokens": [],
        }
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.no_grad():
        out = model(**encoded)
        logprobs = torch.log_softmax(out.logits[0, :-1, :], dim=-1)
    start_char = len(context)
    end_char = len(text)
    token_rows: list[dict[str, Any]] = []
    token_ids = input_ids.tolist()
    for idx, (lo, hi) in enumerate(offsets):
        if hi <= start_char or lo >= end_char:
            continue
        if idx == 0:
            continue
        token_id = token_ids[idx]
        lp = float(logprobs[idx - 1, token_id].detach().cpu())
        token_rows.append(
            {
                "index": idx,
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]),
                "raw_token": tokenizer.convert_ids_to_tokens([token_id])[0],
                "rel_start": max(0, lo - start_char),
                "rel_end": min(len(completion), hi - start_char),
                "logprob": lp,
            }
        )
    lps = [row["logprob"] for row in token_rows]
    return {
        "ok": True,
        "reason": "ok",
        "token_count": len(token_rows),
        "logprob_sum": sum(lps),
        "logprob_mean": mean(lps),
        "tokens": token_rows,
    }


def branch_score(
    model: Any,
    tokenizer: Any,
    prompt: str,
    prefix: str,
    branch: str,
    max_length: int,
) -> dict[str, Any]:
    rows = softmax_logprob_rows(model, tokenizer, prompt + prefix, branch, max_length=max_length)
    return {key: value for key, value in rows.items() if key != "tokens"}


def code_score(
    model: Any,
    tokenizer: Any,
    prompt: str,
    code: str,
    max_length: int,
) -> dict[str, Any]:
    return softmax_logprob_rows(model, tokenizer, prompt, code, max_length=max_length)


def bucket_for_offset(token_map: dict[str, Any], start: int, end: int) -> str:
    midpoint = (start + end) / 2.0
    for token in token_map.get("tokens", []):
        if token["start"] <= midpoint < token["end"]:
            return token.get("bucket", "unknown")
    return "whitespace_or_unknown"


def summarize_forks(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["context"], row["stratum"])].append(row)
    out: dict[str, Any] = {}
    for (context, stratum), items in sorted(grouped.items()):
        prefs = [item["preference_mean"] for item in items if item["ok"]]
        sum_prefs = [item["preference_sum"] for item in items if item["ok"]]
        student_prefs = [item["student_preference_mean"] for item in items if item["ok"]]
        deltas = [item["preference_mean"] - item["student_preference_mean"] for item in items if item["ok"]]
        c_press = [item["correct_positive_pressure_mean"] for item in items if item["ok"] and "correct_positive_pressure_mean" in item]
        w_press = [item["wrong_positive_pressure_mean"] for item in items if item["ok"] and "wrong_positive_pressure_mean" in item]
        key = f"{context}/{stratum}"
        out[key] = {
            "context": context,
            "stratum": stratum,
            "n": len(prefs),
            "mean_preference": mean(prefs),
            "median_preference": median(prefs),
            "mean_sum_preference": mean(sum_prefs),
            "frac_prefers_correct": sum(1 for value in prefs if value > 0.0) / len(prefs) if prefs else 0.0,
            "mean_student_preference": mean(student_prefs),
            "mean_delta_over_student": mean(deltas),
            "frac_delta_over_student_positive": sum(1 for value in deltas if value > 0.0) / len(deltas) if deltas else 0.0,
            "mean_correct_positive_pressure": mean(c_press),
            "mean_wrong_positive_pressure": mean(w_press),
        }
    return out


def summarize_tokens(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["context"], row["candidate_kind"], row["bucket"])].append(row)
    out: dict[str, Any] = {}
    for (context, kind, bucket), items in sorted(grouped.items()):
        gaps = [item["gap"] for item in items]
        positives = [max(0.0, item["gap"]) for item in items]
        key = f"{context}/{kind}/{bucket}"
        out[key] = {
            "context": context,
            "candidate_kind": kind,
            "bucket": bucket,
            "n": len(items),
            "mean_gap": mean(gaps),
            "mean_positive_gap": mean(positives),
            "positive_rate": sum(1 for value in gaps if value > 0.0) / len(gaps) if gaps else 0.0,
        }
    return out


def gate(summary: dict[str, Any]) -> dict[str, Any]:
    weak = summary.get("exec_observation/task_specific", {})
    input_only = summary.get("exec_input_only/task_specific", {})
    shuffled = summary.get("shuffled_exec/task_specific", {})
    full = summary.get("full_reference/task_specific", {})
    weak_mean = float(weak.get("mean_preference", 0.0))
    shuffled_mean = float(shuffled.get("mean_preference", 0.0))
    full_mean = float(full.get("mean_preference", 0.0))
    weak_frac = float(weak.get("frac_prefers_correct", 0.0))
    weak_delta = float(weak.get("mean_delta_over_student", 0.0))
    shuffled_delta = float(shuffled.get("mean_delta_over_student", 0.0))
    passed = bool(
        weak.get("n", 0) >= 5
        and weak_mean > 0.0
        and weak_frac >= 0.55
        and weak_mean > shuffled_mean + 0.10
        and weak_delta > 0.05
        and weak_delta > shuffled_delta + 0.05
    )
    return {
        "passed": passed,
        "reason": (
            "execution observation adds task-specific correct-branch preference beyond student and shuffled control"
            if passed
            else "execution observation does not add task-specific correct-branch preference beyond student and shuffled control"
        ),
        "exec_task_specific_mean": weak_mean,
        "exec_task_specific_frac_prefers_correct": weak_frac,
        "exec_task_specific_delta_over_student": weak_delta,
        "input_only_task_specific_mean": float(input_only.get("mean_preference", 0.0)),
        "input_only_task_specific_delta_over_student": float(input_only.get("mean_delta_over_student", 0.0)),
        "shuffled_exec_task_specific_mean": shuffled_mean,
        "shuffled_exec_task_specific_delta_over_student": shuffled_delta,
        "full_reference_task_specific_mean": full_mean,
        "full_reference_task_specific_delta_over_student": float(full.get("mean_delta_over_student", 0.0)),
        "task_specific_forks": int(weak.get("n", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--fork-out", type=Path, required=True)
    parser.add_argument("--token-out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-pairs", type=int, default=0)
    args = parser.parse_args()

    pairs = load_jsonl(args.pairs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = load_quant_model(args.model_path, for_training=False)
    model.eval()

    fork_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    usage = {"scored_sequences": 0, "prompt_tokens_estimate": 0, "completion_tokens_estimate": 0, "forward_tokens_estimate": 0}
    prompt_cache: dict[tuple[str, str], str] = {}

    def chat_prompt(pair: dict[str, Any], context: str) -> str:
        key = (pair["pair_id"], context)
        if key not in prompt_cache:
            prompt_cache[key] = code_chat_prompt(tokenizer, pair["prompts"][context])
        return prompt_cache[key]

    for pair in pairs:
        prompts = {context: chat_prompt(pair, context) for context in CONTEXTS}
        student_prompt = prompts["student_no_hint"]
        for fork in pair["forks"]:
            student_correct = branch_score(
                model, tokenizer, student_prompt, fork["prefix_text"], fork["correct_branch"], args.max_length
            )
            student_wrong = branch_score(
                model, tokenizer, student_prompt, fork["prefix_text"], fork["wrong_branch"], args.max_length
            )
            usage["scored_sequences"] += 2
            usage["prompt_tokens_estimate"] += estimate_text_tokens(tokenizer, student_prompt + fork["prefix_text"]) * 2
            usage["completion_tokens_estimate"] += estimate_text_tokens(tokenizer, fork["correct_branch"]) + estimate_text_tokens(tokenizer, fork["wrong_branch"])
            for context in TEACHER_CONTEXTS:
                prompt = prompts[context]
                correct = branch_score(model, tokenizer, prompt, fork["prefix_text"], fork["correct_branch"], args.max_length)
                wrong = branch_score(model, tokenizer, prompt, fork["prefix_text"], fork["wrong_branch"], args.max_length)
                usage["scored_sequences"] += 2
                usage["prompt_tokens_estimate"] += estimate_text_tokens(tokenizer, prompt + fork["prefix_text"]) * 2
                usage["completion_tokens_estimate"] += estimate_text_tokens(tokenizer, fork["correct_branch"]) + estimate_text_tokens(tokenizer, fork["wrong_branch"])
                ok = bool(correct["ok"] and wrong["ok"] and student_correct["ok"] and student_wrong["ok"])
                row = {
                    "experiment": EXPERIMENT,
                    "pair_id": pair["pair_id"],
                    "task_id": pair["task_id"],
                    "fork_index": fork["fork_index"],
                    "context": context,
                    "stratum": fork["stratum"],
                    "correct_branch_preview": fork["correct_branch_preview"],
                    "wrong_branch_preview": fork["wrong_branch_preview"],
                    "prefix_tail": fork["prefix_tail"],
                    "ok": ok,
                    "correct_logprob_mean": correct["logprob_mean"],
                    "wrong_logprob_mean": wrong["logprob_mean"],
                    "correct_logprob_sum": correct["logprob_sum"],
                    "wrong_logprob_sum": wrong["logprob_sum"],
                    "student_correct_logprob_mean": student_correct["logprob_mean"],
                    "student_wrong_logprob_mean": student_wrong["logprob_mean"],
                    "preference_mean": correct["logprob_mean"] - wrong["logprob_mean"],
                    "preference_sum": correct["logprob_sum"] - wrong["logprob_sum"],
                    "student_preference_mean": student_correct["logprob_mean"] - student_wrong["logprob_mean"],
                    "correct_positive_pressure_mean": max(0.0, correct["logprob_mean"] - student_correct["logprob_mean"]),
                    "wrong_positive_pressure_mean": max(0.0, wrong["logprob_mean"] - student_wrong["logprob_mean"]),
                    "correct_token_count": correct["token_count"],
                    "wrong_token_count": wrong["token_count"],
                }
                fork_rows.append(row)

        for candidate_kind in ["correct", "wrong"]:
            code = pair[candidate_kind]["code"]
            token_map = pair["token_maps"][candidate_kind]
            student = code_score(model, tokenizer, prompts["student_no_hint"], code, args.max_length)
            usage["scored_sequences"] += 1
            usage["prompt_tokens_estimate"] += estimate_text_tokens(tokenizer, prompts["student_no_hint"])
            usage["completion_tokens_estimate"] += estimate_text_tokens(tokenizer, code)
            student_by_offset = {
                (row["rel_start"], row["rel_end"]): row for row in student.get("tokens", [])
            }
            for context in TEACHER_CONTEXTS:
                teacher = code_score(model, tokenizer, prompts[context], code, args.max_length)
                usage["scored_sequences"] += 1
                usage["prompt_tokens_estimate"] += estimate_text_tokens(tokenizer, prompts[context])
                usage["completion_tokens_estimate"] += estimate_text_tokens(tokenizer, code)
                for row in teacher.get("tokens", []):
                    key = (row["rel_start"], row["rel_end"])
                    student_row = student_by_offset.get(key)
                    if student_row is None:
                        continue
                    bucket = bucket_for_offset(token_map, row["rel_start"], row["rel_end"])
                    token_rows.append(
                        {
                            "experiment": EXPERIMENT,
                            "pair_id": pair["pair_id"],
                            "task_id": pair["task_id"],
                            "context": context,
                            "candidate_kind": candidate_kind,
                            "bucket": bucket,
                            "rel_start": row["rel_start"],
                            "rel_end": row["rel_end"],
                            "token_text": row["token_text"],
                            "teacher_logprob": row["logprob"],
                            "student_logprob": student_row["logprob"],
                            "gap": row["logprob"] - student_row["logprob"],
                        }
                    )
    usage["forward_tokens_estimate"] = usage["prompt_tokens_estimate"] + usage["completion_tokens_estimate"]
    fork_summary = summarize_forks(fork_rows)
    token_summary = summarize_tokens(token_rows)
    summary = {
        "experiment": EXPERIMENT,
        "pairs": len(pairs),
        "fork_rows": len(fork_rows),
        "token_rows": len(token_rows),
        "contexts": TEACHER_CONTEXTS,
        "fork_summary": fork_summary,
        "token_summary": token_summary,
        "gate": gate(fork_summary),
        "usage_estimate": usage,
        "paths": {
            "fork_out": str(args.fork_out),
            "token_out": str(args.token_out),
        },
    }
    write_jsonl(args.fork_out, fork_rows)
    write_jsonl(args.token_out, token_rows)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
