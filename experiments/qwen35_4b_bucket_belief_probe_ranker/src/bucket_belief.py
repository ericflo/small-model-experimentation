from __future__ import annotations

import math
from typing import Any

import numpy as np

from .operator_env import (
    LETTERS,
    Operator,
    all_cases,
    bucket_stats,
    candidate_mask,
    case_to_text,
    decoded_output,
    eval_pair,
    operator_values,
    pair_output_matrix,
    rank_queries_by_expected_split,
)
from .prompts import bucket_belief_prompt


def observed_executions(record: dict[str, Any], operators: list[Operator], used_query_indices: list[int]) -> list[dict[str, Any]]:
    target_left, target_right = record["target_pair"]
    observed = list(record["visible_cases"])
    observed.extend(record["query_pool"][idx] for idx in used_query_indices)
    return [
        {
            "case": case,
            "output": eval_pair(record, operators, target_left, target_right, case),
        }
        for case in observed
    ]


def _example_prior_rank(mask: np.ndarray, mat: np.ndarray, raw_values: set[int] | None) -> int | str:
    if raw_values is None:
        bucket_mask = mask
    else:
        bucket_mask = mask & np.isin(mat, list(raw_values))
    coords = np.argwhere(bucket_mask)
    if coords.size == 0:
        return "none"
    n = mask.shape[1]
    ranks = coords[:, 0] * n + coords[:, 1]
    return int(ranks.min())


def bucket_options(
    record: dict[str, Any],
    mask: np.ndarray,
    mat: np.ndarray,
    target_raw_output: int,
    max_options: int = 8,
) -> tuple[list[dict[str, Any]], str, int]:
    vals, counts = np.unique(mat[mask], return_counts=True)
    total = int(counts.sum())
    items = [
        {
            "raw": int(raw),
            "value": decoded_output(int(raw), record["template"]),
            "count": int(count),
        }
        for raw, count in zip(vals, counts)
    ]
    items.sort(key=lambda row: (-int(row["count"]), str(row["value"])))

    use_other = len(items) > max_options
    top_items = items[: max_options - 1] if use_other else items[:max_options]
    options: list[dict[str, Any]] = []
    label_letter = ""
    survivors_if_taken = 0
    for i, item in enumerate(top_items):
        letter = LETTERS[i]
        count = int(item["count"])
        if int(item["raw"]) == int(target_raw_output):
            label_letter = letter
            survivors_if_taken = count
        options.append(
            {
                "letter": letter,
                "value": item["value"],
                "raw": int(item["raw"]),
                "count": count,
                "fraction": count / max(total, 1),
                "is_other": False,
                "example_prior_rank": _example_prior_rank(mask, mat, {int(item["raw"])}),
            }
        )

    if use_other:
        other_items = items[max_options - 1 :]
        other_raw = {int(item["raw"]) for item in other_items}
        other_count = sum(int(item["count"]) for item in other_items)
        other_letter = LETTERS[len(top_items)]
        if int(target_raw_output) in other_raw:
            label_letter = other_letter
            survivors_if_taken = next(int(item["count"]) for item in other_items if int(item["raw"]) == int(target_raw_output))
        options.append(
            {
                "letter": other_letter,
                "value": "OTHER",
                "raw": None,
                "count": int(other_count),
                "fraction": other_count / max(total, 1),
                "is_other": True,
                "example_prior_rank": _example_prior_rank(mask, mat, other_raw),
            }
        )

    if not label_letter:
        raise RuntimeError("target output was not assigned to any bucket option")
    return options, label_letter, int(survivors_if_taken)


def make_bucket_example(
    record: dict[str, Any],
    operators: list[Operator],
    used_query_indices: list[int],
    query_index: int,
    max_options: int = 8,
) -> dict[str, Any]:
    mask = candidate_mask(record, operators, used_query_indices)
    before = int(mask.sum())
    n = record["library_size"]
    ops = operators[:n]
    cases = all_cases(record)
    case_index = len(record["visible_cases"]) + query_index
    left_values = operator_values(ops, cases, "left")
    right_values = operator_values(ops, cases, "right")
    mat = pair_output_matrix(record, left_values, right_values, case_index)
    target_left, target_right = record["target_pair"]
    target_raw_output = int(mat[target_left, target_right])
    options, label, survivors_if_taken = bucket_options(record, mask, mat, target_raw_output, max_options=max_options)
    vals, counts = np.unique(mat[mask], return_counts=True)
    buckets = {decoded_output(int(v), record["template"]): int(c) for v, c in zip(vals, counts)}
    stats = bucket_stats(buckets)
    reward = math.log2(max(before, 1) / max(survivors_if_taken, 1))
    payload = {
        "candidate_count": before,
        "used_query_indices": list(used_query_indices),
        "observations": observed_executions(record, operators, used_query_indices),
        "probe_case": record["query_pool"][query_index],
        "probe_case_text": case_to_text(record["query_pool"][query_index]),
        "query_index": int(query_index),
        "target_output": decoded_output(target_raw_output, record["template"]),
        "target_raw_output": target_raw_output,
        "bucket_options": options,
    }
    return {
        "prompt": bucket_belief_prompt(record, payload),
        "label": label,
        "label_index": LETTERS.index(label),
        "record_id": record["record_id"],
        "split": record["split"],
        "library_size": record["library_size"],
        "template": record["template"],
        "used_query_indices": list(used_query_indices),
        "query_index": int(query_index),
        "candidate_count": before,
        "survivors_if_taken": survivors_if_taken,
        "target_output": payload["target_output"],
        "bucket_options": options,
        "expected_remaining": float(stats["expected_remaining"]),
        "entropy": float(stats["entropy"]),
        "largest": int(stats["largest"]),
        "reward": float(reward),
    }


def top_split_queries(
    record: dict[str, Any],
    operators: list[Operator],
    used_query_indices: list[int],
    top_k: int,
) -> list[int]:
    used = set(used_query_indices)
    candidates = [idx for idx in range(len(record["query_pool"])) if idx not in used]
    ranked = rank_queries_by_expected_split(record, operators, used_query_indices, candidates)
    return [int(row["query_index"]) for row in ranked[:top_k]]


def predicted_expected_survivors(example: dict[str, Any], probabilities: list[float]) -> float:
    option_indices = [LETTERS.index(option["letter"]) for option in example["bucket_options"]]
    total_prob = sum(float(probabilities[i]) for i in option_indices)
    if total_prob <= 0:
        return float(example["candidate_count"])
    return sum(
        (float(probabilities[i]) / total_prob) * int(option["count"])
        for option, i in zip(example["bucket_options"], option_indices)
    )
