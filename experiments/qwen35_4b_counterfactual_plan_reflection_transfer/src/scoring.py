"""Strict action-answer parsing, exact scoring, and censoring diagnostics."""

from __future__ import annotations

import json
import re
from typing import Any


ANSWER = re.compile(r"\s*ANSWER:\s*(\[.*\])\s*", re.DOTALL)


def parse_answer(text: str) -> list[Any] | None:
    visible = text.rsplit("</think>", 1)[-1]
    match = ANSWER.fullmatch(visible)
    if match is None:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(value, list) or len(value) != 3:
        return None
    return value


def exact_periodic_tail(
    token_ids: list[int], min_repeats: int, max_period: int, min_total_repeated: int
) -> bool:
    """Detect an exact periodic suffix without using decoded lexical content."""
    if len(token_ids) < min_total_repeated:
        return False
    for period in range(1, min(max_period, len(token_ids) // min_repeats) + 1):
        pattern = token_ids[-period:]
        repeats = 1
        cursor = len(token_ids) - 2 * period
        while cursor >= 0 and token_ids[cursor:cursor + period] == pattern:
            repeats += 1
            cursor -= period
        if repeats >= min_repeats and repeats * period >= min_total_repeated:
            return True
    return False


def score_generation_rows(
    generated: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    arm: str,
    candidate_counts: tuple[int, ...],
    answer_max_tokens: int,
    loop_detector: dict[str, int],
) -> list[dict[str, Any]]:
    label_by_id = {row["id"]: row for row in labels}
    if len(label_by_id) != len(labels):
        raise ValueError("duplicate label IDs")
    generated_ids = [row["id"] for row in generated]
    if len(set(generated_ids)) != len(generated_ids):
        raise ValueError("duplicate generated IDs")
    if set(generated_ids) != set(label_by_id):
        raise ValueError("generated/label task IDs differ")
    maximum = max(candidate_counts)
    scored = []
    for row in generated:
        label = label_by_id[row["id"]]
        outputs = row["outputs"]
        if len(outputs) < maximum:
            raise ValueError(f"{row['id']} has fewer than {maximum} candidates")
        candidates = []
        for output in outputs[:maximum]:
            parsed = parse_answer(str(output["text"]))
            candidates.append(
                {
                    "parsed": parsed is not None,
                    "correct": parsed == label["answers"],
                    "answer_limit": bool(output.get("truncated"))
                    or int(output["n_answer_tokens"]) >= answer_max_tokens,
                    "periodic_loop": exact_periodic_tail(
                        list(output.get("retained_thinking_token_ids", [])),
                        min_repeats=int(loop_detector["min_repeats"]),
                        max_period=int(loop_detector["max_period_tokens"]),
                        min_total_repeated=int(loop_detector["min_total_repeated_tokens"]),
                    ),
                    "logical_tokens": int(output["n_sampled_tokens"])
                    + int(output.get("n_injected_tokens", 0)),
                }
            )
        scored_row = {
            "schema_version": 1,
            "task_id": row["id"],
            "family": label["family"],
            "depth": int(label["depth"]),
            "split": label["split"],
            "arm": arm,
        }
        for count in candidate_counts:
            prefix = candidates[:count]
            scored_row[f"coverage_at_{count}"] = float(any(item["correct"] for item in prefix))
            scored_row[f"candidate_accuracy_at_{count}"] = sum(
                item["correct"] for item in prefix
            ) / count
        primary = candidates[:maximum]
        scored_row.update(
            {
                "strict_parse_rate": sum(item["parsed"] for item in primary) / maximum,
                "answer_limit_contact": sum(item["answer_limit"] for item in primary) / maximum,
                "periodic_loop_contact": sum(item["periodic_loop"] for item in primary) / maximum,
                "logical_tokens": sum(item["logical_tokens"] for item in primary),
            }
        )
        scored.append(scored_row)
    return scored


def score_literal_reflection_diagnostic(
    reflection_generated: list[dict[str, Any]],
    action_generated: list[dict[str, Any]],
    base_generated: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    literal_candidate_count: int,
) -> list[dict[str, Any]]:
    """Compare literal plan-then-action against a token-matched base prefix."""
    label_by_id = {row["id"]: row for row in labels}
    reflection_by_id = {row["id"]: row for row in reflection_generated}
    base_by_id = {row["id"]: row for row in base_generated}
    action_by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in action_generated:
        parent = str(row["meta"]["parent_task_id"])
        action_by_parent.setdefault(parent, []).append(row)
    if not (
        set(label_by_id) == set(reflection_by_id) == set(base_by_id) == set(action_by_parent)
    ):
        raise ValueError("literal/base/label task IDs differ")

    def logical(output: dict[str, Any]) -> int:
        return int(output["n_sampled_tokens"]) + int(output.get("n_injected_tokens", 0))

    scored = []
    for task_id in sorted(label_by_id):
        reflection_outputs = reflection_by_id[task_id]["outputs"]
        action_rows = sorted(
            action_by_parent[task_id], key=lambda row: int(row["meta"]["sample_index"])
        )
        if len(reflection_outputs) != literal_candidate_count or len(action_rows) != literal_candidate_count:
            raise ValueError("literal candidate count differs from preregistration")
        if any(len(row["outputs"]) != 1 for row in action_rows):
            raise ValueError("literal action continuation must use n=1 per plan")
        literal_outputs = [row["outputs"][0] for row in action_rows]
        literal_tokens = sum(logical(output) for output in reflection_outputs) + sum(
            logical(output) for output in literal_outputs
        )
        literal_coverage = any(
            parse_answer(str(output["text"])) == label_by_id[task_id]["answers"]
            for output in literal_outputs
        )
        base_outputs = base_by_id[task_id]["outputs"]
        cumulative = 0
        matched_prefix = []
        for output in base_outputs:
            matched_prefix.append(output)
            cumulative += logical(output)
            if cumulative >= literal_tokens:
                break
        if cumulative < literal_tokens:
            raise ValueError("base reserve does not reach literal logical-token spend")
        base_coverage = any(
            parse_answer(str(output["text"])) == label_by_id[task_id]["answers"]
            for output in matched_prefix
        )
        scored.append(
            {
                "schema_version": 1,
                "task_id": task_id,
                "family": label_by_id[task_id]["family"],
                "split": label_by_id[task_id]["split"],
                "literal_candidates": literal_candidate_count,
                "literal_logical_tokens": literal_tokens,
                "literal_coverage": float(literal_coverage),
                "matched_base_candidates": len(matched_prefix),
                "matched_base_logical_tokens": cumulative,
                "matched_base_coverage": float(base_coverage),
                "literal_minus_matched_base": float(literal_coverage) - float(base_coverage),
            }
        )
    return scored
