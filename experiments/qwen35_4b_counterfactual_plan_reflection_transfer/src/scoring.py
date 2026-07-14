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
