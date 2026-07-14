#!/usr/bin/env python3
"""Frozen grading and selection policy for successful-sibling distillation."""

from __future__ import annotations

import hashlib
import re
from collections import Counter


EXPECTED_SKILLS = (
    "induct",
    "execute",
    "trace",
    "verify",
    "repair",
    "optimize",
    "abstain",
    "state",
    "order",
    "probe",
)
QUOTA_PER_SKILL = 4
SAMPLES_PER_FAILURE = 16
MAX_SIBLING_THINKING_TOKENS = 768
SELECTION_SEED = 55116
EOS_TEXT = "<|im_end|>"


def expected_answer(source: dict) -> str:
    value = source.get("answer")
    if not isinstance(value, str) or not value.startswith("ANSWER: ") or "\n" in value:
        raise ValueError(f"malformed oracle answer: {source.get('task_id')}")
    return value.removeprefix("ANSWER: ").strip()


def extract_answer(text: str) -> str | None:
    tail = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    matches = re.findall(r"(?:^|\n)ANSWER:\s*([^\n<]+)", tail)
    return matches[-1].strip() if matches else None


def grade_greedy(source: dict, rollout: dict) -> dict:
    outputs = rollout.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError(f"expected one greedy output: {source.get('task_id')}")
    output = outputs[0]
    text = output.get("text")
    sampled = output.get("n_sampled_tokens")
    thinking = output.get("n_thinking_tokens")
    if not isinstance(text, str) or not isinstance(sampled, int) or not isinstance(thinking, int):
        raise ValueError(f"greedy output lacks text/token accounting: {source.get('task_id')}")
    observed = extract_answer(text)
    target = expected_answer(source)
    cap_contact = bool(
        output.get("truncated") is True
        or output.get("finish_reason") == "length"
        or output.get("thinking_closed") is not True
    )
    missing_answer = observed is None
    wrong_answer = observed is not None and observed != target
    reasons = [
        name
        for name, active in (
            ("cap_contact", cap_contact),
            ("missing_answer", missing_answer),
            ("wrong_answer", wrong_answer),
        )
        if active
    ]
    return {
        "task_id": source["task_id"],
        "skill": source["selection_skill"],
        "kind": source["kind"],
        "hard_failure": bool(reasons),
        "reasons": reasons,
        "observed_answer": observed,
        "expected_answer": target,
        "n_sampled_tokens": sampled,
        "n_thinking_tokens": thinking,
    }


def parse_successful_sibling(source: dict, output: dict) -> dict:
    text = output.get("text")
    sampled = output.get("n_sampled_tokens")
    thinking_tokens = output.get("n_thinking_tokens")
    sample_index = output.get("sample_index")
    if (
        not isinstance(text, str)
        or not isinstance(sampled, int)
        or not isinstance(thinking_tokens, int)
        or not isinstance(sample_index, int)
    ):
        raise ValueError(f"sample lacks text/token accounting: {source.get('task_id')}")
    target = expected_answer(source)
    reasons: list[str] = []
    if output.get("thinking_closed") is not True:
        reasons.append("thinking_not_closed")
    if output.get("truncated") is not False or output.get("finish_reason") != "stop":
        reasons.append("not_naturally_stopped")
    if thinking_tokens > MAX_SIBLING_THINKING_TOKENS:
        reasons.append("over_short_budget")
    if text.count("</think>") != 1:
        reasons.append("noncanonical_close_count")
    observed = extract_answer(text)
    if observed is None:
        reasons.append("missing_answer")
    elif observed != target:
        reasons.append("wrong_answer")

    think = ""
    answer_tail = ""
    if text.count("</think>") == 1:
        think_raw, answer_raw = text.split("</think>", 1)
        think = think_raw.strip()
        answer_tail = answer_raw.strip()
        if not think:
            reasons.append("empty_thinking")
        if EOS_TEXT in think:
            reasons.append("eos_inside_thinking")
        if answer_tail != f"ANSWER: {target}{EOS_TEXT}":
            reasons.append("noncanonical_answer_tail")

    return {
        "sample_index": sample_index,
        "qualified": not reasons,
        "reasons": reasons,
        "observed_answer": observed,
        "expected_answer": target,
        "n_sampled_tokens": sampled,
        "n_thinking_tokens": thinking_tokens,
        "think": think,
        "answer": f"ANSWER: {target}",
        "raw_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def choose_best_sibling(source: dict, rollout: dict) -> tuple[dict | None, list[dict]]:
    outputs = rollout.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != SAMPLES_PER_FAILURE:
        raise ValueError(f"expected {SAMPLES_PER_FAILURE} sibling samples: {source.get('task_id')}")
    graded = [parse_successful_sibling(source, output) for output in outputs]
    qualified = [item for item in graded if item["qualified"]]
    if not qualified:
        return None, graded
    return min(
        qualified,
        key=lambda item: (
            item["n_sampled_tokens"],
            item["n_thinking_tokens"],
            hashlib.sha256(
                f"{SELECTION_SEED}:{source['task_id']}:{item['sample_index']}".encode()
            ).digest(),
        ),
    ), graded


def select_balanced(best_by_task: list[dict]) -> tuple[list[dict], dict[str, int]]:
    availability = Counter(item["skill"] for item in best_by_task)
    if any(availability[skill] < QUOTA_PER_SKILL for skill in EXPECTED_SKILLS):
        return [], {skill: availability[skill] for skill in EXPECTED_SKILLS}
    selected: list[dict] = []
    for skill in EXPECTED_SKILLS:
        candidates = [item for item in best_by_task if item["skill"] == skill]
        candidates.sort(
            key=lambda item: (
                item["best"]["n_sampled_tokens"],
                item["best"]["n_thinking_tokens"],
                hashlib.sha256(f"{SELECTION_SEED}:{item['task_id']}".encode()).digest(),
            )
        )
        selected.extend(candidates[:QUOTA_PER_SKILL])
    return selected, {skill: availability[skill] for skill in EXPECTED_SKILLS}


def training_row(source: dict, selected: dict, sibling_rollout_sha256: str) -> dict:
    best = selected["best"]
    return {
        "messages": source["messages"],
        "think": best["think"],
        "answer": best["answer"],
        "kind": source["kind"],
        "family": "universal_successful_sibling",
        "surface": source["surface"],
        "level": source["level"],
        "n_think_tokens": best["n_thinking_tokens"],
        "row_weight": 1.0,
        "task_id": source["task_id"],
        "selection_skill": source["selection_skill"],
        "teacher": {
            "policy": "authenticated_replay_after_close",
            "sampling_seed": 66117,
            "sample_index": best["sample_index"],
            "n_sampled_tokens": best["n_sampled_tokens"],
            "n_thinking_tokens": best["n_thinking_tokens"],
            "raw_text_sha256": best["raw_text_sha256"],
            "rollout_sha256": sibling_rollout_sha256,
            "selected_shortest_qualified": True,
            "oracle_trace_used": False,
        },
    }
