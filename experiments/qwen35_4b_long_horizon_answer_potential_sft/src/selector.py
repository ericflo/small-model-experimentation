"""Quality-first, diversity-second full-trace banking and SFT records."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import defaultdict
from typing import Any, Mapping, Sequence

ANSWER_BOUNDARY = "</think>\n\nANSWER: "


def _normalized_trigrams(text: str) -> set[tuple[str, str, str]]:
    value = re.sub(r"\b[-+]?\d+(?:\.\d+)?\b", " <num> ", text.lower())
    value = re.sub(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", " <id> ", value)
    tokens = re.findall(r"<num>|<id>|[a-z]+|[^\w\s]", value)
    return {
        (tokens[index], tokens[index + 1], tokens[index + 2])
        for index in range(max(0, len(tokens) - 2))
    }


def structural_distance(left: str, right: str) -> float:
    a = _normalized_trigrams(left)
    b = _normalized_trigrams(right)
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def _stable_tiebreak(seed: int, *parts: str) -> int:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def select_quality_diverse(
    candidates: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    top_candidates: int,
    near_best: float,
    max_selected: int,
) -> list[dict[str, Any]]:
    if max_selected < 1:
        raise ValueError("max_selected must be positive")
    ranked = sorted(
        (dict(row) for row in candidates if math.isfinite(float(row[metric]))),
        key=lambda row: (-float(row[metric]), str(row["trace_id"])),
    )[:top_candidates]
    if not ranked:
        return []
    selected = [ranked[0]]
    while len(selected) < max_selected:
        eligible = [
            row
            for row in ranked
            if row["trace_id"] not in {prior["trace_id"] for prior in selected}
            and float(row[metric]) >= float(ranked[0][metric]) - near_best
        ]
        if not eligible:
            break
        winner = max(
            eligible,
            key=lambda row: (
                min(
                    structural_distance(str(row.get("text", "")), str(prior.get("text", "")))
                    for prior in selected
                ),
                float(row[metric]),
                str(row["trace_id"]),
            ),
        )
        selected.append(winner)
    return selected


def length_matched(
    candidates: Sequence[Mapping[str, Any]],
    target_lengths: Sequence[int],
    *,
    seed: int,
    key: str,
) -> list[dict[str, Any]]:
    available = [dict(row) for row in candidates]
    selected: list[dict[str, Any]] = []
    for index, target in enumerate(target_lengths):
        if not available:
            break
        winner = min(
            available,
            key=lambda row: (
                abs(int(row["n_tokens"]) - int(target)),
                _stable_tiebreak(seed, key, str(index), str(row["trace_id"])),
            ),
        )
        selected.append(winner)
        available = [row for row in available if row["trace_id"] != winner["trace_id"]]
    return selected


def select_task(
    traces: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    rollouts: Sequence[Mapping[str, Any]],
    *,
    selector_config: Mapping[str, Any],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    trace_by_id = {str(row["trace_id"]): dict(row) for row in traces}
    rollout_by_id = {str(row["trace_id"]): dict(row) for row in rollouts}
    joined: list[dict[str, Any]] = []
    for score in scores:
        trace = trace_by_id.get(str(score["trace_id"]))
        if trace is None:
            raise ValueError(f"score has no trace: {score['trace_id']}")
        if not trace["natural_close"] or trace["loop_flag"]:
            continue
        if int(score["full_sequence_tokens"]) > int(selector_config["max_train_length"]):
            continue
        rollout = rollout_by_id.get(str(score["trace_id"]), {})
        joined.append({**trace, **dict(score), **{f"rollout_{k}": v for k, v in rollout.items()}})
    answer = select_quality_diverse(
        joined,
        metric="answer_gain_per_answer_token",
        top_candidates=int(selector_config["top_candidates"]),
        near_best=float(selector_config["near_best_nats_per_answer_token"]),
        max_selected=int(selector_config["max_per_task"]),
    )
    joint = select_quality_diverse(
        joined,
        metric="joint_gain_per_answer_token",
        top_candidates=int(selector_config["top_candidates"]),
        near_best=float(selector_config["near_best_nats_per_answer_token"]),
        max_selected=int(selector_config["max_per_task"]),
    )
    target_lengths = [int(row["n_tokens"]) for row in answer]
    random_natural = length_matched(
        joined, target_lengths, seed=seed, key=f"{joined[0]['task_id'] if joined else 'empty'}::random"
    )
    successes = [row for row in joined if bool(row.get("rollout_any_success", False))]
    success = length_matched(
        successes, target_lengths, seed=seed, key=f"{joined[0]['task_id'] if joined else 'empty'}::success"
    )
    return {
        "answer_potential": answer,
        "joint_potential": joint,
        "random_natural": random_natural,
        "success_rft": success,
        "eligible": joined,
    }


def deranged_sources(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One-to-one within-stratum reassignment with no same-task source."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["family"]), int(row["level"]))].append(dict(row))
    output: list[dict[str, Any]] = []
    for stratum in sorted(groups):
        targets = sorted(
            groups[stratum],
            key=lambda row: (int(row["n_tokens"]), str(row["task_id"]), str(row["trace_id"])),
        )
        if len({str(row["task_id"]) for row in targets}) < 2:
            raise ValueError(f"cannot derange single-task stratum {stratum}")
        valid: list[tuple[int, int]] = []
        for offset in range(1, len(targets)):
            sources = targets[offset:] + targets[:offset]
            if all(str(target["task_id"]) != str(source["task_id"]) for target, source in zip(targets, sources)):
                cost = sum(abs(int(target["n_tokens"]) - int(source["n_tokens"])) for target, source in zip(targets, sources))
                valid.append((cost, offset))
        if not valid:
            raise ValueError(f"no one-to-one task derangement for {stratum}")
        _, offset = min(valid)
        sources = targets[offset:] + targets[:offset]
        for target, source in zip(targets, sources):
            output.append(
                {
                    **target,
                    "shuffle_source_trace_id": source["trace_id"],
                    "shuffle_source_task_id": source["task_id"],
                    "token_ids": list(source["token_ids"]),
                    "text": source.get("text", ""),
                    "n_tokens": int(source["n_tokens"]),
                    "source_kind": "potential_shuffle",
                }
            )
    return output


def sft_record(
    *,
    arm: str,
    item: Mapping[str, Any],
    trace: Mapping[str, Any] | None,
    tokenizer: Any,
    ordinal: int,
    max_length: int,
) -> dict[str, Any]:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": str(item["prompt"])}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    expected = tokenizer.encode("<think>\n", add_special_tokens=False)
    if prompt_ids[-len(expected) :] != expected:
        raise RuntimeError(f"unexpected thinking prompt for {item['id']}")
    trace_ids = [] if trace is None else [int(value) for value in trace["token_ids"]]
    boundary_ids = tokenizer.encode(ANSWER_BOUNDARY, add_special_tokens=False)
    answer_ids = tokenizer.encode(str(item["canonical_answer"]), add_special_tokens=False)
    eos_id = int(tokenizer.eos_token_id)
    total = len(prompt_ids) + len(trace_ids) + len(boundary_ids) + len(answer_ids) + 1
    if total > max_length:
        raise RuntimeError(
            f"selected record exceeds max length instead of being truncated: {item['id']} {total}"
        )
    trace_id = "empty" if trace is None else str(
        trace.get("shuffle_source_trace_id", trace["trace_id"])
    )
    source_task_id = None if trace is None else trace.get(
        "shuffle_source_task_id", trace.get("task_id")
    )
    return {
        "schema_version": 1,
        "record_id": f"{arm}::{item['id']}::{ordinal:02d}::{trace_id}",
        "arm": arm,
        "task_id": item["id"],
        "family": item["family"],
        "level": item["level"],
        "messages": [{"role": "user", "content": item["prompt"]}],
        "prompt_token_ids": prompt_ids,
        "trace_token_ids": trace_ids,
        "answer_boundary_token_ids": boundary_ids,
        "answer_token_ids": answer_ids,
        "eos_token_id": eos_id,
        "canonical_answer": item["canonical_answer"],
        "source_trace_id": None if trace is None else trace_id,
        "source_task_id": source_task_id,
        "source_kind": "empty" if trace is None else trace.get("source_kind"),
        "trace_tokens": len(trace_ids),
        "total_tokens": total,
        "answer_gain_per_answer_token": None if trace is None else trace.get("answer_gain_per_answer_token"),
        "joint_gain_per_answer_token": None if trace is None else trace.get("joint_gain_per_answer_token"),
        "rollout_any_success": None if trace is None else trace.get("rollout_any_success"),
        "shuffle_source_trace_id": None if trace is None else trace.get("shuffle_source_trace_id"),
        "shuffle_source_task_id": None if trace is None else trace.get("shuffle_source_task_id"),
    }


def oversample_to(rows: Sequence[Mapping[str, Any]], target: int, *, seed: int) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("cannot oversample an empty dataset")
    if len(rows) > target:
        raise ValueError("oversampling target is smaller than source")
    output = [dict(row) for row in rows]
    rng = random.Random(seed)
    order = list(range(len(rows)))
    repetition = 1
    while len(output) < target:
        rng.shuffle(order)
        for index in order:
            duplicate = dict(rows[index])
            duplicate["record_id"] = f"{duplicate['record_id']}::repeat{repetition:03d}"
            duplicate["oversampled"] = True
            output.append(duplicate)
            if len(output) == target:
                break
        repetition += 1
    return output
