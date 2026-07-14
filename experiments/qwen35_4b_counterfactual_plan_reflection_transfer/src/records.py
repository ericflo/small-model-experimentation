"""Immutable training/evaluation records and exact target-only token masks."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from taskgen import ACTION_QUESTION, REFLECTION_QUESTION, FAMILIES, build_reflection_arms


AUXILIARY_QUESTION = REFLECTION_QUESTION.replace(
    "Pause before solving.", "Provide exact labels."
)
TRAINING_ARMS = (
    "reflection_correct",
    "reflection_shuffled",
    "auxiliary_plan_label_correct",
    "direct_plan_answer_positive_control",
)


def _sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _plan_thought(ops: list[str]) -> str:
    return "The exact ordered plan is " + " -> ".join(ops) + "."


def _messages(task: dict[str, Any], question: str) -> list[dict[str, str]]:
    return [*task["common_messages"], {"role": "user", "content": question}]


def _schedule_tasks(
    tasks: list[dict[str, Any]], schedule_seed: int, per_family_per_step: int
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    """Return a fixed 18-row optimizer-step schedule (six rows per family)."""
    by_family: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_family.setdefault(task["family"], []).append(task)
    family_names = [family.name for family in FAMILIES]
    if set(by_family) != set(family_names):
        raise ValueError("training rows do not contain every configured family")
    blocks: dict[str, list[list[dict[str, Any]]]] = {}
    for family_index, family_name in enumerate(family_names):
        rows = sorted(by_family[family_name], key=lambda row: row["task_id"])
        random.Random(schedule_seed + family_index).shuffle(rows)
        if len(rows) % per_family_per_step:
            raise ValueError("per-family training count is not optimizer-step divisible")
        blocks[family_name] = [
            rows[start:start + per_family_per_step]
            for start in range(0, len(rows), per_family_per_step)
        ]
    block_count = {len(value) for value in blocks.values()}
    if len(block_count) != 1:
        raise ValueError("families have different training block counts")
    optimizer_groups: list[list[str]] = []
    scheduled: list[dict[str, Any]] = []
    for block_index in range(block_count.pop()):
        group = []
        for family_name in family_names:
            rows = blocks[family_name][block_index]
            scheduled.extend(rows)
            group.extend(row["task_id"] for row in rows)
        optimizer_groups.append(group)
    return scheduled, optimizer_groups


def build_training_records(
    tasks: list[dict[str, Any]],
    shuffle_seed: int,
    schedule_seed: int,
    per_family_per_step: int = 6,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build four arms with a shared, explicit optimizer-step task schedule."""
    scheduled, optimizer_groups = _schedule_tasks(tasks, schedule_seed, per_family_per_step)

    # Reorder as contiguous family blocks for within-(family, optimizer-step)
    # derangement, then map back onto the shared interleaved step schedule.
    group_of = {
        task_id: group_index
        for group_index, group in enumerate(optimizer_groups)
        for task_id in group
    }
    grouped = sorted(
        scheduled,
        key=lambda task: (
            task["family"],
            group_of[task["task_id"]],
            task["task_id"],
        ),
    )
    reflection = build_reflection_arms(
        grouped, shuffle_seed, derangement_group_size=per_family_per_step
    )
    correct_by_id = {row["task_id"]: row for row in reflection["reflection_correct"]}
    shuffled_by_id = {row["task_id"]: row for row in reflection["reflection_shuffled"]}

    arms: dict[str, list[dict[str, Any]]] = {name: [] for name in TRAINING_ARMS}
    for step_index, group in enumerate(optimizer_groups):
        for task_id in group:
            correct = correct_by_id[task_id]
            shuffled = shuffled_by_id[task_id]
            base = {
                "schema_version": 1,
                "task_id": task_id,
                "family": correct["family"],
                "optimizer_group": step_index,
                "truth_ops": list(correct["target_ops"]),
                "truth_plan": correct["target_plan"],
            }
            arms["reflection_correct"].append(
                {
                    **base,
                    "arm": "reflection_correct",
                    "messages": _messages(correct, REFLECTION_QUESTION),
                    "think": _plan_thought(correct["supervision_ops"]),
                    "answer": correct["supervision_plan"],
                    "supervision_ops": list(correct["supervision_ops"]),
                    "supervision_source_task_id": correct["reflection_donor_task_id"],
                }
            )
            arms["reflection_shuffled"].append(
                {
                    **base,
                    "arm": "reflection_shuffled",
                    "messages": _messages(shuffled, REFLECTION_QUESTION),
                    "think": _plan_thought(shuffled["supervision_ops"]),
                    "answer": shuffled["supervision_plan"],
                    "supervision_ops": list(shuffled["supervision_ops"]),
                    "supervision_source_task_id": shuffled["reflection_donor_task_id"],
                }
            )
            arms["auxiliary_plan_label_correct"].append(
                {
                    **base,
                    "arm": "auxiliary_plan_label_correct",
                    "messages": _messages(correct, AUXILIARY_QUESTION),
                    "think": _plan_thought(correct["target_ops"]),
                    "answer": correct["target_plan"],
                    "supervision_ops": list(correct["target_ops"]),
                    "supervision_source_task_id": task_id,
                }
            )
            arms["direct_plan_answer_positive_control"].append(
                {
                    **base,
                    "arm": "direct_plan_answer_positive_control",
                    "messages": _messages(correct, ACTION_QUESTION),
                    "think": _plan_thought(correct["target_ops"]),
                    "answer": correct["target_answer"],
                    "supervision_ops": list(correct["target_ops"]),
                    "supervision_source_task_id": task_id,
                    "noncomparable_positive_control": True,
                }
            )

    expected_ids = [task_id for group in optimizer_groups for task_id in group]
    for arm, rows in arms.items():
        if [row["task_id"] for row in rows] != expected_ids:
            raise ValueError(f"{arm} does not follow the shared task schedule")
    for group_index in range(len(optimizer_groups)):
        correct_targets = sorted(
            (row["think"], row["answer"])
            for row in arms["reflection_correct"]
            if row["optimizer_group"] == group_index
        )
        shuffled_targets = sorted(
            (row["think"], row["answer"])
            for row in arms["reflection_shuffled"]
            if row["optimizer_group"] == group_index
        )
        if correct_targets != shuffled_targets:
            raise ValueError("correct/shuffled target multiset differs within optimizer step")
    receipt = {
        "schema_version": 1,
        "rows_per_arm": len(scheduled),
        "optimizer_groups_per_epoch": len(optimizer_groups),
        "rows_per_optimizer_group": len(optimizer_groups[0]),
        "per_family_per_optimizer_group": per_family_per_step,
        "schedule_sha256": _sha256(optimizer_groups),
        "arm_sha256": {arm: _sha256(rows) for arm, rows in arms.items()},
    }
    return arms, receipt


def encode_training_record(
    record: dict[str, Any],
    tokenizer: Any,
    max_length: int,
    think_weight: float,
    close_weight: float,
) -> dict[str, Any]:
    """Render one row and mask every token outside the final Assistant target."""
    prompt = tokenizer.apply_chat_template(
        record["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not prompt.endswith("<think>\n"):
        raise ValueError("unexpected Qwen thinking-template suffix")
    think = record["think"].strip() + "\n"
    close = "</think>\n\n"
    answer = record["answer"].strip() + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    think_ids = tokenizer(prompt + think, add_special_tokens=False)["input_ids"]
    close_ids = tokenizer(prompt + think + close, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + think + close + answer, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_length:
        raise ValueError("training row exceeds max_length; truncation is forbidden")
    if (
        think_ids[: len(prompt_ids)] != prompt_ids
        or close_ids[: len(think_ids)] != think_ids
        or full_ids[: len(close_ids)] != close_ids
    ):
        raise ValueError("tokenizer merged across a registered loss boundary")
    weights = (
        [0.0] * len(prompt_ids)
        + [think_weight] * (len(think_ids) - len(prompt_ids))
        + [close_weight] * (len(close_ids) - len(think_ids))
        + [1.0] * (len(full_ids) - len(close_ids))
    )
    labels = [token if weight else -100 for token, weight in zip(full_ids, weights, strict=True)]
    target_ids = full_ids[len(prompt_ids):]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "loss_weights": weights,
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
        "think_target_tokens": len(think_ids) - len(prompt_ids),
        "close_target_tokens": len(close_ids) - len(think_ids),
        "answer_target_tokens": len(full_ids) - len(close_ids),
        "input_ids_sha256": _sha256(full_ids),
        "target_ids_sha256": _sha256(target_ids),
        "mask_sha256": _sha256(labels),
    }


def validate_tokenized_parity(
    arms: dict[str, list[dict[str, Any]]], encoded: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    """Fail unless the causal arms are token/step matched as preregistered."""
    causal = ("reflection_correct", "reflection_shuffled")
    for arm in TRAINING_ARMS:
        if len(arms[arm]) != len(encoded[arm]):
            raise ValueError(f"encoded row count differs for {arm}")
    totals = {
        arm: {
            "prompt_tokens": sum(row["prompt_tokens"] for row in encoded[arm]),
            "target_tokens": sum(row["target_tokens"] for row in encoded[arm]),
            "forward_tokens": sum(len(row["input_ids"]) for row in encoded[arm]),
        }
        for arm in TRAINING_ARMS
    }
    if totals[causal[0]] != totals[causal[1]]:
        raise ValueError("correct/shuffled global token totals differ")
    correct_rows = encoded["reflection_correct"]
    auxiliary_rows = encoded["auxiliary_plan_label_correct"]
    for index, (correct_record, auxiliary_record) in enumerate(
        zip(correct_rows, auxiliary_rows, strict=True)
    ):
        if correct_record["prompt_tokens"] != auxiliary_record["prompt_tokens"]:
            raise ValueError(f"reflection/auxiliary prompt token mismatch at row {index}")
        if correct_record["target_ids_sha256"] != auxiliary_record["target_ids_sha256"]:
            raise ValueError(f"reflection/auxiliary target token mismatch at row {index}")
    group_indices = sorted({row["optimizer_group"] for row in arms[causal[0]]})
    group_receipts = []
    for group_index in group_indices:
        receipt = {"optimizer_group": group_index}
        for arm in causal:
            indices = [
                index
                for index, row in enumerate(arms[arm])
                if row["optimizer_group"] == group_index
            ]
            receipt[arm] = {
                "forward_tokens": sum(len(encoded[arm][index]["input_ids"]) for index in indices),
                "target_tokens": sum(encoded[arm][index]["target_tokens"] for index in indices),
            }
        if receipt[causal[0]] != receipt[causal[1]]:
            raise ValueError("correct/shuffled token totals differ within optimizer group")
        group_receipts.append(receipt)
    return {
        "schema_version": 1,
        "totals": totals,
        "optimizer_group_parity": group_receipts,
        "encoded_sha256": {
            arm: _sha256(encoded[arm]) for arm in TRAINING_ARMS
        },
    }
