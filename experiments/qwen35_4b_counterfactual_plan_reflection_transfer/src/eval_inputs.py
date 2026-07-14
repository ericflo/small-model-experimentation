"""Canonical sealed evaluation prompts, labels, receipts, and task mappings."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from taskgen import (
    ACTION_QUESTION,
    REFLECTION_QUESTION,
    build_corpus,
    build_retention_corpus,
)


def jsonl_payload(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def sealed_tasks(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    construction = config["construction"]
    if split == "retention":
        return build_retention_corpus(
            int(construction["per_family"]["retention_per_family_per_depth"]),
            int(construction["retention_seed"]),
        )
    if split not in {"calibration", "qualification", "confirmation"}:
        raise ValueError(f"unsupported sealed evaluation split: {split}")
    counts = {
        name: int(construction["per_family"][name])
        for name in ("train", "calibration", "qualification", "confirmation")
    }
    return build_corpus(counts, int(construction["seed"]))[split]


def action_bundles(
    config: dict[str, Any], split: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = sealed_tasks(config, split)
    prompts = [
        {
            "id": task["task_id"],
            "messages": [
                *task["common_messages"],
                {"role": "user", "content": ACTION_QUESTION},
            ],
            "meta": {
                "split": split,
                "family": task["family"],
                "depth": task["depth"],
                "input_kind": "action",
            },
        }
        for task in tasks
    ]
    labels = [
        {
            "id": task["task_id"],
            "split": split,
            "family": task["family"],
            "depth": task["depth"],
            "answers": task["answers"],
        }
        for task in tasks
    ]
    return prompts, labels


def reflection_prompts(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if split != "qualification":
        raise ValueError("literal reflection is sealed only on qualification")
    return [
        {
            "id": task["task_id"],
            "messages": [
                *task["common_messages"],
                {"role": "user", "content": REFLECTION_QUESTION},
            ],
            "meta": {
                "split": split,
                "family": task["family"],
                "depth": task["depth"],
                "input_kind": "literal_reflection",
            },
        }
        for task in sealed_tasks(config, split)
    ]


def action_receipt(
    config: dict[str, Any], config_sha256: str, split: str
) -> dict[str, Any]:
    prompts, labels = action_bundles(config, split)
    return {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_sha256,
        "input_kind": "action",
        "split": split,
        "rows": len(prompts),
        "prompt_sha256": hashlib.sha256(jsonl_payload(prompts)).hexdigest(),
        "label_sha256": hashlib.sha256(jsonl_payload(labels)).hexdigest(),
        "model_calls": 0,
        "gpu_events": 0,
        "benchmark_reads": 0,
    }


def reflection_receipt(
    config: dict[str, Any], config_sha256: str, split: str
) -> dict[str, Any]:
    prompts = reflection_prompts(config, split)
    return {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_sha256,
        "input_kind": "literal_reflection",
        "split": split,
        "rows": len(prompts),
        "prompt_sha256": hashlib.sha256(jsonl_payload(prompts)).hexdigest(),
        "model_calls": 0,
        "gpu_events": 0,
        "benchmark_reads": 0,
    }


def task_metadata(config: dict[str, Any], split: str) -> dict[str, tuple[str, int]]:
    return {
        str(task["task_id"]): (str(task["family"]), int(task["depth"]))
        for task in sealed_tasks(config, split)
    }


def literal_action_prompts(
    config: dict[str, Any],
    split: str,
    reflection_generated: list[dict[str, Any]],
    candidate_count: int,
) -> list[dict[str, Any]]:
    tasks = sealed_tasks(config, split)
    reflected = {str(row["id"]): row for row in reflection_generated}
    if len(reflected) != len(reflection_generated) or set(reflected) != {
        str(task["task_id"]) for task in tasks
    }:
        raise ValueError("literal reflection rows do not match the sealed tasks")
    prompts: list[dict[str, Any]] = []
    for task in tasks:
        row = reflected[str(task["task_id"])]
        if len(row.get("outputs", [])) != candidate_count:
            raise ValueError("literal reflection candidate count differs from config")
        for sample_index, output in enumerate(row["outputs"]):
            prompts.append(
                {
                    "id": f"{task['task_id']}::literal::{sample_index}",
                    "messages": [
                        *task["common_messages"],
                        {"role": "user", "content": REFLECTION_QUESTION},
                        {"role": "assistant", "content": str(output["text"])},
                        {"role": "user", "content": ACTION_QUESTION},
                    ],
                    "meta": {
                        "split": split,
                        "family": task["family"],
                        "depth": task["depth"],
                        "input_kind": "literal_action",
                        "parent_task_id": task["task_id"],
                        "sample_index": sample_index,
                        "reflection_text_sha256": hashlib.sha256(
                            str(output["text"]).encode()
                        ).hexdigest(),
                    },
                }
            )
    return prompts


def literal_action_receipt(
    *,
    config: dict[str, Any],
    config_sha256: str,
    split: str,
    prompts: list[dict[str, Any]],
    source_reflection_generated_sha256: str,
    source_reflection_metadata_sha256: str,
    source_reflection_input_receipt_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_sha256,
        "input_kind": "literal_action",
        "split": split,
        "rows": len(prompts),
        "prompt_sha256": hashlib.sha256(jsonl_payload(prompts)).hexdigest(),
        "source_reflection_generated_sha256": source_reflection_generated_sha256,
        "source_reflection_metadata_sha256": source_reflection_metadata_sha256,
        "source_reflection_input_receipt_sha256": source_reflection_input_receipt_sha256,
        "model_calls": 0,
        "gpu_events": 0,
        "benchmark_reads": 0,
    }
