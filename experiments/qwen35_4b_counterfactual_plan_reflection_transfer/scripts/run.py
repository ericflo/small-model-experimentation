#!/usr/bin/env python3
"""Fail-closed CPU construction for counterfactual plan reflection transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))


def _install_benchmark_firewall() -> None:
    """Deny Python file and directory access beneath the repository benchmark root."""
    benchmark_root = os.path.realpath(REPO / "benchmarks")

    def forbidden(value: object) -> bool:
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        path = os.path.realpath(os.fsdecode(value))
        try:
            return os.path.commonpath((path, benchmark_root)) == benchmark_root
        except ValueError:
            return False

    def audit(event: str, args: tuple[object, ...]) -> None:
        if event == "open" and args and forbidden(args[0]):
            raise PermissionError("benchmark read firewall: open denied")
        if event in {"os.listdir", "os.scandir"} and args and forbidden(args[0]):
            raise PermissionError(f"benchmark read firewall: {event} denied")

    sys.addaudithook(audit)


_install_benchmark_firewall()

from taskgen import (  # noqa: E402
    build_corpus,
    build_reflection_arms,
    build_retention_corpus,
    validate_corpus,
    validate_retention_corpus,
)
from records import build_training_records  # noqa: E402


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def construct(
    counts: dict[str, int],
    retention_per_family_per_depth: int,
    seed: int,
    retention_seed: int,
    shuffle_seed: int,
    schedule_seed: int,
    per_family_per_optimizer_group: int,
    mode: str,
) -> dict[str, object]:
    tasks = build_corpus(counts=counts, seed=seed)
    validation = validate_corpus(tasks, counts)
    retention = build_retention_corpus(retention_per_family_per_depth, retention_seed)
    retention_validation = validate_retention_corpus(
        retention, retention_per_family_per_depth
    )
    arms = build_reflection_arms(tasks["train"], seed=shuffle_seed)
    training_records, training_receipt = build_training_records(
        tasks["train"],
        shuffle_seed=shuffle_seed,
        schedule_seed=schedule_seed,
        per_family_per_step=per_family_per_optimizer_group,
    )

    for correct, shuffled in zip(arms["reflection_correct"], arms["reflection_shuffled"]):
        if correct["common_messages"] != shuffled["common_messages"]:
            raise ValueError("correct/shuffled common contexts differ")
        if correct["reflection_question"] != shuffled["reflection_question"]:
            raise ValueError("correct/shuffled reflection questions differ")
        if correct["task_id"] != shuffled["task_id"]:
            raise ValueError("correct/shuffled task pairing differs")
        if correct["target_plan"] != shuffled["target_plan"]:
            raise ValueError("shuffling mutated immutable task truth")
        if correct["supervision_plan"] == shuffled["supervision_plan"]:
            raise ValueError("shuffled reflection target was not changed")

    summary = {
        "schema_version": 1,
        "experiment_id": "qwen35_4b_counterfactual_plan_reflection_transfer",
        "mode": mode,
        "counts": {
            **{split: len(rows) for split, rows in tasks.items()},
            "retention": len(retention),
        },
        "families": validation["families"],
        "unique_task_ids": validation["unique_task_ids"],
        "unique_compositions": validation["unique_compositions"],
        "unique_behavior_signatures": validation["unique_behavior_signatures"],
        "cross_split_composition_collisions": validation["cross_split_composition_collisions"],
        "cross_split_behavior_collisions": validation["cross_split_behavior_collisions"],
        "shuffled_derangement_failures": 0,
        "exact_answer_in_reflection_targets": validation["exact_answer_in_reflection_targets"],
        "retention": retention_validation,
        "corpus_sha256": _digest({"depth_3": tasks, "retention": retention}),
        "arms_sha256": _digest(arms),
        "training_records_sha256": _digest(training_records),
        "training_record_receipt": training_receipt,
        "benchmark_firewall": "python_audit_hook_open_listdir_scandir",
        "model_calls": 0,
        "gpu_events": 0,
        "benchmark_reads": 0,
        "authorized_next_stage": "CPU construction and adversarial design review only",
    }
    if summary["cross_split_composition_collisions"] != 0:
        raise ValueError("cross-split composition collision")
    if summary["cross_split_behavior_collisions"] != 0:
        raise ValueError("cross-split behavior collision")
    if summary["exact_answer_in_reflection_targets"] != 0:
        raise ValueError("reflection contains exact answer")
    return summary


def _config() -> dict[str, Any]:
    with (EXP / "configs" / "default.yaml").open() as handle:
        return yaml.safe_load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--construct", action="store_true")
    args = parser.parse_args()
    if args.smoke == args.construct:
        parser.error("select exactly one of --smoke or --construct")
    config = _config()
    construction = config["construction"]
    counts = (
        {"train": 4, "qualification": 3, "confirmation": 3}
        if args.smoke
        else {
            split: int(construction["per_family"][split])
            for split in ("train", "calibration", "qualification", "confirmation")
        }
    )
    result = construct(
        counts=counts,
        retention_per_family_per_depth=(
            1 if args.smoke else int(construction["per_family"]["retention_per_family_per_depth"])
        ),
        seed=int(construction["seed"]),
        retention_seed=int(construction["retention_seed"]),
        shuffle_seed=int(construction["shuffle_seed"]),
        schedule_seed=int(construction["schedule_seed"]),
        per_family_per_optimizer_group=(
            2
            if args.smoke
            else int(config["training"]["schedule"]["per_family_per_optimizer_group"])
        ),
        mode="model_free_smoke" if args.smoke else "model_free_full_construction",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
