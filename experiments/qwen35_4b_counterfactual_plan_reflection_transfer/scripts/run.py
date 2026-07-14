#!/usr/bin/env python3
"""Model-free construction smoke for counterfactual plan reflection transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from taskgen import build_corpus, build_reflection_arms, validate_corpus  # noqa: E402


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def smoke() -> dict[str, object]:
    counts = {"train": 4, "qualification": 3, "confirmation": 3}
    tasks = build_corpus(counts=counts, seed=73_301)
    validation = validate_corpus(tasks, counts)
    arms = build_reflection_arms(tasks["train"], seed=73_319)

    for correct, shuffled in zip(arms["reflection_correct"], arms["reflection_shuffled"]):
        assert correct["common_messages"] == shuffled["common_messages"]
        assert correct["reflection_question"] == shuffled["reflection_question"]
        assert correct["task_id"] == shuffled["task_id"]
        assert correct["target_plan"] != shuffled["target_plan"]
        assert correct["target_ops"] != shuffled["target_ops"]

    summary = {
        "schema_version": 1,
        "experiment_id": "qwen35_4b_counterfactual_plan_reflection_transfer",
        "mode": "model_free_smoke",
        "counts": {split: len(rows) for split, rows in tasks.items()},
        "families": validation["families"],
        "unique_task_ids": validation["unique_task_ids"],
        "unique_compositions": validation["unique_compositions"],
        "unique_behavior_signatures": validation["unique_behavior_signatures"],
        "cross_split_composition_collisions": validation["cross_split_composition_collisions"],
        "cross_split_behavior_collisions": validation["cross_split_behavior_collisions"],
        "shuffled_derangement_failures": 0,
        "exact_answer_in_reflection_targets": validation["exact_answer_in_reflection_targets"],
        "corpus_sha256": _digest(tasks),
        "arms_sha256": _digest(arms),
        "model_calls": 0,
        "gpu_events": 0,
        "benchmark_reads": 0,
        "authorized_next_stage": "CPU construction and adversarial design review only",
    }
    assert summary["cross_split_composition_collisions"] == 0
    assert summary["cross_split_behavior_collisions"] == 0
    assert summary["exact_answer_in_reflection_targets"] == 0
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("only the model-free smoke is authorized before design review")
    print(json.dumps(smoke(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
