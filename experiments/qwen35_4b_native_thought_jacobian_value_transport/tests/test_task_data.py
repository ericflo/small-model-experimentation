from __future__ import annotations

import random
import sys
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from task_data import (  # noqa: E402
    IDENTIFIABLE_FIRST_OPERATIONS,
    OPERATIONS,
    apply_pipeline,
    build_splits,
    make_task,
    parse_alias,
    task_fingerprint,
    task_prompt,
    verify_answer,
)


def test_operation_semantics() -> None:
    assert apply_pipeline([3, -1, 2], [("reverse", None)]) == [2, -1, 3]
    assert apply_pipeline([3, -1, 2], [("sort_asc", None), ("add_k", 2)]) == [1, 4, 5]
    assert apply_pipeline([3, -1, 2], [("running_sum", None)]) == [3, 2, 4]
    assert apply_pipeline([3, -1, 2], [("adjacent_diff", None)]) == [-4, 3]


def test_generated_task_has_visible_first_type_identifiability() -> None:
    task = make_task(
        random.Random(91),
        task_id="unit-task",
        first_name="reverse",
        depth=2,
        visible=8,
        hidden=8,
    )
    assert task["first_op"] == "reverse"
    assert task["visible_matching_first_types"] == ["reverse"]
    assert len(task["visible"]) == len(task["hidden"]) == 8


def test_splits_are_balanced_unique_and_parseable() -> None:
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    splits = build_splits(config)
    assert {name: len(rows) for name, rows in splits.items()} == {
        "seam_calibration": 16,
        "value_fit": 32,
        "causal_confirmation": 32,
    }
    fingerprints = [
        task_fingerprint(task) for rows in splits.values() for task in rows
    ]
    assert len(fingerprints) == len(set(fingerprints)) == 80
    aliases = config["data"]["operation_aliases"]
    for rows in splits.values():
        counts = {
            name: sum(task["first_op"] == name for task in rows)
            for name in IDENTIFIABLE_FIRST_OPERATIONS
        }
        assert max(counts.values()) - min(counts.values()) <= 1
        assert all(task["first_op"] != "negate" for task in rows)
        for task in rows:
            answer = f"<think>work</think>\nFirst: {aliases[task['first_op']]}"
            assert verify_answer(task, answer, aliases)
            assert parse_alias(answer, aliases) == aliases[task["first_op"]]
            assert "Operation aliases:" in task_prompt(task, aliases)
