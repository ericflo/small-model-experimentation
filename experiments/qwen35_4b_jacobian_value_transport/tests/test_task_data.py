from __future__ import annotations

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from task_data import (  # noqa: E402
    apply_pipeline,
    build_splits,
    make_positive_controls,
    parse_first_op,
    verify_first_op,
)


def _config() -> dict:
    return {
        "seeds": {"split": 11, "lens_corpus": 12},
        "data": {
            "lens_fit_prompts": 4,
            "positive_control_items": 4,
            "value_calibration_tasks": 4,
            "iid_eval_tasks": 5,
            "hard_eval_tasks": 3,
            "held_family_tasks_per_family": 2,
            "visible_examples": 4,
            "hidden_examples": 3,
            "anchor_depth": 2,
            "hard_depth": 3,
        },
    }


def test_apply_pipeline() -> None:
    assert apply_pipeline([3, -1, 2], [("sort_asc", None), ("add_k", 2)]) == [1, 4, 5]


def test_build_splits_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    a = build_splits(first, _config())
    b = build_splits(second, _config())
    assert a["task_split_digests"] == b["task_split_digests"]
    ids = []
    for split in ("value_calibration", "iid_eval", "hard_eval", "held_string_eval", "held_register_eval"):
        rows = [json.loads(line) for line in (first / f"{split}.jsonl").read_text().splitlines()]
        ids.extend(row["task_id"] for row in rows)
        assert all(len(row["visible"]) == 4 and len(row["hidden"]) == 3 for row in rows)
    assert len(ids) == len(set(ids))
    assert a["firewall"]["benchmark_content_used"] is False


def test_positive_control_and_first_op_parser() -> None:
    import random

    row = make_positive_controls(random.Random(3), 1)[0]
    assert row["source"] != row["target"]
    task = {"first_op": "reverse"}
    assert parse_first_op("notes\nFirst: reverse") == "reverse"
    assert verify_first_op(task, "First: reverse")
    assert not verify_first_op(task, "First: sort_asc")
