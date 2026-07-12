from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
from task_data import (  # noqa: E402
    IDENTIFIABLE_FIRST_OPERATIONS,
    build_splits,
    matching_depth_one,
    matching_first_types,
    task_fingerprint,
)


def config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def test_fresh_exact_depth_splits() -> None:
    splits = build_splits(config())
    assert {name: len(rows) for name, rows in splits.items()} == {
        "seam_selection": 16, "seam_confirmation": 16,
        "value_fit": 32, "causal_confirmation": 32,
    }
    fingerprints = [task_fingerprint(row) for rows in splits.values() for row in rows]
    assert len(fingerprints) == len(set(fingerprints)) == 96
    for rows in splits.values():
        counts = Counter(row["first_op"] for row in rows)
        support = [counts[name] for name in IDENTIFIABLE_FIRST_OPERATIONS]
        assert max(support) - min(support) <= 1
        for row in rows:
            inputs = [item["input"] for item in row["visible"]]
            outputs = tuple(tuple(item["output"]) for item in row["visible"])
            assert matching_first_types(inputs, outputs) == {row["first_op"]}
            assert matching_depth_one(inputs, outputs) == []
