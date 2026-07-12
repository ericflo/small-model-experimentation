from __future__ import annotations

import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task_data import CONCEPTS, fingerprint, generate_replication_splits  # noqa: E402


def config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))


def test_replication_splits_are_deterministic_balanced_and_disjoint() -> None:
    first = generate_replication_splits(config())
    assert first == generate_replication_splits(config())
    assert {name: len(rows) for name, rows in first.items()} == {
        "control_calibration": 24,
        "confirmation": 48,
    }
    seen = set()
    for rows in first.values():
        counts = {concept: sum(row["source"] == concept for row in rows) for concept in CONCEPTS}
        assert len(set(counts.values())) == 1
        for row in rows:
            assert fingerprint(row) not in seen
            seen.add(fingerprint(row))
