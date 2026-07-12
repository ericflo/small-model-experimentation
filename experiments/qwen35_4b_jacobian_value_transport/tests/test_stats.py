from __future__ import annotations

import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from stats import binary_auc, paired_bootstrap_difference, task_macro_auc  # noqa: E402


def test_binary_auc_ties_and_perfect_order() -> None:
    assert binary_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert binary_auc([0, 1], [0.5, 0.5]) == 0.5


def test_task_macro_auc_excludes_pure_tasks() -> None:
    rows = [
        {"task_id": "a", "label": 0, "score": 0.1},
        {"task_id": "a", "label": 1, "score": 0.9},
        {"task_id": "b", "label": 1, "score": 0.2},
        {"task_id": "b", "label": 1, "score": 0.3},
    ]
    value, count = task_macro_auc(rows, label_key="label", score_key="score")
    assert value == 1.0 and count == 1


def test_paired_bootstrap_is_deterministic() -> None:
    left = [1.0, 1.0, 0.0, 1.0]
    right = [0.0, 0.0, 0.0, 0.0]
    first = paired_bootstrap_difference(left, right, resamples=200, seed=7)
    second = paired_bootstrap_difference(left, right, resamples=200, seed=7)
    assert first == second
    assert first["mean"] == 0.75
