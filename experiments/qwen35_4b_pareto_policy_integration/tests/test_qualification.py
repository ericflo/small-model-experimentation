from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

from analyze_qualification import paired_advantage, retention_check  # noqa: E402


def payload(scores: list[float], stratum: str = "quick") -> dict:
    return {
        "items": [
            {
                "key": f"item-{index}", "family": "family", "kind": "atom",
                "level": 1, "stratum": stratum, "score": score,
            }
            for index, score in enumerate(scores)
        ]
    }


class QualificationTests(unittest.TestCase):
    def test_arbitrarily_small_consistent_positive_delta_can_pass(self) -> None:
        result = paired_advantage(
            payload([0.500001] * 32), payload([0.5] * 32), stratum="quick",
            samples=1000, confidence=0.95, seed=1,
        )
        self.assertGreater(result["paired_macro_delta"], 0.0)
        self.assertGreater(result["one_sided_lcb"], 0.0)

    def test_zero_delta_does_not_pass(self) -> None:
        result = paired_advantage(
            payload([0.5] * 32), payload([0.5] * 32), stratum="quick",
            samples=1000, confidence=0.95, seed=1,
        )
        self.assertFalse(result["positive_mean"])
        self.assertFalse(result["positive_lcb"])

    def test_unstable_positive_mean_does_not_get_positive_lower_bound(self) -> None:
        preferred = [1.0] * 17 + [0.0] * 15
        comparator = [0.5] * 32
        result = paired_advantage(
            payload(preferred), payload(comparator), stratum="quick",
            samples=5000, confidence=0.95, seed=2,
        )
        self.assertGreater(result["paired_macro_delta"], 0.0)
        self.assertLessEqual(result["one_sided_lcb"], 0.0)

    def test_retention_anchor_equality_passes_but_excess_regression_fails(self) -> None:
        cell = {"quick/family/atom/L1"}
        equal = retention_check(
            payload([0.99] * 16), payload([0.99] * 16), stratum="quick",
            retention_cells=cell, transfer_families=set(),
            maximum_anchor_regression=0.02, maximum_transfer_regression=0.02,
        )
        regressed = retention_check(
            payload([0.969] * 16), payload([0.99] * 16), stratum="quick",
            retention_cells=cell, transfer_families=set(),
            maximum_anchor_regression=0.02, maximum_transfer_regression=0.02,
        )
        self.assertTrue(equal["passed"])
        self.assertFalse(regressed["passed"])

    def test_retention_cells_do_not_enter_positive_advantage(self) -> None:
        preferred = payload([0.500001] * 32)
        comparator = payload([0.5] * 32)
        for index in range(32):
            preferred["items"].append({
                "key": f"anchor-{index}", "family": "anchor", "kind": "atom",
                "level": 1, "stratum": "quick", "score": 0.98,
            })
            comparator["items"].append({
                "key": f"anchor-{index}", "family": "anchor", "kind": "atom",
                "level": 1, "stratum": "quick", "score": 0.99,
            })
        result = paired_advantage(
            preferred, comparator, stratum="quick", samples=1000,
            confidence=0.95, seed=3,
            allowed_cells={"quick/family/atom/L1"},
        )
        self.assertGreater(result["one_sided_lcb"], 0.0)
        retention = retention_check(
            preferred, comparator, stratum="quick",
            retention_cells={"quick/anchor/atom/L1"}, transfer_families=set(),
            maximum_anchor_regression=0.02, maximum_transfer_regression=0.02,
        )
        self.assertTrue(retention["passed"])


if __name__ == "__main__":
    unittest.main()
