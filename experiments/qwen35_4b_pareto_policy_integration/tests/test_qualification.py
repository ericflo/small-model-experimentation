from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

from analyze_qualification import paired_advantage  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
