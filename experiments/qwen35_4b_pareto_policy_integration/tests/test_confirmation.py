from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

from analyze_confirmation import _comparison, _routed  # noqa: E402


def payload(offset: float, *, quick_value: float = 0.5, deep_value: float = 0.5) -> dict:
    items = []
    for stratum, level, value in (
        ("quick", 1, quick_value), ("deep", 3, deep_value)
    ):
        for index in range(24):
            items.append({
                "key": f"{stratum}-{index}", "family": f"{stratum}_family",
                "kind": "atom", "level": level, "stratum": stratum,
                "score": value + offset,
            })
    return {"items": items}


class ConfirmationTests(unittest.TestCase):
    def test_tiny_replicated_joint_gain_has_positive_bound(self) -> None:
        cells = {
            "quick/quick_family/atom/L1", "deep/deep_family/atom/L3",
        }
        result = _comparison(
            [payload(0.000001), payload(0.000001)],
            [payload(0.0), payload(0.0)],
            capability_cells=cells, samples=1000, confidence=0.95, seed=1,
        )
        self.assertGreater(result["paired_joint_macro_delta"], 0.0)
        self.assertGreater(result["one_sided_lcb"], 0.0)

    def test_routed_reference_uses_quick_and_deep_strata(self) -> None:
        quick = payload(0.0, quick_value=0.9, deep_value=0.1)
        deep = payload(0.0, quick_value=0.2, deep_value=0.8)
        routed = _routed(quick, deep)
        values = {
            row["stratum"]: row["score"] for row in routed["items"]
        }
        self.assertEqual(values["quick"], 0.9)
        self.assertEqual(values["deep"], 0.8)


if __name__ == "__main__":
    unittest.main()
