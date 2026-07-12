from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

from analyze_confirmation import _comparison, _payload_scores, _visible_router  # noqa: E402


def payload(tag: str, block: int, quick: float, deep: float) -> dict:
    items = []
    for index in range(4):
        items.append(
            {
                "key": f"q-{block}-{index}",
                "family": "caravan",
                "kind": "atom",
                "level": 1,
                "stratum": "quick",
                "score": quick,
            }
        )
        items.append(
            {
                "key": f"d-{block}-{index}",
                "family": "caravan",
                "kind": "episode",
                "level": 3,
                "stratum": "deep",
                "score": deep,
            }
        )
    return {"tag": tag, "block_seed": 100 + block, "items": items}


class ConfirmationMetricTests(unittest.TestCase):
    def test_equal_weight_joint_macro(self):
        scores = _payload_scores(payload("x", 0, quick=0.8, deep=0.2))
        self.assertEqual(scores, {"quick": 0.8, "deep": 0.2, "joint": 0.5})

    def test_replicated_positive_delta_has_positive_lcb(self):
        primary = [payload("p", block, 0.7, 0.6) for block in range(2)]
        control = [payload("c", block, 0.6, 0.5) for block in range(2)]
        result = _comparison(
            primary, control, samples=500, confidence=0.95, seed=1
        )
        self.assertTrue(result["all_block_joint_means_positive"])
        self.assertGreater(result["one_sided_joint_lcb"], 0.0)
        self.assertAlmostEqual(result["pooled_macro_delta"]["joint"], 0.1)

    def test_visible_router_uses_quick_only_on_quick_stratum(self):
        quick = payload("quick", 0, quick=0.9, deep=0.1)
        deep = payload("deep", 0, quick=0.2, deep=0.8)
        routed = _visible_router(quick, deep, "visible")
        scores = _payload_scores(routed)
        self.assertEqual(scores["quick"], 0.9)
        self.assertEqual(scores["deep"], 0.8)
        self.assertAlmostEqual(scores["joint"], 0.85)


if __name__ == "__main__":
    unittest.main()
