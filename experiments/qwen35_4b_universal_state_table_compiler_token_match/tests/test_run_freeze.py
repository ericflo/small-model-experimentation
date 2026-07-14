from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run.py"
SPEC = importlib.util.spec_from_file_location("state_table_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(label: str, kind: str, correct: bool) -> dict:
    return {"adapter": label, "kind": kind, "correct": correct}


class RunFreezeTests(unittest.TestCase):
    def test_model_data_and_seed_identities_are_frozen(self) -> None:
        self.assertEqual(MODULE.CANDIDATE, "state_table_after_close")
        self.assertEqual(MODULE.CONTROL, "replay_after_close")
        self.assertEqual(MODULE.EXPECTED_FORWARD_TOKENS, 286814)
        self.assertEqual(MODULE.EXPECTED_ROWS, 320)
        self.assertEqual(
            MODULE.EXPECTED_RECEIPT_SHA256,
            "163e40a61d0b3f4dc541f56ea32510bacb8ce64f658e00f47e5867da4a45f0b8",
        )
        self.assertEqual(
            MODULE.EXPECTED_DESIGN_RECEIPT_SHA256,
            "0bac3340c1995beb1cff1ea9c3563849ff9f024e3ff6c836894f7f22d50ef837",
        )
        source = SCRIPT.read_text(encoding="utf-8")
        for seed in ("46", "88008", "78138"):
            self.assertIn(seed, source)

    def test_candidate_must_strictly_beat_both_controls_overall_and_on_target(self) -> None:
        payload = {
            "rows": [
                row(MODULE.PARENT_LABEL, "u_execute", True),
                row(MODULE.PARENT_LABEL, "u_count", True),
                row(MODULE.PARENT_LABEL, "u_probe", False),
                row(MODULE.CONTROL, "u_induct", True),
                row(MODULE.CONTROL, "u_count", True),
                row(MODULE.CONTROL, "u_probe", False),
                row(MODULE.CANDIDATE, "u_execute", True),
                row(MODULE.CANDIDATE, "u_probe", True),
                row(MODULE.CANDIDATE, "u_count", True),
            ]
        }
        checks = MODULE.relative_promotion_checks(payload)
        self.assertTrue(all(checks.values()))
        payload["rows"][-2]["correct"] = False
        checks = MODULE.relative_promotion_checks(payload)
        self.assertFalse(checks["beats_parent_target_correct"])
        self.assertFalse(checks["beats_replay_target_correct"])

    def test_expensive_stages_require_the_adversarial_review_verdict(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("PASS_EXPENSIVE_RUN", source)
        self.assertIn("design_review.md", source)
        self.assertIn("benchmark remains sealed", source)

    def test_harness_forbids_multi_stage_runs_and_requires_committed_receipts(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn('"all"', source)
        self.assertNotIn('"train",', source)
        self.assertIn("require_clean_committed_checkpoint", source)
        self.assertIn('f"{CONTROL}.json"', source)
        self.assertIn("PROMOTION_RECEIPT", source)


if __name__ == "__main__":
    unittest.main()
