from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import analyze as A  # noqa: E402


BOOTSTRAP = {"paired_task_resamples": 500, "seed": 88991}
THRESHOLDS = {
    "correct_minus_shuffled_min": 0.10,
    "correct_minus_frozen_min": 0.10,
    "paired_delta_lower_95_gt": 0.0,
    "each_family_correct_minus_shuffled_min": 0.05,
    "each_family_correct_minus_frozen_min": 0.05,
    "strict_parse_rate_min": 0.95,
    "answer_limit_contact_max": 0.01,
    "periodic_loop_contact_max": 0.01,
    "reflection_specific_correct_minus_auxiliary_min": 0.05,
}


def rows(correct: int, shuffled: int, frozen: int, auxiliary: int) -> list[dict]:
    output = []
    family_names = ("list", "string", "register")
    for index in range(60):
        task_id = f"t{index:03d}"
        family = family_names[index % 3]
        for arm, cutoff in (
            ("reflection_correct_action", correct),
            ("reflection_shuffled_action", shuffled),
            ("frozen_action", frozen),
            ("auxiliary_plan_label_correct_action", auxiliary),
        ):
            output.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "arm": arm,
                    "coverage_at_16": int(index < cutoff),
                    "strict_parse_rate": 1.0,
                    "answer_limit_contact": 0.0,
                    "periodic_loop_contact": 0.0,
                }
            )
    return output


class AnalyzeTests(unittest.TestCase):
    def test_calibration_requires_parse_headroom_and_nonbinding_budgets(self) -> None:
        data = rows(30, 20, 20, 25)
        result = A.evaluate_calibration(
            data,
            {
                "base_coverage_at_16_min": 0.05,
                "base_coverage_at_16_max": 0.80,
                "strict_parse_rate_min": 0.95,
                "answer_limit_contact_max": 0.01,
                "periodic_loop_contact_max": 0.01,
            },
        )
        self.assertTrue(result["pass"])

    def test_large_broad_effect_passes_capability_and_mechanism(self) -> None:
        result = A.evaluate_seed_block(rows(50, 25, 20, 35), THRESHOLDS, BOOTSTRAP)
        self.assertTrue(result["capability_pass"])
        self.assertTrue(result["reflection_specific_pass"])

    def test_auxiliary_tie_preserves_capability_but_rejects_reflection_claim(self) -> None:
        result = A.evaluate_seed_block(rows(50, 25, 20, 50), THRESHOLDS, BOOTSTRAP)
        self.assertTrue(result["capability_pass"])
        self.assertFalse(result["reflection_specific_pass"])

    def test_one_family_miss_cannot_be_rescued_by_pooling(self) -> None:
        data = rows(50, 25, 20, 35)
        for row in data:
            if row["family"] == "register" and row["arm"] == "reflection_correct_action":
                row["coverage_at_16"] = next(
                    other["coverage_at_16"]
                    for other in data
                    if other["task_id"] == row["task_id"]
                    and other["arm"] == "reflection_shuffled_action"
                )
        result = A.evaluate_seed_block(data, THRESHOLDS, BOOTSTRAP)
        self.assertFalse(result["capability_pass"])
        self.assertFalse(result["capability_checks"]["each_family_correct_minus_shuffled"])

    def test_pairing_mismatch_fails_closed(self) -> None:
        data = rows(50, 25, 20, 35)
        data.pop()
        with self.assertRaisesRegex(ValueError, "task pairing differs"):
            A.evaluate_seed_block(data, THRESHOLDS, BOOTSTRAP)


if __name__ == "__main__":
    unittest.main()
