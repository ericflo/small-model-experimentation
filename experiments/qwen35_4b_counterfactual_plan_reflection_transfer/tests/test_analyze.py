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
                    "depth": 3,
                    "arm": arm,
                    "coverage_at_16": int(index < cutoff),
                    "strict_parse_rate": 1.0,
                    "answer_limit_contact": 0.0,
                    "periodic_loop_contact": 0.0,
                    "runtime_protocol_sha256": "same-protocol",
                }
            )
    return output


def expected(data: list[dict]) -> dict[str, tuple[str, int]]:
    return {
        row["task_id"]: (row["family"], int(row["depth"])) for row in data
    }


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
            expected(data),
        )
        self.assertTrue(result["pass"])

    def test_large_broad_effect_passes_capability_and_mechanism(self) -> None:
        data = rows(50, 25, 20, 35)
        result = A.evaluate_seed_block(data, THRESHOLDS, BOOTSTRAP, expected(data))
        self.assertTrue(result["capability_pass"])
        self.assertTrue(result["reflection_specific_pass"])

    def test_auxiliary_tie_preserves_capability_but_rejects_reflection_claim(self) -> None:
        data = rows(50, 25, 20, 50)
        result = A.evaluate_seed_block(data, THRESHOLDS, BOOTSTRAP, expected(data))
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
        result = A.evaluate_seed_block(data, THRESHOLDS, BOOTSTRAP, expected(data))
        self.assertFalse(result["capability_pass"])
        self.assertFalse(result["capability_checks"]["each_family_correct_minus_shuffled"])

    def test_pairing_mismatch_fails_closed(self) -> None:
        data = rows(50, 25, 20, 35)
        data.pop()
        with self.assertRaisesRegex(ValueError, "sealed evaluation split"):
            A.evaluate_seed_block(
                data, THRESHOLDS, BOOTSTRAP, expected(rows(50, 25, 20, 35))
            )

    def test_incomplete_exact_task_set_fails_closed(self) -> None:
        data = rows(50, 25, 20, 35)
        task_metadata = expected(data)
        data = [row for row in data if row["task_id"] != "t059"]
        with self.assertRaisesRegex(ValueError, "sealed evaluation split"):
            A.evaluate_seed_block(data, THRESHOLDS, BOOTSTRAP, task_metadata)

    def test_retention_requires_exact_ids_all_families_and_both_depths(self) -> None:
        data = []
        task_metadata = {}
        families = {"list", "string", "register"}
        for depth in (1, 2):
            for family in sorted(families):
                task_id = f"r-{depth}-{family}"
                task_metadata[task_id] = (family, depth)
                for arm, coverage in (
                    ("frozen_action", 0.0),
                    ("reflection_correct_action", 1.0),
                ):
                    data.append(
                        {
                            "task_id": task_id,
                            "family": family,
                            "depth": depth,
                            "arm": arm,
                            "coverage_at_16": coverage,
                            "runtime_protocol_sha256": "same-protocol",
                        }
                    )
        result = A.evaluate_retention(
            data,
            "reflection_correct_action",
            depth_min=-0.05,
            family_min=-0.125,
            expected_task_metadata=task_metadata,
        )
        self.assertTrue(result["pass"])
        incomplete = [row for row in data if row["depth"] == 1]
        with self.assertRaisesRegex(ValueError, "sealed retention"):
            A.evaluate_retention(
                incomplete,
                "reflection_correct_action",
                -0.05,
                -0.125,
                task_metadata,
            )

    def test_exact_task_mapping_and_runtime_protocol_are_required(self) -> None:
        data = rows(50, 25, 20, 35)
        mapping = expected(data)
        data[0]["family"] = "register"
        with self.assertRaisesRegex(ValueError, "task metadata"):
            A.evaluate_seed_block(data, THRESHOLDS, BOOTSTRAP, mapping)

        data = rows(50, 25, 20, 35)
        data[0]["runtime_protocol_sha256"] = "different-protocol"
        with self.assertRaisesRegex(ValueError, "runtime protocol"):
            A.evaluate_seed_block(data, THRESHOLDS, BOOTSTRAP, expected(data))


if __name__ == "__main__":
    unittest.main()
