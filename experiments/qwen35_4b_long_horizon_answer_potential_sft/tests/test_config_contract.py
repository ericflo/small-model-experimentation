from __future__ import annotations

import unittest
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]


class ConfigContractTests(unittest.TestCase):
    def test_planned_candidate_count_and_no_512_cap(self) -> None:
        config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
        sampling = config["sampling"]
        self.assertEqual(135 * sampling["calibration_n"], 8640)
        self.assertEqual(
            135 * sampling["calibration_n"]
            + 1080 * sampling["train_independent_n"]
            + 1080 * sampling["train_branch_n"],
            95040,
        )
        self.assertGreaterEqual(sampling["natural_close_allowance"], 12288)
        self.assertEqual(config["sft"]["max_length"], 16000)

    def test_complete_seed42_matrix_is_registered(self) -> None:
        config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
        self.assertEqual(
            config["sft"]["arms"],
            [
                "empty",
                "random_natural",
                "success_rft",
                "answer_potential",
                "joint_potential",
                "potential_shuffle",
            ],
        )


if __name__ == "__main__":
    unittest.main()
