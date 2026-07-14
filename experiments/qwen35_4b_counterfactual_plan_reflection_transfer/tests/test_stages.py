from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import stages as S  # noqa: E402


class StageReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = EXP / "configs" / "default.yaml"
        cls.config = yaml.safe_load(cls.config_path.read_text())
        cls.screen = int(cls.config["training"]["staged_seeds"]["screen"])
        cls.replication = int(cls.config["training"]["staged_seeds"]["replication"])

    def receipt(self, stage: str, claims: list[dict]) -> dict:
        return {
            "schema_version": 2,
            "experiment_id": self.config["experiment_id"],
            "authorized_stage": stage,
            "config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            "issuer_git_commit": S.git_commit(),
            "issuer_script_sha256": S.sha256_file(EXP / "scripts" / "authorize_stage.py"),
            "prerequisites": claims,
        }

    @staticmethod
    def decision(seed: int, block: str = "qualification", positive=None) -> dict:
        return {
            "kind": "decision",
            "sha256": "a" * 64,
            "block": block,
            "seed": seed,
            "capability_pass": True,
            "reflection_specific_pass": False,
            "positive_control_pass": positive,
        }

    @staticmethod
    def retention(seed: int) -> dict:
        return {
            "kind": "retention",
            "sha256": "b" * 64,
            "seed": seed,
            "arm": "reflection_correct_action",
            "pass": True,
        }

    def validate(self, stage: str, claims: list[dict]) -> None:
        with mock.patch.object(S, "require_clean_worktree"):
            S.validate_stage_receipt(
                self.receipt(stage, claims),
                config=self.config,
                config_path=self.config_path,
                expected_stage=stage,
            )

    def test_calibration_and_screen_cardinality_are_exact(self) -> None:
        self.validate("calibration_generation", [])
        calibration = {"kind": "calibration_gate", "sha256": "c" * 64, "pass": True}
        self.validate("screen_training", [calibration])
        with self.assertRaisesRegex(ValueError, "calibration prerequisite"):
            self.validate("screen_training", [])

    def test_replication_and_confirmation_require_exact_seed_ancestry(self) -> None:
        screen_decision = self.decision(self.screen, positive=True)
        self.validate(
            "replication_training", [screen_decision, self.retention(self.screen)]
        )
        confirmation = [
            screen_decision,
            self.decision(self.replication),
            self.retention(self.screen),
            self.retention(self.replication),
        ]
        self.validate("confirmation", confirmation)
        with self.assertRaisesRegex(ValueError, "both frozen seeds"):
            self.validate(
                "confirmation",
                [screen_decision, screen_decision, self.retention(self.screen), self.retention(self.replication)],
            )

    def test_final_requires_two_confirmation_decisions(self) -> None:
        claims = [
            self.decision(self.screen, block="confirmation"),
            self.decision(self.replication, block="confirmation"),
        ]
        self.validate("final", claims)
        with self.assertRaisesRegex(ValueError, "cardinality"):
            self.validate("final", claims[:1])


if __name__ == "__main__":
    unittest.main()
