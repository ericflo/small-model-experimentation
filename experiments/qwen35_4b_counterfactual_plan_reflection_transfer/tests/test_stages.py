from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import gate_artifacts  # noqa: E402
import stages as S  # noqa: E402


class StageReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = EXP / "configs" / "default.yaml"
        cls.config = yaml.safe_load(cls.config_path.read_text())
        cls.screen = int(cls.config["training"]["staged_seeds"]["screen"])
        cls.replication = int(cls.config["training"]["staged_seeds"]["replication"])
        cls.commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.counter = 0
        self.git_patcher = mock.patch.object(S, "git_commit", return_value=self.commit)
        self.git_patcher.start()

    def tearDown(self) -> None:
        self.git_patcher.stop()
        self.temporary.cleanup()

    def receipt(self, stage: str, claims: list[dict]) -> dict:
        return {
            "schema_version": 3,
            "experiment_id": self.config["experiment_id"],
            "authorized_stage": stage,
            "config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
            "issuer_git_commit": self.commit,
            "issuer_script_sha256": S.sha256_file(EXP / "scripts" / "authorize_stage.py"),
            "prerequisites": claims,
        }

    def claim(self, kind: str, value: dict) -> dict:
        self.counter += 1
        path = self.root / f"{self.counter}-{kind}.json"
        path.write_text(json.dumps(value, sort_keys=True))
        return {"kind": kind, "path": str(path), "sha256": S.sha256_file(path)}

    @staticmethod
    def decision(seed: int, block: str = "qualification", positive=None) -> dict:
        value = {
            "block": block,
            "seed": seed,
            "capability": {"capability_pass": True, "reflection_specific_pass": False},
        }
        if positive is not None:
            value["positive_control"] = {"pass": positive}
        return value

    @staticmethod
    def retention(seed: int) -> dict:
        return {
            "seed": seed,
            "arm": "reflection_correct_action",
            "gate": {"pass": True},
        }

    @staticmethod
    def matched(screen: int, replication: int, passed: bool = True) -> dict:
        return {
            "gate": {
                "pass": passed,
                "budget_pass": passed,
                "by_seed": {
                    str(screen): {"pass": passed},
                    str(replication): {"pass": passed},
                },
            }
        }

    @staticmethod
    def _replay_fixture(path: Path, **_kwargs) -> dict:
        return json.loads(path.read_text())

    def validate(self, stage: str, claims: list[dict]) -> None:
        with (
            mock.patch.object(S, "require_clean_worktree"),
            mock.patch.object(
                gate_artifacts, "validate_gate_artifact", side_effect=self._replay_fixture
            ),
        ):
            S.validate_stage_receipt(
                self.receipt(stage, claims),
                config=self.config,
                config_path=self.config_path,
                expected_stage=stage,
            )

    def test_calibration_and_screen_cardinality_are_exact(self) -> None:
        self.validate("calibration_generation", [])
        calibration = self.claim("calibration_gate", {"gate": {"pass": True}})
        self.validate("screen_training", [calibration])
        with self.assertRaisesRegex(ValueError, "calibration prerequisite"):
            self.validate("screen_training", [])

    def test_replication_and_confirmation_require_exact_seed_ancestry(self) -> None:
        screen_decision = self.claim(
            "decision", self.decision(self.screen, positive=True)
        )
        self.validate(
            "replication_training",
            [screen_decision, self.claim("retention", self.retention(self.screen))],
        )
        confirmation = [
            screen_decision,
            self.claim("decision", self.decision(self.replication)),
            self.claim("retention", self.retention(self.screen)),
            self.claim("retention", self.retention(self.replication)),
        ]
        self.validate("confirmation", confirmation)
        with self.assertRaisesRegex(ValueError, "both frozen seeds"):
            self.validate(
                "confirmation",
                [
                    screen_decision,
                    screen_decision,
                    self.claim("retention", self.retention(self.screen)),
                    self.claim("retention", self.retention(self.replication)),
                ],
            )

    def test_final_requires_two_confirmation_decisions(self) -> None:
        claims = [
            self.claim("decision", self.decision(self.screen, block="confirmation")),
            self.claim("decision", self.decision(self.replication, block="confirmation")),
            self.claim(
                "matched_compute", self.matched(self.screen, self.replication)
            ),
        ]
        self.validate("final", claims)
        with self.assertRaisesRegex(ValueError, "cardinality"):
            self.validate("final", claims[:1])
        failed = [
            *claims[:2],
            self.claim(
                "matched_compute",
                self.matched(self.screen, self.replication, passed=False),
            ),
        ]
        with self.assertRaisesRegex(ValueError, "matched-compute"):
            self.validate("final", failed)

    def test_fabricated_or_changed_prerequisite_hash_cannot_authorize(self) -> None:
        nonexistent = {
            "kind": "calibration_gate",
            "path": str((self.root / "missing.json").resolve()),
            "sha256": "0" * 64,
        }
        with self.assertRaisesRegex(ValueError, "absent or differs"):
            self.validate("screen_training", [nonexistent])
        claim = self.claim("calibration_gate", {"gate": {"pass": True}})
        Path(claim["path"]).write_text('{"gate":{"pass":false}}')
        with self.assertRaisesRegex(ValueError, "absent or differs"):
            self.validate("screen_training", [claim])

    def test_self_consistent_minimal_gate_json_cannot_authorize(self) -> None:
        claim = self.claim(
            "calibration_gate",
            {
                "schema_version": 2,
                "experiment_id": self.config["experiment_id"],
                "config_sha256": hashlib.sha256(self.config_path.read_bytes()).hexdigest(),
                "gate": {"pass": True},
            },
        )
        with (
            mock.patch.object(S, "require_clean_worktree"),
            self.assertRaisesRegex(ValueError, "invocation"),
        ):
            S.validate_stage_receipt(
                self.receipt("screen_training", [claim]),
                config=self.config,
                config_path=self.config_path,
                expected_stage="screen_training",
            )


if __name__ == "__main__":
    unittest.main()
