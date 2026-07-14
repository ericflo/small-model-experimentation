from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import calibration_lock  # noqa: E402
import mechanics_lock  # noqa: E402
from calibration_stage import load_calibration_inputs  # noqa: E402
from transactions import json_bytes, read_canonical  # noqa: E402


class MechanicsLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_calibration_inputs()
        cls.decision = read_canonical(EXP / "runs/calibration/decision.json")

    @staticmethod
    def ci(commit: str) -> dict[str, dict[str, object]]:
        return {
            workflow: {
                "database_id": index + 1,
                "head_sha": commit,
                "status": "completed",
                "conclusion": "success",
                "url": f"https://github.com/example/actions/{index + 1}",
            }
            for index, workflow in enumerate(calibration_lock.REQUIRED_WORKFLOWS)
        }

    def review(self, commit: str) -> dict[str, object]:
        return {
            "schema_version": 1,
            "verdict": "PASS_IMPLEMENTATION",
            "reviewed_commit": commit,
            "reviewer": "fresh-test-reviewer",
            "review_report_sha256": "a" * 64,
            "reviewed_ci": self.ci(commit),
            "adversarial_review_rounds": 1,
            "allowed_tests_passed": 99,
            "allowed_tests_total": 99,
            "experimental_model_requests_reviewed": 0,
            "sampled_model_outputs_reviewed": 0,
            "hidden_files_read": [],
            "qualification_files_read": [],
            "confirmation_files_read": [],
            "benchmark_files_read": [],
        }

    def test_lock_binds_reviewed_runtime_winner_and_frozen_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "calibration.json"
            decision_path = root / "decision.json"
            calibration_path.write_text("{}\n")
            decision_path.write_bytes(json_bytes(self.decision))
            frozen = {
                name: "b" * 40 for name in calibration_lock.FROZEN_MECHANICS_FILES
            }
            calibration = {
                "implementation_commit": "1" * 40,
                "frozen_mechanics_blobs": frozen,
            }
            implementation = "2" * 40
            release = "3" * 40
            critical = {
                name: "c" * 64 for name in mechanics_lock.MECHANICS_CRITICAL_FILES
            }
            with mock.patch.object(
                mechanics_lock, "CALIBRATION_LOCK", calibration_path
            ), mock.patch.object(
                mechanics_lock, "CALIBRATION_DECISION", decision_path
            ):
                value = mechanics_lock.build_mechanics_lock_value(
                    calibration_lock=calibration,
                    calibration_decision=self.decision,
                    calibration_decision_sha256=mechanics_lock.sha256_file(
                        decision_path
                    ),
                    implementation_review=self.review(implementation),
                    critical_files=critical,
                    release_commit=release,
                    release_ci=self.ci(release),
                    inputs=self.inputs,
                )
                self.assertEqual(
                    mechanics_lock.validate_mechanics_lock_value(
                        value,
                        calibration_lock=calibration,
                        calibration_decision=self.decision,
                        inputs=self.inputs,
                    ),
                    value,
                )
                self.assertEqual(
                    value["selected_interface"],
                    "tokenizer_eos_no_think_program_slot",
                )
                self.assertEqual(value["implementation_commit"], implementation)
                self.assertEqual(value["release_commit"], release)
                self.assertNotEqual(
                    value["sampling"]["direct"]["run_seed"],
                    value["sampling"]["suffix_materialized"]["run_seed"],
                )
                mutated = copy.deepcopy(value)
                mutated["experimental_generation_requests_before_lock"] = False
                with self.assertRaisesRegex(RuntimeError, "boundary"):
                    mechanics_lock.validate_mechanics_lock_value(
                        mutated,
                        calibration_lock=calibration,
                        calibration_decision=self.decision,
                        inputs=self.inputs,
                    )

    def test_hidden_authorization_exact_compares_visible_reanalysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "visible.json"
            calibration_decision = root / "calibration_decision.json"
            visible = {
                "decision": "MECHANICS_VISIBLE_SELECTION_FROZEN",
                "selector_uses_hidden": False,
                "hidden_files_read": [],
                "benchmark_files_read": [],
                "tasks": {},
            }
            path.write_bytes(json_bytes(visible))
            calibration_decision.write_bytes(json_bytes(self.decision))
            forged_reanalysis = {**visible, "tasks": {"forged": {}}}
            with mock.patch.object(
                mechanics_lock, "verify_mechanics_lock"
            ), mock.patch.object(
                mechanics_lock, "load_calibration_inputs", return_value=self.inputs
            ), mock.patch.object(
                mechanics_lock, "CALIBRATION_DECISION", calibration_decision
            ), mock.patch.object(
                mechanics_lock, "load_analysis_tokenizer", return_value=object()
            ), mock.patch.object(
                mechanics_lock, "analyze_visible", return_value=forged_reanalysis
            ):
                with self.assertRaisesRegex(RuntimeError, "exact visible analysis"):
                    mechanics_lock.authorize_hidden_read(path)


if __name__ == "__main__":
    unittest.main()
