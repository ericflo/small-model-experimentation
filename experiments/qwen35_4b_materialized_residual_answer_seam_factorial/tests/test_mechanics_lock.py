from __future__ import annotations

import copy
import json
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
from transactions import MODEL_ID, MODEL_REVISION, json_bytes  # noqa: E402


def _complete_calibration_decision(inputs):
    arms = list(inputs.config["interface"]["fixed_winner_priority"])
    metrics = {}
    for index, arm in enumerate(arms):
        successes = 48 if index == 0 else 0
        metrics[arm] = {
            "rows": 48,
            "exact_echo_successes": successes,
            "parse_successes": successes,
            "answer_cap_contacts": 0,
            "thinking_cap_contacts": 0,
            "arity_counts": {"2": 24, "3": 24},
            "by_arity": {
                str(arity): {
                    "rows": 24,
                    "exact_echo_successes": successes // 2,
                    "parse_successes": successes // 2,
                    "answer_cap_contacts": 0,
                    "thinking_cap_contacts": 0,
                }
                for arity in (2, 3)
            },
        }
    return {
        "schema_version": 1,
        "stage": "interface_calibration",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "decision": "CALIBRATION_INTERFACE_SELECTED",
        "winner": arms[0],
        "fixed_priority": arms,
        "qualification": {arm: index == 0 for index, arm in enumerate(arms)},
        "selection_uses_metric_ranking": False,
        "metrics": metrics,
        "scored_rows_sha256": {arm: "a" * 64 for arm in arms},
        "pairing": {},
        "transaction_chain": {},
        "calibration_read_receipt": {},
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
        "implementation_lock_sha256": "b" * 64,
        "live_preflight_sha256": "c" * 64,
    }


class MechanicsLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_calibration_inputs()
        cls.decision = _complete_calibration_decision(cls.inputs)

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

    def test_lock_binds_calibration_winner_and_preoutcome_mechanics_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration.json"
            decision = root / "decision.json"
            calibration.write_text("{}\n")
            decision.write_text(json.dumps(self.decision) + "\n")
            frozen = {
                name: "b" * 40 for name in calibration_lock.FROZEN_MECHANICS_FILES
            }
            calibration_value = {
                "implementation_commit": "a" * 40,
                "frozen_mechanics_blobs": frozen,
            }
            commit = "c" * 40
            with mock.patch.object(mechanics_lock, "CALIBRATION_LOCK", calibration), mock.patch.object(
                mechanics_lock, "CALIBRATION_DECISION", decision
            ):
                value = mechanics_lock.build_mechanics_lock_value(
                    calibration_lock=calibration_value,
                    calibration_decision=self.decision,
                    calibration_decision_sha256=mechanics_lock.sha256_file(decision),
                    authorization_commit=commit,
                    authorization_ci=self.ci(commit),
                    inputs=self.inputs,
                )
                round_trip = json.loads(json.dumps(value, sort_keys=True))
                self.assertEqual(
                    mechanics_lock.validate_mechanics_lock_value(
                        round_trip,
                        calibration_lock=calibration_value,
                        calibration_decision=self.decision,
                        inputs=self.inputs,
                    ),
                    round_trip,
                )
                self.assertEqual(
                    set(value["frozen_mechanics_blobs"]),
                    set(calibration_lock.FROZEN_MECHANICS_FILES),
                )
                self.assertNotEqual(
                    value["sampling"]["direct"]["run_seed"],
                    value["sampling"]["suffix_materialized"]["run_seed"],
                )
                mutated = copy.deepcopy(value)
                mutated["selected_interface"] = "no_think_freeform"
                with self.assertRaisesRegex(RuntimeError, "boundary"):
                    mechanics_lock.validate_mechanics_lock_value(
                        mutated,
                        calibration_lock=calibration_value,
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
