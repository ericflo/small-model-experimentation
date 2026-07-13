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


class MechanicsLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_calibration_inputs()
        cls.decision = {
            "decision": "CALIBRATION_INTERFACE_SELECTED",
            "winner": "think512_freeform",
            "fixed_priority": list(
                cls.inputs.config["interface"]["fixed_winner_priority"]
            ),
            "qualification": {
                arm: arm == "think512_freeform"
                for arm in cls.inputs.config["interface"]["fixed_winner_priority"]
            },
            "selection_uses_metric_ranking": False,
        }

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


if __name__ == "__main__":
    unittest.main()
