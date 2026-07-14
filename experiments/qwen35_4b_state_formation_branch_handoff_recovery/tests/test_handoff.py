from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "handoff.py"
SPEC = importlib.util.spec_from_file_location("_branch_handoff_tested", SOURCE)
assert SPEC and SPEC.loader
handoff = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = handoff
SPEC.loader.exec_module(handoff)


class ContractAndLineageTests(unittest.TestCase):
    def test_contract_covers_review_config_cli_source_and_tests(self) -> None:
        contract = handoff.handoff_source_contract()
        self.assertEqual(
            [entry["path"] for entry in contract["files"]],
            list(handoff.CONTRACT_FILES),
        )
        self.assertEqual(len(contract["source_contract_sha256"]), 64)

    def test_first_recovery_and_successful_g0_are_exact(self) -> None:
        self.assertEqual(
            handoff.first._sha256(handoff.FIRST_SOURCE),
            handoff.EXPECTED_FIRST_SOURCE_FILE,
        )
        lineage = handoff.validate_handoff()
        self.assertEqual(
            lineage["successful_g0_receipt_identity_sha256"],
            handoff.EXPECTED_G0_IDENTITY,
        )

    def test_original_wrapper_false_rejection_is_pathname_only(self) -> None:
        arguments = {
            "stage": "positive-control",
            "capacity": "fullrank",
            "objective": "joint",
            "eval_set": "trigger",
            "seed": 7411,
            "checkpoint": None,
            "initialization_bundle": "init.pt",
            "model_smoke_receipt": "g0.json",
            "positive_control_receipt": None,
            "authorization_receipt": "authorization.json",
            "output": "positive.json",
        }
        with self.assertRaisesRegex(RuntimeError, "failed G0 pair"):
            handoff.first.invoke_producer(arguments, [])


class SlotClassificationTests(unittest.TestCase):
    def test_exact_failed_bytes_cannot_reoccupy_successful_slot(self) -> None:
        failure = handoff.first._regular_bytes(handoff.ARCHIVED_FAILURE)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            slot = Path(directory) / "g0.json"
            handoff.first._publish_or_verify(slot, failure)
            with mock.patch.object(handoff, "SUCCESSFUL_G0", slot):
                with self.assertRaisesRegex(RuntimeError, "retired failure bytes reoccupied"):
                    handoff.validate_handoff()

    def test_any_changed_successful_g0_bytes_are_rejected(self) -> None:
        success = handoff.first._regular_bytes(handoff.SUCCESSFUL_G0)
        changed = success.replace(
            b'"authorizes_positive_control": true',
            b'"authorizes_positive_control": false',
        )
        self.assertNotEqual(success, changed)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            slot = Path(directory) / "g0.json"
            handoff.first._publish_or_verify(slot, changed)
            with mock.patch.object(handoff, "SUCCESSFUL_G0", slot):
                with self.assertRaises(RuntimeError):
                    handoff.validate_handoff()

    def test_reappearing_retired_mirror_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            mirror = Path(directory) / "mirror.json"
            handoff.first._publish_or_verify(mirror, b"sentinel\n")
            with mock.patch.object(handoff, "RETIRED_FAILURE_MIRROR", mirror):
                with self.assertRaisesRegex(RuntimeError, "mirror reappeared"):
                    handoff.validate_handoff()


class InvocationBoundaryTests(unittest.TestCase):
    def test_requires_authorization_and_explicit_output(self) -> None:
        arguments = {
            "stage": "model-smoke",
            "capacity": "fullrank",
            "objective": "joint",
            "eval_set": "trigger",
            "seed": 7412,
            "initialization_bundle": "init.pt",
            "checkpoint": None,
            "authorization_receipt": None,
            "output": "g0.json",
        }
        with self.assertRaisesRegex(RuntimeError, "requires an authorization"):
            handoff.invoke_producer(arguments, [])
        arguments["authorization_receipt"] = "authorization.json"
        arguments["stage"] = "analyze"
        with self.assertRaisesRegex(RuntimeError, "not allowed"):
            handoff.invoke_producer(arguments, [])

    def test_unregistered_setup_shape_stays_rejected(self) -> None:
        arguments = {
            "stage": "model-smoke",
            "capacity": "lora",
            "objective": "joint",
            "eval_set": "trigger",
            "seed": 7412,
            "initialization_bundle": "init.pt",
            "checkpoint": None,
            "authorization_receipt": "authorization.json",
            "output": "g0.json",
        }
        with self.assertRaisesRegex(RuntimeError, "fullrank/joint/trigger"):
            handoff.invoke_producer(arguments, [])


if __name__ == "__main__":
    unittest.main()
