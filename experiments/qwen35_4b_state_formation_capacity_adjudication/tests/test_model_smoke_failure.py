from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import gpu_runner  # noqa: E402
from src.config import load_config  # noqa: E402


class ModelSmokeFailurePersistenceTests(unittest.TestCase):
    SOURCE = "c" * 64

    def setUp(self) -> None:
        self.config = copy.deepcopy(load_config(ROOT / "configs" / "default.yaml"))
        self.temporary = tempfile.TemporaryDirectory()
        self.experiment = Path(self.temporary.name) / "experiment"
        self.experiment.mkdir()
        self.output = self.experiment / "runs" / "setup" / "g0_lora_seed7411.json"
        self.identity = {
            "experiment_id": self.config["experiment_id"],
            "model_id": gpu_runner.MODEL_ID,
            "model_revision": gpu_runner.MODEL_REVISION,
            "backend": "transformers",
            "config_sha256": "1" * 64,
            "source_contract_sha256": self.SOURCE,
            "requirements_training_lock_sha256": "2" * 64,
            "design_receipt_sha256": "3" * 64,
            "design_receipt_identity_sha256": "4" * 64,
            "phase": "lora_g0",
        }
        self.root_patch = mock.patch.object(gpu_runner, "ROOT", self.experiment)
        self.identity_patch = mock.patch.object(
            gpu_runner, "_identity", return_value=self.identity
        )
        self.config_patch = mock.patch.object(
            gpu_runner, "require_confirmatory_config", return_value=None
        )
        self.design_patch = mock.patch.object(
            gpu_runner, "validate_design_receipt", return_value=None
        )
        for patcher in (
            self.root_patch,
            self.identity_patch,
            self.config_patch,
            self.design_patch,
        ):
            patcher.start()

    def tearDown(self) -> None:
        for patcher in (
            self.design_patch,
            self.config_patch,
            self.identity_patch,
            self.root_patch,
        ):
            patcher.stop()
        self.temporary.cleanup()

    def run_smoke(self) -> None:
        gpu_runner.model_smoke(
            self.config,
            self.output,
            capacity="lora",
            model_seed=7411,
            initialization_bundle=self.experiment / "initialization_seed7411.pt",
            authorization_receipt=None,
        )

    def mirror(self) -> Path:
        return (
            self.experiment
            / "runs"
            / "failures"
            / f"g0_lora_seed7411_source_{self.SOURCE[:12]}.json"
        )

    def assert_valid_pair(self) -> dict:
        self.assertTrue(self.output.is_file())
        self.assertTrue(self.mirror().is_file())
        self.assertEqual(self.output.read_bytes(), self.mirror().read_bytes())
        self.assertNotEqual(self.output.stat().st_ino, self.mirror().stat().st_ino)
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        claimed = payload.pop("receipt_identity_sha256")
        self.assertEqual(gpu_runner._canonical_sha256(payload), claimed)
        return payload

    def test_early_exception_persists_fail_closed_pair_and_reraises(self) -> None:
        with mock.patch.object(
            gpu_runner,
            "_model_smoke_attempt",
            side_effect=RuntimeError("synthetic early setup failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic early setup failure"):
                self.run_smoke()
        payload = self.assert_valid_pair()
        self.assertEqual(payload["status"], "SETUP_CONTROL_FAILED")
        self.assertEqual(payload["phase"], "lora_g0")
        self.assertEqual(payload["failure_stage"], "branch_authorization")
        self.assertIsNone(payload["shared_initialization"])
        self.assertEqual(payload["result_payloads_opened"], [])
        for field in (
            "authorizes_positive_control",
            "authorizes_training",
            "authorizes_result_training",
            "authorizes_result_evaluation",
            "training_or_evaluation_started",
            "scientific_evidence",
        ):
            self.assertFalse(payload[field])

    def test_mid_probe_exception_preserves_structured_gradient_diagnostics(self) -> None:
        gradient_summary = {
            "aggregate_exempt": {
                "tensors": 1,
                "with_gradient": 1,
                "finite": 1,
                "nonzero": 0,
                "items": [
                    {
                        "name": "aggregate_logit",
                        "has_gradient": True,
                        "finite": True,
                        "norm": 0.0,
                    }
                ],
            }
        }
        shared = {"status": "SHARED_INITIALIZATION_PREPARED", "seed": 7411}

        def fail_mid(*args, failure_state: dict, **kwargs):
            failure_state.update(
                {
                    "failure_stage": "live_joint_backward_probe",
                    "completed_checks": list(gpu_runner.G0_COMPLETED_CHECKS[:8]),
                    "data_manifest_sha256": "5" * 64,
                    "shared_initialization": shared,
                    "setup": {"shared_initialization": shared},
                    "result_payloads_opened": ["train"],
                    "live_joint_gradient_summary": gradient_summary,
                    "live_joint_dropout_probe": {"calls": 186, "cycles": 3},
                }
            )
            raise RuntimeError("live joint G0 gradient reachability failed")

        with mock.patch.object(gpu_runner, "_model_smoke_attempt", side_effect=fail_mid):
            with self.assertRaisesRegex(RuntimeError, "gradient reachability failed"):
                self.run_smoke()
        payload = self.assert_valid_pair()
        self.assertEqual(payload["live_joint_gradient_summary"], gradient_summary)
        self.assertEqual(payload["live_joint_dropout_probe"], {"calls": 186, "cycles": 3})
        self.assertEqual(payload["shared_initialization"], shared)
        self.assertEqual(payload["result_payloads_opened"], ["train"])

    def test_empty_exception_message_still_persists_nonempty_error(self) -> None:
        with mock.patch.object(
            gpu_runner,
            "_model_smoke_attempt",
            side_effect=RuntimeError(),
        ):
            with self.assertRaises(RuntimeError):
                self.run_smoke()
        payload = self.assert_valid_pair()
        self.assertEqual(payload["error_type"], "RuntimeError")
        self.assertTrue(payload["error"])

    def test_success_leaf_race_is_rejected_without_overwrite(self) -> None:
        def race_leaf(*args, **kwargs):
            self.output.write_bytes(b"racing intruder")
            return {"status": "MODEL_SMOKE_PASS"}

        with mock.patch.object(
            gpu_runner,
            "_model_smoke_attempt",
            side_effect=race_leaf,
        ):
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite receipt"):
                self.run_smoke()
        self.assertEqual(self.output.read_bytes(), b"racing intruder")
        self.assertFalse(self.mirror().exists())

    def test_existing_mirror_blocks_attempt_without_overwrite(self) -> None:
        self.mirror().parent.mkdir(parents=True)
        self.mirror().write_bytes(b"preserved")
        with mock.patch.object(gpu_runner, "_model_smoke_attempt") as attempt:
            with self.assertRaisesRegex(RuntimeError, "existing failure mirror"):
                self.run_smoke()
        attempt.assert_not_called()
        self.assertEqual(self.mirror().read_bytes(), b"preserved")
        self.assertFalse(self.output.exists())

    def test_existing_canonical_blocks_attempt_without_overwrite(self) -> None:
        self.output.parent.mkdir(parents=True)
        self.output.write_bytes(b"preserved canonical")
        with mock.patch.object(gpu_runner, "_model_smoke_attempt") as attempt:
            with self.assertRaisesRegex(RuntimeError, "refusing to resume or overwrite"):
                self.run_smoke()
        attempt.assert_not_called()
        self.assertEqual(self.output.read_bytes(), b"preserved canonical")
        self.assertFalse(self.mirror().exists())

    def test_second_link_failure_can_leave_mirror_but_never_canonical_only(self) -> None:
        canonical = self.experiment / "pair" / "canonical.json"
        mirror = self.experiment / "pair" / "mirror.json"
        real_link = os.link
        calls = 0

        def fail_second(source, destination, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic canonical-link failure")
            return real_link(source, destination, **kwargs)

        with mock.patch.object(gpu_runner.os, "link", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "canonical-link failure"):
                gpu_runner._write_new_json_pair(canonical, mirror, {"value": 1})
        self.assertTrue(mirror.is_file())
        self.assertFalse(canonical.exists())
        self.assertEqual(json.loads(mirror.read_text(encoding="utf-8")), {"value": 1})
        self.assertEqual(list(canonical.parent.glob(".*.tmp-*")), [])

    def test_successful_pair_has_independent_inodes_and_tamper_isolated(self) -> None:
        canonical = self.experiment / "pair" / "canonical.json"
        mirror = self.experiment / "pair" / "mirror.json"
        gpu_runner._write_new_json_pair(canonical, mirror, {"value": 1})
        expected = mirror.read_bytes()
        self.assertEqual(canonical.read_bytes(), expected)
        self.assertNotEqual(canonical.stat().st_ino, mirror.stat().st_ino)
        canonical.write_bytes(b"tampered canonical")
        self.assertEqual(mirror.read_bytes(), expected)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink creation is unavailable")
    def test_symlinked_runs_ancestor_blocks_before_attempt(self) -> None:
        target = self.experiment / "runs-target"
        target.mkdir()
        (self.experiment / "runs").symlink_to(target, target_is_directory=True)
        with mock.patch.object(gpu_runner, "_model_smoke_attempt") as attempt:
            with self.assertRaisesRegex(RuntimeError, "symlinked ancestor"):
                self.run_smoke()
        attempt.assert_not_called()
        self.assertEqual(list(target.rglob("*")), [])

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink creation is unavailable")
    def test_broken_canonical_symlink_blocks_without_following_it(self) -> None:
        self.output.parent.mkdir(parents=True)
        missing = self.output.parent / "missing-target.json"
        self.output.symlink_to(missing)
        with mock.patch.object(gpu_runner, "_model_smoke_attempt") as attempt:
            with self.assertRaisesRegex(RuntimeError, "refusing to resume or overwrite"):
                self.run_smoke()
        attempt.assert_not_called()
        self.assertTrue(self.output.is_symlink())
        self.assertFalse(missing.exists())


class FailedReceiptAuthorizationTests(unittest.TestCase):
    def test_read_receipt_rejects_failure_even_when_explicitly_allowlisted(self) -> None:
        config = load_config(ROOT / "configs" / "default.yaml")
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "failure.json"
            payload = gpu_runner._with_identity(
                {
                    "schema_version": 1,
                    "status": "SETUP_CONTROL_FAILED",
                    **gpu_runner._identity(config, phase="lora_g0"),
                }
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "did not authorize"):
                gpu_runner._read_receipt(
                    path,
                    config,
                    statuses={"SETUP_CONTROL_FAILED"},
                    phases={"lora_g0"},
                    label="synthetic failed G0",
                )

    def test_g0_pass_requires_exact_access_and_authorization_contract(self) -> None:
        config = load_config(ROOT / "configs" / "default.yaml")
        expected = {
            "authorizes_positive_control": True,
            "authorizes_training": False,
            "authorizes_result_training": False,
            "authorizes_result_evaluation": False,
            "benchmark_files_read": 0,
            "result_payloads_opened": ["train"],
            "sealed_contrast_payloads_opened": [],
            "training_or_evaluation_started": False,
            "scientific_evidence": False,
        }
        with mock.patch.object(gpu_runner, "_read_receipt", return_value=expected):
            self.assertEqual(
                gpu_runner._read_g0_pass(Path("unused"), config, capacity="lora"),
                expected,
            )
        for field in expected:
            with self.subTest(field=field):
                tampered = copy.deepcopy(expected)
                tampered.pop(field)
                with mock.patch.object(
                    gpu_runner, "_read_receipt", return_value=tampered
                ):
                    with self.assertRaisesRegex(RuntimeError, field):
                        gpu_runner._read_g0_pass(
                            Path("unused"), config, capacity="lora"
                        )

    def test_g0_progress_marker_enforces_the_registered_order(self) -> None:
        state = {"completed_checks": []}
        gpu_runner._mark_g0_complete(state, "branch_authorization")
        self.assertEqual(state["completed_checks"], ["branch_authorization"])
        with self.assertRaisesRegex(RuntimeError, "check order changed"):
            gpu_runner._mark_g0_complete(state, "shared_initialization")
        state["completed_checks"] = ["shared_initialization"]
        with self.assertRaisesRegex(RuntimeError, "ordered prefix"):
            gpu_runner._mark_g0_complete(state, "train_only_data_manifest")

    def test_live_joint_progress_is_marked_after_receipt_construction(self) -> None:
        source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        stage = source.index(
            'failure_state["failure_stage"] = "live_joint_optimizer_step"'
        )
        optimizer_step = source.index("optimizer.step()", stage)
        joint_probe = source.index("joint_probe = {", optimizer_step)
        final_clip_scale = source.index(
            'float(config["training"]["common_gradient_clip"]),', joint_probe
        )
        mark = source.index(
            '_mark_g0_complete(failure_state, "live_joint_backward_and_optimizer_probe")',
            joint_probe,
        )
        next_stage = source.index(
            'failure_state["failure_stage"] = "optimizer_state_receipt"', mark
        )
        self.assertLess(optimizer_step, joint_probe)
        self.assertLess(final_clip_scale, mark)
        self.assertLess(mark, next_stage)


if __name__ == "__main__":
    unittest.main()
