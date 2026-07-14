from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "recovery.py"
SPEC = importlib.util.spec_from_file_location("_branch_recovery_tested", SOURCE)
assert SPEC and SPEC.loader
recovery = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recovery
SPEC.loader.exec_module(recovery)


def _reference_failure() -> bytes:
    for path in (recovery.ARCHIVED_FAILURE, recovery.FAILURE_CANONICAL):
        if path.is_file():
            return recovery._regular_bytes(path)
    raise RuntimeError("no preserved branch-authorization failure exists")


class SourceAndSeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = recovery.load_producer_context()

    def test_exact_producer_contract_is_pinned(self) -> None:
        self.assertEqual(
            self.context.cli.source_contract_sha256(recovery.PRODUCER_ROOT),
            recovery.EXPECTED_SOURCE,
        )
        self.assertEqual(
            self.context.config_receipt["config_sha256"],
            recovery.EXPECTED_CONFIG_SHA256,
        )

    def test_raw_prefix_and_clean_descendant_normalize_exactly(self) -> None:
        seam = recovery.ExactRegisteredPrefixSeam(self.context)
        self.assertEqual(seam(Path(seam.raw_prefix)), seam.canonical_prefix)
        self.assertEqual(
            seam(Path(seam.raw_prefix) / "lora_joint_seed7411"),
            seam.canonical_prefix / "lora_joint_seed7411",
        )

    def test_unrelated_alias_and_prefix_traversal_remain_rejected(self) -> None:
        seam = recovery.ExactRegisteredPrefixSeam(self.context)
        with self.assertRaisesRegex(RuntimeError, "lexical-canonical"):
            seam(recovery.PRODUCER_ROOT / "data" / ".." / "data")
        with self.assertRaisesRegex(RuntimeError, "descendant"):
            seam(Path(seam.raw_prefix + "/../state_formation_capacity_adjudication"))

    def test_context_manager_restores_after_exception(self) -> None:
        original = self.context.analysis._canonical_expected_path
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with recovery.installed_path_seam(self.context):
                raise RuntimeError("sentinel")
        self.assertIs(self.context.analysis._canonical_expected_path, original)

    def test_recovery_contract_covers_tests_and_review(self) -> None:
        contract = recovery.recovery_source_contract()
        self.assertEqual(
            [entry["path"] for entry in contract["files"]],
            list(recovery.CONTRACT_FILES),
        )
        self.assertEqual(len(contract["source_contract_sha256"]), 64)


class SafeReceiptTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "duplicate JSON key"):
            recovery._strict_json_bytes(b'{"a":1,"a":2}', label="duplicate")

    def test_publish_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "receipt.json"
            recovery._publish_or_verify(path, b"one\n")
            recovery._publish_or_verify(path, b"one\n")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                recovery._publish_or_verify(path, b"two\n")

    def test_failure_pair_validation_accepts_only_exact_preserved_bytes(self) -> None:
        raw = _reference_failure()
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            canonical = Path(directory) / "canonical.json"
            mirror = Path(directory) / "mirror.json"
            recovery._publish_or_verify(canonical, raw)
            recovery._publish_or_verify(mirror, raw)
            with mock.patch.object(recovery, "FAILURE_CANONICAL", canonical), mock.patch.object(
                recovery, "FAILURE_MIRROR", mirror
            ):
                receipt, reopened = recovery._failure_pair()
            self.assertEqual(reopened, raw)
            self.assertEqual(receipt["receipt_identity_sha256"], recovery.EXPECTED_FAILURE_IDENTITY)

    def test_archive_copies_failure_without_retiring_source_pair(self) -> None:
        raw = _reference_failure()
        smoke = {
            "recovery_source_contract_sha256": "1" * 64,
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            canonical = base / "canonical.json"
            mirror = base / "mirror.json"
            archived = base / "archive.json"
            receipt_path = base / "archive_receipt.json"
            recovery._publish_or_verify(canonical, raw)
            recovery._publish_or_verify(mirror, raw)
            with mock.patch.object(recovery, "FAILURE_CANONICAL", canonical), mock.patch.object(
                recovery, "FAILURE_MIRROR", mirror
            ), mock.patch.object(recovery, "ARCHIVED_FAILURE", archived), mock.patch.object(
                recovery, "ARCHIVE_RECEIPT", receipt_path
            ), mock.patch.object(recovery, "_require_smoke", return_value=(smoke, "2" * 64)):
                result = recovery.archive_failure()
            self.assertEqual(result["status"], "BRANCH_AUTHORIZATION_FAILURE_ARCHIVED")
            self.assertEqual(archived.read_bytes(), raw)
            self.assertTrue(canonical.exists())
            self.assertTrue(mirror.exists())

    def test_retirement_requires_full_commit_identifier(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "full lowercase SHA-1"):
            recovery.retire_failure("deadbeef")

    def test_retirement_preserves_archive_and_removes_only_source_pair(self) -> None:
        raw = _reference_failure()
        smoke = {"recovery_source_contract_sha256": "1" * 64}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            canonical = base / "canonical.json"
            mirror = base / "mirror.json"
            archived = base / "archived.json"
            archive_receipt_path = base / "archive_receipt.json"
            started_path = base / "retirement_started.json"
            retirement_path = base / "retirement.json"
            for path in (canonical, mirror, archived):
                recovery._publish_or_verify(path, raw)
            archive = recovery._with_identity({
                "schema_version": 1,
                "status": "BRANCH_AUTHORIZATION_FAILURE_ARCHIVED",
            })
            recovery._publish_json(archive_receipt_path, archive)
            archive_raw = archive_receipt_path.read_bytes()
            blobs = {
                archived: raw,
                archive_receipt_path: archive_raw,
                canonical: raw,
                mirror: raw,
            }
            with mock.patch.object(recovery, "FAILURE_CANONICAL", canonical), mock.patch.object(
                recovery, "FAILURE_MIRROR", mirror
            ), mock.patch.object(recovery, "ARCHIVED_FAILURE", archived), mock.patch.object(
                recovery, "ARCHIVE_RECEIPT", archive_receipt_path
            ), mock.patch.object(recovery, "RETIREMENT_STARTED", started_path), mock.patch.object(
                recovery, "RETIREMENT_RECEIPT", retirement_path
            ), mock.patch.object(recovery, "_require_smoke", return_value=(smoke, "2" * 64)), mock.patch.object(
                recovery, "_git_blob", side_effect=lambda _commit, path: blobs[path]
            ):
                result = recovery.retire_failure("a" * 40)
            self.assertEqual(result["status"], "BRANCH_AUTHORIZATION_FAILURE_RETIRED")
            self.assertFalse(canonical.exists())
            self.assertFalse(mirror.exists())
            self.assertEqual(archived.read_bytes(), raw)
            self.assertTrue(retirement_path.exists())


class InvocationBoundaryTests(unittest.TestCase):
    def test_invocation_requires_registered_stage_authorization_and_output(self) -> None:
        base = {
            "stage": "model-smoke",
            "capacity": "fullrank",
            "objective": "joint",
            "eval_set": "trigger",
            "seed": 7411,
            "authorization_receipt": None,
            "output": "out.json",
        }
        with self.assertRaisesRegex(RuntimeError, "requires an authorization"):
            recovery.invoke_producer(base, [])
        base["stage"] = "analyze"
        base["authorization_receipt"] = "authorization.json"
        with self.assertRaisesRegex(RuntimeError, "not allowed"):
            recovery.invoke_producer(base, [])

    def test_output_leaf_is_stage_specific(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            base = Path(directory)
            common = {"output": str(base), "stage": "train"}
            self.assertEqual(recovery._producer_output_leaf(common), base / "run.json")
            common["stage"] = "evaluate-state"
            self.assertEqual(recovery._producer_output_leaf(common), base / "summary.json")
            common["stage"] = "model-smoke"
            self.assertEqual(recovery._producer_output_leaf(common), base)

    def test_invocation_shapes_exclude_unregistered_cells(self) -> None:
        setup = {
            "stage": "model-smoke",
            "capacity": "lora",
            "objective": "joint",
            "eval_set": "trigger",
            "initialization_bundle": "init.pt",
            "checkpoint": None,
        }
        with self.assertRaisesRegex(RuntimeError, "fullrank/joint/trigger"):
            recovery._validate_invocation_shape(setup)
        state_only_contrast = {
            "stage": "evaluate-state",
            "capacity": "lora",
            "objective": "state_only",
            "eval_set": "contrast",
            "checkpoint": "checkpoint",
            "initialization_bundle": None,
            "model_smoke_receipt": None,
            "positive_control_receipt": None,
        }
        with self.assertRaisesRegex(RuntimeError, "trigger-only"):
            recovery._validate_invocation_shape(state_only_contrast)


if __name__ == "__main__":
    unittest.main()
