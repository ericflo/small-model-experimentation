from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402


def load_archiver():
    spec = importlib.util.spec_from_file_location(
        "capacity_failed_attempt_archiver", ROOT / "scripts" / "archive_failed_attempt.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed-attempt archiver cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FailedAttemptArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_archiver()
        self.config = copy.deepcopy(load_config(ROOT / "configs" / "default.yaml"))
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        self.experiment = (
            self.repo / "experiments" / "qwen35_4b_state_formation_capacity_adjudication"
        )
        self.experiment.mkdir(parents=True)
        (self.repo / "requirements-training.lock.txt").write_text(
            "synthetic-lock\n", encoding="utf-8"
        )
        self.root_patch = mock.patch.object(self.module, "ROOT", self.experiment)
        self.repo_patch = mock.patch.object(self.module, "REPO_ROOT", self.repo)
        self.design_validation = mock.patch.object(
            self.module, "validate_design_receipt", return_value={"status": "DESIGN_FROZEN"}
        )
        self.design_lineage = mock.patch.object(
            self.module,
            "design_lineage",
            return_value={
                "path": "synthetic/design_receipt.json",
                "sha256": "1" * 64,
                "receipt_identity_sha256": "2" * 64,
                "status": "DESIGN_FROZEN",
                "phase": "design_boundary",
            },
        )
        self.source_digest = mock.patch.object(
            self.module, "source_contract_sha256", return_value="3" * 64
        )
        for patcher in (
            self.root_patch,
            self.repo_patch,
            self.design_validation,
            self.design_lineage,
            self.source_digest,
        ):
            patcher.start()

    def tearDown(self) -> None:
        for patcher in (
            self.source_digest,
            self.design_lineage,
            self.design_validation,
            self.repo_patch,
            self.root_patch,
        ):
            patcher.stop()
        self.temporary.cleanup()

    def large_cell(self, cell: str = "lora_joint_seed7411") -> Path:
        return self.repo / "large_artifacts" / self.config["experiment_id"] / cell

    def tracked_cell(self, cell: str = "lora_joint_seed7411") -> Path:
        return self.experiment / "runs" / "training" / cell

    @staticmethod
    def populate(path: Path, label: str) -> None:
        (path / "nested").mkdir(parents=True)
        (path / "root.txt").write_text(f"root-{label}\n", encoding="utf-8")
        (path / "nested" / "payload.bin").write_bytes(f"payload-{label}".encode())

    def test_allowlist_is_exact_and_two_paths_must_be_same_cell_companions(self) -> None:
        allowed = self.module._allowed_paths(self.config)
        self.assertEqual(len(allowed), 42)
        self.assertIn(self.large_cell(), allowed)
        self.assertIn(self.tracked_cell(), allowed)
        self.assertIn(
            self.experiment / "runs" / "lora_joint_seed7411_contrast", allowed
        )
        self.assertNotIn(self.experiment / "runs" / "state_bag_seed7411_trigger", allowed)

        unregistered = self.experiment / "runs" / "unregistered"
        self.populate(unregistered, "wrong")
        with self.assertRaisesRegex(RuntimeError, "noncanonical or unregistered"):
            self.module.archive_failed_attempt(self.config, [unregistered])

        first = self.large_cell("lora_joint_seed7411")
        unrelated = self.tracked_cell("fullrank_state_only_seed7413")
        self.populate(first, "first")
        self.populate(unrelated, "unrelated")
        with self.assertRaisesRegex(RuntimeError, "same-cell.*companion"):
            self.module.archive_failed_attempt(self.config, [first, unrelated])

    def test_archive_preserves_exact_tree_hashes_and_tracked_receipt(self) -> None:
        primary = self.large_cell()
        companion = self.tracked_cell()
        self.populate(primary, "primary")
        self.populate(companion, "companion")
        expected = [self.module._manifest(primary), self.module._manifest(companion)]

        receipt = self.module.archive_failed_attempt(
            self.config, [primary, companion]
        )
        self.assertFalse(primary.exists())
        self.assertFalse(companion.exists())
        self.assertEqual(receipt["status"], "FAILED_ATTEMPT_ARCHIVED")
        self.assertEqual(receipt["attempts"], expected)
        self.assertEqual(
            receipt["attempt_identity_sha256"],
            self.module._canonical_sha256({"attempts": expected}),
        )
        identity_payload = {
            key: value
            for key, value in receipt.items()
            if key != "receipt_identity_sha256"
        }
        self.assertEqual(
            receipt["receipt_identity_sha256"],
            self.module._canonical_sha256(identity_payload),
        )

        archive = self.repo / receipt["archive_path"]
        self.assertTrue((archive / f"source_1_{primary.name}" / "root.txt").is_file())
        self.assertTrue((archive / f"source_2_{companion.name}" / "root.txt").is_file())
        archived_receipt = json.loads(
            (archive / "archive_receipt.json").read_text(encoding="utf-8")
        )
        tracked_receipts = list((self.experiment / "runs" / "failures").glob("*.json"))
        self.assertEqual(len(tracked_receipts), 1)
        self.assertEqual(archived_receipt, receipt)
        self.assertEqual(
            json.loads(tracked_receipts[0].read_text(encoding="utf-8")), receipt
        )

    def test_archive_refuses_overwrite_of_the_same_content_identity(self) -> None:
        primary = self.large_cell()
        self.populate(primary, "same")
        self.module.archive_failed_attempt(self.config, [primary])
        self.populate(primary, "same")
        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            self.module.archive_failed_attempt(self.config, [primary])

    def test_archive_rejects_symlinks_without_moving_source(self) -> None:
        primary = self.large_cell()
        primary.mkdir(parents=True)
        target = primary / "real.txt"
        target.write_text("bytes", encoding="utf-8")
        (primary / "link.txt").symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, "prohibited symlink"):
            self.module.archive_failed_attempt(self.config, [primary])
        self.assertTrue(primary.is_dir())

    def test_completed_evaluation_cannot_be_archived_as_a_failed_attempt(self) -> None:
        output = self.experiment / "runs" / "lora_joint_seed7411_contrast"
        self.populate(output, "complete")
        (output / "summary.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "completed evaluation"):
            self.module.archive_failed_attempt(self.config, [output])
        self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
