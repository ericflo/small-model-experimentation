from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import gpu_runner  # noqa: E402
from src.config import load_config  # noqa: E402


class DurableLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "default.yaml")

    def setUp(self) -> None:
        self.design_patch = mock.patch.object(
            gpu_runner,
            "design_lineage",
            return_value={
                "path": "synthetic/design_receipt.json",
                "sha256": "1" * 64,
                "receipt_identity_sha256": "2" * 64,
                "status": "DESIGN_FROZEN",
                "phase": "design_boundary",
            },
        )
        self.design_patch.start()

    def tearDown(self) -> None:
        self.design_patch.stop()

    def test_receipt_identity_binds_exact_content(self) -> None:
        phase = "lora_g0"
        receipt = gpu_runner._with_identity(
            {
                "schema_version": 1,
                "status": "MODEL_SMOKE_PASS",
                **gpu_runner._identity(self.config, phase=phase),
                "probe": {"finite": True},
            }
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            reopened = gpu_runner._read_receipt(
                path,
                self.config,
                statuses={"MODEL_SMOKE_PASS"},
                phases={phase},
                label="synthetic G0",
            )
            self.assertEqual(reopened["receipt_identity_sha256"], receipt["receipt_identity_sha256"])

            receipt["probe"]["finite"] = False
            path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                gpu_runner._read_receipt(
                    path,
                    self.config,
                    statuses={"MODEL_SMOKE_PASS"},
                    phases={phase},
                    label="synthetic G0",
                )

    def test_every_preserved_failure_receipt_has_canonical_identity(self) -> None:
        paths = sorted((ROOT / "runs" / "failures").glob("*.json"))
        self.assertGreater(len(paths), 0)
        for path in paths:
            with self.subTest(path=path.name):
                payload = json.loads(path.read_text(encoding="utf-8"))
                claimed = payload.pop("receipt_identity_sha256")
                self.assertEqual(gpu_runner._canonical_sha256(payload), claimed)

    def test_lineage_reopens_file_and_rejects_payload_status_and_phase_tamper(self) -> None:
        # _lineage deliberately permits only repository-relative durable files.
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "gate.json"
            receipt = gpu_runner._with_identity(
                {"status": "MODEL_SMOKE_PASS", "phase": "lora_g0", "value": 1}
            )
            path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            entry = gpu_runner._lineage(path, receipt)
            self.assertEqual(gpu_runner.validate_lineage_entry(entry), receipt)

            changed = dict(receipt)
            changed["value"] = 2
            path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "file changed"):
                gpu_runner.validate_lineage_entry(entry)

            stale_identity_entry = dict(entry, sha256=gpu_runner._sha256(path))
            with self.assertRaisesRegex(RuntimeError, "identity"):
                gpu_runner.validate_lineage_entry(stale_identity_entry)

            changed = gpu_runner._with_identity(
                {"status": "OTHER", "phase": "lora_g0", "value": 1}
            )
            path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
            status_entry = dict(
                entry,
                sha256=gpu_runner._sha256(path),
                receipt_identity_sha256=changed["receipt_identity_sha256"],
            )
            with self.assertRaisesRegex(RuntimeError, "status changed"):
                gpu_runner.validate_lineage_entry(status_entry)

            changed = gpu_runner._with_identity(
                {"status": "MODEL_SMOKE_PASS", "phase": "other_phase", "value": 1}
            )
            path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
            phase_entry = dict(
                entry,
                sha256=gpu_runner._sha256(path),
                receipt_identity_sha256=changed["receipt_identity_sha256"],
            )
            with self.assertRaisesRegex(RuntimeError, "phase changed"):
                gpu_runner.validate_lineage_entry(phase_entry)

    def test_checkpoint_identity_and_payload_hashes_fail_closed(self) -> None:
        metadata = {
            "capacity": "lora",
            "objective": "joint",
            "model_seed": 7411,
            "step": 1500,
            "adaptation_state_sha256": "a" * 64,
            "loop_state_sha256": "b" * 64,
            "g0_lineage": {"receipt_identity_sha256": "c" * 64},
            "positive_control_lineage": {"receipt_identity_sha256": "d" * 64},
        }
        identity = gpu_runner._checkpoint_identity(metadata)
        stored = dict(metadata, checkpoint_identity_sha256=identity)
        self.assertEqual(gpu_runner._checkpoint_identity(stored), identity)
        stored["adaptation_state_sha256"] = "e" * 64
        self.assertNotEqual(gpu_runner._checkpoint_identity(stored), identity)
        stored = dict(metadata, checkpoint_identity_sha256=identity)
        stored["g0_lineage"] = {"receipt_identity_sha256": "f" * 64}
        self.assertNotEqual(gpu_runner._checkpoint_identity(stored), identity)

        source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        self.assertIn('for key in ("g0_lineage", "positive_control_lineage"):', source)
        self.assertIn("validate_lineage_entry(metadata[key])", source)
        self.assertIn('metadata["adaptation_state_sha256"]', source)
        self.assertIn('metadata["loop_state_sha256"]', source)

    def test_analysis_loader_confines_checkpoint_and_row_payload_paths(self) -> None:
        source = (ROOT / "src" / "analysis.py").read_text(encoding="utf-8")
        start = source.index("def _load_evaluation(")
        end = source.index("\ndef _cells(", start)
        loader = source[start:end]
        self.assertIn(
            '_resolve_repo_path(str(summary["checkpoint_path"]))',
            loader,
        )
        self.assertIn('f"rows_{mode}.jsonl"', loader)

        authorization_start = source.index("def _load_analysis_authorization(")
        authorization_end = source.index(
            "\ndef _fullrank_minus_lora_contrast(", authorization_start
        )
        authorization_loader = source[authorization_start:authorization_end]
        self.assertIn("_resolve_repo_path(", authorization_loader)

        runner_source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        receipt_start = runner_source.index("def _read_receipt(")
        receipt_end = runner_source.index("\ndef _lineage(", receipt_start)
        self.assertIn("_resolve_repo_path(", runner_source[receipt_start:receipt_end])


if __name__ == "__main__":
    unittest.main()
