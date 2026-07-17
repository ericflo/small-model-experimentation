"""The standalone lineage package: manifest byte pin, stage-8 block, verify.

The complete in-cell reproduction package (stages 1-6 zero-root datasets,
both stage-7 arm streams, the stage-8 replay pool, trainers, merger,
wrappers, provenance receipts) must authenticate against the extended
manifest via ``rebuild_lineage.py --verify-inputs``, and every tamper of
the manifest must refuse.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rebuild_lineage as rl  # noqa: E402


class TestLineagePackage(unittest.TestCase):
    def test_verify_inputs_passes_end_to_end(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(SCRIPTS / "rebuild_lineage.py"),
                "--verify-inputs",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["checked"]["datasets"], 7)
        self.assertEqual(payload["checked"]["arm_streams"], 2)
        self.assertEqual(payload["checked"]["stage8_dataset"], 1)
        self.assertEqual(payload["checked"]["provenance_receipts"], 7)
        self.assertFalse(payload["root_adapter_vendored"])

    def test_manifest_bytes_match_the_pin(self):
        self.assertEqual(rl.sha256_file(rl.MANIFEST), rl.MANIFEST_SHA256)

    def test_manifest_loads_and_validates(self):
        manifest = rl.load_manifest()
        self.assertEqual(len(manifest["stages"]), 7)
        self.assertIn("stage8_replay_compound", manifest)

    def test_stage8_block_carries_the_frozen_design(self):
        manifest = rl.load_manifest()
        stage8 = manifest["stage8_replay_compound"]
        self.assertEqual(stage8["arm"], "replay_compound")
        self.assertEqual(stage8["training_seed"], 86)
        self.assertEqual(stage8["dataset"]["rows"], 2240)
        self.assertEqual(
            stage8["dataset"]["sha256"],
            "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
        )
        self.assertEqual(
            stage8["parent_composite"]["tree_sha256"],
            rl.ARM_COMPOSITE_PINS["count_walk"]["tree_sha256"],
        )
        self.assertEqual(
            stage8["parent_composite"]["weights_sha256"],
            rl.ARM_COMPOSITE_PINS["count_walk"]["weights_sha256"],
        )
        self.assertEqual(stage8["recipe"]["hyperparameters"], rl.ARM_HYPERPARAMETERS)
        self.assertEqual(stage8["recipe"]["optimizer_steps"], 280)
        self.assertEqual(
            stage8["recipe"]["warm_start"],
            "fresh_adapter_on_count_walk_composite_via_model_path",
        )
        # Composite pins are TODO slots: null pre-merge, 64-hex post-merge.
        for key in ("tree_sha256", "weights_sha256", "committed_merge_receipt_sha256"):
            value = stage8["composite"][key]
            self.assertTrue(value is None or rl.SHA_RE.fullmatch(value))

    def test_tampered_manifest_refuses(self):
        original_pin = rl.MANIFEST_SHA256
        try:
            rl.MANIFEST_SHA256 = "0" * 64
            with self.assertRaises(ValueError):
                rl.load_manifest()
        finally:
            rl.MANIFEST_SHA256 = original_pin

    def test_stage8_seed_tamper_refuses(self):
        manifest = rl.load_manifest()
        tampered = json.loads(json.dumps(manifest))
        tampered["stage8_replay_compound"]["training_seed"] = 85
        with self.assertRaises(ValueError):
            rl.validate_stage8(tampered, tampered["stages"][6])

    def test_stage8_dataset_tamper_refuses(self):
        manifest = rl.load_manifest()
        tampered = json.loads(json.dumps(manifest))
        tampered["stage8_replay_compound"]["dataset"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            rl.validate_stage8(tampered, tampered["stages"][6])

    def test_stage8_parent_pin_tamper_refuses(self):
        manifest = rl.load_manifest()
        tampered = json.loads(json.dumps(manifest))
        tampered["stage8_replay_compound"]["parent_composite"]["weights_sha256"] = (
            "0" * 64
        )
        with self.assertRaises(ValueError):
            rl.validate_stage8(tampered, tampered["stages"][6])

    def test_stage8_recipe_tamper_refuses(self):
        manifest = rl.load_manifest()
        tampered = json.loads(json.dumps(manifest))
        tampered["stage8_replay_compound"]["recipe"]["hyperparameters"]["lr"] = 2e-4
        with self.assertRaises(ValueError):
            rl.validate_stage8(tampered, tampered["stages"][6])

    def test_manifest_never_references_a_root_adapter(self):
        text = rl.MANIFEST.read_text(encoding="utf-8")
        self.assertNotIn("root_adapter", text)

    def test_stage7_wrapper_copies_are_byte_identical_to_lifecycle_28(self):
        source = ROOT / "experiments" / "qwen35_4b_count_walk_menders_confirmation"
        for name in ("train_trial.py", "merge_trained_arm.py"):
            with self.subTest(wrapper=name):
                self.assertEqual(
                    (SCRIPTS / "stage7_wrappers" / name).read_bytes(),
                    (source / "scripts" / name).read_bytes(),
                )

    def test_copied_production_scripts_match_their_manifest_pins(self):
        manifest = rl.load_manifest()
        self.assertEqual(
            rl.sha256_file(EXP / "scripts" / "train_think.py"),
            manifest["stages"][6]["trainer_sha256"],
        )
        self.assertEqual(
            rl.sha256_file(EXP / "scripts" / "merge_adapter.py"),
            manifest["merger"]["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
