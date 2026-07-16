"""Lineage-package byte-identity and root-omission contracts.

The copied package (manifest, six stage datasets, three trainer variants,
merger) must be BYTE-IDENTICAL to the source cell's committed standalone
package — the whole point of the rebuild is that the recipe is exactly
the recorded one, minus the root. The blend root adapter must NOT be
vendored into this cell: its omission is the design.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(SCRIPTS))

import gen_design_receipt as gd  # noqa: E402
import rebuild_zero_root as rz  # noqa: E402

SOURCE = ROOT / "experiments" / "qwen35_4b_goal_gate_confirmation"
COPIED_FILES = (
    "data/lineage/lineage_manifest.json",
    "data/lineage/stage01_replay_refresh.jsonl",
    "data/lineage/stage02_designed160.jsonl",
    "data/lineage/stage03_close_xi__targeted_standard.jsonl",
    "data/lineage/stage04_replay_after_close.jsonl",
    "data/lineage/stage05_designed_fresh.jsonl",
    "data/lineage/stage06_hygiene_explore.jsonl",
    "scripts/lineage_trainers/train_think_stage12.py",
    "scripts/lineage_trainers/train_think_close_stage3.py",
    "scripts/lineage_trainers/train_think_stage456.py",
    "scripts/merge_adapter.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TestPackageByteIdentity(unittest.TestCase):
    def test_every_copied_file_is_byte_identical_to_the_source(self):
        for relative in COPIED_FILES:
            with self.subTest(file=relative):
                source = SOURCE / relative
                copy = EXP / relative
                self.assertTrue(source.is_file(), f"source missing: {source}")
                self.assertTrue(copy.is_file(), f"copy missing: {copy}")
                self.assertEqual(
                    sha256_file(source),
                    sha256_file(copy),
                    f"copied package file drifted from the source: {relative}",
                )

    def test_manifest_byte_pin_matches_the_frozen_constant(self):
        self.assertEqual(
            sha256_file(EXP / "data" / "lineage" / "lineage_manifest.json"),
            rz.MANIFEST_SHA256,
        )
        self.assertEqual(rz.MANIFEST_SHA256, gd.LINEAGE_MANIFEST_SHA256)

    def test_dataset_pins_match_the_manifest_recipe(self):
        manifest = json.loads(
            (EXP / "data" / "lineage" / "lineage_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for row in manifest["stages"]:
            name = Path(row["dataset"]["file"]).name
            digest, rows = gd.LINEAGE_DATASETS[name]
            self.assertEqual(digest, row["dataset"]["sha256"])
            self.assertEqual(rows, row["dataset"]["rows"])
        for relative, digest in gd.LINEAGE_TRAINERS.items():
            registered = (
                manifest["merger"]["sha256"]
                if relative == "merge_adapter.py"
                else manifest["trainers"][f"scripts/{relative}"]["sha256"]
            )
            self.assertEqual(digest, registered)


class TestVerifyInputs(unittest.TestCase):
    def test_verify_inputs_passes_on_the_real_package(self):
        manifest = rz.load_manifest()
        checked = rz.verify_inputs(manifest)
        self.assertEqual(
            checked,
            {"datasets": 6, "trainers": 3, "merger": 1, "root_vendored": False},
        )

    def test_manifest_authenticates_against_the_source_experiment(self):
        manifest = rz.load_manifest()
        self.assertEqual(manifest["experiment_id"], "qwen35_4b_goal_gate_confirmation")
        self.assertEqual(len(manifest["stages"]), 6)
        self.assertEqual(
            tuple(row["seed"] for row in manifest["stages"]),
            rz.LINEAGE_TRAINING_SEEDS,
        )


class TestRootOmission(unittest.TestCase):
    def test_the_blend_root_is_not_vendored_into_this_cell(self):
        self.assertFalse(
            rz.FORBIDDEN_ROOT_DIR.exists(),
            "the blend root must not be vendored here: training without "
            "it is the design",
        )
        rz.require_root_not_vendored()  # must not raise

    def test_forbidden_dir_constant_agrees_across_modules(self):
        self.assertEqual(rz.FORBIDDEN_ROOT_DIR, gd.FORBIDDEN_ROOT_DIR)

    def test_root_contrast_hashes_match_the_manifest_documentation(self):
        manifest = rz.load_manifest()
        root = manifest["root_adapter"]
        self.assertEqual(
            gd.ORIGINAL_ROOT_CONTRAST["adapter_weights_sha256"],
            root["weights_sha256"],
        )
        self.assertEqual(
            gd.ORIGINAL_ROOT_CONTRAST["adapter_config_sha256"],
            root["config_sha256"],
        )
        self.assertEqual(gd.ORIGINAL_ROOT_CONTRAST["name"], root["name"])

    def test_rebuild_never_references_the_source_vendored_root_path(self):
        text = (SCRIPTS / "rebuild_zero_root.py").read_text(encoding="utf-8")
        self.assertNotIn(
            "lineage_root/blend",
            text,
            "the rebuild script must not reach into the source cell's "
            "vendored root adapter",
        )

    def test_training_seeds_are_the_inherited_stage_constants(self):
        self.assertEqual(rz.LINEAGE_TRAINING_SEEDS, (42, 43, 44, 47, 51, 55))
        self.assertEqual(gd.LINEAGE_TRAINING_SEEDS, rz.LINEAGE_TRAINING_SEEDS)


if __name__ == "__main__":
    unittest.main()
