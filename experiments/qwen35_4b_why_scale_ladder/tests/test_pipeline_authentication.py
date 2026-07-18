"""Fail-closed base authentication + the frozen ladder training contract.

The base_reserialized composite is authenticated by tree + weights + the in-cell
provenance copy; tampering any of them must abort. The per-rung epoch schedule,
the adapter-config contract, the published-rung gate, and the ladder-manifest sha
pinning are exercised without GPU (the 9 GB weights hash is a train-stage-only
preflight).
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import build_ladder as bl  # noqa: E402
import gen_why_scale_curriculum as gen  # noqa: E402
import train_trial as tt  # noqa: E402

MANIFEST = EXP / "data" / "ladder_manifest.json"


def write_composite(dir_path: Path) -> None:
    for name in tt.MERGED_FILE_NAMES:
        (dir_path / name).write_text(f"content-of-{name}\n", encoding="utf-8")


class TestBaseProvenanceFailClosed(unittest.TestCase):
    def test_real_in_cell_provenance_passes(self):
        tt.check_base_provenance()

    def test_provenance_copy_sha_is_pinned(self):
        self.assertEqual(tt.sha256_file(tt.BASE_PROVENANCE_COPY), tt.MODEL_PATH_RECEIPT_SHA256)

    def test_tampered_weights_sha_in_provenance_aborts(self):
        payload = json.loads(tt.BASE_PROVENANCE_COPY.read_text())
        payload["weight_files"][0]["sha256"] = "0" * 64
        self._assert_bad_provenance(payload)

    def test_tampered_method_in_provenance_aborts(self):
        payload = json.loads(tt.BASE_PROVENANCE_COPY.read_text())
        payload["method"] = "something_else"
        self._assert_bad_provenance(payload)

    def test_tampered_tokenizer_sha_in_provenance_aborts(self):
        payload = json.loads(tt.BASE_PROVENANCE_COPY.read_text())
        payload["tokenizer_sha256"] = "d" * 64
        self._assert_bad_provenance(payload)

    def _assert_bad_provenance(self, payload: dict) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            bad = Path(scratch) / "prov.json"
            bad.write_text(json.dumps(payload), encoding="utf-8")
            original = tt.BASE_PROVENANCE_COPY
            try:
                tt.BASE_PROVENANCE_COPY = bad
                with self.assertRaises(ValueError):
                    tt.check_base_provenance()
            finally:
                tt.BASE_PROVENANCE_COPY = original

    def test_absent_provenance_aborts(self):
        original = tt.BASE_PROVENANCE_COPY
        try:
            tt.BASE_PROVENANCE_COPY = EXP / "data" / "provenance" / "does_not_exist.json"
            with self.assertRaises(ValueError):
                tt.check_base_provenance()
        finally:
            tt.BASE_PROVENANCE_COPY = original


class TestTreeManifest(unittest.TestCase):
    def test_manifest_over_correct_file_set_is_deterministic(self):
        with tempfile.TemporaryDirectory() as scratch:
            comp = Path(scratch) / "composite"
            comp.mkdir()
            write_composite(comp)
            m1 = tt.tree_manifest_sha256(tt.merged_tree_manifest(comp))
            m2 = tt.tree_manifest_sha256(tt.merged_tree_manifest(comp))
            self.assertEqual(m1, m2)

    def test_extra_file_aborts(self):
        with tempfile.TemporaryDirectory() as scratch:
            comp = Path(scratch) / "composite"
            comp.mkdir()
            write_composite(comp)
            (comp / "surprise.bin").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                tt.merged_tree_manifest(comp)

    def test_missing_file_aborts(self):
        with tempfile.TemporaryDirectory() as scratch:
            comp = Path(scratch) / "composite"
            comp.mkdir()
            write_composite(comp)
            (comp / "config.json").unlink()
            with self.assertRaises(ValueError):
                tt.merged_tree_manifest(comp)

    def test_symlink_child_aborts(self):
        with tempfile.TemporaryDirectory() as scratch:
            comp = Path(scratch) / "composite"
            comp.mkdir()
            write_composite(comp)
            (comp / "config.json").unlink()
            (comp / "config.json").symlink_to(comp / "generation_config.json")
            with self.assertRaises(ValueError):
                tt.merged_tree_manifest(comp)

    def test_frozen_file_names_match_per_file_sha_keys(self):
        self.assertEqual(set(tt.MERGED_FILE_SHA256), set(tt.MERGED_FILE_NAMES))


class TestEpochScheduleAndRecipe(unittest.TestCase):
    def test_epoch_schedule_is_size_appropriate(self):
        self.assertEqual(tt.epochs_for(2000), 4)
        self.assertEqual(tt.epochs_for(5000), 2)
        self.assertEqual(tt.epochs_for(10000), 1)
        self.assertEqual(tt.epochs_for(20000), 1)

    def test_total_exposures_are_roughly_comparable(self):
        exposures = {rows: rows * tt.epochs_for(rows) for rows in tt.LADDER_SIZES}
        self.assertEqual(exposures, {2000: 8000, 5000: 10000, 10000: 10000, 20000: 20000})

    def test_optimizer_steps_are_consistent(self):
        for rows in tt.LADDER_SIZES:
            hp = tt.expected_hyperparameters(rows)
            self.assertEqual(hp["seed"], 94101)
            self.assertEqual((hp["rank"], hp["alpha"]), (32, 64))
            self.assertEqual(hp["optimizer_steps"], rows * tt.epochs_for(rows) // 8)

    def test_adapter_config_truth_table(self):
        good = {
            "r": 32, "lora_alpha": 64,
            "base_model_name_or_path": str(tt.MODEL_PATH.resolve()),
            "target_modules": list(tt.LORA_TARGET_MODULES),
        }
        self.assertTrue(tt.validate_adapter_config(good))
        for mutate in (
            {"r": 16}, {"lora_alpha": 32}, {"base_model_name_or_path": "/wrong"},
            {"target_modules": ["q_proj"]},
        ):
            bad = copy.deepcopy(good)
            bad.update(mutate)
            self.assertFalse(tt.validate_adapter_config(bad), msg=f"should reject {mutate}")

    def test_published_rungs_unpinned_is_fail_closed(self):
        for rows in tt.LADDER_SIZES:
            self.assertIsNone(tt.PUBLISHED_RUNG_HASHES[rows])  # TODO-PIN until trained
        with self.assertRaises(ValueError):
            tt.validate_published_rung(2000, require_committed=True)

    def test_base_pins_are_the_reserialized_base(self):
        self.assertEqual(
            tt.MODEL_PATH_TREE_SHA256,
            "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
        )
        self.assertEqual(
            tt.MODEL_PATH_WEIGHTS_SHA256,
            "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
        )
        self.assertEqual(tt.MODEL_PATH_WEIGHTS_SIZE_BYTES, 9_078_620_536)


class TestLadderManifestShaPinning(unittest.TestCase):
    def test_manifest_present_and_pins_generator_and_fixture(self):
        self.assertTrue(MANIFEST.is_file(), "ladder manifest missing (run scripts/run.py --stage gen-ladder)")
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(manifest["construction_seed"], gen.CONSTRUCTION_SEED)
        self.assertEqual(manifest["ladder_sizes"], list(tt.LADDER_SIZES))
        self.assertEqual(manifest["generator_sha256"], bl.sha256_file(bl.GENERATOR))
        self.assertEqual(manifest["contamination_fixture_sha256"], bl.sha256_file(bl.FIXTURE))

    def test_smallest_rung_regenerates_to_the_pinned_sha(self):
        manifest = json.loads(MANIFEST.read_text())
        pinned = {r["rows"]: r["corpus_sha256"] for r in manifest["rungs"]}
        corpus = gen.generate_curriculum(gen.CONSTRUCTION_SEED, 2000)
        payload = "".join(json.dumps(gen.public_row(r), ensure_ascii=False) + "\n" for r in corpus)
        self.assertEqual(gen.sha256_text(payload), pinned[2000])

    def test_train_trial_reads_rung_from_manifest(self):
        for rows in tt.LADDER_SIZES:
            path, sha = tt.corpus_for(rows)
            self.assertEqual(len(sha), 64)
            self.assertTrue(str(path).endswith(f"why_scale_{rows}.jsonl"))


if __name__ == "__main__":
    unittest.main()
