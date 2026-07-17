import importlib.util
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
# This eval-only cell copied its lineage package from lifecycle 30's
# state_track_install cell (which itself carries lifecycle 27's clean chain).
SOURCE = ROOT / "experiments" / "qwen35_4b_state_track_install"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LINEAGE = load_module("stc_rebuild_lineage", "rebuild_lineage.py")
BENCH = load_module("stc_run_benchmark_lineage", "run_benchmark.py")


class LineagePackageTests(unittest.TestCase):
    """The in-cell standalone stage 1-9 package must verify (eval-only doctrine)."""

    def test_manifest_loads_and_every_copied_file_matches_its_pin(self) -> None:
        manifest = LINEAGE.load_manifest()
        checked = LINEAGE.verify_inputs(manifest)
        self.assertEqual(checked["datasets"], 7)
        self.assertEqual(checked["arm_streams"], 2)
        self.assertEqual(checked["stage8_dataset"], 1)
        self.assertEqual(checked["stage9_dataset"], 1)
        self.assertEqual(checked["materialization_inputs"], 2)
        self.assertEqual(checked["token_receipt"], 1)
        self.assertEqual(checked["trainers"], 4)
        self.assertEqual(checked["merger"], 1)
        self.assertEqual(checked["wrappers"], 2)
        self.assertEqual(checked["documentation_copies"], 1)
        self.assertEqual(checked["provenance_receipts"], 7)
        self.assertFalse(checked["root_vendored"])

    def test_stage9_block_is_the_state_track_install_dose(self) -> None:
        manifest = LINEAGE.load_manifest()
        stage9 = manifest["stage9_state_track_install"]
        self.assertEqual(stage9["extended_by"], "qwen35_4b_state_track_install")
        self.assertEqual(LINEAGE.STAGE9_EXTENDED_BY, "qwen35_4b_state_track_install")
        self.assertEqual(stage9["arm"], "state_track")
        self.assertEqual(stage9["training_seed"], 87)
        self.assertEqual(stage9["dataset"]["file"], "data/sft_state_track.jsonl")
        self.assertEqual(stage9["dataset"]["rows"], 160)
        # The candidate the confirmation evaluates was trained on count_walk.
        self.assertEqual(
            stage9["parent_composite"]["tree_sha256"],
            BENCH.FROZEN_TREE_SHA256["count_walk"],
        )

    def test_count_walk_manifest_pins_equal_the_eval_arm_pins(self) -> None:
        manifest = LINEAGE.load_manifest()
        arms = manifest["stage7_confirmation_arms"]["arms"]
        self.assertEqual(
            arms["count_walk"]["composite"]["tree_sha256"],
            BENCH.FROZEN_TREE_SHA256["count_walk"],
        )
        self.assertEqual(
            arms["count_walk"]["composite"]["weights_sha256"],
            BENCH.FROZEN_WEIGHTS_SHA256["count_walk"],
        )

    def test_copied_files_are_byte_identical_to_the_source_cell(self) -> None:
        for relative in (
            "data/lineage/lineage_manifest.json",
            "data/lineage/stage01_replay_refresh.jsonl",
            "data/lineage/stage02_designed160.jsonl",
            "data/lineage/stage03_close_xi__targeted_standard.jsonl",
            "data/lineage/stage04_replay_after_close.jsonl",
            "data/lineage/stage05_designed_fresh.jsonl",
            "data/lineage/stage06_hygiene_explore.jsonl",
            "data/lineage/provenance/merge.json",
            "data/count_walk.jsonl",
            "data/replay_ctl7.jsonl",
            "data/sft_count_walk.jsonl",
            "data/sft_blend.jsonl",
            "data/sft_state_track.jsonl",
            "data/stream_token_receipt.json",
            "scripts/lineage_trainers/train_think_stage12.py",
            "scripts/lineage_trainers/train_think_close_stage3.py",
            "scripts/lineage_trainers/train_think_stage456.py",
            "scripts/train_think.py",
            "scripts/merge_adapter.py",
            "scripts/rebuild_clean_chain.py",
            "scripts/train_trial.py",
            "scripts/merge_trained_arm.py",
        ):
            copy = EXP / relative
            original = SOURCE / relative
            self.assertTrue(copy.is_file(), relative)
            self.assertTrue(original.is_file(), relative)
            self.assertEqual(copy.read_bytes(), original.read_bytes(), relative)

    def test_provenance_copies_match_the_source_cell(self) -> None:
        # The count_walk provenance copy is byte-identical to lifecycle 30's;
        # the state_track and prior-event copies are this cell's verification
        # aids for its two eval arms.
        self.assertEqual(
            (EXP / "data/provenance/count_walk_merge.json").read_bytes(),
            (SOURCE / "data/provenance/count_walk_merge.json").read_bytes(),
        )
        self.assertEqual(
            (EXP / "data/provenance/state_track_merge.json").read_bytes(),
            (SOURCE / "runs/merges/state_track.json").read_bytes(),
        )

    def test_tampered_manifest_refuses(self) -> None:
        original = LINEAGE.MANIFEST_SHA256
        try:
            LINEAGE.MANIFEST_SHA256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                LINEAGE.load_manifest()
        finally:
            LINEAGE.MANIFEST_SHA256 = original

    def test_no_root_adapter_is_vendored(self) -> None:
        LINEAGE.require_root_not_vendored()
        for path in LINEAGE.FORBIDDEN_ROOT_DIRS:
            self.assertFalse(path.exists(), str(path))


if __name__ == "__main__":
    unittest.main()
