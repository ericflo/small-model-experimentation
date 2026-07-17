import importlib.util
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SOURCE = ROOT / "experiments" / "qwen35_4b_count_dont_walk_enumeration"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LINEAGE = load_module("cwmc_rebuild_lineage", "rebuild_lineage.py")
BENCH = load_module("cwmc_run_benchmark_lineage", "run_benchmark.py")


class LineagePackageTests(unittest.TestCase):
    """The in-cell standalone package (review amendment B1) must verify."""

    def test_manifest_loads_and_every_copied_file_matches_its_pin(self) -> None:
        # load_manifest authenticates the extended manifest bytes and
        # internal chaining; verify_inputs recomputes the sha256 of every
        # copied dataset (with row counts), arm stream, materialization
        # input, token receipt, trainer, merger, wrapper, documentation
        # copy, and provenance receipt against the manifest pins.
        manifest = LINEAGE.load_manifest()
        checked = LINEAGE.verify_inputs(manifest)
        self.assertEqual(checked["datasets"], 7)
        self.assertEqual(checked["arm_streams"], 2)
        self.assertEqual(checked["materialization_inputs"], 2)
        self.assertEqual(checked["token_receipt"], 1)
        self.assertEqual(checked["trainers"], 4)
        self.assertEqual(checked["merger"], 1)
        self.assertEqual(checked["wrappers"], 2)
        self.assertEqual(checked["documentation_copies"], 1)
        self.assertEqual(checked["provenance_receipts"], 7)
        self.assertFalse(checked["root_vendored"])

    def test_stage7_block_records_both_arm_streams_and_seed_85(self) -> None:
        manifest = LINEAGE.load_manifest()
        arms = manifest["stage7_confirmation_arms"]
        self.assertEqual(arms["training_seed"], 85)
        self.assertEqual(arms["recipe"]["order"], ["replay_ctl7", "count_walk"])
        self.assertEqual(
            arms["arms"]["replay_ctl7"]["stream"]["sha256"],
            "94e8259ec03800d0a4dcbf8075252c5180a668e2da74569fcf62497cf0f9de5a",
        )
        self.assertEqual(
            arms["arms"]["count_walk"]["stream"]["sha256"],
            "71291542c3c901caccf9586543efb02da319b371244728ecfd1a0fc7cb92ed26",
        )
        self.assertEqual(
            arms["trainer"]["sha256"],
            "e0eca2a230dae5d109d418dcb4cc19af05882a770af14350ffd741a8d5e90f01",
        )
        self.assertEqual(
            arms["merger"]["sha256"],
            "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672",
        )

    def test_manifest_composite_pins_equal_the_eval_arm_pins(self) -> None:
        # The lineage package must reproduce exactly what this cell
        # measures: the arms' tree/weights pins in the manifest equal the
        # eval runner's frozen constants (and the zero-root pins equal
        # the zero_root_parent arm's).
        manifest = LINEAGE.load_manifest()
        arms = manifest["stage7_confirmation_arms"]["arms"]
        for name in ("replay_ctl7", "count_walk"):
            self.assertEqual(
                arms[name]["composite"]["tree_sha256"],
                BENCH.FROZEN_TREE_SHA256[name],
                name,
            )
            self.assertEqual(
                arms[name]["composite"]["weights_sha256"],
                BENCH.FROZEN_WEIGHTS_SHA256[name],
                name,
            )
            self.assertEqual(
                arms[name]["composite"]["committed_merge_receipt_sha256"],
                BENCH.COMMITTED_MERGE_RECEIPTS[name][1],
                name,
            )
        self.assertEqual(
            LINEAGE.ZERO_ROOT_TREE_SHA256,
            BENCH.FROZEN_TREE_SHA256["zero_root_parent"],
        )
        self.assertEqual(
            LINEAGE.ZERO_ROOT_WEIGHTS_SHA256,
            BENCH.FROZEN_WEIGHTS_SHA256["zero_root_parent"],
        )

    def test_copied_files_are_byte_identical_to_the_committed_sources(self) -> None:
        # Verification aid (never the reproduction path): every copied
        # production file matches the committed lifecycle-27 original
        # byte for byte. The manifest itself is excluded — it is the
        # copy EXTENDED with the stage7_confirmation_arms block.
        for relative in (
            "data/lineage/stage01_replay_refresh.jsonl",
            "data/lineage/stage02_designed160.jsonl",
            "data/lineage/stage03_close_xi__targeted_standard.jsonl",
            "data/lineage/stage04_replay_after_close.jsonl",
            "data/lineage/stage05_designed_fresh.jsonl",
            "data/lineage/stage06_hygiene_explore.jsonl",
            "data/lineage/provenance/merge.json",
            "data/lineage/provenance/stage01_replay_refresh.json",
            "data/lineage/provenance/stage02_designed160.json",
            "data/lineage/provenance/stage03_close_xi.json",
            "data/lineage/provenance/stage04_replay_after_close.json",
            "data/lineage/provenance/stage05_designed_fresh.json",
            "data/lineage/provenance/stage06_hygiene_explore.json",
            "data/count_walk.jsonl",
            "data/replay_ctl7.jsonl",
            "data/sft_count_walk.jsonl",
            "data/sft_blend.jsonl",
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
            self.assertEqual(
                copy.read_bytes(), original.read_bytes(), relative
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
