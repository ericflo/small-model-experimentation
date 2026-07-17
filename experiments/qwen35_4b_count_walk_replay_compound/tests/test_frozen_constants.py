"""Cross-module equality of every frozen constant: seeds, pins, recipe.

One drifted copy of a seed, arm order, band, or parent pin between the
gate, the evaluator, the trainer, the merger, the rebuilder, and the
sealed-event runner would silently measure the wrong thing; these tests
require exact equality everywhere.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_design as cd  # noqa: E402
import check_local as cl  # noqa: E402
import eval_local_vllm as ev  # noqa: E402
import gen_local_gate as gg  # noqa: E402
import merge_trained_arm as mt  # noqa: E402
import rebuild_lineage as rl  # noqa: E402
import run_benchmark as rb  # noqa: E402
import train_trial as tt  # noqa: E402


def load_harness():
    spec = importlib.util.spec_from_file_location(
        "replay_compound_run_harness", SCRIPTS / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HARNESS = load_harness()


class TestSeeds(unittest.TestCase):
    def test_screen_seeds_are_the_frozen_fresh_triple_everywhere(self):
        expected = (88060, 88061, 88062)
        self.assertEqual(cl.SCREEN_SEEDS, expected)
        self.assertEqual(tuple(gg.INPUT_SEEDS), expected)
        self.assertEqual(ev.SCREEN_SEEDS, expected)
        self.assertEqual(rb.SCREEN_SEEDS, expected)
        self.assertEqual(HARNESS.SCREEN_SEEDS, expected)
        self.assertEqual(mt.LOCAL_SCREEN_SEEDS, expected)

    def test_aggregate_seed_is_78168_everywhere(self):
        self.assertEqual(cl.AGGREGATE_SEED, 78168)
        self.assertEqual(rb.FROZEN_SEED, 78168)
        self.assertEqual(HARNESS.AGGREGATE_SEED, 78168)

    def test_training_seed_is_86_everywhere(self):
        self.assertEqual(tt.TRAINING_SEED, 86)
        self.assertEqual(rl.STAGE8_SEED, 86)
        self.assertEqual(tt.expected_hyperparameters()["seed"], 86)

    def test_stage7_training_seed_is_untouched(self):
        self.assertEqual(rl.TRAINING_SEED, 85)
        self.assertEqual(rl.STAGE_SEEDS, (42, 43, 44, 47, 51, 55, 85))


class TestArmsAndOrder(unittest.TestCase):
    def test_local_arms_parent_first(self):
        self.assertEqual(cl.ARMS, ("count_walk", "replay_compound"))
        self.assertEqual(ev.LABELS, ("count_walk", "replay_compound"))
        self.assertEqual(HARNESS.LOCAL_LABELS, ("count_walk", "replay_compound"))

    def test_benchmark_order_is_base_parent_candidate(self):
        expected = ("base", "count_walk", "replay_compound")
        self.assertEqual(rb.MODEL_ORDER, expected)
        self.assertEqual(HARNESS.MODEL_ORDER, expected)

    def test_single_candidate(self):
        self.assertEqual(cl.CANDIDATES, ("replay_compound",))
        self.assertEqual(rb.FROZEN_CANDIDATES, ("replay_compound",))
        self.assertEqual(ev.CANDIDATE, "replay_compound")


class TestEventGeometry(unittest.TestCase):
    def test_event_shape(self):
        self.assertEqual(rb.FROZEN_NAME, "compound")
        self.assertEqual(rb.FROZEN_TIER, "medium")
        self.assertEqual(rb.FROZEN_THINK_BUDGET, 1024)
        self.assertEqual(HARNESS.FROZEN_NAME, "compound")
        self.assertEqual(HARNESS.FROZEN_TIER, "medium")
        self.assertEqual(HARNESS.FROZEN_THINK_BUDGET, 1024)

    def test_rows_per_arm(self):
        self.assertEqual(cl.ROWS_PER_ARM, 312)
        self.assertEqual(ev.ROWS_PER_ARM, 312)
        self.assertEqual(gg.ROWS_PER_ARM, 312)
        self.assertEqual(HARNESS.ROWS_PER_ARM, 312)

    def test_gateway_sha_is_identical_everywhere_and_matches_disk(self):
        self.assertEqual(rb.GATEWAY_SHA256, cd.GATEWAY_SHA256)
        self.assertEqual(rb.GATEWAY_SHA256, HARNESS.GATEWAY_SHA256)
        self.assertEqual(rb.sha256_file(rb.GATEWAY), rb.GATEWAY_SHA256)


class TestParentPins(unittest.TestCase):
    TREE = "d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1"
    WEIGHTS = "ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3"
    COMMITTED_RECEIPT = (
        "840edca0638b9e291bb34fde28b4b530df8743faf9b7b18b7f2358ce55ec4c36"
    )
    INNER_RECEIPT = (
        "3c432f110fe96a508d6a75ab34e4a649671a3d7b2d942f3346cab609bef437d7"
    )

    def test_parent_tree_pin_everywhere(self):
        self.assertEqual(tt.MODEL_PATH_TREE_SHA256, self.TREE)
        self.assertEqual(rb.FROZEN_TREE_SHA256["count_walk"], self.TREE)
        self.assertEqual(gg.EXPECTED_PARENT_TREE_SHA256, self.TREE)
        self.assertEqual(ev.EXPECTED_INHERITED_TREE_SHA256["count_walk"], self.TREE)
        self.assertEqual(
            rl.ARM_COMPOSITE_PINS["count_walk"]["tree_sha256"], self.TREE
        )

    def test_parent_weights_pin_everywhere(self):
        self.assertEqual(tt.MODEL_PATH_WEIGHTS_SHA256, self.WEIGHTS)
        self.assertEqual(rb.FROZEN_WEIGHTS_SHA256["count_walk"], self.WEIGHTS)
        self.assertEqual(gg.EXPECTED_PARENT_WEIGHTS_SHA256, self.WEIGHTS)
        self.assertEqual(ev.EXPECTED_WEIGHTS_SHA256["count_walk"], self.WEIGHTS)
        self.assertEqual(
            rl.ARM_COMPOSITE_PINS["count_walk"]["weights_sha256"], self.WEIGHTS
        )
        # The merge wrapper's pre-merge full-weights pin (mirrors
        # train_trial's pre-training check).
        self.assertEqual(mt.BASE_COMPOSITE_WEIGHTS_SHA256, self.WEIGHTS)

    def test_parent_committed_receipt_pin_everywhere(self):
        self.assertEqual(
            tt.PARENT_COMMITTED_MERGE_RECEIPT_SHA256, self.COMMITTED_RECEIPT
        )
        self.assertEqual(
            rb.COUNT_WALK_PARENT_MERGE_RECEIPT_SHA256, self.COMMITTED_RECEIPT
        )
        self.assertEqual(
            gg.EXPECTED_PARENT_MERGE_RECEIPT_SHA256, self.COMMITTED_RECEIPT
        )
        self.assertEqual(
            ev.EXPECTED_RECEIPT_SHA256["count_walk"], self.COMMITTED_RECEIPT
        )

    def test_parent_inner_receipt_pin_everywhere(self):
        self.assertEqual(tt.MODEL_PATH_RECEIPT_SHA256, self.INNER_RECEIPT)
        self.assertEqual(mt.BASE_COMPOSITE_RECEIPT_SHA256, self.INNER_RECEIPT)
        self.assertEqual(
            rb.COUNT_WALK_PARENT_INNER_RECEIPT_SHA256, self.INNER_RECEIPT
        )

    def test_base_pins(self):
        self.assertEqual(
            rb.FROZEN_TREE_SHA256["base"],
            "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
        )
        self.assertEqual(
            rb.FROZEN_WEIGHTS_SHA256["base"],
            "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
        )
        self.assertEqual(
            rb.BASE_MERGE_RECEIPT_SHA256,
            "25aee794cfffe4d58110defc61177edef1f5324e47deb28fbd3cb7ccd61ae54f",
        )

    def test_parent_paths_agree(self):
        self.assertEqual(
            tt.MODEL_PATH.resolve(),
            rb.FROZEN_MODEL_PATHS["count_walk"].resolve(),
        )
        self.assertEqual(
            tt.MODEL_PATH.resolve(), mt.BASE_COMPOSITE.resolve()
        )
        self.assertEqual(tt.MODEL_PATH.resolve(), HARNESS.MODEL_PATH.resolve())
        self.assertEqual(tt.MODEL_PATH.resolve(), ev.MERGED["count_walk"].resolve())


class TestRecipe(unittest.TestCase):
    def test_trainer_hyperparameters_match_the_manifest_recipe(self):
        manifest = rl.load_manifest()
        recipe = manifest["stage8_replay_compound"]["recipe"]
        expected = tt.expected_hyperparameters()
        for key, value in recipe["hyperparameters"].items():
            self.assertEqual(expected[key], value, key)
        self.assertEqual(expected["optimizer_steps"], recipe["optimizer_steps"])

    def test_dataset_pin_matches_everywhere(self):
        sha = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
        self.assertEqual(tt.ARM_FILES["replay_compound"][1], sha)
        self.assertEqual(rl.STAGE8_DATASET_SHA256, sha)
        blend = [row for row in cd.FROZEN_CORPORA if row[0] == "data/sft_blend.jsonl"]
        self.assertEqual(blend[0][1], sha)
        self.assertEqual(blend[0][2], 2240)

    def test_trainer_sha_pin_matches_the_copied_trainer(self):
        self.assertEqual(
            tt.TRAINER_SHA256,
            "e0eca2a230dae5d109d418dcb4cc19af05882a770af14350ffd741a8d5e90f01",
        )
        self.assertEqual(tt.sha256_file(tt.TRAINER), tt.TRAINER_SHA256)

    def test_merger_sha_pin_matches_the_copied_merger(self):
        self.assertEqual(
            mt.MERGER_SHA256,
            "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672",
        )
        self.assertEqual(mt.sha256_file(mt.MERGER), mt.MERGER_SHA256)

    def test_lora_geometry(self):
        self.assertEqual(tt.LORA_RANK, 32)
        self.assertEqual(tt.LORA_ALPHA, 64)
        self.assertEqual(tt.EXPECTED_ROWS, 2240)
        self.assertEqual(tt.OPTIMIZER_STEPS, 280)


class TestConsequenceAndBands(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(cl.RETENTION_CORRECT_BAND, 5)
        self.assertEqual(cl.RETENTION_CAP_BAND, 3)
        self.assertEqual(cl.RETENTION_PARSED_BAND, 3)

    def test_consequence_constants(self):
        self.assertEqual(rb.PER_FAMILY_SLACK, 0.1)
        self.assertEqual(rb.SLACK_EPSILON, 1e-9)
        self.assertEqual(rb.AGG_TIE_EPSILON, 1e-12)

    def test_public_families_are_the_ten(self):
        self.assertEqual(len(rb.PUBLIC_FAMILIES), 10)
        self.assertIn("menders", rb.PUBLIC_FAMILIES)


class TestDesignReceiptConsistency(unittest.TestCase):
    def test_design_receipt_when_present_matches_the_frozen_constants(self):
        receipt_path = EXP / "data" / "local_design_receipt.json"
        if not receipt_path.is_file():
            self.skipTest("local design receipt not yet generated")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["screen_seeds"], [88060, 88061, 88062])
        self.assertEqual(receipt["aggregate_seed"], 78168)
        self.assertEqual(receipt["arms"], ["count_walk", "replay_compound"])
        self.assertEqual(receipt["rows_per_arm"], 312)
        self.assertTrue(receipt["gates"]["bands_two_sided"])
        self.assertEqual(receipt["gates"]["retention_correct_band"], 5)
        self.assertEqual(receipt["gates"]["retention_cap_contact_band"], 3)
        self.assertEqual(receipt["gates"]["retention_parsed_band"], 3)


if __name__ == "__main__":
    unittest.main()
