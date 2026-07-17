"""Clean-chain enforcement and cross-module frozen-constant contracts.

The point of the clean chain is a single installed model whose ENTIRE
lineage is documented and contamination-free: six zero-root stages
(lifecycle 22) + the FRESH enumerative-repair dose as stage 7. These tests
hold:

- NO blend root anywhere: the forbidden artifact-storage directories do
  not exist, the rebuild script fails closed if one appears, and the
  clean-chain manifest never references the undocumented root adapter's
  bytes or a ``root_adapter`` block;
- the standalone package: the six stage datasets are byte-identical to
  the zero-root cell's committed copies, the provenance receipts are
  byte-identical to lifecycle 22's committed originals, and stage 7
  records exactly this cell's frozen dose (stream sha, seed 83, the
  byte-copied proven trainer);
- the duplicated frozen constants (seeds, arm names, parent pins) agree
  across the harness, the gate, the trainer wrapper, the merge wrapper,
  the evaluator, the benchmark runner, and the design-receipt generator.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
ZR = ROOT / "experiments" / "qwen35_4b_zero_root_lineage_rebuild"
sys.path.insert(0, str(SCRIPTS))

import check_design as cd  # noqa: E402
import check_local as cl  # noqa: E402
import rebuild_clean_chain as rc  # noqa: E402


def import_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


harness = import_by_path("enum_repair_harness", SCRIPTS / "run.py")
trial = import_by_path("enum_repair_train_trial_cc", SCRIPTS / "train_trial.py")
merge = import_by_path("enum_repair_merge_arm", SCRIPTS / "merge_trained_arm.py")
bench = import_by_path("enum_repair_run_benchmark_cc", SCRIPTS / "run_benchmark.py")
mat = import_by_path("enum_repair_materialize_cc", SCRIPTS / "materialize_streams.py")
evaluator = import_by_path("enum_repair_eval_local", SCRIPTS / "eval_local_vllm.py")

PARENT_TREE = "414f582950bf60fed2fe462cd141ab98d0f772087b4f9c6bc5aa12f03f379e7d"
PARENT_WEIGHTS = "6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932"
PARENT_COMMITTED_RECEIPT = (
    "e906caea7c4b86f4a3eacb96affb7cc2fa9b7cc11e11b634b651cabc5dd01d2b"
)
PARENT_INNER_RECEIPT = (
    "f8981f4638d901471eb41aff0ffd0bfac88aebd6e3e4d4db1e1c733be16709c0"
)


class TestBlendRootAbsence(unittest.TestCase):
    def test_forbidden_directories_do_not_exist(self):
        for path in rc.FORBIDDEN_ROOT_DIRS:
            with self.subTest(path=str(path)):
                self.assertFalse(path.exists())
        self.assertFalse(cd.FORBIDDEN_ROOT_DIR.exists())
        self.assertIn(cd.FORBIDDEN_ROOT_DIR, rc.FORBIDDEN_ROOT_DIRS)

    def test_rebuild_script_fails_closed_when_a_root_appears(self):
        target = rc.FORBIDDEN_ROOT_DIRS[0]
        self.assertFalse(target.exists())
        target.mkdir(parents=True)
        try:
            with self.assertRaisesRegex(ValueError, "clean chain"):
                rc.require_root_not_vendored()
        finally:
            target.rmdir()
            while not target.exists() and target.parent != ROOT / "large_artifacts":
                target = target.parent
                if target.is_dir() and not any(target.iterdir()):
                    target.rmdir()
                else:
                    break
        self.assertFalse(rc.FORBIDDEN_ROOT_DIRS[0].exists())
        rc.require_root_not_vendored()

    def test_manifest_never_references_the_blend_root(self):
        text = rc.MANIFEST.read_text(encoding="utf-8")
        self.assertNotIn("root_adapter", text)
        self.assertNotIn(cd.BLEND_ROOT_WEIGHTS_SHA256, text)
        self.assertNotIn(cd.BLEND_ROOT_CONFIG_SHA256, text)
        manifest = json.loads(text)
        self.assertEqual(manifest["framing"], "clean_chain")
        self.assertEqual(manifest["stages"][0]["warm_start"], "fresh_zero_init")

    def test_no_warm_start_pathway_in_the_dose_wrappers(self):
        for name in ("train_trial.py", "merge_trained_arm.py"):
            text = (SCRIPTS / name).read_text(encoding="utf-8")
            for token in ("--warm" + "-start", "warm" + "_start"):
                self.assertNotIn(token, text, name)


class TestStandalonePackage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = rc.load_manifest()

    def test_stage_datasets_are_byte_identical_to_the_zero_root_cell(self):
        for row in self.manifest["stages"][:6]:
            name = Path(row["dataset"]["file"]).name
            with self.subTest(dataset=name):
                copy = EXP / "data" / "lineage" / name
                original = ZR / "data" / "lineage" / name
                self.assertEqual(copy.read_bytes(), original.read_bytes())

    def test_provenance_receipts_are_byte_identical_to_lifecycle22(self):
        for name in self.manifest["clean_chain"]["provenance_receipts"]:
            with self.subTest(receipt=name):
                copy = EXP / "data" / "lineage" / "provenance" / name
                original = ZR / "runs" / "lineage" / name
                self.assertEqual(copy.read_bytes(), original.read_bytes())

    def test_stage7_records_this_cells_frozen_dose(self):
        stage7 = self.manifest["stages"][6]
        self.assertEqual(stage7["name"], "enum_repair")
        self.assertEqual(stage7["seed"], 83)
        self.assertEqual(stage7["dataset"]["file"], "data/enum_repair.jsonl")
        self.assertEqual(
            stage7["dataset"]["sha256"],
            trial.ARM_FILES["enum_repair"][1],
        )
        self.assertEqual(
            stage7["trainer_sha256"],
            rc.sha256_file(SCRIPTS / "train_think.py"),
        )
        self.assertEqual(stage7["stage7_base"]["tree_sha256"], PARENT_TREE)
        self.assertEqual(stage7["stage7_base"]["weights_sha256"], PARENT_WEIGHTS)
        self.assertEqual(
            stage7["hyperparameters"],
            {
                "alpha": 64,
                "batch_size": 1,
                "epochs": 1.0,
                "grad_accum": 8,
                "lr": 1e-05,
                "max_length": 4096,
                "rank": 32,
                "w_close": 0.2,
                "w_think": 0.2,
            },
        )
        # The stage-7 composition names the FRESH treatment corpus.
        self.assertIn(cd.TREATMENT_SHA256, stage7["dataset"]["composition"])
        self.assertIn("77190", stage7["dataset"]["composition"])
        self.assertIn("55170", stage7["dataset"]["composition"])

    def test_stage_seeds_are_the_documented_chain_plus_79(self):
        self.assertEqual(rc.STAGE_SEEDS, (42, 43, 44, 47, 51, 55, 83))
        self.assertEqual(
            [row["seed"] for row in self.manifest["stages"]],
            list(rc.STAGE_SEEDS),
        )

    def test_vendored_merger_is_byte_identical_to_the_canonical_merger(self):
        canonical = (
            ROOT
            / "experiments"
            / "qwen35_4b_same_prefix_advantage_routing"
            / "scripts"
            / "merge_adapter.py"
        )
        copy = SCRIPTS / "merge_adapter.py"
        self.assertEqual(copy.read_bytes(), canonical.read_bytes())
        self.assertEqual(merge.MERGER, copy)
        self.assertEqual(rc.sha256_file(copy), merge.MERGER_SHA256)

    def test_vendored_trainer_is_byte_identical_to_the_proven_source(self):
        source = (
            ROOT
            / "experiments"
            / "qwen35_4b_statechain_only_dose"
            / "scripts"
            / "train_think.py"
        )
        copy = SCRIPTS / "train_think.py"
        self.assertEqual(copy.read_bytes(), source.read_bytes())

    def test_manifest_pin_agrees_across_modules(self):
        self.assertEqual(rc.MANIFEST_SHA256, cd.LINEAGE_MANIFEST_SHA256)
        self.assertEqual(rc.sha256_file(rc.MANIFEST), rc.MANIFEST_SHA256)


class TestFrozenConstants(unittest.TestCase):
    def test_seed_constants_agree_across_modules(self):
        self.assertEqual(harness.LOCAL_SEED, 88052)
        self.assertEqual(cl.SEED, 88052)
        self.assertEqual(evaluator.SEED, 88052)
        self.assertEqual(cd.LOCAL_SEED, 88052)
        self.assertEqual(merge.LOCAL_SEED, 88052)
        for module_seeds in (
            harness.SCREEN_SEEDS,
            cl.SCREEN_SEEDS,
            evaluator.SCREEN_SEEDS,
            cd.SCREEN_SEEDS,
        ):
            self.assertEqual(tuple(module_seeds), (88053, 88054, 88055))
        for taken in (88043, 88047, 88049):
            self.assertNotIn(taken, harness.SCREEN_SEEDS)
        self.assertEqual(harness.AGGREGATE_SEED, 78162)
        self.assertEqual(cl.AGGREGATE_SEED, 78162)
        self.assertEqual(cd.AGGREGATE_SEED, 78162)
        self.assertEqual(bench.FROZEN_SEED, 78162)
        self.assertEqual(mat.STREAM_ORDER_SEED, 55170)
        self.assertEqual(cd.NAMESPACE_SEED, 55170)
        self.assertEqual(cd.CONSTRUCTION_SEED, 77190)
        self.assertEqual(cd.TRAINING_SEED, 83)
        self.assertEqual(trial.expected_hyperparameters()["seed"], 83)

    def test_arm_names_agree_across_modules(self):
        self.assertEqual(harness.ARMS, ("replay_ctl6", "enum_repair"))
        self.assertEqual(
            cl.ARMS, ("zero_root_parent", "replay_ctl6", "enum_repair")
        )
        self.assertEqual(evaluator.LABELS, cl.ARMS)
        self.assertEqual(tuple(cd.GATE_ARMS), cl.ARMS)
        self.assertEqual(
            bench.MODEL_ORDER,
            ("base", "zero_root_parent", "replay_ctl6", "enum_repair"),
        )
        self.assertEqual(bench.FROZEN_CANDIDATES, ("enum_repair",))
        self.assertEqual(harness.CANDIDATE_ARMS, ("enum_repair",))

    def test_parent_pins_agree_across_modules(self):
        self.assertEqual(cd.PARENT_TREE_SHA256, PARENT_TREE)
        self.assertEqual(cd.PARENT_WEIGHTS_SHA256, PARENT_WEIGHTS)
        self.assertEqual(cd.PARENT_RECEIPT_SHA256, PARENT_COMMITTED_RECEIPT)
        self.assertEqual(cd.PARENT_INNER_RECEIPT_SHA256, PARENT_INNER_RECEIPT)
        self.assertEqual(trial.MODEL_PATH_TREE_SHA256, PARENT_TREE)
        self.assertEqual(trial.MODEL_PATH_WEIGHTS_SHA256, PARENT_WEIGHTS)
        self.assertEqual(trial.MODEL_PATH_RECEIPT_SHA256, PARENT_INNER_RECEIPT)
        self.assertEqual(merge.BASE_COMPOSITE_RECEIPT_SHA256, PARENT_INNER_RECEIPT)
        self.assertEqual(bench.FROZEN_TREE_SHA256["zero_root_parent"], PARENT_TREE)
        self.assertEqual(
            bench.FROZEN_WEIGHTS_SHA256["zero_root_parent"], PARENT_WEIGHTS
        )
        self.assertEqual(
            bench.ZERO_ROOT_PARENT_MERGE_RECEIPT_SHA256, PARENT_COMMITTED_RECEIPT
        )
        self.assertEqual(
            evaluator.EXPECTED_RECEIPT_SHA256["zero_root_parent"],
            PARENT_COMMITTED_RECEIPT,
        )
        self.assertEqual(
            evaluator.EXPECTED_INHERITED_TREE_SHA256["zero_root_parent"],
            PARENT_TREE,
        )
        self.assertEqual(
            evaluator.EXPECTED_WEIGHTS_SHA256["zero_root_parent"], PARENT_WEIGHTS
        )

    def test_parent_paths_agree_across_modules(self):
        expected = (
            ROOT
            / "large_artifacts"
            / "qwen35_4b_zero_root_lineage_rebuild"
            / "merged"
            / "zero_root_hygiene_explore"
        )
        self.assertEqual(harness.MODEL_PATH, expected)
        self.assertEqual(trial.MODEL_PATH, expected)
        self.assertEqual(merge.BASE_COMPOSITE, expected)
        self.assertEqual(mat.BASE_COMPOSITE, expected)
        self.assertEqual(bench.FROZEN_MODEL_PATHS["zero_root_parent"], expected)
        self.assertEqual(evaluator.MERGED["zero_root_parent"], expected)
        self.assertEqual(cd.PARENT, expected)

    def test_the_zero_root_committed_receipt_matches_its_pin(self):
        committed = ZR / "runs" / "lineage" / "merge.json"
        self.assertEqual(rc.sha256_file(committed), PARENT_COMMITTED_RECEIPT)
        payload = json.loads(committed.read_text(encoding="utf-8"))
        self.assertEqual(payload["output_tree_sha256"], PARENT_TREE)
        self.assertEqual(payload["weights_sha256"], PARENT_WEIGHTS)
        self.assertEqual(payload["inner_merge_receipt_sha256"], PARENT_INNER_RECEIPT)

    def test_corpus_pins_agree(self):
        self.assertEqual(
            harness.CORPUS_HASHES["sft_enum_repair.jsonl"],
            cd.TREATMENT_SHA256,
        )
        self.assertEqual(
            harness.CORPUS_HASHES["sft_blend.jsonl"], cd.REPLAY_SHA256
        )
        self.assertEqual(
            harness.CORPUS_HASHES["corpus_manifest.json"], cd.MANIFEST_SHA256
        )
        self.assertEqual(mat.TREATMENT_SHA256, cd.TREATMENT_SHA256)

    def test_stream_pins_agree_between_wrapper_and_lineage_manifest(self):
        manifest = rc.load_manifest()
        stage7 = manifest["stages"][6]
        self.assertEqual(
            trial.ARM_FILES["enum_repair"][1], stage7["dataset"]["sha256"]
        )
        self.assertIn(
            trial.ARM_FILES["replay_ctl6"][1],
            stage7["stage7_base"]["note"],
        )

    def test_verdict_strings(self):
        self.assertEqual(
            harness.TRAIN_VERDICT, "**Verdict:** `PASS_CONTROL_TRAINING`."
        )
        self.assertEqual(harness.MERGE_VERDICT, "**Verdict:** `PASS_CONTROL_MERGE`.")
        self.assertEqual(harness.LOCAL_VERDICT, "**Verdict:** `PASS_LOCAL_EVENT`.")
        self.assertEqual(
            harness.BENCH_VERDICT, "**Verdict:** `PASS_BENCHMARK_EVENT`."
        )
        self.assertEqual(bench.BENCH_VERDICT, harness.BENCH_VERDICT)


if __name__ == "__main__":
    unittest.main()
