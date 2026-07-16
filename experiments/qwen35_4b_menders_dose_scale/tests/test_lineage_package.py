"""Unit tests for the standalone lineage package (owner directive 2026-07-15).

The standalone-reproducibility gate requires this cell to carry the
complete model-reproduction package: the confirmation cell's six ordered
SFT dataset copies plus the fixed-seed per-stage recipe manifest, the
three trainer variants plus the merger copied in, the vendored frozen
root adapter (a HARD provenance boundary — no committed creation receipt
exists for it), EXTENDED with this cell's own stage 7: the candidate's
training (a fresh rank-32 adapter on the merged composite; dataset =
this cell's own materialized ``feedloop_scale.jsonl``; produced hashes
are post-training TODO-PINs with an explicit pending marker). These
tests hold the copies byte-anchored to their recorded sha256s, the
manifest schema and warm-start chaining sound (via
``rebuild_lineage.load_manifest``, including fail-closed negatives on
tampered manifests), the design-receipt generator's frozen lineage pins
identical to the manifest, the trainer-variant/stage mapping exact, the
candidate stage's pending-pin shape enforced, and ``rebuild_lineage.py
--verify-inputs`` green as an integration check. GPU rebuilding is never
exercised here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import check_design as cd  # noqa: E402
import rebuild_lineage as rl  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


MANIFEST = json.loads(rl.MANIFEST.read_text(encoding="utf-8"))
CONFIRMATION = (
    ROOT / "experiments" / "qwen35_4b_goal_gate_confirmation" / "data" / "lineage"
)


class TestCopiesMatchRecordedHashes(unittest.TestCase):
    def test_dataset_copies_match_manifest_and_receipt_pins(self) -> None:
        self.assertEqual(len(MANIFEST["stages"]), 6)
        for row in MANIFEST["stages"]:
            name = Path(row["dataset"]["file"]).name
            copy_path = EXP / "data" / "lineage" / name
            self.assertTrue(copy_path.is_file(), name)
            digest = sha256_file(copy_path)
            self.assertEqual(digest, row["dataset"]["sha256"], name)
            pinned_digest, pinned_rows = cd.LINEAGE_DATASETS[name]
            self.assertEqual(digest, pinned_digest, name)
            rows = sum(
                1
                for line in copy_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            self.assertEqual(rows, row["dataset"]["rows"], name)
            self.assertEqual(rows, pinned_rows, name)

    def test_six_stage_copies_are_byte_identical_to_the_confirmation_cell(self) -> None:
        for row in MANIFEST["stages"]:
            name = Path(row["dataset"]["file"]).name
            ours = EXP / "data" / "lineage" / name
            theirs = CONFIRMATION / name
            self.assertEqual(sha256_file(ours), sha256_file(theirs), name)

    def test_candidate_dataset_matches_the_materialized_stream(self) -> None:
        candidate = MANIFEST["candidate_stage"]
        path = EXP / candidate["dataset"]["file"]
        self.assertEqual(path.name, "feedloop_scale.jsonl")
        self.assertEqual(sha256_file(path), candidate["dataset"]["sha256"])
        self.assertEqual(candidate["dataset"]["rows"], 2280)
        relative, digest, rows = cd.LINEAGE_CANDIDATE_DATASET
        self.assertEqual(relative, candidate["dataset"]["file"])
        self.assertEqual(digest, candidate["dataset"]["sha256"])
        self.assertEqual(rows, candidate["dataset"]["rows"])

    def test_trainer_and_merger_copies_match_recorded_shas(self) -> None:
        for relative, block in MANIFEST["trainers"].items():
            path = EXP / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(sha256_file(path), block["sha256"], relative)
        merger = EXP / MANIFEST["merger"]["path"]
        self.assertEqual(sha256_file(merger), MANIFEST["merger"]["sha256"])
        for relative, digest in cd.LINEAGE_TRAINERS.items():
            self.assertEqual(
                sha256_file(EXP / "scripts" / relative), digest, relative
            )

    def test_vendored_root_adapter_matches_recorded_hashes(self) -> None:
        root_dir = ROOT / MANIFEST["root_adapter"]["vendored_path"]
        self.assertTrue(root_dir.is_dir())
        expected = MANIFEST["root_adapter"]["files"]
        self.assertEqual(
            {child.name for child in root_dir.iterdir()}, set(expected)
        )
        for name, block in expected.items():
            path = root_dir / name
            self.assertEqual(path.stat().st_size, block["size"], name)
            self.assertEqual(sha256_file(path), block["sha256"], name)

    def test_root_adapter_is_vendored_inside_this_cell(self) -> None:
        vendored = MANIFEST["root_adapter"]["vendored_path"]
        self.assertTrue(
            vendored.startswith("large_artifacts/qwen35_4b_menders_dose_scale/")
        )
        self.assertIn("provenance_boundary", MANIFEST["root_adapter"])
        self.assertIn(
            "no committed creation receipt",
            MANIFEST["root_adapter"]["provenance_boundary"],
        )


class TestManifestSchemaAndChaining(unittest.TestCase):
    def test_load_manifest_accepts_the_published_manifest(self) -> None:
        manifest = rl.load_manifest()
        self.assertEqual(manifest["experiment_id"], EXP.name)
        self.assertEqual(len(manifest["stages"]), 6)
        self.assertEqual(manifest["candidate_stage"]["stage"], 7)

    def test_stage_order_names_and_seeds(self) -> None:
        names = [row["name"] for row in MANIFEST["stages"]]
        self.assertEqual(
            names,
            [
                "replay_refresh", "designed160", "close_xi",
                "replay_after_close", "designed_fresh", "hygiene_explore",
            ],
        )
        seeds = [row["seed"] for row in MANIFEST["stages"]]
        self.assertEqual(seeds, [42, 43, 44, 47, 51, 55])
        self.assertEqual(tuple(seeds), tuple(cd.LINEAGE_TRAINING_SEEDS))
        # The candidate's training seed never collides with the chain seeds.
        self.assertEqual(MANIFEST["candidate_stage"]["seed"], 71)
        self.assertNotIn(71, seeds)

    def test_warm_start_chain_is_root_then_previous_stage(self) -> None:
        self.assertEqual(MANIFEST["stages"][0]["warm_start"], "root_adapter")
        for index, row in enumerate(MANIFEST["stages"][1:], start=2):
            self.assertEqual(row["warm_start"], f"stage {index - 1}")

    def test_trainer_variant_stage_mapping(self) -> None:
        by_stage = {row["stage"]: row for row in MANIFEST["stages"]}
        for stage in (1, 2):
            self.assertIn("train_think_stage12.py", by_stage[stage]["trainer"])
            self.assertNotIn("w_close", by_stage[stage]["hyperparameters"])
            self.assertNotIn("targeted_close_overrides", by_stage[stage])
        self.assertIn("train_think_close_stage3.py", by_stage[3]["trainer"])
        self.assertEqual(by_stage[3]["hyperparameters"]["w_close"], 0.2)
        self.assertEqual(
            by_stage[3]["targeted_close_overrides"],
            {"target_close_kinds": ["u_execute", "u_induct"], "target_w_close": 1.0},
        )
        for stage in (4, 5, 6):
            self.assertIn("train_think_stage456.py", by_stage[stage]["trainer"])
            self.assertEqual(by_stage[stage]["hyperparameters"]["w_close"], 0.2)
            self.assertNotIn("targeted_close_overrides", by_stage[stage])
        candidate = MANIFEST["candidate_stage"]
        self.assertEqual(candidate["trainer"], "scripts/train_think.py")
        self.assertIn(
            7, MANIFEST["trainers"]["scripts/train_think.py"]["stages"]
        )
        self.assertEqual(
            candidate["trainer_sha256"],
            MANIFEST["trainers"]["scripts/train_think.py"]["sha256"],
        )
        self.assertEqual(
            candidate["trainer_sha256"],
            sha256_file(EXP / "scripts" / "train_think.py"),
        )

    def test_common_hyperparameters_are_frozen(self) -> None:
        for row in MANIFEST["stages"] + [MANIFEST["candidate_stage"]]:
            hypers = row["hyperparameters"]
            self.assertEqual(hypers["lr"], 1e-05, row["name"])
            self.assertEqual(hypers["rank"], 32)
            self.assertEqual(hypers["alpha"], 64)
            self.assertEqual(hypers["batch_size"], 1)
            self.assertEqual(hypers["grad_accum"], 8)
            self.assertEqual(hypers["max_length"], 4096)
            self.assertEqual(hypers["epochs"], 1.0)
            self.assertEqual(hypers["w_think"], 0.2)
        self.assertEqual(MANIFEST["candidate_stage"]["hyperparameters"]["w_close"], 0.2)

    def test_final_merge_targets_the_parent_composite(self) -> None:
        final = MANIFEST["final_merge"]
        self.assertEqual(
            final["expected_output"]["weights_sha256"],
            cd.PARENT_WEIGHTS_SHA256,
        )
        self.assertEqual(
            final["expected_output"]["published_tree_sha256"],
            cd.PARENT_TREE_SHA256,
        )
        self.assertEqual(
            final["expected_output"]["content_files_sha256"]["model.safetensors"],
            final["expected_output"]["weights_sha256"],
        )
        self.assertIn(cd.MODEL_REVISION, final["base_model"])

    def test_candidate_stage_shape_and_pending_pins(self) -> None:
        candidate = MANIFEST["candidate_stage"]
        self.assertEqual(candidate["name"], "feedloop_scale")
        # No warm start anywhere in the candidate stage: it trains on the
        # merged composite via --model-path.
        self.assertIn("--model-path", candidate["base"])
        self.assertIn("FRESH", candidate["role"])
        produced = candidate["produced"]
        self.assertIsNone(produced["adapter_config_sha256"])
        self.assertIsNone(produced["adapter_weights_sha256"])
        self.assertIs(produced["pending_fill"], True)
        self.assertIn("TODO-PIN", produced["note"])
        merge = candidate["merge"]
        self.assertEqual(merge["merger"], "scripts/merge_adapter.py")
        self.assertEqual(merge["merger_sha256"], MANIFEST["merger"]["sha256"])
        self.assertIsNone(merge["expected_output"]["weights_sha256"])
        self.assertIs(merge["expected_output"]["pending_fill"], True)

    def test_gpu_rebuild_refuses_while_candidate_pins_are_pending(self) -> None:
        manifest = rl.load_manifest()
        with self.assertRaisesRegex(ValueError, "pending"):
            rl.rebuild(manifest)

    def test_raw_base_and_reserialized_eval_arm_are_distinguished(self) -> None:
        self.assertEqual(MANIFEST["model"]["id"], "Qwen/Qwen3.5-4B")
        self.assertEqual(MANIFEST["model"]["revision"], cd.MODEL_REVISION)
        self.assertIn("EVERY training stage", MANIFEST["model"]["role"])
        self.assertIn("b654e033", MANIFEST["base_reserialized_note"])
        self.assertIn(
            "never a training or merge input", MANIFEST["base_reserialized_note"]
        )

    def test_produced_hashes_are_verification_aids(self) -> None:
        self.assertIs(
            MANIFEST["determinism"]["produced_hashes_are_verification_aids"], True
        )
        for row in MANIFEST["stages"]:
            self.assertIs(row["produced"]["verification_aid"], True)
        # Stage 6's produced adapter is the one the parent merge consumes.
        self.assertEqual(
            MANIFEST["stages"][5]["produced"]["adapter_weights_sha256"],
            "7e28d6d152e7c2dbf7641d8516b9b47a1465b34967476ab01389d941b9563316",
        )


class TestManifestFailsClosed(unittest.TestCase):
    def check_rejects(self, mutate) -> None:
        broken = copy.deepcopy(MANIFEST)
        mutate(broken)
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "lineage_manifest.json"
            path.write_text(
                json.dumps(broken, indent=1, sort_keys=True, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            original = rl.MANIFEST
            rl.MANIFEST = path
            try:
                with self.assertRaises(ValueError):
                    rl.load_manifest()
            finally:
                rl.MANIFEST = original

    def test_broken_warm_start_chain_rejected(self) -> None:
        self.check_rejects(
            lambda m: m["stages"][3].update(warm_start="root_adapter")
        )

    def test_out_of_order_stages_rejected(self) -> None:
        def swap(manifest: dict) -> None:
            stages = manifest["stages"]
            stages[0], stages[1] = stages[1], stages[0]

        self.check_rejects(swap)

    def test_duplicate_seeds_rejected(self) -> None:
        self.check_rejects(lambda m: m["stages"][1].update(seed=42))

    def test_candidate_seed_collision_rejected(self) -> None:
        self.check_rejects(lambda m: m["candidate_stage"].update(seed=55))

    def test_unregistered_trainer_rejected(self) -> None:
        self.check_rejects(
            lambda m: m["stages"][0].update(trainer="scripts/other_trainer.py")
        )

    def test_missing_top_level_key_rejected(self) -> None:
        self.check_rejects(lambda m: m.pop("root_adapter"))

    def test_missing_candidate_stage_rejected(self) -> None:
        self.check_rejects(lambda m: m.pop("candidate_stage"))

    def test_root_weights_file_inconsistency_rejected(self) -> None:
        self.check_rejects(
            lambda m: m["root_adapter"].update(weights_sha256="0" * 64)
        )

    def test_final_merge_weights_inconsistency_rejected(self) -> None:
        self.check_rejects(
            lambda m: m["final_merge"]["expected_output"].update(
                weights_sha256="1" * 64
            )
        )

    def test_candidate_filled_pin_with_stale_pending_flag_rejected(self) -> None:
        def half_filled(manifest: dict) -> None:
            produced = manifest["candidate_stage"]["produced"]
            produced["adapter_config_sha256"] = "2" * 64
            produced["adapter_weights_sha256"] = "3" * 64
            # pending_fill stays True: the pin fill must clear the marker.

        self.check_rejects(half_filled)

    def test_candidate_garbage_pin_rejected(self) -> None:
        self.check_rejects(
            lambda m: m["candidate_stage"]["produced"].update(
                adapter_weights_sha256="not-a-sha"
            )
        )

    def test_candidate_row_count_change_rejected(self) -> None:
        self.check_rejects(
            lambda m: m["candidate_stage"]["dataset"].update(rows=1520)
        )

    def test_wrong_model_revision_rejected(self) -> None:
        self.check_rejects(lambda m: m["model"].update(revision="deadbeef"))

    def test_five_stage_manifest_rejected(self) -> None:
        self.check_rejects(lambda m: m["stages"].pop())


class TestReceiptPinsAgree(unittest.TestCase):
    def test_receipt_dataset_pins_equal_manifest(self) -> None:
        by_name = {
            Path(row["dataset"]["file"]).name: row["dataset"]
            for row in MANIFEST["stages"]
        }
        self.assertEqual(set(cd.LINEAGE_DATASETS), set(by_name))
        for name, (digest, rows) in cd.LINEAGE_DATASETS.items():
            self.assertEqual(digest, by_name[name]["sha256"], name)
            self.assertEqual(rows, by_name[name]["rows"], name)

    def test_receipt_trainer_pins_equal_manifest(self) -> None:
        for relative, block in MANIFEST["trainers"].items():
            key = relative.removeprefix("scripts/")
            self.assertEqual(cd.LINEAGE_TRAINERS[key], block["sha256"], relative)
        self.assertEqual(
            cd.LINEAGE_TRAINERS["merge_adapter.py"], MANIFEST["merger"]["sha256"]
        )

    def test_receipt_root_pins_equal_manifest(self) -> None:
        self.assertEqual(
            cd.LINEAGE_ROOT.relative_to(ROOT).as_posix(),
            MANIFEST["root_adapter"]["vendored_path"],
        )
        for name, (digest, size) in cd.LINEAGE_ROOT_FILES.items():
            block = MANIFEST["root_adapter"]["files"][name]
            self.assertEqual(digest, block["sha256"], name)
            self.assertEqual(size, block["size"], name)

    def test_smoke_wires_verify_inputs(self) -> None:
        source = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
        self.assertIn('"rebuild_lineage.py"), "--verify-inputs"', source)


class TestVerifyInputsIntegration(unittest.TestCase):
    def test_verify_inputs_passes_on_the_real_package(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(EXP / "scripts" / "rebuild_lineage.py"),
                "--verify-inputs",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertIs(payload["ok"], True)
        self.assertEqual(payload["mode"], "verify_inputs")
        self.assertEqual(
            payload["checked"],
            {"datasets": 7, "merger": 1, "root_files": 6, "trainers": 4},
        )

    def test_verify_inputs_function_counts(self) -> None:
        manifest = rl.load_manifest()
        checked = rl.verify_inputs(manifest)
        self.assertEqual(
            checked, {"datasets": 7, "merger": 1, "root_files": 6, "trainers": 4}
        )


if __name__ == "__main__":
    unittest.main()
