"""Unit tests for the standalone lineage package (owner directive 2026-07-15).

The standalone-reproducibility gate requires this eval-only cell to carry
the evaluated composite's complete model-reproduction package: the
hygiene_explore lineage's six ordered SFT dataset copies, the fixed-seed
per-stage recipe manifest, the three trainer variants plus the merger
copied in, and the vendored frozen root adapter (a HARD provenance
boundary — no committed creation receipt exists for it). These tests hold
the copies byte-anchored to their recorded sha256s (and byte-identical to
the confirmation cell's audited package), the manifest schema and
warm-start chaining sound (via ``rebuild_lineage.load_manifest``,
including fail-closed negatives on tampered manifests), the final-merge
pins identical to the eval's composite pins, and ``rebuild_lineage.py
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

import eval_local_vllm as ev  # noqa: E402
import gen_local_gate as gg  # noqa: E402
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
    def test_dataset_copies_match_manifest_pins(self) -> None:
        self.assertEqual(len(MANIFEST["stages"]), 6)
        for row in MANIFEST["stages"]:
            name = Path(row["dataset"]["file"]).name
            copy_path = EXP / "data" / "lineage" / name
            self.assertTrue(copy_path.is_file(), name)
            self.assertEqual(sha256_file(copy_path), row["dataset"]["sha256"], name)
            rows = sum(
                1
                for line in copy_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            self.assertEqual(rows, row["dataset"]["rows"], name)

    def test_six_stage_copies_are_byte_identical_to_the_confirmation_cell(self) -> None:
        for row in MANIFEST["stages"]:
            name = Path(row["dataset"]["file"]).name
            ours = EXP / "data" / "lineage" / name
            theirs = CONFIRMATION / name
            self.assertEqual(sha256_file(ours), sha256_file(theirs), name)

    def test_trainer_and_merger_copies_match_recorded_shas(self) -> None:
        for relative, block in MANIFEST["trainers"].items():
            path = EXP / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(sha256_file(path), block["sha256"], relative)
        merger = EXP / MANIFEST["merger"]["path"]
        self.assertEqual(sha256_file(merger), MANIFEST["merger"]["sha256"])

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
        self.assertEqual(
            MANIFEST["root_adapter"]["weights_sha256"],
            "ad2ef4fae785debedf5e50932a79bda97869d3efc212f53d48ccb04c59e25d21",
        )
        self.assertEqual(
            MANIFEST["root_adapter"]["config_sha256"],
            "cd764ae869b8a55526e283dd133e1940896b428839c6abf2e55d6ae2a0b32635",
        )

    def test_root_adapter_is_vendored_inside_this_cell(self) -> None:
        vendored = MANIFEST["root_adapter"]["vendored_path"]
        self.assertTrue(
            vendored.startswith(
                "large_artifacts/qwen35_4b_repair_verifier_signal_probe/"
            )
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

    def test_stage_order_names_and_seeds(self) -> None:
        names = [row["name"] for row in MANIFEST["stages"]]
        self.assertEqual(
            names,
            [
                "replay_refresh", "designed160", "close_xi",
                "replay_after_close", "designed_fresh", "hygiene_explore",
            ],
        )
        self.assertEqual([row["seed"] for row in MANIFEST["stages"]], [42, 43, 44, 47, 51, 55])

    def test_warm_start_chain_is_root_then_previous_stage(self) -> None:
        self.assertEqual(MANIFEST["stages"][0]["warm_start"], "root_adapter")
        for index, row in enumerate(MANIFEST["stages"][1:], start=2):
            self.assertEqual(row["warm_start"], f"stage {index - 1}")

    def test_final_merge_targets_the_evaluated_composite(self) -> None:
        final = MANIFEST["final_merge"]
        self.assertEqual(
            final["expected_output"]["weights_sha256"],
            gg.EXPECTED_WEIGHTS_SHA256,
        )
        self.assertEqual(
            final["expected_output"]["published_tree_sha256"],
            gg.EXPECTED_TREE_SHA256,
        )
        self.assertEqual(
            final["expected_output"]["content_files_sha256"]["model.safetensors"],
            ev.EXPECTED_WEIGHTS_SHA256,
        )
        self.assertIn("merge_receipt.json", final["expected_output"]["tree_sha_note"])
        self.assertIn(gg.MODEL_REVISION, final["base_model"])

    def test_produced_hashes_are_verification_aids(self) -> None:
        self.assertIs(
            MANIFEST["determinism"]["produced_hashes_are_verification_aids"], True
        )
        for row in MANIFEST["stages"]:
            self.assertIs(row["produced"]["verification_aid"], True)


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

    def test_wrong_experiment_id_rejected(self) -> None:
        self.check_rejects(
            lambda m: m.update(experiment_id="qwen35_4b_goal_gate_confirmation")
        )

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

    def test_unregistered_trainer_rejected(self) -> None:
        self.check_rejects(
            lambda m: m["stages"][0].update(trainer="scripts/other_trainer.py")
        )

    def test_missing_top_level_key_rejected(self) -> None:
        self.check_rejects(lambda m: m.pop("root_adapter"))

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

    def test_wrong_model_revision_rejected(self) -> None:
        self.check_rejects(lambda m: m["model"].update(revision="deadbeef"))

    def test_five_stage_manifest_rejected(self) -> None:
        self.check_rejects(lambda m: m["stages"].pop())


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
            {"datasets": 6, "merger": 1, "root_files": 6, "trainers": 3},
        )

    def test_verify_inputs_function_counts(self) -> None:
        manifest = rl.load_manifest()
        checked = rl.verify_inputs(manifest)
        self.assertEqual(
            checked, {"datasets": 6, "merger": 1, "root_files": 6, "trainers": 3}
        )

    def test_smoke_wires_verify_inputs(self) -> None:
        source = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
        self.assertIn('"rebuild_lineage.py"), "--verify-inputs"', source)

    def test_design_receipt_pins_rebuild_lineage_code(self) -> None:
        self.assertIn("rebuild_lineage", gg.CODE_FILES)
        self.assertEqual(
            gg.CODE_FILES["rebuild_lineage"].name, "rebuild_lineage.py"
        )
        self.assertIn("external_merger", gg.CODE_FILES)


if __name__ == "__main__":
    unittest.main()
