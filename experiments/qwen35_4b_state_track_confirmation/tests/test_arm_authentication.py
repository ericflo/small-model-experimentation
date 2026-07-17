import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BENCH = load_module("stc_run_benchmark_auth", "run_benchmark.py")

FAMILIES = sorted(BENCH.PUBLIC_FAMILIES)


def make_fake_composite(directory: Path) -> Path:
    model = directory / "model"
    model.mkdir()
    for name in sorted(BENCH.MERGED_FILE_NAMES):
        (model / name).write_text(f"fake {name}\n", encoding="utf-8")
    return model


class MergedTreeManifestTests(unittest.TestCase):
    def test_complete_flat_tree_is_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            manifest = BENCH.merged_tree_manifest(model)
            self.assertEqual(
                [row["name"] for row in manifest],
                sorted(BENCH.MERGED_FILE_NAMES),
            )

    def test_missing_file_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            (model / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "file set changed"):
                BENCH.merged_tree_manifest(model)

    def test_unexpected_file_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            (model / "extra.bin").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file set changed"):
                BENCH.merged_tree_manifest(model)

    def test_nested_entry_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            (model / "model.safetensors").unlink()
            (model / "model.safetensors").mkdir()
            with self.assertRaisesRegex(ValueError, "symlink or nested"):
                BENCH.merged_tree_manifest(model)

    def test_missing_directory_refuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a real directory"):
            BENCH.merged_tree_manifest(Path("/nonexistent/composite"))


class AuthenticateModelTreeTests(unittest.TestCase):
    """Failure paths on small fake composites; no 9GB hashing here."""

    def setUp(self) -> None:
        self._trees = dict(BENCH.FROZEN_TREE_SHA256)
        self._weights = dict(BENCH.FROZEN_WEIGHTS_SHA256)
        self._size = BENCH.WEIGHTS_SIZE_BYTES

    def tearDown(self) -> None:
        BENCH.FROZEN_TREE_SHA256.clear()
        BENCH.FROZEN_TREE_SHA256.update(self._trees)
        BENCH.FROZEN_WEIGHTS_SHA256.clear()
        BENCH.FROZEN_WEIGHTS_SHA256.update(self._weights)
        BENCH.WEIGHTS_SIZE_BYTES = self._size

    def test_tree_hash_mismatch_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            with self.assertRaisesRegex(ValueError, "tree changed for count_walk"):
                BENCH.authenticate_model_tree("count_walk", model)

    def test_tampered_tree_pin_refuses_the_real_pin_holder(self) -> None:
        # The boundary drill: a single tampered hex in the frozen tree
        # constant refuses the arm even when every byte on disk is its own.
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            manifest = BENCH.merged_tree_manifest(model)
            observed = BENCH.tree_manifest_sha256(manifest)
            tampered = ("0" if observed[0] != "0" else "1") + observed[1:]
            BENCH.FROZEN_TREE_SHA256["state_track"] = tampered
            with self.assertRaisesRegex(ValueError, "tree changed for state_track"):
                BENCH.authenticate_model_tree("state_track", model)

    def test_weights_mismatch_refuses_after_tree_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            manifest = BENCH.merged_tree_manifest(model)
            BENCH.FROZEN_TREE_SHA256["count_walk"] = BENCH.tree_manifest_sha256(manifest)
            with self.assertRaisesRegex(ValueError, "weights changed for count_walk"):
                BENCH.authenticate_model_tree("count_walk", model)

    def test_arm_provenance_payload_mismatch_refuses(self) -> None:
        # The in-cell provenance copy describes the real composite path, so a
        # fake path with matching tree/weights pins still fails the payload
        # equality clause (inner receipt or path).
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            manifest = BENCH.merged_tree_manifest(model)
            files = {row["name"]: row for row in manifest}
            BENCH.FROZEN_TREE_SHA256["count_walk"] = BENCH.tree_manifest_sha256(manifest)
            BENCH.FROZEN_WEIGHTS_SHA256["count_walk"] = (
                files["model.safetensors"]["sha256"]
            )
            BENCH.WEIGHTS_SIZE_BYTES = files["model.safetensors"]["size"]
            with self.assertRaisesRegex(ValueError, "does not describe the frozen arm"):
                BENCH.authenticate_model_tree("count_walk", model)


class ArmProvenanceTests(unittest.TestCase):
    def test_both_arms_authenticate_against_the_in_cell_copy(self) -> None:
        for label in BENCH.MODEL_ORDER:
            note = BENCH.require_arm_provenance(label, BENCH.FROZEN_MODEL_PATHS[label])
            self.assertIn("in-cell pin", note)

    def test_wrong_model_path_refuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not describe the frozen arm"):
            BENCH.require_arm_provenance(
                "count_walk", BENCH.FROZEN_MODEL_PATHS["state_track"]
            )

    def test_tampered_copy_sha_pin_refuses(self) -> None:
        original = {
            label: dict(block) for label, block in BENCH.ARM_PROVENANCE.items()
        }
        try:
            BENCH.ARM_PROVENANCE["state_track"]["merge_receipt_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "absent or changed"):
                BENCH.require_arm_provenance(
                    "state_track", BENCH.FROZEN_MODEL_PATHS["state_track"]
                )
        finally:
            for label, block in original.items():
                BENCH.ARM_PROVENANCE[label].clear()
                BENCH.ARM_PROVENANCE[label].update(block)

    def test_all_provenance_copies_and_prior_summary_verify(self) -> None:
        notes = BENCH.require_provenance_copies()
        self.assertEqual(set(notes), {"count_walk", "state_track", "prior_event"})

    def test_prior_summary_pin_tamper_refuses(self) -> None:
        original = dict(BENCH.PRIOR_EVENT)
        try:
            BENCH.PRIOR_EVENT["summary_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "absent or changed"):
                BENCH.require_provenance_copies()
        finally:
            BENCH.PRIOR_EVENT.clear()
            BENCH.PRIOR_EVENT.update(original)


class PriorReferenceTests(unittest.TestCase):
    def test_prior_summary_authenticates_as_installed_transfer(self) -> None:
        prior = BENCH.load_prior_reference()
        parent = prior["scores"]["count_walk"]["aggregate"]
        candidate = prior["scores"]["state_track"]["aggregate"]
        self.assertGreater(candidate, parent)
        self.assertAlmostEqual(candidate - parent, 0.02557142857142858, places=10)
        self.assertEqual(
            prior["benchmark_implementation"], BENCH.PRIOR_IMPLEMENTATION
        )

    def test_prior_report_records_paired_delta_never_counted(self) -> None:
        prior = BENCH.load_prior_reference()
        report = BENCH.prior_event_report(prior["scores"])
        self.assertEqual(report["seed"], 78169)
        self.assertFalse(report["counted_in_verdict"])
        self.assertTrue(report["installed_transfer"])
        self.assertAlmostEqual(report["paired_delta"], 0.02557142857142858, places=10)


class GatewayReceiptSchemaTests(unittest.TestCase):
    def _event(self, model: Path, seed: int) -> dict:
        return {
            "schema_version": 1,
            "stage": "menagerie_aggregate_gateway",
            "tier": "medium",
            "think_budget": 1024,
            "seed": seed,
            "backend": "qwen_vllm",
            "model": str(model),
            "model_merge_receipt_sha256": BENCH.sha256_file(
                model / "merge_receipt.json"
            ),
            "benchmark_runner_sha256": "x",
            "benchmark_source_inventory_sha256": "y",
            "benchmark_source_file_count": 3,
            "aggregate": 0.5,
            "per_family": {family: 0.5 for family in FAMILIES},
            "within_budget": True,
            "wall_seconds": 1.0,
        }

    def test_receipt_authentication_rejects_bad_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            path = Path(directory) / "event.json"
            seed = BENCH.SEED_ORDER[0]

            event = self._event(model, seed)
            path.write_text(json.dumps(event), encoding="utf-8")
            self.assertEqual(BENCH.load_event(path, model, seed)["aggregate"], 0.5)

            for mutate in (
                lambda e: e.__setitem__("per_family", {**e["per_family"], "menders": math.nan}),
                lambda e: e.__setitem__("aggregate", math.inf),
                lambda e: e.__setitem__("aggregate", 1.5),
                lambda e: e.__setitem__("extra_key", 1),
                lambda e: e.__setitem__("within_budget", "yes"),
                lambda e: e.__setitem__("wall_seconds", -1.0),
                lambda e: e.__setitem__("wall_seconds", math.nan),
                lambda e: e.__setitem__("tier", "quick"),
                lambda e: e.__setitem__("think_budget", 8192),
                lambda e: e.__setitem__("backend", "hf"),
                lambda e: e.__setitem__("model_merge_receipt_sha256", "0" * 64),
                lambda e: e.__setitem__(
                    "per_family",
                    {f: 0.5 for f in FAMILIES if f != "menders"},
                ),
            ):
                event = self._event(model, seed)
                mutate(event)
                path.write_text(json.dumps(event), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "failed authentication"):
                    BENCH.load_event(path, model, seed)

    def test_receipt_seed_must_be_one_of_the_frozen_six(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            path = Path(directory) / "event.json"
            event = self._event(model, BENCH.PRIOR_EVENT["seed"])
            path.write_text(json.dumps(event), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen six"):
                BENCH.load_event(path, model, BENCH.PRIOR_EVENT["seed"])

    def test_over_budget_arm_is_recorded_not_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            path = Path(directory) / "event.json"
            event = self._event(model, BENCH.SEED_ORDER[0])
            event["within_budget"] = False
            path.write_text(json.dumps(event), encoding="utf-8")
            loaded = BENCH.load_event(path, model, BENCH.SEED_ORDER[0])
            self.assertFalse(loaded["within_budget"])


if __name__ == "__main__":
    unittest.main()
