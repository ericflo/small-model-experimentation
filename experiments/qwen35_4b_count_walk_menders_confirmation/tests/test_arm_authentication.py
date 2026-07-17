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


BENCH = load_module("cwmc_run_benchmark_auth", "run_benchmark.py")

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
            with self.assertRaisesRegex(ValueError, "tree changed for base"):
                BENCH.authenticate_model_tree("base", model)

    def test_tampered_tree_pin_refuses_the_real_pin_holder(self) -> None:
        # Simulates the boundary drill: a single tampered hex in the frozen
        # tree constant refuses the arm even when every byte on disk is the
        # arm's own.
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            manifest = BENCH.merged_tree_manifest(model)
            observed = BENCH.tree_manifest_sha256(manifest)
            tampered = ("0" if observed[0] != "0" else "1") + observed[1:]
            BENCH.FROZEN_TREE_SHA256["base"] = tampered
            with self.assertRaisesRegex(ValueError, "tree changed for base"):
                BENCH.authenticate_model_tree("base", model)

    def test_weights_mismatch_refuses_after_tree_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            manifest = BENCH.merged_tree_manifest(model)
            BENCH.FROZEN_TREE_SHA256["base"] = BENCH.tree_manifest_sha256(manifest)
            # The frozen weights pin still names the real 9GB weights, which
            # the fake cannot match.
            with self.assertRaisesRegex(ValueError, "weights changed for base"):
                BENCH.authenticate_model_tree("base", model)

    def test_base_reserialization_receipt_pin_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            manifest = BENCH.merged_tree_manifest(model)
            files = {row["name"]: row for row in manifest}
            BENCH.FROZEN_TREE_SHA256["base"] = BENCH.tree_manifest_sha256(manifest)
            BENCH.FROZEN_WEIGHTS_SHA256["base"] = files["model.safetensors"]["sha256"]
            BENCH.WEIGHTS_SIZE_BYTES = files["model.safetensors"]["size"]
            with self.assertRaisesRegex(ValueError, "reserialization receipt"):
                BENCH.authenticate_model_tree("base", model)

    def test_trained_arm_receipt_payload_mismatch_refuses(self) -> None:
        # The committed lifecycle-27 receipt describes the real composite
        # path, so a fake path with matching tree/weights pins still fails
        # the payload equality clause.
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            manifest = BENCH.merged_tree_manifest(model)
            files = {row["name"]: row for row in manifest}
            BENCH.FROZEN_TREE_SHA256["count_walk"] = BENCH.tree_manifest_sha256(manifest)
            BENCH.FROZEN_WEIGHTS_SHA256["count_walk"] = (
                files["model.safetensors"]["sha256"]
            )
            BENCH.WEIGHTS_SIZE_BYTES = files["model.safetensors"]["size"]
            with self.assertRaisesRegex(ValueError, "does not describe"):
                BENCH.authenticate_model_tree("count_walk", model)


class ZeroRootProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._receipt = BENCH.ZERO_ROOT_PARENT_MERGE_RECEIPT
        self._sha = BENCH.ZERO_ROOT_PARENT_MERGE_RECEIPT_SHA256

    def tearDown(self) -> None:
        BENCH.ZERO_ROOT_PARENT_MERGE_RECEIPT = self._receipt
        BENCH.ZERO_ROOT_PARENT_MERGE_RECEIPT_SHA256 = self._sha

    def test_committed_receipt_authenticates_the_frozen_parent(self) -> None:
        BENCH.require_zero_root_parent_provenance(
            BENCH.FROZEN_MODEL_PATHS["zero_root_parent"]
        )

    def test_wrong_model_path_refuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not describe the frozen"):
            BENCH.require_zero_root_parent_provenance(
                BENCH.FROZEN_MODEL_PATHS["base"]
            )

    def test_tampered_receipt_bytes_refuse_on_the_sha_pin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = json.loads(self._receipt.read_text(encoding="utf-8"))
            payload["weights_size_bytes"] = 1
            tampered = Path(directory) / "merge.json"
            tampered.write_text(json.dumps(payload), encoding="utf-8")
            BENCH.ZERO_ROOT_PARENT_MERGE_RECEIPT = tampered
            with self.assertRaisesRegex(ValueError, "absent or changed"):
                BENCH.require_zero_root_parent_provenance(
                    BENCH.FROZEN_MODEL_PATHS["zero_root_parent"]
                )

    def test_tampered_payload_refuses_even_with_matching_pin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = json.loads(self._receipt.read_text(encoding="utf-8"))
            payload["weights_size_bytes"] = 1
            tampered = Path(directory) / "merge.json"
            tampered.write_text(json.dumps(payload), encoding="utf-8")
            BENCH.ZERO_ROOT_PARENT_MERGE_RECEIPT = tampered
            BENCH.ZERO_ROOT_PARENT_MERGE_RECEIPT_SHA256 = BENCH.sha256_file(tampered)
            with self.assertRaisesRegex(ValueError, "does not describe the frozen"):
                BENCH.require_zero_root_parent_provenance(
                    BENCH.FROZEN_MODEL_PATHS["zero_root_parent"]
                )


class ProvenanceCopyTests(unittest.TestCase):
    def test_all_four_copies_match_pins_and_sources(self) -> None:
        BENCH.require_provenance_copies()

    def test_tampered_copy_refuses(self) -> None:
        original = dict(BENCH.PROVENANCE_COPIES)
        try:
            key = "data/provenance/prior_event_seed78163_summary.json"
            source, _ = BENCH.PROVENANCE_COPIES[key]
            BENCH.PROVENANCE_COPIES[key] = (source, "0" * 64)
            with self.assertRaisesRegex(ValueError, "absent or changed"):
                BENCH.require_provenance_copies()
        finally:
            BENCH.PROVENANCE_COPIES.clear()
            BENCH.PROVENANCE_COPIES.update(original)


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

    def test_receipt_seed_must_be_one_of_the_frozen_four(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            path = Path(directory) / "event.json"
            event = self._event(model, BENCH.PRIOR_EVENT["seed"])
            path.write_text(json.dumps(event), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen four"):
                BENCH.load_event(path, model, BENCH.PRIOR_EVENT["seed"])

    def test_wrong_seed_inside_receipt_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = make_fake_composite(Path(directory))
            path = Path(directory) / "event.json"
            event = self._event(model, BENCH.SEED_ORDER[1])
            path.write_text(json.dumps(event), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "failed authentication"):
                BENCH.load_event(path, model, BENCH.SEED_ORDER[0])


if __name__ == "__main__":
    unittest.main()
