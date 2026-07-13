from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import initialization  # noqa: E402
from src.config import load_config  # noqa: E402
from src.initialization import (  # noqa: E402
    build_shared_state,
    initialization_seed,
    load_initialization_bundle,
    prepare_initialization_bundle,
    tensor_manifest,
)


class SharedInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "smoke.yaml")

    def setUp(self) -> None:
        lineage = {
            "path": "synthetic/design_receipt.json",
            "sha256": "a" * 64,
            "receipt_identity_sha256": "b" * 64,
            "status": "DESIGN_FROZEN",
            "phase": "design_boundary",
        }
        self.design_validation = mock.patch.object(
            initialization,
            "validate_design_receipt",
            return_value={"status": "DESIGN_FROZEN"},
        )
        self.design_lineage = mock.patch.object(
            initialization, "design_lineage", return_value=lineage
        )
        self.design_validation.start()
        self.design_lineage.start()

    def tearDown(self) -> None:
        self.design_lineage.stop()
        self.design_validation.stop()

    def test_seeded_shared_state_is_deterministic_and_capacity_independent(self) -> None:
        self.assertEqual(initialization_seed(7411), initialization_seed(7411))
        self.assertNotEqual(initialization_seed(7411), initialization_seed(7412))
        first = build_shared_state(self.config, 7411)
        second = build_shared_state(self.config, 7411)
        third = build_shared_state(self.config, 7412)
        first_manifest, first_digest = tensor_manifest(first)
        second_manifest, second_digest = tensor_manifest(second)
        _, third_digest = tensor_manifest(third)
        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_digest, second_digest)
        self.assertNotEqual(first_digest, third_digest)
        self.assertGreater(len(first_manifest), 0)

    def test_saved_bundles_reopen_to_bit_identical_tensors(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            first_path = root / "first.pt"
            second_path = root / "second.pt"
            first_receipt = prepare_initialization_bundle(self.config, 7411, first_path)
            second_receipt = prepare_initialization_bundle(self.config, 7411, second_path)
            first_state, reopened_first = load_initialization_bundle(self.config, 7411, first_path)
            second_state, reopened_second = load_initialization_bundle(self.config, 7411, second_path)

            self.assertEqual(
                first_receipt["metadata"]["tensor_values_sha256"],
                second_receipt["metadata"]["tensor_values_sha256"],
            )
            self.assertEqual(
                reopened_first["metadata"]["tensor_manifest"],
                reopened_second["metadata"]["tensor_manifest"],
            )
            first_manifest, first_digest = tensor_manifest(first_state)
            second_manifest, second_digest = tensor_manifest(second_state)
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_digest, second_digest)
            self.assertEqual(first_digest, first_receipt["metadata"]["tensor_values_sha256"])
            self.assertEqual(first_receipt["metadata"]["model_seed"], 7411)
            self.assertEqual(
                first_receipt["metadata"]["config_sha256"],
                reopened_first["metadata"]["config_sha256"],
            )
            self.assertEqual(
                first_receipt["metadata"]["source_contract_sha256"],
                reopened_first["metadata"]["source_contract_sha256"],
            )

    def test_tensor_or_metadata_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            tensor_tamper = root / "tensor_tamper.pt"
            prepare_initialization_bundle(self.config, 7411, tensor_tamper)
            tensor_payload = torch.load(tensor_tamper, map_location="cpu", weights_only=True)
            tensors = tensor_payload["state"]["state_initializer"]
            first_name = sorted(tensors)[0]
            tensors[first_name].reshape(-1)[0] += 1.0
            torch.save(tensor_payload, tensor_tamper)
            self._refresh_sidecar_hash(tensor_tamper)
            with self.assertRaisesRegex(RuntimeError, "tensor digest mismatch"):
                load_initialization_bundle(self.config, 7411, tensor_tamper)

            metadata_tamper = root / "metadata_tamper.pt"
            prepare_initialization_bundle(self.config, 7411, metadata_tamper)
            metadata_payload = torch.load(metadata_tamper, map_location="cpu", weights_only=True)
            metadata_payload["metadata"]["model_seed"] = 7412
            torch.save(metadata_payload, metadata_tamper)
            self._refresh_sidecar_hash(metadata_tamper)
            with self.assertRaisesRegex(RuntimeError, "model_seed mismatch"):
                load_initialization_bundle(self.config, 7411, metadata_tamper)

    @staticmethod
    def _refresh_sidecar_hash(path: Path) -> None:
        sidecar_path = path.with_suffix(path.suffix + ".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["bundle_sha256"] = initialization._file_sha256(path)
        sidecar["receipt_identity_sha256"] = initialization._canonical_sha256(
            {
                key: value
                for key, value in sidecar.items()
                if key != "receipt_identity_sha256"
            }
        )
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_bundle_creation_refuses_overwrite_and_wrong_seed_reopen(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "bundle.pt"
            prepare_initialization_bundle(self.config, 7411, path)
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                prepare_initialization_bundle(self.config, 7411, path)
            with self.assertRaisesRegex(RuntimeError, "model_seed mismatch"):
                load_initialization_bundle(self.config, 7412, path)


if __name__ == "__main__":
    unittest.main()
