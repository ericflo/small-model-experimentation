from __future__ import annotations

import json
import os
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

    def test_canonical_publication_recovers_only_bundle_and_sidecar_prefixes(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            bundle = root / "initialization_seed7411.pt"
            sidecar = bundle.with_suffix(".pt.json")
            tracked = root / "tracked_initialization_seed7411.json"

            def fail_after_bundle(stage: str) -> None:
                if stage == "bundle_installed":
                    raise RuntimeError("injected crash after bundle")

            canonical_patches = (
                mock.patch.object(initialization, "_canonical_bundle_path", return_value=bundle),
                mock.patch.object(initialization, "_tracked_receipt_path", return_value=tracked),
            )
            with canonical_patches[0], canonical_patches[1], mock.patch.object(
                initialization, "_publication_checkpoint", side_effect=fail_after_bundle
            ):
                with self.assertRaisesRegex(RuntimeError, "injected crash after bundle"):
                    prepare_initialization_bundle(self.config, 7411, bundle)
            self.assertTrue(bundle.is_file())
            self.assertFalse(sidecar.exists())
            self.assertFalse(tracked.exists())

            def fail_after_sidecar(stage: str) -> None:
                if stage == "sidecar_installed":
                    raise RuntimeError("injected crash after sidecar")

            with mock.patch.object(
                initialization, "_canonical_bundle_path", return_value=bundle
            ), mock.patch.object(
                initialization, "_tracked_receipt_path", return_value=tracked
            ), mock.patch.object(
                initialization, "_publication_checkpoint", side_effect=fail_after_sidecar
            ):
                with self.assertRaisesRegex(RuntimeError, "injected crash after sidecar"):
                    prepare_initialization_bundle(self.config, 7411, bundle)
            self.assertTrue(sidecar.is_file())
            self.assertFalse(tracked.exists())

            with mock.patch.object(
                initialization, "_canonical_bundle_path", return_value=bundle
            ), mock.patch.object(
                initialization, "_tracked_receipt_path", return_value=tracked
            ):
                receipt = prepare_initialization_bundle(self.config, 7411, bundle)
                with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                    prepare_initialization_bundle(self.config, 7411, bundle)
            self.assertEqual(sidecar.read_bytes(), tracked.read_bytes())
            self.assertEqual(
                json.loads(sidecar.read_text(encoding="utf-8")), receipt
            )
            self.assertNotEqual(sidecar.stat().st_ino, tracked.stat().st_ino)
            for path in (bundle, sidecar, tracked):
                self.assertEqual(path.stat().st_nlink, 1)

    def test_tracked_mirror_is_the_final_commit_marker_after_injected_crash(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            bundle = root / "initialization_seed7411.pt"
            tracked = root / "tracked_initialization_seed7411.json"

            def fail_after_commit(stage: str) -> None:
                if stage == "tracked_mirror_installed":
                    raise RuntimeError("injected crash after commit")

            with mock.patch.object(
                initialization, "_canonical_bundle_path", return_value=bundle
            ), mock.patch.object(
                initialization, "_tracked_receipt_path", return_value=tracked
            ), mock.patch.object(
                initialization, "_publication_checkpoint", side_effect=fail_after_commit
            ):
                with self.assertRaisesRegex(RuntimeError, "injected crash after commit"):
                    prepare_initialization_bundle(self.config, 7411, bundle)
            self.assertTrue(tracked.is_file())
            with mock.patch.object(
                initialization, "_canonical_bundle_path", return_value=bundle
            ), mock.patch.object(
                initialization, "_tracked_receipt_path", return_value=tracked
            ):
                with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                    prepare_initialization_bundle(self.config, 7411, bundle)

    def test_prefix_collision_hardlink_and_symlink_aliases_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)

            collision = root / "collision.pt"

            def fail_after_bundle(stage: str) -> None:
                if stage == "bundle_installed":
                    raise RuntimeError("injected bundle prefix")

            with mock.patch.object(
                initialization, "_publication_checkpoint", side_effect=fail_after_bundle
            ):
                with self.assertRaisesRegex(RuntimeError, "injected bundle prefix"):
                    prepare_initialization_bundle(self.config, 7411, collision)
            collision.with_suffix(".pt.json").write_bytes(b"collision\n")
            with self.assertRaisesRegex(RuntimeError, "sidecar mismatch"):
                prepare_initialization_bundle(self.config, 7411, collision)

            hardlinked = root / "hardlinked.pt"
            with mock.patch.object(
                initialization, "_publication_checkpoint", side_effect=fail_after_bundle
            ):
                with self.assertRaisesRegex(RuntimeError, "injected bundle prefix"):
                    prepare_initialization_bundle(self.config, 7411, hardlinked)
            os.link(hardlinked, root / "hardlinked-alias.pt")
            with self.assertRaisesRegex(RuntimeError, "stable canonical file"):
                prepare_initialization_bundle(self.config, 7411, hardlinked)

            dangling = root / "dangling.pt"
            dangling.symlink_to(root / "missing-target.pt")
            with self.assertRaisesRegex(RuntimeError, "stable canonical file"):
                prepare_initialization_bundle(self.config, 7411, dangling)

            real_parent = root / "real-parent"
            real_parent.mkdir()
            alias_parent = root / "alias-parent"
            alias_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "without aliases"):
                prepare_initialization_bundle(
                    self.config, 7411, alias_parent / "through-alias.pt"
                )
            self.assertFalse((real_parent / "through-alias.pt").exists())

    def test_writer_failure_cleans_private_stage_and_stale_debris_is_harmless(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            failed = root / "writer-failure.bin"

            def fail_after_prefix(handle) -> None:
                handle.write(b"partial bytes that must never become canonical")
                raise RuntimeError("injected initialization writer failure")

            with self.assertRaisesRegex(
                RuntimeError, "injected initialization writer failure"
            ):
                initialization._publish_new(failed, fail_after_prefix)
            self.assertFalse(os.path.lexists(failed))
            self.assertEqual(list(root.glob(".publish-*.tmp")), [])

            stale = root / ".publish-stale-process.tmp"
            stale.write_bytes(b"abandoned private stage")
            bundle = root / "debris-does-not-block.pt"
            prepare_initialization_bundle(self.config, 7411, bundle)
            sidecar = bundle.with_suffix(".pt.json")
            self.assertEqual(stale.read_bytes(), b"abandoned private stage")
            self.assertEqual(bundle.stat().st_nlink, 1)
            self.assertEqual(sidecar.stat().st_nlink, 1)

    def test_noncanonical_or_outside_output_is_rejected_before_write(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            for alias in (
                f"{root}/nested/../alias.pt",
                f"//{root.as_posix().lstrip('/')}/alias.pt",
            ):
                with self.subTest(alias=alias), self.assertRaisesRegex(
                    RuntimeError, "canonical lexical path"
                ):
                    prepare_initialization_bundle(self.config, 7411, alias)
            self.assertFalse((root / "alias.pt").exists())

        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "outside.pt"
            with self.assertRaisesRegex(RuntimeError, "outside repository workspace"):
                prepare_initialization_bundle(self.config, 7411, outside)
            self.assertFalse(outside.exists())

    def test_loader_rejects_duplicate_or_nonfinite_sidecar_json(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            for name, malformed in (
                ("duplicate.pt", b'{"x":1,"x":2}\n'),
                ("nonfinite.pt", b'{"x":NaN}\n'),
            ):
                with self.subTest(name=name):
                    path = root / name
                    prepare_initialization_bundle(self.config, 7411, path)
                    path.with_suffix(".pt.json").write_bytes(malformed)
                    with self.assertRaisesRegex(RuntimeError, "strict UTF-8 JSON"):
                        load_initialization_bundle(self.config, 7411, path)


if __name__ == "__main__":
    unittest.main()
