from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_pipeline  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data_pipeline import (  # noqa: E402
    build_datasets,
    canonical_rows_receipt,
    load_contrast_access_ledger,
    read_jsonl,
    record_contrast_access,
    validate_data_manifest,
)


EXPECTED_SPLITS = {
    "train",
    "validation",
    "depth_extrapolation",
    "joint_holdout",
    "contrast_validation",
    "contrast_depth",
    "contrast_joint",
}


def load_archiver():
    spec = importlib.util.spec_from_file_location(
        "capacity_failed_attempt_archiver_for_ledger",
        ROOT / "scripts" / "archive_failed_attempt.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed-attempt archiver cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FreshDataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "smoke.yaml")
        cls.default_config = load_config(ROOT / "configs" / "default.yaml")

    def setUp(self) -> None:
        self.design = {
            "path": "synthetic/design_receipt.json",
            "sha256": "a" * 64,
            "receipt_identity_sha256": "b" * 64,
            "status": "DESIGN_FROZEN",
            "phase": "design_boundary",
        }
        self.design_validation = mock.patch.object(
            data_pipeline, "validate_design_receipt", return_value={"status": "DESIGN_FROZEN"}
        )
        self.design_lineage = mock.patch.object(
            data_pipeline, "design_lineage", return_value=self.design
        )
        self.design_validation.start()
        self.design_lineage.start()

    def tearDown(self) -> None:
        self.design_lineage.stop()
        self.design_validation.stop()

    @staticmethod
    def contrast_authorization(
        _directory: Path,
        checkpoint_lineages: dict[tuple[str, int], dict[str, str]],
    ) -> dict[str, str]:
        if data_pipeline.ROOT.resolve() == ROOT.resolve():
            raise AssertionError(
                "contrast authorization fixtures must use a temporary experiment root"
            )
        path = data_pipeline.ROOT / "analysis" / "stage_b_seal.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        per_seed: dict[str, dict] = {}
        for (capacity, seed), lineage in checkpoint_lineages.items():
            per_seed.setdefault(str(seed), {"checkpoint_lineages": {}})[
                "checkpoint_lineages"
            ][f"{capacity}_joint"] = lineage
        receipt = {
            "status": "STAGE_B_CONTRAST_AUTHORIZED",
            "phase": "stage_b_seal_analysis",
            "matching": {
                "status": "STAGE_B_MATCHING_VALID",
                "per_seed": per_seed,
            },
        }
        receipt["receipt_identity_sha256"] = data_pipeline._canonical_sha256(receipt)
        path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        return {
            "path": data_pipeline._repo_relative(path),
            "sha256": data_pipeline._sha256(path),
            "receipt_identity_sha256": receipt["receipt_identity_sha256"],
            "status": receipt["status"],
            "phase": receipt["phase"],
        }

    @staticmethod
    def canonical_contrast_output(
        *, capacity: str = "lora", model_seed: int = 7411
    ) -> Path:
        path = (
            data_pipeline.ROOT
            / "runs"
            / f"{capacity}_joint_seed{model_seed}_contrast"
        )
        path.mkdir(parents=True)
        return path

    def checkpoint_lineage(
        self,
        directory: Path,
        data_root: Path,
        name: str = "checkpoint",
        *,
        capacity: str = "lora",
        objective: str = "joint",
        model_seed: int = 7411,
    ) -> dict[str, str]:
        checkpoint = directory / name
        checkpoint.mkdir()
        metadata_path = checkpoint / "checkpoint.json"
        current_design = self.design
        metadata = {
            "experiment_id": self.config["experiment_id"],
            "model_id": data_pipeline.MODEL_ID,
            "model_revision": data_pipeline.MODEL_REVISION,
            "backend": "transformers",
            "capacity": capacity,
            "objective": objective,
            "model_seed": model_seed,
            "step": int(self.config["training"]["train_steps"]),
            "phase": f"{capacity}_{objective}_training",
            "config_sha256": data_pipeline.config_sha256(self.config),
            "source_contract_sha256": data_pipeline.source_contract_sha256(),
            "requirements_training_lock_sha256": data_pipeline._sha256(
                data_pipeline.REQUIREMENTS_LOCK
            ),
            "data_manifest_sha256": data_pipeline._sha256(
                data_root / "manifest.json"
            ),
            "design_receipt_sha256": current_design["sha256"],
            "design_receipt_identity_sha256": current_design[
                "receipt_identity_sha256"
            ],
        }
        metadata["checkpoint_identity_sha256"] = data_pipeline._canonical_sha256(metadata)
        metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        return {
            "path": data_pipeline._repo_relative(checkpoint),
            "metadata_sha256": data_pipeline._sha256(metadata_path),
            "checkpoint_identity_sha256": metadata["checkpoint_identity_sha256"],
        }

    def test_seven_fresh_deterministic_splits_have_no_cross_split_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            first_root = Path(first_directory)
            second_root = Path(second_directory)
            first = build_datasets(self.config, first_root)
            second = build_datasets(self.config, second_root)

            self.assertEqual(set(first["files"]), EXPECTED_SPLITS)
            self.assertEqual(first["cross_split_structural_duplicates"], 0)
            self.assertEqual(first["benchmark_files_read"], 0)
            self.assertEqual(first["data_contract_sha256"], second["data_contract_sha256"])
            self.assertEqual(
                {name: item["sha256"] for name, item in first["files"].items()},
                {name: item["sha256"] for name, item in second["files"].items()},
            )
            self.assertEqual(
                {name: item["canonical_rows"] for name, item in first["files"].items()},
                {name: item["canonical_rows"] for name, item in second["files"].items()},
            )

            fingerprints: dict[str, str] = {}
            row_ids: set[str] = set()
            for split, receipt in first["files"].items():
                rows = read_jsonl(first_root / receipt["path"])
                self.assertEqual(len(rows), receipt["rows"])
                self.assertEqual(receipt["query_kinds"]["node"], receipt["rows"] // 2)
                self.assertEqual(receipt["query_kinds"]["checksum"], receipt["rows"] // 2)
                for cell in receipt["query_kind_grid"].values():
                    self.assertEqual(cell.get("node"), cell.get("checksum"))
                for row in rows:
                    fingerprint = row["structural_fingerprint"]
                    self.assertNotIn(fingerprint, fingerprints)
                    fingerprints[fingerprint] = split
                    self.assertNotIn(row["id"], row_ids)
                    row_ids.add(row["id"])

            validate_data_manifest(self.config, first_root, first)

    def test_sealed_validation_geometry_is_exactly_768_balanced_rows_at_depths_2_to_4(self) -> None:
        receipt = data_pipeline._expected_metadata(self.default_config)[
            "contrast_validation"
        ]
        self.assertEqual(receipt["rows"], 768)
        self.assertEqual(receipt["depths"], {"2": 256, "3": 256, "4": 256})
        self.assertEqual(
            receipt["families"], {"checksum_branch": 384, "phase_branch": 384}
        )
        self.assertEqual(receipt["templates"], {"ledger": 384, "prose": 384})
        self.assertEqual(receipt["query_kinds"], {"checksum": 384, "node": 384})
        self.assertEqual(len(receipt["query_kind_grid"]), 12)
        for cell in receipt["query_kind_grid"].values():
            self.assertEqual(cell, {"checksum": 32, "node": 32})

    def test_manifest_reopens_payload_and_fails_closed_after_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_datasets(self.config, root)
            validate_data_manifest(self.config, root, manifest)
            train_path = root / manifest["files"]["train"]["path"]
            rows = read_jsonl(train_path)
            rows[0]["prompt"] += " tampered"
            with gzip.open(train_path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            with self.assertRaisesRegex(RuntimeError, "payload changed"):
                validate_data_manifest(self.config, root, manifest)

    def test_default_manifest_validation_never_decompresses_sealed_contrast_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_datasets(self.config, root)
            decompressed: list[str] = []
            original = data_pipeline.canonical_rows_receipt

            def audited(path):
                decompressed.append(Path(path).name)
                return original(path)

            with mock.patch.object(
                data_pipeline, "canonical_rows_receipt", side_effect=audited
            ):
                validate_data_manifest(self.config, root, manifest)
                self.assertEqual(decompressed, [])
                validate_data_manifest(
                    self.config,
                    root,
                    manifest,
                    content_splits={"train", "validation"},
                )
            self.assertEqual(set(decompressed), {"train.jsonl.gz", "validation.jsonl.gz"})

    def test_canonical_receipt_binds_content_and_order(self) -> None:
        rows = [
            {"id": "one", "nested": {"b": 2, "a": 1}},
            {"id": "two", "value": 3},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            baseline = canonical_rows_receipt(path)
            self.assertEqual(baseline["rows"], 2)
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in reversed(rows):
                    handle.write(json.dumps(row) + "\n")
            self.assertNotEqual(
                canonical_rows_receipt(path)["canonical_rows_sha256"],
                baseline["canonical_rows_sha256"],
            )

    def test_generation_refuses_to_overwrite_any_realized_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_datasets(self.config, root)
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                build_datasets(self.config, root)

    def test_contrast_access_ledger_is_identity_bound_and_append_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            directory_path = Path(directory)
            root_patcher = mock.patch.object(data_pipeline, "ROOT", directory_path)
            root_patcher.start()
            self.addCleanup(root_patcher.stop)
            root = directory_path / "data"
            manifest = build_datasets(self.config, root)
            checkpoint_lineage = self.checkpoint_lineage(directory_path, root)
            authorization = self.contrast_authorization(
                directory_path, {("lora", 7411): checkpoint_lineage}
            )
            evaluation_output = self.canonical_contrast_output()
            ledger = load_contrast_access_ledger(self.config, root, manifest)
            self.assertEqual(ledger["events"], [])
            self.assertEqual(
                set(ledger["sealed_splits"]),
                {"contrast_validation", "contrast_depth", "contrast_joint"},
            )
            event = record_contrast_access(
                self.config,
                root,
                manifest,
                authorization=authorization,
                capacity="lora",
                objective="joint",
                model_seed=7411,
                evaluation_output=evaluation_output,
                checkpoint_lineage=checkpoint_lineage,
            )
            self.assertEqual(event["event_index"], 1)
            self.assertEqual(
                event["splits"],
                ["contrast_depth", "contrast_joint", "contrast_validation"],
            )
            self.assertEqual(event["checkpoint_lineage"], checkpoint_lineage)
            self.assertEqual(event["replay_archives"], [])
            ledger_lock = root / f"{data_pipeline.ACCESS_LEDGER_NAME}.lock"
            self.assertTrue(ledger_lock.is_file())
            ledger_lock.write_text(
                "opaque persistent lock inode\n", encoding="utf-8"
            )
            reopened = load_contrast_access_ledger(self.config, root, manifest)
            self.assertEqual(reopened["events"], [event])

            with self.assertRaisesRegex(RuntimeError, "exactly one newly preserved"):
                record_contrast_access(
                    self.config,
                    root,
                    manifest,
                    authorization=authorization,
                    capacity="lora",
                    objective="joint",
                    model_seed=7411,
                    evaluation_output=evaluation_output,
                    checkpoint_lineage=checkpoint_lineage,
                )

            alternate = self.checkpoint_lineage(
                directory_path, root, "alternate_checkpoint"
            )
            with self.assertRaises(RuntimeError):
                record_contrast_access(
                    self.config,
                    root,
                    manifest,
                    authorization=authorization,
                    capacity="lora",
                    objective="joint",
                    model_seed=7411,
                    evaluation_output=evaluation_output,
                    checkpoint_lineage=alternate,
                )

            ledger_path = root / data_pipeline.ACCESS_LEDGER_NAME
            tampered = json.loads(ledger_path.read_text(encoding="utf-8"))
            tampered["events"][0]["capacity"] = "fullrank"
            ledger_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                load_contrast_access_ledger(self.config, root, manifest)

    def test_contrast_replay_requires_one_new_same_cell_archive_and_binds_it(self) -> None:
        archiver = load_archiver()
        source_digest = data_pipeline.source_contract_sha256()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            experiment = (
                repo
                / "experiments"
                / "qwen35_4b_state_formation_capacity_adjudication"
            )
            experiment.mkdir(parents=True)
            requirements_lock = repo / "requirements-training.lock.txt"
            requirements_lock.write_text("synthetic-lock\n", encoding="utf-8")
            patches = (
                mock.patch.object(data_pipeline, "ROOT", experiment),
                mock.patch.object(data_pipeline, "REPO_ROOT", repo),
                mock.patch.object(
                    data_pipeline, "REQUIREMENTS_LOCK", requirements_lock
                ),
                mock.patch.object(archiver, "ROOT", experiment),
                mock.patch.object(archiver, "REPO_ROOT", repo),
                mock.patch.object(
                    archiver,
                    "validate_design_receipt",
                    return_value={"status": "DESIGN_FROZEN"},
                ),
                mock.patch.object(
                    archiver, "design_lineage", return_value=self.design
                ),
                mock.patch.object(
                    archiver,
                    "source_contract_sha256",
                    return_value=source_digest,
                ),
            )
            for patcher in patches:
                patcher.start()
            try:
                data_root = experiment / "data"
                manifest = build_datasets(self.config, data_root)
                fixture_root = experiment / "fixtures"
                fixture_root.mkdir()
                checkpoint_lineage = self.checkpoint_lineage(
                    fixture_root, data_root
                )
                authorization = self.contrast_authorization(
                    fixture_root, {("lora", 7411): checkpoint_lineage}
                )
                evaluation_output = (
                    experiment / "runs" / "lora_joint_seed7411_contrast"
                )
                evaluation_output.mkdir(parents=True)

                initial = record_contrast_access(
                    self.config,
                    data_root,
                    manifest,
                    authorization=authorization,
                    capacity="lora",
                    objective="joint",
                    model_seed=7411,
                    evaluation_output=evaluation_output,
                    checkpoint_lineage=checkpoint_lineage,
                )
                self.assertEqual(initial["replay_archives"], [])

                (evaluation_output / "partial.jsonl").write_text(
                    "incomplete\n", encoding="utf-8"
                )
                shutil.rmtree(evaluation_output)
                evaluation_output.mkdir()
                with self.assertRaisesRegex(
                    RuntimeError, "exactly one newly preserved"
                ):
                    record_contrast_access(
                        self.config,
                        data_root,
                        manifest,
                        authorization=authorization,
                        capacity="lora",
                        objective="joint",
                        model_seed=7411,
                        evaluation_output=evaluation_output,
                        checkpoint_lineage=checkpoint_lineage,
                    )

                (evaluation_output / "partial.jsonl").write_text(
                    "preserved incomplete bytes\n", encoding="utf-8"
                )
                archive_receipt = archiver.archive_failed_attempt(
                    self.config, [evaluation_output]
                )
                self.assertFalse(evaluation_output.exists())
                evaluation_output.mkdir()
                replay = record_contrast_access(
                    self.config,
                    data_root,
                    manifest,
                    authorization=authorization,
                    capacity="lora",
                    objective="joint",
                    model_seed=7411,
                    evaluation_output=evaluation_output,
                    checkpoint_lineage=checkpoint_lineage,
                )
                self.assertEqual(len(replay["replay_archives"]), 1)
                tracked_receipt = next(
                    (experiment / "runs" / "failures").glob("*.json")
                )
                self.assertEqual(
                    replay["replay_archives"][0],
                    {
                        "path": data_pipeline._repo_relative(tracked_receipt),
                        "sha256": data_pipeline._sha256(tracked_receipt),
                        "receipt_identity_sha256": archive_receipt[
                            "receipt_identity_sha256"
                        ],
                        "attempt_identity_sha256": archive_receipt[
                            "attempt_identity_sha256"
                        ],
                        "archive_path": archive_receipt["archive_path"],
                    },
                )
                self.assertEqual(
                    load_contrast_access_ledger(
                        self.config, data_root, manifest
                    )["events"],
                    [replay],
                )

                (evaluation_output / "partial-again.jsonl").write_text(
                    "second incomplete attempt\n", encoding="utf-8"
                )
                shutil.rmtree(evaluation_output)
                evaluation_output.mkdir()
                with self.assertRaisesRegex(
                    RuntimeError, "exactly one newly preserved"
                ):
                    record_contrast_access(
                        self.config,
                        data_root,
                        manifest,
                        authorization=authorization,
                        capacity="lora",
                        objective="joint",
                        model_seed=7411,
                        evaluation_output=evaluation_output,
                        checkpoint_lineage=checkpoint_lineage,
                    )
            finally:
                for patcher in reversed(patches):
                    patcher.stop()

    def test_failed_archive_preseed_cannot_authorize_a_first_contrast_access(self) -> None:
        archiver = load_archiver()
        source_digest = data_pipeline.source_contract_sha256()
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            experiment = (
                repo
                / "experiments"
                / "qwen35_4b_state_formation_capacity_adjudication"
            )
            experiment.mkdir(parents=True)
            requirements_lock = repo / "requirements-training.lock.txt"
            requirements_lock.write_text("synthetic-lock\n", encoding="utf-8")
            patches = (
                mock.patch.object(data_pipeline, "ROOT", experiment),
                mock.patch.object(data_pipeline, "REPO_ROOT", repo),
                mock.patch.object(
                    data_pipeline, "REQUIREMENTS_LOCK", requirements_lock
                ),
                mock.patch.object(archiver, "ROOT", experiment),
                mock.patch.object(archiver, "REPO_ROOT", repo),
                mock.patch.object(
                    archiver,
                    "validate_design_receipt",
                    return_value={"status": "DESIGN_FROZEN"},
                ),
                mock.patch.object(
                    archiver, "design_lineage", return_value=self.design
                ),
                mock.patch.object(
                    archiver,
                    "source_contract_sha256",
                    return_value=source_digest,
                ),
            )
            for patcher in patches:
                patcher.start()
            try:
                data_root = experiment / "data"
                manifest = build_datasets(self.config, data_root)
                fixture_root = experiment / "fixtures"
                fixture_root.mkdir()
                checkpoint_lineage = self.checkpoint_lineage(
                    fixture_root, data_root
                )
                authorization = self.contrast_authorization(
                    fixture_root, {("lora", 7411): checkpoint_lineage}
                )
                evaluation_output = (
                    experiment / "runs" / "lora_joint_seed7411_contrast"
                )
                evaluation_output.mkdir(parents=True)
                (evaluation_output / "preseed-partial.jsonl").write_text(
                    "archive predates access\n", encoding="utf-8"
                )
                archiver.archive_failed_attempt(self.config, [evaluation_output])
                evaluation_output.mkdir()

                with self.assertRaisesRegex(
                    RuntimeError, "predates the first contrast access"
                ):
                    record_contrast_access(
                        self.config,
                        data_root,
                        manifest,
                        authorization=authorization,
                        capacity="lora",
                        objective="joint",
                        model_seed=7411,
                        evaluation_output=evaluation_output,
                        checkpoint_lineage=checkpoint_lineage,
                    )
                self.assertEqual(
                    load_contrast_access_ledger(
                        self.config, data_root, manifest
                    )["events"],
                    [],
                )
            finally:
                for patcher in reversed(patches):
                    patcher.stop()

    def test_contrast_ledger_rejects_a_checkpoint_from_another_cell(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            directory_path = Path(directory)
            root_patcher = mock.patch.object(data_pipeline, "ROOT", directory_path)
            root_patcher.start()
            self.addCleanup(root_patcher.stop)
            root = directory_path / "data"
            manifest = build_datasets(self.config, root)
            mismatched = self.checkpoint_lineage(
                directory_path, root, capacity="fullrank"
            )
            authorization = self.contrast_authorization(
                directory_path, {("lora", 7411): mismatched}
            )
            evaluation_output = self.canonical_contrast_output()
            with self.assertRaisesRegex(RuntimeError, "checkpoint.*cell"):
                record_contrast_access(
                    self.config,
                    root,
                    manifest,
                    authorization=authorization,
                    capacity="lora",
                    objective="joint",
                    model_seed=7411,
                    evaluation_output=evaluation_output,
                    checkpoint_lineage=mismatched,
                )

    def test_contrast_ledger_recomputes_the_stage_b_receipt_identity(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            directory_path = Path(directory)
            root_patcher = mock.patch.object(data_pipeline, "ROOT", directory_path)
            root_patcher.start()
            self.addCleanup(root_patcher.stop)
            root = directory_path / "data"
            manifest = build_datasets(self.config, root)
            checkpoint_lineage = self.checkpoint_lineage(directory_path, root)
            authorization = self.contrast_authorization(
                directory_path, {("lora", 7411): checkpoint_lineage}
            )
            evaluation_output = self.canonical_contrast_output()
            authorization_path = data_pipeline.REPO_ROOT / authorization["path"]
            receipt = json.loads(authorization_path.read_text(encoding="utf-8"))
            receipt["tampered_after_validation"] = True
            authorization_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            authorization["sha256"] = data_pipeline._sha256(authorization_path)
            with self.assertRaisesRegex(RuntimeError, "authorization .*identity"):
                record_contrast_access(
                    self.config,
                    root,
                    manifest,
                    authorization=authorization,
                    capacity="lora",
                    objective="joint",
                    model_seed=7411,
                    evaluation_output=evaluation_output,
                    checkpoint_lineage=checkpoint_lineage,
                )

    def test_append_revalidates_locked_ledger_header_not_only_its_self_hash(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            directory_path = Path(directory)
            root_patcher = mock.patch.object(data_pipeline, "ROOT", directory_path)
            root_patcher.start()
            self.addCleanup(root_patcher.stop)
            root = directory_path / "data"
            manifest = build_datasets(self.config, root)
            checkpoint_lineage = self.checkpoint_lineage(directory_path, root)
            authorization = self.contrast_authorization(
                directory_path, {("lora", 7411): checkpoint_lineage}
            )
            evaluation_output = self.canonical_contrast_output()
            ledger_path = root / data_pipeline.ACCESS_LEDGER_NAME
            changed = json.loads(ledger_path.read_text(encoding="utf-8"))
            changed["sealed_splits"]["contrast_depth"]["path"] = "wrong.jsonl.gz"
            changed["receipt_identity_sha256"] = data_pipeline._ledger_identity(changed)
            ledger_path.write_text(
                json.dumps(changed, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "sealed_splits mismatch"):
                record_contrast_access(
                    self.config,
                    root,
                    manifest,
                    authorization=authorization,
                    capacity="lora",
                    objective="joint",
                    model_seed=7411,
                    evaluation_output=evaluation_output,
                    checkpoint_lineage=checkpoint_lineage,
                )

    def test_contrast_access_rejects_a_noncanonical_evaluation_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            directory_path = Path(directory)
            root_patcher = mock.patch.object(data_pipeline, "ROOT", directory_path)
            root_patcher.start()
            self.addCleanup(root_patcher.stop)
            data_root = directory_path / "data"
            manifest = build_datasets(self.config, data_root)
            checkpoint_lineage = self.checkpoint_lineage(directory_path, data_root)
            authorization = self.contrast_authorization(
                directory_path, {("lora", 7411): checkpoint_lineage}
            )
            noncanonical_output = directory_path / "evaluation"
            noncanonical_output.mkdir()

            with self.assertRaisesRegex(RuntimeError, "exact canonical evaluation output"):
                record_contrast_access(
                    self.config,
                    data_root,
                    manifest,
                    authorization=authorization,
                    capacity="lora",
                    objective="joint",
                    model_seed=7411,
                    evaluation_output=noncanonical_output,
                    checkpoint_lineage=checkpoint_lineage,
                )
            self.assertEqual(
                load_contrast_access_ledger(self.config, data_root, manifest)["events"],
                [],
            )

    def test_contrast_access_rejects_a_copied_stage_b_seal(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            directory_path = Path(directory)
            root_patcher = mock.patch.object(data_pipeline, "ROOT", directory_path)
            root_patcher.start()
            self.addCleanup(root_patcher.stop)
            data_root = directory_path / "data"
            manifest = build_datasets(self.config, data_root)
            checkpoint_lineage = self.checkpoint_lineage(directory_path, data_root)
            authorization = self.contrast_authorization(
                directory_path, {("lora", 7411): checkpoint_lineage}
            )
            canonical_authorization = data_pipeline.REPO_ROOT / authorization["path"]
            copied_authorization = directory_path / "copied_stage_b_seal.json"
            shutil.copyfile(canonical_authorization, copied_authorization)
            copied_lineage = dict(authorization)
            copied_lineage["path"] = data_pipeline._repo_relative(copied_authorization)
            copied_lineage["sha256"] = data_pipeline._sha256(copied_authorization)
            evaluation_output = self.canonical_contrast_output()

            with self.assertRaisesRegex(RuntimeError, "canonical Stage-B seal"):
                record_contrast_access(
                    self.config,
                    data_root,
                    manifest,
                    authorization=copied_lineage,
                    capacity="lora",
                    objective="joint",
                    model_seed=7411,
                    evaluation_output=evaluation_output,
                    checkpoint_lineage=checkpoint_lineage,
                )
            self.assertEqual(
                load_contrast_access_ledger(self.config, data_root, manifest)["events"],
                [],
            )


if __name__ == "__main__":
    unittest.main()
