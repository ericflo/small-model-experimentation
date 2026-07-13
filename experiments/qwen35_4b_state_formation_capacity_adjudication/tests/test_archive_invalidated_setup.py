from __future__ import annotations

import copy
import errno
import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import config_sha256, load_config  # noqa: E402


def load_archiver():
    spec = importlib.util.spec_from_file_location(
        "capacity_setup_invalidation_archiver",
        ROOT / "scripts" / "archive_invalidated_setup.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("setup invalidation archiver cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class InvalidatedSetupArchiveTests(unittest.TestCase):
    OLD_SOURCE = "a" * 64
    NEW_SOURCE = "b" * 64

    def setUp(self) -> None:
        self.module = load_archiver()
        self.config = copy.deepcopy(load_config(ROOT / "configs" / "default.yaml"))
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        self.experiment = (
            self.repo / "experiments" / self.config["experiment_id"]
        )
        self.experiment.mkdir(parents=True)
        self.root_patch = mock.patch.object(self.module, "ROOT", self.experiment)
        self.repo_patch = mock.patch.object(self.module, "REPO_ROOT", self.repo)
        self.source_patch = mock.patch.object(
            self.module,
            "source_contract_sha256",
            return_value=self.NEW_SOURCE,
        )
        for patcher in (self.root_patch, self.repo_patch, self.source_patch):
            patcher.start()
        self.trigger = self._populate_setup(
            g0_names=("g0_lora_seed7411.json", "g0_fullrank_seed7412.json")
        )

    def tearDown(self) -> None:
        for patcher in (self.source_patch, self.repo_patch, self.root_patch):
            patcher.stop()
        self.temporary.cleanup()

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _signed(self, payload: dict) -> dict:
        signed = copy.deepcopy(payload)
        signed["receipt_identity_sha256"] = self.module._canonical_sha256(signed)
        return signed

    def _common(self) -> dict:
        return {
            "source_contract_sha256": self.OLD_SOURCE,
            "config_sha256": config_sha256(self.config),
            "experiment_id": self.config["experiment_id"],
            "model_id": self.module.MODEL_ID,
            "model_revision": self.module.MODEL_REVISION,
        }

    def _populate_setup(self, g0_names: tuple[str, ...]) -> Path:
        data_dir = self.experiment / self.config["paths"]["data_dir"]
        data_dir.mkdir(parents=True)
        (data_dir / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")
        manifest_files = {}
        for index, name in enumerate(self.module.DATA_PAYLOADS):
            path = data_dir / name
            path.write_bytes(f"synthetic-gzip-{index}-{name}".encode("utf-8"))
            split = name.removesuffix(".jsonl.gz")
            manifest_files[split] = {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": self.module._sha256(path),
            }
        manifest = {
            "schema_version": 1,
            **self._common(),
            "benchmark_files_read": 0,
            "files": manifest_files,
        }
        manifest_path = data_dir / "manifest.json"
        self._write_json(manifest_path, manifest)
        manifest_sha256 = self.module._sha256(manifest_path)
        ledger = self._signed(
            {
                "schema_version": 1,
                "experiment_id": self.config["experiment_id"],
                "data_manifest_sha256": manifest_sha256,
                "events": [],
                "sealed_splits": {
                    split: manifest_files[split]
                    for split in self.module.SEALED_SPLITS
                },
            }
        )
        self._write_json(data_dir / "contrast_access_ledger.json", ledger)

        cpu_receipt = {
            "status": "CPU_SMOKE_PASS",
            "scientific_evidence": False,
            "config": {**self._common(), "backend": "transformers"},
        }
        self._write_json(
            self.experiment / "runs" / "cpu_smoke" / "receipt.json",
            cpu_receipt,
        )

        large_root = (
            self.experiment / self.config["paths"]["large_artifacts_dir"]
        ).resolve()
        setup_dir = self.experiment / "runs" / "setup"
        (self.experiment / "runs" / "cpu_smoke" / ".gitkeep").touch()
        setup_dir.mkdir(parents=True, exist_ok=True)
        (setup_dir / ".gitkeep").touch()
        initialization_receipts = {}
        for seed in map(int, self.config["training"]["train_seeds"]):
            bundle = large_root / f"initialization_seed{seed}.pt"
            bundle.parent.mkdir(parents=True, exist_ok=True)
            bundle.write_bytes(f"initialization-{seed}".encode("utf-8"))
            metadata = self._signed(
                {
                    "schema_version": 1,
                    **self._common(),
                    "model_seed": seed,
                    "tensor_values_sha256": f"{seed:064x}"[-64:],
                }
            )
            receipt = self._signed(
                {
                    "schema_version": 1,
                    "status": "SHARED_INITIALIZATION_PREPARED",
                    "phase": "shared_initialization",
                    "bundle_path": bundle.relative_to(self.repo).as_posix(),
                    "bundle_sha256": self.module._sha256(bundle),
                    "metadata": metadata,
                }
            )
            encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            bundle.with_suffix(".pt.json").write_text(encoded, encoding="utf-8")
            setup_dir.mkdir(parents=True, exist_ok=True)
            (setup_dir / f"initialization_seed{seed}.json").write_text(
                encoded,
                encoding="utf-8",
            )
            initialization_receipts[seed] = receipt

        for name in g0_names:
            match = self.module.G0_RE.fullmatch(name)
            if match is None:
                raise AssertionError(name)
            capacity, seed_text = match.groups()
            seed = int(seed_text)
            g0 = self._signed(
                {
                    "schema_version": 1,
                    "status": "MODEL_SMOKE_PASS",
                    "phase": f"{capacity}_g0",
                    "backend": "transformers",
                    **self._common(),
                    "capacity": capacity,
                    "model_seed": seed,
                    "data_manifest_sha256": manifest_sha256,
                    "branch_authorization": (
                        None if capacity == "lora" else {"status": "synthetic authorization"}
                    ),
                    "setup": {
                        "shared_initialization": initialization_receipts[seed]
                    },
                }
            )
            self._write_json(setup_dir / name, g0)

        trigger = self.experiment / "runs" / "failures" / "trigger_failure.json"
        trigger_payload = self._signed(
            {
                "schema_version": 1,
                "status": "SETUP_CONTROL_FAILED_PRESERVED",
                "experiment_id": self.config["experiment_id"],
                "model_id": self.module.MODEL_ID,
                "model_revision": self.module.MODEL_REVISION,
                "source_contract_sha256": self.OLD_SOURCE,
                "benchmark_files_read": 0,
                "sealed_contrast_payloads_opened": [],
                "scientific_evidence": False,
            }
        )
        self._write_json(trigger, trigger_payload)
        return trigger

    def _add_positive_control_failure(self, capacity: str, seed: int) -> tuple[Path, Path]:
        setup_dir = self.experiment / "runs" / "setup"
        canonical = setup_dir / f"positive_control_{capacity}_seed{seed}.json"
        payload = self._signed(
            {
                "schema_version": 1,
                "status": "SETUP_CONTROL_FAILED",
                "phase": f"{capacity}_positive_control",
                "backend": "transformers",
                **self._common(),
                "capacity": capacity,
                "model_seed": seed,
                "data_manifest_sha256": None,
                "g0_lineage": None,
                "branch_authorization": None,
                "shared_initialization": None,
                "setup": None,
                "control_rows": None,
                "oracle_analysis": None,
                "oracle_readout_accuracy": None,
                "failure_stage": "receipt_preflight",
                "error_type": "RuntimeError",
                "error": "synthetic G0 receipt preflight failure",
                "completed_updates": 0,
                "completed_microbatches": 0,
                "training_diagnostics": {
                    "fixed_probe_steps": [],
                    "evaluations": [],
                    "parameter_probes": [],
                    "optimizer_step_probes": [],
                    "dropout_probes": [],
                    "completed_updates": 0,
                    "completed_microbatches": 0,
                },
                "authorizes_training": False,
                "authorizes_result_training": False,
                "authorizes_result_evaluation": False,
                "benchmark_files_read": 0,
                "result_payloads_opened": [],
                "sealed_contrast_payloads_opened": [],
                "scientific_evidence": False,
            }
        )
        self._write_json(canonical, payload)
        mirror = (
            self.experiment
            / "runs"
            / "failures"
            / f"positive_control_{capacity}_seed{seed}_source_{self.OLD_SOURCE[:12]}.json"
        )
        self._write_json(mirror, payload)
        return canonical, mirror

    def _add_positive_control_pass(
        self,
        capacity: str,
        seed: int,
        *,
        evaluation_authority: object = False,
        omit_evaluation_authority: bool = False,
    ) -> Path:
        setup_dir = self.experiment / "runs" / "setup"
        g0_path = setup_dir / f"g0_{capacity}_seed{seed}.json"
        g0 = json.loads(g0_path.read_text(encoding="utf-8"))
        shared_initialization = g0["setup"]["shared_initialization"]
        manifest_path = (
            self.experiment / self.config["paths"]["data_dir"] / "manifest.json"
        )
        payload = {
            "schema_version": 1,
            "status": "POSITIVE_CONTROL_PASS",
            "phase": f"{capacity}_positive_control",
            "backend": "transformers",
            **self._common(),
            "capacity": capacity,
            "model_seed": seed,
            "data_manifest_sha256": self.module._sha256(manifest_path),
            "g0_lineage": {
                "path": g0_path.relative_to(self.repo).as_posix(),
                "sha256": self.module._sha256(g0_path),
                "receipt_identity_sha256": g0["receipt_identity_sha256"],
                "status": "MODEL_SMOKE_PASS",
                "phase": f"{capacity}_g0",
            },
            "branch_authorization": g0["branch_authorization"],
            "shared_initialization": shared_initialization,
            "setup": {"shared_initialization": shared_initialization},
            "authorizes_training": True,
            "authorizes_result_training": True,
            "benchmark_files_read": 0,
            "result_payloads_opened": [],
            "sealed_contrast_payloads_opened": [],
            "scientific_evidence": False,
        }
        if not omit_evaluation_authority:
            payload["authorizes_result_evaluation"] = evaluation_authority
        canonical = setup_dir / f"positive_control_{capacity}_seed{seed}.json"
        self._write_json(canonical, self._signed(payload))
        return canonical

    def _replace_g0_with_failure(self, capacity: str, seed: int) -> tuple[Path, Path]:
        setup_dir = self.experiment / "runs" / "setup"
        canonical = setup_dir / f"g0_{capacity}_seed{seed}.json"
        prior = json.loads(canonical.read_text(encoding="utf-8"))
        payload = self._signed(
            {
                "schema_version": 1,
                "status": "SETUP_CONTROL_FAILED",
                "phase": f"{capacity}_g0",
                "backend": "transformers",
                **self._common(),
                "capacity": capacity,
                "model_seed": seed,
                "data_manifest_sha256": self.module._sha256(
                    self.experiment / "data" / "generated" / "manifest.json"
                ),
                "branch_authorization": prior["branch_authorization"],
                "shared_initialization": prior["setup"]["shared_initialization"],
                "setup": prior["setup"],
                "failure_stage": "live_joint_backward_probe",
                "error_type": "RuntimeError",
                "error": "synthetic live-joint reachability failure",
                "completed_checks": [
                    "branch_authorization",
                    "train_only_data_manifest",
                    "shared_initialization",
                    "pinned_model_and_wrapper_setup",
                    "registered_setup_rows_and_encoding",
                    "pre_optimizer_k1_parity",
                    "zero_function_and_k4_call_geometry",
                    "two_step_state_only_optimizer_probe",
                ],
                "authorizes_positive_control": False,
                "authorizes_training": False,
                "authorizes_result_training": False,
                "authorizes_result_evaluation": False,
                "benchmark_files_read": 0,
                "result_payloads_opened": ["train"],
                "sealed_contrast_payloads_opened": [],
                "training_or_evaluation_started": False,
                "scientific_evidence": False,
            }
        )
        self._write_json(canonical, payload)
        mirror = (
            self.experiment
            / "runs"
            / "failures"
            / f"g0_{capacity}_seed{seed}_source_{self.OLD_SOURCE[:12]}.json"
        )
        self._write_json(mirror, payload)
        return canonical, mirror

    def _source_paths(self) -> set[Path]:
        return {
            item.source
            for item in self.module._inventory(self.config)
        }

    def _archive_root(self) -> Path:
        return (
            self.experiment
            / self.config["paths"]["large_artifacts_dir"]
            / "invalidated_setup"
            / f"source_{self.OLD_SOURCE}"
        ).resolve()

    def _tracked_archive_receipt(self) -> Path:
        return (
            self.experiment
            / "runs"
            / "failures"
            / f"invalidated_setup_source_{self.OLD_SOURCE[:8]}.json"
        )

    def _quarantine_root(self) -> Path:
        return self.module._cleanup_quarantine_root(
            self.config, self.OLD_SOURCE
        )

    def assert_complete_zero_quarantine(self) -> None:
        items = self.module._items_from_file_records(
            self.config,
            json.loads(
                (self._archive_root() / "archive_receipt.json").read_text(
                    encoding="utf-8"
                )
            )["files"],
        )[0]
        records = json.loads(
            (self._archive_root() / "archive_receipt.json").read_text(
                encoding="utf-8"
            )
        )["files"]
        complete, destinations = self.module._quarantine_snapshot(
            self._quarantine_root(), items, records
        )
        self.assertTrue(complete)
        self.assertEqual(set(destinations), {item.archive_path for item in items})

    def _fail_move_patch(
        self,
        *,
        call_number: int = 2,
        message: str = "synthetic move failure",
    ):
        real_move = self.module.move_new_entry
        calls = 0

        def fail_at_selected_move(
            root: Path,
            source: Path,
            destination: Path,
        ) -> None:
            nonlocal calls
            calls += 1
            if calls == call_number:
                raise OSError(errno.EIO, message)
            real_move(root, source, destination)

        return mock.patch.object(
            self.module,
            "move_new_entry",
            side_effect=fail_at_selected_move,
        )

    def test_exact_inventory_is_archived_with_stable_names(self) -> None:
        items = self.module._inventory(self.config)
        self.assertEqual(len(items), 21)
        archive_names = {item.archive_path for item in items}
        self.assertEqual(
            archive_names,
            {
                *(f"data_generated/{name}" for name in self.module.DATA_SETUP_FILES),
                *(
                    f"initialization/initialization_seed{seed}{suffix}"
                    for seed in self.config["training"]["train_seeds"]
                    for suffix in (".pt", ".pt.json")
                ),
                "tracked_receipts/cpu_smoke_receipt.json",
                *(
                    f"tracked_receipts/initialization_seed{seed}.json"
                    for seed in self.config["training"]["train_seeds"]
                ),
                "tracked_receipts/g0_lora_seed7411.json",
                "tracked_receipts/g0_fullrank_seed7412.json",
            },
        )
        source_paths = {item.source for item in items}

        receipt = self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )

        self.assertTrue(all(not path.exists() for path in source_paths))
        self.assertTrue(
            (
                self.experiment
                / self.config["paths"]["data_dir"]
                / ".gitignore"
            ).is_file()
        )
        self.assertEqual(
            {path.name for path in (self.experiment / "runs" / "cpu_smoke").iterdir()},
            {".gitkeep"},
        )
        self.assertEqual(
            {path.name for path in (self.experiment / "runs" / "setup").iterdir()},
            {".gitkeep"},
        )
        self.assert_complete_zero_quarantine()
        self.assertTrue(
            (
                self.experiment
                / self.config["paths"]["large_artifacts_dir"]
            ).is_dir()
        )
        self.assertTrue(self.trigger.is_file())
        archived_names = {
            path.relative_to(self._archive_root()).as_posix()
            for path in self._archive_root().rglob("*")
            if path.is_file()
        }
        self.assertEqual(
            archived_names,
            archive_names | {"archive_receipt.json"},
        )
        self.assertEqual(receipt["total_bytes"], sum(row["bytes"] for row in receipt["files"]))

    def test_content_hashes_and_both_receipt_identities_are_exact(self) -> None:
        expected_records = self.module._file_records(
            self.module._inventory(self.config)
        )
        trigger_sha256 = self.module._sha256(self.trigger)
        receipt = self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )
        self.assertEqual(receipt["files"], expected_records)
        self.assertEqual(
            receipt["files_sha256"],
            self.module._canonical_sha256(expected_records),
        )
        unsigned = {
            key: value
            for key, value in receipt.items()
            if key != "receipt_identity_sha256"
        }
        self.assertEqual(
            receipt["receipt_identity_sha256"],
            self.module._canonical_sha256(unsigned),
        )
        self.assertEqual(receipt["trigger_failure_receipt_sha256"], trigger_sha256)

        external = json.loads(
            (self._archive_root() / "archive_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        tracked_path = self._tracked_archive_receipt()
        tracked = json.loads(tracked_path.read_text(encoding="utf-8"))
        self.assertEqual(external, receipt)
        self.assertEqual(tracked, receipt)
        for record in receipt["files"]:
            archived = self._archive_root() / record["path"]
            self.assertEqual(archived.stat().st_size, record["bytes"])
            self.assertEqual(self.module._sha256(archived), record["sha256"])

    def test_success_removes_only_source_parents_that_become_empty(self) -> None:
        data_dir = self.experiment / self.config["paths"]["data_dir"]
        large_root = (
            self.experiment / self.config["paths"]["large_artifacts_dir"]
        )

        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )

        self.assertEqual(
            {path.name for path in (self.experiment / "runs" / "cpu_smoke").iterdir()},
            {".gitkeep"},
        )
        self.assertEqual(
            {path.name for path in (self.experiment / "runs" / "setup").iterdir()},
            {".gitkeep"},
        )
        self.assertTrue(data_dir.is_dir())
        self.assertEqual(
            {path.name for path in data_dir.iterdir()},
            {".gitignore"},
        )
        self.assertTrue(large_root.is_dir())
        self.assertTrue(self._archive_root().is_dir())

    def test_canonical_positive_control_failure_and_identical_mirror_are_preserved(self) -> None:
        canonical, mirror = self._add_positive_control_failure("lora", 7411)
        items = self.module._inventory(self.config)
        self.assertIn(canonical.resolve(), {item.source.resolve() for item in items})
        receipt = self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            mirror,
        )
        self.assertFalse(canonical.exists())
        self.assertTrue(mirror.is_file())
        archived = self._archive_root() / "tracked_receipts" / canonical.name
        self.assertTrue(archived.is_file())
        self.assertEqual(archived.read_bytes(), mirror.read_bytes())
        self.assertIn(
            f"tracked_receipts/{canonical.name}",
            {record["path"] for record in receipt["files"]},
        )

    def test_legacy_positive_control_pass_without_evaluation_bit_is_archived(self) -> None:
        canonical = self._add_positive_control_pass(
            "lora",
            7411,
            omit_evaluation_authority=True,
        )
        receipt = self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )
        self.assertFalse(canonical.exists())
        archived = self._archive_root() / "tracked_receipts" / canonical.name
        self.assertTrue(archived.is_file())
        self.assertIn(
            f"tracked_receipts/{canonical.name}",
            {record["path"] for record in receipt["files"]},
        )

    def test_positive_control_pass_rejects_unsafe_evaluation_bit(self) -> None:
        canonical = self._add_positive_control_pass("lora", 7411)
        original = json.loads(canonical.read_text(encoding="utf-8"))
        for label, unsafe in (("true", True), ("null", None), ("integer", 0)):
            with self.subTest(label=label):
                payload = copy.deepcopy(original)
                payload.pop("receipt_identity_sha256")
                payload["authorizes_result_evaluation"] = unsafe
                self._write_json(canonical, self._signed(payload))
                with self.assertRaises(RuntimeError):
                    self.module.archive_invalidated_setup(
                        self.config,
                        self.OLD_SOURCE,
                        self.trigger,
                    )
                self.assertFalse(self._archive_root().exists())

    def test_positive_control_failure_auth_error_access_and_progress_are_exact(self) -> None:
        canonical, mirror = self._add_positive_control_failure("lora", 7411)
        original = json.loads(canonical.read_text(encoding="utf-8"))
        mutations = (
            ("unsafe-auth", {"authorizes_result_evaluation": True}),
            ("missing-error", {"error": ""}),
            ("result-access", {"result_payloads_opened": ["train"]}),
            ("future-setup", {"setup": {"shared_initialization": {}}}),
            ("impossible-progress", {"completed_updates": 1}),
        )
        for label, changes in mutations:
            with self.subTest(label=label):
                payload = copy.deepcopy(original)
                payload.pop("receipt_identity_sha256")
                payload.update(changes)
                payload = self._signed(payload)
                self._write_json(canonical, payload)
                self._write_json(mirror, payload)
                with self.assertRaises(RuntimeError):
                    self.module.archive_invalidated_setup(
                        self.config,
                        self.OLD_SOURCE,
                        self.trigger,
                    )
                self.assertFalse(self._archive_root().exists())

    def test_canonical_g0_failure_and_identical_mirror_are_preserved(self) -> None:
        canonical, mirror = self._replace_g0_with_failure("lora", 7411)
        receipt = self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            mirror,
        )
        self.assertFalse(canonical.exists())
        self.assertTrue(mirror.is_file())
        archived = self._archive_root() / "tracked_receipts" / canonical.name
        self.assertEqual(archived.read_bytes(), mirror.read_bytes())
        self.assertIn(
            f"tracked_receipts/{canonical.name}",
            {record["path"] for record in receipt["files"]},
        )

    def test_early_g0_failure_without_unreached_lineage_is_archivable(self) -> None:
        canonical, mirror = self._replace_g0_with_failure("lora", 7411)
        payload = json.loads(canonical.read_text(encoding="utf-8"))
        payload.pop("receipt_identity_sha256")
        payload.update(
            {
                "data_manifest_sha256": None,
                "shared_initialization": None,
                "setup": None,
                "failure_stage": "branch_authorization",
                "completed_checks": [],
                "result_payloads_opened": [],
            }
        )
        payload = self._signed(payload)
        self._write_json(canonical, payload)
        self._write_json(mirror, payload)
        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            mirror,
        )
        archived = self._archive_root() / "tracked_receipts" / canonical.name
        self.assertEqual(archived.read_bytes(), mirror.read_bytes())

    def test_model_setup_failure_may_precede_initialization_validation(self) -> None:
        canonical, mirror = self._replace_g0_with_failure("lora", 7411)
        payload = json.loads(canonical.read_text(encoding="utf-8"))
        payload.pop("receipt_identity_sha256")
        payload.update(
            {
                "shared_initialization": None,
                "setup": None,
                "failure_stage": "model_setup",
                "completed_checks": [
                    "branch_authorization",
                    "train_only_data_manifest",
                ],
            }
        )
        payload = self._signed(payload)
        self._write_json(canonical, payload)
        self._write_json(mirror, payload)
        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            mirror,
        )
        archived = self._archive_root() / "tracked_receipts" / canonical.name
        self.assertEqual(archived.read_bytes(), mirror.read_bytes())

    def test_fullrank_branch_failure_may_precede_authorization(self) -> None:
        canonical, mirror = self._replace_g0_with_failure("fullrank", 7412)
        payload = json.loads(canonical.read_text(encoding="utf-8"))
        payload.pop("receipt_identity_sha256")
        payload.update(
            {
                "branch_authorization": None,
                "data_manifest_sha256": None,
                "shared_initialization": None,
                "setup": None,
                "failure_stage": "branch_authorization",
                "completed_checks": [],
                "result_payloads_opened": [],
            }
        )
        payload = self._signed(payload)
        self._write_json(canonical, payload)
        self._write_json(mirror, payload)
        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            mirror,
        )
        archived = self._archive_root() / "tracked_receipts" / canonical.name
        self.assertEqual(archived.read_bytes(), mirror.read_bytes())

    def test_g0_failure_progress_must_be_an_exact_stage_prefix(self) -> None:
        canonical, mirror = self._replace_g0_with_failure("lora", 7411)
        original = json.loads(canonical.read_text(encoding="utf-8"))
        mutations = (
            ("unknown-stage", {"failure_stage": "invented_future_stage"}),
            (
                "duplicate-check",
                {"completed_checks": [
                    *original["completed_checks"],
                    original["completed_checks"][-1],
                ]},
            ),
            (
                "out-of-order",
                {"completed_checks": [
                    original["completed_checks"][1],
                    original["completed_checks"][0],
                    *original["completed_checks"][2:],
                ]},
            ),
            ("late-empty-access", {"result_payloads_opened": []}),
            ("late-null-manifest", {"data_manifest_sha256": None}),
        )
        for label, changes in mutations:
            with self.subTest(label=label):
                mutated = copy.deepcopy(original)
                mutated.pop("receipt_identity_sha256")
                mutated.update(changes)
                mutated = self._signed(mutated)
                self._write_json(canonical, mutated)
                self._write_json(mirror, mutated)
                with self.assertRaises(RuntimeError):
                    self.module.archive_invalidated_setup(
                        self.config,
                        self.OLD_SOURCE,
                        self.trigger,
                    )

    def test_g0_failure_duplicate_initialization_bindings_must_both_match(self) -> None:
        canonical, mirror = self._replace_g0_with_failure("lora", 7411)
        payload = json.loads(canonical.read_text(encoding="utf-8"))
        payload.pop("receipt_identity_sha256")
        payload["shared_initialization"] = {"wrong": "top-level lineage"}
        payload = self._signed(payload)
        self._write_json(canonical, payload)
        self._write_json(mirror, payload)
        with self.assertRaisesRegex(RuntimeError, "top-level binding"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

    def test_g0_pass_new_access_contract_is_all_or_exact(self) -> None:
        path = self.experiment / "runs" / "setup" / "g0_lora_seed7411.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("receipt_identity_sha256")
        payload["scientific_evidence"] = False
        payload = self._signed(payload)
        self._write_json(path, payload)
        with self.assertRaisesRegex(RuntimeError, "partial access contract"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

    def test_g0_failure_missing_or_changed_mirror_is_rejected_without_deletion(self) -> None:
        canonical, mirror = self._replace_g0_with_failure("lora", 7411)
        source_paths = self._source_paths()
        mirror.unlink()
        with self.assertRaisesRegex(RuntimeError, "lacks its identical tracked mirror"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )
        self.assertTrue(all(path.exists() for path in source_paths))
        self._write_json(mirror, json.loads(canonical.read_text(encoding="utf-8")))
        mirror.write_bytes(mirror.read_bytes() + b" ")
        with self.assertRaisesRegex(RuntimeError, "lacks its identical tracked mirror"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )
        self.assertTrue(all(path.exists() for path in source_paths))

    def test_g0_failure_unsafe_access_or_authorization_is_rejected(self) -> None:
        canonical, mirror = self._replace_g0_with_failure("lora", 7411)
        original = json.loads(canonical.read_text(encoding="utf-8"))
        for field, value, message in (
            ("authorizes_positive_control", True, "authorizes_positive_control mismatch"),
            ("result_payloads_opened", ["validation"], "unsafe result access"),
            ("sealed_contrast_payloads_opened", ["contrast_depth"], "sealed_contrast_payloads_opened mismatch"),
        ):
            with self.subTest(field=field):
                mutated = copy.deepcopy(original)
                mutated.pop("receipt_identity_sha256")
                mutated[field] = value
                mutated = self._signed(mutated)
                self._write_json(canonical, mutated)
                self._write_json(mirror, mutated)
                with self.assertRaisesRegex(RuntimeError, message):
                    self.module.archive_invalidated_setup(
                        self.config,
                        self.OLD_SOURCE,
                        self.trigger,
                    )
                self.assertFalse(self._archive_root().exists())

    def test_g0_failure_wrong_phase_or_initialization_is_rejected(self) -> None:
        canonical, mirror = self._replace_g0_with_failure("lora", 7411)
        original = json.loads(canonical.read_text(encoding="utf-8"))
        mutations = (
            ("phase", "fullrank_g0", "phase mismatch"),
            ("shared_initialization", {"wrong": True}, "wrong initialization lineage"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                mutated = copy.deepcopy(original)
                mutated.pop("receipt_identity_sha256")
                mutated[field] = value
                if field == "shared_initialization":
                    mutated["setup"] = None
                mutated = self._signed(mutated)
                self._write_json(canonical, mutated)
                self._write_json(mirror, mutated)
                with self.assertRaisesRegex(RuntimeError, message):
                    self.module.archive_invalidated_setup(
                        self.config,
                        self.OLD_SOURCE,
                        self.trigger,
                    )
                self.assertFalse(self._archive_root().exists())

    def test_failed_g0_cannot_coexist_with_positive_control(self) -> None:
        self._replace_g0_with_failure("lora", 7411)
        self._add_positive_control_failure("lora", 7411)
        source_paths = self._source_paths()
        with self.assertRaisesRegex(RuntimeError, "cannot coexist with a positive control"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )
        self.assertTrue(all(path.exists() for path in source_paths))

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink creation is unavailable")
    def test_symlinked_g0_failure_mirror_is_rejected(self) -> None:
        _, mirror = self._replace_g0_with_failure("lora", 7411)
        source_paths = self._source_paths()
        target = mirror.with_name("g0-mirror-target.json")
        mirror.rename(target)
        mirror.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, "lacks its identical tracked mirror"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )
        self.assertTrue(all(path.exists() for path in source_paths))
        self.assertFalse(self._archive_root().exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink creation is unavailable")
    def test_symlinked_positive_control_failure_mirror_is_rejected(self) -> None:
        _, mirror = self._add_positive_control_failure("lora", 7411)
        source_paths = self._source_paths()
        target = mirror.with_name("positive-control-mirror-target.json")
        mirror.rename(target)
        mirror.symlink_to(target)

        with self.assertRaisesRegex(RuntimeError, "lacks its identical tracked mirror"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertTrue(all(path.exists() for path in source_paths))
        self.assertFalse(self._archive_root().exists())

    def test_source_mismatch_refuses_without_deleting_any_setup(self) -> None:
        source_paths = self._source_paths()
        manifest_path = self.experiment / "data" / "generated" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_contract_sha256"] = "c" * 64
        self._write_json(manifest_path, manifest)
        with self.assertRaisesRegex(RuntimeError, "source_contract_sha256 mismatch"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )
        self.assertTrue(all(path.exists() for path in source_paths))
        self.assertFalse(self._archive_root().exists())

    def test_nonempty_contrast_ledger_refuses_without_deletion(self) -> None:
        source_paths = self._source_paths()
        ledger_path = (
            self.experiment / "data" / "generated" / "contrast_access_ledger.json"
        )
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger.pop("receipt_identity_sha256")
        ledger["events"] = [{"split": "contrast_depth"}]
        ledger = self._signed(ledger)
        self._write_json(ledger_path, ledger)
        with self.assertRaisesRegex(RuntimeError, "ledger is not empty"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )
        self.assertTrue(all(path.exists() for path in source_paths))
        self.assertFalse(self._archive_root().exists())

    def test_overwrite_and_partial_or_unknown_inventory_are_refused(self) -> None:
        source_paths = self._source_paths()
        self._archive_root().mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "archive receipt is not"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )
        self.assertTrue(all(path.exists() for path in source_paths))

        self._archive_root().rmdir()
        unknown = self.experiment / "data" / "generated" / "unexpected.json"
        unknown.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "partial or unknown"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )
        self.assertTrue(all(path.exists() for path in source_paths))

        unknown.unlink()
        missing = self.experiment / "runs" / "setup" / "initialization_seed7413.json"
        missing.unlink()
        with self.assertRaisesRegex(RuntimeError, "partial or unknown"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )
        self.assertTrue(all(path.exists() for path in source_paths if path != missing))

    def test_precommit_failure_never_deletes_current_setup(self) -> None:
        source_paths = self._source_paths()
        with mock.patch.object(
            self.module,
            "_commit_staged_archive",
            side_effect=RuntimeError("synthetic precommit failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic precommit"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        self.assertTrue(all(path.exists() for path in source_paths))
        self.assertFalse(self._archive_root().exists())
        partials = list(
            self._archive_root().parent.glob(f".source_{self.OLD_SOURCE}.tmp-*")
        )
        self.assertEqual(partials, [])

    def test_stale_private_archive_and_receipt_stages_are_recovered(self) -> None:
        archive_parent = self._archive_root().parent
        archive_parent.mkdir(parents=True)
        stale_archive = archive_parent / f".source_{self.OLD_SOURCE}.tmp-dead"
        stale_archive.mkdir()
        (stale_archive / "partial.bin").write_bytes(b"partial")
        tracked = self._tracked_archive_receipt()
        stale_receipt = tracked.parent / f".{tracked.name}.tmp-dead"
        stale_receipt.write_bytes(b"partial receipt")

        receipt = self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )

        self.assertEqual(receipt["status"], "INVALIDATED_SETUP_ARCHIVED")
        self.assertFalse(stale_archive.exists())
        self.assertFalse(stale_receipt.exists())
        self.assertTrue(self._archive_root().is_dir())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink creation is unavailable")
    def test_unsafe_stale_archive_stage_is_not_cleaned_through(self) -> None:
        source_paths = self._source_paths()
        archive_parent = self._archive_root().parent
        archive_parent.mkdir(parents=True)
        target = self.repo / "stale-stage-target"
        target.mkdir()
        sentinel = target / "sentinel.bin"
        sentinel.write_bytes(b"preserve")
        stale = archive_parent / f".source_{self.OLD_SOURCE}.tmp-hostile"
        stale.symlink_to(target, target_is_directory=True)

        with self.assertRaisesRegex(RuntimeError, "stale.*unsafe"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertEqual(sentinel.read_bytes(), b"preserve")
        self.assertTrue(stale.is_symlink())
        self.assertTrue(all(path.exists() for path in source_paths))

    def test_directory_commit_collision_preserves_competing_destination(self) -> None:
        source_paths = self._source_paths()
        original_rename = self.module.rename_new_entry

        def install_competitor(root: Path, source: Path, destination: Path) -> None:
            destination.mkdir()
            (destination / "competitor.bin").write_bytes(b"competing archive")
            original_rename(root, source, destination)

        with mock.patch.object(
            self.module,
            "rename_new_entry",
            side_effect=install_competitor,
        ):
            with self.assertRaisesRegex(RuntimeError, "overwrite or alias"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )

        self.assertEqual(
            (self._archive_root() / "competitor.bin").read_bytes(),
            b"competing archive",
        )
        self.assertTrue(all(path.exists() for path in source_paths))
        self.assertEqual(
            list(
                self._archive_root().parent.glob(
                    f".source_{self.OLD_SOURCE}.tmp-*"
                )
            ),
            [],
        )

    def test_tracked_receipt_failure_keeps_all_sources_after_archive_commit(self) -> None:
        source_paths = self._source_paths()
        with mock.patch.object(
            self.module,
            "_atomic_tracked_receipt",
            side_effect=RuntimeError("synthetic tracked receipt failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "tracked receipt failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        self.assertTrue(all(path.exists() for path in source_paths))
        self.assertTrue((self._archive_root() / "archive_receipt.json").is_file())
        self.assertFalse(self._tracked_archive_receipt().exists())

        receipt = self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )

        self.assertTrue(self._tracked_archive_receipt().is_file())
        self.assertTrue(all(not path.exists() for path in source_paths))
        self.assertEqual(
            self._tracked_archive_receipt().read_bytes(),
            (self._archive_root() / "archive_receipt.json").read_bytes(),
        )
        self.assertEqual(receipt["status"], "INVALIDATED_SETUP_ARCHIVED")

    def test_mid_zeroization_failure_resumes_from_complete_quarantine(self) -> None:
        source_paths = self._source_paths()
        original_zeroize = self.module._zeroize_quarantined_descriptor
        calls = 0

        def fail_second(descriptor: int, record: dict) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError(errno.EIO, "synthetic second-zeroize failure")
            original_zeroize(descriptor, record)

        with mock.patch.object(
            self.module,
            "_zeroize_quarantined_descriptor",
            side_effect=fail_second,
        ):
            with self.assertRaisesRegex(RuntimeError, "second-zeroize failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )

        self.assertTrue(all(not path.exists() for path in source_paths))
        self.assertTrue(self._archive_root().is_dir())
        self.assertTrue(self._tracked_archive_receipt().is_file())

        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )

        self.assertTrue(all(not path.exists() for path in source_paths))
        self.assertEqual(
            {path.name for path in (self.experiment / "runs" / "cpu_smoke").iterdir()},
            {".gitkeep"},
        )
        self.assertEqual(
            {path.name for path in (self.experiment / "runs" / "setup").iterdir()},
            {".gitkeep"},
        )
        self.assert_complete_zero_quarantine()

    def test_canonical_replacement_race_never_deletes_unvalidated_bytes(self) -> None:
        items = self.module._inventory(self.config)
        target = next(
            item
            for item in items
            if item.source.name == "initialization_seed7411.pt"
        )
        original_bytes = target.source.read_bytes()
        preserved = target.source.with_name("preserved-original-after-race.bin")
        replacement_bytes = b"concurrent replacement that was never validated"
        real_move = self.module.move_new_entry
        raced = False

        def replace_before_move(root: Path, source: Path, destination: Path) -> None:
            nonlocal raced
            if source == target.source:
                source.rename(preserved)
                source.write_bytes(replacement_bytes)
                raced = True
            real_move(root, source, destination)

        with mock.patch.object(
            self.module,
            "move_new_entry",
            side_effect=replace_before_move,
        ):
            with self.assertRaisesRegex(RuntimeError, "differs from its archived record"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )

        self.assertTrue(raced)
        self.assertEqual(preserved.read_bytes(), original_bytes)
        quarantined = self.module._quarantine_path(
            self._quarantine_root(), target
        )
        self.assertEqual(quarantined.read_bytes(), replacement_bytes)
        self.assertFalse(target.source.exists())

    def test_quarantine_path_replacement_preserves_both_bindings(self) -> None:
        items = self.module._inventory(self.config)
        target = items[0]
        target_quarantine = self.module._quarantine_path(
            self._quarantine_root(), target
        )
        real_zeroize = self.module._zeroize_quarantined_descriptor
        preserved = target_quarantine.with_name(
            "preserved-original-after-cleanup-race.bin"
        )
        replacement_bytes = b"do not delete this replacement"
        raced = False

        def replace_quarantine(descriptor: int, record: dict) -> None:
            nonlocal raced
            if not raced:
                target_quarantine.rename(preserved)
                target_quarantine.write_bytes(replacement_bytes)
                raced = True
            real_zeroize(descriptor, record)

        with mock.patch.object(
            self.module,
            "_zeroize_quarantined_descriptor",
            side_effect=replace_quarantine,
        ):
            with self.assertRaisesRegex(RuntimeError, "held artifact changed"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )

        self.assertTrue(raced)
        archived_record = json.loads(
            (self._archive_root() / "archive_receipt.json").read_text(
                encoding="utf-8"
            )
        )["files"][0]
        self.assertEqual(preserved.read_bytes(), b"")
        self.assertGreater(archived_record["bytes"], 0)
        self.assertEqual(target_quarantine.read_bytes(), replacement_bytes)

    def test_quarantine_destination_collision_preserves_source_and_competitor(self) -> None:
        items = self.module._inventory(self.config)
        target = items[0]
        original_bytes = target.source.read_bytes()
        destination = self.module._quarantine_path(
            self._quarantine_root(), target
        )
        competitor = b"competing quarantine"
        real_move = self.module.move_new_entry

        def collide(root: Path, source: Path, new: Path) -> None:
            if source == target.source:
                new.parent.mkdir(parents=True, exist_ok=True)
                new.write_bytes(competitor)
            real_move(root, source, new)

        with mock.patch.object(
            self.module,
            "move_new_entry",
            side_effect=collide,
        ):
            with self.assertRaisesRegex(RuntimeError, "atomic no-clobber"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )

        self.assertEqual(target.source.read_bytes(), original_bytes)
        self.assertEqual(destination.read_bytes(), competitor)

    def test_resume_revalidates_nontrigger_failed_g0_mirror_before_deletion(self) -> None:
        _, mirror = self._replace_g0_with_failure("lora", 7411)
        source_paths = self._source_paths()
        with self._fail_move_patch(message="synthetic second-move failure"):
            with self.assertRaisesRegex(RuntimeError, "second-move failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        remaining_before_resume = {
            path for path in source_paths if os.path.lexists(path)
        }
        self.assertTrue(remaining_before_resume)
        mirror.unlink()

        with self.assertRaisesRegex(RuntimeError, "lacks its identical tracked mirror"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )
        self.assertEqual(
            remaining_before_resume,
            {path for path in source_paths if os.path.lexists(path)},
        )

    def test_empty_source_parents_are_retained_as_regeneration_surface(self) -> None:
        source_paths = self._source_paths()
        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )
        self.assertTrue(all(not path.exists() for path in source_paths))
        for path in (
            self.experiment / "runs" / "cpu_smoke",
            self.experiment / "runs" / "setup",
        ):
            self.assertTrue(path.is_dir())
            self.assertEqual({child.name for child in path.iterdir()}, {".gitkeep"})

    def test_setup_sentinels_are_required_empty_and_inode_distinct(self) -> None:
        sentinels = (
            self.experiment / "runs" / "cpu_smoke" / ".gitkeep",
            self.experiment / "runs" / "setup" / ".gitkeep",
        )
        for sentinel in sentinels:
            with self.subTest(parent=sentinel.parent.name, defect="missing"):
                sentinel.unlink()
                with self.assertRaises(RuntimeError):
                    self.module._inventory(self.config)
                sentinel.touch()
            with self.subTest(parent=sentinel.parent.name, defect="nonempty"):
                sentinel.write_bytes(b"not-structural")
                with self.assertRaises(RuntimeError):
                    self.module._inventory(self.config)
                sentinel.write_bytes(b"")
            with self.subTest(parent=sentinel.parent.name, defect="hardlink"):
                alias = self.repo / f"{sentinel.parent.name}-sentinel-alias"
                os.link(sentinel, alias)
                try:
                    with self.assertRaises(RuntimeError):
                        self.module._inventory(self.config)
                finally:
                    alias.unlink()

    def test_unknown_residue_blocks_resume_before_further_deletion(self) -> None:
        source_paths = self._source_paths()
        with self._fail_move_patch(message="synthetic second-move failure"):
            with self.assertRaisesRegex(RuntimeError, "second-move failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        late = self.experiment / "runs" / "setup" / "late-residue.json"
        late.write_text("{}\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "unknown residue"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        remaining_before_resume = {
            path for path in source_paths if os.path.lexists(path)
        }
        self.assertGreater(len(remaining_before_resume), 0)
        late.unlink()
        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )
        self.assertTrue(all(not path.exists() for path in source_paths))

    def test_tampered_remaining_source_blocks_resume_before_deletion(self) -> None:
        source_paths = self._source_paths()
        with self._fail_move_patch(message="synthetic second-move failure"):
            with self.assertRaisesRegex(RuntimeError, "second-move failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        tampered = sorted(
            (path for path in source_paths if path.exists()),
            key=lambda path: path.as_posix(),
        )[0]
        tampered.write_bytes(b"tampered")
        remaining_before_resume = {
            path for path in source_paths if os.path.lexists(path)
        }

        with self.assertRaisesRegex(RuntimeError, "differs from archive"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertEqual(
            remaining_before_resume,
            {path for path in source_paths if os.path.lexists(path)},
        )

    def test_fsync_failure_after_partial_cleanup_is_resumable(self) -> None:
        source_paths = self._source_paths()
        with self._fail_move_patch(message="synthetic second-move failure"):
            with self.assertRaisesRegex(RuntimeError, "second-move failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        original_fsync = self.module.fsync_canonical_directory
        failed = False

        def fail_once(root: Path, path: Path) -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise self.module.StableArtifactError(
                    "synthetic cleanup fsync failure"
                )
            original_fsync(root, path)

        with mock.patch.object(
            self.module,
            "fsync_canonical_directory",
            side_effect=fail_once,
        ):
            with self.assertRaisesRegex(RuntimeError, "could not be fsynced"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )

        self.assertTrue(any(not path.exists() for path in source_paths))
        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )
        self.assertTrue(all(not path.exists() for path in source_paths))

    def test_zeroized_file_fsync_failure_is_resumable_and_refsynced(self) -> None:
        real_zeroize = self.module._zeroize_quarantined_descriptor
        interrupted = False

        def truncate_then_fail(descriptor: int, record: dict) -> None:
            nonlocal interrupted
            if not interrupted:
                real_fsync = self.module.os.fsync
                failed = False

                def fail_after_flush(descriptor: int) -> None:
                    nonlocal failed
                    real_fsync(descriptor)
                    if not failed and stat.S_ISREG(
                        os.fstat(descriptor).st_mode
                    ):
                        failed = True
                        raise OSError(
                            errno.EIO,
                            "synthetic post-zeroization fsync failure",
                        )

                interrupted = True
                with mock.patch.object(
                    self.module.os,
                    "fsync",
                    side_effect=fail_after_flush,
                ):
                    real_zeroize(descriptor, record)
                return
            real_zeroize(descriptor, record)

        with mock.patch.object(
            self.module,
            "_zeroize_quarantined_descriptor",
            side_effect=truncate_then_fail,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "post-zeroization fsync failure"
            ):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )

        self.assertTrue(interrupted)
        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )
        self.assert_complete_zero_quarantine()

    def test_last_zeroized_file_fsync_failure_reconfirms_completed_quarantine(self) -> None:
        total_files = len(self._source_paths())
        real_zeroize = self.module._zeroize_quarantined_descriptor
        calls = 0

        def fail_after_last_file_flush(descriptor: int, record: dict) -> None:
            nonlocal calls
            calls += 1
            if calls != total_files:
                real_zeroize(descriptor, record)
                return
            real_fsync = self.module.os.fsync
            failed = False

            def flush_then_fail(candidate: int) -> None:
                nonlocal failed
                real_fsync(candidate)
                if candidate == descriptor and not failed:
                    failed = True
                    raise OSError(
                        errno.EIO,
                        "synthetic last-leaf fsync failure",
                    )

            with mock.patch.object(
                self.module.os,
                "fsync",
                side_effect=flush_then_fail,
            ):
                real_zeroize(descriptor, record)

        with mock.patch.object(
            self.module,
            "_zeroize_quarantined_descriptor",
            side_effect=fail_after_last_file_flush,
        ):
            with self.assertRaisesRegex(RuntimeError, "last-leaf fsync failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )

        receipt = json.loads(
            (self._archive_root() / "archive_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        items = self.module._items_from_file_records(
            self.config,
            receipt["files"],
        )[0]
        complete, _ = self.module._quarantine_snapshot(
            self._quarantine_root(),
            items,
            receipt["files"],
        )
        self.assertTrue(complete)
        regenerated = items[0].source
        regenerated.write_bytes(b"regenerated replacement-source bytes")

        reconfirmed = 0

        def count_reconfirmation(descriptor: int, record: dict) -> None:
            nonlocal reconfirmed
            reconfirmed += 1
            real_zeroize(descriptor, record)

        with mock.patch.object(
            self.module,
            "_zeroize_quarantined_descriptor",
            side_effect=count_reconfirmation,
        ):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )
        self.assertEqual(reconfirmed, total_files)
        self.assertEqual(
            regenerated.read_bytes(),
            b"regenerated replacement-source bytes",
        )

    def test_corrupt_last_quarantine_leaf_blocks_every_truncate(self) -> None:
        items = self.module._inventory(self.config)
        records = self.module._file_records(items)
        target = self.module._quarantine_path(
            self._quarantine_root(),
            items[-1],
        )
        real_state = self.module._quarantined_descriptor_state
        injected = False

        def corrupt_before_global_validation(descriptor: int, record: dict) -> bool:
            nonlocal injected
            if not injected:
                target.write_bytes(b"corrupt final quarantine leaf")
                injected = True
            return real_state(descriptor, record)

        with (
            mock.patch.object(
                self.module,
                "_quarantined_descriptor_state",
                side_effect=corrupt_before_global_validation,
            ),
            mock.patch.object(
                self.module.os,
                "ftruncate",
                side_effect=AssertionError("no quarantine leaf may truncate"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "differs from its archived record"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )

        self.assertTrue(injected)
        for item, record in zip(items[:-1], records[:-1], strict=True):
            quarantined = self.module._quarantine_path(
                self._quarantine_root(),
                item,
            )
            self.assertEqual(quarantined.stat().st_size, record["bytes"])
            self.assertEqual(self.module._sha256(quarantined), record["sha256"])

    def test_archive_payload_mutation_blocks_every_quarantine_truncate(self) -> None:
        items = self.module._inventory(self.config)
        records = self.module._file_records(items)
        archive_payload = self._archive_root() / records[-1]["path"]
        real_state = self.module._quarantined_descriptor_state
        injected = False

        def mutate_archive_during_q_validation(
            descriptor: int,
            record: dict,
        ) -> bool:
            nonlocal injected
            if not injected:
                archive_payload.write_bytes(b"mutated durable archive payload")
                injected = True
            return real_state(descriptor, record)

        with (
            mock.patch.object(
                self.module,
                "_quarantined_descriptor_state",
                side_effect=mutate_archive_during_q_validation,
            ),
            mock.patch.object(
                self.module.os,
                "ftruncate",
                side_effect=AssertionError("no quarantine leaf may truncate"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "held artifact changed"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        self.assertTrue(injected)

    def test_archive_receipt_mutation_blocks_every_quarantine_truncate(self) -> None:
        archive_receipt = self._archive_root() / "archive_receipt.json"
        real_state = self.module._quarantined_descriptor_state
        injected = False

        def mutate_receipt_during_q_validation(
            descriptor: int,
            record: dict,
        ) -> bool:
            nonlocal injected
            if not injected:
                archive_receipt.write_bytes(
                    archive_receipt.read_bytes() + b"\n"
                )
                injected = True
            return real_state(descriptor, record)

        with (
            mock.patch.object(
                self.module,
                "_quarantined_descriptor_state",
                side_effect=mutate_receipt_during_q_validation,
            ),
            mock.patch.object(
                self.module.os,
                "ftruncate",
                side_effect=AssertionError("no quarantine leaf may truncate"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "held artifact changed"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        self.assertTrue(injected)

    def test_quarantine_directory_fsync_failure_precedes_every_source_move(self) -> None:
        source_paths = self._source_paths()
        quarantine_root = self._quarantine_root()
        real_fsync = self.module.fsync_canonical_directory
        failed = False

        def fail_quarantine_root_once(root: Path, path: Path) -> None:
            nonlocal failed
            if path == quarantine_root and not failed:
                failed = True
                raise self.module.StableArtifactError(
                    "synthetic quarantine-root fsync failure"
                )
            real_fsync(root, path)

        with mock.patch.object(
            self.module,
            "fsync_canonical_directory",
            side_effect=fail_quarantine_root_once,
        ):
            with self.assertRaisesRegex(RuntimeError, "could not be fsynced"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        self.assertTrue(failed)
        self.assertTrue(all(path.exists() for path in source_paths))

    def test_hardlinked_setup_source_is_rejected_before_archive_commit(self) -> None:
        target = next(
            item.source
            for item in self.module._inventory(self.config)
            if item.source.name == "initialization_seed7411.pt"
        )
        alias = target.with_name("unregistered-hardlink-alias.bin")
        os.link(target, alias)
        try:
            with self.assertRaisesRegex(RuntimeError, "changed before archival"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
            self.assertFalse(self._archive_root().exists())
            self.assertTrue(target.exists())
            self.assertTrue(alias.exists())
        finally:
            alias.unlink()

    def test_tracked_only_state_is_rejected_without_deletion(self) -> None:
        source_paths = self._source_paths()
        self._tracked_archive_receipt().parent.mkdir(parents=True, exist_ok=True)
        self._tracked_archive_receipt().write_text("{}\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "without its archive"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertTrue(all(path.exists() for path in source_paths))

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink creation is unavailable")
    def test_symlinked_archive_parent_is_rejected_before_archival(self) -> None:
        source_paths = self._source_paths()
        large_root = self.module._large_root(self.config)
        target = large_root / "invalidated-setup-target"
        target.mkdir()
        (large_root / "invalidated_setup").symlink_to(
            target,
            target_is_directory=True,
        )

        with self.assertRaisesRegex(RuntimeError, "symlinked path component"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertTrue(all(path.exists() for path in source_paths))

    def test_nonidentical_tracked_receipt_blocks_resume(self) -> None:
        source_paths = self._source_paths()
        with self._fail_move_patch(message="synthetic second-move failure"):
            with self.assertRaisesRegex(RuntimeError, "second-move failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        tracked = self._tracked_archive_receipt()
        tracked.write_bytes(tracked.read_bytes() + b"\n")
        remaining_before_resume = {
            path for path in source_paths if os.path.lexists(path)
        }

        with self.assertRaisesRegex(RuntimeError, "byte-identical"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertEqual(
            remaining_before_resume,
            {path for path in source_paths if os.path.lexists(path)},
        )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink creation is unavailable")
    def test_symlinked_tracked_receipt_blocks_resume(self) -> None:
        source_paths = self._source_paths()
        with self._fail_move_patch(message="synthetic second-move failure"):
            with self.assertRaisesRegex(RuntimeError, "second-move failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        tracked = self._tracked_archive_receipt()
        target = tracked.with_name("invalidation-receipt-target.json")
        tracked.rename(target)
        tracked.symlink_to(target)
        remaining_before_resume = {
            path for path in source_paths if os.path.lexists(path)
        }

        with self.assertRaisesRegex(RuntimeError, "symlinked path component"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertEqual(
            remaining_before_resume,
            {path for path in source_paths if os.path.lexists(path)},
        )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink creation is unavailable")
    def test_symlinked_trigger_receipt_is_rejected_before_archival(self) -> None:
        source_paths = self._source_paths()
        target = self.trigger.with_name("trigger-target.json")
        self.trigger.rename(target)
        self.trigger.symlink_to(target)

        with self.assertRaisesRegex(RuntimeError, "symlinked path component"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertTrue(all(path.exists() for path in source_paths))
        self.assertFalse(self._archive_root().exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink creation is unavailable")
    def test_symlinked_generated_data_root_blocks_resume_before_deletion(self) -> None:
        with self._fail_move_patch(message="synthetic second-move failure"):
            with self.assertRaisesRegex(RuntimeError, "second-move failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        data_dir = self.experiment / self.config["paths"]["data_dir"]
        moved = data_dir.with_name("generated-real")
        data_dir.rename(moved)
        data_dir.symlink_to(moved, target_is_directory=True)
        preserved = {path.name: path.read_bytes() for path in moved.iterdir()}

        with self.assertRaisesRegex(RuntimeError, "symlinked path component"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertEqual(
            {path.name: path.read_bytes() for path in moved.iterdir()},
            preserved,
        )

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink creation is unavailable")
    def test_symlinked_runs_root_blocks_resume_before_deletion(self) -> None:
        with self._fail_move_patch(message="synthetic second-move failure"):
            with self.assertRaisesRegex(RuntimeError, "second-move failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        runs_dir = self.experiment / self.config["paths"]["runs_dir"]
        moved = runs_dir.with_name("runs-real")
        runs_dir.rename(moved)
        runs_dir.symlink_to(moved, target_is_directory=True)
        setup_files = {
            path.relative_to(moved).as_posix(): path.read_bytes()
            for path in moved.rglob("*")
            if path.is_file()
        }

        with self.assertRaisesRegex(RuntimeError, "symlinked path component"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertEqual(
            {
                path.relative_to(moved).as_posix(): path.read_bytes()
                for path in moved.rglob("*")
                if path.is_file()
            },
            setup_files,
        )

    def test_completed_cleanup_rerun_is_idempotent(self) -> None:
        first = self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )
        archive_receipt = self._archive_root() / "archive_receipt.json"
        tracked_receipt = self._tracked_archive_receipt()
        before = {
            "archive_bytes": archive_receipt.read_bytes(),
            "archive_mtime": archive_receipt.stat().st_mtime_ns,
            "tracked_bytes": tracked_receipt.read_bytes(),
            "tracked_mtime": tracked_receipt.stat().st_mtime_ns,
        }
        archived_items = self.module._items_from_file_records(
            self.config, first["files"]
        )[0]
        regenerated = archived_items[0].source
        regenerated.parent.mkdir(parents=True, exist_ok=True)
        regenerated.write_bytes(b"replacement-source regenerated setup")

        with (
            mock.patch.object(
                self.module,
                "_stage_archive",
                side_effect=AssertionError("stage must not rerun"),
            ),
            mock.patch.object(
                self.module,
                "_atomic_tracked_receipt",
                side_effect=AssertionError("receipt must not rewrite"),
            ),
            mock.patch.object(
                self.module,
                "move_new_entry",
                side_effect=AssertionError("completed cleanup must not move sources"),
            ),
            mock.patch.object(
                self.module.os,
                "ftruncate",
                side_effect=AssertionError(
                    "completed cleanup must not truncate zero quarantine"
                ),
            ),
            mock.patch.object(
                self.module,
                "fsync_canonical_directory",
                wraps=self.module.fsync_canonical_directory,
            ) as fsync_calls,
        ):
            second = self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertEqual(second, first)
        self.assertEqual(archive_receipt.read_bytes(), before["archive_bytes"])
        self.assertEqual(archive_receipt.stat().st_mtime_ns, before["archive_mtime"])
        self.assertEqual(tracked_receipt.read_bytes(), before["tracked_bytes"])
        self.assertEqual(tracked_receipt.stat().st_mtime_ns, before["tracked_mtime"])
        self.assertEqual(
            regenerated.read_bytes(), b"replacement-source regenerated setup"
        )
        fsynced_paths = {call.args[1] for call in fsync_calls.call_args_list}
        self.assertTrue(
            {item.source.parent for item in archived_items}.issubset(
                fsynced_paths
            )
        )

    def test_tampered_archive_blocks_completed_rerun(self) -> None:
        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )
        extra = self._archive_root() / "unexpected.bin"
        extra.write_bytes(b"unexpected")

        with self.assertRaisesRegex(RuntimeError, "partial or unknown"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO creation is unavailable")
    def test_special_archive_entry_blocks_completed_rerun(self) -> None:
        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )
        fifo = self._archive_root() / "unexpected.fifo"
        os.mkfifo(fifo)

        with self.assertRaisesRegex(RuntimeError, "unsafe entry"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )


if __name__ == "__main__":
    unittest.main()
