from __future__ import annotations

import copy
import errno
import importlib.util
import json
import os
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
        g0_path = setup_dir / f"g0_{capacity}_seed{seed}.json"
        g0 = json.loads(g0_path.read_text(encoding="utf-8"))
        manifest_path = self.experiment / "data" / "generated" / "manifest.json"
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
                "data_manifest_sha256": self.module._sha256(manifest_path),
                "g0_lineage": {
                    "path": g0_path.relative_to(self.repo).as_posix(),
                    "sha256": self.module._sha256(g0_path),
                    "receipt_identity_sha256": g0["receipt_identity_sha256"],
                    "status": "MODEL_SMOKE_PASS",
                    "phase": f"{capacity}_g0",
                },
                "setup": g0["setup"],
                "authorizes_training": False,
                "authorizes_result_training": False,
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
        self.assertFalse((self.experiment / "runs" / "cpu_smoke").exists())
        self.assertFalse((self.experiment / "runs" / "setup").exists())
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

        self.assertFalse((self.experiment / "runs" / "cpu_smoke").exists())
        self.assertFalse((self.experiment / "runs" / "setup").exists())
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

    def test_mid_cleanup_failure_resumes_from_exact_or_absent_sources(self) -> None:
        source_paths = self._source_paths()
        original_unlink = self.module._unlink_source
        calls = 0

        def fail_second(path: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError(errno.EIO, "synthetic second-unlink failure")
            original_unlink(path)

        with mock.patch.object(self.module, "_unlink_source", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "second-unlink failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )

        existing_after_failure = {path for path in source_paths if path.exists()}
        self.assertGreater(len(existing_after_failure), 0)
        self.assertLess(len(existing_after_failure), len(source_paths))
        self.assertTrue(self._archive_root().is_dir())
        self.assertTrue(self._tracked_archive_receipt().is_file())

        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )

        self.assertTrue(all(not path.exists() for path in source_paths))
        self.assertFalse((self.experiment / "runs" / "cpu_smoke").exists())
        self.assertFalse((self.experiment / "runs" / "setup").exists())

    def test_rmdir_failure_resumes_after_all_sources_are_absent(self) -> None:
        source_paths = self._source_paths()
        original_remove = self.module._remove_empty_source_parent
        failed = False

        def fail_cpu_once(path: Path) -> None:
            nonlocal failed
            if path.name == "cpu_smoke" and not failed:
                failed = True
                raise OSError(errno.EIO, "synthetic rmdir failure")
            original_remove(path)

        with mock.patch.object(
            self.module,
            "_remove_empty_source_parent",
            side_effect=fail_cpu_once,
        ):
            with self.assertRaisesRegex(OSError, "synthetic rmdir failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )

        self.assertTrue(all(not path.exists() for path in source_paths))
        self.assertTrue((self.experiment / "runs" / "cpu_smoke").is_dir())

        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )
        self.assertFalse((self.experiment / "runs" / "cpu_smoke").exists())
        self.assertFalse((self.experiment / "runs" / "setup").exists())

    def test_unknown_residue_blocks_resume_before_further_deletion(self) -> None:
        source_paths = self._source_paths()
        with mock.patch.object(
            self.module,
            "_unlink_source",
            side_effect=OSError(errno.EIO, "synthetic first-unlink failure"),
        ):
            with self.assertRaisesRegex(OSError, "first-unlink failure"):
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

        self.assertTrue(all(path.exists() for path in source_paths))
        late.unlink()
        self.module.archive_invalidated_setup(
            self.config,
            self.OLD_SOURCE,
            self.trigger,
        )
        self.assertTrue(all(not path.exists() for path in source_paths))

    def test_tampered_remaining_source_blocks_resume_before_deletion(self) -> None:
        source_paths = self._source_paths()
        with mock.patch.object(
            self.module,
            "_unlink_source",
            side_effect=OSError(errno.EIO, "synthetic first-unlink failure"),
        ):
            with self.assertRaisesRegex(OSError, "first-unlink failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        tampered = sorted(source_paths, key=lambda path: path.as_posix())[0]
        tampered.write_bytes(b"tampered")

        with self.assertRaisesRegex(RuntimeError, "differs from archive"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertTrue(
            all(path.exists() for path in source_paths if path != tampered)
        )

    def test_fsync_failure_after_partial_cleanup_is_resumable(self) -> None:
        source_paths = self._source_paths()
        with mock.patch.object(
            self.module,
            "_unlink_source",
            side_effect=OSError(errno.EIO, "synthetic first-unlink failure"),
        ):
            with self.assertRaisesRegex(OSError, "first-unlink failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        original_fsync = self.module._fsync_directory
        failed = False

        def fail_once(path: Path) -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise OSError(errno.EIO, "synthetic cleanup fsync failure")
            original_fsync(path)

        with mock.patch.object(
            self.module,
            "_fsync_directory",
            side_effect=fail_once,
        ):
            with self.assertRaisesRegex(OSError, "cleanup fsync failure"):
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
        with mock.patch.object(
            self.module,
            "_unlink_source",
            side_effect=OSError(errno.EIO, "synthetic first-unlink failure"),
        ):
            with self.assertRaisesRegex(OSError, "first-unlink failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        tracked = self._tracked_archive_receipt()
        tracked.write_bytes(tracked.read_bytes() + b"\n")

        with self.assertRaisesRegex(RuntimeError, "byte-identical"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertTrue(all(path.exists() for path in source_paths))

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink creation is unavailable")
    def test_symlinked_tracked_receipt_blocks_resume(self) -> None:
        source_paths = self._source_paths()
        with mock.patch.object(
            self.module,
            "_unlink_source",
            side_effect=OSError(errno.EIO, "synthetic first-unlink failure"),
        ):
            with self.assertRaisesRegex(OSError, "first-unlink failure"):
                self.module.archive_invalidated_setup(
                    self.config,
                    self.OLD_SOURCE,
                    self.trigger,
                )
        tracked = self._tracked_archive_receipt()
        target = tracked.with_name("invalidation-receipt-target.json")
        tracked.rename(target)
        tracked.symlink_to(target)

        with self.assertRaisesRegex(RuntimeError, "symlinked path component"):
            self.module.archive_invalidated_setup(
                self.config,
                self.OLD_SOURCE,
                self.trigger,
            )

        self.assertTrue(all(path.exists() for path in source_paths))

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
        with mock.patch.object(
            self.module,
            "_unlink_source",
            side_effect=OSError(errno.EIO, "synthetic first-unlink failure"),
        ):
            with self.assertRaisesRegex(OSError, "first-unlink failure"):
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
        with mock.patch.object(
            self.module,
            "_unlink_source",
            side_effect=OSError(errno.EIO, "synthetic first-unlink failure"),
        ):
            with self.assertRaisesRegex(OSError, "first-unlink failure"):
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
                "_unlink_source",
                side_effect=AssertionError("no source remains"),
            ),
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
