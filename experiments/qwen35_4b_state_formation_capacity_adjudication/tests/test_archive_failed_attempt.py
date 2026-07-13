from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.attempt_receipts as attempt_receipts  # noqa: E402
from src.attempt_receipts import (  # noqa: E402
    AttemptReceiptError,
    archive_lineage,
    ensure_attempt_output,
    locked_regular,
    prepare_training_attempt,
    start_training_attempt,
    tree_manifest,
)
from src.config import load_config  # noqa: E402
from src.training_receipts import TrainingCell  # noqa: E402


def load_archiver():
    spec = importlib.util.spec_from_file_location(
        "capacity_failed_attempt_archiver", ROOT / "scripts" / "archive_failed_attempt.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed-attempt archiver cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FailedAttemptArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_archiver()
        self.config = copy.deepcopy(load_config(ROOT / "configs" / "default.yaml"))
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        self.experiment = (
            self.repo / "experiments" / self.config["experiment_id"]
        )
        self.experiment.mkdir(parents=True)
        (self.repo / "requirements-training.lock.txt").write_text(
            "synthetic-lock\n", encoding="utf-8"
        )
        self.patchers = (
            mock.patch.object(self.module, "ROOT", self.experiment),
            mock.patch.object(self.module, "REPO_ROOT", self.repo),
            mock.patch.object(
                self.module,
                "validate_design_receipt",
                return_value={"status": "DESIGN_FROZEN"},
            ),
            mock.patch.object(
                self.module,
                "design_lineage",
                return_value={
                    "path": "synthetic/design_receipt.json",
                    "sha256": "1" * 64,
                    "receipt_identity_sha256": "2" * 64,
                    "status": "DESIGN_FROZEN",
                    "phase": "design_boundary",
                },
            ),
            mock.patch.object(
                self.module, "source_contract_sha256", return_value="3" * 64
            ),
        )
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temporary.cleanup()

    def large_cell(self, cell: str = "lora_joint_seed7411") -> Path:
        return self.repo / "large_artifacts" / self.config["experiment_id"] / cell

    def tracked_cell(self, cell: str = "lora_joint_seed7411") -> Path:
        return self.experiment / "runs" / "training" / cell

    @staticmethod
    def populate(path: Path, label: str) -> None:
        (path / "nested").mkdir(parents=True, exist_ok=True)
        (path / "empty").mkdir(exist_ok=True)
        (path / "root.txt").write_text(f"root-{label}\n", encoding="utf-8")
        (path / "nested" / "payload.bin").write_bytes(f"payload-{label}".encode())

    def assert_zeroized_tombstone(self, path: Path) -> None:
        self.assertTrue(path.is_dir())
        self.assertFalse(path.is_symlink())
        for member in path.rglob("*"):
            self.assertFalse(member.is_symlink())
            self.assertTrue(member.is_dir() or member.is_file())
            if member.is_file():
                self.assertEqual(member.stat().st_size, 0)

    def _attempt_spec(self, cell_name: str) -> dict:
        identity = self.module._training_cell_identity(cell_name)
        if identity is None:
            raise AssertionError("test cell must be registered")
        capacity, objective, seed = identity
        stage = self.module._training_cell_stage(capacity, objective)
        cell = TrainingCell(stage, capacity, objective, seed)
        contract = self.module._training_contract(self.config, cell)
        external, tracked = self.module._training_pairs(self.config)[cell_name]
        return {
            "cell_object": cell,
            "header": {
                key: value for key, value in contract.identity.items() if key != "phase"
            },
            "cell": {
                "stage": stage,
                "capacity": capacity,
                "objective": objective,
                "seed": seed,
                "slug": cell_name,
            },
            "canonical_paths": [
                self.module._repo_relative(external),
                self.module._repo_relative(tracked),
            ],
            "context": {
                "test_contract": "durable-started-training-attempt",
                "cell": cell_name,
            },
            "external": external,
            "tracked": tracked,
        }

    def started_training(
        self,
        cell_name: str = "lora_joint_seed7411",
        *,
        present: tuple[str, ...] = ("external", "tracked"),
    ) -> tuple[dict, dict]:
        spec = self._attempt_spec(cell_name)
        authorization = prepare_training_attempt(
            self.repo,
            slug=cell_name,
            header=spec["header"],
            cell=spec["cell"],
            canonical_paths=spec["canonical_paths"],
            context=spec["context"],
            replay_archive=None,
        )
        ensure_attempt_output(spec["external"], authorization)
        start_training_attempt(
            self.repo,
            slug=cell_name,
            header=spec["header"],
            cell=spec["cell"],
            canonical_paths=spec["canonical_paths"],
            authorization=authorization,
        )
        marker = spec["external"] / "attempt.json"
        if "tracked" in present:
            spec["tracked"].mkdir(parents=True)
            shutil.copy2(marker, spec["tracked"] / marker.name)
        for side in present:
            self.populate(spec[side], f"{cell_name}:{side}")
        if "external" not in present:
            shutil.rmtree(spec["external"])
        return spec, authorization

    def _tracked_receipt_path(self, receipt: dict) -> Path:
        label = Path(receipt["archive_path"]).name
        return self.experiment / "runs" / "failures" / f"{label}.json"

    def test_allowlist_is_exact_and_two_paths_must_be_same_cell_companions(self) -> None:
        allowed = self.module._allowed_paths(self.config)
        self.assertEqual(len(allowed), 42)
        self.assertIn(self.large_cell(), allowed)
        self.assertIn(self.tracked_cell(), allowed)
        self.assertIn(self.experiment / "runs" / "lora_joint_seed7411_contrast", allowed)
        self.assertNotIn(self.experiment / "runs" / "state_bag_seed7411_trigger", allowed)

        unregistered = self.experiment / "runs" / "unregistered"
        self.populate(unregistered, "wrong")
        with self.assertRaisesRegex(RuntimeError, "noncanonical or unregistered"):
            self.module.archive_failed_attempt(self.config, [unregistered])

        first = self.large_cell("lora_joint_seed7411")
        unrelated = self.tracked_cell("fullrank_state_only_seed7413")
        self.populate(first, "first")
        self.populate(unrelated, "unrelated")
        with self.assertRaisesRegex(RuntimeError, "same-cell.*companion"):
            self.module.archive_failed_attempt(self.config, [first, unrelated])

    def test_training_archive_requires_durable_started_marker_and_journal(self) -> None:
        primary = self.large_cell()
        self.populate(primary, "unowned")
        with self.assertRaisesRegex(RuntimeError, "durable attempt marker"):
            self.module.archive_failed_attempt(self.config, [primary])
        self.assertTrue(primary.is_dir())

        shutil.rmtree(primary)
        spec, _ = self.started_training()
        journal = (
            self.experiment
            / "runs"
            / "attempts"
            / "training"
            / f"{spec['cell']['slug']}.json"
        )
        payload = json.loads(journal.read_text(encoding="utf-8"))
        payload["events"][-1]["state"] = "PREPARED"
        payload.pop("receipt_identity_sha256")
        payload["receipt_identity_sha256"] = self.module._canonical_sha256(payload)
        journal.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "durable STARTED journal head"):
            self.module.archive_failed_attempt(self.config, [spec["external"]])

    def test_archive_preserves_exact_tree_hashes_and_independent_receipts(self) -> None:
        spec, authorization = self.started_training()
        primary, companion = spec["external"], spec["tracked"]
        authority = self.module._archive_authority(
            self.config, [primary, companion], (primary, companion)
        )
        expected = [
            self.module._bind_archive_authority(self.module._manifest(path), authority)
            for path in (primary, companion)
        ]

        # Selecting only one side still captures both in canonical order.
        receipt = self.module.archive_failed_attempt(self.config, [companion])
        self.assertFalse(primary.exists())
        self.assertFalse(companion.exists())
        self.assertEqual(receipt["status"], "FAILED_ATTEMPT_ARCHIVED")
        self.assertIs(receipt["scientific_evidence"], False)
        self.assertEqual(receipt["attempts"], expected)
        embedded = receipt["attempts"][0]["archive_authority"]
        self.assertEqual(embedded["attempt_authorization"], authorization)
        self.assertEqual(embedded["present_paths"], spec["canonical_paths"])
        self.assertEqual(
            receipt["attempt_identity_sha256"],
            self.module._canonical_sha256({"attempts": expected}),
        )

        archive = self.repo / receipt["archive_path"]
        tracked_receipt = self._tracked_receipt_path(receipt)
        self.assertTrue((archive / f"source_1_{primary.name}" / "root.txt").is_file())
        self.assertTrue((archive / f"source_2_{companion.name}" / "root.txt").is_file())
        self.assertEqual(
            json.loads((archive / "archive_receipt.json").read_text(encoding="utf-8")),
            receipt,
        )
        self.assertEqual(json.loads(tracked_receipt.read_text(encoding="utf-8")), receipt)
        self.assertNotEqual(
            (archive / "archive_receipt.json").stat().st_ino,
            tracked_receipt.stat().st_ino,
        )

    def test_partial_training_sides_are_archived_with_full_canonical_authority(self) -> None:
        cases = (
            ("lora_state_only_seed7411", ("external",)),
            ("fullrank_joint_seed7413", ("tracked",)),
        )
        for cell_name, present in cases:
            with self.subTest(cell=cell_name):
                spec, _ = self.started_training(cell_name, present=present)
                selected = spec[present[0]]
                receipt = self.module.archive_failed_attempt(self.config, [selected])
                self.assertEqual(len(receipt["attempts"]), 1)
                authority = receipt["attempts"][0]["archive_authority"]
                self.assertEqual(authority["canonical_paths"], spec["canonical_paths"])
                self.assertEqual(
                    authority["present_paths"],
                    [self.module._repo_relative(selected)],
                )
                self.assertFalse(selected.exists())

    def test_existing_old_archive_cannot_capture_a_new_retry(self) -> None:
        spec, first_auth = self.started_training("lora_joint_seed7412", present=("external",))
        first = self.module.archive_failed_attempt(self.config, [spec["external"]])
        first_lineage = archive_lineage(
            self.repo, self._tracked_receipt_path(first), first
        )
        second_auth = prepare_training_attempt(
            self.repo,
            slug=spec["cell"]["slug"],
            header=spec["header"],
            cell=spec["cell"],
            canonical_paths=spec["canonical_paths"],
            context=spec["context"],
            replay_archive=first_lineage,
        )
        self.assertNotEqual(
            first_auth["attempt_identity_sha256"],
            second_auth["attempt_identity_sha256"],
        )
        ensure_attempt_output(spec["external"], second_auth)
        start_training_attempt(
            self.repo,
            slug=spec["cell"]["slug"],
            header=spec["header"],
            cell=spec["cell"],
            canonical_paths=spec["canonical_paths"],
            authorization=second_auth,
        )
        self.populate(spec["external"], "second-attempt")
        second = self.module.archive_failed_attempt(self.config, [spec["external"]])
        self.assertNotEqual(first["archive_path"], second["archive_path"])
        self.assertEqual(
            second["attempts"][0]["archive_authority"]["attempt_identity_sha256"],
            second_auth["attempt_identity_sha256"],
        )

    def test_retry_cleanup_tombstone_disambiguates_multiple_historical_archives(self) -> None:
        spec, _ = self.started_training("lora_joint_seed7412", present=("external",))
        first = self.module.archive_failed_attempt(self.config, [spec["external"]])
        lineage = archive_lineage(
            self.repo, self._tracked_receipt_path(first), first
        )
        second_auth = prepare_training_attempt(
            self.repo,
            slug=spec["cell"]["slug"],
            header=spec["header"],
            cell=spec["cell"],
            canonical_paths=spec["canonical_paths"],
            context=spec["context"],
            replay_archive=lineage,
        )
        ensure_attempt_output(spec["external"], second_auth)
        start_training_attempt(
            self.repo,
            slug=spec["cell"]["slug"],
            header=spec["header"],
            cell=spec["cell"],
            canonical_paths=spec["canonical_paths"],
            authorization=second_auth,
        )
        self.populate(spec["external"], "second-crash")
        with mock.patch.dict(
            os.environ,
            {"QWEN35_ARCHIVE_CRASH_AT": "source_1_renamed"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected archive crash"):
                self.module.archive_failed_attempt(
                    self.config, [spec["external"]]
                )
        self.assertFalse(spec["external"].exists())
        second = self.module.archive_failed_attempt(
            self.config, [spec["external"]]
        )
        self.assertNotEqual(first["archive_path"], second["archive_path"])
        self.assertEqual(
            second["attempts"][0]["archive_authority"]["attempt_identity_sha256"],
            second_auth["attempt_identity_sha256"],
        )

    def test_expected_attempt_identity_is_exact_and_exposed_by_cli(self) -> None:
        path = self.experiment / "runs" / "lora_joint_seed7411_trigger"
        invalid = (
            "a" * 63,
            "a" * 65,
            "A" * 64,
            "g" * 64,
            "a" * 32 + "-" + "b" * 31,
        )
        for identity in invalid:
            with self.subTest(identity=identity):
                with self.assertRaisesRegex(
                    RuntimeError, "exactly 64 lowercase hexadecimal"
                ):
                    self.module.archive_failed_attempt(
                        self.config,
                        [path],
                        expected_attempt_identity=identity,
                    )
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        self.module.parse_args(
                            [
                                "--path",
                                str(path),
                                "--attempt-identity",
                                identity,
                            ]
                        )

        canonical = "ab" * 32
        args = self.module.parse_args(
            ["--path", str(path), "--attempt-identity", canonical]
        )
        self.assertEqual(args.expected_attempt_identity, canonical)

    def test_expected_attempt_identity_must_match_fresh_authority(self) -> None:
        output = self.experiment / "runs" / "lora_joint_seed7411_trigger"
        self.populate(output, "fresh-evaluation")
        authority = self.module._archive_authority(self.config, [output], None)
        actual = authority["attempt_identity_sha256"]
        wrong = "0" * 64 if actual != "0" * 64 else "1" * 64

        with self.assertRaisesRegex(
            RuntimeError, "does not match the current durable attempt authority"
        ):
            self.module.archive_failed_attempt(
                self.config,
                [output],
                expected_attempt_identity=wrong,
            )
        self.assertEqual(
            (output / "root.txt").read_text(encoding="utf-8"),
            "root-fresh-evaluation\n",
        )

        receipt = self.module.archive_failed_attempt(
            self.config,
            [output],
            expected_attempt_identity=actual,
        )
        self.assertEqual(
            receipt["attempts"][0]["archive_authority"][
                "attempt_identity_sha256"
            ],
            actual,
        )

    def test_explicit_identity_selects_only_markerless_evaluation_retry(self) -> None:
        output = self.experiment / "runs" / "lora_joint_seed7411_trigger"
        self.populate(output, "first-evaluation")
        first = self.module.archive_failed_attempt(self.config, [output])
        first_authority_identity = first["attempts"][0]["archive_authority"][
            "attempt_identity_sha256"
        ]
        first_tombstone = output.with_name(
            f".{output.name}.archived-{first['attempt_identity_sha256']}-1"
        )
        self.assert_zeroized_tombstone(first_tombstone)

        self.populate(output, "second-evaluation")
        second_authority = self.module._archive_authority(
            self.config, [output], None
        )
        second_authority_identity = second_authority["attempt_identity_sha256"]
        self.assertNotEqual(first_authority_identity, second_authority_identity)
        second_manifest = self.module._bind_archive_authority(
            self.module._manifest(output), second_authority
        )
        second_receipt_identity = self.module._canonical_sha256(
            {"attempts": [second_manifest]}
        )

        with mock.patch.dict(
            os.environ,
            {"QWEN35_ARCHIVE_CRASH_AT": "source_1_renamed"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected archive crash"):
                self.module.archive_failed_attempt(
                    self.config,
                    [output],
                    expected_attempt_identity=second_authority_identity,
                )
        self.assertFalse(output.exists())
        second_tombstone = output.with_name(
            f".{output.name}.archived-{second_receipt_identity}-1"
        )
        self.assertTrue(second_tombstone.is_dir())
        self.assertGreater((second_tombstone / "root.txt").stat().st_size, 0)

        with self.assertRaisesRegex(
            RuntimeError, "multiple failed archives.*exact expected attempt identity"
        ):
            self.module.archive_failed_attempt(self.config, [output])
        self.assertGreater((second_tombstone / "root.txt").stat().st_size, 0)

        wrong = "0" * 64
        if wrong in {first_authority_identity, second_authority_identity}:
            wrong = "f" * 64
        with self.assertRaisesRegex(
            RuntimeError, "no valid archive for the expected attempt identity"
        ):
            self.module.archive_failed_attempt(
                self.config,
                [output],
                expected_attempt_identity=wrong,
            )
        self.assertGreater((second_tombstone / "root.txt").stat().st_size, 0)

        cleanup_identities: list[str] = []
        real_cleanup = self.module._cleanup_sources

        def record_cleanup(
            canonical_paths: list[Path],
            manifests: list[dict],
            *,
            attempt_set_identity: str,
            archive_revalidator=None,
        ) -> None:
            cleanup_identities.append(attempt_set_identity)
            real_cleanup(
                canonical_paths,
                manifests,
                attempt_set_identity=attempt_set_identity,
                archive_revalidator=archive_revalidator,
            )

        with mock.patch.object(
            self.module, "_cleanup_sources", side_effect=record_cleanup
        ):
            second = self.module.archive_failed_attempt(
                self.config,
                [output],
                expected_attempt_identity=second_authority_identity,
            )
        self.assertEqual(cleanup_identities, [second_receipt_identity])
        self.assertEqual(second["attempt_identity_sha256"], second_receipt_identity)
        self.assertNotEqual(second["archive_path"], first["archive_path"])
        self.assertEqual(
            second["attempts"][0]["archive_authority"][
                "attempt_identity_sha256"
            ],
            second_authority_identity,
        )
        self.assert_zeroized_tombstone(first_tombstone)
        self.assert_zeroized_tombstone(second_tombstone)

    def test_cleanup_tombstone_destination_race_is_no_clobber(self) -> None:
        source = self.large_cell("lora_joint_seed7412")
        self.populate(source, "preserved-source")
        manifest = self.module._manifest(source)
        attempt_identity = "a" * 64
        tombstone = source.with_name(
            f".{source.name}.archived-{attempt_identity}-1"
        )
        raced_inode: int | None = None
        real_rename = self.module.rename_new_entry

        def create_raced_destination(root: Path, old: Path, new: Path) -> None:
            nonlocal raced_inode
            new.mkdir()
            raced_inode = new.stat().st_ino
            real_rename(root, old, new)

        with mock.patch.object(
            self.module,
            "rename_new_entry",
            side_effect=create_raced_destination,
        ):
            with self.assertRaisesRegex(RuntimeError, "atomic no-clobber"):
                self.module._cleanup_sources(
                    [source], [manifest], attempt_set_identity=attempt_identity
                )
        self.assertTrue(source.is_dir())
        self.assertTrue(tombstone.is_dir())
        self.assertEqual(tombstone.stat().st_ino, raced_inode)
        self.assertEqual(list(tombstone.iterdir()), [])
        self.assertEqual((source / "root.txt").read_text(encoding="utf-8"), "root-preserved-source\n")

    def test_late_unarchived_payload_is_retained_and_blocks_cleanup(self) -> None:
        spec, _ = self.started_training(
            "lora_joint_seed7411", present=("external",)
        )
        real_rename = self.module.rename_new_entry
        injected = False

        def inject_before_cleanup_rename(root: Path, old: Path, new: Path) -> None:
            nonlocal injected
            if old == spec["external"]:
                (old / "late-unarchived-payload.txt").write_text(
                    "must survive\n", encoding="utf-8"
                )
                injected = True
            real_rename(root, old, new)

        with mock.patch.object(
            self.module,
            "rename_new_entry",
            side_effect=inject_before_cleanup_rename,
        ):
            with self.assertRaisesRegex(RuntimeError, "differs from its archived manifest"):
                self.module.archive_failed_attempt(
                    self.config, [spec["external"]]
                )

        self.assertTrue(injected)
        self.assertFalse(spec["external"].exists())
        tombstones = list(
            spec["external"].parent.glob(
                f".{spec['external'].name}.archived-*"
            )
        )
        self.assertEqual(len(tombstones), 1)
        self.assertEqual(
            (tombstones[0] / "late-unarchived-payload.txt").read_text(
                encoding="utf-8"
            ),
            "must survive\n",
        )
        archives = list(
            (
                self.experiment
                / self.config["paths"]["large_artifacts_dir"]
                / "failed_attempts"
            ).glob(f"{spec['external'].name}-*")
        )
        self.assertEqual(len(archives), 1)
        self.assertEqual(
            list(archives[0].rglob("late-unarchived-payload.txt")), []
        )

    def test_tombstone_path_replacement_fails_without_deleting_either_tree(self) -> None:
        spec, _ = self.started_training(
            "lora_state_only_seed7411", present=("external",)
        )
        real_zeroize = self.module._zeroize_quarantined_tree
        preserved: Path | None = None
        replacement: Path | None = None

        def replace_path(tombstone: Path, manifest: dict) -> None:
            nonlocal preserved, replacement
            if preserved is None:
                preserved = tombstone.with_name(
                    "preserved-original-after-cleanup-race"
                )
                tombstone.rename(preserved)
                tombstone.mkdir()
                (tombstone / "concurrent-replacement.txt").write_text(
                    "do not delete\n", encoding="utf-8"
                )
                replacement = tombstone
            real_zeroize(tombstone, manifest)

        with mock.patch.object(
            self.module,
            "_zeroize_quarantined_tree",
            side_effect=replace_path,
        ):
            with self.assertRaisesRegex(RuntimeError, "differs from its archived manifest"):
                self.module.archive_failed_attempt(
                    self.config, [spec["external"]]
                )

        assert preserved is not None and replacement is not None
        self.assertEqual(
            (preserved / "root.txt").read_text(encoding="utf-8"),
            "root-lora_state_only_seed7411:external\n",
        )
        self.assertEqual(
            (replacement / "concurrent-replacement.txt").read_text(
                encoding="utf-8"
            ),
            "do not delete\n",
        )

    def test_partial_descriptor_zeroization_is_resumable(self) -> None:
        spec, _ = self.started_training(
            "fullrank_joint_seed7411", present=("external",)
        )
        real_ftruncate = os.ftruncate
        calls = 0

        def fail_second(descriptor: int, length: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic partial zeroization failure")
            real_ftruncate(descriptor, length)

        with mock.patch.object(
            self.module.os,
            "ftruncate",
            side_effect=fail_second,
        ):
            with self.assertRaisesRegex(OSError, "partial zeroization"):
                self.module.archive_failed_attempt(
                    self.config, [spec["external"]]
                )

        receipt = self.module.archive_failed_attempt(
            self.config, [spec["external"]]
        )
        tombstones = list(
            spec["external"].parent.glob(
                f".{spec['external'].name}.archived-"
                f"{receipt['attempt_identity_sha256']}-*"
            )
        )
        self.assertEqual(len(tombstones), 1)
        self.assert_zeroized_tombstone(tombstones[0])

    def test_post_tombstone_deletion_retry_uses_exact_started_journal_head(self) -> None:
        spec, _ = self.started_training("lora_joint_seed7412", present=("external",))
        first = self.module.archive_failed_attempt(self.config, [spec["external"]])
        lineage = archive_lineage(
            self.repo, self._tracked_receipt_path(first), first
        )
        second_auth = prepare_training_attempt(
            self.repo,
            slug=spec["cell"]["slug"],
            header=spec["header"],
            cell=spec["cell"],
            canonical_paths=spec["canonical_paths"],
            context=spec["context"],
            replay_archive=lineage,
        )
        ensure_attempt_output(spec["external"], second_auth)
        start_training_attempt(
            self.repo,
            slug=spec["cell"]["slug"],
            header=spec["header"],
            cell=spec["cell"],
            canonical_paths=spec["canonical_paths"],
            authorization=second_auth,
        )
        self.populate(spec["external"], "second-post-delete-crash")
        with mock.patch.dict(
            os.environ,
            {"QWEN35_ARCHIVE_CRASH_AT": "source_1_deleted"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected archive crash"):
                self.module.archive_failed_attempt(
                    self.config, [spec["external"]]
                )
        self.assertFalse(spec["external"].exists())
        tombstones = list(
            spec["external"].parent.glob(
                f".{spec['external'].name}.archived-*"
            )
        )
        self.assertEqual(len(tombstones), 2)
        for tombstone in tombstones:
            self.assert_zeroized_tombstone(tombstone)
        second = self.module.archive_failed_attempt(
            self.config, [spec["external"]]
        )
        self.assertNotEqual(first["archive_path"], second["archive_path"])
        self.assertEqual(
            second["attempts"][0]["archive_authority"]["attempt_identity_sha256"],
            second_auth["attempt_identity_sha256"],
        )

    def test_successful_archive_is_idempotently_resumable_for_same_attempt(self) -> None:
        spec, _ = self.started_training("lora_joint_seed7413", present=("external",))
        first = self.module.archive_failed_attempt(self.config, [spec["external"]])
        archived_source = self.repo / first["archive_path"] / f"source_1_{spec['external'].name}"
        shutil.copytree(archived_source, spec["external"])
        real_zeroize = self.module._zeroize_quarantined_tree
        with mock.patch.object(
            self.module,
            "_zeroize_quarantined_tree",
            wraps=real_zeroize,
        ) as reconfirm:
            second = self.module.archive_failed_attempt(
                self.config, [spec["external"]]
            )
        self.assertEqual(second, first)
        self.assertEqual(reconfirm.call_count, 1)
        self.assertTrue(spec["external"].is_dir())
        self.assertEqual(
            (spec["external"] / "root.txt").read_text(encoding="utf-8"),
            "root-lora_joint_seed7413:external\n",
        )

    def test_every_archive_commit_crash_point_resumes_without_loss(self) -> None:
        cases = (
            ("lora_joint_seed7411", "source_1_copied", ("external", "tracked")),
            ("lora_state_only_seed7412", "source_2_copied", ("external", "tracked")),
            ("fullrank_joint_seed7411", "archive_receipt_written", ("external",)),
            ("fullrank_joint_seed7412", "archive_promoted", ("external",)),
            ("fullrank_state_only_seed7411", "tracked_receipt_written", ("external",)),
            ("fullrank_state_only_seed7412", "archive_verified", ("external",)),
            ("lora_state_only_seed7413", "source_1_renamed", ("external", "tracked")),
            ("fullrank_joint_seed7413", "source_2_renamed", ("external", "tracked")),
        )
        for cell_name, crash_point, present in cases:
            with self.subTest(crash_point=crash_point):
                spec, _ = self.started_training(cell_name, present=present)
                with mock.patch.dict(
                    os.environ,
                    {"QWEN35_ARCHIVE_CRASH_AT": crash_point},
                    clear=False,
                ):
                    with self.assertRaisesRegex(RuntimeError, "injected archive crash"):
                        self.module.archive_failed_attempt(self.config, [spec[present[0]]])
                receipt = self.module.archive_failed_attempt(
                    self.config, [spec[present[0]]]
                )
                self.module.validate_failed_archive(
                    self.repo,
                    self._tracked_receipt_path(receipt),
                    expected_header=self.module._archive_header(self.config),
                )
                self.assertFalse(spec["external"].exists())
                self.assertFalse(spec["tracked"].exists())
                leftovers = [
                    path
                    for parent in (spec["external"].parent, spec["tracked"].parent)
                    if parent.exists()
                    for path in parent.iterdir()
                    if ".archived-" in path.name or path.name.endswith(".staging")
                ]
                self.assertTrue(leftovers)
                self.assertFalse(
                    any(path.name.endswith(".staging") for path in leftovers)
                )
                for path in leftovers:
                    self.assert_zeroized_tombstone(path)

    def test_valid_completed_training_pair_cannot_be_archived(self) -> None:
        spec, _ = self.started_training()
        with mock.patch.object(
            self.module,
            "_terminal_training_pair_audit",
            return_value=(True, spec["cell_object"], ()),
        ):
            with self.assertRaisesRegex(RuntimeError, "valid completed training pair"):
                self.module.archive_failed_attempt(self.config, [spec["tracked"]])
        self.assertTrue(spec["external"].is_dir())
        self.assertTrue(spec["tracked"].is_dir())

    def test_published_terminal_started_window_must_be_finalized_not_archived(self) -> None:
        spec, _ = self.started_training("lora_joint_seed7412")

        def audit(_config, _external, _tracked, *, allow_started_terminal=False):
            if allow_started_terminal:
                return True, spec["cell_object"], ()
            return (
                False,
                spec["cell_object"],
                ("terminal receipts were published before journal completion",),
            )

        with mock.patch.object(
            self.module, "_terminal_training_pair_audit", side_effect=audit
        ):
            with self.assertRaisesRegex(RuntimeError, "finalize.*journal"):
                self.module.archive_failed_attempt(
                    self.config, [spec["external"]]
                )
        self.assertTrue(spec["external"].is_dir())
        self.assertTrue(spec["tracked"].is_dir())

    def test_terminal_audit_errors_are_preserved_as_nonscientific_metadata(self) -> None:
        spec, _ = self.started_training("fullrank_state_only_seed7413")
        errors = ("terminal graph mirror mismatch", "unexpected final member")
        with mock.patch.object(
            self.module,
            "_terminal_training_pair_audit",
            return_value=(False, spec["cell_object"], errors),
        ):
            receipt = self.module.archive_failed_attempt(
                self.config, [spec["external"]]
            )
        authority = receipt["attempts"][0]["archive_authority"]
        self.assertEqual(authority["terminal_graph_errors"], list(errors))
        self.assertFalse(authority["scientific_evidence"])
        self.assertFalse(authority["authorizes_training"])

    def test_unsafe_tree_members_are_rejected_before_any_source_move(self) -> None:
        spec, _ = self.started_training()
        target = spec["external"] / "real.txt"
        target.write_text("bytes", encoding="utf-8")
        (spec["external"] / "link.txt").symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, "symlink|special node"):
            self.module.archive_failed_attempt(self.config, [spec["external"]])
        self.assertTrue(spec["external"].is_dir())

        trigger = self.experiment / "runs" / "lora_joint_seed7412_trigger"
        trigger.mkdir(parents=True)
        os.mkfifo(trigger / "progress.pipe")
        with self.assertRaisesRegex(RuntimeError, "special node"):
            self.module.archive_failed_attempt(self.config, [trigger])
        self.assertTrue(trigger.is_dir())

        socket_output = self.experiment / "runs" / "lora_joint_seed7413_trigger"
        socket_output.mkdir(parents=True)
        endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        previous = Path.cwd()
        try:
            os.chdir(socket_output)
            endpoint.bind("progress.sock")
            os.chdir(previous)
            with self.assertRaisesRegex(RuntimeError, "special node"):
                self.module.archive_failed_attempt(self.config, [socket_output])
        finally:
            os.chdir(previous)
            endpoint.close()

    def test_tree_snapshot_detects_root_rebind_after_leaf_hash(self) -> None:
        source = self.repo / "stable-tree"
        source.mkdir()
        (source / "payload.bin").write_bytes(b"stable bytes")
        displaced = self.repo / "displaced-tree"
        original_digest = attempt_receipts._digest_descriptor
        replaced = False

        def digest_and_replace(descriptor: int) -> str:
            nonlocal replaced
            digest = original_digest(descriptor)
            if not replaced:
                replaced = True
                source.rename(displaced)
                shutil.copytree(displaced, source)
            return digest

        with mock.patch.object(
            attempt_receipts,
            "_digest_descriptor",
            side_effect=digest_and_replace,
        ):
            with self.assertRaisesRegex(
                AttemptReceiptError, "binding changed|membership changed"
            ):
                tree_manifest(source, source_path="synthetic/stable-tree")

    def test_archive_validation_rejects_cross_source_hardlink_alias(self) -> None:
        spec, _ = self.started_training()
        for side in ("external", "tracked"):
            (spec[side] / "identical.bin").write_bytes(b"identical")
        receipt = self.module.archive_failed_attempt(
            self.config, [spec["external"]]
        )
        archive = self.repo / receipt["archive_path"]
        first = archive / f"source_1_{spec['external'].name}" / "identical.bin"
        second = archive / f"source_2_{spec['tracked'].name}" / "identical.bin"
        second.unlink()
        os.link(first, second)
        with self.assertRaisesRegex(
            AttemptReceiptError, "cross-tree hardlink alias"
        ):
            self.module.validate_failed_archive(
                self.repo,
                self._tracked_receipt_path(receipt),
                expected_header=self.module._archive_header(self.config),
            )

    def test_receipt_replacement_during_graph_validation_fails_closed(self) -> None:
        spec, _ = self.started_training(
            "fullrank_joint_seed7411", present=("external",)
        )
        receipt = self.module.archive_failed_attempt(
            self.config, [spec["external"]]
        )
        tracked = self._tracked_receipt_path(receipt)
        archive_receipt = (
            self.repo / receipt["archive_path"] / "archive_receipt.json"
        )
        replacement = json.dumps(
            receipt, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        original_compare = attempt_receipts._compare_tree_manifest
        replaced = False

        def compare_and_replace(observed, expected) -> None:
            nonlocal replaced
            original_compare(observed, expected)
            if replaced:
                return
            replaced = True
            for path in (tracked, archive_receipt):
                temporary = path.with_name(f".{path.name}.replacement")
                temporary.write_bytes(replacement)
                os.replace(temporary, path)

        with mock.patch.object(
            attempt_receipts,
            "_compare_tree_manifest",
            side_effect=compare_and_replace,
        ):
            with self.assertRaisesRegex(
                AttemptReceiptError, "changed|membership"
            ):
                self.module.validate_failed_archive(
                    self.repo,
                    tracked,
                    expected_header=self.module._archive_header(self.config),
                )

    def test_archive_lineage_digest_comes_from_validated_mirror_snapshot(self) -> None:
        spec, _ = self.started_training(
            "fullrank_joint_seed7412", present=("external",)
        )
        receipt = self.module.archive_failed_attempt(
            self.config, [spec["external"]]
        )
        tracked = self._tracked_receipt_path(receipt)
        archive_receipt = (
            self.repo / receipt["archive_path"] / "archive_receipt.json"
        )
        replacement = json.dumps(
            receipt, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        for path in (tracked, archive_receipt):
            temporary = path.with_name(f".{path.name}.replacement")
            temporary.write_bytes(replacement)
            os.replace(temporary, path)
        lineage = archive_lineage(self.repo, tracked, receipt)
        self.assertEqual(
            lineage["sha256"],
            hashlib.sha256(replacement).hexdigest(),
        )

    def test_lock_leaf_replacement_cannot_create_two_cooperating_writers(self) -> None:
        lock = self.repo / "lock-state" / "journal.lock"
        acquired = threading.Event()
        finished = threading.Event()
        errors: list[BaseException] = []

        def second_writer() -> None:
            try:
                with locked_regular(lock):
                    acquired.set()
            except BaseException as exc:  # surfaced in the test thread below
                errors.append(exc)
            finally:
                finished.set()

        thread: threading.Thread | None = None
        with self.assertRaisesRegex(AttemptReceiptError, "pathname changed"):
            with locked_regular(lock):
                lock.unlink()
                lock.write_text("replacement lock inode\n", encoding="utf-8")
                thread = threading.Thread(target=second_writer, daemon=True)
                thread.start()
                self.assertFalse(
                    acquired.wait(0.2),
                    "the replacement lock leaf bypassed the held parent lock",
                )
        assert thread is not None
        self.assertTrue(finished.wait(2.0))
        thread.join(timeout=2.0)
        self.assertTrue(acquired.is_set())
        self.assertEqual(errors, [])

    def test_nested_checkpoint_symlink_is_rejected_before_dereference(self) -> None:
        spec, _ = self.started_training("lora_joint_seed7412", present=("external",))
        target = self.repo / "untrusted-checkpoint-target"
        target.mkdir()
        (target / "checkpoint.json").write_text(
            '{"receipt_identity_sha256":"not-opened"}\n', encoding="utf-8"
        )
        checkpoint = spec["external"] / "checkpoint_000001"
        checkpoint.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "symlink|special node"):
            self.module.archive_failed_attempt(self.config, [spec["external"]])
        self.assertTrue(checkpoint.is_symlink())

    def test_symlinked_canonical_ancestors_are_rejected_lexically(self) -> None:
        canonical_root = self.large_cell().parent
        canonical_root.parent.mkdir(parents=True)
        actual_root = self.repo / "actual-external-training-root"
        actual_root.mkdir()
        canonical_root.symlink_to(actual_root, target_is_directory=True)
        primary = self.large_cell()
        self.populate(primary, "behind-root-symlink")
        with self.assertRaisesRegex(RuntimeError, "symlinked path component"):
            self.module.archive_failed_attempt(self.config, [primary])
        self.assertTrue((actual_root / primary.name / "root.txt").is_file())

    def test_completed_evaluation_cannot_be_archived_as_failed(self) -> None:
        output = self.experiment / "runs" / "lora_joint_seed7411_contrast"
        self.populate(output, "complete")
        (output / "summary.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "completed evaluation"):
            self.module.archive_failed_attempt(self.config, [output])
        self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
