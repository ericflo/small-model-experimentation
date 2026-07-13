from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import analysis, safe_io  # noqa: E402
from src.config import load_config  # noqa: E402


SEEDS = (7411, 7412, 7413)
REQUIRED = {
    "validation": tuple(range(1, 5)),
    "depth_extrapolation": tuple(range(5, 13)),
    "joint_holdout": tuple(range(5, 13)),
}


def synthetic_bundles(
    *,
    trained_pass: bool = True,
    depth_pass: bool = True,
    joint_pass: bool = True,
) -> dict[int, dict[str, list[dict]]]:
    bundles: dict[int, dict[str, list[dict]]] = {}
    for seed in SEEDS:
        intact: list[dict] = []
        disabled: list[dict] = []
        for split, depths in REQUIRED.items():
            for depth in depths:
                if split == "validation" and depth == 1:
                    cell_pass = False  # diagnostic only; must not veto an arm
                elif split == "validation":
                    cell_pass = trained_pass
                elif split == "depth_extrapolation":
                    cell_pass = depth_pass
                else:
                    cell_pass = joint_pass
                correct = 2 if cell_pass else 1  # 2/5 passes the frozen >= .40 gate
                for item in range(5):
                    row_id = f"{split}-depth{depth}-task{item}"
                    intact.append(
                        {
                            "id": row_id,
                            "split": split,
                            "depth": depth,
                            "joint_final_correct": item < correct,
                        }
                    )
                    disabled.append(
                        {
                            "id": row_id,
                            "split": split,
                            "depth": depth,
                            "joint_final_correct": False,
                        }
                    )
        bundles[seed] = {"intact": intact, "disabled": disabled}
    return bundles


def synthetic_contrast_bundles(
    *,
    trained_pass: bool = True,
    depth_pass: bool = True,
    joint_pass: bool = True,
) -> dict[int, dict[str, list[dict]]]:
    bundles = synthetic_bundles(
        trained_pass=trained_pass,
        depth_pass=depth_pass,
        joint_pass=joint_pass,
    )
    split_map = {
        "validation": "contrast_validation",
        "depth_extrapolation": "contrast_depth",
        "joint_holdout": "contrast_joint",
    }
    for bundle in bundles.values():
        for mode in ("intact", "disabled"):
            transformed = []
            for row in bundle[mode]:
                if row["split"] not in split_map:
                    continue
                if row["split"] == "validation" and row["depth"] == 1:
                    continue
                copied = dict(row)
                copied["split"] = split_map[row["split"]]
                copied["id"] = copied["id"].replace(
                    row["split"], copied["split"], 1
                )
                transformed.append(copied)
            bundle[mode] = transformed
    return bundles


def with_missing_required_cell(
    bundles: dict[int, dict[str, list[dict]]],
    *,
    split: str,
    depth: int,
    seed: int = 7411,
) -> dict[int, dict[str, list[dict]]]:
    result = copy.deepcopy(bundles)
    for mode in ("intact", "disabled"):
        result[seed][mode] = [
            row
            for row in result[seed][mode]
            if not (row["split"] == split and row["depth"] == depth)
        ]
    return result


def effect(passes: bool = True) -> dict:
    return {
        "status": (
            "ADAPTATION_REQUIRED" if passes else "ADAPTATION_CONTRAST_UNCERTAIN"
        ),
        "passes": passes,
        "every_required_seed_depth_point_positive": passes,
        "every_seed_point_positive": passes,
        "no_depth_point_negative": passes,
        "every_split_crossed_lcb_positive": passes,
        "splits": {},
    }


class CanonicalAnalysisPathTests(unittest.TestCase):
    def test_lineage_paths_reject_symlink_and_noncanonical_aliases(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            payload = real / "receipt.json"
            payload.write_text("{}\n", encoding="utf-8")
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            canonical = payload.relative_to(analysis.REPO_ROOT).as_posix()
            self.assertEqual(analysis._resolve_repo_path(canonical), payload)
            with self.assertRaisesRegex(RuntimeError, "not canonical"):
                analysis._resolve_repo_path(
                    (alias / "receipt.json").relative_to(analysis.REPO_ROOT).as_posix()
                )
            for value in (
                f"{real.relative_to(analysis.REPO_ROOT).as_posix()}/./receipt.json",
                f"{real.relative_to(analysis.REPO_ROOT).as_posix()}/../real/receipt.json",
                f"{real.relative_to(analysis.REPO_ROOT).as_posix()}//receipt.json",
            ):
                with self.subTest(value=value), self.assertRaisesRegex(
                    RuntimeError, "not canonical"
                ):
                    analysis._resolve_repo_path(value)


class AnalysisPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "default.yaml")

    def _analyze(
        self,
        output: Path,
        *,
        identity_extra: dict | None = None,
    ) -> dict:
        identity = {
            "experiment_id": self.config["experiment_id"],
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "backend": "transformers",
        }
        identity.update(identity_extra or {})

        def phase_identity(_config, phase):
            return {**identity, "phase": phase}

        with mock.patch.object(
            analysis, "validate_design_receipt"
        ), mock.patch.object(
            analysis,
            "_load_cell",
            return_value=(synthetic_bundles(), [{"synthetic": True}]),
        ), mock.patch.object(
            analysis, "_adaptation_effects", return_value=effect(True)
        ), mock.patch.object(
            analysis, "_identity", side_effect=phase_identity
        ):
            return analysis.analyze_phase(
                self.config,
                ROOT / "runs",
                "lora_joint",
                output,
            )

    @staticmethod
    def _staging_entries(directory: Path) -> list[Path]:
        return list(directory.glob(".publish-*.tmp"))

    def test_success_is_exact_single_link_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            parent = Path(directory)
            output = parent / "analysis.json"
            summary = self._analyze(output)
            original = output.read_bytes()
            original_inode = output.stat().st_ino

            self.assertTrue(original.endswith(b"\n"))
            self.assertEqual(json.loads(original.decode("utf-8")), summary)
            self.assertEqual(output.stat().st_nlink, 1)
            self.assertEqual(self._staging_entries(parent), [])

            with self.assertRaisesRegex(
                safe_io.StableArtifactError, "refusing to overwrite"
            ):
                self._analyze(output)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(output.stat().st_ino, original_inode)
            self.assertEqual(output.stat().st_nlink, 1)
            self.assertEqual(self._staging_entries(parent), [])

    def test_racing_collision_preserves_existing_leaf_and_cleans_stage(self) -> None:
        sentinel = b"concurrent-winner\n"
        original_rename = safe_io._rename_noreplace_at

        def collide_then_rename(
            source_directory_fd,
            source_name,
            destination_directory_fd,
            destination_name,
        ):
            descriptor = os.open(
                destination_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=destination_directory_fd,
            )
            try:
                os.write(descriptor, sentinel)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.fsync(destination_directory_fd)
            return original_rename(
                source_directory_fd,
                source_name,
                destination_directory_fd,
                destination_name,
            )

        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            parent = Path(directory)
            output = parent / "analysis.json"
            with mock.patch.object(
                safe_io, "_rename_noreplace_at", side_effect=collide_then_rename
            ):
                with self.assertRaisesRegex(
                    safe_io.StableArtifactError, "refusing to overwrite"
                ):
                    self._analyze(output)
            self.assertEqual(output.read_bytes(), sentinel)
            self.assertEqual(output.stat().st_nlink, 1)
            self.assertEqual(self._staging_entries(parent), [])

    def test_symlink_leaf_ancestor_escape_and_noncanonical_alias_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            parent = Path(directory)
            real = parent / "real"
            real.mkdir()
            ancestor_alias = parent / "ancestor-alias"
            ancestor_alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(safe_io.StableArtifactError):
                self._analyze(ancestor_alias / "analysis.json")
            self.assertFalse((real / "analysis.json").exists())

            outside = parent / "outside.json"
            outside.write_bytes(b"outside\n")
            leaf_alias = parent / "leaf-alias.json"
            leaf_alias.symlink_to(outside)
            with self.assertRaisesRegex(
                safe_io.StableArtifactError, "refusing to overwrite"
            ):
                self._analyze(leaf_alias)
            self.assertTrue(leaf_alias.is_symlink())
            self.assertEqual(outside.read_bytes(), b"outside\n")

            subdirectory = parent / "subdirectory"
            subdirectory.mkdir()
            noncanonical = subdirectory / ".." / "aliased.json"
            with self.assertRaisesRegex(
                safe_io.StableArtifactError, "not canonical"
            ):
                self._analyze(noncanonical)
            self.assertFalse((parent / "aliased.json").exists())
            self.assertEqual(self._staging_entries(parent), [])

        with tempfile.TemporaryDirectory() as outside_directory:
            outside = Path(outside_directory) / "analysis.json"
            with self.assertRaisesRegex(
                safe_io.StableArtifactError, "escapes its trusted root"
            ):
                self._analyze(outside)
            self.assertFalse(outside.exists())

    def test_injected_writer_and_precommit_failures_leave_no_partial_output(self) -> None:
        original_publish_file = safe_io.publish_new_file

        def fail_during_write(root, path, _writer, *, mode=0o600):
            def partial_writer(handle):
                handle.write(b'{"partial":')
                raise RuntimeError("injected analysis write failure")

            return original_publish_file(root, path, partial_writer, mode=mode)

        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            parent = Path(directory)
            write_output = parent / "write-failure.json"
            with mock.patch.object(
                safe_io, "publish_new_file", side_effect=fail_during_write
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "injected analysis write failure"
                ):
                    self._analyze(write_output)
            self.assertFalse(os.path.lexists(write_output))
            self.assertEqual(self._staging_entries(parent), [])

            precommit_output = parent / "precommit-failure.json"
            with mock.patch.object(
                safe_io.os,
                "fsync",
                side_effect=OSError("injected analysis pre-commit failure"),
            ):
                with self.assertRaises(safe_io.StableArtifactError):
                    self._analyze(precommit_output)
            self.assertFalse(os.path.lexists(precommit_output))
            self.assertEqual(self._staging_entries(parent), [])

    def test_nonfinite_summary_is_rejected_before_any_publication(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            parent = Path(directory)
            output = parent / "analysis.json"
            with self.assertRaisesRegex(ValueError, "JSON compliant"):
                self._analyze(output, identity_extra={"nonfinite": float("nan")})
            self.assertFalse(os.path.lexists(output))
            self.assertEqual(self._staging_entries(parent), [])


class FormationSummaryTests(unittest.TestCase):
    def test_all_three_seeds_and_every_required_depth_are_cellwise_gated(self) -> None:
        summary = analysis._formation_summary(synthetic_bundles(), 0.40)
        self.assertTrue(summary["passes"])
        self.assertEqual(summary["status"], "STATE_FORMATION_PASS")
        for seed in map(str, SEEDS):
            self.assertFalse(summary["per_seed"][seed]["validation"]["1"]["required"])
            for depth in (2, 3, 4):
                self.assertTrue(summary["per_seed"][seed]["validation"][str(depth)]["passes"])
            for split in ("depth_extrapolation", "joint_holdout"):
                for depth in range(5, 13):
                    self.assertTrue(summary["per_seed"][seed][split][str(depth)]["passes"])

    def test_simultaneous_failures_use_frozen_earliest_failure_priority(self) -> None:
        cases = (
            ((False, False, False), "TRAINED_DEPTH_MISS"),
            ((True, False, False), "TRAINED_PASS_DEPTH_EXTRAPOLATION_MISS"),
            ((True, True, False), "TRAINED_AND_DEPTH_PASS_JOINT_SHIFT_MISS"),
            ((True, True, True), "STATE_FORMATION_PASS"),
        )
        for (trained, depth, joint), expected in cases:
            with self.subTest(expected=expected):
                summary = analysis._formation_summary(
                    synthetic_bundles(
                        trained_pass=trained,
                        depth_pass=depth,
                        joint_pass=joint,
                    ),
                    0.40,
                )
                self.assertEqual(summary["status"], expected)

    def test_missing_seed_or_required_depth_is_evidence_incomplete_not_a_pass(self) -> None:
        missing_seed = synthetic_bundles()
        del missing_seed[7413]
        summary = analysis._formation_summary(missing_seed, 0.40)
        self.assertFalse(summary["passes"])
        self.assertEqual(summary["status"], "EVIDENCE_INCOMPLETE")

        missing_depth = synthetic_bundles()
        for mode in ("intact", "disabled"):
            missing_depth[7412][mode] = [
                row
                for row in missing_depth[7412][mode]
                if not (row["split"] == "joint_holdout" and row["depth"] == 12)
            ]
        summary = analysis._formation_summary(missing_depth, 0.40)
        self.assertFalse(summary["passes"])
        self.assertEqual(summary["status"], "EVIDENCE_INCOMPLETE")

    def test_sealed_formation_requires_fresh_validation_depths_two_to_four(self) -> None:
        contrast = synthetic_contrast_bundles()
        summary = analysis._formation_summary(contrast, 0.40)
        self.assertTrue(summary["passes"])
        self.assertEqual(summary["matrix"], "contrast")
        for seed in map(str, SEEDS):
            self.assertEqual(
                set(summary["per_seed"][seed]["contrast_validation"]),
                {"2", "3", "4"},
            )
            self.assertTrue(
                all(
                    cell["required"] and cell["passes"]
                    for cell in summary["per_seed"][seed][
                        "contrast_validation"
                    ].values()
                )
            )

        incomplete = copy.deepcopy(contrast)
        for mode in ("intact", "disabled"):
            incomplete[7411][mode] = [
                row
                for row in incomplete[7411][mode]
                if row["split"] != "contrast_validation"
            ]
        summary = analysis._formation_summary(incomplete, 0.40)
        self.assertFalse(summary["passes"])
        self.assertEqual(summary["status"], "EVIDENCE_INCOMPLETE")


class FailureCategoryReplicationTests(unittest.TestCase):
    @staticmethod
    def formation(*, trained: bool, depth: bool, joint: bool) -> dict:
        return {
            "category_passes": {
                "trained": trained,
                "depth": depth,
                "joint": joint,
            }
        }

    def test_every_trigger_failure_must_repeat_but_additional_sealed_failures_are_allowed(self) -> None:
        trigger_joint_miss = self.formation(
            trained=True, depth=True, joint=False
        )
        sealed_trained_only_miss = self.formation(
            trained=False, depth=True, joint=True
        )
        missing = analysis._failure_category_replication(
            trigger_joint_miss, sealed_trained_only_miss
        )
        self.assertFalse(missing["passes"])
        self.assertEqual(
            missing["status"], "TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED"
        )
        self.assertEqual(missing["trigger_failed_categories"], ["joint"])
        self.assertEqual(missing["sealed_failed_categories"], ["trained"])
        self.assertEqual(missing["missing_replications"], ["joint"])

        sealed_joint_plus_trained_miss = self.formation(
            trained=False, depth=True, joint=False
        )
        replicated = analysis._failure_category_replication(
            trigger_joint_miss, sealed_joint_plus_trained_miss
        )
        self.assertTrue(replicated["passes"])
        self.assertEqual(
            replicated["status"], "TRIGGER_FAILURE_CATEGORIES_REPLICATED"
        )
        self.assertEqual(replicated["missing_replications"], [])


class AdaptationDependenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "default.yaml")

    def test_required_not_required_and_uncertain_are_distinct(self) -> None:
        required = synthetic_bundles()
        self.assertEqual(
            analysis._adaptation_effects(self.config, required)["status"],
            "ADAPTATION_REQUIRED",
        )

        not_required = synthetic_bundles()
        for bundle in not_required.values():
            bundle["disabled"] = copy.deepcopy(bundle["intact"])
        self.assertEqual(
            analysis._adaptation_effects(self.config, not_required)["status"],
            "ADAPTATION_NOT_REQUIRED_AT_INFERENCE",
        )

        zero_cell = synthetic_bundles()
        for bundle in zero_cell.values():
            intact_cell = {
                row["id"]: row
                for row in bundle["intact"]
                if row["split"] == "depth_extrapolation" and row["depth"] == 5
            }
            for row in bundle["disabled"]:
                if row["id"] in intact_cell:
                    row["joint_final_correct"] = intact_cell[row["id"]][
                        "joint_final_correct"
                    ]
        self.assertEqual(
            analysis._adaptation_effects(self.config, zero_cell)["status"],
            "ADAPTATION_REQUIRED",
        )

        uncertain = synthetic_bundles()
        for seed in (7411, 7412):
            for row in uncertain[seed]["disabled"]:
                if row["split"] == "depth_extrapolation" and row["depth"] == 5:
                    row["joint_final_correct"] = True
        self.assertEqual(
            analysis._adaptation_effects(self.config, uncertain)["status"],
            "ADAPTATION_CONTRAST_UNCERTAIN",
        )

    def test_formation_pass_fail_quadrants_have_distinct_adaptation_statuses(self) -> None:
        intact_pass_disabled_miss = synthetic_bundles()

        both_pass = synthetic_bundles()
        for bundle in both_pass.values():
            bundle["disabled"] = copy.deepcopy(bundle["intact"])

        intact_miss_disabled_pass = synthetic_bundles(
            trained_pass=False, depth_pass=False, joint_pass=False
        )
        disabled_pass = synthetic_bundles()
        for seed in SEEDS:
            intact_miss_disabled_pass[seed]["disabled"] = copy.deepcopy(
                disabled_pass[seed]["intact"]
            )

        both_miss = synthetic_bundles(
            trained_pass=False, depth_pass=False, joint_pass=False
        )
        cases = (
            (
                intact_pass_disabled_miss,
                "ADAPTATION_REQUIRED",
            ),
            (both_pass, "ADAPTATION_NOT_REQUIRED_AT_INFERENCE"),
            (intact_miss_disabled_pass, "ADAPTATION_DISABLED_REVERSAL"),
            (both_miss, "ADAPTATION_CONTRAST_UNCERTAIN"),
        )
        for bundles, expected in cases:
            with self.subTest(expected=expected):
                result = analysis._adaptation_effects(self.config, bundles)
                self.assertEqual(result["status"], expected)
                self.assertEqual(
                    result["passes"], expected == "ADAPTATION_REQUIRED"
                )

    def test_cross_capacity_rescue_rejects_a_negative_pooled_depth(self) -> None:
        fullrank = synthetic_bundles()
        lora = synthetic_bundles()
        for bundle in lora.values():
            for row in bundle["intact"]:
                row["joint_final_correct"] = False
        robust = analysis._fullrank_minus_lora_contrast(self.config, fullrank, lora)
        self.assertTrue(robust["passes"])

        reversed_cell = copy.deepcopy(fullrank)
        for seed in (7411, 7412):
            for row in reversed_cell[seed]["intact"]:
                if row["split"] == "depth_extrapolation" and row["depth"] == 5:
                    row["joint_final_correct"] = False
            for row in lora[seed]["intact"]:
                if row["split"] == "depth_extrapolation" and row["depth"] == 5:
                    row["joint_final_correct"] = True
        result = analysis._fullrank_minus_lora_contrast(
            self.config, reversed_cell, lora
        )
        self.assertFalse(result["passes"])

    def test_every_crossed_receipt_uses_the_single_registered_bootstrap_seed(self) -> None:
        observed: list[int] = []

        def bootstrap(records, *, resamples, seed):
            observed.append(seed)
            return {
                "point": 1.0,
                "ci95": [0.5, 1.0],
                "model_seeds": sorted(records),
                "tasks": len(next(iter(records.values()))),
                "bootstrap_unit": "crossed_model_seed_by_task",
                "bootstrap_seed": seed,
                "resamples": resamples,
            }

        with mock.patch.object(analysis, "_crossed_bootstrap", side_effect=bootstrap):
            adaptation = analysis._adaptation_effects(
                self.config, synthetic_bundles()
            )
            fullrank = analysis._fullrank_minus_lora_contrast(
                self.config,
                synthetic_bundles(),
                synthetic_bundles(
                    trained_pass=False, depth_pass=False, joint_pass=False
                ),
            )
        self.assertTrue(observed)
        self.assertEqual(set(observed), {75301})
        crossed = [*adaptation["splits"].values(), *fullrank["splits"].values()]
        self.assertTrue(crossed)
        self.assertEqual(
            {receipt["bootstrap_seed"] for receipt in crossed}, {75301}
        )


class BranchEvidenceClosureIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "default.yaml")
        cls.identity = {
            "experiment_id": cls.config["experiment_id"],
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "backend": "transformers",
            "config_sha256": "1" * 64,
            "source_contract_sha256": "2" * 64,
            "requirements_training_lock_sha256": "3" * 64,
            "design_receipt_sha256": "4" * 64,
            "design_receipt_identity_sha256": "5" * 64,
        }

    def test_root_producer_receipt_is_consumed_and_rehashed_evidence_edits_fail(self) -> None:
        bundles = synthetic_bundles(
            trained_pass=False, depth_pass=False, joint_pass=False
        )
        manifests = [{"seed": seed, "kind": "trigger"} for seed in SEEDS]
        adaptation = {
            "status": "ADAPTATION_CONTRAST_UNCERTAIN",
            "passes": False,
            "splits": {
                "validation": {
                    "bootstrap_seed": 75301,
                    "bootstrap_unit": "crossed_model_seed_by_task",
                }
            },
        }
        with tempfile.TemporaryDirectory(dir=analysis.REPO_ROOT) as directory:
            repo = Path(directory).resolve()
            experiment = repo / "experiments" / self.config["experiment_id"]
            runs_dir = experiment / "runs"
            output = experiment / "analysis" / "lora_joint_trigger.json"
            output.parent.mkdir(parents=True)

            def identity(_config, phase):
                return {**self.identity, "phase": phase}

            patches = (
                mock.patch.object(analysis, "_identity", side_effect=identity),
                mock.patch.object(analysis, "validate_design_receipt"),
                mock.patch.object(
                    analysis,
                    "_load_cell",
                    return_value=(bundles, manifests),
                ),
                mock.patch.object(
                    analysis,
                    "_adaptation_effects",
                    return_value=adaptation,
                ),
                mock.patch.object(
                    analysis,
                    "_branch_evidence_inputs",
                    return_value=(self.config, runs_dir),
                ),
            )
            for patcher in patches:
                patcher.start()
            try:
                produced = analysis.analyze_phase(
                    self.config, runs_dir, "lora_joint", output
                )
                canonical = output.relative_to(repo).as_posix()

                def consume() -> dict:
                    return analysis.validate_branch_authorization(
                        repo,
                        output,
                        canonical_relative_path=canonical,
                        branch=analysis.LORA_MISS_BRANCH,
                        expected_identity=self.identity,
                    )

                self.assertEqual(
                    consume()["receipt"]["receipt_identity_sha256"],
                    produced["receipt_identity_sha256"],
                )
                alias = output.with_name("lora_joint_trigger_alias.json")
                os.link(output, alias)
                try:
                    with self.assertRaisesRegex(RuntimeError, "hardlink"):
                        consume()
                finally:
                    alias.unlink()
                with mock.patch.object(
                    analysis,
                    "_load_cell",
                    side_effect=RuntimeError("evaluation summary is missing"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "summary is missing"):
                        consume()
                mutations = {
                    "fabricated_minimal_evidence": lambda item: (
                        item.__setitem__(
                            "formation",
                            {"status": "TRAINED_DEPTH_MISS", "passes": False},
                        ),
                        item.pop("adaptation_effect"),
                        item.__setitem__("input_manifest", []),
                    ),
                    "deleted_manifest": lambda item: item.pop("input_manifest"),
                    "substituted_manifest": lambda item: item.__setitem__(
                        "input_manifest", [{"seed": 9999, "kind": "trigger"}]
                    ),
                    "substituted_formation": lambda item: item["formation"].__setitem__(
                        "status", "TRAINED_PASS_DEPTH_EXTRAPOLATION_MISS"
                    ),
                }
                for name, mutate in mutations.items():
                    forged = copy.deepcopy(produced)
                    forged.pop("receipt_identity_sha256")
                    mutate(forged)
                    forged["receipt_identity_sha256"] = analysis._canonical_sha256(
                        forged
                    )
                    output.write_text(
                        json.dumps(forged, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    with self.subTest(name=name), self.assertRaises(RuntimeError):
                        consume()
            finally:
                for patcher in reversed(patches):
                    patcher.stop()

    def test_firewall_validates_the_exact_preopened_ledger_snapshot(self) -> None:
        config = copy.deepcopy(self.config)
        config["paths"]["data_dir"] = "data"
        root_authorization = {
            "path": "analysis/lora_joint_trigger.json",
            "sha256": "1" * 64,
            "receipt_identity_sha256": "2" * 64,
            "status": "LORA_JOINT_MISS_CONTROLS_REQUIRED",
            "phase": "lora_joint_analysis",
        }
        ledger = {
            "schema_version": 1,
            "status": "CONTRAST_ACCESS_LEDGER",
            "events": [],
            "receipt_identity_sha256": "3" * 64,
        }
        empty_sha256 = analysis._canonical_empty_ledger_sha256(ledger)
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            experiment = repo / "experiments" / config["experiment_id"]
            data_dir = experiment / "data"
            data_dir.mkdir(parents=True)
            (data_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            firewall = {
                "status": "CONTRAST_FIREWALL_UNOPENED",
                "data_manifest_sha256": "d" * 64,
                "ledger_path": (
                    data_dir / "contrast_access_ledger.json"
                ).relative_to(repo).as_posix(),
                "ledger_sha256": empty_sha256,
                "events": 0,
                "authorization": root_authorization,
            }
            with mock.patch.object(analysis, "ROOT", experiment), mock.patch.object(
                analysis, "REPO_ROOT", repo
            ), mock.patch.object(
                analysis, "read_verified_json_object", return_value={"files": {}}
            ), mock.patch.object(
                analysis, "validate_data_manifest"
            ), mock.patch.object(
                analysis,
                "_read_stable_json_object",
                return_value=(ledger, empty_sha256),
            ), mock.patch.object(
                analysis, "load_contrast_access_ledger", return_value=ledger
            ) as loader:
                analysis._validate_stage_b_firewall_evidence(
                    config,
                    experiment / "runs",
                    firewall,
                    root_authorization=root_authorization,
                    stage_b_lineage={"unused": True},
                    branch=analysis.STAGE_B_FULLRANK_MISS_BRANCH,
                )
        self.assertIs(loader.call_args.kwargs["payload"], ledger)
        self.assertEqual(loader.call_args.kwargs["manifest_sha256"], "d" * 64)


class ExactEvaluationMatrixTests(unittest.TestCase):
    """The analyzer must reject plausible-looking partial or mixed result grids."""

    ROW_KEYS = {
        "id", "split", "depth", "family", "template", "query_kind",
        "capacity", "objective", "model_seed", "adaptation_mode",
        "node_target", "phase_target", "checksum_target",
        "node_prediction", "phase_prediction", "checksum_prediction",
        "node_final_correct", "phase_final_correct", "checksum_final_correct",
        "joint_final_correct", "node_trajectory_targets",
        "phase_trajectory_targets", "checksum_trajectory_targets",
        "node_trajectory_predictions", "phase_trajectory_predictions",
        "checksum_trajectory_predictions", "node_trajectory_accuracy",
        "phase_trajectory_accuracy", "checksum_trajectory_accuracy",
        "joint_trajectory_accuracy", "answer_choice_target",
        "answer_choice_prediction", "answer_correct", "full_top_is_answer",
        "answer_token_mass", "state_change_rms_by_transition",
        "mean_state_change_rms", "answer_loss", "state_loss", "fixed_point_loss",
        "prompt_tokens", "base_layer_token_applications",
        "extra_loop_layer_token_applications", "total_layer_token_applications",
        "adaptation_forward_macs", "adaptation_calls",
        "adaptation_call_manifest_sha256", "adaptation_cycles", "compute_proxy",
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "default.yaml")

    @staticmethod
    def rows(
        *,
        eval_set: str = "trigger",
        capacity: str = "lora",
        objective: str = "joint",
        seed: int = 7411,
    ) -> dict[str, list[dict]]:
        if eval_set == "trigger":
            matrix = {
                "validation": (range(1, 5), 256),
                "depth_extrapolation": (range(5, 13), 128),
                "joint_holdout": (range(5, 13), 128),
            }
        else:
            matrix = {
                "contrast_validation": (range(2, 5), 256),
                "contrast_depth": (range(5, 13), 128),
                "contrast_joint": (range(5, 13), 128),
            }
        result: dict[str, list[dict]] = {"intact": [], "disabled": []}
        for mode in result:
            for split, (depths, count) in matrix.items():
                for depth in depths:
                    for item in range(count):
                        prompt_tokens = 10
                        if split in {"joint_holdout", "contrast_joint"}:
                            substrate_cells = (
                                ("braided_branch", "compact", "node"),
                                ("braided_branch", "compact", "checksum"),
                            )
                        else:
                            substrate_cells = tuple(
                                (family, template, query)
                                for family in ("phase_branch", "checksum_branch")
                                for template in ("ledger", "prose")
                                for query in ("node", "checksum")
                            )
                        family, template, query_kind = substrate_cells[
                            item % len(substrate_cells)
                        ]
                        active = mode == "intact"
                        extra_cycles = (depth - 1) if active else 0
                        expected_parameters = (
                            16_232_448 if capacity == "lora" else 892_272_640
                        )
                        result[mode].append(
                            {
                                "id": f"{split}-depth{depth}-task{item}",
                                "split": split,
                                "depth": depth,
                                "family": family,
                                "template": template,
                                "query_kind": query_kind,
                                "capacity": capacity,
                                "objective": objective,
                                "model_seed": seed,
                                "adaptation_mode": mode,
                                "node_target": 0,
                                "phase_target": 0,
                                "checksum_target": 0,
                                "node_prediction": 0,
                                "phase_prediction": 0,
                                "checksum_prediction": 0,
                                "node_final_correct": True,
                                "phase_final_correct": True,
                                "checksum_final_correct": True,
                                "joint_final_correct": True,
                                "node_trajectory_targets": [0] * depth,
                                "phase_trajectory_targets": [0] * depth,
                                "checksum_trajectory_targets": [0] * depth,
                                "node_trajectory_predictions": [0] * depth,
                                "phase_trajectory_predictions": [0] * depth,
                                "checksum_trajectory_predictions": [0] * depth,
                                "node_trajectory_accuracy": 1.0,
                                "phase_trajectory_accuracy": 1.0,
                                "checksum_trajectory_accuracy": 1.0,
                                "joint_trajectory_accuracy": 1.0,
                                "answer_choice_target": 0,
                                "answer_choice_prediction": 0,
                                "answer_correct": True,
                                "full_top_is_answer": True,
                                "answer_token_mass": 0.75,
                                "state_change_rms_by_transition": [0.25] * (depth - 1),
                                "mean_state_change_rms": 0.25 if depth > 1 else 0.0,
                                "answer_loss": 0.0,
                                "state_loss": 0.0,
                                "fixed_point_loss": 0.0,
                                "prompt_tokens": prompt_tokens,
                                "base_layer_token_applications": prompt_tokens * 32,
                                "extra_loop_layer_token_applications": (
                                    prompt_tokens * 8 * (depth - 1)
                                ),
                                "total_layer_token_applications": (
                                    prompt_tokens * (32 + 8 * (depth - 1))
                                ),
                                "adaptation_forward_macs": (
                                    prompt_tokens * expected_parameters * extra_cycles
                                ),
                                "adaptation_calls": 62 * extra_cycles,
                                "adaptation_call_manifest_sha256": (
                                    "a" * 64
                                    if extra_cycles
                                    else hashlib.sha256(b"[]").hexdigest()
                                ),
                                "adaptation_cycles": extra_cycles,
                                "compute_proxy": (
                                    "exact_layer_token_applications_and_adapter_linear_macs;"
                                    "not_hardware_flops"
                                ),
                            }
                        )
        return result

    def validate(
        self,
        rows: dict[str, list[dict]],
        *,
        eval_set: str = "trigger",
        capacity: str = "lora",
        objective: str = "joint",
        seed: int = 7411,
    ) -> None:
        analysis._validate_evaluation_rows(
            self.config,
            rows,
            capacity=capacity,
            objective=objective,
            seed=seed,
            eval_set=eval_set,
        )

    def test_exact_trigger_and_contrast_matrices_pass(self) -> None:
        trigger = self.rows()
        self.assertEqual(set(trigger["intact"][0]), self.ROW_KEYS)
        self.validate(trigger)
        self.validate(self.rows(eval_set="contrast"), eval_set="contrast")
        self.validate(
            self.rows(capacity="fullrank", objective="state_only", seed=7413),
            capacity="fullrank",
            objective="state_only",
            seed=7413,
        )

    def test_validation_cells_reject_255_and_257_rows(self) -> None:
        for eval_set, split in (
            ("trigger", "validation"),
            ("contrast", "contrast_validation"),
        ):
            for depth in ((1, 2, 3, 4) if eval_set == "trigger" else (2, 3, 4)):
                for delta in (-1, 1):
                    rows = self.rows(eval_set=eval_set)
                    for mode in rows:
                        cell = [
                            row
                            for row in rows[mode]
                            if row["split"] == split and row["depth"] == depth
                        ]
                        if delta < 0:
                            rows[mode].remove(cell[-1])
                        else:
                            extra = dict(cell[-1])
                            extra["id"] = f"{split}-depth{depth}-extra-task"
                            rows[mode].append(extra)
                    with self.subTest(
                        eval_set=eval_set,
                        split=split,
                        depth=depth,
                        rows=256 + delta,
                    ), self.assertRaises(RuntimeError):
                        self.validate(rows, eval_set=eval_set)

    def test_every_deep_cell_rejects_127_and_129_rows(self) -> None:
        cases = (
            ("trigger", "depth_extrapolation"),
            ("trigger", "joint_holdout"),
            ("contrast", "contrast_depth"),
            ("contrast", "contrast_joint"),
        )
        for eval_set, split in cases:
            for depth in range(5, 13):
                for delta in (-1, 1):
                    rows = self.rows(eval_set=eval_set)
                    for mode in rows:
                        cell = [
                            row
                            for row in rows[mode]
                            if row["split"] == split and row["depth"] == depth
                        ]
                        if delta < 0:
                            rows[mode].remove(cell[-1])
                        else:
                            extra = dict(cell[-1])
                            extra["id"] = f"{split}-depth{depth}-extra-task"
                            rows[mode].append(extra)
                    with self.subTest(
                        eval_set=eval_set, split=split, depth=depth, rows=128 + delta
                    ):
                        with self.assertRaises(RuntimeError):
                            self.validate(rows, eval_set=eval_set)

    def test_duplicate_ids_and_unexpected_splits_are_rejected(self) -> None:
        duplicate = self.rows()
        for mode in duplicate:
            cell = [
                row
                for row in duplicate[mode]
                if row["split"] == "validation" and row["depth"] == 3
            ]
            cell[-1]["id"] = cell[0]["id"]
        with self.assertRaises(RuntimeError):
            self.validate(duplicate)

        unexpected_split = self.rows()
        for mode in unexpected_split:
            unexpected_split[mode][0]["split"] = "validation_shadow"
        with self.assertRaises(RuntimeError):
            self.validate(unexpected_split)

    def test_split_specific_family_template_and_query_grids_are_exact(self) -> None:
        mutations = (
            ("joint_holdout", "family", "phase_branch"),
            ("joint_holdout", "template", "ledger"),
            ("validation", "family", "braided_branch"),
            ("validation", "template", "compact"),
        )
        for split, field, value in mutations:
            rows = self.rows()
            row = next(item for item in rows["intact"] if item["split"] == split)
            row[field] = value
            with self.subTest(split=split, field=field), self.assertRaises(RuntimeError):
                self.validate(rows)

        unbalanced_query = self.rows()
        row = next(
            item
            for item in unbalanced_query["intact"]
            if item["split"] == "depth_extrapolation"
            and item["query_kind"] == "checksum"
        )
        row["query_kind"] = "node"
        with self.assertRaises(RuntimeError):
            self.validate(unbalanced_query)

        for field, value in (
            ("family", "braided_branch"),
            ("template", "compact"),
        ):
            contrast = self.rows(eval_set="contrast")
            row = next(
                item
                for item in contrast["intact"]
                if item["split"] == "contrast_validation"
            )
            row[field] = value
            with self.subTest(
                split="contrast_validation", field=field
            ), self.assertRaises(RuntimeError):
                self.validate(contrast, eval_set="contrast")

    def test_missing_or_extra_adaptation_mode_is_rejected(self) -> None:
        missing = self.rows()
        del missing["disabled"]
        with self.assertRaises(RuntimeError):
            self.validate(missing)

        extra = self.rows()
        extra["shadow"] = copy.deepcopy(extra["intact"])
        with self.assertRaises(RuntimeError):
            self.validate(extra)

    def test_each_row_must_bind_mode_capacity_objective_seed_and_depth(self) -> None:
        mutations = {
            "adaptation_mode": "disabled",
            "capacity": "fullrank",
            "objective": "state_only",
            "model_seed": 7412,
            "depth": 99,
        }
        for field, value in mutations.items():
            rows = self.rows()
            rows["intact"][0][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.validate(rows)

    def test_exact_schema_rejects_missing_or_extra_diagnostics(self) -> None:
        missing = self.rows()
        del missing["intact"][0]["answer_correct"]
        with self.assertRaises(RuntimeError):
            self.validate(missing)

        extra = self.rows()
        extra["intact"][0]["unregistered_diagnostic"] = 0.0
        with self.assertRaises(RuntimeError):
            self.validate(extra)

    def test_types_reject_string_booleans_bool_integers_and_bad_lists(self) -> None:
        mutations = (
            ("joint_final_correct", "true"),
            ("depth", True),
            ("model_seed", True),
            ("node_target", False),
            ("prompt_tokens", True),
            ("node_trajectory_targets", [False]),
            ("state_change_rms_by_transition", [1]),
        )
        for field, value in mutations:
            rows = self.rows()
            rows["intact"][0][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.validate(rows)

    def test_nonfinite_or_out_of_range_diagnostics_are_rejected(self) -> None:
        mutations = (
            ("node_trajectory_accuracy", float("nan")),
            ("joint_trajectory_accuracy", float("inf")),
            ("answer_token_mass", float("-inf")),
            ("mean_state_change_rms", float("nan")),
            ("answer_loss", float("nan")),
            ("state_loss", float("inf")),
            ("fixed_point_loss", float("-inf")),
            ("answer_token_mass", 1.01),
            ("mean_state_change_rms", -0.01),
            ("state_loss", -0.01),
        )
        for field, value in mutations:
            rows = self.rows()
            rows["intact"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                self.validate(rows)

        transitions = self.rows()
        transitions["intact"][256]["state_change_rms_by_transition"][0] = float("inf")
        with self.assertRaises(RuntimeError):
            self.validate(transitions)

    def test_diagnostic_lengths_values_and_compute_are_self_consistent(self) -> None:
        mutations = (
            ("node_trajectory_targets", [0]),
            ("state_change_rms_by_transition", []),
            ("node_target", 1),
            ("node_final_correct", False),
            ("joint_final_correct", False),
            ("node_trajectory_accuracy", 0.5),
            ("answer_correct", False),
            ("mean_state_change_rms", 0.5),
            ("base_layer_token_applications", 1),
            ("extra_loop_layer_token_applications", 1),
            ("total_layer_token_applications", 1),
            ("adaptation_forward_macs", 1),
            ("adaptation_calls", 1),
            ("adaptation_call_manifest_sha256", "not-a-digest"),
            ("adaptation_cycles", 0),
            ("compute_proxy", "estimated_flops"),
        )
        # Row 256 is the first K=2 row, so neither K-length nor transition-
        # length errors are accidentally valid as they would be for K=1.
        for field, value in mutations:
            rows = self.rows()
            rows["intact"][256][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.validate(rows)

    def test_intact_and_disabled_must_have_the_same_unique_keys(self) -> None:
        rows = self.rows()
        rows["disabled"][0]["id"] = "different-but-still-unique"
        with self.assertRaises(RuntimeError):
            self.validate(rows)

        metadata_swap = self.rows()
        left = metadata_swap["disabled"][0]
        right = next(
            row
            for row in metadata_swap["disabled"]
            if row["split"] == left["split"]
            and row["depth"] == left["depth"]
            and row["family"] != left["family"]
            and row["template"] == left["template"]
            and row["query_kind"] == left["query_kind"]
        )
        left["family"], right["family"] = right["family"], left["family"]
        with self.assertRaises(RuntimeError):
            self.validate(metadata_swap)


class EvaluationCorpusBindingTests(unittest.TestCase):
    """End-to-end loader checks for truth and setup lineage after grid validation."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "default.yaml")

    @staticmethod
    def rows() -> dict[str, list[dict]]:
        base = {
            "id": "validation-task-0",
            "split": "validation",
            "family": "phase_branch",
            "template": "ledger",
            "depth": 2,
            "query_kind": "node",
            "node_trajectory_targets": [1, 2],
            "phase_trajectory_targets": [0, 1],
            "checksum_trajectory_targets": [2, 3],
            "node_target": 2,
            "phase_target": 1,
            "checksum_target": 3,
            "answer_choice_target": 3,
        }
        return {
            mode: [dict(base, adaptation_mode=mode)]
            for mode in ("intact", "disabled")
        }

    @staticmethod
    def setup(*, device: str = "NVIDIA H100 80GB HBM3", free_gib: float = 72.0) -> dict:
        return {
            "environment": {
                "python": "3.12.3",
                "device": {
                    "name": device,
                    "compute_capability": [9, 0],
                    "free_memory_gib_before_load": free_gib,
                },
            },
            "preflight_device": {
                "name": device,
                "compute_capability": [9, 0],
                "free_memory_gib_before_load": free_gib,
            },
            "deterministic_setup_marker": "exact-cell-setup",
        }

    def load(
        self,
        rows: dict[str, list[dict]],
        *,
        g0_setup: dict | None = None,
        positive_control_setup: dict | None = None,
        preflight_error: RuntimeError | None = None,
        events: list[str] | None = None,
    ) -> tuple[dict, dict[str, list[dict]], dict]:
        config = copy.deepcopy(self.config)
        config["paths"]["data_dir"] = "data"
        config["evaluation"]["trigger_splits"] = ["validation"]
        run_setup = self.setup()
        g0_setup = copy.deepcopy(g0_setup or run_setup)
        positive_control_setup = copy.deepcopy(positive_control_setup or run_setup)
        digest = "d" * 64
        receipt_identity = "e" * 64
        checkpoint_identity = "c" * 64
        corpus_row = {
            "id": "validation-task-0",
            "split": "validation",
            "family": "phase_branch",
            "template": "ledger",
            "depth": 2,
            "query_kind": "node",
            "correct_choice": 3,
        }
        target_receipt = {
            "node": [1, 2],
            "phase": [0, 1],
            "checksum": [2, 3],
        }

        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory).resolve()
            experiment_root = repo_root
            data_dir = experiment_root / "data"
            data_dir.mkdir()
            data_manifest = {
                "files": {
                    "validation": {"sha256": digest, "canonical_rows": 1}
                }
            }
            (data_dir / "manifest.json").write_text(
                json.dumps(data_manifest), encoding="utf-8"
            )

            checkpoint_path = repo_root / "checkpoint_001500"
            checkpoint_path.mkdir()
            initialization_path = repo_root / "initialization_bundle"
            shared_initialization = {"bundle_path": str(initialization_path)}
            checkpoint = {
                "capacity": "lora",
                "objective": "joint",
                "model_seed": 7411,
                "step": int(config["training"]["train_steps"]),
                "data_manifest_sha256": digest,
                "checkpoint_identity_sha256": checkpoint_identity,
                "g0_lineage": {"kind": "g0"},
                "positive_control_lineage": {"kind": "positive_control"},
                "branch_authorization_lineage": None,
                "shared_initialization": shared_initialization,
                "adaptation_state_sha256": digest,
                "loop_state_sha256": digest,
            }
            (checkpoint_path / "checkpoint.json").write_text(
                json.dumps(checkpoint), encoding="utf-8"
            )
            for filename in ("adaptation_state.pt", "loop_state.pt"):
                (checkpoint_path / filename).write_bytes(b"payload")

            run = {
                "receipt_identity_sha256": receipt_identity,
                "checkpoint_metadata_sha256": digest,
                "checkpoint_identity_sha256": checkpoint_identity,
                "checkpoint_path": str(checkpoint_path),
                "setup": run_setup,
            }
            (checkpoint_path.parent / "run.json").write_text(
                json.dumps(run), encoding="utf-8"
            )

            runs_dir = repo_root / "runs"
            evaluation_dir = runs_dir / "lora_joint_seed7411_trigger"
            evaluation_dir.mkdir(parents=True)
            for mode in ("intact", "disabled"):
                (evaluation_dir / f"rows_{mode}.jsonl").write_text(
                    "{}\n", encoding="utf-8"
                )
            summary = {
                "status": "STATE_EVALUATION_COMPLETE",
                "capacity": "lora",
                "objective": "joint",
                "model_seed": 7411,
                "eval_set": "trigger",
                "receipt_identity_sha256": receipt_identity,
                "k1_max_logit_abs_error": 0.0,
                "k1_adaptation_calls": 0,
                "data_manifest_sha256": digest,
                "split_payloads": {
                    "validation": {"sha256": digest, "canonical_rows": 1}
                },
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_metadata_sha256": digest,
                "checkpoint_identity_sha256": checkpoint_identity,
                "modes": {
                    mode: {
                        "rows_path": f"rows_{mode}.jsonl",
                        "rows_sha256": digest,
                        "rows": 1,
                    }
                    for mode in ("intact", "disabled")
                },
            }
            (evaluation_dir / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )

            stable_summary_reader = analysis._read_stable_json_object

            def read_summary(path: Path) -> tuple[dict, str]:
                if events is not None:
                    events.append("evaluation_summary")
                return stable_summary_reader(path)

            def reopen_gate(entry: dict) -> dict:
                setups = {
                    "g0": g0_setup,
                    "positive_control": positive_control_setup,
                }
                return {"setup": copy.deepcopy(setups[entry["kind"]])}

            def contract_preflight(*_args, **_kwargs) -> dict:
                if events is not None:
                    events.append("contract_preflight")
                if preflight_error is not None:
                    raise preflight_error
                return {
                    "branch": {},
                    "setup_barrier": {"status": "SETUP_BARRIER_COMPLETE"},
                    "training_barrier": {
                        "status": "REACHED_TRAINING_BARRIER_COMPLETE"
                    },
                    "training_cell": {},
                    "checkpoint_path": checkpoint_path,
                    "checkpoint": copy.deepcopy(checkpoint),
                    "run": copy.deepcopy(run),
                    "setup_gate_receipts": {
                        "g0_lineage": reopen_gate(checkpoint["g0_lineage"]),
                        "positive_control_lineage": reopen_gate(
                            checkpoint["positive_control_lineage"]
                        ),
                    },
                }

            def bind_summary(*_args, **kwargs) -> dict:
                return dict(kwargs["graph_preflight"])

            def read_rows(path: Path, _expected_sha256: str) -> list[dict]:
                if events is not None:
                    events.append("evaluation_rows")
                mode = "intact" if path.name == "rows_intact.jsonl" else "disabled"
                return copy.deepcopy(rows[mode])

            def validate_manifest(*_args, **_kwargs) -> None:
                if events is not None:
                    events.append("manifest_content")

            def read_corpus(*_args) -> list[dict]:
                if events is not None:
                    events.append("task_rows")
                return [copy.deepcopy(corpus_row)]

            patches = (
                mock.patch.object(analysis, "ROOT", experiment_root),
                mock.patch.object(analysis, "REPO_ROOT", repo_root),
                mock.patch.object(analysis, "_identity", return_value={}),
                mock.patch.object(
                    analysis, "_canonical_sha256", return_value=receipt_identity
                ),
                mock.patch.object(analysis, "_sha256", return_value=digest),
                mock.patch.object(
                    analysis, "_read_stable_json_object", side_effect=read_summary
                ),
                mock.patch.object(
                    analysis,
                    "read_verified_json_object",
                    return_value=copy.deepcopy(data_manifest),
                ),
                mock.patch.object(
                    analysis, "_checkpoint_identity", return_value=checkpoint_identity
                ),
                mock.patch.object(
                    analysis,
                    "_evaluation_graph_preflight",
                    side_effect=contract_preflight,
                ),
                mock.patch.object(
                    analysis,
                    "_evaluation_contract_preflight",
                    side_effect=bind_summary,
                ),
                mock.patch.object(
                    analysis, "_resolve_repo_path", side_effect=lambda value: Path(value).resolve()
                ),
                mock.patch.object(
                    analysis, "validate_data_manifest", side_effect=validate_manifest
                ),
                mock.patch.object(
                    analysis,
                    "load_initialization_bundle",
                    return_value=(None, shared_initialization),
                ),
                mock.patch.object(
                    analysis, "_validate_lineage_entry", side_effect=reopen_gate
                ),
                mock.patch.object(
                    analysis, "_validate_training_payloads", return_value={}
                ),
                mock.patch.object(
                    analysis, "_validate_evaluation_rows", return_value={}
                ),
                mock.patch.object(
                    analysis, "_read_verified_jsonl", side_effect=read_rows
                ),
                mock.patch.object(
                    analysis, "read_verified_jsonl_gzip", side_effect=read_corpus
                ),
                mock.patch.object(analysis, "verify_example"),
                mock.patch.object(
                    analysis, "trajectory_targets", return_value=target_receipt
                ),
            )
            for patcher in patches:
                patcher.start()
            try:
                return analysis._load_evaluation(
                    config,
                    runs_dir,
                    capacity="lora",
                    objective="joint",
                    seed=7411,
                    eval_set="trigger",
                )
            finally:
                for patcher in reversed(patches):
                    patcher.stop()

    def test_loader_recomputes_truth_and_rejects_an_internally_consistent_mutation(self) -> None:
        self.assertEqual(self.load(self.rows())[0]["status"], "STATE_EVALUATION_COMPLETE")
        mutated = self.rows()
        for mode in ("intact", "disabled"):
            row = mutated[mode][0]
            row.update(
                {
                    "node_trajectory_targets": [0, 0],
                    "phase_trajectory_targets": [0, 0],
                    "checksum_trajectory_targets": [0, 0],
                    "node_target": 0,
                    "phase_target": 0,
                    "checksum_target": 0,
                    "answer_choice_target": 0,
                }
            )
        with self.assertRaisesRegex(RuntimeError, "exact corpus truth"):
            self.load(mutated)

    def test_loader_reopens_both_setup_gates_and_matches_the_exact_training_setup(self) -> None:
        for label, keyword in (
            ("g0_lineage", "g0_setup"),
            ("positive_control_lineage", "positive_control_setup"),
        ):
            changed = self.setup(device="different GPU")
            with self.subTest(label=label), self.assertRaisesRegex(
                RuntimeError, f"exact {label}"
            ):
                self.load(self.rows(), **{keyword: changed})

        # Availability is volatile and intentionally removed from the stable
        # receipt; all durable device identity fields remain equal.
        self.assertEqual(
            self.load(
                self.rows(),
                g0_setup=self.setup(free_gib=51.0),
                positive_control_setup=self.setup(free_gib=49.0),
            )[0]["status"],
            "STATE_EVALUATION_COMPLETE",
        )

    def test_contract_preflight_fails_before_any_manifest_or_result_content(self) -> None:
        events: list[str] = []
        with self.assertRaisesRegex(RuntimeError, "tracked terminal receipt missing"):
            self.load(
                self.rows(),
                preflight_error=RuntimeError("tracked terminal receipt missing"),
                events=events,
            )
        self.assertEqual(events, ["contract_preflight"])


class AnalysisContractPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "default.yaml")

    @staticmethod
    def lineage(path: str, status: str, phase: str, marker: str) -> dict:
        return {
            "path": path,
            "sha256": marker * 64,
            "receipt_identity_sha256": marker * 64,
            "status": status,
            "phase": phase,
        }

    def test_named_branch_uses_status_specific_canonical_path(self) -> None:
        returned = {
            "lineage": self.lineage(
                "experiments/qwen35_4b_state_formation_capacity_adjudication/"
                "analysis/lora_joint_trigger.json",
                "LORA_JOINT_MISS_CONTROLS_REQUIRED",
                "lora_joint_analysis",
                "1",
            )
        }
        with mock.patch.object(
            analysis, "validate_branch_authorization", return_value=returned
        ) as validator, mock.patch.object(
            analysis, "_identity", return_value={"phase": "lora_joint_analysis"}
        ):
            self.assertEqual(
                analysis._validate_named_branch(
                    self.config, analysis.LORA_MISS_BRANCH
                ),
                returned,
            )
        args, kwargs = validator.call_args
        self.assertEqual(args[0], analysis.REPO_ROOT)
        self.assertEqual(
            args[1], ROOT / "analysis" / "lora_joint_trigger.json"
        )
        self.assertEqual(kwargs["branch"], analysis.LORA_MISS_BRANCH)
        self.assertEqual(
            kwargs["canonical_relative_path"],
            "experiments/qwen35_4b_state_formation_capacity_adjudication/"
            "analysis/lora_joint_trigger.json",
        )

    def test_stage_c_rejects_nested_or_status_only_decoy_authorization(self) -> None:
        decoy = self.lineage(
            "analysis/nested/decoy.json",
            "FULLRANK_STATE_ONLY_REQUIRED",
            "fullrank_joint_analysis",
            "2",
        )
        summary = {
            "training_branch_authorization": decoy,
            "contrast_authorization": None,
        }
        with mock.patch.object(analysis, "_validate_named_branch") as validator:
            with self.assertRaisesRegex(RuntimeError, "canonical registered branch"):
                analysis._evaluation_branch_context(
                    self.config,
                    summary,
                    capacity="fullrank",
                    objective="state_only",
                    eval_set="trigger",
                )
        validator.assert_not_called()

    def test_reached_barrier_is_ordered_and_binds_each_stage_authorization(self) -> None:
        root = self.lineage(
            "experiments/qwen35_4b_state_formation_capacity_adjudication/"
            "analysis/lora_joint_trigger.json",
            "LORA_JOINT_MISS_CONTROLS_REQUIRED",
            "lora_joint_analysis",
            "3",
        )
        context = {
            "reached_stage": "B",
            "root_lora_miss_lineage": root,
            "training_authorization": root,
        }
        calls = []

        def barrier(_root, stage, contracts, *, required_authorization):
            calls.append((stage, contracts, required_authorization))
            return {
                "schema_version": 1,
                "status": "TRAINING_BARRIER_COMPLETE",
                "stage": stage,
                "cells": [{"cell": f"stage-{stage}"}],
                "branch_authorization_lineage": required_authorization,
                "barrier_identity_sha256": stage.lower() * 64,
            }

        contracts = {"sentinel": object()}
        setup_barrier = {"cells": []}
        with mock.patch.object(
            analysis, "_training_contracts", return_value=contracts
        ), mock.patch.object(
            analysis, "evaluation_barrier", side_effect=barrier
        ), mock.patch.object(
            analysis, "_bind_training_cells_to_setup"
        ):
            proof = analysis._reached_training_barrier(
                self.config, context, setup_barrier
            )
        self.assertEqual([item[0] for item in calls], ["A", "B"])
        self.assertIsNone(calls[0][2])
        self.assertEqual(calls[1][2], root)
        self.assertTrue(all(item[1] is contracts for item in calls))
        self.assertEqual(proof["reached_stage"], "B")
        claimed = proof["barrier_identity_sha256"]
        self.assertEqual(
            claimed,
            analysis._canonical_sha256(
                {
                    key: value
                    for key, value in proof.items()
                    if key != "barrier_identity_sha256"
                }
            ),
        )

    def test_stage_b_setup_barrier_reopens_all_six_pairs_in_frozen_order(self) -> None:
        root = self.lineage(
            "experiments/qwen35_4b_state_formation_capacity_adjudication/"
            "analysis/lora_joint_trigger.json",
            "LORA_JOINT_MISS_CONTROLS_REQUIRED",
            "lora_joint_analysis",
            "8",
        )
        calls = []

        def reopen(_config, *, capacity, model_seed, **kwargs):
            calls.append((capacity, model_seed, kwargs["root_lora_miss_lineage"]))
            marker = "a" if capacity == "lora" else "b"
            targets = ["model.layers.12.self_attn.q_proj"]
            target_digest = hashlib.sha256("\n".join(targets).encode()).hexdigest()
            return {
                "g0": {},
                "control": {},
                "g0_lineage": self.lineage(
                    f"runs/setup/g0_{capacity}_seed{model_seed}.json",
                    "MODEL_SMOKE_PASS",
                    f"{capacity}_g0",
                    marker,
                ),
                "control_lineage": self.lineage(
                    f"runs/setup/positive_control_{capacity}_seed{model_seed}.json",
                    "POSITIVE_CONTROL_PASS",
                    f"{capacity}_positive_control",
                    marker,
                ),
                "setup": {
                    "capacity": capacity,
                    "model_seed": model_seed,
                    "tokenizer": {"vocabulary_sha256": "c" * 64},
                    "adaptation_targets": targets,
                    "adaptation_targets_sha256": target_digest,
                    "adaptation_target_manifest": [
                        {"name": targets[0], "shape": [2, 2]}
                    ],
                    "adaptation_target_manifest_sha256": marker * 64,
                    "adaptation_parameters": 1 if capacity == "lora" else 4,
                    "adaptation_zero_function": {"enabled_error": 0.0},
                    "trainable_parameters": {
                        "total": 10 if capacity == "lora" else 13,
                        "tensor_count": 2,
                        "values_sha256": str(model_seed) * 16,
                    },
                    "dropout_control": {"matched_adaptation_dropout": 0.05},
                    "environment": {"device": {"name": "registered GPU"}},
                    "installed_environment_lock": {"sha256": "d" * 64},
                    "preflight_device": {"name": "registered GPU"},
                    "shared_initialization": {"model_seed": model_seed},
                },
            }

        with mock.patch.object(
            analysis, "_reopen_setup_pair", side_effect=reopen
        ), mock.patch.object(
            analysis,
            "strict_stable_setup_receipt",
            side_effect=lambda value: dict(value),
        ):
            proof, cells = analysis._recompute_setup_barrier(
                self.config,
                stage="B",
                data_manifest_sha256="d" * 64,
                root_lora_miss_lineage=root,
            )
        expected_calls = [
            (capacity, seed, root)
            for capacity in ("lora", "fullrank")
            for seed in SEEDS
        ]
        self.assertEqual(calls, expected_calls)
        self.assertEqual(
            list(cells),
            [
                f"{capacity}_seed{seed}"
                for capacity in ("lora", "fullrank")
                for seed in SEEDS
            ],
        )
        self.assertEqual(proof["root_lora_miss_lineage"], root)
        self.assertEqual(
            proof["barrier_identity_sha256"],
            analysis._canonical_sha256(
                {
                    key: value
                    for key, value in proof.items()
                    if key != "barrier_identity_sha256"
                }
            ),
        )

    def test_evaluation_access_claims_are_exact_and_type_strict(self) -> None:
        summary = {
            "authorizes_training": False,
            "authorizes_result_training": False,
            "authorizes_result_evaluation": False,
            "benchmark_files_read": 0,
            "result_payloads_opened": list(
                self.config["evaluation"]["trigger_splits"]
            ),
            "sealed_contrast_payloads_opened": [],
            "training_or_evaluation_started": True,
            "scientific_evidence": True,
            "contrast_access_event": None,
        }
        analysis._validate_evaluation_access_claims(
            self.config, summary, eval_set="trigger"
        )
        mutations = {
            "authorizes_result_evaluation": True,
            "benchmark_files_read": False,
            "result_payloads_opened": [],
            "sealed_contrast_payloads_opened": ["contrast_depth"],
            "training_or_evaluation_started": False,
            "scientific_evidence": False,
        }
        for field, value in mutations.items():
            altered = copy.deepcopy(summary)
            altered[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                RuntimeError, "access claim changed"
            ):
                analysis._validate_evaluation_access_claims(
                    self.config, altered, eval_set="trigger"
                )

    def test_terminal_barrier_failure_and_rehashed_summary_tamper_fail_closed(self) -> None:
        context = {
            "reached_stage": "A",
            "root_lora_miss_lineage": None,
            "training_authorization": None,
        }
        setup_barrier = {"cells": []}
        with mock.patch.object(
            analysis, "_training_contracts", return_value={}
        ), mock.patch.object(
            analysis,
            "evaluation_barrier",
            side_effect=RuntimeError("tracked TRAINING_COMPLETE mirror missing"),
        ):
            with self.assertRaisesRegex(RuntimeError, "tracked TRAINING_COMPLETE"):
                analysis._reached_training_barrier(
                    self.config, context, setup_barrier
                )

        canonical = {
            "schema_version": 1,
            "status": "REACHED_TRAINING_BARRIER_COMPLETE",
            "stages": [{"stage": "A", "cells": []}],
            "reached_stage": "A",
        }
        canonical["barrier_identity_sha256"] = analysis._canonical_sha256(canonical)
        tampered = copy.deepcopy(canonical)
        tampered["stages"][0]["cells"] = [{"cell": "orphan"}]
        tampered["barrier_identity_sha256"] = analysis._canonical_sha256(
            {
                key: value
                for key, value in tampered.items()
                if key != "barrier_identity_sha256"
            }
        )
        setup_proof = {"status": "SETUP_BARRIER_COMPLETE"}
        summary = {
            "data_manifest_sha256": "d" * 64,
            "setup_barrier": setup_proof,
            "training_barrier": tampered,
        }
        with mock.patch.object(
            analysis, "_evaluation_branch_context", return_value=context
        ), mock.patch.object(
            analysis, "_validate_evaluation_access_claims"
        ), mock.patch.object(
            analysis,
            "_recompute_setup_barrier",
            return_value=(setup_proof, {}),
        ), mock.patch.object(
            analysis, "_reached_training_barrier", return_value=canonical
        ), mock.patch.object(
            analysis, "_current_training_cell_proof"
        ) as cell_proof:
            with self.assertRaisesRegex(RuntimeError, "training barrier changed"):
                analysis._evaluation_contract_preflight(
                    self.config,
                    summary,
                    capacity="lora",
                    objective="joint",
                    seed=7411,
                    eval_set="trigger",
                )
        cell_proof.assert_not_called()

    def test_rehashed_target_cell_proof_cannot_replace_barrier_selected_cell(self) -> None:
        context = {
            "reached_stage": "A",
            "root_lora_miss_lineage": None,
            "training_authorization": None,
        }
        setup_proof = {"status": "SETUP_BARRIER_COMPLETE"}
        target = {
            "cell": "lora_joint_seed7411",
            "checkpoint_path": "canonical/checkpoint_001500",
            "checkpoint_metadata_sha256": "1" * 64,
            "checkpoint_identity_sha256": "2" * 64,
        }
        barrier = {
            "status": "REACHED_TRAINING_BARRIER_COMPLETE",
            "stages": [],
        }
        decoy = copy.deepcopy(target)
        decoy["checkpoint_path"] = "decoy/checkpoint_001500"
        decoy["proof_identity_sha256"] = analysis._canonical_sha256(decoy)
        summary = {
            "data_manifest_sha256": "d" * 64,
            "setup_barrier": setup_proof,
            "training_barrier": barrier,
            "target_training_cell_proof": decoy,
        }
        with mock.patch.object(
            analysis, "_evaluation_branch_context", return_value=context
        ), mock.patch.object(
            analysis, "_validate_evaluation_access_claims"
        ), mock.patch.object(
            analysis,
            "_recompute_setup_barrier",
            return_value=(setup_proof, {}),
        ), mock.patch.object(
            analysis, "_reached_training_barrier", return_value=barrier
        ), mock.patch.object(
            analysis, "_current_training_cell_proof", return_value=target
        ):
            with self.assertRaisesRegex(RuntimeError, "target training-cell proof"):
                analysis._evaluation_contract_preflight(
                    self.config,
                    summary,
                    capacity="lora",
                    objective="joint",
                    seed=7411,
                    eval_set="trigger",
                )

    def test_semantic_setup_validators_bind_exact_checkpoint_lineages(self) -> None:
        setup = {"capacity": "fullrank", "model_seed": 7411}
        root = self.lineage(
            "experiments/qwen35_4b_state_formation_capacity_adjudication/"
            "analysis/lora_joint_trigger.json",
            "LORA_JOINT_MISS_CONTROLS_REQUIRED",
            "lora_joint_analysis",
            "4",
        )
        g0_lineage = self.lineage(
            "experiments/qwen35_4b_state_formation_capacity_adjudication/"
            "runs/setup/g0_fullrank_seed7411.json",
            "MODEL_SMOKE_PASS",
            "fullrank_g0",
            "5",
        )
        control_lineage = self.lineage(
            "experiments/qwen35_4b_state_formation_capacity_adjudication/"
            "runs/setup/positive_control_fullrank_seed7411.json",
            "POSITIVE_CONTROL_PASS",
            "fullrank_positive_control",
            "6",
        )
        checkpoint = {
            "setup": setup,
            "data_manifest_sha256": "d" * 64,
            "g0_lineage": g0_lineage,
            "positive_control_lineage": control_lineage,
        }
        g0_receipt = {"status": "MODEL_SMOKE_PASS"}
        control_receipt = {"status": "POSITIVE_CONTROL_PASS"}

        def lineage_for(_root, path, _receipt):
            return (
                g0_lineage
                if Path(path).name.startswith("g0_")
                else control_lineage
            )

        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "g0.json"
            candidate.write_text(json.dumps({"setup": setup}), encoding="utf-8")
            patches = (
                mock.patch.object(
                    analysis, "canonical_repo_path", return_value=candidate
                ),
                mock.patch.object(
                    analysis,
                    "_read_stable_json_object",
                    return_value=({"setup": setup}, "0" * 64),
                ),
                mock.patch.object(
                    analysis, "_expected_positive_control_rows", return_value=[]
                ),
                mock.patch.object(analysis, "_validate_registered_setup_fields"),
                mock.patch.object(
                    analysis, "validate_g0_pass", return_value=g0_receipt
                ),
                mock.patch.object(
                    analysis,
                    "validate_positive_control_pass",
                    return_value=control_receipt,
                ),
                mock.patch.object(
                    analysis, "lineage_entry", side_effect=lineage_for
                ),
                mock.patch.object(
                    analysis, "_identity", return_value={"identity": "bound"}
                ),
                mock.patch.object(
                    analysis,
                    "strict_stable_setup_receipt",
                    side_effect=lambda value: dict(value),
                ),
            )
            started = [patcher.start() for patcher in patches]
            try:
                receipts = analysis._validate_semantic_setup_gates(
                    self.config,
                    capacity="fullrank",
                    seed=7411,
                    checkpoint=checkpoint,
                    branch_context={"root_lora_miss_lineage": root},
                )
            finally:
                for patcher in reversed(patches):
                    patcher.stop()
        g0_validator = started[4]
        control_validator = started[5]
        self.assertEqual(receipts["g0_lineage"], g0_receipt)
        self.assertEqual(receipts["positive_control_lineage"], control_receipt)
        self.assertEqual(
            g0_validator.call_args.kwargs["expected_branch_authorization"], root
        )
        self.assertEqual(g0_validator.call_args.kwargs["expected_setup"], setup)
        self.assertEqual(
            g0_validator.call_args.kwargs["expected_adaptation_parameters"],
            int(
                self.config["architecture"]["adaptation"]["fullrank"][
                    "expected_parameters"
                ]
            ),
        )
        self.assertEqual(
            g0_validator.call_args.kwargs["expected_lora_rank"],
            int(self.config["architecture"]["adaptation"]["lora"]["rank"]),
        )
        self.assertEqual(
            g0_validator.call_args.kwargs["expected_peft_version"], "0.19.1"
        )
        self.assertEqual(
            control_validator.call_args.kwargs["expected_g0_lineage"], g0_lineage
        )
        self.assertEqual(
            control_validator.call_args.kwargs["expected_branch_authorization"],
            root,
        )
        self.assertEqual(
            control_validator.call_args.kwargs["control_query_kinds"],
            ("node", "checksum"),
        )
        self.assertEqual(
            control_validator.call_args.kwargs["control_examples_per_cell"], 2
        )

        wrong_control = copy.deepcopy(control_lineage)
        wrong_control["receipt_identity_sha256"] = "7" * 64
        validated = {
            "g0": g0_receipt,
            "control": control_receipt,
            "g0_lineage": g0_lineage,
            "control_lineage": wrong_control,
            "setup": setup,
        }
        with mock.patch.object(
            analysis, "_reopen_setup_pair", return_value=validated
        ), mock.patch.object(
            analysis,
            "strict_stable_setup_receipt",
            side_effect=lambda value: dict(value),
        ):
            with self.assertRaisesRegex(RuntimeError, "positive-control lineage"):
                analysis._validate_semantic_setup_gates(
                    self.config,
                    capacity="fullrank",
                    seed=7411,
                    checkpoint=checkpoint,
                    branch_context={"root_lora_miss_lineage": root},
                )


class TrainingProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "smoke.yaml")

    @staticmethod
    def _shared_fields(order_sha256: str) -> dict:
        return {
            "training_prompt_tokens": 40,
            "training_layer_token_applications": 1280,
            "training_order_sha256": order_sha256,
            "dropout_schedule_sha256": "1" * 64,
            "dropout_probes": [],
            "train_metrics_sha256": "2" * 64,
            "train_metrics_rows": 2,
            "train_metrics_path": "placeholder",
            "optimizer_steps_sha256": "3" * 64,
            "optimizer_steps_rows": 2,
            "optimizer_steps_path": "placeholder",
            "optimizer_state": {
                "delta_states_complete": True,
                "delta_moment_tensors": 2,
                "delta_parameters_audited": 1,
                "all_required_group_states_complete_and_finite": True,
                "registered_missing_state_exemptions": 0,
            },
            "optimizer_step_receipt": {},
            "setup_sha256": "4" * 64,
            "stable_setup": {},
        }

    def test_self_consistent_but_wrong_training_order_digest_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["paths"]["data_dir"] = "data"
        config["paths"]["large_artifacts_dir"] = "large"
        train_rows = [{"id": f"train-{index}"} for index in range(4)]
        wrong_order = "f" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            checkpoint_path = (
                root
                / "large"
                / "lora_joint_seed7411"
                / "checkpoint_000002"
            )
            checkpoint_path.mkdir(parents=True)
            shared = self._shared_fields(wrong_order)
            checkpoint = {
                "data_manifest_sha256": "d" * 64,
                **copy.deepcopy(shared),
            }
            run = {
                "schema_version": 1,
                "status": "TRAINING_COMPLETE",
                "capacity": "lora",
                "objective": "joint",
                "model_seed": 7411,
                "steps": 2,
                "data_manifest_sha256": "d" * 64,
                **copy.deepcopy(shared),
            }
            self.assertEqual(
                run["training_order_sha256"],
                checkpoint["training_order_sha256"],
            )
            with mock.patch.object(analysis, "ROOT", root), mock.patch.object(
                analysis, "REPO_ROOT", root
            ), mock.patch.object(
                analysis, "_identity", return_value={}
            ), mock.patch.object(
                analysis,
                "read_jsonl",
                side_effect=lambda _path: copy.deepcopy(train_rows),
            ):
                expected = analysis._expected_training_order_sha256(config, 7411)
                self.assertNotEqual(wrong_order, expected)
                with mock.patch.object(
                    analysis,
                    "_expected_training_order_sha256",
                    return_value=expected,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "registered seeded schedule"
                    ):
                        analysis._validate_training_payloads(
                            config,
                            run=run,
                            checkpoint=checkpoint,
                            checkpoint_path=checkpoint_path,
                            capacity="lora",
                            objective="joint",
                            seed=7411,
                        )

    def _assert_self_consistent_wrong_lr_rejected(self, source: str) -> None:
        config = copy.deepcopy(self.config)
        config["paths"]["data_dir"] = "data"
        config["paths"]["large_artifacts_dir"] = "large"
        train_rows = [{"id": f"train-{index}"} for index in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            canonical_run = root / "large" / "lora_joint_seed7411"
            checkpoint_path = canonical_run / "checkpoint_000002"
            checkpoint_path.mkdir(parents=True)
            metrics_path = canonical_run / "train_metrics.jsonl"
            optimizer_path = canonical_run / "optimizer_steps.jsonl"
            with mock.patch.object(analysis, "ROOT", root), mock.patch.object(
                analysis, "REPO_ROOT", root
            ), mock.patch.object(
                analysis, "_identity", return_value={}
            ), mock.patch.object(
                analysis,
                "read_jsonl",
                side_effect=lambda _path: copy.deepcopy(train_rows),
            ):
                order_sha256 = analysis._expected_training_order_sha256(config, 7411)
                metric_rows = []
                for step in (1, 2):
                    metric_rows.append(
                        {
                            "step": step,
                            "capacity": "lora",
                            "objective": "joint",
                            "model_seed": 7411,
                            "loss": 1.0,
                            "answer": 1.0,
                            "state": 1.0,
                            "fixed": 1.0,
                            "adaptation_learning_rate": analysis._training_lr(
                                config, step
                            ),
                            "common_state_learning_rate": analysis._training_lr(
                                config, step
                            ),
                            "preclip_adaptation_gradient_norm": 0.5,
                            "preclip_common_gradient_norm": 0.5,
                            "adaptation_applied_clip_scale": 1.0,
                            "common_state_applied_clip_scale": 1.0,
                            "elapsed_seconds": 1.0,
                            "peak_allocated_gib": 1.0,
                        }
                    )
                optimizer_rows = [
                    {
                        "step": step,
                        "adaptation_preclip_gradient_norm": 0.5,
                        "adaptation_applied_clip_scale": 1.0,
                        "common_state_preclip_gradient_norm": 0.5,
                        "common_state_applied_clip_scale": 1.0,
                        "adaptation_gradient_finite": True,
                        "common_state_gradient_finite": True,
                        "base_trainable_parameters": 0,
                        "adaptation_learning_rate": analysis._training_lr(
                            config, step
                        ),
                        "common_state_learning_rate": analysis._training_lr(
                            config, step
                        ),
                    }
                    for step in (1, 2)
                ]
                if source == "metric":
                    metric_rows[0]["common_state_learning_rate"] *= 0.5
                    self.assertEqual(
                        metric_rows[0]["adaptation_learning_rate"],
                        analysis._training_lr(config, 1),
                    )
                elif source == "optimizer":
                    optimizer_rows[0]["common_state_learning_rate"] *= 0.5
                    self.assertEqual(
                        optimizer_rows[0]["adaptation_learning_rate"],
                        analysis._training_lr(config, 1),
                    )
                else:
                    raise AssertionError(source)
                metrics_path.write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in metric_rows),
                    encoding="utf-8",
                )
                optimizer_path.write_text(
                    "".join(
                        json.dumps(row, sort_keys=True) + "\n"
                        for row in optimizer_rows
                    ),
                    encoding="utf-8",
                )
                optimizer_digest = hashlib.sha256()
                for row in optimizer_rows:
                    optimizer_digest.update(
                        json.dumps(
                            row, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                        + b"\n"
                    )
                setup = {
                    "capacity": "lora",
                    "model_seed": 7411,
                    "shared_initialization": {"receipt": "shared"},
                    "trainable_parameters": {"total": 2},
                    "adaptation_parameters": 1,
                    "adaptation_target_manifest_sha256": "5" * 64,
                    "environment": {
                        "device": {
                            "name": "synthetic GPU",
                            "free_memory_gib_before_load": 70.0,
                        }
                    },
                    "preflight_device": {
                        "name": "synthetic GPU",
                        "free_memory_gib_before_load": 70.0,
                    },
                }
                shared = self._shared_fields(order_sha256)
                shared.update(
                    {
                        "train_metrics_sha256": analysis._sha256(metrics_path),
                        "train_metrics_path": metrics_path.relative_to(root).as_posix(),
                        "optimizer_steps_sha256": analysis._sha256(optimizer_path),
                        "optimizer_steps_path": optimizer_path.relative_to(root).as_posix(),
                        "optimizer_step_receipt": {
                            "schema_version": 1,
                            "steps": 2,
                            "rows": 2,
                            "events_sha256": optimizer_digest.hexdigest(),
                            "group_names": ["adaptation", "common_state"],
                            "clip_thresholds": {
                                "adaptation": 1.0,
                                "common_state": 1.0,
                            },
                            "minimum_applied_clip_scales": {
                                "adaptation": 1.0,
                                "common_state": 1.0,
                            },
                            "all_gradients_finite": True,
                            "base_trainable_parameters": 0,
                            "probes": [
                                optimizer_rows[0],
                                optimizer_rows[0],
                                optimizer_rows[1],
                            ],
                        },
                        "setup_sha256": analysis._canonical_sha256(setup),
                        "stable_setup": analysis._stable_setup_receipt(setup),
                    }
                )
                checkpoint = {
                    "data_manifest_sha256": "d" * 64,
                    "shared_initialization": setup["shared_initialization"],
                    "trainable_parameters": setup["trainable_parameters"],
                    "adaptation_parameters": setup["adaptation_parameters"],
                    "adaptation_target_manifest_sha256": setup[
                        "adaptation_target_manifest_sha256"
                    ],
                    "environment": setup["environment"],
                    **copy.deepcopy(shared),
                }
                run = {
                    "schema_version": 1,
                    "status": "TRAINING_COMPLETE",
                    "capacity": "lora",
                    "objective": "joint",
                    "model_seed": 7411,
                    "steps": 2,
                    "data_manifest_sha256": "d" * 64,
                    "setup": setup,
                    **copy.deepcopy(shared),
                }
                self.assertEqual(
                    run["train_metrics_sha256"], checkpoint["train_metrics_sha256"]
                )
                self.assertEqual(
                    run["optimizer_step_receipt"],
                    checkpoint["optimizer_step_receipt"],
                )
                with mock.patch.object(
                    analysis,
                    "_expected_training_order_sha256",
                    return_value=order_sha256,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "learning-rate schedule changed"
                    ):
                        analysis._validate_training_payloads(
                            config,
                            run=run,
                            checkpoint=checkpoint,
                            checkpoint_path=checkpoint_path,
                            capacity="lora",
                            objective="joint",
                            seed=7411,
                        )

    def test_wrong_common_lr_rejects_even_when_adaptation_lr_and_receipts_are_consistent(self) -> None:
        for source in ("metric", "optimizer"):
            with self.subTest(source=source):
                self._assert_self_consistent_wrong_lr_rejected(source)


class StageBMatchingTests(unittest.TestCase):
    AUTHORIZATION = {
        "path": "synthetic/lora_miss.json",
        "sha256": "1" * 64,
        "receipt_identity_sha256": "2" * 64,
        "status": "LORA_JOINT_MISS_CONTROLS_REQUIRED",
        "phase": "lora_joint_analysis",
    }

    @staticmethod
    def setup(seed: int, shared: dict) -> dict:
        targets = ["layers.12.self_attn.q_proj", "layers.12.mlp.down_proj"]
        target_digest = hashlib.sha256("\n".join(targets).encode()).hexdigest()
        return {
            "model_seed": seed,
            "tokenizer": {"vocabulary_sha256": "a" * 64},
            "adaptation_targets": targets,
            "adaptation_targets_sha256": target_digest,
            "shared_initialization": copy.deepcopy(shared),
            "dropout_control": {"nonadapter_training_dropout_modules": 0},
            "environment": {
                "python": "3.12.3",
                "device": {
                    "name": "NVIDIA H100 80GB HBM3",
                    "compute_capability": [9, 0],
                    "free_memory_gib_before_load": 72.0,
                },
            },
            "installed_environment_lock": {
                "path": "requirements.lock",
                "sha256": "b" * 64,
            },
            "preflight_device": {
                "name": "NVIDIA H100 80GB HBM3",
                "compute_capability": [9, 0],
                "free_memory_gib_before_load": 72.0,
            },
        }

    @staticmethod
    def manifests() -> tuple[list[dict], list[dict], list[dict]]:
        groups = ([], [], [])
        for seed in SEEDS:
            shared = {
                "receipt_identity_sha256": hashlib.sha256(
                    f"shared-{seed}".encode()
                ).hexdigest()
            }
            run = {
                "data_manifest_sha256": "3" * 64,
                "training_order_sha256": hashlib.sha256(
                    f"order-{seed}".encode()
                ).hexdigest(),
                "dropout_schedule_sha256": hashlib.sha256(
                    f"dropout-{seed}".encode()
                ).hexdigest(),
                "dropout_probes": [{"microbatch": 1, "mask": f"mask-{seed}"}],
                "setup": StageBMatchingTests.setup(seed, shared),
            }
            for index, (capacity, objective) in enumerate(
                (("lora", "joint"), ("lora", "state_only"), ("fullrank", "joint"))
            ):
                checkpoint_identity = hashlib.sha256(
                    f"checkpoint-{capacity}-{objective}-{seed}".encode()
                ).hexdigest()
                checkpoint = {
                    "model_seed": seed,
                    "shared_initialization": copy.deepcopy(shared),
                    "g0_lineage": {
                        "synthetic_capacity": capacity,
                        "synthetic_seed": seed,
                    },
                    "positive_control_lineage": {
                        "synthetic_capacity": capacity,
                        "synthetic_seed": seed,
                    },
                    "branch_authorization_lineage": (
                        None if index == 0 else copy.deepcopy(StageBMatchingTests.AUTHORIZATION)
                    ),
                    "checkpoint_identity_sha256": checkpoint_identity,
                }
                groups[index].append(
                    {
                        "summary_path": (
                            f"synthetic/{capacity}_{objective}_seed{seed}/summary.json"
                        ),
                        "summary_sha256": hashlib.sha256(
                            f"summary-{capacity}-{objective}-{seed}".encode()
                        ).hexdigest(),
                        "receipt_identity_sha256": hashlib.sha256(
                            f"receipt-{capacity}-{objective}-{seed}".encode()
                        ).hexdigest(),
                        "checkpoint": checkpoint,
                        "run": copy.deepcopy(run),
                        "checkpoint_path": (
                            f"synthetic/{capacity}_{objective}_seed{seed}/final"
                        ),
                        "checkpoint_metadata_sha256": hashlib.sha256(
                            f"metadata-{capacity}-{objective}-{seed}".encode()
                        ).hexdigest(),
                        "checkpoint_identity_sha256": checkpoint_identity,
                    }
                )
        return groups

    @staticmethod
    def reopen_g0(entry: dict) -> dict:
        seed = entry["synthetic_seed"]
        return {
            "two_step_gradient_probe": [
                {"dropout_probe": {"step": step, "mask": f"mask-{seed}-{step}"}}
                for step in (1, 2)
            ]
        }

    def validate(self, groups) -> dict:
        with mock.patch.object(
            analysis, "_validate_lineage_entry", side_effect=self.reopen_g0
        ):
            return analysis._stage_b_matching_receipt(
                groups[0], groups[1], groups[2], self.AUTHORIZATION
            )

    def test_exact_three_arm_matching_receipt_passes_and_binds_each_seed(self) -> None:
        result = self.validate(self.manifests())
        self.assertEqual(result["status"], "STAGE_B_MATCHING_VALID")
        self.assertEqual(set(result["per_seed"]), set(map(str, SEEDS)))
        for seed in map(str, SEEDS):
            self.assertEqual(
                set(result["per_seed"][seed]["checkpoint_identities"]),
                {"lora_joint", "lora_state_only", "fullrank_joint"},
            )
            self.assertEqual(
                set(result["per_seed"][seed]["checkpoint_lineages"]),
                {"lora_joint", "fullrank_joint"},
            )

    def test_shared_initialization_order_and_realized_dropout_must_match(self) -> None:
        mutations = (
            ("checkpoint", "shared_initialization", {"receipt_identity_sha256": "bad"}),
            ("run", "data_manifest_sha256", "bad"),
            ("run", "training_order_sha256", "bad"),
            ("run", "dropout_schedule_sha256", "bad"),
            ("run", "dropout_probes", [{"microbatch": 1, "mask": "bad"}]),
        )
        for container, field, value in mutations:
            groups = self.manifests()
            groups[2][0][container][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.validate(groups)

    def test_hardware_environment_and_target_setup_must_match_but_free_memory_may_drift(self) -> None:
        mutations = (
            ("adaptation_targets", ["different.target"]),
            ("adaptation_targets_sha256", "bad"),
            ("tokenizer", {"vocabulary_sha256": "bad"}),
            ("dropout_control", {"nonadapter_training_dropout_modules": 1}),
            ("installed_environment_lock", {"sha256": "bad"}),
            (
                "environment",
                {
                    "python": "3.12.3",
                    "device": {
                        "name": "different GPU",
                        "compute_capability": [9, 0],
                        "free_memory_gib_before_load": 72.0,
                    },
                },
            ),
            (
                "preflight_device",
                {
                    "name": "different GPU",
                    "compute_capability": [9, 0],
                    "free_memory_gib_before_load": 72.0,
                },
            ),
        )
        for field, value in mutations:
            groups = self.manifests()
            groups[2][0]["run"]["setup"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                RuntimeError, "hardware/environment/target setup differs"
            ):
                self.validate(groups)

        free_memory_only = self.manifests()
        free_memory_only[2][0]["run"]["setup"]["environment"]["device"][
            "free_memory_gib_before_load"
        ] = 51.0
        free_memory_only[2][0]["run"]["setup"]["preflight_device"][
            "free_memory_gib_before_load"
        ] = 50.0
        self.assertEqual(
            self.validate(free_memory_only)["status"], "STAGE_B_MATCHING_VALID"
        )

    def test_lora_controls_reuse_g0_positive_control_and_exact_miss_authorization(self) -> None:
        mutations = (
            ("g0_lineage", {"synthetic_capacity": "lora", "synthetic_seed": 999}),
            (
                "positive_control_lineage",
                {"synthetic_capacity": "lora", "synthetic_seed": 999},
            ),
            ("branch_authorization_lineage", {"status": "forged"}),
        )
        for field, value in mutations:
            groups = self.manifests()
            groups[1][0]["checkpoint"][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.validate(groups)

    def test_cross_capacity_g0_must_realize_the_same_dropout_probes(self) -> None:
        groups = self.manifests()

        def mismatched(entry):
            reopened = self.reopen_g0(entry)
            if entry["synthetic_capacity"] == "fullrank":
                reopened["two_step_gradient_probe"][1]["dropout_probe"]["mask"] = "bad"
            return reopened

        with mock.patch.object(
            analysis, "_validate_lineage_entry", side_effect=mismatched
        ), self.assertRaisesRegex(RuntimeError, "dropout schedules differ"):
            analysis._stage_b_matching_receipt(
                groups[0], groups[1], groups[2], self.AUTHORIZATION
            )


class StageCMatchingTests(unittest.TestCase):
    @staticmethod
    def fixtures(*, post_contrast: bool = False):
        stage_b_groups = StageBMatchingTests.manifests()
        with mock.patch.object(
            analysis,
            "_validate_lineage_entry",
            side_effect=StageBMatchingTests.reopen_g0,
        ):
            stage_b_matching = analysis._stage_b_matching_receipt(
                *stage_b_groups, StageBMatchingTests.AUTHORIZATION
            )
        stage_b_status = (
            "STAGE_B_CONTRAST_AUTHORIZED"
            if post_contrast
            else "FULLRANK_STATE_ONLY_REQUIRED"
        )
        stage_b_lineage = {
            "path": "synthetic/stage_b.json",
            "sha256": "5" * 64,
            "receipt_identity_sha256": "6" * 64,
            "status": stage_b_status,
            "phase": "stage_b_seal_analysis",
        }
        stage_b_receipt = {
            "status": stage_b_status,
            "authorization": copy.deepcopy(StageBMatchingTests.AUTHORIZATION),
            "matching": stage_b_matching,
        }
        if post_contrast:
            current_lineage = {
                "path": "synthetic/fullrank_joint.json",
                "sha256": "7" * 64,
                "receipt_identity_sha256": "8" * 64,
                "status": "FULLRANK_STATE_ONLY_REQUIRED",
                "phase": "fullrank_joint_analysis",
            }
            current_receipt = {
                "status": "FULLRANK_STATE_ONLY_REQUIRED",
                "authorization": copy.deepcopy(stage_b_lineage),
            }
        else:
            current_lineage = copy.deepcopy(stage_b_lineage)
            current_receipt = stage_b_receipt

        fullrank_controls = copy.deepcopy(stage_b_groups[2])
        for manifest in fullrank_controls:
            seed = manifest["checkpoint"]["model_seed"]
            manifest["checkpoint"]["branch_authorization_lineage"] = copy.deepcopy(
                current_lineage
            )
            identity = hashlib.sha256(f"fullrank-control-{seed}".encode()).hexdigest()
            manifest["checkpoint"]["checkpoint_identity_sha256"] = identity
            manifest["checkpoint_identity_sha256"] = identity
            manifest["checkpoint_path"] = f"synthetic/fullrank_state_only_seed{seed}/final"

        def reopen(entry):
            if entry == current_lineage:
                return current_receipt
            if entry == stage_b_lineage:
                return stage_b_receipt
            raise AssertionError(f"unexpected lineage reopen: {entry}")

        return (
            stage_b_groups[1],
            stage_b_groups[2],
            fullrank_controls,
            current_lineage,
            reopen,
        )

    def validate(self, fixture) -> dict:
        lora, fullrank_joint, fullrank_control, authorization, reopen = fixture
        with mock.patch.object(
            analysis, "_validate_lineage_entry", side_effect=reopen
        ):
            return analysis._stage_c_matching_receipt(
                lora, fullrank_joint, fullrank_control, authorization
            )

    def test_trigger_miss_and_postcontrast_paths_both_reopen_stage_b(self) -> None:
        for post_contrast in (False, True):
            result = self.validate(self.fixtures(post_contrast=post_contrast))
            with self.subTest(post_contrast=post_contrast):
                self.assertEqual(result["status"], "STAGE_C_MATCHING_VALID")
                self.assertEqual(set(result["per_seed"]), set(map(str, SEEDS)))
                self.assertEqual(
                    result["stage_b_authorization"]["phase"],
                    "stage_b_seal_analysis",
                )

    def test_stage_c_rejects_unmatched_stateonly_or_setup_receipts(self) -> None:
        mutations = (
            ("run", "training_order_sha256", "bad"),
            ("run", "dropout_schedule_sha256", "bad"),
            ("run", "dropout_probes", [{"mask": "bad"}]),
            (
                "checkpoint",
                "shared_initialization",
                {"receipt_identity_sha256": "bad"},
            ),
            ("checkpoint", "g0_lineage", {"bad": True}),
            ("checkpoint", "positive_control_lineage", {"bad": True}),
            ("checkpoint", "branch_authorization_lineage", {"bad": True}),
        )
        for container, field, value in mutations:
            fixture = self.fixtures()
            fixture[2][0][container][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.validate(fixture)

        for field, value in (
            ("adaptation_targets", ["different.target"]),
            (
                "preflight_device",
                {
                    "name": "different GPU",
                    "compute_capability": [9, 0],
                    "free_memory_gib_before_load": 72.0,
                },
            ),
        ):
            fixture = self.fixtures()
            fixture[2][0]["run"]["setup"][field] = value
            with self.subTest(setup_field=field), self.assertRaisesRegex(
                RuntimeError, "Stage-C hardware/environment/target setup differs"
            ):
                self.validate(fixture)

    def test_stage_c_rejects_a_predecessor_evaluation_not_in_stage_b_seal(self) -> None:
        fixture = self.fixtures()
        fixture[1][0]["summary_sha256"] = "bad"
        with self.assertRaisesRegex(RuntimeError, "differs from the Stage-B seal"):
            self.validate(fixture)


class BranchTaxonomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "default.yaml")

    def analyze_with_mocks(
        self,
        phase: str,
        loader,
        *,
        adaptation_effects=None,
        contrast=None,
        matching_receipt=None,
        stage_c_matching=None,
        stage_b_trigger_reopen=None,
        stage_b_receipt_mutator=None,
        contrast_firewall=None,
        contrast_ledger=None,
    ) -> dict:
        adaptation_effects = adaptation_effects or (lambda *_: effect(True))

        def lineage(status, phase):
            return {
                "path": f"synthetic/{phase}-{status}.json",
                "sha256": "3" * 64,
                "receipt_identity_sha256": "4" * 64,
                "status": status,
                "phase": phase,
            }

        def authorization_for(_config, _path, branch):
            branches = {
                analysis.LORA_MISS_BRANCH: (
                    "LORA_JOINT_MISS_CONTROLS_REQUIRED",
                    "lora_joint_analysis",
                ),
                analysis.STAGE_B_CONTRAST_BRANCH: (
                    "STAGE_B_CONTRAST_AUTHORIZED",
                    "stage_b_seal_analysis",
                ),
                analysis.STAGE_B_FULLRANK_MISS_BRANCH: (
                    "FULLRANK_STATE_ONLY_REQUIRED",
                    "stage_b_seal_analysis",
                ),
                analysis.POSTCONTRAST_FULLRANK_MISS_BRANCH: (
                    "FULLRANK_STATE_ONLY_REQUIRED",
                    "fullrank_joint_analysis",
                ),
            }
            return lineage(*branches[branch])

        def lora_control_authorization(_config, _path):
            return lineage(
                "LORA_STATE_ONLY_CONTROL_COMPLETE", "lora_control_analysis"
            )

        def reopen_lineage(entry):
            threshold = float(
                self.config["gates"]["min_final_joint_accuracy_each_seed_depth"]
            )
            if entry.get("status") == "LORA_JOINT_MISS_CONTROLS_REQUIRED":
                bundles, _ = loader(
                    self.config, ROOT / "runs", "lora", "joint", "trigger"
                )
                return {"formation": analysis._formation_summary(bundles, threshold)}
            if entry.get("status") == "LORA_STATE_ONLY_CONTROL_COMPLETE":
                bundles, _ = loader(
                    self.config, ROOT / "runs", "lora", "state_only", "trigger"
                )
                return {
                    "authorization": lineage(
                        "LORA_JOINT_MISS_CONTROLS_REQUIRED", "lora_joint_analysis"
                    ),
                    "formation": analysis._formation_summary(bundles, threshold),
                }
            if entry.get("status") in {
                "STAGE_B_CONTRAST_AUTHORIZED",
                "FULLRANK_STATE_ONLY_REQUIRED",
            }:
                control_bundles, _ = loader(
                    self.config, ROOT / "runs", "lora", "state_only", "trigger"
                )
                joint_bundles, _ = loader(
                    self.config, ROOT / "runs", "lora", "joint", "trigger"
                )
                formation = analysis._formation_summary(control_bundles, threshold)
                receipt = {
                    "lora_joint_formation": analysis._formation_summary(
                        joint_bundles, threshold
                    ),
                    "lora_state_only_formation": formation,
                    "lora_state_only_annotation": (
                        "LORA_CAN_FORM_STATE_STATE_ONLY"
                        if formation["passes"] else None
                    ),
                }
                return (
                    stage_b_receipt_mutator(copy.deepcopy(receipt))
                    if stage_b_receipt_mutator is not None
                    else receipt
                )
            return {"status": entry.get("status"), "authorization": None}

        matching_receipt = matching_receipt or (
            lambda *_: {"status": "STAGE_B_MATCHING_VALID", "per_seed": {}}
        )
        stage_c_matching = stage_c_matching or (
            lambda *_: {"status": "STAGE_C_MATCHING_VALID", "per_seed": {}}
        )
        stage_b_trigger_reopen = stage_b_trigger_reopen or (
            lambda *_args, **kwargs: {
                "status": "STAGE_B_TRIGGER_INPUTS_REOPENED",
                "arm": kwargs["arm"],
            }
        )
        contrast_firewall = contrast_firewall or (
            lambda *_: {"status": "CONTRAST_FIREWALL_UNOPENED", "events": 0}
        )
        contrast_ledger = contrast_ledger or (
            lambda *_: {"status": "CONTRAST_ACCESS_LEDGER_COMPLETE", "events": []}
        )
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory, mock.patch.object(
            analysis, "_load_cell", side_effect=loader
        ), mock.patch.object(
            analysis, "_adaptation_effects", side_effect=adaptation_effects
        ), mock.patch.object(
            analysis, "_load_analysis_authorization", side_effect=authorization_for
        ) as authorization_mock, mock.patch.object(
            analysis,
            "_load_lora_control_analysis",
            side_effect=lora_control_authorization,
        ), mock.patch.object(
            analysis, "_validate_lineage_entry", side_effect=reopen_lineage
        ) as lineage_mock, mock.patch.object(
            analysis,
            "_fullrank_minus_lora_contrast",
            return_value=(
                contrast if contrast is not None else {"passes": True, "splits": {}}
            ),
        ), mock.patch.object(
            analysis, "_stage_b_matching_receipt", side_effect=matching_receipt
        ) as matching_mock, mock.patch.object(
            analysis, "_stage_c_matching_receipt", side_effect=stage_c_matching
        ) as stage_c_mock, mock.patch.object(
            analysis, "_require_stage_b_trigger_match", side_effect=stage_b_trigger_reopen
        ) as trigger_reopen_mock, mock.patch.object(
            analysis, "_contrast_firewall_preopen", side_effect=contrast_firewall
        ) as firewall_mock, mock.patch.object(
            analysis, "_contrast_ledger_guard", side_effect=contrast_ledger
        ) as ledger_mock, mock.patch.object(
            analysis,
            "validate_design_receipt",
            return_value={"status": "DESIGN_FROZEN"},
            create=True,
        ), mock.patch.object(
            analysis,
            "design_lineage",
            return_value={
                "path": "synthetic/design_receipt.json",
                "sha256": "1" * 64,
                "receipt_identity_sha256": "2" * 64,
                "status": "DESIGN_FROZEN",
                "phase": "design_boundary",
            },
            create=True,
        ):
            result = analysis.analyze_phase(
                self.config,
                ROOT / "runs",
                phase,
                Path(directory) / f"{phase}.json",
                authorization_receipt=(
                    None
                    if phase == "lora_joint"
                    else (
                        ROOT / "analysis" / "fullrank_joint.json"
                        if phase == "fullrank_control"
                        else Path("synthetic/authorization.json")
                    )
                ),
            )
            self.last_analysis_mocks = {
                "authorization": authorization_mock,
                "lineage": lineage_mock,
                "matching": matching_mock,
                "stage_c_matching": stage_c_mock,
                "trigger_reopen": trigger_reopen_mock,
                "firewall": firewall_mock,
                "ledger": ledger_mock,
            }
            return result

    @staticmethod
    def one_cell_loader(bundles):
        return lambda *args, **kwargs: (bundles, [{"synthetic": True}])

    def test_stage_a_and_lora_control_incomplete_evidence_are_repair_only(self) -> None:
        incomplete = with_missing_required_cell(
            synthetic_bundles(), split="validation", depth=2
        )
        stage_a = self.analyze_with_mocks(
            "lora_joint", self.one_cell_loader(incomplete)
        )
        self.assertEqual(stage_a["formation"]["status"], "EVIDENCE_INCOMPLETE")
        self.assertEqual(stage_a["status"], "EVIDENCE_INVALID_REPAIR_REQUIRED")
        self.assertEqual(stage_a["verdict"], stage_a["status"])
        self.assertEqual(stage_a["next_stage"], "repair_evidence_only")
        self.assertNotIn(
            stage_a["status"],
            {
                "LORA_DOES_NOT_PREVENT_STATE_FORMATION",
                "LORA_JOINT_MISS_CONTROLS_REQUIRED",
            },
        )

        lora_control = self.analyze_with_mocks(
            "lora_control", self.one_cell_loader(incomplete)
        )
        self.assertEqual(lora_control["status"], "BRANCH_EVIDENCE_INCOMPLETE")
        self.assertEqual(lora_control["next_stage"], "repair_evidence_only")
        self.assertNotEqual(
            lora_control["status"], "LORA_STATE_ONLY_CONTROL_COMPLETE"
        )

    def test_stage_b_and_fullrank_control_incomplete_evidence_never_classify(self) -> None:
        lora_miss = synthetic_bundles(joint_pass=False)
        passed = synthetic_bundles()
        incomplete_fullrank = with_missing_required_cell(
            passed, split="depth_extrapolation", depth=5
        )

        def stage_b_loader(_config, _runs, capacity, objective, _eval_set):
            if (capacity, objective) == ("lora", "joint"):
                bundles = lora_miss
            elif (capacity, objective) == ("fullrank", "joint"):
                bundles = incomplete_fullrank
            else:
                bundles = passed
            return bundles, [{"capacity": capacity, "objective": objective}]

        stage_b = self.analyze_with_mocks("stage_b_seal", stage_b_loader)
        self.assertEqual(stage_b["status"], "BRANCH_EVIDENCE_INCOMPLETE")
        self.assertEqual(stage_b["next_stage"], "repair_evidence_only")
        self.assertNotIn(
            stage_b["status"],
            {"STAGE_B_CONTRAST_AUTHORIZED", "FULLRANK_STATE_ONLY_REQUIRED"},
        )

        incomplete_control = with_missing_required_cell(
            passed, split="joint_holdout", depth=12
        )

        def control_loader(_config, _runs, capacity, objective, _eval_set):
            bundles = (
                incomplete_control
                if (capacity, objective) == ("fullrank", "state_only")
                else passed
            )
            return bundles, [{"capacity": capacity, "objective": objective}]

        control = self.analyze_with_mocks("fullrank_control", control_loader)
        self.assertEqual(control["status"], "BRANCH_EVIDENCE_INCOMPLETE")
        self.assertEqual(control["next_stage"], "repair_evidence_only")
        self.assertNotIn(
            control["status"],
            {
                "DIRECT_FULLSHAPE_RECIPE_STATE_ONLY_RESCUE",
                "BOTH_CAPACITIES_FORM_STATE_WITHOUT_ANSWER",
                "FULLRANK_CONTROL_REVERSAL",
                "FULLRANK_RELIEF_NOT_SUFFICIENT_REGISTERED_RECIPE_"
                "BOTTLENECK_UNRESOLVED",
            },
        )

    def test_incomplete_sealed_evidence_defers_failure_category_replication(self) -> None:
        fullrank_trigger = synthetic_bundles()
        lora_trigger_miss = synthetic_bundles(joint_pass=False)
        incomplete_fullrank_sealed = with_missing_required_cell(
            synthetic_contrast_bundles(),
            split="contrast_validation",
            depth=2,
        )
        lora_sealed_miss = synthetic_contrast_bundles(joint_pass=False)

        def loader(_config, _runs, capacity, objective, eval_set):
            if eval_set == "trigger":
                bundles = (
                    lora_trigger_miss
                    if (capacity, objective) == ("lora", "joint")
                    else fullrank_trigger
                )
            elif capacity == "lora":
                bundles = lora_sealed_miss
            else:
                bundles = incomplete_fullrank_sealed
            return bundles, [{"capacity": capacity, "eval_set": eval_set}]

        with mock.patch.object(
            analysis,
            "_failure_category_replication",
            wraps=analysis._failure_category_replication,
        ) as category_helper:
            result = self.analyze_with_mocks("fullrank_joint", loader)
        category_helper.assert_not_called()
        self.assertEqual(result["status"], "BRANCH_EVIDENCE_INCOMPLETE")
        self.assertEqual(result["next_stage"], "repair_evidence_only")
        self.assertEqual(
            result["lora_trigger_failure_replication"]["status"],
            "EVIDENCE_INCOMPLETE",
        )
        self.assertFalse(result["lora_trigger_failure_replication"]["passes"])
        self.assertNotIn(
            result["status"],
            {
                "DIRECT_FULLSHAPE_RECIPE_RESCUE",
                "DIRECT_FULLSHAPE_RECIPE_PASS_CONTRAST_UNCERTAIN",
                "FULLRANK_STATE_ONLY_REQUIRED",
            },
        )

    def test_lora_joint_pass_stops_and_miss_mandates_both_controls(self) -> None:
        passed = self.analyze_with_mocks(
            "lora_joint", self.one_cell_loader(synthetic_bundles())
        )
        self.assertEqual(passed["verdict"], "LORA_DOES_NOT_PREVENT_STATE_FORMATION")
        self.assertEqual(passed["next_stage"], "stop_capacity_branch")
        self.assertEqual(passed["formation"]["status"], "STATE_FORMATION_PASS")

        missed = self.analyze_with_mocks(
            "lora_joint",
            self.one_cell_loader(synthetic_bundles(trained_pass=False, depth_pass=False, joint_pass=False)),
        )
        self.assertEqual(missed["status"], "LORA_JOINT_MISS_CONTROLS_REQUIRED")
        self.assertEqual(missed["next_stage"], "run_lora_state_only_and_fullrank_joint")
        self.assertEqual(missed["formation"]["status"], "TRAINED_DEPTH_MISS")

    def test_stage_b_seal_is_the_only_contrast_authorization_point(self) -> None:
        passed = synthetic_bundles()
        lora_miss = synthetic_bundles(trained_pass=False)

        def loader(_config, _runs, capacity, objective, eval_set):
            self.assertEqual(eval_set, "trigger")
            self.assertIn(
                (capacity, objective),
                {("lora", "joint"), ("lora", "state_only"), ("fullrank", "joint")},
            )
            bundles = lora_miss if (capacity, objective) == ("lora", "joint") else passed
            return bundles, [{"capacity": capacity, "objective": objective}]

        result = self.analyze_with_mocks("stage_b_seal", loader)
        self.assertEqual(result["status"], "STAGE_B_CONTRAST_AUTHORIZED")
        self.assertEqual(result["phase"], "stage_b_seal_analysis")
        self.assertEqual(result["next_stage"], "evaluate_exact_six_joint_contrast_cells")
        self.assertEqual(result["matching"]["status"], "STAGE_B_MATCHING_VALID")
        self.assertEqual(
            result["contrast_firewall"]["status"], "CONTRAST_FIREWALL_UNOPENED"
        )
        self.last_analysis_mocks["matching"].assert_called_once()
        self.last_analysis_mocks["firewall"].assert_called_once()
        self.last_analysis_mocks["ledger"].assert_not_called()
        self.assertEqual(self.last_analysis_mocks["authorization"].call_count, 1)

    def test_stage_b_trigger_miss_skips_contrast_and_mandates_state_only(self) -> None:
        passed = synthetic_bundles()
        lora_miss = synthetic_bundles(trained_pass=False)
        failed = synthetic_bundles(joint_pass=False)

        def loader(_config, _runs, capacity, objective, _eval_set):
            if (capacity, objective) == ("lora", "joint"):
                bundles = lora_miss
            elif capacity == "fullrank":
                bundles = failed
            else:
                bundles = passed
            return bundles, [{"capacity": capacity}]

        result = self.analyze_with_mocks("stage_b_seal", loader)
        self.assertEqual(result["status"], "FULLRANK_STATE_ONLY_REQUIRED")
        self.assertEqual(result["next_stage"], "run_fullrank_state_only_control")
        self.assertEqual(
            result["fullrank_trigger_formation"]["status"],
            "TRAINED_AND_DEPTH_PASS_JOINT_SHIFT_MISS",
        )
        self.last_analysis_mocks["ledger"].assert_not_called()

    def test_stage_b_never_authorizes_when_matching_or_firewall_is_invalid(self) -> None:
        passed = synthetic_bundles()
        lora_miss = synthetic_bundles(trained_pass=False)

        def loader(_config, _runs, capacity, objective, _eval_set):
            bundles = lora_miss if (capacity, objective) == ("lora", "joint") else passed
            return bundles, [{"capacity": capacity, "objective": objective}]

        def invalid_firewall(*_args):
            raise RuntimeError("sealed access ledger was already opened")

        result = self.analyze_with_mocks(
            "stage_b_seal",
            loader,
            contrast_firewall=invalid_firewall,
        )
        self.assertEqual(result["status"], "CONTRAST_FIREWALL_NOT_READY")
        self.assertEqual(result["next_stage"], "repair_stage_b_evidence_only")
        self.assertEqual(result["matching"]["status"], "INVALID")
        self.assertEqual(result["contrast_firewall"]["status"], "INVALID")

    def test_fullrank_analysis_reopens_exact_contrast_ledger_before_classifying(self) -> None:
        trigger = synthetic_bundles()
        lora_trigger_miss = synthetic_bundles(depth_pass=False, joint_pass=False)
        sealed = synthetic_contrast_bundles()
        lora_sealed_miss = synthetic_contrast_bundles(depth_pass=False, joint_pass=False)

        def loader(_config, _runs, capacity, objective, eval_set):
            if eval_set == "trigger":
                bundles = (
                    lora_trigger_miss
                    if (capacity, objective) == ("lora", "joint")
                    else trigger
                )
            elif capacity == "lora":
                bundles = lora_sealed_miss
            else:
                bundles = sealed
            return bundles, [{"eval_set": eval_set}]

        result = self.analyze_with_mocks("fullrank_joint", loader)
        self.assertEqual(result["status"], "DIRECT_FULLSHAPE_RECIPE_RESCUE")
        self.last_analysis_mocks["ledger"].assert_called_once_with(
            self.config, result["authorization"]
        )
        self.assertEqual(self.last_analysis_mocks["trigger_reopen"].call_count, 3)
        self.assertEqual(
            {
                call.kwargs["arm"]
                for call in self.last_analysis_mocks["trigger_reopen"].call_args_list
            },
            {"fullrank_joint", "lora_joint", "lora_state_only"},
        )
        self.assertTrue(
            all(
                call.kwargs["authorization"] == result["authorization"]
                for call in self.last_analysis_mocks[
                    "trigger_reopen"
                ].call_args_list
            )
        )
        self.assertEqual(result["lora_trigger_reopen"]["arm"], "lora_joint")
        self.assertEqual(
            result["lora_state_only_reopen"]["arm"], "lora_state_only"
        )
        self.assertEqual(
            result["contrast_access_ledger"]["status"],
            "CONTRAST_ACCESS_LEDGER_COMPLETE",
        )

    def test_fullrank_analysis_rejects_cached_lora_trigger_category_tamper(self) -> None:
        trigger = synthetic_bundles()
        lora_trigger_miss = synthetic_bundles(depth_pass=False, joint_pass=False)
        sealed = synthetic_contrast_bundles()
        lora_sealed_miss = synthetic_contrast_bundles(
            depth_pass=False, joint_pass=False
        )

        def loader(_config, _runs, capacity, objective, eval_set):
            if eval_set == "trigger":
                bundles = (
                    lora_trigger_miss
                    if (capacity, objective) == ("lora", "joint")
                    else trigger
                )
            elif capacity == "lora":
                bundles = lora_sealed_miss
            else:
                bundles = sealed
            return bundles, [{"eval_set": eval_set}]

        def tamper_cached_lora_formation(receipt):
            receipt["lora_joint_formation"]["category_passes"]["joint"] = True
            return receipt

        with self.assertRaisesRegex(
            RuntimeError, "reopened LoRA trigger formation differs"
        ):
            self.analyze_with_mocks(
                "fullrank_joint",
                loader,
                stage_b_receipt_mutator=tamper_cached_lora_formation,
            )

    def test_fullrank_joint_rescue_uncertain_and_absolute_miss_branches(self) -> None:
        trigger = synthetic_bundles()
        lora_trigger_miss = synthetic_bundles(depth_pass=False, joint_pass=False)
        sealed = synthetic_contrast_bundles()
        lora_sealed_miss = synthetic_contrast_bundles(depth_pass=False, joint_pass=False)

        def passing_loader(_config, _runs, capacity, objective, eval_set):
            if eval_set == "trigger":
                bundles = (
                    lora_trigger_miss
                    if (capacity, objective) == ("lora", "joint")
                    else trigger
                )
            elif capacity == "lora":
                bundles = lora_sealed_miss
            else:
                bundles = sealed
            return bundles, [{"eval_set": eval_set}]

        rescue = self.analyze_with_mocks(
            "fullrank_joint",
            passing_loader,
        )
        self.assertEqual(rescue["status"], "DIRECT_FULLSHAPE_RECIPE_RESCUE")

        uncertain = self.analyze_with_mocks(
            "fullrank_joint",
            passing_loader,
            contrast={"passes": False, "splits": {}},
        )
        self.assertEqual(
            uncertain["status"],
            "DIRECT_FULLSHAPE_RECIPE_PASS_CONTRAST_UNCERTAIN",
        )

        failed = synthetic_contrast_bundles(joint_pass=False)

        def failed_loader(_config, _runs, capacity, objective, eval_set):
            if eval_set == "trigger":
                bundles = (
                    lora_trigger_miss
                    if (capacity, objective) == ("lora", "joint")
                    else trigger
                )
            elif capacity == "lora":
                bundles = lora_sealed_miss
            else:
                bundles = failed
            return bundles, [{"eval_set": eval_set}]

        miss = self.analyze_with_mocks(
            "fullrank_joint", failed_loader
        )
        self.assertEqual(miss["status"], "FULLRANK_STATE_ONLY_REQUIRED")
        self.assertEqual(miss["next_stage"], "run_fullrank_state_only_control")

    def test_lora_trigger_miss_must_replicate_on_sealed_contrast(self) -> None:
        trigger = synthetic_bundles()
        lora_trigger_miss = synthetic_bundles(depth_pass=False, joint_pass=False)
        lora_sealed_pass = synthetic_contrast_bundles()
        for fullrank_pass in (True, False):
            fullrank_sealed = synthetic_contrast_bundles(
                depth_pass=fullrank_pass, joint_pass=fullrank_pass
            )

            def loader(_config, _runs, capacity, objective, eval_set):
                if eval_set == "trigger":
                    bundles = (
                        lora_trigger_miss
                        if (capacity, objective) == ("lora", "joint")
                        else trigger
                    )
                elif capacity == "lora":
                    bundles = lora_sealed_pass
                else:
                    bundles = fullrank_sealed
                return bundles, [{"capacity": capacity, "eval_set": eval_set}]

            with self.subTest(fullrank_pass=fullrank_pass):
                result = self.analyze_with_mocks("fullrank_joint", loader)
                self.assertEqual(
                    result["status"],
                    "LORA_TRIGGER_MISS_NOT_REPLICATED_ON_SEALED_CONTRAST",
                )
                self.assertEqual(result["verdict"], result["status"])
                self.assertEqual(result["next_stage"], "stop_capacity_branch")
                self.assertTrue(result["lora_sealed_contrast_formation"]["passes"])

    def test_fullrank_rescue_requires_the_same_lora_failure_categories_on_sealed_rows(self) -> None:
        fullrank_trigger = synthetic_bundles()
        fullrank_sealed = synthetic_contrast_bundles()
        lora_trigger_joint_miss = synthetic_bundles(joint_pass=False)

        def run(lora_sealed):
            def loader(_config, _runs, capacity, objective, eval_set):
                if eval_set == "trigger":
                    if (capacity, objective) == ("lora", "joint"):
                        bundles = lora_trigger_joint_miss
                    else:
                        bundles = fullrank_trigger
                elif capacity == "lora":
                    bundles = lora_sealed
                else:
                    bundles = fullrank_sealed
                return bundles, [{"capacity": capacity, "eval_set": eval_set}]

            return self.analyze_with_mocks("fullrank_joint", loader)

        trained_only_sealed_miss = synthetic_contrast_bundles(
            trained_pass=False, depth_pass=True, joint_pass=True
        )
        rejected = run(trained_only_sealed_miss)
        self.assertEqual(
            rejected["status"],
            "LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_ON_SEALED_CONTRAST",
        )
        self.assertNotEqual(rejected["status"], "DIRECT_FULLSHAPE_RECIPE_RESCUE")
        self.assertEqual(
            rejected["lora_trigger_failure_replication"]["missing_replications"],
            ["joint"],
        )

        joint_plus_trained_sealed_miss = synthetic_contrast_bundles(
            trained_pass=False, depth_pass=True, joint_pass=False
        )
        rescue = run(joint_plus_trained_sealed_miss)
        self.assertTrue(rescue["lora_trigger_failure_replication"]["passes"])
        self.assertEqual(
            rescue["lora_trigger_failure_replication"]["sealed_failed_categories"],
            ["trained", "joint"],
        )
        self.assertEqual(rescue["status"], "DIRECT_FULLSHAPE_RECIPE_RESCUE")

    def test_state_only_pair_has_all_four_preregistered_terminal_outcomes(self) -> None:
        pass_bundle = synthetic_bundles()
        fail_bundle = synthetic_bundles(trained_pass=False)
        cases = (
            (True, False, "DIRECT_FULLSHAPE_RECIPE_STATE_ONLY_RESCUE"),
            (True, True, "BOTH_CAPACITIES_FORM_STATE_WITHOUT_ANSWER"),
            (False, True, "FULLRANK_CONTROL_REVERSAL"),
            (
                False,
                False,
                "FULLRANK_RELIEF_NOT_SUFFICIENT_REGISTERED_RECIPE_BOTTLENECK_UNRESOLVED",
            ),
        )
        for fullrank_pass, lora_pass, expected in cases:
            def loader(_config, _runs, capacity, _objective, _eval_set):
                selected = fullrank_pass if capacity == "fullrank" else lora_pass
                return (pass_bundle if selected else fail_bundle), [{"capacity": capacity}]

            with self.subTest(expected=expected):
                result = self.analyze_with_mocks("fullrank_control", loader)
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["next_stage"], "stop_capacity_branch")
                self.assertEqual(result["matching"]["status"], "STAGE_C_MATCHING_VALID")
                self.last_analysis_mocks["stage_c_matching"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
