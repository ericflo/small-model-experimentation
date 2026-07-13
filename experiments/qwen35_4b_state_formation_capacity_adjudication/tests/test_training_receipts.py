from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training_receipts import (  # noqa: E402
    EXPERIMENT_ID,
    RUN_IDENTITY_FIELDS,
    STAGE_A_MATRIX,
    STAGE_B_MATRIX,
    STAGE_C_MATRIX,
    TrainingCell,
    TrainingCellState,
    TrainingReceiptContract,
    TrainingReceiptError,
    canonical_training_cell_paths,
    classify_training_cell,
    evaluation_barrier,
    recover_published_training_completion,
    stage_matrix,
    training_barrier,
    training_launch_preflight,
)
from src.gate_receipts import stable_setup_receipt  # noqa: E402
from src.attempt_receipts import (  # noqa: E402
    AttemptReceiptError,
    complete_training_attempt,
    ensure_attempt_output,
    load_training_journal,
    prepare_training_attempt,
    start_training_attempt,
    validate_training_attempt_history,
)


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _lineage(label: str) -> dict:
    return {
        "path": f"receipts/{label}.json",
        "sha256": _digest(f"{label}:bytes"),
        "receipt_identity_sha256": _digest(f"{label}:identity"),
        "status": f"{label.upper()}_PASS",
        "phase": label,
    }


class Fixture:
    def __init__(self, repo: Path, *, steps: int = 2) -> None:
        self.repo = repo
        self.steps = steps

    def identity(self, cell: TrainingCell) -> dict:
        identity = {
            "experiment_id": EXPERIMENT_ID,
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "revision",
            "backend": "transformers",
            "config_sha256": _digest("config"),
            "source_contract_sha256": _digest("source"),
            "requirements_training_lock_sha256": _digest("lock"),
            "design_receipt_sha256": _digest("design-bytes"),
            "design_receipt_identity_sha256": _digest("design-identity"),
            "phase": cell.phase,
        }
        assert set(identity) == RUN_IDENTITY_FIELDS
        return identity

    def contract(self, cell: TrainingCell) -> TrainingReceiptContract:
        return TrainingReceiptContract(
            schema_version=1,
            status="TRAINING_COMPLETE",
            identity=self.identity(cell),
            steps=self.steps,
        )

    def contracts(self, stage: str) -> dict[str, TrainingReceiptContract]:
        return {cell.slug: self.contract(cell) for cell in stage_matrix(stage)}

    def synthetic_cell_proof(
        self, cell: TrainingCell, *, branch: dict | None = None
    ) -> dict:
        digest_fields = {
            name: _digest(f"{cell.slug}:{name}")
            for name in (
                "run_sha256", "receipt_identity_sha256", "train_metrics_sha256",
                "optimizer_steps_sha256", "checkpoint_metadata_sha256",
                "checkpoint_identity_sha256", "adaptation_state_sha256",
                "loop_state_sha256", "stable_setup_sha256",
                "setup_barrier_identity_sha256",
                "training_launch_preflight_identity_sha256",
                "training_attempt_identity_sha256",
                "training_attempt_history_identity_sha256",
            )
        }
        return {
            "cell": cell.slug,
            "stage": cell.stage,
            "capacity": cell.capacity,
            "objective": cell.objective,
            "seed": cell.seed,
            "steps": self.steps,
            "run_path": f"large/{cell.slug}/run.json",
            "tracked_run_path": f"tracked/{cell.slug}/run.json",
            "checkpoint_path": f"large/{cell.slug}/checkpoint",
            "setup_lineage": {"seed": cell.seed},
            "gate_lineages": {},
            "branch_authorization_lineage": branch,
            "prior_training_barrier_identity_sha256s": [],
            **digest_fields,
        }

    def synthetic_training_barrier(
        self, stage: str, *, branch: dict | None
    ) -> dict:
        stage_branch = None if stage == "A" else (branch or _lineage(f"{stage}-branch"))
        proof = {
            "schema_version": 1,
            "status": "TRAINING_BARRIER_COMPLETE",
            "stage": stage,
            "cells": [
                self.synthetic_cell_proof(cell, branch=stage_branch)
                for cell in stage_matrix(stage)
            ],
            "branch_authorization_lineage": stage_branch,
        }
        proof["barrier_identity_sha256"] = _canonical_sha256(proof)
        return proof

    def complete(
        self,
        cell: TrainingCell,
        *,
        branch: dict | None = None,
        journal_complete: bool = True,
        replay_archive: dict | None = None,
    ) -> None:
        contract = self.contract(cell)
        paths = canonical_training_cell_paths(
            self.repo, cell, steps=contract.steps
        )
        # Build the exact embedded barrier/preflight shapes before authorizing
        # the durable marker-only attempt output.
        paths.tracked_dir.mkdir(parents=True)
        # Remove the tracked directory until after PREPARED/STARTED, since a
        # first attempt requires both canonical result paths absent.
        paths.tracked_dir.rmdir()

        device = {
            "name": "synthetic",
            "free_memory_gib_before_load": 42.0,
        }
        setup = {
            "capacity": cell.capacity,
            "model_seed": cell.seed,
            "tokenizer": {"state_token_id": 1},
            "adaptation_targets": ["target"],
            "adaptation_targets_sha256": _digest("targets"),
            "adaptation_target_manifest": [{"target": "target"}],
            "adaptation_target_manifest_sha256": _digest("target-manifest"),
            "adaptation_parameters": 1,
            "adaptation_zero_function": {"nonzero_output_weights": 0},
            "shared_initialization": {
                "status": "SHARED_INITIALIZATION_PREPARED",
                "seed": cell.seed,
                "receipt_identity_sha256": _digest(f"init:{cell.seed}"),
            },
            "trainable_parameters": {"total": 2},
            "dropout_control": {"matched_adaptation_dropout": 0.05},
            "environment": {"device": dict(device)},
            "installed_environment_lock": {"packages": {}},
            "preflight_device": dict(device),
        }
        g0_lineage = _lineage(f"g0-{cell.capacity}-{cell.seed}")
        control_lineage = _lineage(f"control-{cell.capacity}-{cell.seed}")
        stable_setup = stable_setup_receipt(setup)
        setup_barrier = {
            "schema_version": 1,
            "status": "SETUP_BARRIER_COMPLETE",
            "stage": cell.stage,
            "cells": [],
            "root_lora_miss_lineage": branch if cell.stage != "A" else None,
            "common_setup_invariant_sha256": _digest(f"{cell.stage}:common"),
            "capacity_setup_invariant_sha256s": {
                capacity: _digest(f"{cell.stage}:{capacity}")
                for capacity in (("lora",) if cell.stage == "A" else ("lora", "fullrank"))
            },
        }
        for setup_capacity in (("lora",) if cell.stage == "A" else ("lora", "fullrank")):
            for setup_seed in (7411, 7412, 7413):
                is_target = setup_capacity == cell.capacity and setup_seed == cell.seed
                setup_barrier["cells"].append(
                    {
                        "cell": f"{setup_capacity}_seed{setup_seed}",
                        "g0_lineage": g0_lineage if is_target else _lineage(
                            f"g0-{setup_capacity}-{setup_seed}"
                        ),
                        "positive_control_lineage": control_lineage if is_target else _lineage(
                            f"control-{setup_capacity}-{setup_seed}"
                        ),
                        "stable_setup_sha256": (
                            _canonical_sha256(stable_setup)
                            if is_target else _digest(f"stable-{setup_capacity}-{setup_seed}")
                        ),
                    }
                )
        setup_barrier["barrier_identity_sha256"] = _canonical_sha256(setup_barrier)
        prior_training_barriers = []
        for prior_stage in {"A": (), "B": ("A",), "C": ("A", "B")}[cell.stage]:
            prior_training_barriers.append(
                self.synthetic_training_barrier(prior_stage, branch=branch)
            )
        matrix = stage_matrix(cell.stage)
        target_index = matrix.index(cell)
        peer_rows = [
            {
                "cell": peer.slug,
                "state": "COMPLETE" if index < target_index else "ABSENT",
            }
            for index, peer in enumerate(matrix)
            if peer != cell
        ]
        launch = {
            "schema_version": 1,
            "status": "TRAINING_LAUNCH_PREFLIGHT_PASS",
            "stage": cell.stage,
            "target": cell.slug,
            "target_state": "ABSENT",
            "peers": peer_rows,
            "completed_peer_proofs": [
                self.synthetic_cell_proof(peer, branch=branch)
                for index, peer in enumerate(matrix)
                if index < target_index
            ],
            "trigger_outputs_absent": True,
            "branch_authorization_lineage": branch,
        }
        launch["preflight_identity_sha256"] = _canonical_sha256(launch)
        attempt_header = {
            key: value for key, value in contract.identity.items() if key != "phase"
        }
        attempt_cell = {
            "stage": cell.stage,
            "capacity": cell.capacity,
            "objective": cell.objective,
            "seed": cell.seed,
            "slug": cell.slug,
        }
        attempt_paths = [
            paths.external_dir.relative_to(self.repo).as_posix(),
            paths.tracked_dir.relative_to(self.repo).as_posix(),
        ]
        attempt_context = {
            "setup_barrier_identity_sha256": setup_barrier[
                "barrier_identity_sha256"
            ],
            "prior_training_barrier_identity_sha256s": [
                proof["barrier_identity_sha256"] for proof in prior_training_barriers
            ],
            "training_launch_preflight_identity_sha256": launch[
                "preflight_identity_sha256"
            ],
            "training_launch_peer_vector": peer_rows,
            "branch_authorization_lineage": branch,
        }
        attempt_authorization = prepare_training_attempt(
            self.repo,
            slug=cell.slug,
            header=attempt_header,
            cell=attempt_cell,
            canonical_paths=attempt_paths,
            context=attempt_context,
            replay_archive=replay_archive,
        )
        ensure_attempt_output(paths.external_dir, attempt_authorization)
        start_training_attempt(
            self.repo,
            slug=cell.slug,
            header=attempt_header,
            cell=attempt_cell,
            canonical_paths=attempt_paths,
            authorization=attempt_authorization,
        )
        attempt_history = validate_training_attempt_history(
            self.repo,
            slug=cell.slug,
            header=attempt_header,
            cell=attempt_cell,
            canonical_paths=attempt_paths,
            current_authorization=attempt_authorization,
            expected_archive_header={},
        )
        paths.tracked_dir.mkdir(parents=True)
        paths.checkpoint_dir.mkdir()
        paths.tracked_attempt_marker.write_bytes(paths.external_attempt_marker.read_bytes())
        metrics = b'{"step":1}\n{"step":2}\n'
        optimizer = b'{"step":1}\n{"step":2}\n'
        paths.external_metrics.write_bytes(metrics)
        paths.tracked_metrics.write_bytes(metrics)
        paths.external_optimizer_steps.write_bytes(optimizer)
        paths.tracked_optimizer_steps.write_bytes(optimizer)
        paths.adaptation_state.write_bytes(f"adaptation:{cell.slug}".encode())
        paths.loop_state.write_bytes(f"loop:{cell.slug}".encode())
        common = {
            "schema_version": contract.schema_version,
            **dict(contract.identity),
            "capacity": cell.capacity,
            "objective": cell.objective,
            "model_seed": cell.seed,
            "data_manifest_sha256": _digest("data"),
            "training_prompt_tokens": 100,
            "training_layer_token_applications": 200,
            "training_order_sha256": _digest(f"order:{cell.seed}"),
            "dropout_schedule_sha256": _digest(f"dropout:{cell.seed}"),
            "dropout_probes": [],
            "train_metrics_sha256": _sha256(paths.external_metrics),
            "train_metrics_rows": self.steps,
            "train_metrics_path": paths.external_metrics.relative_to(
                self.repo
            ).as_posix(),
            "optimizer_steps_sha256": _sha256(paths.external_optimizer_steps),
            "optimizer_steps_rows": self.steps,
            "optimizer_steps_path": paths.external_optimizer_steps.relative_to(
                self.repo
            ).as_posix(),
            "optimizer_state": {"all_required_states_complete": True},
            "optimizer_step_receipt": {
                "steps": self.steps,
                "rows": self.steps,
            },
            "setup_barrier": setup_barrier,
            "prior_training_barriers": prior_training_barriers,
            "training_launch_preflight": launch,
            "setup": setup,
            "setup_sha256": _canonical_sha256(setup),
            "stable_setup": stable_setup,
            "training_attempt_authorization": attempt_authorization,
            "training_attempt_history": attempt_history,
            "training_attempt_journal_path": paths.attempt_journal.relative_to(
                self.repo
            ).as_posix(),
        }
        metadata = {
            **common,
            "step": self.steps,
            "adaptation_state_sha256": _sha256(paths.adaptation_state),
            "loop_state_sha256": _sha256(paths.loop_state),
            "g0_lineage": g0_lineage,
            "positive_control_lineage": control_lineage,
            "branch_authorization_lineage": branch,
        }
        metadata["checkpoint_identity_sha256"] = _canonical_sha256(metadata)
        _write_json(paths.checkpoint_metadata, metadata)

        run = {
            **common,
            "status": contract.status,
            "steps": self.steps,
            "g0_lineage": metadata["g0_lineage"],
            "positive_control_lineage": metadata["positive_control_lineage"],
            "branch_authorization_lineage": branch,
            "checkpoint_path": paths.checkpoint_dir.relative_to(
                self.repo
            ).as_posix(),
            "checkpoint_metadata_sha256": _sha256(paths.checkpoint_metadata),
            "checkpoint_identity_sha256": metadata[
                "checkpoint_identity_sha256"
            ],
            "tracked_run_path": paths.tracked_run.relative_to(self.repo).as_posix(),
            "tracked_metrics_path": paths.tracked_metrics.relative_to(
                self.repo
            ).as_posix(),
            "tracked_optimizer_steps_path": paths.tracked_optimizer_steps.relative_to(
                self.repo
            ).as_posix(),
            "authorizes_training": False,
            "authorizes_result_training": False,
            "authorizes_result_evaluation": False,
            "benchmark_files_read": 0,
            "result_payloads_opened": ["train"],
            "sealed_contrast_payloads_opened": [],
            "training_or_evaluation_started": True,
            "scientific_evidence": False,
        }
        run["receipt_identity_sha256"] = _canonical_sha256(run)
        _write_json(paths.external_run, run)
        paths.tracked_run.write_bytes(paths.external_run.read_bytes())
        if journal_complete:
            complete_training_attempt(
                self.repo,
                slug=cell.slug,
                header=attempt_header,
                cell=attempt_cell,
                canonical_paths=attempt_paths,
                authorization=attempt_authorization,
                terminal_run_lineage={
                    "path": paths.external_run.relative_to(self.repo).as_posix(),
                    "sha256": _sha256(paths.external_run),
                    "receipt_identity_sha256": run["receipt_identity_sha256"],
                },
            )


class TrainingReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        self.fixture = Fixture(self.repo)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_canonical_matrices_and_paths_are_exact(self) -> None:
        self.assertEqual(
            [cell.slug for cell in STAGE_A_MATRIX],
            [f"lora_joint_seed{seed}" for seed in (7411, 7412, 7413)],
        )
        self.assertEqual(len(STAGE_B_MATRIX), 6)
        self.assertEqual(
            [cell.slug for cell in STAGE_B_MATRIX],
            [
                f"{capacity}_{objective}_seed{seed}"
                for seed in (7411, 7412, 7413)
                for capacity, objective in (
                    ("lora", "state_only"),
                    ("fullrank", "joint"),
                )
            ],
        )
        self.assertEqual(
            [cell.slug for cell in STAGE_C_MATRIX],
            [f"fullrank_state_only_seed{seed}" for seed in (7411, 7412, 7413)],
        )
        cell = STAGE_A_MATRIX[0]
        paths = canonical_training_cell_paths(self.repo, cell, steps=2)
        self.assertEqual(
            paths.external_run.relative_to(self.repo).as_posix(),
            f"large_artifacts/{EXPERIMENT_ID}/{cell.slug}/run.json",
        )
        self.assertEqual(
            paths.tracked_run.relative_to(self.repo).as_posix(),
            f"experiments/{EXPERIMENT_ID}/runs/training/{cell.slug}/run.json",
        )
        self.assertEqual(paths.checkpoint_dir.name, "checkpoint_000002")
        self.assertEqual(
            paths.trigger_output.name, f"{cell.slug}_trigger"
        )

    def _prepared_training_authorization(self) -> tuple[Path, dict]:
        cell = STAGE_A_MATRIX[0]
        paths = canonical_training_cell_paths(self.repo, cell, steps=2)
        canonical_paths = [
            paths.external_dir.relative_to(self.repo).as_posix(),
            paths.tracked_dir.relative_to(self.repo).as_posix(),
        ]
        authorization = prepare_training_attempt(
            self.repo,
            slug=cell.slug,
            header={"experiment_id": EXPERIMENT_ID, "test": "prepared-output"},
            cell={
                "stage": cell.stage,
                "capacity": cell.capacity,
                "objective": cell.objective,
                "seed": cell.seed,
                "slug": cell.slug,
            },
            canonical_paths=canonical_paths,
            context={"test": "prepared-output"},
            replay_archive=None,
        )
        return paths.external_dir, authorization

    def test_relative_prepared_output_publishes_only_the_canonical_leaf(self) -> None:
        output, authorization = self._prepared_training_authorization()
        relative = output.relative_to(self.repo)
        with contextlib.chdir(self.repo):
            marker = ensure_attempt_output(relative, authorization)
        self.assertEqual(
            marker["attempt_identity_sha256"],
            authorization["attempt_identity_sha256"],
        )
        self.assertTrue((output / "attempt.json").is_file())
        duplicated = output.parent / relative / "attempt.json"
        self.assertFalse(duplicated.exists())

    def test_empty_prepared_output_is_exact_mkdir_crash_recovery(self) -> None:
        output, authorization = self._prepared_training_authorization()
        output.mkdir(parents=True)
        marker = ensure_attempt_output(output, authorization)
        self.assertEqual(set(item.name for item in output.iterdir()), {"attempt.json"})
        self.assertEqual(ensure_attempt_output(output, authorization), marker)

    def test_nonempty_markerless_prepared_output_remains_fatal(self) -> None:
        output, authorization = self._prepared_training_authorization()
        output.mkdir(parents=True)
        (output / "unknown.bin").write_bytes(b"not a marker")
        with self.assertRaisesRegex(
            AttemptReceiptError, "PREPARED attempt output is not marker-only"
        ):
            ensure_attempt_output(output, authorization)

    def test_absent_and_complete_classification_with_lineage_passthrough(self) -> None:
        cell = STAGE_A_MATRIX[0]
        contract = self.fixture.contract(cell)
        absent = classify_training_cell(self.repo, cell, contract)
        self.assertIs(absent.state, TrainingCellState.ABSENT)

        self.fixture.complete(cell)
        complete = classify_training_cell(self.repo, cell, contract)
        self.assertIs(complete.state, TrainingCellState.COMPLETE)
        self.assertEqual(complete.proof["steps"], 2)
        self.assertEqual(
            complete.setup_lineage["seed"], cell.seed
        )
        self.assertEqual(
            set(complete.gate_lineages),
            {"g0_lineage", "positive_control_lineage"},
        )
        self.assertIsNone(complete.branch_authorization_lineage)
        self.assertNotIn("torch", classify_training_cell.__globals__)

    def test_published_terminal_receipts_require_and_recover_exact_journal_completion(
        self,
    ) -> None:
        cell = STAGE_A_MATRIX[0]
        contract = self.fixture.contract(cell)
        self.fixture.complete(cell, journal_complete=False)
        paths = canonical_training_cell_paths(
            self.repo, cell, steps=contract.steps
        )
        run = json.loads(paths.external_run.read_text(encoding="utf-8"))
        history_before = copy.deepcopy(run["training_attempt_history"])

        strict = classify_training_cell(self.repo, cell, contract)
        self.assertIs(strict.state, TrainingCellState.INCOMPLETE)
        self.assertTrue(
            any("published before journal completion" in error for error in strict.errors),
            strict.errors,
        )
        provisional = classify_training_cell(
            self.repo, cell, contract, allow_started_terminal=True
        )
        self.assertIs(provisional.state, TrainingCellState.COMPLETE)

        self.assertTrue(
            recover_published_training_completion(self.repo, cell, contract)
        )
        complete = classify_training_cell(self.repo, cell, contract)
        self.assertIs(complete.state, TrainingCellState.COMPLETE)
        self.assertFalse(
            recover_published_training_completion(self.repo, cell, contract),
            "an already COMPLETE journal must not be mutated or reported recovered",
        )
        attempt_header = {
            key: value for key, value in contract.identity.items() if key != "phase"
        }
        journal = load_training_journal(
            self.repo,
            cell.slug,
            header=attempt_header,
            cell={
                "stage": cell.stage,
                "capacity": cell.capacity,
                "objective": cell.objective,
                "seed": cell.seed,
                "slug": cell.slug,
            },
            canonical_paths=[
                paths.external_dir.relative_to(self.repo).as_posix(),
                paths.tracked_dir.relative_to(self.repo).as_posix(),
            ],
        )
        self.assertIsNotNone(journal)
        self.assertEqual(journal["events"][-1]["state"], "COMPLETE")
        self.assertEqual(
            json.loads(paths.external_run.read_text(encoding="utf-8"))[
                "training_attempt_history"
            ],
            history_before,
            "STARTED-to-COMPLETE normalization must not rewrite scientific receipts",
        )

    def test_fully_rehashed_journal_cannot_make_complete_or_prepared_history(self) -> None:
        header = {"experiment_id": EXPERIMENT_ID, "test": "history-state"}
        cell = {
            "stage": "A",
            "capacity": "lora",
            "objective": "joint",
            "seed": 7411,
            "slug": "lora_joint_seed7411",
        }
        external = self.repo / "large_artifacts" / EXPERIMENT_ID / cell["slug"]
        tracked = (
            self.repo
            / "experiments"
            / EXPERIMENT_ID
            / "runs"
            / "training"
            / cell["slug"]
        )
        canonical_paths = [
            external.relative_to(self.repo).as_posix(),
            tracked.relative_to(self.repo).as_posix(),
        ]
        first = prepare_training_attempt(
            self.repo,
            slug=cell["slug"],
            header=header,
            cell=cell,
            canonical_paths=canonical_paths,
            context={"test": "history-state"},
            replay_archive=None,
        )
        ensure_attempt_output(external, first)
        start_training_attempt(
            self.repo,
            slug=cell["slug"],
            header=header,
            cell=cell,
            canonical_paths=canonical_paths,
            authorization=first,
        )
        shutil.rmtree(external)
        prepare_training_attempt(
            self.repo,
            slug=cell["slug"],
            header=header,
            cell=cell,
            canonical_paths=canonical_paths,
            context={"test": "history-state"},
            replay_archive={
                "attempt_identity_sha256": first["attempt_identity_sha256"]
            },
        )
        journal_path = (
            self.repo
            / "experiments"
            / EXPERIMENT_ID
            / "runs"
            / "attempts"
            / "training"
            / f"{cell['slug']}.json"
        )
        for forged_state in ("PREPARED", "COMPLETE"):
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            journal["events"][0]["state"] = forged_state
            journal["events"][0]["terminal_run_lineage"] = (
                {"forged": True} if forged_state == "COMPLETE" else None
            )
            journal.pop("receipt_identity_sha256")
            journal["receipt_identity_sha256"] = _canonical_sha256(journal)
            _write_json(journal_path, journal)
            with self.subTest(forged_state=forged_state), self.assertRaisesRegex(
                RuntimeError, "historical training attempt was not a STARTED crash"
            ):
                load_training_journal(
                    self.repo,
                    cell["slug"],
                    header=header,
                    cell=cell,
                    canonical_paths=canonical_paths,
                )
            # Restore the valid STARTED history before exercising the other
            # fully rehashed forbidden state.
            journal["events"][0]["state"] = "STARTED"
            journal["events"][0]["terminal_run_lineage"] = None
            journal.pop("receipt_identity_sha256")
            journal["receipt_identity_sha256"] = _canonical_sha256(journal)
            _write_json(journal_path, journal)
    def test_external_only_and_tracked_only_are_incomplete(self) -> None:
        cell = STAGE_A_MATRIX[0]
        contract = self.fixture.contract(cell)
        self.fixture.complete(cell)
        paths = canonical_training_cell_paths(self.repo, cell, steps=2)
        shutil.rmtree(paths.tracked_dir)
        self.assertIs(
            classify_training_cell(self.repo, cell, contract).state,
            TrainingCellState.INCOMPLETE,
        )

        shutil.rmtree(paths.external_dir)
        paths.tracked_dir.mkdir(parents=True)
        paths.tracked_run.write_text("{}\n", encoding="utf-8")
        self.assertIs(
            classify_training_cell(self.repo, cell, contract).state,
            TrainingCellState.INCOMPLETE,
        )

    def test_each_mirror_mismatch_is_incomplete(self) -> None:
        for attribute in (
            "tracked_run",
            "tracked_metrics",
            "tracked_optimizer_steps",
        ):
            with self.subTest(attribute=attribute), tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                fixture = Fixture(repo)
                cell = STAGE_A_MATRIX[0]
                fixture.complete(cell)
                path = getattr(
                    canonical_training_cell_paths(repo, cell, steps=2), attribute
                )
                path.write_bytes(path.read_bytes() + b"mismatch\n")
                self.assertIs(
                    classify_training_cell(repo, cell, fixture.contract(cell)).state,
                    TrainingCellState.INCOMPLETE,
                )

    def test_symlinks_hardlink_aliases_and_alias_root_fail_closed(self) -> None:
        cell = STAGE_A_MATRIX[0]
        self.fixture.complete(cell)
        paths = canonical_training_cell_paths(self.repo, cell, steps=2)
        paths.tracked_metrics.unlink()
        paths.tracked_metrics.symlink_to(paths.external_metrics)
        audit = classify_training_cell(self.repo, cell, self.fixture.contract(cell))
        self.assertIs(audit.state, TrainingCellState.INCOMPLETE)
        self.assertIn("symlink", audit.errors[0])

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            fixture = Fixture(repo)
            fixture.complete(cell)
            pair = canonical_training_cell_paths(repo, cell, steps=2)
            pair.tracked_metrics.unlink()
            os.link(pair.external_metrics, pair.tracked_metrics)
            audit = classify_training_cell(repo, cell, fixture.contract(cell))
            self.assertIs(audit.state, TrainingCellState.INCOMPLETE)
            self.assertRegex(audit.errors[0], "hardlink alias|inode alias")

            alias = repo.parent / f"{repo.name}-alias"
            alias.symlink_to(repo, target_is_directory=True)
            try:
                audit = classify_training_cell(alias, cell, fixture.contract(cell))
                self.assertIs(audit.state, TrainingCellState.INCOMPLETE)
            finally:
                alias.unlink()

    def test_terminal_snapshot_rejects_atomic_replacement_at_read_boundary(self) -> None:
        from src import training_receipts as receipts

        cell = STAGE_A_MATRIX[0]
        self.fixture.complete(cell)
        paths = canonical_training_cell_paths(self.repo, cell, steps=2)
        replacement = self.repo / "replacement-run.json"
        replacement.write_bytes(paths.external_run.read_bytes())
        original = receipts.open_stable_regular
        injected = False

        @contextlib.contextmanager
        def replacing(root, path, *args, **kwargs):
            nonlocal injected
            with original(root, path, *args, **kwargs) as handle:
                yield handle
                if Path(path) == paths.external_run and not injected:
                    injected = True
                    os.replace(replacement, paths.external_run)

        with mock.patch.object(receipts, "open_stable_regular", replacing):
            audit = classify_training_cell(
                self.repo, cell, self.fixture.contract(cell)
            )
        self.assertTrue(injected)
        self.assertIs(audit.state, TrainingCellState.INCOMPLETE)
        self.assertRegex(
            audit.errors[0], "canonical path changed|inode changed while it was consumed"
        )

    def test_missing_required_sibling_is_always_incomplete(self) -> None:
        for attribute in (
            "external_run",
            "tracked_run",
            "external_metrics",
            "tracked_metrics",
            "external_optimizer_steps",
            "tracked_optimizer_steps",
            "checkpoint_metadata",
            "adaptation_state",
            "loop_state",
        ):
            with self.subTest(attribute=attribute), tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                fixture = Fixture(repo)
                cell = STAGE_A_MATRIX[0]
                fixture.complete(cell)
                getattr(
                    canonical_training_cell_paths(repo, cell, steps=2), attribute
                ).unlink()
                self.assertIs(
                    classify_training_cell(repo, cell, fixture.contract(cell)).state,
                    TrainingCellState.INCOMPLETE,
                )

    def test_receipt_contract_cell_paths_hashes_and_core_agreement_fail_closed(self) -> None:
        mutations = (
            ("schema", lambda run: run.__setitem__("schema_version", 2)),
            ("status", lambda run: run.__setitem__("status", "RUNNING")),
            ("identity", lambda run: run.__setitem__("model_revision", "wrong")),
            ("cell", lambda run: run.__setitem__("model_seed", 9999)),
            ("steps", lambda run: run.__setitem__("steps", 1)),
            ("path", lambda run: run.__setitem__("train_metrics_path", "alias")),
            ("self_hash", lambda run: run.__setitem__("receipt_identity_sha256", "0" * 64)),
        )
        for label, mutation in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                fixture = Fixture(repo)
                cell = STAGE_A_MATRIX[0]
                fixture.complete(cell)
                paths = canonical_training_cell_paths(repo, cell, steps=2)
                run = json.loads(paths.external_run.read_text())
                mutation(run)
                _write_json(paths.external_run, run)
                paths.tracked_run.write_bytes(paths.external_run.read_bytes())
                self.assertIs(
                    classify_training_cell(repo, cell, fixture.contract(cell)).state,
                    TrainingCellState.INCOMPLETE,
                )

    def test_fully_rehashed_access_setup_and_embedded_proof_forgeries_fail_closed(self) -> None:
        mutations = {
            "terminal_access": lambda payload: payload.__setitem__(
                "authorizes_result_evaluation", True
            ),
            "stable_setup": lambda payload: payload.__setitem__(
                "stable_setup", {"forged": True}
            ),
            "setup_barrier": lambda payload: payload["setup_barrier"].__setitem__(
                "stage", "C"
            ),
            "launch_preflight": lambda payload: payload[
                "training_launch_preflight"
            ].__setitem__("target_state", "COMPLETE"),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                fixture = Fixture(repo)
                cell = STAGE_A_MATRIX[0]
                fixture.complete(cell)
                paths = canonical_training_cell_paths(repo, cell, steps=2)
                metadata = json.loads(paths.checkpoint_metadata.read_text())
                run = json.loads(paths.external_run.read_text())
                mutation(metadata)
                mutation(run)
                metadata.pop("checkpoint_identity_sha256")
                metadata["checkpoint_identity_sha256"] = _canonical_sha256(metadata)
                _write_json(paths.checkpoint_metadata, metadata)
                run["checkpoint_metadata_sha256"] = _sha256(
                    paths.checkpoint_metadata
                )
                run["checkpoint_identity_sha256"] = metadata[
                    "checkpoint_identity_sha256"
                ]
                run.pop("receipt_identity_sha256")
                run["receipt_identity_sha256"] = _canonical_sha256(run)
                _write_json(paths.external_run, run)
                paths.tracked_run.write_bytes(paths.external_run.read_bytes())
                audit = classify_training_cell(repo, cell, fixture.contract(cell))
                self.assertIs(audit.state, TrainingCellState.INCOMPLETE)

        for attribute in ("checkpoint_metadata", "adaptation_state", "loop_state"):
            with self.subTest(checkpoint_edge=attribute), tempfile.TemporaryDirectory() as td:
                repo = Path(td)
                fixture = Fixture(repo)
                cell = STAGE_A_MATRIX[0]
                fixture.complete(cell)
                path = getattr(canonical_training_cell_paths(repo, cell, steps=2), attribute)
                path.write_bytes(path.read_bytes() + b"changed")
                self.assertIs(
                    classify_training_cell(repo, cell, fixture.contract(cell)).state,
                    TrainingCellState.INCOMPLETE,
                )

    def test_barrier_is_deterministic_and_requires_every_cell(self) -> None:
        contracts = self.fixture.contracts("A")
        for cell in STAGE_A_MATRIX:
            self.fixture.complete(cell)
        first = training_barrier(self.repo, "A", contracts)
        second = training_barrier(self.repo, "stage_a", contracts)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "TRAINING_BARRIER_COMPLETE")
        self.assertEqual(len(first["cells"]), 3)
        self.assertEqual(
            first["barrier_identity_sha256"],
            _canonical_sha256(
                {k: v for k, v in first.items() if k != "barrier_identity_sha256"}
            ),
        )

        missing = canonical_training_cell_paths(
            self.repo, STAGE_A_MATRIX[-1], steps=2
        ).loop_state
        missing.unlink()
        with self.assertRaisesRegex(TrainingReceiptError, "incomplete"):
            evaluation_barrier(self.repo, "A", contracts)

    def test_launch_preflight_enforces_absent_target_complete_or_absent_peers_and_no_trigger(self) -> None:
        contracts = self.fixture.contracts("A")
        target = STAGE_A_MATRIX[1]
        with self.assertRaisesRegex(TrainingReceiptError, "out-of-order"):
            training_launch_preflight(self.repo, target, contracts)

        self.fixture.complete(STAGE_A_MATRIX[0])
        proof = training_launch_preflight(self.repo, target, contracts)
        self.assertEqual(proof["status"], "TRAINING_LAUNCH_PREFLIGHT_PASS")
        self.assertEqual(
            proof["peers"],
            [
                {"cell": STAGE_A_MATRIX[0].slug, "state": "COMPLETE"},
                {"cell": STAGE_A_MATRIX[2].slug, "state": "ABSENT"},
            ],
        )

        partial = canonical_training_cell_paths(
            self.repo, STAGE_A_MATRIX[2], steps=2
        ).external_dir
        partial.mkdir(parents=True)
        with self.assertRaisesRegex(TrainingReceiptError, "INCOMPLETE"):
            training_launch_preflight(self.repo, target, contracts)
        partial.rmdir()

        trigger = canonical_training_cell_paths(
            self.repo, STAGE_A_MATRIX[0], steps=2
        ).trigger_output
        trigger.mkdir(parents=True)
        with self.assertRaisesRegex(TrainingReceiptError, "trigger output"):
            training_launch_preflight(self.repo, target, contracts)
        trigger.rmdir()

        self.fixture.complete(target)
        with self.assertRaisesRegex(TrainingReceiptError, "must be ABSENT"):
            training_launch_preflight(self.repo, target, contracts)

    def test_stage_c_evaluation_requires_one_identical_authorization(self) -> None:
        authorization = _lineage("fullrank-control-authorization")
        contracts = self.fixture.contracts("C")
        for cell in STAGE_C_MATRIX:
            self.fixture.complete(cell, branch=authorization)
        proof = evaluation_barrier(
            self.repo,
            "C",
            contracts,
            required_authorization=authorization,
        )
        self.assertEqual(proof["branch_authorization_lineage"], authorization)

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            fixture = Fixture(repo)
            for index, cell in enumerate(STAGE_C_MATRIX):
                fixture.complete(
                    cell,
                    branch=(authorization if index < 2 else _lineage("wrong-auth")),
                )
            with self.assertRaisesRegex(TrainingReceiptError, "share one authorization"):
                evaluation_barrier(repo, "C", fixture.contracts("C"))

    def test_contract_requires_the_full_caller_identity(self) -> None:
        cell = STAGE_A_MATRIX[0]
        identity = self.fixture.identity(cell)
        del identity["source_contract_sha256"]
        with self.assertRaisesRegex(ValueError, "full frozen field set"):
            TrainingReceiptContract(1, "TRAINING_COMPLETE", identity, 2)


if __name__ == "__main__":
    unittest.main()
