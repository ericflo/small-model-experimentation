#!/usr/bin/env python3
"""Content-address and preserve an incomplete canonical attempt before retry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    config_sha256,
    load_config,
    source_contract_sha256,
)
from src.design_boundary import design_lineage, validate_design_receipt  # noqa: E402
from src.attempt_receipts import (  # noqa: E402
    ATTEMPT_MARKER_NAME,
    AttemptReceiptError,
    atomic_write_json,
    canonical_sha256,
    fsync_directory,
    fsync_file,
    locked_regular,
    load_training_journal,
    read_json as read_attempt_json,
    tree_manifest,
    validate_attempt_authorization,
    validate_attempt_marker,
    validate_failed_archive,
    validate_tree_manifest,
)
from src.data_pipeline import (  # noqa: E402
    load_contrast_access_ledger,
    validate_data_manifest,
)
from src.gate_receipts import (  # noqa: E402
    LORA_MISS_BRANCH,
    POSTCONTRAST_FULLRANK_MISS_BRANCH,
    STAGE_B_FULLRANK_MISS_BRANCH,
    reopen_lineage,
    stable_setup_receipt,
    validate_branch_authorization,
)
from src.training_receipts import (  # noqa: E402
    TrainingCell,
    TrainingCellState,
    TrainingReceiptContract,
    canonical_training_cell_paths,
    classify_training_cell,
)
from src.safe_io import (  # noqa: E402
    StableArtifactError,
    open_stable_directory_for_update,
    open_stable_regular,
    read_stable_json_object,
    rename_new_entry,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file_record(path: Path, relative: str) -> dict[str, Any]:
    """Hash one regular file through a no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(
                f"failed attempt contains a non-regular file: {path}"
            )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "path": relative,
            "bytes": info.st_size,
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _absolute_no_follow(path: Path) -> Path:
    """Normalize an absolute path without dereferencing any symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _safe_repo_path(path: Path, label: str) -> Path:
    """Confine a path lexically and reject symlinks before any resolution."""

    repository = _absolute_no_follow(REPO_ROOT)
    absolute = _absolute_no_follow(path)
    if not absolute.is_relative_to(repository):
        raise RuntimeError(f"{label} escapes repository: {absolute}")
    if repository.is_symlink():
        raise RuntimeError(f"{label} has a symlinked repository root")
    current = repository
    for part in absolute.relative_to(repository).parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"{label} has a symlinked path component: {current}")
    return absolute


def _repo_relative(path: Path) -> str:
    absolute = _safe_repo_path(path, "failed attempt path")
    return absolute.relative_to(_absolute_no_follow(REPO_ROOT)).as_posix()


def _manifest(path: Path) -> dict[str, Any]:
    path = _safe_repo_path(path, "failed attempt path")
    try:
        return tree_manifest(path, source_path=_repo_relative(path))
    except AttemptReceiptError as exc:
        raise RuntimeError(f"failed attempt tree is unsafe: {exc}") from exc


def _training_pairs(config: dict[str, Any]) -> dict[str, tuple[Path, Path]]:
    large = _safe_repo_path(
        ROOT / config["paths"]["large_artifacts_dir"],
        "canonical external training root",
    )
    tracked = _safe_repo_path(
        ROOT / "runs" / "training",
        "canonical tracked training root",
    )
    pairs = {}
    for capacity in ("lora", "fullrank"):
        for objective in ("joint", "state_only"):
            for seed in map(int, config["training"]["train_seeds"]):
                cell = f"{capacity}_{objective}_seed{seed}"
                pairs[cell] = (large / cell, tracked / cell)
    return pairs


def _allowed_paths(config: dict[str, Any]) -> set[Path]:
    runs = _safe_repo_path(ROOT / "runs", "canonical tracked runs root")
    paths = set()
    for cell, pair in _training_pairs(config).items():
        paths.update(pair)
        paths.add(runs / f"{cell}_trigger")
        if "_joint_" in cell:
            paths.add(runs / f"{cell}_contrast")
    return paths


def _training_cell_identity(cell: str) -> tuple[str, str, int] | None:
    for capacity in ("lora", "fullrank"):
        for objective in ("joint", "state_only"):
            prefix = f"{capacity}_{objective}_seed"
            if cell.startswith(prefix):
                suffix = cell.removeprefix(prefix)
                if suffix.isdigit():
                    return capacity, objective, int(suffix)
    return None


def _training_cell_stage(capacity: str, objective: str) -> str:
    try:
        return {
            ("lora", "joint"): "A",
            ("lora", "state_only"): "B",
            ("fullrank", "joint"): "B",
            ("fullrank", "state_only"): "C",
        }[(capacity, objective)]
    except KeyError as exc:
        raise RuntimeError("training cell is outside the registered stage matrix") from exc


def _training_contract(
    config: dict[str, Any], cell: TrainingCell
) -> TrainingReceiptContract:
    design = design_lineage(config)
    return TrainingReceiptContract(
        schema_version=1,
        status="TRAINING_COMPLETE",
        identity={
            "experiment_id": config["experiment_id"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "backend": "transformers",
            "config_sha256": config_sha256(config),
            "source_contract_sha256": source_contract_sha256(ROOT),
            "requirements_training_lock_sha256": _sha256(
                REPO_ROOT / "requirements-training.lock.txt"
            ),
            "design_receipt_sha256": design["sha256"],
            "design_receipt_identity_sha256": design["receipt_identity_sha256"],
            "phase": cell.phase,
        },
        steps=int(config["training"]["train_steps"]),
    )


def _canonicalize_audit_error(value: str) -> str:
    repository = _absolute_no_follow(REPO_ROOT).as_posix().rstrip("/")
    return value.replace(f"{repository}/", "")


def _terminal_training_pair_audit(
    config: dict[str, Any],
    external: Path,
    tracked: Path,
    *,
    allow_started_terminal: bool = False,
) -> tuple[bool, TrainingCell, tuple[str, ...]]:
    """Apply the shared terminal graph and the producer's final access contract."""

    identity = _training_cell_identity(external.name)
    if identity is None or tracked.name != external.name:
        raise RuntimeError("training pair is not one registered same-cell pair")
    capacity, objective, seed = identity
    cell = TrainingCell(
        _training_cell_stage(capacity, objective), capacity, objective, seed
    )
    contract = _training_contract(config, cell)
    paths = canonical_training_cell_paths(
        REPO_ROOT, cell, steps=contract.steps
    )
    if paths.external_dir != external or paths.tracked_dir != tracked:
        raise RuntimeError("training pair differs from the canonical terminal graph")

    audit = classify_training_cell(
        REPO_ROOT,
        cell,
        contract,
        allow_started_terminal=allow_started_terminal,
    )
    if audit.state is not TrainingCellState.COMPLETE:
        errors = tuple(_canonicalize_audit_error(item) for item in audit.errors)
        return False, cell, errors or (f"terminal graph state is {audit.state.value}",)

    try:
        # The shared graph validates exact directory membership, distinct
        # inodes, byte-identical mirrors, setup equality, tensor hashes, and
        # lineage shapes.  Reopen the resulting receipts to enforce the
        # remaining producer semantics before declaring the pair unarchivable.
        run = read_stable_json_object(REPO_ROOT, paths.external_run)
        metadata = read_stable_json_object(REPO_ROOT, paths.checkpoint_metadata)
        if not isinstance(run, dict) or not isinstance(metadata, dict):
            raise RuntimeError("terminal receipts are not JSON objects")

        access = {
            "authorizes_training": False,
            "authorizes_result_training": False,
            "authorizes_result_evaluation": False,
            "benchmark_files_read": 0,
            "result_payloads_opened": ["train"],
            "sealed_contrast_payloads_opened": [],
            "training_or_evaluation_started": True,
            "scientific_evidence": False,
        }
        for field, expected in access.items():
            if type(run.get(field)) is not type(expected) or run.get(field) != expected:
                raise RuntimeError(
                    f"terminal run has wrong nonauthorization/access field: {field}"
                )

        for field in (
            "g0_lineage",
            "positive_control_lineage",
            "branch_authorization_lineage",
        ):
            if run.get(field) != metadata.get(field):
                raise RuntimeError(f"terminal run/checkpoint lineage differs: {field}")

        identity_base = {
            key: value
            for key, value in contract.identity.items()
            if key != "phase"
        }
        experiment = f"experiments/{config['experiment_id']}"
        g0_path = f"{experiment}/runs/setup/g0_{capacity}_seed{seed}.json"
        control_path = (
            f"{experiment}/runs/setup/positive_control_{capacity}_seed{seed}.json"
        )
        g0 = reopen_lineage(
            REPO_ROOT,
            run["g0_lineage"],
            expected_identity=identity_base,
            expected_status="MODEL_SMOKE_PASS",
            expected_phase=f"{capacity}_g0",
            canonical_relative_path=g0_path,
        )
        control = reopen_lineage(
            REPO_ROOT,
            run["positive_control_lineage"],
            expected_identity=identity_base,
            expected_status="POSITIVE_CONTROL_PASS",
            expected_phase=f"{capacity}_positive_control",
            canonical_relative_path=control_path,
        )
        if control.get("g0_lineage") != run["g0_lineage"]:
            raise RuntimeError("positive control does not bind the terminal G0 lineage")
        setup_barrier = run.get("setup_barrier")
        if not isinstance(setup_barrier, dict):
            raise RuntimeError("terminal run omits its setup barrier")
        expected_setup_branch = (
            None
            if capacity == "lora"
            else setup_barrier.get("root_lora_miss_lineage")
        )
        if control.get("branch_authorization") != expected_setup_branch:
            raise RuntimeError("positive control does not bind the setup branch lineage")
        if g0.get("branch_authorization") != expected_setup_branch:
            raise RuntimeError("G0 does not bind the setup branch lineage")
        stable = run.get("stable_setup")
        if (
            stable_setup_receipt(g0.get("setup")) != stable
            or stable_setup_receipt(control.get("setup")) != stable
        ):
            raise RuntimeError("terminal setup differs from its reopened gate setup")

        branch = run["branch_authorization_lineage"]
        if cell.stage == "A":
            if branch is not None:
                raise RuntimeError("Stage A terminal graph has branch authorization")
        else:
            if not isinstance(branch, dict):
                raise RuntimeError("conditional terminal graph omits branch authorization")
            branch_path = branch.get("path")
            if capacity == "fullrank" and objective == "state_only":
                if branch_path == f"{experiment}/analysis/stage_b_seal.json":
                    branch_kind = STAGE_B_FULLRANK_MISS_BRANCH
                elif branch_path == f"{experiment}/analysis/fullrank_joint.json":
                    branch_kind = POSTCONTRAST_FULLRANK_MISS_BRANCH
                else:
                    raise RuntimeError(
                        "full-rank state-only terminal graph has a noncanonical branch"
                    )
            else:
                branch_kind = LORA_MISS_BRANCH
            validated = validate_branch_authorization(
                REPO_ROOT,
                REPO_ROOT / str(branch_path),
                canonical_relative_path=str(branch_path),
                branch=branch_kind,
                expected_identity=identity_base,
            )
            if validated["lineage"] != branch:
                raise RuntimeError("terminal branch lineage differs from canonical analysis")
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        return False, cell, (_canonicalize_audit_error(str(exc)),)
    return True, cell, ()


def _is_valid_completed_training_pair(
    config: dict[str, Any], external: Path, tracked: Path
) -> bool:
    try:
        complete, _, _ = _terminal_training_pair_audit(config, external, tracked)
        return complete
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return False


def _durable_training_started_authorization(
    config: dict[str, Any],
    training_pair: tuple[Path, Path],
) -> dict[str, Any]:
    """Return the exact durable STARTED head even after both trees are gone."""

    identity = _training_cell_identity(training_pair[0].name)
    if identity is None:
        raise RuntimeError("training archive marker has no registered cell")
    capacity, objective, seed = identity
    stage = _training_cell_stage(capacity, objective)
    cell_payload = {
        "stage": stage,
        "capacity": capacity,
        "objective": objective,
        "seed": seed,
        "slug": training_pair[0].name,
    }
    canonical_paths = [_repo_relative(path) for path in training_pair]
    design = design_lineage(config)
    header = {
        "experiment_id": config["experiment_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers",
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(ROOT),
        "requirements_training_lock_sha256": _sha256(
            REPO_ROOT / "requirements-training.lock.txt"
        ),
        "design_receipt_sha256": design["sha256"],
        "design_receipt_identity_sha256": design["receipt_identity_sha256"],
    }
    journal = load_training_journal(
        REPO_ROOT,
        training_pair[0].name,
        header=header,
        cell=cell_payload,
        canonical_paths=canonical_paths,
    )
    if journal is None or journal["events"][-1]["state"] != "STARTED":
        raise RuntimeError(
            "training archive lacks a durable attempt marker or durable STARTED journal head"
        )
    authorization = validate_attempt_authorization(
        journal["events"][-1]["authorization"], attempt_kind="training"
    )
    if authorization["canonical_paths"] != canonical_paths:
        raise RuntimeError("training archive STARTED head binds different canonical paths")
    return authorization


def _durable_attempt_authorization(
    config: dict[str, Any],
    resolved: list[Path],
    training_pair: tuple[Path, Path] | None,
) -> dict[str, Any] | None:
    """Reopen the marker and durable STARTED journal/ledger for an attempt."""

    markers = [
        path / ATTEMPT_MARKER_NAME
        for path in resolved
        if os.path.lexists(path / ATTEMPT_MARKER_NAME)
    ]
    first: dict[str, Any] | None = None
    if markers:
        parsed = [read_attempt_json(path) for path in markers]
        authorizations = [
            validate_attempt_authorization(item.get("attempt_authorization"))
            for item in parsed
        ]
        first = authorizations[0]
        if any(item != first for item in authorizations[1:]):
            raise RuntimeError("attempt marker mirrors bind different authorizations")
        for marker in parsed:
            validate_attempt_marker(marker, first)

    if training_pair is not None:
        durable = _durable_training_started_authorization(config, training_pair)
        if first is not None and first["attempt_kind"] != "training":
            raise RuntimeError("training archive marker has the wrong attempt kind")
        if first is not None and durable != first:
            raise RuntimeError("training archive is not the durable STARTED journal head")
        return durable
    else:
        if first is None:
            return None
        if first["attempt_kind"] != "contrast":
            # Trigger evaluations predate the contrast attempt ledger.  They
            # remain archivable but cannot license automatic same-cell replay.
            return None
        context = first.get("context")
        if not isinstance(context, dict):
            raise RuntimeError("contrast attempt authorization omits its context")
        manifest_path = _safe_repo_path(
            REPO_ROOT / str(context.get("data_manifest_path")),
            "prepared data manifest",
        )
        data_dir = manifest_path.parent
        expected_ledger = _safe_repo_path(
            REPO_ROOT / str(context.get("contrast_access_ledger_path")),
            "contrast access ledger",
        )
        if expected_ledger != data_dir / "contrast_access_ledger.json":
            raise RuntimeError("contrast attempt ledger path is noncanonical")
        manifest = read_attempt_json(manifest_path)
        validate_data_manifest(config, data_dir, manifest, content_splits=set())
        ledger = load_contrast_access_ledger(config, data_dir, manifest)
        matches = [
            attempt
            for event in ledger["events"]
            for attempt in event["attempts"]
            if attempt["authorization"] == first
        ]
        if len(matches) != 1 or matches[0]["state"] != "STARTED":
            raise RuntimeError("contrast archive is not a unique durable STARTED event")
    return first


def _archive_authority(
    config: dict[str, Any],
    resolved: list[Path],
    training_pair: tuple[Path, Path] | None,
) -> dict[str, Any]:
    """Create the fail-closed authority compatible with step-zero replay.

    Replay has no independent crash oracle: a crash can leave either side of a
    training pair or one partial evaluation directory.  The authority is thus
    the deterministic pre-move classification of those canonical paths.  It is
    embedded in every attempt manifest so the unchanged top-level receipt
    schema remains consumable by the existing contrast replay firewall.
    """

    durable_attempt = _durable_attempt_authorization(
        config, resolved, training_pair
    )
    if training_pair is not None:
        if durable_attempt is None:
            raise RuntimeError("training archive lacks a durable attempt marker")
        complete, cell, errors = _terminal_training_pair_audit(
            config, *training_pair
        )
        if complete:
            raise RuntimeError(
                "refusing to archive a valid completed training pair as a failed attempt"
            )
        recoverable, _, _ = _terminal_training_pair_audit(
            config, *training_pair, allow_started_terminal=True
        )
        if recoverable:
            raise RuntimeError(
                "refusing to archive fully published terminal training receipts; "
                "finalize the durable training journal instead"
            )
        authority: dict[str, Any] = {
            "schema_version": 1,
            "status": "INCOMPLETE_ATTEMPT_ARCHIVE_AUTHORIZED",
            "phase": "failed_attempt_archive_preflight",
            "attempt_kind": "training",
            "cell": cell.slug,
            "canonical_paths": [_repo_relative(path) for path in training_pair],
            "present_paths": [_repo_relative(path) for path in resolved],
            "terminal_graph_state": "INCOMPLETE",
            "terminal_graph_errors": list(errors),
            "requires_step_zero_replay": True,
            "attempt_identity_sha256": durable_attempt[
                "attempt_identity_sha256"
            ],
            "attempt_authorization": durable_attempt,
        }
    else:
        if len(resolved) != 1:
            raise RuntimeError("evaluation archival requires exactly one canonical path")
        output = resolved[0]
        marker = output / "summary.json"
        if os.path.lexists(marker):
            raise RuntimeError(
                "refusing to archive a completed evaluation as a failed attempt"
            )
        is_contrast = output.name.endswith("_contrast")
        if is_contrast and durable_attempt is None:
            raise RuntimeError("contrast archive lacks a durable STARTED attempt marker")
        synthetic_identity = _canonical_sha256(
            {
                "attempt_kind": "evaluation",
                "canonical_paths": [_repo_relative(output)],
                "terminal_marker_present": False,
                "tree_identity_sha256": _manifest(output)[
                    "tree_identity_sha256"
                ],
            }
        )
        authority = {
            "schema_version": 1,
            "status": "INCOMPLETE_ATTEMPT_ARCHIVE_AUTHORIZED",
            "phase": "failed_attempt_archive_preflight",
            "attempt_kind": "contrast" if is_contrast else "evaluation",
            "canonical_paths": [_repo_relative(output)],
            "terminal_marker": _repo_relative(marker),
            "terminal_marker_present": False,
            "requires_identical_checkpoint_replay": True,
            "attempt_identity_sha256": (
                durable_attempt["attempt_identity_sha256"]
                if durable_attempt is not None
                else synthetic_identity
            ),
            "attempt_authorization": durable_attempt,
        }
    authority.update(
        {
            "authorizes_training": False,
            "authorizes_result_training": False,
            "authorizes_result_evaluation": False,
            "benchmark_files_read": 0,
            "scientific_evidence": False,
        }
    )
    authority["authority_identity_sha256"] = _canonical_sha256(authority)
    return authority


def _bind_archive_authority(
    manifest: dict[str, Any], authority: dict[str, Any]
) -> dict[str, Any]:
    bound = dict(manifest)
    bound["archive_authority"] = dict(authority)
    bound["manifest_identity_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in bound.items()
            if key != "manifest_identity_sha256"
        }
    )
    return bound


def _normalized_archive_paths(
    config: dict[str, Any], paths: list[Path]
) -> tuple[list[Path], tuple[Path, Path] | None]:
    if not 1 <= len(paths) <= 2:
        raise RuntimeError("archive exactly one canonical attempt and at most one companion")
    requested = [
        _safe_repo_path(path, "failed attempt path")
        for path in paths
    ]
    if len(set(requested)) != len(requested):
        raise RuntimeError("failed attempt paths must be distinct")
    allowed = _allowed_paths(config)
    if any(path not in allowed for path in requested):
        raise RuntimeError("refusing to archive a noncanonical or unregistered path")
    selected = [
        (cell, pair)
        for cell, pair in _training_pairs(config).items()
        if any(path in pair for path in requested)
    ]
    if selected:
        cell, pair = selected[0]
        if len(selected) != 1 or any(path not in pair for path in requested):
            raise RuntimeError(
                "two-path archive must contain the exact same-cell tracked companion"
            )
        # Either side is a valid selector, but omission is never a choice:
        # capture every same-cell companion that exists now.
        existing = [path for path in pair if os.path.lexists(path)]
        for path in existing:
            _safe_repo_path(path, "failed training attempt path")
        return existing, pair

    if len(requested) != 1:
        raise RuntimeError(
            "two-path archive must contain the exact same-cell tracked companion"
        )
    return [path for path in requested if os.path.lexists(path)], None


def _archive_header(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FAILED_ATTEMPT_ARCHIVED",
        "experiment_id": config["experiment_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers",
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(ROOT),
        "requirements_training_lock_sha256": _sha256(
            REPO_ROOT / "requirements-training.lock.txt"
        ),
        "design_lineage": design_lineage(config),
        "scientific_evidence": False,
    }


def _crash_point(name: str) -> None:
    if os.environ.get("QWEN35_ARCHIVE_CRASH_AT") == name:
        raise RuntimeError(f"injected archive crash at {name}")


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        directories.append(current)
        for name in names:
            child = current / name
            if child.is_symlink():
                raise RuntimeError("archive copy contains a symlink")
        for name in files:
            child = current / name
            if child.is_symlink() or not child.is_file():
                raise RuntimeError("archive copy contains a non-regular leaf")
            fsync_file(child)
    for directory in reversed(directories):
        fsync_directory(directory)


_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_LOWER_HEX_DIGITS = frozenset("0123456789abcdef")


def _validated_expected_attempt_identity(value: str | None) -> str | None:
    """Return one canonical attempt-authority identity or reject it.

    Attempt identities are security-relevant selectors, not convenient hash
    prefixes.  In particular, a retry after canonical-to-tombstone rename may
    have no live evaluation marker left from which to recover the identity.
    """

    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWER_HEX_DIGITS for character in value)
    ):
        raise RuntimeError(
            "expected attempt identity must be exactly 64 lowercase hexadecimal "
            "characters"
        )
    return value


def _attempt_identity_argument(value: str) -> str:
    """Argparse adapter for the public canonical-identity contract."""

    try:
        validated = _validated_expected_attempt_identity(value)
    except RuntimeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    assert validated is not None
    return validated


def _descriptor_digest(descriptor: int) -> str:
    """Hash a held regular-file description without changing its final offset."""

    offset = os.lseek(descriptor, 0, os.SEEK_CUR)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.lseek(descriptor, offset, os.SEEK_SET)


@contextmanager
def _held_verified_failed_archive(
    archive_root: Path,
    tracked_receipt: Path,
    receipt: dict[str, Any],
) -> Iterator[Callable[[], None]]:
    """Hold every durable archive byte while canonical sources are retired."""

    encoded_receipt = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    receipt_sha256 = hashlib.sha256(encoded_receipt).hexdigest()
    held: list[tuple[int, int, str, str]] = []
    with ExitStack() as stack:
        stack.enter_context(
            open_stable_directory_for_update(REPO_ROOT, archive_root)
        )
        for label, path in (
            ("archive receipt", archive_root / "archive_receipt.json"),
            ("tracked archive receipt", tracked_receipt),
        ):
            handle = stack.enter_context(
                open_stable_regular(
                    REPO_ROOT,
                    path,
                    expected_sha256=receipt_sha256,
                )
            )
            if handle.read() != encoded_receipt:
                raise RuntimeError(f"failed-attempt {label} bytes changed")
            held.append(
                (
                    handle.fileno(),
                    len(encoded_receipt),
                    receipt_sha256,
                    label,
                )
            )
        attempts = receipt.get("attempts")
        if not isinstance(attempts, list):
            raise RuntimeError("failed-attempt archive receipt omits its attempts")
        for index, manifest in enumerate(attempts, start=1):
            if not isinstance(manifest, dict):
                raise RuntimeError("failed-attempt archive manifest is malformed")
            source_path = manifest.get("source_path")
            files = manifest.get("files")
            if not isinstance(source_path, str) or not isinstance(files, list):
                raise RuntimeError("failed-attempt archive manifest is malformed")
            source_name = Path(source_path).name
            for record in files:
                if not isinstance(record, dict):
                    raise RuntimeError("failed-attempt archive file record is malformed")
                relative = record.get("path")
                expected_bytes = record.get("bytes")
                expected_sha256 = record.get("sha256")
                if (
                    not isinstance(relative, str)
                    or not isinstance(expected_bytes, int)
                    or isinstance(expected_bytes, bool)
                    or not isinstance(expected_sha256, str)
                ):
                    raise RuntimeError(
                        "failed-attempt archive file record is malformed"
                    )
                path = (
                    archive_root
                    / f"source_{index}_{source_name}"
                    / relative
                )
                handle = stack.enter_context(
                    open_stable_regular(
                        REPO_ROOT,
                        path,
                        expected_sha256=expected_sha256,
                    )
                )
                if os.fstat(handle.fileno()).st_size != expected_bytes:
                    raise RuntimeError(
                        "failed-attempt archive payload has the wrong byte count"
                    )
                held.append(
                    (
                        handle.fileno(),
                        expected_bytes,
                        expected_sha256,
                        f"archive payload {index}:{relative}",
                    )
                )

        def revalidate() -> None:
            for descriptor, expected_bytes, expected_sha256, label in held:
                info = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_size != expected_bytes
                    or _descriptor_digest(descriptor) != expected_sha256
                ):
                    raise RuntimeError(
                        f"failed-attempt {label} changed before cleanup commit"
                    )

        revalidate()
        yield revalidate


def _inode_identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _directory_flags() -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _regular_update_flags() -> int:
    return (
        os.O_RDWR
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _zeroize_quarantined_tree(
    tombstone: Path,
    manifest: dict[str, Any],
) -> None:
    """Zero only a complete, manifest-exact archived duplicate through held fds.

    Cleanup deliberately retains the tombstone directory and its zero-length
    skeleton.  POSIX has no compare-and-unlink primitive; retaining names while
    truncating only already-opened, fully verified duplicate inodes makes a
    pathname swap fail closed without deleting an unvalidated replacement.
    A partially zeroized tree is accepted on retry, but every nonempty file must
    still match its archived record exactly.
    """

    raw_files = manifest.get("files")
    raw_directories = manifest.get("directory_entries")
    if not isinstance(raw_files, list) or not isinstance(raw_directories, list):
        raise RuntimeError("archive cleanup manifest is malformed")
    expected_files: dict[str, dict[str, Any]] = {}
    for record in raw_files:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "bytes", "sha256"}
            or not isinstance(record["path"], str)
            or not isinstance(record["bytes"], int)
            or isinstance(record["bytes"], bool)
            or record["bytes"] < 0
            or not isinstance(record["sha256"], str)
        ):
            raise RuntimeError("archive cleanup file manifest is malformed")
        expected_files[record["path"]] = record
    if len(expected_files) != len(raw_files):
        raise RuntimeError("archive cleanup file manifest has duplicate paths")

    expected_directories: dict[str, list[dict[str, str]]] = {}
    for record in raw_directories:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "entries"}
            or not isinstance(record["path"], str)
            or not isinstance(record["entries"], list)
        ):
            raise RuntimeError("archive cleanup directory manifest is malformed")
        entries: list[dict[str, str]] = []
        for entry in record["entries"]:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"name", "type"}
                or not isinstance(entry["name"], str)
                or entry["type"] not in {"directory", "regular_file"}
            ):
                raise RuntimeError("archive cleanup directory entry is malformed")
            entries.append(entry)
        expected_directories[record["path"]] = entries
    if len(expected_directories) != len(raw_directories) or "." not in expected_directories:
        raise RuntimeError("archive cleanup directory manifest has invalid paths")

    stack = ExitStack()
    root_descriptor: int | None = None
    held_directories: list[
        tuple[str, int, list[dict[str, str]], dict[str, tuple[int, int]]]
    ] = []
    held_files: list[
        tuple[str, int, int, str, tuple[int, int], bool]
    ] = []
    try:
        root_descriptor = stack.enter_context(
            open_stable_directory_for_update(REPO_ROOT, tombstone)
        )
        root_info = os.fstat(root_descriptor)

        seen_inodes: set[tuple[int, int]] = set()

        def walk(descriptor: int, relative: str) -> None:
            expected_entries = expected_directories.get(relative)
            if expected_entries is None:
                raise RuntimeError("archive cleanup tree has an unregistered directory")
            actual_names = sorted(os.listdir(descriptor))
            expected_names = sorted(item["name"] for item in expected_entries)
            if actual_names != expected_names:
                raise RuntimeError(
                    "archive cleanup tombstone differs from its archived manifest"
                )
            child_identities: dict[str, tuple[int, int]] = {}
            held_directories.append(
                (relative, descriptor, expected_entries, child_identities)
            )
            for expected in sorted(expected_entries, key=lambda item: item["name"]):
                name = expected["name"]
                child_relative = name if relative == "." else f"{relative}/{name}"
                observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if expected["type"] == "directory":
                    if not stat.S_ISDIR(observed.st_mode):
                        raise RuntimeError(
                            "archive cleanup tombstone entry changed type"
                        )
                    child_descriptor = os.open(
                        name,
                        _directory_flags(),
                        dir_fd=descriptor,
                    )
                    opened = os.fstat(child_descriptor)
                    if _inode_identity(opened) != _inode_identity(observed):
                        os.close(child_descriptor)
                        raise RuntimeError(
                            "archive cleanup directory changed while opening"
                        )
                    inode = _inode_identity(opened)
                    if inode in seen_inodes:
                        os.close(child_descriptor)
                        raise RuntimeError(
                            "archive cleanup tree contains a directory alias"
                        )
                    seen_inodes.add(inode)
                    child_identities[name] = inode
                    walk(child_descriptor, child_relative)
                else:
                    record = expected_files.get(child_relative)
                    if record is None or not stat.S_ISREG(observed.st_mode):
                        raise RuntimeError(
                            "archive cleanup tombstone file is unregistered"
                        )
                    child_descriptor = os.open(
                        name,
                        _regular_update_flags(),
                        dir_fd=descriptor,
                    )
                    opened = os.fstat(child_descriptor)
                    inode = _inode_identity(opened)
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or inode != _inode_identity(observed)
                        or opened.st_nlink != 1
                        or inode in seen_inodes
                    ):
                        os.close(child_descriptor)
                        raise RuntimeError(
                            "archive cleanup regular file is aliased or changed"
                        )
                    seen_inodes.add(inode)
                    digest = _descriptor_digest(child_descriptor)
                    is_zeroized = opened.st_size == 0 and digest == _EMPTY_SHA256
                    is_exact = (
                        opened.st_size == record["bytes"]
                        and digest == record["sha256"]
                    )
                    if not (is_zeroized or is_exact):
                        os.close(child_descriptor)
                        raise RuntimeError(
                            "archive cleanup tombstone differs from its archived manifest"
                        )
                    child_identities[name] = inode
                    held_files.append(
                        (
                            child_relative,
                            child_descriptor,
                            descriptor,
                            name,
                            inode,
                            is_zeroized,
                        )
                    )

        seen_inodes.add(_inode_identity(root_info))
        walk(root_descriptor, ".")
        if set(expected_files) != {item[0] for item in held_files}:
            raise RuntimeError("archive cleanup manifest omits or invents a file")

        # Rebind the entire held tree immediately before changing bytes.  A
        # concurrent pathname replacement is retained and causes failure.
        for _, descriptor, expected_entries, identities in held_directories:
            if sorted(os.listdir(descriptor)) != sorted(
                item["name"] for item in expected_entries
            ):
                raise RuntimeError("archive cleanup directory membership changed")
            for name, inode in identities.items():
                rebound = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if _inode_identity(rebound) != inode:
                    raise RuntimeError("archive cleanup child binding changed")

        for relative, descriptor, owner, name, inode, _ in held_files:
            record = expected_files[relative]
            before = os.fstat(descriptor)
            digest = _descriptor_digest(descriptor)
            is_zeroized = before.st_size == 0 and digest == _EMPTY_SHA256
            is_exact = (
                before.st_size == record["bytes"]
                and digest == record["sha256"]
            )
            rebound = os.stat(name, dir_fd=owner, follow_symlinks=False)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or _inode_identity(before) != inode
                or _inode_identity(rebound) != inode
                or not (is_zeroized or is_exact)
            ):
                raise RuntimeError(
                    "archive cleanup file changed immediately before zeroization"
                )
            if not is_zeroized:
                os.ftruncate(descriptor, 0)
            os.fsync(descriptor)
            after = os.fstat(descriptor)
            rebound = os.stat(name, dir_fd=owner, follow_symlinks=False)
            if (
                after.st_size != 0
                or after.st_nlink != 1
                or _inode_identity(after) != inode
                or _inode_identity(rebound) != inode
                or _descriptor_digest(descriptor) != _EMPTY_SHA256
            ):
                raise RuntimeError("archive cleanup file did not zeroize")

        for _, descriptor, expected_entries, identities in held_directories:
            if sorted(os.listdir(descriptor)) != sorted(
                item["name"] for item in expected_entries
            ):
                raise RuntimeError(
                    "archive cleanup directory membership changed after zeroization"
                )
            for name, inode in identities.items():
                rebound = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if _inode_identity(rebound) != inode:
                    raise RuntimeError(
                        "archive cleanup child binding changed after zeroization"
                    )
        for _, descriptor, _, _, inode, _ in held_files:
            final = os.fstat(descriptor)
            if (
                final.st_size != 0
                or final.st_nlink != 1
                or _inode_identity(final) != inode
                or _descriptor_digest(descriptor) != _EMPTY_SHA256
            ):
                raise RuntimeError(
                    "archive cleanup zeroized file changed before commit"
                )
        os.fsync(root_descriptor)
    except (RuntimeError, OSError):
        raise
    finally:
        for _, descriptor, _, _, _, _ in reversed(held_files):
            os.close(descriptor)
        if root_descriptor is not None:
            # Child directory descriptors are present in held_directories, but
            # the root is owned separately.
            child_descriptors = [
                descriptor
                for relative, descriptor, _, _ in held_directories
                if relative != "."
            ]
            for descriptor in reversed(child_descriptors):
                os.close(descriptor)
        stack.close()


def _validate_zeroized_tombstone(
    tombstone: Path,
    manifest: dict[str, Any],
) -> None:
    """Require the exact retained tree shape with every regular file empty."""

    try:
        observed = tree_manifest(
            tombstone,
            source_path=str(manifest.get("source_path")),
        )
    except AttemptReceiptError as exc:
        raise RuntimeError("archive cleanup tombstone is unsafe") from exc
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise RuntimeError("archive cleanup manifest is malformed")
    expected_files = [
        {
            "path": record["path"],
            "bytes": 0,
            "sha256": _EMPTY_SHA256,
        }
        for record in raw_files
    ]
    if (
        observed.get("files") != expected_files
        or observed.get("directory_entries") != manifest.get("directory_entries")
    ):
        raise RuntimeError(
            "archive cleanup canonical path coexists with a nonterminal tombstone"
        )


def _cleanup_sources(
    canonical_paths: list[Path],
    manifests: list[dict[str, Any]],
    *,
    attempt_set_identity: str,
    archive_revalidator: Callable[[], None] | None = None,
) -> None:
    revalidate_archive = archive_revalidator or (lambda: None)
    manifests_by_path = {
        _safe_repo_path(REPO_ROOT / manifest["source_path"], "archive source path"): manifest
        for manifest in manifests
    }
    if len(manifests_by_path) != len(manifests):
        raise RuntimeError("failed archive has duplicate source manifests")
    for index, source in enumerate(canonical_paths, start=1):
        revalidate_archive()
        tombstone = source.with_name(
            f".{source.name}.archived-{attempt_set_identity}-{index}"
        )
        manifest = manifests_by_path.get(source)
        if manifest is None:
            if os.path.lexists(source) or os.path.lexists(tombstone):
                raise RuntimeError(
                    "archive cleanup found a tree absent from the archive manifest"
                )
            continue
        if os.path.lexists(source) and os.path.lexists(tombstone):
            if tombstone.is_symlink() or not tombstone.is_dir():
                raise RuntimeError("archive cleanup tombstone is unsafe")
            # A complete zero tombstone is the immutable cleanup marker.  A
            # canonical tree that reappears later may be a newer generation;
            # completion therefore dominates and the canonical binding is
            # preserved untouched.  Re-run descriptor validation/fsync on the
            # zero skeleton so a prior truncate followed by fsync failure can
            # never be mistaken for durable completion.
            revalidate_archive()
            _zeroize_quarantined_tree(tombstone, manifest)
            revalidate_archive()
            continue
        if os.path.lexists(tombstone):
            if tombstone.is_symlink() or not tombstone.is_dir():
                raise RuntimeError("archive cleanup tombstone is unsafe")
            revalidate_archive()
            _zeroize_quarantined_tree(tombstone, manifest)
            revalidate_archive()
        if os.path.lexists(source):
            revalidate_archive()
            try:
                rename_new_entry(REPO_ROOT, source, tombstone)
            except StableArtifactError as exc:
                raise RuntimeError(
                    "archive cleanup tombstone commit was not an atomic no-clobber rename"
                ) from exc
            _crash_point(f"source_{index}_renamed")
            if tombstone.is_symlink() or not tombstone.is_dir():
                raise RuntimeError("archive cleanup tombstone is unsafe")
            revalidate_archive()
            _zeroize_quarantined_tree(tombstone, manifest)
            revalidate_archive()
        if not os.path.lexists(tombstone):
            raise RuntimeError("archive cleanup lost both canonical and tombstone trees")
        _crash_point(f"source_{index}_deleted")
    _crash_point("cleanup_complete")


def _resume_archived_attempt(
    config: dict[str, Any],
    *,
    label: str,
    canonical_paths: list[Path],
    expected_attempt_identity: str | None,
) -> dict[str, Any] | None:
    failures = ROOT / "runs" / "failures"
    candidates = sorted(failures.glob(f"{label}-*.json")) if failures.is_dir() else []
    matches: list[tuple[Path, dict[str, Any]]] = []
    expected_paths = [_repo_relative(path) for path in canonical_paths]
    # A crash after canonical->tombstone rename can leave no live evaluation
    # marker from which to recover its attempt-authority identity.  Tombstones
    # bind the *receipt/set* identity, while the caller's explicit selector
    # binds the authority identity inside that receipt.  Join those two exact
    # identities rather than comparing one kind of hash to the other.
    tombstone_receipt_identities: set[str] = set()
    for index, source in enumerate(canonical_paths, start=1):
        prefix = f".{source.name}.archived-"
        suffix = f"-{index}"
        if not source.parent.is_dir():
            continue
        for candidate in source.parent.iterdir():
            name = candidate.name
            if name.startswith(prefix) and name.endswith(suffix):
                identity_prefix = name[len(prefix) : -len(suffix)]
                if (
                    len(identity_prefix) != 64
                    or any(
                        character not in _LOWER_HEX_DIGITS
                        for character in identity_prefix
                    )
                ):
                    raise RuntimeError("archive cleanup tombstone identity is malformed")
                tombstone_receipt_identities.add(identity_prefix)
    for receipt_path in candidates:
        try:
            receipt = validate_failed_archive(
                REPO_ROOT, receipt_path, expected_header=_archive_header(config)
            )
        except AttemptReceiptError as exc:
            raise RuntimeError(f"existing failed archive is invalid: {exc}") from exc
        authority = receipt["attempts"][0]["archive_authority"]
        receipt_identity = receipt.get("attempt_identity_sha256")
        if (
            authority.get("canonical_paths") == expected_paths
            and (
                not tombstone_receipt_identities
                or receipt_identity in tombstone_receipt_identities
            )
            and (
                expected_attempt_identity is None
                or authority.get("attempt_identity_sha256")
                == expected_attempt_identity
            )
        ):
            matches.append((receipt_path, receipt))
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(
            "multiple failed archives claim the same canonical attempt; provide "
            "the exact expected attempt identity"
        )
    receipt_path, receipt = matches[0]
    archive_root = _safe_repo_path(
        REPO_ROOT / receipt["archive_path"],
        "failed-attempt archive root",
    )
    with _held_verified_failed_archive(
        archive_root,
        receipt_path,
        receipt,
    ) as revalidate_archive:
        revalidate_archive()
        _cleanup_sources(
            canonical_paths,
            receipt["attempts"],
            attempt_set_identity=receipt["attempt_identity_sha256"],
            archive_revalidator=revalidate_archive,
        )
        revalidate_archive()
    return receipt


def _archive_failed_attempt_transaction(
    config: dict[str, Any],
    paths: list[Path],
    *,
    expected_attempt_identity: str | None = None,
) -> dict[str, Any]:
    expected_attempt_identity = _validated_expected_attempt_identity(
        expected_attempt_identity
    )
    validate_design_receipt(config)
    resolved, training_pair = _normalized_archive_paths(config, paths)
    requested = [_safe_repo_path(path, "failed attempt path") for path in paths]
    canonical_paths = (
        list(training_pair)
        if training_pair is not None
        else requested
    )
    label = canonical_paths[0].name
    authority = (
        _archive_authority(config, resolved, training_pair) if resolved else None
    )
    durable_training = (
        _durable_training_started_authorization(config, training_pair)
        if not resolved and training_pair is not None
        else None
    )
    derived_attempt_identity = (
        str(authority["attempt_identity_sha256"])
        if authority is not None
        else (
            str(durable_training["attempt_identity_sha256"])
            if durable_training is not None
            else None
        )
    )
    if (
        expected_attempt_identity is not None
        and derived_attempt_identity is not None
        and expected_attempt_identity != derived_attempt_identity
    ):
        raise RuntimeError(
            "expected attempt identity does not match the current durable "
            "attempt authority"
        )
    resume_attempt_identity = (
        derived_attempt_identity
        if derived_attempt_identity is not None
        else expected_attempt_identity
    )
    resumed = _resume_archived_attempt(
        config,
        label=label,
        canonical_paths=canonical_paths,
        expected_attempt_identity=resume_attempt_identity,
    )
    if resumed is not None:
        return resumed
    if not resolved:
        if expected_attempt_identity is not None:
            raise RuntimeError(
                "failed attempt directory is missing and has no valid archive for "
                "the expected attempt identity"
            )
        raise RuntimeError("failed attempt directory is missing and has no valid archive")
    assert authority is not None
    manifests = [
        _bind_archive_authority(_manifest(path), authority) for path in resolved
    ]
    attempt_identity = _canonical_sha256({"attempts": manifests})
    label = canonical_paths[0].name
    archive_root = (
        ROOT
        / config["paths"]["large_artifacts_dir"]
        / "failed_attempts"
        / f"{label}-{attempt_identity[:16]}"
    )
    tracked_receipt = (
        ROOT / "runs" / "failures" / f"{label}-{attempt_identity[:16]}.json"
    )
    archive_root = _safe_repo_path(archive_root, "failed-attempt archive path")
    tracked_receipt = _safe_repo_path(
        tracked_receipt, "tracked failed-attempt receipt path"
    )
    receipt = {
        **_archive_header(config),
        "attempt_identity_sha256": attempt_identity,
        "archive_path": _repo_relative(archive_root),
        "attempts": manifests,
    }
    receipt["receipt_identity_sha256"] = _canonical_sha256(receipt)
    archive_parent = archive_root.parent
    archive_parent.mkdir(parents=True, exist_ok=True)
    fsync_directory(archive_parent.parent)
    staging = archive_parent / f".{archive_root.name}.staging"
    if os.path.lexists(archive_root):
        archive_receipt = archive_root / "archive_receipt.json"
        if read_attempt_json(archive_receipt) != receipt:
            raise RuntimeError("existing archive root binds different bytes")
    else:
        if os.path.lexists(staging):
            if staging.is_symlink() or not staging.is_dir():
                raise RuntimeError("failed-attempt staging path is unsafe")
        else:
            staging.mkdir(exist_ok=False)
            fsync_directory(archive_parent)
        for index, (source, manifest) in enumerate(
            zip(resolved, manifests, strict=True), start=1
        ):
            destination = staging / f"source_{index}_{source.name}"
            if os.path.lexists(destination):
                try:
                    validate_tree_manifest(destination, manifest)
                except AttemptReceiptError:
                    shutil.rmtree(destination)
                    fsync_directory(staging)
            if not os.path.lexists(destination):
                shutil.copytree(source, destination, symlinks=True)
                _fsync_tree(destination)
                fsync_directory(staging)
            validate_tree_manifest(destination, manifest)
            _crash_point(f"source_{index}_copied")
        staging_receipt = staging / "archive_receipt.json"
        if os.path.lexists(staging_receipt):
            if read_attempt_json(staging_receipt) != receipt:
                raise RuntimeError("staging archive receipt binds different bytes")
        else:
            atomic_write_json(staging_receipt, receipt, replace=False)
        fsync_directory(staging)
        _crash_point("archive_receipt_written")
        try:
            rename_new_entry(REPO_ROOT, staging, archive_root)
        except StableArtifactError as exc:
            raise RuntimeError(
                "refusing to overwrite or alias a competing failed-attempt archive"
            ) from exc
        _crash_point("archive_promoted")

    if os.path.lexists(tracked_receipt):
        if read_attempt_json(tracked_receipt) != receipt:
            raise RuntimeError("tracked archive receipt binds different bytes")
    else:
        atomic_write_json(tracked_receipt, receipt, replace=False)
    _crash_point("tracked_receipt_written")
    try:
        validate_failed_archive(
            REPO_ROOT, tracked_receipt, expected_header=_archive_header(config)
        )
    except AttemptReceiptError as exc:
        raise RuntimeError(f"failed archive did not verify before cleanup: {exc}") from exc
    _crash_point("archive_verified")
    with _held_verified_failed_archive(
        archive_root,
        tracked_receipt,
        receipt,
    ) as revalidate_archive:
        revalidate_archive()
        _cleanup_sources(
            canonical_paths,
            manifests,
            attempt_set_identity=attempt_identity,
            archive_revalidator=revalidate_archive,
        )
        revalidate_archive()
    return receipt


def archive_failed_attempt(
    config: dict[str, Any],
    paths: list[Path],
    *,
    expected_attempt_identity: str | None = None,
) -> dict[str, Any]:
    """Archive under the same setup/attempt locks held by every producer."""

    execution_lock = _safe_repo_path(
        ROOT / "runs" / "run.lock",
        "execution generation lock",
    )
    with locked_regular(execution_lock):
        return _archive_failed_attempt_transaction(
            config,
            paths,
            expected_attempt_identity=expected_attempt_identity,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--path", action="append", required=True)
    parser.add_argument(
        "--attempt-identity",
        dest="expected_attempt_identity",
        type=_attempt_identity_argument,
        help=(
            "exact 64-character lowercase attempt-authority SHA-256; required "
            "to disambiguate a markerless non-training retry"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    receipt = archive_failed_attempt(
        config,
        [Path(item) for item in args.path],
        expected_attempt_identity=args.expected_attempt_identity,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
