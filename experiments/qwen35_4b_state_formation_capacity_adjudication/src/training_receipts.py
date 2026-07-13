"""Fail-closed, stdlib-only completion proofs for result training.

This module deliberately knows nothing about tensors or ``torch``.  It treats a
training cell as complete only when the external artifacts, tracked mirrors,
fixed-final checkpoint, and every hash/identity edge form one closed graph.
Partial output is never resumable evidence: it is classified as ``INCOMPLETE``.
"""

from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .attempt_receipts import (
    ATTEMPT_MARKER_NAME,
    AttemptReceiptError,
    complete_training_attempt,
    load_training_journal,
    training_attempt_is_marker_only,
    training_journal_path,
    validate_attempt_authorization,
    validate_training_attempt_for_terminal,
)
from .gate_receipts import stable_setup_receipt
from .safe_io import open_stable_regular, read_stable_json_object


EXPERIMENT_ID = "qwen35_4b_state_formation_capacity_adjudication"
EXPERIMENT_RELATIVE = Path("experiments") / EXPERIMENT_ID
LARGE_ARTIFACTS_RELATIVE = Path("large_artifacts") / EXPERIMENT_ID
TRAINING_SEEDS = (7411, 7412, 7413)
DEFAULT_TRAINING_STEPS = 1500

RUN_IDENTITY_FIELDS = frozenset(
    {
        "experiment_id",
        "model_id",
        "model_revision",
        "backend",
        "config_sha256",
        "source_contract_sha256",
        "requirements_training_lock_sha256",
        "design_receipt_sha256",
        "design_receipt_identity_sha256",
        "phase",
    }
)
LINEAGE_FIELDS = frozenset(
    {"path", "sha256", "receipt_identity_sha256", "status", "phase"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class TrainingReceiptError(RuntimeError):
    """A training artifact graph violates its frozen completion contract."""


class TrainingCellState(str, Enum):
    ABSENT = "ABSENT"
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True, order=True)
class TrainingCell:
    """One preregistered result-bearing training cell."""

    stage: str
    capacity: str
    objective: str
    seed: int

    @property
    def slug(self) -> str:
        return f"{self.capacity}_{self.objective}_seed{self.seed}"

    @property
    def phase(self) -> str:
        return f"{self.capacity}_{self.objective}_training"


STAGE_A_MATRIX = tuple(
    TrainingCell("A", "lora", "joint", seed) for seed in TRAINING_SEEDS
)
# Frozen runbook order is seed-major and alternates the two Stage-B arms.  This
# ordering is scientific state: launch preflights below authorize only its exact
# prefix, so changing this comprehension to arm-major would silently change the
# experiment.
STAGE_B_MATRIX = tuple(
    TrainingCell("B", capacity, objective, seed)
    for seed in TRAINING_SEEDS
    for capacity, objective in (("lora", "state_only"), ("fullrank", "joint"))
)
STAGE_C_MATRIX = tuple(
    TrainingCell("C", "fullrank", "state_only", seed) for seed in TRAINING_SEEDS
)
TRAINING_STAGE_MATRICES = MappingProxyType(
    {"A": STAGE_A_MATRIX, "B": STAGE_B_MATRIX, "C": STAGE_C_MATRIX}
)


@dataclass(frozen=True)
class TrainingReceiptContract:
    """Caller-supplied immutable identity expected in a cell's two receipts."""

    schema_version: int
    status: str
    identity: Mapping[str, Any]
    steps: int = DEFAULT_TRAINING_STEPS

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int:
            raise ValueError("schema_version must be an exact integer")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a nonempty string")
        if set(self.identity) != RUN_IDENTITY_FIELDS:
            missing = sorted(RUN_IDENTITY_FIELDS - set(self.identity))
            extra = sorted(set(self.identity) - RUN_IDENTITY_FIELDS)
            raise ValueError(
                f"identity must contain the full frozen field set; missing={missing}, extra={extra}"
            )
        if type(self.steps) is not int or self.steps <= 0:
            raise ValueError("steps must be a positive exact integer")
        object.__setattr__(self, "identity", MappingProxyType(copy.deepcopy(dict(self.identity))))


@dataclass(frozen=True)
class TrainingCellPaths:
    external_dir: Path
    tracked_dir: Path
    external_run: Path
    tracked_run: Path
    external_metrics: Path
    tracked_metrics: Path
    external_optimizer_steps: Path
    tracked_optimizer_steps: Path
    external_attempt_marker: Path
    tracked_attempt_marker: Path
    attempt_journal: Path
    checkpoint_dir: Path
    checkpoint_metadata: Path
    adaptation_state: Path
    loop_state: Path
    trigger_output: Path


@dataclass(frozen=True)
class TrainingCellAudit:
    cell: TrainingCell
    state: TrainingCellState
    errors: tuple[str, ...]
    proof: Mapping[str, Any] | None = None
    setup_lineage: Mapping[str, Any] | None = None
    gate_lineages: Mapping[str, Any] | None = None
    branch_authorization_lineage: Mapping[str, Any] | None = None

    @property
    def complete(self) -> bool:
        return self.state is TrainingCellState.COMPLETE


def stage_matrix(stage: str) -> tuple[TrainingCell, ...]:
    """Return the exact ordered cell matrix for Stage A, B, or C."""

    normalized = str(stage).strip().upper().removeprefix("STAGE_").removeprefix("STAGE ")
    try:
        return TRAINING_STAGE_MATRICES[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown training stage: {stage!r}") from exc


def _canonical_repo_root(repo_root: Path | str) -> Path:
    root = Path(repo_root)
    absolute = Path(os.path.abspath(os.fspath(root)))
    if absolute.resolve(strict=False) != absolute:
        raise TrainingReceiptError("repository root is a symlink or path alias")
    if not absolute.is_dir():
        raise TrainingReceiptError(f"repository root is not a directory: {absolute}")
    return absolute


def canonical_training_cell_paths(
    repo_root: Path | str,
    cell: TrainingCell,
    *,
    steps: int = DEFAULT_TRAINING_STEPS,
) -> TrainingCellPaths:
    """Construct every canonical path for a registered training cell."""

    root = _canonical_repo_root(repo_root)
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive exact integer")
    if cell not in stage_matrix(cell.stage):
        raise ValueError(f"unregistered training cell: {cell!r}")
    external = root / LARGE_ARTIFACTS_RELATIVE / cell.slug
    tracked = root / EXPERIMENT_RELATIVE / "runs" / "training" / cell.slug
    checkpoint = external / f"checkpoint_{steps:06d}"
    return TrainingCellPaths(
        external_dir=external,
        tracked_dir=tracked,
        external_run=external / "run.json",
        tracked_run=tracked / "run.json",
        external_metrics=external / "train_metrics.jsonl",
        tracked_metrics=tracked / "train_metrics.jsonl",
        external_optimizer_steps=external / "optimizer_steps.jsonl",
        tracked_optimizer_steps=tracked / "optimizer_steps.jsonl",
        external_attempt_marker=external / ATTEMPT_MARKER_NAME,
        tracked_attempt_marker=tracked / ATTEMPT_MARKER_NAME,
        attempt_journal=training_journal_path(root, cell.slug),
        checkpoint_dir=checkpoint,
        checkpoint_metadata=checkpoint / "checkpoint.json",
        adaptation_state=checkpoint / "adaptation_state.pt",
        loop_state=checkpoint / "loop_state.pt",
        trigger_output=(
            root / EXPERIMENT_RELATIVE / "runs" / f"{cell.slug}_trigger"
        ),
    )


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open_stable_regular(Path("/"), path) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_object(path: Path) -> dict[str, Any]:
    return read_stable_json_object(Path("/"), path)


@dataclass(frozen=True)
class _FileSnapshot:
    raw: bytes
    sha256: str
    device: int
    inode: int


def _snapshot_regular_files(
    root: Path, files: Mapping[str, Path]
) -> dict[str, _FileSnapshot]:
    """Read every terminal leaf once while all no-follow handles remain open."""

    snapshots: dict[str, _FileSnapshot] = {}
    try:
        with contextlib.ExitStack() as stack:
            handles = {
                label: stack.enter_context(open_stable_regular(root, path))
                for label, path in files.items()
            }
            for label, handle in handles.items():
                raw = handle.read()
                info = os.fstat(handle.fileno())
                snapshots[label] = _FileSnapshot(
                    raw=raw,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    device=int(info.st_dev),
                    inode=int(info.st_ino),
                )
    except RuntimeError as exc:
        raise TrainingReceiptError(
            f"terminal artifact snapshot is unstable: {exc}"
        ) from exc
    seen: dict[tuple[int, int], str] = {}
    for label, snapshot in snapshots.items():
        identity = (snapshot.device, snapshot.inode)
        if identity in seen:
            raise TrainingReceiptError(
                f"artifact inode alias is prohibited: {seen[identity]} and {label}"
            )
        seen[identity] = label
    return snapshots


def _json_object_from_snapshot(snapshot: _FileSnapshot, label: str) -> dict[str, Any]:
    try:
        text = snapshot.raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingReceiptError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TrainingReceiptError(f"{label} is not a JSON object")
    return value


def _nonempty_snapshot_line_count(snapshot: _FileSnapshot, label: str) -> int:
    try:
        text = snapshot.raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingReceiptError(f"{label} is not UTF-8") from exc
    return sum(1 for line in text.splitlines() if line.strip())


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise TrainingReceiptError(f"canonical artifact escapes repository: {path}") from exc


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TrainingReceiptError(f"{label} is not a canonical SHA-256")
    return value


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise TrainingReceiptError(f"{label} mismatch: {value!r} != {expected!r}")


def _require_regular(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise TrainingReceiptError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise TrainingReceiptError(f"{label} is a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise TrainingReceiptError(f"{label} is not a regular file: {path}")
    return info


def _require_directory(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise TrainingReceiptError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise TrainingReceiptError(f"{label} is a symlink: {path}")
    if not stat.S_ISDIR(info.st_mode):
        raise TrainingReceiptError(f"{label} is not a directory: {path}")
    return info


def _require_no_symlink_components(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise TrainingReceiptError(f"artifact path escapes repository: {path}") from exc
    current = root
    for component in relative.parts:
        current /= component
        if not _lexists(current):
            continue
        if stat.S_ISLNK(current.lstat().st_mode):
            raise TrainingReceiptError(f"symlink component is prohibited: {current}")


def _require_exact_members(path: Path, expected: set[str], label: str) -> None:
    actual = {entry.name for entry in path.iterdir()}
    if actual != expected:
        raise TrainingReceiptError(
            f"{label} members mismatch: actual={sorted(actual)}, expected={sorted(expected)}"
        )


def _require_distinct_inodes(files: Mapping[str, tuple[Path, os.stat_result]]) -> None:
    seen: dict[tuple[int, int], str] = {}
    for label, (_, info) in files.items():
        inode = (info.st_dev, info.st_ino)
        if inode in seen:
            raise TrainingReceiptError(
                f"artifact inode alias is prohibited: {seen[inode]} and {label}"
            )
        seen[inode] = label


def _nonempty_line_count(path: Path) -> int:
    with open_stable_regular(Path("/"), path) as handle:
        return sum(1 for line in handle.read().decode("utf-8").splitlines() if line.strip())


def _validate_lineage_shape(value: Any, label: str, *, nullable: bool = False) -> Any:
    if value is None and nullable:
        return None
    if not isinstance(value, dict) or set(value) != LINEAGE_FIELDS:
        raise TrainingReceiptError(f"{label} has the wrong lineage fields")
    _require_hash(value.get("sha256"), f"{label}.sha256")
    _require_hash(
        value.get("receipt_identity_sha256"),
        f"{label}.receipt_identity_sha256",
    )
    for field in ("path", "status", "phase"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise TrainingReceiptError(f"{label}.{field} must be a nonempty string")
    return copy.deepcopy(value)


def _validate_identity(
    receipt: Mapping[str, Any], contract: TrainingReceiptContract, label: str
) -> None:
    _require_exact(receipt.get("schema_version"), contract.schema_version, f"{label}.schema_version")
    for field, expected in contract.identity.items():
        _require_exact(receipt.get(field), expected, f"{label}.{field}")


def _validate_embedded_proof(
    value: Any,
    *,
    status: str,
    identity_field: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingReceiptError(f"{label} is not an object")
    _require_exact(value.get("schema_version"), 1, f"{label}.schema_version")
    _require_exact(value.get("status"), status, f"{label}.status")
    claimed = _require_hash(value.get(identity_field), f"{label}.{identity_field}")
    payload = {key: item for key, item in value.items() if key != identity_field}
    _require_exact(claimed, _canonical_sha256(payload), f"{label} identity")
    return copy.deepcopy(value)


def _validate_cell_proof_shape(value: Any, cell: TrainingCell, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingReceiptError(f"{label} is not an object")
    required = {
        "cell", "stage", "capacity", "objective", "seed", "steps",
        "run_path", "tracked_run_path", "run_sha256", "receipt_identity_sha256",
        "train_metrics_sha256", "optimizer_steps_sha256", "checkpoint_path",
        "checkpoint_metadata_sha256", "checkpoint_identity_sha256",
        "adaptation_state_sha256", "loop_state_sha256", "setup_lineage",
        "stable_setup_sha256", "gate_lineages", "branch_authorization_lineage",
        "setup_barrier_identity_sha256", "prior_training_barrier_identity_sha256s",
        "training_launch_preflight_identity_sha256", "training_attempt_identity_sha256",
        "training_attempt_history_identity_sha256",
    }
    if set(value) != required:
        raise TrainingReceiptError(f"{label} fields changed")
    for field, expected in (
        ("cell", cell.slug), ("stage", cell.stage), ("capacity", cell.capacity),
        ("objective", cell.objective), ("seed", cell.seed),
    ):
        _require_exact(value.get(field), expected, f"{label}.{field}")
    for field in (
        "run_sha256", "receipt_identity_sha256", "train_metrics_sha256",
        "optimizer_steps_sha256", "checkpoint_metadata_sha256",
        "checkpoint_identity_sha256", "adaptation_state_sha256", "loop_state_sha256",
        "stable_setup_sha256", "setup_barrier_identity_sha256",
        "training_launch_preflight_identity_sha256", "training_attempt_identity_sha256",
        "training_attempt_history_identity_sha256",
    ):
        _require_hash(value.get(field), f"{label}.{field}")
    prior = value.get("prior_training_barrier_identity_sha256s")
    if not isinstance(prior, list) or any(
        not isinstance(item, str) or _SHA256.fullmatch(item) is None for item in prior
    ):
        raise TrainingReceiptError(f"{label} prior-barrier identities are malformed")
    return copy.deepcopy(value)


def _expected_setup_cells(stage: str) -> tuple[str, ...]:
    capacities = ("lora",) if stage == "A" else ("lora", "fullrank")
    return tuple(f"{capacity}_seed{seed}" for capacity in capacities for seed in TRAINING_SEEDS)


def _validate_setup_barrier_shape(
    value: Any,
    *,
    cell: TrainingCell,
    gates: Mapping[str, Any],
    stable_setup_sha256: str,
) -> dict[str, Any]:
    proof = _validate_embedded_proof(
        value,
        status="SETUP_BARRIER_COMPLETE",
        identity_field="barrier_identity_sha256",
        label="run.setup_barrier",
    )
    required = {
        "schema_version", "status", "stage", "cells", "root_lora_miss_lineage",
        "common_setup_invariant_sha256", "capacity_setup_invariant_sha256s",
        "barrier_identity_sha256",
    }
    if set(proof) != required:
        raise TrainingReceiptError("run.setup_barrier fields changed")
    _require_exact(proof.get("stage"), cell.stage, "run.setup_barrier.stage")
    rows = proof.get("cells")
    expected_slugs = _expected_setup_cells(cell.stage)
    if not isinstance(rows, list) or [
        row.get("cell") if isinstance(row, dict) else None for row in rows
    ] != list(expected_slugs):
        raise TrainingReceiptError("run.setup_barrier has the wrong ordered setup matrix")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "cell", "g0_lineage", "positive_control_lineage", "stable_setup_sha256"
        }:
            raise TrainingReceiptError(f"run.setup_barrier.cells[{index}] fields changed")
        _validate_lineage_shape(row["g0_lineage"], f"setup cell {index} G0")
        _validate_lineage_shape(
            row["positive_control_lineage"], f"setup cell {index} positive control"
        )
        _require_hash(row["stable_setup_sha256"], f"setup cell {index} stable setup")
    setup_key = f"{cell.capacity}_seed{cell.seed}"
    setup_match = next(row for row in rows if row["cell"] == setup_key)
    _require_exact(setup_match["g0_lineage"], gates["g0_lineage"], "setup-barrier G0")
    _require_exact(
        setup_match["positive_control_lineage"],
        gates["positive_control_lineage"],
        "setup-barrier positive control",
    )
    _require_exact(
        setup_match["stable_setup_sha256"], stable_setup_sha256, "setup-barrier stable setup"
    )
    _require_hash(proof.get("common_setup_invariant_sha256"), "setup common invariant")
    capacity_hashes = proof.get("capacity_setup_invariant_sha256s")
    expected_capacities = {"lora"} if cell.stage == "A" else {"lora", "fullrank"}
    if not isinstance(capacity_hashes, dict) or set(capacity_hashes) != expected_capacities:
        raise TrainingReceiptError("setup capacity invariants have the wrong keys")
    for capacity, digest in capacity_hashes.items():
        _require_hash(digest, f"setup {capacity} invariant")
    if cell.stage == "A":
        _require_exact(proof.get("root_lora_miss_lineage"), None, "Stage-A setup root")
    else:
        _validate_lineage_shape(proof.get("root_lora_miss_lineage"), "setup root LoRA miss")
    return proof


def _validate_training_barrier_shape(value: Any, expected_stage: str, label: str) -> dict[str, Any]:
    proof = _validate_embedded_proof(
        value,
        status="TRAINING_BARRIER_COMPLETE",
        identity_field="barrier_identity_sha256",
        label=label,
    )
    if set(proof) != {
        "schema_version", "status", "stage", "cells",
        "branch_authorization_lineage", "barrier_identity_sha256",
    }:
        raise TrainingReceiptError(f"{label} fields changed")
    _require_exact(proof.get("stage"), expected_stage, f"{label}.stage")
    cells = stage_matrix(expected_stage)
    rows = proof.get("cells")
    if not isinstance(rows, list) or len(rows) != len(cells):
        raise TrainingReceiptError(f"{label} has the wrong cell count")
    for index, (row, expected_cell) in enumerate(zip(rows, cells, strict=True)):
        _validate_cell_proof_shape(row, expected_cell, f"{label}.cells[{index}]")
    branch = proof.get("branch_authorization_lineage")
    if expected_stage == "A":
        _require_exact(branch, None, f"{label} Stage-A branch")
    else:
        _validate_lineage_shape(branch, f"{label} branch")
    return proof


def _validate_launch_shape(
    value: Any,
    *,
    cell: TrainingCell,
    branch: Mapping[str, Any] | None,
) -> dict[str, Any]:
    launch = _validate_embedded_proof(
        value,
        status="TRAINING_LAUNCH_PREFLIGHT_PASS",
        identity_field="preflight_identity_sha256",
        label="run.training_launch_preflight",
    )
    if set(launch) != {
        "schema_version", "status", "stage", "target", "target_state", "peers",
        "completed_peer_proofs", "trigger_outputs_absent",
        "branch_authorization_lineage", "preflight_identity_sha256",
    }:
        raise TrainingReceiptError("training launch proof fields changed")
    for field, expected in (
        ("stage", cell.stage), ("target", cell.slug), ("target_state", "ABSENT"),
        ("trigger_outputs_absent", True), ("branch_authorization_lineage", branch),
    ):
        _require_exact(launch.get(field), expected, f"training launch {field}")
    matrix = stage_matrix(cell.stage)
    target_index = matrix.index(cell)
    expected_peers = [peer for peer in matrix if peer != cell]
    peers = launch.get("peers")
    if not isinstance(peers, list) or len(peers) != len(expected_peers):
        raise TrainingReceiptError("training launch peer matrix changed")
    complete_slugs: list[str] = []
    for index, (row, peer) in enumerate(zip(peers, expected_peers, strict=True)):
        if not isinstance(row, dict) or set(row) != {"cell", "state"}:
            raise TrainingReceiptError(f"training launch peer {index} fields changed")
        _require_exact(row["cell"], peer.slug, f"training launch peer {index} cell")
        expected_state = "COMPLETE" if matrix.index(peer) < target_index else "ABSENT"
        _require_exact(
            row["state"], expected_state, f"training launch peer {index} state"
        )
        if row["state"] == "COMPLETE":
            complete_slugs.append(peer.slug)
    completed = launch.get("completed_peer_proofs")
    if not isinstance(completed, list) or [
        item.get("cell") if isinstance(item, dict) else None for item in completed
    ] != complete_slugs:
        raise TrainingReceiptError("training launch completed-peer proofs changed")
    by_slug = {peer.slug: peer for peer in expected_peers}
    for index, proof in enumerate(completed):
        _validate_cell_proof_shape(
            proof, by_slug[proof["cell"]], f"training launch completed peer {index}"
        )
    return launch


def _validate_complete_cell(
    root: Path,
    cell: TrainingCell,
    contract: TrainingReceiptContract,
    paths: TrainingCellPaths,
    *,
    allow_started_terminal: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    if contract.identity["phase"] != cell.phase:
        raise TrainingReceiptError(
            f"caller identity phase does not match cell: {contract.identity['phase']!r}"
        )
    if contract.identity["experiment_id"] != EXPERIMENT_ID:
        raise TrainingReceiptError("caller identity has the wrong experiment_id")

    for path in (
        paths.external_dir,
        paths.tracked_dir,
        paths.checkpoint_dir,
        paths.external_run,
        paths.tracked_run,
        paths.external_metrics,
        paths.tracked_metrics,
        paths.external_optimizer_steps,
        paths.tracked_optimizer_steps,
        paths.external_attempt_marker,
        paths.tracked_attempt_marker,
        paths.checkpoint_metadata,
        paths.adaptation_state,
        paths.loop_state,
    ):
        _require_no_symlink_components(root, path)

    _require_directory(paths.external_dir, "external training directory")
    _require_directory(paths.tracked_dir, "tracked training directory")
    _require_directory(paths.checkpoint_dir, "fixed-final checkpoint directory")
    _require_exact_members(
        paths.external_dir,
        {
            "run.json",
            "train_metrics.jsonl",
            "optimizer_steps.jsonl",
            ATTEMPT_MARKER_NAME,
            paths.checkpoint_dir.name,
        },
        "external training directory",
    )
    _require_exact_members(
        paths.tracked_dir,
        {"run.json", "train_metrics.jsonl", "optimizer_steps.jsonl", ATTEMPT_MARKER_NAME},
        "tracked training directory",
    )
    _require_exact_members(
        paths.checkpoint_dir,
        {"checkpoint.json", "adaptation_state.pt", "loop_state.pt"},
        "fixed-final checkpoint directory",
    )

    regular_paths = {
        "external run": paths.external_run,
        "tracked run": paths.tracked_run,
        "external metrics": paths.external_metrics,
        "tracked metrics": paths.tracked_metrics,
        "external optimizer steps": paths.external_optimizer_steps,
        "tracked optimizer steps": paths.tracked_optimizer_steps,
        "external attempt marker": paths.external_attempt_marker,
        "tracked attempt marker": paths.tracked_attempt_marker,
        "checkpoint metadata": paths.checkpoint_metadata,
        "adaptation state": paths.adaptation_state,
        "loop state": paths.loop_state,
    }
    snapshots = _snapshot_regular_files(root, regular_paths)

    for external_label, tracked_label, label in (
        ("external run", "tracked run", "run receipt"),
        ("external metrics", "tracked metrics", "training metrics"),
        ("external optimizer steps", "tracked optimizer steps", "optimizer-step log"),
        ("external attempt marker", "tracked attempt marker", "attempt marker"),
    ):
        external_snapshot = snapshots[external_label]
        tracked_snapshot = snapshots[tracked_label]
        if external_snapshot.raw != tracked_snapshot.raw:
            raise TrainingReceiptError(f"external/tracked {label} mirrors differ")
        if (external_snapshot.device, external_snapshot.inode) == (
            tracked_snapshot.device,
            tracked_snapshot.inode,
        ):
            raise TrainingReceiptError(
                f"external/tracked {label} mirrors share an inode"
            )

    run = _json_object_from_snapshot(snapshots["external run"], "external run")
    tracked_run = _json_object_from_snapshot(snapshots["tracked run"], "tracked run")
    if run != tracked_run:
        raise TrainingReceiptError("parsed external/tracked run receipts differ")
    _validate_identity(run, contract, "run")
    _require_exact(run.get("status"), contract.status, "run.status")
    claimed_run_identity = _require_hash(
        run.get("receipt_identity_sha256"), "run.receipt_identity_sha256"
    )
    run_without_identity = {
        key: value for key, value in run.items() if key != "receipt_identity_sha256"
    }
    if claimed_run_identity != _canonical_sha256(run_without_identity):
        raise TrainingReceiptError("run receipt self-hash mismatch")

    _require_exact(run.get("capacity"), cell.capacity, "run.capacity")
    _require_exact(run.get("objective"), cell.objective, "run.objective")
    _require_exact(run.get("model_seed"), cell.seed, "run.model_seed")
    _require_exact(run.get("steps"), contract.steps, "run.steps")
    terminal_access = {
        "authorizes_training": False,
        "authorizes_result_training": False,
        "authorizes_result_evaluation": False,
        "benchmark_files_read": 0,
        "result_payloads_opened": ["train"],
        "sealed_contrast_payloads_opened": [],
        "training_or_evaluation_started": True,
        "scientific_evidence": False,
    }
    for field, expected in terminal_access.items():
        _require_exact(run.get(field), expected, f"run.{field}")

    expected_paths = {
        "train_metrics_path": _repo_relative(paths.external_metrics, root),
        "optimizer_steps_path": _repo_relative(paths.external_optimizer_steps, root),
        "checkpoint_path": _repo_relative(paths.checkpoint_dir, root),
        "tracked_run_path": _repo_relative(paths.tracked_run, root),
        "tracked_metrics_path": _repo_relative(paths.tracked_metrics, root),
        "tracked_optimizer_steps_path": _repo_relative(
            paths.tracked_optimizer_steps, root
        ),
    }
    for field, expected in expected_paths.items():
        _require_exact(run.get(field), expected, f"run.{field}")

    metrics_sha256 = snapshots["external metrics"].sha256
    optimizer_sha256 = snapshots["external optimizer steps"].sha256
    _require_exact(run.get("train_metrics_sha256"), metrics_sha256, "run.train_metrics_sha256")
    _require_exact(
        run.get("optimizer_steps_sha256"),
        optimizer_sha256,
        "run.optimizer_steps_sha256",
    )
    metrics_rows = _nonempty_snapshot_line_count(
        snapshots["external metrics"], "external metrics"
    )
    optimizer_rows = _nonempty_snapshot_line_count(
        snapshots["external optimizer steps"], "external optimizer steps"
    )
    _require_exact(run.get("train_metrics_rows"), metrics_rows, "run.train_metrics_rows")
    _require_exact(
        run.get("optimizer_steps_rows"), optimizer_rows, "run.optimizer_steps_rows"
    )
    _require_exact(optimizer_rows, contract.steps, "optimizer-step row count")

    metadata_sha256 = snapshots["checkpoint metadata"].sha256
    _require_exact(
        run.get("checkpoint_metadata_sha256"),
        metadata_sha256,
        "run.checkpoint_metadata_sha256",
    )
    metadata = _json_object_from_snapshot(
        snapshots["checkpoint metadata"], "checkpoint metadata"
    )
    _validate_identity(metadata, contract, "checkpoint")
    _require_exact(metadata.get("capacity"), cell.capacity, "checkpoint.capacity")
    _require_exact(metadata.get("objective"), cell.objective, "checkpoint.objective")
    _require_exact(metadata.get("model_seed"), cell.seed, "checkpoint.model_seed")
    _require_exact(metadata.get("step"), contract.steps, "checkpoint.step")
    checkpoint_identity = _require_hash(
        metadata.get("checkpoint_identity_sha256"),
        "checkpoint.checkpoint_identity_sha256",
    )
    checkpoint_without_identity = {
        key: value
        for key, value in metadata.items()
        if key != "checkpoint_identity_sha256"
    }
    if checkpoint_identity != _canonical_sha256(checkpoint_without_identity):
        raise TrainingReceiptError("checkpoint metadata self-hash mismatch")
    _require_exact(
        run.get("checkpoint_identity_sha256"),
        checkpoint_identity,
        "run/checkpoint identity",
    )

    _require_exact(
        metadata.get("train_metrics_path"),
        expected_paths["train_metrics_path"],
        "checkpoint.train_metrics_path",
    )
    _require_exact(
        metadata.get("optimizer_steps_path"),
        expected_paths["optimizer_steps_path"],
        "checkpoint.optimizer_steps_path",
    )
    _require_exact(
        metadata.get("adaptation_state_sha256"),
        snapshots["adaptation state"].sha256,
        "checkpoint.adaptation_state_sha256",
    )
    _require_exact(
        metadata.get("loop_state_sha256"),
        snapshots["loop state"].sha256,
        "checkpoint.loop_state_sha256",
    )

    shared_fields = (
        "schema_version",
        *sorted(RUN_IDENTITY_FIELDS),
        "capacity",
        "objective",
        "model_seed",
        "data_manifest_sha256",
        "training_prompt_tokens",
        "training_layer_token_applications",
        "training_order_sha256",
        "dropout_schedule_sha256",
        "dropout_probes",
        "train_metrics_sha256",
        "train_metrics_rows",
        "train_metrics_path",
        "optimizer_steps_sha256",
        "optimizer_steps_rows",
        "optimizer_steps_path",
        "optimizer_state",
        "optimizer_step_receipt",
        "setup_barrier",
        "prior_training_barriers",
        "training_launch_preflight",
        "setup",
        "setup_sha256",
        "stable_setup",
        "training_attempt_authorization",
        "training_attempt_history",
        "training_attempt_journal_path",
    )
    for field in shared_fields:
        if field not in run or field not in metadata or run[field] != metadata[field]:
            raise TrainingReceiptError(f"run/checkpoint core field mismatch: {field}")

    setup = run.get("setup")
    if not isinstance(setup, dict):
        raise TrainingReceiptError("run.setup is not an object")
    _require_exact(
        run.get("setup_sha256"), _canonical_sha256(setup), "run.setup_sha256"
    )
    try:
        recomputed_stable_setup = stable_setup_receipt(setup)
    except RuntimeError as exc:
        raise TrainingReceiptError(f"run.setup stable projection is invalid: {exc}") from exc
    _require_exact(
        run.get("stable_setup"),
        recomputed_stable_setup,
        "run.stable_setup",
    )
    setup_lineage = setup.get("shared_initialization")
    if not isinstance(setup_lineage, dict):
        raise TrainingReceiptError("setup omits shared_initialization lineage")

    optimizer_receipt = run.get("optimizer_step_receipt")
    if not isinstance(optimizer_receipt, dict):
        raise TrainingReceiptError("optimizer_step_receipt is not an object")
    _require_exact(
        optimizer_receipt.get("steps"), contract.steps, "optimizer_step_receipt.steps"
    )
    _require_exact(
        optimizer_receipt.get("rows"), contract.steps, "optimizer_step_receipt.rows"
    )
    if not isinstance(run.get("optimizer_state"), dict):
        raise TrainingReceiptError("optimizer_state is not an object")

    gates = {
        "g0_lineage": _validate_lineage_shape(
            metadata.get("g0_lineage"), "checkpoint.g0_lineage"
        ),
        "positive_control_lineage": _validate_lineage_shape(
            metadata.get("positive_control_lineage"),
            "checkpoint.positive_control_lineage",
        ),
    }
    branch = _validate_lineage_shape(
        metadata.get("branch_authorization_lineage"),
        "checkpoint.branch_authorization_lineage",
        nullable=True,
    )
    for field, expected in (
        ("g0_lineage", gates["g0_lineage"]),
        ("positive_control_lineage", gates["positive_control_lineage"]),
        ("branch_authorization_lineage", branch),
    ):
        _require_exact(run.get(field), expected, f"run/checkpoint {field}")

    setup_barrier = _validate_setup_barrier_shape(
        run.get("setup_barrier"),
        cell=cell,
        gates=gates,
        stable_setup_sha256=_canonical_sha256(recomputed_stable_setup),
    )

    prior = run.get("prior_training_barriers")
    if not isinstance(prior, list):
        raise TrainingReceiptError("run.prior_training_barriers is not a list")
    expected_prior_stages = {"A": (), "B": ("A",), "C": ("A", "B")}[cell.stage]
    if len(prior) != len(expected_prior_stages):
        raise TrainingReceiptError("run.prior_training_barriers has the wrong length")
    for index, expected_stage in enumerate(expected_prior_stages):
        _validate_training_barrier_shape(
            prior[index], expected_stage, f"run.prior_training_barriers[{index}]"
        )

    launch = _validate_launch_shape(
        run.get("training_launch_preflight"), cell=cell, branch=branch
    )

    attempt_authorization = validate_attempt_authorization(
        run.get("training_attempt_authorization"), attempt_kind="training"
    )
    expected_attempt_cell = {
        "stage": cell.stage,
        "capacity": cell.capacity,
        "objective": cell.objective,
        "seed": cell.seed,
        "slug": cell.slug,
    }
    _require_exact(
        attempt_authorization.get("cell"), expected_attempt_cell, "training attempt cell"
    )
    expected_attempt_paths = [
        _repo_relative(paths.external_dir, root),
        _repo_relative(paths.tracked_dir, root),
    ]
    _require_exact(
        attempt_authorization.get("canonical_paths"),
        expected_attempt_paths,
        "training attempt paths",
    )
    expected_attempt_context = {
        "setup_barrier_identity_sha256": setup_barrier["barrier_identity_sha256"],
        "prior_training_barrier_identity_sha256s": [
            proof["barrier_identity_sha256"] for proof in prior
        ],
        "training_launch_preflight_identity_sha256": launch[
            "preflight_identity_sha256"
        ],
        "training_launch_peer_vector": copy.deepcopy(launch["peers"]),
        "branch_authorization_lineage": branch,
    }
    _require_exact(
        attempt_authorization.get("context"),
        expected_attempt_context,
        "training attempt context",
    )
    expected_journal_path = _repo_relative(paths.attempt_journal, root)
    _require_exact(
        run.get("training_attempt_journal_path"),
        expected_journal_path,
        "training attempt journal path",
    )
    attempt_header = {
        key: value for key, value in contract.identity.items() if key != "phase"
    }
    attempt_cell = expected_attempt_cell
    run_lineage = {
        "path": _repo_relative(paths.external_run, root),
        "sha256": snapshots["external run"].sha256,
        "receipt_identity_sha256": claimed_run_identity,
    }
    try:
        attempt_graph = validate_training_attempt_for_terminal(
            root,
            slug=cell.slug,
            header=attempt_header,
            cell=attempt_cell,
            canonical_paths=expected_attempt_paths,
            authorization=attempt_authorization,
            external_marker=paths.external_attempt_marker,
            tracked_marker=paths.tracked_attempt_marker,
            run_lineage=run_lineage,
            expected_archive_header={
                "schema_version": 1,
                "status": "FAILED_ATTEMPT_ARCHIVED",
                "experiment_id": contract.identity["experiment_id"],
                "model_id": contract.identity["model_id"],
                "model_revision": contract.identity["model_revision"],
                "backend": contract.identity["backend"],
                "config_sha256": contract.identity["config_sha256"],
                "source_contract_sha256": contract.identity[
                    "source_contract_sha256"
                ],
                "requirements_training_lock_sha256": contract.identity[
                    "requirements_training_lock_sha256"
                ],
                "scientific_evidence": False,
            },
            expected_history=run.get("training_attempt_history"),
            require_complete=not allow_started_terminal,
            external_marker_snapshot={
                "raw": snapshots["external attempt marker"].raw,
                "device": snapshots["external attempt marker"].device,
                "inode": snapshots["external attempt marker"].inode,
            },
            tracked_marker_snapshot={
                "raw": snapshots["tracked attempt marker"].raw,
                "device": snapshots["tracked attempt marker"].device,
                "inode": snapshots["tracked attempt marker"].inode,
            },
        )
    except AttemptReceiptError as exc:
        raise TrainingReceiptError(f"training attempt graph is invalid: {exc}") from exc

    proof = {
        "cell": cell.slug,
        "stage": cell.stage,
        "capacity": cell.capacity,
        "objective": cell.objective,
        "seed": cell.seed,
        "steps": contract.steps,
        "run_path": _repo_relative(paths.external_run, root),
        "tracked_run_path": _repo_relative(paths.tracked_run, root),
        "run_sha256": snapshots["external run"].sha256,
        "receipt_identity_sha256": claimed_run_identity,
        "train_metrics_sha256": metrics_sha256,
        "optimizer_steps_sha256": optimizer_sha256,
        "checkpoint_path": _repo_relative(paths.checkpoint_dir, root),
        "checkpoint_metadata_sha256": metadata_sha256,
        "checkpoint_identity_sha256": checkpoint_identity,
        "adaptation_state_sha256": metadata["adaptation_state_sha256"],
        "loop_state_sha256": metadata["loop_state_sha256"],
        "setup_lineage": copy.deepcopy(setup_lineage),
        "stable_setup_sha256": _canonical_sha256(run["stable_setup"]),
        "gate_lineages": copy.deepcopy(gates),
        "branch_authorization_lineage": copy.deepcopy(branch),
        "setup_barrier_identity_sha256": setup_barrier["barrier_identity_sha256"],
        "prior_training_barrier_identity_sha256s": [
            proof["barrier_identity_sha256"] for proof in prior
        ],
        "training_launch_preflight_identity_sha256": launch[
            "preflight_identity_sha256"
        ],
        "training_attempt_identity_sha256": attempt_authorization[
            "attempt_identity_sha256"
        ],
        "training_attempt_history_identity_sha256": attempt_graph["history"][
            "history_identity_sha256"
        ],
    }
    # Recheck directory closure after every leaf snapshot and graph edge has
    # been consumed.  A concurrent extra member must not sneak in after the
    # initial membership gate while leaving all registered inodes unchanged.
    _require_exact_members(
        paths.external_dir,
        {
            "run.json",
            "train_metrics.jsonl",
            "optimizer_steps.jsonl",
            ATTEMPT_MARKER_NAME,
            paths.checkpoint_dir.name,
        },
        "external training directory",
    )
    _require_exact_members(
        paths.tracked_dir,
        {"run.json", "train_metrics.jsonl", "optimizer_steps.jsonl", ATTEMPT_MARKER_NAME},
        "tracked training directory",
    )
    _require_exact_members(
        paths.checkpoint_dir,
        {"checkpoint.json", "adaptation_state.pt", "loop_state.pt"},
        "fixed-final checkpoint directory",
    )
    return proof, copy.deepcopy(setup_lineage), gates, branch, run


def classify_training_cell(
    repo_root: Path | str,
    cell: TrainingCell,
    contract: TrainingReceiptContract,
    *,
    allow_started_terminal: bool = False,
) -> TrainingCellAudit:
    """Classify a cell as ABSENT, COMPLETE, or fail-closed INCOMPLETE."""

    try:
        root = _canonical_repo_root(repo_root)
        paths = canonical_training_cell_paths(root, cell, steps=contract.steps)
    except (OSError, ValueError, TrainingReceiptError) as exc:
        return TrainingCellAudit(cell, TrainingCellState.INCOMPLETE, (str(exc),))

    external_present = _lexists(paths.external_dir)
    tracked_present = _lexists(paths.tracked_dir)
    if not external_present and not tracked_present:
        return TrainingCellAudit(cell, TrainingCellState.ABSENT, ())
    try:
        proof, setup, gates, branch, _ = _validate_complete_cell(
            root,
            cell,
            contract,
            paths,
            allow_started_terminal=allow_started_terminal,
        )
    except (OSError, UnicodeError, ValueError, TypeError, TrainingReceiptError) as exc:
        return TrainingCellAudit(
            cell, TrainingCellState.INCOMPLETE, (str(exc) or type(exc).__name__,)
        )
    return TrainingCellAudit(
        cell,
        TrainingCellState.COMPLETE,
        (),
        MappingProxyType(proof),
        MappingProxyType(setup),
        MappingProxyType(gates),
        MappingProxyType(branch) if branch is not None else None,
    )


def recover_published_training_completion(
    repo_root: Path | str,
    cell: TrainingCell,
    contract: TrainingReceiptContract,
    *,
    required_authorization: Mapping[str, Any] | None = None,
    expected_setup_barrier_identity_sha256: str | None = None,
    expected_prior_training_barrier_identity_sha256s: list[str] | None = None,
) -> bool:
    """Finalize the one safe crash window after exact terminal publication.

    A producer crash can occur after both immutable ``run.json`` mirrors are
    durable but before the journal transition from STARTED to COMPLETE.  That
    state is not scientific completion, and it is not a failed attempt that
    should be replayed.  Reopen the entire terminal graph while permitting
    only that journal state, bind the exact published run lineage, durably
    complete the journal, then require the ordinary fail-closed graph to pass.
    """

    try:
        root = _canonical_repo_root(repo_root)
        paths = canonical_training_cell_paths(root, cell, steps=contract.steps)
    except (OSError, ValueError, TrainingReceiptError):
        return False
    if not (_lexists(paths.external_dir) and _lexists(paths.tracked_dir)):
        return False
    try:
        proof, _, _, branch, run = _validate_complete_cell(
            root,
            cell,
            contract,
            paths,
            allow_started_terminal=True,
        )
        if cell.stage == "A":
            if required_authorization is not None or branch is not None:
                return False
        elif (
            required_authorization is None
            or dict(branch or {}) != dict(required_authorization)
        ):
            return False
        if (
            expected_setup_barrier_identity_sha256 is not None
            and proof["setup_barrier_identity_sha256"]
            != expected_setup_barrier_identity_sha256
        ):
            return False
        if (
            expected_prior_training_barrier_identity_sha256s is not None
            and proof["prior_training_barrier_identity_sha256s"]
            != expected_prior_training_barrier_identity_sha256s
        ):
            return False
        authorization = validate_attempt_authorization(
            run.get("training_attempt_authorization"), attempt_kind="training"
        )
        header = {
            key: value for key, value in contract.identity.items() if key != "phase"
        }
        cell_payload = {
            "stage": cell.stage,
            "capacity": cell.capacity,
            "objective": cell.objective,
            "seed": cell.seed,
            "slug": cell.slug,
        }
        canonical_paths = [
            _repo_relative(paths.external_dir, root),
            _repo_relative(paths.tracked_dir, root),
        ]
        journal = load_training_journal(
            root,
            cell.slug,
            header=header,
            cell=cell_payload,
            canonical_paths=canonical_paths,
        )
        if journal is None or journal["events"][-1]["authorization"] != authorization:
            return False
        if journal["events"][-1]["state"] != "STARTED":
            return False
        run_lineage = {
            "path": _repo_relative(paths.external_run, root),
            "sha256": proof["run_sha256"],
            "receipt_identity_sha256": run["receipt_identity_sha256"],
        }
        complete_training_attempt(
            root,
            slug=cell.slug,
            header=header,
            cell=cell_payload,
            canonical_paths=canonical_paths,
            authorization=authorization,
            terminal_run_lineage=run_lineage,
        )
        # No success is returned until the normal (COMPLETE-required) graph
        # reopens after the durable journal transition.
        _validate_complete_cell(root, cell, contract, paths)
    except (
        AttemptReceiptError,
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        TrainingReceiptError,
    ):
        return False
    return True


def _contract_for(
    contracts: Mapping[Any, TrainingReceiptContract], cell: TrainingCell
) -> TrainingReceiptContract:
    if cell in contracts:
        contract = contracts[cell]
    elif cell.slug in contracts:
        contract = contracts[cell.slug]
    else:
        raise TrainingReceiptError(f"missing caller contract for {cell.slug}")
    if not isinstance(contract, TrainingReceiptContract):
        raise TrainingReceiptError(f"invalid caller contract for {cell.slug}")
    return contract


def _stage_authorization(
    stage: str,
    audits: tuple[TrainingCellAudit, ...],
    required_authorization: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    branches = [audit.branch_authorization_lineage for audit in audits]
    if stage == "A":
        if any(branch is not None for branch in branches):
            raise TrainingReceiptError("Stage A training unexpectedly has branch authorization")
        if required_authorization is not None:
            raise TrainingReceiptError("Stage A cannot require branch authorization")
        return None
    if any(branch is None for branch in branches):
        raise TrainingReceiptError(f"Stage {stage} training omits branch authorization")
    first = dict(branches[0] or {})
    if any(dict(branch or {}) != first for branch in branches[1:]):
        raise TrainingReceiptError(f"Stage {stage} cells do not share one authorization")
    if required_authorization is not None and first != dict(required_authorization):
        raise TrainingReceiptError(f"Stage {stage} authorization differs from the caller contract")
    return first


def training_barrier(
    repo_root: Path | str,
    stage: str,
    contracts: Mapping[Any, TrainingReceiptContract],
    *,
    required_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove that every training cell in a reached stage is immutably complete."""

    cells = stage_matrix(stage)
    stage_name = cells[0].stage
    audits = tuple(
        classify_training_cell(repo_root, cell, _contract_for(contracts, cell))
        for cell in cells
    )
    failures = [
        f"{audit.cell.slug}={audit.state.value}: {'; '.join(audit.errors)}"
        for audit in audits
        if not audit.complete
    ]
    if failures:
        raise TrainingReceiptError(
            f"Stage {stage_name} training barrier is incomplete: " + " | ".join(failures)
        )
    authorization = _stage_authorization(
        stage_name, audits, required_authorization
    )
    proof: dict[str, Any] = {
        "schema_version": 1,
        "status": "TRAINING_BARRIER_COMPLETE",
        "stage": stage_name,
        "cells": [dict(audit.proof or {}) for audit in audits],
        "branch_authorization_lineage": copy.deepcopy(authorization),
    }
    proof["barrier_identity_sha256"] = _canonical_sha256(proof)
    return proof


def evaluation_barrier(
    repo_root: Path | str,
    stage: str,
    contracts: Mapping[Any, TrainingReceiptContract],
    *,
    required_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require the whole current-stage training matrix before any evaluation."""

    return training_barrier(
        repo_root,
        stage,
        contracts,
        required_authorization=required_authorization,
    )


def training_launch_preflight(
    repo_root: Path | str,
    target: TrainingCell,
    contracts: Mapping[Any, TrainingReceiptContract],
    *,
    target_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authorize only the next cell in the frozen stage-matrix order.

    The exact matrix prefix before the target must be COMPLETE, the target and
    exact suffix must be ABSENT, and no result-evaluation output for the current
    stage may exist.  A marker-only PREPARED target is recoverably ABSENT because
    no payload has yet been opened.  Stages B/C additionally bind every completed
    peer and the new target to the same branch authorization.
    """

    cells = stage_matrix(target.stage)
    if target not in cells:
        raise TrainingReceiptError(f"target is not registered in Stage {target.stage}")
    target_contract = _contract_for(contracts, target)
    target_audit = classify_training_cell(repo_root, target, target_contract)
    root = _canonical_repo_root(repo_root)
    target_paths = canonical_training_cell_paths(
        root, target, steps=target_contract.steps
    )
    target_header = {
        key: value for key, value in target_contract.identity.items() if key != "phase"
    }
    target_cell = {
        "stage": target.stage,
        "capacity": target.capacity,
        "objective": target.objective,
        "seed": target.seed,
        "slug": target.slug,
    }
    target_canonical_paths = [
        _repo_relative(target_paths.external_dir, root),
        _repo_relative(target_paths.tracked_dir, root),
    ]
    target_marker_only = False
    if target_audit.state is TrainingCellState.INCOMPLETE:
        try:
            target_marker_only = training_attempt_is_marker_only(
                root,
                slug=target.slug,
                header=target_header,
                cell=target_cell,
                canonical_paths=target_canonical_paths,
            )
        except (OSError, ValueError, AttemptReceiptError):
            target_marker_only = False
    if target_audit.state is not TrainingCellState.ABSENT and not target_marker_only:
        raise TrainingReceiptError(
            f"training target must be ABSENT, got {target_audit.state.value}: "
            + "; ".join(target_audit.errors)
        )

    target_index = cells.index(target)
    peer_audits = []
    for peer_index, peer in enumerate(cells):
        paths = canonical_training_cell_paths(
            root, peer, steps=_contract_for(contracts, peer).steps
        )
        _require_no_symlink_components(root, paths.trigger_output)
        if _lexists(paths.trigger_output):
            raise TrainingReceiptError(
                f"trigger output exists before stage training is sealed: {paths.trigger_output}"
            )
        if peer == target:
            continue
        audit = classify_training_cell(root, peer, _contract_for(contracts, peer))
        if audit.state is TrainingCellState.INCOMPLETE:
            raise TrainingReceiptError(
                f"training peer is INCOMPLETE: {peer.slug}: {'; '.join(audit.errors)}"
            )
        required_state = (
            TrainingCellState.COMPLETE
            if peer_index < target_index
            else TrainingCellState.ABSENT
        )
        if audit.state is not required_state:
            raise TrainingReceiptError(
                f"out-of-order training launch: {peer.slug} must be "
                f"{required_state.value} before {target.slug}, got {audit.state.value}"
            )
        peer_audits.append(audit)

    completed = tuple(
        audit for audit in peer_audits if audit.state is TrainingCellState.COMPLETE
    )
    if target.stage == "A":
        if target_authorization is not None:
            raise TrainingReceiptError("Stage A target cannot have branch authorization")
        for audit in completed:
            if audit.branch_authorization_lineage is not None:
                raise TrainingReceiptError("Stage A peer unexpectedly has branch authorization")
    else:
        if target_authorization is None:
            raise TrainingReceiptError(
                f"Stage {target.stage} target requires explicit branch authorization"
            )
        for audit in completed:
            if dict(audit.branch_authorization_lineage or {}) != dict(target_authorization):
                raise TrainingReceiptError(
                    f"Stage {target.stage} peer authorization differs from target"
                )

    proof: dict[str, Any] = {
        "schema_version": 1,
        "status": "TRAINING_LAUNCH_PREFLIGHT_PASS",
        "stage": target.stage,
        "target": target.slug,
        "target_state": TrainingCellState.ABSENT.value,
        "peers": [
            {"cell": audit.cell.slug, "state": audit.state.value}
            for audit in peer_audits
        ],
        "completed_peer_proofs": [
            dict(audit.proof or {}) for audit in completed
        ],
        "trigger_outputs_absent": True,
        "branch_authorization_lineage": copy.deepcopy(target_authorization),
    }
    proof["preflight_identity_sha256"] = _canonical_sha256(proof)
    return proof


# Short aliases retained for call sites that speak in terms of completion proofs.
TrainingState = TrainingCellState
classify_training_completion = classify_training_cell
prove_training_barrier = training_barrier
require_evaluation_barrier = evaluation_barrier
