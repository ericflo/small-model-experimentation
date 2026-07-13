#!/usr/bin/env python3
"""Durably archive one complete, source-invalidated canonical setup."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from contextlib import ExitStack
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    G0_COMPLETED_CHECKS,
    G0_FAILURE_STAGE_PREFIX_LENGTHS,
    MODEL_ID,
    MODEL_REVISION,
    config_sha256,
    load_config,
    source_contract_sha256,
)
from src.attempt_receipts import locked_regular, tree_manifest  # noqa: E402
from src.safe_io import (  # noqa: E402
    StableArtifactError,
    ensure_canonical_directory,
    fsync_canonical_directory,
    move_new_entry,
    open_stable_directory_for_update,
    open_stable_regular,
    open_stable_regular_for_update,
    publish_new_bytes,
    rename_new_entry,
)


SHA256_RE = re.compile(r"[0-9a-f]{64}")
G0_RE = re.compile(r"g0_(lora|fullrank)_seed([0-9]+)\.json")
POSITIVE_CONTROL_RE = re.compile(
    r"positive_control_(lora|fullrank)_seed([0-9]+)\.json"
)
INITIALIZATION_PAYLOAD_RE = re.compile(
    r"initialization_seed([0-9]+)\.pt(?:\.json)?"
)
INITIALIZATION_RECEIPT_RE = re.compile(r"initialization_seed([0-9]+)\.json")
DATA_PAYLOADS = (
    "contrast_depth.jsonl.gz",
    "contrast_joint.jsonl.gz",
    "contrast_validation.jsonl.gz",
    "depth_extrapolation.jsonl.gz",
    "joint_holdout.jsonl.gz",
    "train.jsonl.gz",
    "validation.jsonl.gz",
)
SEALED_SPLITS = (
    "contrast_depth",
    "contrast_joint",
    "contrast_validation",
)
DATA_SETUP_FILES = (
    "contrast_access_ledger.json",
    *DATA_PAYLOADS,
    "manifest.json",
)
G0_PASS_ACCESS_CONTRACT = {
    "authorizes_positive_control": True,
    "authorizes_training": False,
    "authorizes_result_training": False,
    "authorizes_result_evaluation": False,
    "benchmark_files_read": 0,
    "result_payloads_opened": ["train"],
    "sealed_contrast_payloads_opened": [],
    "training_or_evaluation_started": False,
    "scientific_evidence": False,
}


@dataclass(frozen=True)
class ArchiveItem:
    source: Path
    archive_path: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _absolute_no_follow(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _safe_repo_path(path: Path, label: str) -> Path:
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
    absolute = _safe_repo_path(path, "setup path")
    return absolute.relative_to(_absolute_no_follow(REPO_ROOT)).as_posix()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_identity(payload: dict[str, Any], label: str) -> None:
    claimed = payload.get("receipt_identity_sha256")
    if not isinstance(claimed, str) or not SHA256_RE.fullmatch(claimed):
        raise RuntimeError(f"{label} lacks a canonical receipt identity")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "receipt_identity_sha256"
    }
    actual = _canonical_sha256(unsigned)
    if claimed != actual:
        raise RuntimeError(
            f"{label} receipt identity mismatch: claimed {claimed}, actual {actual}"
        )


def _expect(payload: dict[str, Any], key: str, expected: Any, label: str) -> None:
    actual = payload.get(key)
    if actual != expected:
        raise RuntimeError(
            f"{label} {key} mismatch: expected {expected!r}, found {actual!r}"
        )


def _require_exact_directory(
    path: Path,
    expected_names: set[str],
    label: str,
) -> None:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} directory is missing or unsafe: {path}")
    children = list(path.iterdir())
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise RuntimeError(f"{label} contains a non-file or symlink")
    actual_names = {child.name for child in children}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unknown = sorted(actual_names - expected_names)
        raise RuntimeError(
            f"{label} inventory is partial or unknown; "
            f"missing={missing}, unknown={unknown}"
        )


def _large_root(config: dict[str, Any]) -> Path:
    return _safe_repo_path(
        ROOT / config["paths"]["large_artifacts_dir"],
        "external setup root",
    )


def _inventory(config: dict[str, Any]) -> list[ArchiveItem]:
    seeds = tuple(map(int, config["training"]["train_seeds"]))
    data_dir = _safe_repo_path(
        ROOT / config["paths"]["data_dir"],
        "generated-data setup root",
    )
    _require_exact_directory(
        data_dir,
        {".gitignore", *DATA_SETUP_FILES},
        "generated-data setup",
    )

    runs_dir = _safe_repo_path(
        ROOT / config["paths"]["runs_dir"],
        "tracked runs root",
    )
    cpu_dir = _safe_repo_path(runs_dir / "cpu_smoke", "CPU setup receipt root")
    _require_exact_directory(cpu_dir, {"receipt.json"}, "CPU setup receipt")

    setup_dir = _safe_repo_path(runs_dir / "setup", "tracked setup receipt root")
    if setup_dir.is_symlink() or not setup_dir.is_dir():
        raise RuntimeError("tracked setup receipt directory is missing or unsafe")
    tracked_initialization = {f"initialization_seed{seed}.json" for seed in seeds}
    g0_names: set[str] = set()
    positive_control_names: set[str] = set()
    unknown_setup: list[str] = []
    for child in setup_dir.iterdir():
        match = G0_RE.fullmatch(child.name)
        if child.is_symlink() or not child.is_file():
            unknown_setup.append(child.name)
        elif child.name in tracked_initialization:
            continue
        elif match is not None and int(match.group(2)) in seeds:
            g0_names.add(child.name)
        elif (
            (control_match := POSITIVE_CONTROL_RE.fullmatch(child.name)) is not None
            and int(control_match.group(2)) in seeds
        ):
            positive_control_names.add(child.name)
        else:
            unknown_setup.append(child.name)
    missing_initialization = sorted(
        name for name in tracked_initialization if not (setup_dir / name).is_file()
    )
    if missing_initialization or unknown_setup:
        raise RuntimeError(
            "tracked setup inventory is partial or unknown; "
            f"missing={missing_initialization}, unknown={sorted(unknown_setup)}"
        )

    large_root = _large_root(config)
    if large_root.is_symlink() or not large_root.is_dir():
        raise RuntimeError("external setup root is missing or unsafe")
    external_names = {
        name
        for child in large_root.iterdir()
        if child.is_file() and name_is_initialization(name := child.name)
    }
    unsafe_initialization = [
        child.name
        for child in large_root.iterdir()
        if name_is_initialization(child.name)
        and (child.is_symlink() or not child.is_file())
    ]
    expected_external = {
        suffix
        for seed in seeds
        for suffix in (
            f"initialization_seed{seed}.pt",
            f"initialization_seed{seed}.pt.json",
        )
    }
    if external_names != expected_external or unsafe_initialization:
        raise RuntimeError(
            "external initialization inventory is partial or unknown; "
            f"missing={sorted(expected_external - external_names)}, "
            f"unknown={sorted(external_names - expected_external)}, "
            f"unsafe={sorted(unsafe_initialization)}"
        )

    items = [
        ArchiveItem(data_dir / name, f"data_generated/{name}")
        for name in DATA_SETUP_FILES
    ]
    for seed in seeds:
        for suffix in (
            f"initialization_seed{seed}.pt",
            f"initialization_seed{seed}.pt.json",
        ):
            items.append(ArchiveItem(large_root / suffix, f"initialization/{suffix}"))
    items.append(
        ArchiveItem(cpu_dir / "receipt.json", "tracked_receipts/cpu_smoke_receipt.json")
    )
    items.extend(
        ArchiveItem(setup_dir / name, f"tracked_receipts/{name}")
        for name in sorted(
            tracked_initialization | g0_names | positive_control_names
        )
    )
    return sorted(items, key=lambda item: item.archive_path)


def name_is_initialization(name: str) -> bool:
    return name.startswith("initialization_seed")


def _validate_common_binding(
    payload: dict[str, Any],
    *,
    source: str,
    config_digest: str,
    experiment_id: str,
    label: str,
) -> None:
    _expect(payload, "source_contract_sha256", source, label)
    _expect(payload, "config_sha256", config_digest, label)
    _expect(payload, "experiment_id", experiment_id, label)
    _expect(payload, "model_id", MODEL_ID, label)
    _expect(payload, "model_revision", MODEL_REVISION, label)


def _validate_manifest(
    path: Path,
    *,
    source: str,
    config_digest: str,
    experiment_id: str,
) -> tuple[dict[str, Any], str]:
    payload = _load_json(path, "data manifest")
    _validate_common_binding(
        payload,
        source=source,
        config_digest=config_digest,
        experiment_id=experiment_id,
        label="data manifest",
    )
    _expect(payload, "benchmark_files_read", 0, "data manifest")
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != {
        name.removesuffix(".jsonl.gz") for name in DATA_PAYLOADS
    }:
        raise RuntimeError("data manifest has a partial or unknown payload inventory")
    for split, record in files.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"data manifest split {split} is not an object")
        expected_name = f"{split}.jsonl.gz"
        _expect(record, "path", expected_name, f"data manifest split {split}")
        payload_path = path.parent / expected_name
        _expect(record, "bytes", payload_path.stat().st_size, f"data manifest split {split}")
        _expect(record, "sha256", _sha256(payload_path), f"data manifest split {split}")
    return payload, _sha256(path)


def _validate_ledger(
    path: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    experiment_id: str,
) -> None:
    payload = _load_json(path, "contrast access ledger")
    _validate_identity(payload, "contrast access ledger")
    _expect(payload, "experiment_id", experiment_id, "contrast access ledger")
    _expect(
        payload,
        "data_manifest_sha256",
        manifest_sha256,
        "contrast access ledger",
    )
    if payload.get("events") != []:
        raise RuntimeError("contrast access ledger is not empty; sealed data was accessed")
    sealed = payload.get("sealed_splits")
    if not isinstance(sealed, dict) or set(sealed) != set(SEALED_SPLITS):
        raise RuntimeError("contrast access ledger sealed inventory is partial or unknown")
    for split in SEALED_SPLITS:
        record = sealed[split]
        expected = manifest["files"][split]
        if not isinstance(record, dict):
            raise RuntimeError(f"contrast access ledger split {split} is not an object")
        for key in ("path", "bytes", "sha256"):
            _expect(record, key, expected[key], f"contrast access ledger split {split}")


def _validate_cpu_receipt(
    path: Path,
    *,
    source: str,
    config_digest: str,
    experiment_id: str,
) -> None:
    payload = _load_json(path, "CPU setup receipt")
    _expect(payload, "status", "CPU_SMOKE_PASS", "CPU setup receipt")
    _expect(payload, "scientific_evidence", False, "CPU setup receipt")
    nested = payload.get("config")
    if not isinstance(nested, dict):
        raise RuntimeError("CPU setup receipt lacks its config binding")
    _validate_common_binding(
        nested,
        source=source,
        config_digest=config_digest,
        experiment_id=experiment_id,
        label="CPU setup receipt config",
    )
    _expect(nested, "backend", "transformers", "CPU setup receipt config")


def _validate_initialization_receipts(
    config: dict[str, Any],
    *,
    source: str,
    config_digest: str,
    experiment_id: str,
) -> dict[int, dict[str, Any]]:
    large_root = _large_root(config)
    setup_dir = _safe_repo_path(
        ROOT / config["paths"]["runs_dir"] / "setup",
        "tracked setup receipt root",
    )
    receipts: dict[int, dict[str, Any]] = {}
    for seed in map(int, config["training"]["train_seeds"]):
        bundle = large_root / f"initialization_seed{seed}.pt"
        external = bundle.with_suffix(".pt.json")
        tracked = setup_dir / f"initialization_seed{seed}.json"
        if (
            bundle.is_symlink()
            or not bundle.is_file()
            or external.is_symlink()
            or not external.is_file()
            or tracked.is_symlink()
            or not tracked.is_file()
        ):
            raise RuntimeError(f"seed {seed} initialization artifacts are unsafe")
        if external.read_bytes() != tracked.read_bytes():
            raise RuntimeError(f"seed {seed} initialization sidecar and mirror differ")
        payload = _load_json(tracked, f"seed {seed} initialization receipt")
        _validate_identity(payload, f"seed {seed} initialization receipt")
        _expect(
            payload,
            "status",
            "SHARED_INITIALIZATION_PREPARED",
            f"seed {seed} initialization receipt",
        )
        _expect(
            payload,
            "bundle_path",
            _repo_relative(bundle),
            f"seed {seed} initialization receipt",
        )
        _expect(
            payload,
            "bundle_sha256",
            _sha256(bundle),
            f"seed {seed} initialization receipt",
        )
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"seed {seed} initialization receipt lacks metadata")
        _validate_identity(metadata, f"seed {seed} initialization metadata")
        _validate_common_binding(
            metadata,
            source=source,
            config_digest=config_digest,
            experiment_id=experiment_id,
            label=f"seed {seed} initialization metadata",
        )
        _expect(metadata, "model_seed", seed, f"seed {seed} initialization metadata")
        receipts[seed] = payload
    return receipts


def _validate_g0_receipts(
    config: dict[str, Any],
    initialization: dict[int, dict[str, Any]],
    *,
    source: str,
    config_digest: str,
    experiment_id: str,
    manifest_sha256: str,
) -> None:
    runs_dir = _safe_repo_path(
        ROOT / config["paths"]["runs_dir"],
        "tracked runs root",
    )
    setup_dir = _safe_repo_path(runs_dir / "setup", "tracked setup receipt root")
    failures_dir = _safe_repo_path(runs_dir / "failures", "tracked failures root")
    for path in sorted(setup_dir.glob("g0_*.json")):
        match = G0_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"unknown canonical G0 receipt name: {path.name}")
        capacity, seed_text = match.groups()
        seed = int(seed_text)
        payload = _load_json(path, f"G0 receipt {path.name}")
        _validate_identity(payload, f"G0 receipt {path.name}")
        _validate_common_binding(
            payload,
            source=source,
            config_digest=config_digest,
            experiment_id=experiment_id,
            label=f"G0 receipt {path.name}",
        )
        status = payload.get("status")
        if status not in {"MODEL_SMOKE_PASS", "SETUP_CONTROL_FAILED"}:
            raise RuntimeError(f"G0 receipt {path.name} has invalid status")
        _expect(payload, "phase", f"{capacity}_g0", f"G0 receipt {path.name}")
        _expect(payload, "backend", "transformers", f"G0 receipt {path.name}")
        _expect(payload, "capacity", capacity, f"G0 receipt {path.name}")
        _expect(payload, "model_seed", seed, f"G0 receipt {path.name}")
        if status == "SETUP_CONTROL_FAILED":
            failure_stage = payload.get("failure_stage")
            completed_checks = payload.get("completed_checks")
            if not isinstance(failure_stage, str) or (
                failure_stage not in G0_FAILURE_STAGE_PREFIX_LENGTHS
            ):
                raise RuntimeError(f"G0 failure {path.name} has unknown failure stage")
            if not isinstance(completed_checks, list):
                raise RuntimeError(f"G0 failure {path.name} lacks completed checks")
            prefix_length = len(completed_checks)
            if (
                completed_checks != list(G0_COMPLETED_CHECKS[:prefix_length])
                or prefix_length not in G0_FAILURE_STAGE_PREFIX_LENGTHS[failure_stage]
            ):
                raise RuntimeError(
                    f"G0 failure {path.name} has impossible completed-check progress"
                )
            expected_access = [] if failure_stage == "branch_authorization" else ["train"]
            if payload.get("result_payloads_opened") != expected_access:
                raise RuntimeError(f"G0 failure {path.name} reports unsafe result access")
            expected_manifest = manifest_sha256 if prefix_length >= 2 else None
            _expect(
                payload,
                "data_manifest_sha256",
                expected_manifest,
                f"G0 receipt {path.name}",
            )
        else:
            _expect(
                payload,
                "data_manifest_sha256",
                manifest_sha256,
                f"G0 receipt {path.name}",
            )
            present_access_fields = set(G0_PASS_ACCESS_CONTRACT).intersection(payload)
            if present_access_fields:
                if present_access_fields != set(G0_PASS_ACCESS_CONTRACT):
                    raise RuntimeError(
                        f"G0 pass {path.name} has a partial access contract"
                    )
                for field, expected in G0_PASS_ACCESS_CONTRACT.items():
                    actual = payload.get(field)
                    if type(actual) is not type(expected) or actual != expected:
                        raise RuntimeError(
                            f"G0 pass {path.name} has invalid {field}"
                        )
        branch_authorization = payload.get("branch_authorization")
        if capacity == "lora" and branch_authorization is not None:
            raise RuntimeError(f"G0 receipt {path.name} has unexpected authorization")
        if (
            capacity == "fullrank"
            and not isinstance(branch_authorization, dict)
            and not (
                status == "SETUP_CONTROL_FAILED"
                and len(completed_checks) == 0
            )
        ):
            raise RuntimeError(f"G0 receipt {path.name} lacks full-rank authorization")
        setup = payload.get("setup")
        if status == "MODEL_SMOKE_PASS":
            if (
                not isinstance(setup, dict)
                or setup.get("shared_initialization") != initialization[seed]
            ):
                raise RuntimeError(
                    f"G0 receipt {path.name} has wrong initialization lineage"
                )
        else:
            prefix_length = len(completed_checks)
            expected_top_level = initialization[seed] if prefix_length >= 3 else None
            if payload.get("shared_initialization") != expected_top_level:
                raise RuntimeError(
                    f"G0 failure {path.name} has wrong initialization lineage "
                    "in its top-level binding"
                )
            if prefix_length < 4 and setup is not None:
                raise RuntimeError(
                    f"G0 failure {path.name} claims setup before completing model setup"
                )
            if prefix_length >= 4 and (
                not isinstance(setup, dict)
                or setup.get("shared_initialization") != initialization[seed]
            ):
                raise RuntimeError(
                    f"G0 failure {path.name} has wrong setup initialization lineage"
                )
        if status == "SETUP_CONTROL_FAILED":
            for field in (
                "authorizes_positive_control",
                "authorizes_training",
                "authorizes_result_training",
                "authorizes_result_evaluation",
                "training_or_evaluation_started",
                "scientific_evidence",
            ):
                _expect(payload, field, False, f"G0 receipt {path.name}")
            _expect(payload, "benchmark_files_read", 0, f"G0 receipt {path.name}")
            _expect(
                payload,
                "sealed_contrast_payloads_opened",
                [],
                f"G0 receipt {path.name}",
            )
            for field in ("failure_stage", "error_type", "error"):
                if not isinstance(payload.get(field), str) or not payload[field]:
                    raise RuntimeError(f"G0 failure {path.name} lacks {field}")
            mirror = failures_dir / (
                f"g0_{capacity}_seed{seed}_source_{source[:12]}.json"
            )
            if (
                mirror.is_symlink()
                or not mirror.is_file()
                or mirror.read_bytes() != path.read_bytes()
            ):
                raise RuntimeError(
                    f"G0 failure {path.name} lacks its identical tracked mirror"
                )
            positive_control = (
                setup_dir / f"positive_control_{capacity}_seed{seed}.json"
            )
            if os.path.lexists(positive_control):
                raise RuntimeError(
                    f"failed G0 {path.name} cannot coexist with a positive control"
                )


def _validate_positive_control_failure_progress(
    config: dict[str, Any],
    payload: dict[str, Any],
    *,
    capacity: str,
    initialization: dict[str, Any],
    manifest_sha256: str,
    expected_g0_lineage: dict[str, Any],
    expected_branch_authorization: Any,
    label: str,
) -> None:
    """Validate exactly the state reachable at each durable failure stage."""

    stages = (
        "receipt_preflight",
        "branch_authorization",
        "data_manifest",
        "oracle_analysis",
        "model_setup",
        "initial_diagnostics",
        "state_path_overfit",
        "final_optimizer_audit",
        "fixed_final_overfit_gate",
    )
    stage = payload.get("failure_stage")
    if type(stage) is not str or stage not in stages:
        raise RuntimeError(f"{label} has an unknown failure stage")
    for field in ("error_type", "error"):
        if type(payload.get(field)) is not str or not payload[field]:
            raise RuntimeError(f"{label} lacks {field}")
    for field in (
        "authorizes_training",
        "authorizes_result_training",
        "authorizes_result_evaluation",
        "scientific_evidence",
    ):
        if type(payload.get(field)) is not bool or payload[field] is not False:
            raise RuntimeError(f"{label} has unsafe {field}")
    if type(payload.get("benchmark_files_read")) is not int or payload[
        "benchmark_files_read"
    ] != 0:
        raise RuntimeError(f"{label} reports benchmark access")
    for field in ("result_payloads_opened", "sealed_contrast_payloads_opened"):
        if payload.get(field) != []:
            raise RuntimeError(f"{label} reports unsafe {field}")

    completed_updates = payload.get("completed_updates")
    completed_microbatches = payload.get("completed_microbatches")
    diagnostics = payload.get("training_diagnostics")
    if (
        type(completed_updates) is not int
        or type(completed_microbatches) is not int
        or completed_updates < 0
        or completed_microbatches < 0
        or not isinstance(diagnostics, dict)
    ):
        raise RuntimeError(f"{label} has malformed training progress")
    if (
        diagnostics.get("completed_updates") != completed_updates
        or diagnostics.get("completed_microbatches") != completed_microbatches
        or any(
            not isinstance(diagnostics.get(field), list)
            for field in (
                "fixed_probe_steps",
                "evaluations",
                "parameter_probes",
                "optimizer_step_probes",
                "dropout_probes",
            )
        )
    ):
        raise RuntimeError(f"{label} diagnostics/progress binding changed")
    updates = int(config["training"]["positive_control"]["updates"])
    accumulation = int(config["training"]["gradient_accumulation"])
    total_microbatches = updates * accumulation
    if completed_updates > updates or completed_microbatches > total_microbatches:
        raise RuntimeError(f"{label} progress exceeds the registered control geometry")
    if stage in stages[:6] and (completed_updates or completed_microbatches):
        raise RuntimeError(f"{label} claims progress before state-path overfit")
    if stage == "state_path_overfit" and not (
        completed_updates * accumulation
        <= completed_microbatches
        <= min(total_microbatches, (completed_updates + 1) * accumulation)
    ):
        raise RuntimeError(f"{label} has impossible partial optimizer progress")
    if stage in {"final_optimizer_audit", "fixed_final_overfit_gate"} and (
        completed_updates != updates or completed_microbatches != total_microbatches
    ):
        raise RuntimeError(f"{label} claims a final stage before all updates")

    stage_index = stages.index(stage)
    expected_g0 = None if stage == "receipt_preflight" else expected_g0_lineage
    _expect(payload, "g0_lineage", expected_g0, label)
    branch_expected = (
        expected_branch_authorization
        if capacity == "fullrank" and stage_index >= stages.index("data_manifest")
        else None
    )
    _expect(payload, "branch_authorization", branch_expected, label)
    expected_manifest = (
        manifest_sha256 if stage_index >= stages.index("oracle_analysis") else None
    )
    _expect(payload, "data_manifest_sha256", expected_manifest, label)

    control_rows = payload.get("control_rows")
    oracle_analysis = payload.get("oracle_analysis")
    oracle_accuracy = payload.get("oracle_readout_accuracy")
    if stage_index < stages.index("oracle_analysis"):
        if any(value is not None for value in (control_rows, oracle_analysis, oracle_accuracy)):
            raise RuntimeError(f"{label} claims oracle evidence before oracle analysis")
    elif stage == "oracle_analysis":
        if control_rows is None and (oracle_analysis is not None or oracle_accuracy is not None):
            raise RuntimeError(f"{label} has impossible partial oracle progress")
        if oracle_analysis is None and oracle_accuracy is not None:
            raise RuntimeError(f"{label} has an accuracy without oracle analysis")
        if control_rows is not None and not isinstance(control_rows, dict):
            raise RuntimeError(f"{label} control-row receipt is malformed")
        if oracle_analysis is not None and not isinstance(oracle_analysis, dict):
            raise RuntimeError(f"{label} oracle analysis is malformed")
        if oracle_accuracy is not None and (
            type(oracle_accuracy) is not float or not math.isfinite(oracle_accuracy)
        ):
            raise RuntimeError(f"{label} oracle accuracy is malformed")
    elif (
        not isinstance(control_rows, dict)
        or not isinstance(oracle_analysis, dict)
        or type(oracle_accuracy) is not float
        or not math.isfinite(oracle_accuracy)
    ):
        raise RuntimeError(f"{label} omits completed oracle evidence")

    setup = payload.get("setup")
    shared_initialization = payload.get("shared_initialization")
    if stage_index < stages.index("model_setup"):
        if setup is not None or shared_initialization is not None:
            raise RuntimeError(f"{label} claims setup before model setup")
    elif stage == "model_setup" and setup is None:
        if shared_initialization is not None:
            raise RuntimeError(f"{label} has initialization without setup")
    elif (
        not isinstance(setup, dict)
        or setup.get("shared_initialization") != initialization
        or shared_initialization != initialization
    ):
        raise RuntimeError(f"{label} has wrong initialization lineage")


def _validate_positive_control_receipts(
    config: dict[str, Any],
    initialization: dict[int, dict[str, Any]],
    *,
    source: str,
    config_digest: str,
    experiment_id: str,
    manifest_sha256: str,
) -> None:
    runs_dir = _safe_repo_path(
        ROOT / config["paths"]["runs_dir"],
        "tracked runs root",
    )
    setup_dir = _safe_repo_path(runs_dir / "setup", "tracked setup receipt root")
    failures_dir = _safe_repo_path(runs_dir / "failures", "tracked failures root")
    for path in sorted(setup_dir.glob("positive_control_*.json")):
        match = POSITIVE_CONTROL_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"unknown canonical positive-control name: {path.name}")
        capacity, seed_text = match.groups()
        seed = int(seed_text)
        payload = _load_json(path, f"positive-control receipt {path.name}")
        _validate_identity(payload, f"positive-control receipt {path.name}")
        _validate_common_binding(
            payload,
            source=source,
            config_digest=config_digest,
            experiment_id=experiment_id,
            label=f"positive-control receipt {path.name}",
        )
        status = payload.get("status")
        if status not in {"POSITIVE_CONTROL_PASS", "SETUP_CONTROL_FAILED"}:
            raise RuntimeError(f"positive-control receipt {path.name} has invalid status")
        _expect(
            payload,
            "phase",
            f"{capacity}_positive_control",
            f"positive-control receipt {path.name}",
        )
        _expect(payload, "capacity", capacity, f"positive-control receipt {path.name}")
        _expect(payload, "model_seed", seed, f"positive-control receipt {path.name}")
        _expect(payload, "backend", "transformers", f"positive-control receipt {path.name}")
        g0_path = setup_dir / f"g0_{capacity}_seed{seed}.json"
        g0 = _load_json(g0_path, f"positive-control G0 {g0_path.name}")
        expected_g0_lineage = {
            "path": _repo_relative(g0_path),
            "sha256": _sha256(g0_path),
            "receipt_identity_sha256": g0["receipt_identity_sha256"],
            "status": "MODEL_SMOKE_PASS",
            "phase": f"{capacity}_g0",
        }
        label = f"positive-control receipt {path.name}"
        if status == "POSITIVE_CONTROL_PASS":
            _expect(payload, "data_manifest_sha256", manifest_sha256, label)
            _expect(payload, "g0_lineage", expected_g0_lineage, label)
            _expect(payload, "branch_authorization", g0.get("branch_authorization"), label)
            setup = payload.get("setup")
            if (
                not isinstance(setup, dict)
                or setup.get("shared_initialization") != initialization[seed]
                or payload.get("shared_initialization") != initialization[seed]
            ):
                raise RuntimeError(f"{label} has wrong initialization lineage")
            for field, expected in {
                "authorizes_training": True,
                "authorizes_result_training": True,
                "authorizes_result_evaluation": False,
                "scientific_evidence": False,
            }.items():
                if type(payload.get(field)) is not bool or payload[field] is not expected:
                    raise RuntimeError(f"{label} has unsafe {field}")
            if type(payload.get("benchmark_files_read")) is not int or payload[
                "benchmark_files_read"
            ] != 0:
                raise RuntimeError(f"{label} reports benchmark access")
            for field in ("result_payloads_opened", "sealed_contrast_payloads_opened"):
                if payload.get(field) != []:
                    raise RuntimeError(f"{label} reports unsafe {field}")
        else:
            _validate_positive_control_failure_progress(
                config,
                payload,
                capacity=capacity,
                initialization=initialization[seed],
                manifest_sha256=manifest_sha256,
                expected_g0_lineage=expected_g0_lineage,
                expected_branch_authorization=g0.get("branch_authorization"),
                label=label,
            )
            mirror = failures_dir / (
                f"positive_control_{capacity}_seed{seed}_source_{source[:12]}.json"
            )
            if (
                mirror.is_symlink()
                or not mirror.is_file()
                or mirror.read_bytes() != path.read_bytes()
            ):
                raise RuntimeError(
                    f"positive-control failure {path.name} lacks its identical tracked mirror"
                )


def _validate_trigger_failure(
    path: Path,
    *,
    source: str,
    experiment_id: str,
) -> tuple[dict[str, Any], Path]:
    canonical = _safe_repo_path(path, "trigger failure receipt")
    failures_dir = _safe_repo_path(
        ROOT / "runs" / "failures",
        "tracked failures root",
    )
    if canonical.parent != failures_dir or canonical.suffix != ".json":
        raise RuntimeError("trigger failure must be a canonical tracked failure receipt")
    if canonical.name.startswith("invalidated_setup_source_"):
        raise RuntimeError("an invalidation archive cannot trigger another invalidation")
    payload = _load_json(canonical, "trigger failure receipt")
    _validate_identity(payload, "trigger failure receipt")
    status = payload.get("status")
    if not isinstance(status, str) or "FAIL" not in status:
        raise RuntimeError("trigger receipt is not a preserved failure")
    _expect(payload, "experiment_id", experiment_id, "trigger failure receipt")
    _expect(payload, "model_id", MODEL_ID, "trigger failure receipt")
    _expect(payload, "model_revision", MODEL_REVISION, "trigger failure receipt")
    _expect(payload, "source_contract_sha256", source, "trigger failure receipt")
    _expect(payload, "scientific_evidence", False, "trigger failure receipt")
    _expect(payload, "benchmark_files_read", 0, "trigger failure receipt")
    if payload.get("sealed_contrast_payloads_opened", []) != []:
        raise RuntimeError("trigger failure reports sealed contrast access")
    return payload, canonical


def _file_records(items: list[ArchiveItem]) -> list[dict[str, Any]]:
    return [
        {
            "path": item.archive_path,
            "bytes": item.source.stat().st_size,
            "sha256": _sha256(item.source),
        }
        for item in items
    ]


def _items_from_file_records(
    config: dict[str, Any],
    raw_records: Any,
) -> tuple[list[ArchiveItem], list[dict[str, Any]]]:
    if not isinstance(raw_records, list) or not raw_records:
        raise RuntimeError("archive receipt files must be a nonempty list")
    seeds = set(map(int, config["training"]["train_seeds"]))
    data_dir = _safe_repo_path(
        ROOT / config["paths"]["data_dir"],
        "generated-data cleanup root",
    )
    runs_dir = _safe_repo_path(
        ROOT / config["paths"]["runs_dir"],
        "tracked runs root",
    )
    setup_dir = _safe_repo_path(runs_dir / "setup", "tracked setup cleanup root")
    cpu_dir = _safe_repo_path(runs_dir / "cpu_smoke", "CPU setup cleanup root")
    cpu_receipt = cpu_dir / "receipt.json"
    large_root = _large_root(config)
    mandatory_paths = {
        *(f"data_generated/{name}" for name in DATA_SETUP_FILES),
        *(
            f"initialization/initialization_seed{seed}{suffix}"
            for seed in seeds
            for suffix in (".pt", ".pt.json")
        ),
        "tracked_receipts/cpu_smoke_receipt.json",
        *(
            f"tracked_receipts/initialization_seed{seed}.json"
            for seed in seeds
        ),
    }
    items: list[ArchiveItem] = []
    records: list[dict[str, Any]] = []
    paths: list[str] = []
    for raw in raw_records:
        if not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
            raise RuntimeError("archive receipt file record has an invalid schema")
        archive_path = raw["path"]
        byte_count = raw["bytes"]
        digest = raw["sha256"]
        if not isinstance(archive_path, str) or not archive_path:
            raise RuntimeError("archive receipt file path is invalid")
        if (
            not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count < 0
        ):
            raise RuntimeError(f"archive byte count is invalid: {archive_path!r}")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise RuntimeError(f"archive SHA-256 is invalid: {archive_path!r}")

        source: Path | None = None
        if archive_path.startswith("data_generated/"):
            name = archive_path.removeprefix("data_generated/")
            if name in DATA_SETUP_FILES:
                source = data_dir / name
        elif archive_path.startswith("initialization/"):
            name = archive_path.removeprefix("initialization/")
            match = INITIALIZATION_PAYLOAD_RE.fullmatch(name)
            if match is not None and int(match.group(1)) in seeds:
                source = large_root / name
        elif archive_path == "tracked_receipts/cpu_smoke_receipt.json":
            source = cpu_receipt
        elif archive_path.startswith("tracked_receipts/"):
            name = archive_path.removeprefix("tracked_receipts/")
            initialization_match = INITIALIZATION_RECEIPT_RE.fullmatch(name)
            g0_match = G0_RE.fullmatch(name)
            control_match = POSITIVE_CONTROL_RE.fullmatch(name)
            if (
                initialization_match is not None
                and int(initialization_match.group(1)) in seeds
            ):
                source = setup_dir / name
            elif g0_match is not None and int(g0_match.group(2)) in seeds:
                source = setup_dir / name
            elif control_match is not None and int(control_match.group(2)) in seeds:
                source = setup_dir / name
        if source is None:
            raise RuntimeError(
                f"archive receipt contains a noncanonical setup path: {archive_path!r}"
            )
        paths.append(archive_path)
        records.append(
            {"path": archive_path, "bytes": byte_count, "sha256": digest}
        )
        items.append(ArchiveItem(source, archive_path))

    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RuntimeError("archive receipt file paths are not unique canonical order")
    missing = mandatory_paths - set(paths)
    if missing:
        raise RuntimeError(
            f"archive receipt omits mandatory setup files: {sorted(missing)}"
        )
    for archive_path in paths:
        name = archive_path.removeprefix("tracked_receipts/")
        match = POSITIVE_CONTROL_RE.fullmatch(name)
        if match is not None:
            capacity, seed = match.groups()
            required_g0 = f"tracked_receipts/g0_{capacity}_seed{seed}.json"
            if required_g0 not in paths:
                raise RuntimeError(
                    f"archive positive control lacks its G0 receipt: {archive_path}"
                )
    return items, records


def _validate_all(
    config: dict[str, Any],
    items: list[ArchiveItem],
    invalidated_source: str,
    trigger_failure: Path,
) -> tuple[dict[str, Any], Path, str]:
    if not SHA256_RE.fullmatch(invalidated_source):
        raise RuntimeError("--invalidated-source must be an exact lowercase SHA-256")
    replacement_source = source_contract_sha256(ROOT)
    if replacement_source == invalidated_source:
        raise RuntimeError("refusing to archive setup before its source is invalidated")
    experiment_id = config["experiment_id"]
    if config["model"]["id"] != MODEL_ID or config["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("config violates the frozen model identity")
    config_digest = config_sha256(config)
    data_dir = ROOT / config["paths"]["data_dir"]
    manifest, manifest_sha256 = _validate_manifest(
        data_dir / "manifest.json",
        source=invalidated_source,
        config_digest=config_digest,
        experiment_id=experiment_id,
    )
    _validate_ledger(
        data_dir / "contrast_access_ledger.json",
        manifest,
        manifest_sha256,
        experiment_id,
    )
    _validate_cpu_receipt(
        ROOT / config["paths"]["runs_dir"] / "cpu_smoke" / "receipt.json",
        source=invalidated_source,
        config_digest=config_digest,
        experiment_id=experiment_id,
    )
    initialization = _validate_initialization_receipts(
        config,
        source=invalidated_source,
        config_digest=config_digest,
        experiment_id=experiment_id,
    )
    _validate_g0_receipts(
        config,
        initialization,
        source=invalidated_source,
        config_digest=config_digest,
        experiment_id=experiment_id,
        manifest_sha256=manifest_sha256,
    )
    _validate_positive_control_receipts(
        config,
        initialization,
        source=invalidated_source,
        config_digest=config_digest,
        experiment_id=experiment_id,
        manifest_sha256=manifest_sha256,
    )
    trigger, resolved_trigger = _validate_trigger_failure(
        trigger_failure,
        source=invalidated_source,
        experiment_id=experiment_id,
    )
    inventory_sources = {_absolute_no_follow(item.source) for item in items}
    if resolved_trigger in inventory_sources:
        raise RuntimeError("trigger failure cannot be part of the setup inventory")
    return trigger, resolved_trigger, replacement_source


def _validate_existing_archive_receipt(
    config: dict[str, Any],
    invalidated_source: str,
    trigger_failure: Path,
    archive_root: Path,
) -> tuple[
    dict[str, Any],
    bytes,
    list[ArchiveItem],
    list[dict[str, Any]],
]:
    archive_root = _safe_repo_path(
        archive_root,
        "existing invalidation archive root",
    )
    if not SHA256_RE.fullmatch(invalidated_source):
        raise RuntimeError("--invalidated-source must be an exact lowercase SHA-256")
    if archive_root.is_symlink() or not archive_root.is_dir():
        raise RuntimeError("existing invalidation archive is missing or unsafe")
    receipt_path = archive_root / "archive_receipt.json"
    receipt = _load_json(receipt_path, "archive receipt")
    receipt_bytes = receipt_path.read_bytes()
    _validate_identity(receipt, "archive receipt")
    experiment_id = config["experiment_id"]
    _expect(receipt, "schema_version", 1, "archive receipt")
    _expect(
        receipt,
        "status",
        "INVALIDATED_SETUP_ARCHIVED",
        "archive receipt",
    )
    _expect(
        receipt,
        "phase",
        "setup_source_invalidation",
        "archive receipt",
    )
    _expect(receipt, "experiment_id", experiment_id, "archive receipt")
    _expect(receipt, "model_id", MODEL_ID, "archive receipt")
    _expect(receipt, "model_revision", MODEL_REVISION, "archive receipt")
    _expect(receipt, "backend", "transformers", "archive receipt")
    _expect(receipt, "config_sha256", config_sha256(config), "archive receipt")
    _expect(
        receipt,
        "invalidated_source_contract_sha256",
        invalidated_source,
        "archive receipt",
    )
    _expect(receipt, "archive_path", _repo_relative(archive_root), "archive receipt")
    _expect(receipt, "benchmark_files_read", 0, "archive receipt")
    _expect(receipt, "scientific_evidence", False, "archive receipt")
    replacement_source = receipt.get("replacement_source_contract_sha256")
    if (
        not isinstance(replacement_source, str)
        or not SHA256_RE.fullmatch(replacement_source)
        or replacement_source == invalidated_source
    ):
        raise RuntimeError("archive receipt replacement source is invalid")
    if source_contract_sha256(ROOT) == invalidated_source:
        raise RuntimeError("refusing cleanup while the invalidated source is current")

    trigger, resolved_trigger = _validate_trigger_failure(
        trigger_failure,
        source=invalidated_source,
        experiment_id=experiment_id,
    )
    _expect(
        receipt,
        "trigger_failure_receipt",
        _repo_relative(resolved_trigger),
        "archive receipt",
    )
    _expect(
        receipt,
        "trigger_failure_receipt_sha256",
        _sha256(resolved_trigger),
        "archive receipt",
    )
    _expect(
        receipt,
        "trigger_failure_receipt_identity_sha256",
        trigger["receipt_identity_sha256"],
        "archive receipt",
    )
    items, records = _items_from_file_records(config, receipt.get("files"))
    _expect(
        receipt,
        "files_sha256",
        _canonical_sha256(records),
        "archive receipt",
    )
    _expect(
        receipt,
        "total_bytes",
        sum(record["bytes"] for record in records),
        "archive receipt",
    )
    return receipt, receipt_bytes, items, records


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_fsynced(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _write_fsynced(path: Path, encoded: bytes, *, exclusive: bool) -> None:
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _stage_archive(
    items: list[ArchiveItem],
    records: list[dict[str, Any]],
    receipt: dict[str, Any],
    archive_parent: Path,
    invalidated_source: str,
) -> Path:
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".source_{invalidated_source}.tmp-",
            dir=archive_parent,
        )
    )
    try:
        for item, record in zip(items, records, strict=True):
            destination = temporary / item.archive_path
            _copy_fsynced(item.source, destination)
            if (
                destination.stat().st_size != record["bytes"]
                or _sha256(destination) != record["sha256"]
            ):
                raise RuntimeError(f"staged archive verification failed: {item.archive_path}")
        encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _write_fsynced(temporary / "archive_receipt.json", encoded, exclusive=True)
        for directory in sorted(
            {temporary, *(path.parent for path in temporary.rglob("*") if path.is_file())},
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return temporary


def _verify_sources(items: list[ArchiveItem], records: list[dict[str, Any]]) -> None:
    for item, record in zip(items, records, strict=True):
        info = item.source.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
        ):
            raise RuntimeError(f"setup source changed before archival: {item.source}")
        if (
            item.source.stat().st_size != record["bytes"]
            or _sha256(item.source) != record["sha256"]
        ):
            raise RuntimeError(f"setup source changed before archival: {item.source}")


def _commit_staged_archive(temporary: Path, archive_root: Path) -> None:
    try:
        rename_new_entry(REPO_ROOT, temporary, archive_root)
    except StableArtifactError as exc:
        raise RuntimeError(
            "refusing to overwrite or alias an existing invalidated setup archive"
        ) from exc


def _atomic_tracked_receipt(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        publish_new_bytes(REPO_ROOT, path, encoded)
    except StableArtifactError as exc:
        raise RuntimeError(
            "refusing to overwrite or alias an invalidation receipt"
        ) from exc


def _cleanup_stale_staging(
    archive_parent: Path,
    invalidated_source: str,
    tracked_receipt: Path,
) -> None:
    """Remove only private stages left by this exact locked transaction."""

    partial_archives = (
        sorted(archive_parent.glob(f".source_{invalidated_source}.tmp-*"))
        if archive_parent.is_dir()
        else []
    )
    partial_receipts = sorted(
        tracked_receipt.parent.glob(f".{tracked_receipt.name}.tmp-*")
    )
    for partial in partial_archives:
        info = partial.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("stale invalidation archive stage is unsafe")
        try:
            tree_manifest(partial, source_path="stale_invalidation_stage")
        except Exception as exc:
            raise RuntimeError("stale invalidation archive stage is unsafe") from exc
        shutil.rmtree(partial)
        _fsync_directory(archive_parent)
    for partial in partial_receipts:
        info = partial.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
        ):
            raise RuntimeError("stale invalidation receipt stage is unsafe")
        partial.unlink()
        _fsync_directory(tracked_receipt.parent)


def _verify_archive(
    archive_root: Path,
    records: list[dict[str, Any]],
    receipt: dict[str, Any],
    tracked_receipt: Path,
) -> None:
    archive_root = _safe_repo_path(archive_root, "durable invalidation archive root")
    tracked_receipt = _safe_repo_path(
        tracked_receipt,
        "tracked invalidation receipt",
    )
    _verify_archive_content(archive_root, records, receipt)
    archive_receipt = archive_root / "archive_receipt.json"
    tracked_payload = _load_json(tracked_receipt, "tracked archive receipt")
    _validate_identity(tracked_payload, "tracked archive receipt")
    if archive_receipt.read_bytes() != tracked_receipt.read_bytes():
        raise RuntimeError("durable invalidation receipts are not byte-identical")
    if tracked_payload != receipt:
        raise RuntimeError("durable invalidation receipts differ from the staged receipt")


def _verify_archive_content(
    archive_root: Path,
    records: list[dict[str, Any]],
    receipt: dict[str, Any],
) -> None:
    archive_root = _safe_repo_path(archive_root, "durable invalidation archive root")
    if archive_root.is_symlink() or not archive_root.is_dir():
        raise RuntimeError("durable invalidation archive is missing or unsafe")
    archived_payload = _load_json(archive_root / "archive_receipt.json", "archive receipt")
    _validate_identity(archived_payload, "archive receipt")
    if archived_payload != receipt:
        raise RuntimeError("durable archive receipt differs from the transaction receipt")
    expected_names = {record["path"] for record in records} | {"archive_receipt.json"}
    expected_directories = {
        parent.as_posix()
        for name in expected_names
        for parent in Path(name).parents
        if parent.as_posix() != "."
    }
    entries = list(archive_root.rglob("*"))
    if any(
        path.is_symlink() or (not path.is_file() and not path.is_dir())
        for path in entries
    ):
        raise RuntimeError("durable archive contains an unsafe entry")
    actual_names = {
        path.relative_to(archive_root).as_posix()
        for path in entries
        if path.is_file()
    }
    actual_directories = {
        path.relative_to(archive_root).as_posix()
        for path in entries
        if path.is_dir()
    }
    if actual_names != expected_names or actual_directories != expected_directories:
        raise RuntimeError("durable archive inventory is partial or unknown")
    for record in records:
        path = archive_root / record["path"]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"durable archive verification failed: {record['path']}")


def _validate_archived_setup_failure_mirrors(
    config: dict[str, Any],
    archive_root: Path,
    records: list[dict[str, Any]],
    invalidated_source: str,
) -> None:
    """Require every archived setup failure's live source-qualified mirror."""

    failures_dir = _safe_repo_path(
        ROOT / config["paths"]["runs_dir"] / "failures",
        "tracked failures root",
    )
    for record in records:
        archive_path = str(record["path"])
        relative = Path(archive_path)
        if relative.parent.as_posix() != "tracked_receipts":
            continue
        g0_match = G0_RE.fullmatch(relative.name)
        control_match = POSITIVE_CONTROL_RE.fullmatch(relative.name)
        if g0_match is None and control_match is None:
            continue
        canonical = archive_root / relative
        payload = _load_json(canonical, f"archived setup receipt {relative.name}")
        if payload.get("status") != "SETUP_CONTROL_FAILED":
            continue
        match = g0_match if g0_match is not None else control_match
        assert match is not None
        capacity, seed = match.groups()
        prefix = "g0" if g0_match is not None else "positive_control"
        mirror = _safe_repo_path(
            failures_dir
            / f"{prefix}_{capacity}_seed{seed}_source_{invalidated_source[:12]}.json",
            "setup-failure mirror",
        )
        if (
            mirror.is_symlink()
            or not mirror.is_file()
            or mirror.read_bytes() != canonical.read_bytes()
        ):
            raise RuntimeError(
                f"archived setup failure {relative.name} lacks its identical tracked mirror"
            )


def _validate_source_exact_or_absent(
    item: ArchiveItem,
    record: dict[str, Any],
) -> None:
    if not os.path.lexists(item.source):
        return
    info = item.source.lstat()
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
    ):
        raise RuntimeError(f"setup cleanup source is unsafe: {item.source}")
    if (
        item.source.stat().st_size != record["bytes"]
        or _sha256(item.source) != record["sha256"]
    ):
        raise RuntimeError(f"setup cleanup source differs from archive: {item.source}")


_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _descriptor_digest(descriptor: int) -> str:
    offset = os.lseek(descriptor, 0, os.SEEK_CUR)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        return digest.hexdigest()
    finally:
        os.lseek(descriptor, offset, os.SEEK_SET)


def _inode_identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _require_descriptor_bytes(
    descriptor: int,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> None:
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size != expected_bytes
        or _descriptor_digest(descriptor) != expected_sha256
    ):
        raise RuntimeError(f"{label} changed before setup cleanup zeroization")


def _expected_tree_entries(
    file_paths: set[str],
) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {".": {}}
    for raw in sorted(file_paths):
        relative = Path(raw)
        if (
            relative.is_absolute()
            or relative.as_posix() != raw
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise RuntimeError("setup cleanup tree path is noncanonical")
        parent = "."
        for component in relative.parts[:-1]:
            child = component if parent == "." else f"{parent}/{component}"
            prior = entries[parent].get(component)
            if prior not in {None, "directory"}:
                raise RuntimeError("setup cleanup tree path changes type")
            entries[parent][component] = "directory"
            entries.setdefault(child, {})
            parent = child
        leaf = relative.parts[-1]
        if leaf in entries[parent]:
            raise RuntimeError("setup cleanup tree paths collide")
        entries[parent][leaf] = "regular_file"
    return entries


def _hold_exact_tree_bindings(
    root_descriptor: int,
    file_descriptors: dict[str, int],
    stack: ExitStack,
    *,
    label: str,
) -> list[tuple[int, dict[str, tuple[tuple[int, int], str]]]]:
    """Hold every directory and bind exact leaf inodes below one held root."""

    expected = _expected_tree_entries(set(file_descriptors))
    held: list[tuple[int, dict[str, tuple[tuple[int, int], str]]]] = []
    seen = {_inode_identity(os.fstat(root_descriptor))}

    def walk(descriptor: int, relative: str) -> None:
        expected_entries = expected.get(relative)
        if expected_entries is None:
            raise RuntimeError(f"{label} contains an unregistered directory")
        if sorted(os.listdir(descriptor)) != sorted(expected_entries):
            raise RuntimeError(f"{label} inventory changed before zeroization")
        bindings: dict[str, tuple[tuple[int, int], str]] = {}
        held.append((descriptor, bindings))
        for name, expected_type in sorted(expected_entries.items()):
            child_relative = name if relative == "." else f"{relative}/{name}"
            observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            inode = _inode_identity(observed)
            if inode in seen:
                raise RuntimeError(f"{label} contains an inode alias")
            seen.add(inode)
            if expected_type == "directory":
                if not stat.S_ISDIR(observed.st_mode):
                    raise RuntimeError(f"{label} entry changed type")
                child_descriptor = os.open(
                    name,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
                stack.callback(os.close, child_descriptor)
                opened = os.fstat(child_descriptor)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or _inode_identity(opened) != inode
                ):
                    raise RuntimeError(f"{label} directory changed while opening")
                bindings[name] = (inode, "directory")
                walk(child_descriptor, child_relative)
                continue
            held_descriptor = file_descriptors.get(child_relative)
            if held_descriptor is None:
                raise RuntimeError(f"{label} has an unregistered regular file")
            held_info = os.fstat(held_descriptor)
            if (
                not stat.S_ISREG(observed.st_mode)
                or not stat.S_ISREG(held_info.st_mode)
                or held_info.st_nlink != 1
                or _inode_identity(held_info) != inode
            ):
                raise RuntimeError(f"{label} regular-file binding changed")
            bindings[name] = (inode, "regular_file")

    walk(root_descriptor, ".")
    return held


def _revalidate_held_tree_bindings(
    held: list[tuple[int, dict[str, tuple[tuple[int, int], str]]]],
    *,
    label: str,
) -> None:
    for descriptor, bindings in held:
        if sorted(os.listdir(descriptor)) != sorted(bindings):
            raise RuntimeError(f"{label} directory membership changed")
        for name, (inode, expected_type) in bindings.items():
            observed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if (
                _inode_identity(observed) != inode
                or (expected_type == "directory" and not stat.S_ISDIR(observed.st_mode))
                or (
                    expected_type == "regular_file"
                    and (not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1)
                )
            ):
                raise RuntimeError(f"{label} child binding changed")


def _cleanup_quarantine_root(
    config: dict[str, Any], invalidated_source: str
) -> Path:
    return _safe_repo_path(
        _large_root(config)
        / "invalidated_setup_cleanup"
        / f"source_{invalidated_source}",
        "setup cleanup quarantine root",
    )


def _quarantine_path(root: Path, item: ArchiveItem) -> Path:
    relative = Path(item.archive_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("setup cleanup archive path is noncanonical")
    return _safe_repo_path(root / relative, "setup cleanup quarantine path")


def _quarantined_descriptor_state(
    descriptor: int,
    record: dict[str, Any],
) -> bool:
    """Return whether a held leaf is zero, accepting only exact-or-zero state."""

    info = os.fstat(descriptor)
    digest = _descriptor_digest(descriptor)
    is_zeroized = info.st_size == 0 and digest == _EMPTY_SHA256
    is_exact = info.st_size == record["bytes"] and digest == record["sha256"]
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not (
        is_zeroized or is_exact
    ):
        raise RuntimeError(
            "setup cleanup quarantine differs from its archived record"
        )
    return is_zeroized


def _zeroize_quarantined_descriptor(
    descriptor: int,
    record: dict[str, Any],
) -> None:
    """Zero one already-held exact archived duplicate inode."""

    inode = _inode_identity(os.fstat(descriptor))
    is_zeroized = _quarantined_descriptor_state(descriptor, record)
    before = os.fstat(descriptor)
    if before.st_nlink != 1 or _inode_identity(before) != inode:
        raise RuntimeError("setup cleanup quarantine changed before zeroization")
    if not is_zeroized:
        os.ftruncate(descriptor, 0)
    os.fsync(descriptor)
    after = os.fstat(descriptor)
    if (
        after.st_size != 0
        or after.st_nlink != 1
        or _inode_identity(after) != inode
        or _descriptor_digest(descriptor) != _EMPTY_SHA256
    ):
        raise RuntimeError("setup cleanup quarantine did not zeroize exactly")


def _zeroize_quarantined_file(
    path: Path,
    record: dict[str, Any],
) -> None:
    """Safely hold and zero one quarantined file (focused helper/tests)."""

    with open_stable_regular_for_update(REPO_ROOT, path) as descriptor:
        _zeroize_quarantined_descriptor(descriptor, record)


def _validate_quarantined_file_state(
    path: Path,
    record: dict[str, Any],
) -> None:
    try:
        with open_stable_regular(REPO_ROOT, path) as handle:
            _quarantined_descriptor_state(handle.fileno(), record)
    except StableArtifactError as exc:
        raise RuntimeError("setup cleanup quarantine is unsafe") from exc


def _quarantine_snapshot(
    root: Path,
    items: list[ArchiveItem],
    records: list[dict[str, Any]],
) -> tuple[bool, dict[str, Path]]:
    """Validate the exact transaction-owned quarantine inventory."""

    destinations = {
        item.archive_path: _quarantine_path(root, item)
        for item in items
    }
    if not os.path.lexists(root):
        return False, destinations
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("setup cleanup quarantine root is unsafe")
    try:
        observed = tree_manifest(
            root,
            source_path="invalidated_setup_cleanup_quarantine",
        )
    except Exception as exc:
        raise RuntimeError("setup cleanup quarantine is unsafe") from exc
    observed_files = observed.get("files")
    observed_directories = observed.get("directory_entries")
    if not isinstance(observed_files, list) or not isinstance(
        observed_directories, list
    ):
        raise RuntimeError("setup cleanup quarantine snapshot is malformed")
    expected_paths = set(destinations)
    observed_paths = {str(row.get("path")) for row in observed_files}
    if not observed_paths.issubset(expected_paths):
        raise RuntimeError("setup cleanup quarantine contains unknown residue")
    expected_directories = {"."}
    for relative in expected_paths:
        for parent in Path(relative).parents:
            if parent.as_posix() != ".":
                expected_directories.add(parent.as_posix())
    observed_directory_paths = {
        str(row.get("path")) for row in observed_directories
    }
    if not observed_directory_paths.issubset(expected_directories):
        raise RuntimeError("setup cleanup quarantine contains unknown directories")
    if observed_paths != expected_paths:
        return False, destinations
    records_by_path = {str(record["path"]): record for record in records}
    if set(records_by_path) != expected_paths:
        raise RuntimeError("setup cleanup records and quarantine paths differ")
    complete = all(
        row.get("bytes") == 0 and row.get("sha256") == _EMPTY_SHA256
        for row in observed_files
    )
    return complete, destinations


def _prepare_quarantine_directories(
    root: Path,
    destinations: dict[str, Path],
) -> None:
    directories = {root}
    for destination in destinations.values():
        current = destination.parent
        while current != root:
            if not current.is_relative_to(root):
                raise RuntimeError("setup cleanup quarantine path escapes its root")
            directories.add(current)
            current = current.parent
    for directory in sorted(directories, key=lambda path: len(path.parts)):
        try:
            ensure_canonical_directory(REPO_ROOT, directory)
        except StableArtifactError as exc:
            raise RuntimeError(
                "setup cleanup quarantine directory could not be created durably"
            ) from exc
    for directory in sorted(
        directories,
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            fsync_canonical_directory(REPO_ROOT, directory)
        except StableArtifactError as exc:
            raise RuntimeError(
                "setup cleanup quarantine directory could not be fsynced"
            ) from exc


def _source_parent_paths(items: list[ArchiveItem]) -> list[Path]:
    return sorted(
        {item.source.parent for item in items},
        key=lambda path: path.as_posix(),
    )


def _fsync_source_parents(items: list[ArchiveItem]) -> None:
    for parent in _source_parent_paths(items):
        try:
            fsync_canonical_directory(REPO_ROOT, parent)
        except StableArtifactError as exc:
            raise RuntimeError(
                f"setup cleanup source parent could not be fsynced: {parent}"
            ) from exc


def _confirm_durable_zero_quarantine(
    quarantine_root: Path,
    items: list[ArchiveItem],
    records: list[dict[str, Any]],
) -> dict[str, Path]:
    """Re-fsync and rebind a pathname-complete zero quarantine on recovery."""

    complete, destinations = _quarantine_snapshot(
        quarantine_root, items, records
    )
    if not complete:
        raise RuntimeError("setup cleanup quarantine is not complete")
    _prepare_quarantine_directories(quarantine_root, destinations)
    try:
        with ExitStack() as stack:
            root_descriptor = stack.enter_context(
                open_stable_directory_for_update(REPO_ROOT, quarantine_root)
            )
            held_files: list[tuple[int, dict[str, Any]]] = []
            file_descriptors: dict[str, int] = {}
            for item, record in zip(items, records, strict=True):
                descriptor = stack.enter_context(
                    open_stable_regular_for_update(
                        REPO_ROOT,
                        destinations[item.archive_path],
                    )
                )
                if not _quarantined_descriptor_state(descriptor, record):
                    raise RuntimeError(
                        "setup cleanup completed quarantine contains nonzero bytes"
                    )
                held_files.append((descriptor, record))
                file_descriptors[item.archive_path] = descriptor
            held_tree = _hold_exact_tree_bindings(
                root_descriptor,
                file_descriptors,
                stack,
                label="setup cleanup quarantine",
            )
            _revalidate_held_tree_bindings(
                held_tree,
                label="setup cleanup quarantine",
            )
            for descriptor, record in held_files:
                _zeroize_quarantined_descriptor(descriptor, record)
            for descriptor, _ in reversed(held_tree):
                os.fsync(descriptor)
            _revalidate_held_tree_bindings(
                held_tree,
                label="setup cleanup quarantine",
            )
    except StableArtifactError as exc:
        raise RuntimeError(
            "setup cleanup completed quarantine changed during durable confirmation"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"setup cleanup completed quarantine could not be fsynced: {exc}"
        ) from exc
    _prepare_quarantine_directories(quarantine_root, destinations)
    complete, _ = _quarantine_snapshot(quarantine_root, items, records)
    if not complete:
        raise RuntimeError(
            "setup cleanup quarantine changed after durable confirmation"
        )
    return destinations


def _validate_cleanup_inventory(
    config: dict[str, Any],
    items: list[ArchiveItem],
    records: list[dict[str, Any]],
    archive_root: Path,
    invalidated_source: str,
    quarantine_root: Path,
) -> None:
    _validate_archived_setup_failure_mirrors(
        config, archive_root, records, invalidated_source
    )
    _, destinations = _quarantine_snapshot(quarantine_root, items, records)
    for item, record in zip(items, records, strict=True):
        destination = destinations[item.archive_path]
        if os.path.lexists(item.source) and os.path.lexists(destination):
            raise RuntimeError(
                "setup cleanup has both canonical and quarantined sources"
            )
        if os.path.lexists(item.source):
            _validate_source_exact_or_absent(item, record)
        elif not os.path.lexists(destination):
            raise RuntimeError(
                f"setup cleanup lost canonical and quarantined source: {item.source}"
            )
        else:
            _validate_quarantined_file_state(destination, record)

    data_dir = _safe_repo_path(
        ROOT / config["paths"]["data_dir"],
        "generated-data cleanup root",
    )
    if data_dir.is_symlink() or not data_dir.is_dir():
        raise RuntimeError("generated-data cleanup root is missing or unsafe")
    data_items = {item.source.name for item in items if item.source.parent == data_dir}
    data_children = list(data_dir.iterdir())
    if any(child.is_symlink() for child in data_children):
        raise RuntimeError("generated-data cleanup root contains a symlink")
    actual_data = {child.name for child in data_children}
    if ".gitignore" not in actual_data or not (data_dir / ".gitignore").is_file():
        raise RuntimeError("generated-data cleanup root lacks its tracked .gitignore")
    unknown_data = actual_data - ({".gitignore"} | data_items)
    if unknown_data:
        raise RuntimeError(
            f"generated-data cleanup root contains unknown residue: {sorted(unknown_data)}"
        )

    runs_dir = _safe_repo_path(
        ROOT / config["paths"]["runs_dir"],
        "tracked runs root",
    )
    for name in ("cpu_smoke", "setup"):
        parent = runs_dir / name
        if not os.path.lexists(parent):
            continue
        if parent.is_symlink() or not parent.is_dir():
            raise RuntimeError(f"tracked setup cleanup root is unsafe: {parent}")
        allowed = {item.source.name for item in items if item.source.parent == parent}
        children = list(parent.iterdir())
        if any(child.is_symlink() for child in children):
            raise RuntimeError(f"tracked setup cleanup root contains a symlink: {parent}")
        unknown = {child.name for child in children} - allowed
        if unknown:
            raise RuntimeError(
                f"tracked setup cleanup root contains unknown residue: {sorted(unknown)}"
            )

    large_root = _large_root(config)
    if large_root.is_symlink() or not large_root.is_dir():
        raise RuntimeError("external setup cleanup root is missing or unsafe")
    allowed_initialization = {
        item.source.name for item in items if item.source.parent == large_root
    }
    unexpected_initialization = {
        child.name
        for child in large_root.iterdir()
        if name_is_initialization(child.name)
        and child.name not in allowed_initialization
    }
    if unexpected_initialization:
        raise RuntimeError(
            "external setup cleanup root contains unknown initialization residue: "
            f"{sorted(unexpected_initialization)}"
        )


def _verify_cleanup_postconditions(
    config: dict[str, Any],
    items: list[ArchiveItem],
    records: list[dict[str, Any]],
    archive_root: Path,
    quarantine_root: Path,
) -> None:
    remaining = [str(item.source) for item in items if os.path.lexists(item.source)]
    if remaining:
        raise RuntimeError(f"setup cleanup left canonical sources: {remaining}")
    data_dir = _safe_repo_path(
        ROOT / config["paths"]["data_dir"],
        "generated-data cleanup root",
    )
    if (
        data_dir.is_symlink()
        or not data_dir.is_dir()
        or {path.name for path in data_dir.iterdir()} != {".gitignore"}
        or not (data_dir / ".gitignore").is_file()
        or (data_dir / ".gitignore").is_symlink()
    ):
        raise RuntimeError("generated-data cleanup postcondition failed")
    runs_dir = _safe_repo_path(
        ROOT / config["paths"]["runs_dir"],
        "tracked runs root",
    )
    for name in ("cpu_smoke", "setup"):
        path = runs_dir / name
        if path.is_symlink() or not path.is_dir() or list(path.iterdir()):
            raise RuntimeError(f"tracked setup cleanup postcondition failed: {name}")
    large_root = _large_root(config)
    if large_root.is_symlink() or not large_root.is_dir():
        raise RuntimeError("external cleanup postcondition failed")
    remaining_initialization = [
        child.name
        for child in large_root.iterdir()
        if name_is_initialization(child.name)
    ]
    if remaining_initialization:
        raise RuntimeError(
            f"external initialization cleanup postcondition failed: {remaining_initialization}"
        )
    if archive_root.is_symlink() or not archive_root.is_dir():
        raise RuntimeError("durable archive disappeared during cleanup")
    complete, _ = _quarantine_snapshot(quarantine_root, items, records)
    if not complete:
        raise RuntimeError("setup cleanup quarantine is not complete")


def _delete_sources(
    config: dict[str, Any],
    items: list[ArchiveItem],
    records: list[dict[str, Any]],
    archive_root: Path,
    invalidated_source: str,
    receipt_bytes: bytes,
) -> None:
    quarantine_root = _cleanup_quarantine_root(config, invalidated_source)
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    tracked_receipt = _safe_repo_path(
        ROOT
        / config["paths"]["runs_dir"]
        / "failures"
        / f"invalidated_setup_source_{invalidated_source[:8]}.json",
        "tracked invalidation receipt",
    )
    complete, destinations = _quarantine_snapshot(
        quarantine_root, items, records
    )
    if complete:
        # Completion dominates: canonical names may now belong to a regenerated
        # setup under the replacement source and must remain untouched.
        _validate_archived_setup_failure_mirrors(
            config, archive_root, records, invalidated_source
        )
        _confirm_durable_zero_quarantine(
            quarantine_root,
            items,
            records,
        )
        _fsync_source_parents(items)
        return
    _validate_cleanup_inventory(
        config,
        items,
        records,
        archive_root,
        invalidated_source,
        quarantine_root,
    )
    _prepare_quarantine_directories(quarantine_root, destinations)
    try:
        with ExitStack() as stack:
            archive_root_descriptor = stack.enter_context(
                open_stable_directory_for_update(REPO_ROOT, archive_root)
            )
            quarantine_root_descriptor = stack.enter_context(
                open_stable_directory_for_update(REPO_ROOT, quarantine_root)
            )
            archive_handles: dict[str, tuple[Any, int, str]] = {}
            for item, record in zip(items, records, strict=True):
                handle = stack.enter_context(
                    open_stable_regular(
                        REPO_ROOT,
                        archive_root / item.archive_path,
                        expected_sha256=record["sha256"],
                    )
                )
                if os.fstat(handle.fileno()).st_size != record["bytes"]:
                    raise RuntimeError(
                        "setup cleanup archived copy has the wrong byte count"
                    )
                archive_handles[item.archive_path] = (
                    handle,
                    int(record["bytes"]),
                    str(record["sha256"]),
                )
            archive_receipt_handle = stack.enter_context(
                open_stable_regular(
                    REPO_ROOT,
                    archive_root / "archive_receipt.json",
                    expected_sha256=receipt_sha256,
                )
            )
            tracked_receipt_handle = stack.enter_context(
                open_stable_regular(
                    REPO_ROOT,
                    tracked_receipt,
                    expected_sha256=receipt_sha256,
                )
            )
            if (
                archive_receipt_handle.read() != receipt_bytes
                or tracked_receipt_handle.read() != receipt_bytes
            ):
                raise RuntimeError(
                    "setup cleanup archive receipts are not byte-identical"
                )

            # Move every still-canonical exact source before truncating any
            # duplicate.  A crash leaves each item in exactly one recoverable
            # source/quarantine location.
            for item, record in zip(items, records, strict=True):
                destination = destinations[item.archive_path]
                _validate_archived_setup_failure_mirrors(
                    config, archive_root, records, invalidated_source
                )
                if os.path.lexists(item.source):
                    _validate_source_exact_or_absent(item, record)
                    try:
                        move_new_entry(REPO_ROOT, item.source, destination)
                    except StableArtifactError as exc:
                        raise RuntimeError(
                            "setup cleanup quarantine commit was not atomic no-clobber"
                        ) from exc

            _, destinations = _quarantine_snapshot(
                quarantine_root, items, records
            )
            held_quarantine: list[tuple[int, dict[str, Any]]] = []
            quarantine_descriptors: dict[str, int] = {}
            for item, record in zip(items, records, strict=True):
                destination = destinations[item.archive_path]
                descriptor = stack.enter_context(
                    open_stable_regular_for_update(REPO_ROOT, destination)
                )
                held_quarantine.append((descriptor, record))
                quarantine_descriptors[item.archive_path] = descriptor

            # Validate the complete held inventory before the first truncate.
            for descriptor, record in held_quarantine:
                _quarantined_descriptor_state(descriptor, record)
            quarantine_tree = _hold_exact_tree_bindings(
                quarantine_root_descriptor,
                quarantine_descriptors,
                stack,
                label="setup cleanup quarantine",
            )
            archive_descriptors = {
                path: handle.fileno()
                for path, (handle, _, _) in archive_handles.items()
            }
            archive_descriptors["archive_receipt.json"] = (
                archive_receipt_handle.fileno()
            )
            archive_tree = _hold_exact_tree_bindings(
                archive_root_descriptor,
                archive_descriptors,
                stack,
                label="setup cleanup archive",
            )

            def revalidate_archive() -> None:
                _revalidate_held_tree_bindings(
                    archive_tree,
                    label="setup cleanup archive",
                )
                for path, (handle, size, digest) in archive_handles.items():
                    _require_descriptor_bytes(
                        handle.fileno(),
                        expected_bytes=size,
                        expected_sha256=digest,
                        label=f"setup cleanup archive payload {path}",
                    )
                for label, handle in (
                    ("archive receipt", archive_receipt_handle),
                    ("tracked archive receipt", tracked_receipt_handle),
                ):
                    _require_descriptor_bytes(
                        handle.fileno(),
                        expected_bytes=len(receipt_bytes),
                        expected_sha256=receipt_sha256,
                        label=f"setup cleanup {label}",
                    )
                _validate_archived_setup_failure_mirrors(
                    config, archive_root, records, invalidated_source
                )

            _revalidate_held_tree_bindings(
                quarantine_tree,
                label="setup cleanup quarantine",
            )
            revalidate_archive()
            for descriptor, record in held_quarantine:
                _quarantined_descriptor_state(descriptor, record)
            for descriptor, record in held_quarantine:
                revalidate_archive()
                _revalidate_held_tree_bindings(
                    quarantine_tree,
                    label="setup cleanup quarantine",
                )
                _quarantined_descriptor_state(descriptor, record)
                _zeroize_quarantined_descriptor(descriptor, record)
            for descriptor, record in held_quarantine:
                if not _quarantined_descriptor_state(descriptor, record):
                    raise RuntimeError(
                        "setup cleanup quarantine was not fully zeroized"
                    )
            _revalidate_held_tree_bindings(
                quarantine_tree,
                label="setup cleanup quarantine",
            )
            revalidate_archive()
            if len(archive_handles) != len(items):
                raise RuntimeError("setup cleanup did not hold every archive copy")
    except StableArtifactError as exc:
        raise RuntimeError(
            "setup cleanup held artifact changed while sources were retired"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"setup cleanup mutation failed: {exc}") from exc

    _prepare_quarantine_directories(quarantine_root, destinations)
    _fsync_source_parents(items)
    _validate_archived_setup_failure_mirrors(
        config, archive_root, records, invalidated_source
    )
    _verify_cleanup_postconditions(
        config,
        items,
        records,
        archive_root,
        quarantine_root,
    )


def _resume_existing_archive(
    config: dict[str, Any],
    invalidated_source: str,
    trigger_failure: Path,
    archive_root: Path,
    tracked_receipt: Path,
) -> dict[str, Any]:
    receipt, receipt_bytes, items, records = _validate_existing_archive_receipt(
        config,
        invalidated_source,
        trigger_failure,
        archive_root,
    )
    _verify_archive_content(archive_root, records, receipt)
    _validate_archived_setup_failure_mirrors(
        config, archive_root, records, invalidated_source
    )
    if os.path.lexists(tracked_receipt):
        _verify_archive(archive_root, records, receipt, tracked_receipt)
    else:
        _atomic_tracked_receipt(tracked_receipt, receipt_bytes)
        _verify_archive(archive_root, records, receipt, tracked_receipt)
    _delete_sources(
        config,
        items,
        records,
        archive_root,
        invalidated_source,
        receipt_bytes,
    )
    _verify_archive(archive_root, records, receipt, tracked_receipt)
    return receipt


def _archive_invalidated_setup_transaction(
    config: dict[str, Any],
    invalidated_source: str,
    trigger_failure: Path,
) -> dict[str, Any]:
    if not SHA256_RE.fullmatch(invalidated_source):
        raise RuntimeError("--invalidated-source must be an exact lowercase SHA-256")
    large_root = _large_root(config)
    archive_parent = _safe_repo_path(
        large_root / "invalidated_setup",
        "invalidation archive parent",
    )
    archive_root = _safe_repo_path(
        archive_parent / f"source_{invalidated_source}",
        "invalidation archive root",
    )
    failures_dir = _safe_repo_path(
        ROOT / config["paths"]["runs_dir"] / "failures",
        "tracked failures root",
    )
    tracked_receipt = _safe_repo_path(
        failures_dir / f"invalidated_setup_source_{invalidated_source[:8]}.json",
        "tracked invalidation receipt",
    )
    _cleanup_stale_staging(
        archive_parent,
        invalidated_source,
        tracked_receipt,
    )
    archive_exists = os.path.lexists(archive_root)
    tracked_exists = os.path.lexists(tracked_receipt)
    if tracked_exists and not archive_exists:
        raise RuntimeError("tracked invalidation receipt exists without its archive")
    if archive_exists:
        return _resume_existing_archive(
            config,
            invalidated_source,
            trigger_failure,
            archive_root,
            tracked_receipt,
        )

    items = _inventory(config)
    trigger, resolved_trigger, replacement_source = _validate_all(
        config,
        items,
        invalidated_source,
        trigger_failure,
    )
    records = _file_records(items)
    files_identity = _canonical_sha256(records)
    if archive_parent.is_symlink():
        raise RuntimeError("refusing a symlinked invalidation archive parent")
    archive_parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(archive_parent.parent)

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "INVALIDATED_SETUP_ARCHIVED",
        "phase": "setup_source_invalidation",
        "experiment_id": config["experiment_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers",
        "config_sha256": config_sha256(config),
        "invalidated_source_contract_sha256": invalidated_source,
        "replacement_source_contract_sha256": replacement_source,
        "archive_path": _repo_relative(archive_root),
        "trigger_failure_receipt": _repo_relative(resolved_trigger),
        "trigger_failure_receipt_sha256": _sha256(resolved_trigger),
        "trigger_failure_receipt_identity_sha256": trigger[
            "receipt_identity_sha256"
        ],
        "files": records,
        "files_sha256": files_identity,
        "total_bytes": sum(record["bytes"] for record in records),
        "benchmark_files_read": 0,
        "scientific_evidence": False,
    }
    receipt["receipt_identity_sha256"] = _canonical_sha256(receipt)
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")

    temporary: Path | None = None
    try:
        temporary = _stage_archive(
            items,
            records,
            receipt,
            archive_parent,
            invalidated_source,
        )
        _verify_sources(items, records)
        _commit_staged_archive(temporary, archive_root)
        temporary = None
        _atomic_tracked_receipt(tracked_receipt, encoded)
        _verify_archive(archive_root, records, receipt, tracked_receipt)
        _delete_sources(
            config,
            items,
            records,
            archive_root,
            invalidated_source,
            encoded,
        )
        _verify_archive(archive_root, records, receipt, tracked_receipt)
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return receipt


def archive_invalidated_setup(
    config: dict[str, Any],
    invalidated_source: str,
    trigger_failure: Path,
) -> dict[str, Any]:
    """Run one source-invalidation transaction under its stable parent lock."""

    if not SHA256_RE.fullmatch(invalidated_source):
        raise RuntimeError("--invalidated-source must be an exact lowercase SHA-256")
    failures_dir = _safe_repo_path(
        ROOT / config["paths"]["runs_dir"] / "failures",
        "tracked failures root",
    )
    failures_dir.mkdir(parents=True, exist_ok=True)
    lock_path = failures_dir / (
        f".invalidated_setup_source_{invalidated_source[:8]}.lock"
    )
    execution_generation_lock = _safe_repo_path(
        ROOT / "runs" / "run.lock",
        "execution generation lock",
    )
    try:
        with locked_regular(execution_generation_lock):
            with locked_regular(lock_path):
                return _archive_invalidated_setup_transaction(
                    config,
                    invalidated_source,
                    trigger_failure,
                )
    except StableArtifactError as exc:
        raise RuntimeError("invalidated setup transaction path is unsafe") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--invalidated-source", required=True)
    parser.add_argument("--trigger-failure", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = archive_invalidated_setup(
        load_config(args.config),
        args.invalidated_source,
        Path(args.trigger_failure),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
