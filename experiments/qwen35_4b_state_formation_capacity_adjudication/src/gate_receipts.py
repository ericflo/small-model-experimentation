"""Fail-closed contracts for durable gate and branch receipts.

The basic receipt/path contracts deliberately depend only on the Python
standard library.  Conditional branch authorization additionally performs a
lazy, read-only import of :mod:`src.analysis`: a branch is not authoritative
merely because its status and self-hash look plausible.  Its exact canonical
evaluation, training, setup, and row evidence is recomputed before the branch
is returned to any consumer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .oracle_control import validate_oracle_analysis_receipt
from .safe_io import open_stable_regular


IDENTITY_FIELDS = (
    "experiment_id",
    "model_id",
    "model_revision",
    "backend",
    "config_sha256",
    "source_contract_sha256",
    "requirements_training_lock_sha256",
    "design_receipt_sha256",
    "design_receipt_identity_sha256",
)
LINEAGE_FIELDS = frozenset(
    {"path", "sha256", "receipt_identity_sha256", "status", "phase"}
)
STABLE_SETUP_FIELDS = frozenset(
    {
        "capacity",
        "model_seed",
        "tokenizer",
        "adaptation_targets",
        "adaptation_targets_sha256",
        "adaptation_target_manifest",
        "adaptation_target_manifest_sha256",
        "adaptation_parameters",
        "adaptation_zero_function",
        "shared_initialization",
        "trainable_parameters",
        "dropout_control",
        "environment",
        "installed_environment_lock",
        "preflight_device",
    }
)

LORA_MISS_BRANCH = "lora_miss"
STAGE_B_CONTRAST_BRANCH = "stage_b_contrast"
STAGE_B_FULLRANK_MISS_BRANCH = "stage_b_fullrank_miss"
POSTCONTRAST_FULLRANK_MISS_BRANCH = "postcontrast_fullrank_miss"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PINNED_SNAPSHOT_FILES = frozenset(
    {
        "chat_template.jinja",
        "config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)
_BRANCH_SPECS = {
    LORA_MISS_BRANCH: {
        "filename": "lora_joint_trigger.json",
        "status": "LORA_JOINT_MISS_CONTROLS_REQUIRED",
        "phase": "lora_joint_analysis",
        "analysis_phase": "lora_joint",
        "next_stage": "run_lora_state_only_and_fullrank_joint",
    },
    STAGE_B_CONTRAST_BRANCH: {
        "filename": "stage_b_seal.json",
        "status": "STAGE_B_CONTRAST_AUTHORIZED",
        "phase": "stage_b_seal_analysis",
        "analysis_phase": "stage_b_seal",
        "next_stage": "evaluate_exact_six_joint_contrast_cells",
    },
    STAGE_B_FULLRANK_MISS_BRANCH: {
        "filename": "stage_b_seal.json",
        "status": "FULLRANK_STATE_ONLY_REQUIRED",
        "phase": "stage_b_seal_analysis",
        "analysis_phase": "stage_b_seal",
        "next_stage": "run_fullrank_state_only_control",
    },
    POSTCONTRAST_FULLRANK_MISS_BRANCH: {
        "filename": "fullrank_joint.json",
        "status": "FULLRANK_STATE_ONLY_REQUIRED",
        "phase": "fullrank_joint_analysis",
        "analysis_phase": "fullrank_joint",
        "next_stage": "run_fullrank_state_only_control",
    },
}


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    if any(type(key) is not str for key in value):
        _fail(f"{label} has a non-string key")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        _fail(f"{label} mismatch")


def _finite_number(
    value: Any, label: str, *, minimum: float | None = None
) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        _fail(f"{label} must be finite")
    number = float(value)
    if minimum is not None and number < minimum:
        _fail(f"{label} is below its registered minimum")
    return number


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON mapping with the experiment's canonical serialization."""

    _require_mapping(payload, "canonical payload")
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("canonical payload is not finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def receipt_with_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a receipt whose self-identity binds every supplied field."""

    receipt = copy.deepcopy(dict(_require_mapping(payload, "receipt payload")))
    if "receipt_identity_sha256" in receipt:
        _fail("receipt payload already contains receipt_identity_sha256")
    receipt["receipt_identity_sha256"] = canonical_sha256(receipt)
    return receipt


def _validate_self_identity(receipt: Mapping[str, Any], label: str) -> None:
    claimed = _require_sha256(
        receipt.get("receipt_identity_sha256"),
        f"{label} receipt_identity_sha256",
    )
    payload = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_identity_sha256"
    }
    if claimed != canonical_sha256(payload):
        _fail(f"{label} receipt identity mismatch")


def _identity_base(
    expected_identity: Mapping[str, Any], *, expected_phase: str | None = None
) -> dict[str, Any]:
    expected = dict(_require_mapping(expected_identity, "expected identity"))
    permitted = set(IDENTITY_FIELDS) | {"phase"}
    if set(expected) not in (set(IDENTITY_FIELDS), permitted):
        _fail("expected identity has the wrong fields")
    if "phase" in expected:
        if expected_phase is not None:
            _require_exact(expected["phase"], expected_phase, "expected identity phase")
        expected.pop("phase")
    for field in IDENTITY_FIELDS:
        if type(expected[field]) is not str:
            _fail(f"expected identity {field} must be a string")
    for field in (
        "config_sha256",
        "source_contract_sha256",
        "requirements_training_lock_sha256",
        "design_receipt_sha256",
        "design_receipt_identity_sha256",
    ):
        _require_sha256(expected[field], f"expected identity {field}")
    return expected


def validate_receipt_identity(
    receipt: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    *,
    expected_status: str,
    expected_phase: str,
) -> dict[str, Any]:
    """Validate explicit schema/status, full provenance identity, and self-hash."""

    receipt = _require_mapping(receipt, "receipt")
    base = _identity_base(expected_identity, expected_phase=expected_phase)
    _require_exact(receipt.get("schema_version"), 1, "receipt schema_version")
    _require_exact(receipt.get("status"), expected_status, "receipt status")
    _require_exact(receipt.get("phase"), expected_phase, "receipt phase")
    for field, expected in base.items():
        _require_exact(receipt.get(field), expected, f"receipt {field}")
    _validate_self_identity(receipt, "gate")
    return dict(receipt)


def _canonical_relative(value: str) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        _fail("repository path must be a nonempty POSIX relative path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        _fail(f"repository path is not canonical: {value!r}")
    return pure


def canonical_repo_path(
    repo_root: str | os.PathLike[str],
    value: str,
    *,
    require_file: bool = True,
) -> Path:
    """Resolve one canonical repo-relative path while rejecting all symlinks."""

    root = Path(repo_root).resolve(strict=True)
    if not root.is_dir():
        _fail("repository root is not a directory")
    pure = _canonical_relative(value)
    current = root
    for part in pure.parts:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            raise RuntimeError(f"repository path is missing: {value}") from exc
        if stat.S_ISLNK(mode):
            _fail(f"repository path uses a symlink: {value}")
    if current.resolve(strict=True) != current:
        _fail(f"repository path is an alias: {value}")
    if require_file and not stat.S_ISREG(os.stat(current).st_mode):
        _fail(f"repository path is not a regular file: {value}")
    return current


def _require_canonical_actual_path(
    repo_root: str | os.PathLike[str],
    actual_path: str | os.PathLike[str],
    canonical_relative_path: str,
) -> Path:
    expected = canonical_repo_path(repo_root, canonical_relative_path)
    raw = os.fspath(actual_path)
    actual = Path(raw)
    if actual.is_absolute():
        if actual.as_posix() != expected.as_posix():
            _fail("receipt path is not the registered canonical path")
    elif actual.as_posix() != canonical_relative_path:
        _fail("receipt path is not the registered canonical path")
    return expected


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON receipt has duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    _fail(f"JSON receipt has nonfinite constant: {value}")


def _read_stable_json(
    repo_root: str | os.PathLike[str], path: Path
) -> tuple[dict[str, Any], str]:
    """Read and hash a receipt from one no-follow, single-link inode."""

    try:
        with open_stable_regular(repo_root, path) as handle:
            raw = handle.read()
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_pairs_without_duplicates,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"receipt is not stable UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        _fail(f"receipt root is not a mapping: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _path_to_canonical_relative(
    repo_root: str | os.PathLike[str], path: str | os.PathLike[str]
) -> str:
    root = Path(repo_root).resolve(strict=True)
    raw = os.fspath(path)
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            value = candidate.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError("lineage path escapes repository") from exc
        if candidate.as_posix() != (root / value).as_posix():
            _fail("lineage path is an alias")
    else:
        value = candidate.as_posix()
    canonical_repo_path(root, value)
    return value


def lineage_entry(
    repo_root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Create exact durable lineage only for a canonical on-disk receipt."""

    relative = _path_to_canonical_relative(repo_root, path)
    canonical = canonical_repo_path(repo_root, relative)
    receipt = _require_mapping(receipt, "lineage receipt")
    _validate_self_identity(receipt, "lineage")
    if type(receipt.get("status")) is not str or type(receipt.get("phase")) is not str:
        _fail("lineage receipt requires explicit status and phase")
    observed, observed_sha256 = _read_stable_json(repo_root, canonical)
    if observed != dict(receipt):
        _fail("lineage receipt differs from its on-disk file")
    return {
        "path": relative,
        "sha256": observed_sha256,
        "receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "status": receipt["status"],
        "phase": receipt["phase"],
    }


def reopen_lineage(
    repo_root: str | os.PathLike[str],
    entry: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_status: str | None = None,
    expected_phase: str | None = None,
    canonical_relative_path: str | None = None,
) -> dict[str, Any]:
    """Reopen exact lineage, verify bytes/self-hash, and optionally full identity."""

    entry = _require_mapping(entry, "lineage entry")
    if set(entry) != LINEAGE_FIELDS:
        _fail("lineage entry has the wrong fields")
    path_value = entry.get("path")
    if type(path_value) is not str:
        _fail("lineage path must be a string")
    if canonical_relative_path is not None and path_value != canonical_relative_path:
        _fail("lineage path is not the expected canonical path")
    if type(entry.get("status")) is not str or not entry["status"]:
        _fail("lineage status must be a nonempty string")
    if type(entry.get("phase")) is not str or not entry["phase"]:
        _fail("lineage phase must be a nonempty string")
    path = canonical_repo_path(repo_root, path_value)
    _require_sha256(entry.get("sha256"), "lineage sha256")
    _require_sha256(
        entry.get("receipt_identity_sha256"),
        "lineage receipt_identity_sha256",
    )
    receipt, receipt_sha256 = _read_stable_json(repo_root, path)
    if receipt_sha256 != entry["sha256"]:
        _fail("lineage file changed")
    _validate_self_identity(receipt, "lineage")
    _require_exact(
        receipt.get("receipt_identity_sha256"),
        entry["receipt_identity_sha256"],
        "lineage receipt identity",
    )
    _require_exact(receipt.get("status"), entry.get("status"), "lineage status")
    _require_exact(receipt.get("phase"), entry.get("phase"), "lineage phase")
    if expected_status is not None:
        _require_exact(entry.get("status"), expected_status, "lineage expected status")
    if expected_phase is not None:
        _require_exact(entry.get("phase"), expected_phase, "lineage expected phase")
    if expected_identity is not None:
        if expected_status is None or expected_phase is None:
            _fail("full lineage identity requires expected status and phase")
        validate_receipt_identity(
            receipt,
            expected_identity,
            expected_status=expected_status,
            expected_phase=expected_phase,
        )
    return receipt


def stable_setup_receipt(setup: Mapping[str, Any]) -> dict[str, Any]:
    """Return all 15 setup fields, excluding only two volatile VRAM leaves."""

    setup = _require_mapping(setup, "setup receipt")
    if set(setup) != STABLE_SETUP_FIELDS:
        _fail("setup receipt fields changed")
    stable = copy.deepcopy(dict(setup))
    environment = _require_mapping(stable["environment"], "setup environment")
    device = _require_mapping(environment.get("device"), "setup environment device")
    preflight = _require_mapping(stable["preflight_device"], "setup preflight device")
    volatile = "free_memory_gib_before_load"
    if volatile not in device or volatile not in preflight:
        _fail("setup receipt omits a registered free-memory leaf")
    stable["environment"]["device"].pop(volatile)
    stable["preflight_device"].pop(volatile)
    return stable


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(f"{label} must be a positive integer")
    return value


def _validate_tokenizer_snapshot(
    value: Any, *, expected_identity: Mapping[str, Any]
) -> None:
    tokenizer = _require_mapping(value, "setup tokenizer")
    required = {
        "state_token_id",
        "answer_token_ids",
        "pinned_snapshot",
        "runtime_model_config_commit_hash",
    }
    if set(tokenizer) != required:
        _fail("setup tokenizer fields changed")
    state_token = tokenizer.get("state_token_id")
    if type(state_token) is not int or state_token < 0:
        _fail("setup tokenizer state token is invalid")
    answer_tokens = tokenizer.get("answer_token_ids")
    if (
        type(answer_tokens) is not list
        or len(answer_tokens) != 4
        or any(type(token) is not int or token < 0 for token in answer_tokens)
        or len(set(answer_tokens)) != 4
        or state_token in answer_tokens
    ):
        _fail("setup tokenizer answer-token geometry changed")

    identity = _identity_base(expected_identity)
    revision = identity["model_revision"]
    runtime_revision = tokenizer.get("runtime_model_config_commit_hash")
    if runtime_revision not in (None, revision):
        _fail("setup tokenizer runtime model revision changed")
    snapshot = _require_mapping(tokenizer.get("pinned_snapshot"), "pinned snapshot")
    if set(snapshot) != {
        "model_id",
        "requested_revision",
        "resolved_revision",
        "snapshot_layout",
        "files",
        "files_sha256",
    }:
        _fail("pinned snapshot fields changed")
    _require_exact(snapshot.get("model_id"), identity["model_id"], "snapshot model")
    _require_exact(snapshot.get("requested_revision"), revision, "snapshot requested revision")
    _require_exact(snapshot.get("resolved_revision"), revision, "snapshot resolved revision")
    _require_exact(
        snapshot.get("snapshot_layout"), f"snapshots/{revision}", "snapshot layout"
    )
    files = snapshot.get("files")
    if type(files) is not list or not files:
        _fail("pinned snapshot file list is empty")
    filenames: list[str] = []
    shard_count = 0
    for index, item in enumerate(files):
        item = _require_mapping(item, f"pinned snapshot file {index}")
        if set(item) != {"filename", "resolved_revision", "bytes", "sha256"}:
            _fail(f"pinned snapshot file {index} fields changed")
        filename = item.get("filename")
        if (
            type(filename) is not str
            or not filename
            or PurePosixPath(filename).name != filename
        ):
            _fail(f"pinned snapshot file {index} name is invalid")
        filenames.append(filename)
        _require_exact(
            item.get("resolved_revision"), revision, f"pinned snapshot file {index} revision"
        )
        _positive_int(item.get("bytes"), f"pinned snapshot file {index} bytes")
        _require_sha256(item.get("sha256"), f"pinned snapshot file {index} digest")
        if filename.endswith(".safetensors"):
            shard_count += 1
    if filenames != sorted(filenames) or len(filenames) != len(set(filenames)):
        _fail("pinned snapshot files are not uniquely sorted")
    if not _PINNED_SNAPSHOT_FILES <= set(filenames) or shard_count <= 0:
        _fail("pinned snapshot omits a registered model/tokenizer file")
    _require_exact(
        snapshot.get("files_sha256"),
        canonical_sha256({"files": files}),
        "pinned snapshot file manifest digest",
    )


def _validate_trainable_receipt(
    value: Any,
    *,
    adaptation_parameters: int,
    adaptation_tensors: int,
    label: str,
) -> dict[str, int]:
    receipt = _require_mapping(value, label)
    if set(receipt) != {
        "total",
        "adaptation",
        "common",
        "tensor_count",
        "names_sha256",
        "values_sha256",
    }:
        _fail(f"{label} fields changed")
    _require_exact(
        receipt.get("adaptation"), adaptation_parameters, f"{label} adaptation parameters"
    )
    common = _positive_int(receipt.get("common"), f"{label} common parameters")
    _require_exact(
        receipt.get("total"), adaptation_parameters + common, f"{label} total parameters"
    )
    tensor_count = _positive_int(receipt.get("tensor_count"), f"{label} tensor count")
    if tensor_count <= adaptation_tensors:
        _fail(f"{label} omits common-state tensors")
    _require_sha256(receipt.get("names_sha256"), f"{label} names digest")
    _require_sha256(receipt.get("values_sha256"), f"{label} values digest")
    return {
        "common_parameters": common,
        "common_tensors": tensor_count - adaptation_tensors,
        "tensor_count": tensor_count,
    }


def _validate_setup_semantics(
    setup_value: Any,
    *,
    expected_identity: Mapping[str, Any],
    capacity: str,
    model_seed: int,
    expected_adaptation_targets: int,
    expected_adaptation_parameters: int,
    expected_adaptation_dropout: float,
    expected_lora_rank: int,
) -> dict[str, int]:
    setup = _require_mapping(setup_value, "setup")
    if set(setup) != STABLE_SETUP_FIELDS:
        _fail("setup receipt fields changed")
    if capacity not in {"lora", "fullrank"}:
        _fail("setup capacity is not registered")
    _require_exact(setup.get("capacity"), capacity, "setup capacity")
    _require_exact(setup.get("model_seed"), model_seed, "setup model seed")
    target_count = _positive_int(
        expected_adaptation_targets, "registered adaptation target count"
    )
    expected_parameters = _positive_int(
        expected_adaptation_parameters, "registered adaptation parameter count"
    )
    rank = _positive_int(expected_lora_rank, "registered LoRA rank")
    dropout = _finite_number(
        expected_adaptation_dropout, "registered adaptation dropout", minimum=0.0
    )
    if dropout >= 1.0:
        _fail("registered adaptation dropout must be below one")
    _validate_tokenizer_snapshot(setup.get("tokenizer"), expected_identity=expected_identity)

    targets = setup.get("adaptation_targets")
    if (
        type(targets) is not list
        or len(targets) != target_count
        or any(type(target) is not str or not target for target in targets)
        or targets != sorted(targets)
        or len(set(targets)) != target_count
    ):
        _fail("setup adaptation targets are not the registered ordered unique list")
    targets_digest = hashlib.sha256("\n".join(targets).encode("utf-8")).hexdigest()
    _require_exact(
        setup.get("adaptation_targets_sha256"),
        targets_digest,
        "setup adaptation-target digest",
    )

    manifest = setup.get("adaptation_target_manifest")
    if type(manifest) is not list or len(manifest) != target_count:
        _fail("setup adaptation target manifest geometry changed")
    manifest_parameter_sum = 0
    adaptation_tensors = 0
    for index, item in enumerate(manifest):
        item = _require_mapping(item, f"setup adaptation manifest item {index}")
        if set(item) != {"key", "target", "shapes", "dtype", "parameters"}:
            _fail(f"setup adaptation manifest item {index} fields changed")
        _require_exact(item.get("key"), f"d{index:03d}", f"setup target {index} key")
        _require_exact(item.get("target"), targets[index], f"setup target {index} order")
        _require_exact(item.get("dtype"), "torch.float32", f"setup target {index} dtype")
        shapes = item.get("shapes")
        expected_shape_count = 2 if capacity == "lora" else 1
        if type(shapes) is not list or len(shapes) != expected_shape_count:
            _fail(f"setup target {index} shape geometry changed")
        item_parameters = 0
        for shape_index, shape in enumerate(shapes):
            if (
                type(shape) is not list
                or len(shape) != 2
                or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            ):
                _fail(f"setup target {index} shape {shape_index} is invalid")
            item_parameters += shape[0] * shape[1]
        if capacity == "lora" and (shapes[0][0] != rank or shapes[1][1] != rank):
            _fail(f"setup target {index} LoRA rank geometry changed")
        _require_exact(
            item.get("parameters"), item_parameters, f"setup target {index} parameters"
        )
        manifest_parameter_sum += item_parameters
        adaptation_tensors += expected_shape_count
    _require_exact(
        setup.get("adaptation_target_manifest_sha256"),
        canonical_sha256({"targets": manifest}),
        "setup adaptation manifest digest",
    )
    _require_exact(
        manifest_parameter_sum,
        expected_parameters,
        "setup registered adaptation parameter sum",
    )
    _require_exact(
        setup.get("adaptation_parameters"),
        manifest_parameter_sum,
        "setup adaptation parameters",
    )
    _require_exact(
        setup.get("adaptation_zero_function"),
        {"nonzero_output_weights": 0, "max_abs_output_weight": 0.0},
        "setup adaptation zero function",
    )

    dropout_receipt = _require_mapping(setup.get("dropout_control"), "setup dropout")
    if set(dropout_receipt) != {
        "active_nn_dropout_modules",
        "model_config_dropout_values",
        "matched_adaptation_dropout",
    }:
        _fail("setup dropout-control fields changed")
    _require_exact(
        dropout_receipt.get("active_nn_dropout_modules"), [], "setup active dropout modules"
    )
    config_dropout = _require_mapping(
        dropout_receipt.get("model_config_dropout_values"), "setup model dropout values"
    )
    if any(
        type(name) is not str
        or type(value) not in (int, float)
        or not math.isfinite(float(value))
        or float(value) != 0.0
        for name, value in config_dropout.items()
    ):
        _fail("setup model-config dropout is not fully disabled")
    _require_exact(
        dropout_receipt.get("matched_adaptation_dropout"),
        dropout,
        "setup matched adaptation dropout",
    )
    geometry = _validate_trainable_receipt(
        setup.get("trainable_parameters"),
        adaptation_parameters=expected_parameters,
        adaptation_tensors=adaptation_tensors,
        label="setup trainable receipt",
    )
    return {
        **geometry,
        "adaptation_parameters": expected_parameters,
        "adaptation_tensors": adaptation_tensors,
    }


def _validate_dropout_geometry(
    receipt: Any, *, expected_calls: int, expected_cycles: int, label: str
) -> None:
    receipt = _require_mapping(receipt, label)
    required = {
        "calls",
        "cycles",
        "cycle_order_identical",
        "each_cycle_exact_target_set",
        "call_manifest_sha256",
        "cycle_manifest_sha256s",
        "mask_sha256",
    }
    if set(receipt) != required:
        _fail(f"{label} fields changed")
    _require_exact(receipt.get("calls"), expected_calls, f"{label} calls")
    _require_exact(receipt.get("cycles"), expected_cycles, f"{label} cycles")
    _require_exact(
        receipt.get("cycle_order_identical"), True, f"{label} cycle order"
    )
    _require_exact(
        receipt.get("each_cycle_exact_target_set"),
        True,
        f"{label} target set",
    )
    _require_sha256(receipt.get("call_manifest_sha256"), f"{label} call manifest")
    _require_sha256(receipt.get("mask_sha256"), f"{label} mask")
    manifests = receipt.get("cycle_manifest_sha256s")
    if type(manifests) is not list or len(manifests) != expected_cycles:
        _fail(f"{label} cycle manifest geometry changed")
    for index, digest in enumerate(manifests):
        _require_sha256(digest, f"{label} cycle manifest {index}")
    if len(set(manifests)) != 1:
        _fail(f"{label} cycle manifests are not identical")


def _registered_clip_scale(preclip_norm: float, maximum_norm: float) -> float:
    norm = _finite_number(preclip_norm, "preclip gradient norm", minimum=0.0)
    maximum = _finite_number(maximum_norm, "registered gradient clip", minimum=0.0)
    if maximum <= 0.0:
        _fail("registered gradient clip must be positive")
    return min(1.0, maximum / (norm + 1e-6))


def _validate_peft_formula_reference(
    value: Any,
    *,
    capacity: str,
    expected_peft_version: str,
    expected_scale: float,
    expected_dropout: float,
) -> None:
    if capacity == "fullrank":
        _require_exact(value, None, "full-rank PEFT formula reference")
        return
    reference = _require_mapping(value, "LoRA PEFT formula reference")
    if set(reference) != {
        "peft_version",
        "scale",
        "device",
        "actual_adaptation_bank_hook",
        "exact_fp32_dropout_disabled",
        "live_bfloat16_dropout_0_05",
    }:
        _fail("LoRA PEFT formula-reference fields changed")
    if type(expected_peft_version) is not str or not expected_peft_version:
        _fail("registered PEFT version is invalid")
    _require_exact(reference.get("peft_version"), expected_peft_version, "PEFT version")
    _require_exact(reference.get("scale"), expected_scale, "PEFT scale")
    _require_exact(reference.get("device"), "cuda", "PEFT parity device")
    _require_exact(
        reference.get("actual_adaptation_bank_hook"), True, "PEFT actual-hook flag"
    )
    regimes = (
        (
            "exact_fp32_dropout_disabled",
            "torch.float32",
            0.0,
            False,
            1e-6,
            1e-5,
        ),
        (
            "live_bfloat16_dropout_0_05",
            "torch.bfloat16",
            expected_dropout,
            True,
            2e-3,
            1e-2,
        ),
    )
    for name, dtype, dropout, autocast, atol, rtol in regimes:
        regime = _require_mapping(reference.get(name), f"PEFT {name}")
        if set(regime) != {
            "passes",
            "dtype",
            "dropout",
            "autocast",
            "output_shape_dtype_equal",
            "max_output_abs_error",
            "max_a_gradient_abs_error",
            "max_b_gradient_abs_error",
            "atol",
            "rtol",
            "custom_dropout_receipt",
        }:
            _fail(f"PEFT {name} fields changed")
        for field, expected in (
            ("passes", True),
            ("dtype", dtype),
            ("dropout", dropout),
            ("autocast", autocast),
            ("output_shape_dtype_equal", True),
            ("atol", atol),
            ("rtol", rtol),
        ):
            _require_exact(regime.get(field), expected, f"PEFT {name} {field}")
        for field in (
            "max_output_abs_error",
            "max_a_gradient_abs_error",
            "max_b_gradient_abs_error",
        ):
            error = _finite_number(regime.get(field), f"PEFT {name} {field}", minimum=0.0)
            if error > atol:
                _fail(f"PEFT {name} {field} exceeds its absolute tolerance")
        _validate_dropout_geometry(
            regime.get("custom_dropout_receipt"),
            expected_calls=1,
            expected_cycles=1,
            label=f"PEFT {name} dropout",
        )


def _validate_optimizer_geometry(
    value: Any,
    *,
    setup: Mapping[str, Any],
    adaptation_parameters: int,
    adaptation_tensors: int,
    missing_common_state_exemptions: int,
    label: str,
) -> None:
    receipt = _require_mapping(value, label)
    if set(receipt) != {
        "tensors",
        "bytes_by_dtype",
        "total_bytes",
        "total_gib",
        "delta_parameters_audited",
        "delta_moment_tensors",
        "delta_moment_bytes",
        "delta_states_complete",
        "delta_state_manifest_sha256",
        "groups",
        "all_required_group_states_complete_and_finite",
        "registered_missing_state_exemptions",
    }:
        _fail(f"{label} fields changed")
    trainable = _require_mapping(setup.get("trainable_parameters"), f"{label} setup trainable")
    total_tensors = _positive_int(trainable.get("tensor_count"), f"{label} total trainable tensors")
    common_tensors = total_tensors - adaptation_tensors
    if common_tensors <= missing_common_state_exemptions:
        _fail(f"{label} common-state tensor geometry is invalid")
    common_parameters = _positive_int(
        trainable.get("common"), f"{label} common parameter count"
    )
    if type(missing_common_state_exemptions) is not int or missing_common_state_exemptions < 0:
        _fail(f"{label} missing-state exemption count is invalid")
    active_tensors = total_tensors - missing_common_state_exemptions
    delta_moment_bytes = adaptation_parameters * 2 * 4
    # The only registered missing state is the scalar aggregate_logit.
    common_moment_bytes = (common_parameters - missing_common_state_exemptions) * 2 * 4
    total_moment_bytes = delta_moment_bytes + common_moment_bytes
    total_bytes = total_moment_bytes + active_tensors * 4
    expected = {
        "tensors": active_tensors * 3,
        "bytes_by_dtype": {"torch.float32": total_bytes},
        "total_bytes": total_bytes,
        "total_gib": total_bytes / (1024**3),
        "delta_parameters_audited": adaptation_tensors,
        "delta_moment_tensors": adaptation_tensors * 2,
        "delta_moment_bytes": delta_moment_bytes,
        "delta_states_complete": True,
        "all_required_group_states_complete_and_finite": True,
        "registered_missing_state_exemptions": missing_common_state_exemptions,
    }
    for field, expected_value in expected.items():
        _require_exact(receipt.get(field), expected_value, f"{label} {field}")
    _require_sha256(receipt.get("delta_state_manifest_sha256"), f"{label} delta manifest")
    groups = receipt.get("groups")
    if type(groups) is not list or len(groups) != 2:
        _fail(f"{label} optimizer group geometry changed")
    group_specs = (
        (
            "adaptation",
            adaptation_tensors,
            adaptation_tensors * 2,
            delta_moment_bytes,
            0,
        ),
        (
            "common_state",
            common_tensors,
            (common_tensors - missing_common_state_exemptions) * 2,
            common_moment_bytes,
            missing_common_state_exemptions,
        ),
    )
    for index, spec in enumerate(group_specs):
        group = _require_mapping(groups[index], f"{label} group {index}")
        if set(group) != {
            "group_name",
            "parameters",
            "moment_tensors",
            "moment_bytes",
            "registered_missing_state_exemptions",
            "state_manifest_sha256",
            "required_states_complete_and_finite",
        }:
            _fail(f"{label} group {index} fields changed")
        name, parameters, moments, moment_bytes, exemptions = spec
        for field, expected_value in (
            ("group_name", name),
            ("parameters", parameters),
            ("moment_tensors", moments),
            ("moment_bytes", moment_bytes),
            ("registered_missing_state_exemptions", exemptions),
            ("required_states_complete_and_finite", True),
        ):
            _require_exact(group.get(field), expected_value, f"{label} group {name} {field}")
        _require_sha256(group.get("state_manifest_sha256"), f"{label} group {name} manifest")


def _validate_gradient_group(group: Any, label: str, *, require_nonzero: bool) -> None:
    group = _require_mapping(group, label)
    if set(group) != {"tensors", "with_gradient", "finite", "nonzero", "items"}:
        _fail(f"{label} fields changed")
    tensors = group.get("tensors")
    with_gradient = group.get("with_gradient")
    finite_count = group.get("finite")
    nonzero = group.get("nonzero")
    if any(type(value) is not int or value < 0 for value in (tensors, with_gradient, finite_count, nonzero)):
        _fail(f"{label} gradient counts are invalid")
    if tensors <= 0 or with_gradient != tensors or finite_count != tensors:
        _fail(f"{label} gradient reachability is incomplete")
    if require_nonzero and nonzero != tensors:
        _fail(f"{label} gradients are not all nonzero")
    items = group.get("items")
    if type(items) is not list or len(items) != tensors:
        _fail(f"{label} gradient item geometry changed")
    names: set[str] = set()
    observed_with_gradient = observed_finite = observed_nonzero = 0
    for index, item in enumerate(items):
        item = _require_mapping(item, f"{label} item {index}")
        if set(item) != {"name", "has_gradient", "finite", "norm"}:
            _fail(f"{label} item {index} fields changed")
        name = item.get("name")
        if type(name) is not str or not name or name in names:
            _fail(f"{label} item {index} name is invalid")
        names.add(name)
        has_gradient = item.get("has_gradient")
        finite = item.get("finite")
        if type(has_gradient) is not bool or type(finite) is not bool:
            _fail(f"{label} item {index} flags are invalid")
        if not has_gradient and finite:
            _fail(f"{label} item {index} cannot be finite without a gradient")
        norm = item.get("norm")
        if finite:
            norm_value = _finite_number(
                norm, f"{label} item {index} norm", minimum=0.0
            )
        elif norm is not None:
            _fail(f"{label} item {index} nonfinite norm must be null")
        else:
            norm_value = 0.0
        observed_with_gradient += int(has_gradient)
        observed_finite += int(finite)
        observed_nonzero += int(finite and norm_value > 0.0)
    if (
        observed_with_gradient != with_gradient
        or observed_finite != finite_count
        or observed_nonzero != nonzero
    ):
        _fail(f"{label} gradient counts do not match its items")


def _validate_g0_gradient_summary(
    summary: Any, *, label: str, include_aggregate: bool
) -> None:
    summary = _require_mapping(summary, label)
    expected_fields = {
        "adaptation",
        "initializer",
        "step",
        "sufficiency",
        "damping",
        "aggregate_exempt",
        "all_required_tensors_finite_nonzero",
        "base_gradient_tensors",
    }
    if set(summary) != expected_fields:
        _fail(f"{label} fields changed")
    _require_exact(summary.get("base_gradient_tensors"), 0, f"{label} base gradients")
    for name in ("adaptation", "initializer", "step", "sufficiency", "damping"):
        _validate_gradient_group(summary[name], f"{label} {name}", require_nonzero=True)
    if include_aggregate:
        _validate_gradient_group(
            summary["aggregate_exempt"],
            f"{label} aggregate",
            require_nonzero=True,
        )
    _require_exact(
        summary.get("all_required_tensors_finite_nonzero"),
        True,
        f"{label} required-gradient flag",
    )


def _validate_lineage_binding(
    repo_root: str | os.PathLike[str],
    observed: Any,
    expected: Mapping[str, Any] | None,
    *,
    expected_identity: Mapping[str, Any],
    label: str,
) -> None:
    if expected is None:
        _require_exact(observed, None, label)
        return
    expected = dict(_require_mapping(expected, f"expected {label}"))
    _require_exact(observed, expected, label)
    base_identity = _identity_base(expected_identity)
    reopen_lineage(
        repo_root,
        expected,
        expected_identity=base_identity,
        expected_status=str(expected["status"]),
        expected_phase=str(expected["phase"]),
        canonical_relative_path=str(expected["path"]),
    )


def validate_g0_pass(
    repo_root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    *,
    canonical_relative_path: str,
    expected_identity: Mapping[str, Any],
    capacity: str,
    model_seed: int,
    data_manifest_sha256: str,
    expected_setup: Mapping[str, Any],
    expected_branch_authorization: Mapping[str, Any] | None,
    k1_max_logit_abs_error: float,
    train_k: int,
    max_recurrence: int,
    expected_adaptation_targets: int,
    expected_adaptation_parameters: int,
    expected_adaptation_dropout: float,
    expected_adaptation_scale: float,
    expected_lora_rank: int,
    expected_peft_version: str,
    adaptation_gradient_clip: float,
    common_gradient_clip: float,
    worst_depth_seed: int,
    min_free_memory_gib: float = 4.0,
) -> dict[str, Any]:
    """Validate a canonical G0 pass, including its scientific no-evidence gate."""

    expected_name = f"g0_{capacity}_seed{model_seed}.json"
    canonical = _canonical_relative(canonical_relative_path)
    if canonical.name != expected_name or canonical.parent.name != "setup":
        _fail("G0 canonical path does not match its result cell")
    actual = _require_canonical_actual_path(repo_root, path, canonical_relative_path)
    receipt, receipt_sha256 = _read_stable_json(repo_root, actual)
    validate_receipt_identity(
        receipt,
        expected_identity,
        expected_status="MODEL_SMOKE_PASS",
        expected_phase=f"{capacity}_g0",
    )
    _require_exact(receipt.get("capacity"), capacity, "G0 capacity")
    _require_exact(receipt.get("model_seed"), model_seed, "G0 model seed")
    _require_sha256(data_manifest_sha256, "expected data manifest")
    _require_exact(
        receipt.get("data_manifest_sha256"), data_manifest_sha256, "G0 data manifest"
    )
    _require_exact(
        stable_setup_receipt(receipt.get("setup")),
        stable_setup_receipt(expected_setup),
        "G0 stable setup",
    )
    _require_exact(receipt["setup"].get("capacity"), capacity, "G0 setup capacity")
    _require_exact(receipt["setup"].get("model_seed"), model_seed, "G0 setup seed")
    setup_geometry = _validate_setup_semantics(
        receipt["setup"],
        expected_identity=expected_identity,
        capacity=capacity,
        model_seed=model_seed,
        expected_adaptation_targets=expected_adaptation_targets,
        expected_adaptation_parameters=expected_adaptation_parameters,
        expected_adaptation_dropout=expected_adaptation_dropout,
        expected_lora_rank=expected_lora_rank,
    )
    _validate_peft_formula_reference(
        receipt.get("peft_formula_reference"),
        capacity=capacity,
        expected_peft_version=expected_peft_version,
        expected_scale=expected_adaptation_scale,
        expected_dropout=expected_adaptation_dropout,
    )
    _validate_lineage_binding(
        repo_root,
        receipt.get("branch_authorization"),
        expected_branch_authorization,
        expected_identity=expected_identity,
        label="G0 branch authorization",
    )
    access = {
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
    for field, expected in access.items():
        _require_exact(receipt.get(field), expected, f"G0 {field}")

    threshold = _finite_number(
        k1_max_logit_abs_error, "registered K=1 threshold", minimum=0.0
    )
    for field in (
        "k1_max_logit_abs_error_before_optimizer",
        "k1_max_logit_abs_error_after_optimizer",
    ):
        value = _finite_number(receipt.get(field), f"G0 {field}", minimum=0.0)
        if value > threshold:
            _fail(f"G0 {field} exceeds the registered threshold")
    _require_exact(receipt.get("k1_adaptation_calls"), 0, "G0 pre-optimizer K=1 calls")
    _require_exact(
        receipt.get("k1_adaptation_calls_after_optimizer"),
        0,
        "G0 post-optimizer K=1 calls",
    )
    _require_exact(
        receipt.get("zero_function_enabled_minus_disabled_error"),
        0.0,
        "G0 zero-function error",
    )

    if type(train_k) is not int or train_k <= 1:
        _fail("registered train K is invalid")
    if type(expected_adaptation_targets) is not int or expected_adaptation_targets <= 0:
        _fail("registered adaptation target count is invalid")
    train_cycles = train_k - 1
    train_calls = train_cycles * expected_adaptation_targets
    probes = receipt.get("two_step_gradient_probe")
    if type(probes) is not list or len(probes) != 2:
        _fail("G0 two-step probe geometry changed")
    for index, probe in enumerate(probes, start=1):
        probe = _require_mapping(probe, f"G0 gradient probe {index}")
        _require_exact(probe.get("step"), index, f"G0 gradient probe {index} step")
        _validate_dropout_geometry(
            probe.get("dropout_probe"),
            expected_calls=train_calls,
            expected_cycles=train_cycles,
            label=f"G0 gradient probe {index} dropout",
        )
        _require_exact(
            _require_mapping(probe.get("gradients"), f"G0 gradient probe {index}").get(
                "base_gradient_tensors"
            ),
            0,
            f"G0 gradient probe {index} base gradients",
        )
        _finite_number(probe.get("loss"), f"G0 gradient probe {index} loss")
        _finite_number(
            probe.get("preclip_adaptation_gradient_norm"),
            f"G0 gradient probe {index} adaptation norm",
            minimum=0.0,
        )
        _finite_number(
            probe.get("preclip_common_gradient_norm"),
            f"G0 gradient probe {index} common norm",
            minimum=0.0,
        )
    _validate_g0_gradient_summary(
        probes[1]["gradients"], label="G0 second-step gradients", include_aggregate=False
    )

    joint = _require_mapping(receipt.get("live_joint_backward_probe"), "G0 joint probe")
    _require_exact(joint.get("objective"), "joint", "G0 joint objective")
    _validate_dropout_geometry(
        joint.get("dropout_probe"),
        expected_calls=train_calls,
        expected_cycles=train_cycles,
        label="G0 joint dropout",
    )
    _validate_g0_gradient_summary(
        joint.get("gradients"), label="G0 joint gradients", include_aggregate=True
    )
    _require_exact(
        joint.get("all_joint_trainable_groups_finite_nonzero"),
        True,
        "G0 joint-gradient flag",
    )
    for field in (
        "loss",
        "answer_loss",
        "elapsed_seconds",
        "peak_allocated_gib",
        "preclip_adaptation_gradient_norm",
        "preclip_common_gradient_norm",
        "adaptation_applied_clip_scale",
        "common_state_applied_clip_scale",
    ):
        _finite_number(joint.get(field), f"G0 joint {field}", minimum=0.0)
    _require_exact(
        joint.get("adaptation_applied_clip_scale"),
        _registered_clip_scale(
            joint["preclip_adaptation_gradient_norm"], adaptation_gradient_clip
        ),
        "G0 joint adaptation clip scale",
    )
    _require_exact(
        joint.get("common_state_applied_clip_scale"),
        _registered_clip_scale(
            joint["preclip_common_gradient_norm"], common_gradient_clip
        ),
        "G0 joint common-state clip scale",
    )

    _validate_optimizer_geometry(
        receipt.get("optimizer_state"),
        setup=receipt["setup"],
        adaptation_parameters=setup_geometry["adaptation_parameters"],
        adaptation_tensors=setup_geometry["adaptation_tensors"],
        missing_common_state_exemptions=0,
        label="G0 optimizer state",
    )
    timed = _require_mapping(receipt.get("timed_ten_step_probe"), "G0 timing probe")
    _require_exact(timed.get("steps"), 10, "G0 timing steps")
    losses = timed.get("losses")
    if type(losses) is not list or len(losses) != 10:
        _fail("G0 timing loss geometry changed")
    for index, value in enumerate(losses):
        _finite_number(value, f"G0 timing loss {index}")
    timed_elapsed = _finite_number(
        timed.get("elapsed_seconds"), "G0 timing elapsed", minimum=0.0
    )
    _require_exact(
        timed.get("seconds_per_step"),
        timed_elapsed / 10,
        "G0 timing per-step",
    )

    if type(max_recurrence) is not int or max_recurrence <= train_k:
        _fail("registered maximum recurrence is invalid")
    _require_exact(receipt.get("worst_depth"), max_recurrence, "G0 worst depth")
    _finite_number(
        receipt.get("worst_forward_seconds"), "G0 worst forward time", minimum=0.0
    )
    _validate_dropout_geometry(
        receipt.get("worst_call_receipt"),
        expected_calls=(max_recurrence - 1) * expected_adaptation_targets,
        expected_cycles=max_recurrence - 1,
        label="G0 worst-depth dropout",
    )
    worst_row = _require_mapping(receipt.get("worst_setup_row"), "G0 worst setup row")
    if set(worst_row) != {
        "id",
        "seed",
        "structural_fingerprint",
        "cross_result_structural_overlap",
    }:
        _fail("G0 worst setup row fields changed")
    if type(worst_row.get("id")) is not str or not worst_row["id"]:
        _fail("G0 worst setup row id is invalid")
    _require_exact(worst_row.get("seed"), worst_depth_seed, "G0 worst setup row seed")
    if not str(worst_row.get("id", "")).startswith(
        f"setup_g0_worst_depth-{worst_depth_seed}-"
    ):
        _fail("G0 worst setup row id is detached from its registered seed")
    _require_sha256(
        worst_row.get("structural_fingerprint"), "G0 worst-row structural fingerprint"
    )
    _require_exact(
        worst_row.get("cross_result_structural_overlap"),
        0,
        "G0 worst-row result overlap",
    )
    roundtrip = _require_mapping(
        receipt.get("checkpoint_roundtrip"), "G0 checkpoint roundtrip"
    )
    for field in (
        "destructive_adaptation_digest_changed",
        "destructive_common_digest_changed",
        "restored_adaptation_digest_equal",
        "restored_common_digest_equal",
    ):
        _require_exact(roundtrip.get(field), True, f"G0 roundtrip {field}")
    _require_exact(
        roundtrip.get("max_logit_abs_error"), 0.0, "G0 roundtrip logit error"
    )
    rng = _require_mapping(
        receipt.get("common_initialization_rng_isolation"), "G0 RNG isolation"
    )
    if set(rng) != {
        "tensor_manifest_equal",
        "tensor_values_sha256",
        "expected_tensor_values_sha256",
    }:
        _fail("G0 RNG-isolation fields changed")
    _require_exact(rng.get("tensor_manifest_equal"), True, "G0 RNG tensor manifest")
    _require_sha256(rng.get("tensor_values_sha256"), "G0 RNG tensor values")
    shared_initialization = _require_mapping(
        receipt["setup"].get("shared_initialization"), "G0 shared initialization"
    )
    initialization_metadata = _require_mapping(
        shared_initialization.get("metadata"), "G0 shared initialization metadata"
    )
    initialization_digest = _require_sha256(
        initialization_metadata.get("tensor_values_sha256"),
        "G0 shared initialization tensor digest",
    )
    _require_exact(
        rng.get("expected_tensor_values_sha256"),
        initialization_digest,
        "G0 RNG tensor digest",
    )
    _require_exact(
        rng.get("tensor_values_sha256"), initialization_digest, "G0 rebuilt RNG tensor digest"
    )
    elapsed = _finite_number(
        receipt.get("elapsed_seconds"), "G0 total elapsed time", minimum=0.0
    )
    allocated = _finite_number(
        receipt.get("peak_allocated_gib"), "G0 peak allocated memory", minimum=0.0
    )
    reserved = _finite_number(
        receipt.get("peak_reserved_gib"), "G0 peak reserved memory", minimum=0.0
    )
    worst_seconds = _finite_number(
        receipt.get("worst_forward_seconds"), "G0 worst forward time", minimum=0.0
    )
    if elapsed <= 0.0 or allocated <= 0.0 or reserved < allocated or worst_seconds <= 0.0:
        _fail("G0 top-level timing/memory receipt is invalid")
    if allocated < float(joint["peak_allocated_gib"]):
        _fail("G0 top-level peak memory is below the live joint probe")
    if elapsed < float(joint["elapsed_seconds"]) + timed_elapsed + worst_seconds:
        _fail("G0 total elapsed time is below its timed subprobes")
    _finite_number(
        receipt.get("free_memory_gib_after_g0"),
        "G0 free memory",
        minimum=_finite_number(
            min_free_memory_gib, "registered minimum free memory", minimum=0.0
        ),
    )
    return receipt


def _validate_positive_control_evaluations(
    diagnostics: Mapping[str, Any], *, rows: int, updates: int
) -> tuple[int, float]:
    expected_steps = [0, 1, 16, 64, 128, updates]
    _require_exact(
        diagnostics.get("fixed_probe_steps"), expected_steps, "positive-control probe steps"
    )
    evaluations = diagnostics.get("evaluations")
    if type(evaluations) is not list or len(evaluations) != 2 * len(expected_steps):
        _fail("positive-control evaluation geometry changed")
    final_correct = -1
    final_accuracy = -1.0
    for index, (step, mode) in enumerate(
        (item for step in expected_steps for item in ((step, "intact"), (step, "disabled")))
    ):
        evaluation = _require_mapping(evaluations[index], "positive-control evaluation")
        _require_exact(evaluation.get("step"), step, "positive-control evaluation step")
        _require_exact(
            evaluation.get("adaptation_mode"), mode, "positive-control adaptation mode"
        )
        overall = _require_mapping(
            evaluation.get("overall"), "positive-control evaluation overall"
        )
        _require_exact(overall.get("rows"), rows, "positive-control evaluation rows")
        counts = _require_mapping(
            overall.get("terminal_correct_counts"),
            "positive-control terminal counts",
        )
        correct = counts.get("joint")
        if type(correct) is not int or not 0 <= correct <= rows:
            _fail("positive-control terminal joint count is invalid")
        accuracy = _finite_number(
            overall.get("joint_final_accuracy"),
            "positive-control joint accuracy",
            minimum=0.0,
        )
        if accuracy > 1.0 or accuracy != correct / rows:
            _fail("positive-control joint accuracy/count mismatch")
        if step == updates and mode == "intact":
            final_correct, final_accuracy = correct, accuracy
    return final_correct, final_accuracy


def _registered_factorial_grid(
    *,
    families: Sequence[str],
    templates: Sequence[str],
    depths: Sequence[int],
    query_kinds: Sequence[str],
    examples_per_cell: int,
) -> dict[str, int]:
    dimensions: tuple[tuple[Any, ...], ...] = tuple(
        tuple(value) for value in (families, templates, depths, query_kinds)
    )
    if any(not dimension for dimension in dimensions):
        _fail("positive-control factorial dimension is empty")
    family_values, template_values, depth_values, query_values = dimensions
    for label, values in (
        ("family", family_values),
        ("template", template_values),
        ("query kind", query_values),
    ):
        if (
            any(type(value) is not str or not value or "|" in value for value in values)
            or len(set(values)) != len(values)
        ):
            _fail(f"positive-control {label} dimension is invalid")
    if (
        any(type(depth) is not int or depth <= 1 for depth in depth_values)
        or len(set(depth_values)) != len(depth_values)
    ):
        _fail("positive-control depth dimension is invalid")
    count = _positive_int(examples_per_cell, "positive-control examples per cell")
    return {
        f"{family}|{template}|depth={depth}|query={query}": count
        for family in family_values
        for template in template_values
        for depth in depth_values
        for query in query_values
    }


def _validate_positive_optimizer_probes(
    diagnostics: Mapping[str, Any],
    *,
    updates: int,
    accumulation: int,
    learning_rate: float,
    adaptation_gradient_clip: float,
    common_gradient_clip: float,
) -> None:
    steps = [step for step in (1, 16, 64, 128, updates) if step <= updates]
    # Preserve order while allowing a smaller smoke update count.
    steps = list(dict.fromkeys(steps))
    probes = diagnostics.get("optimizer_step_probes")
    if type(probes) is not list or len(probes) != len(steps):
        _fail("positive-control optimizer-step probe geometry changed")
    observed_adaptation_scales: list[float] = []
    observed_common_scales: list[float] = []
    for index, step in enumerate(steps):
        probe = _require_mapping(probes[index], f"positive-control optimizer probe {step}")
        _require_exact(probe.get("step"), step, f"positive-control optimizer probe {step} step")
        _require_exact(
            probe.get("microbatch_start"),
            (step - 1) * accumulation + 1,
            f"positive-control optimizer probe {step} microbatch start",
        )
        _require_exact(
            probe.get("microbatch_end"),
            step * accumulation,
            f"positive-control optimizer probe {step} microbatch end",
        )
        _require_exact(
            probe.get("microbatches"), accumulation, f"positive-control optimizer probe {step} microbatches"
        )
        for field in (
            "adaptation_gradient_finite",
            "common_state_gradient_finite",
        ):
            _require_exact(probe.get(field), True, f"positive-control optimizer probe {step} {field}")
        _require_exact(
            probe.get("base_trainable_parameters"),
            0,
            f"positive-control optimizer probe {step} base trainables",
        )
        _require_exact(
            probe.get("adaptation_learning_rate"),
            learning_rate,
            f"positive-control optimizer probe {step} adaptation learning rate",
        )
        _require_exact(
            probe.get("common_state_learning_rate"),
            learning_rate,
            f"positive-control optimizer probe {step} common learning rate",
        )
        adaptation_norm = _finite_number(
            probe.get("adaptation_preclip_gradient_norm"),
            f"positive-control optimizer probe {step} adaptation norm",
            minimum=0.0,
        )
        common_norm = _finite_number(
            probe.get("common_state_preclip_gradient_norm"),
            f"positive-control optimizer probe {step} common norm",
            minimum=0.0,
        )
        if adaptation_norm <= 0.0 or common_norm <= 0.0:
            _fail(f"positive-control optimizer probe {step} has a zero gradient norm")
        adaptation_scale = _registered_clip_scale(
            adaptation_norm, adaptation_gradient_clip
        )
        common_scale = _registered_clip_scale(common_norm, common_gradient_clip)
        _require_exact(
            probe.get("adaptation_applied_clip_scale"),
            adaptation_scale,
            f"positive-control optimizer probe {step} adaptation clip scale",
        )
        _require_exact(
            probe.get("common_state_applied_clip_scale"),
            common_scale,
            f"positive-control optimizer probe {step} common clip scale",
        )
        observed_adaptation_scales.append(adaptation_scale)
        observed_common_scales.append(common_scale)
    minimums = _require_mapping(
        diagnostics.get("minimum_applied_clip_scales"),
        "positive-control minimum clip scales",
    )
    if set(minimums) != {"adaptation", "common_state"}:
        _fail("positive-control minimum clip-scale fields changed")
    for name, values in (
        ("adaptation", observed_adaptation_scales),
        ("common_state", observed_common_scales),
    ):
        minimum = _finite_number(
            minimums.get(name), f"positive-control minimum {name} clip scale", minimum=0.0
        )
        if minimum <= 0.0 or minimum > 1.0 or minimum > min(values):
            _fail(f"positive-control minimum {name} clip scale is inconsistent")


def validate_positive_control_pass(
    repo_root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    *,
    canonical_relative_path: str,
    expected_identity: Mapping[str, Any],
    capacity: str,
    model_seed: int,
    data_manifest_sha256: str,
    expected_setup: Mapping[str, Any],
    expected_branch_authorization: Mapping[str, Any] | None,
    expected_g0_lineage: Mapping[str, Any],
    expected_control_rows: Sequence[Mapping[str, Any]],
    control_seed: int,
    control_rows: int,
    control_updates: int,
    gradient_accumulation: int,
    min_oracle_readout_accuracy: float,
    min_overfit_final_joint_accuracy: float,
    expected_adaptation_targets: int,
    expected_adaptation_parameters: int,
    expected_adaptation_dropout: float,
    expected_lora_rank: int,
    control_families: Sequence[str],
    control_templates: Sequence[str],
    control_depths: Sequence[int],
    control_query_kinds: Sequence[str],
    control_examples_per_cell: int,
    learning_rate: float,
    adaptation_gradient_clip: float,
    common_gradient_clip: float,
) -> dict[str, Any]:
    """Validate the canonical fresh positive-control pass for one result cell."""

    expected_name = f"positive_control_{capacity}_seed{model_seed}.json"
    canonical = _canonical_relative(canonical_relative_path)
    if canonical.name != expected_name or canonical.parent.name != "setup":
        _fail("positive-control canonical path does not match its result cell")
    actual = _require_canonical_actual_path(repo_root, path, canonical_relative_path)
    receipt, receipt_sha256 = _read_stable_json(repo_root, actual)
    validate_receipt_identity(
        receipt,
        expected_identity,
        expected_status="POSITIVE_CONTROL_PASS",
        expected_phase=f"{capacity}_positive_control",
    )
    _require_exact(receipt.get("capacity"), capacity, "positive-control capacity")
    _require_exact(receipt.get("model_seed"), model_seed, "positive-control model seed")
    _require_exact(
        receipt.get("data_manifest_sha256"),
        data_manifest_sha256,
        "positive-control data manifest",
    )
    stable_expected = stable_setup_receipt(expected_setup)
    _require_exact(
        stable_setup_receipt(receipt.get("setup")),
        stable_expected,
        "positive-control stable setup",
    )
    _require_exact(
        receipt["setup"].get("capacity"), capacity, "positive-control setup capacity"
    )
    _require_exact(
        receipt["setup"].get("model_seed"), model_seed, "positive-control setup seed"
    )
    setup_geometry = _validate_setup_semantics(
        receipt["setup"],
        expected_identity=expected_identity,
        capacity=capacity,
        model_seed=model_seed,
        expected_adaptation_targets=expected_adaptation_targets,
        expected_adaptation_parameters=expected_adaptation_parameters,
        expected_adaptation_dropout=expected_adaptation_dropout,
        expected_lora_rank=expected_lora_rank,
    )
    _require_exact(
        receipt.get("shared_initialization"),
        receipt["setup"].get("shared_initialization"),
        "positive-control shared initialization",
    )
    _validate_lineage_binding(
        repo_root,
        receipt.get("branch_authorization"),
        expected_branch_authorization,
        expected_identity=expected_identity,
        label="positive-control branch authorization",
    )
    expected_g0 = dict(_require_mapping(expected_g0_lineage, "expected G0 lineage"))
    _require_exact(receipt.get("g0_lineage"), expected_g0, "positive-control G0 lineage")
    expected_g0_path = canonical.parent / f"g0_{capacity}_seed{model_seed}.json"
    base_identity = _identity_base(expected_identity)
    g0 = reopen_lineage(
        repo_root,
        expected_g0,
        expected_identity=base_identity,
        expected_status="MODEL_SMOKE_PASS",
        expected_phase=f"{capacity}_g0",
        canonical_relative_path=expected_g0_path.as_posix(),
    )
    for field, expected in (
        ("capacity", capacity),
        ("model_seed", model_seed),
        ("data_manifest_sha256", data_manifest_sha256),
        ("branch_authorization", expected_branch_authorization),
    ):
        _require_exact(g0.get(field), expected, f"bound G0 {field}")
    _require_exact(
        stable_setup_receipt(g0.get("setup")), stable_expected, "bound G0 stable setup"
    )
    for field, expected in {
        "authorizes_positive_control": True,
        "authorizes_training": False,
        "authorizes_result_training": False,
        "authorizes_result_evaluation": False,
        "benchmark_files_read": 0,
        "result_payloads_opened": ["train"],
        "sealed_contrast_payloads_opened": [],
        "training_or_evaluation_started": False,
        "scientific_evidence": False,
    }.items():
        _require_exact(g0.get(field), expected, f"bound G0 {field}")

    access = {
        "authorizes_training": True,
        "authorizes_result_training": True,
        "authorizes_result_evaluation": False,
        "benchmark_files_read": 0,
        "result_payloads_opened": [],
        "sealed_contrast_payloads_opened": [],
        "scientific_evidence": False,
    }
    for field, expected in access.items():
        _require_exact(receipt.get(field), expected, f"positive-control {field}")

    rows_receipt = _require_mapping(receipt.get("control_rows"), "positive-control rows")
    if set(rows_receipt) != {
        "seed",
        "rows",
        "grid",
        "canonical_rows_sha256",
        "cross_result_structural_overlap",
    }:
        _fail("positive-control row receipt fields changed")
    _require_exact(rows_receipt.get("seed"), control_seed, "positive-control row seed")
    _require_exact(rows_receipt.get("rows"), control_rows, "positive-control row count")
    _require_exact(
        rows_receipt.get("cross_result_structural_overlap"),
        0,
        "positive-control result overlap",
    )
    if type(expected_control_rows) is not list:
        _fail("expected positive-control rows must be the exact generated list")
    canonical_rows = hashlib.sha256()
    for row in expected_control_rows:
        if not isinstance(row, Mapping):
            _fail("expected positive-control row is not a mapping")
        try:
            canonical_rows.update(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError) as exc:
            raise GateReceiptError(
                "expected positive-control rows are not finite canonical JSON"
            ) from exc
    _require_exact(
        rows_receipt.get("canonical_rows_sha256"),
        canonical_rows.hexdigest(),
        "positive-control canonical rows",
    )
    grid = _require_mapping(rows_receipt.get("grid"), "positive-control grid")
    expected_grid = _registered_factorial_grid(
        families=control_families,
        templates=control_templates,
        depths=control_depths,
        query_kinds=control_query_kinds,
        examples_per_cell=control_examples_per_cell,
    )
    _require_exact(grid, expected_grid, "positive-control exact factorial grid")
    _require_exact(sum(expected_grid.values()), control_rows, "positive-control factorial rows")

    microbatches = control_updates * gradient_accumulation
    for field, expected in (
        ("overfit_rows", control_rows),
        ("overfit_updates", control_updates),
        ("overfit_gradient_accumulation", gradient_accumulation),
        ("overfit_microbatches", microbatches),
    ):
        _require_exact(receipt.get(field), expected, f"positive-control {field}")
    oracle_analysis = validate_oracle_analysis_receipt(
        expected_control_rows,
        _require_mapping(
            receipt.get("oracle_analysis"), "positive-control oracle analysis"
        ),
    )
    oracle = _finite_number(
        receipt.get("oracle_readout_accuracy"),
        "positive-control oracle accuracy",
        minimum=min_oracle_readout_accuracy,
    )
    if oracle > 1.0:
        _fail("positive-control oracle accuracy exceeds one")
    _require_exact(
        oracle,
        oracle_analysis["terminal_joint_accuracy"],
        "positive-control oracle scalar alias",
    )
    if oracle_analysis["threshold"] < min_oracle_readout_accuracy:
        _fail("positive-control oracle-analysis threshold is below the registered gate")
    diagnostics = _require_mapping(
        receipt.get("training_diagnostics"), "positive-control diagnostics"
    )
    geometry = _require_mapping(
        diagnostics.get("geometry"), "positive-control diagnostic geometry"
    )
    expected_geometry = {
        "rows": control_rows,
        "optimizer_updates": control_updates,
        "gradient_accumulation": gradient_accumulation,
        "singleton_microbatches": microbatches,
        "loss_divisor": gradient_accumulation,
        "optimizer_zero_grad_calls": control_updates + 1,
        "adaptation_clip_calls": control_updates,
        "common_state_clip_calls": control_updates,
        "optimizer_step_calls": control_updates,
        "early_stopping": False,
        "checkpoint_selection": False,
    }
    _require_exact(geometry, expected_geometry, "positive-control diagnostic geometry")
    _require_exact(
        diagnostics.get("completed_updates"),
        control_updates,
        "positive-control completed updates",
    )
    _require_exact(
        diagnostics.get("completed_microbatches"),
        microbatches,
        "positive-control completed microbatches",
    )
    _validate_positive_optimizer_probes(
        diagnostics,
        updates=control_updates,
        accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        adaptation_gradient_clip=adaptation_gradient_clip,
        common_gradient_clip=common_gradient_clip,
    )
    final_correct, final_accuracy = _validate_positive_control_evaluations(
        diagnostics, rows=control_rows, updates=control_updates
    )
    _require_exact(
        receipt.get("overfit_final_joint_correct"),
        final_correct,
        "positive-control final correct",
    )
    _require_exact(
        receipt.get("overfit_final_joint_accuracy"),
        final_accuracy,
        "positive-control final accuracy",
    )
    if final_accuracy < min_overfit_final_joint_accuracy:
        _fail("positive-control final accuracy is below its registered threshold")
    _require_exact(
        diagnostics.get("parameter_values_changed"),
        True,
        "positive-control parameter-change flag",
    )
    initial_trainable = diagnostics.get("initial_trainable_parameters")
    _require_exact(
        initial_trainable,
        receipt["setup"].get("trainable_parameters"),
        "positive-control initial trainable receipt",
    )
    _validate_trainable_receipt(
        initial_trainable,
        adaptation_parameters=setup_geometry["adaptation_parameters"],
        adaptation_tensors=setup_geometry["adaptation_tensors"],
        label="positive-control initial trainables",
    )
    final_trainable = diagnostics.get("final_trainable_parameters")
    _validate_trainable_receipt(
        final_trainable,
        adaptation_parameters=setup_geometry["adaptation_parameters"],
        adaptation_tensors=setup_geometry["adaptation_tensors"],
        label="positive-control final trainables",
    )
    for field in ("total", "adaptation", "common", "tensor_count", "names_sha256"):
        _require_exact(
            _require_mapping(final_trainable, "positive-control final trainables").get(field),
            _require_mapping(initial_trainable, "positive-control initial trainables").get(field),
            f"positive-control final trainable {field}",
        )
    if final_trainable.get("values_sha256") == initial_trainable.get("values_sha256"):
        _fail("positive-control final trainable values did not change")
    _validate_optimizer_geometry(
        diagnostics.get("optimizer_state"),
        setup=receipt["setup"],
        adaptation_parameters=setup_geometry["adaptation_parameters"],
        adaptation_tensors=setup_geometry["adaptation_tensors"],
        missing_common_state_exemptions=1,
        label="positive-control optimizer state",
    )
    deltas = _require_mapping(
        diagnostics.get("final_parameter_delta_norms"),
        "positive-control parameter deltas",
    )
    for group in ("adaptation_output", "common_state"):
        details = _require_mapping(deltas.get(group), f"positive-control {group} delta")
        if _finite_number(
            details.get("l2_delta_norm"), f"positive-control {group} delta", minimum=0.0
        ) <= 0.0:
            _fail(f"positive-control {group} did not move")
    return receipt


def _analysis_sibling(canonical: PurePosixPath, filename: str) -> str:
    return (canonical.parent / filename).as_posix()


def _require_registered_formation(
    value: Any, label: str, *, passes: bool
) -> dict[str, Any]:
    formation = _require_mapping(value, label)
    _require_exact(formation.get("passes"), passes, f"{label} passes")
    if type(formation.get("status")) is not str or not formation["status"]:
        _fail(f"{label} status must be a nonempty producer status")
    return dict(formation)


def _validate_branch_evidence(
    repo_root: str | os.PathLike[str],
    receipt: Mapping[str, Any],
    *,
    branch: str,
    expected_identity: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> None:
    """Recompute the exact producer evidence without writing an analysis file.

    Importing lazily avoids making ordinary setup-receipt/archive operations
    import the model-bearing analysis module.  The import is reached only for a
    conditional scientific authorization, where status-only validation would
    be unsafe.
    """

    try:
        from .analysis import validate_branch_evidence_receipt
    except (ImportError, AttributeError) as exc:  # pragma: no cover - packaging failure
        raise RuntimeError("branch evidence validator is unavailable") from exc
    validate_branch_evidence_receipt(
        repo_root,
        receipt,
        branch=branch,
        expected_identity=expected_identity,
        lineage=lineage,
    )


def validate_branch_authorization(
    repo_root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    *,
    canonical_relative_path: str,
    branch: str,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one canonical branch and only its registered named ancestry.

    The return value includes the current ``receipt`` and ``lineage``, the
    canonical ``root_lora_miss_lineage``, and ``stage_b_lineage`` when that
    stage has been reached.  No recursive search is performed, so a valid
    lineage hidden in an arbitrary nested decoy cannot authorize a branch.
    """

    spec = _BRANCH_SPECS.get(branch)
    if spec is None:
        _fail(f"unknown branch authorization contract: {branch!r}")
    canonical = _canonical_relative(canonical_relative_path)
    if canonical.name != spec["filename"] or canonical.parent.name != "analysis":
        _fail("branch receipt is not at its status-specific canonical path")
    actual = _require_canonical_actual_path(repo_root, path, canonical_relative_path)
    receipt, receipt_sha256 = _read_stable_json(repo_root, actual)
    validate_receipt_identity(
        receipt,
        expected_identity,
        expected_status=str(spec["status"]),
        expected_phase=str(spec["phase"]),
    )
    _require_exact(
        receipt.get("analysis_phase"), spec["analysis_phase"], "branch analysis phase"
    )
    _require_exact(receipt.get("verdict"), spec["status"], "branch verdict")
    _require_exact(receipt.get("next_stage"), spec["next_stage"], "branch next stage")
    current_lineage = {
        "path": canonical.as_posix(),
        "sha256": receipt_sha256,
        "receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "status": receipt["status"],
        "phase": receipt["phase"],
    }
    base_identity = _identity_base(expected_identity, expected_phase=str(spec["phase"]))

    if branch == LORA_MISS_BRANCH:
        _require_exact(receipt.get("authorization"), None, "LoRA-miss authorization")
        _require_registered_formation(receipt.get("formation"), "LoRA formation", passes=False)
        _validate_branch_evidence(
            repo_root,
            receipt,
            branch=branch,
            expected_identity=base_identity,
            lineage=current_lineage,
        )
        return {
            "receipt": receipt,
            "lineage": current_lineage,
            "root_lora_miss_lineage": current_lineage,
            "stage_b_lineage": None,
        }

    if branch in (STAGE_B_CONTRAST_BRANCH, STAGE_B_FULLRANK_MISS_BRANCH):
        root_path = _analysis_sibling(canonical, "lora_joint_trigger.json")
        root_entry = receipt.get("authorization")
        root_receipt = reopen_lineage(
            repo_root,
            _require_mapping(root_entry, "Stage-B LoRA-miss authorization"),
            expected_identity=base_identity,
            expected_status="LORA_JOINT_MISS_CONTROLS_REQUIRED",
            expected_phase="lora_joint_analysis",
            canonical_relative_path=root_path,
        )
        root_validated = validate_branch_authorization(
            repo_root,
            canonical_repo_path(repo_root, root_path),
            canonical_relative_path=root_path,
            branch=LORA_MISS_BRANCH,
            expected_identity=base_identity,
        )
        _require_exact(
            root_entry,
            root_validated["lineage"],
            "Stage-B root LoRA-miss lineage",
        )
        # This direct named control ancestry is part of the Stage-B seal.  A
        # lineage placed elsewhere in the receipt is intentionally ignored.
        control_path = _analysis_sibling(canonical, "lora_control.json")
        control_entry = receipt.get("lora_control_analysis")
        control = reopen_lineage(
            repo_root,
            _require_mapping(control_entry, "Stage-B LoRA-control analysis"),
            expected_identity=base_identity,
            expected_status="LORA_STATE_ONLY_CONTROL_COMPLETE",
            expected_phase="lora_control_analysis",
            canonical_relative_path=control_path,
        )
        _require_exact(
            control.get("authorization"), root_entry, "LoRA-control root authorization"
        )
        _require_exact(
            control.get("analysis_phase"), "lora_control", "LoRA-control analysis phase"
        )
        _require_exact(
            control.get("next_stage"),
            "continue_mandatory_stage_b_seal",
            "LoRA-control next stage",
        )
        control_formation = _require_registered_formation(
            control.get("formation"),
            "LoRA-control formation",
            passes=bool(_require_mapping(control.get("formation"), "LoRA-control formation").get("passes")),
        )
        expected_control_verdict = (
            "LORA_CAN_FORM_STATE_STATE_ONLY"
            if control_formation["passes"]
            else control_formation["status"]
        )
        _require_exact(
            control.get("verdict"), expected_control_verdict, "LoRA-control verdict"
        )
        expected_annotation = (
            "LORA_CAN_FORM_STATE_STATE_ONLY" if control_formation["passes"] else None
        )
        _require_exact(
            control.get("lora_state_only_annotation"),
            expected_annotation,
            "LoRA-control state-only annotation",
        )
        root_formation = _require_registered_formation(
            root_receipt.get("formation"), "root LoRA formation", passes=False
        )
        stage_lora_formation = _require_registered_formation(
            receipt.get("lora_joint_formation"), "Stage-B LoRA formation", passes=False
        )
        _require_exact(
            stage_lora_formation,
            root_formation,
            "Stage-B direct LoRA-miss formation equality",
        )
        stage_control_formation = _require_registered_formation(
            receipt.get("lora_state_only_formation"),
            "Stage-B LoRA state-only formation",
            passes=control_formation["passes"],
        )
        _require_exact(
            stage_control_formation,
            control_formation,
            "Stage-B direct LoRA-control formation equality",
        )
        _require_exact(
            receipt.get("lora_state_only_annotation"),
            expected_annotation,
            "Stage-B LoRA state-only annotation",
        )
        _require_registered_formation(
            receipt.get("fullrank_trigger_formation"),
            "Stage-B full-rank formation",
            passes=branch == STAGE_B_CONTRAST_BRANCH,
        )
        matching = _require_mapping(receipt.get("matching"), "Stage-B matching")
        _require_exact(
            matching.get("status"), "STAGE_B_MATCHING_VALID", "Stage-B matching status"
        )
        firewall = _require_mapping(
            receipt.get("contrast_firewall"), "Stage-B contrast firewall"
        )
        _require_exact(
            firewall.get("status"),
            "CONTRAST_FIREWALL_UNOPENED",
            "Stage-B contrast firewall status",
        )
        _validate_branch_evidence(
            repo_root,
            receipt,
            branch=branch,
            expected_identity=base_identity,
            lineage=current_lineage,
        )
        return {
            "receipt": receipt,
            "lineage": current_lineage,
            "root_lora_miss_lineage": root_validated["lineage"],
            "stage_b_lineage": current_lineage,
        }

    stage_path = _analysis_sibling(canonical, "stage_b_seal.json")
    stage_entry = receipt.get("authorization")
    reopen_lineage(
        repo_root,
        _require_mapping(stage_entry, "postcontrast Stage-B authorization"),
        expected_identity=base_identity,
        expected_status="STAGE_B_CONTRAST_AUTHORIZED",
        expected_phase="stage_b_seal_analysis",
        canonical_relative_path=stage_path,
    )
    stage_validated = validate_branch_authorization(
        repo_root,
        canonical_repo_path(repo_root, stage_path),
        canonical_relative_path=stage_path,
        branch=STAGE_B_CONTRAST_BRANCH,
        expected_identity=base_identity,
    )
    _require_exact(stage_entry, stage_validated["lineage"], "postcontrast Stage-B lineage")
    _require_registered_formation(
        receipt.get("lora_sealed_contrast_formation"),
        "postcontrast LoRA sealed formation",
        passes=False,
    )
    _require_registered_formation(
        receipt.get("trigger_formation"),
        "postcontrast full-rank trigger formation",
        passes=True,
    )
    _require_registered_formation(
        receipt.get("sealed_contrast_formation"),
        "postcontrast full-rank sealed formation",
        passes=False,
    )
    _validate_branch_evidence(
        repo_root,
        receipt,
        branch=branch,
        expected_identity=base_identity,
        lineage=current_lineage,
    )
    return {
        "receipt": receipt,
        "lineage": current_lineage,
        "root_lora_miss_lineage": stage_validated["root_lora_miss_lineage"],
        "stage_b_lineage": stage_validated["lineage"],
    }
