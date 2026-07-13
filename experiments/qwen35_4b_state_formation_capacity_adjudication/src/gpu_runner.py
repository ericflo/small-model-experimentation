"""Pinned Transformers runner for matched state-formation capacity arms."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.hub import cached_file, extract_commit_hash
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

from .adaptation import microbatch_dropout_seed
from .attempt_receipts import (
    ATTEMPT_MARKER_NAME,
    AttemptReceiptError,
    canonical_path as canonical_attempt_path,
    complete_training_attempt,
    ensure_attempt_output,
    prepare_training_attempt,
    required_training_replay_archive,
    repo_relative as attempt_repo_relative,
    start_training_attempt,
    validate_training_attempt_history,
)
from .config import (
    G0_COMPLETED_CHECKS,
    G0_FAILURE_STAGE_PREFIX_LENGTHS,
    MODEL_ID,
    MODEL_REVISION,
    config_sha256,
    requirements_training_lock_bytes,
    require_confirmatory_config,
    source_contract_sha256,
)
from .data_pipeline import (
    load_contrast_access_ledger,
    record_contrast_access,
    start_contrast_access,
    validate_data_manifest,
    validated_data_rows,
)
from .design_boundary import design_lineage, validate_design_receipt
from .gate_receipts import (
    LORA_MISS_BRANCH,
    POSTCONTRAST_FULLRANK_MISS_BRANCH,
    STAGE_B_CONTRAST_BRANCH,
    STAGE_B_FULLRANK_MISS_BRANCH,
    canonical_repo_path,
    stable_setup_receipt as strict_stable_setup_receipt,
    validate_branch_authorization,
    validate_g0_pass,
    validate_positive_control_pass,
)
from .initialization import build_shared_state, load_initialization_bundle, tensor_manifest
from .mechanics import recurrent_compute_receipt
from .optimizer_receipts import optimizer_state_receipt
from .oracle_control import (
    analyze_positive_control_records,
    generate_control_rows,
    produce_oracle_analysis_receipt,
)
from .safe_io import (
    open_stable_directory_for_update,
    open_stable_regular,
    publish_new_bytes,
    publish_new_file,
    read_stable_bytes,
    read_stable_json_object,
    read_verified_json_object,
)
from .state_loop_model import StateLoopModel
from .substrate import LETTERS, generate_example, trajectory_targets, verify_example
from .training_receipts import (
    STAGE_A_MATRIX,
    STAGE_B_MATRIX,
    STAGE_C_MATRIX,
    TrainingCell,
    TrainingReceiptContract,
    canonical_training_cell_paths,
    evaluation_barrier,
    recover_published_training_completion,
    training_barrier,
    training_launch_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
REQUIREMENTS_LOCK = REPO_ROOT / "requirements-training.lock.txt"
ANSWER_STRINGS = tuple(f" {letter}" for letter in LETTERS)
EXPECTED_GPU_NAME = "NVIDIA RTX 6000 Ada Generation"
EXPECTED_COMPUTE_CAPABILITY = "8.9"
PINNED_SNAPSHOT_FILES = (
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _write_json(path: Path, payload: object) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("JSON receipt payload must be an object")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    publish_new_bytes(_publication_root(path), path, encoded, mode=0o644)


def _publication_root(path: Path) -> Path:
    """Choose the canonical trusted root for a production or isolated test path."""

    lexical = Path(os.path.abspath(os.fspath(path)))
    return REPO_ROOT if lexical.is_relative_to(REPO_ROOT) else lexical.parent


def _atomic_copy(source: Path, destination: Path) -> None:
    _copy_new_fsynced(source, destination)


def _copy_new_fsynced(source: Path, destination: Path) -> None:
    """Copy to a new inode without replacement and durably bind its directory."""

    _require_no_symlink_ancestors(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestors(destination)
    lexical = Path(os.path.abspath(os.fspath(source)))
    trusted_root = REPO_ROOT if lexical.is_relative_to(REPO_ROOT) else lexical.parent
    def copy(writer: Any) -> None:
        with open_stable_regular(trusted_root, lexical) as reader:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)

    publish_new_file(
        _publication_root(destination), destination, copy, mode=0o644
    )


def _fsync_file(path: Path) -> None:
    lexical = Path(os.path.abspath(os.fspath(path)))
    trusted_root = REPO_ROOT if lexical.is_relative_to(REPO_ROOT) else lexical.parent
    with open_stable_regular(trusted_root, lexical) as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_no_symlink_ancestors(path: Path) -> None:
    current = path.parent
    while True:
        if current.is_symlink():
            raise RuntimeError(f"refusing to persist through symlinked ancestor: {current}")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _write_new_json_pair(
    canonical: Path,
    mirror: Path,
    payload: Mapping[str, Any],
) -> None:
    """Publish a mirror-first, independently inoded immutable receipt pair."""

    _require_no_symlink_ancestors(canonical)
    _require_no_symlink_ancestors(mirror)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestors(canonical)
    _require_no_symlink_ancestors(mirror)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    publish_new_bytes(_publication_root(mirror), mirror, encoded, mode=0o644)
    _receipt_pair_checkpoint("mirror_published")
    publish_new_bytes(_publication_root(canonical), canonical, encoded, mode=0o644)


def _receipt_pair_checkpoint(stage: str) -> None:
    """Fault-injection seam after the durable mirror-first pair prefix."""

    del stage


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably install one new JSON file without ever replacing a leaf."""

    _require_no_symlink_ancestors(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestors(path)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    publish_new_bytes(_publication_root(path), path, encoded, mode=0o644)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    _require_no_symlink_ancestors(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestors(path)
    existed = os.path.lexists(path)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("JSONL destination is not one unaliased regular file")
        with os.fdopen(descriptor, "a", encoding="utf-8", closefd=False) as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        rebound = path.lstat()
        if (
            not stat.S_ISREG(rebound.st_mode)
            or (rebound.st_dev, rebound.st_ino) != (info.st_dev, info.st_ino)
        ):
            raise RuntimeError("JSONL destination changed while appending")
    finally:
        os.close(descriptor)
    if not existed:
        _fsync_directory(path.parent)


def _clip_scale(preclip_norm: float, maximum_norm: float) -> float:
    if not math.isfinite(preclip_norm) or preclip_norm < 0.0 or maximum_norm <= 0.0:
        raise RuntimeError("invalid gradient-clipping receipt input")
    return min(1.0, maximum_norm / (preclip_norm + 1e-6))


def _stable_size_sha256(trusted_root: Path, path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with open_stable_regular(trusted_root, path) as handle:
        size = os.fstat(handle.fileno()).st_size
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return int(size), digest.hexdigest()


def _sha256(path: Path) -> str:
    lexical = Path(os.path.abspath(os.fspath(path)))
    trusted_root = REPO_ROOT if lexical.is_relative_to(REPO_ROOT) else lexical.parent
    return _stable_size_sha256(trusted_root, lexical)[1]


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _requirements_sha256() -> str:
    return _requirements_snapshot()[1]


def _requirements_snapshot() -> tuple[list[str], str]:
    raw = requirements_training_lock_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError("requirements lock is not UTF-8") from exc
    return lines, hashlib.sha256(raw).hexdigest()


def _locked_requirement_version(package: str) -> str:
    """Return the one exact version pinned for ``package`` in the run lock."""

    prefix = f"{package}=="
    lines, _ = _requirements_snapshot()
    matches = [
        line[len(prefix) :].strip()
        for line in lines
        if line.startswith(prefix)
    ]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError(
            f"training lock does not pin exactly one {package} version"
        )
    return matches[0]


def _installed_environment_lock_receipt() -> dict[str, Any]:
    lines, requirements_sha256 = _requirements_snapshot()
    expected = {}
    for line in lines:
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^ ;]+)$", line)
        if match:
            expected[match.group(1)] = match.group(2)
    expected["causal_conv1d"] = "1.6.2.post1"
    installed = {}
    for package, required in sorted(expected.items()):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"pinned training package is missing: {package}") from exc
        if actual != required:
            raise RuntimeError(
                f"training environment drift for {package}: {actual} != {required}"
            )
        installed[package] = actual
    return {
        "requirements_training_lock_sha256": requirements_sha256,
        "causal_conv1d_out_of_band_pin": "1.6.2.post1",
        "packages": installed,
        "packages_sha256": _canonical_sha256({"packages": installed}),
    }


def _repo_relative(path: Path) -> str:
    try:
        return attempt_repo_relative(REPO_ROOT, path)
    except AttemptReceiptError as exc:
        raise RuntimeError(f"lineage path is not canonical: {path}") from exc


def _resolve_repo_path(value: str) -> Path:
    try:
        candidate = Path(value)
        if candidate.is_absolute():
            lexical = Path(os.path.abspath(value))
            if value != lexical.as_posix():
                raise RuntimeError(f"lineage path is not lexical-canonical: {value}")
            relative = lexical.relative_to(REPO_ROOT).as_posix()
        else:
            relative = value
        return canonical_attempt_path(REPO_ROOT, relative)
    except (AttemptReceiptError, ValueError) as exc:
        raise RuntimeError(f"lineage path is not canonical: {value}") from exc


def _require_new_output(path: Path, *, directory: bool, kind: str) -> None:
    if os.path.lexists(path):
        raise RuntimeError(f"refusing to resume or overwrite {kind}: {path}")
    if directory:
        path.mkdir(parents=True, exist_ok=False)
        _fsync_directory(path.parent)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _fsync_directory(path.parent)


def _require_cuda(minimum_gib: float = 44.0) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for every model-bearing stage")
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / (1024**3)
    free_bytes, runtime_total_bytes = torch.cuda.mem_get_info(0)
    free_gib = free_bytes / (1024**3)
    compute_capability = f"{properties.major}.{properties.minor}"
    if properties.name != EXPECTED_GPU_NAME or compute_capability != EXPECTED_COMPUTE_CAPABILITY:
        raise RuntimeError(
            "the experiment is frozen to NVIDIA RTX 6000 Ada Generation / compute capability 8.9; "
            f"found {properties.name!r} / {compute_capability}"
        )
    if total_gib < minimum_gib or free_gib < minimum_gib:
        raise RuntimeError(
            f"at least {minimum_gib:.0f} GiB total and free VRAM is required; "
            f"found {total_gib:.1f} total/{free_gib:.1f} free"
        )
    if not is_flash_linear_attention_available() or not is_causal_conv1d_available():
        raise RuntimeError("Qwen3.5 fast-path extensions are unavailable; rebuild the pinned environment")
    queried = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip().splitlines()
    if len(queried) != 1:
        raise RuntimeError("the registered run requires exactly one physical GPU")
    gpu_uuid, queried_name = [part.strip() for part in queried[0].split(",", 1)]
    if queried_name != EXPECTED_GPU_NAME or not gpu_uuid.startswith("GPU-"):
        raise RuntimeError("nvidia-smi GPU identity differs from the registered device")
    return {
        "name": properties.name,
        "uuid": gpu_uuid,
        "total_memory_gib": total_gib,
        "free_memory_gib_before_load": free_gib,
        "runtime_total_memory_gib": runtime_total_bytes / (1024**3),
        "compute_capability": compute_capability,
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
    }


def _environment_receipt(*, include_device: bool) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if include_device:
        receipt["device"] = _require_cuda(0.0)
    return receipt


def _identity(config: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    design = design_lineage(config)
    return {
        "experiment_id": config["experiment_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers",
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(ROOT),
        "requirements_training_lock_sha256": _requirements_sha256(),
        "design_receipt_sha256": design["sha256"],
        "design_receipt_identity_sha256": design["receipt_identity_sha256"],
        "phase": phase,
    }


def _failed_archive_header(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FAILED_ATTEMPT_ARCHIVED",
        "experiment_id": config["experiment_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers",
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(ROOT),
        "requirements_training_lock_sha256": _requirements_sha256(),
        "design_lineage": design_lineage(config),
        "scientific_evidence": False,
    }


def _with_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(payload)
    receipt["receipt_identity_sha256"] = _canonical_sha256(receipt)
    return receipt


def _read_receipt(
    path: Path,
    config: Mapping[str, Any],
    *,
    statuses: set[str],
    phases: set[str],
    label: str,
) -> dict[str, Any]:
    path = _resolve_repo_path(str(path))
    if not path.is_file():
        raise RuntimeError(f"{label} receipt is missing: {path}")
    receipt = read_stable_json_object(REPO_ROOT, path)
    if type(receipt.get("schema_version")) is not int or receipt.get("schema_version") != 1:
        raise RuntimeError(f"{label} receipt schema is invalid")
    status = receipt.get("status")
    if type(status) is not str:
        raise RuntimeError(f"{label} receipt requires explicit status")
    if status == "SETUP_CONTROL_FAILED":
        raise RuntimeError(f"{label} did not authorize this stage: {status!r}")
    if status not in statuses or receipt.get("phase") not in phases:
        raise RuntimeError(f"{label} did not authorize this stage: {status!r}/{receipt.get('phase')!r}")
    claimed = receipt.get("receipt_identity_sha256")
    payload = {key: value for key, value in receipt.items() if key != "receipt_identity_sha256"}
    if claimed != _canonical_sha256(payload):
        raise RuntimeError(f"{label} receipt identity mismatch")
    expected = _identity(config, phase=str(receipt["phase"]))
    for key, value in expected.items():
        if type(receipt.get(key)) is not type(value) or receipt.get(key) != value:
            raise RuntimeError(f"{label} identity mismatch for {key}")
    return receipt


def _read_g0_pass(
    path: Path,
    config: Mapping[str, Any],
    *,
    capacity: str,
) -> dict[str, Any]:
    """Read a G0 pass only when its access and authorization claims are exact."""

    receipt = _read_receipt(
        path,
        config,
        statuses={"MODEL_SMOKE_PASS"},
        phases={f"{capacity}_g0"},
        label=f"{capacity} G0",
    )
    expected = {
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
    for field, value in expected.items():
        actual = receipt.get(field)
        if type(actual) is not type(value) or actual != value:
            raise RuntimeError(f"{capacity} G0 has invalid {field}")
    return receipt


def _lineage(path: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    status = receipt.get("status")
    if type(status) is not str or not status:
        raise RuntimeError("lineage source requires an explicit nonempty status")
    return {
        "path": _repo_relative(path),
        "sha256": _sha256(path),
        "receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "status": status,
        "phase": receipt["phase"],
    }


def validate_lineage_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    required = {"path", "sha256", "receipt_identity_sha256", "status", "phase"}
    if set(entry) != required:
        raise RuntimeError("gate lineage entry has the wrong fields")
    path = _resolve_repo_path(str(entry["path"]))
    if not path.is_file() or _sha256(path) != entry["sha256"]:
        raise RuntimeError(f"gate lineage file changed: {path}")
    receipt = read_verified_json_object(REPO_ROOT, path, str(entry["sha256"]))
    claimed = receipt.get("receipt_identity_sha256")
    payload = {
        key: value for key, value in receipt.items()
        if key != "receipt_identity_sha256"
    }
    if claimed != _canonical_sha256(payload):
        raise RuntimeError("gate lineage receipt self-identity changed")
    if claimed != entry["receipt_identity_sha256"]:
        raise RuntimeError("gate lineage receipt identity changed")
    if type(receipt.get("status")) is not str or receipt.get("status") != entry["status"]:
        raise RuntimeError("gate lineage status changed")
    if receipt.get("phase") != entry["phase"]:
        raise RuntimeError("gate lineage phase changed")
    return receipt


def _render(tokenizer: Any, row: Mapping[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _validate_tokenizer(tokenizer: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    state_ids = tokenizer.encode(config["architecture"]["state_token"], add_special_tokens=False)
    if len(state_ids) != 1 or state_ids[0] == tokenizer.unk_token_id:
        raise RuntimeError(f"state token is not one known token: {state_ids}")
    answer_ids = []
    for answer in ANSWER_STRINGS:
        ids = tokenizer.encode(answer, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"answer {answer!r} is not a single token: {ids}")
        answer_ids.append(ids[0])
    if len(set(answer_ids)) != 4:
        raise RuntimeError("answer token IDs are not distinct")
    return {"state_token_id": state_ids[0], "answer_token_ids": answer_ids}


def _pinned_snapshot_receipt() -> dict[str, Any]:
    """Prove every model/tokenizer payload resolves through the pinned snapshot.

    Transformers 5.13.0's Qwen3.5 causal-LM wrapper retains a derived text
    configuration whose ``_commit_hash`` is ``None`` even when
    ``from_pretrained`` was given an exact commit.  Revision provenance must
    therefore come from the resolved cache paths, not that lossy runtime field.
    """

    receipts: dict[str, dict[str, Any]] = {}
    snapshot_roots: set[Path] = set()

    resolved_content: dict[str, tuple[Path, Path]] = {}

    def resolve(filename: str) -> Path:
        if Path(filename).name != filename:
            raise RuntimeError(f"pinned snapshot filename is not basename-only: {filename}")
        resolved = cached_file(
            MODEL_ID,
            filename,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
        if not resolved:
            raise RuntimeError(f"pinned snapshot file is missing: {filename}")
        path = Path(os.path.abspath(os.fspath(resolved)))
        if path.name != filename:
            raise RuntimeError(
                f"resolved pinned snapshot basename changed: {path.name!r} != {filename!r}"
            )
        resolved_revision = extract_commit_hash(str(path), None)
        if resolved_revision != MODEL_REVISION:
            raise RuntimeError(
                f"{filename} resolved outside the pinned model revision: "
                f"{resolved_revision!r} != {MODEL_REVISION!r}"
            )
        snapshot_root = path.parent
        if (
            snapshot_root.name != resolved_revision
            or snapshot_root.parent.name != "snapshots"
        ):
            raise RuntimeError("resolved pinned file is outside the canonical snapshot root")
        model_cache_root = snapshot_root.parent.parent
        try:
            with open_stable_directory_for_update(
                model_cache_root,
                snapshot_root,
            ) as snapshot_descriptor:
                before = os.stat(
                    filename,
                    dir_fd=snapshot_descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(before.st_mode):
                    if before.st_nlink != 1:
                        raise RuntimeError(
                            f"resolved pinned snapshot symlink is aliased: {filename}"
                        )
                    target = os.readlink(filename, dir_fd=snapshot_descriptor)
                    match = re.fullmatch(
                        r"\.\./\.\./blobs/([0-9a-f]{40}|[0-9a-f]{64})",
                        target,
                    )
                    if match is None:
                        raise RuntimeError(
                            "resolved pinned snapshot symlink is not one canonical "
                            "content-addressed blob"
                        )
                    content_root = model_cache_root
                    content_path = model_cache_root / "blobs" / match.group(1)
                elif stat.S_ISREG(before.st_mode):
                    if before.st_nlink != 1:
                        raise RuntimeError(
                            f"resolved pinned snapshot file is aliased: {filename}"
                        )
                    target = None
                    content_root = snapshot_root
                    content_path = path
                else:
                    raise RuntimeError(
                        f"resolved pinned snapshot file is not readable: {path}"
                    )
                size, digest = _stable_size_sha256(content_root, content_path)
                after = os.stat(
                    filename,
                    dir_fd=snapshot_descriptor,
                    follow_symlinks=False,
                )
                if (
                    (after.st_dev, after.st_ino, after.st_mode, after.st_nlink)
                    != (before.st_dev, before.st_ino, before.st_mode, before.st_nlink)
                    or (
                        target is not None
                        and os.readlink(filename, dir_fd=snapshot_descriptor) != target
                    )
                ):
                    raise RuntimeError(
                        f"resolved pinned snapshot entry changed while hashing: {filename}"
                    )
        except OSError as exc:
            raise RuntimeError(
                f"resolved pinned snapshot file is not readable: {path}"
            ) from exc
        snapshot_roots.add(snapshot_root)
        receipts[filename] = {
            "filename": filename,
            "resolved_revision": resolved_revision,
            "bytes": size,
            "sha256": digest,
        }
        resolved_content[filename] = (content_root, content_path)
        return path

    resolved_paths = {filename: resolve(filename) for filename in PINNED_SNAPSHOT_FILES}
    index_path = resolved_paths["model.safetensors.index.json"]
    index_root, index_content_path = resolved_content[
        "model.safetensors.index.json"
    ]
    try:
        index = read_verified_json_object(
            index_root,
            index_content_path,
            receipts["model.safetensors.index.json"]["sha256"],
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as exc:
        raise RuntimeError("pinned safetensors index is unreadable") from exc
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise RuntimeError("pinned safetensors index has no weight map")
    shard_names = sorted(set(weight_map.values()))
    if not all(
        isinstance(name, str)
        and Path(name).name == name
        and name.endswith(".safetensors")
        for name in shard_names
    ):
        raise RuntimeError("pinned safetensors index contains an invalid shard path")
    for shard_name in shard_names:
        resolve(shard_name)
    if len(snapshot_roots) != 1:
        raise RuntimeError("pinned model files resolved through mixed snapshot roots")
    files = [receipts[name] for name in sorted(receipts)]
    resolved_revisions = {entry["resolved_revision"] for entry in files}
    if resolved_revisions != {MODEL_REVISION}:
        raise RuntimeError("pinned model files resolved through mixed revisions")
    receipt = {
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": next(iter(resolved_revisions)),
        "snapshot_layout": f"snapshots/{MODEL_REVISION}",
        "files": files,
    }
    receipt["files_sha256"] = _canonical_sha256({"files": files})
    return receipt


def _load_base(config: Mapping[str, Any]) -> tuple[Any, nn.Module, dict[str, Any]]:
    if transformers.__version__ != str(config["model"]["transformers_version"]):
        raise RuntimeError("Transformers version drift")
    snapshot_receipt = _pinned_snapshot_receipt()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    token_receipt = _validate_tokenizer(tokenizer, config)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
        use_safetensors=True,
    ).cuda()
    model.config.use_cache = False
    runtime_commit_hash = getattr(model.config, "_commit_hash", None)
    if runtime_commit_hash not in {None, MODEL_REVISION}:
        raise RuntimeError("loaded model revision differs from the pinned revision")
    token_receipt["pinned_snapshot"] = snapshot_receipt
    token_receipt["runtime_model_config_commit_hash"] = runtime_commit_hash
    model.requires_grad_(False)
    return tokenizer, model, token_receipt


def _discover_targets(model: nn.Module, start: int, end: int) -> list[str]:
    pattern = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.")
    targets = []
    for name, module in model.named_modules():
        match = pattern.search(name)
        if match and isinstance(module, nn.Linear) and start <= int(match.group(1)) < end:
            targets.append(name)
    if not targets:
        raise RuntimeError("no loop-block linear targets discovered")
    return sorted(targets)


def _parameter_groups(wrapper: StateLoopModel) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    adaptation = [parameter for parameter in wrapper.adaptation.parameters() if parameter.requires_grad]
    adaptation_ids = {id(parameter) for parameter in adaptation}
    common = [
        parameter for parameter in wrapper.parameters()
        if parameter.requires_grad and id(parameter) not in adaptation_ids
    ]
    if not adaptation or not common:
        raise RuntimeError("adaptation/common trainable parameter partition is empty")
    return adaptation, common


def _build_optimizer(
    adaptation_parameters: Sequence[nn.Parameter],
    common_parameters: Sequence[nn.Parameter],
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    adaptation_ids = {id(parameter) for parameter in adaptation_parameters}
    common_ids = {id(parameter) for parameter in common_parameters}
    if not adaptation_ids or not common_ids or adaptation_ids & common_ids:
        raise RuntimeError("optimizer parameter groups must be nonempty and disjoint")
    return torch.optim.AdamW(
        [
            {
                "params": list(adaptation_parameters),
                "group_name": "adaptation",
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
            },
            {
                "params": list(common_parameters),
                "group_name": "common_state",
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
            },
        ]
    )


def _trainable_receipt(wrapper: StateLoopModel) -> dict[str, Any]:
    named = sorted(
        ((name, parameter) for name, parameter in wrapper.named_parameters() if parameter.requires_grad),
        key=lambda pair: pair[0],
    )
    digest = hashlib.sha256()
    adaptation_parameters = common_parameters = 0
    for name, parameter in named:
        if name.startswith("adaptation."):
            adaptation_parameters += parameter.numel()
        else:
            common_parameters += parameter.numel()
        digest.update(name.encode("utf-8"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        flat = parameter.detach().contiguous().reshape(-1)
        for chunk in flat.split(8 * 1024 * 1024):
            digest.update(chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return {
        "total": adaptation_parameters + common_parameters,
        "adaptation": adaptation_parameters,
        "common": common_parameters,
        "tensor_count": len(named),
        "names_sha256": hashlib.sha256(
            "\n".join(name for name, _ in named).encode("utf-8")
        ).hexdigest(),
        "values_sha256": digest.hexdigest(),
    }


def _nonadapter_dropout_receipt(wrapper: StateLoopModel) -> dict[str, Any]:
    active = []
    for name, module in wrapper.named_modules():
        if isinstance(module, nn.Dropout) and float(module.p) > 0.0:
            active.append({"name": name, "p": float(module.p)})
    if active:
        raise RuntimeError(f"uncontrolled nn.Dropout modules remain active: {active}")
    config_values = {}
    for name, value in vars(wrapper.text.config).items():
        if "dropout" in name.lower() and isinstance(value, (int, float)):
            config_values[name] = float(value)
    active_config = {name: value for name, value in config_values.items() if value != 0.0}
    if active_config:
        raise RuntimeError(f"uncontrolled model-config dropout remains active: {active_config}")
    return {
        "active_nn_dropout_modules": active,
        "model_config_dropout_values": dict(sorted(config_values.items())),
        "matched_adaptation_dropout": wrapper.adaptation.dropout,
    }


def _peft_formula_parity_receipt() -> dict[str, Any]:
    """Compare the actual hook with PEFT in exact and live-like regimes."""
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear as PeftLoraLinear

    from .adaptation import AdaptationBank

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    def probe(dtype: torch.dtype, dropout: float, *, autocast_enabled: bool) -> dict[str, Any]:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(99731)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(99731)
            reference_base = nn.Linear(11, 7, bias=False, dtype=dtype, device=device)
            custom_base = nn.Sequential()
            custom_base.add_module(
                "target", nn.Linear(11, 7, bias=False, dtype=dtype, device=device)
            )
            with torch.no_grad():
                custom_base.target.weight.copy_(reference_base.weight)
            reference_base.weight.requires_grad_(False)
            custom_base.target.weight.requires_grad_(False)
            peft_config = LoraConfig(
                r=5, lora_alpha=10, lora_dropout=dropout, init_lora_weights=False
            )
            reference = PeftLoraLinear(
                reference_base, "default", config=peft_config, r=5,
                lora_alpha=10, lora_dropout=dropout, init_lora_weights=False,
            ).to(device)
            custom = AdaptationBank(
                custom_base, ["target"], capacity="lora", model_seed=99731,
                config={
                    "lora": {"rank": 5, "dropout": dropout, "scale": 2.0},
                    "fullrank": {"dropout": dropout, "scale": 2.0},
                },
            ).to(device)
            a_value = torch.randn_like(reference.lora_A["default"].weight)
            b_value = torch.randn_like(reference.lora_B["default"].weight)
            with torch.no_grad():
                reference.lora_A["default"].weight.copy_(a_value)
                reference.lora_B["default"].weight.copy_(b_value)
                custom.down["d000"].weight.copy_(a_value.float())
                custom.up["d000"].weight.copy_(b_value.float())
            reference.train()
            custom.train()
            value = torch.randn(3, 11, device=device, dtype=dtype)
            dropout_seed = 88421
            if device.type == "cuda":
                torch.cuda.manual_seed_all(dropout_seed)
            else:
                torch.manual_seed(dropout_seed)
            with torch.autocast(
                device.type, dtype=torch.bfloat16,
                enabled=autocast_enabled and device.type == "cuda",
            ):
                reference_output = reference(value)
            if device.type == "cuda":
                custom.begin_microbatch(dropout_seed, capture_masks=True)
            else:
                torch.manual_seed(dropout_seed)
                custom.reset_call_count()
            with custom.enabled(True), torch.autocast(
                device.type, dtype=torch.bfloat16,
                enabled=autocast_enabled and device.type == "cuda",
            ):
                custom_output = custom_base.target(value)
            custom_dropout = custom.end_microbatch()
            reference_output.sum().backward()
            custom_output.sum().backward()
            output_error = float(
                (reference_output - custom_output).detach().abs().max().float().cpu()
            )
            a_gradient_error = float(
                (
                    reference.lora_A["default"].weight.grad.float()
                    - custom.down["d000"].weight.grad.float()
                ).abs().max().cpu()
            )
            b_gradient_error = float(
                (
                    reference.lora_B["default"].weight.grad.float()
                    - custom.up["d000"].weight.grad.float()
                ).abs().max().cpu()
            )
            shapes_equal = (
                reference_output.shape == custom_output.shape
                and reference.lora_A["default"].weight.grad.shape
                == custom.down["d000"].weight.grad.shape
                and reference.lora_B["default"].weight.grad.shape
                == custom.up["d000"].weight.grad.shape
            )
            dtypes_equal = reference_output.dtype == custom_output.dtype
            custom.close()
        atol, rtol = ((1e-6, 1e-5) if dtype == torch.float32 else (2e-3, 1e-2))
        passes = (
            shapes_equal and dtypes_equal
            and torch.allclose(reference_output, custom_output, atol=atol, rtol=rtol)
            and torch.allclose(
                reference.lora_A["default"].weight.grad.float(),
                custom.down["d000"].weight.grad.float(), atol=atol, rtol=rtol,
            )
            and torch.allclose(
                reference.lora_B["default"].weight.grad.float(),
                custom.up["d000"].weight.grad.float(), atol=atol, rtol=rtol,
            )
        )
        return {
            "passes": bool(passes), "dtype": str(dtype), "dropout": dropout,
            "autocast": autocast_enabled and device.type == "cuda",
            "output_shape_dtype_equal": shapes_equal and dtypes_equal,
            "max_output_abs_error": output_error,
            "max_a_gradient_abs_error": a_gradient_error,
            "max_b_gradient_abs_error": b_gradient_error,
            "atol": atol, "rtol": rtol, "custom_dropout_receipt": custom_dropout,
        }

    exact = probe(torch.float32, 0.0, autocast_enabled=False)
    live_like = probe(torch.bfloat16, 0.05, autocast_enabled=True)
    if not exact["passes"] or not live_like["passes"]:
        raise RuntimeError("actual custom LoRA hook differs from pinned PEFT Linear")
    return {
        "peft_version": __import__("peft").__version__,
        "scale": 2.0,
        "device": device.type,
        "actual_adaptation_bank_hook": True,
        "exact_fp32_dropout_disabled": exact,
        "live_bfloat16_dropout_0_05": live_like,
    }


def _build_new(
    config: Mapping[str, Any],
    *,
    capacity: str,
    model_seed: int,
    initialization_bundle: Path,
    failure_state: dict[str, Any] | None = None,
) -> tuple[Any, StateLoopModel, dict[str, Any]]:
    if capacity not in {"lora", "fullrank"}:
        raise ValueError(capacity)
    preflight_device = _require_cuda()
    installed_environment_lock = _installed_environment_lock_receipt()
    shared_state, init_receipt = load_initialization_bundle(
        config, model_seed, initialization_bundle
    )
    if failure_state is not None:
        failure_state["shared_initialization"] = init_receipt
        _mark_g0_complete(failure_state, "shared_initialization")
    tokenizer, base, token_receipt = _load_base(config)
    arch = config["architecture"]
    targets = _discover_targets(base, int(arch["loop_start"]), int(arch["loop_end"]))
    wrapper = StateLoopModel(
        base,
        config,
        targets,
        capacity=capacity,
        model_seed=model_seed,
        shared_state=shared_state,
    ).cuda()
    if wrapper.hidden_size != int(arch["expected_hidden_size"]):
        raise RuntimeError("loaded model hidden size changed")
    live_manifest, live_digest = tensor_manifest(wrapper.extra_state_dict())
    init_metadata = init_receipt["metadata"]
    if (
        live_manifest != init_metadata["tensor_manifest"]
        or live_digest != init_metadata["tensor_values_sha256"]
    ):
        raise RuntimeError("wrapper did not load the exact shared initialization")
    target_manifest = wrapper.adaptation.target_manifest()
    expected = arch["adaptation"][capacity]
    parameters = sum(int(item["parameters"]) for item in target_manifest)
    if len(target_manifest) != int(expected["expected_targets"]):
        raise RuntimeError("adaptation target count changed")
    if parameters != int(expected["expected_parameters"]):
        raise RuntimeError(f"{capacity} parameter count changed: {parameters}")
    zero_receipt = wrapper.adaptation.zero_function_receipt()
    if zero_receipt != {"nonzero_output_weights": 0, "max_abs_output_weight": 0.0}:
        raise RuntimeError("adaptation does not begin as the exact zero function")
    setup = {
        "capacity": capacity,
        "model_seed": int(model_seed),
        "tokenizer": token_receipt,
        "adaptation_targets": targets,
        "adaptation_targets_sha256": hashlib.sha256("\n".join(targets).encode()).hexdigest(),
        "adaptation_target_manifest": target_manifest,
        "adaptation_target_manifest_sha256": _canonical_sha256({"targets": target_manifest}),
        "adaptation_parameters": parameters,
        "adaptation_zero_function": zero_receipt,
        "shared_initialization": init_receipt,
        "trainable_parameters": _trainable_receipt(wrapper),
        "dropout_control": _nonadapter_dropout_receipt(wrapper),
        "environment": _environment_receipt(include_device=True),
        "installed_environment_lock": installed_environment_lock,
        "preflight_device": preflight_device,
    }
    return tokenizer, wrapper, setup


def _encode_row(
    tokenizer: Any,
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    k: int,
    device: torch.device,
) -> dict[str, Any]:
    rendered = _render(tokenizer, row)
    encoded = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    token_receipt = _validate_tokenizer(tokenizer, config)
    base_ids = input_ids[0].tolist()
    for answer, answer_id in zip(ANSWER_STRINGS, token_receipt["answer_token_ids"]):
        contextual = tokenizer.encode(rendered + answer, add_special_tokens=False)
        if contextual[:-1] != base_ids or contextual[-1] != answer_id:
            raise RuntimeError(f"answer tokenization is not prefix-stable: {answer!r}")
    state_mask = input_ids.eq(int(token_receipt["state_token_id"]))
    expected_slots = int(config["architecture"]["state_slots"])
    if int(state_mask.sum()) != expected_slots:
        raise RuntimeError("tokenized prompt has the wrong state-slot count")
    query_character = rendered.index("Query:")
    query_token = len(tokenizer.encode(rendered[:query_character], add_special_tokens=False))
    if int(torch.where(state_mask)[1].max()) >= query_token:
        raise RuntimeError("state slots are not causally before the query")
    targets = trajectory_targets(dict(row), k)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "state_mask": state_mask,
        "answer_positions": torch.tensor([input_ids.shape[1] - 1], device=device),
        "answer_targets": torch.tensor(
            [token_receipt["answer_token_ids"][int(row["correct_choice"])]], device=device
        ),
        "state_targets": {
            name: torch.tensor([values], device=device, dtype=torch.long)
            for name, values in targets.items()
        },
        "depths": torch.tensor([int(row["depth"])], device=device),
        "answer_token_ids": token_receipt["answer_token_ids"],
        "prompt_tokens": int(input_ids.shape[1]),
    }


def _forward(
    wrapper: StateLoopModel, batch: Mapping[str, Any], *, k: int, compute_answer: bool
) -> Any:
    return wrapper(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        state_mask=batch["state_mask"],
        answer_positions=batch["answer_positions"],
        answer_targets=batch["answer_targets"] if compute_answer else None,
        state_targets=batch["state_targets"],
        depths=batch["depths"],
        k=k,
        compute_answer=compute_answer,
    )


def _objective_loss(output: Any, config: Mapping[str, Any], objective: str) -> torch.Tensor:
    weights = config["training"]["objectives"][objective]
    terms = []
    if float(weights["answer_loss_weight"]) > 0:
        if output.answer_loss is None or output.answer_logits is None:
            raise RuntimeError("joint objective requires an answer graph")
        terms.append(float(weights["answer_loss_weight"]) * output.answer_loss)
    elif output.answer_loss is not None or output.answer_logits is not None:
        raise RuntimeError("state-only objective computed a prohibited answer graph")
    terms.append(float(weights["state_loss_weight"]) * output.state_loss)
    terms.append(float(weights["fixed_point_loss_weight"]) * output.fixed_point_loss)
    return sum(terms[1:], terms[0]) if terms else output.state_loss.new_zeros(())


def _choice_prediction(output: Any, answer_ids: Sequence[int]) -> tuple[int, bool, float]:
    logits = output.answer_logits[0]
    choices = logits[torch.tensor(answer_ids, device=logits.device)]
    probabilities = torch.softmax(logits.float(), dim=-1)
    return (
        int(choices.argmax()),
        int(logits.argmax()) in set(answer_ids),
        float(probabilities[list(answer_ids)].sum()),
    )


def _k1_parity(wrapper: StateLoopModel, batch: Mapping[str, Any]) -> float:
    wrapper.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        loop = _forward(wrapper, batch, k=1, compute_answer=True)
        direct = wrapper.core(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            use_cache=False, logits_to_keep=1,
        ).logits[:, -1, :]
    return float((loop.answer_logits - direct).abs().max().cpu())


def _gradient_receipt(wrapper: StateLoopModel) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {
        "adaptation": [], "initializer": [], "step": [], "sufficiency": [],
        "damping": [], "aggregate_exempt": [],
    }
    base_gradient_tensors = 0
    for name, parameter in wrapper.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None and name.startswith("base_model."):
                base_gradient_tensors += 1
            continue
        if name.startswith("adaptation."):
            group = "adaptation"
        elif "state_initializer" in name:
            group = "initializer"
        elif "step_encoder" in name:
            group = "step"
        elif "sufficiency" in name:
            group = "sufficiency"
        elif name == "damping_logit":
            group = "damping"
        elif name == "aggregate_logit":
            group = "aggregate_exempt"
        elif name.startswith("base_model."):
            base_gradient_tensors += int(parameter.grad is not None)
            continue
        else:
            raise RuntimeError(f"unclassified trainable tensor: {name}")
        gradient = parameter.grad
        finite = gradient is not None and bool(torch.isfinite(gradient).all())
        norm = float(gradient.detach().float().norm().cpu()) if finite else None
        groups[group].append(
            {"name": name, "has_gradient": gradient is not None, "finite": finite, "norm": norm}
        )
    receipt: dict[str, Any] = {}
    for key, values in groups.items():
        receipt[key] = {
            "tensors": len(values),
            "with_gradient": sum(item["has_gradient"] for item in values),
            "finite": sum(item["finite"] for item in values),
            "nonzero": sum(
                item["norm"] is not None and item["norm"] > 0.0 for item in values
            ),
            "items": values,
        }
    receipt["base_gradient_tensors"] = base_gradient_tensors
    required = ("adaptation", "initializer", "step", "sufficiency", "damping")
    receipt["all_required_tensors_finite_nonzero"] = all(
        receipt[key]["tensors"] > 0
        and receipt[key]["nonzero"] == receipt[key]["tensors"]
        and receipt[key]["finite"] == receipt[key]["tensors"]
        for key in required
    )
    return receipt


def _load_data_manifest(
    config: Mapping[str, Any], *, content_splits: set[str] | frozenset[str]
) -> tuple[Path, dict[str, Any], str]:
    expected_data_dir = Path(os.path.abspath(
        ROOT / str(config["paths"]["data_dir"])
    ))
    try:
        data_dir = canonical_attempt_path(
            REPO_ROOT,
            expected_data_dir.relative_to(REPO_ROOT).as_posix(),
        )
    except (AttemptReceiptError, ValueError) as exc:
        raise RuntimeError("prepared data directory is missing or aliased") from exc
    path = data_dir / "manifest.json"
    try:
        with open_stable_regular(REPO_ROOT, path) as handle:
            raw = handle.read()
    except Exception as exc:
        raise RuntimeError(f"prepared data manifest is missing or aliased: {path}") from exc
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        manifest = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("prepared data manifest is not strict UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("prepared data manifest is not an object")
    validate_data_manifest(config, data_dir, manifest, content_splits=content_splits)
    return data_dir, manifest, manifest_sha256


def _authorization_details_for(
    config: Mapping[str, Any], capacity: str, objective: str, path: Path | None
) -> dict[str, Any] | None:
    """Validate the one status-specific canonical authorization for a cell.

    Branch ancestry is deliberately interpreted only by ``gate_receipts``.
    In particular, no arbitrary nested lineage can stand in for a direct named
    authorization edge.
    """

    if capacity == "lora" and objective == "joint":
        if path is not None:
            raise RuntimeError("LoRA joint is the first scientific stage and accepts no authorization")
        return None
    if path is None:
        raise RuntimeError(f"{capacity}/{objective} requires an upstream analysis authorization")
    if capacity == "lora" and objective == "state_only":
        branch = LORA_MISS_BRANCH
        filename = "lora_joint_trigger.json"
    elif capacity == "fullrank" and objective == "joint":
        branch = LORA_MISS_BRANCH
        filename = "lora_joint_trigger.json"
    else:
        if path.name == "stage_b_seal.json":
            branch = STAGE_B_FULLRANK_MISS_BRANCH
            filename = "stage_b_seal.json"
        elif path.name == "fullrank_joint.json":
            branch = POSTCONTRAST_FULLRANK_MISS_BRANCH
            filename = "fullrank_joint.json"
        else:
            raise RuntimeError(
                "fullrank/state_only requires the canonical Stage-B or postcontrast miss"
            )
    canonical = _repo_relative(ROOT / "analysis" / filename)
    expected_identity = _identity(config, phase=str(branch))
    expected_identity.pop("phase")
    return validate_branch_authorization(
        REPO_ROOT,
        path,
        canonical_relative_path=canonical,
        branch=branch,
        expected_identity=expected_identity,
    )


def _authorization_for(
    config: Mapping[str, Any], capacity: str, objective: str, path: Path | None
) -> dict[str, Any] | None:
    details = _authorization_details_for(config, capacity, objective, path)
    return None if details is None else dict(details["receipt"])


def _contrast_authorization_details(
    config: Mapping[str, Any], path: Path | None
) -> dict[str, Any]:
    if path is None:
        raise RuntimeError("sealed contrast evaluation requires the Stage-B seal")
    expected_identity = _identity(config, phase="stage_b_seal_analysis")
    expected_identity.pop("phase")
    return validate_branch_authorization(
        REPO_ROOT,
        path,
        canonical_relative_path=_repo_relative(ROOT / "analysis" / "stage_b_seal.json"),
        branch=STAGE_B_CONTRAST_BRANCH,
        expected_identity=_identity(config, phase="stage_b_seal_analysis"),
    )


def _setup_receipt_path(kind: str, capacity: str, model_seed: int) -> Path:
    if kind not in {"g0", "positive_control"}:
        raise ValueError(f"unknown setup receipt kind: {kind!r}")
    return ROOT / "runs" / "setup" / f"{kind}_{capacity}_seed{model_seed}.json"


def _validate_registered_setup_fields(
    config: Mapping[str, Any], setup: Mapping[str, Any], *, capacity: str, model_seed: int
) -> None:
    adaptation = config["architecture"]["adaptation"][capacity]
    targets = setup.get("adaptation_targets")
    manifest = setup.get("adaptation_target_manifest")
    expected_targets = int(adaptation["expected_targets"])
    if type(targets) is not list or len(targets) != expected_targets:
        raise RuntimeError("setup adaptation-target count changed")
    if any(type(target) is not str or not target for target in targets):
        raise RuntimeError("setup adaptation-target names are invalid")
    if len(set(targets)) != len(targets):
        raise RuntimeError("setup adaptation targets are not unique")
    if setup.get("adaptation_targets_sha256") != hashlib.sha256(
        "\n".join(targets).encode("utf-8")
    ).hexdigest():
        raise RuntimeError("setup adaptation-target digest changed")
    if type(manifest) is not list or len(manifest) != expected_targets:
        raise RuntimeError("setup adaptation-target manifest geometry changed")
    if setup.get("adaptation_target_manifest_sha256") != _canonical_sha256(
        {"targets": manifest}
    ):
        raise RuntimeError("setup adaptation-target manifest digest changed")
    expected_parameters = int(adaptation["expected_parameters"])
    if setup.get("adaptation_parameters") != expected_parameters:
        raise RuntimeError("setup adaptation parameter count changed")
    if setup.get("capacity") != capacity or setup.get("model_seed") != model_seed:
        raise RuntimeError("setup cell identity changed")
    shared = setup.get("shared_initialization")
    if not isinstance(shared, Mapping):
        raise RuntimeError("setup shared initialization is missing")
    if set(shared) != {
        "schema_version",
        "status",
        "phase",
        "bundle_path",
        "bundle_sha256",
        "metadata",
        "receipt_identity_sha256",
    }:
        raise RuntimeError("setup shared-initialization receipt fields changed")
    if (
        shared.get("schema_version") != 1
        or shared.get("status") != "SHARED_INITIALIZATION_PREPARED"
        or shared.get("phase") != "shared_initialization"
    ):
        raise RuntimeError("setup shared-initialization receipt status changed")
    if shared.get("receipt_identity_sha256") != _canonical_sha256(
        {
            key: value
            for key, value in shared.items()
            if key != "receipt_identity_sha256"
        }
    ):
        raise RuntimeError("setup shared-initialization identity changed")
    expected_bundle_relative = (
        Path("large_artifacts")
        / config["experiment_id"]
        / f"initialization_seed{model_seed}.pt"
    ).as_posix()
    if shared.get("bundle_path") != expected_bundle_relative:
        raise RuntimeError("setup initialization bundle path changed")
    bundle_path = canonical_repo_path(REPO_ROOT, expected_bundle_relative)
    if shared.get("bundle_sha256") != _sha256(bundle_path):
        raise RuntimeError("setup initialization bundle bytes changed")
    metadata = shared.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("setup shared-initialization metadata is missing")
    metadata_identity = metadata.get("receipt_identity_sha256")
    if metadata_identity != _canonical_sha256(
        {
            key: value
            for key, value in metadata.items()
            if key != "receipt_identity_sha256"
        }
    ):
        raise RuntimeError("setup shared-initialization metadata identity changed")
    expected_shared = {
        "experiment_id": config["experiment_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(ROOT),
        "requirements_training_lock_sha256": _requirements_sha256(),
        "model_seed": model_seed,
    }
    for field, expected in expected_shared.items():
        if type(metadata.get(field)) is not type(expected) or metadata.get(field) != expected:
            raise RuntimeError(f"setup shared-initialization {field} changed")


def _require_supplied_initialization_bundle(
    setup: Mapping[str, Any], supplied: Path
) -> Path:
    shared = setup.get("shared_initialization")
    if not isinstance(shared, Mapping) or type(shared.get("bundle_path")) is not str:
        raise RuntimeError("setup omits its initialization bundle path")
    expected = canonical_repo_path(REPO_ROOT, str(shared["bundle_path"]))
    if supplied.absolute() != expected:
        raise RuntimeError("supplied initialization bundle is not the setup-bound path")
    if shared.get("bundle_sha256") != _sha256(expected):
        raise RuntimeError("supplied initialization bundle bytes changed")
    return expected


def _validate_setup_cell(
    config: Mapping[str, Any],
    *,
    capacity: str,
    model_seed: int,
    data_manifest_sha256: str,
    root_lora_miss_lineage: Mapping[str, Any] | None,
    g0_path: Path | None = None,
    control_path: Path | None = None,
) -> dict[str, Any]:
    """Reopen one canonical setup pair and prove its scientific pass semantics."""

    adaptation = config["architecture"]["adaptation"][capacity]
    lora_rank = int(config["architecture"]["adaptation"]["lora"]["rank"])
    g0_path = g0_path or _setup_receipt_path("g0", capacity, model_seed)
    control_path = control_path or _setup_receipt_path(
        "positive_control", capacity, model_seed
    )
    expected_branch = root_lora_miss_lineage if capacity == "fullrank" else None
    g0 = _read_g0_pass(g0_path, config, capacity=capacity)
    setup = g0.get("setup")
    if not isinstance(setup, Mapping):
        raise RuntimeError("G0 receipt omits its deterministic setup")
    _validate_registered_setup_fields(
        config, setup, capacity=capacity, model_seed=model_seed
    )
    g0 = validate_g0_pass(
        REPO_ROOT,
        g0_path,
        canonical_relative_path=_repo_relative(
            _setup_receipt_path("g0", capacity, model_seed)
        ),
        expected_identity=_identity(config, phase=f"{capacity}_g0"),
        capacity=capacity,
        model_seed=model_seed,
        data_manifest_sha256=data_manifest_sha256,
        expected_setup=setup,
        expected_branch_authorization=expected_branch,
        k1_max_logit_abs_error=float(config["gates"]["k1_max_logit_abs_error"]),
        train_k=int(config["training"]["train_k"]),
        max_recurrence=int(config["architecture"]["max_recurrence"]),
        expected_adaptation_targets=int(
            adaptation["expected_targets"]
        ),
        expected_adaptation_parameters=int(adaptation["expected_parameters"]),
        expected_adaptation_dropout=float(adaptation["dropout"]),
        expected_adaptation_scale=float(adaptation["scale"]),
        expected_lora_rank=lora_rank,
        expected_peft_version=_locked_requirement_version("peft"),
        adaptation_gradient_clip=float(
            config["training"]["adaptation_gradient_clip"]
        ),
        common_gradient_clip=float(config["training"]["common_gradient_clip"]),
        worst_depth_seed=int(config["training"]["g0_control"]["worst_depth_seed"]),
        min_free_memory_gib=4.0,
    )
    g0_lineage = _lineage(g0_path, g0)
    control_config = config["training"]["positive_control"]
    _, control_manifest, observed_manifest_sha256 = _load_data_manifest(
        config, content_splits=set()
    )
    if observed_manifest_sha256 != data_manifest_sha256:
        raise RuntimeError("positive-control validator data manifest changed")
    expected_control_rows, _ = generate_control_rows(config, control_manifest)
    control = validate_positive_control_pass(
        REPO_ROOT,
        control_path,
        canonical_relative_path=_repo_relative(
            _setup_receipt_path("positive_control", capacity, model_seed)
        ),
        expected_identity=_identity(config, phase=f"{capacity}_positive_control"),
        capacity=capacity,
        model_seed=model_seed,
        data_manifest_sha256=data_manifest_sha256,
        expected_setup=setup,
        expected_branch_authorization=expected_branch,
        expected_g0_lineage=g0_lineage,
        expected_control_rows=expected_control_rows,
        control_seed=int(control_config["seed"]),
        control_rows=int(control_config["rows"]),
        control_updates=int(control_config["updates"]),
        gradient_accumulation=int(config["training"]["gradient_accumulation"]),
        min_oracle_readout_accuracy=float(
            control_config["min_oracle_readout_accuracy"]
        ),
        min_overfit_final_joint_accuracy=float(
            control_config["min_overfit_final_joint_accuracy"]
        ),
        expected_adaptation_targets=int(adaptation["expected_targets"]),
        expected_adaptation_parameters=int(adaptation["expected_parameters"]),
        expected_adaptation_dropout=float(adaptation["dropout"]),
        expected_lora_rank=lora_rank,
        control_families=tuple(map(str, config["substrate"]["train_families"])),
        control_templates=tuple(
            map(str, config["substrate"]["train_templates"])
        ),
        control_depths=tuple(map(int, control_config["depths"])),
        control_query_kinds=("node", "checksum"),
        control_examples_per_cell=int(control_config["examples_per_cell"]),
        learning_rate=float(config["training"]["learning_rate"]),
        adaptation_gradient_clip=float(
            config["training"]["adaptation_gradient_clip"]
        ),
        common_gradient_clip=float(config["training"]["common_gradient_clip"]),
    )
    return {
        "g0": g0,
        "control": control,
        "g0_lineage": g0_lineage,
        "control_lineage": _lineage(control_path, control),
        "setup": dict(setup),
    }


def _setup_barrier(
    config: Mapping[str, Any],
    *,
    stage: str,
    data_manifest_sha256: str,
    root_lora_miss_lineage: Mapping[str, Any] | None,
    target: TrainingCell | None = None,
    target_g0_path: Path | None = None,
    target_control_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Require every preregistered setup pair before a stage can start."""

    if stage not in {"A", "B", "C"}:
        raise ValueError(f"unknown setup-barrier stage: {stage!r}")
    capacities = ("lora",) if stage == "A" else ("lora", "fullrank")
    if stage == "A" and root_lora_miss_lineage is not None:
        raise RuntimeError("Stage A setup cannot have a LoRA-miss authorization")
    if stage in {"B", "C"} and root_lora_miss_lineage is None:
        raise RuntimeError(f"Stage {stage} setup requires the root LoRA-miss authorization")
    cells: dict[str, dict[str, Any]] = {}
    per_seed_shared: dict[int, Mapping[str, Any]] = {}
    common_invariant: Mapping[str, Any] | None = None
    capacity_invariants: dict[str, Mapping[str, Any]] = {}
    for capacity_name in capacities:
        for seed in map(int, config["training"]["train_seeds"]):
            is_target = (
                target is not None
                and target.capacity == capacity_name
                and target.seed == seed
            )
            validated = _validate_setup_cell(
                config,
                capacity=capacity_name,
                model_seed=seed,
                data_manifest_sha256=data_manifest_sha256,
                root_lora_miss_lineage=root_lora_miss_lineage,
                g0_path=target_g0_path if is_target else None,
                control_path=target_control_path if is_target else None,
            )
            slug = f"{capacity_name}_seed{seed}"
            cells[slug] = validated
            setup = validated["setup"]
            stable = strict_stable_setup_receipt(setup)
            common = {
                "tokenizer": stable["tokenizer"],
                "adaptation_targets": stable["adaptation_targets"],
                "adaptation_targets_sha256": stable["adaptation_targets_sha256"],
                "dropout_control": stable["dropout_control"],
                "environment": stable["environment"],
                "installed_environment_lock": stable["installed_environment_lock"],
                "preflight_device": stable["preflight_device"],
            }
            if common_invariant is None:
                common_invariant = common
            elif common != common_invariant:
                raise RuntimeError("setup cells do not share one model/environment invariant")
            trainable = stable["trainable_parameters"]
            if not isinstance(trainable, Mapping):
                raise RuntimeError("setup trainable-parameter receipt is malformed")
            capacity_invariant = {
                "adaptation_target_manifest": stable[
                    "adaptation_target_manifest"
                ],
                "adaptation_target_manifest_sha256": stable[
                    "adaptation_target_manifest_sha256"
                ],
                "adaptation_parameters": stable["adaptation_parameters"],
                "adaptation_zero_function": stable["adaptation_zero_function"],
                "trainable_parameters_without_values": {
                    key: value
                    for key, value in trainable.items()
                    if key != "values_sha256"
                },
            }
            prior_capacity_invariant = capacity_invariants.get(capacity_name)
            if prior_capacity_invariant is None:
                capacity_invariants[capacity_name] = capacity_invariant
            elif capacity_invariant != prior_capacity_invariant:
                raise RuntimeError(
                    f"{capacity_name} setup geometry changes across seeds"
                )
            shared = setup["shared_initialization"]
            if seed in per_seed_shared and per_seed_shared[seed] != shared:
                raise RuntimeError(
                    f"LoRA/full-rank setup does not share exact initialization for seed {seed}"
                )
            per_seed_shared[seed] = shared
    proof: dict[str, Any] = {
        "schema_version": 1,
        "status": "SETUP_BARRIER_COMPLETE",
        "stage": stage,
        "cells": [
            {
                "cell": slug,
                "g0_lineage": validated["g0_lineage"],
                "positive_control_lineage": validated["control_lineage"],
                "stable_setup_sha256": _canonical_sha256(
                    strict_stable_setup_receipt(validated["setup"])
                ),
            }
            for slug, validated in cells.items()
        ],
        "root_lora_miss_lineage": (
            dict(root_lora_miss_lineage) if root_lora_miss_lineage else None
        ),
        "common_setup_invariant_sha256": _canonical_sha256(common_invariant or {}),
        "capacity_setup_invariant_sha256s": {
            capacity_name: _canonical_sha256(invariant)
            for capacity_name, invariant in capacity_invariants.items()
        },
    }
    proof["barrier_identity_sha256"] = _canonical_sha256(proof)
    return proof, cells


def _training_stage(capacity: str, objective: str) -> str:
    mapping = {
        ("lora", "joint"): "A",
        ("lora", "state_only"): "B",
        ("fullrank", "joint"): "B",
        ("fullrank", "state_only"): "C",
    }
    try:
        return mapping[(capacity, objective)]
    except KeyError as exc:
        raise ValueError(f"unregistered training cell: {capacity}/{objective}") from exc


def _training_contracts(
    config: Mapping[str, Any], stage: str
) -> dict[TrainingCell, TrainingReceiptContract]:
    matrices = {"A": STAGE_A_MATRIX, "B": STAGE_B_MATRIX, "C": STAGE_C_MATRIX}
    try:
        cells = matrices[stage]
    except KeyError as exc:
        raise ValueError(f"unknown training stage: {stage!r}") from exc
    steps = int(config["training"]["train_steps"])
    return {
        cell: TrainingReceiptContract(
            schema_version=1,
            status="TRAINING_COMPLETE",
            identity=_identity(config, phase=cell.phase),
            steps=steps,
        )
        for cell in cells
    }


def _stage_required_authorization(
    stage: str, authorization_details: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    if stage == "A":
        return None
    if authorization_details is None:
        raise RuntimeError(f"Stage {stage} requires branch authorization")
    field = "root_lora_miss_lineage" if stage == "B" else "lineage"
    lineage = authorization_details.get(field)
    if not isinstance(lineage, Mapping):
        raise RuntimeError(f"Stage {stage} authorization omits {field}")
    return lineage


def _bind_training_cells_to_setup(
    cell_proofs: Sequence[Mapping[str, Any]], setup_barrier: Mapping[str, Any]
) -> None:
    setup_rows = setup_barrier.get("cells")
    if not isinstance(setup_rows, list):
        raise RuntimeError("setup barrier omits its cells")
    setup_by_cell: dict[str, Mapping[str, Any]] = {}
    for row in setup_rows:
        if not isinstance(row, Mapping) or type(row.get("cell")) is not str:
            raise RuntimeError("setup barrier cell is malformed")
        if row["cell"] in setup_by_cell:
            raise RuntimeError("setup barrier has a duplicate cell")
        setup_by_cell[str(row["cell"])] = row
    for proof in cell_proofs:
        if not isinstance(proof, Mapping):
            raise RuntimeError("training cell proof is malformed")
        capacity = proof.get("capacity")
        seed = proof.get("seed")
        if type(capacity) is not str or type(seed) is not int:
            raise RuntimeError("training cell proof identity is malformed")
        setup_key = f"{capacity}_seed{seed}"
        expected = setup_by_cell.get(setup_key)
        if expected is None:
            raise RuntimeError(f"training cell has no setup-barrier cell: {setup_key}")
        gates = proof.get("gate_lineages")
        if not isinstance(gates, Mapping):
            raise RuntimeError("training cell proof omits gate lineages")
        if gates.get("g0_lineage") != expected.get("g0_lineage"):
            raise RuntimeError("training cell changed its canonical G0 lineage")
        if gates.get("positive_control_lineage") != expected.get(
            "positive_control_lineage"
        ):
            raise RuntimeError("training cell changed its canonical positive-control lineage")
        if proof.get("stable_setup_sha256") != expected.get(
            "stable_setup_sha256"
        ):
            raise RuntimeError("training cell changed its deterministic setup")


def _prior_training_barriers(
    config: Mapping[str, Any],
    *,
    target_stage: str,
    authorization_details: Mapping[str, Any] | None,
    setup_barrier: Mapping[str, Any],
) -> list[dict[str, Any]]:
    reached = ("A", "B", "C")
    target_index = reached.index(target_stage)
    proofs = []
    for stage in reached[:target_index]:
        proof = training_barrier(
            REPO_ROOT,
            stage,
            _training_contracts(config, stage),
            required_authorization=_stage_required_authorization(
                stage, authorization_details
            ),
        )
        _bind_training_cells_to_setup(proof["cells"], setup_barrier)
        proofs.append(proof)
    return proofs


def _reached_training_barrier(
    config: Mapping[str, Any],
    *,
    reached_stage: str,
    authorization_details: Mapping[str, Any] | None,
    setup_barrier: Mapping[str, Any],
) -> dict[str, Any]:
    stages = ("A", "B", "C")
    final_index = stages.index(reached_stage)
    proofs = []
    for stage in stages[: final_index + 1]:
        proof = evaluation_barrier(
            REPO_ROOT,
            stage,
            _training_contracts(config, stage),
            required_authorization=_stage_required_authorization(
                stage, authorization_details
            ),
        )
        _bind_training_cells_to_setup(proof["cells"], setup_barrier)
        proofs.append(proof)
    reached: dict[str, Any] = {
        "schema_version": 1,
        "status": "REACHED_TRAINING_BARRIER_COMPLETE",
        "stages": proofs,
        "reached_stage": reached_stage,
    }
    reached["barrier_identity_sha256"] = _canonical_sha256(reached)
    return reached


def _target_training_cell_proof(
    reached_barrier: Mapping[str, Any], target: TrainingCell
) -> dict[str, Any]:
    matches = []
    for stage in reached_barrier.get("stages", []):
        if not isinstance(stage, Mapping):
            raise RuntimeError("reached training barrier has a malformed stage")
        for proof in stage.get("cells", []):
            if isinstance(proof, Mapping) and proof.get("cell") == target.slug:
                matches.append(dict(proof))
    if len(matches) != 1:
        raise RuntimeError("reached training barrier does not uniquely bind target cell")
    return matches[0]


def _g0_failure_mirror_path(
    capacity: str,
    model_seed: int,
    source_contract: str,
) -> Path:
    return (
        ROOT
        / "runs"
        / "failures"
        / f"g0_{capacity}_seed{model_seed}_source_{source_contract[:12]}.json"
    )


def _positive_control_failure_mirror_path(
    capacity: str,
    model_seed: int,
    source_contract: str,
) -> Path:
    return (
        ROOT
        / "runs"
        / "failures"
        / (
            f"positive_control_{capacity}_seed{model_seed}_source_"
            f"{source_contract[:12]}.json"
        )
    )


def _strict_json_object_snapshot(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    """Read one immutable strict-JSON object and retain its exact bytes."""

    try:
        raw = read_stable_bytes(_publication_root(path), path)
        value = _strict_json_object_bytes(raw, label=label)
    except Exception as exc:
        raise RuntimeError(f"{label} is not one stable strict-JSON receipt") from exc
    return value, raw


def _strict_json_object_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except Exception as exc:
        raise RuntimeError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _validate_recoverable_setup_failure(
    receipt: Mapping[str, Any],
    *,
    kind: str,
    capacity: str,
    model_seed: int,
    common_identity: Mapping[str, Any],
) -> None:
    """Validate a stranded failure before copying it to the missing pair leaf.

    Recovery never turns the receipt into an authorization.  It only restores
    the byte-identical independently-inoded mirror/canonical pair required by
    the invalidated-setup archiver.
    """

    common_keys = set(common_identity)
    if kind == "g0":
        exact_keys = common_keys | {
            "schema_version", "status", "capacity", "model_seed",
            "data_manifest_sha256", "branch_authorization",
            "shared_initialization", "setup", "failure_stage", "error_type",
            "error", "completed_checks", "live_joint_gradient_summary",
            "live_joint_dropout_probe", "authorizes_positive_control",
            "authorizes_training", "authorizes_result_training",
            "authorizes_result_evaluation", "benchmark_files_read",
            "result_payloads_opened", "sealed_contrast_payloads_opened",
            "training_or_evaluation_started", "scientific_evidence",
            "receipt_identity_sha256",
        }
    elif kind == "positive_control":
        exact_keys = common_keys | {
            "schema_version", "status", "capacity", "model_seed",
            "data_manifest_sha256", "g0_lineage", "branch_authorization",
            "shared_initialization", "setup", "control_rows",
            "oracle_analysis", "oracle_readout_accuracy", "failure_stage",
            "error_type", "error", "completed_updates",
            "completed_microbatches", "training_diagnostics",
            "authorizes_training", "authorizes_result_training",
            "authorizes_result_evaluation", "benchmark_files_read",
            "result_payloads_opened", "sealed_contrast_payloads_opened",
            "scientific_evidence", "receipt_identity_sha256",
        }
    else:
        raise RuntimeError(f"unknown setup-failure kind: {kind!r}")
    if set(receipt) != exact_keys:
        raise RuntimeError(f"stranded {kind} failure receipt fields are incomplete")
    expected_scalars = {
        "schema_version": 1,
        "status": "SETUP_CONTROL_FAILED",
        "capacity": capacity,
        "model_seed": int(model_seed),
        **common_identity,
    }
    for field, expected in expected_scalars.items():
        actual = receipt.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError(f"stranded {kind} failure has invalid {field}")
    claimed = receipt.get("receipt_identity_sha256")
    unsigned = {
        key: value for key, value in receipt.items()
        if key != "receipt_identity_sha256"
    }
    if type(claimed) is not str or claimed != _canonical_sha256(unsigned):
        raise RuntimeError(f"stranded {kind} failure identity mismatch")
    for field in ("failure_stage", "error_type", "error"):
        if type(receipt.get(field)) is not str or not receipt[field]:
            raise RuntimeError(f"stranded {kind} failure lacks {field}")
    false_fields = [
        "authorizes_training", "authorizes_result_training",
        "authorizes_result_evaluation", "scientific_evidence",
    ]
    if kind == "g0":
        false_fields.extend(("authorizes_positive_control", "training_or_evaluation_started"))
    for field in false_fields:
        if type(receipt.get(field)) is not bool or receipt[field] is not False:
            raise RuntimeError(f"stranded {kind} failure has unsafe {field}")
    if type(receipt.get("benchmark_files_read")) is not int or receipt[
        "benchmark_files_read"
    ] != 0:
        raise RuntimeError(f"stranded {kind} failure reports benchmark access")
    if receipt.get("sealed_contrast_payloads_opened") != []:
        raise RuntimeError(f"stranded {kind} failure reports sealed contrast access")
    if kind == "positive_control":
        if receipt.get("result_payloads_opened") != []:
            raise RuntimeError("stranded positive-control failure reports result access")
        stages = {
            "receipt_preflight", "branch_authorization", "data_manifest",
            "oracle_analysis", "model_setup", "initial_diagnostics",
            "state_path_overfit",
            "final_optimizer_audit", "fixed_final_overfit_gate",
        }
        if receipt["failure_stage"] not in stages:
            raise RuntimeError("stranded positive-control failure stage is unknown")
        for field in ("completed_updates", "completed_microbatches"):
            if type(receipt.get(field)) is not int or receipt[field] < 0:
                raise RuntimeError(f"stranded positive-control failure has invalid {field}")
        if not isinstance(receipt.get("training_diagnostics"), dict):
            raise RuntimeError("stranded positive-control failure lacks diagnostics")
    else:
        completed = receipt.get("completed_checks")
        stage = receipt["failure_stage"]
        if (
            type(completed) is not list
            or stage not in G0_FAILURE_STAGE_PREFIX_LENGTHS
            or completed != list(G0_COMPLETED_CHECKS[: len(completed)])
            or len(completed) not in G0_FAILURE_STAGE_PREFIX_LENGTHS[stage]
        ):
            raise RuntimeError("stranded G0 failure has impossible progress")
        expected_access = [] if stage == "branch_authorization" else ["train"]
        if receipt.get("result_payloads_opened") != expected_access:
            raise RuntimeError("stranded G0 failure reports unsafe result access")
        manifest = receipt.get("data_manifest_sha256")
        if len(completed) < 2:
            if manifest is not None:
                raise RuntimeError("stranded G0 failure claims an early data manifest")
        elif (
            type(manifest) is not str
            or len(manifest) != 64
            or any(character not in "0123456789abcdef" for character in manifest)
        ):
            raise RuntimeError("stranded G0 failure lacks its data manifest digest")


def _stable_setup_failure_pair(
    canonical: Path,
    mirror: Path,
    *,
    kind: str,
    capacity: str,
    model_seed: int,
    common_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Hold and verify the final pair as identical independent inodes."""

    with open_stable_regular(_publication_root(canonical), canonical) as canonical_file:
        with open_stable_regular(_publication_root(mirror), mirror) as mirror_file:
            canonical_raw = canonical_file.read()
            mirror_raw = mirror_file.read()
            if canonical_raw != mirror_raw:
                raise RuntimeError("setup-failure canonical and mirror bytes differ")
            canonical_info = os.fstat(canonical_file.fileno())
            mirror_info = os.fstat(mirror_file.fileno())
            if (canonical_info.st_dev, canonical_info.st_ino) == (
                mirror_info.st_dev, mirror_info.st_ino
            ):
                raise RuntimeError("setup-failure pair shares one inode")
    receipt = _strict_json_object_bytes(canonical_raw, label=f"{kind} failure")
    _validate_recoverable_setup_failure(
        receipt,
        kind=kind,
        capacity=capacity,
        model_seed=model_seed,
        common_identity=common_identity,
    )
    return receipt


def _recover_setup_failure_pair(
    canonical: Path,
    mirror: Path,
    *,
    kind: str,
    capacity: str,
    model_seed: int,
    common_identity: Mapping[str, Any],
) -> None:
    """Complete a valid one-leaf crash prefix, then require archival."""

    canonical_present = os.path.lexists(canonical)
    mirror_present = os.path.lexists(mirror)
    if not canonical_present and not mirror_present:
        return
    if canonical_present and mirror_present:
        receipt = _stable_setup_failure_pair(
            canonical, mirror, kind=kind, capacity=capacity,
            model_seed=model_seed, common_identity=common_identity,
        )
    else:
        present = canonical if canonical_present else mirror
        missing = mirror if canonical_present else canonical
        receipt, raw = _strict_json_object_snapshot(
            present, label=f"stranded {kind} failure"
        )
        _validate_recoverable_setup_failure(
            receipt,
            kind=kind,
            capacity=capacity,
            model_seed=model_seed,
            common_identity=common_identity,
        )
        missing.parent.mkdir(parents=True, exist_ok=True)
        publish_new_bytes(_publication_root(missing), missing, raw, mode=0o644)
        receipt = _stable_setup_failure_pair(
            canonical, mirror, kind=kind, capacity=capacity,
            model_seed=model_seed, common_identity=common_identity,
        )
    raise RuntimeError(
        f"preserved prior {kind} setup failure requires invalidated-setup archival: "
        f"{receipt['error_type']}: {receipt['error']}"
    )


def _mark_g0_complete(failure_state: dict[str, Any], check: str) -> None:
    """Append exactly the next registered G0 check and reject progress drift."""

    completed = failure_state.get("completed_checks")
    if not isinstance(completed, list):
        raise RuntimeError("G0 failure state lacks a completed-check list")
    expected_prefix = list(G0_COMPLETED_CHECKS[: len(completed)])
    if completed != expected_prefix:
        raise RuntimeError("G0 completed checks are not the registered ordered prefix")
    if len(completed) >= len(G0_COMPLETED_CHECKS):
        raise RuntimeError("G0 completed-check sequence is already complete")
    expected = G0_COMPLETED_CHECKS[len(completed)]
    if check != expected:
        raise RuntimeError(f"G0 check order changed: expected {expected}, found {check}")
    completed.append(check)


def model_smoke(
    config: Mapping[str, Any],
    output: Path,
    *,
    capacity: str,
    model_seed: int,
    initialization_bundle: Path,
    authorization_receipt: Path | None,
) -> None:
    require_confirmatory_config(config)
    validate_design_receipt(config)
    if model_seed not in set(map(int, config["training"]["train_seeds"])):
        raise RuntimeError("model-smoke seed is not preregistered")
    expected_output = (
        ROOT
        / config["paths"]["runs_dir"]
        / "setup"
        / f"g0_{capacity}_seed{model_seed}.json"
    )
    if output.resolve() != expected_output.resolve():
        raise RuntimeError(f"model-smoke output is not canonical: {output}")
    _require_no_symlink_ancestors(output)
    common_identity = _identity(config, phase=f"{capacity}_g0")
    failure_mirror = _g0_failure_mirror_path(
        capacity, model_seed, common_identity["source_contract_sha256"]
    )
    _require_no_symlink_ancestors(failure_mirror)
    _recover_setup_failure_pair(
        output,
        failure_mirror,
        kind="g0",
        capacity=capacity,
        model_seed=model_seed,
        common_identity=common_identity,
    )
    _require_new_output(output, directory=False, kind="model-smoke receipt")
    failure_state: dict[str, Any] = {
        "failure_stage": "branch_authorization",
        "completed_checks": [],
        "data_manifest_sha256": None,
        "branch_authorization": None,
        "setup": None,
        "shared_initialization": None,
        "result_payloads_opened": [],
    }
    try:
        receipt = _model_smoke_attempt(
            config,
            capacity=capacity,
            model_seed=model_seed,
            initialization_bundle=initialization_bundle,
            authorization_receipt=authorization_receipt,
            failure_state=failure_state,
        )
    except Exception as exc:
        failure = _with_identity({
            "schema_version": 1,
            "status": "SETUP_CONTROL_FAILED",
            **common_identity,
            "capacity": capacity,
            "model_seed": int(model_seed),
            "data_manifest_sha256": failure_state["data_manifest_sha256"],
            "branch_authorization": failure_state["branch_authorization"],
            "shared_initialization": failure_state["shared_initialization"],
            "setup": failure_state["setup"],
            "failure_stage": failure_state["failure_stage"],
            "error_type": type(exc).__name__,
            "error": str(exc) or repr(exc) or type(exc).__name__,
            "completed_checks": failure_state["completed_checks"],
            "live_joint_gradient_summary": failure_state.get(
                "live_joint_gradient_summary"
            ),
            "live_joint_dropout_probe": failure_state.get(
                "live_joint_dropout_probe"
            ),
            "authorizes_positive_control": False,
            "authorizes_training": False,
            "authorizes_result_training": False,
            "authorizes_result_evaluation": False,
            "benchmark_files_read": 0,
            "result_payloads_opened": failure_state["result_payloads_opened"],
            "sealed_contrast_payloads_opened": [],
            "training_or_evaluation_started": False,
            "scientific_evidence": False,
        })
        if os.path.lexists(output) or os.path.lexists(failure_mirror):
            raise RuntimeError(
                "model-smoke failure cannot overwrite a competing receipt"
            ) from exc
        _write_new_json_pair(output, failure_mirror, failure)
        raise
    _write_new_json(output, receipt)


def _model_smoke_attempt(
    config: Mapping[str, Any],
    *,
    capacity: str,
    model_seed: int,
    initialization_bundle: Path,
    authorization_receipt: Path | None,
    failure_state: dict[str, Any],
) -> dict[str, Any]:
    if capacity == "fullrank":
        authorization = _authorization_for(
            config, capacity, "joint", authorization_receipt
        )
    elif authorization_receipt is not None:
        raise RuntimeError("LoRA model smoke accepts no branch authorization")
    else:
        authorization = None
    failure_state["branch_authorization"] = (
        _lineage(authorization_receipt, authorization)
        if authorization_receipt and authorization else None
    )
    _mark_g0_complete(failure_state, "branch_authorization")
    failure_state["failure_stage"] = "data_manifest"
    failure_state["result_payloads_opened"] = ["train"]
    data_dir, data_manifest, data_manifest_sha256 = _load_data_manifest(
        config, content_splits=set()
    )
    training_rows = validated_data_rows(
        config, data_dir, data_manifest, ("train",)
    )["train"]
    failure_state["data_manifest_sha256"] = data_manifest_sha256
    _mark_g0_complete(failure_state, "train_only_data_manifest")
    failure_state["failure_stage"] = "model_setup"
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    tokenizer, wrapper, setup = _build_new(
        config, capacity=capacity, model_seed=model_seed,
        initialization_bundle=initialization_bundle,
        failure_state=failure_state,
    )
    failure_state["setup"] = setup
    failure_state["shared_initialization"] = setup["shared_initialization"]
    _mark_g0_complete(failure_state, "pinned_model_and_wrapper_setup")
    failure_state["failure_stage"] = "setup_rows_and_encoding"
    row = next(item for item in training_rows if int(item["depth"]) == 4)
    substrate = config["substrate"]
    architecture = config["architecture"]
    worst = generate_example(
        seed=int(config["training"]["g0_control"]["worst_depth_seed"]),
        split="setup_g0_worst_depth",
        family=str(substrate["train_families"][0]),
        template=str(substrate["train_templates"][0]),
        depth=int(architecture["max_recurrence"]),
        node_count=int(substrate["node_count"]),
        checksum_modulus=int(substrate["checksum_modulus"]),
        num_choices=int(substrate["num_choices"]),
        state_token=str(architecture["state_token"]),
        state_slots=int(architecture["state_slots"]),
        max_attempts=int(substrate["max_generation_attempts"]),
        query_kind="node",
    )
    verify_example(worst, str(architecture["state_token"]), int(architecture["state_slots"]))
    result_fingerprints = {
        fingerprint
        for metadata in data_manifest["files"].values()
        for fingerprint in metadata["structural_fingerprints"]
    }
    if worst["structural_fingerprint"] in result_fingerprints:
        raise RuntimeError("setup-only G0 worst-depth row overlaps result data")
    batch = _encode_row(tokenizer, row, config, k=4, device=torch.device("cuda"))
    k1_batch = _encode_row(tokenizer, row, config, k=1, device=torch.device("cuda"))
    worst_k = int(worst["depth"])
    if worst_k != int(config["architecture"]["max_recurrence"]):
        raise RuntimeError("G0 worst row is not the registered maximum recurrence")
    worst_batch = _encode_row(tokenizer, worst, config, k=worst_k, device=torch.device("cuda"))
    _mark_g0_complete(failure_state, "registered_setup_rows_and_encoding")

    failure_state["failure_stage"] = "pre_optimizer_k1_parity"
    wrapper.eval()
    wrapper.adaptation.reset_call_count()
    parity_before = _k1_parity(wrapper, k1_batch)
    k1_adaptation_calls = wrapper.adaptation.active_call_count
    allowed = float(config["gates"]["k1_max_logit_abs_error"])
    if parity_before > allowed or k1_adaptation_calls != 0:
        raise RuntimeError(f"K=1 parity failed before optimizer: {parity_before}")
    _mark_g0_complete(failure_state, "pre_optimizer_k1_parity")
    failure_state["failure_stage"] = "zero_function_and_k4_call_geometry"
    wrapper.adaptation.reset_call_count()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        enabled = _forward(wrapper, batch, k=4, compute_answer=True)
    calls = wrapper.adaptation.active_call_count
    expected_calls = 3 * int(config["architecture"]["adaptation"][capacity]["expected_targets"])
    if calls != expected_calls:
        raise RuntimeError(f"K=4 adaptation calls changed: {calls} != {expected_calls}")
    with torch.no_grad(), wrapper.adaptation.suspended(), torch.autocast("cuda", dtype=torch.bfloat16):
        disabled = _forward(wrapper, batch, k=4, compute_answer=True)
    zero_function_error = float((enabled.answer_logits - disabled.answer_logits).abs().max().cpu())
    if zero_function_error != 0.0:
        raise RuntimeError("zero-function adaptation changed initial logits")
    _mark_g0_complete(failure_state, "zero_function_and_k4_call_geometry")

    failure_state["failure_stage"] = "two_step_state_only_optimizer_probe"
    adaptation_parameters, common_parameters = _parameter_groups(wrapper)
    optimizer = _build_optimizer(
        adaptation_parameters,
        common_parameters,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    gradient_steps = []
    wrapper.train()
    for probe_step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        dropout_seed = microbatch_dropout_seed(
            model_seed, probe_step, str(row["id"]), 4
        )
        wrapper.adaptation.begin_microbatch(dropout_seed, capture_masks=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probe_output = _forward(wrapper, batch, k=4, compute_answer=False)
            loss = _objective_loss(probe_output, config, "state_only")
        loss.backward()
        dropout_probe = wrapper.adaptation.end_microbatch()
        if (
            dropout_probe["calls"] != expected_calls
            or dropout_probe["cycles"] != 3
            or not dropout_probe["cycle_order_identical"]
            or not dropout_probe["each_cycle_exact_target_set"]
        ):
            raise RuntimeError("K=4 G0 ordered adaptation schedule changed")
        gradients = _gradient_receipt(wrapper)
        if gradients["base_gradient_tensors"] != 0:
            raise RuntimeError("frozen base parameters received gradients")
        if probe_step == 2 and not gradients["all_required_tensors_finite_nonzero"]:
            raise RuntimeError(f"two-step gradient reachability failed: {gradients}")
        adaptation_norm = torch.nn.utils.clip_grad_norm_(
            adaptation_parameters, float(config["training"]["adaptation_gradient_clip"])
        )
        common_norm = torch.nn.utils.clip_grad_norm_(
            common_parameters, float(config["training"]["common_gradient_clip"])
        )
        if not bool(torch.isfinite(adaptation_norm)) or not bool(torch.isfinite(common_norm)):
            raise RuntimeError("G0 clipping norm is nonfinite")
        optimizer.step()
        gradient_steps.append(
            {
                "step": probe_step,
                "loss": float(loss.detach().cpu()),
                "dropout_probe": dropout_probe,
                "gradients": gradients,
                "preclip_adaptation_gradient_norm": float(adaptation_norm.detach().cpu()),
                "preclip_common_gradient_norm": float(common_norm.detach().cpu()),
            }
        )
    _mark_g0_complete(failure_state, "two_step_state_only_optimizer_probe")
    failure_state["failure_stage"] = "live_joint_backward_probe"
    optimizer.zero_grad(set_to_none=True)
    joint_dropout_seed = microbatch_dropout_seed(model_seed, 3, str(row["id"]), 4)
    wrapper.adaptation.begin_microbatch(joint_dropout_seed, capture_masks=True)
    torch.cuda.synchronize()
    joint_started = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        joint_output = _forward(wrapper, batch, k=4, compute_answer=True)
        joint_loss = _objective_loss(joint_output, config, "joint")
    if (
        joint_output.answer_loss is None
        or not bool(torch.isfinite(joint_output.answer_loss))
        or not bool(torch.isfinite(joint_loss))
    ):
        raise RuntimeError("live joint G0 answer/objective loss is nonfinite")
    joint_loss.backward()
    torch.cuda.synchronize()
    joint_elapsed = time.time() - joint_started
    joint_dropout_probe = wrapper.adaptation.end_microbatch()
    failure_state["live_joint_dropout_probe"] = joint_dropout_probe
    if (
        joint_dropout_probe["calls"] != expected_calls
        or joint_dropout_probe["cycles"] != 3
        or not joint_dropout_probe["cycle_order_identical"]
        or not joint_dropout_probe["each_cycle_exact_target_set"]
    ):
        raise RuntimeError("live joint G0 adaptation schedule changed")
    joint_gradients = _gradient_receipt(wrapper)
    failure_state["live_joint_gradient_summary"] = joint_gradients
    joint_required_groups = (
        "adaptation", "initializer", "step", "sufficiency", "damping",
        "aggregate_exempt",
    )
    joint_gradients_complete = all(
        joint_gradients[group]["tensors"] > 0
        and joint_gradients[group]["finite"] == joint_gradients[group]["tensors"]
        and joint_gradients[group]["nonzero"] == joint_gradients[group]["tensors"]
        for group in joint_required_groups
    )
    if joint_gradients["base_gradient_tensors"] != 0 or not joint_gradients_complete:
        raise RuntimeError(f"live joint G0 gradient reachability failed: {joint_gradients}")
    failure_state["failure_stage"] = "live_joint_optimizer_step"
    joint_adaptation_norm = torch.nn.utils.clip_grad_norm_(
        adaptation_parameters, float(config["training"]["adaptation_gradient_clip"])
    )
    joint_common_norm = torch.nn.utils.clip_grad_norm_(
        common_parameters, float(config["training"]["common_gradient_clip"])
    )
    if (
        not bool(torch.isfinite(joint_adaptation_norm))
        or not bool(torch.isfinite(joint_common_norm))
    ):
        raise RuntimeError("live joint G0 clipping norm is nonfinite")
    optimizer.step()
    joint_probe = {
        "objective": "joint",
        "loss": float(joint_loss.detach().cpu()),
        "answer_loss": float(joint_output.answer_loss.detach().cpu()),
        "elapsed_seconds": joint_elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "dropout_probe": joint_dropout_probe,
        "gradients": joint_gradients,
        "all_joint_trainable_groups_finite_nonzero": joint_gradients_complete,
        "preclip_adaptation_gradient_norm": float(joint_adaptation_norm.detach().cpu()),
        "preclip_common_gradient_norm": float(joint_common_norm.detach().cpu()),
        "adaptation_applied_clip_scale": _clip_scale(
            float(joint_adaptation_norm.detach().cpu()),
            float(config["training"]["adaptation_gradient_clip"]),
        ),
        "common_state_applied_clip_scale": _clip_scale(
            float(joint_common_norm.detach().cpu()),
            float(config["training"]["common_gradient_clip"]),
        ),
    }
    _mark_g0_complete(failure_state, "live_joint_backward_and_optimizer_probe")
    failure_state["failure_stage"] = "optimizer_state_receipt"
    optimizer_receipt = optimizer_state_receipt(
        optimizer, delta_parameters=adaptation_parameters
    )
    _mark_g0_complete(failure_state, "optimizer_state_receipt")

    # A fixed ten-step state-only timing probe catches practical OOM/nonfinite
    # behavior without producing a scientific checkpoint.
    failure_state["failure_stage"] = "timed_ten_step_probe"
    torch.cuda.synchronize()
    timed_started = time.time()
    timed_losses = []
    for timed_index in range(1, 11):
        optimizer.zero_grad(set_to_none=True)
        microbatch_index = timed_index + 3
        wrapper.adaptation.begin_microbatch(
            microbatch_dropout_seed(model_seed, microbatch_index, str(row["id"]), 4)
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            timed_output = _forward(wrapper, batch, k=4, compute_answer=False)
            timed_loss = _objective_loss(timed_output, config, "state_only")
        if not bool(torch.isfinite(timed_loss)):
            raise RuntimeError("timed G0 loss is nonfinite")
        timed_loss.backward()
        timed_dropout = wrapper.adaptation.end_microbatch()
        if timed_dropout["calls"] != expected_calls:
            raise RuntimeError("timed G0 adaptation call schedule changed")
        timed_adaptation_norm = torch.nn.utils.clip_grad_norm_(
            adaptation_parameters, float(config["training"]["adaptation_gradient_clip"])
        )
        timed_common_norm = torch.nn.utils.clip_grad_norm_(
            common_parameters, float(config["training"]["common_gradient_clip"])
        )
        if (
            not bool(torch.isfinite(timed_adaptation_norm))
            or not bool(torch.isfinite(timed_common_norm))
        ):
            raise RuntimeError("timed G0 clipping norm is nonfinite")
        optimizer.step()
        timed_losses.append(float(timed_loss.detach().cpu()))
    torch.cuda.synchronize()
    timed_seconds = time.time() - timed_started
    _mark_g0_complete(failure_state, "timed_ten_step_probe")

    failure_state["failure_stage"] = "post_optimizer_k1_parity"
    wrapper.adaptation.reset_call_count()
    parity_after = _k1_parity(wrapper, k1_batch)
    k1_adaptation_calls_after_optimizer = wrapper.adaptation.active_call_count
    if parity_after > allowed or k1_adaptation_calls_after_optimizer != 0:
        raise RuntimeError(f"K=1 parity failed after optimizer: {parity_after}")
    _mark_g0_complete(failure_state, "post_optimizer_k1_parity")

    failure_state["failure_stage"] = "worst_depth_probe"
    wrapper.train()
    worst_dropout_seed = microbatch_dropout_seed(
        model_seed, 99, str(worst["id"]), worst_k
    )
    wrapper.adaptation.begin_microbatch(worst_dropout_seed, capture_masks=True)
    torch.cuda.synchronize()
    worst_started = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        worst_output = _forward(wrapper, worst_batch, k=worst_k, compute_answer=True)
    torch.cuda.synchronize()
    worst_call_receipt = wrapper.adaptation.end_microbatch()
    expected_worst_calls = (worst_k - 1) * int(
        config["architecture"]["adaptation"][capacity]["expected_targets"]
    )
    if worst_call_receipt["calls"] != expected_worst_calls:
        raise RuntimeError(
            f"K={worst_k} adaptation calls changed: "
            f"{worst_call_receipt['calls']} != {expected_worst_calls}"
        )
    if (
        worst_call_receipt["cycles"] != worst_k - 1
        or not worst_call_receipt["cycle_order_identical"]
        or not worst_call_receipt["each_cycle_exact_target_set"]
    ):
        raise RuntimeError("worst-depth ordered adaptation schedule changed")
    worst_tensors = [
        worst_output.answer_logits,
        worst_output.node_logits,
        worst_output.phase_logits,
        worst_output.checksum_logits,
        *worst_output.states,
    ]
    if any(tensor is None or not bool(torch.isfinite(tensor).all()) for tensor in worst_tensors):
        raise RuntimeError("worst-depth output is nonfinite")
    worst_seconds = time.time() - worst_started
    _mark_g0_complete(failure_state, "worst_depth_probe")

    # Rebuild the serialized common state after unrelated RNG consumption and
    # require exact tensor equality with the per-seed bundle.
    failure_state["failure_stage"] = "common_initialization_rng_isolation"
    _ = torch.rand(4096)
    rebuilt_manifest, rebuilt_digest = tensor_manifest(build_shared_state(config, model_seed))
    init_metadata = setup["shared_initialization"]["metadata"]
    rng_isolation = {
        "tensor_manifest_equal": rebuilt_manifest == init_metadata["tensor_manifest"],
        "tensor_values_sha256": rebuilt_digest,
        "expected_tensor_values_sha256": init_metadata["tensor_values_sha256"],
    }
    if not rng_isolation["tensor_manifest_equal"] or rebuilt_digest != init_metadata["tensor_values_sha256"]:
        raise RuntimeError("common initialization changed after unrelated RNG consumption")
    _mark_g0_complete(failure_state, "common_initialization_rng_isolation")

    failure_state["failure_stage"] = "checkpoint_roundtrip"
    wrapper.eval()
    with tempfile.TemporaryDirectory(prefix=f"state-capacity-{capacity}-") as temporary:
        adaptation_path = Path(temporary) / "adaptation.pt"
        common_path = Path(temporary) / "common.pt"
        torch.save(wrapper.delta_state_dict(), adaptation_path)
        torch.save(wrapper.extra_state_dict(), common_path)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            reference = _forward(
                wrapper, batch, k=4, compute_answer=True
            ).answer_logits.detach().clone()
        before_adaptation_digest = tensor_manifest(wrapper.delta_state_dict())[1]
        before_common_digest = tensor_manifest(wrapper.extra_state_dict())[1]
        with torch.no_grad():
            for parameter in wrapper.adaptation.parameters():
                parameter.zero_()
            for parameter in common_parameters:
                parameter.zero_()
        corrupted_adaptation_digest = tensor_manifest(wrapper.delta_state_dict())[1]
        corrupted_common_digest = tensor_manifest(wrapper.extra_state_dict())[1]
        if (
            corrupted_adaptation_digest == before_adaptation_digest
            or corrupted_common_digest == before_common_digest
        ):
            raise RuntimeError("checkpoint roundtrip probe did not destructively change tensors")
        adaptation_sha256 = _sha256(adaptation_path)
        common_sha256 = _sha256(common_path)
        with open_stable_regular(
            adaptation_path.parent,
            adaptation_path,
            expected_sha256=adaptation_sha256,
        ) as handle:
            restored_adaptation = torch.load(handle, map_location="cpu", weights_only=True)
        with open_stable_regular(
            common_path.parent,
            common_path,
            expected_sha256=common_sha256,
        ) as handle:
            restored_common = torch.load(handle, map_location="cpu", weights_only=True)
        wrapper.load_delta_state_dict(restored_adaptation)
        wrapper.load_extra_state_dict(restored_common)
        restored_adaptation_digest = tensor_manifest(wrapper.delta_state_dict())[1]
        restored_common_digest = tensor_manifest(wrapper.extra_state_dict())[1]
        if (
            restored_adaptation_digest != before_adaptation_digest
            or restored_common_digest != before_common_digest
        ):
            raise RuntimeError("checkpoint payload did not restore exact tensor digests")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            restored = _forward(wrapper, batch, k=4, compute_answer=True).answer_logits
        roundtrip_error = float((reference - restored).abs().max().cpu())
        roundtrip = {
            "adaptation_bytes": adaptation_path.stat().st_size,
            "common_bytes": common_path.stat().st_size,
            "adaptation_sha256": adaptation_sha256,
            "common_sha256": common_sha256,
            "destructive_adaptation_digest_changed": True,
            "destructive_common_digest_changed": True,
            "restored_adaptation_digest_equal": True,
            "restored_common_digest_equal": True,
            "max_logit_abs_error": roundtrip_error,
        }
    if roundtrip_error != 0.0:
        raise RuntimeError("checkpoint payload roundtrip changed recurrent logits")
    _mark_g0_complete(failure_state, "checkpoint_roundtrip")
    failure_state["failure_stage"] = "memory_headroom"
    free_memory_after_g0 = torch.cuda.mem_get_info(0)[0] / (1024**3)
    if free_memory_after_g0 < 4.0:
        raise RuntimeError(
            f"G0 leaves insufficient registered VRAM headroom: {free_memory_after_g0:.2f} GiB"
        )
    _mark_g0_complete(failure_state, "memory_headroom")
    failure_state["failure_stage"] = "receipt_construction"

    receipt = _with_identity({
        "schema_version": 1,
        "status": "MODEL_SMOKE_PASS",
        **_identity(config, phase=f"{capacity}_g0"),
        "capacity": capacity,
        "model_seed": int(model_seed),
        "data_manifest_sha256": data_manifest_sha256,
        "setup": setup,
        "peft_formula_reference": _peft_formula_parity_receipt()
        if capacity == "lora" else None,
        "branch_authorization": _lineage(authorization_receipt, authorization)
        if authorization_receipt and authorization else None,
        "k1_max_logit_abs_error_before_optimizer": parity_before,
        "k1_adaptation_calls": k1_adaptation_calls,
        "k1_max_logit_abs_error_after_optimizer": parity_after,
        "k1_adaptation_calls_after_optimizer": k1_adaptation_calls_after_optimizer,
        "zero_function_enabled_minus_disabled_error": zero_function_error,
        "two_step_gradient_probe": gradient_steps,
        "live_joint_backward_probe": joint_probe,
        "optimizer_state": optimizer_receipt,
        "timed_ten_step_probe": {
            "steps": 10,
            "elapsed_seconds": timed_seconds,
            "seconds_per_step": timed_seconds / 10,
            "losses": timed_losses,
        },
        "worst_depth": worst_k,
        "worst_setup_row": {
            "id": worst["id"],
            "seed": int(config["training"]["g0_control"]["worst_depth_seed"]),
            "structural_fingerprint": worst["structural_fingerprint"],
            "cross_result_structural_overlap": 0,
        },
        "worst_call_receipt": worst_call_receipt,
        "worst_forward_seconds": worst_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
        "checkpoint_roundtrip": roundtrip,
        "common_initialization_rng_isolation": rng_isolation,
        "free_memory_gib_after_g0": free_memory_after_g0,
        "elapsed_seconds": time.time() - started,
        "authorizes_positive_control": True,
        "authorizes_training": False,
        "authorizes_result_training": False,
        "authorizes_result_evaluation": False,
        "benchmark_files_read": 0,
        "result_payloads_opened": ["train"],
        "sealed_contrast_payloads_opened": [],
        "training_or_evaluation_started": False,
        "scientific_evidence": False,
    })
    return receipt


def _oracle_readout_control(wrapper: StateLoopModel, config: Mapping[str, Any]) -> float:
    wrapper.sufficiency.train()
    hidden = int(config["architecture"]["expected_hidden_size"])
    slots = int(config["architecture"]["state_slots"])
    rows = []
    targets = []
    for node in range(16):
        for phase in range(2):
            for checksum in range(8):
                vector = torch.zeros(hidden, dtype=torch.float32)
                vector[node] = 1.0
                vector[16 + phase] = 1.0
                vector[18 + checksum] = 1.0
                rows.append(vector.repeat(slots, 1))
                targets.append((node, phase, checksum))
    states = torch.stack(rows).cuda().to(torch.float32).unsqueeze(1)
    node_targets = torch.tensor([item[0] for item in targets], device="cuda")
    phase_targets = torch.tensor([item[1] for item in targets], device="cuda")
    checksum_targets = torch.tensor([item[2] for item in targets], device="cuda")
    optimizer = torch.optim.AdamW(wrapper.sufficiency.parameters(), lr=0.02, weight_decay=0.0)
    for _ in range(256):
        optimizer.zero_grad(set_to_none=True)
        node, phase, checksum = wrapper.sufficiency(states)
        loss = (
            F.cross_entropy(node[:, 0], node_targets)
            + F.cross_entropy(phase[:, 0], phase_targets)
            + F.cross_entropy(checksum[:, 0], checksum_targets)
        ) / 3
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        node, phase, checksum = wrapper.sufficiency(states)
        correct = (
            node[:, 0].argmax(-1).eq(node_targets)
            & phase[:, 0].argmax(-1).eq(phase_targets)
            & checksum[:, 0].argmax(-1).eq(checksum_targets)
        )
    return float(correct.float().mean().cpu())


def _state_accuracy_counts(
    node_logits: torch.Tensor,
    phase_logits: torch.Tensor,
    checksum_logits: torch.Tensor,
    state_targets: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Return strict, serializable trajectory and terminal state counts."""

    logits = {
        "node": node_logits,
        "phase": phase_logits,
        "checksum": checksum_logits,
    }
    expected_classes = {"node": 16, "phase": 2, "checksum": 8}
    if set(state_targets) != set(logits):
        raise RuntimeError("state-accuracy targets have the wrong fields")
    reference_shape: tuple[int, int] | None = None
    predictions: dict[str, torch.Tensor] = {}
    correct: dict[str, torch.Tensor] = {}
    histograms: dict[str, dict[str, list[int]]] = {}
    for name, values in logits.items():
        target = state_targets[name]
        if values.ndim != 3 or target.ndim != 2:
            raise RuntimeError(f"{name} state logits/targets have the wrong rank")
        if (
            tuple(values.shape[:2]) != tuple(target.shape)
            or values.shape[-1] != expected_classes[name]
        ):
            raise RuntimeError(f"{name} state logits/targets have incompatible shapes")
        if not values.is_floating_point() or target.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        }:
            raise RuntimeError(f"{name} state logits/targets have invalid dtypes")
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError(f"{name} state logits are nonfinite")
        if target.numel() == 0:
            raise RuntimeError("state-accuracy input is empty")
        if bool((target < 0).any()) or bool((target >= values.shape[-1]).any()):
            raise RuntimeError(f"{name} state target is outside the registered classes")
        shape = (int(target.shape[0]), int(target.shape[1]))
        if reference_shape is None:
            reference_shape = shape
        elif shape != reference_shape:
            raise RuntimeError("state heads do not share exact batch/step geometry")
        prediction = values.argmax(dim=-1)
        predictions[name] = prediction
        correct[name] = prediction.eq(target)
        histograms[name] = {
            "prediction": torch.bincount(
                prediction.flatten(), minlength=values.shape[-1]
            ).cpu().tolist(),
            "target": torch.bincount(
                target.flatten(), minlength=values.shape[-1]
            ).cpu().tolist(),
        }
    assert reference_shape is not None
    joint = correct["node"] & correct["phase"] & correct["checksum"]
    terminal = {
        name: int(values[:, -1].sum().item()) for name, values in correct.items()
    }
    terminal["joint"] = int(joint[:, -1].sum().item())
    terminal["rows"] = reference_shape[0]
    trajectory = {name: int(values.sum().item()) for name, values in correct.items()}
    trajectory["joint"] = int(joint.sum().item())
    trajectory["steps"] = reference_shape[0] * reference_shape[1]
    return {
        "batch_size": reference_shape[0],
        "steps": reference_shape[1],
        "terminal": terminal,
        "trajectory": trajectory,
        "histograms": histograms,
        "predictions": {
            name: values.detach().cpu().tolist() for name, values in predictions.items()
        },
        "targets": {
            name: state_targets[name].detach().cpu().tolist() for name in logits
        },
    }


def _positive_control_probe_steps(updates: int) -> tuple[int, ...]:
    if updates <= 0:
        raise RuntimeError("positive-control updates must be positive")
    return tuple(
        sorted(step for step in {0, 1, 16, 64, 128, updates} if step <= updates)
    )


def _positive_control_schedule(
    rows: Sequence[Mapping[str, Any]], updates: int, accumulation: int, model_seed: int
) -> list[dict[str, Any]]:
    """Materialize the frozen singleton-microbatch order for CPU regression tests."""

    if not rows or updates <= 0 or accumulation <= 0:
        raise RuntimeError("positive-control schedule geometry must be positive")
    events = []
    for microbatch_index in range(1, updates * accumulation + 1):
        row = rows[(microbatch_index - 1) % len(rows)]
        event = {
            "optimizer_step": (microbatch_index - 1) // accumulation + 1,
            "microbatch_in_step": (microbatch_index - 1) % accumulation + 1,
            "microbatch_index": microbatch_index,
            "row_index": (microbatch_index - 1) % len(rows),
            "id": str(row["id"]),
            "k": int(row["depth"]),
        }
        event["dropout_seed"] = microbatch_dropout_seed(
            model_seed, microbatch_index, event["id"], event["k"]
        )
        events.append(event)
    return events


def _parameter_norm_receipt(wrapper: StateLoopModel) -> dict[str, Any]:
    groups: dict[str, list[torch.Tensor]] = {
        "adaptation_input": [],
        "adaptation_output": [],
        "common_state": [],
    }
    for name, parameter in wrapper.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("adaptation.down."):
            group = "adaptation_input"
        elif name.startswith(("adaptation.up.", "adaptation.deltas.")):
            group = "adaptation_output"
        elif name.startswith("adaptation."):
            raise RuntimeError(f"unclassified adaptation parameter: {name}")
        else:
            group = "common_state"
        groups[group].append(parameter)
    receipt: dict[str, Any] = {}
    for name, parameters in groups.items():
        squared = 0.0
        nonzero_tensors = 0
        parameter_count = 0
        for parameter in parameters:
            norm = float(parameter.detach().float().norm().cpu())
            if not math.isfinite(norm):
                raise RuntimeError(f"nonfinite {name} parameter norm")
            squared += norm * norm
            nonzero_tensors += int(norm > 0.0)
            parameter_count += parameter.numel()
        receipt[name] = {
            "tensors": len(parameters),
            "parameters": parameter_count,
            "nonzero_tensors": nonzero_tensors,
            "l2_norm": math.sqrt(squared),
        }
    if not groups["adaptation_output"] or not groups["common_state"]:
        raise RuntimeError("positive-control parameter roles are incomplete")
    return receipt


def _rng_state_sha256(cuda_devices: Sequence[int] = ()) -> str:
    digest = hashlib.sha256()
    digest.update(torch.get_rng_state().contiguous().numpy().tobytes())
    for device in cuda_devices:
        state = torch.cuda.get_rng_state(int(device))
        digest.update(state.contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


@contextlib.contextmanager
def _positive_control_diagnostic_context(
    wrapper: StateLoopModel, *, adaptation_enabled: bool
):
    cuda_devices = sorted(
        {
            int(parameter.device.index)
            for parameter in wrapper.parameters()
            if parameter.device.type == "cuda" and parameter.device.index is not None
        }
    )
    rng_before = _rng_state_sha256(cuda_devices)
    was_training = wrapper.training
    wrapper.eval()
    adaptation_context = (
        contextlib.nullcontext()
        if adaptation_enabled else wrapper.adaptation.suspended()
    )
    try:
        with (
            torch.random.fork_rng(devices=cuda_devices),
            adaptation_context,
            torch.no_grad(),
        ):
            yield
    finally:
        wrapper.train(was_training)
        if _rng_state_sha256(cuda_devices) != rng_before:
            raise RuntimeError("positive-control diagnostic changed global RNG state")


def _parameter_delta_baseline(wrapper: StateLoopModel) -> dict[str, torch.Tensor]:
    baseline = {}
    for name, parameter in wrapper.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("adaptation.up.", "adaptation.deltas.")):
            if bool(torch.count_nonzero(parameter.detach()).cpu()):
                raise RuntimeError("positive-control output adaptation did not start at zero")
            continue
        baseline[name] = parameter.detach().cpu().clone()
    return baseline


def _parameter_delta_norm_receipt(
    wrapper: StateLoopModel, baseline: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    groups = {
        "adaptation_input": {"tensors": 0, "parameters": 0, "squared": 0.0},
        "adaptation_output": {"tensors": 0, "parameters": 0, "squared": 0.0},
        "common_state": {"tensors": 0, "parameters": 0, "squared": 0.0},
    }
    seen = set()
    for name, parameter in wrapper.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("adaptation.down."):
            group = "adaptation_input"
        elif name.startswith(("adaptation.up.", "adaptation.deltas.")):
            group = "adaptation_output"
        elif name.startswith("adaptation."):
            raise RuntimeError(f"unclassified adaptation parameter: {name}")
        else:
            group = "common_state"
        if name in baseline:
            reference = baseline[name]
            if reference.shape != parameter.shape or reference.dtype != parameter.dtype:
                raise RuntimeError("positive-control delta baseline changed")
            norm = float((parameter.detach().cpu() - reference).float().norm())
            seen.add(name)
        elif group == "adaptation_output":
            norm = float(parameter.detach().float().norm().cpu())
        else:
            raise RuntimeError(f"positive-control delta baseline is missing {name}")
        if not math.isfinite(norm):
            raise RuntimeError("positive-control parameter delta norm is nonfinite")
        groups[group]["tensors"] += 1
        groups[group]["parameters"] += parameter.numel()
        groups[group]["squared"] += norm * norm
    if seen != set(baseline):
        raise RuntimeError("positive-control delta baseline has stale tensors")
    return {
        name: {
            "tensors": int(values["tensors"]),
            "parameters": int(values["parameters"]),
            "l2_delta_norm": math.sqrt(float(values["squared"])),
        }
        for name, values in groups.items()
    }


def _evaluate_positive_control(
    wrapper: StateLoopModel,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    capacity: str,
    model_seed: int,
    probe_step: int,
    adaptation_enabled: bool,
) -> dict[str, Any]:
    records = []
    mode = "intact" if adaptation_enabled else "disabled"
    with _positive_control_diagnostic_context(
        wrapper, adaptation_enabled=adaptation_enabled
    ):
        for row_index, row in enumerate(rows, start=1):
            k = int(row["depth"])
            batch = _encode_row(
                tokenizer, row, config, k=k, device=torch.device("cuda")
            )
            diagnostic_index = probe_step * len(rows) + row_index
            wrapper.adaptation.begin_microbatch(
                microbatch_dropout_seed(
                    model_seed,
                    diagnostic_index,
                    f"positive-probe|{mode}|{row['id']}",
                    k,
                )
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model_output = _forward(
                    wrapper, batch, k=k, compute_answer=False
                )
                objective_loss = _objective_loss(model_output, config, "state_only")
            call_receipt = wrapper.adaptation.end_microbatch()
            expected_calls = (
                (k - 1)
                * int(config["architecture"]["adaptation"][capacity]["expected_targets"])
                if adaptation_enabled else 0
            )
            if call_receipt["calls"] != expected_calls:
                raise RuntimeError("positive-control diagnostic adaptation calls changed")
            if expected_calls and (
                call_receipt["cycles"] != k - 1
                or not call_receipt["cycle_order_identical"]
                or not call_receipt["each_cycle_exact_target_set"]
            ):
                raise RuntimeError("positive-control diagnostic adaptation order changed")
            losses = {
                "objective_loss": float(objective_loss.detach().float().cpu()),
                "state_loss": float(model_output.state_loss.detach().float().cpu()),
                "fixed_point_loss": float(
                    model_output.fixed_point_loss.detach().float().cpu()
                ),
            }
            if not all(math.isfinite(value) for value in losses.values()):
                raise RuntimeError("positive-control diagnostic loss is nonfinite")
            state = _state_accuracy_counts(
                model_output.node_logits,
                model_output.phase_logits,
                model_output.checksum_logits,
                batch["state_targets"],
            )
            if state["batch_size"] != 1 or state["steps"] != k:
                raise RuntimeError("positive-control diagnostic state geometry changed")
            records.append(
                {
                    "id": str(row["id"]),
                    "depth": k,
                    "family": str(row["family"]),
                    "template": str(row["template"]),
                    "query_kind": str(row["query_kind"]),
                    "state": state,
                    **losses,
                }
            )
    analysis = analyze_positive_control_records(records, expected_rows=rows)
    return {
        "step": int(probe_step),
        "adaptation_mode": mode,
        "rng_state_restored": True,
        **analysis,
    }


def _positive_control_rows(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate the exact fresh 48-row factorial setup-only control grid."""

    return generate_control_rows(config, manifest)


def _require_setup_binding(
    receipt: Mapping[str, Any], *, capacity: str, model_seed: int,
    data_manifest_sha256: str, live_setup: Mapping[str, Any], label: str,
    expected_setup_authorization: Mapping[str, Any] | None,
) -> None:
    expected = {
        "capacity": capacity,
        "model_seed": int(model_seed),
        "data_manifest_sha256": data_manifest_sha256,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"{label} {key} does not match this result cell")
    bound_setup = receipt.get("setup")
    if not isinstance(bound_setup, Mapping):
        raise RuntimeError(f"{label} does not bind its exact deterministic setup")
    if _stable_setup_receipt(bound_setup) != _stable_setup_receipt(live_setup):
        raise RuntimeError(f"{label} deterministic setup differs from this result cell")
    if receipt.get("branch_authorization") != (
        dict(expected_setup_authorization) if expected_setup_authorization else None
    ):
        raise RuntimeError(f"{label} setup branch authorization mismatch")


def _stable_setup_receipt(setup: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "capacity", "model_seed", "tokenizer", "adaptation_targets",
        "adaptation_targets_sha256", "adaptation_target_manifest",
        "adaptation_target_manifest_sha256", "adaptation_parameters",
        "adaptation_zero_function", "shared_initialization", "trainable_parameters",
        "dropout_control", "environment", "installed_environment_lock",
        "preflight_device",
    }
    if set(setup) != required:
        raise RuntimeError("setup receipt fields changed")

    def stable_device(device: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in device.items()
            if key != "free_memory_gib_before_load"
        }

    environment = dict(setup["environment"])
    environment["device"] = stable_device(environment["device"])
    return {
        **{
            key: setup[key]
            for key in required - {"environment", "preflight_device"}
        },
        "environment": environment,
        "preflight_device": stable_device(setup["preflight_device"]),
    }


def positive_control(
    config: Mapping[str, Any],
    output: Path,
    *,
    capacity: str,
    model_seed: int,
    initialization_bundle: Path,
    model_smoke_receipt: Path,
    authorization_receipt: Path | None,
) -> None:
    require_confirmatory_config(config)
    adaptation = config["architecture"]["adaptation"][capacity]
    lora_rank = int(config["architecture"]["adaptation"]["lora"]["rank"])
    if model_seed not in set(map(int, config["training"]["train_seeds"])):
        raise RuntimeError("positive-control seed is not preregistered")
    expected_output = _setup_receipt_path("positive_control", capacity, model_seed)
    if output.resolve() != expected_output.resolve():
        raise RuntimeError(f"positive-control output is not canonical: {output}")
    _require_no_symlink_ancestors(output)
    common_identity = _identity(config, phase=f"{capacity}_positive_control")
    failure_mirror = _positive_control_failure_mirror_path(
        capacity, model_seed, common_identity["source_contract_sha256"]
    )
    _require_no_symlink_ancestors(failure_mirror)
    _recover_setup_failure_pair(
        output,
        failure_mirror,
        kind="positive_control",
        capacity=capacity,
        model_seed=model_seed,
        common_identity=common_identity,
    )
    _require_new_output(output, directory=False, kind="positive-control receipt")
    g0: dict[str, Any] | None = None
    authorization: dict[str, Any] | None = None
    authorization_details: dict[str, Any] | None = None
    data_manifest_sha256: str | None = None
    setup: dict[str, Any] | None = None
    control_rows_receipt: dict[str, Any] | None = None
    oracle_analysis: dict[str, Any] | None = None
    oracle_accuracy: float | None = None
    rows: list[dict[str, Any]] = []
    completed_updates = 0
    completed_microbatches = 0
    failure_stage = "receipt_preflight"
    diagnostics: dict[str, Any] = {
        "fixed_probe_steps": [],
        "evaluations": [],
        "parameter_probes": [],
        "optimizer_step_probes": [],
        "dropout_probes": [],
        "completed_updates": 0,
        "completed_microbatches": 0,
    }
    wrapper: StateLoopModel | None = None
    try:
        g0 = _read_g0_pass(
            model_smoke_receipt,
            config,
            capacity=capacity,
        )
        failure_stage = "branch_authorization"
        if capacity == "fullrank":
            authorization_details = _authorization_details_for(
                config, capacity, "joint", authorization_receipt
            )
            assert authorization_details is not None
            authorization = dict(authorization_details["receipt"])
        elif authorization_receipt is not None:
            raise RuntimeError("LoRA positive control accepts no branch authorization")
        failure_stage = "data_manifest"
        _, manifest, data_manifest_sha256 = _load_data_manifest(
            config, content_splits=set()
        )
        failure_stage = "oracle_analysis"
        rows, control_rows_receipt = generate_control_rows(config, manifest)
        oracle_analysis = produce_oracle_analysis_receipt(rows)
        oracle_accuracy = float(oracle_analysis["terminal_joint_accuracy"])
        min_oracle = float(
            config["training"]["positive_control"]["min_oracle_readout_accuracy"]
        )
        if oracle_accuracy < min_oracle:
            raise RuntimeError(
                f"exact 48-row oracle analysis failed: {oracle_accuracy} < {min_oracle}"
            )
        expected_setup = g0.get("setup")
        if not isinstance(expected_setup, Mapping):
            raise RuntimeError("G0 receipt omits its deterministic setup")
        _validate_registered_setup_fields(
            config, expected_setup, capacity=capacity, model_seed=model_seed
        )
        g0 = validate_g0_pass(
            REPO_ROOT,
            model_smoke_receipt,
            canonical_relative_path=_repo_relative(
                _setup_receipt_path("g0", capacity, model_seed)
            ),
            expected_identity=_identity(config, phase=f"{capacity}_g0"),
            capacity=capacity,
            model_seed=model_seed,
            data_manifest_sha256=data_manifest_sha256,
            expected_setup=expected_setup,
            expected_branch_authorization=(
                authorization_details["root_lora_miss_lineage"]
                if authorization_details is not None else None
            ),
            k1_max_logit_abs_error=float(
                config["gates"]["k1_max_logit_abs_error"]
            ),
            train_k=int(config["training"]["train_k"]),
            max_recurrence=int(config["architecture"]["max_recurrence"]),
            expected_adaptation_targets=int(
                adaptation["expected_targets"]
            ),
            expected_adaptation_parameters=int(adaptation["expected_parameters"]),
            expected_adaptation_dropout=float(adaptation["dropout"]),
            expected_adaptation_scale=float(adaptation["scale"]),
            expected_lora_rank=lora_rank,
            expected_peft_version=_locked_requirement_version("peft"),
            adaptation_gradient_clip=float(
                config["training"]["adaptation_gradient_clip"]
            ),
            common_gradient_clip=float(
                config["training"]["common_gradient_clip"]
            ),
            worst_depth_seed=int(
                config["training"]["g0_control"]["worst_depth_seed"]
            ),
            min_free_memory_gib=4.0,
        )
        _require_supplied_initialization_bundle(expected_setup, initialization_bundle)
        failure_stage = "model_setup"
        tokenizer, wrapper, setup = _build_new(
            config, capacity=capacity, model_seed=model_seed,
            initialization_bundle=initialization_bundle,
        )
        setup_authorization = (
            authorization_details["root_lora_miss_lineage"]
            if authorization_details is not None else None
        )
        _require_setup_binding(
            g0, capacity=capacity, model_seed=model_seed,
            data_manifest_sha256=data_manifest_sha256,
            live_setup=setup, label=f"{capacity} G0",
            expected_setup_authorization=setup_authorization,
        )
        # Reopen the seed-matched shared tensors even though the stdlib oracle
        # is model-free.  This explicit restore is part of the frozen control
        # boundary: the learned overfit run always starts from registered init.
        shared_state, _ = load_initialization_bundle(
            config, model_seed, initialization_bundle
        )
        wrapper.load_extra_state_dict(shared_state)
        wrapper.zero_grad(set_to_none=True)
        initial_trainable = _trainable_receipt(wrapper)
        if initial_trainable != setup["trainable_parameters"]:
            raise RuntimeError("positive-control oracle reset did not restore shared initialization")
        delta_baseline = _parameter_delta_baseline(wrapper)

        training = config["training"]
        updates = int(training["positive_control"]["updates"])
        accumulation = int(training["gradient_accumulation"])
        schedule = _positive_control_schedule(rows, updates, accumulation, model_seed)
        total_microbatches = updates * accumulation
        if len(schedule) != total_microbatches:
            raise RuntimeError("positive-control schedule length changed")
        probe_steps = _positive_control_probe_steps(updates)
        dropout_probe_indices = {1, max(1, total_microbatches // 2), total_microbatches}
        diagnostics.update(
            {
                "fixed_probe_steps": list(probe_steps),
                "geometry": {
                    "rows": len(rows),
                    "optimizer_updates": updates,
                    "gradient_accumulation": accumulation,
                    "singleton_microbatches": total_microbatches,
                    "loss_divisor": accumulation,
                    "optimizer_zero_grad_calls": updates + 1,
                    "adaptation_clip_calls": updates,
                    "common_state_clip_calls": updates,
                    "optimizer_step_calls": updates,
                    "early_stopping": False,
                    "checkpoint_selection": False,
                },
                "row_order_sha256": hashlib.sha256().hexdigest(),
                "dropout_schedule_sha256": hashlib.sha256().hexdigest(),
                "optimizer_steps_sha256": hashlib.sha256().hexdigest(),
            }
        )
        adaptation_parameters, common_parameters = _parameter_groups(wrapper)
        base_trainable_parameters = sum(
            parameter.numel()
            for name, parameter in wrapper.named_parameters()
            if name.startswith("base_model.") and parameter.requires_grad
        )
        if base_trainable_parameters != 0:
            raise RuntimeError("positive-control base model has trainable parameters")
        optimizer = _build_optimizer(
            adaptation_parameters,
            common_parameters,
            learning_rate=float(training["learning_rate"]),
            weight_decay=0.0,
        )
        if [group.get("group_name") for group in optimizer.param_groups] != [
            "adaptation", "common_state"
        ]:
            raise RuntimeError("positive-control optimizer group order changed")
        wrapper.train()
        optimizer.zero_grad(set_to_none=True)
        order_digest = hashlib.sha256()
        dropout_digest = hashlib.sha256()
        optimizer_digest = hashlib.sha256()
        exposures = {str(row["id"]): 0 for row in rows}
        depth_exposures = {str(depth): 0 for depth in map(int, training["positive_control"]["depths"])}
        optimizer_probe_steps = set(probe_steps) - {0}
        minimum_clip_scales = {"adaptation": 1.0, "common_state": 1.0}

        def record_probe(step: int) -> None:
            before_norms = _parameter_norm_receipt(wrapper)
            before_deltas = _parameter_delta_norm_receipt(wrapper, delta_baseline)
            intact = _evaluate_positive_control(
                wrapper, tokenizer, rows, config, capacity=capacity,
                model_seed=model_seed, probe_step=step, adaptation_enabled=True,
            )
            disabled = _evaluate_positive_control(
                wrapper, tokenizer, rows, config, capacity=capacity,
                model_seed=model_seed, probe_step=step, adaptation_enabled=False,
            )
            after_norms = _parameter_norm_receipt(wrapper)
            after_deltas = _parameter_delta_norm_receipt(wrapper, delta_baseline)
            if before_norms != after_norms or before_deltas != after_deltas:
                raise RuntimeError("positive-control diagnostic changed trainable parameters")
            diagnostics["evaluations"].extend((intact, disabled))
            diagnostics["parameter_probes"].append(
                {
                    "step": step,
                    "parameter_norms": before_norms,
                    "parameter_delta_norms": before_deltas,
                    "diagnostic_parameter_state_unchanged": True,
                }
            )

        failure_stage = "initial_diagnostics"
        record_probe(0)
        failure_stage = "state_path_overfit"
        for update in range(1, updates + 1):
            totals = {"objective_loss": 0.0, "state_loss": 0.0, "fixed_point_loss": 0.0}
            for _ in range(accumulation):
                event = schedule[completed_microbatches]
                row = rows[int(event["row_index"])]
                k = int(row["depth"])
                batch = _encode_row(
                    tokenizer, row, config, k=k, device=torch.device("cuda")
                )
                dropout_seed = int(event["dropout_seed"])
                capture = int(event["microbatch_index"]) in dropout_probe_indices
                wrapper.adaptation.begin_microbatch(
                    dropout_seed, capture_masks=capture
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model_output = _forward(
                        wrapper, batch, k=k, compute_answer=False
                    )
                    objective_loss = _objective_loss(
                        model_output, config, "state_only"
                    )
                    scaled_loss = objective_loss / accumulation
                losses = {
                    "objective_loss": float(objective_loss.detach().float().cpu()),
                    "state_loss": float(model_output.state_loss.detach().float().cpu()),
                    "fixed_point_loss": float(
                        model_output.fixed_point_loss.detach().float().cpu()
                    ),
                }
                if not all(math.isfinite(value) for value in losses.values()):
                    raise RuntimeError(
                        f"nonfinite positive-control loss at microbatch "
                        f"{event['microbatch_index']}"
                    )
                scaled_loss.backward()
                dropout_receipt = wrapper.adaptation.end_microbatch()
                expected_calls = (k - 1) * int(
                    config["architecture"]["adaptation"][capacity]["expected_targets"]
                )
                if (
                    dropout_receipt["calls"] != expected_calls
                    or dropout_receipt["cycles"] != k - 1
                    or not dropout_receipt["cycle_order_identical"]
                    or not dropout_receipt["each_cycle_exact_target_set"]
                ):
                    raise RuntimeError("positive-control adaptation schedule changed")
                order_event = {
                    key: event[key]
                    for key in (
                        "optimizer_step", "microbatch_in_step", "microbatch_index",
                        "row_index", "id", "k",
                    )
                }
                order_digest.update(
                    json.dumps(
                        order_event, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8") + b"\n"
                )
                dropout_event = {
                    **order_event,
                    "dropout_seed": dropout_seed,
                    "calls": dropout_receipt["calls"],
                    "call_manifest_sha256": dropout_receipt["call_manifest_sha256"],
                }
                dropout_digest.update(
                    json.dumps(
                        dropout_event, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8") + b"\n"
                )
                if capture:
                    diagnostics["dropout_probes"].append(
                        {**dropout_event, "mask_sha256": dropout_receipt["mask_sha256"]}
                    )
                exposures[str(row["id"])] += 1
                depth_exposures[str(k)] += 1
                completed_microbatches += 1
                diagnostics["completed_microbatches"] = completed_microbatches
                diagnostics["row_order_sha256"] = order_digest.hexdigest()
                diagnostics["dropout_schedule_sha256"] = dropout_digest.hexdigest()
                diagnostics["row_exposures"] = dict(sorted(exposures.items()))
                diagnostics["depth_exposures"] = dict(sorted(depth_exposures.items()))
                for name, value in losses.items():
                    totals[name] += value / accumulation

            gradient_probe = _gradient_receipt(wrapper) if update in optimizer_probe_steps else None
            if gradient_probe is not None:
                required_groups = ("adaptation", "initializer", "step", "sufficiency", "damping")
                if gradient_probe["base_gradient_tensors"] != 0 or any(
                    gradient_probe[name]["tensors"] == 0
                    or gradient_probe[name]["with_gradient"] != gradient_probe[name]["tensors"]
                    or gradient_probe[name]["finite"] != gradient_probe[name]["tensors"]
                    for name in required_groups
                ):
                    raise RuntimeError("positive-control gradient probe is incomplete or nonfinite")
            adaptation_norm = torch.nn.utils.clip_grad_norm_(
                adaptation_parameters, float(training["adaptation_gradient_clip"])
            )
            common_norm = torch.nn.utils.clip_grad_norm_(
                common_parameters, float(training["common_gradient_clip"])
            )
            adaptation_norm_value = float(adaptation_norm.detach().float().cpu())
            common_norm_value = float(common_norm.detach().float().cpu())
            if (
                not math.isfinite(adaptation_norm_value)
                or not math.isfinite(common_norm_value)
                or adaptation_norm_value <= 0.0
                or common_norm_value <= 0.0
            ):
                raise RuntimeError(f"invalid positive-control gradient norm at update {update}")
            optimizer_event = {
                "step": update,
                "microbatch_start": (update - 1) * accumulation + 1,
                "microbatch_end": update * accumulation,
                **totals,
                "adaptation_preclip_gradient_norm": adaptation_norm_value,
                "adaptation_applied_clip_scale": _clip_scale(
                    adaptation_norm_value, float(training["adaptation_gradient_clip"])
                ),
                "common_state_preclip_gradient_norm": common_norm_value,
                "common_state_applied_clip_scale": _clip_scale(
                    common_norm_value, float(training["common_gradient_clip"])
                ),
                "adaptation_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "common_state_learning_rate": float(optimizer.param_groups[1]["lr"]),
                "microbatches": accumulation,
                "adaptation_gradient_finite": True,
                "common_state_gradient_finite": True,
                "base_trainable_parameters": base_trainable_parameters,
                "gradient_probe": gradient_probe,
            }
            if (
                optimizer_event["adaptation_learning_rate"]
                != float(training["learning_rate"])
                or optimizer_event["common_state_learning_rate"]
                != float(training["learning_rate"])
            ):
                raise RuntimeError("positive-control constant learning rate changed")
            optimizer_digest.update(
                json.dumps(
                    optimizer_event, sort_keys=True, separators=(",", ":")
                ).encode("utf-8") + b"\n"
            )
            minimum_clip_scales["adaptation"] = min(
                minimum_clip_scales["adaptation"],
                optimizer_event["adaptation_applied_clip_scale"],
            )
            minimum_clip_scales["common_state"] = min(
                minimum_clip_scales["common_state"],
                optimizer_event["common_state_applied_clip_scale"],
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_updates = update
            diagnostics["completed_updates"] = completed_updates
            diagnostics["optimizer_steps_sha256"] = optimizer_digest.hexdigest()
            diagnostics["minimum_applied_clip_scales"] = minimum_clip_scales
            if update in optimizer_probe_steps:
                diagnostics["optimizer_step_probes"].append(optimizer_event)
                record_probe(update)

        failure_stage = "final_optimizer_audit"
        if completed_updates != updates or completed_microbatches != total_microbatches:
            raise RuntimeError("positive-control training geometry was incomplete")
        exposure_values = sorted(exposures.values())
        if exposure_values[-1] - exposure_values[0] > 1:
            raise RuntimeError("positive-control row-cycle exposure imbalance changed")
        if updates == 256 and accumulation == 16 and len(rows) == 48:
            if exposure_values.count(85) != 32 or exposure_values.count(86) != 16:
                raise RuntimeError("confirmatory positive-control row exposures changed")
            if depth_exposures != {"2": 1368, "3": 1368, "4": 1360}:
                raise RuntimeError("confirmatory positive-control depth exposures changed")
        if len(diagnostics["dropout_probes"]) != len(dropout_probe_indices):
            raise RuntimeError("positive-control dropout probes are incomplete")
        optimizer_state = optimizer_state_receipt(
            optimizer,
            delta_parameters=adaptation_parameters,
            allowed_missing_parameters=[wrapper.aggregate_logit],
        )
        if (
            not optimizer_state["delta_states_complete"]
            or not optimizer_state["all_required_group_states_complete_and_finite"]
        ):
            raise RuntimeError("positive-control optimizer state is incomplete")
        final_trainable = _trainable_receipt(wrapper)
        if final_trainable["values_sha256"] == initial_trainable["values_sha256"]:
            raise RuntimeError("positive-control trainable tensors did not change")
        final_parameter_deltas = _parameter_delta_norm_receipt(wrapper, delta_baseline)
        if (
            final_parameter_deltas["adaptation_output"]["l2_delta_norm"] <= 0.0
            or final_parameter_deltas["common_state"]["l2_delta_norm"] <= 0.0
        ):
            raise RuntimeError("positive-control required parameter groups did not move")
        diagnostics.update(
            {
                "optimizer_state": optimizer_state,
                "initial_trainable_parameters": initial_trainable,
                "final_trainable_parameters": final_trainable,
                "final_parameter_delta_norms": final_parameter_deltas,
                "parameter_values_changed": True,
            }
        )
        final_intact = next(
            item for item in diagnostics["evaluations"]
            if item["step"] == updates and item["adaptation_mode"] == "intact"
        )
        overfit_accuracy = float(final_intact["overall"]["joint_final_accuracy"])
        threshold = float(
            training["positive_control"]["min_overfit_final_joint_accuracy"]
        )
        failure_stage = "fixed_final_overfit_gate"
        if overfit_accuracy < threshold:
            raise RuntimeError(
                f"tiny state-path overfit failed: {overfit_accuracy} < {threshold}"
            )
        receipt = _with_identity({
            "schema_version": 1,
            "status": "POSITIVE_CONTROL_PASS",
            **common_identity,
            "capacity": capacity,
            "model_seed": int(model_seed),
            "data_manifest_sha256": data_manifest_sha256,
            "g0_lineage": _lineage(model_smoke_receipt, g0),
            "branch_authorization": _lineage(authorization_receipt, authorization)
            if authorization_receipt and authorization else None,
            "shared_initialization": setup["shared_initialization"],
            "setup": setup,
            "control_rows": control_rows_receipt,
            "oracle_analysis": oracle_analysis,
            "oracle_readout_accuracy": oracle_accuracy,
            "overfit_rows": len(rows),
            "overfit_updates": updates,
            "overfit_gradient_accumulation": accumulation,
            "overfit_microbatches": total_microbatches,
            "overfit_final_joint_accuracy": overfit_accuracy,
            "overfit_final_joint_correct": int(
                final_intact["overall"]["terminal_correct_counts"]["joint"]
            ),
            "training_diagnostics": diagnostics,
            "authorizes_training": True,
            "authorizes_result_training": True,
            "authorizes_result_evaluation": False,
            "benchmark_files_read": 0,
            "result_payloads_opened": [],
            "sealed_contrast_payloads_opened": [],
            "scientific_evidence": False,
        })
        _write_json(output, receipt)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "SETUP_CONTROL_FAILED",
            **common_identity,
            "capacity": capacity,
            "model_seed": int(model_seed),
            "data_manifest_sha256": data_manifest_sha256,
            "g0_lineage": _lineage(model_smoke_receipt, g0) if g0 else None,
            "branch_authorization": _lineage(authorization_receipt, authorization)
            if authorization_receipt and authorization else None,
            "shared_initialization": setup["shared_initialization"] if setup else None,
            "setup": setup,
            "control_rows": control_rows_receipt,
            "oracle_analysis": oracle_analysis,
            "oracle_readout_accuracy": oracle_accuracy,
            "failure_stage": failure_stage,
            "error_type": type(exc).__name__,
            "error": str(exc) or repr(exc) or type(exc).__name__,
            "completed_updates": completed_updates,
            "completed_microbatches": completed_microbatches,
            "training_diagnostics": diagnostics,
            "authorizes_training": False,
            "authorizes_result_training": False,
            "authorizes_result_evaluation": False,
            "benchmark_files_read": 0,
            "result_payloads_opened": [],
            "sealed_contrast_payloads_opened": [],
            "scientific_evidence": False,
        }
        if os.path.lexists(output):
            raise RuntimeError(
                "positive-control failure cannot overwrite an existing canonical receipt"
            ) from exc
        failure_receipt = _with_identity(failure)
        if os.path.lexists(failure_mirror):
            raise RuntimeError(
                "refusing to overwrite positive-control failure mirror: "
                f"{failure_mirror}"
            ) from exc
        _write_new_json_pair(output, failure_mirror, failure_receipt)
        raise


def _schedule(step: int, total_steps: int, warmup_fraction: float) -> float:
    warmup = max(1, int(total_steps * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _metrics_line_count(path: Path) -> int:
    raw = read_stable_bytes(REPO_ROOT, path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"metrics log is not UTF-8: {path}") from exc
    return sum(1 for line in text.splitlines() if line.strip())


def _checkpoint_identity(metadata: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in metadata.items() if key != "checkpoint_identity_sha256"}
    )


def _save_checkpoint(
    wrapper: StateLoopModel,
    config: Mapping[str, Any],
    output: Path,
    *,
    capacity: str,
    objective: str,
    model_seed: int,
    step: int,
    setup: Mapping[str, Any],
    data_manifest_sha256: str,
    g0_lineage: Mapping[str, Any],
    control_lineage: Mapping[str, Any],
    authorization_lineage: Mapping[str, Any] | None,
    training_prompt_tokens: int,
    training_layer_token_applications: int,
    training_order_sha256: str,
    dropout_schedule_sha256: str,
    dropout_probes: list[dict[str, Any]],
    metrics_path: Path,
    optimizer_steps_path: Path,
    optimizer_state: Mapping[str, Any],
    optimizer_step_receipt: Mapping[str, Any],
    setup_barrier: Mapping[str, Any],
    prior_training_barriers: Sequence[Mapping[str, Any]],
    training_launch_preflight_receipt: Mapping[str, Any],
    training_attempt_authorization: Mapping[str, Any],
    training_attempt_history: Mapping[str, Any],
    training_attempt_journal_path: str,
) -> dict[str, Any]:
    _require_new_output(output, directory=True, kind="checkpoint")
    adaptation_path = output / "adaptation_state.pt"
    common_path = output / "loop_state.pt"
    torch.save(wrapper.delta_state_dict(), adaptation_path)
    torch.save(wrapper.extra_state_dict(), common_path)
    metadata = {
        "schema_version": 1,
        **_identity(config, phase=f"{capacity}_{objective}_training"),
        "capacity": capacity,
        "objective": objective,
        "model_seed": int(model_seed),
        "step": int(step),
        "trainable_parameters": setup["trainable_parameters"],
        "adaptation_parameters": setup["adaptation_parameters"],
        "adaptation_target_manifest_sha256": setup["adaptation_target_manifest_sha256"],
        "shared_initialization": setup["shared_initialization"],
        "environment": setup["environment"],
        "setup": dict(setup),
        "setup_sha256": _canonical_sha256(setup),
        "stable_setup": _stable_setup_receipt(setup),
        "data_manifest_sha256": data_manifest_sha256,
        "g0_lineage": dict(g0_lineage),
        "positive_control_lineage": dict(control_lineage),
        "branch_authorization_lineage": dict(authorization_lineage) if authorization_lineage else None,
        "training_prompt_tokens": training_prompt_tokens,
        "training_layer_token_applications": training_layer_token_applications,
        "training_order_sha256": training_order_sha256,
        "dropout_schedule_sha256": dropout_schedule_sha256,
        "dropout_probes": dropout_probes,
        "train_metrics_sha256": _sha256(metrics_path),
        "train_metrics_rows": _metrics_line_count(metrics_path),
        "train_metrics_path": _repo_relative(metrics_path),
        "optimizer_steps_sha256": _sha256(optimizer_steps_path),
        "optimizer_steps_rows": _metrics_line_count(optimizer_steps_path),
        "optimizer_steps_path": _repo_relative(optimizer_steps_path),
        "optimizer_state": dict(optimizer_state),
        "optimizer_step_receipt": dict(optimizer_step_receipt),
        "setup_barrier": dict(setup_barrier),
        "prior_training_barriers": [
            dict(proof) for proof in prior_training_barriers
        ],
        "training_launch_preflight": dict(training_launch_preflight_receipt),
        "training_attempt_authorization": dict(training_attempt_authorization),
        "training_attempt_history": dict(training_attempt_history),
        "training_attempt_journal_path": training_attempt_journal_path,
        "adaptation_state_sha256": _sha256(adaptation_path),
        "loop_state_sha256": _sha256(common_path),
    }
    metadata["checkpoint_identity_sha256"] = _checkpoint_identity(metadata)
    _write_json(output / "checkpoint.json", metadata)
    return metadata


def train(
    config: Mapping[str, Any],
    *,
    capacity: str,
    objective: str,
    model_seed: int,
    output_dir: Path,
    initialization_bundle: Path,
    model_smoke_receipt: Path,
    positive_control_receipt: Path,
    authorization_receipt: Path | None,
) -> None:
    require_confirmatory_config(config)
    if capacity not in {"lora", "fullrank"} or objective not in {"joint", "state_only"}:
        raise ValueError(f"invalid training cell: {capacity}/{objective}")
    if model_seed not in set(map(int, config["training"]["train_seeds"])):
        raise RuntimeError("model seed is not preregistered")
    stage = _training_stage(capacity, objective)
    target = TrainingCell(stage, capacity, objective, model_seed)
    authorization_details = _authorization_details_for(
        config, capacity, objective, authorization_receipt
    )
    authorization = (
        None
        if authorization_details is None
        else dict(authorization_details["receipt"])
    )
    target_authorization = _stage_required_authorization(
        stage, authorization_details
    )

    # Receipt-only preflight comes before any result payload, model load, or
    # output creation.  Every setup pair required by this stage must pass its
    # full scientific contract, and every prior reached stage must have a
    # closed terminal training graph.
    data_dir, data_manifest, data_manifest_sha256 = _load_data_manifest(
        config, content_splits=set()
    )
    root_lora_miss = (
        authorization_details["root_lora_miss_lineage"]
        if authorization_details is not None else None
    )
    setup_barrier_proof, setup_cells = _setup_barrier(
        config,
        stage=stage,
        data_manifest_sha256=data_manifest_sha256,
        root_lora_miss_lineage=root_lora_miss,
        target=target,
        target_g0_path=model_smoke_receipt,
        target_control_path=positive_control_receipt,
    )
    prior_training_barrier_proofs = _prior_training_barriers(
        config,
        target_stage=stage,
        authorization_details=authorization_details,
        setup_barrier=setup_barrier_proof,
    )
    canonical_paths = canonical_training_cell_paths(
        REPO_ROOT, target, steps=int(config["training"]["train_steps"])
    )
    if output_dir.absolute() != canonical_paths.external_dir:
        raise RuntimeError(
            f"training output is not the canonical result cell: {output_dir}"
        )
    output_dir = canonical_paths.external_dir
    contracts = _training_contracts(config, stage)
    # The only in-place recovery is the exact post-publication crash window:
    # both immutable terminal receipts already validate, while their durable
    # attempt journal is still STARTED.  Finalize that journal without opening
    # training rows or loading a model, and never create a second result.
    if recover_published_training_completion(
        REPO_ROOT,
        target,
        contracts[target],
        required_authorization=target_authorization,
        expected_setup_barrier_identity_sha256=setup_barrier_proof[
            "barrier_identity_sha256"
        ],
        expected_prior_training_barrier_identity_sha256s=[
            proof["barrier_identity_sha256"]
            for proof in prior_training_barrier_proofs
        ],
    ):
        return
    launch_preflight_proof = training_launch_preflight(
        REPO_ROOT,
        target,
        contracts,
        target_authorization=target_authorization,
    )
    _bind_training_cells_to_setup(
        launch_preflight_proof["completed_peer_proofs"], setup_barrier_proof
    )
    attempt_header = {
        key: value
        for key, value in _identity(config, phase=target.phase).items()
        if key != "phase"
    }
    attempt_cell = {
        "stage": target.stage,
        "capacity": target.capacity,
        "objective": target.objective,
        "seed": target.seed,
        "slug": target.slug,
    }
    attempt_paths = [
        _repo_relative(canonical_paths.external_dir),
        _repo_relative(canonical_paths.tracked_dir),
    ]
    attempt_context = {
        "setup_barrier_identity_sha256": setup_barrier_proof[
            "barrier_identity_sha256"
        ],
        "prior_training_barrier_identity_sha256s": [
            proof["barrier_identity_sha256"]
            for proof in prior_training_barrier_proofs
        ],
        "training_launch_preflight_identity_sha256": launch_preflight_proof[
            "preflight_identity_sha256"
        ],
        "training_launch_peer_vector": list(launch_preflight_proof["peers"]),
        "branch_authorization_lineage": target_authorization,
    }
    try:
        replay_archive = required_training_replay_archive(
            REPO_ROOT,
            slug=target.slug,
            header=attempt_header,
            cell=attempt_cell,
            canonical_paths=attempt_paths,
            expected_archive_header=_failed_archive_header(config),
        )
        training_attempt_authorization = prepare_training_attempt(
            REPO_ROOT,
            slug=target.slug,
            header=attempt_header,
            cell=attempt_cell,
            canonical_paths=attempt_paths,
            context=attempt_context,
            replay_archive=replay_archive,
        )
        ensure_attempt_output(output_dir, training_attempt_authorization)
        start_training_attempt(
            REPO_ROOT,
            slug=target.slug,
            header=attempt_header,
            cell=attempt_cell,
            canonical_paths=attempt_paths,
            authorization=training_attempt_authorization,
        )
        training_attempt_history = validate_training_attempt_history(
            REPO_ROOT,
            slug=target.slug,
            header=attempt_header,
            cell=attempt_cell,
            canonical_paths=attempt_paths,
            current_authorization=training_attempt_authorization,
            expected_archive_header=_failed_archive_header(config),
        )
    except AttemptReceiptError as exc:
        raise RuntimeError(f"training attempt authorization failed: {exc}") from exc

    # Only after all receipt/barrier proofs may the train payload be opened.
    train_rows = validated_data_rows(
        config, data_dir, data_manifest, ("train",)
    )["train"]
    target_setup = setup_cells[f"{capacity}_seed{model_seed}"]
    g0 = target_setup["g0"]
    control = target_setup["control"]
    _require_supplied_initialization_bundle(
        target_setup["setup"], initialization_bundle
    )
    random.seed(model_seed)
    tokenizer, wrapper, setup = _build_new(
        config, capacity=capacity, model_seed=model_seed,
        initialization_bundle=initialization_bundle,
    )
    if capacity == "fullrank":
        setup_authorization = root_lora_miss
        if not isinstance(setup_authorization, Mapping):
            raise RuntimeError("full-rank setup lacks the canonical root LoRA-miss lineage")
    else:
        setup_authorization = None
    _require_setup_binding(
        g0, capacity=capacity, model_seed=model_seed,
        data_manifest_sha256=data_manifest_sha256,
        live_setup=setup, label=f"{capacity} G0",
        expected_setup_authorization=setup_authorization,
    )
    _require_setup_binding(
        control, capacity=capacity, model_seed=model_seed,
        data_manifest_sha256=data_manifest_sha256,
        live_setup=setup,
        label=f"{capacity} positive control",
        expected_setup_authorization=setup_authorization,
    )
    if control.get("g0_lineage") != _lineage(model_smoke_receipt, g0):
        raise RuntimeError("positive control does not bind the exact supplied G0 receipt")
    rng = random.Random(model_seed)
    rng.shuffle(train_rows)
    tracked_run_dir = canonical_paths.tracked_dir
    training = config["training"]
    total_steps = int(training["train_steps"])
    accumulation = int(training["gradient_accumulation"])
    total_microbatches = total_steps * accumulation
    probe_indices = {1, max(1, total_microbatches // 2), total_microbatches}
    train_k = int(training["train_k"])
    adaptation_parameters, common_parameters = _parameter_groups(wrapper)
    optimizer = _build_optimizer(
        adaptation_parameters,
        common_parameters,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    metrics_path = output_dir / "train_metrics.jsonl"
    optimizer_steps_path = output_dir / "optimizer_steps.jsonl"
    wrapper.train()
    optimizer.zero_grad(set_to_none=True)
    example_index = 0
    prompt_tokens = 0
    layer_token_applications = 0
    order_digest = hashlib.sha256()
    dropout_digest = hashlib.sha256()
    dropout_probes: list[dict[str, Any]] = []
    optimizer_step_digest = hashlib.sha256()
    optimizer_step_probes: list[dict[str, Any]] = []
    optimizer_probe_steps = {1, max(1, total_steps // 2), total_steps}
    minimum_clip_scales = {"adaptation": 1.0, "common_state": 1.0}
    loop_layers = int(config["architecture"]["loop_end"]) - int(config["architecture"]["loop_start"])
    started = time.time()
    for step in range(1, total_steps + 1):
        totals = {"loss": 0.0, "answer": 0.0, "state": 0.0, "fixed": 0.0}
        for _ in range(accumulation):
            row = train_rows[example_index % len(train_rows)]
            example_index += 1
            batch = _encode_row(tokenizer, row, config, k=train_k, device=torch.device("cuda"))
            compute = recurrent_compute_receipt(
                sequence_tokens=batch["prompt_tokens"],
                total_layers=(
                    int(config["architecture"]["expected_num_layers"])
                    if objective == "joint" else int(config["architecture"]["loop_end"])
                ),
                loop_layers=loop_layers,
                k=train_k,
            ).total_layer_token_applications
            event = {
                "microbatch_index": example_index,
                "id": row["id"],
                "k": train_k,
            }
            encoded_event = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
            order_digest.update(encoded_event + b"\n")
            dropout_seed = microbatch_dropout_seed(model_seed, example_index, str(row["id"]), train_k)
            capture = example_index in probe_indices
            wrapper.adaptation.begin_microbatch(dropout_seed, capture_masks=capture)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model_output = _forward(
                    wrapper, batch, k=train_k, compute_answer=(objective == "joint")
                )
                loss = _objective_loss(model_output, config, objective)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"nonfinite loss at step {step}")
                scaled = loss / accumulation
            scaled.backward()
            dropout_receipt = wrapper.adaptation.end_microbatch()
            expected_calls = 3 * int(config["architecture"]["adaptation"][capacity]["expected_targets"])
            if dropout_receipt["calls"] != expected_calls:
                raise RuntimeError("training adaptation call schedule changed")
            if (
                dropout_receipt["cycles"] != train_k - 1
                or not dropout_receipt["cycle_order_identical"]
                or not dropout_receipt["each_cycle_exact_target_set"]
            ):
                raise RuntimeError("training ordered adaptation schedule changed")
            schedule_event = {
                **event,
                "prompt_tokens": batch["prompt_tokens"],
                "dropout_seed": dropout_seed,
                "calls": dropout_receipt["calls"],
                "call_manifest_sha256": dropout_receipt["call_manifest_sha256"],
            }
            dropout_digest.update(
                json.dumps(schedule_event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            )
            if capture:
                dropout_probes.append({**schedule_event, "mask_sha256": dropout_receipt["mask_sha256"]})
            prompt_tokens += batch["prompt_tokens"]
            layer_token_applications += compute
            totals["loss"] += float(loss.detach().cpu()) / accumulation
            if model_output.answer_loss is not None:
                totals["answer"] += float(model_output.answer_loss.detach().cpu()) / accumulation
            totals["state"] += float(model_output.state_loss.detach().cpu()) / accumulation
            totals["fixed"] += float(model_output.fixed_point_loss.detach().cpu()) / accumulation
        adaptation_norm = torch.nn.utils.clip_grad_norm_(
            adaptation_parameters, float(training["adaptation_gradient_clip"])
        )
        common_norm = torch.nn.utils.clip_grad_norm_(
            common_parameters, float(training["common_gradient_clip"])
        )
        if not bool(torch.isfinite(adaptation_norm)) or not bool(torch.isfinite(common_norm)):
            raise RuntimeError(f"nonfinite gradient norm at step {step}")
        multiplier = _schedule(step - 1, total_steps, float(training["warmup_fraction"]))
        for group in optimizer.param_groups:
            group["lr"] = float(training["learning_rate"]) * multiplier
        group_learning_rates = {
            str(group["group_name"]): float(group["lr"])
            for group in optimizer.param_groups
        }
        if set(group_learning_rates) != {"adaptation", "common_state"}:
            raise RuntimeError("optimizer group names changed during training")
        adaptation_norm_value = float(adaptation_norm.detach().cpu())
        common_norm_value = float(common_norm.detach().cpu())
        optimizer_step_event = {
            "step": step,
            "adaptation_preclip_gradient_norm": adaptation_norm_value,
            "adaptation_applied_clip_scale": _clip_scale(
                adaptation_norm_value, float(training["adaptation_gradient_clip"])
            ),
            "common_state_preclip_gradient_norm": common_norm_value,
            "common_state_applied_clip_scale": _clip_scale(
                common_norm_value, float(training["common_gradient_clip"])
            ),
            "adaptation_gradient_finite": True,
            "common_state_gradient_finite": True,
            "base_trainable_parameters": 0,
            "adaptation_learning_rate": group_learning_rates["adaptation"],
            "common_state_learning_rate": group_learning_rates["common_state"],
        }
        encoded_optimizer_step = json.dumps(
            optimizer_step_event, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        optimizer_step_digest.update(encoded_optimizer_step + b"\n")
        _append_jsonl(optimizer_steps_path, optimizer_step_event)
        minimum_clip_scales["adaptation"] = min(
            minimum_clip_scales["adaptation"],
            optimizer_step_event["adaptation_applied_clip_scale"],
        )
        minimum_clip_scales["common_state"] = min(
            minimum_clip_scales["common_state"],
            optimizer_step_event["common_state_applied_clip_scale"],
        )
        if step in optimizer_probe_steps:
            optimizer_step_probes.append(optimizer_step_event)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 10 == 0 or step == total_steps:
            _append_jsonl(metrics_path, {
                "step": step,
                "capacity": capacity,
                "objective": objective,
                "model_seed": model_seed,
                **totals,
                "adaptation_learning_rate": group_learning_rates["adaptation"],
                "common_state_learning_rate": group_learning_rates["common_state"],
                "preclip_adaptation_gradient_norm": float(adaptation_norm.detach().cpu()),
                "preclip_common_gradient_norm": float(common_norm.detach().cpu()),
                "adaptation_applied_clip_scale": optimizer_step_event[
                    "adaptation_applied_clip_scale"
                ],
                "common_state_applied_clip_scale": optimizer_step_event[
                    "common_state_applied_clip_scale"
                ],
                "elapsed_seconds": time.time() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
            })
    optimizer_step_receipt = {
        "schema_version": 1,
        "steps": total_steps,
        "rows": _metrics_line_count(optimizer_steps_path),
        "events_sha256": optimizer_step_digest.hexdigest(),
        "group_names": [group["group_name"] for group in optimizer.param_groups],
        "clip_thresholds": {
            "adaptation": float(training["adaptation_gradient_clip"]),
            "common_state": float(training["common_gradient_clip"]),
        },
        "minimum_applied_clip_scales": minimum_clip_scales,
        "all_gradients_finite": True,
        "base_trainable_parameters": 0,
        "probes": optimizer_step_probes,
    }
    if optimizer_step_receipt["rows"] != total_steps:
        raise RuntimeError("optimizer-step receipt does not cover every training step")
    optimizer_state = optimizer_state_receipt(
        optimizer,
        delta_parameters=adaptation_parameters,
        allowed_missing_parameters=(
            [wrapper.aggregate_logit] if objective == "state_only" else []
        ),
    )
    checkpoint = output_dir / f"checkpoint_{total_steps:06d}"
    g0_lineage = _lineage(model_smoke_receipt, g0)
    control_lineage = _lineage(positive_control_receipt, control)
    authorization_lineage = (
        _lineage(authorization_receipt, authorization)
        if authorization_receipt and authorization else None
    )
    checkpoint_metadata = _save_checkpoint(
        wrapper,
        config,
        checkpoint,
        capacity=capacity,
        objective=objective,
        model_seed=model_seed,
        step=total_steps,
        setup=setup,
        data_manifest_sha256=data_manifest_sha256,
        g0_lineage=g0_lineage,
        control_lineage=control_lineage,
        authorization_lineage=authorization_lineage,
        training_prompt_tokens=prompt_tokens,
        training_layer_token_applications=layer_token_applications,
        training_order_sha256=order_digest.hexdigest(),
        dropout_schedule_sha256=dropout_digest.hexdigest(),
        dropout_probes=dropout_probes,
        metrics_path=metrics_path,
        optimizer_steps_path=optimizer_steps_path,
        optimizer_state=optimizer_state,
        optimizer_step_receipt=optimizer_step_receipt,
        setup_barrier=setup_barrier_proof,
        prior_training_barriers=prior_training_barrier_proofs,
        training_launch_preflight_receipt=launch_preflight_proof,
        training_attempt_authorization=training_attempt_authorization,
        training_attempt_history=training_attempt_history,
        training_attempt_journal_path=_repo_relative(canonical_paths.attempt_journal),
    )
    tracked_run_path = tracked_run_dir / "run.json"
    tracked_metrics_path = tracked_run_dir / "train_metrics.jsonl"
    tracked_optimizer_steps_path = tracked_run_dir / "optimizer_steps.jsonl"
    tracked_attempt_marker = tracked_run_dir / ATTEMPT_MARKER_NAME
    run = _with_identity({
        "schema_version": 1,
        "status": "TRAINING_COMPLETE",
        **_identity(config, phase=f"{capacity}_{objective}_training"),
        "capacity": capacity,
        "objective": objective,
        "model_seed": model_seed,
        "steps": total_steps,
        "data_manifest_sha256": data_manifest_sha256,
        "training_prompt_tokens": prompt_tokens,
        "training_layer_token_applications": layer_token_applications,
        "training_order_sha256": order_digest.hexdigest(),
        "dropout_schedule_sha256": dropout_digest.hexdigest(),
        "dropout_probes": dropout_probes,
        "train_metrics_sha256": _sha256(metrics_path),
        "train_metrics_rows": _metrics_line_count(metrics_path),
        "train_metrics_path": _repo_relative(metrics_path),
        "optimizer_steps_sha256": _sha256(optimizer_steps_path),
        "optimizer_steps_rows": _metrics_line_count(optimizer_steps_path),
        "optimizer_steps_path": _repo_relative(optimizer_steps_path),
        "optimizer_state": optimizer_state,
        "optimizer_step_receipt": optimizer_step_receipt,
        "setup_barrier": setup_barrier_proof,
        "prior_training_barriers": prior_training_barrier_proofs,
        "training_launch_preflight": launch_preflight_proof,
        "training_attempt_authorization": training_attempt_authorization,
        "training_attempt_history": training_attempt_history,
        "training_attempt_journal_path": _repo_relative(
            canonical_paths.attempt_journal
        ),
        "checkpoint_path": _repo_relative(checkpoint),
        "checkpoint_metadata_sha256": _sha256(checkpoint / "checkpoint.json"),
        "checkpoint_identity_sha256": checkpoint_metadata["checkpoint_identity_sha256"],
        "elapsed_seconds": time.time() - started,
        "setup": setup,
        "setup_sha256": _canonical_sha256(setup),
        "stable_setup": _stable_setup_receipt(setup),
        "g0_lineage": g0_lineage,
        "positive_control_lineage": control_lineage,
        "branch_authorization_lineage": authorization_lineage,
        "tracked_run_path": _repo_relative(tracked_run_path),
        "tracked_metrics_path": _repo_relative(tracked_metrics_path),
        "tracked_optimizer_steps_path": _repo_relative(tracked_optimizer_steps_path),
        "authorizes_training": False,
        "authorizes_result_training": False,
        "authorizes_result_evaluation": False,
        "benchmark_files_read": 0,
        "result_payloads_opened": ["train"],
        "sealed_contrast_payloads_opened": [],
        "training_or_evaluation_started": True,
        "scientific_evidence": False,
    })
    # The external terminal receipt is installed last.  Therefore its
    # existence implies a complete, independently persisted tracked mirror;
    # mirror-only state after a crash remains visibly incomplete and must be
    # archived before a step-zero replay.
    for durable_path in (
        metrics_path,
        optimizer_steps_path,
        checkpoint / "adaptation_state.pt",
        checkpoint / "loop_state.pt",
        checkpoint / "checkpoint.json",
    ):
        _fsync_file(durable_path)
    _fsync_directory(checkpoint)
    _fsync_directory(output_dir)
    tracked_run_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(tracked_run_dir.parent)
    _copy_new_fsynced(output_dir / ATTEMPT_MARKER_NAME, tracked_attempt_marker)
    _copy_new_fsynced(metrics_path, tracked_metrics_path)
    _copy_new_fsynced(optimizer_steps_path, tracked_optimizer_steps_path)
    if (
        read_stable_bytes(REPO_ROOT, tracked_metrics_path)
        != read_stable_bytes(REPO_ROOT, metrics_path)
        or read_stable_bytes(REPO_ROOT, tracked_optimizer_steps_path)
        != read_stable_bytes(REPO_ROOT, optimizer_steps_path)
        or read_stable_bytes(REPO_ROOT, tracked_attempt_marker)
        != read_stable_bytes(REPO_ROOT, output_dir / ATTEMPT_MARKER_NAME)
    ):
        raise RuntimeError("tracked training payload mirror differs before finalization")
    _write_new_json_pair(output_dir / "run.json", tracked_run_path, run)
    if os.environ.get("QWEN35_TRAINING_CRASH_AT") == "terminal_receipts_published":
        raise RuntimeError(
            "injected training crash after terminal receipt publication"
        )
    complete_training_attempt(
        REPO_ROOT,
        slug=target.slug,
        header=attempt_header,
        cell=attempt_cell,
        canonical_paths=attempt_paths,
        authorization=training_attempt_authorization,
        terminal_run_lineage={
            "path": _repo_relative(output_dir / "run.json"),
            "sha256": _sha256(output_dir / "run.json"),
            "receipt_identity_sha256": run["receipt_identity_sha256"],
        },
    )


def _load_checkpoint(
    config: Mapping[str, Any],
    checkpoint: Path,
    *,
    capacity: str,
    objective: str,
    model_seed: int,
    expected_metadata_sha256: str,
) -> tuple[Any, StateLoopModel, dict[str, Any], str]:
    metadata_path = checkpoint / "checkpoint.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"checkpoint metadata is missing: {metadata_path}")
    metadata = read_verified_json_object(
        REPO_ROOT, metadata_path, expected_metadata_sha256
    )
    expected = {
        **_identity(config, phase=f"{capacity}_{objective}_training"),
        "capacity": capacity,
        "objective": objective,
        "model_seed": model_seed,
        "step": int(config["training"]["train_steps"]),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"checkpoint {key} mismatch")
    if metadata.get("checkpoint_identity_sha256") != _checkpoint_identity(metadata):
        raise RuntimeError("checkpoint identity mismatch")
    for key in ("g0_lineage", "positive_control_lineage"):
        validate_lineage_entry(metadata[key])
    if metadata.get("branch_authorization_lineage") is not None:
        validate_lineage_entry(metadata["branch_authorization_lineage"])
    adaptation_path = checkpoint / "adaptation_state.pt"
    common_path = checkpoint / "loop_state.pt"
    init_path = _resolve_repo_path(str(metadata["shared_initialization"]["bundle_path"]))
    tokenizer, wrapper, setup = _build_new(
        config, capacity=capacity, model_seed=model_seed,
        initialization_bundle=init_path,
    )
    if setup["shared_initialization"] != metadata["shared_initialization"]:
        raise RuntimeError("checkpoint/live shared-initialization receipt mismatch")
    if setup["trainable_parameters"] != metadata["trainable_parameters"]:
        raise RuntimeError("checkpoint/live initial trainable tensor receipt mismatch")
    if setup["adaptation_parameters"] != metadata["adaptation_parameters"]:
        raise RuntimeError("checkpoint/live adaptation parameter count mismatch")
    if setup["adaptation_target_manifest_sha256"] != metadata["adaptation_target_manifest_sha256"]:
        raise RuntimeError("checkpoint/live adaptation target manifest mismatch")
    if _stable_setup_receipt(setup) != metadata.get("stable_setup"):
        raise RuntimeError("checkpoint/live deterministic setup or device identity mismatch")
    with open_stable_regular(
        REPO_ROOT,
        adaptation_path,
        expected_sha256=str(metadata["adaptation_state_sha256"]),
    ) as handle:
        adaptation_state = torch.load(handle, map_location="cpu", weights_only=True)
    with open_stable_regular(
        REPO_ROOT,
        common_path,
        expected_sha256=str(metadata["loop_state_sha256"]),
    ) as handle:
        common_state = torch.load(handle, map_location="cpu", weights_only=True)
    wrapper.load_delta_state_dict(adaptation_state)
    wrapper.load_extra_state_dict(common_state)
    return tokenizer, wrapper, metadata, expected_metadata_sha256


def _trigger_launch_preflight(
    config: Mapping[str, Any], target: TrainingCell
) -> dict[str, Any]:
    """Require the exact frozen trigger-evaluation prefix for one stage."""

    matrices = {"A": STAGE_A_MATRIX, "B": STAGE_B_MATRIX, "C": STAGE_C_MATRIX}
    matrix = matrices[target.stage]
    target_index = matrix.index(target)
    vector: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for index, cell in enumerate(matrix):
        output = canonical_training_cell_paths(
            REPO_ROOT, cell, steps=int(config["training"]["train_steps"])
        ).trigger_output
        if index >= target_index:
            if os.path.lexists(output):
                raise RuntimeError(
                    f"out-of-order trigger evaluation output exists: {cell.slug}"
                )
            vector.append({"cell": cell.slug, "state": "ABSENT"})
            continue
        summary_path = output / "summary.json"
        if not output.is_dir() or not summary_path.is_file():
            raise RuntimeError(
                f"earlier trigger evaluation is not terminal: {cell.slug}"
            )
        summary = read_stable_json_object(REPO_ROOT, summary_path)
        claimed = summary.get("receipt_identity_sha256")
        if claimed != _canonical_sha256(
            {key: value for key, value in summary.items() if key != "receipt_identity_sha256"}
        ):
            raise RuntimeError(f"earlier trigger receipt identity changed: {cell.slug}")
        expected = {
            "status": "STATE_EVALUATION_COMPLETE",
            "eval_set": "trigger",
            "capacity": cell.capacity,
            "objective": cell.objective,
            "model_seed": cell.seed,
        }
        if any(summary.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"earlier trigger receipt targets another cell: {cell.slug}")
        proof = {
            "cell": cell.slug,
            "summary_path": _repo_relative(summary_path),
            "summary_sha256": _sha256(summary_path),
            "receipt_identity_sha256": claimed,
        }
        vector.append({"cell": cell.slug, "state": "COMPLETE"})
        completed.append(proof)
    proof = {
        "schema_version": 1,
        "status": "TRIGGER_LAUNCH_PREFLIGHT_PASS",
        "stage": target.stage,
        "target": target.slug,
        "cells": vector,
        "completed": completed,
    }
    proof["preflight_identity_sha256"] = _canonical_sha256(proof)
    return proof


def evaluate_state(
    config: Mapping[str, Any],
    *,
    checkpoint: Path,
    capacity: str,
    objective: str,
    model_seed: int,
    eval_set: str,
    output_dir: Path,
    authorization_receipt: Path | None,
) -> None:
    require_confirmatory_config(config)
    if eval_set not in {"trigger", "contrast"}:
        raise ValueError(eval_set)
    if capacity not in {"lora", "fullrank"} or objective not in {"joint", "state_only"}:
        raise ValueError(f"invalid evaluation cell: {capacity}/{objective}")
    if model_seed not in set(map(int, config["training"]["train_seeds"])):
        raise RuntimeError("evaluation seed is not preregistered")
    cell_stage = _training_stage(capacity, objective)
    cell = TrainingCell(cell_stage, capacity, objective, model_seed)
    if eval_set == "contrast":
        if capacity not in {"lora", "fullrank"} or objective != "joint":
            raise RuntimeError("sealed contrast evaluation is registered only for joint capacity arms")
        authorization_details = _contrast_authorization_details(
            config, authorization_receipt
        )
        reached_stage = "B"
    else:
        authorization_details = _authorization_details_for(
            config, capacity, objective, authorization_receipt
        )
        reached_stage = cell_stage
    authorization = (
        None
        if authorization_details is None
        else dict(authorization_details["receipt"])
    )
    split_names = (
        list(config["evaluation"]["trigger_splits"])
        if eval_set == "trigger" else list(config["evaluation"]["sealed_contrast_splits"])
    )

    # No result row, model payload, or output path is opened before the
    # complete setup and terminal-training graphs are proven.
    data_dir, manifest, data_manifest_sha256 = _load_data_manifest(
        config, content_splits=set()
    )
    root_lora_miss = (
        authorization_details["root_lora_miss_lineage"]
        if authorization_details is not None else None
    )
    setup_barrier_proof, _ = _setup_barrier(
        config,
        stage=reached_stage,
        data_manifest_sha256=data_manifest_sha256,
        root_lora_miss_lineage=root_lora_miss,
    )
    training_barrier_proof = _reached_training_barrier(
        config,
        reached_stage=reached_stage,
        authorization_details=authorization_details,
        setup_barrier=setup_barrier_proof,
    )
    target_training_proof = _target_training_cell_proof(
        training_barrier_proof, cell
    )
    canonical_paths = canonical_training_cell_paths(
        REPO_ROOT, cell, steps=int(config["training"]["train_steps"])
    )
    if checkpoint.absolute() != canonical_paths.checkpoint_dir:
        raise RuntimeError("evaluation checkpoint is not the canonical fixed-final cell")
    checkpoint = canonical_paths.checkpoint_dir
    expected_output = (
        canonical_paths.trigger_output
        if eval_set == "trigger"
        else ROOT / "runs" / f"{capacity}_joint_seed{model_seed}_contrast"
    )
    if output_dir.absolute() != expected_output:
        raise RuntimeError("evaluation output is not the canonical result cell")
    output_dir = expected_output
    trigger_launch_preflight = (
        _trigger_launch_preflight(config, cell) if eval_set == "trigger" else None
    )
    authorization_lineage = (
        dict(authorization_details["lineage"])
        if eval_set == "contrast" and authorization_details is not None else None
    )
    tokenizer, wrapper, checkpoint_metadata, checkpoint_sha256 = _load_checkpoint(
        config,
        checkpoint,
        capacity=capacity,
        objective=objective,
        model_seed=model_seed,
        expected_metadata_sha256=str(
            target_training_proof["checkpoint_metadata_sha256"]
        ),
    )
    if checkpoint_metadata.get("data_manifest_sha256") != data_manifest_sha256:
        raise RuntimeError("checkpoint/evaluation data manifest mismatch")
    expected_training_authorization = (
        dict(authorization_details["lineage"])
        if eval_set == "trigger" and authorization_details is not None else None
    )
    if eval_set == "trigger" and (
        checkpoint_metadata.get("branch_authorization_lineage")
        != expected_training_authorization
    ):
        raise RuntimeError("trigger evaluation/checkpoint branch authorization mismatch")

    # Contrast PREPARES its durable access identity before output creation,
    # then installs the bound marker and records STARTED before the first
    # operation permitted to decompress sealed payload.  A crash at every
    # boundary is therefore either marker-only recoverable or archive-bound.
    if eval_set == "contrast":
        checkpoint_lineage = {
            "path": _repo_relative(checkpoint),
            "metadata_sha256": checkpoint_sha256,
            "checkpoint_identity_sha256": checkpoint_metadata[
                "checkpoint_identity_sha256"
            ],
        }
        authorized_checkpoint = (
            authorization.get("matching", {})
            .get("per_seed", {})
            .get(str(model_seed), {})
            .get("checkpoint_lineages", {})
            .get(f"{capacity}_joint")
        )
        if authorized_checkpoint != checkpoint_lineage:
            raise RuntimeError(
                "contrast checkpoint is not the exact fixed final authorized by Stage-B seal"
            )
        access_event = record_contrast_access(
            config,
            data_dir,
            manifest,
            authorization=authorization_lineage,
            capacity=capacity,
            objective=objective,
            model_seed=model_seed,
            evaluation_output=output_dir,
            checkpoint_lineage=checkpoint_lineage,
        )
        try:
            attempt_authorization = access_event["attempts"][-1]["authorization"]
            ensure_attempt_output(output_dir, attempt_authorization)
        except (KeyError, IndexError, TypeError, AttemptReceiptError) as exc:
            raise RuntimeError(f"contrast PREPARED output marker failed: {exc}") from exc
        access_event = start_contrast_access(
            config,
            data_dir,
            manifest,
            capacity=capacity,
            objective=objective,
            model_seed=model_seed,
            evaluation_output=output_dir,
        )
        evaluation_rows = validated_data_rows(
            config, data_dir, manifest, tuple(split_names)
        )
    else:
        access_event = None
        _require_new_output(output_dir, directory=True, kind="state evaluation")
        evaluation_rows = validated_data_rows(
            config, data_dir, manifest, tuple(split_names)
        )
    summaries: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats()
    evaluation_started = time.time()
    wrapper.eval()
    for mode in ("intact", "disabled"):
        rows_path = output_dir / f"rows_{mode}.jsonl"
        split_summaries = {}
        context = contextlib.nullcontext() if mode == "intact" else wrapper.adaptation.suspended()
        with context, torch.no_grad():
            for split in split_names:
                rows = evaluation_rows[split]
                by_depth: dict[int, dict[str, float]] = {}
                answer_interface = 0
                answer_correct_total = 0
                for row_index, row in enumerate(rows, start=1):
                    k = int(row["depth"])
                    batch = _encode_row(tokenizer, row, config, k=k, device=torch.device("cuda"))
                    wrapper.adaptation.begin_microbatch(
                        microbatch_dropout_seed(
                            model_seed, row_index, f"{mode}|{row['id']}", k
                        )
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        model_output = _forward(wrapper, batch, k=k, compute_answer=True)
                    call_receipt = wrapper.adaptation.end_microbatch()
                    expected_adaptation_calls = (
                        (k - 1)
                        * int(config["architecture"]["adaptation"][capacity]["expected_targets"])
                        if mode == "intact" else 0
                    )
                    if call_receipt["calls"] != expected_adaptation_calls:
                        raise RuntimeError("evaluation adaptation call count changed")
                    if expected_adaptation_calls and (
                        call_receipt["cycles"] != k - 1
                        or not call_receipt["cycle_order_identical"]
                        or not call_receipt["each_cycle_exact_target_set"]
                    ):
                        raise RuntimeError("evaluation ordered adaptation schedule changed")
                    losses = {
                        "answer_loss": float(model_output.answer_loss.detach().float().cpu()),
                        "state_loss": float(model_output.state_loss.detach().float().cpu()),
                        "fixed_point_loss": float(
                            model_output.fixed_point_loss.detach().float().cpu()
                        ),
                    }
                    if not all(math.isfinite(value) for value in losses.values()):
                        raise RuntimeError("evaluation loss diagnostic is nonfinite")
                    node_predictions = model_output.node_logits[0].argmax(-1).tolist()
                    phase_predictions = model_output.phase_logits[0].argmax(-1).tolist()
                    checksum_predictions = model_output.checksum_logits[0].argmax(-1).tolist()
                    node_targets = batch["state_targets"]["node"][0].tolist()
                    phase_targets = batch["state_targets"]["phase"][0].tolist()
                    checksum_targets = batch["state_targets"]["checksum"][0].tolist()
                    if not all(
                        len(values) == k for values in (
                            node_predictions, phase_predictions, checksum_predictions,
                            node_targets, phase_targets, checksum_targets,
                        )
                    ):
                        raise RuntimeError("state trajectory output length changed")
                    node_step_correct = [a == b for a, b in zip(node_predictions, node_targets)]
                    phase_step_correct = [a == b for a, b in zip(phase_predictions, phase_targets)]
                    checksum_step_correct = [
                        a == b for a, b in zip(checksum_predictions, checksum_targets)
                    ]
                    joint_step_correct = [
                        node and phase and checksum
                        for node, phase, checksum in zip(
                            node_step_correct, phase_step_correct, checksum_step_correct
                        )
                    ]
                    node_prediction = int(node_predictions[-1])
                    phase_prediction = int(phase_predictions[-1])
                    checksum_prediction = int(checksum_predictions[-1])
                    node_target = int(node_targets[-1])
                    phase_target = int(phase_targets[-1])
                    checksum_target = int(checksum_targets[-1])
                    node_correct = node_prediction == node_target
                    phase_correct = phase_prediction == phase_target
                    checksum_correct = checksum_prediction == checksum_target
                    joint_correct = node_correct and phase_correct and checksum_correct
                    answer_prediction, full_top_answer, answer_mass = _choice_prediction(
                        model_output, batch["answer_token_ids"]
                    )
                    answer_correct = answer_prediction == int(row["correct_choice"])
                    answer_interface += full_top_answer
                    answer_correct_total += answer_correct
                    state_change_rms = [
                        float(
                            (right.float() - left.float()).pow(2).mean().sqrt().cpu()
                        )
                        for left, right in zip(model_output.states, model_output.states[1:])
                    ]
                    mean_state_change = (
                        sum(state_change_rms) / len(state_change_rms) if state_change_rms else 0.0
                    )
                    compute = recurrent_compute_receipt(
                        sequence_tokens=batch["prompt_tokens"],
                        total_layers=int(config["architecture"]["expected_num_layers"]),
                        loop_layers=(
                            int(config["architecture"]["loop_end"])
                            - int(config["architecture"]["loop_start"])
                        ),
                        k=k,
                    )
                    adaptation_forward_macs = (
                        batch["prompt_tokens"]
                        * (k - 1)
                        * int(checkpoint_metadata["adaptation_parameters"])
                        if mode == "intact" else 0
                    )
                    cell = by_depth.setdefault(
                        k,
                        {
                            "n": 0, "node": 0, "phase": 0, "checksum": 0, "joint": 0,
                            "trajectory_steps": 0, "node_steps": 0, "phase_steps": 0,
                            "checksum_steps": 0, "joint_steps": 0, "answer": 0,
                            "state_change_sum": 0.0, "layer_token_applications": 0,
                            "adaptation_calls": 0, "adaptation_forward_macs": 0,
                        },
                    )
                    cell["n"] += 1
                    cell["node"] += node_correct
                    cell["phase"] += phase_correct
                    cell["checksum"] += checksum_correct
                    cell["joint"] += joint_correct
                    cell["trajectory_steps"] += k
                    cell["node_steps"] += sum(node_step_correct)
                    cell["phase_steps"] += sum(phase_step_correct)
                    cell["checksum_steps"] += sum(checksum_step_correct)
                    cell["joint_steps"] += sum(joint_step_correct)
                    cell["answer"] += answer_correct
                    cell["state_change_sum"] += mean_state_change
                    cell["layer_token_applications"] += compute.total_layer_token_applications
                    cell["adaptation_calls"] += call_receipt["calls"]
                    cell["adaptation_forward_macs"] += adaptation_forward_macs
                    _append_jsonl(rows_path, {
                        "id": row["id"],
                        "split": split,
                        "depth": k,
                        "family": row["family"],
                        "template": row["template"],
                        "query_kind": row["query_kind"],
                        "capacity": capacity,
                        "objective": objective,
                        "model_seed": model_seed,
                        "adaptation_mode": mode,
                        "node_target": node_target,
                        "phase_target": phase_target,
                        "checksum_target": checksum_target,
                        "node_prediction": node_prediction,
                        "phase_prediction": phase_prediction,
                        "checksum_prediction": checksum_prediction,
                        "node_final_correct": node_correct,
                        "phase_final_correct": phase_correct,
                        "checksum_final_correct": checksum_correct,
                        "joint_final_correct": joint_correct,
                        "node_trajectory_targets": node_targets,
                        "phase_trajectory_targets": phase_targets,
                        "checksum_trajectory_targets": checksum_targets,
                        "node_trajectory_predictions": node_predictions,
                        "phase_trajectory_predictions": phase_predictions,
                        "checksum_trajectory_predictions": checksum_predictions,
                        "node_trajectory_accuracy": sum(node_step_correct) / k,
                        "phase_trajectory_accuracy": sum(phase_step_correct) / k,
                        "checksum_trajectory_accuracy": sum(checksum_step_correct) / k,
                        "joint_trajectory_accuracy": sum(joint_step_correct) / k,
                        "answer_choice_target": int(row["correct_choice"]),
                        "answer_choice_prediction": answer_prediction,
                        "answer_correct": answer_correct,
                        "full_top_is_answer": full_top_answer,
                        "answer_token_mass": answer_mass,
                        "state_change_rms_by_transition": state_change_rms,
                        "mean_state_change_rms": mean_state_change,
                        **losses,
                        "prompt_tokens": batch["prompt_tokens"],
                        "base_layer_token_applications": (
                            compute.base_layer_token_applications
                        ),
                        "extra_loop_layer_token_applications": (
                            compute.recurrent_layer_token_applications
                        ),
                        "total_layer_token_applications": (
                            compute.total_layer_token_applications
                        ),
                        "adaptation_forward_macs": adaptation_forward_macs,
                        "adaptation_calls": call_receipt["calls"],
                        "adaptation_cycles": call_receipt["cycles"],
                        "adaptation_call_manifest_sha256": (
                            call_receipt["call_manifest_sha256"]
                        ),
                        "compute_proxy": (
                            "exact_layer_token_applications_and_adapter_linear_macs;"
                            "not_hardware_flops"
                        ),
                    })
                split_summaries[split] = {
                    "n": len(rows),
                    "answer_interface_rate": answer_interface / len(rows),
                    "answer_accuracy": answer_correct_total / len(rows),
                    "by_depth": {
                        str(depth): {
                            "n": counts["n"],
                            "node_final_accuracy": counts["node"] / counts["n"],
                            "phase_final_accuracy": counts["phase"] / counts["n"],
                            "checksum_final_accuracy": counts["checksum"] / counts["n"],
                            "joint_final_accuracy": counts["joint"] / counts["n"],
                            "node_trajectory_accuracy": (
                                counts["node_steps"] / counts["trajectory_steps"]
                            ),
                            "phase_trajectory_accuracy": (
                                counts["phase_steps"] / counts["trajectory_steps"]
                            ),
                            "checksum_trajectory_accuracy": (
                                counts["checksum_steps"] / counts["trajectory_steps"]
                            ),
                            "joint_trajectory_accuracy": (
                                counts["joint_steps"] / counts["trajectory_steps"]
                            ),
                            "answer_accuracy": counts["answer"] / counts["n"],
                            "mean_state_change_rms": (
                                counts["state_change_sum"] / counts["n"]
                            ),
                            "total_layer_token_applications": int(
                                counts["layer_token_applications"]
                            ),
                            "adaptation_calls": int(counts["adaptation_calls"]),
                            "adaptation_forward_macs": int(
                                counts["adaptation_forward_macs"]
                            ),
                        }
                        for depth, counts in sorted(by_depth.items())
                    },
                }
        summaries[mode] = {
            "rows_path": rows_path.name,
            "rows_sha256": _sha256(rows_path),
            "rows": _metrics_line_count(rows_path),
            "splits": split_summaries,
        }
    for mode in ("intact", "disabled"):
        _fsync_file(output_dir / f"rows_{mode}.jsonl")
    _fsync_directory(output_dir)
    parity_row = evaluation_rows[split_names[0]][0]
    parity_batch = _encode_row(tokenizer, parity_row, config, k=1, device=torch.device("cuda"))
    wrapper.adaptation.reset_call_count()
    parity_error = _k1_parity(wrapper, parity_batch)
    parity_calls = wrapper.adaptation.active_call_count
    if (
        parity_error > float(config["gates"]["k1_max_logit_abs_error"])
        or parity_calls != 0
    ):
        raise RuntimeError(f"post-training K=1 parity failed: {parity_error}")
    torch.cuda.synchronize()
    summary = _with_identity({
        "schema_version": 1,
        "status": "STATE_EVALUATION_COMPLETE",
        **_identity(config, phase=f"{capacity}_{objective}_{eval_set}_evaluation"),
        "capacity": capacity,
        "objective": objective,
        "model_seed": model_seed,
        "eval_set": eval_set,
        "checkpoint_path": _repo_relative(checkpoint),
        "checkpoint_metadata_sha256": checkpoint_sha256,
        "checkpoint_identity_sha256": checkpoint_metadata["checkpoint_identity_sha256"],
        "data_manifest_sha256": data_manifest_sha256,
        "setup_barrier": setup_barrier_proof,
        "training_barrier": training_barrier_proof,
        "target_training_cell_proof": target_training_proof,
        "trigger_launch_preflight": trigger_launch_preflight,
        "split_payloads": {
            split: {
                "sha256": manifest["files"][split]["sha256"],
                "canonical_rows": manifest["files"][split]["canonical_rows"],
            }
            for split in split_names
        },
        "contrast_authorization": authorization_lineage,
        "training_branch_authorization": (
            expected_training_authorization if eval_set == "trigger"
            else checkpoint_metadata.get("branch_authorization_lineage")
        ),
        "contrast_access_event": access_event,
        "authorizes_training": False,
        "authorizes_result_training": False,
        "authorizes_result_evaluation": False,
        "benchmark_files_read": 0,
        "result_payloads_opened": (
            list(split_names) if eval_set == "trigger" else []
        ),
        "sealed_contrast_payloads_opened": (
            list(split_names) if eval_set == "contrast" else []
        ),
        "training_or_evaluation_started": True,
        "scientific_evidence": True,
        "k1_max_logit_abs_error": parity_error,
        "k1_adaptation_calls": parity_calls,
        "evaluation_elapsed_seconds": time.time() - evaluation_started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
        "modes": summaries,
    })
    _write_json(output_dir / "summary.json", summary)
