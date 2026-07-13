"""Content-addressed shared loop-state initialization bundles."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator, Mapping

import torch

from .config import (
    MODEL_ID,
    MODEL_REVISION,
    config_sha256,
    requirements_training_lock_bytes,
    source_contract_sha256,
)
from .design_boundary import design_lineage, validate_design_receipt
from .safe_io import (
    StableArtifactError,
    open_stable_regular,
    publish_new_bytes,
    publish_new_file,
    read_stable_bytes,
)
from .state_loop_model import LowRankStateAdapter, SinusoidalStepEncoder, StateSufficiencyHeads


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
REQUIREMENTS_LOCK = REPO_ROOT / "requirements-training.lock.txt"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _exact_value(actual: Any, expected: Any) -> bool:
    """JSON-like equality that never conflates booleans with integers."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_value(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _file_sha256(path: Path) -> str:
    lexical = _canonical_workspace_path(path, label="hashed initialization artifact")
    return hashlib.sha256(read_stable_bytes(REPO_ROOT, lexical)).hexdigest()


def _requirements_sha256() -> str:
    return hashlib.sha256(requirements_training_lock_bytes()).hexdigest()


def _canonical_workspace_path(
    value: str | os.PathLike[str], *, label: str
) -> Path:
    """Return one lexical workspace path without resolving aliases or symlinks."""

    raw = os.fspath(value)
    candidate = Path(raw)
    if not raw or "\x00" in raw or "\\" in raw:
        raise RuntimeError(f"{label} is not a canonical lexical path")
    if candidate.is_absolute():
        lexical = Path(os.path.abspath(raw))
        if raw.startswith("//") or raw != lexical.as_posix():
            raise RuntimeError(f"{label} is not a canonical lexical path")
    else:
        if (
            raw != candidate.as_posix()
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise RuntimeError(f"{label} is not a canonical lexical path")
        lexical = Path(os.path.abspath(raw))
    try:
        relative = lexical.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"{label} is outside repository workspace: {lexical}") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"{label} is not a canonical workspace artifact")
    return lexical


def _repo_relative(path: Path) -> str:
    resolved = _canonical_workspace_path(path, label="initialization lineage path")
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"initialization path is outside repository workspace: {resolved}") from exc


def _canonical_bundle_path(config: Mapping[str, Any], model_seed: int) -> Path:
    return Path(os.path.abspath(
        ROOT
        / str(config["paths"]["large_artifacts_dir"])
        / f"initialization_seed{int(model_seed)}.pt"
    ))


def _tracked_receipt_path(model_seed: int) -> Path:
    return Path(os.path.abspath(
        ROOT / "runs" / "setup" / f"initialization_seed{int(model_seed)}.json"
    ))


def _directory_flags() -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


@contextmanager
def _open_canonical_parent(path: Path) -> Iterator[tuple[int, str]]:
    """Open/create a destination parent descriptor below the trusted repo root."""

    path = _canonical_workspace_path(path, label="initialization publication path")
    relative = path.relative_to(REPO_ROOT)
    descriptors: list[int] = []
    try:
        current = os.open(REPO_ROOT, _directory_flags())
        descriptors.append(current)
        for component in relative.parts[:-1]:
            try:
                following = os.open(component, _directory_flags(), dir_fd=current)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=current)
                    os.fsync(current)
                except FileExistsError:
                    pass
                following = os.open(component, _directory_flags(), dir_fd=current)
            info = os.fstat(following)
            if not stat.S_ISDIR(info.st_mode):
                os.close(following)
                raise RuntimeError("initialization publication parent is not a directory")
            descriptors.append(following)
            current = following
        yield current, relative.parts[-1]
    except OSError as exc:
        raise RuntimeError(
            f"initialization publication path cannot be opened without aliases: {path}"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _publish_new(
    path: Path,
    writer: Callable[[BinaryIO], None],
) -> None:
    """Fsync and atomically no-clobber-install one new single-link inode."""

    path = _canonical_workspace_path(path, label="initialization publication path")
    # Parent creation is descriptor-relative and no-follow.  Publication then
    # reopens the complete chain and holds it through RENAME_NOREPLACE.
    with _open_canonical_parent(path):
        pass
    try:
        publish_new_file(REPO_ROOT, path, writer, mode=0o600)
    except StableArtifactError as exc:
        if os.path.lexists(path):
            raise RuntimeError(
                f"refusing to overwrite initialization artifact: {path}"
            ) from exc
        raise RuntimeError(
            f"initialization publication could not be committed safely: {path}"
        ) from exc


def _publish_bytes(path: Path, encoded: bytes) -> None:
    path = _canonical_workspace_path(path, label="initialization publication path")
    with _open_canonical_parent(path):
        pass
    try:
        publish_new_bytes(REPO_ROOT, path, encoded, mode=0o600)
    except StableArtifactError as exc:
        if os.path.lexists(path):
            raise RuntimeError(
                f"refusing to overwrite initialization artifact: {path}"
            ) from exc
        raise RuntimeError(
            f"initialization publication could not be committed safely: {path}"
        ) from exc


def _publication_checkpoint(stage: str) -> None:
    """Test seam immediately after each durable publication boundary."""

    del stage


def initialization_seed(model_seed: int) -> int:
    payload = f"shared-loop-state-init-v1|{int(model_seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _flatten(state: Mapping[str, Any], prefix: str = "") -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_flatten(value, name))
        elif isinstance(value, torch.Tensor):
            result[name] = value.detach().cpu().contiguous()
        else:
            raise TypeError(f"non-tensor initialization value at {name}: {type(value)}")
    return result


def tensor_manifest(state: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    flat = _flatten(state)
    digest = hashlib.sha256()
    manifest = []
    for name in sorted(flat):
        tensor = flat[name]
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        item = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        manifest.append(item)
        digest.update(name.encode("utf-8"))
        digest.update(item["dtype"].encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(raw)
    return manifest, digest.hexdigest()


def build_shared_state(config: Mapping[str, Any], model_seed: int) -> dict[str, Any]:
    arch = config["architecture"]
    substrate = config["substrate"]
    hidden_size = int(arch["expected_hidden_size"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed(model_seed))
        initializer = LowRankStateAdapter(hidden_size, int(arch["state_adapter_rank"]))
        step_encoder = SinusoidalStepEncoder(int(arch["step_encoding_dim"]), hidden_size)
        sufficiency = StateSufficiencyHeads(
            hidden_size,
            int(substrate["node_count"]),
            int(substrate["checksum_modulus"]),
        )
    damping = float(arch["damping_initial"])
    aggregate = float(arch["aggregate_last_initial"])
    return {
        "state_initializer": initializer.state_dict(),
        "step_encoder": step_encoder.state_dict(),
        "sufficiency": sufficiency.state_dict(),
        "scalars": {
            "damping_logit": torch.tensor(torch.logit(torch.tensor(damping)).item()),
            "aggregate_logit": torch.tensor(torch.logit(torch.tensor(aggregate)).item()),
        },
    }


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _inspect_bundle(
    path: Path,
    expected_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate and hash one existing bundle from the same stable descriptor."""

    digest = hashlib.sha256()
    try:
        with open_stable_regular(REPO_ROOT, path) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu", weights_only=True)
    except StableArtifactError as exc:
        raise RuntimeError(f"initialization bundle is not a stable canonical file: {path}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"metadata", "state"}:
        raise RuntimeError("initialization bundle payload is malformed")
    metadata = payload.get("metadata")
    state = payload.get("state")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        raise RuntimeError("initialization bundle payload is malformed")
    if not _exact_value(dict(metadata), dict(expected_metadata)):
        raise RuntimeError("initialization durable prefix metadata mismatch")
    manifest, value_digest = tensor_manifest(state)
    if (
        not _exact_value(manifest, expected_metadata.get("tensor_manifest"))
        or value_digest != expected_metadata.get("tensor_values_sha256")
    ):
        raise RuntimeError("initialization durable prefix tensor digest mismatch")
    return dict(state), digest.hexdigest()


def _receipt_payload(
    *,
    metadata: Mapping[str, Any],
    output: Path,
    bundle_sha256: str,
) -> dict[str, Any]:
    receipt = {
        "schema_version": 1,
        "status": "SHARED_INITIALIZATION_PREPARED",
        "phase": "shared_initialization",
        "metadata": dict(metadata),
        "bundle_path": _repo_relative(output),
        "bundle_sha256": bundle_sha256,
    }
    receipt["receipt_identity_sha256"] = _canonical_sha256(receipt)
    return receipt


def prepare_initialization_bundle(
    config: Mapping[str, Any], model_seed: int, output: Path
) -> dict[str, Any]:
    validate_design_receipt(config)
    if int(model_seed) not in set(map(int, config["training"]["train_seeds"])):
        raise RuntimeError("initialization seed is not preregistered")
    output = _canonical_workspace_path(output, label="initialization output")
    tracked_receipt = _tracked_receipt_path(model_seed)
    is_canonical = output == _canonical_bundle_path(config, model_seed)
    receipt_path = output.with_suffix(output.suffix + ".json")
    bundle_initially_present = os.path.lexists(output)
    sidecar_initially_present = os.path.lexists(receipt_path)
    tracked_initially_present = is_canonical and os.path.lexists(tracked_receipt)
    if sidecar_initially_present and not bundle_initially_present:
        raise RuntimeError("initialization durable prefix has a sidecar without its bundle")
    if tracked_initially_present and not sidecar_initially_present:
        raise RuntimeError("initialization durable prefix has a commit marker without its sidecar")

    state = build_shared_state(config, model_seed)
    manifest, value_digest = tensor_manifest(state)
    metadata = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "model_seed": int(model_seed),
        "initialization_seed": initialization_seed(model_seed),
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(),
        "requirements_training_lock_sha256": _requirements_sha256(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "design_lineage": design_lineage(config),
        "tensor_manifest": manifest,
        "tensor_values_sha256": value_digest,
    }
    metadata["receipt_identity_sha256"] = _canonical_sha256(metadata)

    if not bundle_initially_present:
        _publish_new(
            output,
            lambda handle: torch.save({"metadata": metadata, "state": state}, handle),
        )
        _publication_checkpoint("bundle_installed")
    _, bundle_sha256 = _inspect_bundle(output, metadata)
    receipt = _receipt_payload(
        metadata=metadata,
        output=output,
        bundle_sha256=bundle_sha256,
    )
    encoded_receipt = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    if sidecar_initially_present:
        if read_stable_bytes(REPO_ROOT, receipt_path) != encoded_receipt:
            raise RuntimeError("initialization durable prefix sidecar mismatch")
    else:
        _publish_bytes(receipt_path, encoded_receipt)
        _publication_checkpoint("sidecar_installed")

    if is_canonical:
        if tracked_initially_present:
            if read_stable_bytes(REPO_ROOT, tracked_receipt) != encoded_receipt:
                raise RuntimeError("initialization durable commit marker mismatch")
            raise RuntimeError(f"refusing to overwrite initialization bundle: {output}")
        _publish_bytes(tracked_receipt, encoded_receipt)
        _publication_checkpoint("tracked_mirror_installed")
    elif sidecar_initially_present:
        raise RuntimeError(f"refusing to overwrite initialization bundle: {output}")
    return receipt


def load_initialization_bundle(
    config: Mapping[str, Any], model_seed: int, path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_design_receipt(config)
    if int(model_seed) not in set(map(int, config["training"]["train_seeds"])):
        raise RuntimeError("initialization seed is not preregistered")
    path = _canonical_workspace_path(path, label="initialization bundle path")
    receipt_path = path.with_suffix(path.suffix + ".json")
    receipt_bytes = read_stable_bytes(REPO_ROOT, receipt_path)
    receipt = _strict_json_object(receipt_bytes, label="initialization sidecar")
    if set(receipt) != {
        "schema_version",
        "status",
        "phase",
        "metadata",
        "bundle_path",
        "bundle_sha256",
        "receipt_identity_sha256",
    }:
        raise RuntimeError("initialization sidecar fields changed")
    if path == _canonical_bundle_path(config, model_seed):
        tracked_receipt = _tracked_receipt_path(model_seed)
        tracked_bytes = read_stable_bytes(REPO_ROOT, tracked_receipt)
        if tracked_bytes != receipt_bytes:
            raise RuntimeError("tracked initialization receipt mirror is missing or changed")
    receipt_identity = receipt.get("receipt_identity_sha256")
    receipt_payload = {
        key: value for key, value in receipt.items() if key != "receipt_identity_sha256"
    }
    if receipt_identity != _canonical_sha256(receipt_payload):
        raise RuntimeError("initialization sidecar identity mismatch")
    expected_sidecar = {
        "schema_version": 1,
        "status": "SHARED_INITIALIZATION_PREPARED",
        "phase": "shared_initialization",
        "bundle_path": _repo_relative(path),
        "bundle_sha256": receipt.get("bundle_sha256"),
    }
    for key, value in expected_sidecar.items():
        if not _exact_value(receipt.get(key), value):
            raise RuntimeError(f"initialization sidecar {key} mismatch")
    with open_stable_regular(
        REPO_ROOT,
        path,
        expected_sha256=str(receipt["bundle_sha256"]),
    ) as bundle_handle:
        payload = torch.load(bundle_handle, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {"metadata", "state"}:
        raise RuntimeError("initialization bundle payload is malformed")
    metadata = payload.get("metadata")
    state = payload.get("state")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        raise RuntimeError("initialization bundle payload is malformed")
    if set(metadata) != {
        "schema_version",
        "experiment_id",
        "model_seed",
        "initialization_seed",
        "config_sha256",
        "source_contract_sha256",
        "requirements_training_lock_sha256",
        "model_id",
        "model_revision",
        "design_lineage",
        "tensor_manifest",
        "tensor_values_sha256",
        "receipt_identity_sha256",
    }:
        raise RuntimeError("initialization bundle metadata fields changed")
    expected = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "model_seed": int(model_seed),
        "initialization_seed": initialization_seed(model_seed),
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(),
        "requirements_training_lock_sha256": _requirements_sha256(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "design_lineage": design_lineage(config),
    }
    for key, value in expected.items():
        if not _exact_value(metadata.get(key), value):
            raise RuntimeError(f"initialization bundle {key} mismatch")
    identity_payload = {key: value for key, value in metadata.items() if key != "receipt_identity_sha256"}
    if metadata.get("receipt_identity_sha256") != _canonical_sha256(identity_payload):
        raise RuntimeError("initialization bundle receipt identity mismatch")
    manifest, digest = tensor_manifest(state)
    if (
        not _exact_value(manifest, metadata.get("tensor_manifest"))
        or digest != metadata.get("tensor_values_sha256")
    ):
        raise RuntimeError("initialization bundle tensor digest mismatch")
    if not _exact_value(receipt.get("metadata"), metadata):
        raise RuntimeError("initialization sidecar/internal metadata mismatch")
    return dict(state), receipt
