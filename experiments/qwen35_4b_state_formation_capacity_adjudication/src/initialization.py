"""Content-addressed shared loop-state initialization bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import MODEL_ID, MODEL_REVISION, config_sha256, source_contract_sha256
from .design_boundary import design_lineage, validate_design_receipt
from .state_loop_model import LowRankStateAdapter, SinusoidalStepEncoder, StateSufficiencyHeads


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
REQUIREMENTS_LOCK = REPO_ROOT / "requirements-training.lock.txt"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"initialization path is outside repository workspace: {resolved}") from exc


def _canonical_bundle_path(config: Mapping[str, Any], model_seed: int) -> Path:
    return (
        ROOT
        / str(config["paths"]["large_artifacts_dir"])
        / f"initialization_seed{int(model_seed)}.pt"
    ).resolve()


def _tracked_receipt_path(model_seed: int) -> Path:
    return (ROOT / "runs" / "setup" / f"initialization_seed{int(model_seed)}.json").resolve()


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


def prepare_initialization_bundle(
    config: Mapping[str, Any], model_seed: int, output: Path
) -> dict[str, Any]:
    validate_design_receipt(config)
    if int(model_seed) not in set(map(int, config["training"]["train_seeds"])):
        raise RuntimeError("initialization seed is not preregistered")
    output = output.resolve()
    tracked_receipt = _tracked_receipt_path(model_seed)
    is_canonical = output == _canonical_bundle_path(config, model_seed)
    if (
        output.exists()
        or output.with_suffix(output.suffix + ".json").exists()
        or (is_canonical and tracked_receipt.exists())
    ):
        raise RuntimeError(f"refusing to overwrite initialization bundle: {output}")
    state = build_shared_state(config, model_seed)
    manifest, value_digest = tensor_manifest(state)
    metadata = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "model_seed": int(model_seed),
        "initialization_seed": initialization_seed(model_seed),
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(),
        "requirements_training_lock_sha256": _file_sha256(REQUIREMENTS_LOCK),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "design_lineage": design_lineage(config),
        "tensor_manifest": manifest,
        "tensor_values_sha256": value_digest,
    }
    metadata["receipt_identity_sha256"] = _canonical_sha256(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "state": state}, output)
    receipt = {
        "schema_version": 1,
        "status": "SHARED_INITIALIZATION_PREPARED",
        "phase": "shared_initialization",
        "metadata": metadata,
        "bundle_path": _repo_relative(output),
        "bundle_sha256": _file_sha256(output),
    }
    receipt["receipt_identity_sha256"] = _canonical_sha256(receipt)
    receipt_path = output.with_suffix(output.suffix + ".json")
    encoded_receipt = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    receipt_path.write_text(encoded_receipt, encoding="utf-8")
    if is_canonical:
        tracked_receipt.parent.mkdir(parents=True, exist_ok=True)
        tracked_receipt.write_text(encoded_receipt, encoding="utf-8")
    return receipt


def load_initialization_bundle(
    config: Mapping[str, Any], model_seed: int, path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_design_receipt(config)
    if int(model_seed) not in set(map(int, config["training"]["train_seeds"])):
        raise RuntimeError("initialization seed is not preregistered")
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"initialization bundle is missing: {path}")
    receipt_path = path.with_suffix(path.suffix + ".json")
    if not receipt_path.is_file():
        raise RuntimeError(f"initialization sidecar is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if path == _canonical_bundle_path(config, model_seed):
        tracked_receipt = _tracked_receipt_path(model_seed)
        if (
            not tracked_receipt.is_file()
            or tracked_receipt.read_bytes() != receipt_path.read_bytes()
        ):
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
        "bundle_sha256": _file_sha256(path),
    }
    for key, value in expected_sidecar.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"initialization sidecar {key} mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata")
    state = payload.get("state")
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping):
        raise RuntimeError("initialization bundle payload is malformed")
    expected = {
        "experiment_id": config["experiment_id"],
        "model_seed": int(model_seed),
        "initialization_seed": initialization_seed(model_seed),
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(),
        "requirements_training_lock_sha256": _file_sha256(REQUIREMENTS_LOCK),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "design_lineage": design_lineage(config),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"initialization bundle {key} mismatch")
    identity_payload = {key: value for key, value in metadata.items() if key != "receipt_identity_sha256"}
    if metadata.get("receipt_identity_sha256") != _canonical_sha256(identity_payload):
        raise RuntimeError("initialization bundle receipt identity mismatch")
    manifest, digest = tensor_manifest(state)
    if manifest != metadata.get("tensor_manifest") or digest != metadata.get("tensor_values_sha256"):
        raise RuntimeError("initialization bundle tensor digest mismatch")
    if receipt.get("metadata") != metadata:
        raise RuntimeError("initialization sidecar/internal metadata mismatch")
    return dict(state), receipt
