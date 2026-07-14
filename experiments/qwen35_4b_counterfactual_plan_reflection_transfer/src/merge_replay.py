"""Exact replay proving a merged checkpoint equals pinned base plus retained LoRA."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

from checkpoint_lineage import (
    _read_safetensors_header,
    canonical_sha256,
    load_pinned_structure,
    sha256_file,
)


ADAPTER_PREFIX = "base_model.model.model.layers."
BASE_PREFIX = "model.language_model.layers."


def _tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _adapter_model_key(adapter_key: str) -> str:
    if not adapter_key.startswith(ADAPTER_PREFIX) or not adapter_key.endswith(
        ".lora_A.weight"
    ):
        raise ValueError(f"unexpected retained adapter key: {adapter_key}")
    core = adapter_key[len(ADAPTER_PREFIX) : -len(".lora_A.weight")]
    return f"{BASE_PREFIX}{core}.weight"


def _target_module_keys(keys: set[str], target_modules: set[str]) -> set[str]:
    return {
        key
        for key in keys
        if key.startswith(BASE_PREFIX)
        and key.endswith(".weight")
        and key.rsplit(".", 2)[-2] in target_modules
    }


def verify_tensor_equations(
    *,
    base: Mapping[str, Any],
    merged: Mapping[str, Any],
    adapter: Mapping[str, Any],
    target_modules: set[str],
    scale: float,
) -> dict[str, Any]:
    """Compare every tensor exactly and replay every LoRA update in float32."""
    import torch

    base_keys = set(base)
    merged_keys = set(merged)
    adapter_keys = set(adapter)
    if base_keys != merged_keys:
        raise ValueError("merged tensor keys differ from the pinned base tensor keys")
    a_keys = {key for key in adapter_keys if key.endswith(".lora_A.weight")}
    b_keys = {key for key in adapter_keys if key.endswith(".lora_B.weight")}
    if (
        not a_keys
        or {key.replace(".lora_A.", ".lora_B.") for key in a_keys} != b_keys
        or adapter_keys != a_keys | b_keys
    ):
        raise ValueError("retained adapter tensor set is not exact LoRA A/B pairs")
    adapter_by_model = {_adapter_model_key(key): key for key in a_keys}
    expected_adapted = _target_module_keys(base_keys, target_modules)
    if set(adapter_by_model) != expected_adapted:
        raise ValueError("retained adapter modules differ from the pinned target-module set")
    adapted_hashes: dict[str, str] = {}
    unchanged_digest = hashlib.sha256()
    for key in sorted(base_keys):
        base_tensor = base[key]
        merged_tensor = merged[key]
        if base_tensor.shape != merged_tensor.shape or base_tensor.dtype != merged_tensor.dtype:
            raise ValueError(f"merged tensor shape/dtype differs from pinned base at {key}")
        if key in adapter_by_model:
            key_a = adapter_by_model[key]
            key_b = key_a.replace(".lora_A.", ".lora_B.")
            tensor_a = adapter[key_a]
            tensor_b = adapter[key_b]
            if (
                base_tensor.ndim != 2
                or tensor_a.ndim != 2
                or tensor_b.ndim != 2
                or tensor_a.shape[0] != tensor_b.shape[1]
                or tensor_a.shape[1] != base_tensor.shape[1]
                or tensor_b.shape[0] != base_tensor.shape[0]
            ):
                raise ValueError(f"LoRA tensor shapes do not compose with pinned base at {key}")
            expected = (
                base_tensor.float() + (tensor_b.float() @ tensor_a.float()) * scale
            ).to(base_tensor.dtype)
            if _tensor_sha256(expected) != _tensor_sha256(merged_tensor):
                raise ValueError(f"merged tensor does not equal pinned base plus LoRA at {key}")
            adapted_hashes[key] = _tensor_sha256(merged_tensor)
            del tensor_a, tensor_b, expected
        else:
            if _tensor_sha256(base_tensor) != _tensor_sha256(merged_tensor):
                raise ValueError(f"unmodified merged tensor differs from pinned base at {key}")
            unchanged_digest.update(key.encode())
            unchanged_digest.update(b"\0")
            unchanged_digest.update(bytes.fromhex(_tensor_sha256(merged_tensor)))
        del base_tensor, merged_tensor
    return {
        "equation": "merged == base_dtype(base.float32 + (B.float32 @ A.float32) * alpha/r)",
        "checked_tensor_count": len(base_keys),
        "adapted_module_count": len(adapted_hashes),
        "unchanged_tensor_count": len(base_keys) - len(adapted_hashes),
        "adapted_tensor_sha256": adapted_hashes,
        "unchanged_tensor_aggregate_sha256": unchanged_digest.hexdigest(),
    }


class _SafeTensorMapping(Mapping[str, Any]):
    def __init__(self, root: Path, weight_map: dict[str, str], stack: ExitStack):
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise ValueError("safetensors is required for exact merge replay") from error
        self._weight_map = dict(weight_map)
        self._handles = {
            name: stack.enter_context(
                safe_open(str(root / name), framework="pt", device="cpu")
            )
            for name in sorted(set(weight_map.values()))
        }

    def __iter__(self):
        return iter(self._weight_map)

    def __len__(self) -> int:
        return len(self._weight_map)

    def __getitem__(self, key: str) -> Any:
        return self._handles[self._weight_map[key]].get_tensor(key)


class _AdapterMapping(Mapping[str, Any]):
    def __init__(self, path: Path, stack: ExitStack):
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise ValueError("safetensors is required for exact merge replay") from error
        self._handle = stack.enter_context(
            safe_open(str(path), framework="pt", device="cpu")
        )
        self._keys = tuple(self._handle.keys())

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, key: str) -> Any:
        return self._handle.get_tensor(key)


def _base_snapshot(structure: dict[str, Any]) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ValueError("huggingface_hub is required to locate pinned base tensors") from error
    return Path(
        snapshot_download(
            repo_id=structure["model_id"],
            revision=structure["model_revision"],
            allow_patterns=["config.json", "model.safetensors.index.json", "*.safetensors"],
            local_files_only=True,
        )
    )


def authenticate_base_snapshot(
    base_snapshot: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Locate and fully authenticate the exact cached source checkpoint."""
    structure = load_pinned_structure()
    base_root = _base_snapshot(structure) if base_snapshot is None else base_snapshot
    base_config = base_root / "config.json"
    base_index_path = base_root / "model.safetensors.index.json"
    if (
        sha256_file(base_config) != structure["base_config_sha256"]
        or sha256_file(base_index_path) != structure["base_index_sha256"]
    ):
        raise ValueError("cached base config/index differs from the pinned revision")
    base_index = json.loads(base_index_path.read_text())
    if set(base_index) != {"metadata", "weight_map"}:
        raise ValueError("cached base weight-index schema changed")
    base_shards = set(base_index["weight_map"].values())
    if base_shards != set(structure["source_shard_sha256"]):
        raise ValueError("cached base shard names differ from the pinned revision")
    base_header_hashes: dict[str, str] = {}
    for name in sorted(base_shards):
        shard = base_root / name
        if (
            shard.stat().st_size != int(structure["source_shard_size"][name])
            or sha256_file(shard) != structure["source_shard_sha256"][name]
        ):
            raise ValueError("cached base shard bytes differ from the pinned revision")
        _tensors, details = _read_safetensors_header(shard)
        base_header_hashes[name] = details["header_sha256"]
    if base_header_hashes != structure["source_shard_header_sha256"]:
        raise ValueError("cached base shard headers differ from the pinned revision")
    return base_root, base_index, structure


def base_snapshot_commitment(base_snapshot: Path | None = None) -> dict[str, Any]:
    """Return the exact content commitment after authenticating all base bytes."""
    _root, _index, structure = authenticate_base_snapshot(base_snapshot)
    return {
        "schema_version": 1,
        "model_id": structure["model_id"],
        "model_revision": structure["model_revision"],
        "config_sha256": structure["base_config_sha256"],
        "index_sha256": structure["base_index_sha256"],
        "shard_sha256": structure["source_shard_sha256"],
        "shard_size": structure["source_shard_size"],
    }


def verify_merge_equation(
    *,
    merged_path: Path,
    adapter_tree: Path,
    recipe: dict[str, Any],
    base_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Authenticate pinned base files and exactly replay the entire merge."""
    base_root, base_index, structure = authenticate_base_snapshot(base_snapshot)
    merged_index = json.loads(
        (merged_path / "model.safetensors.index.json").read_text()
    )
    if set(merged_index) != {
        "metadata", "weight_map"
    }:
        raise ValueError("merged weight-index schema changed")
    adapter_path = adapter_tree / "adapter_model.safetensors"
    with ExitStack() as stack:
        result = verify_tensor_equations(
            base=_SafeTensorMapping(base_root, base_index["weight_map"], stack),
            merged=_SafeTensorMapping(merged_path, merged_index["weight_map"], stack),
            adapter=_AdapterMapping(adapter_path, stack),
            target_modules=set(recipe["target_modules"]),
            scale=float(recipe["lora_alpha"]) / float(recipe["lora_rank"]),
        )
    return {
        "schema_version": 1,
        "model_id": structure["model_id"],
        "model_revision": structure["model_revision"],
        "base_config_sha256": structure["base_config_sha256"],
        "base_index_sha256": structure["base_index_sha256"],
        "base_shard_sha256": structure["source_shard_sha256"],
        "adapter_sha256": sha256_file(adapter_path),
        "merged_index_sha256": sha256_file(
            merged_path / "model.safetensors.index.json"
        ),
        "recipe_sha256": canonical_sha256(recipe),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        **result,
    }
