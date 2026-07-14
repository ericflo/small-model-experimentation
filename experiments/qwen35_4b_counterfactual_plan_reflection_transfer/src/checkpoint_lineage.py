"""Static inventories for authentic merged Qwen checkpoints and retained LoRA sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MIN_MERGED_SHARD_BYTES = 5_000_000_000
MIN_MERGED_TENSOR_BYTES = 5_000_000_000
MIN_MERGED_TENSORS = 100
MIN_ADAPTER_BYTES = 1_000_000


def _safetensor_keys(path: Path) -> set[str]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ValueError("safetensors is required to authenticate checkpoint tensors") from error
    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        return set(tensors.keys())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merged_checkpoint_inventory(path: Path) -> dict[str, Any]:
    required_assets = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    }
    missing = sorted(name for name in required_assets if not (path / name).is_file())
    if missing:
        raise ValueError(f"merged checkpoint lacks required model/tokenizer assets: {missing}")
    config = json.loads((path / "config.json").read_text())
    if not isinstance(config, dict) or not isinstance(config.get("architectures"), list):
        raise ValueError("merged checkpoint config lacks an architecture identity")
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    if set(index) != {"metadata", "weight_map"}:
        raise ValueError("merged checkpoint weight index schema changed")
    weight_map = index["weight_map"]
    metadata = index["metadata"]
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or not isinstance(metadata, dict)
        or not isinstance(metadata.get("total_size"), int)
    ):
        raise ValueError("merged checkpoint weight index is incomplete")
    referenced = set(weight_map.values())
    if any(
        not isinstance(name, str)
        or "/" in name
        or not name.startswith("model-")
        or not name.endswith(".safetensors")
        for name in referenced
    ):
        raise ValueError("merged checkpoint index has an unexpected shard layout")
    observed_shards = {item.name for item in path.glob("model-*.safetensors")}
    if referenced != observed_shards:
        raise ValueError("merged checkpoint shard set differs from its weight index")
    observed_keys: set[str] = set()
    for shard_name in sorted(referenced):
        shard_path = path / shard_name
        if not shard_path.is_file():
            raise ValueError("merged checkpoint references a missing shard")
        keys = _safetensor_keys(shard_path)
        if observed_keys & keys:
            raise ValueError("merged checkpoint repeats tensor keys across shards")
        if any(weight_map.get(key) != shard_name for key in keys):
            raise ValueError("merged checkpoint tensor placement differs from its index")
        observed_keys.update(keys)
    if observed_keys != set(weight_map):
        raise ValueError("merged checkpoint tensor keys differ from its weight index")
    shard_file_bytes = sum((path / name).stat().st_size for name in referenced)
    tensor_bytes = int(metadata["total_size"])
    if (
        len(observed_keys) < MIN_MERGED_TENSORS
        or shard_file_bytes < MIN_MERGED_SHARD_BYTES
        or tensor_bytes < MIN_MERGED_TENSOR_BYTES
    ):
        raise ValueError("merged checkpoint is too small to be the pinned 4B composite model")
    return {
        "weight_index_sha256": sha256_file(index_path),
        "config_sha256": sha256_file(path / "config.json"),
        "tokenizer_json_sha256": sha256_file(path / "tokenizer.json"),
        "shard_count": len(referenced),
        "tensor_count": len(observed_keys),
        "shard_file_bytes": shard_file_bytes,
        "indexed_tensor_bytes": tensor_bytes,
    }


def adapter_tensor_inventory(adapter_path: Path) -> dict[str, Any]:
    if not adapter_path.is_file() or adapter_path.stat().st_size < MIN_ADAPTER_BYTES:
        raise ValueError("retained source adapter tensor file is absent or implausibly small")
    keys = _safetensor_keys(adapter_path)
    a_keys = {key for key in keys if key.endswith(".lora_A.weight")}
    b_keys = {key for key in keys if key.endswith(".lora_B.weight")}
    expected_b = {key.replace(".lora_A.", ".lora_B.") for key in a_keys}
    if not a_keys or expected_b != b_keys or keys != a_keys | b_keys:
        raise ValueError("retained source adapter is not an exact LoRA A/B tensor set")
    return {
        "sha256": sha256_file(adapter_path),
        "file_bytes": adapter_path.stat().st_size,
        "tensor_count": len(keys),
        "module_count": len(a_keys),
    }
