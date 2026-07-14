"""Static inventories for authentic merged Qwen checkpoints and retained LoRA sources."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Any


STRUCTURE_PATH = Path(__file__).resolve().parents[1] / "configs" / "pinned_model_structure.json"
MIN_ADAPTER_BYTES = 1_000_000
MIN_PHYSICAL_ALLOCATION_FRACTION = 0.99
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}
PINNED_CONFIG_KEYS = {
    "architectures",
    "image_token_id",
    "model_type",
    "text_config",
    "tie_word_embeddings",
    "video_token_id",
    "vision_config",
    "vision_end_token_id",
    "vision_start_token_id",
}
PINNED_TEXT_CONFIG_KEYS = {
    "attention_bias", "attention_dropout", "attn_output_gate", "dtype", "eos_token_id",
    "full_attention_interval", "head_dim", "hidden_act", "hidden_size",
    "initializer_range", "intermediate_size", "layer_types", "linear_conv_kernel_dim",
    "linear_key_head_dim", "linear_num_key_heads", "linear_num_value_heads",
    "linear_value_head_dim", "mamba_ssm_dtype", "max_position_embeddings",
    "mlp_only_layers", "model_type", "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings", "num_attention_heads", "num_hidden_layers",
    "num_key_value_heads", "rms_norm_eps", "rope_parameters", "tie_word_embeddings",
    "use_cache", "vocab_size",
}
PINNED_VISION_CONFIG_KEYS = {
    "deepstack_visual_indexes", "depth", "hidden_act", "hidden_size", "in_channels",
    "initializer_range", "intermediate_size", "model_type", "num_heads",
    "num_position_embeddings", "out_hidden_size", "patch_size", "spatial_merge_size",
    "temporal_patch_size",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_pinned_structure() -> dict[str, Any]:
    value = json.loads(STRUCTURE_PATH.read_text())
    required = {
        "schema_version", "model_id", "model_revision", "model_type", "architecture",
        "base_config_sha256", "base_index_sha256", "config_structure_sha256",
        "tensor_count", "tensor_bytes", "tensor_inventory_sha256", "dtype_counts",
        "source_shard_header_sha256", "source_shard_sha256", "source_shard_size",
    }
    if set(value) != required or value["schema_version"] != 1:
        raise ValueError("pinned model structure schema changed")
    return value


def config_structure(config: dict[str, Any]) -> dict[str, Any]:
    if not PINNED_CONFIG_KEYS <= set(config):
        raise ValueError("merged checkpoint config lacks pinned structural fields")
    text = config["text_config"]
    vision = config["vision_config"]
    if (
        not isinstance(text, dict)
        or not PINNED_TEXT_CONFIG_KEYS <= set(text)
        or not isinstance(vision, dict)
        or not PINNED_VISION_CONFIG_KEYS <= set(vision)
    ):
        raise ValueError("merged checkpoint nested config lacks pinned structural fields")
    result = {
        key: config[key]
        for key in sorted(PINNED_CONFIG_KEYS - {"text_config", "vision_config"})
    }
    result["text_config"] = {
        key: text[key] for key in sorted(PINNED_TEXT_CONFIG_KEYS)
    }
    result["vision_config"] = {
        key: vision[key] for key in sorted(PINNED_VISION_CONFIG_KEYS)
    }
    return result


def _read_safetensors_header(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    logical_bytes = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensors shard lacks an eight-byte header length")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes < 2 or header_bytes > logical_bytes - 8:
            raise ValueError("safetensors shard header length is invalid")
        encoded = handle.read(header_bytes)
    try:
        raw = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("safetensors shard header is invalid JSON") from error
    if not isinstance(raw, dict):
        raise ValueError("safetensors shard header is not an object")
    metadata = raw.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("safetensors shard metadata is malformed")
    tensors: dict[str, dict[str, Any]] = {}
    spans: list[tuple[int, int, str]] = []
    for key, entry in raw.items():
        if not isinstance(key, str) or not isinstance(entry, dict) or set(entry) != {
            "dtype", "shape", "data_offsets"
        }:
            raise ValueError("safetensors tensor metadata schema changed")
        dtype = entry["dtype"]
        shape = entry["shape"]
        offsets = entry["data_offsets"]
        if (
            dtype not in DTYPE_BYTES
            or not isinstance(shape, list)
            or any(not isinstance(value, int) or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or value < 0 for value in offsets)
            or offsets[1] < offsets[0]
        ):
            raise ValueError("safetensors tensor metadata is invalid")
        nbytes = math.prod(shape) * DTYPE_BYTES[dtype]
        if offsets[1] - offsets[0] != nbytes:
            raise ValueError("safetensors tensor offsets differ from shape/dtype bytes")
        tensors[key] = {"dtype": dtype, "shape": shape, "nbytes": nbytes}
        spans.append((offsets[0], offsets[1], key))
    cursor = 0
    for start, end, _key in sorted(spans):
        if start != cursor:
            raise ValueError("safetensors tensor payload is overlapping or non-contiguous")
        cursor = end
    if logical_bytes != 8 + header_bytes + cursor:
        raise ValueError("safetensors logical file size differs from its tensor payload")
    allocated_bytes = int(getattr(path.stat(), "st_blocks", 0)) * 512
    if allocated_bytes < int(logical_bytes * MIN_PHYSICAL_ALLOCATION_FRACTION):
        raise ValueError("safetensors shard is sparse or physically incomplete")
    return tensors, {
        "logical_bytes": logical_bytes,
        "allocated_bytes": allocated_bytes,
        "header_sha256": hashlib.sha256(prefix + encoded).hexdigest(),
    }


def merged_checkpoint_inventory(
    path: Path, *, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    expected = load_pinned_structure() if expected is None else expected
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
    config_hash = canonical_sha256(config_structure(config))
    if (
        config.get("model_type") != expected["model_type"]
        or config.get("architectures") != [expected["architecture"]]
        or config_hash != expected["config_structure_sha256"]
    ):
        raise ValueError("merged checkpoint config differs from pinned Qwen3.5-4B structure")
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
        not isinstance(name, str) or "/" in name or not name.endswith(".safetensors")
        for name in referenced
    ):
        raise ValueError("merged checkpoint index has an unexpected shard layout")
    observed_shards = {item.name for item in path.glob("*.safetensors")}
    if referenced != observed_shards:
        raise ValueError("merged checkpoint shard set differs from its weight index")
    inventory: dict[str, dict[str, Any]] = {}
    shard_details: dict[str, dict[str, Any]] = {}
    for shard_name in sorted(referenced):
        tensors, details = _read_safetensors_header(path / shard_name)
        if set(inventory) & set(tensors):
            raise ValueError("merged checkpoint repeats tensor keys across shards")
        if any(weight_map.get(key) != shard_name for key in tensors):
            raise ValueError("merged checkpoint tensor placement differs from its index")
        inventory.update(tensors)
        shard_details[shard_name] = details
    if set(inventory) != set(weight_map):
        raise ValueError("merged checkpoint tensor keys differ from its weight index")
    tensor_bytes = sum(value["nbytes"] for value in inventory.values())
    dtype_counts = dict(sorted(Counter(value["dtype"] for value in inventory.values()).items()))
    inventory_sha256 = canonical_sha256(inventory)
    if (
        len(inventory) != int(expected["tensor_count"])
        or tensor_bytes != int(expected["tensor_bytes"])
        or inventory_sha256 != expected["tensor_inventory_sha256"]
        or dtype_counts != expected["dtype_counts"]
        or metadata["total_size"] != tensor_bytes
    ):
        raise ValueError("merged checkpoint tensor inventory differs from pinned Qwen3.5-4B")
    return {
        "weight_index_sha256": sha256_file(index_path),
        "config_sha256": sha256_file(path / "config.json"),
        "config_structure_sha256": config_hash,
        "tokenizer_json_sha256": sha256_file(path / "tokenizer.json"),
        "shard_count": len(referenced),
        "tensor_count": len(inventory),
        "tensor_bytes": tensor_bytes,
        "tensor_inventory_sha256": inventory_sha256,
        "dtype_counts": dtype_counts,
        "logical_shard_bytes": sum(value["logical_bytes"] for value in shard_details.values()),
        "allocated_shard_bytes": sum(value["allocated_bytes"] for value in shard_details.values()),
        "shard_header_sha256": {
            name: value["header_sha256"] for name, value in sorted(shard_details.items())
        },
    }


def adapter_tensor_inventory(adapter_path: Path) -> dict[str, Any]:
    if not adapter_path.is_file() or adapter_path.stat().st_size < MIN_ADAPTER_BYTES:
        raise ValueError("retained source adapter tensor file is absent or implausibly small")
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ValueError("safetensors is required to authenticate adapter tensors") from error
    with safe_open(str(adapter_path), framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
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
