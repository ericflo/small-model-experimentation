"""Deterministic mixed-dtype-preserving tensor-level LoRA merge."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from merge_replay import _adapter_model_key, _target_module_keys


def _validate_contract(contract: dict[str, Any], shard_names: set[str]) -> None:
    required = {
        "implementation",
        "shard_policy",
        "expected_shards",
        "unchanged_tensor_policy",
        "adapted_tensor_math",
        "local_trust_remote_code",
        "physical_allocation",
    }
    if (
        set(contract) != required
        or contract["implementation"] != "tensor_level_safetensors"
        or contract["shard_policy"] != "preserve_exact_pinned_source_index"
        or contract["unchanged_tensor_policy"] != "exact_value_shape_and_source_dtype"
        or contract["adapted_tensor_math"]
        != "base_dtype(base.float32 + (B.float32 @ A.float32) * alpha/rank)"
        or contract["local_trust_remote_code"] is not False
        or contract["physical_allocation"] != "allocated_bytes_gte_logical_bytes"
        or contract["expected_shards"] != sorted(shard_names)
        or len(shard_names) != 2
    ):
        raise ValueError("tensor merge contract differs from the frozen two-shard policy")


def write_tensor_level_merge(
    *,
    base_root: Path,
    base_index: dict[str, Any],
    adapter_path: Path,
    output: Path,
    recipe: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Copy source shards, replacing only exact LoRA targets in source dtype."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    if output.exists():
        raise ValueError(f"tensor merge output already exists: {output}")
    if set(base_index) != {"metadata", "weight_map"}:
        raise ValueError("base weight-index schema changed")
    weight_map = base_index["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("base weight-index mapping is empty")
    shard_names = set(weight_map.values())
    _validate_contract(contract, shard_names)

    with safe_open(str(adapter_path), framework="pt", device="cpu") as adapter:
        adapter_keys = set(adapter.keys())
        a_keys = {key for key in adapter_keys if key.endswith(".lora_A.weight")}
        b_keys = {key for key in adapter_keys if key.endswith(".lora_B.weight")}
        if (
            not a_keys
            or {key.replace(".lora_A.", ".lora_B.") for key in a_keys} != b_keys
            or adapter_keys != a_keys | b_keys
        ):
            raise ValueError("adapter tensor set is not exact LoRA A/B pairs")
        adapter_by_model = {_adapter_model_key(key): key for key in a_keys}
        expected_adapted = _target_module_keys(
            set(weight_map), set(recipe["target_modules"])
        )
        if set(adapter_by_model) != expected_adapted:
            raise ValueError("adapter modules differ from the exact pinned target set")
        scale = float(recipe["lora_alpha"]) / float(recipe["lora_rank"])

        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(base_root / "config.json", output / "config.json")
        shutil.copyfile(
            base_root / "model.safetensors.index.json",
            output / "model.safetensors.index.json",
        )

        adapted = 0
        unchanged = 0
        for shard_name in sorted(shard_names):
            shard_keys = {
                key for key, assigned in weight_map.items() if assigned == shard_name
            }
            with safe_open(
                str(base_root / shard_name), framework="pt", device="cpu"
            ) as source:
                if set(source.keys()) != shard_keys:
                    raise ValueError("base shard contents differ from its exact index")
                tensors: dict[str, Any] = {}
                for key in sorted(shard_keys):
                    base_tensor = source.get_tensor(key)
                    if key in adapter_by_model:
                        key_a = adapter_by_model[key]
                        key_b = key_a.replace(".lora_A.", ".lora_B.")
                        tensor_a = adapter.get_tensor(key_a)
                        tensor_b = adapter.get_tensor(key_b)
                        if (
                            base_tensor.ndim != 2
                            or tensor_a.ndim != 2
                            or tensor_b.ndim != 2
                            or tensor_a.shape[0] != tensor_b.shape[1]
                            or tensor_a.shape[1] != base_tensor.shape[1]
                            or tensor_b.shape[0] != base_tensor.shape[0]
                        ):
                            raise ValueError(f"LoRA shapes do not compose at {key}")
                        tensors[key] = (
                            base_tensor.float()
                            + (tensor_b.float() @ tensor_a.float()) * scale
                        ).to(base_tensor.dtype)
                        adapted += 1
                    else:
                        # The mmap-backed tensor is serialized in its original dtype;
                        # no model-wide dtype coercion or reinitialization occurs.
                        tensors[key] = base_tensor
                        unchanged += 1
                save_file(
                    tensors,
                    str(output / shard_name),
                    metadata=source.metadata(),
                )
                del tensors
        if adapted != len(expected_adapted) or unchanged != len(weight_map) - adapted:
            raise ValueError("tensor merge did not visit the exact frozen tensor sets")
    return {
        "schema_version": 1,
        "implementation": contract["implementation"],
        "shard_policy": contract["shard_policy"],
        "shards": sorted(shard_names),
        "adapted_module_count": adapted,
        "unchanged_tensor_count": unchanged,
        "equation": contract["adapted_tensor_math"],
    }
