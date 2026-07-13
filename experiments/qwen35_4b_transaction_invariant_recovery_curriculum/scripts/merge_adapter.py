#!/usr/bin/env python3
"""Merge a recovery QLoRA delta into the registered warm-start composite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER_PREFIX = "base_model.model.model.layers."


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--expected-base-weight-sha256", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.base_model.resolve() == args.out.resolve():
        raise SystemExit("merge output must differ from the immutable warm start")
    base_config = json.loads((args.base_model / "config.json").read_text())
    base_merge = json.loads((args.base_model / "merge_receipt.json").read_text())
    text_config = base_config.get("text_config", {})
    lineage = base_merge.get("model_lineage", base_merge.get("base_model"))
    revision = base_merge.get("model_revision", base_merge.get("base_revision"))
    recorded_weights = {
        item.get("name"): item.get("sha256")
        for item in base_merge.get("weight_files", [])
    }
    observed_weight_sha256 = sha256_file(args.base_model / "model.safetensors")
    if (
        base_config.get("model_type") != "qwen3_5"
        or text_config.get("model_type") != "qwen3_5_text"
        or text_config.get("vocab_size") != 248320
        or text_config.get("hidden_size") != 2560
        or text_config.get("num_hidden_layers") != 32
        or lineage != MODEL_ID
        or revision != MODEL_REVISION
        or recorded_weights.get("model.safetensors") != args.expected_base_weight_sha256
        or observed_weight_sha256 != args.expected_base_weight_sha256
    ):
        raise SystemExit("base model is not the registered Qwen/Qwen3.5-4B warm start")
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    scale = float(config["lora_alpha"]) / float(config["r"])
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="cpu",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    tails: dict[str, list[str]] = {}
    for name, module in model.named_modules():
        if ".layers." in name and hasattr(module, "weight"):
            tails.setdefault(name.split(".layers.", 1)[1], []).append(name)
    applied = 0
    nonzero = 0
    norm_sum = 0.0
    norm_max = 0.0
    with safe_open(str(args.adapter / "adapter_model.safetensors"), framework="pt") as tensors:
        for key_a in [key for key in tensors.keys() if key.endswith(".lora_A.weight")]:
            if not key_a.startswith(ADAPTER_PREFIX):
                raise SystemExit(f"unexpected adapter key: {key_a}")
            tail = key_a[len(ADAPTER_PREFIX):-len(".lora_A.weight")]
            paths = tails.get(tail, [])
            if len(paths) != 1:
                raise SystemExit(f"missing/ambiguous composite module {tail}: {paths}")
            module = model.get_submodule(paths[0])
            lora_a = tensors.get_tensor(key_a).to(device=device, dtype=torch.float32)
            lora_b = tensors.get_tensor(key_a.replace(".lora_A.", ".lora_B.")).to(
                device=device, dtype=torch.float32
            )
            delta = (lora_b @ lora_a) * scale
            if tuple(delta.shape) != tuple(module.weight.shape):
                raise SystemExit(f"shape mismatch at {tail}: {delta.shape} != {module.weight.shape}")
            base = module.weight.data.to(device=device, dtype=torch.float32)
            module.weight.data = (base + delta).to(device="cpu", dtype=torch.bfloat16)
            norm = float(torch.linalg.vector_norm(delta).item())
            applied += 1
            nonzero += norm > 0
            norm_sum += norm
            norm_max = max(norm_max, norm)
            del base, delta, lora_a, lora_b
    if applied == 0 or nonzero != applied:
        raise SystemExit(f"invalid merge coverage: applied={applied} nonzero={nonzero}")
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    tokenizer.save_pretrained(str(args.out))
    try:
        AutoProcessor.from_pretrained(
            args.base_model, local_files_only=True, trust_remote_code=True
        ).save_pretrained(str(args.out))
    except Exception as exc:  # noqa: BLE001
        print(f"[merge] optional processor skipped: {exc}", flush=True)
    receipt = {
        "method": "explicit_composite_lora_merge",
        "model_lineage": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_model": str(args.base_model.resolve()),
        "base_weight_sha256": observed_weight_sha256,
        "base_config_sha256": sha256_file(args.base_model / "config.json"),
        "base_merge_receipt_sha256": sha256_file(args.base_model / "merge_receipt.json"),
        "adapter": str(args.adapter.resolve()),
        "adapter_config_sha256": sha256_file(args.adapter / "adapter_config.json"),
        "adapter_weights_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "applied_lora_modules": applied,
        "nonzero_lora_modules": nonzero,
        "delta_frobenius_norm_sum": norm_sum,
        "delta_frobenius_norm_max": norm_max,
        "scale": scale,
        "merge_device": str(device),
        "fp32_tf32_allowed": torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None,
        "weight_files": [
            {"name": path.name, "sha256": sha256_file(path)}
            for path in sorted(args.out.glob("*.safetensors"))
        ],
    }
    (args.out / "merge_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
