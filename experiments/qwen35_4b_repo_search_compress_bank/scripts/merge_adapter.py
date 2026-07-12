#!/usr/bin/env python3
"""Explicitly merge a QLoRA adapter into the full Qwen3.5-4B composite."""

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
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    scale = float(config["lora_alpha"]) / float(config["r"])
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
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
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, use_fast=True
    )
    tokenizer.save_pretrained(str(args.out))
    try:
        AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
        ).save_pretrained(str(args.out))
    except Exception as exc:  # noqa: BLE001
        print(f"[merge] optional processor skipped: {exc}", flush=True)
    receipt = {
        "method": "explicit_composite_lora_merge",
        "base_model": MODEL_ID,
        "base_revision": MODEL_REVISION,
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
