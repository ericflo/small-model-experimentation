#!/usr/bin/env python3
"""Explicitly merge a convex mixture of two same-base LoRA deltas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER_PREFIX = "base_model.model.model.layers."


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _config(adapter: Path) -> tuple[dict, float]:
    payload = json.loads((adapter / "adapter_config.json").read_text())
    return payload, float(payload["lora_alpha"]) / float(payload["r"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick-adapter", type=Path, required=True)
    parser.add_argument("--deep-adapter", type=Path, required=True)
    parser.add_argument("--deep-weight", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.deep_weight <= 1.0:
        raise SystemExit("deep weight must be in [0, 1]")
    quick_config, quick_scale = _config(args.quick_adapter)
    deep_config, deep_scale = _config(args.deep_adapter)
    if quick_config.get("target_modules") != deep_config.get("target_modules"):
        raise SystemExit("adapter target modules differ")
    print(f"[weighted-merge] loading {MODEL_ID}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="cpu",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    tail_to_path = {}
    for name, module in model.named_modules():
        if ".layers." in name and hasattr(module, "weight"):
            tail_to_path.setdefault(name.split(".layers.", 1)[1], []).append(name)
    applied = nonzero = 0
    norm_sum = 0.0
    with safe_open(str(args.quick_adapter / "adapter_model.safetensors"), "pt") as quick, safe_open(
        str(args.deep_adapter / "adapter_model.safetensors"), "pt"
    ) as deep:
        quick_keys = sorted(key for key in quick.keys() if key.endswith(".lora_A.weight"))
        deep_keys = sorted(key for key in deep.keys() if key.endswith(".lora_A.weight"))
        if quick_keys != deep_keys:
            raise SystemExit("adapter tensor layouts differ")
        for key_a in quick_keys:
            core = key_a[len(ADAPTER_PREFIX):-len(".lora_A.weight")]
            paths = tail_to_path.get(core, [])
            if len(paths) != 1:
                raise SystemExit(f"missing or ambiguous composite module {core}: {paths}")
            quick_delta = (
                quick.get_tensor(key_a.replace(".lora_A.", ".lora_B.")).to(device, torch.float32)
                @ quick.get_tensor(key_a).to(device, torch.float32)
            ) * quick_scale
            deep_delta = (
                deep.get_tensor(key_a.replace(".lora_A.", ".lora_B.")).to(device, torch.float32)
                @ deep.get_tensor(key_a).to(device, torch.float32)
            ) * deep_scale
            delta = (1.0 - args.deep_weight) * quick_delta + args.deep_weight * deep_delta
            module = model.get_submodule(paths[0])
            module.weight.data = (
                module.weight.data.to(device, torch.float32) + delta
            ).to("cpu", torch.bfloat16)
            norm = float(torch.linalg.vector_norm(delta).item())
            nonzero += norm > 0.0
            norm_sum += norm
            applied += 1
            del quick_delta, deep_delta, delta
    if applied == 0 or nonzero != applied:
        raise SystemExit(f"invalid weighted merge: {nonzero}/{applied} nonzero")
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, use_fast=True
    ).save_pretrained(args.out)
    receipt = {
        "method": "explicit_convex_lora_delta_merge",
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "quick_adapter": str(args.quick_adapter.resolve()),
        "quick_config_sha256": _digest(args.quick_adapter / "adapter_config.json"),
        "quick_weights_sha256": _digest(args.quick_adapter / "adapter_model.safetensors"),
        "deep_adapter": str(args.deep_adapter.resolve()),
        "deep_config_sha256": _digest(args.deep_adapter / "adapter_config.json"),
        "deep_weights_sha256": _digest(args.deep_adapter / "adapter_model.safetensors"),
        "deep_weight": args.deep_weight,
        "applied_lora_modules": applied, "nonzero_lora_modules": nonzero,
        "delta_frobenius_norm_sum": norm_sum,
        "merge_device": str(device),
        "fp32_tf32_allowed": (
            bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else None
        ),
        "weight_files": [
            {"name": path.name, "sha256": _digest(path)}
            for path in sorted(args.out.glob("*.safetensors"))
        ],
    }
    (args.out / "merge_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
