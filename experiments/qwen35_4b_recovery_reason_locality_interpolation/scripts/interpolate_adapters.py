#!/usr/bin/env python3
"""Merge an action-to-reason LoRA interpolation into the frozen apex model."""

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


def adapter_config(path: Path) -> dict:
    return json.loads((path / "adapter_config.json").read_text())


def validate_configs(action: dict, reason: dict) -> float:
    scalar_keys = ("r", "lora_alpha", "fan_in_fan_out", "bias")
    for key in scalar_keys:
        if action.get(key) != reason.get(key):
            raise SystemExit(f"adapter configuration mismatch at {key}")
    if set(action.get("target_modules", [])) != set(reason.get("target_modules", [])):
        raise SystemExit("adapter target-module sets differ")
    if action.get("base_model_name_or_path") != reason.get("base_model_name_or_path"):
        raise SystemExit("adapter base checkpoints differ")
    return float(action["lora_alpha"]) / float(action["r"])


def interpolate_delta(
    action_delta: torch.Tensor, reason_delta: torch.Tensor, reason_lambda: float
) -> torch.Tensor:
    if not 0.0 <= reason_lambda <= 1.0:
        raise ValueError("reason_lambda must be in [0, 1]")
    if action_delta.shape != reason_delta.shape:
        raise ValueError("endpoint delta shapes differ")
    return action_delta + reason_lambda * (reason_delta - action_delta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--action-adapter", type=Path, required=True)
    parser.add_argument("--reason-adapter", type=Path, required=True)
    parser.add_argument("--reason-lambda", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lam = float(args.reason_lambda)
    if not 0.0 <= lam <= 1.0:
        raise SystemExit("--reason-lambda must be in [0, 1]")
    if args.base_model.resolve() == args.out.resolve():
        raise SystemExit("merge output must differ from the immutable warm start")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out}")

    base_config = json.loads((args.base_model / "config.json").read_text())
    base_merge = json.loads((args.base_model / "merge_receipt.json").read_text())
    if (
        base_config.get("model_type") != "qwen3_5"
        or base_merge.get("base_model") != MODEL_ID
        or base_merge.get("base_revision") != MODEL_REVISION
    ):
        raise SystemExit("base model is not the registered Qwen/Qwen3.5-4B warm start")
    action_config = adapter_config(args.action_adapter)
    reason_config = adapter_config(args.reason_adapter)
    lora_scale = validate_configs(action_config, reason_config)

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    tails: dict[str, list[str]] = {}
    for name, module in model.named_modules():
        if ".layers." in name and hasattr(module, "weight"):
            tails.setdefault(name.split(".layers.", 1)[1], []).append(name)

    applied = 0
    norm_sums = {"action": 0.0, "reason": 0.0, "contrast": 0.0, "mixed": 0.0}
    norm_max = 0.0
    with (
        safe_open(str(args.action_adapter / "adapter_model.safetensors"), framework="pt") as action,
        safe_open(str(args.reason_adapter / "adapter_model.safetensors"), framework="pt") as reason,
    ):
        if set(action.keys()) != set(reason.keys()):
            raise SystemExit("adapter tensor-key sets differ")
        keys_a = sorted(key for key in action.keys() if key.endswith(".lora_A.weight"))
        for key_a in keys_a:
            if not key_a.startswith(ADAPTER_PREFIX):
                raise SystemExit(f"unexpected adapter key: {key_a}")
            tail = key_a[len(ADAPTER_PREFIX):-len(".lora_A.weight")]
            paths = tails.get(tail, [])
            if len(paths) != 1:
                raise SystemExit(f"missing/ambiguous composite module {tail}: {paths}")
            key_b = key_a.replace(".lora_A.", ".lora_B.")
            action_a = action.get_tensor(key_a).to(device=device, dtype=torch.float32)
            action_b = action.get_tensor(key_b).to(device=device, dtype=torch.float32)
            reason_a = reason.get_tensor(key_a).to(device=device, dtype=torch.float32)
            reason_b = reason.get_tensor(key_b).to(device=device, dtype=torch.float32)
            action_delta = (action_b @ action_a) * lora_scale
            reason_delta = (reason_b @ reason_a) * lora_scale
            contrast = reason_delta - action_delta
            mixed_delta = interpolate_delta(action_delta, reason_delta, lam)
            module = model.get_submodule(paths[0])
            if tuple(mixed_delta.shape) != tuple(module.weight.shape):
                raise SystemExit(
                    f"shape mismatch at {tail}: {mixed_delta.shape} != {module.weight.shape}"
                )
            base = module.weight.data.to(device=device, dtype=torch.float32)
            module.weight.data = (base + mixed_delta).to(device="cpu", dtype=torch.bfloat16)
            norms = {
                "action": float(torch.linalg.vector_norm(action_delta).item()),
                "reason": float(torch.linalg.vector_norm(reason_delta).item()),
                "contrast": float(torch.linalg.vector_norm(contrast).item()),
                "mixed": float(torch.linalg.vector_norm(mixed_delta).item()),
            }
            for key, value in norms.items():
                norm_sums[key] += value
            norm_max = max(norm_max, norms["mixed"])
            applied += 1
            del (
                base,
                action_a,
                action_b,
                reason_a,
                reason_b,
                action_delta,
                reason_delta,
                contrast,
                mixed_delta,
            )
    if applied == 0 or norm_sums["mixed"] <= 0:
        raise SystemExit(f"invalid interpolation coverage: applied={applied}")

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
        "schema_version": 1,
        "method": "explicit_action_to_reason_delta_interpolation",
        "formula": "W_apex + (1-lambda)*delta_action + lambda*delta_reason",
        "reason_lambda": lam,
        "model_lineage": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_model": str(args.base_model.resolve()),
        "base_config_sha256": sha256_file(args.base_model / "config.json"),
        "base_merge_receipt_sha256": sha256_file(args.base_model / "merge_receipt.json"),
        "action_adapter": str(args.action_adapter.resolve()),
        "reason_adapter": str(args.reason_adapter.resolve()),
        "action_adapter_config_sha256": sha256_file(args.action_adapter / "adapter_config.json"),
        "reason_adapter_config_sha256": sha256_file(args.reason_adapter / "adapter_config.json"),
        "action_adapter_weights_sha256": sha256_file(
            args.action_adapter / "adapter_model.safetensors"
        ),
        "reason_adapter_weights_sha256": sha256_file(
            args.reason_adapter / "adapter_model.safetensors"
        ),
        "applied_lora_modules": applied,
        "lora_scale": lora_scale,
        "delta_frobenius_norm_sums": norm_sums,
        "mixed_delta_frobenius_norm_max": norm_max,
        "merge_device": str(device),
        "fp32_tf32_allowed": (
            torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None
        ),
        "weight_files": [
            {"name": path.name, "sha256": sha256_file(path)}
            for path in sorted(args.out.glob("*.safetensors"))
        ],
    }
    (args.out / "interpolation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
