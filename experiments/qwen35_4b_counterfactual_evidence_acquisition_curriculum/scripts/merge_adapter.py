#!/usr/bin/env python3
"""Merge a recovery QLoRA delta into the registered warm-start composite."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER_PREFIX = "base_model.model.model.layers."
EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

from harness import (  # noqa: E402
    TOKENIZER_FILE_NAMES,
    tokenizer_provenance,
    validate_model_execution_lock,
    validate_registered_checkpoint,
)
from train import validate_training_authorization  # noqa: E402


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
    parser.add_argument(
        "--arm",
        choices=["evidence_binding", "explicit_redundant", "shuffled_binding"],
        required=True,
    )
    parser.add_argument("--expected-training-receipt-sha256", required=True)
    parser.add_argument("--design-lock", type=Path, required=True)
    parser.add_argument("--training-authorization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate_model_execution_lock(
            EXP, args.design_lock, "scripts/merge_adapter.py"
        )
    except ValueError as exc:
        raise SystemExit(f"merge is not design-locked: {exc}") from exc
    if args.base_model.resolve() == args.out.resolve():
        raise SystemExit("merge output must differ from the immutable warm start")
    design_lock = json.loads(args.design_lock.read_text())
    frozen_order = design_lock.get("frozen_file_order")
    frozen_files = design_lock.get("frozen_files")
    design_commit = design_lock.get("design_commit")
    if (
        design_lock.get("status") != "locked"
        or design_lock.get("experiment_id") != EXP.name
        or not isinstance(frozen_order, list)
        or not isinstance(frozen_files, dict)
        or set(frozen_order) != set(frozen_files)
        or len(frozen_order) != len(frozen_files)
        or not isinstance(design_commit, str)
        or len(design_commit) != 40
        or any(character not in "0123456789abcdef" for character in design_commit)
        or frozen_files.get("scripts/merge_adapter.py")
        != sha256_file(Path(__file__).resolve())
    ):
        raise SystemExit("merge implementation differs from the design lock")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", design_commit, "HEAD"],
        cwd=ROOT, check=False,
    ).returncode:
        raise SystemExit("design commit is not an ancestor of merge HEAD")
    training_receipt_path = args.adapter / "training_receipt.json"
    if sha256_file(training_receipt_path) != args.expected_training_receipt_sha256:
        raise SystemExit("training receipt hash differs from the registered merge input")
    training_receipt = json.loads(training_receipt_path.read_text())
    validate_training_authorization(args.training_authorization, args.design_lock)
    if (
        training_receipt.get("arm") != args.arm
        or training_receipt.get("adapter_path") != str(args.adapter.resolve())
        or training_receipt.get("design_lock_sha256") != sha256_file(args.design_lock)
        or training_receipt.get("start_checkpoint", {}).get("weight_sha256")
        != args.expected_base_weight_sha256
        or training_receipt.get("adapter_config_sha256")
        != sha256_file(args.adapter / "adapter_config.json")
        or training_receipt.get("adapter_weights_sha256")
        != sha256_file(args.adapter / "adapter_model.safetensors")
        or training_receipt.get("training_authorization_sha256")
        != sha256_file(args.training_authorization)
    ):
        raise SystemExit("adapter/training receipt/design-lock chain is invalid")
    registered_cfg = yaml.safe_load(
        (EXP / "configs" / "default.yaml").read_text(encoding="utf-8")
    )
    try:
        base_checkpoint = validate_registered_checkpoint(
            EXP, args.base_model, registered_cfg, args.design_lock, "start"
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"base checkpoint registration is invalid: {exc}") from exc
    if (
        base_checkpoint["model_weight_sha256"]
        != args.expected_base_weight_sha256
    ):
        raise SystemExit("base weight hash differs from the registered invocation")
    base_config = json.loads((args.base_model / "config.json").read_text())
    base_merge = json.loads((args.base_model / "merge_receipt.json").read_text())
    text_config = base_config.get("text_config", {})
    lineage = base_merge.get("model_lineage", base_merge.get("base_model"))
    revision = base_merge.get("model_revision", base_merge.get("base_revision"))
    recorded_weights = {
        item.get("name"): item.get("sha256")
        for item in base_merge.get("weight_files", [])
    }
    observed_weight_sha256 = base_checkpoint["model_weight_sha256"]
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
    try:
        base_tokenizer = tokenizer_provenance(args.base_model)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"base tokenizer provenance is invalid: {exc}") from exc
    registered_model = registered_cfg["model"]
    if (
        base_tokenizer["tokenizer_manifest_sha256"]
        != registered_model["start_tokenizer_manifest_sha256"]
        or base_tokenizer["tokenizer_compatibility_sha256"]
        != registered_model["tokenizer_compatibility_sha256"]
    ):
        raise SystemExit("base tokenizer differs from the frozen config identity")
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
    if (
        training_receipt.get("lora_module_count") != applied
        or int(training_receipt.get("trainable_parameter_count", 0)) <= 0
    ):
        raise SystemExit("merge coverage differs from recorded trainable geometry")
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True)
    try:
        AutoProcessor.from_pretrained(
            args.base_model, local_files_only=True, trust_remote_code=True
        ).save_pretrained(str(args.out))
    except Exception as exc:  # noqa: BLE001
        print(f"[merge] optional processor skipped: {exc}", flush=True)
    # Processor serialization may rewrite tokenizer metadata.  Restore the exact
    # registered base bytes, then reject any extra tokenizer-like artifacts.
    for name in TOKENIZER_FILE_NAMES:
        shutil.copyfile(args.base_model / name, args.out / name)
    try:
        output_tokenizer = tokenizer_provenance(args.out)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"merged tokenizer provenance is invalid: {exc}") from exc
    if output_tokenizer != base_tokenizer:
        raise SystemExit("merged tokenizer files differ from the registered warm start")
    receipt = {
        "method": "explicit_composite_lora_merge",
        "model_lineage": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_model": str(args.base_model.resolve()),
        "base_weight_sha256": observed_weight_sha256,
        "base_config_sha256": sha256_file(args.base_model / "config.json"),
        "base_generation_config_sha256": sha256_file(
            args.base_model / "generation_config.json"
        ),
        "base_merge_receipt_sha256": sha256_file(args.base_model / "merge_receipt.json"),
        "base_tokenizer_files": base_tokenizer["tokenizer_files"],
        "base_tokenizer_manifest_sha256": base_tokenizer[
            "tokenizer_manifest_sha256"
        ],
        "base_tokenizer_compatibility_sha256": base_tokenizer[
            "tokenizer_compatibility_sha256"
        ],
        "adapter": str(args.adapter.resolve()),
        "arm": args.arm,
        "training_receipt_sha256": args.expected_training_receipt_sha256,
        "training_authorization_sha256": training_receipt[
            "training_authorization_sha256"
        ],
        "design_lock_sha256": sha256_file(args.design_lock),
        "adapter_config_sha256": sha256_file(args.adapter / "adapter_config.json"),
        "adapter_weights_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "applied_lora_modules": applied,
        "nonzero_lora_modules": nonzero,
        "delta_frobenius_norm_sum": norm_sum,
        "delta_frobenius_norm_max": norm_max,
        "scale": scale,
        "merge_device": str(device),
        "fp32_tf32_allowed": torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None,
        "output_config_sha256": sha256_file(args.out / "config.json"),
        "output_generation_config_sha256": sha256_file(
            args.out / "generation_config.json"
        ),
        "weight_files": [
            {"name": path.name, "sha256": sha256_file(path)}
            for path in sorted(args.out.glob("*.safetensors"))
        ],
        **output_tokenizer,
    }
    (args.out / "merge_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
