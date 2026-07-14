#!/usr/bin/env python3
"""Merge a trained LoRA into the full composite Qwen3.5-4B checkpoint for vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from firewall import install_benchmark_firewall  # noqa: E402

install_benchmark_firewall(EXP.parents[1])

ADAPTER_PREFIX = "base_model.model.model.layers."


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path, excluded: set[str] | None = None) -> str:
    excluded = excluded or set()
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        if relative in excluded:
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    if config["authorization"]["evaluation"] is not True:
        raise SystemExit("adapter merge/evaluation is not authorized by the committed config")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    source_receipt_path = args.adapter / "training_receipt.json"
    source_receipt = json.loads(source_receipt_path.read_text())
    model_id = config["model"]["id"]
    revision = config["model"]["revision"]
    if (
        source_receipt["model_id"] != model_id
        or source_receipt["model_revision"] != revision
    ):
        raise ValueError("adapter training receipt has the wrong base identity")
    peft_config = json.loads((args.adapter / "adapter_config.json").read_text())
    if int(peft_config["r"]) != int(config["training"]["recipe"]["lora_rank"]):
        raise ValueError("adapter rank differs from preregistration")
    scale = float(peft_config["lora_alpha"]) / float(peft_config["r"])
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    tail_to_path: dict[str, list[str]] = {}
    for name, module in model.named_modules():
        if ".layers." in name and hasattr(module, "weight"):
            tail = name.split(".layers.", 1)[1]
            tail_to_path.setdefault(tail, []).append(name)

    applied = 0
    adapter_path = args.adapter / "adapter_model.safetensors"
    with safe_open(str(adapter_path), "pt") as tensors:
        keys = [key for key in tensors.keys() if key.endswith(".lora_A.weight")]
        for key_a in keys:
            if not key_a.startswith(ADAPTER_PREFIX):
                raise ValueError(f"unexpected adapter key layout: {key_a}")
            core = key_a[len(ADAPTER_PREFIX):-len(".lora_A.weight")]
            paths = tail_to_path.get(core, [])
            if len(paths) != 1:
                raise ValueError(f"ambiguous or missing composite module for {core!r}: {paths}")
            module = model.get_submodule(paths[0])
            lora_a = tensors.get_tensor(key_a).float()
            lora_b = tensors.get_tensor(key_a.replace(".lora_A.", ".lora_B.")).float()
            delta = (lora_b @ lora_a) * scale
            if delta.shape != module.weight.shape:
                raise ValueError(
                    f"shape mismatch at {core}: {delta.shape} vs {module.weight.shape}"
                )
            module.weight.data = (module.weight.data.float() + delta).to(torch.bfloat16)
            applied += 1
    if applied == 0:
        raise ValueError("no LoRA deltas applied")

    args.output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(args.output), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, use_fast=True
    )
    tokenizer.save_pretrained(str(args.output))
    try:
        AutoProcessor.from_pretrained(
            model_id, revision=revision, trust_remote_code=True
        ).save_pretrained(str(args.output))
    except Exception as error:  # noqa: BLE001 - processor is irrelevant to text-only serving
        (args.output / "processor_save_warning.txt").write_text(str(error) + "\n")
    receipt = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "model_id": model_id,
        "model_revision": revision,
        "source_training_receipt_sha256": _sha256_file(source_receipt_path),
        "source_adapter_sha256": _sha256_file(adapter_path),
        "applied_lora_modules": applied,
        "merged_tree_sha256": _sha256_tree(args.output),
    }
    (args.output / "merge_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
