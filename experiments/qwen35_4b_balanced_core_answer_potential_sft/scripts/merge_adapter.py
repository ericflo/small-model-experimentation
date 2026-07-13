#!/usr/bin/env python3
"""Merge a QLoRA adapter into the FULL composite Qwen3.5-4B checkpoint.

Rationale (2026-07-10): vLLM 0.24 runtime LoRA is a SILENT NO-OP for
Qwen3.5-4B (in-process probe: identical outputs with and without the
lora_request — the adapter's `model.layers.*` tensor names never match the
served composite module tree, and vLLM skips them without erroring). A
text-only merge_and_unload checkpoint does not load either (vLLM's Qwen3.5
class requires the composite config). So we merge the LoRA delta into the
composite checkpoint by EXPLICIT name mapping: adapter keys
`base_model.model.model.layers.N.<mod>.lora_{A,B}.weight` map onto the
composite's text-layer weights, W += (B @ A) * (alpha / r).

Deploy the result via menagerie --model-id or the experiment runner's
model_override. Run under the repo .venv.
"""

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
EXPECTED_LORA_PAIRS = 128
EXPECTED_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    peft_config = json.loads((args.adapter / "adapter_config.json").read_text())
    scale = float(peft_config["lora_alpha"]) / float(peft_config["r"])
    adapter_path = args.adapter / "adapter_model.safetensors"
    training_receipt = json.loads(
        (args.adapter / "training_receipt.json").read_text(encoding="utf-8")
    )
    if (
        training_receipt.get("model") != MODEL_ID
        or training_receipt.get("revision") != MODEL_REVISION
    ):
        raise SystemExit("training receipt model/revision violates merge contract")
    expected_config_sha = training_receipt.get("artifacts", {}).get(
        "adapter_config.json", {}
    ).get("sha256")
    if sha256_file(args.adapter / "adapter_config.json") != expected_config_sha:
        raise SystemExit("adapter config does not match training receipt")
    contract = training_receipt.get("training_contract", {})
    if (
        peft_config.get("base_model_name_or_path") != MODEL_ID
        or int(peft_config.get("r", -1)) != int(contract.get("rank", -2))
        or int(peft_config.get("lora_alpha", -1)) != int(contract.get("alpha", -2))
        or float(peft_config.get("lora_dropout", -1))
        != float(contract.get("dropout", -2))
        or set(peft_config.get("target_modules", [])) != EXPECTED_TARGETS
        or peft_config.get("bias") != "none"
        or peft_config.get("task_type") != "CAUSAL_LM"
        or peft_config.get("use_rslora") is not False
        or peft_config.get("use_dora") is not False
    ):
        raise SystemExit("adapter configuration violates the frozen merge contract")
    manifest_digest = hashlib.sha256()
    with safe_open(str(adapter_path), framework="pt", device="cpu") as tensors:
        all_keys = sorted(tensors.keys())
        for key in all_keys:
            value = tensors.get_tensor(key)
            manifest_digest.update(
                f"{key}\0{tuple(value.shape)}\0{value.dtype}\0".encode("utf-8")
            )
    keys_a = [key for key in all_keys if key.endswith(".lora_A.weight")]
    keys_b = [key for key in all_keys if key.endswith(".lora_B.weight")]
    expected_b = {key.replace(".lora_A.", ".lora_B.") for key in keys_a}
    if len(all_keys) != 2 * EXPECTED_LORA_PAIRS:
        raise SystemExit(f"unexpected adapter tensor count: {len(all_keys)}")
    if len(keys_a) != EXPECTED_LORA_PAIRS or len(keys_b) != EXPECTED_LORA_PAIRS:
        raise SystemExit(
            f"incomplete LoRA A/B tensors: A={len(keys_a)} B={len(keys_b)}"
        )
    if set(keys_b) != expected_b or set(all_keys) != set(keys_a) | set(keys_b):
        raise SystemExit("adapter has unpaired or unexpected non-LoRA tensor keys")
    if any(
        not key.startswith(ADAPTER_PREFIX)
        or key[: -len(".lora_A.weight")].split(".")[-1] not in EXPECTED_TARGETS
        for key in keys_a
    ):
        raise SystemExit("adapter keys do not match the frozen target-module set")
    manifest = {
        "sha256": manifest_digest.hexdigest(),
        "tensors": len(all_keys),
    }
    if training_receipt.get("adapter_tensor_manifest") != manifest:
        raise SystemExit("adapter tensor manifest does not match training receipt")
    expected_adapter_sha = training_receipt.get("artifacts", {}).get(
        "adapter_model.safetensors", {}
    ).get("sha256")
    if sha256_file(adapter_path) != expected_adapter_sha:
        raise SystemExit("adapter weights do not match training receipt")

    print(f"[merge] loading composite {MODEL_ID} bf16 ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="cpu")

    # Map "N.<suffix>" tails (as the adapter names them) to the composite's
    # full module paths. Qwen3.5 is a hybrid: not every layer has attention
    # projections, so discovery must go by name, not by layer index.
    tail_to_path: dict[str, list[str]] = {}
    for name, module in model.named_modules():
        if ".layers." in name and hasattr(module, "weight"):
            tail = name.split(".layers.", 1)[1]
            tail_to_path.setdefault(tail, []).append(name)
    sample = next(iter(tail_to_path))
    print(f"[merge] example composite tail: {sample!r} -> {tail_to_path[sample]}", flush=True)

    applied = 0
    with safe_open(str(adapter_path), framework="pt", device="cpu") as tensors:
        for key_a in keys_a:
            if not key_a.startswith(ADAPTER_PREFIX):
                raise SystemExit(f"unexpected adapter key layout: {key_a}")
            core = key_a[len(ADAPTER_PREFIX):-len(".lora_A.weight")]
            paths = tail_to_path.get(core, [])
            if len(paths) != 1:
                raise SystemExit(f"ambiguous or missing composite module for {core!r}: {paths}")
            module = model.get_submodule(paths[0])
            lora_a = tensors.get_tensor(key_a).float()
            lora_b = tensors.get_tensor(key_a.replace(".lora_A.", ".lora_B.")).float()
            delta = (lora_b @ lora_a) * scale
            if delta.shape != module.weight.shape:
                raise SystemExit(
                    f"shape mismatch at {core}: {delta.shape} vs {module.weight.shape}"
                )
            module.weight.data = (module.weight.data.float() + delta).to(torch.bfloat16)
            applied += 1
    print(f"[merge] applied {applied} LoRA deltas (expected 7 modules x layers)", flush=True)
    if applied != EXPECTED_LORA_PAIRS:
        raise SystemExit(
            f"incomplete merge: applied {applied}, expected {EXPECTED_LORA_PAIRS}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving composite merged model to {args.out} ...", flush=True)
    model.save_pretrained(str(args.out), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                        trust_remote_code=True, use_fast=True)
    tok.save_pretrained(str(args.out))
    try:
        AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
        ).save_pretrained(str(args.out))
    except Exception as exc:  # noqa: BLE001 - processor optional for text use
        print(f"[merge] processor save skipped: {exc}", flush=True)
    (args.out / "merge_application_receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "adapter_sha256": expected_adapter_sha,
                "adapter_tensor_manifest": manifest,
                "applied_lora_pairs": applied,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("[merge] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
