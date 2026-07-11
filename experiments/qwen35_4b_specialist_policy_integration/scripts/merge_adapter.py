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
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER_PREFIX = "base_model.model.model.layers."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        default=MODEL_ID,
        help="official pinned model id or prior local merged composite",
    )
    args = parser.parse_args()

    peft_config = json.loads((args.adapter / "adapter_config.json").read_text())
    scale = float(peft_config["lora_alpha"]) / float(peft_config["r"])

    base_source = (
        str(Path(args.base_model).resolve())
        if Path(args.base_model).exists()
        else args.base_model
    )
    if base_source != MODEL_ID and not Path(base_source, "config.json").exists():
        raise SystemExit(f"local base composite missing config.json: {base_source}")
    source_kwargs = (
        {"revision": MODEL_REVISION}
        if base_source == MODEL_ID
        else {"local_files_only": True}
    )

    print(f"[merge] loading composite {base_source} bf16 ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base_source, **source_kwargs, trust_remote_code=True,
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
    with safe_open(str(args.adapter / "adapter_model.safetensors"), "pt") as tensors:
        keys = [k for k in tensors.keys() if k.endswith(".lora_A.weight")]
        for key_a in keys:
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
    if applied == 0:
        raise SystemExit("no deltas applied — refusing to save a no-op checkpoint")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving composite merged model to {args.out} ...", flush=True)
    model.save_pretrained(str(args.out), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(base_source, **source_kwargs,
                                        trust_remote_code=True, use_fast=True)
    tok.save_pretrained(str(args.out))
    try:
        AutoProcessor.from_pretrained(
            base_source, **source_kwargs, trust_remote_code=True
        ).save_pretrained(str(args.out))
    except Exception as exc:  # noqa: BLE001 - processor optional for text use
        print(f"[merge] processor save skipped: {exc}", flush=True)
    import hashlib

    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    receipt = {
        "method": "explicit_composite_lora_merge",
        "base_model": base_source,
        "base_revision": MODEL_REVISION if base_source == MODEL_ID else None,
        "adapter": str(args.adapter.resolve()),
        "adapter_config_sha256": digest(args.adapter / "adapter_config.json"),
        "adapter_weights_sha256": digest(args.adapter / "adapter_model.safetensors"),
        "applied_lora_modules": applied,
        "scale": scale,
        "weight_files": [
            {"name": path.name, "sha256": digest(path)}
            for path in sorted(args.out.glob("*.safetensors"))
        ],
    }
    (args.out / "merge_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[merge] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
