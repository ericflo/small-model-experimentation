#!/usr/bin/env python3
"""Reserialize the pinned base into the same composite form as merged adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
METHOD = "pinned_base_composite_reserialization"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_existing(root: Path) -> dict:
    receipt_path = root / "merge_receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        payload.get("method") != METHOD
        or payload.get("model_lineage") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or not payload.get("weight_files")
    ):
        raise ValueError("existing base receipt has the wrong lineage or method")
    for row in payload["weight_files"]:
        path = root / row["name"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"existing materialized base weight changed: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    receipt_path = output / "merge_receipt.json"
    if receipt_path.is_file():
        print(json.dumps(validate_existing(output), indent=2, sort_keys=True))
        return 0
    if output.exists() and any(output.iterdir()):
        parser.error("refusing a nonempty base directory without a valid receipt")

    # Loading and saving through the exact Transformers composite path used by the
    # explicit LoRA merger keeps config, generation config, tokenizer serialization,
    # and shard layout equivalent across all benchmark arms. A direct Hub hard-link
    # is weight-identical but not packaging-identical and is therefore only archival.
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    config = model.config.to_dict()
    text_config = config.get("text_config", {})
    if config.get("model_type") != "qwen3_5" or text_config.get("model_type") != "qwen3_5_text":
        raise SystemExit("loaded base is not the pinned Qwen3.5 composite")
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, use_fast=True
    )
    tokenizer.save_pretrained(str(output))
    processor_saved = False
    try:
        AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
        ).save_pretrained(str(output))
        processor_saved = True
    except Exception as exc:  # noqa: BLE001 - processor is optional for text-only use
        print(f"[base] processor save skipped: {type(exc).__name__}", flush=True)

    weights = [
        {"name": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(output.glob("*.safetensors"))
    ]
    if not weights:
        raise SystemExit("reserialized base has no safetensor weights")
    receipt = {
        "schema_version": 1,
        "method": METHOD,
        "model_lineage": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "processor_saved": processor_saved,
        "weight_files": weights,
        "config_sha256": sha256_file(output / "config.json"),
        "generation_config_sha256": sha256_file(output / "generation_config.json"),
        "tokenizer_sha256": sha256_file(output / "tokenizer.json"),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
