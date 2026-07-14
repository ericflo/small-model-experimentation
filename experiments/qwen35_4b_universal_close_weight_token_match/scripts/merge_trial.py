#!/usr/bin/env python3
"""Run the proven explicit composite LoRA merge and authenticate its receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
MERGER = ROOT / "experiments" / "qwen35_4b_same_prefix_advantage_routing" / "scripts" / "merge_adapter.py"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_merge(output: Path, adapter: Path) -> dict:
    receipt_path = output / "merge_receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        payload.get("method") != "explicit_composite_lora_merge"
        or payload.get("base_model") != MODEL_ID
        or payload.get("base_revision") != MODEL_REVISION
        or Path(payload.get("adapter", "")).resolve() != adapter.resolve()
        or payload.get("adapter_config_sha256") != sha256_file(adapter / "adapter_config.json")
        or payload.get("adapter_weights_sha256") != sha256_file(adapter / "adapter_model.safetensors")
        or int(payload.get("applied_lora_modules", 0)) <= 0
        or payload.get("nonzero_lora_modules") != payload.get("applied_lora_modules")
    ):
        raise ValueError("merge receipt failed lineage/application checks")
    for row in payload.get("weight_files", []):
        path = output / row["name"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"merged weight changed: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.name):
        parser.error("unsafe merge name")
    adapter = args.adapter.resolve()
    output = args.out.resolve()
    if not (adapter / "adapter_model.safetensors").is_file():
        parser.error("adapter is incomplete")
    run_receipt = EXP / "runs" / "merges" / f"{args.name}.json"
    log_path = EXP / "runs" / "merges" / f"{args.name}.log"
    if (output / "merge_receipt.json").is_file() and run_receipt.is_file():
        payload = validate_merge(output, adapter)
        if json.loads(run_receipt.read_text(encoding="utf-8")).get("merge_receipt_sha256") != sha256_file(output / "merge_receipt.json"):
            parser.error("existing experiment merge receipt disagrees")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if output.exists() or run_receipt.exists() or log_path.exists():
        parser.error("refusing to overwrite a partial merge; preserve and use a new name")

    command = [str(PYTHON), str(MERGER), "--adapter", str(adapter), "--out", str(output)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line); sys.stdout.flush(); log.write(line); log.flush()
        returncode = process.wait()
    if returncode != 0:
        raise SystemExit(f"merge failed with exit {returncode}; preserved {log_path}")
    merge = validate_merge(output, adapter)
    receipt = {
        "schema_version": 1,
        "name": args.name,
        "adapter": str(adapter),
        "adapter_weights_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "merged": str(output),
        "merge_receipt_sha256": sha256_file(output / "merge_receipt.json"),
        "weight_files": merge["weight_files"],
        "merger": str(MERGER),
        "merger_sha256": sha256_file(MERGER),
        "command": command,
    }
    run_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
