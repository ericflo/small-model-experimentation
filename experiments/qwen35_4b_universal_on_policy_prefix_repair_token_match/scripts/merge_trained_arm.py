#!/usr/bin/env python3
"""Explicitly merge one authenticated trained arm for same-backend vLLM use."""

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
MERGER = (
    ROOT
    / "experiments"
    / "qwen35_4b_same_prefix_advantage_routing"
    / "scripts"
    / "merge_adapter.py"
)
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LARGE = ROOT / "large_artifacts" / EXP.name
FROZEN_ADAPTERS = {
    "replay_after_close": LARGE / "adapters" / "replay_after_close",
    "prefix_repair_after_close": LARGE / "adapters" / "prefix_repair_after_close",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"not a JSON object: {path}")
    return payload


def committed_at_head(path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return committed.returncode == 0 and committed.stdout == path.read_bytes()


def authenticate_adapter(name: str, *, require_committed: bool = True) -> dict:
    sys.path.insert(0, str(EXP / "scripts"))
    from train_trial import (  # noqa: PLC0415
        validate_candidate_checkpoint,
        validate_control_prerequisite,
    )

    if name == "replay_after_close":
        checkpoint = validate_control_prerequisite(require_committed=require_committed)
    elif name == "prefix_repair_after_close":
        checkpoint = validate_candidate_checkpoint(require_committed=require_committed)
    else:
        raise ValueError("merge name is not a preregistered trained arm")
    expected = FROZEN_ADAPTERS[name].resolve()
    if Path(checkpoint["adapter"]).resolve() != expected:
        raise ValueError("authenticated adapter path changed")
    return checkpoint


def validate_external_merge(output: Path, adapter: Path) -> dict:
    receipt_path = output / "merge_receipt.json"
    payload = load_json(receipt_path)
    if (
        payload.get("method") != "explicit_composite_lora_merge"
        or payload.get("base_model") != MODEL_ID
        or payload.get("base_revision") != MODEL_REVISION
        or Path(payload.get("adapter", "")).resolve() != adapter.resolve()
        or payload.get("adapter_config_sha256")
        != sha256_file(adapter / "adapter_config.json")
        or payload.get("adapter_weights_sha256")
        != sha256_file(adapter / "adapter_model.safetensors")
        or int(payload.get("applied_lora_modules", 0)) <= 0
        or payload.get("nonzero_lora_modules") != payload.get("applied_lora_modules")
    ):
        raise ValueError("merge receipt failed lineage/application checks")
    files = payload.get("weight_files", [])
    if not files:
        raise ValueError("merge receipt contains no weight files")
    for row in files:
        path = output / row["name"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"merged weight changed: {path}")
    return payload


def validate_published_merge(
    name: str, *, require_committed: bool = True
) -> dict:
    checkpoint = authenticate_adapter(name, require_committed=require_committed)
    adapter = FROZEN_ADAPTERS[name].resolve()
    output = (LARGE / "merged" / name).resolve()
    run_receipt = EXP / "runs" / "merges" / f"{name}.json"
    if not run_receipt.is_file() or (
        require_committed and not committed_at_head(run_receipt)
    ):
        raise ValueError("trained-arm merge receipt is not committed at HEAD")
    payload = load_json(run_receipt)
    external = validate_external_merge(output, adapter)
    log = Path(payload.get("log", ""))
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_id") != EXP.name
        or payload.get("name") != name
        or Path(payload.get("adapter", "")).resolve() != adapter
        or payload.get("adapter_config_sha256")
        != checkpoint["adapter_config_sha256"]
        or payload.get("adapter_weights_sha256")
        != checkpoint["adapter_weights_sha256"]
        or payload.get("training_receipt_sha256") != checkpoint["receipt_sha256"]
        or Path(payload.get("merged", "")).resolve() != output
        or payload.get("merge_receipt_sha256")
        != sha256_file(output / "merge_receipt.json")
        or payload.get("weight_files") != external["weight_files"]
        or payload.get("preflight_git_status") != ""
        or not log.is_file()
        or (require_committed and not committed_at_head(log))
        or payload.get("log_sha256") != sha256_file(log)
    ):
        raise ValueError("published trained-arm merge violates its frozen contract")
    return payload


def normalize_log(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def run_text(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.name):
        parser.error("unsafe merge name")
    if args.name not in FROZEN_ADAPTERS:
        parser.error("merge name is not a preregistered trained arm")
    adapter = args.adapter.resolve()
    output = args.out.resolve()
    if adapter != FROZEN_ADAPTERS[args.name].resolve():
        parser.error("adapter path does not match the frozen merge arm")
    if output != (LARGE / "merged" / args.name).resolve():
        parser.error("merged output does not match the frozen arm path")
    try:
        checkpoint = authenticate_adapter(args.name)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    run_receipt = EXP / "runs" / "merges" / f"{args.name}.json"
    log_path = EXP / "runs" / "merges" / f"{args.name}.log"
    if output.exists() or run_receipt.exists() or log_path.exists():
        parser.error("refusing to overwrite a trained-arm merge artifact")
    preflight_git_head = run_text(["git", "rev-parse", "HEAD"])
    preflight_git_status = run_text(["git", "status", "--short"])
    if preflight_git_status:
        parser.error("merge requires a clean incrementally committed worktree")

    command = [
        str(PYTHON),
        "-B",
        str(MERGER),
        "--adapter",
        str(adapter),
        "--out",
        str(output),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()
    normalize_log(log_path)
    if returncode != 0:
        raise SystemExit(f"merge failed with exit {returncode}; preserved {log_path}")
    merge = validate_external_merge(output, adapter)
    receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "name": args.name,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter": str(adapter),
        "adapter_config_sha256": checkpoint["adapter_config_sha256"],
        "adapter_weights_sha256": checkpoint["adapter_weights_sha256"],
        "training_receipt": checkpoint["receipt"],
        "training_receipt_sha256": checkpoint["receipt_sha256"],
        "merged": str(output),
        "merge_receipt_sha256": sha256_file(output / "merge_receipt.json"),
        "weight_files": merge["weight_files"],
        "applied_lora_modules": merge["applied_lora_modules"],
        "nonzero_lora_modules": merge["nonzero_lora_modules"],
        "merger": str(MERGER),
        "merger_sha256": sha256_file(MERGER),
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "preflight_git_head": preflight_git_head,
        "preflight_git_status": preflight_git_status,
        "command": command,
    }
    run_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
