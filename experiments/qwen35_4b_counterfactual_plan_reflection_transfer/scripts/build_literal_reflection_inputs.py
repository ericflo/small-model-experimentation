#!/usr/bin/env python3
"""Build the dedicated sealed literal-reflection prompt bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from eval_inputs import jsonl_payload, reflection_prompts, reflection_receipt  # noqa: E402
from firewall import install_benchmark_firewall  # noqa: E402
from runtime_contract import require_detached_execution_worktree  # noqa: E402

install_benchmark_firewall(EXP.parents[1])


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    require_detached_execution_worktree(EXP.parents[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("qualification",), required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.prompts.resolve() == args.receipt.resolve():
        raise ValueError("prompt and receipt paths must differ")
    config_path = EXP / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["authorization"]["cpu_construction"] is not True:
        raise SystemExit("CPU construction is not authorized")
    prompts = reflection_prompts(config, args.split)
    receipt = reflection_receipt(
        config, hashlib.sha256(config_path.read_bytes()).hexdigest(), args.split
    )
    _write_exclusive(args.prompts, jsonl_payload(prompts))
    _write_exclusive(
        args.receipt, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
