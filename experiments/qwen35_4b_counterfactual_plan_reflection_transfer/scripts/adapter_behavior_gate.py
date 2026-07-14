#!/usr/bin/env python3
"""Prove a receipt-bound merged adapter changes greedy Qwen behavior or logprob."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from adapter_gate_artifacts import build_adapter_gate_artifact  # noqa: E402
from firewall import install_benchmark_firewall  # noqa: E402

install_benchmark_firewall(EXP.parents[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-generated", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--merged-generated", type=Path, required=True)
    parser.add_argument("--merged-metadata", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = EXP / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    receipt = build_adapter_gate_artifact(
        arm=args.arm,
        seed=args.seed,
        base_generated_path=args.base_generated,
        base_metadata_path=args.base_metadata,
        merged_generated_path=args.merged_generated,
        merged_metadata_path=args.merged_metadata,
        input_receipt_path=args.input_receipt,
        labels_path=args.labels,
        config=config,
        config_path=config_path,
        experiment_root=EXP,
    )
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
