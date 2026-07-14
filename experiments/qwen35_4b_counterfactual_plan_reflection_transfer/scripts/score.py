#!/usr/bin/env python3
"""Score one immutable vLLM generation bundle into task-level gate rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from firewall import install_benchmark_firewall  # noqa: E402
from score_artifacts import build_score_rows, jsonl_payload  # noqa: E402

install_benchmark_firewall(EXP.parents[1])

def _write_exclusive(path: Path, rows: list[dict]) -> str:
    import hashlib

    payload = jsonl_payload(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--adapter-gate-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = EXP / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    scored = build_score_rows(
        config=config,
        config_path=config_path,
        experiment_root=EXP,
        generated_path=args.generated,
        metadata_path=args.metadata,
        input_receipt_path=args.input_receipt,
        labels_path=args.labels,
        arm=args.arm,
        training_seed=args.training_seed,
        adapter_gate_receipt_path=args.adapter_gate_receipt,
    )
    digest = _write_exclusive(args.output, scored)
    print(json.dumps({"rows": len(scored), "output_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
