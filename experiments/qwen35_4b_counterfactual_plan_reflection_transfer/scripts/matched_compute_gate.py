#!/usr/bin/env python3
"""Build the final two-seed matched-compute promotion gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if sys.flags.no_site != 1:
    raise SystemExit("stage must start with the pinned interpreter and -I -B -S")

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from runtime_contract import (  # noqa: E402
    bootstrap_runtime_environment,
    require_detached_execution_worktree,
    seal_runtime_environment,
)

bootstrap_runtime_environment(EXP.parents[1], "training")

import yaml

from firewall import install_benchmark_firewall  # noqa: E402
from matched_compute import build_matched_compute_artifact  # noqa: E402

install_benchmark_firewall(EXP.parents[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, action="append", required=True)
    parser.add_argument("--reservoir-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_detached_execution_worktree(EXP.parents[1])
    config_path = EXP / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    result = build_matched_compute_artifact(
        confirmation_decision_paths=args.confirmation,
        reservoir_manifest_path=args.reservoir_manifest,
        config=config,
        config_path=config_path,
        experiment_root=EXP,
    )
    seal_runtime_environment(EXP.parents[1], "training")
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({"gate": result["gate"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
