#!/usr/bin/env python3
"""Apply preregistered qualification/confirmation gates to scored arm bundles."""

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
from gate_artifacts import build_decision_artifact  # noqa: E402

install_benchmark_firewall(EXP.parents[1])


def _write_exclusive(path: Path, value: object) -> str:
    import hashlib

    seal_runtime_environment(EXP.parents[1], "training")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    require_detached_execution_worktree(EXP.parents[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=("qualification", "confirmation"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    config_path = EXP / "configs" / "default.yaml"
    result = build_decision_artifact(
        block=args.block,
        seed=args.seed,
        stage_receipt_path=args.stage_receipt,
        score_paths=args.scores,
        config=config,
        config_path=config_path,
        experiment_root=EXP,
    )
    digest = _write_exclusive(args.output, result)
    print(
        json.dumps(
            {
                "output_sha256": digest,
                "capability_pass": result["capability"]["capability_pass"],
                "reflection_specific_pass": result["capability"]["reflection_specific_pass"],
                "positive_control_pass": result.get("positive_control", {}).get("pass"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
