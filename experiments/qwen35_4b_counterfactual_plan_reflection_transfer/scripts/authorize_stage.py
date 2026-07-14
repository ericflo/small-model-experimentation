#!/usr/bin/env python3
"""Issue hash-bound stage receipts only after every prerequisite gate passes."""

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

from firewall import install_benchmark_firewall  # noqa: E402

install_benchmark_firewall(EXP.parents[1])


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_decision(path: Path, block: str, seed: int, positive: bool) -> None:
    value = _read(path)
    if value.get("block") != block or int(value.get("seed", -1)) != seed:
        raise ValueError(f"{path} has the wrong block or seed")
    if value.get("capability", {}).get("capability_pass") is not True:
        raise ValueError(f"{path} did not pass the capability gate")
    if positive and value.get("positive_control", {}).get("pass") is not True:
        raise ValueError(f"{path} did not pass the positive-control gate")


def _require_retention(path: Path, seed: int) -> None:
    value = _read(path)
    if (
        value.get("arm") != "reflection_correct_action"
        or int(value.get("seed", -1)) != seed
        or value.get("gate", {}).get("pass") is not True
    ):
        raise ValueError(f"{path} did not pass correct-reflection retention for seed {seed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("screen_training", "replication_training", "confirmation", "final"),
        required=True,
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--qualification", type=Path, action="append", default=[])
    parser.add_argument("--confirmation", type=Path, action="append", default=[])
    parser.add_argument("--retention", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = EXP / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    screen = int(config["training"]["staged_seeds"]["screen"])
    replication = int(config["training"]["staged_seeds"]["replication"])
    inputs: list[Path] = []
    if args.stage == "screen_training":
        if args.calibration is None:
            raise ValueError("screen training requires calibration")
        value = _read(args.calibration)
        if value.get("gate", {}).get("pass") is not True:
            raise ValueError("calibration did not pass")
        inputs = [args.calibration]
    elif args.stage == "replication_training":
        if len(args.qualification) != 1 or len(args.retention) != 1:
            raise ValueError("replication training requires screen qualification and retention")
        _require_decision(args.qualification[0], "qualification", screen, positive=True)
        _require_retention(args.retention[0], screen)
        inputs = [*args.qualification, *args.retention]
    elif args.stage == "confirmation":
        if len(args.qualification) != 2 or len(args.retention) != 2:
            raise ValueError("confirmation requires two qualification and two retention passes")
        by_seed = {int(_read(path).get("seed", -1)): path for path in args.qualification}
        retention_by_seed = {int(_read(path).get("seed", -1)): path for path in args.retention}
        if set(by_seed) != {screen, replication} or set(retention_by_seed) != {screen, replication}:
            raise ValueError("confirmation prerequisites do not contain both frozen seeds")
        _require_decision(by_seed[screen], "qualification", screen, positive=True)
        _require_decision(by_seed[replication], "qualification", replication, positive=False)
        _require_retention(retention_by_seed[screen], screen)
        _require_retention(retention_by_seed[replication], replication)
        inputs = [*args.qualification, *args.retention]
    else:
        if len(args.confirmation) != 2:
            raise ValueError("final authorization requires both confirmation decisions")
        by_seed = {int(_read(path).get("seed", -1)): path for path in args.confirmation}
        if set(by_seed) != {screen, replication}:
            raise ValueError("final prerequisites do not contain both frozen seeds")
        _require_decision(by_seed[screen], "confirmation", screen, positive=False)
        _require_decision(by_seed[replication], "confirmation", replication, positive=False)
        inputs = list(args.confirmation)
    config_sha256 = _sha(config_path)
    for path in inputs:
        value = _read(path)
        if (
            value.get("experiment_id") != config["experiment_id"]
            or value.get("config_sha256") != config_sha256
        ):
            raise ValueError(f"{path} is not bound to the current experiment config")
    receipt = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "authorized_stage": args.stage,
        "config_sha256": config_sha256,
        "inputs": {str(path.resolve()): _sha(path) for path in inputs},
    }
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
