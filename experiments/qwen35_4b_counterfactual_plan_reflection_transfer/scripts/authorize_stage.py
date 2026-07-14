#!/usr/bin/env python3
"""Issue hash-bound stage receipts only after every prerequisite gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from firewall import install_benchmark_firewall  # noqa: E402
from gate_artifacts import validate_gate_artifact  # noqa: E402
from stages import git_commit  # noqa: E402

install_benchmark_firewall(EXP.parents[1])


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_decision(
    path: Path,
    block: str,
    seed: int,
    positive: bool,
    *,
    config: dict,
    config_path: Path,
) -> dict:
    value = validate_gate_artifact(
        path,
        kind="decision",
        config=config,
        config_path=config_path,
        experiment_root=EXP,
    )
    if value.get("block") != block or int(value.get("seed", -1)) != seed:
        raise ValueError(f"{path} has the wrong block or seed")
    if value.get("capability", {}).get("capability_pass") is not True:
        raise ValueError(f"{path} did not pass the capability gate")
    if positive and value.get("positive_control", {}).get("pass") is not True:
        raise ValueError(f"{path} did not pass the positive-control gate")
    return value


def _require_retention(
    path: Path, seed: int, *, config: dict, config_path: Path
) -> dict:
    value = validate_gate_artifact(
        path,
        kind="retention",
        config=config,
        config_path=config_path,
        experiment_root=EXP,
    )
    if (
        value.get("arm") != "reflection_correct_action"
        or int(value.get("seed", -1)) != seed
        or value.get("gate", {}).get("pass") is not True
    ):
        raise ValueError(f"{path} did not pass correct-reflection retention for seed {seed}")
    return value


def _claim(kind: str, path: Path) -> dict:
    resolved = path.resolve()
    return {"kind": kind, "path": str(resolved), "sha256": _sha(resolved)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "calibration_generation",
            "screen_training",
            "replication_training",
            "confirmation",
            "final",
        ),
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
    claims: list[dict] = []
    if subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout:
        raise ValueError("stage receipts require a clean worktree")
    if args.stage == "calibration_generation":
        if config["authorization"]["evaluation"] is not True:
            raise ValueError("calibration generation is not authorized")
        if args.calibration is not None or args.qualification or args.confirmation or args.retention:
            raise ValueError("calibration generation accepts no prerequisites")
    elif args.stage == "screen_training":
        if config["authorization"]["training"] is not True:
            raise ValueError("screen training is not authorized")
        if args.qualification or args.confirmation or args.retention:
            raise ValueError("screen training accepts only calibration")
        if args.calibration is None:
            raise ValueError("screen training requires calibration")
        value = validate_gate_artifact(
            args.calibration,
            kind="calibration_gate",
            config=config,
            config_path=config_path,
            experiment_root=EXP,
        )
        if value.get("gate", {}).get("pass") is not True:
            raise ValueError("calibration did not pass")
        inputs = [args.calibration]
        claims = [
            {
                **_claim("calibration_gate", args.calibration),
            }
        ]
    elif args.stage == "replication_training":
        if config["authorization"]["training"] is not True:
            raise ValueError("replication training is not authorized")
        if args.calibration is not None or args.confirmation:
            raise ValueError("replication training received unrelated prerequisites")
        if len(args.qualification) != 1 or len(args.retention) != 1:
            raise ValueError("replication training requires screen qualification and retention")
        decision_value = _require_decision(
            args.qualification[0], "qualification", screen, positive=True,
            config=config, config_path=config_path,
        )
        retention_value = _require_retention(
            args.retention[0], screen, config=config, config_path=config_path
        )
        inputs = [*args.qualification, *args.retention]
        claims = [
            _claim("decision", args.qualification[0]),
            _claim("retention", args.retention[0]),
        ]
    elif args.stage == "confirmation":
        if config["authorization"]["evaluation"] is not True:
            raise ValueError("confirmation is not authorized")
        if args.calibration is not None or args.confirmation:
            raise ValueError("confirmation received unrelated prerequisites")
        if len(args.qualification) != 2 or len(args.retention) != 2:
            raise ValueError("confirmation requires two qualification and two retention passes")
        by_seed = {int(_read(path).get("seed", -1)): path for path in args.qualification}
        retention_by_seed = {int(_read(path).get("seed", -1)): path for path in args.retention}
        if set(by_seed) != {screen, replication} or set(retention_by_seed) != {screen, replication}:
            raise ValueError("confirmation prerequisites do not contain both frozen seeds")
        screen_value = _require_decision(
            by_seed[screen], "qualification", screen, positive=True,
            config=config, config_path=config_path,
        )
        replication_value = _require_decision(
            by_seed[replication], "qualification", replication, positive=False,
            config=config, config_path=config_path,
        )
        screen_retention = _require_retention(
            retention_by_seed[screen], screen, config=config, config_path=config_path
        )
        replication_retention = _require_retention(
            retention_by_seed[replication], replication,
            config=config, config_path=config_path,
        )
        inputs = [*args.qualification, *args.retention]
        claims = [
            _claim("decision", by_seed[screen]),
            _claim("decision", by_seed[replication]),
            _claim("retention", retention_by_seed[screen]),
            _claim("retention", retention_by_seed[replication]),
        ]
    else:
        if args.calibration is not None or args.qualification or args.retention:
            raise ValueError("final stage accepts only confirmation decisions")
        if len(args.confirmation) != 2:
            raise ValueError("final authorization requires both confirmation decisions")
        by_seed = {int(_read(path).get("seed", -1)): path for path in args.confirmation}
        if set(by_seed) != {screen, replication}:
            raise ValueError("final prerequisites do not contain both frozen seeds")
        screen_value = _require_decision(
            by_seed[screen], "confirmation", screen, positive=False,
            config=config, config_path=config_path,
        )
        replication_value = _require_decision(
            by_seed[replication], "confirmation", replication, positive=False,
            config=config, config_path=config_path,
        )
        inputs = list(args.confirmation)
        claims = [
            _claim("decision", by_seed[screen]),
            _claim("decision", by_seed[replication]),
        ]
    config_sha256 = _sha(config_path)
    for path in inputs:
        value = _read(path)
        if (
            value.get("experiment_id") != config["experiment_id"]
            or value.get("config_sha256") != config_sha256
        ):
            raise ValueError(f"{path} is not bound to the current experiment config")
    receipt = {
        "schema_version": 3,
        "experiment_id": config["experiment_id"],
        "authorized_stage": args.stage,
        "config_sha256": config_sha256,
        "issuer_git_commit": git_commit(),
        "issuer_script_sha256": _sha(Path(__file__).resolve()),
        "prerequisites": claims,
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
