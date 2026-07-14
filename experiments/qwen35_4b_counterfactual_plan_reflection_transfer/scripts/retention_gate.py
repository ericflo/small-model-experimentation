#!/usr/bin/env python3
"""Apply exact depth/family retention non-inferiority gates to one adapter seed."""

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

from analyze import evaluate_retention  # noqa: E402
from firewall import install_benchmark_firewall  # noqa: E402
from taskgen import FAMILIES, build_retention_corpus  # noqa: E402

install_benchmark_firewall(EXP.parents[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    if args.arm not in {
        "reflection_correct_action",
        "reflection_shuffled_action",
        "auxiliary_plan_label_correct_action",
    }:
        raise ValueError("retention arm is not preregistered")
    if args.seed not in set(config["training"]["staged_seeds"].values()):
        raise ValueError("retention seed is not preregistered")
    stage_receipt = json.loads(args.stage_receipt.read_text())
    expected_stage = (
        "screen_training"
        if args.seed == config["training"]["staged_seeds"]["screen"]
        else "replication_training"
    )
    config_path = EXP / "configs" / "default.yaml"
    if (
        stage_receipt.get("experiment_id") != config["experiment_id"]
        or stage_receipt.get("authorized_stage") != expected_stage
        or stage_receipt.get("config_sha256")
        != hashlib.sha256(config_path.read_bytes()).hexdigest()
    ):
        raise ValueError("stage receipt does not authorize this retention seed")
    rows = []
    for path in args.scores:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if {row["split"] for row in rows} != {"retention"}:
        raise ValueError("retention score bundle has the wrong split")
    for row in rows:
        expected_seed = None if row["arm"] == "frozen_action" else args.seed
        if row.get("training_seed") != expected_seed:
            raise ValueError("retention score seed differs from the requested adapter seed")
    construction = config["construction"]
    tasks = build_retention_corpus(
        int(construction["per_family"]["retention_per_family_per_depth"]),
        int(construction["retention_seed"]),
    )
    thresholds = config["decision_gates"]["retention_noninferiority"]
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "arm": args.arm,
        "seed": args.seed,
        "stage_receipt_sha256": hashlib.sha256(args.stage_receipt.read_bytes()).hexdigest(),
        "gate": evaluate_retention(
            rows,
            args.arm,
            depth_min=float(thresholds["each_depth_delta_min"]),
            family_min=float(thresholds["each_family_delta_min"]),
            expected_task_ids={task["task_id"] for task in tasks},
            expected_families={family.name for family in FAMILIES},
        ),
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
