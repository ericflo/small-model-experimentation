#!/usr/bin/env python3
"""Apply the frozen pre-training action-interface/headroom calibration gate."""

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

from analyze import evaluate_calibration  # noqa: E402
from firewall import install_benchmark_firewall  # noqa: E402
from eval_inputs import task_metadata  # noqa: E402

install_benchmark_firewall(EXP.parents[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    rows = [
        json.loads(line) for line in args.scores.read_text().splitlines() if line.strip()
    ]
    if {row["split"] for row in rows} != {"calibration"}:
        raise ValueError("calibration score bundle has the wrong split")
    if any(row["arm"] != "frozen_action" or row.get("training_seed") is not None for row in rows):
        raise ValueError("calibration must contain only receipt-bound frozen rows")
    expected_task_metadata = task_metadata(config, "calibration")
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(
            (EXP / "configs" / "default.yaml").read_bytes()
        ).hexdigest(),
        "gate": evaluate_calibration(
            rows,
            config["decision_gates"]["calibration_before_training"],
            expected_task_metadata,
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
