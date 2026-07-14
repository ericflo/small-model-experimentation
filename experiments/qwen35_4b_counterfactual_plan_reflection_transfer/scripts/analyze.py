#!/usr/bin/env python3
"""Apply preregistered qualification/confirmation gates to scored arm bundles."""

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

from analyze import evaluate_positive_control, evaluate_seed_block  # noqa: E402
from firewall import install_benchmark_firewall  # noqa: E402

install_benchmark_firewall(EXP.parents[1])


def _rows(paths: list[Path]) -> list[dict]:
    result = []
    for path in paths:
        result.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return result


def _write_exclusive(path: Path, value: object) -> str:
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=("qualification", "confirmation"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    if args.seed not in set(config["training"]["staged_seeds"].values()):
        raise SystemExit("seed is not preregistered")
    rows = _rows(args.scores)
    if {row["split"] for row in rows} != {args.block}:
        raise ValueError("score bundle contains the wrong evaluation block")
    decision = config["decision_gates"]
    thresholds = dict(decision["per_seed_qualification_and_confirmation"])
    thresholds["reflection_specific_correct_minus_auxiliary_min"] = decision[
        "reflection_specific_mechanism"
    ]["correct_minus_auxiliary_min"]
    capability = evaluate_seed_block(rows, thresholds, decision["bootstrap"])
    result = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "block": args.block,
        "seed": args.seed,
        "capability": capability,
        "score_file_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.scores
        },
    }
    positive_arm = "direct_plan_answer_positive_control_action"
    if args.seed == config["training"]["staged_seeds"]["screen"] and any(
        row["arm"] == positive_arm for row in rows
    ):
        result["positive_control"] = evaluate_positive_control(
            rows,
            decision["positive_control_sanity_on_qualification"],
            decision["bootstrap"],
        )
    digest = _write_exclusive(args.output, result)
    print(
        json.dumps(
            {
                "output_sha256": digest,
                "capability_pass": capability["capability_pass"],
                "reflection_specific_pass": capability["reflection_specific_pass"],
                "positive_control_pass": result.get("positive_control", {}).get("pass"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
