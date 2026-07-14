#!/usr/bin/env python3
"""Fail-closed design-stage harness for the search-scaffold experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
REQUIRED_DESIGN_FILES = (
    "README.md",
    "idea_intake.md",
    "configs/default.yaml",
    "reports/artifact_manifest.yaml",
    "reports/report.md",
)


def smoke() -> None:
    missing = [name for name in REQUIRED_DESIGN_FILES if not (EXP / name).is_file()]
    if missing:
        raise RuntimeError(f"missing design files: {missing}")
    config = (EXP / "configs/default.yaml").read_text()
    required = (
        f"model_id: {MODEL_ID}",
        "construction_seed: 77111",
        "training_seed: 45",
        "local_seed: 88007",
        "conditional_aggregate_seed: 78137",
        "status: design_feasibility_only",
    )
    absent = [item for item in required if item not in config]
    if absent:
        raise RuntimeError(f"design config is incomplete: {absent}")
    if "benchmark" in config.lower():
        raise RuntimeError("design-stage config must not contain a benchmark path")
    print("design smoke passed: fixed model and fresh seeds; scientific stages remain locked")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="check the scaffold without running the full experiment")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return 0
    parser.error("scientific stages are locked until feasibility and design review pass")
    return 2


if __name__ == "__main__":
    sys.exit(main())
