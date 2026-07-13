#!/usr/bin/env python3
"""Model-free scaffold receipt for the residual answer-seam factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
CONFIG = EXP / "configs" / "default.yaml"
SMOKE = EXP / "runs" / "smoke" / "summary.json"
DESIGN_SMOKE = EXP / "runs" / "design_v2" / "smoke.json"
CONSTRUCTION = EXP / "runs" / "construction" / "summary.json"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LEGACY_SMOKE_SHA256 = "fe03944be013af43706b2d072dc4953f9a4022065bf3f43f7ed9f15092c56593"


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoke() -> dict[str, object]:
    if not SMOKE.is_file() or SMOKE.is_symlink() or sha256_file(SMOKE) != LEGACY_SMOKE_SHA256:
        raise RuntimeError("the append-only scaffold-v1 receipt changed")
    summary = json.loads(SMOKE.read_text())
    if summary["interface_arms"] != [
        "current_forced_close_freeform",
        "no_think_short_structured",
        "forced_close_program_slot",
    ]:
        raise RuntimeError("the scaffold-v1 arm inventory changed")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def design_smoke() -> dict[str, object]:
    config = yaml.safe_load(CONFIG.read_text())
    model = config["model"]
    if model["id"] != MODEL_ID or model["revision"] != MODEL_REVISION:
        raise RuntimeError("the one-model boundary changed")
    arms = config["interface"]["arms"]
    expected_arms = [
        "think512_freeform",
        "think512_program_slot",
        "no_think_freeform",
        "no_think_program_slot",
    ]
    if arms != expected_arms or config["interface"]["fixed_winner_priority"] != expected_arms:
        raise RuntimeError("the complete 2x2 interface or fixed priority changed")
    calibration = config["interface"]["calibration"]
    if calibration != {
        "rows": 48,
        "suffix_rows": 24,
        "direct_rows": 24,
        "exact_echo_successes_min": 44,
        "parse_successes_min": 44,
        "answer_cap_contacts_max": 2,
        "each_arity_exact_successes_min": 22,
        "each_arity_parse_successes_min": 22,
        "each_arity_answer_cap_contacts_max": 1,
    }:
        raise RuntimeError("the integer calibration gates changed")
    boundaries = config["boundaries"]
    for key in (
        "hidden_files_read",
        "qualification_files_read",
        "confirmation_files_read",
        "benchmark_files_read",
    ):
        if boundaries[key] != []:
            raise RuntimeError(f"forbidden read boundary changed: {key}")
    if not CONSTRUCTION.is_file() or CONSTRUCTION.is_symlink():
        raise RuntimeError("design-v2 smoke requires frozen fresh construction")
    summary = {
        "schema_version": 2,
        "stage": "design_v2_smoke",
        "decision": "DESIGN_V2_MODEL_FREE_PASS",
        "supersedes_scaffold_for_future_implementation": True,
        "legacy_scaffold_receipt_sha256": LEGACY_SMOKE_SHA256,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "config_sha256": sha256_file(CONFIG),
        "construction_summary_sha256": sha256_file(CONSTRUCTION),
        "interface_arms": arms,
        "fixed_winner_priority": config["interface"]["fixed_winner_priority"],
        "calibration_integer_gates": calibration,
        "calibration_tasks": config["data"]["calibration_tasks"],
        "mechanics_tasks": config["data"]["mechanics_tasks"],
        "model_loaded": False,
        "model_calls": 0,
        "sampled_model_outputs": 0,
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
    }
    data = canonical_bytes(summary)
    if DESIGN_SMOKE.exists():
        if DESIGN_SMOKE.is_symlink() or DESIGN_SMOKE.read_bytes() != data:
            raise RuntimeError("frozen design-v2 smoke differs")
    else:
        DESIGN_SMOKE.parent.mkdir(parents=True, exist_ok=True)
        DESIGN_SMOKE.write_bytes(data)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="write or verify the model-free scaffold receipt")
    parser.add_argument(
        "--design-smoke",
        action="store_true",
        help="write or verify the append-only design-v2 receipt",
    )
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return 0
    if args.design_smoke:
        design_smoke()
        return 0
    parser.error("scientific stages remain sealed pending design review")
    return 2


if __name__ == "__main__":
    sys.exit(main())
