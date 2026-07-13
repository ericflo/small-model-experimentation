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
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoke() -> dict[str, object]:
    config = yaml.safe_load(CONFIG.read_text())
    if config["model"] != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": "bfloat16",
    }:
        raise RuntimeError("the one-model boundary changed")
    arms = config["interface_arms"]
    if arms != [
        "current_forced_close_freeform",
        "no_think_short_structured",
        "forced_close_program_slot",
    ] or len(set(arms)) != 3:
        raise RuntimeError("the interface factorial changed")
    gates = config["interface_gates"]
    if gates != {
        "exact_echo_rate_min": 0.90,
        "parse_rate_min": 0.90,
        "cap_contact_rate_max": 0.05,
    }:
        raise RuntimeError("the echo-first interface gates changed")
    boundaries = config["boundaries"]
    if not boundaries["calibration_disjoint_from_mechanics"]:
        raise RuntimeError("calibration/mechanics isolation is required")
    for key in (
        "hidden_files_read",
        "qualification_files_read",
        "confirmation_files_read",
        "benchmark_files_read",
    ):
        if boundaries[key] != []:
            raise RuntimeError(f"forbidden read boundary changed: {key}")
    summary = {
        "schema_version": 1,
        "stage": "scaffold_smoke",
        "decision": "SCAFFOLD_PASS",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "config_sha256": sha256_file(CONFIG),
        "interface_arms": arms,
        "interface_gates": gates,
        "calibration_tasks": boundaries["calibration_tasks"],
        "mechanics_tasks": boundaries["mechanics_tasks"],
        "calibration_disjoint_from_mechanics": True,
        "model_loaded": False,
        "model_calls": 0,
        "sampled_model_outputs": 0,
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
    }
    data = canonical_bytes(summary)
    if SMOKE.exists():
        if SMOKE.is_symlink() or SMOKE.read_bytes() != data:
            raise RuntimeError("frozen scaffold receipt differs")
    else:
        SMOKE.parent.mkdir(parents=True, exist_ok=True)
        SMOKE.write_bytes(data)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="write or verify the model-free scaffold receipt")
    args = parser.parse_args()

    if args.smoke:
        smoke()
        return 0
    parser.error("scientific stages remain sealed pending design review")
    return 2


if __name__ == "__main__":
    sys.exit(main())
