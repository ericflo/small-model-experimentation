#!/usr/bin/env python3
"""Fail-closed staged runner for Jacobian counterfactual branching."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from branch_geometry import (  # noqa: E402
    balanced_j_branches,
    geometry_receipt,
    gram_matched_non_j,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def cpu_smoke() -> None:
    config_path = EXP / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
    lens_path = EXP / config["lens"]["path"]
    if sha256(lens_path) != config["lens"]["sha256"]:
        raise RuntimeError("frozen lens hash changed")
    lens = torch.load(lens_path, map_location="cpu", weights_only=True)
    if tuple(lens["concepts"][:12]) != tuple(config["data"]["operation_aliases"].values()):
        raise RuntimeError("public alias/lens concept order changed")
    manifest_path = EXP / "data" / "procedural" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("all_disjoint") is not True:
        raise RuntimeError("fresh data disjointness failed")
    expected = {
        "mechanics": int(config["data"]["mechanics_tasks"]),
        "qualification": int(config["data"]["qualification_tasks"]),
        "confirmation": int(config["data"]["confirmation_tasks"]),
    }
    if {name: int(value["rows"]) for name, value in manifest["splits"].items()} != expected:
        raise RuntimeError("fresh data cardinality changed")
    geometry = {}
    for layer in config["lens"]["band"]:
        directions = lens["directions"][layer]
        if int(torch.linalg.matrix_rank(directions.float()).item()) != 24:
            raise RuntimeError(f"lens layer {layer} lost rank")
        for alpha in config["lens"]["alpha_multipliers"]:
            j = balanced_j_branches(
                directions,
                public_concepts=int(config["lens"]["public_alias_concepts"]),
                target_rms_norm=float(config["lens"]["replicated_median_delta_norms"][layer]) * float(alpha),
            )
            non_j = gram_matched_non_j(
                directions,
                j,
                seed=int(config["seeds"]["non_j_geometry"]) + int(layer),
                rtol=float(config["lens"]["pseudoinverse_rtol"]),
            )
            receipt = geometry_receipt(
                directions, j, non_j, rtol=float(config["lens"]["pseudoinverse_rtol"])
            )
            if receipt["j"]["maximum_sum_abs"] > float(config["controls"]["branch_sum_norm_max"]):
                raise RuntimeError("J branches are not zero sum")
            if receipt["non_j"]["maximum_sum_abs"] > float(config["controls"]["branch_sum_norm_max"]):
                raise RuntimeError("non-J branches are not zero sum")
            if receipt["gram_relative_error"] > float(config["controls"]["gram_relative_error_max"]):
                raise RuntimeError("non-J branch Gram mismatch")
            geometry[f"layer_{layer}_alpha_{alpha}"] = receipt
    result = {
        "schema_version": 1,
        "stage": "cpu-smoke",
        "passed": True,
        "model_loaded": False,
        "outcomes_loaded": False,
        "confirmation_opened": False,
        "lens_sha256": sha256(lens_path),
        "config_sha256": sha256(config_path),
        "data_manifest_sha256": sha256(manifest_path),
        "fresh_tasks": expected,
        "ancestor_unique_fingerprints": manifest["ancestor_unique_fingerprints"],
        "geometry": geometry,
        "downstream_available": False,
    }
    write_json(EXP / "runs" / "smoke" / "cpu.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("smoke", "model-smoke", "mechanics", "qualification", "confirmation"),
    )
    args = parser.parse_args()
    if args.stage == "smoke":
        cpu_smoke()
        return
    raise RuntimeError(f"stage {args.stage!r} is unavailable before implementation boundary")


if __name__ == "__main__":
    main()
