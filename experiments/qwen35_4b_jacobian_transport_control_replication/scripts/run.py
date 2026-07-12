#!/usr/bin/env python3
"""Stage-gated quantization-aware J-transport replication harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from io_utils import read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from task_data import CONCEPTS, fingerprint, generate_replication_splits  # noqa: E402


CONFIG_PATH = EXP / "configs" / "default.yaml"
DATA_DIR = EXP / "data" / "procedural"
RUNS_DIR = EXP / "runs"


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def _parent_fingerprints() -> set[str]:
    fingerprints: set[str] = set()
    parents = (
        ROOT / "experiments" / "qwen35_4b_context_local_jacobian_clamp" / "data" / "procedural",
        ROOT / "experiments" / "qwen35_4b_jacobian_value_transport" / "data" / "procedural",
    )
    for directory in parents:
        for path in sorted(directory.glob("*.jsonl")):
            for row in read_jsonl(path):
                if {"mapping", "source", "target", "wrong"}.issubset(row):
                    fingerprints.add(fingerprint(row))
    return fingerprints


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
    lens_path = EXP / config["lens"]["path"]
    lens_hash = sha256_file(lens_path)
    if lens_hash != config["lens"]["sha256"]:
        raise RuntimeError("frozen parent lens hash mismatch")
    splits = generate_replication_splits(config)
    parent = _parent_fingerprints()
    overlap = {
        name: sorted(fingerprint(row) for row in rows if fingerprint(row) in parent)
        for name, rows in splits.items()
    }
    if any(overlap.values()):
        raise RuntimeError(f"fresh replication rows overlap parent data: {overlap}")
    paths = {}
    for name, rows in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = path
    manifest = {
        "schema_version": 1,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "frozen_lens_sha256": lens_hash,
        "parent_fingerprint_count": len(parent),
        "parent_overlap_count": sum(len(values) for values in overlap.values()),
        "splits": {
            name: {
                "items": len(rows),
                "path": str(paths[name].relative_to(EXP)),
                "sha256": sha256_file(paths[name]),
                "unique_fingerprints": len({fingerprint(row) for row in rows}),
                "source_counts": {
                    concept: sum(row["source"] == concept for row in rows)
                    for concept in CONCEPTS
                },
            }
            for name, rows in splits.items()
        },
        "scientific_result": False,
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    receipt = {
        "schema_version": 1,
        "stage": "cpu_smoke",
        "passed": True,
        "scientific_result": False,
        "lens_sha256": lens_hash,
        "parent_overlap_count": 0,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "band": config["intervention"]["band"],
        "random_arms": config["intervention"]["random_arms"],
        "norm_tolerance": config["intervention"]["norm_relative_tolerance"],
        "projection_tolerance": config["intervention"]["realized_span_projection_max"],
    }
    write_json(RUNS_DIR / "smoke" / "data_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def unavailable(stage: str) -> None:
    raise RuntimeError(f"stage {stage!r} is not implemented; refusing a placeholder result")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "model-smoke", "control-calibration", "confirmation", "full"),
        default="smoke",
    )
    args = parser.parse_args()
    config = load_config()
    if args.stage == "smoke":
        run_smoke(config)
        return 0
    unavailable(args.stage)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
