#!/usr/bin/env python3
"""Stage-gated native-thought Jacobian value-transport harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from task_data import (  # noqa: E402
    IDENTIFIABLE_FIRST_OPERATIONS,
    build_splits,
    task_fingerprint,
)


CONFIG_PATH = EXP / "configs" / "default.yaml"
DATA_DIR = EXP / "data" / "procedural"
RUNS_DIR = EXP / "runs"


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parent_fingerprints() -> set[str]:
    directory = ROOT / "experiments" / "qwen35_4b_jacobian_value_transport" / "data" / "procedural"
    values = set()
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if {"depth", "visible", "hidden", "first_op"}.issubset(row):
                values.add(task_fingerprint(row))
    return values


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
    lens_path = EXP / config["lens"]["path"]
    observed_lens_hash = sha256_file(lens_path)
    if observed_lens_hash != config["lens"]["sha256"]:
        raise RuntimeError("frozen replicated lens hash mismatch")
    import torch

    lens_state = torch.load(lens_path, map_location="cpu", weights_only=True)
    lens_concepts = tuple(str(value) for value in lens_state["concepts"])
    aliases = dict(config["data"]["operation_aliases"])
    if (
        len(set(aliases.values())) != len(aliases)
        or not set(aliases.values()).issubset(lens_concepts)
    ):
        raise RuntimeError("operation aliases must be unique frozen-lens concepts")
    splits = build_splits(config)
    parent = _parent_fingerprints()
    all_fingerprints: list[str] = []
    split_receipts = {}
    for split, rows in splits.items():
        path = DATA_DIR / f"{split}.jsonl"
        write_jsonl(path, rows)
        fingerprints = [task_fingerprint(row) for row in rows]
        all_fingerprints.extend(fingerprints)
        counts = Counter(str(row["first_op"]) for row in rows)
        split_receipts[split] = {
            "items": len(rows),
            "path": str(path.relative_to(EXP)),
            "sha256": sha256_file(path),
            "unique_fingerprints": len(set(fingerprints)),
            "parent_overlap": sum(value in parent for value in fingerprints),
            "first_op_counts": {
                name: counts[name] for name in IDENTIFIABLE_FIRST_OPERATIONS
            },
        }
    if len(all_fingerprints) != len(set(all_fingerprints)):
        raise RuntimeError("procedural split fingerprints overlap")
    if any(receipt["parent_overlap"] for receipt in split_receipts.values()):
        raise RuntimeError("fresh tasks overlap the direct Jacobian parent")
    manifest = {
        "schema_version": 1,
        "scientific_result": False,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "frozen_lens_sha256": observed_lens_hash,
        "frozen_lens_concepts": len(lens_concepts),
        "operation_aliases": aliases,
        "identifiable_first_operations": list(IDENTIFIABLE_FIRST_OPERATIONS),
        "parent_fingerprint_count": len(parent),
        "total_unique_fingerprints": len(set(all_fingerprints)),
        "splits": split_receipts,
        "firewall": {
            "benchmark_content_used": False,
            "fresh_procedural_only": True,
            "visible_first_type_identifiable_by_exhaustive_enumeration": True,
        },
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    result = {
        "schema_version": 1,
        "stage": "cpu_smoke",
        "passed": True,
        "scientific_result": False,
        "lens_sha256": observed_lens_hash,
        "items": len(all_fingerprints),
        "unique_fingerprints": len(set(all_fingerprints)),
        "parent_overlap": 0,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "first_target_types": len(IDENTIFIABLE_FIRST_OPERATIONS),
        "alias_count": len(aliases),
    }
    write_json(RUNS_DIR / "smoke" / "data_receipt.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def unavailable(stage: str) -> None:
    raise RuntimeError(f"stage {stage!r} is not implemented; refusing a placeholder result")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "smoke",
            "model-smoke",
            "seam-calibration",
            "prefix-value",
            "control-calibration",
            "causal-confirmation",
        ),
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
