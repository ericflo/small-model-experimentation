#!/usr/bin/env python3
"""Fail-closed staged harness for semantic-anchor coordinate branching."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
CONFIG = EXP / "configs" / "default.yaml"
sys.path.insert(0, str(EXP / "src"))

from task_data import (  # noqa: E402
    behavior_fingerprint,
    build_splits,
    public_mechanics,
    task_fingerprint,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def ancestor_behavior_fingerprints() -> set[str]:
    """Read experiment procedural artifacts only; benchmark contents stay forbidden."""

    values: set[str] = set()
    for path in sorted((ROOT / "experiments").glob("*/data/procedural/*.jsonl")):
        if EXP in path.parents:
            continue
        for line in path.read_text().splitlines():
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if {"depth", "visible", "hidden", "first_op"}.issubset(row):
                values.add(behavior_fingerprint(row))
    return values


def validate_design_boundary(config: dict) -> dict:
    boundary = config["design_boundary"]
    if boundary.get("status") != "anchored":
        raise RuntimeError("scientific design boundary is not anchored")
    commit = str(boundary["commit"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    paths = {
        "readme_sha256": EXP / "README.md",
        "preregistration_sha256": EXP / "reports" / "preregistration.md",
        "design_review_sha256": EXP / "reports" / "design_review.md",
        "data_manifest_sha256": EXP / "data" / "procedural" / "manifest.json",
        "mechanics_public_sha256": EXP / "data" / "procedural" / "mechanics_public.jsonl",
        "lens_sha256": EXP / "assets" / "context_lens.pt",
    }
    expected = {key: str(value) for key, value in boundary["hashes"].items()}
    local = {key: sha256(path) for key, path in paths.items()}
    committed = {}
    for key, path in paths.items():
        relative = path.relative_to(ROOT).as_posix()
        content = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        committed[key] = hashlib.sha256(content).hexdigest()
    if not ancestor or local != expected or committed != expected:
        raise RuntimeError("immutable scientific design boundary changed")
    return {
        "passed": True,
        "commit": commit,
        "design_is_ancestor": ancestor,
        "hashes": expected,
    }


def diagnostic_results(values: list[int], k: int) -> dict[str, list[int]]:
    """Frozen one-step mechanics consequences, independent of task gold."""

    shift = k % len(values)
    return {
        "reverse": values[::-1],
        "sort_asc": sorted(values),
        "sort_desc": sorted(values, reverse=True),
        "abs_all": [abs(value) for value in values],
        "square": [value * value for value in values],
        "negate": [-value for value in values],
        "running_sum": [sum(values[: index + 1]) for index in range(len(values))],
        "adjacent_diff": [
            values[index + 1] - values[index] for index in range(len(values) - 1)
        ],
        "add_k": [value + k for value in values],
        "mul_k": [value * k for value in values],
        "take_k": values[:k],
        "rotate_k": values[shift:] + values[:shift],
    }


def validate_config(config: dict) -> dict:
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
    aliases = tuple(config["data"]["alias_tokens"])
    operations = tuple(config["data"]["operation_names"])
    labels = tuple(config["anchor"]["result_labels"])
    if len(aliases) != 12 or len(set(aliases)) != 12:
        raise RuntimeError("anchor requires 12 unique aliases")
    if len(operations) != 12 or len(set(operations)) != 12:
        raise RuntimeError("anchor requires 12 unique operations")
    if len(labels) != 12 or len(set(labels)) != 12 or set(labels) & set(aliases):
        raise RuntimeError("result labels must be 12 unique non-alias concepts")
    if tuple(config["lens"]["band"]) != (4, 5, 6, 7, 8):
        raise RuntimeError("frozen intervention band changed")
    results = diagnostic_results(
        [int(value) for value in config["anchor"]["diagnostic_input"]],
        int(config["anchor"]["diagnostic_parameter"]),
    )
    if tuple(results) != operations or len({tuple(value) for value in results.values()}) != 12:
        raise RuntimeError("diagnostic consequences are incomplete or non-unique")
    return {
        "aliases": aliases,
        "operations": operations,
        "result_labels": labels,
        "diagnostic_results": results,
    }


def smoke() -> dict:
    config = yaml.safe_load(CONFIG.read_text())
    validated = validate_config(config)
    design = validate_design_boundary(config)
    lens_path = EXP / config["lens"]["path"]
    observed = sha256(lens_path)
    if observed != config["lens"]["sha256"]:
        raise RuntimeError("frozen context lens hash changed")
    lens = torch.load(lens_path, map_location="cpu", weights_only=True)
    concepts = tuple(str(value) for value in lens["concepts"])
    if concepts[:12] != validated["aliases"]:
        raise RuntimeError("public alias order differs from frozen lens")
    if concepts[12:] != validated["result_labels"]:
        raise RuntimeError("result-label order differs from frozen lens")
    band = tuple(int(value) for value in config["lens"]["band"])
    if not set(band).issubset(int(value) for value in lens["source_layers"]):
        raise RuntimeError("frozen lens lacks an intervention layer")
    ranks = {
        str(layer): int(torch.linalg.matrix_rank(lens["directions"][layer].float()).item())
        for layer in band
    }
    if any(rank != 24 for rank in ranks.values()):
        raise RuntimeError("frozen context lens lost full concept rank")
    splits = build_splits(config)
    ancestors = ancestor_behavior_fingerprints()
    overlap = {
        split: sorted(
            behavior_fingerprint(task)
            for task in rows
            if behavior_fingerprint(task) in ancestors
        )
        for split, rows in splits.items()
    }
    if any(overlap.values()):
        raise RuntimeError(f"fresh task behavior overlaps ancestor data: {overlap}")
    data_dir = EXP / "data" / "procedural"
    paths = {}
    for split, rows in splits.items():
        path = data_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        paths[split] = path
    public_rows = [public_mechanics(task) for task in splits["mechanics"]]
    public_path = data_dir / "mechanics_public.jsonl"
    write_jsonl(public_path, public_rows)
    allowed_public = {
        "task_id", "visible", "alias_to_operation", "source_alias",
        "result_label_by_operation",
    }
    if any(set(row) != allowed_public for row in public_rows):
        raise RuntimeError("mechanics public schema leaks a sealed task field")
    manifest = {
        "schema_version": 1,
        "seed": int(config["seeds"]["split"]),
        "ancestor_behavior_fingerprints": len(ancestors),
        "ancestor_overlap_count": 0,
        "all_disjoint": True,
        "total_new_unique_behaviors": len({
            behavior_fingerprint(task) for rows in splits.values() for task in rows
        }),
        "total_new_unique_fingerprints": len({
            task_fingerprint(task) for rows in splits.values() for task in rows
        }),
        "splits": {
            split: {
                "rows": len(rows),
                "path": str(paths[split].relative_to(ROOT)),
                "sha256": sha256(paths[split]),
                "unique_behaviors": len({behavior_fingerprint(task) for task in rows}),
                "unique_fingerprints": len({task_fingerprint(task) for task in rows}),
            }
            for split, rows in splits.items()
        },
        "mechanics_public": {
            "rows": len(public_rows),
            "path": str(public_path.relative_to(ROOT)),
            "sha256": sha256(public_path),
            "fields": sorted(allowed_public),
            "sealed_fields_present": False,
        },
        "benchmarks_read": False,
        "scientific_result": False,
    }
    write_json(data_dir / "manifest.json", manifest)
    result = {
        "schema_version": 1,
        "stage": "cpu_smoke",
        "passed": True,
        "model_loaded": False,
        "outcomes_loaded": False,
        "benchmarks_read": False,
        "lens_sha256": observed,
        "lens_rank_by_layer": ranks,
        "aliases": list(validated["aliases"]),
        "result_labels": list(validated["result_labels"]),
        "diagnostic_results": validated["diagnostic_results"],
        "mechanics_rows_planned": int(config["data"]["mechanics_tasks"]) * 11,
        "fresh_task_manifest_sha256": sha256(data_dir / "manifest.json"),
        "ancestor_behavior_fingerprints": len(ancestors),
        "ancestor_overlap_count": 0,
        "fresh_split_rows": {split: len(rows) for split, rows in splits.items()},
        "design_boundary": design,
        "implementation_boundary_status": config["implementation_boundary"]["status"],
        "downstream_available": False,
    }
    path = EXP / "runs" / "smoke" / "cpu.json"
    write_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("smoke", "model-smoke", "mechanics", "qualification", "confirmation"),
    )
    args = parser.parse_args()

    if args.stage == "smoke":
        smoke()
        return 0
    raise RuntimeError(
        f"stage {args.stage!r} is unavailable before an audited implementation boundary"
    )


if __name__ == "__main__":
    sys.exit(main())
