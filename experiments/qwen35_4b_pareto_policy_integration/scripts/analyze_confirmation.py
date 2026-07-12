#!/usr/bin/env python3
"""Analyze the preregistered two-block Pareto-integration confirmation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, sha256_file, write_json  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell(row: dict) -> str:
    return f"{row['stratum']}/{row['family']}/{row['kind']}/L{int(row['level'])}"


def _item_map(payload: dict) -> dict[str, dict]:
    rows = {str(row["key"]): row for row in payload["items"]}
    if len(rows) != len(payload["items"]):
        raise ValueError("duplicate confirmation item keys")
    return rows


def _pooled(payloads: list[dict]) -> dict:
    items = []
    for block_index, payload in enumerate(payloads):
        for row in payload["items"]:
            copied = dict(row)
            copied["key"] = f"b{block_index}/{row['key']}"
            copied["block_index"] = block_index
            items.append(copied)
    return {"items": items}


def _routed(quick: dict, deep: dict) -> dict:
    quick_map = _item_map(quick)
    deep_map = _item_map(deep)
    if set(quick_map) != set(deep_map):
        raise ValueError("cannot route mismatched source-policy items")
    return {
        "items": [
            dict(
                quick_map[key] if quick_map[key]["stratum"] == "quick" else deep_map[key]
            )
            for key in sorted(quick_map)
        ]
    }


def _joint_macro(rows: list[tuple[dict, float]]) -> float:
    cells: dict[str, list[float]] = defaultdict(list)
    for row, value in rows:
        cells[_cell(row)].append(float(value))
    by_stratum: dict[str, list[float]] = defaultdict(list)
    for cell, values in cells.items():
        by_stratum[cell.split("/", 1)[0]].append(sum(values) / len(values))
    if set(by_stratum) != {"quick", "deep"}:
        raise ValueError(f"joint macro requires both strata, got {sorted(by_stratum)}")
    return sum(sum(values) / len(values) for values in by_stratum.values()) / 2.0


def _paired_rows(
    preferred: dict, comparator: dict, allowed_cells: set[str] | None = None
) -> list[tuple[dict, float]]:
    left = _item_map(preferred)
    right = _item_map(comparator)
    if set(left) != set(right):
        raise ValueError("paired confirmation item mismatch")
    return [
        (left[key], float(left[key]["score"]) - float(right[key]["score"]))
        for key in sorted(left)
        if allowed_cells is None or _cell(left[key]) in allowed_cells
    ]


def _bootstrap_lcb(
    rows: list[tuple[dict, float]], *, samples: int, confidence: float, seed: int
) -> float:
    cells: dict[str, list[float]] = defaultdict(list)
    for row, value in rows:
        cells[_cell(row)].append(float(value))
    by_stratum: dict[str, list[list[float]]] = defaultdict(list)
    for cell, values in cells.items():
        by_stratum[cell.split("/", 1)[0]].append(values)
    if set(by_stratum) != {"quick", "deep"}:
        raise ValueError("bootstrap requires quick and deep capability cells")
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        stratum_draws = []
        for stratum in ("quick", "deep"):
            cell_draws = [
                sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                for values in by_stratum[stratum]
            ]
            stratum_draws.append(sum(cell_draws) / len(cell_draws))
        draws.append(sum(stratum_draws) / 2.0)
    draws.sort()
    index = max(0, min(samples - 1, math.floor((1.0 - confidence) * samples)))
    return float(draws[index])


def _comparison(
    preferred_blocks: list[dict], comparator_blocks: list[dict],
    *, capability_cells: set[str], samples: int, confidence: float, seed: int,
) -> dict:
    pooled_rows = _paired_rows(
        _pooled(preferred_blocks), _pooled(comparator_blocks), capability_cells
    )
    block_deltas = [
        _joint_macro(_paired_rows(left, right, capability_cells))
        for left, right in zip(preferred_blocks, comparator_blocks)
    ]
    mean = _joint_macro(pooled_rows)
    lcb = _bootstrap_lcb(
        pooled_rows, samples=samples, confidence=confidence, seed=seed
    )
    return {
        "n": len(pooled_rows), "paired_joint_macro_delta": mean,
        "one_sided_lcb": lcb, "block_deltas": block_deltas,
        "positive_mean": mean > 0.0, "positive_lcb": lcb > 0.0,
    }


def _retention(
    primary: dict, routed: dict, retention_cells: set[str], maximum_regression: float
) -> dict:
    rows = _paired_rows(primary, routed, retention_cells)
    by_cell: dict[str, list[float]] = defaultdict(list)
    for row, delta in rows:
        by_cell[_cell(row)].append(delta)
    expected = retention_cells
    if set(by_cell) != expected:
        raise ValueError(
            f"confirmation retention-cell mismatch: "
            f"missing={sorted(expected - set(by_cell))[:5]}"
        )
    cells = {
        cell: {
            "n": len(values), "mean_delta": sum(values) / len(values),
            "maximum_allowed_regression": maximum_regression,
            "passed": sum(values) / len(values) >= -maximum_regression,
        }
        for cell, values in sorted(by_cell.items())
    }
    return {"cells": cells, "passed": all(row["passed"] for row in cells.values())}


def _score(payload: dict, cells: set[str] | None = None) -> float:
    return _joint_macro([
        (row, float(row["score"])) for row in payload["items"]
        if cells is None or _cell(row) in cells
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--calibration", type=Path, default=EXP / "analysis" / "calibration.json"
    )
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "confirmation.json"
    )
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    manifest = _load(args.manifest)
    calibration = _load(args.calibration)
    capability_cells = set(calibration["capability_cells"])
    retention_cells = set(calibration["retention_cells"])
    expected_seeds = [int(value) for value in config["seeds"]["confirmatory_blocks"]]
    arms = {
        name: [_load(Path(path)) for path in paths]
        for name, paths in manifest["arms"].items()
    }
    protocol = {
        "manifest_config_matches": manifest.get("config_sha256") == sha256_file(config_path),
        "manifest_blocks_match": manifest.get("block_seeds") == expected_seeds,
        "calibration_gate_passed": bool(calibration.get("gate", {}).get("passed")),
        "all_arms_have_two_blocks": all(len(payloads) == len(expected_seeds) for payloads in arms.values()),
        "all_block_seeds_match": all(
            [int(payload["block_seed"]) for payload in payloads] == expected_seeds
            for payloads in arms.values()
        ),
        "all_confirmatory_scope": all(
            payload.get("scope") == "confirmatory"
            for payloads in arms.values() for payload in payloads
        ),
        "greedy_except_sample_more": all(
            payload.get("decode") == ("sample8" if name == manifest["sample_more_arm"] else "greedy")
            for name, payloads in arms.items() for payload in payloads
        ),
        "calibration_roles_partition_cells": (
            capability_cells.isdisjoint(retention_cells)
            and capability_cells | retention_cells == set(calibration["cells"])
        ),
    }
    # Every arm must evaluate the exact same procedural item keys in each block.
    key_sets = []
    for block_index in range(len(expected_seeds)):
        key_sets.append({
            tuple(sorted(_item_map(payloads[block_index]))) for payloads in arms.values()
        })
    protocol["all_arms_paired"] = all(len(values) == 1 for values in key_sets)

    primary_name = manifest["primary_arm"]
    primary = arms[primary_name]
    samples = int(config["evaluation"]["paired_bootstrap_samples"])
    confidence = float(config["evaluation"]["confidence"])
    strict_comparators = [*manifest["source_arms"], *manifest["control_arms"]]
    comparisons = {
        name: _comparison(
            primary, arms[name], capability_cells=capability_cells,
            samples=samples, confidence=confidence, seed=11000 + index,
        )
        for index, name in enumerate(strict_comparators)
    }
    sample_more_name = manifest["sample_more_arm"]
    sample_more = _comparison(
        primary, arms[sample_more_name], capability_cells=capability_cells,
        samples=samples, confidence=confidence, seed=12000,
    )
    replicate_results = {}
    for index, name in enumerate(manifest["replicate_arms"]):
        replicate_results[name] = {
            source: _comparison(
                arms[name], arms[source], capability_cells=capability_cells,
                samples=samples, confidence=confidence, seed=13000 + index * 10 + source_index,
            )
            for source_index, source in enumerate(manifest["source_arms"])
        }

    routed_blocks = [
        _routed(arms[manifest["quick_arm"]][index], arms[manifest["deep_arm"]][index])
        for index in range(len(expected_seeds))
    ]
    retention = _retention(
        _pooled(primary), _pooled(routed_blocks), retention_cells,
        float(config["decision"]["maximum_retention_regression"]),
    )
    score_table = {
        name: {
            "capability_joint_macro": _score(_pooled(payloads), capability_cells),
            "all_cell_joint_macro": _score(_pooled(payloads)),
        }
        for name, payloads in arms.items()
    }
    score_table["routed_two_checkpoint_reference"] = {
        "capability_joint_macro": _score(_pooled(routed_blocks), capability_cells),
        "all_cell_joint_macro": _score(_pooled(routed_blocks)),
    }
    checks = {
        "protocol": all(protocol.values()),
        "primary_positive_lcb_vs_each_source_and_control": all(
            row["positive_mean"] and row["positive_lcb"] for row in comparisons.values()
        ),
        # Replication is directional rather than a seed-selection contest: all
        # preregistered training seeds must remain above both source policies.
        "every_replication_positive_vs_both_sources": all(
            row["positive_mean"]
            for comparisons_by_source in replicate_results.values()
            for row in comparisons_by_source.values()
        ),
        "retention": retention["passed"],
        "primary_above_sample_more": sample_more["positive_mean"],
    }
    result = {
        "stage": "two_block_confirmatory_analysis", "config": str(config_path),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "primary_arm": primary_name, "protocol_checks": protocol,
        "capability_cells": sorted(capability_cells),
        "retention_cells": sorted(retention_cells),
        "strict_comparisons": comparisons,
        "sample_more_comparison": sample_more,
        "replicate_comparisons": replicate_results,
        "retention": retention, "score_table": score_table,
        "checks": checks, "gate": {"passed": all(checks.values())},
        "downstream_authorization": (
            "benchmark_cli" if all(checks.values()) else "stop_before_benchmark_cli"
        ),
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
