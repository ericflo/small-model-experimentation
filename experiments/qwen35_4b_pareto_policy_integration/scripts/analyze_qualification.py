#!/usr/bin/env python3
"""Qualify a complementary quick/deep policy pair with no effect-size floor."""

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

from io_utils import load_config, write_json  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _item_map(payload: dict) -> dict[str, dict]:
    rows = {str(row["key"]): row for row in payload["items"]}
    if len(rows) != len(payload["items"]):
        raise ValueError("duplicate evaluation item keys")
    return rows


def _cell(row: dict) -> tuple[str, str, int]:
    return str(row["family"]), str(row["kind"]), int(row["level"])


def _cell_key(row: dict) -> str:
    return f"{row['stratum']}/{row['family']}/{row['kind']}/L{int(row['level'])}"


def _macro(rows: list[tuple[dict, float]]) -> float:
    cells: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row, value in rows:
        cells[_cell(row)].append(float(value))
    if not cells:
        raise ValueError("empty paired stratum")
    return sum(sum(values) / len(values) for values in cells.values()) / len(cells)


def stratified_bootstrap_lcb(
    rows: list[tuple[dict, float]], *, samples: int, confidence: float, seed: int
) -> float:
    """One-sided lower bound for an equal-cell macro of paired deltas."""
    if samples < 100:
        raise ValueError("bootstrap requires at least 100 samples")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1)")
    cells: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row, value in rows:
        cells[_cell(row)].append(float(value))
    rng = random.Random(seed)
    draws = []
    groups = list(cells.values())
    for _ in range(samples):
        cell_means = [
            sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            for values in groups
        ]
        draws.append(sum(cell_means) / len(cell_means))
    draws.sort()
    index = max(0, min(samples - 1, math.floor((1.0 - confidence) * samples)))
    return float(draws[index])


def paired_advantage(
    preferred: dict,
    comparator: dict,
    *,
    stratum: str,
    samples: int,
    confidence: float,
    seed: int,
    allowed_cells: set[str] | None = None,
) -> dict:
    left = _item_map(preferred)
    right = _item_map(comparator)
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))[:5]
        missing_right = sorted(set(left) - set(right))[:5]
        raise ValueError(
            f"paired item mismatch: missing preferred={missing_left}, comparator={missing_right}"
        )
    rows = [
        (left[key], float(left[key]["score"]) - float(right[key]["score"]))
        for key in sorted(left)
        if left[key]["stratum"] == stratum
        and (allowed_cells is None or _cell_key(left[key]) in allowed_cells)
    ]
    mean = _macro(rows)
    lcb = stratified_bootstrap_lcb(
        rows, samples=samples, confidence=confidence, seed=seed
    )
    by_cell: dict[str, list[float]] = defaultdict(list)
    for row, delta in rows:
        by_cell["/".join(map(str, _cell(row)))].append(delta)
    return {
        "n": len(rows),
        "paired_macro_delta": mean,
        "one_sided_lcb": lcb,
        "positive_mean": mean > 0.0,
        "positive_lcb": lcb > 0.0,
        "by_cell": {
            key: {"n": len(values), "mean_delta": sum(values) / len(values)}
            for key, values in sorted(by_cell.items())
        },
    }


def retention_check(
    preferred: dict,
    comparator: dict,
    *,
    stratum: str,
    retention_cells: set[str],
    transfer_families: set[str],
    maximum_anchor_regression: float,
    maximum_transfer_regression: float,
) -> dict:
    """Require non-regression without asking saturated cells to improve."""
    left = _item_map(preferred)
    right = _item_map(comparator)
    if set(left) != set(right):
        raise ValueError("paired item mismatch in retention check")
    by_cell: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left):
        row = left[key]
        cell = _cell_key(row)
        if row["stratum"] == stratum and cell in retention_cells:
            by_cell[cell].append(float(row["score"]) - float(right[key]["score"]))
    expected = {cell for cell in retention_cells if cell.startswith(f"{stratum}/")}
    if set(by_cell) != expected:
        raise ValueError(
            f"retention-cell mismatch for {stratum}: "
            f"missing={sorted(expected - set(by_cell))[:5]} "
            f"unexpected={sorted(set(by_cell) - expected)[:5]}"
        )
    cells = {}
    for cell, values in sorted(by_cell.items()):
        family = cell.split("/", 2)[1]
        limit = (
            maximum_transfer_regression
            if family in transfer_families
            else maximum_anchor_regression
        )
        mean_delta = sum(values) / len(values)
        cells[cell] = {
            "n": len(values),
            "mean_delta": mean_delta,
            "maximum_allowed_regression": limit,
            "passed": mean_delta >= -limit,
        }
    return {
        "n": sum(value["n"] for value in cells.values()),
        "cells": cells,
        "passed": all(value["passed"] for value in cells.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--quick-policy-scores", type=Path, nargs=2, required=True)
    parser.add_argument("--deep-policy-scores", type=Path, nargs=2, required=True)
    parser.add_argument(
        "--calibration", type=Path, default=EXP / "analysis" / "calibration.json"
    )
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "specialist_qualification.json"
    )
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    samples = int(config["evaluation"]["paired_bootstrap_samples"])
    confidence = float(config["evaluation"]["confidence"])
    quick_payloads = [_load(path) for path in args.quick_policy_scores]
    deep_payloads = [_load(path) for path in args.deep_policy_scores]
    calibration = _load(args.calibration)
    capability_cells = set(calibration.get("capability_cells", []))
    retention_cells = set(calibration.get("retention_cells", []))
    calibration_cells = set(calibration.get("cells", {}))
    expected_blocks = [int(value) for value in config["seeds"]["qualification_blocks"]]
    protocol = {
        "quick_block_seeds": [int(row["block_seed"]) for row in quick_payloads] == expected_blocks,
        "deep_block_seeds": [int(row["block_seed"]) for row in deep_payloads] == expected_blocks,
        "all_qualification_scope": all(
            row.get("scope") == "qualification" for row in quick_payloads + deep_payloads
        ),
        "all_greedy": all(row.get("decode") == "greedy" for row in quick_payloads + deep_payloads),
        "all_same_families": len({tuple(row.get("families", [])) for row in quick_payloads + deep_payloads}) == 1,
        "calibration_gate_passed": bool(calibration.get("gate", {}).get("passed")),
        "calibration_roles_partition_cells": (
            capability_cells.isdisjoint(retention_cells)
            and capability_cells | retention_cells == calibration_cells
        ),
        "quick_capability_cells_exist": any(
            cell.startswith("quick/") for cell in capability_cells
        ),
        "deep_capability_cells_exist": any(
            cell.startswith("deep/") for cell in capability_cells
        ),
    }
    block_results = {"quick": [], "deep": []}
    for index, (quick, deep) in enumerate(zip(quick_payloads, deep_payloads)):
        block_results["quick"].append(
            paired_advantage(
                quick, deep, stratum="quick", samples=samples,
                confidence=confidence, seed=1000 + index,
                allowed_cells=capability_cells,
            )
        )
        block_results["deep"].append(
            paired_advantage(
                deep, quick, stratum="deep", samples=samples,
                confidence=confidence, seed=2000 + index,
                allowed_cells=capability_cells,
            )
        )

    # Pooled inference retains block identity in the cell key by prefixing item
    # keys; this prevents coincident procedural ids from overwriting one another.
    pooled_payloads = {}
    for name, payloads in (("quick", quick_payloads), ("deep", deep_payloads)):
        items = []
        for block_index, payload in enumerate(payloads):
            for row in payload["items"]:
                copied = dict(row)
                copied["key"] = f"b{block_index}/{row['key']}"
                items.append(copied)
        pooled_payloads[name] = {"items": items}
    pooled = {
        "quick": paired_advantage(
            pooled_payloads["quick"], pooled_payloads["deep"], stratum="quick",
            samples=samples, confidence=confidence, seed=3000,
            allowed_cells=capability_cells,
        ),
        "deep": paired_advantage(
            pooled_payloads["deep"], pooled_payloads["quick"], stratum="deep",
            samples=samples, confidence=confidence, seed=4000,
            allowed_cells=capability_cells,
        ),
    }
    transfer = set(config["strata"]["transfer_families"])
    retention = {
        "quick": retention_check(
            pooled_payloads["quick"], pooled_payloads["deep"], stratum="quick",
            retention_cells=retention_cells, transfer_families=transfer,
            maximum_anchor_regression=float(config["qualification"]["maximum_anchor_regression"]),
            maximum_transfer_regression=float(config["qualification"]["maximum_transfer_regression"]),
        ),
        "deep": retention_check(
            pooled_payloads["deep"], pooled_payloads["quick"], stratum="deep",
            retention_cells=retention_cells, transfer_families=transfer,
            maximum_anchor_regression=float(config["qualification"]["maximum_anchor_regression"]),
            maximum_transfer_regression=float(config["qualification"]["maximum_transfer_regression"]),
        ),
    }
    decisions = {}
    for stratum in ("quick", "deep"):
        decisions[stratum] = {
            "positive_pooled_mean": pooled[stratum]["positive_mean"],
            "positive_pooled_lcb": pooled[stratum]["positive_lcb"],
            "every_block_positive": all(
                row["paired_macro_delta"] > 0.0 for row in block_results[stratum]
            ),
            "retention_passed": retention[stratum]["passed"],
        }
        decisions[stratum]["passed"] = all(decisions[stratum].values())
    passed = all(protocol.values()) and all(row["passed"] for row in decisions.values())
    result = {
        "stage": "specialist_qualification",
        "config": str(config_path),
        "decision_rule": "paired delta > 0 with both blocks positive and one-sided stratified-bootstrap LCB > 0; no effect-size floor",
        "protocol_checks": protocol,
        "block_results": block_results,
        "pooled": pooled,
        "calibration_roles": {
            "capability_cells": sorted(capability_cells),
            "retention_cells": sorted(retention_cells),
        },
        "retention": retention,
        "decisions": decisions,
        "retention_anchor_families": config["strata"]["retention_anchor_families"],
        "gate": {"passed": passed},
        "downstream_authorization": "teacher_audit" if passed else "stop_before_teacher_audit",
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
