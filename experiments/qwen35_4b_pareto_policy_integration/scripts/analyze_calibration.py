#!/usr/bin/env python3
"""Describe quick/deep crossover and saturated cells without tuning a gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--quick", type=Path, required=True)
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=EXP / "analysis" / "calibration.json")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    quick = json.loads(args.quick.read_text())
    deep = json.loads(args.deep.read_text())
    qmap = {row["key"]: row for row in quick["items"]}
    dmap = {row["key"]: row for row in deep["items"]}
    protocol = {
        "paired_items": set(qmap) == set(dmap),
        "quick_scope": quick.get("scope") == "calibration",
        "deep_scope": deep.get("scope") == "calibration",
        "same_seed": quick.get("block_seed") == deep.get("block_seed") == config["seeds"]["calibration"],
        "greedy": quick.get("decode") == deep.get("decode") == "greedy",
    }
    cells = defaultdict(lambda: {"quick": [], "deep": []})
    for key in sorted(set(qmap) & set(dmap)):
        row = qmap[key]
        cell = f"{row['stratum']}/{row['family']}/{row['kind']}/L{row['level']}"
        cells[cell]["quick"].append(float(qmap[key]["score"]))
        cells[cell]["deep"].append(float(dmap[key]["score"]))
    threshold = float(config["qualification"]["saturation_score"])
    minimum_deficits = int(config["qualification"]["minimum_effective_deficit_items"])
    explicit_anchors = set(config["strata"]["retention_anchor_families"])
    transfer = set(config["strata"]["transfer_families"])
    # Episode cells contain six calibration instances per level, so the frozen
    # eight-deficit headroom rule must pool the registered levels within a
    # family/kind/stratum. The score-saturation test remains level-specific.
    deficit_pools = defaultdict(lambda: {"quick": [], "deep": []})
    for cell, values in cells.items():
        pool = "/".join(cell.split("/")[:3])
        for policy in ("quick", "deep"):
            deficit_pools[pool][policy].extend(values[policy])
    table = {}
    for cell, values in sorted(cells.items()):
        quick_mean = sum(values["quick"]) / len(values["quick"])
        deep_mean = sum(values["deep"]) / len(values["deep"])
        family = cell.split("/", 2)[1]
        pool = "/".join(cell.split("/")[:3])
        effective_deficits = {
            policy: sum(
                score < 1.0 - 1e-12 for score in deficit_pools[pool][policy]
            )
            for policy in ("quick", "deep")
        }
        reasons = []
        if max(quick_mean, deep_mean) >= threshold:
            reasons.append("calibration_score_saturated")
        if min(effective_deficits.values()) < minimum_deficits:
            reasons.append("fewer_than_minimum_effective_deficit_items")
        if family in explicit_anchors:
            reasons.append("registered_retention_anchor_family")
        if family in transfer:
            reasons.append("never_trained_transfer_family")
        table[cell] = {
            "n": len(values["quick"]), "quick_mean": quick_mean,
            "deep_mean": deep_mean, "deep_minus_quick": deep_mean - quick_mean,
            "effective_deficit_items": effective_deficits,
            "effective_deficit_pool": pool,
            "role": "retention" if reasons else "capability",
            "retention_reasons": reasons,
        }
    capability_cells = [key for key, value in table.items() if value["role"] == "capability"]
    retention_cells = [key for key, value in table.items() if value["role"] == "retention"]
    role_protocol = {
        "quick_capability_cells_exist": any(key.startswith("quick/") for key in capability_cells),
        "deep_capability_cells_exist": any(key.startswith("deep/") for key in capability_cells),
        "all_registered_anchors_retention_only": all(
            value["role"] == "retention"
            for key, value in table.items()
            if key.split("/", 2)[1] in explicit_anchors
        ),
        "all_transfer_cells_retention_only": all(
            value["role"] == "retention"
            for key, value in table.items()
            if key.split("/", 2)[1] in transfer
        ),
    }
    protocol.update(role_protocol)
    result = {
        "stage": "descriptive_calibration", "config": str(config_path),
        "protocol_checks": protocol, "cells": table,
        "capability_cells": capability_cells,
        "retention_cells": retention_cells,
        "gate": {"passed": all(protocol.values())},
        "downstream_authorization": "qualification" if all(protocol.values()) else "stop_for_protocol_error",
        "note": "descriptive only; no threshold or family was selected from these outcomes",
    }
    write_json(args.out, result)
    print(json.dumps({key: value for key, value in result.items() if key != "cells"}, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
