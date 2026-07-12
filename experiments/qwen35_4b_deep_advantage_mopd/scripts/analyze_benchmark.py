#!/usr/bin/env python3
"""Analyze all frozen aggregate-only benchmark events without item access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, sha256_file, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", action="append", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "benchmark.json"
    )
    args = parser.parse_args()
    config, config_path = load_config()
    first = int(config["benchmark"]["first_seed"])
    quick_n = int(config["benchmark"]["quick_events"])
    medium_n = int(config["benchmark"]["medium_events"])
    expected_seeds = {
        "quick": list(range(first, first + quick_n)),
        "medium": list(range(first + quick_n, first + quick_n + medium_n)),
    }
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.event]
    keyed = {(row["tier"], int(row["seed"]), row["label"]): row for row in rows}
    expected = {
        (tier, seed, label)
        for tier, seeds in expected_seeds.items()
        for seed in seeds
        for label in ("primary", "soup", "visible")
    }
    protocol = {
        "exact_event_inventory": set(keyed) == expected and len(rows) == len(expected),
        "aggregate_only_schema": all(
            row.get("stage") == "aggregate_only_menagerie_event"
            and row.get("config_sha256") == sha256_file(config_path)
            and isinstance(row.get("aggregate"), (int, float))
            for row in rows
        ),
    }
    comparisons = {}
    for tier, seeds in expected_seeds.items():
        comparisons[tier] = {}
        for comparator in ("soup", "visible"):
            deltas = [
                float(keyed[(tier, seed, "primary")]["aggregate"])
                - float(keyed[(tier, seed, comparator)]["aggregate"])
                for seed in seeds
            ]
            comparisons[tier][comparator] = {
                "n": len(deltas),
                "deltas": deltas,
                "mean_delta": sum(deltas) / len(deltas),
                "all_events_positive": all(value > 0.0 for value in deltas),
            }
    checks = {
        "protocol": all(protocol.values()),
        "primary_beats_soup_and_visible_every_event": all(
            row["mean_delta"] > 0.0 and row["all_events_positive"]
            for tier in comparisons.values()
            for row in tier.values()
        ),
    }
    result = {
        "schema_version": 1,
        "stage": "aggregate_only_menagerie_confirmation",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "expected_seeds": expected_seeds,
        "event_artifacts": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.event
        ],
        "protocol_checks": protocol,
        "comparisons": comparisons,
        "checks": checks,
        "gate": {"passed": all(checks.values())},
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
