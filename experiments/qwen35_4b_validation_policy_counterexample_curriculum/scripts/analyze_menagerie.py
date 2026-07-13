#!/usr/bin/env python3
"""Apply the frozen paired quick/medium Menagerie gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--log", type=Path, default=EXP / "runs" / "menagerie_log.jsonl")
    parser.add_argument("--out", type=Path, default=EXP / "analysis" / "menagerie_gate.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())["menagerie"]
    events = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
    expected = {name: int(seed) for name, seed in cfg["paired_seeds"].items()}
    selected = {
        tier: next(
            (row for row in events if row["tier"] == tier and int(row["seed"]) == seed),
            None,
        )
        for tier, seed in expected.items()
    }
    if any(row is None for row in selected.values()):
        raise SystemExit("missing preregistered Menagerie event")
    deltas = {tier: float(row["delta"]) for tier, row in selected.items()}
    checks = {
        "one_positive_tier": max(deltas.values()) >= float(cfg["minimum_positive_tier_delta"]),
        "no_tier_regression": min(deltas.values()) >= -float(cfg["maximum_tier_regression"]),
        "aggregate_only_storage": all(
            row.get("firewall_storage") == "aggregate_and_per_family_only"
            for row in selected.values()
        ),
    }
    result = {
        "schema_version": 1,
        "stage": "menagerie",
        "seeds": expected,
        "deltas": deltas,
        "events": selected,
        "checks": checks,
        "gate": {"passed": all(checks.values())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
