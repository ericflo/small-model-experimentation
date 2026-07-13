#!/usr/bin/env python3
"""Run one firewall-clean paired Menagerie incumbent/candidate event."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
MENAGERIE = ROOT / "benchmarks" / "menagerie" / "run.py"
LOG = EXP / "runs" / "menagerie_log.jsonl"
BASELINE_SEED = 31337


def used_seeds() -> set[int]:
    seeds = {BASELINE_SEED}
    for path in (ROOT / "experiments").glob("*/runs/menagerie_log.jsonl"):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if "seed" in payload:
                seeds.add(int(payload["seed"]))
    return seeds


def run_arm(tier: str, seed: int, arm: str, model: Path) -> dict:
    raw = EXP / "runs" / "menagerie" / f"{tier}_seed{seed}_{arm}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(MENAGERIE),
        "--tier", tier,
        "--seed", str(seed),
        "--model-id", str(model.resolve()),
        "--out", str(raw),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=MENAGERIE.parent,
        env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    if completed.returncode:
        raise SystemExit(f"Menagerie {arm} failed with exit {completed.returncode}")
    payload = json.loads(raw.read_text())
    per_family = {}
    for family, stats in payload["per_family"].items():
        if isinstance(stats, dict):
            value = stats.get("score", stats.get("mean"))
            if value is None:
                raise SystemExit(f"unrecognized aggregate entry for {family}")
            per_family[family] = float(value)
        else:
            per_family[family] = float(stats)
    aggregate_only = {
        "aggregate": float(payload["aggregate"]),
        "per_family": per_family,
        "within_budget": payload.get("within_budget"),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    raw.write_text(json.dumps(aggregate_only, indent=2, sort_keys=True) + "\n")
    return aggregate_only


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["quick", "medium"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    if args.seed in used_seeds():
        raise SystemExit(f"Menagerie seed {args.seed} is already present in the public log union")
    for name, path in (("incumbent", args.incumbent), ("candidate", args.candidate)):
        if not (path / "config.json").is_file():
            raise SystemExit(f"{name} is not a merged checkpoint: {path}")

    event = {
        "schema_version": 1,
        "tier": args.tier,
        "seed": args.seed,
        "arms": {
            "incumbent": run_arm(args.tier, args.seed, "incumbent", args.incumbent),
            "candidate": run_arm(args.tier, args.seed, "candidate", args.candidate),
        },
        "firewall_storage": "aggregate_and_per_family_only",
    }
    event["delta"] = (
        event["arms"]["candidate"]["aggregate"]
        - event["arms"]["incumbent"]["aggregate"]
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    print(json.dumps(event, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
