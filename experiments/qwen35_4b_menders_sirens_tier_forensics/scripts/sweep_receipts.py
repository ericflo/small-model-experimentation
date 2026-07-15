#!/usr/bin/env python3
"""Sweep every committed menagerie gateway receipt for per-family scores.

Analysis-only forensics: no model, no GPU, no seed is consumed, and the
`benchmarks/` directory is never read — the inputs are the gateway OUTPUT
receipts already committed under `experiments/*/runs/`. The cell exists to
explain two frozen constants observed by the universal/axis line at the
quick tier with think budget 1,024: `menders` = 0 and `sirens` = 0.500 for
every arm at every seed. The sweep builds the complete per-family score
table across every historical receipt (all tiers, all seasons), so the
constants can be checked against the medium tier directly.

Output: `runs/receipt_table.json` — one row per (receipt file, arm) with
tier, seed, aggregate, and the ten public family scores, plus provenance
(experiment, relative path, sha256). The companion `analyze_constants.py`
consumes this table; this script only collects.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
OUT = EXP / "runs" / "receipt_table.json"

FAMILIES = (
    "chronicle",
    "lockpick",
    "menders",
    "mirage",
    "rites",
    "siftstack",
    "sirens",
    "stockade",
    "toolsmith",
    "warren",
)

TIER_SEED_RE = re.compile(r"(quick|medium|slow|huge)_(?:tb(\d+)_)?seed(\d+)")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk_for_families(obj):
    """Yield every dict that carries all ten public family keys."""
    if isinstance(obj, dict):
        if all(family in obj for family in FAMILIES):
            yield obj
        for value in obj.values():
            yield from walk_for_families(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_for_families(value)


def find_aggregate(payload: dict, family_block: dict) -> float | None:
    if isinstance(payload.get("aggregate"), (int, float)):
        return float(payload["aggregate"])
    # Some receipts nest {aggregate, per_family} per arm; find the sibling.
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("per_family") is family_block and isinstance(
                node.get("aggregate"), (int, float)
            ):
                return float(node["aggregate"])
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return None


def classify(path: Path) -> tuple[str | None, int | None, int | None]:
    """(tier, think_budget, seed) from the receipt path, else Nones."""
    match = TIER_SEED_RE.search(str(path))
    if not match:
        return None, None, None
    tier = match.group(1)
    budget = int(match.group(2)) if match.group(2) else None
    seed = int(match.group(3))
    return tier, budget, seed


def arm_label(path: Path, payload: dict) -> str:
    for key in ("arm", "model_label", "candidate", "label"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return path.stem.split("seed")[-1].split("_", 1)[-1] if "seed" in path.stem else path.stem


def main() -> int:
    rows = []
    scanned = 0
    for receipt in sorted(REPO.glob("experiments/*/runs/**/*.json")):
        if EXP in receipt.parents:
            continue
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        scanned += 1
        blocks = list(walk_for_families(payload))
        if not blocks:
            continue
        tier, budget, seed = classify(receipt)
        relative = str(receipt.relative_to(REPO))
        digest = sha256_file(receipt)
        for block in blocks:
            if not all(isinstance(block[f], (int, float)) for f in FAMILIES):
                continue
            rows.append(
                {
                    "experiment": receipt.relative_to(REPO).parts[1],
                    "receipt": relative,
                    "receipt_sha256": digest,
                    "tier": tier,
                    "think_budget": budget,
                    "seed": seed,
                    "arm": arm_label(receipt, payload),
                    "aggregate": find_aggregate(payload, block),
                    "families": {f: float(block[f]) for f in FAMILIES},
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_data_read": False,
                "source": "committed gateway receipts under experiments/*/runs only",
                "files_scanned": scanned,
                "rows": rows,
            },
            indent=1,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"scanned {scanned} json files; {len(rows)} family-score rows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
