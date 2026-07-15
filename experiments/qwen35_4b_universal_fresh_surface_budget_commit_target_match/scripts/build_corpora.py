#!/usr/bin/env python3
"""Freeze the two fresh-surface treatment corpora and their construction manifest.

Arm D is the 160-row designed160 distribution rendered on the six fresh surfaces.
Arm B keeps a deterministic 120-row largest-remainder subset of arm D and swaps the
other 40 designed rows for 40 budget lessons from the same construction seed, so the
two arms differ by exactly that one substitution.  Everything is a pure function of
the frozen construction seed; `--check` regenerates and byte-compares.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_fresh_curriculum as fresh  # noqa: E402

CONSTRUCTION_SEED = 77116
ARM_D_PATH = EXP / "data" / "sft_fresh_designed160.jsonl"
ARM_B_PATH = EXP / "data" / "sft_fresh_budget160.jsonl"
MANIFEST_PATH = EXP / "data" / "corpus_manifest.json"
MIN_BUDGET_HITS = 14
MAX_BUDGET_HITS = 30


def deterministic_rank(namespace: str, index: int, line: str) -> bytes:
    return hashlib.sha256(f"{CONSTRUCTION_SEED}:{namespace}:{index}:".encode() + line.encode()).digest()


def subset_designed(rows: list[dict]) -> list[dict]:
    lines = [json.dumps(fresh.public_row(row), sort_keys=True, ensure_ascii=False) for row in rows]
    by_kind: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_kind.setdefault(row["kind"].removeprefix("u_"), []).append(index)
    selected: list[int] = []
    for skill, quota in sorted(fresh.ARM_B_DESIGNED_QUOTAS.items()):
        candidates = sorted(
            by_kind[skill],
            key=lambda index: deterministic_rank(f"arm-b-designed-{skill}", index, lines[index]),
        )
        if len(candidates) < quota:
            raise ValueError(f"not enough {skill} rows for the arm-b quota")
        selected.extend(candidates[:quota])
    return [rows[index] for index in sorted(selected)]


def build() -> tuple[dict[Path, bytes], dict]:
    designed = fresh.generate_curriculum(fresh.ARM_D_MIX, CONSTRUCTION_SEED)
    fresh.validate_generated(designed)
    fresh.check_banned_vocabulary(designed)
    if len(designed) != 160:
        raise ValueError(f"arm D must have 160 rows, got {len(designed)}")

    budget = fresh.generate_curriculum(fresh.BUDGET_MIX, CONSTRUCTION_SEED)
    fresh.validate_generated(budget)
    fresh.check_banned_vocabulary(budget)
    if len(budget) != 40:
        raise ValueError(f"budget block must have 40 rows, got {len(budget)}")
    hits = sum(1 for row in budget if row["_audit"]["outcome"] == "hit")
    if not MIN_BUDGET_HITS <= hits <= MAX_BUDGET_HITS:
        raise ValueError(f"budget outcome balance out of range: {hits} hits of 40")

    subset = subset_designed(designed)
    if len(subset) != 120:
        raise ValueError(f"arm B designed subset must have 120 rows, got {len(subset)}")
    arm_b = subset + budget
    combined = fresh.validate_generated(arm_b)
    arm_d_summary = fresh.validate_generated(designed)

    def encode(rows: list[dict]) -> bytes:
        return "".join(
            json.dumps(fresh.public_row(row), ensure_ascii=False) + "\n" for row in rows
        ).encode("utf-8")

    outputs = {ARM_D_PATH: encode(designed), ARM_B_PATH: encode(arm_b)}
    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "construction_seed": CONSTRUCTION_SEED,
        "surfaces": sorted(fresh.SURFACE_POOLS),
        "separators": list(fresh.SEPARATORS),
        "attributes": list(fresh.ATTRIBUTES),
        "banned_vocabulary_checked": True,
        "arm_d": {
            "path": ARM_D_PATH.relative_to(EXP).as_posix(),
            "rows": len(designed),
            "mix": fresh.ARM_D_MIX,
            "kinds": arm_d_summary["kinds"],
            "sha256": hashlib.sha256(outputs[ARM_D_PATH]).hexdigest(),
        },
        "arm_b": {
            "path": ARM_B_PATH.relative_to(EXP).as_posix(),
            "rows": len(arm_b),
            "designed_subset_rows": len(subset),
            "designed_subset_quotas": fresh.ARM_B_DESIGNED_QUOTAS,
            "budget_rows": len(budget),
            "budget_hits": hits,
            "budget_exhausts": len(budget) - hits,
            "kinds": combined["kinds"],
            "sha256": hashlib.sha256(outputs[ARM_B_PATH]).hexdigest(),
        },
        "arm_b_subset_of_arm_d": True,
        "budget_levels": dict(sorted(Counter(row["level"] for row in budget).items())),
    }
    return outputs, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs, manifest = build()
    targets = dict(outputs)
    targets[MANIFEST_PATH] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if args.check:
        for path, expected in targets.items():
            if not path.is_file() or path.read_bytes() != expected:
                raise SystemExit(f"frozen corpus artifact is absent or changed: {path}")
    else:
        conflicts = [path for path in targets if path.exists()]
        if conflicts:
            parser.error(f"refusing to overwrite frozen corpus artifact: {conflicts[0]}")
        for path, value in targets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(json.dumps({
        "arm_d_sha256": manifest["arm_d"]["sha256"],
        "arm_b_sha256": manifest["arm_b"]["sha256"],
        "budget_hits": manifest["arm_b"]["budget_hits"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
