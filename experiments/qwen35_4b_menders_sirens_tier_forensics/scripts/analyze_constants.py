#!/usr/bin/env python3
"""Adjudicate the two frozen constants and the goal-gate feasibility by tier.

Consumes `runs/receipt_table.json` (from sweep_receipts.py). Cleaning rules,
frozen before results were interpreted:

- Drop any family block containing a value outside [0, 1] — those are DELTA
  tables inside summary receipts, not scores.
- Deduplicate identical (seed, tier, families) tuples, preferring rows whose
  arm label is not the filename fallback "summary"; summary receipts embed
  byte-identical copies of the per-arm blocks.
- An arm is class "base" when its label contains "base", else "treated".

Outputs `runs/constants_analysis.json`:

1. Constant check per tier: distributions of menders and sirens, split by
   arm class, with the exact counter-examples to "menders = 0 and
   sirens = 0.500 for every arm at every seed".
2. Base per-family profile per tier: min/median/max per family, plus per-seed
   INTERIOR flags — a family is interior when base is strictly above 0 and
   strictly below 1, i.e. a strict win against base is arithmetically
   available in both directions of difficulty.
3. Goal-gate feasibility per tier: for each base row (one per seed/event),
   how many families are ties-at-floor (base = 0 where our line's arms also
   sit at 0), ties-at-ceiling (base = 1), and interior; the goal gate wants
   ALL TEN families strictly beatable — floor families need the candidate to
   score at all, ceiling families make strict wins impossible.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
TABLE = EXP / "runs" / "receipt_table.json"
OUT = EXP / "runs" / "constants_analysis.json"

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


def clean_rows(rows: list[dict]) -> list[dict]:
    keep = {}
    for row in rows:
        # Summary receipts embed copies of the per-arm blocks AND delta
        # tables whose values can land inside [0, 1]; every real arm has
        # its own per-arm receipt file, so summary files add only phantoms.
        if Path(row["receipt"]).name.startswith("summary"):
            continue
        values = row["families"]
        if any(not (0.0 <= values[f] <= 1.0) for f in FAMILIES):
            continue
        key = (
            row["seed"],
            row["tier"],
            tuple(round(values[f], 9) for f in FAMILIES),
        )
        incumbent = keep.get(key)
        if incumbent is None or (
            incumbent["arm"] == "summary" and row["arm"] != "summary"
        ):
            keep[key] = row
    return list(keep.values())


def arm_class(row: dict) -> str:
    return "base" if "base" in row["arm"].lower() else "treated"


def dist(values: list[float]) -> dict:
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "zero": sum(1 for v in values if v == 0.0),
        "one": sum(1 for v in values if v == 1.0),
        "exactly_half": sum(1 for v in values if v == 0.5),
    }


def main() -> int:
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    rows = clean_rows(table["rows"])

    constant_check = {}
    for tier in ("quick", "medium"):
        tier_rows = [r for r in rows if r["tier"] == tier]
        block = {}
        for cls in ("base", "treated"):
            sub = [r for r in tier_rows if arm_class(r) == cls]
            if not sub:
                continue
            block[cls] = {
                "menders": dist([r["families"]["menders"] for r in sub]),
                "sirens": dist([r["families"]["sirens"] for r in sub]),
            }
        counterexamples = [
            {
                "receipt": r["receipt"],
                "seed": r["seed"],
                "arm": r["arm"],
                "think_budget": r["think_budget"],
                "menders": r["families"]["menders"],
                "sirens": r["families"]["sirens"],
            }
            for r in tier_rows
            if r["think_budget"] == 1024
            and (r["families"]["menders"] > 0 or r["families"]["sirens"] != 0.5)
        ]
        block["quick_tb1024_counterexamples_to_frozen_constants"] = counterexamples
        constant_check[tier] = block

    base_profile = {}
    feasibility = {}
    for tier in ("quick", "medium"):
        base_rows = [
            r for r in rows if r["tier"] == tier and arm_class(r) == "base"
        ]
        per_family = {}
        for family in FAMILIES:
            values = [r["families"][family] for r in base_rows]
            if values:
                per_family[family] = dist(values)
        base_profile[tier] = {"events": len(base_rows), "families": per_family}

        per_event = []
        for r in base_rows:
            floors = [f for f in FAMILIES if r["families"][f] == 0.0]
            ceilings = [f for f in FAMILIES if r["families"][f] == 1.0]
            per_event.append(
                {
                    "receipt": r["receipt"],
                    "seed": r["seed"],
                    "floor_families": floors,
                    "ceiling_families": ceilings,
                    "interior_count": len(FAMILIES)
                    - len(floors)
                    - len(ceilings),
                }
            )
        all_interior = sum(
            1
            for e in per_event
            if not e["ceiling_families"] and not e["floor_families"]
        )
        no_ceiling = sum(1 for e in per_event if not e["ceiling_families"])
        feasibility[tier] = {
            "base_events": len(per_event),
            "events_with_no_base_ceiling_family": no_ceiling,
            "events_with_all_ten_interior": all_interior,
            "per_event": per_event,
        }

    # Paired goal-gate adjudication: within one (experiment, tier, seed)
    # event, compare every treated arm against the base arm on all ten
    # families; the goal gate is a strict win on every family.
    paired = {}
    for tier in ("quick", "medium"):
        events = defaultdict(list)
        for r in rows:
            if r["tier"] == tier:
                events[(r["experiment"], r["seed"])].append(r)
        outcomes = []
        for (experiment, seed), members in sorted(events.items()):
            bases = [m for m in members if arm_class(m) == "base"]
            treats = [m for m in members if arm_class(m) == "treated"]
            if len(bases) != 1 or not treats:
                continue
            base = bases[0]
            for arm in treats:
                wins = [
                    f
                    for f in FAMILIES
                    if arm["families"][f] > base["families"][f]
                ]
                losses = [
                    f
                    for f in FAMILIES
                    if arm["families"][f] < base["families"][f]
                ]
                ties = [
                    f
                    for f in FAMILIES
                    if arm["families"][f] == base["families"][f]
                ]
                outcomes.append(
                    {
                        "experiment": experiment,
                        "seed": seed,
                        "arm": arm["arm"],
                        "strict_wins": len(wins),
                        "losses": losses,
                        "ties": ties,
                        "goal_gate_pass": len(wins) == len(FAMILIES),
                    }
                )
        paired[tier] = {
            "paired_arm_events": len(outcomes),
            "goal_gate_passes": sum(
                1 for o in outcomes if o["goal_gate_pass"]
            ),
            "near_misses_9_of_10": sum(
                1 for o in outcomes if o["strict_wins"] == 9
            ),
            "strict_win_distribution": {
                str(k): sum(1 for o in outcomes if o["strict_wins"] == k)
                for k in range(11)
            },
            "passes": [o for o in outcomes if o["goal_gate_pass"]],
            "blocking_families_in_9_win_events": sorted(
                {
                    f
                    for o in outcomes
                    if o["strict_wins"] == 9
                    for f in o["losses"] + o["ties"]
                }
            ),
        }

    OUT.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_data_read": False,
                "cleaned_rows": len(rows),
                "constant_check": constant_check,
                "base_profile": base_profile,
                "goal_gate_feasibility": feasibility,
                "paired_goal_gate": paired,
            },
            indent=1,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"cleaned rows: {len(rows)}")
    for tier in ("quick", "medium"):
        fb = feasibility[tier]
        print(
            f"{tier}: base events {fb['base_events']}, "
            f"no-ceiling {fb['events_with_no_base_ceiling_family']}, "
            f"all-interior {fb['events_with_all_ten_interior']}"
        )
        cx = constant_check[tier][
            "quick_tb1024_counterexamples_to_frozen_constants"
        ]
        print(f"  tb1024 counterexamples: {len(cx)}")
        pg = paired[tier]
        print(
            f"  paired arm-events {pg['paired_arm_events']}, "
            f"goal-gate passes {pg['goal_gate_passes']}, "
            f"9/10 near-misses {pg['near_misses_9_of_10']}"
        )
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
