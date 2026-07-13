#!/usr/bin/env python3
"""Analyze the compute-response study: medium delta vs deployed think budget.

Reads runs/menagerie_log.jsonl, selects the compute-response events (medium
tier, seeds in the study range, non-null think_budget), pairs base vs merged
per seed, and reports the mean merged-minus-base delta at each budget. The
decisive read: if delta climbs monotonically with budget, C54's serial-compute
diagnosis is confirmed and the medium wall is budget-tunable; if flat, the wall
is capability, not compute, and C54 must be corrected.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
LOG = EXP / "runs" / "menagerie_log.jsonl"
STUDY_SEEDS = set(range(54001, 54100))


def main() -> int:
    # (budget, seed) -> {arm: aggregate}
    by_cell: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for line in LOG.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("tier") != "medium" or rec.get("seed") not in STUDY_SEEDS:
            continue
        tb = rec.get("think_budget")
        if tb is None:
            continue
        by_cell[(int(tb), int(rec["seed"]))][rec["arm"]] = float(rec["aggregate"])

    per_budget: dict[int, list[float]] = defaultdict(list)
    per_budget_base: dict[int, list[float]] = defaultdict(list)
    per_budget_merged: dict[int, list[float]] = defaultdict(list)
    for (tb, seed), arms in sorted(by_cell.items()):
        if "base" in arms and "merged" in arms:
            delta = arms["merged"] - arms["base"]
            per_budget[tb].append(delta)
            per_budget_base[tb].append(arms["base"])
            per_budget_merged[tb].append(arms["merged"])
            print(f"  tb={tb:>5} seed={seed}: base={arms['base']:.4f} "
                  f"merged={arms['merged']:.4f} delta={delta:+.4f}")

    print("\n=== compute-response: medium delta vs think budget ===")
    print(f"{'budget':>8} {'n':>3} {'base_mean':>10} {'merged_mean':>12} "
          f"{'delta_mean':>11} {'delta_sd':>9}")
    trend = []
    for tb in sorted(per_budget):
        deltas = per_budget[tb]
        n = len(deltas)
        dmean = statistics.mean(deltas)
        dsd = statistics.pstdev(deltas) if n > 1 else float("nan")
        bmean = statistics.mean(per_budget_base[tb])
        mmean = statistics.mean(per_budget_merged[tb])
        trend.append((tb, dmean))
        print(f"{tb:>8} {n:>3} {bmean:>10.4f} {mmean:>12.4f} "
              f"{dmean:>+11.4f} {dsd:>9.4f}")

    if len(trend) >= 2:
        lo_tb, lo_d = trend[0]
        hi_tb, hi_d = trend[-1]
        print(f"\nslope: delta {lo_d:+.4f} @tb{lo_tb} -> {hi_d:+.4f} @tb{hi_tb} "
              f"({hi_d - lo_d:+.4f} over {hi_tb // lo_tb}x budget)")
        if hi_d - lo_d > 0.03:
            print("READ: delta RISES with budget -> serial-compute wall (C54 confirmed); "
                  "medium is budget-tunable.")
        elif abs(hi_d - lo_d) <= 0.03:
            print("READ: delta ~FLAT with budget -> wall is CAPABILITY, not compute; "
                  "C54 serial-compute claim must be corrected.")
        else:
            print("READ: delta FALLS with budget -> merged advantage is budget-specific; investigate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
