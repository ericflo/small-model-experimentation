#!/usr/bin/env python3
"""Learnable-band calibration: base greedy success per (source, family, level) cell.

Runs under .venv-vllm. Greedy, think@1024 (deployment quick budget), 12 items
per cell. Harvest uses only cells with success in (band_low, band_high).

../../.venv-vllm/bin/python scripts/band_calibrate.py [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import harness  # noqa: E402
import tasks  # noqa: E402

# Calibration seed blocks live at the top of the closed harvest ranges so
# harvest slices (which count up from the range base) can never collide.
GYM_CAL_SEED = 72901
CODE_CAL_SEED = 73901


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 cells only")
    args = parser.parse_args()

    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    hp = cfg["harvest_pivot"]
    per_cell = int(hp["band_items_per_cell"])

    cells: list[tuple[str, str, int]] = []
    for family, level in tasks.gym_cells(list(hp["gym_levels"])):
        cells.append(("gym", family, level))
    for depth in hp["code_depths"]:
        cells.append(("code", "listxform", int(depth)))
    if args.smoke:
        cells = cells[:1] + cells[-1:]

    items: list[tasks.TaskItem] = []
    for idx, (source, family, level) in enumerate(cells):
        if source == "gym":
            items.extend(tasks.make_gym_items(family, level, GYM_CAL_SEED + idx, per_cell))
        else:
            items.extend(tasks.make_code_items(level, CODE_CAL_SEED + idx, per_cell))

    runner = harness.make_runner(cfg["engine"])
    started = time.time()
    rows = harness.run_atoms(
        runner,
        [{"id": it.item_id, "family": it.family, "level": it.level,
          "prompt": it.prompt, "gold": it.payload.get("gold"),
          "answer_domain": it.payload.get("answer_domain")} for it in items
         if it.source == "gym"],
        k=1, think_budget=int(hp["think_budget"]),
        answer_max_tokens=int(hp["answer_max_tokens"]),
        run_seed=int(hp["run_seed"]), greedy=True,
    ) if any(it.source == "gym" for it in items) else []

    scores: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        key = ("gym", row["family"], int(row["level"]))
        scores.setdefault(key, []).append(float(row["outputs"][0]["score"]))

    code_items = [it for it in items if it.source == "code"]
    if code_items:
        from vllm_runner import SamplingConfig
        records = [{"id": it.item_id, "messages": [{"role": "user", "content": it.prompt}]}
                   for it in code_items]
        sampling = SamplingConfig(
            thinking="budget", thinking_budget=int(hp["think_budget"]),
            n=1, answer_max_tokens=int(hp["answer_max_tokens"]),
            greedy=True, run_seed=int(hp["run_seed"]))
        out_rows, _ = runner.generate(records, sampling)
        for it, row in zip(code_items, out_rows):
            key = ("code", "listxform", it.level)
            scores.setdefault(key, []).append(
                tasks.score_item(it, row["outputs"][0]["text"]))

    lo, hi = float(hp["band_low"]), float(hp["band_high"])
    table = []
    for (source, family, level), vals in sorted(scores.items()):
        rate = sum(vals) / len(vals)
        table.append({
            "source": source, "family": family, "level": level,
            "n": len(vals), "success": rate,
            "in_band": bool(lo < rate < hi),
        })
        print(f"{source:4s} {family:14s} L{level}: {rate:.3f} "
              f"({'IN' if lo < rate < hi else 'out'})")

    dest = EXP / "runs" / "band_calibration.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "think_budget": hp["think_budget"], "items_per_cell": per_cell,
        "band": [lo, hi], "wall_s": time.time() - started, "cells": table,
    }, indent=2))
    print(f"wrote {dest} ({time.time() - started:.0f}s)")
    in_band = sum(1 for c in table if c["in_band"])
    print(f"in-band cells: {in_band}/{len(table)}")
    return 0 if in_band >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
