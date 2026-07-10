#!/usr/bin/env python3
"""Build grouped, search-relevant calibration candidates from exact oracle maps."""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import families as F  # noqa: E402
import experiment_common as C  # noqa: E402


def _pick_live_parents(
    live: dict[tuple[str, ...], int], length: int, count: int
) -> list[tuple[str, ...]]:
    candidates = [(prefix, n) for prefix, n in live.items() if len(prefix) == length and n > 0]
    if not candidates:
        return []
    candidates.sort(key=lambda item: (-item[1], item[0]))
    picked = [candidates[0][0]]
    if count > 1 and len(candidates) > 1:
        picked.append(min(candidates, key=lambda item: (item[1], item[0]))[0])
    return picked[:count]


def _pick_dead_parent(
    live: dict[tuple[str, ...], int], length: int, rng: random.Random
) -> tuple[str, ...] | None:
    if length == 0:
        return None
    # Rejection is exact and cheap at calibration depth <=4.
    for _ in range(10_000):
        prefix = tuple(rng.choice(F.TYPES) for _ in range(length))
        if live.get(prefix, 0) == 0:
            return prefix
    raise RuntimeError("could not sample a dead calibration parent")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    suffix = "_smoke" if args.smoke else ""
    cfg = C.load_config()
    cal_cfg = cfg["calibration"]
    tasks = C.load_jsonl(EXP / "data" / f"calibration_tasks{suffix}.jsonl")
    oracle_rows = C.load_jsonl(EXP / "data" / f"calibration_oracle{suffix}.jsonl")
    oracle_by_id = {row["task_id"]: row for row in oracle_rows}
    rng = random.Random(8117)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        depth = int(task["depth"])
        oracle = oracle_by_id[task_id]
        live, fills = C.live_prefix_maps(C.success_rows(oracle), depth)
        group_index = 0
        for parent_len in range(depth):
            parents: list[tuple[str, tuple[str, ...]]] = [
                ("live", prefix)
                for prefix in _pick_live_parents(
                    live,
                    parent_len,
                    int(cal_cfg["live_parents_per_task_per_slot"]),
                )
            ]
            if int(cal_cfg["dead_parents_per_task_per_slot"]):
                dead = _pick_dead_parent(live, parent_len, rng)
                if dead is not None:
                    parents.append(("dead", dead))
            for parent_kind, parent in parents:
                group_id = f"{task_id}:p{parent_len}:g{group_index:02d}"
                group_index += 1
                for child_index, operation in enumerate(F.TYPES):
                    prefix = parent + (operation,)
                    record = C.calibration_record(
                        task,
                        prefix,
                        record_id=f"{group_id}:c{child_index:02d}",
                        parent_group=group_id,
                        parent_kind=parent_kind,
                        child_operation=operation,
                        child_index=child_index,
                        live=bool(live.get(prefix, 0)),
                        completion_skeleton_count=int(live.get(prefix, 0)),
                        completion_parameter_fill_count=int(fills.get(prefix, 0)),
                        choices=list(F.TYPES),
                    )
                    rows.append(record)
    if not rows:
        raise RuntimeError("calibration candidate set is empty")
    C.write_jsonl(EXP / "data" / f"calibration_candidates{suffix}.jsonl", rows)
    live_rate = sum(bool(row["live"]) for row in rows) / len(rows)
    groups = len({row["parent_group"] for row in rows})
    mixed = sum(
        0 < sum(bool(row["live"]) for row in group) < len(group)
        for _, group_iter in itertools.groupby(rows, key=lambda row: row["parent_group"])
        for group in [list(group_iter)]
    )
    print(
        f"[calibration] {len(rows)} children in {groups} sibling groups; "
        f"mixed={mixed}; live_rate={live_rate:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
