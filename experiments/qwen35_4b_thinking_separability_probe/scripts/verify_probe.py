#!/usr/bin/env python3
"""Torch-free verification: label each generated solution with full-test + visible-test pass."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import tasks as T  # noqa: E402


def main() -> int:
    task_meta = json.loads((EXP / "data" / "tasks.json").read_text())
    tmap = {int(tid): T.Task(int(tid), "", m["test_list"], m.get("test_imports", []))
            for tid, m in task_meta.items()}
    recs = [json.loads(l) for l in (EXP / "data" / "records.jsonl").read_text().splitlines() if l.strip()]

    def vone(r):
        task = tmap[int(r["task_id"])]
        return (T.verify(r["code"], task)[0], T.verify_visible(r["code"], task)[0])

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(vone, recs))

    with (EXP / "data" / "labels.jsonl").open("w") as f:
        for r, (full, vis) in zip(recs, res):
            f.write(json.dumps({"cond": r["cond"], "task_id": r["task_id"], "sample": r["sample"],
                                "row": r["row"], "full_pass": bool(full), "visible_pass": bool(vis)}) + "\n")
    # quick per-cond pass rates
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])
    for r, (full, vis) in zip(recs, res):
        a = agg[r["cond"]]; a[0] += 1; a[1] += int(full); a[2] += int(vis)
    for c, (n, f_, v) in agg.items():
        print(f"  {c:14s} n={n} full_pass={f_/n:.3f} visible_pass={v/n:.3f}")
    print(f"wrote {EXP/'data'/'labels.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
