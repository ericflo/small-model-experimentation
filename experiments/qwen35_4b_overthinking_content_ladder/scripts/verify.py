#!/usr/bin/env python3
"""Torch-free verification of the multi-budget ladder records."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import tasks as T  # noqa: E402


def main() -> int:
    tm = json.loads((EXP / "data" / "tasks.json").read_text())
    tmap = {int(k): T.Task(int(k), "", v["test_list"], v.get("test_imports", [])) for k, v in tm.items()}
    recs = [json.loads(l) for l in (EXP / "data" / "records.jsonl").read_text().splitlines() if l.strip()]

    def vone(r):
        t = tmap[int(r["task_id"])]
        return (T.verify(r["code"], t)[0], T.verify_visible(r["code"], t)[0])

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(vone, recs))
    with (EXP / "data" / "labels.jsonl").open("w") as f:
        for r, (full, vis) in zip(recs, res):
            f.write(json.dumps({"cond": r["cond"], "budget": r["budget"], "task_id": r["task_id"],
                                "sample": r["sample"], "full_pass": bool(full), "visible_pass": bool(vis)}) + "\n")
    agg = defaultdict(lambda: [0, 0])
    for r, (full, _) in zip(recs, res):
        a = agg[(r["budget"], r["cond"])]; a[0] += 1; a[1] += int(full)
    for (b, c), (nn, f) in sorted(agg.items()):
        print(f"  budget={b:5d} {c:10s} full_pass={f/nn:.3f}")
    print(f"wrote {EXP/'data'/'labels.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
