#!/usr/bin/env python3
"""Merge the generator-verifier candidate pool with a fresh visible-test label per candidate.

Reads gv_records (code + P(A) verifier signals) + gv_labels (full_pass) + tasks.json; computes
visible_pass = passes the FIRST assert (the deployable signal the earlier controller used); writes
data/pool.jsonl. Torch-free.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import tasks as T  # noqa: E402


def main() -> int:
    tm = json.loads((EXP / "data" / "tasks.json").read_text())
    tmap = {int(k): T.Task(int(k), "", v["test_list"], v.get("test_imports", [])) for k, v in tm.items()}
    recs = [json.loads(l) for l in (EXP / "data" / "gv_records.jsonl").read_text().splitlines() if l.strip()]
    full = {(r["task_id"], r["sample"]): r["full_pass"]
            for r in (json.loads(l) for l in (EXP / "data" / "gv_labels.jsonl").read_text().splitlines() if l.strip())}

    def vis(r):
        return T.verify_visible(r["code"], tmap[int(r["task_id"])])[0]

    with ThreadPoolExecutor(max_workers=8) as ex:
        visible = list(ex.map(vis, recs))

    with (EXP / "data" / "pool.jsonl").open("w") as f:
        for r, vp in zip(recs, visible):
            f.write(json.dumps({
                "task_id": r["task_id"], "sample": r["sample"], "code": r["code"],
                "full_pass": bool(full[(r["task_id"], r["sample"])]), "visible_pass": bool(vp),
                "pa_think": r["pa_think"], "pa_nothink": r["pa_nothink"],
            }) + "\n")
    n = len(recs)
    vp_rate = sum(visible) / n
    fp = sum(1 for r, vp in zip(recs, visible) if vp and not full[(r["task_id"], r["sample"])])
    print(f"pool: {n} candidates; visible_pass rate {vp_rate:.3f}; "
          f"visible-pass-but-full-FAIL (C2 false-passes) {fp} ({fp/max(1,sum(visible)):.3f} of visible-passers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
