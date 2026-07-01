#!/usr/bin/env python3
"""Torch-free execution labeling of the candidate solutions."""
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
    recs = [json.loads(l) for l in (EXP / "data" / "records.jsonl").read_text().splitlines() if l.strip()]

    def one(r):
        return T.verify(r["code"], tmap[int(r["task_id"])])[0]

    with ThreadPoolExecutor(max_workers=8) as ex:
        passes = list(ex.map(one, recs))
    with (EXP / "data" / "labels.jsonl").open("w") as f:
        for r, ok in zip(recs, passes):
            f.write(json.dumps({"task_id": r["task_id"], "sample": r["sample"], "full_pass": bool(ok)}) + "\n")
    n = len(passes)
    print(f"labeled {n} candidates; pass rate {sum(passes)/n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
