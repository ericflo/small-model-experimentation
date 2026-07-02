#!/usr/bin/env python3
"""Phase 0: behavioral min-depth audit of existing substrate tasks (CPU-only).

A nominal depth-d composition may be behaviorally equivalent to a shallower pipeline
(sort_asc∘reverse ≡ sort_desc; negate∘negate ≡ id; filter∘filter ≡ tighter filter). For every existing
M1/M2/C12 task, find the true behavioral min-depth over ALL (visible+hidden) examples via exact
enumeration at depths 1-2 and generous beam search at depth 3+, then restratify recorded solves.

ADVANCE PREDICTIONS (logged in prereg before running):
  P0a: >=20% of nominal depth-3 tasks collapse to min-depth <=2.
  P0b: recorded depth-3 monolithic solves concentrate on collapsed tasks.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/ericflo/Development/small-model-experimentation/experiments")
sys.path.insert(0, str(REPO / "qwen35_4b_decompose_compose_frontier" / "src"))
import gen_tasks as G  # noqa: E402
import decompose_lib as D  # noqa: E402

ALL_OPS = D.brute_ops() if hasattr(D, "brute_ops") else [c for n in D.NAMES for c in D.expand(n)]


def states_of(task):
    ex = task["visible"] + task["hidden"]
    return tuple(tuple(e["input"]) for e in ex), tuple(tuple(e["output"]) for e in ex)


def min_depth(task, cap=4):
    """Exact BFS over primitive pipelines (dedup by behavioral state) up to depth cap."""
    inp, target = states_of(task)
    if inp == target:
        return 0
    level = {inp: True}
    seen = {inp}
    for depth in range(1, cap + 1):
        nxt = {}
        for state in level:
            for op, k in ALL_OPS:
                new = D.apply_prim(op, k, state)
                if new is None or new in seen:
                    continue
                if new == target:
                    return depth
                seen.add(new)
                nxt[new] = True
        level = nxt
        if not level or len(seen) > 400_000:
            break
    return None  # > cap (or search exhausted)


def audit(tasks_file, recs_file, key):
    tasks = {t["task_id"]: t for t in (json.loads(l) for l in open(tasks_file))}
    recs = [json.loads(l) for l in open(recs_file)] if recs_file else []
    out = []
    for tid, t in tasks.items():
        cap = min(t["depth"], 3)  # we only need to know if it's SHALLOWER than nominal
        md = min_depth(t, cap=cap if t["depth"] > 1 else 1)
        eff = md if md is not None else t["depth"]  # None => no shallower pipeline found => nominal
        out.append({"task_id": tid, "nominal": t["depth"], "min_depth": eff,
                    "collapsed": eff < t["depth"]})
    coll = defaultdict(lambda: [0, 0])
    for r in out:
        c = coll[r["nominal"]]; c[0] += 1; c[1] += int(r["collapsed"])
    print(f"\n== {key} ==")
    for d in sorted(coll):
        n, c = coll[d]
        print(f"  nominal d{d}: {c}/{n} collapse to shallower ({c/n:.0%})")
    if recs:
        mdmap = {r["task_id"]: r for r in out}
        solved_by = defaultdict(lambda: [0, 0])  # (nominal_d, collapsed) -> [n, solves]
        for rec in recs:
            m = mdmap.get(rec["task_id"])
            if m is None or m["nominal"] < 2:
                continue
            kk = (m["nominal"], m["collapsed"])
            fld = "passk" if "passk" in rec else ("hidden_pass" if "hidden_pass" in rec else "passk_full")
            solved_by[kk][0] += 1; solved_by[kk][1] += int(rec.get(fld, 0))
        print("  solves by (nominal_d, collapsed): ", end="")
        for (dd, cc), (n, s) in sorted(solved_by.items()):
            print(f"d{dd}{'C' if cc else 'F'}:{s}/{n}", end="  ")
        print()
    return out


def main():
    NR = REPO / "qwen35_4b_neurosymbolic_repl_substrate"
    DC = REPO / "qwen35_4b_decompose_compose_frontier"
    results = {}
    results["M1_baseline"] = audit(NR / "data" / "tasks.jsonl", NR / "data" / "baseline_records.jsonl", "M1 baseline (d1-6)")
    results["M2_repl"] = audit(NR / "data" / "repl_tasks.jsonl", NR / "data" / "repl_records.jsonl", "M2 REPL (d1-4)")
    results["C12_search"] = audit(DC / "data" / "search_tasks.jsonl", None, "C12 search tasks (d2-3)")
    outp = Path(__file__).parent / "min_depth_audit.json"
    outp.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
