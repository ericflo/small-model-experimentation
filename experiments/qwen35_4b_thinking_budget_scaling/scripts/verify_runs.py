#!/usr/bin/env python3
"""Torch-free verification pass: read generations.jsonl, execute candidates, write summary.

Run as a separate process so the candidate-sandbox forks never inherit a CUDA context.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import tasks as T  # noqa: E402
import metrics as M  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="{}")
    args = ap.parse_args()

    task_meta = json.loads(Path(args.tasks).read_text())
    tmap = {int(tid): T.Task(int(tid), "", m["test_list"], m.get("test_imports", []))
            for tid, m in task_meta.items()}
    recs = [json.loads(l) for l in Path(args.gen).read_text().splitlines() if l.strip()]

    def vone(r):
        task = tmap[int(r["task_id"])]
        full = T.verify(r["code"], task)[0]
        vis = T.verify_visible(r["code"], task)[0] if r["kind"] == "sample" else False
        return full, vis

    with ThreadPoolExecutor(max_workers=8) as ex:
        for r, (f, v) in zip(recs, ex.map(vone, recs)):
            r["pass"], r["visible"] = f, v

    verified_path = Path(args.out).parent / "verified.jsonl"
    with verified_path.open("w") as vf:
        for r in recs:
            vf.write(json.dumps(r) + "\n")

    # group: cond -> task_id -> records
    conds: list[str] = []
    g = defaultdict(lambda: defaultdict(lambda: {"greedy": None, "samples": {}}))
    task_order: list[int] = []
    for r in recs:
        c, tid = r["cond"], int(r["task_id"])
        if c not in conds:
            conds.append(c)
        if tid not in task_order:
            task_order.append(tid)
        if r["kind"] == "greedy":
            g[c][tid]["greedy"] = r
        else:
            g[c][tid]["samples"][r["s"]] = r

    summary = {}
    for c in conds:
        tids = [t for t in task_order if t in g[c]]
        greedy_pass = [bool(g[c][t]["greedy"]["pass"]) for t in tids]
        sampled_pass, sampled_vis, n_think, n_gen, forced = [], [], [], [], []
        for t in tids:
            smp = g[c][t]["samples"]
            ks = sorted(smp)
            sampled_pass.append([bool(smp[s]["pass"]) for s in ks])
            sampled_vis.append([bool(smp[s]["visible"]) for s in ks])
            n_think += [smp[s]["n_think"] for s in ks]
            n_gen += [smp[s]["n_gen"] for s in ks]
            forced += [bool(smp[s]["forced"]) for s in ks]
        forced_frac = (sum(forced) / len(forced)) if forced else 0.0
        summary[c] = M.summarize_condition(greedy_pass, sampled_pass, sampled_vis,
                                           n_think, n_gen, forced_frac, args.k)

    meta = json.loads(args.meta)
    meta["conditions"] = summary
    Path(args.out).write_text(json.dumps(meta, indent=2))

    # pretty print
    print("\n=== THINKING-BUDGET SWEEP SUMMARY ===")
    hdr = f"{'condition':18s} {'greedy@1':>8s} {'pass@1':>7s} {'pass@'+str(args.k):>7s} {'sel@1':>6s} {'think':>6s} {'total':>6s} {'forced':>6s}"
    print(hdr)
    for c in conds:
        s = summary[c]
        print(f"{c:18s} {s['greedy_pass@1']:8.3f} {s['sampled_pass@1']:7.3f} "
              f"{s['pass@'+str(args.k)+'_oracle']:7.3f} {s['visible_selector@1']:6.3f} "
              f"{s['mean_think_tokens']:6.0f} {s['mean_total_tokens']:6.0f} {s['forced_close_frac']:6.2f}")
    print(f"\nwrote {args.out} and {verified_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
