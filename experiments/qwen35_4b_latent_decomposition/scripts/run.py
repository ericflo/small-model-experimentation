#!/usr/bin/env python3
"""Dissect the decomposition failure and test whether banking fixes it.
Phase RANK  : per-step ground-truth next-op ranking accuracy (does the model know the next op? lookahead?).
Phase SEARCH: end-to-end hidden-generalizing coverage vs interpreter budget (model-guided vs brute vs random).
Phase ABLATION: 2x2 {model,brute,random} x {pruned,unpruned} -- does distance-pruning do the model's work?
Run once per guide (base / banked_640 / banked_1280); brute+random computed in the base run (model-free)."""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP / "src"))
import families as FAM  # noqa: E402
import gen_lib as GL  # noqa: E402
import decompose as D  # noqa: E402

fam = FAM.FAMILIES["list"]


def parse_repr(r):
    if "(" in r:
        name, k = r.split("(")[0], int(r[r.index("(") + 1:-1])
    else:
        name, k = r, None
    return D.OP_REPRS.index(FAM.op_repr(name, k))


def rank_phase(probe, tasks, K=6):
    ranks = {1: [], 2: [], 3: []}
    hit1 = {1: 0, 2: 0, 3: 0}
    hitk = {1: 0, 2: 0, 3: 0}
    for t in tasks:
        ops = [parse_repr(r) for r in t["target_ops"]]
        states = [list(ex["input"]) for ex in t["visible"]]
        target = [list(ex["output"]) for ex in t["visible"]]
        for step, gt in enumerate(ops, 1):
            sc = D.score_ops(probe, D.propose_prompt(states, target))
            order = sorted(range(len(sc)), key=lambda i: -sc[i])
            rank = order.index(gt) + 1
            ranks[step].append(rank)
            if rank == 1:
                hit1[step] += 1
            if rank <= K:
                hitk[step] += 1
            op, p = D.OPS[gt]
            states = [FAM.apply_op(fam, op, p, s) for s in states]
    out = {}
    for s in (1, 2, 3):
        n = len(ranks[s])
        out[s] = {"top1": hit1[s] / n, f"top{K}": hitk[s] / n,
                  "mean_rank": statistics.mean(ranks[s]), "median_rank": statistics.median(ranks[s]), "n": n}
    return out


def search_coverage(probe, tasks, mode, beam, k, rng, pruned=True):
    visn = hidn = interp = fwd = 0
    for t in tasks:
        r = D.model_search(probe, t, k=k, beam=beam, max_depth=3, mode=mode, rng=rng, pruned=pruned)
        interp += r["n_interp"]; fwd += r["n_fwd"]
        if r["seq"] is not None:
            visn += 1
            if D.verify_hidden(r["seq"], t["hidden"]):
                hidn += 1
    n = len(tasks)
    return {"mode": mode, "beam": beam, "k": k, "pruned": pruned, "n": n,
            "coverage_hidden": hidn / n, "coverage_visible": visn / n,
            "interp_per_task": interp / n, "fwd_per_task": fwd / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--search", action="store_true", help="run the (expensive) guided-search coverage phase")
    ap.add_argument("--with-controls", action="store_true", help="also run brute+random (model-free); use in base run")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in (EXP / "data" / "eval_frozen_d3.jsonl").read_text().splitlines() if l.strip()][:args.n]
    probe = GL.Probe()
    if args.adapter:
        from peft import PeftModel
        probe.model = PeftModel.from_pretrained(probe.model, args.adapter).eval()
        print(f"[{args.tag}] loaded adapter {args.adapter}", flush=True)
    rng = random.Random(0)
    (EXP / "runs").mkdir(exist_ok=True)

    t0 = time.time()
    rank = rank_phase(probe, tasks)
    print(f"[{args.tag}] RANK per-step next-op accuracy (chance top1={1/32:.3f}, top6={6/32:.3f}):", flush=True)
    for s in (1, 2, 3):
        r = rank[s]
        print(f"   step {s}: top1 {r['top1']:.3f} | top6 {r['top6']:.3f} | mean rank {r['mean_rank']:.1f} | median {r['median_rank']:.0f}", flush=True)
    (EXP / "runs" / f"rank_{args.tag}.json").write_text(json.dumps(rank, indent=1))

    if args.search:
        guided = []
        for beam in (8,):
            r = search_coverage(probe, tasks, "model", beam, k=beam, rng=rng)
            guided.append(r)
            print(f"[{args.tag}] model-guided beam={beam}: hidden-cov {r['coverage_hidden']:.3f} | {r['interp_per_task']:.0f} interp/task", flush=True)
        (EXP / "runs" / f"search_{args.tag}.json").write_text(json.dumps(guided, indent=1))

    if args.with_controls:
        for mode, beams in (("brute", (2, 4, 8, 16, 32, 64)), ("random", (2, 4, 8, 16, 32))):
            rows = []
            for beam in beams:
                kk = len(D.OPS) if mode == "brute" else beam
                r = search_coverage(probe, tasks, mode, beam, k=kk, rng=rng)
                rows.append(r)
                print(f"[{mode}] beam={beam}: hidden-cov {r['coverage_hidden']:.3f} | {r['interp_per_task']:.0f} interp/task", flush=True)
            (EXP / "runs" / f"search_{mode}.json").write_text(json.dumps(rows, indent=1))
        abl = []
        for mode in ("brute", "random"):  # CPU-cheap (no model): full n, beam=8
            for pruned in (True, False):
                kk = len(D.OPS) if mode == "brute" else 8
                r = search_coverage(probe, tasks, mode, 8, k=kk, rng=rng, pruned=pruned)
                abl.append(r)
                print(f"[ablation] {mode} pruned={pruned}: hidden-cov {r['coverage_hidden']:.3f} | {r['interp_per_task']:.0f} interp/task", flush=True)
        for pruned in (True, False):  # model part bounded: beam=4, first 40 tasks
            r = search_coverage(probe, tasks[:40], "model", 4, k=4, rng=rng, pruned=pruned)
            abl.append(r)
            print(f"[ablation] model(n40,beam4) pruned={pruned}: hidden-cov {r['coverage_hidden']:.3f} | {r['interp_per_task']:.0f} interp/task", flush=True)
        (EXP / "runs" / "ablation.json").write_text(json.dumps(abl, indent=1))

    print(f"[{args.tag}] done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
