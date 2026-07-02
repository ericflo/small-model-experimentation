#!/usr/bin/env python3
"""Harvest frontier-exceeding solutions via brute decompose-search (fast, no model, interpreter-only) on a
large fresh pool, for banking. Also emits a held-out monolithic eval set. Solutions use G.prompt_for so the
training prompt matches eval_lora's prompt."""
import json, sys, time
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP/"src"))
import gen_tasks as G, decompose_lib as D
def build(depths, per, seed, kv=10):
    import random; rng=random.Random(seed); ts=[]; tid=0
    for d in depths:
        m=0
        while m<per:
            t=G.make_task(tid,d,rng,k_visible=kv,m_hidden=8)
            if t: ts.append(t); tid+=1; m+=1
    return ts
def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,default=888)
    ap.add_argument("--out",default="data/frontier_train.jsonl"); ap.add_argument("--gen-eval",action="store_true")
    a=ap.parse_args()
    pool=build([2,3],250,seed=a.seed); dz=D.Decomposer(probe=None); t0=time.time()
    found=[]
    for t in pool:
        r=dz.search(t,"brute",beam_width=60,top_p=0,max_depth=t["depth"],call_budget=6000)
        if r["solved"] and D.verify_hidden(t,r["prims"]):
            found.append({"task_id":t["task_id"],"depth":t["depth"],"prompt":G.prompt_for(t),"code":D.prims_to_code([tuple(x) for x in r["prims"]])})
    (EXP/a.out).write_text("\n".join(json.dumps(r) for r in found)+"\n")
    if a.gen_eval:
        ev=build([2,3],40,seed=404)
        (EXP/"data"/"frontier_eval.jsonl").write_text("\n".join(json.dumps(t) for t in ev)+"\n")
    from collections import Counter
    print(f"harvested {len(found)} {dict(Counter(r['depth'] for r in found))} from {len(pool)} tasks (seed {a.seed}) -> {a.out} [{time.time()-t0:.0f}s]")
main()
