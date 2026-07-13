#!/usr/bin/env python3
"""C58 iterated repair: does looping localize->isolated-recompute->rerun fix
MULTI-error chains for more gain than a single repair? Each iteration: decode with
the accumulated substitutions, localize the lowest-confidence NOT-yet-repaired
step, isolated-recompute it, add it to the substitution set. Reports whole-chain
final-correct after each iteration. Reuses repair_e2e's decode. Run under .venv."""
from __future__ import annotations
import argparse, statistics, sys, json
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts")); sys.path.insert(0, str(EXP / "src"))
import chain_family as CF  # noqa
import gen_lib as GL  # noqa
from repair_e2e import decode, dists, CONDS  # noqa


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n-per-cond",type=int,default=140)
    ap.add_argument("--seed",type=int,default=800); ap.add_argument("--iters",type=int,default=3); a=ap.parse_args()
    p=GL.Probe(); nl=lambda s:p.tok(s,add_special_tokens=False).input_ids
    tasks=[CF.gen_chain(k,d,a.seed*100000+ci*1000+i) for ci,(k,d) in enumerate(CONDS) for i in range(a.n_per_cond)]
    subst={}; repaired_steps={j:set() for j in range(len(tasks))}
    print(f"[iter] {len(tasks)} chains", flush=True)
    for it in range(a.iters+1):
        vals,confs,ok=decode(p,tasks,nl,subst=subst)
        print(f"  iter {it}: final-correct {statistics.mean(ok):.3f}  (repaired steps so far: {sum(len(v) for v in repaired_steps.values())})", flush=True)
        if it==a.iters: break
        # localize lowest-conf not-yet-repaired step per chain; isolated-recompute
        iso=[]; targets=[]
        for j,t in enumerate(tasks):
            cand=[i for i in range(t["depth"]) if i not in repaired_steps[j]]
            if not cand: targets.append(None); iso.append(p._ids(p.prompt("x",enable_thinking=False))); continue
            Lg=min(cand,key=lambda i:confs[j][i]); targets.append(Lg)
            prev_d=vals[j][Lg-1] if Lg>0 else t["start"]
            iit=CF.gen_chain("familiar",1,0,k=t["k"]); iit["order"]=t["order"]; iit["start"]=prev_d; iit["nxt"]=t["nxt"]
            iso.append(p._ids(p.prompt(CF.render(iit),enable_thinking=False))+nl("Step 1: "))
        idist=dists(p,iso)
        for j,t in enumerate(tasks):
            L=targets[j]
            if L is None: continue
            repaired_steps[j].add(L)
            newd=str(max(range(10),key=lambda x:idist[j][x]))
            if newd!=vals[j][L]:
                subst.setdefault(j,{})[L]=newd
    (EXP/"runs"/"repair_iter.json").write_text(json.dumps({"n":len(tasks),"iters":a.iters},indent=2)+"\n")


if __name__=="__main__": main()
