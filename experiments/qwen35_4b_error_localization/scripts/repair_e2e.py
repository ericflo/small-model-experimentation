#!/usr/bin/env python3
"""C57 x C58 end-to-end: does a localize->isolated-recompute->rerun loop lift
whole-chain accuracy? For each familiar +k chain: greedy scaffold-decode (baseline
final-correct); pick ONE step to repair by a policy (localizer = argmin per-step
confidence; random; oracle = the true first local error); ISOLATED-recompute that
step (depth-1 chain from its prev, same scaffold); if the recomputed digit differs,
substitute it and RE-RUN the chain downstream greedily; measure the new
final-correct. Reports baseline vs after-repair for each policy. Fast toy chains,
HF, one forward pass per step. Run under .venv."""
from __future__ import annotations
import argparse, json, random, statistics, sys
from pathlib import Path
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts")); sys.path.insert(0, str(EXP / "src"))
import chain_family as CF  # noqa
import gen_lib as GL  # noqa
DIGIT_IDS = [15 + d for d in range(10)]
CONDS = [("familiar", 4), ("familiar", 5), ("familiar", 6), ("familiar", 7)]


@torch.no_grad()
def dists(p, prefixes, bs=64):
    out=[None]*len(prefixes); pad=p.tok.pad_token_id
    order=sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
    for s in range(0,len(order),bs):
        sub=order[s:s+bs]; seqs=[prefixes[i] for i in sub]; ml=max(len(x) for x in seqs)
        ids=torch.tensor([[pad]*(ml-len(x))+x for x in seqs], device=p.device)
        o=p.model(input_ids=ids, attention_mask=(ids!=pad).long(), logits_to_keep=1)
        pr=torch.softmax(o.logits[:,-1,DIGIT_IDS].float(),dim=1).cpu().tolist()
        for i,v in zip(sub,pr): out[i]=v
    return out


def decode(p, tasks, nl, subst=None):
    """Greedy scaffold-decode. subst[j] = {step_index0: forced_digit} overrides."""
    prefixes=[p._ids(p.prompt(CF.render(t), enable_thinking=False))+nl("Step 1: ") for t in tasks]
    prev=[t["start"] for t in tasks]; vals=[[] for _ in tasks]
    confs=[[] for _ in tasks]
    maxd=max(t["depth"] for t in tasks)
    for i in range(1,maxd+1):
        active=[j for j,t in enumerate(tasks) if i<=t["depth"]]
        ds=dists(p,[prefixes[j] for j in active])
        for j,dist in zip(active,ds):
            forced = subst.get(j,{}).get(i-1) if subst else None
            d = int(forced) if forced is not None else max(range(10),key=lambda x:dist[x])
            val=str(d); vals[j].append(val); confs[j].append(dist[d]); 
            t=tasks[j]
            prefixes[j]=prefixes[j]+nl(f"{val}\nStep {i+1}: " if i<t["depth"] else f"{val}")
            prev[j]=val
    final_ok=[int(vals[j][t["depth"]-1]==t["chain"][t["depth"]]) for j,t in enumerate(tasks)]
    return vals, confs, final_ok


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n-per-cond",type=int,default=140)
    ap.add_argument("--seed",type=int,default=800); a=ap.parse_args()
    rng=random.Random(1); p=GL.Probe(); nl=lambda s:p.tok(s,add_special_tokens=False).input_ids
    tasks=[CF.gen_chain(k,d,a.seed*100000+ci*1000+i) for ci,(k,d) in enumerate(CONDS) for i in range(a.n_per_cond)]
    vals,confs,base_ok=decode(p,tasks,nl)
    print(f"[e2e] {len(tasks)} chains | baseline final-correct {statistics.mean(base_ok):.3f}", flush=True)

    # choose repair target per policy
    def localizer(j): return min(range(tasks[j]["depth"]), key=lambda i: confs[j][i])
    def randstep(j):  return rng.randrange(tasks[j]["depth"])
    def oracle(j):
        errs=[i for i in range(tasks[j]["depth"]) if vals[j][i]!=tasks[j]["nxt"].get(vals[j][i-1] if i>0 else tasks[j]["start"],"")]
        return errs[0] if errs else localizer(j)

    for name,pol in [("localizer(min-conf)",localizer),("random-step",randstep),("oracle-first-error",oracle)]:
        # isolated-recompute the target step (depth-1 chain from its prev)
        iso=[]
        for j,t in enumerate(tasks):
            L=pol(j); prev_d = vals[j][L-1] if L>0 else t["start"]
            it=CF.gen_chain("familiar",1,0,k=t["k"]); it["order"]=t["order"]; it["start"]=prev_d; it["nxt"]=t["nxt"]
            iso.append(p._ids(p.prompt(CF.render(it),enable_thinking=False))+nl("Step 1: "))
        idist=dists(p,iso)
        subst={}
        for j,t in enumerate(tasks):
            L=pol(j); newd=str(max(range(10),key=lambda x:idist[j][x]))
            if newd!=vals[j][L]:
                subst[j]={L:newd}
        # re-run chains that changed
        _,_,rep_ok=decode(p,tasks,nl,subst=subst)
        changed=len(subst)
        print(f"  {name:>22}: repaired {changed:>3} chains -> final-correct {statistics.mean(rep_ok):.3f} "
              f"(delta {statistics.mean(rep_ok)-statistics.mean(base_ok):+.3f})")
    (EXP/"runs"/"repair_e2e.json").write_text(json.dumps({"baseline":round(statistics.mean(base_ok),3),"n":len(tasks)},indent=2)+"\n")


if __name__=="__main__": main()
