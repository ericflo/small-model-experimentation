#!/usr/bin/env python3
"""Phase 2: does banking the model's OWN reasoning (Tself) beat banking answers (Aself)? And does the model's
own (likely rationalized) reasoning bank as well as an explicit plan (Tsynth)? All on identical tasks/code."""
import json, os
from math import comb, sqrt
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
EXP = Path(__file__).resolve().parents[1]
def cov(c,n,k): k=min(k,n); return 0.0 if c==0 else (1.0 if n-c<k else 1-comb(n-c,k)/comb(n,k))
def dep(tag):
    p=EXP/"runs"/f"eval_{tag}.json"
    if not p.exists(): return None
    r=[x for x in json.load(open(p))["records"] if x["depth"]==3]; K=r[0]["K"]; n=len(r)
    return {"cov16":sum(cov(x["cov_full"],x["K"],16) for x in r)/n, "greedy":sum(x["greedy_full"] for x in r)/n, "n":n}
cells=[("p2_Aself_nt","A_self=answers(nt)"),("p2_Tself_th","T_self=model's own thoughts(think)"),
       ("p2_Tsynth_th","T_synth=explicit plan(think)"),("p2_Tselfcorrupt_th","T_selfcorrupt=mismatched(think)")]
d={}; print("\n=== PHASE 2 deployability (depth-3 held-out, n=80) ===")
print(f"{'cell':>34} {'cov@16':>8} {'greedy@1':>9}")
for tag,lab in cells:
    x=dep(tag)
    if not x: print(f"{lab:>34}: pending"); continue
    d[tag]=x; print(f"{lab:>34} {x['cov16']:>8.3f} {x['greedy']:>9.3f}")
v={"deploy":{t:{"cov16":round(d[t]["cov16"],3),"greedy":round(d[t]["greedy"],3)} for t in d}}
if "p2_Aself_nt" in d and "p2_Tself_th" in d:
    v["Tself_beats_Aself_cov16"]=round(d["p2_Tself_th"]["cov16"]-d["p2_Aself_nt"]["cov16"],3)
if "p2_Tself_th" in d and "p2_Tsynth_th" in d:
    v["Tself_vs_Tsynth_cov16"]=round(d["p2_Tself_th"]["cov16"]-d["p2_Tsynth_th"]["cov16"],3)
if "p2_Tself_th" in d and "p2_Tselfcorrupt_th" in d:
    v["Tself_content_causal_cov16"]=round(d["p2_Tself_th"]["cov16"]-d["p2_Tselfcorrupt_th"]["cov16"],3)
(EXP/"runs"/"verdict_phase2.json").write_text(json.dumps(v,indent=1))
print("\n=== VERDICT ==="); print(json.dumps(v,indent=1))
fig,ax=plt.subplots(figsize=(9,4.8))
order=["p2_Aself_nt","p2_Tself_th","p2_Tsynth_th","p2_Tselfcorrupt_th"]
labs=["A_self\nanswers","T_self\nmodel's own\nthoughts","T_synth\nexplicit\nplan","T_selfcorrupt\nmismatched\nthoughts"]
cols=["#94a3b8","#16a34a","#4a3aa7","#f59e0b"]
xs=[i for i,t in enumerate(order) if t in d]
ax.bar(xs,[d[order[i]]["cov16"] for i in xs],0.38,label="coverage@16",color=[cols[i] for i in xs])
ax.bar([x+0.4 for x in xs],[d[order[i]]["greedy"] for i in xs],0.38,label="greedy@1",color=[cols[i] for i in xs],alpha=0.55,hatch="//")
ax.set_xticks([x+0.2 for x in xs]); ax.set_xticklabels([labs[i] for i in xs],fontsize=8)
ax.set_ylabel("depth-3 solve rate (held-out)"); ax.legend(fontsize=9); ax.grid(alpha=0.25,axis="y")
ax.set_title("Phase 2: banking the model's OWN reasoning vs answers vs explicit plans (matched tasks/code)")
fig.tight_layout(); (EXP/"analysis").mkdir(exist_ok=True)
fig.savefig(EXP/"analysis"/"bank_thoughts_phase2.png",dpi=130,bbox_inches="tight")
print("wrote analysis/bank_thoughts_phase2.png and runs/verdict_phase2.json")
