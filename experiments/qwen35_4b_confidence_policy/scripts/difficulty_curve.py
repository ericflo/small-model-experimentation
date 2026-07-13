#!/usr/bin/env python3
"""Is the conf-select-vs-majority advantage difficulty-dependent? Pool the two
cached code pools (MBPP hard-ish + HumanEval easy), bin TASKS by difficulty (the
fraction of the 9 candidates that pass), and measure conf-select minus
majority-vote accuracy at k=6 per bin. Pure post-hoc, deterministic."""
import json, random, statistics
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
POOLS = {
    "mbpp": EXP.parents[1] / "experiments/qwen35_4b_code_confidence/runs/code_conf.json",
    "humaneval": EXP.parents[1] / "experiments/qwen35_4b_humaneval_code_confidence/runs/humaneval_code_conf.json",
}
R, K, SEED = 200, 6, 31337


def pt(c): 
    v = c.get("p_true"); return float(v) if v is not None else 0.0

def majority(cands):
    votable = [c for c in cands if c.get("parse_ok")] or cands
    sigs = Counter(c["behavior_signature"] for c in votable)
    top = max(sigs.values()); win = {s for s,n in sigs.items() if n==top}
    return max([c for c in votable if c["behavior_signature"] in win], key=pt)

def main():
    rng = random.Random(SEED)
    tasks = []
    for name, p in POOLS.items():
        for t in json.loads(p.read_text()):
            cs = t["cands"]
            passrate = statistics.mean(c["full_pass"] for c in cs)
            tasks.append((name, passrate, cs))
    print(f"pooled tasks: {len(tasks)} (mbpp {sum(1 for n,_,_ in tasks if n=='mbpp')}, "
          f"humaneval {sum(1 for n,_,_ in tasks if n=='humaneval')})")
    bins = [(0.0,0.34,"hard"),(0.34,0.67,"medium"),(0.67,1.01,"easy")]
    print(f"\n{'difficulty':>10} {'n':>4} {'passrate':>9} {'conf':>7} {'majority':>9} {'conf-maj':>9}")
    rows=[]
    for lo,hi,lab in bins:
        grp=[cs for _,pr,cs in tasks if lo<=pr<hi]
        if not grp: continue
        cacc,macc=[],[]
        for cs in grp:
            subs=[cs] if K>=len(cs) else [rng.sample(cs,K) for _ in range(R)]
            cacc.append(statistics.mean(max(s,key=pt)["full_pass"] for s in subs))
            macc.append(statistics.mean(majority(s)["full_pass"] for s in subs))
        cm,mm=statistics.mean(cacc),statistics.mean(macc)
        pr=statistics.mean(statistics.mean(c["full_pass"] for c in cs) for cs in grp)
        print(f"{lab:>10} {len(grp):>4} {pr:>9.3f} {cm:>7.3f} {mm:>9.3f} {cm-mm:>+9.3f}")
        rows.append({"bin":lab,"n":len(grp),"passrate":round(pr,3),"conf":round(cm,3),
                     "majority":round(mm,3),"conf_minus_maj":round(cm-mm,3)})
    (EXP/"runs"/"difficulty_curve.json").write_text(json.dumps({"bins":rows},indent=2)+"\n")
    print("\nwrote runs/difficulty_curve.json")

if __name__=="__main__": main()
