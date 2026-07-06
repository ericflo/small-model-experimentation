#!/usr/bin/env python3
"""Split harvest_phase2.jsonl into matched training arms (identical canonical code; only the trace differs):
Aself=answers, Tself=model's own thoughts, Tsynth=explicit plan, Tselfcorrupt=mismatched model thoughts."""
import json, random
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
recs = [json.loads(l) for l in (EXP/"data"/"harvest_phase2.jsonl").read_text().splitlines() if l.strip()]
n = len(recs)
def w(name, rows): (EXP/"data"/name).write_text("\n".join(json.dumps(r) for r in rows)+"\n")
w("train_Aself.jsonl",  [{"prompt":r["prompt"],"code":r["code"],"depth":3} for r in recs])
w("train_Tself.jsonl",  [{"prompt":r["prompt"],"thinking":r["model_thinking"],"code":r["code"],"depth":3} for r in recs])
w("train_Tsynth.jsonl", [{"prompt":r["prompt"],"thinking":r["synth_thinking"],"code":r["code"],"depth":3} for r in recs])
rng=random.Random(13); perm=list(range(n))
for _ in range(200):
    rng.shuffle(perm)
    if all(perm[i]!=i for i in range(n)): break
w("train_Tselfcorrupt.jsonl", [{"prompt":recs[i]["prompt"],"thinking":recs[perm[i]]["model_thinking"],"code":recs[i]["code"],"depth":3} for i in range(n)])
import statistics
ml=[len(r["model_thinking"]) for r in recs]; sl=[len(r["synth_thinking"]) for r in recs]
print(f"phase2 built: {n} matched tasks | model-thinking median {statistics.median(ml):.0f} chars, synth-plan median {statistics.median(sl):.0f} chars")
