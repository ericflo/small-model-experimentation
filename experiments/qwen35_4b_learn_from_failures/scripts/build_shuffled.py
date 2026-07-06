#!/usr/bin/env python3
"""Shuffled-DPO control (load-bearing): same chosen, but rejecteds re-paired across tasks (derangement) --
destroys the same-task correct-vs-wrong signal while preserving the contrastive geometry, negative exposure,
and length statistics. Real DPO must beat this to attribute a gain to discrimination, not loss-shape."""
import json, random
from pathlib import Path
EXP=Path(__file__).resolve().parents[1]
p=[json.loads(l) for l in (EXP/"data"/"pairs.jsonl").read_text().splitlines() if l.strip()]
n=len(p); rng=random.Random(7); perm=list(range(n))
for _ in range(500):
    rng.shuffle(perm)
    if all(perm[i]!=i for i in range(n)): break
out=[{"prompt":p[i]["prompt"],"chosen":p[i]["chosen"],"rejected":p[perm[i]]["rejected"],"depth":3} for i in range(n)]
(EXP/"data"/"pairs_shuffled.jsonl").write_text("\n".join(json.dumps(x) for x in out)+"\n")
print(f"shuffled control: {n} pairs (rejecteds deranged across tasks)")
