#!/usr/bin/env python3
"""Brute-full structure-search DEPLOY at depth-4 (no model, pure CPU): enumerate all 16^4 skeletons, value-fill
against true visible, execution-consensus deploy. The model-guided (bank-fill) deploy is capped at banked_d4
structure-cov=0.10 (gate). Confirms brute still dominates at depth-4."""
import json, random, sys, time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src")); sys.path.insert(0, str(EXP / "scripts"))
import families as FAM  # noqa: E402
from bank_fill import brute_candidates, consensus_deploy, exec_seq, solves  # noqa: E402
FAM_L = FAM.FAMILIES["list"]

tasks = [json.loads(l) for l in (EXP / "data" / "eval_frozen_d4.jsonl").read_text().splitlines() if l.strip()]
tasks = [t for t in tasks if not FAM.min_depth_leq(
    FAM_L, [e["input"] for e in t["visible"] + t["hidden"]],
    [e["output"] for e in t["visible"] + t["hidden"]], t["depth"] - 1)]
rng = random.Random(2024)
probes = [[rng.randint(-9, 9) for _ in range(rng.randint(4, 8))] for _ in range(16)]
dep = cov = 0; t0 = time.time(); costs = []
for i, t in enumerate(tasks):
    c = brute_candidates(t, t["depth"]); costs.append(len({x[0] for x in c}))
    cov += int(any(solves(f, t["hidden"]) for _, f in c))
    dep += int(bool(consensus_deploy(c, probes, t)))
    if (i + 1) % 15 == 0:
        print(f"  {i+1}/{len(tasks)} [{time.time()-t0:.0f}s] deploy-so-far {dep/(i+1):.3f}", flush=True)
n = len(tasks)
print(f"[brute d4] n={n} | brute-full DEPLOY {dep/n:.3f} | coverage {cov/n:.3f} | "
      f"mean surviving skeletons/task {sum(costs)/n:.1f} | enumerated 16^4={16**4}/task | {time.time()-t0:.0f}s total", flush=True)
json.dump({"n": n, "brute_deploy": round(dep / n, 3), "brute_cov": round(cov / n, 3), "enum_per_task": 16**4},
          open(EXP / "runs" / "brute_d4.json", "w"), indent=1)
