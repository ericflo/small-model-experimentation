#!/usr/bin/env python3
"""Compute-optimal confidence policy: accuracy-vs-compute frontier (post-hoc).

Reuses the cached MBPP candidate pool (qwen35_4b_code_confidence: 244 tasks x 9
candidates, each with full_pass ground truth, p_true single-token judge,
mean_logprob, visible_all_pass execution signal, behavior_signature). No new
inference. For each compute level k (candidates used), average over random
size-k subsets and compare deployable selection policies at MATCHED compute:
  * majority   : most common behavior_signature (ties -> highest p_true)
  * conf       : argmax p_true (C46)
  * logprob    : argmax mean_logprob (the weaker readout C46 dominated)
  * first      : the greedy/first candidate (k=1 control)
  * oracle     : any full_pass in the subset (pass@k ceiling)
  * exec       : a candidate that passes visible tests, else fall back to conf
Also the risk-coverage curve for conf+ABSTAIN at fixed k (sweep max-p_true
threshold): selective accuracy vs coverage. CPU-only, deterministic.
"""
from __future__ import annotations
import json, random, statistics
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
SRC = EXP.parents[1] / "experiments" / "qwen35_4b_code_confidence" / "runs" / "code_conf.json"
R_SUBSETS = 60   # random size-k subsets averaged per task
SEED = 31337


def pt(c):  # unparseable candidates (empty code, no judge) -> zero confidence
    v = c.get("p_true")
    return float(v) if v is not None else 0.0


def majority(cands):
    # parse-failed empties share behavior_signature "" but are never a real
    # consensus; exclude them from the vote when any parseable cand exists.
    votable = [c for c in cands if c.get("parse_ok")] or cands
    sigs = Counter(c["behavior_signature"] for c in votable)
    top = max(sigs.values())
    winners = {s for s, n in sigs.items() if n == top}
    pool = [c for c in votable if c["behavior_signature"] in winners]
    return max(pool, key=pt)  # tie-break by confidence


def pick(cands, policy):
    if policy == "majority": return majority(cands)
    if policy == "conf":     return max(cands, key=pt)
    if policy == "logprob":  return max(cands, key=lambda c: c["mean_logprob"])
    if policy == "first":    return cands[0]
    if policy == "oracle":   return max(cands, key=lambda c: c["full_pass"])
    if policy == "exec":
        ex = [c for c in cands if c["visible_all_pass"]]
        return max(ex, key=pt) if ex else max(cands, key=pt)
    raise ValueError(policy)


def main():
    tasks = json.loads(SRC.read_text())
    rng = random.Random(SEED)
    policies = ["first", "majority", "logprob", "conf", "exec", "oracle"]
    KS = list(range(1, 10))
    frontier = {p: {} for p in policies}
    for k in KS:
        acc = {p: [] for p in policies}
        for t in tasks:
            cands = t["cands"]
            subs = ([cands] if k == len(cands) else
                    [rng.sample(cands, k) for _ in range(R_SUBSETS)])
            for p in policies:
                acc[p].append(statistics.mean(pick(s, p)["full_pass"] for s in subs))
        for p in policies:
            frontier[p][k] = round(statistics.mean(acc[p]), 4)

    print("=== accuracy vs compute (k candidates) ===")
    print("  k " + "".join(f"{p:>10}" for p in policies))
    for k in KS:
        print(f"{k:>3} " + "".join(f"{frontier[p][k]:>10.4f}" for p in policies))

    # Risk-coverage for conf+abstain at k=8 (near-full pool)
    k = 8
    print(f"\n=== conf+abstain risk-coverage @k={k} (sweep max-p_true threshold) ===")
    rows = []
    for t in tasks:
        subs = [rng.sample(t["cands"], k) for _ in range(R_SUBSETS)]
        for s in subs:
            sel = max(s, key=pt)
            rows.append((pt(sel), sel["full_pass"]))
    rows.sort(reverse=True)
    n = len(rows)
    print(f"{'thresh':>8} {'coverage':>9} {'sel_acc':>8}")
    rc = []
    for thr in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        att = [r for r in rows if r[0] >= thr]
        cov = len(att) / n
        sa = statistics.mean(r[1] for r in att) if att else float("nan")
        rc.append({"thr": thr, "coverage": round(cov, 4), "sel_acc": round(sa, 4)})
        print(f"{thr:>8.2f} {cov:>9.3f} {sa:>8.4f}")

    # Solvability AUROC of max-p_true (does confidence predict a solvable task?)
    per_task = []
    for t in tasks:
        mp = max((c.get("p_true") or 0.0) for c in t["cands"])
        solvable = any(c["full_pass"] for c in t["cands"])
        per_task.append((mp, solvable))
    pos = [m for m, s in per_task if s]; neg = [m for m, s in per_task if not s]
    auroc = (sum(1 for a in pos for b in neg if a > b) +
             0.5 * sum(1 for a in pos for b in neg if a == b)) / (len(pos) * len(neg))
    print(f"\nsolvability AUROC (max p_true): {auroc:.4f}  (pos={len(pos)} neg={len(neg)})")

    out = {"frontier": frontier, "risk_coverage_k8": rc,
           "solvability_auroc": round(auroc, 4), "n_tasks": len(tasks),
           "base_pass_rate": round(statistics.mean(
               any(c["full_pass"] for c in t["cands"]) for t in tasks), 4)}
    (EXP / "runs" / "frontier.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {EXP/'runs'/'frontier.json'}")


if __name__ == "__main__":
    main()
