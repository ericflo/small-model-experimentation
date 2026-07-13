#!/usr/bin/env python3
"""The compute-optimal confidence policy: confidence-GATED adaptive sampling.

The difficulty curve showed conf-select helps only on hard tasks. So a
deployable allocator should spend samples only there: take the greedy answer,
read its P(True); if high-confidence, COMMIT it (1 sample); else sample K more
and conf-select (1+K samples). Sweeping the gate threshold traces an
accuracy-vs-average-compute frontier. Compare it to the UNIFORM-k frontier
(always sample k, conf-select). Does confidence-gated allocation dominate uniform
sampling at matched average compute? Post-hoc on the cached MBPP pool. CPU-only.
"""
import json, random, statistics
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
SRC = EXP.parents[1] / "experiments/qwen35_4b_code_confidence/runs/code_conf.json"
R, SEED = 300, 31337


def pt(c):
    v = c.get("p_true"); return float(v) if v is not None else 0.0


def main():
    rng = random.Random(SEED)
    tasks = json.loads(SRC.read_text())
    # each task: a 'greedy' candidate + sampled ones
    def greedy(t): return next((c for c in t["cands"] if c["tag"] == "greedy"), t["cands"][0])
    def sampled(t): return [c for c in t["cands"] if c["tag"] != "greedy"]

    # UNIFORM-k frontier: always sample k, conf-select (k samples/task)
    print("=== uniform-k (conf-select) ===")
    print(f"{'avg_samples':>12} {'accuracy':>9}")
    uni = []
    for k in range(1, 10):
        accs = []
        for t in tasks:
            pool = t["cands"]
            subs = [pool] if k >= len(pool) else [rng.sample(pool, k) for _ in range(R)]
            accs.append(statistics.mean(max(s, key=pt)["full_pass"] for s in subs))
        uni.append((k, statistics.mean(accs)))
        print(f"{k:>12} {statistics.mean(accs):>9.4f}")

    # ADAPTIVE frontier: commit greedy if p_true>=thr, else sample K=8 + conf-select
    print("\n=== confidence-gated adaptive (K=8 on low-confidence) ===")
    print(f"{'threshold':>10} {'avg_samples':>12} {'accuracy':>9}")
    ada = []
    K = 8
    for thr in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]:
        accs, samp = [], []
        for t in tasks:
            g = greedy(t); sp = sampled(t)
            if pt(g) >= thr:
                accs.append(float(g["full_pass"])); samp.append(1.0)
            else:
                pool = sp if len(sp) >= K else t["cands"]
                subs = [pool] if K >= len(pool) else [rng.sample(pool, K) for _ in range(R)]
                accs.append(statistics.mean(max(s, key=pt)["full_pass"] for s in subs))
                samp.append(1.0 + K)
        ada.append((statistics.mean(samp), statistics.mean(accs)))
        print(f"{thr:>10.2f} {statistics.mean(samp):>12.2f} {statistics.mean(accs):>9.4f}")

    # Compare: at each adaptive avg-compute, interpolate the uniform accuracy.
    print("\n=== adaptive vs uniform at matched average compute ===")
    def uni_at(c):
        for (k1, a1), (k2, a2) in zip(uni, uni[1:]):
            if k1 <= c <= k2:
                return a1 + (a2 - a1) * (c - k1) / (k2 - k1)
        return uni[-1][1] if c > uni[-1][0] else uni[0][1]
    wins = 0
    for c, a in ada:
        u = uni_at(c)
        d = a - u
        wins += d > 0.002
        print(f"  compute {c:>5.2f}: adaptive {a:.4f} vs uniform {u:.4f}  delta {d:+.4f}")
    print(f"\nadaptive strictly beats uniform at {wins}/{len(ada)} operating points")
    (EXP / "runs" / "adaptive_compute.json").write_text(json.dumps(
        {"uniform": [{"k": k, "acc": round(a, 4)} for k, a in uni],
         "adaptive": [{"avg_samples": round(c, 3), "acc": round(a, 4)} for c, a in ada]},
        indent=2) + "\n")
    print("wrote runs/adaptive_compute.json")


if __name__ == "__main__":
    main()
