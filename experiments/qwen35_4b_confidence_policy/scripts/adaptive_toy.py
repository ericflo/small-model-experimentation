#!/usr/bin/env python3
"""Cross-domain check: does confidence-gated adaptive ALLOCATION generalize from
code to (toy) REASONING? Reuses the original C41 pool
(qwen35_4b_confidence_guided_compute: 240 records x k=12 samples; per-sample
P(answer) 'p' + correctness 'c'; a greedy answer with greedy_p/greedy_correct).
Confidence here is the answer-token probability (C40), not a P(True) judge.
Adaptive: commit greedy if greedy_p>=thr, else sample K=8 + conf-select (argmax
p). Compare to uniform-k conf-select at matched average compute. Post-hoc, CPU."""
import json, random, statistics
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
SRC = EXP.parents[1] / "experiments/qwen35_4b_confidence_guided_compute/runs/sampling_records.json"
R, SEED = 300, 31337


def conf_select(samples):
    return max(samples, key=lambda s: s["p"])["c"]


def main():
    rng = random.Random(SEED)
    recs = json.loads(SRC.read_text())["records"]
    print(f"records: {len(recs)} | conditions: {dict((k,sum(1 for r in recs if r['cond']==k)) for k in set(r['cond'] for r in recs))}")

    # uniform-k conf-select
    print("\n=== uniform-k (conf-select by P(answer)) ===")
    uni = []
    for k in range(1, 13):
        acc = []
        for r in recs:
            s = r["samples"]
            subs = [s] if k >= len(s) else [rng.sample(s, k) for _ in range(R)]
            acc.append(statistics.mean(conf_select(x) for x in subs))
        uni.append((k, statistics.mean(acc)))
    for k, a in uni:
        print(f"  k={k:>2} acc={a:.4f}")

    # adaptive: commit greedy if greedy_p>=thr else sample K=8 + conf-select
    print("\n=== confidence-gated adaptive (K=8 on low-confidence) ===")
    K = 8
    ada = []
    for thr in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]:
        acc, samp = [], []
        for r in recs:
            if r["greedy_p"] >= thr:
                acc.append(float(r["greedy_correct"])); samp.append(1.0)
            else:
                s = r["samples"]
                subs = [s] if K >= len(s) else [rng.sample(s, K) for _ in range(R)]
                acc.append(statistics.mean(conf_select(x) for x in subs)); samp.append(1.0 + K)
        ada.append((statistics.mean(samp), statistics.mean(acc)))
        print(f"  thr={thr:>4} avg_samples={statistics.mean(samp):>5.2f} acc={statistics.mean(acc):.4f}")

    def uni_at(c):
        for (k1, a1), (k2, a2) in zip(uni, uni[1:]):
            if k1 <= c <= k2:
                return a1 + (a2 - a1) * (c - k1) / (k2 - k1)
        return uni[-1][1] if c > uni[-1][0] else uni[0][1]
    print("\n=== adaptive vs uniform at matched average compute ===")
    wins = 0
    for c, a in ada:
        u = uni_at(c); d = a - u; wins += d > 0.002
        print(f"  compute {c:>5.2f}: adaptive {a:.4f} vs uniform {u:.4f}  delta {d:+.4f}")
    print(f"\nadaptive beats uniform at {wins}/{len(ada)} operating points")
    (EXP / "runs" / "adaptive_toy.json").write_text(json.dumps(
        {"uniform": [{"k": k, "acc": round(a, 4)} for k, a in uni],
         "adaptive": [{"avg_samples": round(c, 3), "acc": round(a, 4)} for c, a in ada]}, indent=2) + "\n")
    print("wrote runs/adaptive_toy.json")


if __name__ == "__main__":
    main()
