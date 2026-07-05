#!/usr/bin/env python3
"""Analyze the 3 arms: saturation (depth-3 dose curve past 640, vs distinct functions), data-vs-compute
(upsampled-40 2x2), depth-4 rung (banked_d4 vs SCAFFOLD-only baseline). Wilson CIs + CI-overlap verdicts."""
from __future__ import annotations

import json
from collections import defaultdict
from math import comb, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
# distinct-function counts per depth-3 dose (measured in setup; C23 low end + this run's high end)
DISTINCT = {40: 39, 160: 153, 640: 555, 1280: 1156, 2560: 2305}


def cov_at_k(c, n, k):
    k = min(k, n)
    return 0.0 if c == 0 else (1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 3), round(max(0, c - h), 3), round(min(1, c + h), 3))


def load(tag, depth):
    p = EXP / "runs" / f"eval_{tag}.json"
    if not p.exists():
        return None
    rs = [r for r in json.load(open(p))["records"] if r["depth"] == depth]
    if not rs:
        return None
    n, K = len(rs), rs[0]["K"]
    ns = sum(1 for r in rs if r["cov_full"] >= 1)
    return {"n": n, "K": K, "n_solved": ns, "covK": sum(cov_at_k(r["cov_full"], r["K"], K) for r in rs) / n,
            "per_sample": sum(r["cov_full"] for r in rs) / (n * K), "greedy": sum(r["greedy_full"] for r in rs) / n,
            "wilson": wilson(ns, n), "curve": [sum(cov_at_k(r["cov_full"], r["K"], k) for r in rs) / n for k in range(1, K + 1)]}


def overlap(a, b):
    return not (a["wilson"][1] > b["wilson"][2] or b["wilson"][1] > a["wilson"][2])


def main():
    v = {}
    # ---- ARM 1: saturation ----
    doses = [(0, "base_th3"), (40, "b40_th3"), (160, "b160_th3"), (640, "b640_th3"), (1280, "b1280_th3"), (2560, "b2560_th3")]
    arm1 = {N: load(tag, 3) for N, tag in doses if load(tag, 3)}
    print("\n=== ARM 1: saturation — depth-3 think cov@16 vs dose (n=80) ===")
    print(f"{'N':>5} {'distinct':>8} {'covK':>6} {'Wilson95':>16} {'per-sample':>11} {'greedy@1':>9}")
    for N in sorted(arm1):
        m = arm1[N]; p, lo, hi = m["wilson"]
        print(f"{N:>5} {DISTINCT.get(N, N):>8} {m['covK']:>6.3f} {f'[{lo:.2f},{hi:.2f}]':>16} {m['per_sample']:>11.3f} {m['greedy']:>9.3f}")
    nz = [N for N in sorted(arm1) if N > 0]
    if 640 in arm1 and len(nz) >= 2:
        top = nz[-1]; prev = nz[-2]  # e.g. top=1280, prev=640
        v["arm1_saturation"] = {"covK": {N: arm1[N]["covK"] for N in sorted(arm1)},
                                "distinct": {N: DISTINCT.get(N, N) for N in sorted(arm1)},
                                "top_dose": top, f"rise_{prev}_to_{top}": round(arm1[top]["covK"] - arm1[prev]["covK"], 3),
                                "top_two_CIs_overlap(plateau)": overlap(arm1[top], arm1[prev]),
                                "still_rising_top_vs_prev": bool(arm1[top]["wilson"][1] > arm1[prev]["wilson"][2]),
                                "note": "2560 dose dropped (training ~2x slower than estimated); top dose = 1280 (1156 distinct funcs)"}

    # ---- ARM 2: data-diversity vs compute (2x2) ----
    n40, up40, n640 = arm1.get(40), load("b_up40_th3", 3), arm1.get(640)
    if up40 and n40 and n640:
        print("\n=== ARM 2: data-diversity vs compute (depth-3 cov@16) ===")
        print(f"  N=40 dose   (40 distinct, 120 visits): {n40['covK']:.3f} {n40['wilson']}")
        print(f"  up40        (40 distinct, 1920 visits): {up40['covK']:.3f} {up40['wilson']}")
        print(f"  train_640   (640 distinct, 1920 visits): {n640['covK']:.3f} {n640['wilson']}")
        diversity_wins = bool(n640["wilson"][1] > up40["wilson"][2])   # 640 > up40 at fixed compute
        compute_helps = bool(up40["wilson"][1] > n40["wilson"][2])     # up40 > N=40 at fixed diversity
        v["arm2_data_vs_compute"] = {"N40_covK": n40["covK"], "up40_covK": up40["covK"], "train640_covK": n640["covK"],
                                     "diversity_wins_at_fixed_compute": diversity_wins,
                                     "compute_helps_at_fixed_diversity": compute_helps,
                                     "interpretation": ("data-diversity dominates" if diversity_wins and not compute_helps
                                                        else "both diversity and compute contribute" if diversity_wins and compute_helps
                                                        else "compute/exposure dominates" if compute_helps and not diversity_wins
                                                        else "inconclusive")}

    # ---- ARM 3: depth-4 rung ----
    base4, scaf4, bank4 = load("base_d4", 4), load("scaffold_d4", 4), load("banked_d4_d4", 4)
    guard3 = load("banked_d4_d3guard", 3)
    if scaf4 and bank4:
        print("\n=== ARM 3: depth-4 rung (cov@16, n=60) ===")
        print(f"  raw base:            {base4['covK']:.3f} {base4['wilson'] if base4 else ''}")
        print(f"  scaffold (d1+2+640d3): {scaf4['covK']:.3f} {scaf4['wilson']}  <- the attribution baseline")
        print(f"  banked_d4 (+320 d4):  {bank4['covK']:.3f} {bank4['wilson']}  greedy@1 {bank4['greedy']:.3f}")
        if guard3:
            print(f"  depth-3 guardrail (banked_d4 on d3): {guard3['covK']:.3f} (scaffold forgetting check)")
        installs = bool(bank4["wilson"][1] > scaf4["wilson"][2])  # banked_d4 > scaffold, non-overlapping
        v["arm3_depth4"] = {"base_covK": base4["covK"] if base4 else None, "scaffold_covK": scaf4["covK"],
                            "banked_d4_covK": bank4["covK"], "banked_d4_greedy": bank4["greedy"],
                            "depth3_guardrail": guard3["covK"] if guard3 else None,
                            "recipe_repeats(installs_vs_scaffold)": installs,
                            "delta_vs_scaffold": round(bank4["covK"] - scaf4["covK"], 3)}
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "verdict.json").write_text(json.dumps(v, indent=1))
    print("\n=== VERDICTS ===")
    print(json.dumps(v, indent=1))

    # ---- figure: 3 panels ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    # Arm1 saturation
    ax = axes[0]
    Ns = sorted(arm1); xs = [max(1, DISTINCT.get(N, N)) for N in Ns]
    cov = [arm1[N]["covK"] for N in Ns]
    lo = [arm1[N]["covK"] - arm1[N]["wilson"][1] for N in Ns]; hi = [arm1[N]["wilson"][2] - arm1[N]["covK"] for N in Ns]
    ax.errorbar(xs, cov, yerr=[lo, hi], fmt="o-", color="#4a3aa7", lw=2, capsize=4)
    ax.set_xscale("log"); ax.set_xlabel("distinct depth-3 functions banked"); ax.set_ylabel("depth-3 cov@16 (think)")
    ax.set_title("Arm 1: saturation?"); ax.set_ylim(-0.02, max(cov) + 0.12); ax.grid(alpha=0.25)
    for N, x, y in zip(Ns, xs, cov):
        ax.annotate(f"N={N}", (x, y), fontsize=7, xytext=(0, 6), textcoords="offset points", ha="center")
    # Arm2 2x2
    ax = axes[1]
    if up40 and n40 and n640:
        bars = [("N=40\n40d,120v", n40), ("up40\n40d,1920v", up40), ("640\n640d,1920v", n640)]
        ax.bar(range(3), [m["covK"] for _, m in bars], color=["#94a3b8", "#eda100", "#4a3aa7"])
        ax.errorbar(range(3), [m["covK"] for _, m in bars],
                    yerr=[[m["covK"]-m["wilson"][1] for _,m in bars],[m["wilson"][2]-m["covK"] for _,m in bars]],
                    fmt="none", ecolor="black", capsize=4)
        ax.set_xticks(range(3)); ax.set_xticklabels([l for l, _ in bars], fontsize=8)
    ax.set_title("Arm 2: data-diversity vs compute"); ax.set_ylabel("depth-3 cov@16"); ax.grid(alpha=0.25, axis="y")
    # Arm3 depth-4
    ax = axes[2]
    if scaf4 and bank4:
        labels = ["raw base", "scaffold\n(640 d3)", "banked_d4\n(+320 d4)"]
        vals = [base4["covK"] if base4 else 0, scaf4["covK"], bank4["covK"]]
        ax.bar(range(3), vals, color=["#cbd5e1", "#94a3b8", "#16a34a"])
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Arm 3: does depth-4 install?"); ax.set_ylabel("depth-4 cov@16"); ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Depth scaling & controls: saturation, data-vs-compute, depth-4 rung", y=1.02, fontsize=12)
    fig.tight_layout()
    (EXP / "analysis").mkdir(exist_ok=True)
    fig.savefig(EXP / "analysis" / "scaling_controls.png", dpi=130, bbox_inches="tight")
    print("\nwrote analysis/scaling_controls.png and runs/verdict.json")


if __name__ == "__main__":
    main()
