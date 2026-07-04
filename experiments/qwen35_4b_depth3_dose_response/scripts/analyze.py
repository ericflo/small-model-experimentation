#!/usr/bin/env python3
"""Depth-3 dose-response: is the weak install data-limited (rises with N) or a representational cap
(plateaus)? Think coverage@16 dose curve (base/40/160/640) with Wilson CIs + a dense per-sample solve rate;
no-think (deployable) at the top dose; depth-2 guardrail. CI-overlap decision, not a bare p-value."""
from __future__ import annotations

import json
from collections import defaultdict
from math import comb, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
DOSES = [40, 160, 640]


def cov_at_k(c, n, k):
    k = min(k, n)
    return 0.0 if c == 0 else (1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 3), round(max(0, c - h), 3), round(min(1, c + h), 3))


def load(tag, depth=3):
    p = EXP / "runs" / f"eval_{tag}.json"
    if not p.exists():
        return None
    rs = [r for r in json.load(open(p))["records"] if r["depth"] == depth]
    if not rs:
        return None
    n, K = len(rs), rs[0]["K"]
    n_solved = sum(1 for r in rs if r["cov_full"] >= 1)         # tasks with >=1 correct sample (coverage@K)
    total_hits = sum(r["cov_full"] for r in rs)                 # dense: correct samples over n*K
    return {"n": n, "K": K, "n_solved": n_solved,
            "covK": sum(cov_at_k(r["cov_full"], r["K"], K) for r in rs) / n,
            "per_sample_rate": total_hits / (n * K),
            "greedy": sum(r["greedy_full"] for r in rs) / n,
            "wilson": wilson(n_solved, n),
            "curve": [sum(cov_at_k(r["cov_full"], r["K"], k) for r in rs) / n for k in range(1, K + 1)]}


def main():
    base = load("base_th3")
    dose = {N: load(f"b{N}_th3") for N in DOSES}
    xs = [0] + DOSES
    pts = [base] + [dose[N] for N in DOSES]
    K = base["K"]

    print(f"\n=== Depth-3 dose-response: think coverage@{K} (frozen paired held-out, n={base['n']}) ===")
    print(f"{'N(depth3)':>9} {'covK':>6} {'Wilson95':>16} {'per-sample':>11} {'greedy@1':>9} {'#solved':>8}")
    for x, m in zip(xs, pts):
        p, lo, hi = m["wilson"]
        print(f"{x:>9} {m['covK']:>6.3f} {f'[{lo:.2f},{hi:.2f}]':>16} {m['per_sample_rate']:>11.3f} "
              f"{m['greedy']:>9.3f} {m['n_solved']:>4}/{m['n']}")

    b40, b640 = dose[40], dose[640]
    rise = round(b640["covK"] - b40["covK"], 3)
    top_lo = b640["wilson"][1]
    low_hi = b40["wilson"][2]
    # DATA-LIMITED: material rise AND top-dose lower CI above low-dose upper CI (non-overlapping, ordered up)
    data_limited = bool(rise >= 0.10 and top_lo > low_hi)
    # CAP: top dose within noise of 160 (CIs overlap) and no material rise
    cap = bool(not data_limited and abs(b640["covK"] - dose[160]["covK"]) <= 0.05
               and b640["wilson"][1] <= dose[160]["wilson"][2] and dose[160]["wilson"][1] <= b640["wilson"][2])

    nt_base, nt_640 = load("base_nt3"), load("b640_nt3")
    v = {
        "think_depth3": {str(x): {"covK": m["covK"], "wilson": m["wilson"], "per_sample": round(m["per_sample_rate"], 3),
                                  "greedy": m["greedy"]} for x, m in zip(xs, pts)},
        "rise_40_to_640": rise, "top_lowerCI": top_lo, "low_upperCI": low_hi,
        "deployable_nothink_depth3": None if not nt_640 else {
            "base_covK": nt_base["covK"] if nt_base else None, "banked640_covK": nt_640["covK"],
            "banked640_per_sample": round(nt_640["per_sample_rate"], 3), "banked640_greedy": nt_640["greedy"],
            "reaches_deployable": bool(nt_640["covK"] > 0.05)},
        "VERDICT": "DATA-LIMITED" if data_limited else ("REPRESENTATIONAL-CAP" if cap else "INCONCLUSIVE"),
    }
    # depth-2 guardrail
    g_b, g_640 = load("base_th2", 2), load("b640_th2", 2)
    if g_b and g_640:
        v["depth2_guardrail"] = {"base_covK": g_b["covK"], "banked640_covK": g_640["covK"]}
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "verdict.json").write_text(json.dumps(v, indent=1))
    print(f"\nrise 40->640: {rise:+.3f} | top-dose lowerCI {top_lo} vs low-dose upperCI {low_hi}")
    if nt_640:
        print(f"deployable no-think@640 depth-3: covK {nt_640['covK']:.3f}, per-sample {nt_640['per_sample_rate']:.3f}, greedy {nt_640['greedy']:.3f}")
    print(f"\n=== VERDICT: {v['VERDICT']} ===")

    # figure: dose curve (think covK + per-sample + no-think) with Wilson error bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    cov = [m["covK"] for m in pts]
    lo = [m["covK"] - m["wilson"][1] for m in pts]
    hi = [m["wilson"][2] - m["covK"] for m in pts]
    xplot = [1] + DOSES  # base plotted at x=1 for log axis
    ax1.errorbar(xplot, cov, yerr=[lo, hi], fmt="o-", color="#4a3aa7", lw=2, capsize=4, label="think coverage@16")
    ax1.plot(xplot, [m["per_sample_rate"] for m in pts], "s--", color="#1baf7a", lw=1.8, label="think per-sample solve rate")
    if nt_640 and nt_base:
        ax1.plot([1, 640], [nt_base["covK"], nt_640["covK"]], "^:", color="#eda100", lw=1.8, label="no-think coverage@16 (deployable)")
    ax1.set_xscale("log"); ax1.set_xticks(xplot); ax1.set_xticklabels(["base(0)", "40", "160", "640"])
    ax1.set_xlabel("# distinct depth-3 tool-pairs banked (fixed epochs)"); ax1.set_ylabel("depth-3 solve rate")
    ax1.set_ylim(-0.02, max(0.3, max(cov) + 0.08)); ax1.grid(alpha=0.25); ax1.legend(fontsize=8)
    ax1.set_title(f"Depth-3 dose-response — {v['VERDICT']}")
    for ax, m, N in [(ax2, dose[640], 640)]:
        ks = list(range(1, K + 1))
        ax.plot(ks, base["curve"], "o-", color="#94a3b8", lw=2, ms=3, label="base")
        for NN, col in zip(DOSES, ["#c4b5fd", "#8b5cf6", "#4a3aa7"]):
            ax.plot(ks, dose[NN]["curve"], "o-", color=col, lw=2, ms=3, label=f"banked N={NN}")
        ax.set_xlabel("k"); ax.set_ylabel("depth-3 coverage@k (think)"); ax.set_ylim(-0.02, max(0.3, max(cov) + 0.08))
        ax.grid(alpha=0.25); ax.legend(fontsize=8); ax.set_title("depth-3 coverage@k by dose")
    fig.suptitle("Is the depth-3 install data-limited or a representational cap?", y=1.02, fontsize=12)
    fig.tight_layout()
    (EXP / "analysis").mkdir(exist_ok=True)
    fig.savefig(EXP / "analysis" / "dose_response.png", dpi=130, bbox_inches="tight")
    print("wrote analysis/dose_response.png and runs/verdict.json")


if __name__ == "__main__":
    main()
