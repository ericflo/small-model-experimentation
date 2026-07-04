#!/usr/bin/env python3
"""Wall-climbing analysis: did banking depth-1+2 (banked1) unlock deeper coverage vs base? Emits per-depth
coverage@K + greedy, the unlock verdict (P1-P3), and a chart. If banked2 exists, adds the Round-2 rung."""
from __future__ import annotations

import json
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]


def cov_at_k(c, n, k):
    k = min(k, n)
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load(tag):
    p = EXP / "runs" / f"eval_{tag}.json"
    if not p.exists():
        return None
    recs = json.load(open(p))["records"]
    b = defaultdict(list)
    for r in recs:
        b[r["depth"]].append(r)
    out = {}
    for d, rs in b.items():
        n, K = len(rs), rs[0]["K"]
        out[d] = {"greedy": sum(r["greedy_full"] for r in rs) / n,
                  "covK": sum(cov_at_k(r["cov_full"], r["K"], K) for r in rs) / n,
                  "uniq": sum(r["n_unique"] for r in rs) / n, "n": n, "K": K,
                  "curve": [sum(cov_at_k(r["cov_full"], r["K"], k) for r in rs) / n for k in range(1, K + 1)]}
    return out


def main():
    arms = {t: load(t) for t in ("base", "banked1", "banked2")}
    arms = {t: v for t, v in arms.items() if v}
    base, b1 = arms.get("base"), arms.get("banked1")
    depths = sorted(base)
    K = base[depths[0]]["K"]

    print(f"\n=== Wall climbing: coverage@{K} (think, held-out) ===")
    print(f"{'depth':>5} {'base_cov':>9} {'banked1_cov':>12} {'Δ_cov':>7} {'base_g@1':>9} {'bank1_g@1':>10}")
    per = {}
    for d in depths:
        dc = b1[d]["covK"] - base[d]["covK"]
        per[d] = {"base_cov": round(base[d]["covK"], 3), "banked1_cov": round(b1[d]["covK"], 3),
                  "d_cov": round(dc, 3), "base_greedy": round(base[d]["greedy"], 3),
                  "banked1_greedy": round(b1[d]["greedy"], 3),
                  "uniq_base": round(base[d]["uniq"], 1), "uniq_banked1": round(b1[d]["uniq"], 1)}
        if "banked2" in arms and d in arms["banked2"]:
            per[d]["banked2_cov"] = round(arms["banked2"][d]["covK"], 3)
        print(f"{d:>5} {base[d]['covK']:>9.2f} {b1[d]['covK']:>12.2f} {dc:>+7.2f} "
              f"{base[d]['greedy']:>9.2f} {b1[d]['greedy']:>10.2f}")

    v = {"per_depth": per}
    v["P1_install"] = bool(3 in per and per[3] and 2 in per and (per[2]["banked1_cov"] - per[2]["base_cov"]) >= 0.10)
    if 3 in per:
        v["P2_unlock_depth3"] = {"base_cov": per[3]["base_cov"], "banked1_cov": per[3]["banked1_cov"],
                                 "delta": per[3]["d_cov"],
                                 "unlocked": bool(per[3]["d_cov"] >= 0.05 and per[3]["base_cov"] <= 0.10),
                                 "depth_local": bool(per[3]["d_cov"] <= 0.03)}
    if 4 in per:
        v["P3_no_two_rung_leap"] = {"banked1_depth4_cov": per[4]["banked1_cov"],
                                    "stays_near_zero": bool(per[4]["banked1_cov"] <= 0.03)}
    unlocked = bool(v.get("P2_unlock_depth3", {}).get("unlocked"))
    v["VERDICT"] = "CLIMBABLE" if unlocked else "DEPTH-LOCAL"
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "verdict.json").write_text(json.dumps(v, indent=1))
    print("\n=== VERDICT:", v["VERDICT"], "===")
    print(json.dumps(v, indent=1))

    # figure: coverage@k curves per depth (base vs banked1[, banked2]) + a bar of cov@K by depth
    ncol = len(depths) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0))
    ks = list(range(1, K + 1))
    for ax, d in zip(axes[:len(depths)], depths):
        ax.plot(ks, base[d]["curve"], "o-", color="#94a3b8", lw=2, ms=3, label="base")
        ax.plot(ks, b1[d]["curve"], "o-", color="#16a34a", lw=2, ms=3, label="banked1 (depth 1+2)")
        if "banked2" in arms and d in arms["banked2"]:
            ax.plot(ks, arms["banked2"][d]["curve"], "o-", color="#4a3aa7", lw=2, ms=3, label="banked2 (+depth 3)")
        ax.set_title(f"depth {d}" + (" (UNLOCK test)" if d == 3 else ""))
        ax.set_xlabel("k"); ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.25)
        if d == depths[0]:
            ax.set_ylabel("coverage@k"); ax.legend(fontsize=8)
    axb = axes[-1]
    x = range(len(depths)); w = 0.35
    axb.bar([i - w / 2 for i in x], [base[d]["covK"] for d in depths], w, label="base", color="#94a3b8")
    axb.bar([i + w / 2 for i in x], [b1[d]["covK"] for d in depths], w, label="banked1", color="#16a34a")
    axb.set_xticks(list(x)); axb.set_xticklabels([f"d{d}" for d in depths])
    axb.set_title(f"coverage@{K} by depth"); axb.set_ylim(0, 1.05); axb.legend(fontsize=8); axb.grid(alpha=0.25, axis="y")
    fig.suptitle(f"Does banking depth-1+2 unlock deeper coverage? — {v['VERDICT']}", y=1.02, fontsize=12)
    fig.tight_layout()
    (EXP / "analysis").mkdir(exist_ok=True)
    fig.savefig(EXP / "analysis" / "wall_climbing.png", dpi=130, bbox_inches="tight")
    print("\nwrote analysis/wall_climbing.png and runs/verdict.json")


if __name__ == "__main__":
    main()
