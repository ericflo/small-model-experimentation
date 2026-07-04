#!/usr/bin/env python3
"""Tool-seeded banking analysis: did seeding banking with interpreter-found depth-3 solutions cross the
depth-3 wall (vs C21's self-banking, which stayed 0.00)? Think (primary, matched to C21) + no-think
(deployable) arms, on the frozen paired held-out set. Verdict on P2/P4 + chart."""
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
    return 0.0 if c == 0 else (1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k))


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
        out[d] = {"n": n, "K": K,
                  "greedy": sum(r["greedy_full"] for r in rs) / n,
                  "covK": sum(cov_at_k(r["cov_full"], r["K"], K) for r in rs) / n,
                  "n_unlocked": sum(1 for r in rs if r["cov_full"] >= 1),
                  "curve": [sum(cov_at_k(r["cov_full"], r["K"], k) for r in rs) / n for k in range(1, K + 1)]}
    return out


def main():
    A = {t: load(t) for t in ("base_think", "banked_think", "base_nt", "banked_nt")}
    A = {t: v for t, v in A.items() if v}
    depths = sorted(A["base_think"])
    K = A["base_think"][depths[0]]["K"]

    def line(mode):
        bt, bk = A.get(f"base_{mode}"), A.get(f"banked_{mode}")
        print(f"\n=== {mode.upper()} : coverage@{K} / greedy@1 (frozen paired held-out) ===")
        print(f"{'depth':>5} {'base_cov':>9} {'bank_cov':>9} {'Δcov':>7} {'base_g@1':>9} {'bank_g@1':>9} {'#unlocked(bank)':>16}")
        rows = {}
        for d in depths:
            dc = bk[d]["covK"] - bt[d]["covK"]
            rows[d] = {"base_cov": round(bt[d]["covK"], 3), "bank_cov": round(bk[d]["covK"], 3), "d_cov": round(dc, 3),
                       "base_greedy": round(bt[d]["greedy"], 3), "bank_greedy": round(bk[d]["greedy"], 3),
                       "n_unlocked_bank": bk[d]["n_unlocked"], "n_base": bt[d]["n"]}
            print(f"{d:>5} {bt[d]['covK']:>9.2f} {bk[d]['covK']:>9.2f} {dc:>+7.2f} "
                  f"{bt[d]['greedy']:>9.2f} {bk[d]['greedy']:>9.2f} {bk[d]['n_unlocked']:>10}/{bk[d]['n']}")
        return rows

    think = line("think")
    nt = line("nt") if "base_nt" in A else {}

    v = {"think": think, "nothink": nt, "K": K}
    if 3 in think:
        t3 = think[3]
        # base depth-3 is 0/n; its 95% upper CI ~ 3/n (rule of three). A significant unlock clears that.
        base_upper_ci = round(3.0 / max(1, t3["n_base"]), 3)
        v["P2_unlock_depth3_think"] = {
            "base_cov": t3["base_cov"], "banked_cov": t3["bank_cov"], "delta": t3["d_cov"],
            "n_distinct_unlocked": t3["n_unlocked_bank"], "n": t3["n_base"],
            "base_upper_95ci": base_upper_ci,
            "significant_unlock": bool(t3["bank_cov"] > base_upper_ci and t3["n_unlocked_bank"] >= 3),
            "clears_strong_bar_0.15": bool(t3["bank_cov"] >= 0.15 and t3["d_cov"] >= 0.10 and t3["n_unlocked_bank"] >= 5)}
    if nt and 3 in nt:
        v["P4_deployable_depth3_nothink"] = {"base_greedy": nt[3]["base_greedy"], "banked_greedy": nt[3]["bank_greedy"],
                                             "base_cov": nt[3]["base_cov"], "banked_cov": nt[3]["bank_cov"],
                                             "banked_into_weights": bool(nt[3]["bank_greedy"] > nt[3]["base_greedy"] + 0.03
                                                                         or nt[3]["bank_cov"] > nt[3]["base_cov"] + 0.05)}
    if 4 in think:
        v["P4_next_rung_depth4_think"] = {"base_cov": think[4]["base_cov"], "banked_cov": think[4]["bank_cov"],
                                          "delta": think[4]["d_cov"]}
    p2 = v.get("P2_unlock_depth3_think", {})
    deployable = bool(v.get("P4_deployable_depth3_nothink", {}).get("banked_into_weights"))
    if p2.get("clears_strong_bar_0.15"):
        v["VERDICT"] = "CROSSED (tools explore + banking installs)"
    elif p2.get("significant_unlock"):
        v["VERDICT"] = ("CROSSED-BUT-WEAK (significant depth-3 unlock vs base 0, but modest and "
                        + ("deployable" if deployable else "test-time-dominated") + ")")
    else:
        v["VERDICT"] = "NOT-CROSSED"
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "verdict.json").write_text(json.dumps(v, indent=1))
    print("\n=== VERDICT:", v["VERDICT"], "===")
    print(json.dumps({k: v[k] for k in v if k.startswith("P")}, indent=1))

    # figure: think coverage@k curves per depth (base vs banked) + a bar of depth-3 cov (C21 vs this)
    bt, bk = A["base_think"], A["banked_think"]
    fig, axes = plt.subplots(1, len(depths) + 1, figsize=(4.2 * (len(depths) + 1), 4.0))
    ks = list(range(1, K + 1))
    for ax, d in zip(axes[:len(depths)], depths):
        ax.plot(ks, bt[d]["curve"], "o-", color="#94a3b8", lw=2, ms=3, label="base")
        ax.plot(ks, bk[d]["curve"], "o-", color="#4a3aa7", lw=2, ms=3, label="banked_tool")
        ax.set_title(f"depth {d}" + (" (WALL — the test)" if d == 3 else ""))
        ax.set_xlabel("k"); ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.25)
        if d == depths[0]:
            ax.set_ylabel("coverage@k (think)"); ax.legend(fontsize=8)
    axb = axes[-1]
    labels = ["C21 self-bank\n(depth1+2)", "tool-seeded\n(this)"]
    d3base = think[3]["base_cov"] if 3 in think else 0
    d3bank = think[3]["bank_cov"] if 3 in think else 0
    axb.bar([0, 1], [0.0, d3bank], color=["#94a3b8", "#4a3aa7"])
    axb.bar([0], [d3base], color="#cbd5e1")
    axb.set_xticks([0, 1]); axb.set_xticklabels(labels, fontsize=9)
    axb.set_title("depth-3 coverage@16: self-banking vs tool-seeded"); axb.set_ylim(0, max(0.3, d3bank + 0.05))
    axb.grid(alpha=0.25, axis="y")
    fig.suptitle(f"Tool-seeded banking across the depth-3 wall — {v['VERDICT']}", y=1.02, fontsize=12)
    fig.tight_layout()
    (EXP / "analysis").mkdir(exist_ok=True)
    fig.savefig(EXP / "analysis" / "tool_seeded_banking.png", dpi=130, bbox_inches="tight")
    print("\nwrote analysis/tool_seeded_banking.png and runs/verdict.json")


if __name__ == "__main__":
    main()
