#!/usr/bin/env python3
"""Compare base vs banked (no-think AND think) on held-out tasks. The sample-more baseline is base-think
coverage@K. Questions: (1) does banking beat sample-more (banked greedy@1 vs base-think coverage@K)?
(2) CONCENTRATION (greedy rises toward the base ceiling) vs EXPANSION (banked coverage exceeds base-think
coverage — new programs proposed)?"""
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


def stats(tag):
    path = EXP / "runs" / f"eval_{tag}.json"
    if not path.exists():
        return None
    recs = json.load(open(path))["records"]
    b = defaultdict(list)
    for r in recs:
        b[r["depth"]].append(r)
    out = {}
    for d, rs in b.items():
        n, K = len(rs), rs[0]["K"]
        out[d] = {"greedy": sum(r["greedy_full"] for r in rs) / n,
                  "covK": sum(cov_at_k(r["cov_full"], r["K"], K) for r in rs) / n,
                  "uniq": sum(r["n_unique"] for r in rs) / n, "K": K,
                  "curve": [sum(cov_at_k(r["cov_full"], r["K"], k) for r in rs) / n for k in range(1, K + 1)]}
    return out


def main():
    arms = {t: stats(t) for t in ("base", "banked", "base_think", "banked_think")}
    arms = {t: v for t, v in arms.items() if v}
    depths = sorted(next(iter(arms.values())))
    K = arms[next(iter(arms))][depths[0]]["K"]

    def g(tag, d, key):
        return arms.get(tag, {}).get(d, {}).get(key)

    print("\n=== Banking on held-out list tasks (greedy@1 / coverage@%d) ===" % K)
    print(f"{'d':>2} | {'base_th g@1':>11} {'SAMPLE-MORE':>11} | {'bank_nt g@1':>11} {'bank_th g@1':>11} "
          f"{'bank_th cov':>11} | {'verdict':>22}")
    print(f"{'':>2} | {'(1shot base)':>11} {'base_th cov':>11} | {'cheap deploy':>11} {'strong deploy':>11} "
          f"{'bank ceiling':>11} |")
    verdict = {"K": K, "per_depth": {}}
    for d in depths:
        smore = g("base_think", d, "covK")           # sample-more ceiling
        base1 = g("base_think", d, "greedy")          # honest single-shot baseline
        bnt = g("banked", d, "greedy")                # cheap banked deploy
        bth = g("banked_think", d, "greedy")          # strong banked deploy
        bth_cov = g("banked_think", d, "covK")
        best_bank_greedy = max([x for x in (bnt, bth) if x is not None], default=None)
        # concentration vs expansion (think mode, apples-to-apples)
        d_greedy = (bth - base1) if (bth is not None and base1 is not None) else None
        d_cov = (bth_cov - smore) if (bth_cov is not None and smore is not None) else None
        verd = "?"
        if d_cov is not None and d_greedy is not None:
            if d_cov >= 0.10:
                verd = "EXPANSION"
            elif d_greedy > d_cov + 0.05 and d_greedy > 0.05:
                verd = "concentration"
            elif abs(d_greedy) <= 0.05 and abs(d_cov) <= 0.05:
                verd = "no-move"
            else:
                verd = "mixed"
        beats = (best_bank_greedy is not None and smore is not None and best_bank_greedy >= smore - 1e-9)
        verdict["per_depth"][d] = {
            "base_think_greedy": base1, "sample_more_covK": smore,
            "banked_nt_greedy": bnt, "banked_think_greedy": bth, "banked_think_covK": bth_cov,
            "d_greedy_think": None if d_greedy is None else round(d_greedy, 3),
            "d_cov_vs_samplemore": None if d_cov is None else round(d_cov, 3),
            "banked_beats_sample_more": bool(beats), "verdict": verd,
            "uniq_base_think": g("base_think", d, "uniq"), "uniq_banked_think": g("banked_think", d, "uniq")}
        fmt = lambda x: f"{x:.2f}" if x is not None else "  - "
        print(f"{d:>2} | {fmt(base1):>11} {fmt(smore):>11} | {fmt(bnt):>11} {fmt(bth):>11} "
              f"{fmt(bth_cov):>11} | {verd:>22}")

    (EXP / "runs" / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print("\n=== verdict.json ===")
    print(json.dumps(verdict["per_depth"], indent=1))

    # figure: think-mode coverage overlays (base vs banked) + greedy/ceiling bars
    have_think = "base_think" in arms and "banked_think" in arms
    ncol = len(depths) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0))
    ks = list(range(1, K + 1))
    for ax, d in zip(axes[:len(depths)], depths):
        if have_think:
            ax.plot(ks, arms["base_think"][d]["curve"], "o-", color="#94a3b8", lw=2, ms=3, label="base (think)")
            ax.plot(ks, arms["banked_think"][d]["curve"], "o-", color="#16a34a", lw=2, ms=3, label="banked (think)")
            ax.scatter([1], [arms["base_think"][d]["greedy"]], color="#475569", s=45, zorder=5, marker="D")
            ax.scatter([1], [arms["banked_think"][d]["greedy"]], color="#065f46", s=45, zorder=5, marker="D")
        if "banked" in arms:
            ax.plot(ks, arms["banked"][d]["curve"], "--", color="#f59e0b", lw=1.5, label="banked (no-think)")
        ax.set_title(f"depth {d}" + ("" if d <= 3 else " (untrained)"))
        ax.set_xlabel("k"); ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.25)
        if d == depths[0]:
            ax.set_ylabel("coverage@k"); ax.legend(fontsize=7)
    axb = axes[-1]
    x = range(len(depths)); w = 0.2
    if have_think:
        axb.bar([i - 1.5 * w for i in x], [arms["base_think"][d]["greedy"] for d in depths], w,
                label="base think greedy@1", color="#94a3b8")
        axb.bar([i - 0.5 * w for i in x], [arms["base_think"][d]["covK"] for d in depths], w,
                label=f"sample-more (base think cov@{K})", color="#fbbf24")
        axb.bar([i + 0.5 * w for i in x], [arms["banked_think"][d]["greedy"] for d in depths], w,
                label="banked think greedy@1", color="#16a34a")
        axb.bar([i + 1.5 * w for i in x], [arms["banked_think"][d]["covK"] for d in depths], w,
                label=f"banked think cov@{K}", color="#86efac")
    axb.set_xticks(list(x)); axb.set_xticklabels([f"d{d}" for d in depths])
    axb.set_title("single-shot vs sample-more vs banked"); axb.set_ylim(0, 1.05)
    axb.legend(fontsize=7); axb.grid(alpha=0.25, axis="y")
    fig.suptitle("Does banking beat sample-more? concentration vs expansion (diamond = greedy@1)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(EXP / "analysis" / "banking_coverage.png", dpi=130, bbox_inches="tight")
    print("\nwrote analysis/banking_coverage.png and runs/verdict.json")


if __name__ == "__main__":
    main()
