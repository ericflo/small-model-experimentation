#!/usr/bin/env python3
"""Analyze: does thinking breach the lookahead wall? Headline = step-1 think->rank top-1 vs budget (vs no-think
0.025 and chance 0.031). Steps 2/3 shown but flagged (state-materialization). Wilson CIs."""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
CHANCE = 1 / 32


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 3), round(min(1, c + h), 3))


def main():
    rows = json.loads((EXP / "runs" / "results.json").read_text())
    n = rows[0]["n"]
    budgets = sorted({r["budget"] for r in rows})
    steps = sorted({r["step"] for r in rows})
    tab = {(r["step"], r["budget"]): r for r in rows}

    print(f"\n=== next-op top-1 ranking accuracy (n={n}, chance {CHANCE:.3f}) ===")
    print("step\\B  " + "  ".join(f"B={b:>5}" for b in budgets))
    for s in steps:
        print(f"step {s}: " + "  ".join(f"{tab[(s,b)]['top1']:.3f} " if (s, b) in tab else "  --  " for b in budgets))

    s1 = {b: tab[(1, b)] for b in budgets if (1, b) in tab}
    base1 = s1.get(0, {}).get("top1", 0)
    top1_max = max(m["top1"] for m in s1.values())
    lift = round(top1_max - base1, 3)
    # significance of the best-budget step-1 vs no-think (non-overlapping Wilson)
    best_b = max(s1, key=lambda b: s1[b]["top1"])
    lo_best, hi_best = wilson(s1[best_b]["top1"], n)
    lo0, hi0 = wilson(base1, n)
    breach = bool(top1_max >= 0.10 and lo_best > hi0)
    # contamination: do steps 2/3 lift more than step 1 at the top budget?
    hi_bud = budgets[-1]
    d1 = tab[(1, hi_bud)]["top1"] - tab[(1, 0)]["top1"] if (1, hi_bud) in tab and (1, 0) in tab else None
    d3 = tab[(3, hi_bud)]["top1"] - tab[(3, 0)]["top1"] if (3, hi_bud) in tab and (3, 0) in tab else None

    v = {"step1_by_budget": {b: s1[b]["top1"] for b in s1},
         "step1_wilson": {b: wilson(s1[b]["top1"], n) for b in s1},
         "no_think_step1": base1, "best_budget": best_b, "step1_lift": lift,
         "breaches_wall(step1 >=0.10 & CI above no-think)": breach,
         "contamination_step1_lift_vs_step3_lift": {"step1": d1, "step3": d3},
         "verdict": ("THINKING BREACHES the lookahead wall (step-1 lift, significant)" if breach
                     else "thinking does NOT breach the wall (step-1 stays ~chance)")}
    (EXP / "runs" / "verdict.json").write_text(json.dumps(v, indent=1))
    print("\n=== VERDICT ===\n" + json.dumps(v, indent=1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    # panel 1: step-1 top1 vs budget with Wilson CIs
    xs = list(s1.keys())
    ys = [s1[b]["top1"] for b in xs]
    lo = [ys[i] - wilson(ys[i], n)[0] for i in range(len(xs))]
    hi = [wilson(ys[i], n)[1] - ys[i] for i in range(len(xs))]
    ax1.errorbar([str(b) for b in xs], ys, yerr=[lo, hi], fmt="o-", color="#4a3aa7", lw=2, capsize=5, ms=7)
    ax1.axhline(CHANCE, ls="--", color="red", lw=1, label=f"chance ({CHANCE:.3f})")
    ax1.set_xlabel("thinking budget (tokens)"); ax1.set_ylabel("STEP-1 next-op top-1 (think->rank)")
    ax1.set_title("Does thinking breach the lookahead wall? (step 1 = clean test)")
    ax1.set_ylim(0, max(0.2, max(ys) + 0.08)); ax1.legend(fontsize=8); ax1.grid(alpha=0.25)
    # panel 2: per-step top1 grouped by budget
    width = 0.8 / max(1, len(budgets))
    colors = ["#94a3b8", "#8b5cf6", "#4a3aa7", "#1e1b4b"]
    for i, b in enumerate(budgets):
        vals = [tab[(s, b)]["top1"] if (s, b) in tab else 0 for s in steps]
        ax2.bar([x + (i - len(budgets) / 2) * width for x in steps], vals, width, label=f"B={b}", color=colors[i % 4])
    ax2.axhline(CHANCE, ls="--", color="red", lw=1)
    ax2.set_xticks(steps); ax2.set_xticklabels([f"step {s}\n({4-s} away)" for s in steps])
    ax2.set_ylabel("next-op top-1"); ax2.set_title("Steps 2/3 get the true intermediate state (contaminated)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25, axis="y")
    fig.suptitle("Can thinking breach the lookahead wall? think->RANK, channel-matched to C25", y=1.02, fontsize=11)
    fig.tight_layout()
    (EXP / "analysis").mkdir(exist_ok=True)
    fig.savefig(EXP / "analysis" / "thinking_lookahead.png", dpi=130, bbox_inches="tight")
    print("wrote analysis/thinking_lookahead.png and runs/verdict.json")


if __name__ == "__main__":
    main()
