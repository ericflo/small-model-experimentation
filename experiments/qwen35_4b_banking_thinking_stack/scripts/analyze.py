#!/usr/bin/env python3
"""2x2 {base, banked} x {no-think, think} on per-step next-op ranking. base cells from C26 (same n=40 slice,
same think_rank harness, same held-out); banked cells from this run. Question: do banking (installs planning,
C25) and thinking (amplifies recognition, C26) STACK? Answer per axis: no stacking on planning (step-1);
additive stacking on recognition (step-3). HONEST SCOPE: banking is no-think SFT, so 'thinking adds no planning'
is about test-time thinking on a no-think-trained model -- motivating the bank-the-thoughts follow-up."""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
CHANCE = 1 / 32
N = 40
# base cells re-used from C26 (identical n=40 first-40 slice + think_rank harness + held-out)
BASE = {1: {0: 0.025, 1024: 0.050, 2048: 0.075},
        2: {0: 0.000, 1024: 0.125, 2048: 0.325},
        3: {0: 0.275, 1024: 0.600, 2048: 0.600}}


def wilson(p, n=N, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 3), round(min(1, c + h), 3))


def main():
    rows = json.loads((EXP / "runs" / "results_banked1280.json").read_text())
    BANK = {s: {} for s in (1, 2, 3)}
    for r in rows:
        BANK[r["step"]][r["budget"]] = r["top1"]

    print("\n=== 2x2 per-step next-op top-1 (n=40, chance %.3f) ===" % CHANCE)
    print(f"{'step':>6} {'base+NT':>8} {'base+T2048':>11} {'bank+NT':>8} {'bank+T2048':>11}")
    for s in (1, 2, 3):
        print(f"{s:>6} {BASE[s][0]:>8.3f} {BASE[s][2048]:>11.3f} {BANK[s][0]:>8.3f} {BANK[s][2048]:>11.3f}")

    def effects(s, B=2048):
        bank_eff = BANK[s][0] - BASE[s][0]                 # banking at no-think
        think_eff = BASE[s][B] - BASE[s][0]                # thinking on base
        additive_pred = BASE[s][0] + bank_eff + think_eff  # if orthogonal/additive
        observed = BANK[s][B]                              # banked + think
        return {"bank_effect": round(bank_eff, 3), "think_effect_on_base": round(think_eff, 3),
                "additive_prediction": round(additive_pred, 3), "observed_banked_think": round(observed, 3),
                "interaction(obs-pred)": round(observed - additive_pred, 3)}

    v = {"step1_planning": effects(1), "step3_recognition": effects(3),
         "banked_think_step1_vs_banked_nothink": {"nothink": BANK[1][0], "think2048": BANK[1][2048],
                                                  "think_adds_planning": bool(BANK[1][2048] - BANK[1][0] > 0.05)},
         "verdict": ("Levers STACK additively on RECOGNITION (step-3) but NOT on PLANNING (step-1): banking owns "
                     "the planning lift, test-time thinking adds ~0 to planning on base OR banked. SCOPE: banking "
                     "is no-think SFT -> 'thinking adds no planning' is test-time-only; motivates bank-the-thoughts."),
         "caveat": "n=40, one seed/budget; base cells inherited from C26 (identical slice/harness). Banked adapter "
                   "was trained NO-THINK; its thinking channel is intact (traces coherent, median ~2500 chars)."}
    (EXP / "runs" / "verdict.json").write_text(json.dumps(v, indent=1))
    print("\n=== VERDICT ===\n" + json.dumps(v, indent=1))

    # figure: step-1 (planning) and step-3 (recognition) 2x2 bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    cells = [("base\n+no-think", BASE, 0, "#cbd5e1"), ("base\n+think", BASE, 2048, "#94a3b8"),
             ("banked\n+no-think", BANK, 0, "#a78bfa"), ("banked\n+think", BANK, 2048, "#4a3aa7")]
    for ax, s, title in ((ax1, 1, "STEP 1 = PLANNING (goal 3 ops away)"), (ax2, 3, "STEP 3 = RECOGNITION (goal 1 op away)")):
        vals = [d[s][b] for (_, d, b, _) in cells]
        cols = [c for (_, _, _, c) in cells]
        bars = ax.bar(range(4), vals, color=cols)
        ax.errorbar(range(4), vals, yerr=[[v - wilson(v)[0] for v in vals], [wilson(v)[1] - v for v in vals]],
                    fmt="none", ecolor="black", capsize=4)
        ax.axhline(CHANCE, ls="--", color="red", lw=1)
        ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in cells], fontsize=8)
        ax.set_ylim(0, 1.0 if s == 3 else 0.32); ax.set_ylabel("next-op top-1"); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)
    ax1.text(0.5, 0.9, "banking lifts;\nthinking adds ~0\n=> NO stacking", transform=ax1.transAxes, fontsize=8,
             ha="center", va="top", bbox=dict(boxstyle="round", fc="#fef3c7", ec="none"))
    ax2.text(0.5, 0.35, "both lift, ~additive\n0.28+0.25+0.33≈0.85\n=> STACK", transform=ax2.transAxes, fontsize=8,
             ha="center", va="top", bbox=dict(boxstyle="round", fc="#dcfce7", ec="none"))
    fig.suptitle("Do banking + thinking stack? Additively on RECOGNITION, not at all on PLANNING "
                 "(banked model was trained no-think)", y=1.02, fontsize=10.5)
    fig.tight_layout()
    (EXP / "analysis").mkdir(exist_ok=True)
    fig.savefig(EXP / "analysis" / "stack.png", dpi=130, bbox_inches="tight")
    print("wrote analysis/stack.png and runs/verdict.json")


if __name__ == "__main__":
    main()
