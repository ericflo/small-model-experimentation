#!/usr/bin/env python3
"""Solve-rate vs interpreter-call-budget: model-guided decompose vs brute-force, with the monolithic
(sampling) frontier baseline, by depth."""
from __future__ import annotations

import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def main():
    s = json.loads((EXP / "runs" / "search_summary.json").read_text())
    depths = s["depths"]
    budgets = s["budgets"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(depths), figsize=(5.5 * len(depths), 4.6), squeeze=False)
    for ax, depth in zip(axes[0], depths):
        d = s[f"depth{depth}"]
        gc = [d["guided_curve"][str(b)] if str(b) in d["guided_curve"] else d["guided_curve"][b] for b in budgets]
        bc = [d["brute_curve"][str(b)] if str(b) in d["brute_curve"] else d["brute_curve"][b] for b in budgets]
        ax.plot(budgets, gc, "o-", color="#e76f51", lw=2, label=f"guided (model-ranked, ~{d['guided_mean_calls']:.0f} calls)")
        ax.plot(budgets, bc, "s-", color="#264653", label=f"brute-force (~{d['brute_mean_calls']:.0f} calls)")
        ax.axhline(d["monolithic_hidden"], color="#2a9d8f", ls="--", lw=1.5, label=f"monolithic sampling {d['monolithic_hidden']:.2f}")
        ax.set_xscale("log")
        ax.set_xlabel("interpreter-call budget (log)")
        ax.set_ylabel("hidden-generalizing solve rate")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"depth {depth} (n={d['n']})")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    fig.suptitle("Decompose-and-compose: does model guidance beat brute-force + crack the frontier?")
    fig.tight_layout()
    fig.savefig(EXP / "analysis" / "search_curve.png", dpi=130)
    print("wrote analysis/search_curve.png")


if __name__ == "__main__":
    main()
