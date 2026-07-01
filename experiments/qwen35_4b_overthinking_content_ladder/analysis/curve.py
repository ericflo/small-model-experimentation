#!/usr/bin/env python3
"""Per-budget ladder + the coherence-advantage (real - shuffle) curve vs thinking budget."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def main():
    labels = [json.loads(l) for l in (EXP / "data" / "labels.jsonl").read_text().splitlines() if l.strip()]
    agg = defaultdict(lambda: [0, 0])  # (budget, cond) -> [n, pass]
    for r in labels:
        a = agg[(r["budget"], r["cond"])]; a[0] += 1; a[1] += int(r["full_pass"])
    rate = {k: v[1] / v[0] for k, v in agg.items()}
    budgets = sorted({b for (b, c) in rate if c != "no_think"})
    no_think = rate.get((0, "no_think"), float("nan"))

    # table
    conds = ["filler", "shuffle", "foreign", "real"]
    rows = ["| budget | no_think | filler | shuffle | foreign | real | coherence (real-shuffle) | content-used (real-foreign) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    curve = {}
    for b in budgets:
        r = {c: rate.get((b, c), float("nan")) for c in conds}
        coh = r["real"] - r["shuffle"]
        curve[b] = coh
        rows.append(f"| {b} | {no_think:.3f} | {r['filler']:.3f} | {r['shuffle']:.3f} | {r['foreign']:.3f} | "
                    f"{r['real']:.3f} | {coh:+.3f} | {r['real']-r['foreign']:+.3f} |")
    table = "\n".join(rows)
    (EXP / "analysis").mkdir(exist_ok=True)
    (EXP / "analysis" / "curve.md").write_text(table + "\n")
    print(table)

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # (1) coherence advantage vs budget
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(budgets, [curve[b] for b in budgets], "o-", color="#2a9d8f", lw=2)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    for b in budgets:
        ax.annotate(f"{curve[b]:+.3f}", (b, curve[b]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("thinking budget (tokens)")
    ax.set_ylabel("coherence advantage: real − shuffle (full-pass)")
    ax.set_title("Does coherent order stop mattering as the budget grows? (overthinking)")
    ax.set_xticks(budgets)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(EXP / "analysis" / "coherence_vs_budget.png", dpi=130)

    # (2) full ladder per budget
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
    order = ["no_think", "filler", "shuffle", "foreign", "real"]
    cols = {"no_think": "#264653", "filler": "#a8dadc", "shuffle": "#f4a261", "foreign": "#e63946", "real": "#2a9d8f"}
    for c in order:
        ys = [no_think if c == "no_think" else rate.get((b, c), float("nan")) for b in budgets]
        ax2.plot(budgets, ys, "o-", color=cols[c], label=c, lw=1.8)
    ax2.set_xlabel("thinking budget (tokens)"); ax2.set_ylabel("MBPP full-pass")
    ax2.set_xticks(budgets); ax2.set_ylim(0, 1)
    ax2.set_title("Content ladder vs budget"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(EXP / "analysis" / "ladder_vs_budget.png", dpi=130)
    print("\nwrote analysis/curve.md, coherence_vs_budget.png, ladder_vs_budget.png")


if __name__ == "__main__":
    main()
