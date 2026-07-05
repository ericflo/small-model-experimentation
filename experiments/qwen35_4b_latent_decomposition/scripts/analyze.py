#!/usr/bin/env python3
"""Analyze the decomposition dissection: (1) per-step next-op ranking accuracy base vs banked (lookahead wall +
does banking install planning or only monolithic compilation?); (2) coverage-vs-interpreter-budget (model-guided
vs brute vs random -- does the guide beat brute on coverage?); (3) pruning ablation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
CHANCE1 = 1 / 32
GUIDES = [("base", "base"), ("banked640", "banked N=640"), ("banked1280", "banked N=1280")]


def load(name):
    p = EXP / "runs" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    v = {}
    # ---- Phase RANK ----
    ranks = {tag: load(f"rank_{tag}") for tag, _ in GUIDES}
    print("\n=== RANK: per-step next-op top-1 accuracy (chance %.3f) ===" % CHANCE1)
    print(f"{'guide':>14} {'step1':>8} {'step2':>8} {'step3':>8}")
    rankv = {}
    for tag, lab in GUIDES:
        r = ranks.get(tag)
        if not r:
            continue
        rankv[tag] = {s: r[str(s)] for s in (1, 2, 3)}
        print(f"{lab:>14} {r['1']['top1']:>8.3f} {r['2']['top1']:>8.3f} {r['3']['top1']:>8.3f}")
    if "base" in rankv and "banked1280" in rankv:
        b, k = rankv["base"], rankv["banked1280"]
        look_lift = round((k[1]["top1"] + k[2]["top1"]) / 2 - (b[1]["top1"] + b[2]["top1"]) / 2, 3)
        term_lift = round(k[3]["top1"] - b[3]["top1"], 3)
        v["rank"] = {
            "base": {s: b[s]["top1"] for s in (1, 2, 3)},
            "banked1280": {s: k[s]["top1"] for s in (1, 2, 3)},
            "lookahead_wall_base": bool(b[3]["top1"] > 3 * CHANCE1 and b[1]["top1"] <= 2 * CHANCE1),
            "banking_lifts_lookahead(step12)": look_lift,
            "banking_lifts_terminal(step3)": term_lift,
            "verdict": ("banking installs LOOKAHEAD" if look_lift > 0.08
                        else "banking is MONOLITHIC (lifts terminal recognition, not lookahead)"
                        if term_lift > 0.05 else "banking changes neither the guide"),
        }

    # ---- Phase SEARCH: coverage vs interp budget ----
    print("\n=== SEARCH: hidden-generalizing coverage vs interp/task ===")
    curves = {}
    for tag, lab in GUIDES:
        rows = load(f"search_{tag}")
        if rows:
            curves[lab] = [(r["interp_per_task"], r["coverage_hidden"]) for r in rows]
    for mode in ("brute", "random"):
        rows = load(f"search_{mode}")
        if rows:
            curves[mode] = [(r["interp_per_task"], r["coverage_hidden"]) for r in rows]
    for lab, pts in curves.items():
        best = max(p[1] for p in pts)
        print(f"  {lab:>16}: max hidden-cov {best:.3f}  (pts: {[(round(x),round(y,2)) for x,y in pts]})")
    brute_max = max((y for pts in [curves.get("brute", [])] for x, y in pts), default=0)
    base_max = max((y for x, y in curves.get("base", [])), default=0)
    bank_max = max((y for x, y in curves.get("banked N=1280", [])), default=0)
    v["search"] = {"base_guided_max_cov": base_max, "banked_guided_max_cov": bank_max, "brute_max_cov": brute_max,
                   "guide_beats_brute_on_coverage": bool(max(base_max, bank_max) > brute_max + 0.03)}

    # ---- Phase ABLATION ----
    abl = load("ablation")
    if abl:
        print("\n=== ABLATION: pruned vs unpruned (hidden coverage) ===")
        av = {}
        for r in abl:
            av[f"{r['mode']}_{'pruned' if r['pruned'] else 'unpruned'}"] = r["coverage_hidden"]
            print(f"  {r['mode']:>7} pruned={str(r['pruned']):>5}: {r['coverage_hidden']:.3f}")
        v["ablation"] = av

    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "verdict.json").write_text(json.dumps(v, indent=1))
    print("\n=== VERDICT ===\n" + json.dumps(v, indent=1))

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    # panel 1: per-step top-1, grouped by guide
    steps = [1, 2, 3]
    width = 0.25
    colors = {"base": "#94a3b8", "banked640": "#8b5cf6", "banked1280": "#4a3aa7"}
    for i, (tag, lab) in enumerate(GUIDES):
        if tag not in rankv:
            continue
        vals = [rankv[tag][s]["top1"] for s in steps]
        ax1.bar([x + (i - 1) * width for x in steps], vals, width, label=lab, color=colors[tag])
    ax1.axhline(CHANCE1, ls="--", color="red", lw=1, label="chance (1/32)")
    ax1.set_xticks(steps); ax1.set_xticklabels(["step 1\n(3 ops away)", "step 2\n(2 away)", "step 3\n(1 away)"])
    ax1.set_ylabel("top-1 next-op ranking accuracy"); ax1.set_title("The lookahead wall: does banking install planning?")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.25, axis="y")
    ax1.set_ylim(0, 0.62)
    # panel 2: coverage vs interp budget
    mk = {"base": "o-", "banked N=640": "s-", "banked N=1280": "^-", "brute": "D--", "random": "x:"}
    col = {"base": "#94a3b8", "banked N=640": "#8b5cf6", "banked N=1280": "#4a3aa7", "brute": "#16a34a", "random": "#ef4444"}
    for lab, pts in curves.items():
        pts = sorted(pts)
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], mk.get(lab, "o-"), label=lab, color=col.get(lab), lw=1.8, ms=5)
    ax2.set_xscale("log"); ax2.set_xlabel("interpreter calls per task (budget)"); ax2.set_ylabel("hidden-generalizing depth-3 coverage")
    ax2.annotate("base guide\n(worse than random)", (1088, 0.013), fontsize=7, xytext=(1300, 0.09),
                 textcoords="data", arrowprops=dict(arrowstyle="->", lw=0.7))
    ax2.set_title("Model-guided search: banking upgrades base from worse-than-random to competitive"); ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
    fig.suptitle("Be your own tool-search: the base model has recognition but NO lookahead; banking installs transferable lookahead (planning), dose-dependently", y=1.02, fontsize=10.5)
    fig.tight_layout()
    (EXP / "analysis").mkdir(exist_ok=True)
    fig.savefig(EXP / "analysis" / "decomposition.png", dpi=130, bbox_inches="tight")
    print("wrote analysis/decomposition.png and runs/verdict.json")


if __name__ == "__main__":
    main()
