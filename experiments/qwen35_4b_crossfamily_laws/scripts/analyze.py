#!/usr/bin/env python3
"""Cross-family analysis: are the C13-C15 constants family-invariant? Emits a table + figure and a
verdict against the pre-registered predictions P-L1..P-L4."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
FAMS = ["list", "string", "register"]
RUNGS = ["transcription", "sim", "bare"]
COL = {"list": "#2563eb", "string": "#16a34a", "register": "#dc2626"}


def load(fam):
    return json.load(open(EXP / "runs" / f"ladder_{fam}.json"))["records"]


def by_depth(recs):
    b = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for x in recs:
        for rg in RUNGS:
            c = b[x["depth"]][rg]; c[0] += 1; c[1] += int(x[rg])
    return {d: {rg: b[d][rg][1] / b[d][rg][0] for rg in RUNGS} for d in sorted(b)}


def main():
    data = {f: by_depth(load(f)) for f in FAMS}
    depths = sorted(data["list"])

    # ---- table -----------------------------------------------------------------------------------
    print("\n=== Cross-family capability ladder ===")
    for rg in RUNGS:
        print(f"\n{rg}:")
        print("  depth  " + "  ".join(f"{f:>9}" for f in FAMS))
        for d in depths:
            print(f"  {d:>5}  " + "  ".join(f"{data[f][d][rg]:>9.2f}" for f in FAMS))

    # ---- normalized simulation decay (per-op retention vs each family's own peak) -----------------
    print("\n=== simulation decay, normalized to each family's peak (single-step retention) ===")
    norm = {}
    for f in FAMS:
        peak = max(data[f][d]["sim"] for d in depths) or 1e-9
        norm[f] = {d: data[f][d]["sim"] / peak for d in depths}
        print(f"  {f:>9}: " + "  ".join(f"d{d}={norm[f][d]:.2f}" for d in depths))
    spread = {d: max(norm[f][d] for f in FAMS) - min(norm[f][d] for f in FAMS) for d in depths}
    print("  cross-family spread per depth: " + "  ".join(f"d{d}={spread[d]:.2f}" for d in depths))

    # ---- verdict on locked predictions ----------------------------------------------------------
    print("\n=== Pre-registered verdicts ===")
    v = {}
    # P-L1 transcription law: >=0.85 at every depth, both new families; refuted if <0.6 at depth<=3
    p1 = {}
    for f in ("string", "register"):
        lo = min(data[f][d]["transcription"] for d in depths)
        lo3 = min(data[f][d]["transcription"] for d in depths if d <= 3)
        p1[f] = {"min_all": round(lo, 2), "min_d<=3": round(lo3, 2),
                 "hold": lo >= 0.85, "refuted": lo3 < 0.6}
    v["P-L1_transcription"] = p1
    # P-L2 decay: monotone-ish + d4<=0.5 both; strong: normalized spread <=0.15 all depths
    v["P-L2_decay"] = {
        "d4_sim": {f: round(data[f][4]["sim"], 2) for f in ("string", "register")},
        "d4<=0.5_both": all(data[f][4]["sim"] <= 0.5 for f in ("string", "register")),
        "max_norm_spread": round(max(spread.values()), 2),
        "strong_invariant(<=0.15)": max(spread.values()) <= 0.15,
    }
    # P-L3 generation wall: bare<0.15 at depth>=3 both; trans-bare gap >=0.6 both
    p3 = {}
    for f in ("string", "register"):
        bare_hi = max(data[f][d]["bare"] for d in depths if d >= 3)
        gap = min(data[f][d]["transcription"] - data[f][d]["bare"] for d in depths if d >= 3)
        p3[f] = {"max_bare_d>=3": round(bare_hi, 2), "min_gap_d>=3": round(gap, 2),
                 "wall": bare_hi < 0.15, "gap>=0.6": gap >= 0.6}
    v["P-L3_generation_wall"] = p3
    # P-L4 ordering at depth 3: trans > sim > bare each family
    p4 = {}
    for f in FAMS:
        d3 = data[f][3]
        p4[f] = {"trans": round(d3["transcription"], 2), "sim": round(d3["sim"], 2),
                 "bare": round(d3["bare"], 2),
                 "order_holds": d3["transcription"] >= d3["sim"] >= d3["bare"]}
    v["P-L4_ordering_d3"] = p4

    laws = (all(p1[f]["hold"] and not p1[f]["refuted"] for f in ("string", "register"))
            and all(p3[f]["wall"] and p3[f]["gap>=0.6"] for f in ("string", "register"))
            and all(p4[f]["order_holds"] for f in FAMS))
    v["VERDICT"] = "LAWS" if laws else "SCOPED"
    v["strong_invariant_decay"] = v["P-L2_decay"]["strong_invariant(<=0.15)"]
    print(json.dumps(v, indent=1))
    (EXP / "runs" / "verdict.json").write_text(json.dumps(v, indent=1))

    # ---- figure ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    titles = {"transcription": "Transcription (plan→code)", "sim": "Simulation (mental exec)",
              "bare": "Identification (I/O→code)"}
    for ax, rg in zip(axes, RUNGS):
        for f in FAMS:
            ax.plot(depths, [data[f][d][rg] for d in depths], "o-", color=COL[f], label=f, lw=2, ms=6)
        ax.set_title(titles[rg], fontsize=11)
        ax.set_xlabel("composition depth"); ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(depths); ax.grid(alpha=0.25)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(title="family", fontsize=9)
    fig.suptitle(f"Cross-family capability ladder — verdict: {v['VERDICT']}  "
                 f"(strong decay-invariant: {v['strong_invariant_decay']})", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(EXP / "analysis" / "crossfamily_ladder.png", dpi=130, bbox_inches="tight")
    print(f"\nwrote analysis/crossfamily_ladder.png and runs/verdict.json — VERDICT: {v['VERDICT']}")


if __name__ == "__main__":
    main()
