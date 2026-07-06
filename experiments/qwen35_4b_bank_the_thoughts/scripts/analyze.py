#!/usr/bin/env python3
"""Bank-the-thoughts (Phase 1, synthetic decomposition traces). Does banking the PLAN (T) beat banking the
ANSWER (A) on (i) deployable depth-3 (coverage@16 + greedy@1) and (ii) step-1 planning ranking (the
rationalization-robust primary)? T_corrupt (same code, mismatched plan) is the content-causality control."""
from __future__ import annotations

import json
from math import comb, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]


def cov_at_k(c, n, k):
    k = min(k, n)
    return 0.0 if c == 0 else (1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k))


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 3), round(min(1, c + h), 3))


def deploy(tag, depth=3):
    p = EXP / "runs" / f"eval_deploy_{tag}.json"
    if not p.exists():
        return None
    rs = [r for r in json.load(open(p))["records"] if r["depth"] == depth]
    if not rs:
        return None
    n, K = len(rs), rs[0]["K"]
    return {"n": n, "cov16": sum(cov_at_k(r["cov_full"], r["K"], 16) for r in rs) / n,
            "greedy": sum(r["greedy_full"] for r in rs) / n,
            "per_sample": sum(r["cov_full"] for r in rs) / (n * K)}


def step1(tag):
    p = EXP / "runs" / f"results_s1_{tag}.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    return {r["budget"]: r["top1"] for r in rows if r["step"] == 1}


def main():
    v = {}
    print("\n=== DEPLOYABILITY: depth-3 (frozen held-out) ===")
    print(f"{'cell':>16} {'cov@16':>8} {'greedy@1':>9} {'per-sample':>11}")
    cells = [("base_nt", "base (no-think)"), ("A_nt", "A=answers (no-think)"),
             ("T_th", "T=plans (think)"), ("Tcorrupt_th", "T_corrupt (think)"), ("T_nt", "T (no-think)")]
    dep = {}
    for tag, lab in cells:
        d = deploy(tag)
        if not d:
            continue
        dep[tag] = d
        print(f"{lab:>16} {d['cov16']:>8.3f} {d['greedy']:>9.3f} {d['per_sample']:>11.3f}")
    v["deploy"] = {tag: {"cov16": round(dep[tag]["cov16"], 3), "greedy": round(dep[tag]["greedy"], 3)} for tag in dep}
    if "A_nt" in dep and "T_th" in dep:
        n = dep["A_nt"]["n"]
        v["T_beats_A_greedy"] = bool(dep["T_th"]["greedy"] - dep["A_nt"]["greedy"] > 0.05)
        v["T_beats_A_cov16"] = bool(dep["T_th"]["cov16"] - dep["A_nt"]["cov16"] > 0.05)
        v["content_causal_greedy(T vs Tcorrupt)"] = (
            round(dep["T_th"]["greedy"] - dep["Tcorrupt_th"]["greedy"], 3) if "Tcorrupt_th" in dep else None)

    print("\n=== STEP-1 PLANNING RANKING (rationalization-robust; top-1, chance 0.031) ===")
    print(f"{'model':>10} {'no-think(B0)':>13} {'think(B2048)':>13}")
    s1 = {}
    for tag in ("base", "A", "T", "Tcorrupt"):
        s = step1(tag)
        if s:
            s1[tag] = s
            print(f"{tag:>10} {s.get(0, float('nan')):>13.3f} {s.get(2048, float('nan')):>13.3f}")
    v["step1"] = {t: {str(b): s1[t][b] for b in s1[t]} for t in s1}
    if "A" in s1 and "T" in s1:
        v["T_installs_planning(step1 think > A)"] = bool(
            s1["T"].get(2048, 0) - s1["A"].get(2048, 0) > 0.08)

    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "verdict.json").write_text(json.dumps(v, indent=1))
    print("\n=== VERDICT ===\n" + json.dumps(v, indent=1))

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    order = ["base_nt", "A_nt", "T_th", "Tcorrupt_th", "T_nt"]
    labs = ["base", "A=answers", "T=plans\n(think)", "T_corrupt\n(think)", "T=plans\n(no-think)"]
    cols = ["#cbd5e1", "#94a3b8", "#16a34a", "#f59e0b", "#86efac"]
    xs = [i for i, t in enumerate(order) if t in dep]
    ax1.bar(xs, [dep[order[i]]["cov16"] for i in xs], width=0.38, label="coverage@16", color=[cols[i] for i in xs])
    ax1.bar([x + 0.4 for x in xs], [dep[order[i]]["greedy"] for i in xs], width=0.38, label="greedy@1 (deployable)",
            color=[cols[i] for i in xs], alpha=0.55, hatch="//")
    ax1.set_xticks([x + 0.2 for x in xs]); ax1.set_xticklabels([labs[i] for i in xs], fontsize=8)
    ax1.set_ylabel("depth-3 solve rate (held-out)"); ax1.set_title("Deployability: does banking PLANS beat banking ANSWERS?")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.25, axis="y")
    if s1:
        mods = [m for m in ("base", "A", "T", "Tcorrupt") if m in s1]
        x = range(len(mods)); w = 0.38
        ax2.bar([i - w / 2 for i in x], [s1[m].get(0, 0) for m in mods], w, label="no-think", color="#94a3b8")
        ax2.bar([i + w / 2 for i in x], [s1[m].get(2048, 0) for m in mods], w, label="think(2048)", color="#4a3aa7")
        ax2.axhline(1 / 32, ls="--", color="red", lw=1, label="chance")
        ax2.set_xticks(list(x)); ax2.set_xticklabels(mods)
        ax2.set_ylabel("step-1 next-op top-1"); ax2.set_title("Step-1 PLANNING (rationalization-robust primary)")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.25, axis="y")
    fig.suptitle("Bank-the-thoughts (synthetic decomposition plans): does banking the PLAN install usable planning?", y=1.02, fontsize=10.5)
    fig.tight_layout()
    (EXP / "analysis").mkdir(exist_ok=True)
    fig.savefig(EXP / "analysis" / "bank_thoughts.png", dpi=130, bbox_inches="tight")
    print("wrote analysis/bank_thoughts.png and runs/verdict.json")


if __name__ == "__main__":
    main()
