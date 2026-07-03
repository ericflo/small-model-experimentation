#!/usr/bin/env python3
"""Analyze the coverage-vs-selection anatomy: per-depth table, coverage@k curves, the wall MAP
(selection-bound vs coverage-bound + crossover depth), and verdicts on P1-P4."""
from __future__ import annotations

import json
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
COV0 = 0.10       # coverage floor below which a depth is COVERAGE-bound
GAP = 0.15        # coverage-first@1 margin to call a depth SELECTION-bound


def cov_at_k(c, n, k):
    k = min(k, n)
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    d = json.load(open(EXP / "runs" / "anatomy.json"))
    K = d["K"]
    recs = d["records"]
    fams = sorted({r["family"] for r in recs})
    by = defaultdict(list)
    for r in recs:
        by[(r["family"], r["depth"])].append(r)

    def agg(rs):
        n = len(rs)
        m = {
            "n": n,
            "first1": sum(r["first1"] for r in rs) / n,
            "vfilter": sum(r["vfilter_full"] for r in rs) / n,
            "mverify": sum(r["mverify_full"] for r in rs) / n,
            "rndVP": sum((r["n_vpass_full"] / r["n_vpass"]) if r["n_vpass"] else 0.0 for r in rs) / n,
            "vpass_rate": sum((r["n_vpass"] / r["K"]) for r in rs) / n,
        }
        for k in (1, 2, 4, 8, 16, 32):
            if k <= K:
                m[f"cov{k}"] = sum(cov_at_k(r["cov_full"], r["K"], k) for r in rs) / n
        m["covK"] = m[f"cov{K}"]
        return m

    stats = {(f, dep): agg(by[(f, dep)]) for (f, dep) in by}
    depths = sorted({dep for (_, dep) in by})

    print(f"\n=== Coverage vs Selection (K={K}) ===")
    hdr = f"{'fam':>9} {'d':>2} {'n':>3} {'first@1':>8} {'vfilter':>8} {'mverify':>8} {'rndVP':>7} " \
          f"{'cov@2':>7} {'cov@8':>7} {'cov@'+str(K):>7} {'regime':>16}"
    print(hdr)
    verdict = {"per_depth": {}, "crossover_dstar": {}}
    for f in fams:
        dstar = None
        for dep in depths:
            m = stats[(f, dep)]
            cov = m["covK"]
            gap = cov - m["first1"]
            # Selection is (empirically) free: vfilter recovers coverage. So regimes are about COVERAGE,
            # not selection. "sampling-gap" = single-shot underdeploys but sample+filter recovers it.
            if cov < COV0:
                regime = "coverage-wall"
                if dstar is None:
                    dstar = dep
            elif gap >= GAP:
                regime = "sampling-gap"
            else:
                regime = "easy"
            best_sel = max(m["vfilter"], m["mverify"])
            recov = (best_sel - m["first1"]) / gap if gap > 1e-9 else float("nan")
            verdict["per_depth"][f"{f}-d{dep}"] = {
                "first1": round(m["first1"], 3), "vfilter": round(m["vfilter"], 3),
                "mverify": round(m["mverify"], 3), "covK": round(cov, 3),
                "rndVP": round(m["rndVP"], 3), "gap": round(gap, 3),
                "best_selector_recovers_frac_of_gap": None if recov != recov else round(recov, 2),
                "regime": regime}
            print(f"{f:>9} {dep:>2} {m['n']:>3} {m['first1']:>8.2f} {m['vfilter']:>8.2f} "
                  f"{m['mverify']:>8.2f} {m['rndVP']:>7.2f} {m.get('cov2',0):>7.2f} "
                  f"{m.get('cov8',0):>7.2f} {cov:>7.2f} {regime:>16}")
        verdict["crossover_dstar"][f] = dstar

    # --- pre-registered verdicts (list) ---
    L = {dep: stats[("list", dep)] for dep in depths if ("list", dep) in stats}
    v = {}
    if 2 in L:
        v["P1_cov_rises_d2"] = bool(L[2]["covK"] >= 3 * max(L[2]["first1"], 1e-9)) and L[2]["first1"] < 0.5
    if 3 in L:
        v["P1_cov_d3_gt_0.10"] = bool(L[3]["covK"] > 0.10)
    v["P2_dstar_list"] = verdict["crossover_dstar"].get("list")
    # P3 selection gap where coverage>0.15
    p3 = {}
    for dep in depths:
        for f in fams:
            m = stats.get((f, dep))
            if m and m["covK"] > 0.15:
                p3[f"{f}-d{dep}"] = {"covK": round(m["covK"], 2), "vfilter": round(m["vfilter"], 2),
                                     "selection_loss_cov_minus_vfilter": round(m["covK"] - m["vfilter"], 2)}
    v["P3_selection_gaps"] = p3
    # P4 verifier > random-among-visible-passers at d2,d3 (list)
    p4 = {}
    for dep in (2, 3):
        m = stats.get(("list", dep))
        if m:
            p4[f"list-d{dep}"] = {"mverify": round(m["mverify"], 2), "rndVP": round(m["rndVP"], 2),
                                  "vfilter": round(m["vfilter"], 2),
                                  "mverify>rndVP": bool(m["mverify"] > m["rndVP"] + 1e-9),
                                  "mverify>=vfilter": bool(m["mverify"] >= m["vfilter"] - 1e-9)}
    v["P4_verifier_elicits"] = p4
    # selection-freeness + overfit traps: does visible-pass imply hidden-pass?
    tot_vp = sum(r["n_vpass"] for r in recs)
    tot_vpf = sum(r["n_vpass_full"] for r in recs)
    traps = defaultdict(lambda: [0, 0])  # (fam,depth) -> [overfit-only tasks, tasks-with-vp]
    for r in recs:
        if r["n_vpass"] > 0:
            traps[(r["family"], r["depth"])][1] += 1
            if r["n_vpass_full"] == 0:
                traps[(r["family"], r["depth"])][0] += 1
    v["selection_free"] = {
        "visible_pass_implies_hidden_pass_rate": round(tot_vpf / tot_vp, 3) if tot_vp else None,
        "max_cov_minus_vfilter": round(max(stats[k]["covK"] - stats[k]["vfilter"] for k in stats), 3),
        "overfit_trap_tasks": {f"{f}-d{dep}": f"{traps[(f,dep)][0]}/{traps[(f,dep)][1]}"
                               for (f, dep) in sorted(traps)},
    }
    verdict["prereg"] = v
    (EXP / "runs" / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print("\n=== Pre-registered verdicts ===")
    print(json.dumps(v, indent=1))
    print("\ncrossover d* (selection->coverage bound):", verdict["crossover_dstar"])

    # --- figures ---
    fig, axes = plt.subplots(1, len(fams) + 1, figsize=(5.5 * (len(fams) + 1), 4.3))
    colors = plt.cm.viridis([0.15, 0.4, 0.65, 0.9])
    ks = list(range(1, K + 1))
    for ax, f in zip(axes[:len(fams)], fams):
        for i, dep in enumerate(depths):
            rs = by.get((f, dep))
            if not rs:
                continue
            curve = [sum(cov_at_k(r["cov_full"], r["K"], k) for r in rs) / len(rs) for k in ks]
            ax.plot(ks, curve, "-", color=colors[i % 4], lw=2, label=f"depth {dep}")
            ax.scatter([1], [curve[0]], color=colors[i % 4], s=25, zorder=5)
        ax.axhline(COV0, ls=":", color="gray", lw=1)
        ax.set_title(f"{f}: coverage@k (sample-more ceiling)")
        ax.set_xlabel("k (samples)"); ax.set_ylabel("coverage (any correct)")
        ax.set_ylim(-0.03, 1.05); ax.grid(alpha=0.25); ax.legend(fontsize=8)

    # elicitation-gap bars for list
    axg = axes[-1]
    width = 0.2
    xs = range(len(depths))
    f = "list"
    b_first = [stats[(f, dep)]["first1"] if (f, dep) in stats else 0 for dep in depths]
    b_vf = [stats[(f, dep)]["vfilter"] if (f, dep) in stats else 0 for dep in depths]
    b_mv = [stats[(f, dep)]["mverify"] if (f, dep) in stats else 0 for dep in depths]
    b_cov = [stats[(f, dep)]["covK"] if (f, dep) in stats else 0 for dep in depths]
    axg.bar([x - 1.5 * width for x in xs], b_first, width, label="first@1 (single-shot)", color="#94a3b8")
    axg.bar([x - 0.5 * width for x in xs], b_vf, width, label="vfilter (execute-filter)", color="#3b82f6")
    axg.bar([x + 0.5 * width for x in xs], b_mv, width, label="mverify (model verifier)", color="#16a34a")
    axg.bar([x + 1.5 * width for x in xs], b_cov, width, label=f"coverage@{K} (ceiling)", color="#f59e0b")
    axg.set_xticks(list(xs)); axg.set_xticklabels([f"d{dep}" for dep in depths])
    axg.set_title("list: single-shot vs selectors vs ceiling")
    axg.set_ylabel("deployable accuracy"); axg.set_ylim(0, 1.05); axg.legend(fontsize=8); axg.grid(alpha=0.25, axis="y")

    fig.suptitle(f"Anatomy of the generation wall — coverage vs selection  (crossover d*: {verdict['crossover_dstar']})",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(EXP / "analysis" / "wall_anatomy.png", dpi=130, bbox_inches="tight")
    print("\nwrote analysis/wall_anatomy.png and runs/verdict.json")


if __name__ == "__main__":
    main()
