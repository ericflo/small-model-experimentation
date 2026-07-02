#!/usr/bin/env python3
"""Grid analysis vs the pre-registered predictions (reports/prereg.md): P1 (k=0 deep survives), P2 (k
drives at fixed d), P3 (two-parameter reliability model beats depth-only), P5 (false-pass rises with k),
P6 (thinking length), P9 (first-op rank degrades with k not d)."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def main():
    recs = [json.loads(l) for l in (EXP / "data" / "grid_records.jsonl").read_text().splitlines()]
    cells = defaultdict(list)
    for r in recs:
        cells[(r["depth"], r["n_destr"])].append(r)

    def rate(rs, f):
        return sum(f(r) for r in rs) / len(rs)

    table = {}
    print("=== GRID: greedy / pass@6 (n) ===")
    print("d\\k " + "".join(f"{k:>14d}" for k in range(4)))
    for d in (1, 2, 3, 4, 5):
        row = []
        for k in range(4):
            rs = cells.get((d, k))
            if not rs:
                row.append(" " * 14); continue
            g, pk = rate(rs, lambda r: r["greedy_full"]), rate(rs, lambda r: r["passk_full"])
            table[(d, k)] = {"n": len(rs), "greedy": round(g, 3), "passk": round(pk, 3)}
            row.append(f"  {g:.2f}/{pk:.2f}({len(rs)})")
        print(f" {d} " + "".join(f"{x:>14s}" for x in row))

    # P1: k=0 at d4/d5 >= 0.4 pass@6
    p1 = {d: table.get((d, 0), {}).get("passk") for d in (4, 5)}
    print(f"\nP1 (k=0 deep survives >=0.4): d4 {p1[4]}, d5 {p1[5]} -> "
          f"{'CONFIRMED' if all(v is not None and v >= 0.4 for v in p1.values()) else 'REFUTED/PARTIAL'}")
    # P2: monotone in k at d3; drop >= 0.3 k0->k2
    d3 = [table.get((3, k), {}).get("passk") for k in range(4)]
    mono = all(a >= b for a, b in zip(d3, d3[1:]) if a is not None and b is not None)
    drop = (d3[0] - d3[2]) if (d3[0] is not None and d3[2] is not None) else None
    print(f"P2 (k drives at d3): passk by k {d3}, monotone={mono}, k0->k2 drop={drop} -> "
          f"{'CONFIRMED' if mono and drop is not None and drop >= 0.3 else 'CHECK'}")

    # P3: logistic fits — depth-only vs (nT, nD)
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        X_d = np.array([[r["depth"]] for r in recs], float)
        X_tk = np.array([[r["depth"] - r["n_destr"], r["n_destr"]] for r in recs], float)
        y = np.array([int(r["passk_full"]) for r in recs])
        def ll(X):
            m = LogisticRegression(C=1e6).fit(X, y)
            p = m.predict_proba(X)[:, 1]
            return float(np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))), m
        ll_d, m_d = ll(X_d); ll_tk, m_tk = ll(X_tk)
        aic_d, aic_tk = 2 * 2 - 2 * ll_d, 2 * 3 - 2 * ll_tk
        print(f"P3 (model comparison): depth-only AIC {aic_d:.1f} vs (nT,nD) AIC {aic_tk:.1f} -> "
              f"{'(nT,nD) WINS' if aic_tk < aic_d - 2 else 'no clear win'}; "
              f"coef transparent {m_tk.coef_[0][0]:.2f}, destructive {m_tk.coef_[0][1]:.2f}")
        p3 = {"aic_depth": round(aic_d, 1), "aic_tk": round(aic_tk, 1),
              "coef_T": round(float(m_tk.coef_[0][0]), 3), "coef_D": round(float(m_tk.coef_[0][1]), 3)}
    except Exception as e:
        p3 = {"error": str(e)}; print("P3 fit failed:", e)

    # P5: false-pass rate among visible-passing candidates, by k
    fp = defaultdict(lambda: [0, 0])
    for r in recs:
        for v, fl in zip(r["cand_vis"], r["cand_full"]):
            if v:
                c = fp[r["n_destr"]]; c[0] += 1; c[1] += int(not fl)
    fp_tab = {k: (f"{c[1]}/{c[0]}" if c[0] else "0/0") for k, c in sorted(fp.items())}
    print(f"P5 (false-pass by k): {fp_tab}")

    # P6: thinking length by depth & k
    nt = defaultdict(list)
    for r in recs:
        nt[(r["depth"], r["n_destr"])].append(r["mean_nthink"])
    # P9: first-op rank medians
    import statistics
    p9 = {}
    for cell in [(3, 0), (2, 2), (2, 0), (3, 2), (4, 0)]:
        rs = cells.get(cell)
        if rs:
            ranks = [r["first_op_rank"] for r in rs if r["first_op_rank"]]
            p9[str(cell)] = statistics.median(ranks) if ranks else None
    print(f"P9 (first-op median rank): {p9} -> predict (3,0) better than (2,2)")

    out = {"cells": {f"{d},{k}": v for (d, k), v in table.items()}, "P1": p1, "P2": {"by_k_d3": d3, "monotone": mono, "drop": drop},
           "P3": p3, "P5": fp_tab, "P9": p9}
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "grid_analysis.json").write_text(json.dumps(out, indent=2))

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, col, mk in [(0, "#2a9d8f", "o"), (1, "#e9c46a", "s"), (2, "#e76f51", "^"), (3, "#264653", "d")]:
        ds = [d for d in (1, 2, 3, 4, 5) if (d, k) in table]
        if ds:
            ax1.plot(ds, [table[(d, k)]["passk"] for d in ds], mk + "-", color=col, label=f"k={k} destructive")
    ax1.set_xlabel("composition depth d"); ax1.set_ylabel("pass@6 (hidden, verified-depth tasks)")
    ax1.set_ylim(0, 1.02); ax1.set_title("Does depth or information destruction set the wall?")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ks = sorted({k for (_, k) in table})
    width = 0.2
    for i, d in enumerate((2, 3, 4)):
        vals = [table.get((d, k), {}).get("passk", 0) for k in ks]
        ax2.bar([k + (i - 1) * width for k in ks], vals, width, label=f"d={d}",
                color=["#8d99ae", "#e9c46a", "#e76f51"][i])
    ax2.set_xlabel("# destructive ops (k)"); ax2.set_ylabel("pass@6"); ax2.set_xticks(ks)
    ax2.set_title("Destruction effect at fixed depth"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(EXP / "analysis" / "grid_main.png", dpi=130)
    print("wrote runs/grid_analysis.json, analysis/grid_main.png")


if __name__ == "__main__":
    main()
