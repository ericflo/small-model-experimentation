#!/usr/bin/env python3
"""Head-to-head deployable selectors on the k=8 candidate pool: does the thinking-verifier break the
C2 visible-test false-pass wall, and at what cost?
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
# rough token costs (tokens/op) for the cost picture; stated assumptions, not measured per-item.
GEN_TOK = 120        # a no-think candidate answer
VERIFY_TOK = 500     # a thinking-verification (budget 1024, mostly forced ~<=1024, mean ~500)


def main():
    pool = [json.loads(l) for l in (EXP / "data" / "pool.jsonl").read_text().splitlines() if l.strip()]
    by_task = defaultdict(list)
    for r in pool:
        by_task[r["task_id"]].append(r)
    tasks = list(by_task.values())
    k = max(len(v) for v in tasks)

    def acc(selector):
        return float(np.mean([1 if selector(v)["full_pass"] else 0 for v in tasks]))

    # selectors: each takes a task's candidate list, returns the chosen candidate
    def s_pass1(v):  # expected single sample -> use sample 0 (deployable, 1 generation)
        return sorted(v, key=lambda r: r["sample"])[0]

    def s_visible(v):  # first (lowest-sample) candidate passing the visible test, else sample 0
        vp = sorted([r for r in v if r["visible_pass"]], key=lambda r: r["sample"])
        return vp[0] if vp else s_pass1(v)

    def s_verifier(key):
        def f(v):
            return max(v, key=lambda r: r[key])
        return f

    def s_visible_plus(key):
        def f(v):
            vp = [r for r in v if r["visible_pass"]]
            pool_ = vp if vp else v
            return max(pool_, key=lambda r: r[key])
        return f

    pass1 = float(np.mean([r["full_pass"] for r in pool]))
    oracle = float(np.mean([1 if any(r["full_pass"] for r in v) else 0 for v in tasks]))

    results = {
        "pass@1 (random single)": (pass1, GEN_TOK),
        "visible-only (first visible-pass)": (acc(s_visible), k * GEN_TOK),
        "no-think verifier (max P_A)": (acc(s_verifier("pa_nothink")), k * GEN_TOK),  # +~0 gen tokens
        "thinking verifier (max P_A)": (acc(s_verifier("pa_think")), k * GEN_TOK + k * VERIFY_TOK),
        "visible + no-think verifier": (acc(s_visible_plus("pa_nothink")), k * GEN_TOK),
        "visible + thinking verifier": (acc(s_visible_plus("pa_think")), k * GEN_TOK + k * VERIFY_TOK),
        "ORACLE pass@k (non-deployable)": (oracle, k * GEN_TOK),
    }

    # C2 false-pass breakdown
    vis = [r for r in pool if r["visible_pass"]]
    false_pass = [r for r in vis if not r["full_pass"]]
    fp_rate = len(false_pass) / max(1, len(vis))
    # among visible-passers, can the thinking-verifier rank true-pass above false-pass?
    from sklearn.metrics import roc_auc_score
    yv = np.array([int(r["full_pass"]) for r in vis])
    auroc_think = roc_auc_score(yv, [r["pa_think"] for r in vis]) if len(set(yv)) == 2 else float("nan")
    auroc_nt = roc_auc_score(yv, [r["pa_nothink"] for r in vis]) if len(set(yv)) == 2 else float("nan")

    out = {
        "n_tasks": len(tasks), "k": k, "pass@1": round(pass1, 3), "oracle": round(oracle, 3),
        "selectors": {name: {"accuracy": round(a, 3), "est_tokens_per_task": t} for name, (a, t) in results.items()},
        "c2_false_pass": {"visible_pass_rate": round(len(vis) / len(pool), 3),
                          "false_pass_rate_among_visible": round(fp_rate, 3),
                          "n_false_passes": len(false_pass),
                          "among_visible_passers_verifier_auroc_think": round(auroc_think, 3),
                          "among_visible_passers_verifier_auroc_nothink": round(auroc_nt, 3)},
    }
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "summary.json").write_text(json.dumps(out, indent=2))

    print("\n=== SELECTOR SHOWDOWN (deployable; k=%d pool) ===" % k)
    for name, (a, t) in results.items():
        print(f"  {name:34s} acc={a:.3f}   ~{t} tok/task")
    print(f"\nC2 false-pass wall: visible-pass rate {len(vis)/len(pool):.3f}; "
          f"{fp_rate:.3f} of visible-passers FULL-FAIL ({len(false_pass)} false-passes)")
    print(f"Among visible-passers, verifier AUROC (rank true>false): think {auroc_think:.3f}, no-think {auroc_nt:.3f}")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["pass@1 (random single)", "visible-only (first visible-pass)", "no-think verifier (max P_A)",
             "thinking verifier (max P_A)", "visible + thinking verifier", "ORACLE pass@k (non-deployable)"]
    accs = [results[o][0] for o in order]
    labels = ["pass@1", "visible\nonly", "no-think\nverifier", "thinking\nverifier", "visible +\nthink-verifier", "oracle"]
    cols = ["#264653", "#e76f51", "#e9c46a", "#2a9d8f", "#1d6f63", "#8d99ae"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    b = ax.bar(labels, accs, color=cols, edgecolor="k", lw=0.5)
    for r, a in zip(b, accs):
        ax.text(r.get_x() + r.get_width() / 2, a + 0.008, f"{a:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(oracle, color="#8d99ae", ls=":", lw=1)
    ax.set_ylim(0, 1); ax.set_ylabel("deployable MBPP accuracy (selected candidate)")
    ax.set_title("Selection signals on the k=8 pool: does self-verification beat the visible-test wall?")
    fig.tight_layout(); fig.savefig(EXP / "analysis" / "selectors.png", dpi=130)
    print("\nwrote runs/summary.json, analysis/selectors.png")


if __name__ == "__main__":
    main()
