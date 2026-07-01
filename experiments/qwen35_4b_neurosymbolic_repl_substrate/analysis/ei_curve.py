#!/usr/bin/env python3
"""M4 expert-iteration trajectory: held-out accuracy per round + the flywheel's fuel (coverage, #pairs)."""
from __future__ import annotations

import json
import re
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load(f):
    p = EXP / "runs" / f
    return json.loads(p.read_text()) if p.exists() else None


def main():
    frozen = load("eval_frozen_big.json")
    rounds = []
    r = 1
    while (EXP / "runs" / f"ei_eval_{r}.json").exists():
        rounds.append(load(f"ei_eval_{r}.json")); r += 1
    if frozen is None or not rounds:
        print("missing eval files"); return

    # per-round coverage + accumulated pairs from the loop log
    log = (EXP / "runs" / "ei_run.log").read_text() if (EXP / "runs" / "ei_run.log").exists() else ""
    cov = re.findall(r"round solved (\d+)/(\d+)", log)
    tot = re.findall(r"total (\d+)\)", log)

    xs = list(range(len(rounds) + 1))  # 0 = frozen
    g = [frozen["think_greedy@1"]["overall"]] + [rd["think_greedy@1"]["overall"] for rd in rounds]
    pk = [frozen["think_pass@5"]["overall"]] + [rd["think_pass@5"]["overall"] for rd in rounds]
    depths = sorted(frozen["think_greedy@1"]["by_depth"], key=int)
    bydepth = {d: [frozen["think_greedy@1"]["by_depth"][d]] + [rd["think_greedy@1"]["by_depth"].get(d, 0) for rd in rounds] for d in depths}

    summary = {"rounds": len(rounds), "greedy@1_by_round": [round(x, 3) for x in g],
               "pass@5_by_round": [round(x, 3) for x in pk],
               "by_depth_greedy": {d: [round(x, 3) for x in v] for d, v in bydepth.items()},
               "train_pool_coverage": [f"{a}/{b}" for a, b in cov], "accumulated_pairs": tot}
    (EXP / "runs" / "ei_summary.json").write_text(json.dumps(summary, indent=2))
    print("=== EXPERT-ITERATION TRAJECTORY (held-out greedy@1) ===")
    print("  round:   " + "  ".join(f"{i}" for i in xs))
    print("  greedy@1:" + "  ".join(f"{x:.3f}" for x in g))
    print("  pass@5:  " + "  ".join(f"{x:.3f}" for x in pk))
    for d in depths:
        print(f"  depth {d}: " + "  ".join(f"{x:.3f}" for x in bydepth[d]))
    print(f"  coverage: {summary['train_pool_coverage']}  pairs: {summary['accumulated_pairs']}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(xs, g, "o-", color="#e76f51", lw=2, label="greedy@1 (deployable)")
    ax1.plot(xs, pk, "s--", color="#2a9d8f", label="pass@5 (coverage)")
    ax1.axhline(g[0], color="#8d99ae", ls=":", lw=1, label="frozen")
    for i, v in zip(xs, g):
        ax1.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax1.set_xlabel("expert-iteration round (0 = frozen)"); ax1.set_ylabel("held-out accuracy (n=135)")
    ax1.set_xticks(xs); ax1.set_ylim(0, max(pk) + 0.08); ax1.set_title("Does self-training compound across rounds?")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    cols = ["#264653", "#e9c46a", "#e76f51", "#8d99ae"]
    for i, d in enumerate(depths):
        ax2.plot(xs, bydepth[d], "o-", color=cols[i % len(cols)], label=f"depth {d}")
    ax2.set_xlabel("round"); ax2.set_ylabel("greedy@1 by depth"); ax2.set_xticks(xs); ax2.set_ylim(0, 1)
    ax2.set_title("Does the frontier (depth 3) crack?"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(EXP / "analysis" / "ei_trajectory.png", dpi=130)
    print("wrote runs/ei_summary.json, analysis/ei_trajectory.png")


if __name__ == "__main__":
    main()
