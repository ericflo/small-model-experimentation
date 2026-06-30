#!/usr/bin/env python3
"""Evaluate thinking-budget controllers vs fixed budgets on a deployable accuracy-vs-cost Pareto.

Offline: reuses greedy generations from qwen35_4b_thinking_budget_scaling (copied into data/).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import controller as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-reverify", action="store_true", help="use cached visible-test results")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cells = C.load_cells(reverify=not args.no_reverify)
    n = len(cells)
    print(f"loaded {n} tasks x {len(C.BUDGET_ORDER)} budgets", flush=True)
    if args.smoke:
        print("smoke: visible-test reverify + load OK")
        return 0

    fixed = {b: C.fixed_budget(cells, b) for b in C.BUDGET_ORDER}

    ladders = {
        "esc[256-512-1024]": ["think_256", "think_512", "think_1024"],
        "esc[256-512-1024-2048]": ["think_256", "think_512", "think_1024", "think_2048"],
        "esc[nothink-256-512-1024]": ["no_think", "think_256", "think_512", "think_1024"],
        "esc[nothink-1024] (2-tier)": ["no_think", "think_1024"],
    }
    controllers = {}
    for name, lad in ladders.items():
        controllers[name] = C.escalation(cells, lad, cumulative=True)
        controllers[name + " +continue"] = C.escalation(cells, lad, cumulative=False)

    oracle = C.oracle_ceiling(cells, ["no_think", "think_256", "think_512", "think_1024", "think_2048"])

    out = {"n_tasks": n, "fixed": fixed, "controllers": controllers, "oracle_ceiling": oracle}
    run_dir = EXP / "runs"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(out, indent=2))

    # table
    def row(name, d):
        return (f"| {name} | {d['accuracy']:.3f} | {d['mean_think']:.0f} | "
                f"{d.get('false_visible_commit', float('nan')):.3f} |")
    lines = ["| strategy | deployable acc | mean think tok | false-visible-commit |",
             "| --- | ---: | ---: | ---: |"]
    lines += [row("fixed:" + b, fixed[b]) for b in C.BUDGET_ORDER]
    lines += [row(k, v) for k, v in controllers.items()]
    lines.append(f"| ORACLE ceiling (non-deployable) | {oracle['accuracy']:.3f} | {oracle['mean_think']:.0f} | - |")
    table = "\n".join(lines)
    (EXP / "analysis").mkdir(exist_ok=True)
    (EXP / "analysis" / "pareto_table.md").write_text(table + "\n")
    print("\n" + table, flush=True)

    # figure: accuracy vs mean thinking tokens
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    fx = [fixed[b]["mean_think"] for b in C.BUDGET_ORDER]
    fy = [fixed[b]["accuracy"] for b in C.BUDGET_ORDER]
    ax.plot(fx, fy, "o-", color="#264653", label="fixed budget", zorder=3)
    for b in C.BUDGET_ORDER:
        ax.annotate(b.replace("think_", "").replace("no_think", "0"),
                    (fixed[b]["mean_think"], fixed[b]["accuracy"]), fontsize=7,
                    xytext=(0, 6), textcoords="offset points", ha="center")
    markers = ["s", "^", "D", "v", "P", "X", "*", "h"]
    for (name, d), m in zip(controllers.items(), markers):
        style = "#e76f51" if "+continue" not in name else "#f4a261"
        ax.scatter(d["mean_think"], d["accuracy"], marker=m, s=70, color=style,
                   edgecolor="k", linewidth=0.4, label=name, zorder=4)
    ax.scatter(oracle["mean_think"], oracle["accuracy"], marker="*", s=240, color="#2a9d8f",
               edgecolor="k", label="oracle ceiling (non-deployable)", zorder=5)
    ax.set_xlabel("mean thinking tokens (cost)")
    ax.set_ylabel("deployable full-test accuracy")
    ax.set_title(f"Thinking-budget controllers vs fixed budgets (MBPP, n={n})")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(EXP / "analysis" / "pareto.png", dpi=130)
    print("\nwrote runs/summary.json, analysis/pareto_table.md, analysis/pareto.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
