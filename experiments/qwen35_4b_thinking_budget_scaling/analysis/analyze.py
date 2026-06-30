#!/usr/bin/env python3
"""Analyze a thinking-budget sweep: summary table + scaling-curve figures.

Reads runs/<tag>/summary.json (+ verified.jsonl for pass@k-vs-k curves) and writes
analysis/<tag>_summary.md and figures under analysis/.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def nominal_budget(name: str) -> float:
    if name == "no_think":
        return 0.0
    if name.startswith("think_") and name[6:].isdigit():
        return float(name[6:])
    return float("inf")  # unbudgeted / controls sort last


def order_conditions(conds: list[str]) -> list[str]:
    return sorted(conds, key=lambda c: (nominal_budget(c), c))


def write_table(summary: dict, k: int, out_md: Path) -> str:
    conds = order_conditions(list(summary))
    rows = ["| condition | greedy@1 | sampled pass@1 | pass@%d (oracle) | visible-sel@1 | oracle−deployable | think tok | total tok | forced |" % k,
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in conds:
        s = summary[c]
        rows.append(f"| {c} | {s['greedy_pass@1']:.3f} | {s['sampled_pass@1']:.3f} | "
                    f"{s['pass@%d_oracle' % k]:.3f} | {s['visible_selector@1']:.3f} | "
                    f"{s['oracle_minus_deployable']:.3f} | {s['mean_think_tokens']:.0f} | "
                    f"{s['mean_total_tokens']:.0f} | {s['forced_close_frac']:.2f} |")
    md = "\n".join(rows)
    out_md.write_text(md + "\n")
    return md


def pass_at_k(n, c, kk):
    if kk > n:
        kk = n
    if n - c < kk:
        return 1.0
    p = 1.0
    for i in range(kk):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def passk_curves(verified: Path, k: int):
    """Return {cond: {kk: mean pass@kk}} computed from sampled candidates."""
    by = defaultdict(lambda: defaultdict(list))  # cond -> task -> [pass bool]
    for line in verified.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "sample":
            by[r["cond"]][r["task_id"]].append(bool(r["pass"]))
    out = {}
    for cond, tasks in by.items():
        curve = {}
        for kk in range(1, k + 1):
            vals = [pass_at_k(len(v), sum(v), kk) for v in tasks.values()]
            curve[kk] = sum(vals) / len(vals) if vals else 0.0
        out[cond] = curve
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    run_dir = EXP / "runs" / args.tag
    summary_path = run_dir / "summary.json"
    meta = json.loads(summary_path.read_text())
    summary = meta["conditions"]
    k = meta.get("k", 8)

    out_md = EXP / "analysis" / f"{args.tag}_summary.md"
    md = write_table(summary, k, out_md)
    print(md)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = order_conditions(list(summary))
    x = [summary[c]["mean_think_tokens"] for c in conds]
    greedy = [summary[c]["greedy_pass@1"] for c in conds]
    pk = [summary[c][f"pass@{k}_oracle"] for c in conds]
    sel = [summary[c]["visible_selector@1"] for c in conds]
    p1 = [summary[c]["sampled_pass@1"] for c in conds]

    # Figure 1: scaling curve (accuracy vs mean thinking tokens)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, pk, "o-", label=f"pass@{k} (oracle ceiling)", color="#2a9d8f")
    ax.plot(x, sel, "s-", label="visible-test selector@1 (deployable)", color="#e76f51")
    ax.plot(x, greedy, "^--", label="greedy@1 (deployable)", color="#264653")
    ax.plot(x, p1, "d:", label="sampled pass@1", color="#999999")
    for xi, c in zip(x, conds):
        ax.annotate(c.replace("think_", "").replace("no_think", "0"),
                    (xi, -0.04), ha="center", va="top", fontsize=7, rotation=0,
                    annotation_clip=False)
    ax.set_xlabel("mean thinking tokens used")
    ax.set_ylabel("MBPP accuracy")
    ax.set_title(f"Qwen3.5-4B thinking-budget scaling (MBPP, n={meta.get('n_tasks')})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(EXP / "analysis" / f"{args.tag}_scaling_curve.png", dpi=130)

    # Figure 2: accuracy vs total compute (tokens) — efficiency view
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    xt = [summary[c]["mean_total_tokens"] for c in conds]
    ax2.plot(xt, pk, "o-", label=f"pass@{k} (oracle)", color="#2a9d8f")
    ax2.plot(xt, sel, "s-", label="visible-selector@1 (deployable)", color="#e76f51")
    ax2.plot(xt, greedy, "^--", label="greedy@1", color="#264653")
    ax2.set_xlabel("mean total generated tokens (compute proxy)")
    ax2.set_ylabel("MBPP accuracy")
    ax2.set_title("Accuracy vs compute")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(EXP / "analysis" / f"{args.tag}_accuracy_vs_compute.png", dpi=130)

    # Figure 3: pass@k vs k per budget (if verified.jsonl present)
    vpath = run_dir / "verified.jsonl"
    if vpath.exists():
        curves = passk_curves(vpath, k)
        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        for c in order_conditions(list(curves)):
            ks = sorted(curves[c])
            ax3.plot(ks, [curves[c][kk] for kk in ks], "o-", label=c, markersize=3)
        ax3.set_xlabel("k (samples)")
        ax3.set_ylabel(f"pass@k")
        ax3.set_title("Oracle ceiling growth with samples, per thinking budget")
        ax3.legend(fontsize=7)
        ax3.grid(alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(EXP / "analysis" / f"{args.tag}_passk_vs_k.png", dpi=130)

    print(f"\nwrote analysis/{args.tag}_summary.md and figures to analysis/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
