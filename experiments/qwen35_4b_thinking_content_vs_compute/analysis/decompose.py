#!/usr/bin/env python3
"""Combine behavioral full-pass + probe separability into the content-vs-compute decomposition.

Ladder: no_think -> foreign -> shuffle -> real. Attributes the thinking gain into:
  compute+scaffold (foreign - no_think), relevance/presence (shuffle - foreign), order (real - shuffle).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ORDER = ["no_think", "foreign", "shuffle", "real"]


def behavioral():
    labels = [json.loads(l) for l in (EXP / "data" / "labels.jsonl").read_text().splitlines() if l.strip()]
    agg = defaultdict(lambda: [0, 0, 0])
    for r in labels:
        a = agg[r["cond"]]; a[0] += 1; a[1] += int(r["full_pass"]); a[2] += int(r["visible_pass"])
    return {c: (a[1] / a[0], a[2] / a[0], a[0]) for c, a in agg.items()}


def main():
    beh = behavioral()
    pr = {}
    p = EXP / "runs" / "probe_results.json"
    if p.exists():
        pr = json.loads(p.read_text())

    rows = ["| rung | full-pass | visible-pass | best-layer probe AUC | what it adds vs the rung below |",
            "| --- | ---: | ---: | ---: | --- |"]
    adds = {"no_think": "(baseline)", "foreign": "compute + scaffold (irrelevant thinking)",
            "shuffle": "token-presence / relevance (relevant tokens, scrambled)",
            "real": "coherent order"}
    for c in ORDER:
        if c not in beh:
            continue
        fp, vp, nn = beh[c]
        auc = pr.get(c, {}).get("best_layer_auc")
        rows.append(f"| {c} | {fp:.3f} | {vp:.3f} | {auc if auc is not None else '-'} | {adds[c]} |")
    table = "\n".join(rows)

    # decomposition deltas (behavioral full-pass)
    def d(a, b):
        return beh[a][0] - beh[b][0] if a in beh and b in beh else float("nan")
    decomp = [
        "\n## Decomposition of the behavioral thinking gain (full-pass)",
        f"- compute + scaffold (foreign - no_think): {d('foreign','no_think'):+.3f}",
        f"- token-presence / relevance (shuffle - foreign): {d('shuffle','foreign'):+.3f}",
        f"- coherent order (real - shuffle): {d('real','shuffle'):+.3f}",
        f"- total (real - no_think): {d('real','no_think'):+.3f}",
    ]
    out = table + "\n" + "\n".join(decomp) + "\n"
    (EXP / "analysis" / "decomposition.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
