#!/usr/bin/env python3
"""Task-level analysis of a thinking-budget sweep: where does thinking help?

- fail->pass flips vs regressions on the deployable greedy answer, per budget vs no_think
- difficulty slices by no_think oracle (pass@8): does thinking help hard tasks or easy ones?
Reads runs/<tag>/verified.jsonl, writes analysis/<tag>_deeper.md.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def nominal(c):
    if c == "no_think":
        return 0
    if c.startswith("think_") and c[6:].isdigit():
        return int(c[6:])
    return 10**9


def load(tag):
    recs = [json.loads(l) for l in (EXP / "runs" / tag / "verified.jsonl").read_text().splitlines() if l.strip()]
    greedy = defaultdict(dict)   # cond -> task -> pass(bool)
    samples = defaultdict(lambda: defaultdict(list))  # cond -> task -> [pass]
    for r in recs:
        if r["kind"] == "greedy":
            greedy[r["cond"]][r["task_id"]] = bool(r["pass"])
        else:
            samples[r["cond"]][r["task_id"]].append(bool(r["pass"]))
    return greedy, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    greedy, samples = load(args.tag)
    conds = sorted(greedy, key=nominal)
    base = "no_think"
    if base not in greedy:
        base = conds[0]
    tasks = sorted(greedy[base])

    out = [f"# Deeper analysis ({args.tag})\n",
           "## Deployable greedy flips vs no_think\n",
           "| budget | greedy@1 | fail→pass | pass→fail | net |",
           "| --- | ---: | ---: | ---: | ---: |"]
    for c in conds:
        if c == base:
            acc = sum(greedy[base].values()) / len(tasks)
            out.append(f"| {c} | {acc:.3f} | – | – | – |")
            continue
        up = sum(1 for t in tasks if not greedy[base].get(t) and greedy[c].get(t))
        dn = sum(1 for t in tasks if greedy[base].get(t) and not greedy[c].get(t))
        acc = sum(greedy[c].get(t, False) for t in tasks) / len(tasks)
        out.append(f"| {c} | {acc:.3f} | {up} | {dn} | {up - dn:+d} |")

    # difficulty slices by no_think oracle pass@8
    def bucket(t):
        s = samples[base].get(t, [])
        if not s:
            return "n/a"
        frac = sum(s) / len(s)
        if frac == 0:
            return "never (0/8)"
        if frac == 1:
            return "always (8/8)"
        return "sometimes (1-7/8)"

    buckets = defaultdict(list)
    for t in tasks:
        buckets[bucket(t)].append(t)
    out += ["\n## Difficulty slices (by no_think oracle pass@8)\n",
            "Greedy@1 within each slice, no_think vs best thinking budget.\n",
            "| slice | n | no_think greedy | best-think greedy | Δ |",
            "| --- | ---: | ---: | ---: | ---: |"]
    think_conds = [c for c in conds if c != base]
    for name in ["always (8/8)", "sometimes (1-7/8)", "never (0/8)"]:
        ts = buckets.get(name, [])
        if not ts:
            continue
        b = sum(greedy[base].get(t, False) for t in ts) / len(ts)
        best = max(sum(greedy[c].get(t, False) for t in ts) / len(ts) for c in think_conds)
        out.append(f"| {name} | {len(ts)} | {b:.3f} | {best:.3f} | {best - b:+.3f} |")

    md = "\n".join(out) + "\n"
    (EXP / "analysis" / f"{args.tag}_deeper.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
