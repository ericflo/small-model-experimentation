#!/usr/bin/env python3
"""What are the FAILING episodes actually doing? Tool-call forensics on trajectories we already have.

The termination story so far is negative-shaped: episodes die at the 600s wall, and it is NOT doom
looping (loop_scan.py: 0% periodic repetition across 4.3M chars). That leaves the positive question
unanswered -- what is the model spending the wall on? A first glance at one episode showed 134 `bash`
calls against 2 `edit` calls, which if typical is a very different pathology from repetition: the model
probes the environment endlessly instead of committing an implementation.

This matters for choosing an intervention. "Stop looping" (FTPO/antidoom) targets repetition. "Commit
when ready" targets a model that has a good patch and keeps going. "Stop probing" targets a model that
never gets to a patch at all. They are different fixes and the trajectories can tell us which we have.

Costs nothing: reads logged pi event streams, no GPU, no model.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTD = HERE.parents[2] / "large_artifacts" / "qwen35_4b_realrepo_agentic_instrument"


def episode_stats(path: Path):
    """Tool-call counts, bash command shapes, and the last edit, from one trajectory."""
    tools = Counter()
    bash_cmds = []
    edits = 0
    last_edit = None
    turns = 0
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                t = ev.get("type", "")
                if t == "turn_start":
                    turns += 1
                if t == "tool_execution_start":
                    name = ev.get("toolName") or "?"
                    tools[name] += 1
                    args = ev.get("args") or {}
                    if name == "bash":
                        c = args.get("command") or args.get("cmd") or ""
                        if isinstance(c, str):
                            bash_cmds.append(c.strip()[:120])
                    if name in ("edit", "write", "str_replace"):
                        edits += 1
                        last_edit = args
    except OSError:
        return None
    return {"tools": tools, "bash_cmds": bash_cmds, "n_edits": edits,
            "turns": turns, "last_edit": last_edit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="base,warmstart")
    ap.add_argument("--out", default=str(HERE.parent / "reports" / "behavior_scan.json"))
    a = ap.parse_args()
    out = {}
    for label in a.labels.split(","):
        label = label.strip()
        rep = OUTD / "reports" / f"pi_{label}_test.json"
        if not rep.exists():
            continue
        eps = json.loads(rep.read_text()).get("episodes", [])
        groups = {"solved": [], "timeout": [], "failed_no_timeout": []}
        for e in eps:
            if not e.get("traj"):
                continue
            st = episode_stats(OUTD / e["traj"])
            if st is None:
                continue
            key = ("solved" if e.get("solved")
                   else "timeout" if e.get("pi_exit") == 124 else "failed_no_timeout")
            groups[key].append((e, st))

        arm = {}
        print(f"\n=== [{label}] ===", flush=True)
        for key, rows in groups.items():
            if not rows:
                continue
            bash = [len(s["bash_cmds"]) for _, s in rows]
            edit = [s["n_edits"] for _, s in rows]
            turns = [s["turns"] for _, s in rows]
            # repeated-command rate: how much of the bash traffic is the SAME command re-issued?
            rep_rates = []
            for _, s in rows:
                if s["bash_cmds"]:
                    c = Counter(s["bash_cmds"])
                    rep_rates.append(1 - len(c) / len(s["bash_cmds"]))
            arm[key] = {
                "n": len(rows),
                "bash_calls_median": statistics.median(bash) if bash else 0,
                "bash_calls_max": max(bash) if bash else 0,
                "edits_median": statistics.median(edit) if edit else 0,
                "edits_zero_frac": round(sum(1 for x in edit if x == 0) / len(edit), 3),
                "turns_median": statistics.median(turns) if turns else 0,
                "repeated_bash_frac_median": round(statistics.median(rep_rates), 3) if rep_rates else 0,
            }
            print(f"  {key:18s} n={arm[key]['n']:3d} | bash median {arm[key]['bash_calls_median']:5.0f} "
                  f"(max {arm[key]['bash_calls_max']:4d}) | edits median {arm[key]['edits_median']:3.0f} "
                  f"| never edited {arm[key]['edits_zero_frac']:.0%} | turns {arm[key]['turns_median']:3.0f} "
                  f"| repeated-bash {arm[key]['repeated_bash_frac_median']:.0%}", flush=True)
        # what commands dominate the timed-out episodes?
        if groups["timeout"]:
            allc = Counter()
            for _, s in groups["timeout"]:
                allc.update(s["bash_cmds"])
            arm["top_bash_in_timeouts"] = allc.most_common(8)
            print("  most-issued commands in timed-out episodes:", flush=True)
            for cmd, n in allc.most_common(6):
                print(f"      {n:4d}x  {cmd[:96]}", flush=True)
        out[label] = arm
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"\nsaved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
