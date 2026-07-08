#!/usr/bin/env python3
"""THINK-judge the full candidate pool (post-diagnostic pivot): the no-think judge is within-task chance
on this substrate, but CoT judging rescues it (0.49 -> 0.80 within-task; runs/judge_think_diag.json).
Adds p_true_think + think_forced_close to every pool record, in place."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP / "src"))
import common as C  # noqa: E402
import families as FAM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    suf = "_smoke" if args.smoke else ""

    fam = FAM.FAMILIES["list"]
    tasks = [json.loads(l) for l in (EXP / "data" / f"train_tasks{suf}.jsonl").read_text().splitlines() if l.strip()]
    pp = EXP / "data" / f"pool{suf}.jsonl"
    pool = [json.loads(l) for l in pp.read_text().splitlines() if l.strip()]
    todo = [i for i, x in enumerate(pool) if "p_true_think" not in x]
    print(f"[poolthink] {len(todo)}/{len(pool)} candidates to think-judge (budget {args.budget})", flush=True)
    if not todo:
        return

    import gen_lib as GL
    p = GL.Probe()
    t0 = time.time()
    jp = [p.judge_prompt(C.ident_prompt(fam, tasks[pool[i]["ti"]]), pool[i]["code"], enable_thinking=True)
          for i in todo]
    pts, forced = p.judge_think(jp, budget=args.budget, gen_batch=32, logit_batch=16)
    for i, v, f in zip(todo, pts, forced):
        pool[i]["p_true_think"] = round(float(v), 6)
        pool[i]["think_forced_close"] = bool(f)
    pp.write_text("\n".join(json.dumps(x) for x in pool) + "\n")
    frate = sum(forced) / max(1, len(forced))
    print(f"[poolthink] done: {len(todo)} judged, forced-close rate {frate:.2f} [{time.time()-t0:.0f}s]",
          flush=True)


if __name__ == "__main__":
    main()
