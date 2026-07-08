#!/usr/bin/env python3
"""DIAGNOSTIC (post-gate pivot): the no-think judge's WITHIN-task AUROC is at chance (0.471) on this
substrate -- it reads task difficulty, not candidate correctness. Hypothesis (C44 serial-compute law):
judging 'does this transform reproduce the examples' requires mentally EXECUTING the candidate, which a
single forward pass cannot do. Test: THINK-mode judging (judge_think) on a subsample of the already-graded
pool -- all correct candidates + stratified incorrect. Compare within-task + pooled AUROC vs the stored
no-think P(True). Decides whether the banking experiment proceeds with a think-P(True) filter."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP / "src"))
import common as C  # noqa: E402
import families as FAM  # noqa: E402
from harvest_pool import auroc  # noqa: E402


def within_auroc(cands, key):
    by = {}
    for x in cands:
        by.setdefault(x["ti"], []).append((x[key], int(x["full_pass"])))
    per = []
    for sy in by.values():
        yy = [y for _, y in sy]
        if 0 < sum(yy) < len(yy):
            a = auroc([s for s, _ in sy], yy)
            if a is not None:
                per.append(a)
    return (round(sum(per) / len(per), 3), len(per)) if per else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--neg-per-task", type=int, default=6, help="incorrect candidates kept per task")
    ap.add_argument("--seed", type=int, default=303)
    args = ap.parse_args()

    fam = FAM.FAMILIES["list"]
    tasks = [json.loads(l) for l in (EXP / "data" / "train_tasks.jsonl").read_text().splitlines() if l.strip()]
    pool = [json.loads(l) for l in (EXP / "data" / "pool.jsonl").read_text().splitlines() if l.strip()]

    # subsample: ALL correct + up to neg-per-task incorrect per task (keeps within-task cells mixed)
    rng = random.Random(args.seed)
    sub = [x for x in pool if x["full_pass"]]
    by_task = {}
    for x in pool:
        if not x["full_pass"]:
            by_task.setdefault(x["ti"], []).append(x)
    for ti, xs in by_task.items():
        rng.shuffle(xs)
        sub += xs[:args.neg_per_task]
    rng.shuffle(sub)
    print(f"[diag] {len(sub)} candidates ({sum(x['full_pass'] for x in sub)} correct) from {len(pool)}-pool",
          flush=True)

    import gen_lib as GL
    p = GL.Probe()
    t0 = time.time()
    jp = [p.judge_prompt(C.ident_prompt(fam, tasks[x["ti"]]), x["code"], enable_thinking=True) for x in sub]
    pts, forced = p.judge_think(jp, budget=args.budget, gen_batch=32, logit_batch=16)
    for x, v, f in zip(sub, pts, forced):
        x["p_true_think"] = round(float(v), 6)
        x["think_forced_close"] = bool(f)
    frate = sum(forced) / max(1, len(forced))
    print(f"[diag] think-judged {len(sub)} | forced-close (budget-truncated thinking) rate {frate:.2f} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    ys = [int(x["full_pass"]) for x in sub]
    res = {"n": len(sub), "n_correct": sum(ys), "budget": args.budget,
           "forced_close_rate": round(sum(1 for x in sub if x["think_forced_close"]) / len(sub), 3),
           "pooled_auroc": {"nothink": round(auroc([x["p_true"] for x in sub], ys), 3),
                            "think": round(auroc([x["p_true_think"] for x in sub], ys), 3)}}
    for key, name in [("p_true", "nothink"), ("p_true_think", "think")]:
        a, nmix = within_auroc(sub, key)
        res.setdefault("within_auroc", {})[name] = a
        res["n_mixed_tasks"] = nmix
    for name, key in [("nothink", "p_true"), ("think", "p_true_think")]:
        cor = [x[key] for x in sub if x["full_pass"]]
        inc = [x[key] for x in sub if not x["full_pass"]]
        res.setdefault("mean_p_true", {})[name] = {
            "correct": round(sum(cor) / max(1, len(cor)), 3), "incorrect": round(sum(inc) / max(1, len(inc)), 3)}
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "judge_think_diag.json").write_text(json.dumps(res, indent=1))
    (EXP / "data" / "diag_sub.jsonl").write_text("\n".join(json.dumps(x) for x in sub) + "\n")
    print(json.dumps(res, indent=1))
    verdict = ("RESCUED: think-judging restores within-task discrimination -> rebuild conf arms with "
               "think-P(True) and proceed" if (res["within_auroc"]["think"] or 0) >= 0.65 else
               "NOT RESCUED: within-task discrimination stays weak even WITH serial compute -> the negative "
               "is deeper than serial compute; document")
    print(f"[diag] {verdict}")


if __name__ == "__main__":
    main()
