#!/usr/bin/env python3
"""Orchestrate the verifier-free banking pipeline (design-review-hardened). Idempotent stages:
1. harvest_pool -> data/pool.jsonl (+_smoke variants; smoke can NEVER poison a full run)
1b. judge_pool_think -> adds p_true_think to the pool (the no-think judge is within-task CHANCE on this
    substrate; CoT judging rescues it 0.49->0.80 -- runs/judge_think_diag.json). Conf arms rank on it.
2. build_arms   -> data/train_{exec,conf_strat,conf_global,rand}.jsonl (matched size, hard)
   GATE: proceed iff pool AUROC >= 0.65 AND conf_strat purity - rand purity >= 0.10 AND n_pairs >= 60
         (below the gate, "the judge does not transfer to this substrate" is the finding -- stop cheap).
3. train x4     -> runs/lora_*
4. eval         -> runs/eval_{arm}_{mode}.json  (n-per-depth 25, depths 1-3, frozen paired;
                   think-mode for base/exec/conf_strat/rand; conf_global nothink-only ablation)
5. calib        -> runs/calib_{arm}.json (fixed judge set) + runs/calib_self_{arm}.json (own distribution)
Run analyze.py separately."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# mandatory for these workloads per docs/compute_environment.md; its absence OOM'd the first full run
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

EXP = Path(__file__).resolve().parents[1]
S = EXP / "scripts"
ARMS = ["exec", "conf_strat", "conf_global", "rand"]
THINK_ARMS = ["base", "exec", "conf_strat", "rand"]


def sh(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    rc = subprocess.call([sys.executable] + cmd)
    if rc != 0:
        sys.exit(rc)


def smoke():
    sh([str(S / "harvest_pool.py"), "--smoke"])
    sh([str(S / "judge_pool_think.py"), "--smoke", "--budget", "128"])
    sh([str(S / "build_arms.py"), "--smoke"])
    sh([str(S / "train_lora.py"), "--train", "data/train_conf_strat_smoke.jsonl", "--out", "runs/lora_smoke", "--smoke"])
    sh([str(S / "eval_ladder.py"), "--tag", "smbase_nothink", "--smoke"])
    sh([str(S / "eval_ladder.py"), "--tag", "smconf_nothink", "--smoke", "--adapter", "runs/lora_smoke"])
    sh([str(S / "calib_eval.py"), "--tag", "smbase", "--K", "4", "--eval-file", "data/eval_frozen_smoke.jsonl",
        "--judge-set", "data/judge_set_smoke.jsonl"])
    sh([str(S / "calib_eval.py"), "--tag", "smconf", "--K", "4", "--eval-file", "data/eval_frozen_smoke.jsonl",
        "--judge-set", "data/judge_set_smoke.jsonl", "--adapter", "runs/lora_smoke"])
    sh([str(S / "calib_eval.py"), "--tag", "smconf", "--self-dist", "--self-K", "2",
        "--eval-file", "data/eval_frozen_smoke.jsonl", "--adapter", "runs/lora_smoke"])
    sh([str(S / "analyze.py"), "--arms", "smbase,smconf", "--modes", "nothink",
        "--summary-file", "runs/arms_summary_smoke.json"])
    print("\nSMOKE OK: pool + arms + train + eval(+adapter) + calib(fixed+self) + analyze all ran end-to-end")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true", help="override the post-arms gate")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return

    tt = EXP / "data" / "train_tasks.jsonl"
    if tt.exists():
        n_tasks = sum(1 for l in tt.read_text().splitlines() if l.strip())
        assert n_tasks == 90, f"train_tasks.jsonl has {n_tasks} tasks (expected 90) -- stale/smoke artifact?"
    if not (EXP / "data" / "pool.jsonl").exists():
        sh([str(S / "harvest_pool.py")])
    first = json.loads((EXP / "data" / "pool.jsonl").read_text().splitlines()[0])
    if "p_true_think" not in first:
        sh([str(S / "judge_pool_think.py")])
    if not (EXP / "data" / "train_exec.jsonl").exists():
        sh([str(S / "build_arms.py")])

    summ = json.load(open(EXP / "runs" / "arms_summary.json"))
    g = summ["gate"]
    purity_gap = summ["conf_strat"]["purity_full_pass"] - summ["rand"]["purity_full_pass"]
    checks = [(f"pool AUROC {g['pool_auroc_pooled']} >= 0.65", (g["pool_auroc_pooled"] or 0) >= 0.65),
              (f"conf_strat-rand purity gap {purity_gap:.3f} >= 0.10", purity_gap >= 0.10),
              (f"n_pairs {g['n_pairs']} >= 60", g["n_pairs"] >= 60)]
    print("\n[GATE] " + " | ".join(f"{'PASS' if ok else 'FAIL'}: {msg}" for msg, ok in checks), flush=True)
    if not all(ok for _, ok in checks) and not args.force:
        sys.exit("GATE FAILED -- the informative negative is 'P(True) judge does not transfer to this "
                 "substrate / pool too weak'. Document it; do not burn training+evals. (--force to override.)")

    for arm in ARMS:
        if not (EXP / "runs" / f"lora_{arm}").exists():
            sh([str(S / "train_lora.py"), "--train", f"data/train_{arm}.jsonl", "--out", f"runs/lora_{arm}"])

    evals = [("base", None)] + [(a, f"runs/lora_{a}") for a in ARMS]
    for tag, adapter in evals:
        for mode in ["nothink", "think"]:
            if mode == "think" and tag not in THINK_ARMS:
                continue
            if (EXP / "runs" / f"eval_{tag}_{mode}.json").exists():
                continue
            cmd = [str(S / "eval_ladder.py"), "--tag", f"{tag}_{mode}", "--K", "16",
                   "--n-per-depth", "25", "--depths", "1", "2", "3"]
            if adapter:
                cmd += ["--adapter", adapter]
            if mode == "think":
                cmd += ["--think"]
            sh(cmd)

    THINK_CALIB = ["base", "exec", "conf_strat"]
    SELF_THINK = ["base", "conf_strat"]
    for tag, adapter in evals:
        jobs = [(f"calib_{tag}.json", [])]
        if tag in THINK_CALIB:
            jobs.append((f"calib_think_{tag}.json", ["--think-judge"]))
        if tag in SELF_THINK:
            jobs.append((f"calib_self_think_{tag}.json", ["--self-dist", "--think-judge"]))
        for out, extra in jobs:
            if (EXP / "runs" / out).exists():
                continue
            cmd = [str(S / "calib_eval.py"), "--tag", tag] + extra
            if adapter:
                cmd += ["--adapter", adapter]
            sh(cmd)
    print("\nPIPELINE COMPLETE -- run analyze.py")


if __name__ == "__main__":
    main()
