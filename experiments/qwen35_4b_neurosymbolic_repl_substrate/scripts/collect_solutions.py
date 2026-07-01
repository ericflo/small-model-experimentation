#!/usr/bin/env python3
"""Collect the 4B's OWN verified solutions on a large FRESH pool -> single-shot training data (M3).

Solve a fresh training pool with thinking sampling (+ optional REPL augmentation for misses); keep only
execution-verified-correct (prompt -> code) pairs. Also emit a held-out eval set (fresh seed, includes an
unseen extrapolation depth). No teacher: every solution is the 4B's own, execution-filtered.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import gen_tasks as G  # noqa: E402
import code_env as E  # noqa: E402


def extract(p, prompt, seq_ids):
    text = p.tok.decode(seq_ids[len(p._ids(prompt)):], skip_special_tokens=False)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    c, _ = E.extract_candidate_code(text, "transform")
    return c or ""


def solves(task, code):
    return bool(code) and bool(E.execute_public_and_asserts(code, G.to_public_cases(task), G.to_hidden_asserts(task))["full_pass"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--per-depth", type=int, default=120)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--max-per-task", type=int, default=2)
    ap.add_argument("--train-seed", type=int, default=202)
    ap.add_argument("--eval-seed", type=int, default=303)
    ap.add_argument("--eval-depths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--eval-per-depth", type=int, default=20)
    args = ap.parse_args()

    train_tasks = G.build_dataset(args.train_depths, args.per_depth, seed=args.train_seed)
    eval_tasks = G.build_dataset(args.eval_depths, args.eval_per_depth, seed=args.eval_seed)
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "eval_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in eval_tasks) + "\n")
    print(f"train pool {len(train_tasks)} (depths {args.train_depths}), eval {len(eval_tasks)} (depths {args.eval_depths})", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    prompts = [G.prompt_for(t) for t in train_tasks]
    rep = [pr for pr in prompts for _ in range(args.k)]
    print(f"sampling {len(rep)} drafts (k={args.k}, thinking, budget {args.budget})...", flush=True)
    gens = p.gen_sequences(rep, think=True, budget=args.budget, greedy=False, batch_size=48)

    pairs, per_depth_solved = [], defaultdict(int)
    solved_tasks = set()
    for ti, t in enumerate(train_tasks):
        seen, kept = set(), 0
        for j in range(args.k):
            code = extract(p, rep[ti * args.k + j], gens[ti * args.k + j]["seq_ids"])
            if code and code not in seen and solves(t, code):
                seen.add(code)
                pairs.append({"task_id": t["task_id"], "depth": t["depth"], "prompt": prompts[ti], "code": code})
                kept += 1
                if kept >= args.max_per_task:
                    break
        if kept:
            solved_tasks.add(t["task_id"])
            per_depth_solved[t["depth"]] += 1

    (EXP / "data" / "train.jsonl").write_text("\n".join(json.dumps(r) for r in pairs) + "\n")
    print(f"collected {len(pairs)} verified (prompt->code) pairs from {len(solved_tasks)}/{len(train_tasks)} solved tasks")
    for d in sorted(per_depth_solved):
        n = sum(1 for t in train_tasks if t["depth"] == d)
        print(f"  depth {d}: {per_depth_solved[d]}/{n} tasks solved")
    print("wrote data/train.jsonl, data/eval_tasks.jsonl")


if __name__ == "__main__":
    main()
