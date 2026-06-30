#!/usr/bin/env python3
"""Extract answer-token activations under thinking conditions for the separability probe.

Generates k samples per MBPP task under each condition (capturing full token sequences),
extracts per-layer last-token hidden states, and saves activations (to large_artifacts,
gitignored) + small records for the torch-free verifier and the probe.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import tasks as T  # noqa: E402

ACTS_DIR = EXP.parents[1] / "large_artifacts" / "qwen35_4b_thinking_separability_probe"

CONDS = {
    "no_think": dict(think=False, budget=None, shuffle=False),
    "think_512": dict(think=True, budget=512, shuffle=False),
    "shuffle_512": dict(think=True, budget=512, shuffle=True),
    "think_1024": dict(think=True, budget=1024, shuffle=False),
    "shuffle_1024": dict(think=True, budget=1024, shuffle=True),
}


def batch_for(cfg):
    if not cfg["think"]:
        return 64
    return 48 if (cfg["budget"] or 0) <= 1024 else 40


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--conditions", default=",".join(CONDS))
    args = ap.parse_args()
    if args.smoke:
        args.tasks, args.k, args.conditions = 4, 2, "no_think,think_1024"

    conds = args.conditions.split(",")
    ACTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXP / "data").mkdir(exist_ok=True)
    tasks = T.load_mbpp(split="test", limit=args.tasks)
    print(f"{len(tasks)} tasks, k={args.k}, conditions={conds}", flush=True)
    (EXP / "data" / "tasks.json").write_text(json.dumps(
        {t.task_id: {"test_list": t.test_list, "test_imports": t.test_imports} for t in tasks}))

    import probe_lib as PL  # late import
    p = PL.Probe()
    print(f"model loaded in {p.load_secs:.0f}s, {p.n_layers} layers", flush=True)

    def user_prompt(t):
        anchor = t.test_list[0] if t.test_list else ""
        return (f"{t.prompt}\n\nYour function must satisfy this example:\n{anchor}\n"
                f"Define the function with the exact name used above.")

    rec_f = (EXP / "data" / "records.jsonl").open("w")
    wall0 = time.time()
    for cond in conds:
        cfg = CONDS[cond]
        t0 = time.time()
        prompts = [p.prompt(user_prompt(t), enable_thinking=cfg["think"]) for t in tasks]
        rep = [pr for pr in prompts for _ in range(args.k)]
        gens = p.gen_sequences(rep, think=cfg["think"], budget=cfg["budget"], shuffle=cfg["shuffle"],
                               batch_size=batch_for(cfg))
        seqs = [g["seq_ids"] for g in gens]
        acts = p.activations(seqs, batch_size=8)  # [N, L+1, H] float16 (small batch: long seqs)
        np.save(ACTS_DIR / f"acts_{cond}.npy", acts)
        for i, t in enumerate(tasks):
            for s in range(args.k):
                idx = i * args.k + s
                g = gens[idx]
                code = T.extract_code(p.tok.decode(g["seq_ids"], skip_special_tokens=False))
                rec_f.write(json.dumps({
                    "cond": cond, "task_id": t.task_id, "sample": s, "row": idx,
                    "code": code, "n_think": g["n_think"], "forced": g["forced"],
                    "seq_len": len(g["seq_ids"])}) + "\n")
        rec_f.flush()
        print(f"  [{cond}] {acts.shape} acts, mean_think={np.mean([g['n_think'] for g in gens]):.0f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    rec_f.close()
    print(f"done in {time.time()-wall0:.0f}s; acts in {ACTS_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
