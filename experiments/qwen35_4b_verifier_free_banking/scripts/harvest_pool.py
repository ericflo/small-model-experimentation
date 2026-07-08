#!/usr/bin/env python3
"""Harvest ONE shared candidate pool for all banking arms: sample K think-mode completions/task on TRAIN
tasks (C18-identical sampling), then annotate every unique (task, code) candidate with BOTH the oracle
(execution grade -- used only by the ceiling arm + post-hoc reporting) and the model's own P(True) judge
readout (no execution). Arm construction happens downstream in build_arms.py so every arm filters the
SAME pool (sampling noise shared). Prints a P(True)-vs-full_pass AUROC canary."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP / "src"))
import common as C  # noqa: E402
import families as FAM  # noqa: E402
import code_env as E  # noqa: E402


def auroc(s, y):
    pairs = sorted(zip(s, y))
    n1 = sum(y); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(1 for k in range(i, j) if pairs[k][1])
        i = j
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    depth_tasks = [(1, 20), (2, 35), (3, 35)]  # C18-identical schedule
    suf = ""
    if args.smoke:
        depth_tasks, args.K, suf = [(1, 2), (2, 2)], 8, "_smoke"  # separate paths: smoke must never poison full

    fam = FAM.FAMILIES["list"]
    rng = random.Random(args.seed)
    tasks = C.gen_tasks(fam, depth_tasks, rng, start_id=0)
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / f"train_tasks{suf}.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    print(f"[pool] {len(tasks)} train tasks, K={args.K}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    t0 = time.time()
    prompts = [p.prompt(C.ident_prompt(fam, t), enable_thinking=True) for t in tasks]
    rep = [pr for pr in prompts for _ in range(args.K)]
    gens = p.gen_sequences(rep, think=True, budget=args.budget, greedy=False, answer_max=420, batch_size=48)
    print(f"[pool] sampling done [{time.time()-t0:.0f}s]", flush=True)

    codes_per = []
    for ti in range(len(tasks)):
        cs = []
        for j in range(args.K):
            idx = ti * args.K + j
            txt = p.tok.decode(gens[idx]["seq_ids"][len(p._ids(rep[idx])):], skip_special_tokens=False)
            txt = txt.split("</think>")[-1] if "</think>" in txt else txt
            c, _ = E.extract_candidate_code(txt, "transform")
            cs.append(c or "")
        codes_per.append(cs)

    # oracle grade every unique candidate (thread pool) -- ceiling arm + post-hoc reporting only
    keys = [(ti, c) for ti in range(len(tasks)) for c in sorted(set(codes_per[ti])) if c]
    with ThreadPoolExecutor(max_workers=16) as ex:
        res = dict(zip(keys, ex.map(lambda k: C.grade(k[1], tasks[k[0]]), keys)))
    print(f"[pool] graded {len(keys)} unique [{time.time()-t0:.0f}s]", flush=True)

    # P(True) judge every unique candidate (no execution) -- the verifier-free signal
    jp = [p.judge_prompt(C.ident_prompt(fam, tasks[ti]), c, enable_thinking=False) for ti, c in keys]
    ptrue = p.judge_nothink(jp, batch_size=16)
    pt = {k: v for k, v in zip(keys, ptrue)}
    print(f"[pool] judged {len(keys)} unique [{time.time()-t0:.0f}s]", flush=True)

    freq_per = [Counter(cs) for cs in codes_per]
    pool = []
    for ti, c in keys:
        vis, full, _sig = res[(ti, c)]
        pool.append({"ti": ti, "task_id": tasks[ti]["task_id"], "depth": tasks[ti]["depth"], "code": c,
                     "freq": freq_per[ti][c], "visible_pass": bool(vis), "full_pass": bool(full),
                     "p_true": round(float(pt[(ti, c)]), 6)})
    (EXP / "data" / f"pool{suf}.jsonl").write_text("\n".join(json.dumps(x) for x in pool) + "\n")

    ys = [int(x["full_pass"]) for x in pool]
    ss = [x["p_true"] for x in pool]
    a = auroc(ss, ys)
    by_d = Counter((x["depth"], x["full_pass"]) for x in pool)
    n_solved = len({x["ti"] for x in pool if x["full_pass"]})
    print(f"[pool] {len(pool)} candidates | pass-rate {sum(ys)/max(1,len(ys)):.2f} | solved tasks "
          f"{n_solved}/{len(tasks)} | CANARY pooled AUROC P(True) vs full_pass = "
          f"{a if a is None else round(a,3)} | by (depth,pass): {dict(by_d)}", flush=True)
    print(f"wrote data/pool{suf}.jsonl")


if __name__ == "__main__":
    main()
