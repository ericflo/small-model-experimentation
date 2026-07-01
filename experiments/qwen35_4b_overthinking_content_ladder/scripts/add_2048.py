#!/usr/bin/env python3
"""Recover the budget-2048 ladder (conservative batches) and APPEND to records.jsonl.

The full sweep crashed mid-2048 (CUDA 'device not ready' in the fla kernel: answer-regen over
~2000-token thinking prefixes at batch 48). This regenerates real/shuffle/foreign/filler at 2048
with small batches.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import tasks as T  # noqa: E402

BUDGET = 2048


def user_prompt(t):
    anchor = t.test_list[0] if t.test_list else ""
    return (f"{t.prompt}\n\nYour function must satisfy this example:\n{anchor}\n"
            f"Define the function with the exact name used above.")


def main() -> int:
    recs = [json.loads(l) for l in (EXP / "data" / "records.jsonl").read_text().splitlines() if l.strip()]
    if any(r["budget"] == BUDGET for r in recs):
        print("budget-2048 records already present; remove them first"); return 0
    tasks = T.load_mbpp(split="test", limit=100)
    n, k = len(tasks), 8

    import ladder_lib as LL
    p = LL.Probe()
    print(f"model loaded in {p.load_secs:.0f}s", flush=True)
    dot = p.tok(".", add_special_tokens=False).input_ids[-1]
    rep = [p.prompt(user_prompt(t), enable_thinking=True) for t in tasks for _ in range(k)]
    prompt_ids = [p._ids(x) for x in rep]
    for_src = [(((i // k + 1) % n) * k + i % k) for i in range(len(rep))]

    t0 = time.time()
    real = p.gen_real(rep, budget=BUDGET, batch_size=20)
    print(f"  real done, mean_think={sum(r['n_think'] for r in real)/len(real):.0f} [{time.time()-t0:.0f}s]", flush=True)
    sh_pre = []
    for i, r in enumerate(real):
        tt = r["think_tokens"]
        sh = [tt[j] for j in torch.randperm(len(tt)).tolist()] if tt else []
        sh_pre.append(prompt_ids[i] + sh + p.close_ids)
    shuffle = p.gen_answer(sh_pre, batch_size=12)
    foreign = p.gen_answer([prompt_ids[i] + real[for_src[i]]["think_tokens"] + p.close_ids for i in range(len(real))], batch_size=12)
    filler = p.gen_answer([prompt_ids[i] + [dot] * real[i]["n_think"] + p.close_ids for i in range(len(real))], batch_size=12)
    print(f"  shuffle/foreign/filler done [{time.time()-t0:.0f}s]", flush=True)

    conds = {"real": real, "shuffle": shuffle, "foreign": foreign, "filler": filler}
    with (EXP / "data" / "records.jsonl").open("a") as f:
        for cond, gens in conds.items():
            for i, t in enumerate(tasks):
                for s in range(k):
                    idx = i * k + s
                    code = T.extract_code(p.tok.decode(gens[idx]["seq_ids"], skip_special_tokens=False))
                    f.write(json.dumps({"cond": cond, "budget": BUDGET, "task_id": t.task_id,
                                        "sample": s, "code": code}) + "\n")
    print("appended budget-2048 records", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
