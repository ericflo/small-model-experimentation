#!/usr/bin/env python3
"""Overthinking content ladder: run the no_think/filler/shuffle/foreign/real ladder across thinking
budgets and ask whether the coherent-content advantage (real - shuffle) shrinks as the budget grows.

Behavioral-only (no activations). Real thinking generated once per budget (capturing thinking tokens);
filler/shuffle/foreign reuse those tokens / matched length and regenerate only the answer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import tasks as T  # noqa: E402


def user_prompt(t):
    anchor = t.test_list[0] if t.test_list else ""
    return (f"{t.prompt}\n\nYour function must satisfy this example:\n{anchor}\n"
            f"Define the function with the exact name used above.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budgets", default="512,1024,2048")
    args = ap.parse_args()
    if args.smoke:
        args.tasks, args.k, args.budgets = 4, 2, "512"
    budgets = [int(b) for b in args.budgets.split(",")]

    (EXP / "data").mkdir(exist_ok=True)
    tasks = T.load_mbpp(split="test", limit=args.tasks)
    n, k = len(tasks), args.k
    print(f"{n} tasks, k={k}, budgets={budgets}", flush=True)
    (EXP / "data" / "tasks.json").write_text(json.dumps(
        {t.task_id: {"test_list": t.test_list, "test_imports": t.test_imports} for t in tasks}))

    import ladder_lib as LL
    p = LL.Probe()
    print(f"model loaded in {p.load_secs:.0f}s", flush=True)
    dot = p.tok(".", add_special_tokens=False).input_ids[-1]
    rec_f = (EXP / "data" / "records.jsonl").open("w")
    wall0 = time.time()

    def write(cond, budget, seqs):
        for i, t in enumerate(tasks):
            for s in range(k):
                idx = i * k + s
                code = T.extract_code(p.tok.decode(seqs[idx], skip_special_tokens=False))
                rec_f.write(json.dumps({"cond": cond, "budget": budget, "task_id": t.task_id,
                                        "sample": s, "code": code}) + "\n")
        rec_f.flush()

    # no_think once (budget-independent)
    nt = p.gen_sequences([p.prompt(user_prompt(t), enable_thinking=False) for t in tasks for _ in range(k)],
                         think=False, budget=None, batch_size=64)
    write("no_think", 0, [g["seq_ids"] for g in nt])
    print(f"  [no_think] done [{time.time()-wall0:.0f}s]", flush=True)

    think_prompts = [p.prompt(user_prompt(t), enable_thinking=True) for t in tasks]
    rep = [tp for tp in think_prompts for _ in range(k)]
    prompt_ids = [p._ids(x) for x in rep]
    for_src = [(((i // k + 1) % n) * k + i % k) for i in range(len(rep))]

    for budget in budgets:
        t0 = time.time()
        # answer-regen forwards over the whole thinking prefix, so long budgets need small batches
        # (batch 48 over ~2000-token prefixes triggers a CUDA "device not ready" in the fla kernel).
        gbs = 48 if budget <= 1024 else 24
        abs_ = 48 if budget <= 512 else (32 if budget <= 1024 else 12)
        real = p.gen_real(rep, budget=budget, batch_size=gbs)
        write("real", budget, [r["seq_ids"] for r in real])
        # shuffle
        sh_pre = []
        for i, r in enumerate(real):
            tt = r["think_tokens"]
            sh = [tt[j] for j in torch.randperm(len(tt)).tolist()] if tt else []
            sh_pre.append(prompt_ids[i] + sh + p.close_ids)
        write("shuffle", budget, [g["seq_ids"] for g in p.gen_answer(sh_pre, batch_size=abs_)])
        # foreign
        fo_pre = [prompt_ids[i] + real[for_src[i]]["think_tokens"] + p.close_ids for i in range(len(real))]
        write("foreign", budget, [g["seq_ids"] for g in p.gen_answer(fo_pre, batch_size=abs_)])
        # filler (contentless dots, matched to real thinking length)
        fi_pre = [prompt_ids[i] + [dot] * real[i]["n_think"] + p.close_ids for i in range(len(real))]
        write("filler", budget, [g["seq_ids"] for g in p.gen_answer(fi_pre, batch_size=abs_)])
        mt = sum(r["n_think"] for r in real) / len(real)
        print(f"  [budget {budget}] real/shuffle/foreign/filler done, mean_think={mt:.0f} [{time.time()-t0:.0f}s]", flush=True)

    rec_f.close()
    print(f"done in {time.time()-wall0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
