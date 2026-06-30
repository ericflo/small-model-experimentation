#!/usr/bin/env python3
"""Foreign-thinking ladder: decompose the thinking gain into compute/scaffold vs token-presence
vs coherent order, behaviorally AND in answer-token separability.

Ladder (all at a fixed thinking budget): no_think | foreign | shuffle | real.
- real:    the model's own coherent thinking for THIS task.
- shuffle: the same thinking tokens, permuted (destroys order, keeps relevance + count + scaffold).
- foreign: a DIFFERENT task's thinking spliced in (destroys relevance, keeps count + scaffold + compute).
- no_think: no thinking at all.
Real thinking is generated once (capturing its thinking tokens); shuffle/foreign reuse those tokens
and regenerate only the answer, so the conditions share the same thinking-token multiset.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import tasks as T  # noqa: E402

ACTS_DIR = EXP.parents[1] / "large_artifacts" / "qwen35_4b_thinking_content_vs_compute"


def user_prompt(t):
    anchor = t.test_list[0] if t.test_list else ""
    return (f"{t.prompt}\n\nYour function must satisfy this example:\n{anchor}\n"
            f"Define the function with the exact name used above.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=int, default=512)
    args = ap.parse_args()
    if args.smoke:
        args.tasks, args.k = 4, 2

    ACTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXP / "data").mkdir(exist_ok=True)
    tasks = T.load_mbpp(split="test", limit=args.tasks)
    n, k = len(tasks), args.k
    print(f"{n} tasks, k={k}, budget={args.budget}", flush=True)
    (EXP / "data" / "tasks.json").write_text(json.dumps(
        {t.task_id: {"test_list": t.test_list, "test_imports": t.test_imports} for t in tasks}))

    import ladder_lib as LL
    p = LL.Probe()
    print(f"model loaded in {p.load_secs:.0f}s, {p.n_layers} layers", flush=True)
    wall0 = time.time()

    think_prompts = [p.prompt(user_prompt(t), enable_thinking=True) for t in tasks]
    rep = [tp for tp in think_prompts for _ in range(k)]
    prompt_ids = [p._ids(x) for x in rep]

    # real thinking (captures thinking tokens)
    t0 = time.time()
    real = p.gen_real(rep, budget=args.budget, batch_size=48)
    print(f"  [real] generated, mean_think={np.mean([r['n_think'] for r in real]):.0f} [{time.time()-t0:.0f}s]", flush=True)

    # shuffle: permute each item's own thinking tokens
    shuf_prefix = []
    for i, r in enumerate(real):
        tt = r["think_tokens"]
        sh = [tt[j] for j in torch.randperm(len(tt)).tolist()] if tt else []
        shuf_prefix.append(prompt_ids[i] + sh + p.close_ids)
    t0 = time.time(); shuffle = p.gen_answer(shuf_prefix, batch_size=48)
    print(f"  [shuffle] answers regenerated [{time.time()-t0:.0f}s]", flush=True)

    # foreign: cyclic task shift (task t -> task (t+1)%n), same sample slot
    for_src = [(((i // k + 1) % n) * k + i % k) for i in range(len(real))]
    for_prefix = [prompt_ids[i] + real[for_src[i]]["think_tokens"] + p.close_ids for i in range(len(real))]
    t0 = time.time(); foreign = p.gen_answer(for_prefix, batch_size=48)
    print(f"  [foreign] answers regenerated [{time.time()-t0:.0f}s]", flush=True)

    # no_think
    nt_rep = [p.prompt(user_prompt(t), enable_thinking=False) for t in tasks for _ in range(k)]
    t0 = time.time(); no_think = p.gen_sequences(nt_rep, think=False, budget=None, batch_size=64)
    print(f"  [no_think] generated [{time.time()-t0:.0f}s]", flush=True)

    conds = {"no_think": [g["seq_ids"] for g in no_think],
             "foreign": [g["seq_ids"] for g in foreign],
             "shuffle": [g["seq_ids"] for g in shuffle],
             "real": [g["seq_ids"] for g in real]}
    n_think = {"no_think": [0] * len(real),
               "foreign": [real[for_src[i]]["n_think"] for i in range(len(real))],
               "shuffle": [real[i]["n_think"] for i in range(len(real))],
               "real": [real[i]["n_think"] for i in range(len(real))]}

    rec_f = (EXP / "data" / "records.jsonl").open("w")
    for cond, seqs in conds.items():
        t0 = time.time()
        acts = p.activations(seqs, batch_size=8)
        np.save(ACTS_DIR / f"acts_{cond}.npy", acts)
        for i, t in enumerate(tasks):
            for s in range(k):
                idx = i * k + s
                code = T.extract_code(p.tok.decode(seqs[idx], skip_special_tokens=False))
                rec_f.write(json.dumps({"cond": cond, "task_id": t.task_id, "sample": s, "row": idx,
                                        "code": code, "n_think": n_think[cond][idx],
                                        "seq_len": len(seqs[idx])}) + "\n")
        rec_f.flush()
        print(f"  [{cond}] acts {acts.shape} [{time.time()-t0:.0f}s]", flush=True)
    rec_f.close()
    print(f"done in {time.time()-wall0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
