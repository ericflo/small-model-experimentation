#!/usr/bin/env python3
"""Harvest the model's OWN (correct, incorrect) depth-3 samples for preference training. Sample banked_1280
(has coverage) k times no-think per fresh depth-3 task (disjoint from the frozen held-out); keep tasks that
produced BOTH a verified-correct and a verified-wrong program. Emit pairs.jsonl {prompt, chosen, rejected}
(model's own correct vs its own plausible-wrong) and sft.jsonl {prompt, code} (the same correct, for the SFT
arm) -- matched chosen data so DPO's only extra signal is the negative."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP / "src"))
import common as C  # noqa: E402
import families as FAM  # noqa: E402
import gen_lib as GL  # noqa: E402
import code_env as E  # noqa: E402

fam = FAM.FAMILIES["list"]
PROBE = [[((i * 7 + j * 5) % 19) - 9 for j in range(4 + (i % 5))] for i in range(24)]


def parse_repr(r):
    if "(" in r:
        return (r.split("(")[0], int(r[r.index("(") + 1:-1]))
    return (r, None)


def fsig(ops):
    out = []
    for x in PROBE:
        st = list(x)
        for op, k in ops:
            st = FAM.apply_op(fam, op, k, st)
            if st is None:
                break
        out.append(repr(st))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=303)
    args = ap.parse_args()

    heldout = [json.loads(l) for l in (EXP / "data" / "eval_frozen_d3.jsonl").read_text().splitlines() if l.strip()]
    excl_f = {fsig([parse_repr(r) for r in t["target_ops"]]) for t in heldout}
    excl_o = {tuple(t["target_ops"]) for t in heldout}

    rng = random.Random(args.seed); tid = 500_000; tasks = []
    while len(tasks) < args.pool and tid < 500_000 + 400_000:
        tid += 1
        t = FAM.make_task(fam, tid, 3, rng, k_visible=8, m_hidden=8)
        if not t or fsig([parse_repr(r) for r in t["target_ops"]]) in excl_f or tuple(t["target_ops"]) in excl_o:
            continue
        tasks.append(t)
    print(f"[pairs] {len(tasks)} fresh true-depth-3 tasks (disjoint from held-out)", flush=True)

    probe = GL.Probe()
    from peft import PeftModel
    probe.model = PeftModel.from_pretrained(probe.model, args.adapter).eval()
    print(f"[pairs] loaded {args.adapter}; sampling k={args.k} no-think", flush=True)

    # k no-think samples per task (flatten to prompts, batched)
    flat_prompts, flat_task = [], []
    for ti, t in enumerate(tasks):
        pt = probe.prompt(C.ident_prompt(fam, t), enable_thinking=False)
        for _ in range(args.k):
            flat_prompts.append(pt); flat_task.append(ti)
    outs = probe.gen_sequences(flat_prompts, think=False, budget=None, greedy=False, batch_size=48)

    correct = [[] for _ in tasks]
    wrong = [[] for _ in tasks]
    for ti, pt, out in zip(flat_task, flat_prompts, outs):
        ans = probe.tok.decode(out["seq_ids"][len(probe._ids(pt)):]).strip()
        code, _ = E.extract_candidate_code(ans, "transform")
        if not code:
            continue
        _, full, _ = C.grade(code, tasks[ti])
        (correct if full else wrong)[ti].append(code)

    pairs, sft = [], []
    for ti, t in enumerate(tasks):
        cs = list(dict.fromkeys(correct[ti]))  # dedup, keep order
        ws = list(dict.fromkeys(wrong[ti]))
        if cs and ws:
            prompt = C.ident_prompt(fam, t)
            pairs.append({"prompt": prompt, "chosen": cs[0], "rejected": ws[0], "depth": 3})
            sft.append({"prompt": prompt, "code": cs[0], "depth": 3})
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "pairs.jsonl").write_text("\n".join(json.dumps(x) for x in pairs) + "\n")
    (EXP / "data" / "sft.jsonl").write_text("\n".join(json.dumps(x) for x in sft) + "\n")
    n_solved = sum(1 for c in correct if c)
    print(f"[pairs] {n_solved}/{len(tasks)} tasks had >=1 correct sample; {len(pairs)} (chosen,rejected) pairs kept", flush=True)


if __name__ == "__main__":
    main()
