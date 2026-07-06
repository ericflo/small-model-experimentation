#!/usr/bin/env python3
"""Can THINKING breach the lookahead wall (C25)? Channel-matched test: think for B tokens, close </think>,
then run the SAME 32-way likelihood ranking C25 used (parse-immune, directly comparable to base 0.013).
Headline = STEP 1 (goal 3 ops away, no intermediate state materialized -- the only clean lookahead test)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP / "src"))
import families as FAM  # noqa: E402
import gen_lib as GL  # noqa: E402
import decompose as D  # noqa: E402

fam = FAM.FAMILIES["list"]


@torch.no_grad()
def score_ops_prefix(probe, prefix_ids, op_reprs=D.OP_REPRS, chunk=4):
    """Length-normalized logprob of each op repr as the next answer, given an arbitrary token prefix.
    Chunked over ops: with a long (thinking) prefix the full-vocab logits tensor is huge, so score a few
    ops per forward to stay in memory."""
    pad = probe.tok.pad_token_id
    op_cids = [probe.tok(r, add_special_tokens=False).input_ids for r in op_reprs]
    scores = []
    for c0 in range(0, len(op_reprs), chunk):
        cids_chunk = op_cids[c0:c0 + chunk]
        seqs = [prefix_ids + c for c in cids_chunk]
        clens = [len(c) for c in cids_chunk]
        maxlen = max(len(s) for s in seqs)
        max_cl = max(clens)
        inp = torch.tensor([[pad] * (maxlen - len(s)) + s for s in seqs], device=probe.device)
        attn = (inp != pad).long()
        logits = probe.model(input_ids=inp, attention_mask=attn).logits
        sl = logits[:, maxlen - max_cl - 1:maxlen - 1].float()
        logp = torch.log_softmax(sl, dim=-1)
        tgt = inp[:, maxlen - max_cl:maxlen]
        lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = torch.zeros(inp.size(0), max_cl, device=probe.device)
        for b, cl in enumerate(clens):
            mask[b, max_cl - cl:] = 1.0
        scores.extend(((lp * mask).sum(-1) / mask.sum(-1)).tolist())
        del logits, sl, logp
    return scores


def think_then_rank(probe, user_prompt, budget, op_reprs=D.OP_REPRS):
    """budget==0: no-think ranking (C25). budget>0: generate a thinking trace of <=budget tokens, close
    </think>, then rank the ops given prompt+thinking. Returns (scores, thinking_text)."""
    if budget == 0:
        ptext = probe.prompt(user_prompt, enable_thinking=False)
        return score_ops_prefix(probe, probe._ids(ptext), op_reprs), ""
    ptext = probe.prompt(user_prompt, enable_thinking=True)
    ptext_ids = probe._ids(ptext)
    out = probe.gen_sequences([ptext], think=True, budget=budget, batch_size=8)[0]
    gen = out["seq_ids"][len(ptext_ids):]
    n_think = out["n_think"]
    prefix_ids = ptext_ids + gen[:n_think] + probe.close_ids
    thinking = probe.tok.decode(gen[:n_think])
    return score_ops_prefix(probe, prefix_ids, op_reprs), thinking


def parse_repr(r):
    if "(" in r:
        name, k = r.split("(")[0], int(r[r.index("(") + 1:-1])
    else:
        name, k = r, None
    return D.OP_REPRS.index(FAM.op_repr(name, k))


def gt_ops(task):
    return [parse_repr(r) for r in task["target_ops"]]


def states_at_step(task, step):
    """The true intermediate lists just before applying op `step` (1-indexed). step==1 -> raw inputs."""
    states = [list(ex["input"]) for ex in task["visible"]]
    for gt in gt_ops(task)[:step - 1]:
        op, p = D.OPS[gt]
        states = [FAM.apply_op(fam, op, p, s) for s in states]
    return states, [list(ex["output"]) for ex in task["visible"]]


if __name__ == "__main__":  # smoke: step-1 no-think-rank vs think-rank@1024 on 20 tasks
    import json
    import time
    tasks = [json.loads(l) for l in (EXP / "data" / "eval_frozen_d3.jsonl").read_text().splitlines() if l.strip()][:20]
    probe = GL.Probe()
    for budget in (0, 1024):
        t0 = time.time(); hit = 0
        for t in tasks:
            states, target = states_at_step(t, 1)  # step 1: raw inputs, goal 3 ops away
            sc, _ = think_then_rank(probe, D.propose_prompt(states, target), budget)
            rank1 = max(range(len(sc)), key=lambda i: sc[i])
            if rank1 == gt_ops(t)[0]:
                hit += 1
        print(f"[budget={budget}] STEP-1 top1 {hit/len(tasks):.3f} ({hit}/{len(tasks)}) | {time.time()-t0:.0f}s", flush=True)
