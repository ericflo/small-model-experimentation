#!/usr/bin/env python3
"""'Be your own tool-search': model-guided beam search over the model's OWN op-proposals, no training,
no brute-enumeration. At each step the fixed model ranks the 32 DSL ops by likelihood given (current
lists -> goal lists); the interpreter applies the top-k and verifies partial/final states. Tests whether
depth-3 composition is LATENT (elicitable at test time) vs a true capability gap.

mode='model'  : proposals = model's top-k likelihood-ranked ops (the latent search heuristic)
mode='random' : proposals = k random ops (ablation: isolates the model's contribution from beam+pruning)
mode='oracle' : proposals = all ops (= brute-search upper bound at this beam)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import families as FAM  # noqa: E402
import gen_lib as GL  # noqa: E402

fam = FAM.FAMILIES["list"]
OPS = FAM.all_ops(fam)                              # 32 (op, param)
OP_REPRS = [FAM.op_repr(o, p) for (o, p) in OPS]

DSL_DESC = (
    "You transform a list of integers by applying operations one at a time. Operations:\n"
    "  reverse            reverse the list\n"
    "  sort_asc           sort ascending\n"
    "  sort_desc          sort descending\n"
    "  unique_stable      drop later duplicates, keep first occurrence order\n"
    "  dedup_adjacent     collapse runs of equal adjacent elements\n"
    "  abs_all            absolute value of each element\n"
    "  square             square each element\n"
    "  negate             negate each element\n"
    "  running_sum        running cumulative sum\n"
    "  adjacent_diff      differences of adjacent elements (length shrinks by 1)\n"
    "  add_k(k)           add k to each element, k in {-3,-2,-1,1,2,3}\n"
    "  mul_k(k)           multiply each element by k, k in {-2,2,3}\n"
    "  mod_k(k)           each element mod k (result in 0..k-1), k in {2,3,4}\n"
    "  take_k(k)          keep the first k elements, k in {1,2,3,4}\n"
    "  drop_k(k)          drop the first k elements, k in {1,2,3,4}\n"
    "  rotate_k(k)        rotate left by k, k in {1,2,3}\n"
)


def propose_prompt(states, target, n_show=4):
    lines = [DSL_DESC, "Apply operations to move each list toward its goal.", ""]
    lines.append("Current list -> goal list:")
    for s, t in list(zip(states, target))[:n_show]:
        lines.append(f"  {s} -> {t}")
    lines.append("")
    lines.append("What is the single next operation to apply now? Answer with only the operation name, "
                 "e.g. sort_asc or add_k(2).")
    return "\n".join(lines)


@torch.no_grad()
def score_ops(probe, user_prompt):
    """Length-normalized logprob of each of the 32 op reprs as the assistant answer. Batched, one forward."""
    ptext = probe.prompt(user_prompt, enable_thinking=False)
    pids = probe._ids(ptext)
    seqs, clens = [], []
    for r in OP_REPRS:
        cids = probe.tok(r, add_special_tokens=False).input_ids
        seqs.append(pids + cids); clens.append(len(cids))
    maxlen = max(len(s) for s in seqs)
    max_cl = max(clens)
    pad = probe.tok.pad_token_id
    inp = torch.tensor([[pad] * (maxlen - len(s)) + s for s in seqs], device=probe.device)  # left pad, [B, L]
    attn = (inp != pad).long()
    logits = probe.model(input_ids=inp, attention_mask=attn).logits  # [B, L, V] bf16
    # completion tokens are the last cl (right-aligned); logits at position p-1 predict token p.
    # Only softmax the max_cl positions that can be completion tokens -> small [B, max_cl, V] float. Vectorized.
    sl = logits[:, maxlen - max_cl - 1:maxlen - 1].float()            # [B, max_cl, V]
    logp = torch.log_softmax(sl, dim=-1)
    tgt = inp[:, maxlen - max_cl:maxlen]                              # [B, max_cl]
    lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)              # [B, max_cl] logprob of each token
    mask = torch.zeros(inp.size(0), max_cl, device=probe.device)
    for b, cl in enumerate(clens):
        mask[b, max_cl - cl:] = 1.0                                   # keep only real completion positions
    return ((lp * mask).sum(-1) / mask.sum(-1)).tolist()             # length-normalized, ONE sync


def apply_seq_states(states, op, p):
    out = []
    for s in states:
        r = FAM.apply_op(fam, op, p, list(s))
        if r is None:
            return None
        out.append(r)
    return out


def distance(states, target):
    d = 0
    for s, t in zip(states, target):
        if len(s) != len(t):
            d += max(len(s), len(t)) * 4 + abs(len(s) - len(t))
        else:
            d += sum(1 for a, b in zip(s, t) if a != b)
    return d


def solved(states, target):
    return all(list(s) == list(t) for s, t in zip(states, target))


def verify_hidden(op_seq, hidden):
    for ex in hidden:
        s = list(ex["input"])
        for (op, p) in op_seq:
            s = FAM.apply_op(fam, op, p, s)
            if s is None:
                return False
        if s != list(ex["output"]):
            return False
    return True


def model_search(probe, task, *, k=6, beam=4, max_depth=3, mode="model", rng=None, pruned=True):
    """Model-guided (or brute/random) beam search. Termination uses VISIBLE examples ONLY (no peeking at
    hidden); the returned op_seq's generalization is graded on hidden by the caller.
    Returns dict: {seq (visible-solving, or None), n_fwd (model forward passes), n_interp (op applications)}.
    """
    vis = task["visible"]
    inputs = [list(ex["input"]) for ex in vis]
    target = [list(ex["output"]) for ex in vis]
    frontier = [([], inputs)]
    n_fwd = 0
    n_interp = 0
    for _ in range(max_depth):
        cands = []
        for (seq, states) in frontier:
            if mode == "model":
                sc = score_ops(probe, propose_prompt(states, target)); n_fwd += 1
                idx = sorted(range(len(OPS)), key=lambda i: -sc[i])[:k]
            elif mode == "random":
                idx = rng.sample(range(len(OPS)), k)
            else:  # brute: all ops (budget-matched control -- the honesty bar)
                idx = list(range(len(OPS)))
            for i in idx:
                op, p = OPS[i]
                ns = apply_seq_states(states, op, p); n_interp += len(inputs)
                if ns is None:
                    continue
                cands.append((seq + [(op, p)], ns))
        # VISIBLE-only solution check (no hidden peeking)
        for (seq, states) in cands:
            if solved(states, target):
                return {"seq": seq, "n_fwd": n_fwd, "n_interp": n_interp}
        # prune to beam (distance-to-target) unless ablating the pruning
        if pruned:
            cands.sort(key=lambda sc: distance(sc[1], target))
            frontier = cands[:beam]
        else:
            frontier = cands[:128]  # unpruned ablation: keep frontier in arbitrary order (no distance oracle), capped
        if not frontier:
            break
    return {"seq": None, "n_fwd": n_fwd, "n_interp": n_interp}


if __name__ == "__main__":  # smoke: model vs brute vs random on a few held-out depth-3 tasks (hidden-graded)
    import json
    import time
    tasks = [json.loads(l) for l in (EXP / "data" / "eval_frozen_d3.jsonl").read_text().splitlines() if l.strip()][:8]
    probe = GL.Probe()
    rng = random.Random(0)
    for mode in ("model", "brute", "random"):
        t0 = time.time(); vis_n = 0; hid_n = 0; fwd = 0; interp = 0
        for t in tasks:
            r = model_search(probe, t, k=6, beam=4, max_depth=3, mode=mode, rng=rng)
            fwd += r["n_fwd"]; interp += r["n_interp"]
            if r["seq"] is not None:
                vis_n += 1
                if verify_hidden(r["seq"], t["hidden"]):
                    hid_n += 1
        print(f"[{mode}] visible-solved {vis_n}/{len(tasks)} hidden-generalizing {hid_n}/{len(tasks)} | "
              f"{fwd} fwd {interp} interp | {time.time()-t0:.0f}s", flush=True)
