#!/usr/bin/env python3
"""C42 targeted-repair test: at a located single slip, is the CORRECT digit
reachable, and can a verifier-free move recover it? Reproduces the greedy
scaffolded decode (familiar +k chains), finds single-slip chains, and at the slip
step L measures: (1) the RANK of the correct successor d* in the model's step-L
distribution (is the fix 'almost known'?); (2) temperature-resample recovery (does
sampling n=8 from the step-L dist surface d*?); (3) confidence-select (is d* the
argmax? = the no-op control, must be wrong by construction); (4) ISOLATED
re-computation (fresh one-step prompt 'prev moved k forward = ?' -> does the model
get d* out of the flawed chain context?). Reports fix rates vs the oracle ceiling.
Fast: toy digit chains, HF, one forward pass per step. Run under .venv."""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts")); sys.path.insert(0, str(EXP / "src"))
import chain_family as CF  # noqa
import gen_lib as GL  # noqa
DIGIT_IDS = [15 + d for d in range(10)]
CONDS = [("familiar", 4), ("familiar", 5), ("familiar", 6), ("familiar", 7)]


@torch.no_grad()
def digit_dists(p, prefixes, bs=64):
    out = [None] * len(prefixes); pad = p.tok.pad_token_id
    order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
    for s in range(0, len(order), bs):
        sub = order[s:s+bs]; seqs = [prefixes[i] for i in sub]; ml = max(len(x) for x in seqs)
        ids = torch.tensor([[pad]*(ml-len(x))+x for x in seqs], device=p.device)
        o = p.model(input_ids=ids, attention_mask=(ids != pad).long(), logits_to_keep=1)
        pr = torch.softmax(o.logits[:, -1, DIGIT_IDS].float(), dim=1).cpu().tolist()
        for i, v in zip(sub, pr): out[i] = v
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-per-cond", type=int, default=140)
    ap.add_argument("--seed", type=int, default=800); a = ap.parse_args()
    p = GL.Probe()
    tasks = [CF.gen_chain(k, d, a.seed*100000 + ci*1000 + i)
             for ci, (k, d) in enumerate(CONDS) for i in range(a.n_per_cond)]
    nl = lambda s: p.tok(s, add_special_tokens=False).input_ids
    prefixes = [p._ids(p.prompt(CF.render(t), enable_thinking=False)) + nl("Step 1: ") for t in tasks]
    prev = [t["start"] for t in tasks]
    steps = [[] for _ in tasks]  # per task: list of {val, dist, prev}
    maxd = max(t["depth"] for t in tasks)
    for i in range(1, maxd+1):
        active = [j for j, t in enumerate(tasks) if i <= t["depth"]]
        dists = digit_dists(p, [prefixes[j] for j in active])
        for j, dist in zip(active, dists):
            d = max(range(10), key=lambda x: dist[x]); val = str(d)
            steps[j].append({"val": val, "dist": dist, "prev": prev[j]})
            prev[j] = val
            t = tasks[j]
            prefixes[j] = prefixes[j] + nl(f"{val}\nStep {i+1}: " if i < t["depth"] else f"{val}")

    # single-slip chains
    slips = []  # (task, slip_index0, correct_digit)
    for t, st in zip(tasks, steps):
        errs = [i for i, s in enumerate(st) if s["val"] != t["nxt"].get(s["prev"], "")]
        if len(errs) == 1:
            L = errs[0]; slips.append((t, L, t["nxt"][st[L]["prev"]], st))
    print(f"[repair] {len(slips)} single-slip chains of {len(tasks)}", flush=True)

    ranks, avail8, confsel, greedy_ok = [], [], [], []
    for t, L, dstar, st in slips:
        dist = st[L]["dist"]
        order = sorted(range(10), key=lambda x: -dist[x])
        rank = order.index(int(dstar)) + 1        # 1 = argmax (the wrong greedy)
        ranks.append(rank)
        greedy_ok.append(rank == 1)               # 0 by construction (slip)
        # temperature-resample recovery: prob at least one of 8 samples == d* (T=1.0 sampling from dist)
        pstar = dist[int(dstar)]
        avail8.append(1 - (1 - pstar) ** 8)
        confsel.append(rank == 1)                 # highest-conf pick = argmax = wrong (no-op)

    # isolated re-computation: a depth-1 chain from the slip's prev, using the
    # EXACT same "Step 1: " scaffold that works in the chain (not an abstract
    # re-phrasing, which the model mis-parses as backward).
    iso_prompts, iso_star = [], []
    for t, L, dstar, st in slips:
        iso_t = CF.gen_chain("familiar", 1, 0, k=t["k"])
        iso_t["order"] = t["order"]; iso_t["start"] = st[L]["prev"]
        iso_t["nxt"] = t["nxt"]
        iso_prompts.append(p._ids(p.prompt(CF.render(iso_t), enable_thinking=False)) + nl("Step 1: "))
        iso_star.append(dstar)
    # decode the ACTUAL greedy token (robust to space-prefixed digits) and compare
    @torch.no_grad()
    def greedy_tok(prefixes, bs=64):
        out = [None]*len(prefixes); pad = p.tok.pad_token_id
        order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
        for s in range(0, len(order), bs):
            sub = order[s:s+bs]; seqs=[prefixes[i] for i in sub]; ml=max(len(x) for x in seqs)
            ids = torch.tensor([[pad]*(ml-len(x))+x for x in seqs], device=p.device)
            o = p.model(input_ids=ids, attention_mask=(ids!=pad).long(), logits_to_keep=1)
            am = o.logits[:, -1, :].argmax(-1).cpu().tolist()
            for i, tid in zip(sub, am): out[i]=tid
        return out
    iso_tok = greedy_tok(iso_prompts)
    iso_ok = [p.tok.decode([tid]).strip() == ds for tid, ds in zip(iso_tok, iso_star)]

    print(f"\n=== C42 repair (single-slip chains, n={len(slips)}) ===")
    print(f"correct-digit RANK at the slip: mean {statistics.mean(ranks):.2f}, median {statistics.median(ranks)}, "
          f"rank<=2 {statistics.mean(r<=2 for r in ranks):.2f}, rank<=3 {statistics.mean(r<=3 for r in ranks):.2f}")
    print(f"greedy re-try recovers d*:        {statistics.mean(greedy_ok):.3f}  (no-op control, 0 by construction)")
    print(f"confidence-select recovers d*:    {statistics.mean(confsel):.3f}  (argmax = the slip)")
    print(f"temperature-resample avail (n=8): {statistics.mean(avail8):.3f}  (P at least one sample == d*)")
    print(f"ISOLATED re-computation recovers: {statistics.mean(iso_ok):.3f}  (fresh one-step prompt)")
    print(f"oracle ceiling:                   1.000")
    (EXP/"runs"/"repair.json").write_text(json.dumps({
        "n_single_slip": len(slips), "rank_mean": round(statistics.mean(ranks),3),
        "rank_le2": round(statistics.mean(r<=2 for r in ranks),3),
        "rank_le3": round(statistics.mean(r<=3 for r in ranks),3),
        "greedy_recover": round(statistics.mean(greedy_ok),3),
        "resample_avail8": round(statistics.mean(avail8),3),
        "isolated_recompute": round(statistics.mean(iso_ok),3)}, indent=2)+"\n")
    print("wrote runs/repair.json")


if __name__ == "__main__":
    main()
