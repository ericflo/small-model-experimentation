#!/usr/bin/env python3
"""Can the model localize its own errors in multi-step reasoning? Extends C40 per-STEP via SCAFFOLDED DECODING:
we force the 'Step i: <digit>' format and read the model's live per-step commitment. At each step we read P(m_i)
(softmax over the 10 digit tokens, one forward pass -- C40 trick), take the greedy digit, then scaffold the next
step. This yields, with NO prose and NO truncation: per-step value m_i, confidence P(m_i), and LOCAL correctness
(m_i == successor of the model's OWN previous output m_{i-1}). analyze.py asks: does per-step confidence predict
local correctness beyond a position baseline, and does the lowest-confidence step pinpoint the FIRST local error
(localization) -> targeted repair. Batched across chains, one forward pass per step-depth."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts")); sys.path.insert(0, str(EXP / "src"))
import chain_family as CF  # noqa: E402
DIGIT_IDS = [15 + d for d in range(10)]
# Familiar (natural-order) chains only: the model applies +k but occasionally SLIPS (genuine per-step errors,
# ~15-25%, single-slip-dominant). Novel/reversal collapse under forced-scaffold to a SYSTEMATIC wrong rule
# (natural-successor intrusion -> every step "wrong", no single origin to localize -- the review's failure case).
CONDS = [("familiar", 4), ("familiar", 5), ("familiar", 6), ("familiar", 7)]


@torch.no_grad()
def digit_probs(p, prefixes, batch_size=64):
    out = [None] * len(prefixes); pad = p.tok.pad_token_id
    order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
    for s in range(0, len(order), batch_size):
        sub = order[s:s + batch_size]; seqs = [prefixes[i] for i in sub]; ml = max(len(x) for x in seqs)
        ids = torch.tensor([[pad] * (ml - len(x)) + x for x in seqs], device=p.device)
        o = p.model(input_ids=ids, attention_mask=(ids != pad).long(), logits_to_keep=1)
        pr = torch.softmax(o.logits[:, -1, DIGIT_IDS].float(), dim=1).cpu().tolist()
        for i, v in zip(sub, pr): out[i] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cond", type=int, default=140)
    ap.add_argument("--seed", type=int, default=800)
    args = ap.parse_args()
    import gen_lib as GL
    p = GL.Probe()
    tasks = []
    for ci, (kind, depth) in enumerate(CONDS):
        for i in range(args.n_per_cond):
            tasks.append(CF.gen_chain(kind, depth, args.seed * 100000 + ci * 1000 + i))
    maxd = max(t["depth"] for t in tasks)
    # scaffolded decode: prefix starts with prompt + "Step 1: "
    tok_nl = lambda s: p.tok(s, add_special_tokens=False).input_ids
    prefixes = [p._ids(p.prompt(CF.render(t), enable_thinking=False)) + tok_nl("Step 1: ") for t in tasks]
    recs = [{"kind": t["kind"], "depth": t["depth"], "start": t["start"], "steps": []} for t in tasks]
    prev = [t["start"] for t in tasks]
    for i in range(1, maxd + 1):
        active = [j for j, t in enumerate(tasks) if i <= t["depth"]]
        dists = digit_probs(p, [prefixes[j] for j in active])
        for j, dist in zip(active, dists):
            d = max(range(10), key=lambda x: dist[x]); val = str(d); pc = dist[d]
            t = tasks[j]
            recs[j]["steps"].append({"i": i, "val": val, "p": round(pc, 4),
                                     "local_correct": int(val == t["nxt"].get(prev[j], "")), "prev": prev[j]})
            prev[j] = val
            cont = f"{val}\nStep {i + 1}: " if i < t["depth"] else f"{val}"
            prefixes[j] = prefixes[j] + tok_nl(cont)
    for t, r in zip(tasks, recs):
        model_chain = [t["start"]] + [s["val"] for s in r["steps"]]
        r["final_correct"] = int(model_chain[-1] == t["chain"][t["depth"]])
        fe = [s["i"] for s in r["steps"] if not s["local_correct"]]
        r["first_local_error"] = fe[0] if fe else None
        r["n_local_errors"] = len(fe)
    allsteps = [s for r in recs for s in r["steps"]]
    lc = sum(s["local_correct"] for s in allsteps) / len(allsteps)
    fc = sum(r["final_correct"] for r in recs) / len(recs)
    print(f"[loc] {len(recs)} chains | per-step local-correct={lc:.2f} | final-correct={fc:.2f}", flush=True)
    for kind, depth in CONDS:
        rc = [r for r in recs if r["kind"] == kind and r["depth"] == depth]
        print(f"   {kind} d{depth}: final-correct={sum(r['final_correct'] for r in rc)/len(rc):.2f} "
              f"has-error={sum(r['first_local_error'] is not None for r in rc)/len(rc):.2f}", flush=True)
    (EXP / "runs").mkdir(exist_ok=True)
    json.dump(recs, open(EXP / "runs" / "localize.json", "w"))
    print("[loc] wrote runs/localize.json", flush=True)


if __name__ == "__main__":
    main()
