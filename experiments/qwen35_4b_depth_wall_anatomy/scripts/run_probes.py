#!/usr/bin/env python3
"""Phase 3 probes: locate the identification deficit precisely.

  segmented : intermediates presented PRE-SEGMENTED as per-step transition blocks (each block is a pure
              depth-1 identification; no segmentation required). Prediction logged in prereg addendum:
              if segmentation was the deficit, solve rates recover toward the d1 rate (~0.7+).
  twoafc    : given the true pipeline and a one-op decoy (both stated), pick which is consistent with the
              I/O examples — identification as DISCRIMINATION (letter-logit A/B read, no generation).

Cells: same verified grid tasks, d {2,3,4} x k {0,2}.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import gen_tasks as G  # noqa: E402
import gen_factorial as F  # noqa: E402
import code_env as E  # noqa: E402
import decompose_lib as D  # noqa: E402


def steps_of(task):
    out = []
    for s in task["target_ops"]:
        op = s.split("(")[0]
        k = int(s[s.index("(") + 1:-1]) if "(" in s else None
        out.append((op, k))
    return out


def seg_prompt(task):
    """Per-step transition blocks over the visible examples — segmentation done FOR the model."""
    steps = steps_of(task)
    states = [tuple(ex["input"]) for ex in task["visible"][:5]]
    blocks = []
    for i, (op, k) in enumerate(steps):
        nxt = [D.apply_prim(op, k, (s,))[0] for s in states]
        lines = "\n".join(f"  {list(a)} -> {list(b)}" for a, b in zip(states, nxt))
        blocks.append(f"Step {i+1} examples (input -> output of this step alone):\n{lines}")
        states = nxt
    return ("A function `transform` applies a fixed sequence of simple list operations. Below, EACH STEP "
            "is shown separately with its own input->output examples:\n\n" + "\n\n".join(blocks) +
            "\n\nInfer what each step does and implement `def transform(xs):` applying all steps in order. "
            "Respond with only the function in one ```python code block.")


def decoy_pipeline(task, rng):
    """Copy the true pipeline, replace one random op with a same-arity different op (params resampled)."""
    steps = steps_of(task)
    i = rng.randrange(len(steps))
    old = steps[i][0]
    pool = [n for n in D.NAMES if n != old]
    new = rng.choice(pool)
    k = D.PARAM_OPTS[new][rng.randrange(len(D.PARAM_OPTS[new]))] if D.ARITY[new] else None
    steps = list(steps)
    steps[i] = (new, k)
    return steps


def pipe_str(steps):
    return " then ".join((f"{op}({k})" if k is not None else op) for op, k in steps)


def twoafc(p, tasks, rng, batch_size=16):
    """Letter-logit A/B: which stated pipeline matches the examples? Returns per-task correctness."""
    A = p.tok("A", add_special_tokens=False).input_ids[-1]
    B = p.tok("B", add_special_tokens=False).input_ids[-1]
    ans = p.tok("Answer: ", add_special_tokens=False).input_ids
    prefixes, truth = [], []
    for t in tasks:
        true_s, dec_s = pipe_str(steps_of(t)), pipe_str(decoy_pipeline(t, rng))
        a_is_true = rng.random() < 0.5
        pa, pb = (true_s, dec_s) if a_is_true else (dec_s, true_s)
        ex = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"][:6])
        user = (f"Examples of `transform`:\n{ex}\n\nWhich pipeline produces exactly this behaviour?\n"
                f"A) {pa}\nB) {pb}\n\n(Step meanings: " +
                ", ".join(f"{n}: {D.DESC[n]}" for n in sorted({s.split('(')[0] for s in t['target_ops']})) +
                ", plus standard list ops.) Answer with the single letter A or B.")
        prefixes.append(p._ids(p.prompt(user, enable_thinking=False)) + ans)
        truth.append(a_is_true)
    correct = []
    pad = p.tok.pad_token_id
    order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
    res = [None] * len(prefixes)
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            sub = order[s:s + batch_size]
            seqs = [prefixes[i] for i in sub]
            m = max(len(x) for x in seqs)
            ids = torch.tensor([[pad] * (m - len(x)) + x for x in seqs], device=p.device)
            attn = (ids != pad).long()
            lg = p.model(input_ids=ids, attention_mask=attn, logits_to_keep=1).logits[:, -1, :].float()
            pick_a = (lg[:, A] > lg[:, B]).cpu().tolist()
            for i, pa_ in zip(sub, pick_a):
                res[i] = pa_
    for got_a, a_true in zip(res, truth):
        correct.append(bool(got_a) == bool(a_true))
    return correct


def extract(p, prompt, seq_ids):
    txt = p.tok.decode(seq_ids[len(p._ids(prompt)):], skip_special_tokens=False)
    if "</think>" in txt:
        txt = txt.split("</think>")[-1]
    c, _ = E.extract_candidate_code(txt, "transform")
    return c or ""


def grade(task, code):
    return bool(code) and bool(E.execute_public_and_asserts(code, G.to_public_cases(task), G.to_hidden_asserts(task))["full_pass"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--k-samples", type=int, default=4)
    ap.add_argument("--budget", type=int, default=512)
    args = ap.parse_args()
    CELLS = [(2, 0), (2, 2), (3, 0), (3, 2), (4, 0), (4, 2)]
    if args.smoke:
        CELLS, args.per_cell, args.k_samples = [(2, 0), (3, 2)], 2, 2

    allt = [json.loads(l) for l in (EXP / "data" / "grid_tasks.jsonl").read_text().splitlines()]
    tasks = []
    for (d, k) in CELLS:
        tasks += [t for t in allt if t["depth"] == d and t["n_destr"] == k][:args.per_cell]
    print(f"{len(tasks)} tasks over {CELLS}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    print(f"model loaded {p.load_secs:.0f}s", flush=True)
    rng = random.Random(4242)
    t0 = time.time()

    # segmented generation
    prompts = [p.prompt(seg_prompt(t), enable_thinking=True) for t in tasks]
    rep = [pr for pr in prompts for _ in range(args.k_samples)]
    # segmented prompts are long (per-step blocks); batch 48 trips the fla kernel -> use 20
    gens = p.gen_sequences(rep, think=True, budget=args.budget, greedy=False, batch_size=20)
    seg_ok = []
    for ti, t in enumerate(tasks):
        oks = [grade(t, extract(p, rep[ti * args.k_samples + j], gens[ti * args.k_samples + j]["seq_ids"]))
               for j in range(args.k_samples)]
        seg_ok.append(any(oks))
    print(f"segmented done [{time.time()-t0:.0f}s]", flush=True)

    afc = twoafc(p, tasks, rng)
    print(f"2AFC done [{time.time()-t0:.0f}s]", flush=True)

    with (EXP / "data" / "probe_records.jsonl").open("w") as f:
        for t, so, ac in zip(tasks, seg_ok, afc):
            f.write(json.dumps({"task_id": t["task_id"], "depth": t["depth"], "n_destr": t["n_destr"],
                                "segmented_passk": so, "twoafc_correct": ac}) + "\n")
    print("\n=== PHASE 3 PROBES (segmented pass@%d | 2AFC accuracy) ===" % args.k_samples)
    for (d, k) in CELLS:
        cell = [(so, ac) for t, so, ac in zip(tasks, seg_ok, afc) if t["depth"] == d and t["n_destr"] == k]
        if cell:
            print(f"  d{d}k{k}: segmented {sum(c[0] for c in cell)/len(cell):.2f} | 2AFC {sum(c[1] for c in cell)/len(cell):.2f}  (n={len(cell)})")
    print("wrote data/probe_records.jsonl")


if __name__ == "__main__":
    main()
