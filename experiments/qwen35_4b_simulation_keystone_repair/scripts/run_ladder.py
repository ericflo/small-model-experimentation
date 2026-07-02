#!/usr/bin/env python3
"""Phase 2: retest the C13 capability ladder under an (optional) adapter, on fresh verified tasks.

Rungs (protocols mirror qwen35_4b_depth_wall_anatomy): bare identification (pass@4, thinking),
plan-given transcription (pass@2), segmented identification (pass@4), no-think 2AFC (logit),
thinking 2AFC (greedy). Cells: d {2,3,4} x k {0,2}.
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
sys.path.insert(0, str(EXP / "scripts"))
import gen_tasks as G  # noqa: E402
import gen_factorial as F  # noqa: E402
import code_env as E  # noqa: E402
import decompose_lib as D  # noqa: E402
from run_simbench import steps_of, true_chain  # noqa: E402


# --- prompt builders (mirroring depth_wall_anatomy) ---------------------------------------------
def prompt_bare(t):
    return G.prompt_for(t)


def prompt_plan(t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    plan = " then ".join(t["target_ops"])
    return ("You are given input/output examples of a Python function `transform` on lists of integers.\n\n"
            f"Examples:\n{lines}\n\nThe transformation is exactly this pipeline of steps, in order: {plan}.\n"
            "Step meanings: " + ", ".join(f"{n}: {D.DESC[n]}" for n in sorted({o.split('(')[0] for o in t['target_ops']})) + ".\n\n"
            "Implement `def transform(xs):` applying these steps in order. Respond with only the function "
            "in one ```python code block.")


def prompt_seg(t):
    steps = steps_of(t)
    states = [tuple(ex["input"]) for ex in t["visible"][:5]]
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
    steps = steps_of(task)
    i = rng.randrange(len(steps))
    old = steps[i][0]
    new = rng.choice([n for n in D.NAMES if n != old])
    k = D.PARAM_OPTS[new][rng.randrange(len(D.PARAM_OPTS[new]))] if D.ARITY[new] else None
    steps = list(steps)
    steps[i] = (new, k)
    return steps


def pipe_str(steps):
    return " then ".join((f"{op}({k})" if k is not None else op) for op, k in steps)


def afc_user(t, rng):
    true_s, dec_s = pipe_str(steps_of(t)), pipe_str(decoy_pipeline(t, rng))
    a_is_true = rng.random() < 0.5
    pa, pb = (true_s, dec_s) if a_is_true else (dec_s, true_s)
    ex = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"][:6])
    user = (f"Examples of `transform`:\n{ex}\n\nWhich pipeline produces exactly this behaviour?\n"
            f"A) {pa}\nB) {pb}\n\n")
    return user, a_is_true


def extract(p, prompt, seq_ids):
    txt = p.tok.decode(seq_ids[len(p._ids(prompt)):], skip_special_tokens=False)
    if "</think>" in txt:
        txt = txt.split("</think>")[-1]
    c, _ = E.extract_candidate_code(txt, "transform")
    return c or ""


def grade(task, code):
    return bool(code) and bool(E.execute_public_and_asserts(code, G.to_public_cases(task), G.to_hidden_asserts(task))["full_pass"])


def gen_pass(p, tasks, builder, k_samples, budget, batch):
    prompts = [p.prompt(builder(t), enable_thinking=True) for t in tasks]
    rep = [pr for pr in prompts for _ in range(k_samples)]
    gens = p.gen_sequences(rep, think=True, budget=budget, greedy=False, batch_size=batch)
    out = []
    for ti, t in enumerate(tasks):
        oks = [grade(t, extract(p, rep[ti * k_samples + j], gens[ti * k_samples + j]["seq_ids"]))
               for j in range(k_samples)]
        out.append(any(oks))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--tasks-file", type=str, default="data/ladder_tasks.jsonl")
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    CELLS = [(2, 0), (2, 2), (3, 0), (3, 2), (4, 0), (4, 2)]

    tpath = EXP / args.tasks_file
    if tpath.exists():
        allt = [json.loads(l) for l in tpath.read_text().splitlines()]
    else:
        allt = F.build_grid(CELLS, args.per_cell, seed=909, verify_depth=True)
        tpath.write_text("\n".join(json.dumps(t) for t in allt) + "\n")
    tasks = []
    for (d, k) in CELLS:
        tasks += [t for t in allt if t["depth"] == d and t["n_destr"] == k][: (2 if args.smoke else args.per_cell)]
    if args.smoke:
        tasks = [t for t in tasks if (t["depth"], t["n_destr"]) in ((2, 0), (3, 2))]
    print(f"{len(tasks)} ladder tasks", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    if args.adapter:
        from peft import PeftModel
        p.model = PeftModel.from_pretrained(p.model, args.adapter)
        p.model.eval()
    tag = args.adapter or "base"
    print(f"model: {tag}", flush=True)
    t0 = time.time()
    res = {}

    res["bare"] = gen_pass(p, tasks, prompt_bare, 4, args.budget, 48)
    print(f"  bare done [{time.time()-t0:.0f}s]", flush=True)
    res["plan_given"] = gen_pass(p, tasks, prompt_plan, 2, args.budget, 32)
    print(f"  plan done [{time.time()-t0:.0f}s]", flush=True)
    res["segmented"] = gen_pass(p, tasks, prompt_seg, 4, args.budget, 20)
    print(f"  segmented done [{time.time()-t0:.0f}s]", flush=True)

    # 2AFC: no-think logit + thinking greedy (same items/decoys: fixed rng)
    rng = random.Random(4242)
    afc_items = [afc_user(t, rng) for t in tasks]
    A = p.tok("A", add_special_tokens=False).input_ids[-1]
    B = p.tok("B", add_special_tokens=False).input_ids[-1]
    ans = p.tok("Answer: ", add_special_tokens=False).input_ids
    prefixes = [p._ids(p.prompt(u + "Answer with the single letter A or B.", enable_thinking=False)) + ans
                for u, _ in afc_items]
    nt = [None] * len(prefixes)
    pad = p.tok.pad_token_id
    order = sorted(range(len(prefixes)), key=lambda i: len(prefixes[i]))
    with torch.no_grad():
        for s in range(0, len(order), 16):
            sub = order[s:s + 16]
            seqs = [prefixes[i] for i in sub]
            m = max(len(x) for x in seqs)
            ids = torch.tensor([[pad] * (m - len(x)) + x for x in seqs], device=p.device)
            attn = (ids != pad).long()
            lg = p.model(input_ids=ids, attention_mask=attn, logits_to_keep=1).logits[:, -1, :].float()
            for i, pick_a in zip(sub, (lg[:, A] > lg[:, B]).cpu().tolist()):
                nt[i] = pick_a
    res["afc_nothink"] = [bool(g) == bool(a) for g, (_, a) in zip(nt, afc_items)]
    print(f"  2AFC no-think done [{time.time()-t0:.0f}s]", flush=True)

    tprompts = [p.prompt(u + "Work through the examples step by step, then answer with the single letter A or B.",
                         enable_thinking=True) for u, _ in afc_items]
    gens = p.gen_sequences(tprompts, think=True, budget=args.budget, greedy=True, batch_size=24)
    tk = []
    for pr, g, (_, a_true) in zip(tprompts, gens, afc_items):
        txt = p.tok.decode(g["seq_ids"][len(p._ids(pr)):], skip_special_tokens=False)
        ansx = txt.split("</think>")[-1] if "</think>" in txt else txt
        picked = next((ch for ch in ansx if ch in "AB"), None)
        tk.append(picked is not None and (picked == "A") == bool(a_true))
    res["afc_think"] = tk
    print(f"  2AFC think done [{time.time()-t0:.0f}s]", flush=True)

    recs = []
    for i, t in enumerate(tasks):
        recs.append({"task_id": t["task_id"], "depth": t["depth"], "n_destr": t["n_destr"],
                     **{k: res[k][i] for k in res}})
    Path(EXP / args.out).write_text(json.dumps({"tag": tag, "records": recs}, indent=1))
    print(f"\n=== LADDER ({tag}) ===")
    rungs = ["bare", "segmented", "afc_nothink", "afc_think", "plan_given"]
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in recs:
        for rg in rungs:
            c = by[(r["depth"], r["n_destr"])][rg]; c[0] += 1; c[1] += int(r[rg])
    print(f"{'cell':>7} " + " ".join(f"{rg:>11}" for rg in rungs))
    for cell in sorted(by):
        row = " ".join(f"{by[cell][rg][1]/by[cell][rg][0]:>11.2f}" for rg in rungs)
        print(f"  d{cell[0]}k{cell[1]} {row}")
    overall = {rg: sum(r[rg] for r in recs) / len(recs) for rg in rungs}
    print("  OVERALL " + " ".join(f"{overall[rg]:>11.2f}" for rg in rungs))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
