#!/usr/bin/env python3
"""Context composition: can explicit orchestration / few-shot demonstration compose capabilities that
weight-training cannot (C14)? Conditions per reports/prereg.md, on the keystone ladder tasks.

2AFC:  plain@1024 (budget control), orchestrated, ICL  x  {base, SIM adapter}
Ident: orchestrated generate-and-test  x  {base, SIM}
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

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


def afc_core(t, rng):
    true_s, dec_s = pipe_str(steps_of(t)), pipe_str(decoy_pipeline(t, rng))
    a_is_true = rng.random() < 0.5
    pa, pb = (true_s, dec_s) if a_is_true else (dec_s, true_s)
    ex = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"][:6])
    menu = ", ".join(f"{n}: {D.DESC[n]}" for n in D.NAMES)
    return pa, pb, ex, menu, a_is_true


def afc_plain(t, rng):
    pa, pb, ex, menu, a = afc_core(t, rng)
    u = (f"Examples of `transform`:\n{ex}\n\nWhich pipeline produces exactly this behaviour?\n"
         f"A) {pa}\nB) {pb}\n\n(Operation meanings: {menu}.)\n"
         "Work through the examples step by step, then finish with a line `Answer: A` or `Answer: B`.")
    return u, a


ORCH = ("Use this exact procedure:\n"
        "1. Take the FIRST example's input. Apply pipeline A one operation at a time, writing the "
        "intermediate list after each operation as `Step 1: [...]`, `Step 2: [...]`, etc.\n"
        "2. Compare your final list with that example's shown output.\n"
        "3. Do the same for pipeline B.\n"
        "4. If both match, repeat on the second example.\n"
        "The correct pipeline matches exactly. Finish with a line `Answer: A` or `Answer: B`.")


def afc_orch(t, rng):
    pa, pb, ex, menu, a = afc_core(t, rng)
    u = (f"Examples of `transform`:\n{ex}\n\nWhich pipeline produces exactly this behaviour?\n"
         f"A) {pa}\nB) {pb}\n\n(Operation meanings: {menu}.)\n\n{ORCH}")
    return u, a


def build_icl_demos(rng):
    """Two worked simulate-both-compare demos on fresh d2 tasks (disjoint from eval)."""
    demos = []
    while len(demos) < 2:
        t = F.make_controlled_task(9000 + len(demos), 2, len(demos) % 2, rng, k_visible=4, m_hidden=2)
        if t is None:
            continue
        true_steps = steps_of(t)
        dec_steps = decoy_pipeline(t, rng)
        a_is_true = len(demos) == 0
        pa, pb = (true_steps, dec_steps) if a_is_true else (dec_steps, true_steps)
        ex0 = t["visible"][0]
        lines = [f"Examples of `transform`:"]
        lines += [f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"][:3]]
        lines += [f"Which pipeline produces exactly this behaviour?", f"A) {pipe_str(pa)}", f"B) {pipe_str(pb)}", ""]
        work = [f"Checking A on {ex0['input']!r}:"]
        st = tuple(ex0["input"])
        okA = True
        for i, (op, k) in enumerate(pa):
            st = D.apply_prim(op, k, (st,))[0]
            work.append(f"Step {i+1}: {list(st)!r}")
        okA = list(st) == ex0["output"]
        work.append(f"Final {list(st)!r} {'==' if okA else '!='} expected {ex0['output']!r} -> A {'matches' if okA else 'does not match'}.")
        st = tuple(ex0["input"])
        for i, (op, k) in enumerate(pb):
            st = D.apply_prim(op, k, (st,))[0]
        okB = list(st) == ex0["output"]
        work.append(f"Checking B the same way gives {list(st)!r} -> B {'matches' if okB else 'does not match'}.")
        ansL = "A" if okA and not okB else ("B" if okB and not okA else None)
        if ansL is None:
            continue
        work.append(f"Answer: {ansL}")
        demos.append("\n".join(lines) + "\n" + "\n".join(work))
    return "\n\n---\n\n".join(demos)


def afc_icl(t, rng, demos):
    u, a = afc_plain(t, rng)
    return ("Here are two worked examples of how to solve this kind of question:\n\n" + demos +
            "\n\n---\n\nNow your question:\n\n" + u), a


IDENT_ORCH = ("Solve by generate-and-test: (1) propose a candidate pipeline of the operations above; "
              "(2) apply it to the FIRST example's input one operation at a time, writing `Step i: [...]` "
              "lines; (3) if the final list does not equal the shown output, revise your candidate and "
              "test again; (4) once it matches, verify mentally on a second example. Then output ONLY the "
              "final function as `def transform(xs):` in one ```python code block.")


def ident_orch(t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    menu = ", ".join(f"{n}: {D.DESC[n]}" for n in D.NAMES)
    return (f"Infer the rule mapping input list to output list.\n\nExamples:\n{lines}\n\n"
            f"(Available operations: {menu}.)\n\n{IDENT_ORCH}")


ANS_RE = re.compile(r"Answer:\s*([AB])")


def parse_afc(txt):
    m = list(ANS_RE.finditer(txt))
    if m:
        return m[-1].group(1) == "A"
    letters = [c for c in txt if c in "AB"]
    return (letters[-1] == "A") if letters else None


def extract(p, prompt, seq_ids):
    txt = p.tok.decode(seq_ids[len(p._ids(prompt)):], skip_special_tokens=False)
    if "</think>" in txt:
        txt = txt.split("</think>")[-1]
    return txt


def grade_code(task, txt):
    c, _ = E.extract_candidate_code(txt, "transform")
    return bool(c) and bool(E.execute_public_and_asserts(c, G.to_public_cases(task), G.to_hidden_asserts(task))["full_pass"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--conds", nargs="+", required=True,
                    help="subset of: afc_plain afc_orch afc_icl ident_orch")
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--k-ident", type=int, default=2)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    allt = [json.loads(l) for l in (EXP / "data" / "ladder_tasks.jsonl").read_text().splitlines()]
    tasks = allt[:8] if args.smoke else allt
    import gen_lib as GL
    p = GL.Probe()
    if args.adapter:
        from peft import PeftModel
        p.model = PeftModel.from_pretrained(p.model, args.adapter)
        p.model.eval()
    tag = ("SIM" if args.adapter else "base")
    print(f"{len(tasks)} tasks | model={tag} | conds={args.conds}", flush=True)
    out = {"tag": tag, "conds": {}}
    t0 = time.time()

    for cond in args.conds:
        rng = random.Random(4242)  # SAME decoys/order as keystone for every 2AFC condition
        if cond == "afc_icl":
            demos = build_icl_demos(random.Random(777))
        recs = []
        if cond.startswith("afc"):
            items = []
            for t in tasks:
                if cond == "afc_plain":
                    items.append(afc_plain(t, rng))
                elif cond == "afc_orch":
                    items.append(afc_orch(t, rng))
                else:
                    items.append(afc_icl(t, rng, demos))
            prompts = [p.prompt(u, enable_thinking=True) for u, _ in items]
            gens = p.gen_sequences(prompts, think=True, budget=args.budget, greedy=True,
                                   answer_max=1100, batch_size=16)
            for t, pr, g, (_, a_true) in zip(tasks, prompts, gens, items):
                txt = extract(p, pr, g["seq_ids"])
                got = parse_afc(txt)
                recs.append({"task_id": t["task_id"], "depth": t["depth"], "n_destr": t["n_destr"],
                             "correct": (got == a_true) if got is not None else False,
                             "parsed": got is not None})
        else:  # ident_orch
            prompts = [p.prompt(ident_orch(t), enable_thinking=True) for t in tasks]
            rep = [pr for pr in prompts for _ in range(args.k_ident)]
            gens = p.gen_sequences(rep, think=True, budget=args.budget, greedy=False,
                                   answer_max=900, batch_size=16)
            for ti, t in enumerate(tasks):
                oks = [grade_code(t, extract(p, rep[ti * args.k_ident + j], gens[ti * args.k_ident + j]["seq_ids"]))
                       for j in range(args.k_ident)]
                recs.append({"task_id": t["task_id"], "depth": t["depth"], "n_destr": t["n_destr"],
                             "correct": any(oks), "parsed": True})
        out["conds"][cond] = recs
        acc = sum(r["correct"] for r in recs) / len(recs)
        pr_ = sum(r["parsed"] for r in recs) / len(recs)
        print(f"  {cond}: acc {acc:.2f} (parse {pr_:.2f}) [{time.time()-t0:.0f}s]", flush=True)

    Path(EXP / args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
