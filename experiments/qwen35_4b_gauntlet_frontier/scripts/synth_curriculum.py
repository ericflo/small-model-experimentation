#!/usr/bin/env python3
"""SYNTHETIC CURRICULUM for the induction circuit (glyphgate) — DESIGNED pedagogy,
not harvested/oracle-answer traces. Programmatically generates correct, truth-blind
training text that BUILDS the composable features composed-rule induction needs:
  (P) PRIMITIVE lessons: each op in isolation, worked -> a clean feature per op;
  (S) SINGLE-rule induction: try-op -> check-against-every-example -> keep;
  (C) COMPOSITION-DECOMPOSITION (the circuit C56 lacked): 'output=step2(step1(x)).
      For each candidate step1, apply it to the probe INPUTS to get intermediates,
      then find the ONE step2 mapping all intermediates->outputs' -- the SEARCH made
      explicit with worked intermediates, incl. a dead-end and the found decomposition.
Curriculum-ordered P->S->C. Ground truth from the gym's own _apply (we wrote it).
CPU-only. Emits SFT rows matching train_think.py."""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP / "src"))
from gym.families import glyphgate as G  # noqa
from gym import base as B  # noqa

GL = G.GLYPHS; NG = G.N_GLYPHS
KINDS = ("rev", "swap", "setg", "shift", "crot")
ANS = B.ATOM_ANSWER_INSTRUCTION


def rstr(rng, L): return tuple(rng.randrange(NG) for _ in range(L))
def fmt(s): return G._fmt(s)


def find_single(ins, outs):
    """Return the single op (from KINDS) mapping every ins[i]->outs[i], or None.
    Uses the gym's own pools -- ground truth."""
    L = len(ins[0])
    for r in G._singles_pool(L, KINDS):
        if all(G._apply(r, i) == o for i, o in zip(ins, outs)):
            return r
    return None


def prim_lesson(rng, L):
    op = list(G._gen_single(rng, L, KINDS)); inp = rstr(rng, L); out = G._apply(op, inp)
    prompt = (f"{G._ALPHABET_LINE}\nApply this single operation to the glyph string "
              f"{fmt(inp)}:\n  - {G._step_desc(tuple(op))}.\nWrite the resulting string.\n"
              f"Write glyph strings dash-joined, e.g. za-ke-ro.\n\n{ANS}")
    think = (f"The operation is: {G._step_desc(tuple(op))}. I apply it to {fmt(inp)} directly. "
             f"That gives {fmt(out)}.")
    return {"prompt": prompt, "think": think, "answer": f"ANSWER: {fmt(out)}", "kind": "curr_primitive", "level": 1}


def probes_for(rng, rule, n, L):
    seen, ps = set(), []
    while len(ps) < n and len(seen) < 200:
        c = rstr(rng, L)
        if c not in seen:
            seen.add(c); ps.append((c, G._apply(rule, c)))
    return ps


def single_lesson(rng, L):
    op = list(G._gen_single(rng, L, KINDS)); rule = op
    ps = probes_for(rng, rule, 3, L); q = rstr(rng, L); a = G._apply(rule, q)
    plog = "\n".join(f"  {fmt(i)} -> {fmt(o)}" for i, o in ps)
    prompt = (f"A sealed glyph machine maps {L}-glyph strings by one hidden operation.\n{G._ALPHABET_LINE}\n"
              f"Probe log (input -> output):\n{plog}\n\nWhat does the machine print for {fmt(q)}?\n"
              f"Write glyph strings dash-joined.\n\n{ANS}")
    ins = [i for i, _ in ps]; outs = [o for _, o in ps]
    lines = ["The rule is ONE operation. I test each family against EVERY probe pair and keep the one that fits."]
    for k in KINDS:
        cand = next((r for r in G._singles_pool(L, (k,)) if all(G._apply(r, i) == o for i, o in ps)), None)
        if tuple(op) == tuple(cand) if cand else False:
            lines.append(f"Try {k}: {G._step_desc(tuple(op))} maps every probe correctly. KEEP it.")
            break
        lines.append(f"Try {k}: no parameterization fits all probes. Drop it.")
    lines.append(f"So the rule is {G._step_desc(tuple(op))}. Apply to {fmt(q)}: gives {fmt(a)}.")
    return {"prompt": prompt, "think": " ".join(lines), "answer": f"ANSWER: {fmt(a)}", "kind": "curr_single", "level": 3}


def comp_lesson(rng, L):
    for _ in range(20):
        op1 = list(G._gen_single(rng, L, KINDS)); op2 = list(G._gen_single(rng, L, KINDS))
        rule = ["comp", op1, op2]
        ps = probes_for(rng, rule, 4, L)
        if any(G._apply(rule, i) != i for i, o in ps):
            break
    q = rstr(rng, L); a = G._apply(rule, q)
    ins = [i for i, _ in ps]; outs = [o for _, o in ps]
    plog = "\n".join(f"  {fmt(i)} -> {fmt(o)}" for i, o in ps)
    prompt = (f"A sealed glyph machine maps {L}-glyph strings by a hidden rule that CHAINS TWO operations "
              f"(the second reads the result of the first).\n{G._ALPHABET_LINE}\n"
              f"Probe log (input -> output):\n{plog}\n\nWhat does the machine print for {fmt(q)}?\n"
              f"Write glyph strings dash-joined.\n\n{ANS}")
    # DECOMPOSITION SEARCH made explicit: fix step1, compute intermediates, find step2.
    lines = ["output = step2(step1(input)). To find both, I FIX a candidate first step, apply it to every "
             "probe INPUT to get intermediates, then check whether ONE operation maps all intermediates to the outputs."]
    # one honest dead-end (a candidate step1 whose residual has no single step2), then the correct one.
    dead = None
    for cand in G._singles_pool(L, KINDS):
        if tuple(cand) == tuple(op1):
            continue
        mids = [G._apply(cand, i) for i in ins]
        if find_single(mids, outs) is None:
            dead = (cand, mids); break
    if dead:
        cand, mids = dead
        lines.append(f"Try step1 = {G._step_desc(tuple(cand))}: intermediates " +
                     ", ".join(f"{fmt(i)}->{fmt(m)}" for i, m in zip(ins, mids)) +
                     ". No single operation maps those intermediates to the outputs. Dead end.")
    mids = [G._apply(op1, i) for i in ins]
    step2 = find_single(mids, outs)
    lines.append(f"Try step1 = {G._step_desc(tuple(op1))}: intermediates " +
                 ", ".join(f"{fmt(i)}->{fmt(m)}" for i, m in zip(ins, mids)) + ".")
    lines.append(f"Now one operation maps every intermediate to its output: {G._step_desc(tuple(step2))} "
                 f"({', '.join(f'{fmt(m)}->{fmt(o)}' for m, o in zip(mids, outs))}). So step2 = that.")
    qmid = G._apply(op1, q)
    lines.append(f"Rule found: first {G._step_desc(tuple(op1))}, then {G._step_desc(tuple(step2))}. "
                 f"Apply to {fmt(q)}: {fmt(q)} -> {fmt(qmid)} -> {fmt(a)}.")
    return {"prompt": prompt, "think": " ".join(lines), "answer": f"ANSWER: {fmt(a)}", "kind": "curr_comp", "level": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prim", type=int, default=300); ap.add_argument("--n-single", type=int, default=300)
    ap.add_argument("--n-comp", type=int, default=600); ap.add_argument("--seed", type=int, default=71001)
    ap.add_argument("--out", type=Path, default=EXP / "data" / "sft_curriculum_induction.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed); rows = []
    for _ in range(a.n_prim):   rows.append(prim_lesson(rng, rng.choice([4, 5, 6])))
    for _ in range(a.n_single): rows.append(single_lesson(rng, rng.choice([4, 5])))
    for _ in range(a.n_comp):   rows.append(comp_lesson(rng, rng.choice([5, 6])))
    for r in rows:
        r["family"] = "glyphgate"; r["messages"] = [{"role": "user", "content": r.pop("prompt")}]
        r["n_think_tokens"] = max(1, len(r["think"]) // 4); r["row_weight"] = 1.0
    rng.shuffle(rows)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"curriculum: {len(rows)} rows | {dict(Counter(r['kind'] for r in rows))}")


if __name__ == "__main__":
    main()
