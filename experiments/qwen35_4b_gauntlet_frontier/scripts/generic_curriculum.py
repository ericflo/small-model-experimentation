#!/usr/bin/env python3
"""GENERIC compositional-induction curriculum — teaches a UNIVERSAL decomposition
circuit on surfaces that look NOTHING like the glyphgate eval, to test whether a
transferable feature (not memorized format) can be installed. The abstract skill:
a sequence over some alphabet is mapped by a hidden rule that CHAINS two primitive
ops (shift-in-cycle / reverse / swap-positions / overwrite-position / conditional-
rotate); given input->output probes, DECOMPOSE by fixing step-1, computing
intermediates, and finding the single step-2 that maps them. Rendered over MANY
surfaces (digits, letters, invented words; sizes 6-10) -- deliberately disjoint
from the glyph vocabulary. Truth-blind (all values computed). CPU-only."""
from __future__ import annotations
import argparse, json, random
from collections import Counter
from itertools import combinations
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ANS = ("End with exactly one line:\nANSWER: <the sequence, items joined by '-'>")

SURFACES = {
    "digits":  list("0123456789"),
    "letters": list("abcdefghij"),
    "romans":  ["I","II","III","IV","V","VI","VII","VIII","IX","X"],
    "words":   ["ash","bly","cor","dun","elm","fro","gil","hax","ivo","jum"],
    "greek":   ["al","be","ga","de","ep","ze","et","th","io","ka"],
}
KINDS = ("shift", "rev", "swap", "setg", "crot")


def apply(op, s, N):
    k = op[0]
    if k == "shift": return tuple((x + op[1]) % N for x in s)
    if k == "rev":   return tuple(reversed(s))
    if k == "swap":  o = list(s); i, j = op[1], op[2]; o[i], o[j] = o[j], o[i]; return tuple(o)
    if k == "setg":  return s[:op[1]] + (op[2],) + s[op[1]+1:]
    if k == "crot":  return s[1:] + s[:1] if (s[0] in (op[1], op[2])) else s[-1:] + s[:-1]
    raise ValueError(k)


def gen_op(rng, L, N):
    k = rng.choice(KINDS)
    if k == "shift": return ("shift", rng.randint(1, N-1))
    if k == "rev":   return ("rev",)
    if k == "swap":  i, j = sorted(rng.sample(range(L), 2)); return ("swap", i, j)
    if k == "setg":  return ("setg", rng.randrange(L), rng.randrange(N))
    return ("crot",) + tuple(sorted(rng.sample(range(N), 2)))


def singles(L, N):
    out = [("rev",)]
    out += [("shift", k) for k in range(1, N)]
    out += [("swap", i, j) for i, j in combinations(range(L), 2)]
    out += [("setg", p, x) for p in range(L) for x in range(N)]
    out += [("crot", a, b) for a, b in combinations(range(N), 2)]
    return out


def find_single(ins, outs, L, N):
    for r in singles(L, N):
        if all(apply(r, i, N) == o for i, o in zip(ins, outs)): return r
    return None


def desc(op, items, order):
    k = op[0]
    if k == "shift": return f"advance every item {op[1]} step(s) forward in the stated order"
    if k == "rev":   return "reverse the whole sequence"
    if k == "swap":  return f"swap the items at positions {op[1]+1} and {op[2]+1}"
    if k == "setg":  return f"overwrite position {op[1]+1} with {items[op[2]]}"
    return f"if the first item is {items[op[1]]} or {items[op[2]]} rotate one step left, else one step right"


def rnd_seq(rng, L, N): return tuple(rng.randrange(N) for _ in range(L))


def render(s, items): return "-".join(items[x] for x in s)


def header(surface, items, order):
    return (f"Items and their fixed cycle order: {' '.join(items[x] for x in order)} "
            f"(after the last it wraps to the first).")


def comp_lesson(rng):
    sname = rng.choice(list(SURFACES)); items = SURFACES[sname]; N = len(items)
    order = list(range(N)); rng.shuffle(order)  # a stated cycle order (perm)
    L = rng.choice([5, 6, 7])
    # remap: op indices are positions in `order`? keep simple: alphabet is 0..N-1, order lists them
    for _ in range(30):
        op1 = gen_op(rng, L, N); op2 = gen_op(rng, L, N)
        ins = []
        seen = set()
        while len(ins) < 4 and len(seen) < 100:
            c = rnd_seq(rng, L, N)
            if c not in seen: seen.add(c); ins.append(c)
        outs = [apply(op2, apply(op1, i, N), N) for i in ins]
        if any(o != i for i, o in zip(ins, outs)): break
    q = rnd_seq(rng, L, N); a = apply(op2, apply(op1, q, N), N)
    plog = "\n".join(f"  {render(i, items)} -> {render(o, items)}" for i, o in zip(ins, outs))
    prompt = (f"A hidden rule maps sequences of {L} items by CHAINING TWO operations (the second reads "
              f"the result of the first). Each operation is one of: advance every item a fixed number "
              f"of steps forward in the order; reverse the whole sequence; swap two fixed positions; "
              f"overwrite one fixed position with a fixed item; or a first-item-conditional one-step "
              f"rotation.\n{header(sname, items, order)}\nProbe log (input -> output):\n{plog}\n\n"
              f"What is the output for {render(q, items)}?\n{ANS}")
    lines = ["output = step2(step1(input)). I FIX a candidate first step, apply it to every probe INPUT "
             "to get intermediates, then check whether ONE operation maps all intermediates to the outputs."]
    dead = None
    for cand in singles(L, N):
        if tuple(cand) == tuple(op1): continue
        mids = [apply(cand, i, N) for i in ins]
        if find_single(mids, outs, L, N) is None: dead = (cand, mids); break
    if dead:
        cand, mids = dead
        lines.append("Try step1 = " + desc(cand, items, order) + ": intermediates " +
                     ", ".join(f"{render(i,items)}->{render(m,items)}" for i, m in zip(ins, mids)) +
                     ". No single operation maps those to the outputs. Dead end.")
    mids = [apply(op1, i, N) for i in ins]; step2 = find_single(mids, outs, L, N)
    lines.append("Try step1 = " + desc(op1, items, order) + ": intermediates " +
                 ", ".join(f"{render(i,items)}->{render(m,items)}" for i, m in zip(ins, mids)) + ".")
    lines.append("Now one operation maps every intermediate to its output: " + desc(step2, items, order) +
                 " (" + ", ".join(f"{render(m,items)}->{render(o,items)}" for m, o in zip(mids, outs)) + "). So step2 is that.")
    qm = apply(op1, q, N)
    lines.append("Rule: first " + desc(op1, items, order) + ", then " + desc(step2, items, order) +
                 f". Apply to {render(q,items)}: -> {render(qm,items)} -> {render(a,items)}.")
    return {"messages": [{"role": "user", "content": prompt}], "think": " ".join(lines),
            "answer": f"ANSWER: {render(a, items)}", "kind": "gen_comp", "family": "generic_induction",
            "level": 5, "n_think_tokens": max(1, len(" ".join(lines))//4), "row_weight": 1.0, "_surface": sname}


def prim_lesson(rng):
    sname = rng.choice(list(SURFACES)); items = SURFACES[sname]; N = len(items); L = rng.choice([5, 6, 7])
    order = list(range(N)); rng.shuffle(order)
    op = gen_op(rng, L, N); inp = rnd_seq(rng, L, N); out = apply(op, inp, N)
    prompt = (f"Apply this single operation to the sequence {render(inp, items)}:\n  - {desc(op, items, order)}.\n"
              f"{header(sname, items, order)}\n{ANS}")
    think = f"The operation: {desc(op, items, order)}. Applied to {render(inp,items)} it gives {render(out,items)}."
    return {"messages": [{"role": "user", "content": prompt}], "think": think, "answer": f"ANSWER: {render(out,items)}",
            "kind": "gen_primitive", "family": "generic_induction", "level": 1,
            "n_think_tokens": max(1, len(think)//4), "row_weight": 1.0, "_surface": sname}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prim", type=int, default=400); ap.add_argument("--n-comp", type=int, default=900)
    ap.add_argument("--seed", type=int, default=73001)
    ap.add_argument("--out", type=Path, default=EXP / "data" / "sft_generic_induction.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed); rows = [prim_lesson(rng) for _ in range(a.n_prim)] + [comp_lesson(rng) for _ in range(a.n_comp)]
    rng.shuffle(rows); a.out.write_text("".join(json.dumps({k:v for k,v in r.items() if k!="_surface"}, ensure_ascii=False)+"\n" for r in rows))
    print(f"generic curriculum: {len(rows)} rows | kinds {dict(Counter(r['kind'] for r in rows))} | surfaces {dict(Counter(r['_surface'] for r in rows))}")


if __name__ == "__main__":
    main()
