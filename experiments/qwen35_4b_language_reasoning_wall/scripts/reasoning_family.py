#!/usr/bin/env python3
"""Contamination-free multi-step SIMULATION substrate (tests C13's mental-simulation wall in language, NOT the
C32/C36 proposal wall). A random successor-chain over made-up pronounceable entities + distractor chains,
shuffled. Same chain rendered in modalities x relations: LINGUISTIC-semantic ('Kel is directly followed by
Vor.'), LINGUISTIC-symbolic ('Kel gorps Vor.' -- made-up token, contamination-clean control), FORMAL ('nxt =
{..}; apply D times'). Shortcut-hardened: the answer is INTERIOR (chain longer than depth, so answer is never the
sink), the start is a RANDOM interior node (depth != line number), distractors are confusable (same name
distribution). Paired: one task -> all renderings."""
from __future__ import annotations
import random

CONS = "bcdfghjklmnpqrstvwz"
VOW = "aeiou"


def gen_names(n, rng):
    names = []
    seen = set()
    while len(names) < n:
        nm = rng.choice(CONS).upper() + rng.choice(VOW) + rng.choice(CONS)
        if rng.random() < 0.5:
            nm += rng.choice(VOW) + rng.choice(CONS)
        # reject substrings/superstrings of existing names (avoid match/tokenization artifacts)
        if nm in seen or any(nm in o or o in nm for o in names):
            continue
        seen.add(nm); names.append(nm)
    return names


def gen_task(depth, seed, n_distract=6, extra=(2, 4)):
    rng = random.Random(seed)
    main_len = depth + rng.randint(*extra)          # main chain LONGER than depth -> answer is interior
    start_i = rng.randint(0, main_len - 1 - depth)   # random interior start; start+depth <= main_len-1
    d_lens = [rng.randint(depth, depth + 3) for _ in range(n_distract)]
    pool = gen_names(main_len + sum(d_lens) + 8, rng)
    rng.shuffle(pool); it = iter(pool)
    main = [next(it) for _ in range(main_len)]
    chains = [main] + [[next(it) for _ in range(dl)] for dl in d_lens]
    facts = [(a, b) for ch in chains for a, b in zip(ch, ch[1:])]
    rng.shuffle(facts)
    start = main[start_i]; answer = main[start_i + depth]
    return {"depth": depth, "main": main, "start": start, "answer": answer, "facts": facts,
            "entities": [e for ch in chains for e in ch], "start_i": start_i, "main_len": main_len}


def _q(t, step_word):
    d = t["depth"]; s = "step" if d == 1 else "steps"
    return (f"Question: Starting from {t['start']} and moving forward {d} {s} "
            f"(each '{step_word}' is one step), which name do you reach?\n"
            f"Answer with exactly one name on the last line as `Answer: <name>`.")


def render_linguistic_semantic(t):
    lines = "\n".join(f"{a} is directly followed by {b}." for a, b in t["facts"])
    return f"Here is a list of ordering facts:\n{lines}\n\n{_q(t, 'is directly followed by')}"


def render_linguistic_symbolic(t):
    lines = "\n".join(f"{a} gorps {b}." for a, b in t["facts"])
    return f"Here is a list of facts using the relation 'gorps':\n{lines}\n\n{_q(t, 'gorps')}"


def render_formal(t):
    entries = ", ".join(f"{a!r}: {b!r}" for a, b in t["facts"])
    d = t["depth"]
    return (f"You are given a dictionary `nxt`:\nnxt = {{{entries}}}\n\n"
            f"Question: Start from {t['start']!r} and look it up in `nxt`, then look up the result in `nxt`, "
            f"and repeat -- {d} lookup(s) in total.\n"
            f"Answer with exactly one value on the last line as `Answer: <value>`.")


RENDERERS = {"ling_sem": render_linguistic_semantic, "ling_sym": render_linguistic_symbolic, "formal": render_formal}


if __name__ == "__main__":
    t = gen_task(3, 7)
    for k in RENDERERS:
        print(f"=== {k} ===\n{RENDERERS[k](t)}\n")
    print("start_i", t["start_i"], "main_len", t["main_len"], "answer", t["answer"], "(interior:", t["answer"] != t["main"][-1], ")")
