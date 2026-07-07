"""Execution-SAFE substrate for retrieval-vs-induction (the char-cipher floored: 4B can't assemble ciphered
strings, application-only 0.20). Single-VALUE output = 'k steps forward in a cyclic order'. THE CRUX (review):
FAMILIAR order (natural 0..9, model knows the successor) vs NOVEL order (a STATED random cyclic order) -- SAME
1-param rule ('advance k'), matched description-length, differ ONLY in familiarity. Few-shot examples (x -> the
item k steps after x); query an item NOT in the examples. Generalization to the unseen query = did the model
apply the RULE. If familiar generalizes but novel-given-order does not, ICL = retrieval of familiar structure, not
induction of a novel (even fully-stated) rule."""
from __future__ import annotations
import random

DIGITS = [str(i) for i in range(10)]


def gen_task(kind, seed, k=None, n_examples=5):
    rng = random.Random(seed)
    if kind == "familiar":
        order = DIGITS[:]                    # natural order 0..9 (model knows it)
        stated = False
    elif kind == "novel":
        order = DIGITS[:]; rng.shuffle(order)  # random cyclic order, STATED in the prompt
        stated = True
    else:
        raise ValueError(kind)
    k = k if k is not None else rng.randint(2, 8)
    nxt = {order[i]: order[(i + k) % 10] for i in range(10)}
    # examples cover a SUBSET; query item is NOT among example inputs (tests generalization, not memorization)
    perm = DIGITS[:]; rng.shuffle(perm)
    ex_items = perm[:n_examples]
    query = perm[n_examples]                 # unseen input
    examples = [(x, nxt[x]) for x in ex_items]
    return {"kind": kind, "order": order, "stated": stated, "k": k, "examples": examples,
            "query": query, "answer": nxt[query], "seen_inputs": set(ex_items)}


def render(t, rule_stated=False):
    ex = "\n".join(f"{x} -> {y}" for x, y in t["examples"])
    order_note = ""
    if t["stated"]:
        order_note = (f"The digits are arranged in this fixed circular order: {' '.join(t['order'])} "
                      f"(after the last, it wraps back to the first).\n")
    if rule_stated:
        body = f"A rule maps each digit to the digit {t['k']} steps after it in the order (wrapping around)."
    else:
        body = f"A fixed rule maps each digit to another digit. Here are examples:\n{ex}"
    return (f"{order_note}{body}\n\n"
            f"Apply the SAME rule to: {t['query']}\n"
            f"Do NOT write any code, function, or ``` block -- this is a mental puzzle. "
            f"Reason briefly in plain words, then end with exactly `Answer: <digit>`.")


if __name__ == "__main__":
    for kind in ("familiar", "novel"):
        t = gen_task(kind, 5, k=3)
        print(f"=== {kind} (k={t['k']}) ===")
        print(render(t)); print("answer:", t["answer"], "\n")
