"""Meta-induction episodes: can SFT install the SKILL of inducing a novel rule from examples? Each episode gives a
STATED scrambled digit order + k worked examples of a hidden rule + a query; the model must INFER the rule and
apply it. Base fails this (C39: scrambled-order induction ~0.12=chance). Two rule FAMILIES over positions in the
stated order:
  shift  (in-family, TRAIN):  f(order[i]) = order[(i + k) % 10]           -- a=1
  affine (OUT-OF-FAMILY test): f(order[i]) = order[(a*i + b) % 10], a in {3,7,9} coprime to 10
Random orders + params per episode => held-out episodes are genuinely novel (nothing to memorize). Query disjoint
from example inputs => forces rule-use, not lookup."""
from __future__ import annotations
import random

DIGITS = [str(i) for i in range(10)]


def gen_episode(family, seed, n_examples=6):
    rng = random.Random(seed)
    order = DIGITS[:]; rng.shuffle(order)
    pos = {d: i for i, d in enumerate(order)}
    if family == "shift":
        a, b = 1, rng.randint(2, 8)
    elif family == "affine":
        a, b = rng.choice([3, 7, 9]), rng.randint(1, 9)
    else:
        raise ValueError(family)
    f = {d: order[(a * pos[d] + b) % 10] for d in DIGITS}
    perm = DIGITS[:]; rng.shuffle(perm)
    ex_in = perm[:n_examples]; query = perm[n_examples]
    examples = [(x, f[x]) for x in ex_in]
    return {"family": family, "order": order, "a": a, "b": b, "examples": examples,
            "query": query, "answer": f[query]}


def render(ep):
    ex = "\n".join(f"{x} -> {y}" for x, y in ep["examples"])
    return (f"The digits are arranged in this fixed order: {' '.join(ep['order'])}.\n"
            f"A hidden rule maps each digit to another digit. Here are examples:\n{ex}\n\n"
            f"Apply the SAME rule to: {ep['query']}\n"
            f"Reason briefly in plain words (no code), then end with exactly `Answer: <digit>`.")


TARGET = lambda ep: f"Answer: {ep['answer']}"   # answer-only SFT target (cleanest 'install the skill' test)

if __name__ == "__main__":
    for fam in ("shift", "affine"):
        ep = gen_episode(fam, 1)
        print(f"=== {fam} (a={ep['a']} b={ep['b']}) ===\n{render(ep)}\n-> answer {ep['answer']}\n")


def render_execute(ep):
    """EXECUTE-mode: state the rule explicitly (the review's gate -- base must be able to APPLY the rule; else an
    induction failure is really an execution floor). Rule over 0-indexed positions in the stated order."""
    a, b = ep["a"], ep["b"]
    if a == 1:
        rule = f"move each digit {b} steps forward in this order (wrapping around at the end)"
    else:
        rule = (f"for each digit, find its position p (0-indexed) in the order, compute ({a}*p + {b}) mod 10, "
                f"and output the digit at that position")
    return (f"The digits are arranged in this fixed order: {' '.join(ep['order'])}.\n"
            f"Rule: {rule}.\n\nApply the rule to: {ep['query']}\n"
            f"Reason briefly in plain words (no code), then end with exactly `Answer: <digit>`.")


def render_strategy(ep):
    """Base + STRATEGY hint (no SFT): hand over the PROCEDURE (how to find the rule) but NOT the answer. If the base
    can now induce, the wall was a missing-strategy gap elicitable by prompting = serial-compute/latent, not a
    knowledge-absent limit that must be SFT-injected."""
    return (f"The digits are arranged in this fixed order: {' '.join(ep['order'])}.\n"
            f"A hidden rule maps each digit to another digit. Here are examples:\n"
            + "\n".join(f"{x} -> {y}" for x, y in ep["examples"]) +
            f"\n\nHint: to find the rule, take one example x -> y, find the 0-indexed positions of x and y in the "
            f"order, and compute the shift k = (position of y - position of x) mod 10. Then apply: find the query's "
            f"position p and the answer is the digit at position (p + k) mod 10.\n\n"
            f"Apply the rule to: {ep['query']}\nReason step by step in plain words (no code), then end with exactly "
            f"`Answer: <digit>`.")
