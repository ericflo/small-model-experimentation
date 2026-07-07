"""Multi-step successor chains with per-STEP ground truth, for the error-localization experiment. 'Advance k steps
in a cyclic order', applied depth times, the model showing each intermediate (Step i: <digit>). Reuses the C37/C40
order machinery. Mix of familiar (few errors) and novel/reversal orders (more errors) to get a useful per-step
error rate. Ground truth per step: the correct successor of the model's OWN previous output (local correctness)."""
from __future__ import annotations
import random

DIGITS = [str(i) for i in range(10)]


def _order(kind, rng):
    if kind == "familiar": return DIGITS[:]
    if kind == "reversal": return DIGITS[::-1]
    if kind == "novel":    o = DIGITS[:]; rng.shuffle(o); return o
    raise ValueError(kind)


def gen_chain(kind, depth, seed, k=None):
    rng = random.Random(seed)
    order = _order(kind, rng)
    k = k if k is not None else rng.randint(2, 8)
    nxt = {order[i]: order[(i + k) % 10] for i in range(10)}
    start = rng.choice(DIGITS)
    chain = [start]
    for _ in range(depth):
        chain.append(nxt[chain[-1]])
    return {"kind": kind, "depth": depth, "order": order, "k": k, "nxt": nxt, "start": start, "chain": chain}


def render(t):
    order_note = (f"The digits are arranged in this fixed circular order: {' '.join(t['order'])} "
                  f"(after the last, it wraps back to the first).\n")
    body = (f"Starting from {t['start']}, repeatedly move {t['k']} steps forward in this order, "
            f"{t['depth']} times in total.")
    fmt = (f"Show your work: write each intermediate result on its own line as `Step <i>: <digit>` "
           f"for i = 1 to {t['depth']}. Do NOT write any code.")
    return f"{order_note}{body}\n{fmt}"


if __name__ == "__main__":
    t = gen_chain("novel", 5, 3, k=3)
    print(render(t)); print("true chain:", t["chain"])
