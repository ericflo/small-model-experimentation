"""Format-EQUALIZED single-value substrate for the metacognition experiment (review-hardened). Every condition
shows an explicit order block (so block-PRESENCE cannot be the surface cue); conditions differ only in block
CONTENT and in whether the rule is stated (execute) or must be induced (few-shot). Kinds:
  familiar  -> natural order 0..9 (retrievable; the model knows the successor)
  novel     -> a random scrambled order (must be induced; C39 = chance)
  reversal  -> reversed order 9..0 : the DISSOCIATION condition -- LOOKS non-natural (scrambled-looking surface)
               but is a simple regular reflection the model can induce. If confidence drops here as much as on
               'novel', confidence is a surface heuristic; if it stays high where the model actually succeeds,
               that is competence-assessment beyond surface."""
from __future__ import annotations
import random

DIGITS = [str(i) for i in range(10)]


def _order(kind, rng):
    if kind == "familiar":  return DIGITS[:]
    if kind == "reversal":  return DIGITS[::-1]
    if kind == "novel":     o = DIGITS[:]; rng.shuffle(o); return o
    raise ValueError(kind)


def gen_task(kind, seed, k=None, n_examples=5):
    rng = random.Random(seed)
    order = _order(kind, rng)
    k = k if k is not None else rng.randint(2, 8)
    nxt = {order[i]: order[(i + k) % 10] for i in range(10)}
    perm = DIGITS[:]; rng.shuffle(perm)
    ex_items = perm[:n_examples]
    query = perm[n_examples]
    examples = [(x, nxt[x]) for x in ex_items]
    gap = min(abs(int(query) - int(s)) for s in ex_items)          # numeric gap query->nearest seen input
    feats = {"k": k, "n_examples": n_examples, "query": int(query), "gap_to_seen": gap,
             "n_distinct_seen": len(set(ex_items)), "kind": kind}
    return {"kind": kind, "order": order, "k": k, "examples": examples, "query": query,
            "answer": nxt[query], "natural_succ": DIGITS[(int(query) + k) % 10], "feats": feats}


def render(t, rule_stated=False):
    ex = "\n".join(f"{x} -> {y}" for x, y in t["examples"])
    order_note = (f"The digits are arranged in this fixed circular order: {' '.join(t['order'])} "
                  f"(after the last, it wraps back to the first).\n")
    if rule_stated:
        body = f"A rule maps each digit to the digit {t['k']} steps after it in the order (wrapping around)."
    else:
        body = f"A fixed rule maps each digit to another digit. Here are examples:\n{ex}"
    return (f"{order_note}{body}\n\nApply the SAME rule to: {t['query']}\n"
            f"Do NOT write any code. Reason briefly in plain words, then end with exactly `Answer: <digit>`.")


if __name__ == "__main__":
    for kind in ("familiar", "reversal", "novel"):
        t = gen_task(kind, 5, k=3)
        print(f"=== {kind} (k=3) ===\n{render(t)}\nanswer={t['answer']} natural_succ={t['natural_succ']}\n")
