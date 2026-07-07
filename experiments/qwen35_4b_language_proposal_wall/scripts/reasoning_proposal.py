"""Relational-composition INDUCTION (proposal) substrate -- the linguistic analog of C32's structure-proposal
wall. R made-up relations, each a random bijection over made-up entities; a HIDDEN rule = a fixed sequence of D
relations. Give the full relation KB + k examples (start -> answer applying the hidden rule) and a query whose
start is NOT in the examples -> the model must INDUCE which relations compose (in what order) and apply it. Same
KB/rule rendered LINGUISTICALLY ('Kel's gorm is Vor.') vs FORMALLY (dicts). Tests whether the structure-PROPOSAL
wall (can't infer a depth-3 op-sequence, C32/C36) persists in language, in contrast to SIMULATION which does not
(C37). Contamination-free, verifiable, non-guessable."""
from __future__ import annotations
import random, sys
from itertools import product
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reasoning_family import gen_names  # noqa: E402

RELNAMES = ["gorm", "plit", "vash", "drex", "quon", "selb"]


def _apply(maps, rule, x):
    for r in rule:
        x = maps[r][x]
    return x


def _compose(maps, rule):
    ents = list(next(iter(maps.values())))
    return {e: _apply(maps, rule, e) for e in ents}


def _is_min_depth(maps, rule, rels, depth):
    """Reject if the composed bijection equals a shorter (<depth) composition on ALL entities (C13 pathology)."""
    target = _compose(maps, rule)
    for d in range(0, depth):
        for sr in product(rels, repeat=d):
            if _compose(maps, list(sr)) == target:
                return False
    return True


def gen_task(depth, seed, n_rel=4, n_ent=16, k_examples=5):
    rng = random.Random(seed)
    ents = gen_names(n_ent, rng)
    rels = RELNAMES[:n_rel]
    maps = {r: dict(zip(ents, rng.sample(ents, len(ents)))) for r in rels}
    rule = [rng.choice(rels) for _ in range(depth)]
    if not _is_min_depth(maps, rule, rels, depth):
        return None  # nominal-depth != true-depth (shallower-equivalent) -> reject
    order = ents[:]; rng.shuffle(order)
    q_start = order[-1]
    # grow examples until the hidden rule is the UNIQUE depth-D rule consistent with them (well-posed induction)
    all_rules = [list(c) for c in product(rels, repeat=depth)]
    for k in range(k_examples, n_ent):
        ex_starts = order[:k]
        examples = [(s, _apply(maps, rule, s)) for s in ex_starts]
        consistent = [r for r in all_rules if all(_apply(maps, r, s) == a for s, a in examples)]
        # also require the surviving rules to AGREE on the query (answer well-defined)
        if len(consistent) == 1 or len({_apply(maps, r, q_start) for r in consistent}) == 1:
            return {"depth": depth, "ents": ents, "rels": rels, "maps": maps, "rule": rule,
                    "examples": examples, "query": q_start, "answer": _apply(maps, rule, q_start),
                    "n_consistent": len(consistent)}
    return None  # could not make well-posed (rare)


def render_linguistic(t):
    facts = "\n".join(f"{e}'s {r} is {t['maps'][r][e]}." for r in t["rels"] for e in t["ents"])
    ex = "\n".join(f"- Starting from {s}, the rule gives {a}." for s, a in t["examples"])
    rels = ", ".join(t["rels"])
    d = t["depth"]
    return (f"Here are relationship facts among some names:\n{facts}\n\n"
            f"A hidden rule maps a starting name to a result name. The rule is a fixed sequence of {d} "
            f"relation-step(s), each chosen from ({rels}), applied in order. Here are examples of the rule:\n{ex}\n\n"
            f"Question: Applying the SAME hidden rule, starting from {t['query']}, which name do you reach?\n"
            f"Answer with exactly one name on the last line as `Answer: <name>`.")


def render_formal(t):
    dicts = "\n".join(f"{r} = {{" + ", ".join(f"{e!r}: {v!r}" for e, v in t['maps'][r].items()) + "}" for r in t["rels"])
    ex = "\n".join(f"rule({s!r}) == {a!r}" for s, a in t["examples"])
    rels = ", ".join(t["rels"])
    d = t["depth"]
    return (f"You are given dictionaries:\n{dicts}\n\n"
            f"A hidden function `rule` is a composition of {d} of these dictionaries (each maps a name to a name), "
            f"applied in order (chosen from {rels}). Here are examples:\n{ex}\n\n"
            f"Question: What is rule({t['query']!r})?\n"
            f"Answer with exactly one value on the last line as `Answer: <value>`.")


def render_application(t):
    """APPLICATION-ONLY control: the rule is GIVEN explicitly (no induction) -> pure simulation on this substrate.
    The ceiling that isolates induction: a low induction score is a PROPOSAL wall only if this is near-ceiling."""
    facts = "\n".join(f"{e}'s {r} is {t['maps'][r][e]}." for r in t["rels"] for e in t["ents"])
    rule_str = ", then ".join(t["rule"])
    d = t["depth"]
    return (f"Here are relationship facts among some names:\n{facts}\n\n"
            f"Apply this rule to a starting name: {rule_str} (apply each relation-step in order, {d} step(s)).\n"
            f"Question: Starting from {t['query']}, which name do you reach?\n"
            f"Answer with exactly one name on the last line as `Answer: <name>`.")


RENDERERS = {"ling": render_linguistic, "formal": render_formal, "app": render_application}

if __name__ == "__main__":
    t = gen_task(3, 5)
    print("rule:", t["rule"], "| query:", t["query"], "| answer:", t["answer"])
    print("\n=== LINGUISTIC ===\n" + render_linguistic(t)[:1200])
