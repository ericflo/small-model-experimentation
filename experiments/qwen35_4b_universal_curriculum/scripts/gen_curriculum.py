#!/usr/bin/env python3
"""Generic MULTI-SKILL synthetic curriculum — engineered pedagogy that installs
UNIVERSAL circuits, on surfaces deliberately disjoint from any benchmark (digits/
letters/romans/invented words/greek). The transfer target is the held-out menagerie
we never train on or read. Skills, each taught over all surfaces with worked,
truth-blind reasoning:
  INDUCT   - infer a hidden COMPOSED rule from examples via decomposition SEARCH
             (fix step-1, compute intermediates, find the step-2 that maps them).
  EXECUTE  - apply a STATED multi-step procedure to an input, step by step.
  SELECT   - pick the item(s) satisfying a conjunction of stated constraints,
             checking each constraint explicitly (constrained search).
The circuits (compose/decompose, execute-a-procedure, check-constraints) are the
generic features we hope transfer. CPU-only, seconds."""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter
from itertools import combinations
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
# reuse the op machinery + surfaces from the induction generator (proven)
sys.path.insert(0, str(EXP.parents[0] / "qwen35_4b_gauntlet_frontier" / "scripts"))
import generic_curriculum as GC  # apply, gen_op, singles, find_single, desc, render, SURFACES, KINDS  # noqa

ANS = "End with exactly one line:\nANSWER: <answer>"
SURF = GC.SURFACES


def rseq(rng, L, N): return tuple(rng.randrange(N) for _ in range(L))


def induct_lesson(rng):
    r = GC.comp_lesson(rng)  # already the decomposition-search pedagogy, surface-varied
    r["kind"] = "u_induct"; return r


def execute_lesson(rng):
    sname = rng.choice(list(SURF)); items = SURF[sname]; N = len(items); L = rng.choice([5, 6, 7])
    order = list(range(N))
    steps = [GC.gen_op(rng, L, N) for _ in range(rng.choice([2, 3]))]
    inp = rseq(rng, L, N); cur = inp; trace = [GC.render(cur, items)]
    for op in steps:
        cur = GC.apply(op, cur, N); trace.append(GC.render(cur, items))
    plan = "; then ".join(GC.desc(op, items, order) for op in steps)
    prompt = (f"Apply this procedure, in order, to the sequence {GC.render(inp, items)}:\n"
              f"  {plan}.\n{GC.header(sname, items, order)}\n{ANS}")
    lines = [f"I execute the {len(steps)} steps in order, carrying the result forward."]
    for k, (op, before, after) in enumerate(zip(steps, trace[:-1], trace[1:]), 1):
        lines.append(f"Step {k} ({GC.desc(op, items, order)}): {before} -> {after}.")
    lines.append(f"Final result: {trace[-1]}.")
    return {"messages": [{"role": "user", "content": prompt}], "think": " ".join(lines),
            "answer": f"ANSWER: {trace[-1]}", "kind": "u_execute", "family": "universal",
            "level": 3, "n_think_tokens": max(1, len(" ".join(lines)) // 4), "row_weight": 1.0}


def select_lesson(rng):
    sname = rng.choice(list(SURF)); items = SURF[sname]; N = len(items)
    n_items = rng.choice([5, 6, 7]); attrs = ["size", "shade", "weight"]
    # each candidate is an id + 3 numeric attributes; a conjunction of constraints picks exactly one.
    cand = []
    for i in range(n_items):
        cand.append({"id": items[i % N] + str(i), "size": rng.randint(1, 9),
                     "shade": rng.randint(1, 9), "weight": rng.randint(1, 9)})
    # build a constraint set satisfied by exactly one; retry
    for _ in range(40):
        tgt = rng.choice(cand)
        cons = []
        for at in rng.sample(attrs, 2):
            if rng.random() < 0.5: cons.append((at, ">=", tgt[at]))
            else: cons.append((at, "<=", tgt[at]))
        def ok(c): return all((c[a] >= v) if op == ">=" else (c[a] <= v) for a, op, v in cons)
        winners = [c for c in cand if ok(c)]
        if len(winners) == 1:
            break
    else:
        winners = [tgt]
    clines = "\n".join(f"  - {c['id']}: size {c['size']}, shade {c['shade']}, weight {c['weight']}" for c in cand)
    ctext = " and ".join(f"{a} {op} {v}" for a, op, v in cons)
    prompt = (f"Among these items, exactly one satisfies ALL constraints. Find its id.\nItems:\n{clines}\n"
              f"Constraints: {ctext}.\n{ANS}")
    lines = ["I check each item against every constraint; keep only those passing ALL."]
    for c in cand:
        res = ", ".join(f"{a} {c[a]} {op} {v}? {'yes' if ((c[a]>=v) if op=='>=' else (c[a]<=v)) else 'NO'}" for a, op, v in cons)
        passes = all((c[a] >= v) if op == ">=" else (c[a] <= v) for a, op, v in cons)
        lines.append(f"{c['id']}: {res} -> {'PASS' if passes else 'fail'}.")
    lines.append(f"Only {winners[0]['id']} passes all constraints.")
    return {"messages": [{"role": "user", "content": prompt}], "think": " ".join(lines),
            "answer": f"ANSWER: {winners[0]['id']}", "kind": "u_select", "family": "universal",
            "level": 3, "n_think_tokens": max(1, len(" ".join(lines)) // 4), "row_weight": 1.0}


def trace_lesson(rng):
    """Multi-hop dereference: follow a pointer chain K times. The generic
    pointer-chasing / navigation circuit (graph & maze tasks share it)."""
    sname = rng.choice(list(SURF)); items = SURF[sname]; N = len(items)
    m = rng.choice([5, 6, 7]); nodes = rng.sample(range(N), m)
    link = {n: rng.choice(nodes) for n in nodes}
    start = rng.choice(nodes); K = rng.choice([2, 3, 4])
    cur = start; path = [cur]
    for _ in range(K):
        cur = link[cur]; path.append(cur)
    linktext = "\n".join(f"  - {items[n]} -> {items[link[n]]}" for n in nodes)
    prompt = (f"Follow the links. Each line 'A -> B' means A points to B. Start at {items[start]} "
              f"and follow the arrow {K} times; report where you land.\nLinks:\n{linktext}\n{ANS}")
    lines = [f"I start at {items[start]} and follow {K} arrows one at a time."]
    for step, (a, b) in enumerate(zip(path[:-1], path[1:]), 1):
        lines.append(f"Step {step}: {items[a]} -> {items[b]}.")
    lines.append(f"After {K} arrows I land on {items[path[-1]]}.")
    return {"messages": [{"role": "user", "content": prompt}], "think": " ".join(lines),
            "answer": f"ANSWER: {items[path[-1]]}", "kind": "u_trace", "family": "universal",
            "level": 3, "n_think_tokens": max(1, len(" ".join(lines)) // 4), "row_weight": 1.0}


def verify_lesson(rng):
    """Check a CANDIDATE result against a stated procedure, step by step, and
    judge YES/NO. Installs the explicit-verification circuit (the meta-skill
    behind the P(True) judge, C46)."""
    sname = rng.choice(list(SURF)); items = SURF[sname]; N = len(items); L = rng.choice([5, 6, 7])
    order = list(range(N))
    steps = [GC.gen_op(rng, L, N) for _ in range(rng.choice([2, 3]))]
    inp = rseq(rng, L, N); cur = inp
    for op in steps:
        cur = GC.apply(op, cur, N)
    true = cur
    correct = rng.random() < 0.5
    if correct:
        cand = true
    else:
        cl = list(true); pos = rng.randrange(L)
        cl[pos] = (cl[pos] + rng.randint(1, N - 1)) % N; cand = tuple(cl)
    plan = "; then ".join(GC.desc(op, items, order) for op in steps)
    prompt = (f"Someone applied this procedure to {GC.render(inp, items)} and claims the result is "
              f"{GC.render(cand, items)}. Is that claim correct?\n  Procedure: {plan}.\n"
              f"{GC.header(sname, items, order)}\nEnd with exactly one line:\nANSWER: YES or ANSWER: NO")
    lines = ["I recompute the procedure myself, then compare to the claim."]
    c = inp
    for k, op in enumerate(steps, 1):
        nx = GC.apply(op, c, N)
        lines.append(f"Step {k} ({GC.desc(op, items, order)}): {GC.render(c, items)} -> {GC.render(nx, items)}.")
        c = nx
    verdict = "matches" if correct else "does NOT match"
    lines.append(f"My result {GC.render(true, items)} {verdict} the claim {GC.render(cand, items)}.")
    return {"messages": [{"role": "user", "content": prompt}], "think": " ".join(lines),
            "answer": f"ANSWER: {'YES' if correct else 'NO'}", "kind": "u_verify", "family": "universal",
            "level": 3, "n_think_tokens": max(1, len(" ".join(lines)) // 4), "row_weight": 1.0}


def count_lesson(rng):
    """Tally how many items satisfy a stated predicate. The generic
    count-under-a-predicate circuit."""
    sname = rng.choice(list(SURF)); items = SURF[sname]; N = len(items)
    n_items = rng.choice([5, 6, 7]); attrs = ["size", "shade", "weight"]
    cand = [{"id": items[i % N] + str(i), "size": rng.randint(1, 9),
             "shade": rng.randint(1, 9), "weight": rng.randint(1, 9)} for i in range(n_items)]
    at = rng.choice(attrs); op = rng.choice([">=", "<="]); v = rng.randint(3, 7)
    def ok(c): return (c[at] >= v) if op == ">=" else (c[at] <= v)
    k = sum(1 for c in cand if ok(c))
    clines = "\n".join(f"  - {c['id']}: size {c['size']}, shade {c['shade']}, weight {c['weight']}" for c in cand)
    prompt = (f"Count how many items satisfy the condition.\nItems:\n{clines}\n"
              f"Condition: {at} {op} {v}.\n{ANS}")
    lines = [f"I check each item's {at} against {op} {v} and tally the ones that pass."]
    run = 0
    for c in cand:
        p = ok(c); run += int(p)
        lines.append(f"{c['id']}: {at} {c[at]} {op} {v}? {'yes' if p else 'no'} (running total {run}).")
    lines.append(f"Total satisfying: {k}.")
    return {"messages": [{"role": "user", "content": prompt}], "think": " ".join(lines),
            "answer": f"ANSWER: {k}", "kind": "u_count", "family": "universal",
            "level": 3, "n_think_tokens": max(1, len(" ".join(lines)) // 4), "row_weight": 1.0}


# skill name -> (generator, default count). The default mix (induct/execute/select
# at these counts, in this order, seed 77001) reproduces v1's sft_universal.jsonl
# byte-for-byte; new skills are opt-in via --mix so they never perturb v1.
SKILLS = {
    "induct": (induct_lesson, 600), "execute": (execute_lesson, 400),
    "select": (select_lesson, 400), "trace": (trace_lesson, 400),
    "verify": (verify_lesson, 400), "count": (count_lesson, 400),
}
DEFAULT_MIX = "induct=600,execute=400,select=400"


def parse_mix(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, cnt = part.partition("=")
        name = name.strip()
        if name not in SKILLS:
            raise SystemExit(f"unknown skill {name!r}; known: {sorted(SKILLS)}")
        out.append((name, int(cnt) if cnt else SKILLS[name][1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default=DEFAULT_MIX,
                    help="comma list skill=count, e.g. 'induct=600,execute=400,trace=300'")
    ap.add_argument("--seed", type=int, default=77001)
    ap.add_argument("--out", type=Path, default=EXP / "data" / "sft_universal.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    mix = parse_mix(a.mix)
    rows = []
    for name, count in mix:
        gen = SKILLS[name][0]
        rows += [gen(rng) for _ in range(count)]
    for r in rows:
        r["family"] = "universal"
    rng.shuffle(rows)
    a.out.write_text("".join(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n" for r in rows))
    print(f"universal curriculum: {len(rows)} rows | kinds {dict(Counter(r['kind'] for r in rows))} | mix {mix}")


if __name__ == "__main__":
    main()
