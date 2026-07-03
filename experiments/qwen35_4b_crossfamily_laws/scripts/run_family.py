#!/usr/bin/env python3
"""Family-generic capability ladder: transcription (plan-given), simulation microbenchmark, bare
identification, and orchestrated 2AFC. Run per family; compare the C13-C15 constants across families.
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
import families as FAM  # noqa: E402
import code_env as E  # noqa: E402


def ops_of(task):
    out = []
    for s in task["target_ops"]:
        op = s.split("(")[0]
        k = int(s[s.index("(") + 1:-1]) if "(" in s else None
        out.append((op, k))
    return out


def true_chain(fam, ops, x):
    chain, st = [], x
    for op, k in ops:
        st = FAM.apply_op(fam, op, k, st)
        chain.append(st)
    return chain


def desc_menu(fam):
    return ", ".join(f"{n}" for n in fam["prims"])


# ---- prompt builders (family-parameterized) ---------------------------------------------------
def p_bare(fam, t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    return (f"Infer the Python function `transform` from these input/output examples:\n{lines}\n\n"
            f"Write `{fam['sig']}` reproducing this for all such inputs. Only a ```python code block.")


def p_plan(fam, t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    plan = " then ".join(t["target_ops"])
    used = sorted({o.split("(")[0] for o in t["target_ops"]})
    meanings = "; ".join(f"{n}: applies `{FAM.snippet(fam, n, 'k' if fam['prims'][n][1] else None)}`" for n in used)
    return (f"A Python function `transform` on the given state is exactly this pipeline, in order: {plan}.\n"
            f"Step definitions (each mutates `{fam['var']}`): {meanings}.\n\nExamples:\n{lines}\n\n"
            f"Implement `{fam['sig']}` applying the steps in order. Only a ```python code block.")


def p_sim(fam, t, x):
    ops = ops_of(t)
    plan = "\n".join(f"  Step {i+1}: {FAM.op_repr(op, k)} — `{FAM.snippet(fam, op, k)}`" for i, (op, k) in enumerate(ops))
    return ("Apply the following operations to the input state IN YOUR HEAD (no code), one at a time.\n\n"
            f"Operations (each mutates `{fam['var']}`):\n{plan}\n\nInput {fam['var']} = {x!r}\n\n"
            "Write the state after EACH step, one line per step as `Step 1: <value>` ... No other text.")


STEP_RE = re.compile(r"[Ss]tep\s*(\d+)\s*[::]\s*(.+)")
BRACKET_RE = re.compile(r"\[[^\[\]]*\]")


def _val_for_family(raw, family):
    raw = raw.strip().strip("`").strip()
    if family == "string":
        # value is a bare/quoted token; take first whitespace-delimited chunk, strip quotes
        tok = raw.split()[0] if raw.split() else ""
        tok = tok.strip("'\"").strip(".,")
        return tok if all(c.isalpha() for c in tok) and tok else None
    m = BRACKET_RE.search(raw)  # list / register: first bracketed expr on the line
    if not m:
        return None
    try:
        v = eval(m.group(0), {}, {})  # noqa: S307 (trusted: bracket-only substring)
        return v if isinstance(v, list) and all(isinstance(x, int) for x in v) else None
    except Exception:
        return None


def parse_last_states(text, n, family):
    """Extract the value on each `Step i:` line; return the last n (one per pipeline step)."""
    by_step = {}
    for m in STEP_RE.finditer(text):
        v = _val_for_family(m.group(2), family)
        if v is not None:
            by_step[int(m.group(1))] = v  # last write wins if a step is restated
    if not by_step:
        return None
    ordered = [by_step[i] for i in sorted(by_step)]
    return ordered[-n:] if len(ordered) >= n else None


def extract(p, prompt, seq_ids):
    txt = p.tok.decode(seq_ids[len(p._ids(prompt)):], skip_special_tokens=False)
    return txt.split("</think>")[-1] if "</think>" in txt else txt


def grade(task, txt):
    c, _ = E.extract_candidate_code(txt, "transform")
    return bool(c) and bool(E.execute_public_and_asserts(c, FAM.to_public_cases(task), FAM.to_hidden_asserts(task))["full_pass"])


def gen_pass(p, prompts, k, budget, batch, tasks, amax):
    rep = [pr for pr in prompts for _ in range(k)]
    gens = p.gen_sequences(rep, think=True, budget=budget, greedy=(k == 1), answer_max=amax, batch_size=batch)
    out = []
    for ti, t in enumerate(tasks):
        out.append(any(grade(t, extract(p, rep[ti * k + j], gens[ti * k + j]["seq_ids"])) for j in range(k)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["list", "string", "register"])
    ap.add_argument("--n-per-depth", type=int, default=25)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--seed", type=int, default=303)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_per_depth, args.depths = 3, [2, 4]

    fam = FAM.FAMILIES[args.family]
    rng = random.Random(args.seed)
    tasks = []
    for d in args.depths:
        made = 0
        while made < args.n_per_depth:
            t = FAM.make_task(fam, len(tasks), d, rng)
            if t:
                tasks.append(t); made += 1
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / f"tasks_{args.family}.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    print(f"[{args.family}] {len(tasks)} verified tasks, depths {args.depths}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    t0 = time.time()

    # transcription (plan-given), pass@1 greedy
    trans = gen_pass(p, [p.prompt(p_plan(fam, t), enable_thinking=True) for t in tasks], 1, args.budget, 24, tasks, 700)
    print(f"  transcription done [{time.time()-t0:.0f}s]", flush=True)
    # bare identification, pass@4 sampled
    bare = gen_pass(p, [p.prompt(p_bare(fam, t), enable_thinking=True) for t in tasks], 4, args.budget, 48, tasks, 512)
    print(f"  identification done [{time.time()-t0:.0f}s]", flush=True)
    # simulation microbenchmark (thinking), output exact-match
    sprompts = [p.prompt(p_sim(fam, t, t["visible"][0]["input"]), enable_thinking=True) for t in tasks]
    sgens = p.gen_sequences(sprompts, think=True, budget=args.budget, greedy=True, answer_max=400, batch_size=20)
    sim = []
    for t, pr, g in zip(tasks, sprompts, sgens):
        chain = true_chain(fam, ops_of(t), t["visible"][0]["input"])
        parsed = parse_last_states(extract(p, pr, g["seq_ids"]), len(chain), args.family)
        sim.append(bool(parsed) and parsed[-1] == chain[-1])
    print(f"  simulation done [{time.time()-t0:.0f}s]", flush=True)

    recs = [{"task_id": t["task_id"], "depth": t["depth"], "transcription": tr, "bare": b, "sim": s}
            for t, tr, b, s in zip(tasks, trans, bare, sim)]
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / f"ladder_{args.family}.json").write_text(json.dumps({"family": args.family, "records": recs}, indent=1))

    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in recs:
        for rung in ("transcription", "bare", "sim"):
            c = by[r["depth"]][rung]; c[0] += 1; c[1] += int(r[rung])
    print(f"\n=== {args.family.upper()} ladder ===")
    print(f"{'depth':>6} {'transcription':>14} {'simulation':>11} {'bare-ident':>11}")
    for d in sorted(by):
        row = by[d]
        print(f"{d:>6} " + "".join(f"{row[rg][1]/row[rg][0]:>14.2f}" for rg in ("transcription", "sim", "bare")))
    ov = {rg: sum(r[rg] for r in recs) / len(recs) for rg in ("transcription", "sim", "bare")}
    print(f"  ALL   transcription {ov['transcription']:.2f}  simulation {ov['sim']:.2f}  bare {ov['bare']:.2f}")
    print(f"wrote runs/ladder_{args.family}.json")


if __name__ == "__main__":
    main()
