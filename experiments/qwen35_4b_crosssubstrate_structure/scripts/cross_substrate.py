#!/usr/bin/env python3
"""Do the RECENT structure findings generalize beyond the list DSL? Family-generic replication of C32 (the wall
is STRUCTURE not values) and C34 (brute-force structure-search DOMINATES the model at deploy) on the STRING and
REGISTER substrates (+ LIST anchor). For a family, at depth-3 (min-depth-verified): base model greedy@1/cov@8 +
format-immune STRUCTURE-coverage (does the model program's BEHAVIOR match the true op-type skeleton with any
params?); oracle-skeletonfill (true structure + value-search = ceiling); random-skeletonfill@R (value-fungibility
control); brute-full structure-search + value-fill + execution-consensus deploy. If the pattern holds (base ~0,
structure-cov = concrete-cov, oracle ~1, random low, brute near-solves), these are MODEL-LEVEL laws, not
list-DSL artifacts."""
from __future__ import annotations
import argparse, ast, json, random, sys
from collections import Counter
from itertools import product
from math import comb
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import families as FAM  # noqa: E402
import code_env as E  # noqa: E402


def fam_of(name): return FAM.FAMILIES[name]
def types_of(fam): return list(fam["prims"])
def opts_of(fam): return {n: o for n, (_, a, o) in fam["prims"].items() if a}


def ident_prompt(fam, t):
    var = fam["var"]
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    return (f"Infer the Python function `transform` from these input/output examples:\n{lines}\n\n"
            f"Write `{fam['sig']}` reproducing this for all such inputs. Only a ```python code block.")


def exec_seq(fam, ops, x):
    st = x
    for op, k in ops:
        st = FAM.apply_op(fam, op, k, st)
        if st is None:
            return None
    return st


def solves(fam, ops, exs):
    return all(exec_seq(fam, ops, e["input"]) == e["output"] for e in exs)


def all_fills(fam, optypes):
    OPTS = opts_of(fam)
    ranges = [OPTS[op] if op in OPTS else [None] for op in optypes]
    return [list(zip(optypes, combo)) for combo in product(*ranges)]


def fills_visible(fam, optypes, task):
    return [f for f in all_fills(fam, optypes) if solves(fam, f, task["visible"])]


def skeletonfill_hidden(fam, optypes, task):
    return any(solves(fam, f, task["hidden"]) for f in fills_visible(fam, optypes, task))


def to_public(t): return [{"call_expr": f"transform({e['input']!r})", "expected_expr": f"{e['output']!r}"} for e in t["visible"]]
def to_hidden(t): return [f"assert transform({e['input']!r}) == {e['output']!r}" for e in t["hidden"]]


def py_solves(code, t):
    if not code: return False
    try: return bool(E.execute_public_and_asserts(code, to_public(t), to_hidden(t))["full_pass"])
    except Exception: return False


def model_outputs(code, vis_inputs):
    if not code: return None
    public = [{"call_expr": f"transform({x!r})", "expected_expr": "None"} for x in vis_inputs]
    try:
        r = E.execute_public_and_asserts(code, public, [])
        outs = []
        for po in r.get("public_outputs", []):
            try: outs.append(ast.literal_eval(po))
            except Exception: outs.append("__ERR__")
        return outs if len(outs) == len(vis_inputs) else None
    except Exception:
        return None


def true_types(t): return [s.split("(")[0] for s in t["target_ops"]]


def model_structure_correct(fam, code, t):
    vis = [e["input"] for e in t["visible"]]
    beh = model_outputs(code, vis)
    if beh is None or "__ERR__" in beh: return False
    for f in all_fills(fam, true_types(t)):
        skel = [exec_seq(fam, f, x) for x in vis]
        if None not in skel and skel == beh: return True
    return False


def brute_candidates(fam, task, depth):
    TYPES = types_of(fam); vis = task["visible"]; x1 = vis[0]["input"]; o1 = vis[0]["output"]; out = []
    for skel in product(TYPES, repeat=depth):
        sk = list(skel)
        for fill in all_fills(fam, sk):
            if exec_seq(fam, fill, x1) != o1: continue
            if solves(fam, fill, vis): out.append((tuple(sk), fill))
    return out


def make_probes(name, rng, m=16):
    if name == "string":
        return ["".join(rng.choice("abcdefghijklmnop") for _ in range(rng.randint(4, 8))) for _ in range(m)]
    if name == "register":
        return [[rng.randint(-9, 9) for _ in range(3)] for _ in range(m)]
    return [[rng.randint(-9, 9) for _ in range(rng.randint(4, 8))] for _ in range(m)]


def consensus_deploy(fam, cands, probes, task):
    if not cands: return None
    sig2f = {}
    for _, f in cands:
        sig = tuple(str(exec_seq(fam, f, x)) for x in probes)
        sig2f.setdefault(sig, []).append(f)
    best = max(sig2f, key=lambda s: len(sig2f[s]))
    return solves(fam, sig2f[best][0], task["hidden"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["list", "string", "register"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--randR", type=int, nargs="+", default=[8, 50, 200])
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=41)
    args = ap.parse_args()
    fam = fam_of(args.family)

    rng = random.Random(args.seed); tasks = []; tid = 0
    while len(tasks) < args.n and tid < 900_000:
        tid += 1
        t = FAM.make_task(fam, tid, args.depth, rng, k_visible=8, m_hidden=6)
        if not t: continue
        inp = [e["input"] for e in t["visible"] + t["hidden"]]; out = [e["output"] for e in t["visible"] + t["hidden"]]
        if args.depth > 1 and FAM.min_depth_leq(fam, inp, out, args.depth - 1): continue
        tasks.append(t)
    print(f"[cs {args.family}] {len(tasks)} min-depth-verified depth-{args.depth} tasks | space={len(types_of(fam))**args.depth}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    prompts = [p.prompt(ident_prompt(fam, t), enable_thinking=False) for t in tasks]
    gg = p.gen_sequences(prompts, think=False, budget=None, greedy=True, answer_max=256, batch_size=64)
    greedy = []; struct_g = []
    for t, pr, g in zip(tasks, prompts, gg):
        code, _ = E.extract_candidate_code(p.tok.decode(g["seq_ids"][len(p._ids(pr)):]).strip(), "transform")
        greedy.append(py_solves(code, t)); struct_g.append(model_structure_correct(fam, code, t))
    flat, fidx = [], []
    for i, pr in enumerate(prompts):
        for _ in range(args.k): flat.append(pr); fidx.append(i)
    gs = p.gen_sequences(flat, think=False, budget=None, greedy=False, answer_max=256, batch_size=64)
    cov = [0] * len(tasks); struct_cov = [0] * len(tasks)
    for i, pr, g in zip(fidx, flat, gs):
        code, _ = E.extract_candidate_code(p.tok.decode(g["seq_ids"][len(p._ids(pr)):]).strip(), "transform")
        cov[i] += int(py_solves(code, tasks[i])); struct_cov[i] += int(model_structure_correct(fam, code, tasks[i]))
    print(f"[cs {args.family}] base model done; running search", flush=True)

    srng = random.Random(999); probes = make_probes(args.family, random.Random(2024))
    rows = []
    for i, t in enumerate(tasks):
        oracle = skeletonfill_hidden(fam, true_types(t), t)
        rand = {}
        for R in args.randR:
            hit = False
            for _ in range(R):
                sk = [srng.choice(types_of(fam)) for _ in range(args.depth)]
                if skeletonfill_hidden(fam, sk, t): hit = True; break
            rand[R] = hit
        brute = brute_candidates(fam, t, args.depth)
        rows.append({"greedy": greedy[i], "cov": cov[i], "struct_greedy": struct_g[i], "struct_cov": struct_cov[i],
                     "oracle": oracle, "rand": rand,
                     "brute_cov": any(solves(fam, f, t["hidden"]) for _, f in brute),
                     "brute_deploy": consensus_deploy(fam, brute, probes, t),
                     "brute_skels": len({c[0] for c in brute})})
    (EXP / "runs").mkdir(exist_ok=True)
    json.dump({"family": args.family, "n": len(tasks), "k": args.k, "depth": args.depth,
               "space": len(types_of(fam))**args.depth, "randR": args.randR, "rows": rows},
              open(EXP / "runs" / f"cs_{args.family}.json", "w"), indent=1)
    n = len(tasks)
    def ck(c): kk = min(args.k, args.k); return 0.0 if c == 0 else (1.0 if args.k-c < kk else 1-comb(args.k-c, kk)/comb(args.k, kk))
    print(f"[cs {args.family}] base greedy@1 {sum(greedy)/n:.3f} cov@{args.k} {sum(ck(r['cov']) for r in rows)/n:.3f} "
          f"| STRUCT-cov {sum(ck(r['struct_cov']) for r in rows)/n:.3f} (value tax {sum(ck(r['struct_cov'])-ck(r['cov']) for r in rows)/n:+.3f}) "
          f"| oracle-skelfill {sum(r['oracle'] for r in rows)/n:.3f} "
          f"| random " + " ".join(f"R{R}={sum(r['rand'][R] for r in rows)/n:.3f}" for R in args.randR) +
          f" | BRUTE-deploy {sum(bool(r['brute_deploy']) for r in rows)/n:.3f} (cov {sum(r['brute_cov'] for r in rows)/n:.3f}, ~{sum(r['brute_skels'] for r in rows)/n:.1f} skels/task)", flush=True)
    print(f"[cs {args.family}] wrote runs/cs_{args.family}.json", flush=True)


if __name__ == "__main__":
    main()
