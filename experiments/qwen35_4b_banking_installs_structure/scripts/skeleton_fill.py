#!/usr/bin/env python3
"""Is the depth-3 wall STRUCTURE or VALUES? Pivot after review+smoke (op-seq GENERATION is broken: model can't
emit op-sequences, 0.00 even at depth-1). Use the model's NATIVE PYTHON coverage as the baseline, and test the
question via pure search: oracle-skeletonfill (TRUE op-type skeleton + enumerate param values + execute-filter
on visible + check hidden = "if you KNOW the structure, does value-search finish?") vs random-skeletonfill @
matched budget (R random skeletons, filled = "is random structure enough / is the DSL value-fungible?"). On
MIN-DEPTH-verified true-depth tasks. If monolithic~0, oracle~1, random LOW => the wall is STRUCTURE (values are
searchable once structure is known; the DSL is NOT value-fungible so structure genuinely matters)."""
from __future__ import annotations
import argparse, ast, json, random, sys
from itertools import product
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import families as FAM  # noqa: E402
import code_env as E  # noqa: E402
from gen_skeletons import exec_seq, solves, fills_visible, TYPES  # noqa: E402
FAM_L = FAM.FAMILIES["list"]


def ident_prompt(t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    return (f"Infer the Python function `transform` from these input/output examples:\n{lines}\n\n"
            f"Write `def transform(xs):` reproducing this for all such inputs. Only a ```python code block.")


def to_public(t): return [{"call_expr": f"transform({e['input']!r})", "expected_expr": f"{e['output']!r}"} for e in t["visible"]]
def to_hidden(t): return [f"assert transform({e['input']!r}) == {e['output']!r}" for e in t["hidden"]]


def py_solves(code, t):
    if not code: return False
    try: return bool(E.execute_public_and_asserts(code, to_public(t), to_hidden(t))["full_pass"])
    except Exception: return False


def skeletonfill_solves_hidden(optypes, task):
    for f in fills_visible(optypes, task):
        if solves(f, task["hidden"]): return True
    return False


def true_types(t): return [s.split("(")[0] for s in t["target_ops"]]

from gen_skeletons import OPTS  # noqa: E402


def all_fills(optypes):
    ranges = [OPTS[op] if op in OPTS else [None] for op in optypes]
    return [list(zip(optypes, combo)) for combo in product(*ranges)]


def model_outputs(code, vis_inputs):
    """Model program's outputs on the visible inputs (one sandboxed call), or None."""
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


def model_structure_correct(code, t):
    """Format-immune: does the model's Python behave like the TRUE op-type skeleton with SOME (any) params?
    True => model got the STRUCTURE right (any param) => a value error, not a structure error."""
    vis = [e["input"] for e in t["visible"]]
    beh = model_outputs(code, vis)
    if beh is None or "__ERR__" in beh: return False
    for f in all_fills(true_types(t)):
        skel = [exec_seq(f, x) for x in vis]
        if None not in skel and skel == beh: return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-depth", type=int, default=120)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--randR", type=int, nargs="+", default=[8, 50])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seed", type=int, default=71)
    args = ap.parse_args()

    rng = random.Random(args.seed); tasks = []; per = {d: 0 for d in args.depths}; tid = 0
    while any(per[d] < args.n_per_depth for d in args.depths) and tid < 900_000:
        tid += 1; d = args.depths[tid % len(args.depths)]
        if per[d] >= args.n_per_depth: continue
        t = FAM.make_task(FAM_L, tid, d, rng, k_visible=8, m_hidden=6)
        if not t: continue
        # explicit min-depth verification: true-depth-d means NOT solvable at depth <= d-1
        inputs = [e["input"] for e in t["visible"] + t["hidden"]]
        outputs = [e["output"] for e in t["visible"] + t["hidden"]]
        if d > 1 and FAM.min_depth_leq(FAM_L, inputs, outputs, d - 1):
            continue  # collapsed; skip
        tasks.append(t); per[d] += 1
    print(f"[sf] {len(tasks)} min-depth-VERIFIED tasks, per-depth {per}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    # monolithic Python: greedy@1 + cov@k (no-think, deployable)
    prompts = [p.prompt(ident_prompt(t), enable_thinking=False) for t in tasks]
    gg = p.gen_sequences(prompts, think=False, budget=None, greedy=True, answer_max=256, batch_size=64)
    mono_greedy = []; mono_struct_greedy = []
    for t, pr, g in zip(tasks, prompts, gg):
        code, _ = E.extract_candidate_code(p.tok.decode(g["seq_ids"][len(p._ids(pr)):]).strip(), "transform")
        mono_greedy.append(py_solves(code, t))
        mono_struct_greedy.append(model_structure_correct(code, t))
    flat, fidx = [], []
    for i, pr in enumerate(prompts):
        for _ in range(args.k): flat.append(pr); fidx.append(i)
    gs = p.gen_sequences(flat, think=False, budget=None, greedy=False, answer_max=256, batch_size=64)
    mono_cov = [0] * len(tasks); mono_struct_cov = [0] * len(tasks)
    for i, pr, g in zip(fidx, flat, gs):
        code, _ = E.extract_candidate_code(p.tok.decode(g["seq_ids"][len(p._ids(pr)):]).strip(), "transform")
        mono_cov[i] += int(py_solves(code, tasks[i]))
        mono_struct_cov[i] += int(model_structure_correct(code, tasks[i]))
    print("[sf] monolithic Python + structure-decomposition done", flush=True)

    # oracle + random skeletonfill (pure search)
    srng = random.Random(999)
    rows = []
    for i, t in enumerate(tasks):
        d = t["depth"]
        oracle = skeletonfill_solves_hidden(true_types(t), t)
        rand = {}
        for R in args.randR:
            hit = False
            for _ in range(R):
                sk = [srng.choice(TYPES) for _ in range(d)]
                if skeletonfill_solves_hidden(sk, t): hit = True; break
            rand[R] = hit
        rows.append({"depth": d, "mono_greedy": mono_greedy[i], "mono_cov": mono_cov[i],
                     "mono_struct_greedy": mono_struct_greedy[i], "mono_struct_cov": mono_struct_cov[i],
                     "oracle_skelfill": oracle, "rand_skelfill": rand})
    (EXP / "runs").mkdir(exist_ok=True)
    json.dump({"rows": rows, "k": args.k, "randR": args.randR}, open(EXP / "runs" / "skelfill_results.json", "w"), indent=1)
    # console summary
    from math import comb
    def cov(c, n, kk): kk = min(kk, n); return 0.0 if c == 0 else (1.0 if n-c < kk else 1-comb(n-c, kk)/comb(n, kk))
    for d in args.depths:
        rs = [r for r in rows if r["depth"] == d]; n = len(rs)
        mg = sum(r["mono_greedy"] for r in rs)/n
        mc = sum(cov(r["mono_cov"], args.k, args.k) for r in rs)/n
        msc = sum(cov(r["mono_struct_cov"], args.k, args.k) for r in rs)/n
        osk = sum(r["oracle_skelfill"] for r in rs)/n
        print(f"depth {d} (n={n}): monolithic-Python greedy@1 {mg:.3f} cov@{args.k} {mc:.3f} | "
              f"model-STRUCTURE-cov@{args.k} {msc:.3f} (right op-types, any param) | "
              f"oracle-skeletonfill {osk:.3f} | " +
              " ".join(f"rand-skelfill@R{R} {sum(r['rand_skelfill'][R] for r in rs)/n:.3f}" for R in args.randR), flush=True)
    print("[sf] wrote runs/skelfill_results.json", flush=True)


if __name__ == "__main__":
    main()
