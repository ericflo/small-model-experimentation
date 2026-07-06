#!/usr/bin/env python3
"""Is the depth-3 wall STRUCTURE or VALUES? Prompt the base model to output the op-SEQUENCE (few-shot format +
16-op menu), by depth. Decompose: direct-cov (op-seq as generated solves the task = structure AND values) vs
skeletonfill-cov (model's op-TYPE skeleton + enumerate param values + execute-filter on visible + check hidden =
structure from model, values from search). oracle-skeletonfill (TRUE skeleton + value-search) = ceiling. Plus a
matched-budget RANDOM-skeleton control (does the model's structure beat random skeletons at equal fill budget?).
If skeletonfill >> direct, the wall is VALUE-binding."""
from __future__ import annotations
import argparse, ast, json, random, re, sys
from itertools import product
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import families as FAM  # noqa: E402
FAM_L = FAM.FAMILIES["list"]
TYPES = list(FAM_L["prims"])
OPTS = {name: opts for name, (_, arity, opts) in FAM_L["prims"].items() if arity}  # arity-1 param options
MENU = ", ".join(f"{n}(K)" if a else n for n, (_, a, _) in FAM_L["prims"].items())

FEWSHOT = (
    "Example 1:\n"
    "transform([3, -1, 4]) == [4, 0, 5]\n"
    "transform([0, 2]) == [1, 3]\n"
    "Pipeline: [\"add_k(1)\"]\n\n"
    "Example 2:\n"
    "transform([2, 1, 3, 1]) == [3, 1]\n"
    "transform([5, 4, 4, 2]) == [5, 4]\n"
    "Pipeline: [\"unique_stable\", \"take_k(2)\"]\n\n")


def opseq_prompt(t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    return (f"A transform is a PIPELINE of operations applied left-to-right to the input list.\n"
            f"Operations (K is an integer parameter): {MENU}.\n\n{FEWSHOT}"
            f"Now infer the pipeline for:\n{lines}\n\n"
            f"Give the exact pipeline as a JSON list of operation strings on the LAST line, "
            f"prefixed `Pipeline:` (e.g. Pipeline: [\"sort_desc\", \"take_k(2)\"]).")


def parse_opseq(text):
    text = text.split("</think>")[-1] if "</think>" in text else text
    ms = re.findall(r"\[([^\[\]]*)\]", text)
    if not ms:
        return None
    for chunk in reversed(ms):
        try:
            items = ast.literal_eval("[" + chunk + "]")
        except Exception:
            items = [x.strip().strip("'\"") for x in chunk.split(",") if x.strip()]
        ops = []
        ok = True
        for it in items:
            it = str(it).strip().strip("'\"")
            if not it:
                continue
            op = it.split("(")[0].strip()
            if op not in TYPES:
                ok = False; break
            k = None
            if "(" in it and ")" in it:
                try:
                    k = int(it[it.index("(") + 1:it.index(")")])
                except Exception:
                    k = None
            ops.append((op, k))
        if ok and ops:
            return ops
    return None


def exec_seq(ops, x):
    st = list(x)
    for op, k in ops:
        st = FAM.apply_op(FAM_L, op, k, st)
        if st is None:
            return None
    return st


def solves(ops, exs):
    return all(exec_seq(ops, e["input"]) == e["output"] for e in exs)


def fills_visible(optypes, task):
    ranges = [OPTS[op] if op in OPTS else [None] for op in optypes]
    out = []
    for combo in product(*ranges):
        ops = list(zip(optypes, combo))
        if solves(ops, task["visible"]):
            out.append(ops)
    return out


def skeletonfill(optypes, task):
    """Return (any_fill_solves_hidden, first_visible_passer_solves_hidden, n_visible_passers)."""
    fills = fills_visible(optypes, task)
    if not fills:
        return (False, False, 0)
    anyhid = any(solves(f, task["hidden"]) for f in fills)
    deploy = solves(fills[0], task["hidden"])
    return (anyhid, deploy, len(fills))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-depth", type=int, default=100)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--think", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed); tasks = []; per = {d: 0 for d in args.depths}; tid = 0
    while any(per[d] < args.n_per_depth for d in args.depths) and tid < 600_000:
        tid += 1; d = args.depths[tid % len(args.depths)]
        if per[d] >= args.n_per_depth:
            continue
        t = FAM.make_task(FAM_L, tid, d, rng, k_visible=8, m_hidden=6)
        if t:
            tasks.append(t); per[d] += 1
    print(f"[skel] {len(tasks)} tasks, per-depth {per} | think={args.think}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    flat, fidx = [], []
    for i, t in enumerate(tasks):
        pr = p.prompt(opseq_prompt(t), enable_thinking=args.think)
        flat.append(pr); fidx.append((i, "greedy"))
        for _ in range(args.k):
            flat.append(pr); fidx.append((i, "sample"))
    greedy_mask = [m == "greedy" for _, m in fidx]
    budget = 512 if args.think else None
    # generate greedy and samples separately for correct decoding
    outs = [None] * len(flat)
    gi = [j for j in range(len(flat)) if greedy_mask[j]]
    si = [j for j in range(len(flat)) if not greedy_mask[j]]
    for idxs, gr in ((gi, True), (si, False)):
        res = p.gen_sequences([flat[j] for j in idxs], think=args.think, budget=budget, greedy=gr,
                              answer_max=200, batch_size=64)
        for j, r in zip(idxs, res):
            outs[j] = r
    samples = {i: {"greedy": None, "samples": []} for i in range(len(tasks))}
    for j, (i, m) in enumerate(fidx):
        txt = p.tok.decode(outs[j]["seq_ids"][len(p._ids(flat[j])):], skip_special_tokens=False)
        ops = parse_opseq(txt)
        if m == "greedy":
            samples[i]["greedy"] = ops
        else:
            samples[i]["samples"].append(ops)
    (EXP / "data").mkdir(exist_ok=True)
    json.dump({"tasks": tasks, "samples": {str(i): samples[i] for i in samples},
               "meta": {"k": args.k, "think": args.think, "depths": args.depths}},
              open(EXP / "data" / f"skel_{'think' if args.think else 'nothink'}.json", "w"))
    print(f"[skel] wrote data/skel_{'think' if args.think else 'nothink'}.json", flush=True)


if __name__ == "__main__":
    main()
