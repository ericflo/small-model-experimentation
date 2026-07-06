#!/usr/bin/env python3
"""Capture residual-stream activations while the model reads identification I/O examples, plus behavioral
baselines (identification pass@1, first-op naming). Saves activations + labels for the probe step."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import families as FAM  # noqa: E402
import code_env as E  # noqa: E402

FAM_L = FAM.FAMILIES["list"]
NAMES = list(FAM_L["prims"])  # ~16 primitive names (label space)


def ident_prompt(t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    return (f"Infer the Python function `transform` from these input/output examples:\n{lines}\n\n"
            f"Write `def transform(xs):` reproducing this for all such inputs. Only a ```python code block.")


def name_prompt(t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    menu = ", ".join(NAMES)
    return (f"These examples come from applying a pipeline of operations to the input list, in order:\n{lines}\n\n"
            f"Available operations: {menu}.\n\n"
            f"Which operation is applied FIRST (directly to the input, before any other)? "
            f"Answer with exactly one operation name on the last line as `First: <name>`.")


def ops_of(t):
    out = []
    for s in t["target_ops"]:
        op = s.split("(")[0]
        out.append(op)
    return out


def to_public(t):
    return [{"call_expr": f"transform({e['input']!r})", "expected_expr": f"{e['output']!r}"} for e in t["visible"]]


def to_hidden(t):
    return [f"assert transform({e['input']!r}) == {e['output']!r}" for e in t["hidden"]]


FIRST_RE = re.compile(r"First:\s*([a-z_]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-depth", type=int, default=500)
    ap.add_argument("--n-behavioral", type=int, default=150, help="behavioral baselines on this many/depth")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_per_depth, args.n_behavioral, args.depths = 12, 12, [1, 3]

    rng = random.Random(args.seed)
    tasks = []
    for d in args.depths:
        made = 0
        while made < args.n_per_depth:
            t = FAM.make_task(FAM_L, len(tasks), d, rng, k_visible=8, m_hidden=6)
            if t:
                tasks.append(t); made += 1
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    print(f"{len(tasks)} verified tasks, depths {args.depths}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    t0 = time.time()

    # --- activations at last prompt token (no-think prompt) ---
    seqs = [p._ids(p.prompt(ident_prompt(t), enable_thinking=False)) for t in tasks]
    A = p.activations(seqs, batch_size=16)  # [N, L+1, H]
    np.save(EXP / "data" / "acts.npy", A)
    print(f"activations {A.shape} [{time.time()-t0:.0f}s]", flush=True)

    # --- labels ---
    first_op = [ops_of(t)[0] for t in tasks]
    present = np.zeros((len(tasks), len(NAMES)), dtype=np.int8)
    for i, t in enumerate(tasks):
        for op in set(ops_of(t)):
            present[i, NAMES.index(op)] = 1
    np.save(EXP / "data" / "present.npy", present)

    # behavioral subset: first n_behavioral tasks per depth
    beh_idx = []
    cnt = {}
    for i, t in enumerate(tasks):
        cnt[t["depth"]] = cnt.get(t["depth"], 0) + 1
        if cnt[t["depth"]] <= args.n_behavioral:
            beh_idx.append(i)
    beh_set = set(beh_idx)

    # --- behavioral: identification pass@1 (think, greedy) ---
    iprompts = [p.prompt(ident_prompt(tasks[i]), enable_thinking=True) for i in beh_idx]
    ig = p.gen_sequences(iprompts, think=True, budget=512, greedy=True, answer_max=420, batch_size=24)
    solved = [None] * len(tasks)
    for i, pr, g in zip(beh_idx, iprompts, ig):
        t = tasks[i]
        txt = p.tok.decode(g["seq_ids"][len(p._ids(pr)):], skip_special_tokens=False)
        txt = txt.split("</think>")[-1] if "</think>" in txt else txt
        c, _ = E.extract_candidate_code(txt, "transform")
        solved[i] = bool(c) and bool(E.execute_public_and_asserts(c, to_public(t), to_hidden(t))["full_pass"])
    print(f"identification pass@1 done ({len(beh_idx)}) [{time.time()-t0:.0f}s]", flush=True)

    # --- behavioral: first-op naming (think, greedy) ---
    nprompts = [p.prompt(name_prompt(tasks[i]), enable_thinking=True) for i in beh_idx]
    ng = p.gen_sequences(nprompts, think=True, budget=512, greedy=True, answer_max=200, batch_size=24)
    named = [None] * len(tasks)
    named_ok = [None] * len(tasks)
    for i, pr, g in zip(beh_idx, nprompts, ng):
        t = tasks[i]
        txt = p.tok.decode(g["seq_ids"][len(p._ids(pr)):], skip_special_tokens=False)
        txt = txt.split("</think>")[-1] if "</think>" in txt else txt
        m = list(FIRST_RE.finditer(txt))
        pick = m[-1].group(1) if m else ""
        if pick not in NAMES:  # fall back to any mentioned name
            pick = next((n for n in NAMES if n in txt), "")
        named[i] = pick
        named_ok[i] = pick == ops_of(t)[0]
    print(f"first-op naming done ({len(beh_idx)}) [{time.time()-t0:.0f}s]", flush=True)

    (EXP / "data" / "labels.json").write_text(json.dumps({
        "names": NAMES,
        "depth": [t["depth"] for t in tasks],
        "first_op": first_op,
        "solved": solved,
        "named": named,
        "named_ok": named_ok,
    }, indent=1))
    # quick console summary of behavioral rates (over the behavioral subset only)
    from collections import defaultdict
    by = defaultdict(lambda: [0, 0, 0])
    for i in beh_idx:
        c = by[tasks[i]["depth"]]; c[0] += 1; c[1] += int(solved[i]); c[2] += int(named_ok[i])
    print(f"{'depth':>5} {'n':>4} {'ident@1':>8} {'name1st':>8}")
    for d in sorted(by):
        c = by[d]
        print(f"{d:>5} {c[0]:>4} {c[1]/c[0]:>8.2f} {c[2]/c[0]:>8.2f}")
    print("wrote data/{acts.npy, present.npy, labels.json}")


if __name__ == "__main__":
    main()
