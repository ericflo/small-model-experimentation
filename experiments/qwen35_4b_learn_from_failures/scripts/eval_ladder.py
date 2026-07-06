#!/usr/bin/env python3
"""Evaluate base or banked model on HELD-OUT tasks (disjoint from TRAIN) in a no-think harness:
greedy@1 (deployable single-shot) + coverage@k (no-think sampling). Grade vs hidden examples."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP / "src"))
import common as C  # noqa: E402
import families as FAM  # noqa: E402
import code_env as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--n-per-depth", type=int, default=20)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seed", type=int, default=909)
    ap.add_argument("--think", action="store_true", help="eval in think-mode (for the sample-more baseline)")
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--eval-file", type=str, default="data/eval_frozen.jsonl",
                    help="frozen held-out set: generated once, reused by every arm (paired comparison)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_per_depth, args.depths, args.K = 3, [1, 2], 6
        args.eval_file = "data/eval_frozen_smoke.jsonl"

    fam = FAM.FAMILIES["list"]
    # --- behavioral function-signature dedup: a rule maps a FIXED probe set to outputs; exclude any eval task
    # whose function matches ANY training task's (catches alternate decompositions, not just equal target_ops) ---
    PROBE = [[((i * 7 + j * 5) % 19) - 9 for j in range(4 + (i % 5))] for i in range(24)]

    def parse_ops(target_ops):
        out = []
        for s in target_ops:
            op = s.split("(")[0]
            k = int(s[s.index("(") + 1:-1]) if "(" in s else None
            out.append((op, k))
        return out

    def func_sig(target_ops):
        sig = []
        for x in PROBE:
            st = list(x)
            for op, k in parse_ops(target_ops):
                st = FAM.apply_op(fam, op, k, st)
                if st is None:
                    break
            sig.append(repr(st))
        return tuple(sig)

    train_sigs, train_ops = set(), set()  # function-signature AND op-composition (skeleton) of every train rule
    tp = EXP / "data" / "train_tasks.jsonl"
    if tp.exists():
        for line in tp.read_text().splitlines():
            if line.strip():
                to = json.loads(line)["target_ops"]
                train_sigs.add(func_sig(to)); train_ops.add(tuple(to))

    (EXP / "data").mkdir(exist_ok=True)
    efile = EXP / args.eval_file
    if efile.exists():
        tasks = [json.loads(l) for l in efile.read_text().splitlines() if l.strip()]
        # report leakage of THIS train set (superset changes per experiment) into the frozen eval set
        leak_fn = sum(1 for t in tasks if func_sig(t["target_ops"]) in train_sigs)
        leak_op = sum(1 for t in tasks if tuple(t["target_ops"]) in train_ops)
        print(f"[{args.tag}] loaded {len(tasks)} FROZEN eval tasks from {efile.name} (paired) | "
              f"leakage vs train: {leak_fn} func-sig, {leak_op} op-composition (must be 0)", flush=True)
    else:
        rng = random.Random(args.seed)
        tasks, guard = [], 0
        for d in args.depths:
            made = 0
            while made < args.n_per_depth and guard < args.n_per_depth * 400:
                guard += 1
                t = FAM.make_task(fam, 10_000 + len(tasks), d, rng, k_visible=8, m_hidden=8)
                # exclude if the eval task's FUNCTION or its op-composition is in ANY training rule
                if t is None or func_sig(t["target_ops"]) in train_sigs or tuple(t["target_ops"]) in train_ops:
                    continue
                tasks.append(t); made += 1
        efile.write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
        print(f"[{args.tag}] generated {len(tasks)} FROZEN eval tasks (dedup vs {len(train_sigs)} train rules "
              f"by function-signature AND op-composition; 0 leakage by construction) -> {efile.name}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    if args.adapter:
        from peft import PeftModel
        p.model = PeftModel.from_pretrained(p.model, args.adapter)
        p.model.eval()
        print(f"[{args.tag}] loaded adapter {args.adapter}", flush=True)
    t0 = time.time()

    th = bool(args.think)
    bud = args.budget if th else 0
    bs1, bsk = (24, 48) if th else (48, 64)
    prompts = [p.prompt(C.ident_prompt(fam, t), enable_thinking=th) for t in tasks]
    # greedy@1 (deployable single-shot)
    g1 = p.gen_sequences(prompts, think=th, budget=bud, greedy=True, answer_max=420, batch_size=bs1)
    # coverage: K sampled
    rep = [pr for pr in prompts for _ in range(args.K)]
    gk = p.gen_sequences(rep, think=th, budget=bud, greedy=False, answer_max=420, batch_size=bsk)
    print(f"[{args.tag}] generation done [{time.time()-t0:.0f}s]", flush=True)

    def code_of(prompt, seq):
        txt = p.tok.decode(seq[len(p._ids(prompt)):], skip_special_tokens=False)
        txt = txt.split("</think>")[-1] if "</think>" in txt else txt
        c, _ = E.extract_candidate_code(txt, "transform")
        return c or ""

    recs = []
    grade_keys = set()
    per_task = []
    for ti, t in enumerate(tasks):
        gcode = code_of(prompts[ti], g1[ti]["seq_ids"])
        kcodes = [code_of(rep[ti * args.K + j], gk[ti * args.K + j]["seq_ids"]) for j in range(args.K)]
        per_task.append((gcode, kcodes))
        for c in [gcode] + kcodes:
            grade_keys.add((ti, c))
    keys = list(grade_keys)
    with ThreadPoolExecutor(max_workers=16) as ex:
        res = dict(zip(keys, ex.map(lambda k: C.grade(k[1], tasks[k[0]]), keys)))

    for ti, t in enumerate(tasks):
        gcode, kcodes = per_task[ti]
        cov_full = sum(1 for c in kcodes if res[(ti, c)][1])
        recs.append({"tag": args.tag, "depth": t["depth"], "K": args.K,
                     "greedy_full": bool(res[(ti, gcode)][1]),
                     "cov_full": cov_full, "n_unique": len({c for c in kcodes if c})})
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / f"eval_{args.tag}.json").write_text(json.dumps({"records": recs}, indent=1))

    from collections import defaultdict
    by = defaultdict(list)
    for r in recs:
        by[r["depth"]].append(r)
    print(f"\n=== {args.tag} (no-think) ===")
    print(f"{'depth':>5} {'n':>3} {'greedy@1':>9} {'cov_full/K':>11} {'uniq':>5}")
    for d in sorted(by):
        rs = by[d]
        n = len(rs)
        g = sum(r["greedy_full"] for r in rs) / n
        cf = sum(r["cov_full"] for r in rs) / n
        uq = sum(r["n_unique"] for r in rs) / n
        print(f"{d:>5} {n:>3} {g:>9.2f} {cf:>7.1f}/{rs[0]['K']} {uq:>5.1f}")
    print(f"wrote runs/eval_{args.tag}.json [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
