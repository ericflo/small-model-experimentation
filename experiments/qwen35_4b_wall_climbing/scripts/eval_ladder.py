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
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_per_depth, args.depths, args.K = 3, [1, 2], 6

    fam = FAM.FAMILIES["list"]
    # held-out tasks, disjoint from TRAIN (exclude identical depth+target_ops)
    train_keys = set()
    tp = EXP / "data" / "train_tasks.jsonl"
    if tp.exists():
        for line in tp.read_text().splitlines():
            if line.strip():
                tt = json.loads(line)
                train_keys.add((tt["depth"], tuple(tt["target_ops"])))
    rng = random.Random(args.seed)
    tasks = []
    for d in args.depths:
        made = 0
        while made < args.n_per_depth:
            t = FAM.make_task(fam, 10_000 + len(tasks), d, rng, k_visible=8, m_hidden=8)
            if t is None or (t["depth"], tuple(t["target_ops"])) in train_keys:
                continue
            tasks.append(t); made += 1
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "eval_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    print(f"[{args.tag}] {len(tasks)} held-out tasks (disjoint from train), K={args.K}", flush=True)

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
