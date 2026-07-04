#!/usr/bin/env python3
"""Harvest the fixed 4B's OWN execution-verified identification solutions on TRAIN tasks, to bank via SFT.
Sample K think-mode completions/task, keep hidden-correct programs (cap per task), write {prompt, code}."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
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
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--cap-per-task", type=int, default=12)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--depth-tasks", type=str, default="1:30,2:100",
                    help="comma list of depth:count for the training harvest")
    ap.add_argument("--out-tag", type=str, default="", help="suffix for train.jsonl / train_tasks.jsonl")
    ap.add_argument("--adapter", type=str, default=None, help="harvest with a banked adapter loaded (round 2+)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    depth_tasks = [(int(d), int(n)) for d, n in (x.split(":") for x in args.depth_tasks.split(","))]
    if args.smoke:
        depth_tasks, args.K = [(1, 2), (2, 2)], 8

    fam = FAM.FAMILIES["list"]
    rng = random.Random(args.seed)
    start_id = 50_000 * (hash(args.out_tag) % 7 + 1)  # keep round-2 tasks disjoint from round-1
    tasks = C.gen_tasks(fam, depth_tasks, rng, start_id=start_id)
    (EXP / "data").mkdir(exist_ok=True)
    tag = args.out_tag
    (EXP / "data" / f"train_tasks{tag}.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    print(f"[harvest{tag}] {len(tasks)} train tasks, depths {depth_tasks}, K={args.K}"
          f"{' (adapter '+args.adapter+')' if args.adapter else ''}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    if args.adapter:
        from peft import PeftModel
        p.model = PeftModel.from_pretrained(p.model, args.adapter)
        p.model.eval()
    t0 = time.time()
    prompts = [p.prompt(C.ident_prompt(fam, t), enable_thinking=True) for t in tasks]
    rep = [pr for pr in prompts for _ in range(args.K)]
    gens = p.gen_sequences(rep, think=True, budget=args.budget, greedy=False, answer_max=420, batch_size=48)
    print(f"[harvest] sampling done [{time.time()-t0:.0f}s]", flush=True)

    # extract candidate codes per task
    codes_per = []
    for ti in range(len(tasks)):
        cs = []
        for j in range(args.K):
            idx = ti * args.K + j
            txt = p.tok.decode(gens[idx]["seq_ids"][len(p._ids(rep[idx])):], skip_special_tokens=False)
            txt = txt.split("</think>")[-1] if "</think>" in txt else txt
            c, _ = E.extract_candidate_code(txt, "transform")
            cs.append(c or "")
        codes_per.append(cs)

    # grade unique (thread pool)
    keys = list({(ti, c) for ti in range(len(tasks)) for c in codes_per[ti]})
    with ThreadPoolExecutor(max_workers=16) as ex:
        res = dict(zip(keys, ex.map(lambda k: C.grade(k[1], tasks[k[0]]), keys)))
    print(f"[harvest] graded {len(keys)} unique [{time.time()-t0:.0f}s]", flush=True)

    # keep verified (full_pass), cap per task, dedup by code, prefer most-frequent
    pairs, by_depth = [], Counter()
    for ti, t in enumerate(tasks):
        freq = Counter(codes_per[ti])
        seen, kept = set(), 0
        for c, _ in freq.most_common():
            if kept >= args.cap_per_task:
                break
            if c in seen or not c:
                continue
            seen.add(c)
            if res[(ti, c)][1]:  # full_pass
                pairs.append({"prompt": C.ident_prompt(fam, t), "code": c, "depth": t["depth"]})
                kept += 1
                by_depth[t["depth"]] += 1
    rng.shuffle(pairs)
    (EXP / "data" / f"train{tag}.jsonl").write_text("\n".join(json.dumps(x) for x in pairs) + "\n")
    n_solved = sum(1 for ti, t in enumerate(tasks) if any(res[(ti, c)][1] for c in set(codes_per[ti]) if c))
    print(f"[harvest{tag}] {len(pairs)} verified SFT pairs from {n_solved}/{len(tasks)} solved tasks; "
          f"by depth {dict(by_depth)} [{time.time()-t0:.0f}s]", flush=True)
    print(f"wrote data/train{tag}.jsonl")


if __name__ == "__main__":
    main()
