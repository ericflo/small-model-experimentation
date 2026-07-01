#!/usr/bin/env python3
"""Generator-verifier gap: is checking easier than doing?

Generate k no_think candidate solutions per MBPP task (execution-labeled), then have the SAME frozen
4B judge each candidate correct/incorrect as a black-box (A/B logit -> P(correct)), at no-think and
thinking-on. Compare intrinsic verification skill to generation skill, and whether thinking helps
verification asymmetrically. Foreign-solution judgments (a different task's candidate) are a floor control.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import tasks as T  # noqa: E402


def user_prompt(t):
    anchor = t.test_list[0] if t.test_list else ""
    return (f"{t.prompt}\n\nYour function must satisfy this example:\n{anchor}\n"
            f"Define the function with the exact name used above.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=int, default=1024)
    args = ap.parse_args()
    if args.smoke:
        args.tasks, args.k = 4, 4

    (EXP / "data").mkdir(exist_ok=True)
    tasks = T.load_mbpp(split="test", limit=args.tasks)
    n, k = len(tasks), args.k
    print(f"{n} tasks, k={k}, judge budget={args.budget}", flush=True)
    (EXP / "data" / "tasks.json").write_text(json.dumps(
        {t.task_id: {"test_list": t.test_list, "test_imports": t.test_imports} for t in tasks}))

    import judge_lib as JL
    p = JL.Probe()
    print(f"model loaded in {p.load_secs:.0f}s", flush=True)
    wall0 = time.time()

    # 1. generation: k no_think candidates per task
    gen_prompts = [p.prompt(user_prompt(t), enable_thinking=False) for t in tasks]
    rep = [gp for gp in gen_prompts for _ in range(k)]
    cands = p.gen_sequences(rep, think=False, budget=None, batch_size=64)
    codes = []
    for i in range(len(rep)):
        plen = len(p._ids(rep[i]))
        codes.append(T.extract_code(p.tok.decode(cands[i]["seq_ids"][plen:], skip_special_tokens=True)))
    print(f"  generated {len(codes)} candidates [{time.time()-wall0:.0f}s]", flush=True)

    task_text = [user_prompt(t) for t in tasks]
    # 2. verification of OWN candidates: no-think + thinking
    t0 = time.time()
    pa_nt = p.judge_nothink([p.judge_prompt(task_text[i // k], codes[i], enable_thinking=False) for i in range(len(rep))])
    print(f"  judged own (no-think) [{time.time()-t0:.0f}s]", flush=True)
    t0 = time.time()
    pa_th, forced = p.judge_think([p.judge_prompt(task_text[i // k], codes[i], enable_thinking=True) for i in range(len(rep))], budget=args.budget)
    print(f"  judged own (think) [{time.time()-t0:.0f}s]", flush=True)

    # 3. foreign control: task t judges task (t+1)'s candidate (same sample slot)
    for_src = [(((i // k + 1) % n) * k + i % k) for i in range(len(rep))]
    t0 = time.time()
    fpa_nt = p.judge_nothink([p.judge_prompt(task_text[i // k], codes[for_src[i]], enable_thinking=False) for i in range(len(rep))])
    fpa_th, _ = p.judge_think([p.judge_prompt(task_text[i // k], codes[for_src[i]], enable_thinking=True) for i in range(len(rep))], budget=args.budget)
    print(f"  judged foreign (both) [{time.time()-t0:.0f}s]", flush=True)

    with (EXP / "data" / "records.jsonl").open("w") as f:
        for i in range(len(rep)):
            f.write(json.dumps({
                "task_id": tasks[i // k].task_id, "sample": i % k, "code": codes[i],
                "pa_nothink": pa_nt[i], "pa_think": pa_th[i], "forced_think": bool(forced[i]),
                "foreign_from": tasks[for_src[i] // k].task_id,
                "foreign_pa_nothink": fpa_nt[i], "foreign_pa_think": fpa_th[i],
            }) + "\n")
    print(f"done in {time.time()-wall0:.0f}s; wrote data/records.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
