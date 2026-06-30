#!/usr/bin/env python3
"""Thinking-budget scaling sweep for Qwen3.5-4B on MBPP.

Measures, per thinking-token budget, the deployable line (greedy pass@1, visible-test
selector) vs the oracle ceiling (pass@k) — the oracle/deployable decomposition the
corpus applies to sample/evidence budgets but never to native thinking tokens.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import tasks as T  # noqa: E402


@dataclass
class Cond:
    name: str
    think: bool
    budget: int | None
    shuffle: bool = False

    def batch(self) -> int:
        # generate() auto-subdivides on CUDA OOM and we use expandable_segments,
        # so we can run larger batches for throughput and fall back safely.
        if not self.think:
            return 48
        b = self.budget or 4096
        if b <= 512:
            return 32
        if b <= 1024:
            return 16
        return 10


def build_conditions(budgets: list[str], controls: bool, only_controls: bool = False) -> list[Cond]:
    conds: list[Cond] = []
    if not only_controls:
        for b in budgets:
            if b == "no_think":
                conds.append(Cond("no_think", think=False, budget=None))
            elif b == "unbudgeted":
                conds.append(Cond("think_unbudgeted", think=True, budget=None))
            else:
                conds.append(Cond(f"think_{b}", think=True, budget=int(b)))
    if controls or only_controls:
        conds.append(Cond("shuffle_512", think=True, budget=512, shuffle=True))
        conds.append(Cond("shuffle_2048", think=True, budget=2048, shuffle=True))
    return conds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", type=int, default=120)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--k", type=int, default=8, help="samples for pass@k / oracle ceiling")
    ap.add_argument("--budgets", default="no_think,256,512,1024,2048,unbudgeted")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--only-controls", dest="only_controls", action="store_true",
                    help="generate only the shuffled-thinking controls")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.tasks, args.k, args.budgets, args.controls = 5, 2, "no_think,512", False
        tag = "smoke"
    else:
        tag = "full"

    out_dir = Path(args.out) if args.out else (EXP / "runs" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets = args.budgets.split(",")
    conds = build_conditions(budgets, args.controls, args.only_controls)

    tasks = T.load_mbpp(split="test", limit=args.tasks, offset=args.offset)
    print(f"[{tag}] {len(tasks)} MBPP tasks, k={args.k}, conditions={[c.name for c in conds]}", flush=True)

    # task metadata for the torch-free verifier
    (out_dir / "tasks.json").write_text(json.dumps(
        {t.task_id: {"test_list": t.test_list, "test_imports": t.test_imports} for t in tasks}))

    import runtime as RT  # imported late so --help is fast / CPU-only paths work
    rt = RT.Runtime()
    print(f"model loaded in {rt.load_secs:.0f}s", flush=True)

    def user_prompt(task: T.Task) -> str:
        anchor = task.test_list[0] if task.test_list else ""
        return (f"{task.prompt}\n\nYour function must satisfy this example:\n{anchor}\n"
                f"Define the function with the exact name used above.")

    gen_path = out_dir / "generations.jsonl"
    gen_f = gen_path.open("w")
    wall0 = time.time()

    # ---- GPU phase: generate everything, write raw records (no verification) ----
    for cond in conds:
        t0 = time.time()
        prompts_think = [rt.prompt(user_prompt(t), enable_thinking=cond.think) for t in tasks]
        greedy = rt.generate(prompts_think, think=cond.think, budget=cond.budget,
                             greedy=True, shuffle_think=cond.shuffle, batch_size=cond.batch())
        rep_prompts = [p for p in prompts_think for _ in range(args.k)]
        sampled = rt.generate(rep_prompts, think=cond.think, budget=cond.budget,
                              greedy=False, shuffle_think=cond.shuffle, batch_size=cond.batch())
        for i, t in enumerate(tasks):
            gen_f.write(json.dumps({
                "cond": cond.name, "task_id": t.task_id, "kind": "greedy", "s": 0,
                "code": T.extract_code(greedy[i].text), "n_think": greedy[i].n_think,
                "n_gen": greedy[i].n_gen, "forced": greedy[i].forced}) + "\n")
            for s in range(args.k):
                g = sampled[i * args.k + s]
                gen_f.write(json.dumps({
                    "cond": cond.name, "task_id": t.task_id, "kind": "sample", "s": s,
                    "code": T.extract_code(g.text), "n_think": g.n_think,
                    "n_gen": g.n_gen, "forced": g.forced}) + "\n")
        gen_f.flush()
        mt = sum(g.n_think for g in sampled) / max(1, len(sampled))
        tt = sum(g.n_gen for g in sampled) / max(1, len(sampled))
        ff = sum(g.forced for g in sampled) / max(1, len(sampled))
        print(f"  [gen] {cond.name:18s} think_tok={mt:.0f} tot_tok={tt:.0f} forced={ff:.2f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    gen_f.close()
    gen_secs = time.time() - wall0
    print(f"generation done in {gen_secs:.0f}s; freeing GPU and verifying...", flush=True)

    # ---- free GPU, then verify in a separate torch-free process (fork-safe) ----
    del rt
    import gc, torch  # noqa: E402
    gc.collect(); torch.cuda.empty_cache()
    import subprocess
    rc = subprocess.run([sys.executable, str(EXP / "scripts" / "verify_runs.py"),
                         "--gen", str(gen_path), "--tasks", str(out_dir / "tasks.json"),
                         "--k", str(args.k), "--out", str(out_dir / "summary.json"),
                         "--meta", json.dumps({"tag": tag, "n_tasks": len(tasks), "k": args.k,
                                               "budgets": budgets, "controls": args.controls,
                                               "model": RT.MODEL_ID, "gen_secs": round(gen_secs, 1)})]).returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
