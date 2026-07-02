#!/usr/bin/env python3
"""Phase 0: simulator microbenchmark (frozen model; falsification gate).

Given a STATED pipeline and one input list, the model writes the full state chain (no code).
Graded: final-output exact match (primary) + first-divergence step (diagnostic). Optionally loads a
LoRA adapter (Phase-2 reuse). Pre-registered predictions P-K0a/P-K0b in reports/prereg.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import gen_tasks as G  # noqa: E402
import gen_factorial as F  # noqa: E402
import decompose_lib as D  # noqa: E402


def steps_of(task):
    out = []
    for s in task["target_ops"]:
        op = s.split("(")[0]
        k = int(s[s.index("(") + 1:-1]) if "(" in s else None
        out.append((op, k))
    return out


def true_chain(steps, xs):
    chain = []
    st = tuple(xs)
    for op, k in steps:
        st = D.apply_prim(op, k, (st,))[0]
        chain.append(list(st))
    return chain


def sim_prompt(task, xs):
    steps = steps_of(task)
    plan = "\n".join(f"  Step {i+1}: {(f'{op}({k})' if k is not None else op)} — {D.DESC[op]}"
                     for i, (op, k) in enumerate(steps))
    return ("Apply the following sequence of list operations to the input, IN YOUR HEAD (no code).\n\n"
            f"Operations:\n{plan}\n\nInput list: {list(xs)!r}\n\n"
            "Write the resulting list after EACH step, exactly one line per step, in the form\n"
            "`Step 1: [..]` ... through the final step. No other text after the lines.")


LIST_RE = re.compile(r"\[[-0-9,\s]*\]")


def parse_chain(text, n_steps):
    lists = []
    for m in LIST_RE.finditer(text):
        try:
            lists.append(json.loads(m.group(0)))
        except Exception:
            lists.append(None)
    # take the LAST n_steps parseable lists (models sometimes restate the input first)
    lists = [l for l in lists if l is not None]
    if len(lists) >= n_steps:
        return lists[-n_steps:]
    return lists if lists else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-per-cell", type=int, default=25)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--tasks-file", type=str, default=None, help="reuse an existing task set")
    ap.add_argument("--out", type=str, default="runs/simbench.json")
    ap.add_argument("--arms", nargs="+", default=["nothink", "think"])
    args = ap.parse_args()
    cells = [(d, k) for d in (1, 2, 3, 4, 5) for k in ((0,) if d < 2 else (0, 2))]
    if args.smoke:
        cells, args.n_per_cell = [(2, 0), (4, 2)], 3

    if args.tasks_file:
        tasks = [json.loads(l) for l in Path(EXP / args.tasks_file).read_text().splitlines()]
    else:
        # verification unnecessary: simulation difficulty = STATED pipeline length
        tasks = F.build_grid(cells, args.n_per_cell, seed=555, verify_depth=False)
        (EXP / "data").mkdir(exist_ok=True)
        (EXP / "data" / "simbench_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    print(f"{len(tasks)} tasks over {cells}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    if args.adapter:
        from peft import PeftModel
        p.model = PeftModel.from_pretrained(p.model, args.adapter)
        p.model.eval()
    tag = args.adapter or "frozen"
    print(f"model loaded ({tag})", flush=True)

    results = {}
    t0 = time.time()
    for arm in args.arms:
        think = arm == "think"
        prompts, truths, metas = [], [], []
        for t in tasks:
            xs = t["visible"][0]["input"]
            steps = steps_of(t)
            prompts.append(p.prompt(sim_prompt(t, xs), enable_thinking=think))
            truths.append(true_chain(steps, xs))
            metas.append(t)
        gens = p.gen_sequences(prompts, think=think, budget=args.budget if think else None,
                               greedy=True, answer_max=400, batch_size=24)
        recs = []
        for pr, g, truth, t in zip(prompts, gens, truths, metas):
            txt = p.tok.decode(g["seq_ids"][len(p._ids(pr)):], skip_special_tokens=False)
            if "</think>" in txt:
                txt = txt.split("</think>")[-1]
            parsed = parse_chain(txt, len(truth))
            out_ok = bool(parsed) and parsed[-1] == truth[-1]
            div = None
            if parsed and len(parsed) == len(truth):
                for i, (a, b) in enumerate(zip(parsed, truth)):
                    if a != b:
                        div = i + 1
                        break
            recs.append({"task_id": t["task_id"], "depth": t["depth"], "n_destr": t["n_destr"],
                         "output_ok": out_ok, "first_div": div, "parsed": bool(parsed)})
        results[arm] = recs
        print(f"  {arm} done [{time.time()-t0:.0f}s]", flush=True)

    (EXP / "runs").mkdir(exist_ok=True)
    Path(EXP / args.out).write_text(json.dumps({"tag": tag, "results": results}, indent=1))
    print(f"\n=== SIMULATOR MICROBENCHMARK ({tag}; output exact-match) ===")
    for arm, recs in results.items():
        by = defaultdict(lambda: [0, 0])
        for r in recs:
            c = by[(r["depth"], r["n_destr"])]; c[0] += 1; c[1] += int(r["output_ok"])
        row = "  ".join(f"d{d}k{k}:{s/n:.2f}" for (d, k), (n, s) in sorted(by.items()))
        print(f"  {arm:8s} {row}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
