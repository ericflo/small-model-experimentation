#!/usr/bin/env python3
"""Shared K-sample eval for one arm on the frozen eval tasks.

Arms share the identical eval: C36 ident_prompt (+ the VERBATIM scaffold text prepended for the
scaffold arm only), K sampled think passes at the pinned budget/decode (configs/default.yaml:
temperature 0.8 / top_p 0.95, budget 1024, answer_max 512) + 1 greedy pass. Each candidate's
```python block (post-</think> region) is graded: legacy C36 skeleton metric, probe-robust
skeleton metric, mimicry flag, full-solve on hidden. Also records per-sample generated tokens,
n_think, forced-close, and per-task prompt tokens (compute-parity accounting).

--nothink-anchor (base only): a cheap C36-continuity pass (no-think, K=8, answer_max=256 --
C36's historical decode) tying this frozen eval to the measured wall numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hv_common as C  # noqa: E402

EXP = C.EXP
sys.path.insert(0, str(EXP / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["base", "scaffold", "c45_zero", "dsl_sft"])
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--nothink-anchor", action="store_true",
                    help="base-only: no-think C36-continuity pass instead of the think eval")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budget-override", type=int, default=None,
                    help="pre-committed contingency: re-probe at a larger think budget")
    ap.add_argument("--depths-only", type=int, nargs="*", default=None,
                    help="restrict to these depths (contingency probes d3 only)")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="suffix for the output json (e.g. _b2048)")
    args = ap.parse_args()
    if args.nothink_anchor and args.arm != "base":
        ap.error("--nothink-anchor is base-only")
    cfg = C.load_cfg()
    sfx = C.sfx(args.smoke)
    tag = f"{args.arm}_nothink" if args.nothink_anchor else args.arm
    out_path = EXP / "runs" / f"eval_{tag}{args.out_suffix}{sfx}.json"
    if out_path.exists():
        print(f"[eval:{tag}] {out_path.name} exists; skip", flush=True)
        return

    think = not args.nothink_anchor
    if args.smoke:
        K, budget, answer_max = 2, 192, 192
    else:
        K = cfg["eval"]["K"] if think else 8
        budget = cfg["eval"]["budget"]
        answer_max = cfg["eval"]["answer_max"] if think else 256
    if args.budget_override:
        budget = args.budget_override
    batch = 32 if think else 64  # 48 OOM'd on the scaffold arm's longer prompts (2026-07-08)
    if think and budget >= 2048:
        batch = 16  # KV at 2048-think + long prompts; compute doc caps long-seq batches

    tasks = C.load_jsonl(EXP / "data" / f"eval_tasks{sfx}.jsonl")
    if args.depths_only:
        tasks = [t for t in tasks if t["depth"] in set(args.depths_only)]
        print(f"[eval:{tag}] depth filter {args.depths_only}: {len(tasks)} tasks", flush=True)
    scaffold = (EXP / "configs" / "scaffold_prompt.txt").read_text()  # VERBATIM frozen treatment
    users = []
    for t in tasks:
        fam = C.fam_of(t["family"])
        u = C.ident_prompt(fam, t)
        if args.arm == "scaffold":
            u = scaffold.strip() + "\n\n" + u
        users.append(u)

    import gen_lib as GL
    # pinned decode for the think eval (overrides gen_lib's default think sampling)
    GL.THINK_SAMPLING = dict(do_sample=True, temperature=cfg["eval"]["temperature"],
                             top_p=cfg["eval"]["top_p"], top_k=20)
    p = GL.Probe()
    if args.adapter:
        from peft import PeftModel
        p.model = PeftModel.from_pretrained(p.model, str(args.adapter)).eval()
        print(f"[eval:{tag}] adapter loaded: {args.adapter}", flush=True)
    print(f"[eval:{tag}] model ready in {p.load_secs:.0f}s | {len(tasks)} tasks | K={K} "
          f"think={think} budget={budget if think else '-'} answer_max={answer_max}", flush=True)

    prompts = [p.prompt(u, enable_thinking=think) for u in users]
    prompt_toks = [len(p._ids(pr)) for pr in prompts]

    t0 = time.time()
    gg = p.gen_sequences(prompts, think=think, budget=budget if think else None,
                         greedy=True, answer_max=answer_max, batch_size=batch)
    print(f"[eval:{tag}] greedy pass done ({time.time() - t0:.0f}s)", flush=True)
    flat, fidx = [], []
    for i, pr in enumerate(prompts):
        for _ in range(K):
            flat.append(pr)
            fidx.append(i)
    t0 = time.time()
    gs = p.gen_sequences(flat, think=think, budget=budget if think else None,
                         greedy=False, answer_max=answer_max, batch_size=batch)
    print(f"[eval:{tag}] {len(flat)} sampled passes done ({time.time() - t0:.0f}s)", flush=True)

    fills_by_task = [C.fills_with_outputs(C.fam_of(t["family"]), t) for t in tasks]
    grade_cache = {}

    def grade(i, g, pr):
        gen_ids = g["seq_ids"][len(p._ids(pr)):]
        text = p.tok.decode(gen_ids)
        code = C.extract_code(text, think_mode=think)
        key = (i, code)
        if key not in grade_cache:
            grade_cache[key] = C.grade_candidate(C.fam_of(tasks[i]["family"]), tasks[i], code,
                                                 fills_by_task[i])
        r = dict(grade_cache[key])
        r.update({"gen_tokens": len(gen_ids), "n_think": g.get("n_think", 0),
                  "forced": bool(g.get("forced", False))})
        return r

    rows = []
    for i, t in enumerate(tasks):
        rows.append({"task_id": t["task_id"], "family": t["family"], "depth": t["depth"],
                     "prompt_tokens": prompt_toks[i], "greedy": grade(i, gg[i], prompts[i]),
                     "samples": []})
    for i, pr, g in zip(fidx, flat, gs):
        rows[i]["samples"].append(grade(i, g, pr))
    print(f"[eval:{tag}] grading done ({len(grade_cache)} unique candidates sandboxed)", flush=True)

    def cov(flag):
        d3 = [r for r in rows if r["depth"] == 3]
        pool = d3 or rows
        return sum(any(s[flag] for s in r["samples"]) for r in pool) / len(pool)

    out = {"arm": args.arm, "tag": tag, "think": think, "K": K, "budget": budget if think else None,
           "answer_max": answer_max, "temperature": cfg["eval"]["temperature"] if think else None,
           "top_p": cfg["eval"]["top_p"] if think else None,
           "adapter": str(args.adapter) if args.adapter else None, "n_tasks": len(tasks),
           "rows": rows}
    out_path.parent.mkdir(exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"[eval:{tag}] depth-3 coverage@{K}: probe-robust {cov('probe_robust'):.3f} | "
          f"legacy {cov('legacy'):.3f} | full {cov('full'):.3f}", flush=True)
    print(f"[eval:{tag}] wrote {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
