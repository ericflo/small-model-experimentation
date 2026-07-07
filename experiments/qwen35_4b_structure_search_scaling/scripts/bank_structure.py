#!/usr/bin/env python3
"""Does banking install STRUCTURE? C32 showed the compositional wall is structure-proposal (the base can't
propose the depth-3 op-sequence; failures are wrong-skeleton) while values are trivially searchable once
structure is known. C22-24 showed banking crosses depth-3. So banking must install the STRUCTURE the base lacked.
Run C32's format-immune structure-coverage (does the model program's BEHAVIOR match the true op-type skeleton
with ANY params?) on BASE vs the BANKED model, on held-out depth-3 tasks (banked_1280's frozen eval, disjoint
from its training). If banked structure-coverage jumps from ~0 (base) to high, banking installs generalizable
STRUCTURE."""
from __future__ import annotations
import argparse, json, sys
from math import comb
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import families as FAM  # noqa: E402
import code_env as E  # noqa: E402
from skeleton_fill import ident_prompt, py_solves, model_structure_correct  # noqa: E402
FAM_L = FAM.FAMILIES["list"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--eval-file", type=Path, default=EXP / "data" / "eval_frozen_d3.jsonl")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in args.eval_file.read_text().splitlines() if l.strip()]
    # keep min-depth-verified true-depth-3
    keep = []
    for t in tasks:
        inp = [e["input"] for e in t["visible"] + t["hidden"]]
        out = [e["output"] for e in t["visible"] + t["hidden"]]
        if not FAM.min_depth_leq(FAM_L, inp, out, t["depth"] - 1):
            keep.append(t)
    tasks = keep
    print(f"[bank] {args.tag}: {len(tasks)} min-depth-verified tasks | adapter={args.adapter}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    if args.adapter:
        from peft import PeftModel
        p.model = PeftModel.from_pretrained(p.model, args.adapter).eval()

    prompts = [p.prompt(ident_prompt(t), enable_thinking=False) for t in tasks]
    gg = p.gen_sequences(prompts, think=False, budget=None, greedy=True, answer_max=256, batch_size=64)
    greedy, struct_greedy = [], []
    for t, pr, g in zip(tasks, prompts, gg):
        code, _ = E.extract_candidate_code(p.tok.decode(g["seq_ids"][len(p._ids(pr)):]).strip(), "transform")
        greedy.append(py_solves(code, t)); struct_greedy.append(model_structure_correct(code, t))
    flat, fidx = [], []
    for i, pr in enumerate(prompts):
        for _ in range(args.k): flat.append(pr); fidx.append(i)
    gs = p.gen_sequences(flat, think=False, budget=None, greedy=False, answer_max=256, batch_size=64)
    cov = [0] * len(tasks); struct_cov = [0] * len(tasks)
    for i, pr, g in zip(fidx, flat, gs):
        code, _ = E.extract_candidate_code(p.tok.decode(g["seq_ids"][len(p._ids(pr)):]).strip(), "transform")
        cov[i] += int(py_solves(code, tasks[i])); struct_cov[i] += int(model_structure_correct(code, tasks[i]))

    rows = [{"greedy": greedy[i], "struct_greedy": struct_greedy[i], "cov": cov[i], "struct_cov": struct_cov[i]}
            for i in range(len(tasks))]
    (EXP / "runs").mkdir(exist_ok=True)
    json.dump({"tag": args.tag, "k": args.k, "n": len(tasks), "rows": rows},
              open(EXP / "runs" / f"bank_{args.tag}.json", "w"), indent=1)

    def covk(c, n, k): k = min(k, n); return 0.0 if c == 0 else (1.0 if n-c < k else 1-comb(n-c, k)/comb(n, k))
    n = len(tasks)
    print(f"[bank] {args.tag}: greedy@1 {sum(greedy)/n:.3f} | cov@{args.k} {sum(covk(r['cov'],args.k,args.k) for r in rows)/n:.3f} "
          f"| STRUCTURE-cov@{args.k} {sum(covk(r['struct_cov'],args.k,args.k) for r in rows)/n:.3f} "
          f"| struct-greedy {sum(struct_greedy)/n:.3f}", flush=True)
    print(f"[bank] wrote runs/bank_{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
