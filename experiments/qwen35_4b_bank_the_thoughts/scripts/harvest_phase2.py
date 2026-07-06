#!/usr/bin/env python3
"""Phase 2: harvest the model's OWN verified reasoning, on tasks it can solve, and pair each with (a) its own
thinking, (b) a synthetic explicit plan, (c) the answer only -- all sharing ONE canonical code target. Lets us
ask, on identical tasks: does banking the model's own reasoning (T_self) beat banking answers (A_self)? And
does the model's own (likely rationalized) reasoning bank as well as an explicit plan (T_synth)?"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP / "src"))
import common as C  # noqa: E402
import families as FAM  # noqa: E402
import gen_lib as GL  # noqa: E402
import code_env as E  # noqa: E402
import synth_traces as ST  # noqa: E402

fam = FAM.FAMILIES["list"]
PROBE = [[((i * 7 + j * 5) % 19) - 9 for j in range(4 + (i % 5))] for i in range(24)]


def parse_repr(r):
    if "(" in r:
        return (r.split("(")[0], int(r[r.index("(") + 1:-1]))
    return (r, None)


def fsig(ops):
    out = []
    for x in PROBE:
        st = list(x)
        for op, k in ops:
            st = FAM.apply_op(fam, op, k, st)
            if st is None:
                break
        out.append(repr(st))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=303)
    args = ap.parse_args()

    heldout = [json.loads(l) for l in (EXP / "data" / "eval_frozen_d3.jsonl").read_text().splitlines() if l.strip()]
    excl_f = {fsig([parse_repr(r) for r in t["target_ops"]]) for t in heldout}
    excl_o = {tuple(t["target_ops"]) for t in heldout}

    rng = random.Random(args.seed); tid = 500_000; tasks = []
    while len(tasks) < args.pool and tid < 500_000 + 400_000:
        tid += 1
        t = FAM.make_task(fam, tid, 3, rng, k_visible=8, m_hidden=8)
        if not t or fsig([parse_repr(r) for r in t["target_ops"]]) in excl_f or tuple(t["target_ops"]) in excl_o:
            continue
        tasks.append(t)
    print(f"[p2] {len(tasks)} fresh true-depth-3 tasks (disjoint from held-out)", flush=True)

    probe = GL.Probe()
    from peft import PeftModel
    probe.model = PeftModel.from_pretrained(probe.model, args.adapter).eval()
    print(f"[p2] loaded {args.adapter}", flush=True)

    prompts = [probe.prompt(C.ident_prompt(fam, t), enable_thinking=True) for t in tasks]
    outs = probe.gen_sequences(prompts, think=True, budget=args.budget, greedy=True, batch_size=16)

    recs = []
    for t, pt, out in zip(tasks, prompts, outs):
        pid = probe._ids(pt)
        gen = out["seq_ids"][len(pid):]
        nt = out["n_think"]
        model_thinking = probe.tok.decode(gen[:nt]).strip()
        answer = probe.tok.decode(gen[nt:]).strip()
        code, _ = E.extract_candidate_code(answer, "transform")
        vis, full, _ = C.grade(code, t)          # did the model's OWN thinking reach a correct answer?
        if not (full and model_thinking):
            continue
        ops = [parse_repr(r) for r in t["target_ops"]]
        recs.append({"prompt": C.ident_prompt(fam, t),
                     "model_thinking": model_thinking,
                     "synth_thinking": ST.make_trace(t, ops),
                     "code": FAM.reference_code(fam, ops),      # ONE canonical code target for all arms
                     "target_ops": t["target_ops"], "depth": 3})
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "harvest_phase2.jsonl").write_text("\n".join(json.dumps(x) for x in recs) + "\n")
    import statistics
    ml = [len(x["model_thinking"]) for x in recs]
    print(f"[p2] {len(recs)}/{len(tasks)} verified self-reasoning traces | model-thinking chars median "
          f"{statistics.median(ml) if ml else 0:.0f}", flush=True)


if __name__ == "__main__":
    main()
