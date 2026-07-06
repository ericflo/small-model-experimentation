#!/usr/bin/env python3
"""Harvest the model's OWN execution-verified reasoning traces. banked_1280 (has depth-3 coverage) generates
GREEDY think-mode on fresh depth-3 tasks (disjoint from the frozen held-out); keep traces whose emitted code
passes visible+hidden. Emit matched training sets: T = {prompt -> <thinking> -> code}, A = {prompt -> code}
(identical prompt+code, T only adds the trace). Self-training: traces are the model's own, verified by the
interpreter (a verifier, not an external teacher)."""
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

fam = FAM.FAMILIES["list"]
PROBE = [[((i * 7 + j * 5) % 19) - 9 for j in range(4 + (i % 5))] for i in range(24)]


def parse(tops):
    return [(s.split("(")[0], int(s[s.index("(") + 1:-1]) if "(" in s else None) for s in tops]


def fsig(tops):
    out = []
    for x in PROBE:
        st = list(x)
        for op, k in parse(tops):
            st = FAM.apply_op(fam, op, k, st)
            if st is None:
                break
        out.append(repr(st))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=303)
    args = ap.parse_args()

    heldout = [json.loads(l) for l in (EXP / "data" / "eval_frozen_d3.jsonl").read_text().splitlines() if l.strip()]
    excl_f = {fsig(t["target_ops"]) for t in heldout}
    excl_o = {tuple(t["target_ops"]) for t in heldout}

    # fresh true-depth-3 tasks, disjoint from held-out
    rng = random.Random(args.seed); tid = 500_000; tasks = []
    while len(tasks) < args.pool and tid < 500_000 + 400_000:
        tid += 1
        t = FAM.make_task(fam, tid, 3, rng, k_visible=8, m_hidden=8)
        if not t or fsig(t["target_ops"]) in excl_f or tuple(t["target_ops"]) in excl_o:
            continue
        tasks.append(t)
    print(f"[harvest] {len(tasks)} fresh true-depth-3 tasks (disjoint from held-out)", flush=True)

    probe = GL.Probe()
    from peft import PeftModel
    probe.model = PeftModel.from_pretrained(probe.model, args.adapter).eval()
    print(f"[harvest] loaded {args.adapter}", flush=True)

    prompts = [probe.prompt(C.ident_prompt(fam, t), enable_thinking=True) for t in tasks]
    outs = probe.gen_sequences(prompts, think=True, budget=args.budget, greedy=True, batch_size=16)

    thoughts, answers = [], []
    n_correct = 0
    for t, pt, out in zip(tasks, prompts, outs):
        pid = probe._ids(pt)
        gen = out["seq_ids"][len(pid):]
        nt = out["n_think"]
        thinking = probe.tok.decode(gen[:nt]).strip()
        answer = probe.tok.decode(gen[nt:]).strip()
        code, _ = E.extract_candidate_code(answer, "transform")
        vis, full, _ = C.grade(code, t)
        if full and thinking:
            n_correct += 1
            prompt = C.ident_prompt(fam, t)
            thoughts.append({"prompt": prompt, "thinking": thinking, "code": code, "depth": 3})
            answers.append({"prompt": prompt, "code": code, "depth": 3})
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "harvest_thoughts.jsonl").write_text("\n".join(json.dumps(x) for x in thoughts) + "\n")
    (EXP / "data" / "harvest_answers.jsonl").write_text("\n".join(json.dumps(x) for x in answers) + "\n")
    import statistics
    tl = [len(x["thinking"]) for x in thoughts]
    print(f"[harvest] {n_correct}/{len(tasks)} greedy think-traces verified correct -> banked "
          f"{len(thoughts)} thought/answer pairs (thinking chars: median {statistics.median(tl) if tl else 0:.0f})", flush=True)


if __name__ == "__main__":
    main()
