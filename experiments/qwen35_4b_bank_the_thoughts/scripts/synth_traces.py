#!/usr/bin/env python3
"""Phase 1: SYNTHETIC forward-decomposition traces (not the model's own thoughts -- Phase 2 does those).
For each execution-verified depth-3 op-sequence, build a genuine forward plan: input -> op1 -> state -> op2 ->
state -> op3 -> output, then the code. These are real PLANS (not rationalizations), short, and free. Emits
matched harvest_thoughts.jsonl (T: prompt->thinking->code) and harvest_answers.jsonl (A: prompt->code), so the
existing build_train.py / train_lora_think.py / eval pipeline runs unchanged."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP / "src"))
import common as C  # noqa: E402
import families as FAM  # noqa: E402

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


def make_trace(task, ops):
    """Forward-decomposition narration on the first 2 visible examples."""
    lines = ["I need to find a sequence of operations mapping each input to its output. Let me trace the "
             "transformation step by step on the examples."]
    for ex in task["visible"][:2]:
        st = list(ex["input"])
        lines.append(f"\nInput: {st}")
        for op, k in ops:
            st = FAM.apply_op(fam, op, k, list(st))
            lines.append(f"  apply {FAM.op_repr(op, k)} -> {st}")
        lines.append(f"  target: {ex['output']}  (matches)")
    lines.append("\nSo the operations, in order, are: " + ", ".join(FAM.op_repr(o, k) for o, k in ops) + ".")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=404)
    args = ap.parse_args()

    heldout = [json.loads(l) for l in (EXP / "data" / "eval_frozen_d3.jsonl").read_text().splitlines() if l.strip()]
    excl_f = {fsig([parse_repr(r) for r in t["target_ops"]]) for t in heldout}
    excl_o = {tuple(t["target_ops"]) for t in heldout}

    rng = random.Random(args.seed); tid = 700_000
    thoughts, answers = [], []
    while len(thoughts) < args.n and tid < 700_000 + 400_000:
        tid += 1
        t = FAM.make_task(fam, tid, 3, rng, k_visible=8, m_hidden=8)
        if not t:
            continue
        ops = [parse_repr(r) for r in t["target_ops"]]
        if fsig(ops) in excl_f or tuple(t["target_ops"]) in excl_o:
            continue
        code = FAM.reference_code(fam, ops)
        vis, full, _ = C.grade(code, t)   # sanity: reference code passes its own task
        if not full:
            continue
        prompt = C.ident_prompt(fam, t)
        thoughts.append({"prompt": prompt, "thinking": make_trace(t, ops), "code": code, "depth": 3})
        answers.append({"prompt": prompt, "code": code, "depth": 3})
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "harvest_thoughts.jsonl").write_text("\n".join(json.dumps(x) for x in thoughts) + "\n")
    (EXP / "data" / "harvest_answers.jsonl").write_text("\n".join(json.dumps(x) for x in answers) + "\n")
    import statistics
    tl = [len(x["thinking"]) for x in thoughts]
    print(f"[synth] {len(thoughts)} matched decomposition traces (disjoint from held-out) | "
          f"thinking chars median {statistics.median(tl):.0f}")


if __name__ == "__main__":
    main()
