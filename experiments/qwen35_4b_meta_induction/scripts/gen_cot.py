#!/usr/bin/env python3
"""Generate reasoning (chain-of-thought) SFT targets that DEMONSTRATE the induction procedure for shift episodes:
find the two positions, derive the shift, apply to the query. Tests whether teaching the PROCEDURE installs
induction where answer-only (C43) only partially did -- distinguishing a serial-compute limit from a knowledge
limit."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import episode_gen as EG
EXP = Path(__file__).resolve().parents[1]


def cot(ep):
    order = ep["order"]; pos = {d: i for i, d in enumerate(order)}
    x0, y0 = ep["examples"][0]
    i, j = pos[x0], pos[y0]; k = (j - i) % 10
    q = ep["query"]; p = pos[q]; r = (p + k) % 10; ans = order[r]
    assert ans == ep["answer"]
    return (f"Look at the example {x0} -> {y0}. In the given order, {x0} is at position {i} and {y0} is at "
            f"position {j}, so the rule shifts each digit forward by ({j} - {i}) mod 10 = {k} positions. "
            f"Apply this to {q}: {q} is at position {p}, and ({p} + {k}) mod 10 = {r}, so the answer is the "
            f"digit at position {r}, which is {ans}.\nAnswer: {ans}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-train", type=int, default=4000); a = ap.parse_args()
    with open(EXP / "data" / "train_shift_cot.jsonl", "w") as f:
        for s in range(a.n_train):
            ep = EG.gen_episode("shift", s)
            f.write(json.dumps({"prompt": EG.render(ep), "target": cot(ep), "answer": ep["answer"]}) + "\n")
    print("wrote train_shift_cot.jsonl", a.n_train)
    print("--- sample CoT ---"); print(cot(EG.gen_episode("shift", 0)))
