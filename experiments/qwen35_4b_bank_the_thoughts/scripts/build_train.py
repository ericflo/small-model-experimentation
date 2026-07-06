#!/usr/bin/env python3
"""Build the matched training sets from the harvest. A and T share IDENTICAL {prompt, code}; T adds the
model's own reasoning trace. T_corrupt is the content-causality ablation: same code, but each prompt is paired
with a DIFFERENT task's thinking (so the reasoning is fluent but wrong-for-this-task) -- if T ~= T_corrupt, the
thinking content is inert ritual and only its presence/length mattered."""
from __future__ import annotations

import json
import random
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def main():
    thoughts = [json.loads(l) for l in (EXP / "data" / "harvest_thoughts.jsonl").read_text().splitlines() if l.strip()]
    answers = [json.loads(l) for l in (EXP / "data" / "harvest_answers.jsonl").read_text().splitlines() if l.strip()]
    assert len(thoughts) == len(answers)
    # verify A and T share byte-identical {prompt, code}
    for a, t in zip(answers, thoughts):
        assert a["prompt"] == t["prompt"] and a["code"] == t["code"], "A/T target mismatch"

    # A = answers (no thinking key -> trainer uses no-think target)
    (EXP / "data" / "train_A.jsonl").write_text("\n".join(json.dumps(x) for x in answers) + "\n")
    # T = thoughts (thinking key -> trainer uses think target)
    (EXP / "data" / "train_T.jsonl").write_text("\n".join(json.dumps(x) for x in thoughts) + "\n")
    # T_corrupt = same code, mismatched thinking (derangement)
    n = len(thoughts)
    rng = random.Random(13)
    perm = list(range(n))
    for _ in range(200):
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            break
    corrupt = [{"prompt": thoughts[i]["prompt"], "thinking": thoughts[perm[i]]["thinking"],
                "code": thoughts[i]["code"], "depth": 3} for i in range(n)]
    (EXP / "data" / "train_Tcorrupt.jsonl").write_text("\n".join(json.dumps(x) for x in corrupt) + "\n")

    import statistics
    tl = [len(x["thinking"]) for x in thoughts]
    print(f"built: train_A={len(answers)} train_T={len(thoughts)} train_Tcorrupt={len(corrupt)} pairs "
          f"(A/T code byte-identical) | thinking chars: median {statistics.median(tl):.0f} mean {statistics.mean(tl):.0f}")


if __name__ == "__main__":
    main()
