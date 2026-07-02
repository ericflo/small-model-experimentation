#!/usr/bin/env python3
"""Phase 1 data (CPU): SIM (pipeline+input -> state chain) vs PROD (I/O examples -> reference code).
Matched token counts; depths 1-3; held-out primitives {rotate_k, dedup_adjacent, running_max} excluded
from BOTH arms' training pipelines (they appear only in Phase-2 held-out evals)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import gen_tasks as G  # noqa: E402
import gen_factorial as F  # noqa: E402
from run_simbench import steps_of, true_chain, sim_prompt  # noqa: E402

HELD_OUT = {"rotate_k", "dedup_adjacent", "running_max"}


def build_tasks(n_per_depth, seed):
    rng = random.Random(seed)
    tasks, tid = [], 0
    while tid < n_per_depth * 3:
        d = 1 + tid % 3
        k = rng.randint(0, min(d, 2))
        t = F.make_controlled_task(tid, d, k, rng, k_visible=8, m_hidden=6)
        if t is None:
            continue
        if any(op.split("(")[0] in HELD_OUT for op in t["target_ops"]):
            continue
        tasks.append(t)
        tid += 1
    return tasks


def main():
    n_per_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    tasks = build_tasks(n_per_depth, seed=808)
    sim, prod = [], []
    for t in tasks:
        xs = t["visible"][0]["input"]
        chain = true_chain(steps_of(t), xs)
        target = "\n".join(f"Step {i+1}: {st!r}" for i, st in enumerate(chain))
        sim.append({"task_id": t["task_id"], "depth": t["depth"],
                    "prompt": sim_prompt(t, xs), "target": target})
        prod.append({"task_id": t["task_id"], "depth": t["depth"],
                     "prompt": G.prompt_for(t), "target": "```python\n" + G.reference_code(t) + "\n```"})
    # token matching: trim the larger arm to within ~2%
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)

    def toks(rs):
        return sum(len(tok(r["prompt"] + r["target"], add_special_tokens=False).input_ids) for r in rs)

    ts, tp = toks(sim), toks(prod)
    if ts > tp:
        while toks(sim) > tp * 1.02 and len(sim) > 10:
            sim.pop()
    else:
        while toks(prod) > ts * 1.02 and len(prod) > 10:
            prod.pop()
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "train_sim.jsonl").write_text("\n".join(json.dumps(r) for r in sim) + "\n")
    (EXP / "data" / "train_prod.jsonl").write_text("\n".join(json.dumps(r) for r in prod) + "\n")
    print(f"SIM: {len(sim)} records, {toks(sim)} tokens | PROD: {len(prod)} records, {toks(prod)} tokens")


if __name__ == "__main__":
    main()
