#!/usr/bin/env python3
"""Add the FILLER (pause-token / dot-by-dot) arm to the content-vs-compute ladder.

Reuses the existing run's data: per (task, sample) we know the real thinking length (n_think); the
filler condition prepends that many contentless "." tokens inside <think>, forces </think>, and
regenerates the answer. This isolates PURE forward compute + scaffold (no content to follow), the
piece foreign (misleading content) could not. Appends filler records + activations; re-verify + probe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import tasks as T  # noqa: E402

ACTS_DIR = EXP.parents[1] / "large_artifacts" / "qwen35_4b_thinking_content_vs_compute"


def user_prompt(t):
    anchor = t.test_list[0] if t.test_list else ""
    return (f"{t.prompt}\n\nYour function must satisfy this example:\n{anchor}\n"
            f"Define the function with the exact name used above.")


def main() -> int:
    # real thinking lengths per row, from the existing records
    recs = [json.loads(l) for l in (EXP / "data" / "records.jsonl").read_text().splitlines() if l.strip()]
    if any(r["cond"] == "filler" for r in recs):
        print("filler already present; remove its records to regenerate"); return 0
    real = {r["row"]: r for r in recs if r["cond"] == "real"}
    n = max(r["task_id"] for r in recs)  # not used; tasks loaded below
    tasks = T.load_mbpp(split="test", limit=100)
    k = 8
    assert len(real) == len(tasks) * k, f"expected {len(tasks)*k} real rows, got {len(real)}"

    import ladder_lib as LL
    p = LL.Probe()
    print(f"model loaded in {p.load_secs:.0f}s", flush=True)
    dot = p.tok(".", add_special_tokens=False).input_ids[-1]
    print("filler token id ('.'):", dot, flush=True)

    think_prompts = [p.prompt(user_prompt(t), enable_thinking=True) for t in tasks]
    prompt_ids = [p._ids(think_prompts[i // k]) for i in range(len(tasks) * k)]

    # filler prefix: prompt + n_think dots + </think>
    prefixes = []
    for row in range(len(tasks) * k):
        cnt = int(real[row]["n_think"])
        prefixes.append(prompt_ids[row] + [dot] * cnt + p.close_ids)
    filler = p.gen_answer(prefixes, batch_size=48)
    seqs = [g["seq_ids"] for g in filler]
    acts = p.activations(seqs, batch_size=8)
    np.save(ACTS_DIR / "acts_filler.npy", acts)
    print(f"filler acts {acts.shape}", flush=True)

    with (EXP / "data" / "records.jsonl").open("a") as f:
        for i, t in enumerate(tasks):
            for s in range(k):
                row = i * k + s
                code = T.extract_code(p.tok.decode(seqs[row], skip_special_tokens=False))
                f.write(json.dumps({"cond": "filler", "task_id": t.task_id, "sample": s, "row": row,
                                    "code": code, "n_think": int(real[row]["n_think"]),
                                    "seq_len": len(seqs[row])}) + "\n")
    print("appended filler records", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
