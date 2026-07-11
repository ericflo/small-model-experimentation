#!/usr/bin/env python3
"""Build oracle-trace SFT rows (programmatic distillation, C48 precedent).

Emits think-channel rows from each residual family's hand-coded solving
procedure: think = oracle_trace(item), answer = the terse ANSWER line from
oracle_atom(item). Provenance: gym generators + hand-coded solvers only.
CPU-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gym import base  # noqa: E402
from gym.families import load as load_family  # noqa: E402

TRACE_FAMILIES = (
    "loomfix", "glyphgate", "stallwright", "packhouse",
    "patchwheel", "kilnrite", "burrowmaze",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-seed", type=int, default=17001,
                        help="fresh train-pool namespace for trace items")
    parser.add_argument("--per-level", type=int, default=40)
    parser.add_argument("--out", type=Path, default=EXP / "data" / "sft_oracle_traces.jsonl")
    args = parser.parse_args()

    rows, skipped = [], 0
    for name in TRACE_FAMILIES:
        family = load_family(name)
        for level in family.LEVELS:
            for item in family.gen_atoms(args.gen_seed, level, args.per_level):
                trace = family.oracle_trace(item)
                reply = family.oracle_atom(item)
                answer = base.extract_answer(reply)
                if answer is None or family.score_atom(item, f"{trace}\n\nANSWER: {answer}") < 1.0:
                    skipped += 1
                    continue
                rows.append({
                    "family": name,
                    "level": level,
                    "kind": "oracle_trace",
                    "messages": [{"role": "user", "content": item["prompt"]}],
                    "think": trace,
                    "answer": f"ANSWER: {answer}",
                    "n_think_tokens": max(1, len(trace) // 4),
                })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(rows)} trace rows ({skipped} skipped)",
          dict(Counter(r["family"] for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
