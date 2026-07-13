#!/usr/bin/env python3
"""Focused induction+exploration oracle-trace SFT (the C55 retain-delta probe).

The maxed-budget diagnostic localized the wall: at think budget 8192 the base
model does single-rule induction fine (glyphgate L1-L3) but hits a hard 0.0
floor at composed-rule induction (L4-L6), and the broad `apex` install HURTS
induction. This builder concentrates the training signal exactly there: oracle
hypothesize-and-verify traces for glyphgate (active induction) and burrowmaze
(exploration), heavily weighted to the hard levels where base fails, plus a
modest broad replay slice so co-training-from-base does not forget general
ability. Provenance: gym generators + hand-coded solvers only (no larger model).
Every emitted trace is self-verified to score 1.0. CPU-only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gym import base  # noqa: E402
from gym.families import load as load_family  # noqa: E402

# Concentrate on the levels where base fails at 8192 (L4-L6 = 0.0; L3 shaky).
LEVEL_WEIGHTS = {1: 15, 2: 25, 3: 60, 4: 110, 5: 110, 6: 110}
FOCUS_FAMILIES = ("glyphgate", "burrowmaze")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-seed", type=int, default=41001, help="fresh train-pool namespace")
    ap.add_argument("--families", nargs="*", default=list(FOCUS_FAMILIES))
    ap.add_argument("--replay-n", type=int, default=900,
                    help="broad sft_blend rows mixed in to preserve general ability")
    ap.add_argument("--replay-src", type=Path, default=EXP / "data" / "sft_blend.jsonl")
    ap.add_argument("--out", type=Path, default=EXP / "data" / "sft_induction.jsonl")
    args = ap.parse_args()

    rng = random.Random(41001)
    rows, skipped = [], 0
    for name in args.families:
        family = load_family(name)
        for level in family.LEVELS:
            n = LEVEL_WEIGHTS.get(level, 40)
            for item in family.gen_atoms(args.gen_seed, level, n):
                trace = family.oracle_trace(item)
                answer = base.extract_answer(family.oracle_atom(item))
                if answer is None or family.score_atom(item, f"{trace}\n\nANSWER: {answer}") < 1.0:
                    skipped += 1
                    continue
                # Up-weight the hard levels the model actually fails on, so the
                # loss concentrates on the composed-rule induction wall.
                weight = 1.0 + 0.25 * max(0, level - 3)  # L1-3:1.0, L4:1.25, L5:1.5, L6:1.75
                rows.append({
                    "family": name, "level": level, "kind": "induction_trace",
                    "messages": [{"role": "user", "content": item["prompt"]}],
                    "think": trace, "answer": f"ANSWER: {answer}",
                    "n_think_tokens": max(1, len(trace) // 4),
                    "row_weight": round(weight, 3),
                })

    replay = load_jsonl(args.replay_src)
    rng.shuffle(replay)
    replay = replay[: args.replay_n]
    for r in replay:
        r.setdefault("row_weight", 1.0)
        r["kind"] = r.get("kind", "replay")
    all_rows = rows + replay
    rng.shuffle(all_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as h:
        for r in all_rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"induction rows: {len(rows)} (skipped {skipped}) + replay {len(replay)} = {len(all_rows)}")
    print("by family:", dict(Counter(r["family"] for r in rows)))
    print("by level:", dict(sorted(Counter(r["level"] for r in rows).items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
