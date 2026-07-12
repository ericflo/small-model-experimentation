#!/usr/bin/env python3
"""Assemble the 'effskin' breakthrough arm dataset.

Three complementary signals against the C53 serial-compute ceiling:
  * EFFICIENCY (compression): the model's own shortest correct traces on
    hard residual items + brevity-advantage + wrong-answer contrast rows
    (data/sft_efficiency.jsonl). Attacks the ~63% solvable-but-too-slow.
  * SKIN-SHUFFLE: oracle procedure traces with FRESH pseudo-vocabulary on
    every row (data/sft_skinshuffle.jsonl). Forces mechanic-binding and
    supplies the procedure for the ~37% unreachable-at-K tail.
  * REPLAY: a stability slice of the blend recipe so warm-start does not
    forget the emission policy.
CPU-only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-n", type=int, default=1100)
    ap.add_argument("--out", type=Path, default=EXP / "data" / "sft_effskin.jsonl")
    args = ap.parse_args()

    rng = random.Random(31337)
    efficiency = load(EXP / "data" / "sft_efficiency.jsonl")
    skin = load(EXP / "data" / "sft_skinshuffle.jsonl")
    blend = load(EXP / "data" / "sft_blend.jsonl")
    rng.shuffle(blend)
    replay = blend[: args.replay_n]

    rows = efficiency + skin + replay
    rng.shuffle(rows)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print("effskin rows:", len(rows),
          "| efficiency:", len(efficiency), "skin:", len(skin), "replay:", len(replay))
    print("kinds:", dict(Counter(r.get("kind", "replay") for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
