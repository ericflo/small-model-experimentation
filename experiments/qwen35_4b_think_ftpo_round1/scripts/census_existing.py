#!/usr/bin/env python3
"""Zero-GPU loop census over existing gauntlet base-model completions.

Formalizes the design-review finding that killed the v1 loop-repair premise:
runs the mining fingerprint detector (published thresholds) over the gauntlet's
committed greedy base generations and writes runs/census_existing.json.

python3 scripts/census_existing.py
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import loopdetect  # noqa: E402

GAUNTLET_RUNS = REPO / "experiments/qwen35_4b_gauntlet_breadth_round1/runs"


def think_text(text: str) -> str:
    return text.split("</think>")[0] if "</think>" in text else text


def census_atoms(path: Path) -> dict:
    total = flagged = 0
    chars = 0
    examples = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            for output in row.get("outputs", []):
                text = think_text(output.get("text", ""))
                total += 1
                chars += len(text)
                hit = loopdetect.find_inner_repetition(text)
                if hit is not None:
                    flagged += 1
                    if len(examples) < 5:
                        examples.append({
                            "id": row.get("id"), "period": hit.period,
                            "repeats": hit.repeats, "snippet": hit.snippet[:80],
                        })
    return {"total": total, "flagged": flagged,
            "rate": flagged / max(total, 1),
            "mean_think_chars": chars / max(total, 1), "examples": examples}


def census_episode_turns(path: Path) -> dict:
    total = flagged = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            for turn in row.get("turns", []):
                text = think_text(turn.get("text", ""))
                total += 1
                if loopdetect.find_inner_repetition(text) is not None:
                    flagged += 1
    return {"total": total, "flagged": flagged, "rate": flagged / max(total, 1)}


def main() -> int:
    out = {
        "detector": "fingerprint, published thresholds (>=4 repeats, >=60 chars, period<=1024)",
        "note": ("Census over existing GREEDY base-model gym completions "
                 "(think@1024 atoms; multi-turn episode turns). This is the "
                 "empirical basis for descoping the v1 loop-repair arm at "
                 "deployed budgets; loops dominate only at 16k+ per the "
                 "verified-macro ladder."),
        "sources": {},
    }
    out["sources"]["eval_gym_base_atoms_greedy_think1024"] = census_atoms(
        GAUNTLET_RUNS / "eval_gym_base/atom_rows.jsonl.gz")
    out["sources"]["eval_gym_base_episode_turns_greedy"] = census_episode_turns(
        GAUNTLET_RUNS / "eval_gym_base/episode_rows.jsonl.gz")
    harvest = GAUNTLET_RUNS / "harvest_round1_fast"
    for shard in sorted(harvest.glob("atoms_rows*.jsonl.gz"))[:1]:
        out["sources"]["harvest_round1_atoms_temp0.8"] = census_atoms(shard)
    dest = EXP / "runs" / "census_existing.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    for name, stats in out["sources"].items():
        print(f"{name}: {stats['flagged']}/{stats['total']} ({100*stats['rate']:.2f}%)")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
