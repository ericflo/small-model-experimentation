#!/usr/bin/env python3
"""Materialize the frozen non-coding contexts used by the locality audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

STEMS = (
    "A calm morning often begins with",
    "The clearest way to explain a new idea is",
    "When a garden has had too little rain,",
    "A reliable train schedule helps travelers",
    "The main reason people label storage boxes is",
    "Before serving a warm meal, a careful host",
    "A short walk after lunch can make the afternoon",
    "When two friends disagree, a useful first step is",
    "A library stays easy to use when visitors",
    "The safest response to an unfamiliar noise is",
    "A well-written recipe tells the reader",
    "When clouds gather quickly near sunset,",
)

PREFIXES = (
    "Continue this sentence naturally in a few words:\n",
    "Write the next short phrase:\n",
    "Complete the thought plainly:\n",
    "Add a concise continuation:\n",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXP / "data" / "locality_contexts.json")
    parser.add_argument("--seed", type=int, default=73500)
    args = parser.parse_args()
    contexts = []
    for prefix_index, prefix in enumerate(PREFIXES):
        for stem_index, stem in enumerate(STEMS):
            content = prefix + stem
            contexts.append({
                "id": f"locality-s{args.seed}-{prefix_index:02d}-{stem_index:02d}",
                "messages": [{"role": "user", "content": content}],
                "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            })
    payload = {
        "schema_version": 1,
        "seed": args.seed,
        "contexts": contexts,
        "count": len(contexts),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "count": len(contexts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
