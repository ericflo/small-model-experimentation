#!/usr/bin/env python3
"""Materialize a fresh frozen non-coding block for the transaction curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

STEMS = (
    "A neighborhood bakery smells busiest when",
    "To keep a bicycle chain running smoothly, a rider",
    "When the museum closes for the evening, staff members",
    "A paper map remains useful on a hike because",
    "Before hanging a framed photograph, it helps to",
    "A quiet reading room feels comfortable when visitors",
    "When a soup tastes too salty, a cook might",
    "The first warm afternoon of spring often brings",
    "A reusable water bottle is easier to clean if",
    "When a package arrives in the rain, the recipient",
    "A community garden benefits when each volunteer",
    "Before a long train journey, many travelers",
)

PREFIXES = (
    "Supply a brief natural continuation only:\n",
    "Finish this everyday sentence in a short phrase:\n",
    "Continue the thought plainly and concisely:\n",
    "Write only the next natural few words:\n",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXP / "data" / "locality_contexts.json")
    parser.add_argument("--seed", type=int, default=86900)
    args = parser.parse_args()
    contexts = []
    for prefix_index, prefix in enumerate(PREFIXES):
        for stem_index, stem in enumerate(STEMS):
            content = prefix + stem
            contexts.append({
                "id": f"transaction-locality-s{args.seed}-{prefix_index:02d}-{stem_index:02d}",
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
