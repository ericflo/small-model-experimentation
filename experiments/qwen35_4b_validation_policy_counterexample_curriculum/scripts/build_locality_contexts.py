#!/usr/bin/env python3
"""Materialize a fresh frozen non-coding block for the policy curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

STEMS = (
    "A ceramic mug cools more slowly when",
    "Before lending a favorite novel, its owner may",
    "On a foggy morning, harbor lights help sailors",
    "A wooden drawer slides quietly after someone",
    "When houseplants lean toward one window, a gardener",
    "A picnic blanket stays drier if the group",
    "Before tuning an old piano, the technician usually",
    "Fresh bread keeps its crust best when it",
    "During a power outage, a battery lantern can",
    "A wool scarf dries without stretching when",
    "When a chalkboard becomes dusty, the teacher",
    "Before storing winter boots for summer, people often",
)

PREFIXES = (
    "Complete the ordinary thought with a few words:\n",
    "Add one concise, natural ending to this sentence:\n",
    "Respond solely with a plain short continuation:\n",
    "Give the next sensible phrase and nothing more:\n",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXP / "data" / "locality_contexts.json")
    parser.add_argument("--seed", type=int, default=87600)
    args = parser.parse_args()
    contexts = []
    for prefix_index, prefix in enumerate(PREFIXES):
        for stem_index, stem in enumerate(STEMS):
            content = prefix + stem
            contexts.append({
                "id": f"policy-locality-s{args.seed}-{prefix_index:02d}-{stem_index:02d}",
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
