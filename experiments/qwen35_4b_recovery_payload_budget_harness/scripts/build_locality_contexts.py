#!/usr/bin/env python3
"""Materialize the fresh locality block for the payload-harness candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

STEMS = (
    "A ferry timetable is easiest to follow when",
    "After a heavy snowfall, neighbors often",
    "A handwritten invitation feels welcoming because it",
    "When indoor plants lean toward a window, their owner",
    "A good trail marker should remain visible even when",
    "Before lending a favorite book, a careful reader",
    "At a busy café, a clear menu helps customers",
    "A wool blanket stays in good condition if it",
    "When a local meeting runs long, the chairperson can",
    "A picnic basket is easier to carry when",
    "To remember a new neighbor's name, it can help to",
    "When morning fog begins to lift, distant buildings",
)

PREFIXES = (
    "Give the next natural phrase only:\n",
    "Complete this ordinary sentence briefly:\n",
    "Add a plain short continuation:\n",
    "Finish the thought without explanation:\n",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXP / "data" / "locality_contexts.json")
    parser.add_argument("--seed", type=int, default=85300)
    args = parser.parse_args()
    contexts = []
    for prefix_index, prefix in enumerate(PREFIXES):
        for stem_index, stem in enumerate(STEMS):
            content = prefix + stem
            contexts.append({
                "id": f"payload-locality-s{args.seed}-{prefix_index:02d}-{stem_index:02d}",
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
