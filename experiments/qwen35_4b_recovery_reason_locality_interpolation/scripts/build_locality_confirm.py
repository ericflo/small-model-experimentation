#!/usr/bin/env python3
"""Materialize the frozen, independent locality-confirmation contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

STEMS = (
    "At a neighborhood market, fresh fruit usually",
    "A patient teacher can make a difficult lesson",
    "When the wind becomes stronger near the coast,",
    "A museum map is most helpful when it",
    "Before planting seeds in early spring, gardeners",
    "A shared calendar prevents confusion by",
    "When a bicycle chain begins to squeak, the rider",
    "A thoughtful thank-you note often mentions",
    "To keep a small room feeling orderly, it helps to",
    "During a long journey, regular breaks allow passengers",
    "A community notice should make its main point",
    "When bread has finished baking, a careful cook",
)

PREFIXES = (
    "Supply a natural brief ending:\n",
    "Continue in plain everyday language:\n",
    "Finish this sentence with a short phrase:\n",
    "Write only a concise continuation:\n",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXP / "data" / "locality_confirm.json")
    parser.add_argument("--seed", type=int, default=85200)
    args = parser.parse_args()
    contexts = []
    for prefix_index, prefix in enumerate(PREFIXES):
        for stem_index, stem in enumerate(STEMS):
            content = prefix + stem
            contexts.append({
                "id": f"locality-confirm-s{args.seed}-{prefix_index:02d}-{stem_index:02d}",
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
