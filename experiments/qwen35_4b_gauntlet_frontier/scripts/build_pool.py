#!/usr/bin/env python3
"""Fast-data-search substrate: merge ALL cached SFT components into ONE
deduplicated pool, tagged by (kind, family, level, source), so a data-design
'recipe' is just a fast weighted sample over it -- no regeneration. This is the
'harvest once, remix fast' move: the expensive generation is already done across
~20 component datasets; we pool them and search the MIX. CPU-only, seconds."""
from __future__ import annotations
import hashlib, json
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
DATA = EXP / "data"
OUT = EXP / "data" / "cached_pool.jsonl"
# exclude already-assembled mix files that are just concatenations of others
SKIP = {"cached_pool.jsonl"}


def row_key(r):
    msg = json.dumps(r.get("messages", ""), sort_keys=True)
    return hashlib.md5((msg + "\x00" + str(r.get("think", "")) + "\x00" + str(r.get("answer", ""))).encode()).hexdigest()


def main():
    seen, pool = set(), []
    for f in sorted(DATA.glob("sft_*.jsonl")):
        if f.name in SKIP:
            continue
        src = f.stem.replace("sft_", "")
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            k = row_key(r)
            if k in seen:
                continue
            seen.add(k)
            r["_src"] = src
            r.setdefault("kind", "unknown")
            pool.append(r)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool))
    print(f"pooled {len(pool)} unique rows (from all sft_*.jsonl)")
    print("by kind:", dict(Counter(r.get("kind", "?") for r in pool).most_common(12)))
    print("by family:", dict(Counter(r.get("family", "?") for r in pool).most_common(14)))
    print("by level:", dict(sorted(Counter(r.get("level", "?") for r in pool).items(), key=lambda x: str(x[0]))))


if __name__ == "__main__":
    main()
