#!/usr/bin/env python3
"""Build fixed target-free procedural prompts for checkpoint installation gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gym.families import load as load_family  # noqa: E402
from io_utils import load_config, write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, _ = load_config(args.config)
    seed = int(config["seeds"]["installation_canary"])
    records = []
    families = list(config["strata"]["trained_families"])
    for stratum, level, offset in (("quick", 1, 0), ("deep", 4, 10_000)):
        for family_index, family_name in enumerate(families):
            family = load_family(family_name)
            if level not in family.LEVELS:
                continue
            item = family.gen_atoms(seed + offset + family_index * 100, level, 1)[0]
            records.append(
                {
                    "id": f"canary-{stratum}-{family_name}-{item['id']}",
                    "messages": [{"role": "user", "content": item["prompt"]}],
                    "meta": {
                        "stratum": stratum,
                        "family": family_name,
                        "kind": "atom",
                        "level": level,
                    },
                }
            )
            if sum(row["meta"]["stratum"] == stratum for row in records) == 4:
                break
    if len(records) != 8:
        raise SystemExit(f"expected eight installation prompts, found {len(records)}")
    records.sort(key=lambda row: row["id"])
    write_jsonl(args.out, records)
    print(json.dumps({"rows": len(records), "seed": seed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

