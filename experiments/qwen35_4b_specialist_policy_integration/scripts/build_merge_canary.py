#!/usr/bin/env python3
"""Build frozen visible-prefix canaries for the incumbent installation gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gym.families import load  # noqa: E402
from io_utils import load_config, write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--out", type=Path, default=EXP / "data" / "incumbent_merge_canary.jsonl"
    )
    args = parser.parse_args()
    config, _ = load_config(args.config)
    rows = []
    seed_base = int(config["seeds"]["proxy_eval_base"]) - 5_000
    for index, family_name in enumerate(config["split"]["train_families"]):
        seed = seed_base + index * 1_000
        episode = load(family_name).Episode(seed, 2)
        rows.append(
            {
                "id": f"incumbent-canary-{family_name}",
                "messages": [
                    {"role": "system", "content": episode.system_prompt()},
                    {"role": "user", "content": episode.initial_observation()},
                ],
                "meta": {"family": family_name, "level": 2, "seed": seed},
            }
        )
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} frozen visible-prefix canaries to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
