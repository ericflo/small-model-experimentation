#!/usr/bin/env python3
"""Gym-internal greedy@1 evaluation on held-out seeds (base or adapter).

Run under .venv-vllm. Evaluates ALL 12 families (including the two held-out
transfer families) on generation seeds disjoint from the harvest, greedy, at
the deployed think budget. Writes a compact per-cell score table.

  ../../.venv-vllm/bin/python scripts/eval_gym.py --tag base
  ../../.venv-vllm/bin/python scripts/eval_gym.py --tag round1 --adapter <dir>
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import harness  # noqa: E402
from gym.families import ALL_FAMILIES, load as load_family  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--tag", required=True, help="run label, e.g. base / round1")
    parser.add_argument("--adapter", type=str, default=None,
                        help="DEPRECATED for vLLM evals: runtime LoRA silently no-ops "
                             "on Qwen3.5-4B; use --merged")
    parser.add_argument("--merged", type=str, default=None,
                        help="merged full-checkpoint dir (scripts/merge_adapter.py)")
    parser.add_argument("--families", nargs="*", default=list(ALL_FAMILIES))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--think-budget", type=int, default=None,
                        help="override eval_gym.think_budget (e.g. 8192 for the "
                             "maxed-budget diagnostic)")
    parser.add_argument("--atoms-per-level", type=int, default=None,
                        help="override eval_gym.atoms_per_level")
    parser.add_argument("--episode-levels", nargs="*", type=int, default=None,
                        help="override eval_gym.episode_levels ([] to skip episodes)")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    eval_cfg = config["eval_gym"]
    out_dir = args.out_dir or (EXP / "runs" / f"eval_gym_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_atoms = 2 if args.smoke else int(args.atoms_per_level or eval_cfg["atoms_per_level"])
    n_episodes = 1 if args.smoke else int(eval_cfg["episodes_per_level"])
    if args.episode_levels is not None:
        episode_levels = [int(l) for l in args.episode_levels]
    else:
        episode_levels = [int(l) for l in eval_cfg.get("episode_levels", [1, 2, 3])]
    gen_seed = int(eval_cfg["gen_seed"])

    sampling = dict(
        think_budget=int(args.think_budget or eval_cfg["think_budget"]),
        answer_max_tokens=int(eval_cfg["answer_max_tokens"]),
        run_seed=gen_seed,
        greedy=True,
    )

    runner = harness.make_runner(config["engine"], adapter=args.adapter,
                                 model_override=args.merged)
    started = time.perf_counter()

    atom_items = []
    episode_specs = []
    for family_name in args.families:
        family = load_family(family_name)
        for level in family.LEVELS:
            atom_items.extend(family.gen_atoms(gen_seed, level, n_atoms))
        if getattr(family, "HAS_EPISODES", False):
            for level in episode_levels:
                if level in family.LEVELS:
                    for index in range(n_episodes):
                        episode_specs.append((family_name, level, gen_seed + 500 + index))

    print(f"[eval_gym] {len(atom_items)} atoms, {len(episode_specs)} episodes", flush=True)
    atom_rows = harness.run_atoms(runner, atom_items, k=1, **sampling)
    episode_rows = (
        harness.run_episodes(runner, episode_specs, k=1, **sampling)
        if episode_specs
        else []
    )
    elapsed = time.perf_counter() - started

    cells: dict = defaultdict(lambda: {"n": 0, "score_sum": 0.0, "parse_fail": 0,
                                       "forced_close": 0})
    for row in atom_rows:
        output = row["outputs"][0]
        cell = cells[(row["family"], "atom", row["level"])]
        cell["n"] += 1
        cell["score_sum"] += output["score"]
        cell["parse_fail"] += output["answer_value"] is None
        cell["forced_close"] += output["forced_close"]
    for row in episode_rows:
        cell = cells[(row["family"], "episode", row["level"])]
        cell["n"] += 1
        cell["score_sum"] += row["score"]
        cell["forced_close"] += sum(t["forced_close"] for t in row["turns"]) > 0

    table = {}
    for (family, kind, level), cell in sorted(cells.items()):
        table[f"{family}/{kind}/L{level}"] = {
            "n": cell["n"],
            "mean": round(cell["score_sum"] / max(1, cell["n"]), 4),
            "parse_fail": cell["parse_fail"],
            "forced_close": cell["forced_close"],
        }

    result = {
        "tag": args.tag,
        "adapter": args.adapter,
        "merged": args.merged,
        "gen_seed": gen_seed,
        "think_budget": sampling["think_budget"],
        "seconds": round(elapsed, 1),
        "cells": table,
    }
    (out_dir / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    with gzip.open(out_dir / "atom_rows.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in atom_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with gzip.open(out_dir / "episode_rows.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in episode_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(table, indent=1))
    print(f"[eval_gym] wrote {out_dir} in {elapsed:.0f}s", flush=True)
    runner.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
