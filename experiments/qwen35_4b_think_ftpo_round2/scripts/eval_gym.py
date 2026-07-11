#!/usr/bin/env python3
"""Gym-internal evaluation: atoms L1-L4 + episodes across all 12 families.

One engine (= one arm) per invocation; brinework/spindle are the never-
harvested held-out families and are reported separately.

  ../../.venv-vllm/bin/python scripts/eval_gym.py --arm base
  ../../.venv-vllm/bin/python scripts/eval_gym.py --arm pivot --model <merged_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import harness  # noqa: E402
from gym.families import ALL_FAMILIES, HELDOUT_FAMILIES, load as load_family  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--atoms-per-cell", type=int, default=None)
    parser.add_argument("--episodes-per-family", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    ev = cfg["eval"]
    gen_seed = int(ev["gym_eval_seed"])
    n_atoms = args.atoms_per_cell or int(ev["gym_eval_atoms_per_cell"])
    n_eps = args.episodes_per_family or int(ev["gym_eval_episodes_per_family"])

    atom_items = []
    for family in ALL_FAMILIES:
        family_mod = load_family(family)
        for level in (1, 2, 3, 4):
            for item in family_mod.gen_atoms(gen_seed + level, level, n_atoms):
                atom_items.append({**item, "family": family, "level": level})

    episode_specs = []
    for family in ALL_FAMILIES:
        family_mod = load_family(family)
        if getattr(family_mod, "HAS_EPISODES", False):
            for i in range(n_eps):
                episode_specs.append((family, 2, gen_seed + 50 + i))

    runner = harness.make_runner(cfg["engine"], model_override=args.model)
    started = time.time()
    atom_rows = harness.run_atoms(
        runner, atom_items, k=1, think_budget=1024, answer_max_tokens=512,
        run_seed=9100, greedy=True)
    episode_rows = harness.run_episodes(
        runner, episode_specs, k=1, think_budget=1024, answer_max_tokens=512,
        run_seed=9101, greedy=True)

    per_family: dict[str, list[float]] = {}
    for row in atom_rows:
        per_family.setdefault(row["family"], []).append(row["outputs"][0]["score"])
    for row in episode_rows:
        per_family.setdefault(row["family"], []).append(row["score"])

    family_means = {f: sum(v) / len(v) for f, v in sorted(per_family.items())}
    trained = [f for f in family_means if f not in HELDOUT_FAMILIES]
    heldout = [f for f in family_means if f in HELDOUT_FAMILIES]
    out = {
        "arm": args.arm, "model": args.model or "base",
        "per_family": family_means,
        "aggregate_all": sum(family_means.values()) / len(family_means),
        "aggregate_trained": sum(family_means[f] for f in trained) / len(trained),
        "aggregate_heldout": (sum(family_means[f] for f in heldout) / len(heldout))
                             if heldout else None,
        "forced_close_rate_atoms": sum(
            r["outputs"][0]["forced_close"] for r in atom_rows) / max(len(atom_rows), 1),
        "n_atoms": len(atom_rows), "n_episodes": len(episode_rows),
        "wall_s": time.time() - started,
    }
    dest = EXP / "runs" / f"eval_gym_{args.arm}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
