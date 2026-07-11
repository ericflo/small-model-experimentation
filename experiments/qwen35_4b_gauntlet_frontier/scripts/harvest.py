#!/usr/bin/env python3
"""Harvest stage: sample the model on gym tasks and score everything.

Run under .venv-vllm:
  ../../.venv-vllm/bin/python scripts/harvest.py --stage both [--smoke]

Writes gzip JSONL row files plus a yield summary under --out-dir.
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
from gym.families import load as load_family  # noqa: E402


def write_gz_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def atom_yield_summary(rows: list[dict]) -> dict:
    cells: dict[tuple, dict] = defaultdict(
        lambda: {"items": 0, "samples": 0, "correct": 0, "natural_close": 0,
                 "correct_and_closed": 0, "items_with_keeper": 0}
    )
    for row in rows:
        cell = cells[(row["family"], row["level"])]
        cell["items"] += 1
        keeper = False
        for output in row["outputs"]:
            cell["samples"] += 1
            ok = output["score"] >= 1.0
            closed = not output["forced_close"]
            cell["correct"] += ok
            cell["natural_close"] += closed
            if ok and closed:
                cell["correct_and_closed"] += 1
                keeper = True
        cell["items_with_keeper"] += keeper
    return {f"{fam}-L{lvl}": stats for (fam, lvl), stats in sorted(cells.items())}


def episode_yield_summary(rows: list[dict]) -> dict:
    cells: dict[tuple, dict] = defaultdict(
        lambda: {"rollouts": 0, "success": 0, "score_sum": 0.0, "turns_sum": 0,
                 "forced_close_turns": 0, "total_turns": 0}
    )
    for row in rows:
        cell = cells[(row["family"], row["level"])]
        cell["rollouts"] += 1
        cell["success"] += row["score"] >= 1.0
        cell["score_sum"] += row["score"]
        cell["turns_sum"] += row["n_turns"]
        for turn in row["turns"]:
            cell["total_turns"] += 1
            cell["forced_close_turns"] += turn["forced_close"]
    out = {}
    for (fam, lvl), cell in sorted(cells.items()):
        out[f"{fam}-L{lvl}"] = {
            **{k: v for k, v in cell.items() if k != "score_sum"},
            "mean_score": round(cell["score_sum"] / max(1, cell["rollouts"]), 4),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--stage", choices=["atoms", "episodes", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--adapter", type=str, default=None,
                        help="DEPRECATED for vLLM: runtime LoRA no-ops on Qwen3.5-4B (C49); use --merged")
    parser.add_argument("--merged", type=str, default=None,
                        help="harvest with a merged composite checkpoint (later rounds)")
    parser.add_argument("--smoke", action="store_true", help="tiny counts, fast shakeout")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    harvest_cfg = config["harvest"]
    out_dir = args.out_dir or (EXP / "runs" / f"harvest_round{config['round']}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sampling_common = dict(
        answer_max_tokens=int(harvest_cfg["answer_max_tokens"]),
        run_seed=int(harvest_cfg["run_seed"]),
        temperature=float(harvest_cfg["temperature"]),
        top_p=float(harvest_cfg["top_p"]),
        top_k=int(harvest_cfg["top_k"]),
    )
    atom_think = int(harvest_cfg["think_budget"])
    episode_think = int(harvest_cfg.get("episode_think_budget", harvest_cfg["think_budget"]))
    think_by_family = {
        str(name): int(budget)
        for name, budget in (harvest_cfg.get("think_budget_by_family") or {}).items()
    }

    runner = harness.make_runner(config["engine"], adapter=args.adapter,
                                 model_override=args.merged)
    summary: dict = {"config": str(args.config), "smoke": args.smoke,
                     "adapter": args.adapter, "merged": args.merged}

    if args.stage in ("atoms", "both"):
        atoms_cfg = harvest_cfg["atoms"]
        per_level = {int(k): int(v) for k, v in atoms_cfg["per_level"].items()}
        if args.smoke:
            per_level = {level: min(4, count) for level, count in per_level.items()}
        k = 2 if args.smoke else int(atoms_cfg["k"])
        summary["atoms"] = {"k": k, "families": {}}
        # One shard per family: incremental writes + resumability. A shard
        # whose output file already exists is skipped.
        by_family_plans = {
            str(name): {int(k): int(v) for k, v in plan.items()}
            for name, plan in (atoms_cfg.get("per_level_by_family") or {}).items()
        }
        for family_name in atoms_cfg["families"]:
            shard_path = out_dir / f"atoms_rows_{family_name}.jsonl.gz"
            if shard_path.exists():
                print(f"[harvest] atoms/{family_name}: exists, skipping", flush=True)
                continue
            family = load_family(family_name)
            family_plan = by_family_plans.get(family_name, per_level)
            if args.smoke:
                family_plan = {level: min(4, count) for level, count in family_plan.items()}
            items = []
            for level, count in family_plan.items():
                items.extend(family.gen_atoms(int(atoms_cfg["gen_seed"]), level, count))
            family_think = think_by_family.get(family_name, atom_think)
            print(f"[harvest] atoms/{family_name}: {len(items)} items x k={k} "
                  f"(think {family_think})", flush=True)
            started = time.perf_counter()
            rows = harness.run_atoms(runner, items, k=k, think_budget=family_think, **sampling_common)
            elapsed = time.perf_counter() - started
            write_gz_jsonl(shard_path, rows)
            shard_summary = {
                "items": len(items), "seconds": round(elapsed, 1),
                "yield": atom_yield_summary(rows),
            }
            summary["atoms"]["families"][family_name] = shard_summary
            print(f"[harvest] atoms/{family_name} done in {elapsed:.0f}s: "
                  + json.dumps(shard_summary["yield"]), flush=True)

    if args.stage in ("episodes", "both"):
        episodes_cfg = harvest_cfg["episodes"]
        per_level = {int(k): int(v) for k, v in episodes_cfg["per_level"].items()}
        if args.smoke:
            per_level = {level: min(3, count) for level, count in per_level.items()}
        k = 2 if args.smoke else int(episodes_cfg["k"])
        summary["episodes"] = {"k": k, "families": {}}
        ep_by_family_plans = {
            str(name): {int(k): int(v) for k, v in plan.items()}
            for name, plan in (episodes_cfg.get("per_level_by_family") or {}).items()
        }
        for family_name in episodes_cfg["families"]:
            shard_path = out_dir / f"episodes_rows_{family_name}.jsonl.gz"
            if shard_path.exists():
                print(f"[harvest] episodes/{family_name}: exists, skipping", flush=True)
                continue
            family_plan = ep_by_family_plans.get(family_name, per_level)
            if args.smoke:
                family_plan = {level: min(3, count) for level, count in family_plan.items()}
            specs = []
            for level, count in family_plan.items():
                for index in range(count):
                    specs.append((family_name, level, int(episodes_cfg["seed_base"]) + index))
            print(f"[harvest] episodes/{family_name}: {len(specs)} specs x k={k}", flush=True)
            started = time.perf_counter()
            rows = harness.run_episodes(
                runner, specs, k=k, think_budget=episode_think, **sampling_common
            )
            elapsed = time.perf_counter() - started
            write_gz_jsonl(shard_path, rows)
            shard_summary = {
                "specs": len(specs), "seconds": round(elapsed, 1),
                "yield": episode_yield_summary(rows),
            }
            summary["episodes"]["families"][family_name] = shard_summary
            print(f"[harvest] episodes/{family_name} done in {elapsed:.0f}s: "
                  + json.dumps(shard_summary["yield"]), flush=True)

    (out_dir / "harvest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[harvest] wrote {out_dir}", flush=True)
    runner.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
