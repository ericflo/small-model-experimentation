#!/usr/bin/env python3
"""Frozen greedy proxy evaluation for one merged curriculum checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gym.families import ALL_FAMILIES, load as load_family  # noqa: E402
from harness import make_runner, run_atoms  # noqa: E402
from io_utils import (  # noqa: E402
    load_config,
    make_specs,
    sha256_file,
    split_receipt,
    write_json,
    write_jsonl,
)
from rollout import collect_policy_episodes, summarize_trajectories  # noqa: E402


def _atom_summary(rows: list[dict]) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    parse: dict[str, list[float]] = defaultdict(list)
    forced: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        output = row["outputs"][0]
        grouped[row["family"]].append(float(output["score"]))
        parse[row["family"]].append(float(output["answer_value"] is not None))
        forced[row["family"]].append(float(output["forced_close"]))
    by_family = {
        family: {
            "n": len(values),
            "mean_score": sum(values) / len(values),
            "parse_rate": sum(parse[family]) / len(parse[family]),
            "forced_close_rate": sum(forced[family]) / len(forced[family]),
        }
        for family, values in sorted(grouped.items())
    }
    return {
        "family_macro_score": sum(row["mean_score"] for row in by_family.values()) / len(by_family),
        "family_macro_parse_rate": sum(row["parse_rate"] for row in by_family.values()) / len(by_family),
        "family_macro_forced_close_rate": sum(row["forced_close_rate"] for row in by_family.values()) / len(by_family),
        "by_family": by_family,
    }


def _slice_macro(summary: dict, families: list[str]) -> float:
    rows = [summary["by_family"][family]["mean_score"] for family in families]
    return sum(rows) / len(rows) if rows else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    if not (args.model / "config.json").exists():
        raise SystemExit(f"merged checkpoint missing config.json: {args.model}")
    out_dir = args.out_dir or EXP / "runs" / "proxy_eval" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    train_families = list(config["split"]["train_families"])
    transfer_families = list(config["split"]["transfer_families"])
    process_families = train_families + transfer_families
    if args.smoke:
        process_families = process_families[:2]
        episode_levels = [1]
        episodes_per_level = 1
        atom_families = list(ALL_FAMILIES)[:2]
        atom_levels = [1]
        atoms_per_level = 1
    else:
        episode_levels = [int(level) for level in config["proxy_eval"]["levels"]]
        episodes_per_level = int(config["proxy_eval"]["episodes_per_level"])
        atom_families = list(ALL_FAMILIES)
        atom_levels = [int(level) for level in config["proxy_eval"]["atom_levels"]]
        atoms_per_level = int(config["proxy_eval"]["atoms_per_level"])

    specs = make_specs(
        process_families,
        {level: episodes_per_level for level in episode_levels},
        config["seeds"]["proxy_eval_base"],
    )
    atom_seed = int(config["seeds"]["atom_retention"])
    atom_items = []
    for family_name in atom_families:
        family = load_family(family_name)
        for level in atom_levels:
            if level in family.LEVELS:
                atom_items.extend(family.gen_atoms(atom_seed, level, atoms_per_level))

    started = time.perf_counter()
    runner = make_runner(config["engine"], model_override=str(args.model.resolve()))
    try:
        trajectories, episode_summary = collect_policy_episodes(
            runner,
            specs,
            rollouts_per_episode=1,
            think_budget=config["proxy_eval"]["thinking_budget"],
            answer_max_tokens=config["proxy_eval"]["answer_max_tokens"],
            run_seed=config["seeds"]["proxy_eval_base"] + 71,
            greedy=True,
        )
        atom_rows = run_atoms(
            runner,
            atom_items,
            k=1,
            think_budget=config["proxy_eval"]["thinking_budget"],
            answer_max_tokens=config["proxy_eval"]["answer_max_tokens"],
            run_seed=config["seeds"]["atom_retention"] + 73,
            greedy=True,
        )
    finally:
        runner.close()

    atom_summary = _atom_summary(atom_rows)
    episode_summary["train_family_macro_score"] = _slice_macro(
        episode_summary, [family for family in train_families if family in episode_summary["by_family"]]
    )
    episode_summary["transfer_family_macro_score"] = _slice_macro(
        episode_summary,
        [family for family in transfer_families if family in episode_summary["by_family"]],
    )
    result = {
        "stage": "proxy_eval",
        "tag": args.tag,
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "split": split_receipt(config, specs),
        "episode_summary": episode_summary,
        "atom_summary": atom_summary,
        "wall_seconds": time.perf_counter() - started,
        "smoke": bool(args.smoke),
    }
    write_jsonl(out_dir / "episode_rows.jsonl.gz", trajectories)
    write_jsonl(out_dir / "atom_rows.jsonl.gz", atom_rows)
    write_json(out_dir / "scores.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
