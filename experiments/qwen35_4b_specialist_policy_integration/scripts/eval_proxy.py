#!/usr/bin/env python3
"""Frozen paired whitebox evaluation for one merged Qwen3.5-4B checkpoint."""

from __future__ import annotations

import argparse
import json
import math
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
from rollout import collect_policy_episodes  # noqa: E402


CORRECTION_MARKERS = (
    "check",
    "verify",
    "reconsider",
    "backtrack",
    "revise",
    "mistake",
    "instead",
)


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
    if not by_family:
        return {"family_macro_score": None, "by_family": {}}
    return {
        "family_macro_score": sum(row["mean_score"] for row in by_family.values()) / len(by_family),
        "family_macro_parse_rate": sum(row["parse_rate"] for row in by_family.values()) / len(by_family),
        "family_macro_forced_close_rate": sum(row["forced_close_rate"] for row in by_family.values()) / len(by_family),
        "by_family": by_family,
    }


def _lower_bound_entropy(logprob_rows: list | None) -> list[float]:
    """Entropy of reported top tokens with the unreported tail as one bucket."""
    values: list[float] = []
    for token_row in logprob_rows or []:
        if not isinstance(token_row, dict):
            continue
        probs = []
        for item in token_row.values():
            if isinstance(item, dict) and item.get("logprob") is not None:
                probs.append(math.exp(float(item["logprob"])))
        mass = min(1.0, sum(probs))
        tail = max(0.0, 1.0 - mass)
        entropy = -sum(p * math.log(max(p, 1e-30)) for p in probs)
        if tail > 0.0:
            entropy -= tail * math.log(tail)
        values.append(entropy)
    return values


def _behavior_diagnostics(rows: list[dict]) -> dict:
    entropies: list[float] = []
    turn_count = 0
    correction_turns = 0
    sampled_tokens = 0
    logical_input_tokens = 0
    forced_turns = 0
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_key"]].append(row)
        for turn in row["turns"]:
            turn_count += 1
            policy = turn["policy"]
            text = str(policy.get("text", "")).lower()
            correction_turns += any(marker in text for marker in CORRECTION_MARKERS)
            forced_turns += bool(policy.get("forced_close"))
            sampled_tokens += int(policy.get("n_sampled_tokens") or 0)
            logical_input_tokens += int(policy.get("n_stage1_prompt_tokens") or 0)
            logical_input_tokens += int(policy.get("n_stage2_prompt_tokens") or 0)
            entropies.extend(_lower_bound_entropy(policy.get("stage1_logprobs")))
            entropies.extend(_lower_bound_entropy(policy.get("stage2_logprobs")))
    unique_valid = []
    execution_coverage = []
    for members in by_episode.values():
        sequences = {
            tuple(turn["action"] for turn in member["turns"])
            for member in members
            if all(turn["action_ok"] for turn in member["turns"])
        }
        unique_valid.append(len(sequences))
        execution_coverage.append(any(float(member["score"]) >= 0.999 for member in members))
    return {
        "turns": turn_count,
        "sampled_tokens": sampled_tokens,
        "logical_model_input_tokens": logical_input_tokens,
        "mean_reported_top20_tail_lumped_entropy_nats": (
            sum(entropies) / len(entropies) if entropies else None
        ),
        "entropy_token_positions": len(entropies),
        "correction_marker_turn_rate": correction_turns / turn_count if turn_count else 0.0,
        "forced_close_turn_rate": forced_turns / turn_count if turn_count else 0.0,
        "mean_unique_valid_action_sequences_per_episode": (
            sum(unique_valid) / len(unique_valid) if unique_valid else 0.0
        ),
        "execution_filtered_coverage": (
            sum(execution_coverage) / len(execution_coverage) if execution_coverage else 0.0
        ),
    }


def _best_of_k(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["episode_key"]].append(row)
    by_family: dict[str, list[float]] = defaultdict(list)
    for members in groups.values():
        by_family[members[0]["family"]].append(max(float(row["score"]) for row in members))
    family_rows = {
        family: {"n": len(values), "mean_score": sum(values) / len(values)}
        for family, values in sorted(by_family.items())
    }
    return {
        "n_episode_groups": len(groups),
        "family_macro_score": (
            sum(row["mean_score"] for row in family_rows.values()) / len(family_rows)
            if family_rows else 0.0
        ),
        "by_family": family_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--scope", choices=("calibration", "confirmatory", "custom"), default="calibration")
    parser.add_argument("--decode", choices=("greedy", "sample8"), default="greedy")
    parser.add_argument("--families", nargs="*")
    parser.add_argument("--levels", type=int, nargs="*")
    parser.add_argument("--episodes-per-level", type=int)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--no-atoms", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    if not (args.model / "config.json").exists():
        raise SystemExit(f"merged checkpoint missing config.json: {args.model}")
    out_dir = args.out_dir or EXP / "runs" / "proxy_eval" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    all_process = list(config["split"]["train_families"]) + list(config["split"]["transfer_families"])
    process_families = list(args.families) if args.families else all_process
    invalid = set(process_families) - set(all_process)
    if invalid:
        raise SystemExit(f"families outside frozen process split: {sorted(invalid)}")

    if args.smoke:
        process_families = process_families[:1]
        episode_levels = [1]
        episodes_per_level = 1
    else:
        episode_levels = args.levels or [int(level) for level in config["proxy_eval"]["levels"]]
        if args.episodes_per_level is not None:
            episodes_per_level = int(args.episodes_per_level)
        elif args.scope == "confirmatory":
            episodes_per_level = int(config["proxy_eval"]["confirmatory_episodes_per_level"])
        else:
            episodes_per_level = int(config["proxy_eval"]["calibration_episodes_per_level"])
    if episodes_per_level < 1:
        raise SystemExit("episodes-per-level must be positive")

    specs = make_specs(
        process_families,
        {level: episodes_per_level for level in episode_levels},
        int(config["seeds"]["proxy_eval_base"]) + int(args.seed_offset),
    )
    atom_items = []
    if not args.no_atoms:
        atom_seed = int(config["seeds"]["atom_retention"]) + int(args.seed_offset)
        atom_levels = [1] if args.smoke else [int(level) for level in config["proxy_eval"]["atom_levels"]]
        atoms_per_level = 1 if args.smoke else int(config["proxy_eval"]["atoms_per_level"])
        for family_name in ALL_FAMILIES:
            family = load_family(family_name)
            generator = getattr(family, "gen_atoms", None)
            if not callable(generator):
                continue
            for level in atom_levels:
                if level in family.LEVELS:
                    atom_items.extend(generator(atom_seed, level, atoms_per_level))

    started = time.perf_counter()
    runner = make_runner(config["engine"], model_override=str(args.model.resolve()))
    try:
        trajectories, episode_summary = collect_policy_episodes(
            runner,
            specs,
            rollouts_per_episode=8 if args.decode == "sample8" else 1,
            think_budget=config["proxy_eval"]["thinking_budget"],
            answer_max_tokens=config["proxy_eval"]["answer_max_tokens"],
            run_seed=int(config["seeds"]["proxy_eval_base"]) + int(args.seed_offset) + 71,
            greedy=args.decode == "greedy",
            temperature=config["collection"]["temperature"],
            top_p=config["collection"]["top_p"],
            top_k=config["collection"]["top_k"],
            logprobs=20,
        )
        atom_rows = run_atoms(
            runner,
            atom_items,
            k=1,
            think_budget=config["proxy_eval"]["thinking_budget"],
            answer_max_tokens=config["proxy_eval"]["answer_max_tokens"],
            run_seed=int(config["seeds"]["atom_retention"]) + int(args.seed_offset) + 73,
            greedy=True,
        ) if atom_items else []
    finally:
        runner.close()

    merge_receipt = args.model / "merge_receipt.json"
    result = {
        "stage": "proxy_eval",
        "tag": args.tag,
        "scope": args.scope,
        "decode": args.decode,
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "model_fingerprint": {
            "config_sha256": sha256_file(args.model / "config.json"),
            "merge_receipt_sha256": sha256_file(merge_receipt) if merge_receipt.exists() else None,
            "merge_receipt": (
                json.loads(merge_receipt.read_text(encoding="utf-8"))
                if merge_receipt.exists() else None
            ),
        },
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "split": split_receipt(config, specs),
        "families": process_families,
        "levels": episode_levels,
        "episodes_per_level": episodes_per_level,
        "episode_summary": episode_summary,
        "best_of_k_summary": _best_of_k(trajectories),
        "behavior_diagnostics": _behavior_diagnostics(trajectories),
        "atom_summary": _atom_summary(atom_rows),
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
