#!/usr/bin/env python3
"""Collect grouped on-policy trajectories and exact terminal execution rewards."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from curriculum import semantic_group_diagnostics  # noqa: E402
from harness import make_runner  # noqa: E402
from io_utils import (  # noqa: E402
    load_config,
    make_specs,
    resolve_repo_path,
    sha256_file,
    split_receipt,
    domain_families,
    training_seed,
    write_json,
    write_jsonl,
)
from rollout import (  # noqa: E402
    collect_expert_demonstrations,
    collect_policy_episodes,
    summarize_trajectories,
    visited_dagger_rows,
)


def _cap_anchor_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], int(row["level"]))].append(row)
    rng = random.Random(seed)
    queues = []
    for key in sorted(grouped):
        values = grouped[key]
        rng.shuffle(values)
        queues.append(deque(values))
    output = []
    while queues and len(output) < n:
        next_queues = []
        for queue in queues:
            if queue and len(output) < n:
                output.append(queue.popleft())
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--domain", choices=("discover", "control", "tools", "compose", "joint"),
        default="joint",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--anchor-out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    artifacts_root = resolve_repo_path(config["model"]["artifacts_root"])
    model_path = (args.model or artifacts_root / "merged" / "dagger").resolve()
    if not (model_path / "config.json").exists():
        raise SystemExit(f"merged DAgger checkpoint missing: {model_path}")

    families = domain_families(config, args.domain)
    domain_index = ["discover", "control", "tools", "compose", "joint"].index(args.domain)
    seed_offset = domain_index * 10_000_000
    output_dir = args.output_dir or EXP / "runs" / "rl_collection" / args.domain
    anchor_out = args.anchor_out or EXP / "data" / f"rl_anchor_{args.domain}.jsonl"
    per_level = {1: 1} if args.smoke else config["rl_collection"]["per_level"]
    use_families = families[:2] if args.smoke else families
    specs = make_specs(
        use_families, per_level, config["seeds"]["rl_collect_base"] + seed_offset
    )

    runner = make_runner(config["engine"], model_override=str(model_path))
    try:
        trajectories, _ = collect_policy_episodes(
            runner,
            specs,
            rollouts_per_episode=(2 if args.smoke else config["rl_collection"]["group_size"]),
            think_budget=config["rl_collection"]["thinking_budget"],
            answer_max_tokens=config["rl_collection"]["answer_max_tokens"],
            run_seed=config["seeds"]["rl_collect_base"] + seed_offset + 37,
            greedy=False,
            temperature=config["rl_collection"]["temperature"],
            top_p=config["rl_collection"]["top_p"],
            top_k=config["rl_collection"]["top_k"],
        )
    finally:
        runner.close()

    expert_trajectories, _ = collect_expert_demonstrations(specs)
    oracle_turns = {row["episode_key"]: int(row["n_turns"]) for row in expert_trajectories}
    penalty = float(config["rl_collection"]["success_turn_penalty"])
    for trajectory in trajectories:
        score = float(trajectory["score"])
        minimum = oracle_turns[trajectory["episode_key"]]
        excess = max(0, int(trajectory["n_turns"]) - minimum)
        reward = score * max(0.0, 1.0 - penalty * excess) if score > 0.0 else 0.0
        trajectory["terminal_score"] = score
        trajectory["oracle_min_turns"] = minimum
        trajectory["success_excess_turns"] = excess if score > 0.0 else 0
        trajectory["reward"] = reward

    groups: dict[str, list[dict]] = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory["episode_key"]].append(trajectory)
    min_std = float(config["rl_collection"]["min_outcome_std"])
    group_rows = []
    for key in sorted(groups):
        members = groups[key]
        rewards = [float(row["reward"]) for row in members]
        mean = statistics.fmean(rewards)
        std = statistics.pstdev(rewards)
        active = std >= min_std
        for row in members:
            row["group_reward_mean"] = mean
            row["group_reward_std"] = std
            row["advantage"] = (float(row["reward"]) - mean) / std if active else 0.0
            row["advantage_active"] = active
        diagnostic_members = [dict(row, score=row["reward"]) for row in members]
        diagnostic = semantic_group_diagnostics(diagnostic_members)
        diagnostic.update(
            {
                "episode_key": key,
                "family": members[0]["family"],
                "level": members[0]["level"],
                "ep_seed": members[0]["ep_seed"],
                "reward_mean": mean,
                "reward_std": std,
                "advantage_active": active,
            }
        )
        group_rows.append(diagnostic)

    anchors = visited_dagger_rows(trajectories)
    # Match the intended supervised guard/control pool without letting long
    # trajectories dominate a family/level cell.
    anchor_cap = min(1200, max(64, len(anchors) // 3))
    anchors = _cap_anchor_rows(anchors, anchor_cap, training_seed(config) + seed_offset + 17)
    for row in anchors:
        row["kind"] = "rl_visited_anchor"
        row["source"] = "rl_policy_visited"

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectories.jsonl.gz"
    write_jsonl(trajectory_path, trajectories)
    write_jsonl(output_dir / "group_diagnostics.jsonl", group_rows)
    write_jsonl(anchor_out, anchors)
    summary = summarize_trajectories(trajectories, group_rows)
    active_groups = sum(bool(row["advantage_active"]) for row in group_rows)
    receipt = {
        "stage": "rl_collection",
        "domain": args.domain,
        "families": families,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model": str(model_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "split": split_receipt(config, specs),
        "n_episode_groups": len(group_rows),
        "n_trajectories": len(trajectories),
        "group_size": 2 if args.smoke else config["rl_collection"]["group_size"],
        "n_active_advantage_groups": active_groups,
        "active_advantage_group_fraction": active_groups / len(group_rows) if group_rows else 0.0,
        "n_anchor_rows": len(anchors),
        "reward_definition": {
            "terminal_environment_score": True,
            "success_turn_penalty": penalty,
            "failed_trajectory_dense_reward": False,
        },
        "advantage_counts": dict(
            sorted(Counter("active" if row["advantage_active"] else "zero" for row in trajectories).items())
        ),
        "summary": summary,
        "trajectory_sha256": sha256_file(trajectory_path),
        "anchor_path": str(anchor_out.resolve()),
        "anchor_sha256": sha256_file(anchor_out),
        "smoke": bool(args.smoke),
    }
    write_json(output_dir / "summary.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
