#!/usr/bin/env python3
"""Collect incumbent-visited states, label with live experts, and build DAgger SFT."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from curriculum import semantic_group_diagnostics  # noqa: E402
from harness import make_runner  # noqa: E402
from io_utils import (  # noqa: E402
    canonical_hash,
    load_config,
    make_specs,
    read_jsonl,
    resolve_repo_path,
    sha256_file,
    split_receipt,
    write_json,
    write_jsonl,
)
from rollout import (  # noqa: E402
    collect_expert_demonstrations,
    collect_policy_episodes,
    summarize_trajectories,
    visited_dagger_rows,
)


def _combined_group_rows(trajectories: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory["episode_key"]].append(trajectory)
    rows = []
    for key in sorted(groups):
        members = groups[key]
        row = semantic_group_diagnostics(members)
        row.update(
            {
                "episode_key": key,
                "family": members[0]["family"],
                "level": members[0]["level"],
                "ep_seed": members[0]["ep_seed"],
            }
        )
        rows.append(row)
    return rows


def _select_visited_rows(rows: list[dict], group_rows: list[dict], config: dict) -> list[dict]:
    group_meta = {row["episode_key"]: row for row in group_rows}
    max_per_family = int(config["collection"]["max_rows_per_family"])
    max_per_episode = int(config["collection"]["max_rows_per_episode"])
    entropy_cap = float(config["collection"]["confident_failure_max_operator_entropy"])
    score_cap = float(config["collection"]["confident_failure_max_mean_score"])

    cells: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        meta = group_meta[row["episode_key"]]
        if meta["operator_entropy"] <= entropy_cap and meta["mean_score"] <= score_cap:
            priority = 0  # confident failure: DAgger's unique acquisition seat
        elif meta["outcome_variance"] > 1e-12:
            priority = 1  # an outcome-sensitive fork
        elif meta["mean_score"] < 0.999:
            priority = 2
        else:
            priority = 3
        enriched = dict(row)
        enriched["group_diagnostics"] = {
            key: meta[key]
            for key in (
                "operator_entropy",
                "mean_score",
                "outcome_variance",
                "outcome_varentropy",
                "constant_outcome",
            )
        }
        enriched["acquisition_priority"] = priority
        cells[(row["family"], priority, int(row["level"]))].append(enriched)

    selected: list[dict] = []
    for family in sorted({row["family"] for row in rows}):
        queues = []
        for priority in range(4):
            for level in sorted({int(row["level"]) for row in rows if row["family"] == family}):
                cell = sorted(
                    cells.get((family, priority, level), []),
                    key=lambda row: (row["episode_key"], row.get("rollout", 0), row["turn"]),
                )
                if cell:
                    queues.append(deque(cell))
        episode_counts: Counter[str] = Counter()
        while queues and sum(row["family"] == family for row in selected) < max_per_family:
            next_queues = []
            made_progress = False
            for queue in queues:
                while queue and episode_counts[queue[0]["episode_key"]] >= max_per_episode:
                    queue.popleft()
                if queue:
                    row = queue.popleft()
                    selected.append(row)
                    episode_counts[row["episode_key"]] += 1
                    made_progress = True
                if queue:
                    next_queues.append(queue)
                if sum(row["family"] == family for row in selected) >= max_per_family:
                    break
            if not made_progress:
                break
            queues = next_queues
    return selected


def _dedup(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    kept = []
    for row in rows:
        key = canonical_hash(
            {
                "messages": row["messages"],
                "think": row["think"],
                "answer": row["answer"],
            }
        )
        if key in seen:
            continue
        seen.add(key)
        row = dict(row)
        row["content_sha256"] = key
        kept.append(row)
    return kept


def _stratified_replay(config: dict, n: int) -> list[dict]:
    source_path = resolve_repo_path(config["model"]["incumbent_data"])
    excluded = set(config["split"]["replay_excluded_families"])
    source = [row for row in read_jsonl(source_path) if row.get("family") not in excluded]
    cells: dict[tuple[str, str], deque] = {}
    rng = random.Random(int(config["seeds"]["training"]) + 991)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in source:
        grouped[(str(row.get("kind", "unknown")), str(row.get("family", "unknown")))].append(row)
    for key, values in sorted(grouped.items()):
        rng.shuffle(values)
        cells[key] = deque(values)
    output = []
    queues = list(cells.values())
    while queues and len(output) < n:
        next_queues = []
        for queue in queues:
            if queue and len(output) < n:
                row = dict(queue.popleft())
                row["source_kind"] = row.get("kind")
                row["kind"] = f"c53_replay_{row.get('kind', 'unknown')}"
                row["source"] = "c53_replay"
                output.append(row)
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path, default=EXP / "runs" / "dagger_collection")
    parser.add_argument("--data-out", type=Path, default=EXP / "data" / "dagger_train.jsonl")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    artifacts_root = resolve_repo_path(config["model"]["artifacts_root"])
    model_path = (args.model or artifacts_root / "merged" / "incumbent_blend").resolve()
    if not (model_path / "config.json").exists():
        raise SystemExit(f"merged incumbent missing: {model_path}")

    train_families = list(config["split"]["train_families"])
    if args.smoke:
        sampled_specs = make_specs(train_families[:2], {1: 1}, config["seeds"]["dagger_visit_base"])
        demo_specs = make_specs(train_families[:2], {1: 1}, config["seeds"]["oracle_demo_base"])
    else:
        sampled_specs = make_specs(
            train_families,
            config["collection"]["dagger_per_level"],
            config["seeds"]["dagger_visit_base"],
        )
        demo_specs = make_specs(
            train_families,
            config["collection"]["oracle_demo_per_level"],
            config["seeds"]["oracle_demo_base"],
        )

    runner = make_runner(config["engine"], model_override=str(model_path))
    try:
        greedy_rows, _ = collect_policy_episodes(
            runner,
            sampled_specs,
            rollouts_per_episode=1,
            think_budget=config["collection"]["thinking_budget"],
            answer_max_tokens=config["collection"]["answer_max_tokens"],
            run_seed=config["seeds"]["dagger_visit_base"] + 17,
            greedy=True,
            rollout_offset=0,
        )
        sampled_rows, _ = collect_policy_episodes(
            runner,
            sampled_specs,
            rollouts_per_episode=config["collection"]["dagger_sampled_rollouts_per_episode"],
            think_budget=config["collection"]["thinking_budget"],
            answer_max_tokens=config["collection"]["answer_max_tokens"],
            run_seed=config["seeds"]["dagger_visit_base"] + 29,
            greedy=False,
            temperature=config["collection"]["temperature"],
            top_p=config["collection"]["top_p"],
            top_k=config["collection"]["top_k"],
            rollout_offset=1,
        )
    finally:
        runner.close()

    trajectories = greedy_rows + sampled_rows
    group_rows = _combined_group_rows(trajectories)
    summary = summarize_trajectories(trajectories, group_rows)
    expert_trajectories, expert_rows = collect_expert_demonstrations(demo_specs)
    visited = _select_visited_rows(visited_dagger_rows(trajectories), group_rows, config)

    max_demo = math.floor(
        len(visited)
        * float(config["collection"]["expert_demo_max_fraction"])
        / (1.0 - float(config["collection"]["expert_demo_max_fraction"]))
    )
    expert_rows = sorted(expert_rows, key=lambda row: row["id"])[:max_demo]
    incremental = _dedup(visited + expert_rows)
    target_replay = math.ceil(
        len(incremental)
        * float(config["collection"]["replay_fraction"])
        / (1.0 - float(config["collection"]["replay_fraction"]))
    )
    replay = _stratified_replay(
        config,
        min(int(config["collection"]["replay_rows"]), target_replay),
    )
    dataset = _dedup(incremental + replay)

    transfer = set(config["split"]["transfer_families"])
    leaked = [
        row["id"]
        for row in dataset
        if row.get("family") in transfer and row.get("source") != "c53_replay"
    ]
    if leaked:
        raise SystemExit(f"incremental transfer-family leakage: {leaked[:5]}")
    if any(row.get("family") in transfer for row in replay):
        raise SystemExit("transfer-family row entered new-stage replay")
    demo_fraction = (
        sum(row.get("source") == "expert_demo" for row in incremental) / len(incremental)
        if incremental
        else 0.0
    )
    if demo_fraction > float(config["collection"]["expert_demo_max_fraction"]) + 1e-9:
        raise SystemExit(f"expert-demo fraction {demo_fraction:.4f} exceeds cap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "trajectories.jsonl.gz", trajectories)
    write_jsonl(args.output_dir / "expert_trajectories.jsonl.gz", expert_trajectories)
    write_jsonl(args.output_dir / "group_diagnostics.jsonl", group_rows)
    write_jsonl(args.data_out, dataset)
    counts = Counter(row.get("kind", "unknown") for row in dataset)
    family_counts = Counter(row.get("family", "unknown") for row in dataset)
    receipt = {
        "stage": "dagger_collection",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model": str(model_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "split": split_receipt(config, sampled_specs + demo_specs),
        "n_trajectories": len(trajectories),
        "n_expert_trajectories": len(expert_trajectories),
        "n_visited_rows_before_cap": len(visited_dagger_rows(trajectories)),
        "n_visited_rows_selected": len(visited),
        "n_expert_rows": len(expert_rows),
        "expert_demo_fraction_incremental": demo_fraction,
        "n_replay_rows": len(replay),
        "n_train_rows": len(dataset),
        "kind_counts": dict(sorted(counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "trajectory_summary": summary,
        "data_sha256": sha256_file(args.data_out),
        "smoke": bool(args.smoke),
    }
    write_json(args.output_dir / "summary.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
