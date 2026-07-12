#!/usr/bin/env python3
"""Generate fresh failed student states before any teacher branch exists."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

import harness  # noqa: E402
from eval_policy import _engine_protocol  # noqa: E402
from gym.families import load as load_family  # noqa: E402
from io_utils import load_config, read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from state_replay import (  # noqa: E402
    build_atom_state,
    build_episode_state,
    select_balanced_states,
    state_digest,
)


def _write_gzip(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _candidate_items(config: dict, seed: int, *, smoke: bool) -> tuple[list[dict], list[tuple[str, int, int]]]:
    families = list(config["strata"]["trained_families"])
    if smoke:
        families = families[:2]
    generation = config["generation"]
    atom_n = 2 if smoke else int(generation["atom_candidates_per_family_level"])
    episode_n = 1 if smoke else int(generation["episode_candidates_per_family_level"])
    levels = [
        *config["strata"]["quick_atom_levels"],
        *config["strata"]["deep_atom_levels"],
    ]
    atom_items = []
    for family_index, family_name in enumerate(families):
        family = load_family(family_name)
        for level in levels:
            if int(level) not in family.LEVELS:
                continue
            item_seed = seed + family_index * 100_000 + int(level) * 1_000
            atom_items.extend(family.gen_atoms(item_seed, int(level), atom_n))
    episode_specs = []
    for family_index, family_name in enumerate(families):
        family = load_family(family_name)
        if not getattr(family, "HAS_EPISODES", False):
            continue
        for level in config["strata"]["deep_episode_levels"]:
            if int(level) not in family.LEVELS:
                continue
            for index in range(episode_n):
                episode_specs.append(
                    (
                        family_name,
                        int(level),
                        seed + 50_000_000 + family_index * 100_000 + int(level) * 1_000 + index,
                    )
                )
    return atom_items, episode_specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("qualification", "training"), default="qualification")
    parser.add_argument("--round", type=int)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    expected_seeds = [int(value) for value in config["seeds"]["route_qualification_blocks"]]
    if not args.smoke and args.mode == "qualification":
        if args.block not in range(len(expected_seeds)):
            raise SystemExit(f"qualification block must index {expected_seeds}")
        if args.seed != expected_seeds[args.block]:
            raise SystemExit(f"block {args.block} seed {args.seed} != frozen {expected_seeds[args.block]}")
    if not args.smoke and args.mode == "training":
        if args.round is None or not 0 <= args.round < int(config["mopd"]["rounds"]):
            raise SystemExit("training state generation needs a valid --round")
        maximum = int(config["mopd"]["candidate_batches_per_round_maximum"])
        if not 0 <= args.batch < maximum:
            raise SystemExit(f"training batch must be in [0, {maximum - 1}]")
        expected_seed = int(config["seeds"]["rollout_rounds"][args.round]) + args.batch * 10
        expected_block = 100 + args.round * maximum + args.batch
        if args.seed != expected_seed or args.block != expected_block:
            raise SystemExit(
                f"training state namespace mismatch: block/seed {args.block}/{args.seed} "
                f"!= {expected_block}/{expected_seed}"
            )
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"refusing non-empty state directory: {args.out_dir}")
    if not (args.model / "merge_receipt.json").is_file():
        raise SystemExit("student model must be an explicitly merged composite")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atom_items, episode_specs = _candidate_items(config, args.seed, smoke=args.smoke)
    generation = config["generation"]
    runner = harness.make_runner(config["engine"], model_override=str(args.model.resolve()))
    started = time.perf_counter()
    common = dict(
        think_budget=int(generation["thinking_budget"]),
        answer_max_tokens=int(generation["answer_max_tokens"]),
        run_seed=args.seed,
        greedy=False,
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        top_k=int(generation["top_k"]),
    )
    atom_rows = harness.run_atoms(runner, atom_items, k=1, **common)
    episode_rows = (
        harness.run_episodes(runner, episode_specs, k=1, progress=True, **common)
        if episode_specs
        else []
    )
    elapsed = time.perf_counter() - started
    summaries = getattr(runner, "eval_summaries", [])
    runner.close()
    protocol = _engine_protocol(
        summaries,
        engine_cfg=config["engine"],
        model=args.model,
        model_config_sha256=sha256_file(args.model / "config.json"),
    )
    if not all(protocol.values()):
        raise SystemExit(f"state generation engine protocol failed: {protocol}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    close_id = int(tokenizer.convert_tokens_to_ids("</think>"))
    candidates = []
    anchor_candidates = []
    for row in atom_rows:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        state = build_atom_state(
            row,
            block=args.block,
            prompt_token_ids=prompt_ids,
            think_close_token_id=close_id,
            prefix_fraction=float(generation["atom_prefix_fraction"]),
            prefix_min_tokens=int(generation["atom_prefix_min_tokens"]),
            prefix_max_tokens=int(generation["atom_prefix_max_tokens"]),
            failure_ceiling=float(generation["state_failure_ceiling"]),
        )
        if state is not None:
            candidates.append(state)
        if args.mode == "training":
            anchor = build_atom_state(
                row,
                block=args.block,
                prompt_token_ids=prompt_ids,
                think_close_token_id=close_id,
                prefix_fraction=float(generation["atom_prefix_fraction"]),
                prefix_min_tokens=int(generation["atom_prefix_min_tokens"]),
                prefix_max_tokens=int(generation["atom_prefix_max_tokens"]),
                failure_ceiling=float(generation["state_failure_ceiling"]),
                require_failure=False,
            )
            if anchor is not None:
                anchor_candidates.append(anchor)
    for row in episode_rows:
        state = build_episode_state(
            row,
            block=args.block,
            failure_ceiling=float(generation["state_failure_ceiling"]),
        )
        if state is not None:
            candidates.append(state)
        if args.mode == "training":
            anchor = build_episode_state(
                row,
                block=args.block,
                failure_ceiling=float(generation["state_failure_ceiling"]),
                require_failure=False,
            )
            if anchor is not None:
                anchor_candidates.append(anchor)

    if args.smoke:
        atom_required = min(2, sum(row["kind"] == "atom" for row in candidates))
        episode_required = min(1, sum(row["kind"] == "episode" for row in candidates))
    else:
        atom_required = int(generation["qualification_atom_states_per_block"])
        episode_required = int(generation["qualification_episode_states_per_block"])
    selected = select_balanced_states(
        candidates, atom_count=atom_required, episode_count=episode_required
    )
    anchors = []
    if args.mode == "training":
        anchors = select_balanced_states(
            anchor_candidates,
            atom_count=min(atom_required // 3, sum(row["kind"] == "atom" for row in anchor_candidates)),
            episode_count=min(
                episode_required // 2,
                sum(row["kind"] == "episode" for row in anchor_candidates),
            ),
        )
    _write_gzip(args.out_dir / "student_atom_rows.jsonl.gz", atom_rows)
    _write_gzip(args.out_dir / "student_episode_rows.jsonl.gz", episode_rows)
    write_jsonl(args.out_dir / "states.jsonl", selected)
    if args.mode == "training":
        write_jsonl(args.out_dir / "anchors.jsonl", anchors)
    state_hash = state_digest({"states": selected})
    sampled_tokens = sum(
        int(output["n_sampled_tokens"])
        for row in atom_rows
        for output in row["outputs"]
    ) + sum(
        int(turn["n_sampled_tokens"])
        for row in episode_rows
        for turn in row["turns"]
    )
    receipt = {
        "schema_version": 1,
        "stage": "student_state_generation",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model": str(args.model.resolve()),
        "model_merge_receipt_sha256": sha256_file(args.model / "merge_receipt.json"),
        "block": args.block,
        "seed": args.seed,
        "mode": args.mode,
        "round": args.round,
        "batch": args.batch,
        "candidate_counts": {
            "atom": sum(row["kind"] == "atom" for row in candidates),
            "episode": sum(row["kind"] == "episode" for row in candidates),
        },
        "selected_counts": {
            "atom": sum(row["kind"] == "atom" for row in selected),
            "episode": sum(row["kind"] == "episode" for row in selected),
        },
        "anchor_counts": {
            "atom": sum(row["kind"] == "atom" for row in anchors),
            "episode": sum(row["kind"] == "episode" for row in anchors),
        },
        "state_ids_sha256": state_hash,
        "states_sha256": sha256_file(args.out_dir / "states.jsonl"),
        "anchors_sha256": (
            sha256_file(args.out_dir / "anchors.jsonl") if args.mode == "training" else None
        ),
        "sampled_tokens": sampled_tokens,
        "wall_seconds": elapsed,
        "engine_protocol": protocol,
        "runner_summaries": summaries,
        "smoke": bool(args.smoke),
    }
    write_json(args.out_dir / "receipt.json", receipt)
    # Re-open the exact artifact to catch incomplete writes before returning.
    if len(read_jsonl(args.out_dir / "states.jsonl")) != len(selected):
        raise RuntimeError("state artifact failed round-trip validation")
    print(json.dumps({key: value for key, value in receipt.items() if key != "runner_summaries"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
