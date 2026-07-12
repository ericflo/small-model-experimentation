#!/usr/bin/env python3
"""Generate selection and audit continuations from exact frozen states."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

import harness  # noqa: E402
from eval_policy import _engine_protocol  # noqa: E402
from gym.families import load as load_family  # noqa: E402
from io_utils import load_config, read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from vllm_runner import SamplingConfig  # noqa: E402


POLICY_OFFSET = {"quick": 100_000, "deep": 200_000, "student": 300_000}


def _score_atom(tokenizer, state: dict, output: dict) -> float:
    family = load_family(state["family"])
    item = {
        "id": state["source_id"],
        "family": state["family"],
        "level": int(state["level"]),
        "prompt": state["prompt"],
        "gold": state["gold"],
        "answer_domain": state.get("answer_domain"),
    }
    text = tokenizer.decode(
        [*state["student_prefix_ids"], *[int(value) for value in output["token_ids"]]],
        skip_special_tokens=False,
    )
    return float(family.score_atom(item, text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--policy", choices=tuple(POLICY_OFFSET), required=True)
    parser.add_argument("--block-seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    states = read_jsonl(args.states)
    if not states:
        raise SystemExit("empty state artifact")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f"refusing non-empty branch directory: {args.out_dir}")
    if not (args.model / "merge_receipt.json").is_file():
        raise SystemExit("branch policy must be an explicitly merged composite")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    route = config["route"]
    total_branches = int(route["selection_branches_per_policy"]) + int(
        route["audit_branches_per_policy"]
    )
    generation = config["generation"]
    run_seed = int(args.block_seed) + POLICY_OFFSET[args.policy]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    runner = harness.make_runner(config["engine"], model_override=str(args.model.resolve()))
    started = time.perf_counter()
    results = []
    atom_states = [row for row in states if row["kind"] == "atom"]
    by_prefix: dict[int, list[dict]] = defaultdict(list)
    for state in atom_states:
        by_prefix[int(state["prefix_length"])].append(state)
    for prefix_length in sorted(by_prefix):
        records = [
            {
                "id": f"{args.policy}::{state['state_id']}",
                "prompt_token_ids": [int(value) for value in state["exact_prompt_token_ids"]],
                "meta": {
                    "state_id": state["state_id"],
                    "family": state["family"],
                    "kind": "atom",
                    "level": int(state["level"]),
                    "prefix_length": prefix_length,
                },
            }
            for state in by_prefix[prefix_length]
        ]
        sampling = SamplingConfig(
            thinking="budget",
            thinking_budget=int(generation["thinking_budget"]) - prefix_length,
            n=total_branches,
            answer_max_tokens=int(generation["answer_max_tokens"]),
            greedy=False,
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            top_k=int(generation["top_k"]),
            run_seed=run_seed,
            allow_custom_prompts=True,
        )
        rows, summary = runner.generate(records, sampling)
        runner.eval_summaries = getattr(runner, "eval_summaries", []) + [summary]
        state_map = {state["state_id"]: state for state in by_prefix[prefix_length]}
        for row in rows:
            state_id = row["meta"]["state_id"]
            state = state_map[state_id]
            for output in row["outputs"]:
                results.append(
                    {
                        "state_id": state_id,
                        "branch_index": int(output["sample_index"]),
                        "policy": args.policy,
                        "family": state["family"],
                        "kind": "atom",
                        "level": int(state["level"]),
                        "score": _score_atom(tokenizer, state, output),
                        "output": {
                            key: output.get(key)
                            for key in (
                                "sample_index", "seed_stage1", "seed_stage2", "text",
                                "token_ids", "n_thinking_tokens", "n_answer_tokens",
                                "n_sampled_tokens", "thinking_closed", "forced_close",
                                "finish_reason", "truncated", "injected_token_ids",
                            )
                        },
                    }
                )
    episode_states = [row for row in states if row["kind"] == "episode"]
    if episode_states:
        results.extend(
            harness.run_episode_continuations(
                runner,
                episode_states,
                branches_per_state=total_branches,
                think_budget=int(generation["thinking_budget"]),
                answer_max_tokens=int(generation["answer_max_tokens"]),
                run_seed=run_seed,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                top_k=int(generation["top_k"]),
                policy_tag=args.policy,
            )
        )
    elapsed = time.perf_counter() - started
    summaries = getattr(runner, "eval_summaries", [])
    runner.close()
    del runner
    gc.collect()
    protocol = _engine_protocol(
        summaries,
        engine_cfg=config["engine"],
        model=args.model,
        model_config_sha256=sha256_file(args.model / "config.json"),
    )
    if not all(protocol.values()):
        raise SystemExit(f"branch engine protocol failed: {protocol}")
    results.sort(key=lambda row: (row["state_id"], int(row["branch_index"])))
    expected = len(states) * total_branches
    if len(results) != expected:
        raise SystemExit(f"branch output count {len(results)} != {expected}")
    identities = [(row["state_id"], int(row["branch_index"])) for row in results]
    if len(identities) != len(set(identities)):
        raise SystemExit("duplicate state/branch identity")
    output_path = args.out_dir / "branches.jsonl.gz"
    write_jsonl(output_path, results)
    sampled_tokens = sum(
        int(row.get("output", {}).get("n_sampled_tokens") or 0)
        + sum(int(turn["n_sampled_tokens"]) for turn in row.get("turns", []))
        for row in results
    )
    receipt = {
        "schema_version": 1,
        "stage": "state_branch_generation",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "states": str(args.states.resolve()),
        "states_sha256": sha256_file(args.states),
        "model": str(args.model.resolve()),
        "model_merge_receipt_sha256": sha256_file(args.model / "merge_receipt.json"),
        "policy": args.policy,
        "block_seed": int(args.block_seed),
        "run_seed": run_seed,
        "selection_branches": int(route["selection_branches_per_policy"]),
        "audit_branches": int(route["audit_branches_per_policy"]),
        "rows": len(results),
        "sampled_tokens": sampled_tokens,
        "branches_sha256": sha256_file(output_path),
        "wall_seconds": elapsed,
        "engine_protocol": protocol,
        "runner_summaries": summaries,
    }
    write_json(args.out_dir / "receipt.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "runner_summaries"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
