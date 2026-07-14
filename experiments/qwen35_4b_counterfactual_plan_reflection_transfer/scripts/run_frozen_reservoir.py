#!/usr/bin/env python3
"""Generate compute-stopped frozen confirmation blocks without reading outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if sys.flags.no_site != 1:
    raise SystemExit("reservoir generation must start with the pinned interpreter and -I -B -S")

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from runtime_contract import (  # noqa: E402
    bootstrap_runtime_environment,
    require_detached_execution_worktree,
)

bootstrap_runtime_environment(EXP.parents[1], "vllm")

import yaml

from firewall import install_benchmark_firewall  # noqa: E402

install_benchmark_firewall(EXP.parents[1])

from matched_compute import (  # noqa: E402
    artifact_ref,
    cumulative_reservoir_compute,
    first_budget_prefix,
    target_compute_budget,
)
from vllm_runner import (  # noqa: E402
    EngineConfig,
    SamplingConfig,
    VLLMRunner,
    _read_jsonl,
    _validate_cli_stage_receipt,
    _write_json_atomic,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_pairs(
    training_paths: list[Path], metadata_paths: list[Path], expected_seeds: set[int]
) -> tuple[list[tuple[dict, dict, dict]], list[dict]]:
    receipts: dict[int, tuple[Path, dict]] = {}
    metadata: dict[int, tuple[Path, dict]] = {}
    for path in training_paths:
        value = json.loads(path.read_text())
        seed = int(value.get("seed", -1))
        if seed in receipts:
            raise ValueError("duplicate reservoir target training seed")
        receipts[seed] = (path, value)
    for path in metadata_paths:
        value = json.loads(path.read_text())
        override = value.get("model_override")
        seed = int(override.get("source_seed", -1)) if isinstance(override, dict) else -1
        if seed in metadata:
            raise ValueError("duplicate reservoir target confirmation seed")
        metadata[seed] = (path, value)
    if set(receipts) != expected_seeds or set(metadata) != expected_seeds:
        raise ValueError("reservoir target paths do not contain both frozen seeds")
    pairs = []
    references = []
    for seed in sorted(expected_seeds):
        training_path, training_receipt = receipts[seed]
        tokenizer_path = training_path.parent / "source_tokenizer_receipt.json"
        if (
            not tokenizer_path.is_file()
            or _sha256_file(tokenizer_path)
            != training_receipt.get("tokenizer_receipt_sha256")
            or _sha256_file(tokenizer_path)
            != training_receipt.get("copied_tokenizer_receipt_sha256")
        ):
            raise ValueError("training compute lacks its exact source tokenizer receipt")
        tokenizer_receipt = json.loads(tokenizer_path.read_text())
        pairs.append((training_receipt, tokenizer_receipt, metadata[seed][1]))
        references.append(
            {
                "seed": seed,
                "training_receipt": artifact_ref(training_path),
                "source_tokenizer_receipt": artifact_ref(tokenizer_path),
                "correct_confirmation_metadata": artifact_ref(metadata[seed][0]),
            }
        )
    return pairs, references


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, action="append", required=True)
    parser.add_argument("--correct-metadata", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = EXP / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["authorization"]["evaluation"] is not True:
        raise SystemExit("frozen reservoir evaluation is not authorized")
    if args.output_dir.exists():
        raise SystemExit(f"output already exists: {args.output_dir}")
    worktree = require_detached_execution_worktree(EXP.parents[1])
    config_sha256 = _sha256_file(config_path)
    input_receipt = json.loads(args.input_receipt.read_text())
    if (
        set(input_receipt)
        != {
            "schema_version", "experiment_id", "config_sha256", "input_kind",
            "split", "rows", "prompt_sha256", "label_sha256", "model_calls",
            "gpu_events", "benchmark_reads",
        }
        or input_receipt.get("schema_version") != 2
        or input_receipt.get("experiment_id") != config["experiment_id"]
        or input_receipt.get("config_sha256") != config_sha256
        or input_receipt.get("input_kind") != "action"
        or input_receipt.get("split") != "confirmation"
        or input_receipt.get("rows")
        != int(config["construction"]["per_family"]["confirmation"])
        * len(config["construction"]["families"])
        or input_receipt.get("model_calls") != 0
        or input_receipt.get("gpu_events") != 0
        or input_receipt.get("benchmark_reads") != 0
    ):
        raise ValueError("reservoir input receipt differs from sealed confirmation")
    records, input_sha256 = _read_jsonl(args.input)
    if input_sha256 != input_receipt["prompt_sha256"]:
        raise ValueError("reservoir input differs from sealed confirmation prompts")
    generation_stage = _validate_cli_stage_receipt(
        args.stage_receipt, records, model_override=None
    )
    if generation_stage["authorized_stage"] != "confirmation":
        raise ValueError("reservoir did not use the confirmation stage receipt")
    expected_seeds = set(map(int, config["training"]["staged_seeds"].values()))
    target_pairs, target_references = _target_pairs(
        args.training_receipt, args.correct_metadata, expected_seeds
    )
    target_budget = target_compute_budget(target_pairs, expected_seeds)
    contract = config["evaluation"]["frozen_sample_more"]
    evaluation = config["evaluation"]
    frozen_engine = evaluation["engine"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    engine = EngineConfig(
        max_model_len=int(frozen_engine["max_model_len"]),
        gpu_memory_utilization=float(frozen_engine["gpu_memory_utilization"]),
        max_num_seqs=int(frozen_engine["max_num_seqs"]),
        max_num_batched_tokens=int(frozen_engine["max_num_batched_tokens"]),
        cudagraph_capture_sizes=tuple(frozen_engine["cudagraph_capture_sizes"]),
        enable_prefix_caching=False,
        enforce_eager=False,
    )
    runner: VLLMRunner | None = None
    blocks: list[dict] = []
    metadata_values: list[dict] = []
    try:
        runner = VLLMRunner(engine)
        for index, run_seed in enumerate(contract["block_run_seeds"]):
            sampling = SamplingConfig(
                thinking="budget",
                thinking_budget=int(evaluation["thinking_budget"]),
                n=int(contract["block_candidate_count"]),
                answer_max_tokens=int(evaluation["answer_max_tokens"]),
                temperature=float(evaluation["temperature"]),
                top_p=float(evaluation["top_p"]),
                top_k=int(evaluation["top_k"]),
                run_seed=int(run_seed),
            )
            rows, summary = runner.generate(records, sampling)
            generated_path = args.output_dir / f"block-{index:02d}.jsonl"
            metadata_path = args.output_dir / f"block-{index:02d}.meta.json"
            summary["input"] = {
                "description": str(args.input.resolve()),
                "sha256": input_sha256,
            }
            summary["generation_stage"] = generation_stage
            generated_sha256 = _write_json_atomic(generated_path, rows, jsonl=True)
            summary["output"] = {
                "description": "generated JSONL",
                "sha256": generated_sha256,
                "rows": len(rows),
            }
            _write_json_atomic(metadata_path, summary)
            blocks.append(
                {
                    "index": index,
                    "run_seed": int(run_seed),
                    "generated": artifact_ref(generated_path),
                    "metadata": artifact_ref(metadata_path),
                }
            )
            metadata_values.append(summary)
            cumulative = cumulative_reservoir_compute(metadata_values)
            if first_budget_prefix(cumulative, target_budget) is not None:
                break
    finally:
        if runner is not None:
            runner.close()
    cumulative = cumulative_reservoir_compute(metadata_values)
    prefix = first_budget_prefix(cumulative, target_budget)
    stop = {
        "decision": (
            "FIRST_COMPLETE_BLOCK_REACHES_BOTH_BUDGETS"
            if prefix == len(blocks) - 1
            else "MAXIMUM_BLOCKS_EXHAUSTED_WITHOUT_BOTH_BUDGETS"
        ),
        "first_budget_prefix_index": prefix,
        "completed_blocks": len(blocks),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_sha256,
        "producer": {
            "script_sha256": _sha256_file(Path(__file__).resolve()),
            "module_sha256": _sha256_file(EXP / "src" / "matched_compute.py"),
            "runner_sha256": _sha256_file(EXP / "src" / "vllm_runner.py"),
            "git_commit": worktree["git_commit"],
        },
        "worktree": worktree,
        "stage_receipt": artifact_ref(args.stage_receipt),
        "input": artifact_ref(args.input),
        "input_receipt": artifact_ref(args.input_receipt),
        "targets": target_references,
        "target_budget": target_budget,
        "block_contract": contract,
        "blocks": blocks,
        "cumulative_compute": cumulative,
        "stop": stop,
        "outcome_blindness": {
            "accepted_label_or_score_paths": False,
            "read_correctness_fields": False,
            "stop_fields": [
                "training_receipt.compute",
                "correct_confirmation_metadata.counts",
                "correct_confirmation_metadata.timing",
                "frozen_block_metadata.counts",
                "frozen_block_metadata.timing",
            ],
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps({"manifest": artifact_ref(manifest_path), "stop": stop}, indent=2))
    if prefix is None:
        raise SystemExit("maximum reservoir blocks did not reach both compute budgets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
