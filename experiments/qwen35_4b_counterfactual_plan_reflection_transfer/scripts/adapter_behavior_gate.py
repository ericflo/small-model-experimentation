#!/usr/bin/env python3
"""Prove a receipt-bound merged adapter changes greedy Qwen behavior or logprob."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from firewall import install_benchmark_firewall  # noqa: E402

install_benchmark_firewall(EXP.parents[1])


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-generated", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--merged-generated", type=Path, required=True)
    parser.add_argument("--merged-metadata", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    if args.arm not in config["training"]["arms"]:
        raise ValueError("adapter gate arm is not preregistered")
    if args.seed not in set(config["training"]["staged_seeds"].values()):
        raise ValueError("adapter gate seed is not preregistered")
    if (
        args.arm == config["training"]["positive_control"]["arm"]
        and args.seed != config["training"]["staged_seeds"]["screen"]
    ):
        raise ValueError("positive-control replication seed is not authorized")
    base_meta = json.loads(args.base_metadata.read_text())
    merged_meta = json.loads(args.merged_metadata.read_text())
    input_receipt = json.loads(args.input_receipt.read_text())
    base_generated_sha256 = hashlib.sha256(args.base_generated.read_bytes()).hexdigest()
    merged_generated_sha256 = hashlib.sha256(args.merged_generated.read_bytes()).hexdigest()
    runner_sha256 = hashlib.sha256((EXP / "src" / "vllm_runner.py").read_bytes()).hexdigest()
    if input_receipt.get("split") != "calibration":
        raise ValueError("adapter gate must use the sealed calibration split")
    if base_meta["input"]["sha256"] != input_receipt["prompt_sha256"]:
        raise ValueError("base adapter-gate input differs from its receipt")
    if merged_meta["input"]["sha256"] != input_receipt["prompt_sha256"]:
        raise ValueError("merged adapter-gate input differs from its receipt")
    if (
        base_meta.get("output", {}).get("sha256") != base_generated_sha256
        or merged_meta.get("output", {}).get("sha256") != merged_generated_sha256
        or int(base_meta.get("output", {}).get("rows", -1)) != int(input_receipt["rows"])
        or int(merged_meta.get("output", {}).get("rows", -1)) != int(input_receipt["rows"])
    ):
        raise ValueError("adapter-gate generation differs from runner metadata")
    if base_meta["model_override"] is not None or merged_meta["model_override"] is None:
        raise ValueError("adapter gate did not compare base against a merged override")
    if (
        base_meta.get("base_model") != config["model"]["id"]
        or base_meta.get("model_revision") != config["model"]["revision"]
        or base_meta.get("runner_sha256") != runner_sha256
    ):
        raise ValueError("adapter gate used the wrong base, revision, or runner")
    if (
        merged_meta["model_override"].get("source_arm") != args.arm
        or merged_meta["model_override"].get("source_seed") != args.seed
    ):
        raise ValueError("merged override source arm/seed differs from adapter gate")
    for key in ("base_model", "model_revision", "runner_sha256", "sampling"):
        if base_meta[key] != merged_meta[key]:
            raise ValueError(f"adapter gate base/merged {key} differs")
    gate = config["evaluation"]["adapter_gate"]
    sampling = base_meta["sampling"]
    expected = {
        "thinking": "budget",
        "thinking_budget": int(gate["thinking_budget"]),
        "answer_max_tokens": int(gate["answer_max_tokens"]),
        "n": int(gate["candidate_count"]),
        "greedy": bool(gate["greedy"]),
        "logprobs": int(gate["logprobs"]),
        "run_seed": int(gate["run_seed"]),
    }
    if any(sampling[key] != value for key, value in expected.items()):
        raise ValueError("adapter-gate sampling differs from preregistration")
    engine = config["evaluation"]["engine"]
    for metadata in (base_meta, merged_meta):
        for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
            if int(metadata["engine"][key]) != int(engine[key]):
                raise ValueError(f"adapter-gate engine {key} differs")
        if metadata["engine"]["cudagraph_capture_sizes"] != engine["cudagraph_capture_sizes"]:
            raise ValueError("adapter-gate CUDA-graph geometry differs")
        if float(metadata["engine"]["gpu_memory_utilization"]) != float(
            engine["gpu_memory_utilization"]
        ):
            raise ValueError("adapter-gate GPU memory utilization differs")
        if bool(metadata["engine"]["enable_prefix_caching"]) != bool(
            engine["prefix_caching"]
        ):
            raise ValueError("adapter-gate prefix caching differs")
        if bool(metadata["engine_args"]["async_scheduling"]) != bool(
            engine["async_scheduling"]
        ):
            raise ValueError("adapter-gate async scheduling differs")
        if metadata["runtime"]["git_dirty"]:
            raise ValueError("adapter gate ran from a dirty worktree")
        lock_path = EXP.parents[1] / "requirements-vllm.lock.txt"
        if metadata["runtime"]["environment_lock"]["sha256"] != hashlib.sha256(
            lock_path.read_bytes()
        ).hexdigest():
            raise ValueError("adapter-gate environment lock differs")
    base_rows = {row["id"]: row for row in _read_jsonl(args.base_generated)}
    merged_rows = {row["id"]: row for row in _read_jsonl(args.merged_generated)}
    if set(base_rows) != set(merged_rows) or len(base_rows) != int(input_receipt["rows"]):
        raise ValueError("adapter-gate task IDs differ")
    changed = []
    for task_id in sorted(base_rows):
        base_output = base_rows[task_id]["outputs"][0]
        merged_output = merged_rows[task_id]["outputs"][0]
        if (
            base_output["token_ids"] != merged_output["token_ids"]
            or base_output["sampled_cumulative_logprob"]
            != merged_output["sampled_cumulative_logprob"]
        ):
            changed.append(task_id)
    if not changed:
        raise ValueError("merged adapter is an exact ON/OFF no-op on the frozen gate")
    receipt = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(
            (EXP / "configs" / "default.yaml").read_bytes()
        ).hexdigest(),
        "arm": args.arm,
        "seed": args.seed,
        "pass": True,
        "changed_tasks": len(changed),
        "total_tasks": len(base_rows),
        "model_override": merged_meta["model_override"],
        "base_generated_sha256": base_generated_sha256,
        "merged_generated_sha256": merged_generated_sha256,
        "base_metadata_sha256": hashlib.sha256(args.base_metadata.read_bytes()).hexdigest(),
        "merged_metadata_sha256": hashlib.sha256(args.merged_metadata.read_bytes()).hexdigest(),
    }
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
