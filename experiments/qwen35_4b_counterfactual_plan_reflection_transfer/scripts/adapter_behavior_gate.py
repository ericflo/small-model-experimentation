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
from provenance import (  # noqa: E402
    validate_action_inputs,
    validate_generation_protocol,
    validate_sampling,
)
from vllm_runner import SamplingConfig  # noqa: E402

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
    parser.add_argument("--labels", type=Path, required=True)
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
    config_path = EXP / "configs" / "default.yaml"
    _, expected_task_metadata, sealed = validate_action_inputs(
        config=config,
        config_path=config_path,
        receipt_path=args.input_receipt,
        labels_path=args.labels,
        expected_split="calibration",
    )
    if base_meta["input"]["sha256"] != sealed["prompt_sha256"]:
        raise ValueError("base adapter-gate input differs from sealed reconstruction")
    if merged_meta["input"]["sha256"] != sealed["prompt_sha256"]:
        raise ValueError("merged adapter-gate input differs from sealed reconstruction")
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
        merged_meta["model_override"].get("source_arm") != args.arm
        or merged_meta["model_override"].get("source_seed") != args.seed
    ):
        raise ValueError("merged override source arm/seed differs from adapter gate")
    gate = config["evaluation"]["adapter_gate"]
    expected_sampling = SamplingConfig(
        thinking="budget",
        thinking_budget=int(gate["thinking_budget"]),
        answer_max_tokens=int(gate["answer_max_tokens"]),
        n=int(gate["candidate_count"]),
        greedy=bool(gate["greedy"]),
        logprobs=int(gate["logprobs"]),
        run_seed=int(gate["run_seed"]),
    )
    validate_sampling(base_meta, expected_sampling)
    validate_sampling(merged_meta, expected_sampling)
    protocols = {
        validate_generation_protocol(
            metadata=base_meta,
            config=config,
            experiment_root=EXP,
            generated_path=args.base_generated,
            expected_rows=int(input_receipt["rows"]),
            expect_merged=False,
            expected_stage="calibration_generation",
            expected_split="calibration",
            expected_input_kind="action",
            expected_source_seed=None,
        ),
        validate_generation_protocol(
            metadata=merged_meta,
            config=config,
            experiment_root=EXP,
            generated_path=args.merged_generated,
            expected_rows=int(input_receipt["rows"]),
            expect_merged=True,
            expected_stage=(
                "screen_training"
                if args.seed == config["training"]["staged_seeds"]["screen"]
                else "replication_training"
            ),
            expected_split="calibration",
            expected_input_kind="action",
            expected_source_seed=args.seed,
        ),
    }
    if len(protocols) != 1:
        raise ValueError("adapter gate base/merged runtime protocols differ")
    runtime_protocol_sha256 = protocols.pop()
    base_rows = {row["id"]: row for row in _read_jsonl(args.base_generated)}
    merged_rows = {row["id"]: row for row in _read_jsonl(args.merged_generated)}
    if set(base_rows) != set(merged_rows) or len(base_rows) != int(input_receipt["rows"]):
        raise ValueError("adapter-gate task IDs differ")
    if set(base_rows) != set(expected_task_metadata):
        raise ValueError("adapter-gate task IDs differ from sealed calibration")
    for task_id in expected_task_metadata:
        family, depth = expected_task_metadata[task_id]
        for row in (base_rows[task_id], merged_rows[task_id]):
            if (
                row.get("meta", {}).get("family") != family
                or int(row.get("meta", {}).get("depth", -1)) != depth
            ):
                raise ValueError("adapter-gate task metadata differs from sealed calibration")
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
        "schema_version": 2,
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
        "runtime_protocol_sha256": runtime_protocol_sha256,
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
