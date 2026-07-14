#!/usr/bin/env python3
"""Score one immutable vLLM generation bundle into task-level gate rows."""

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

from scoring import score_generation_rows  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_exclusive(path: Path, rows: list[dict]) -> str:
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--adapter-gate-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    evaluation = config["evaluation"]
    if args.arm not in set(evaluation["arms"]):
        raise ValueError("arm is not preregistered")
    metadata = json.loads(args.metadata.read_text())
    input_receipt = json.loads(args.input_receipt.read_text())
    generated_sha256 = hashlib.sha256(args.generated.read_bytes()).hexdigest()
    if (
        metadata.get("output", {}).get("sha256") != generated_sha256
        or int(metadata.get("output", {}).get("rows", -1)) != int(input_receipt["rows"])
    ):
        raise ValueError("generation output differs from its runner metadata")
    runner_path = EXP / "src" / "vllm_runner.py"
    if metadata["runner_sha256"] != hashlib.sha256(runner_path.read_bytes()).hexdigest():
        raise ValueError("generation used a different vLLM runner")
    if metadata["input"]["sha256"] != input_receipt["prompt_sha256"]:
        raise ValueError("generation input differs from the construction receipt")
    if hashlib.sha256(args.labels.read_bytes()).hexdigest() != input_receipt["label_sha256"]:
        raise ValueError("label bundle differs from the construction receipt")
    if (
        metadata["base_model"] != config["model"]["id"]
        or metadata["model_revision"] != config["model"]["revision"]
    ):
        raise ValueError("generation used the wrong base model identity")
    sampling = metadata["sampling"]
    expected_sampling = {
        "thinking": "budget",
        "thinking_budget": int(evaluation["thinking_budget"]),
        "n": int(evaluation["primary_candidate_count"]),
        "answer_max_tokens": int(evaluation["answer_max_tokens"]),
        "temperature": float(evaluation["temperature"]),
        "top_p": float(evaluation["top_p"]),
        "top_k": int(evaluation["top_k"]),
    }
    for key, expected in expected_sampling.items():
        if sampling[key] != expected:
            raise ValueError(f"generation sampling {key} differs from preregistration")
    expected_seed = int(evaluation["sample_seeds"][input_receipt["split"]])
    if int(sampling["run_seed"]) != expected_seed:
        raise ValueError("generation run seed differs from the frozen split seed")
    engine = evaluation["engine"]
    for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        if int(metadata["engine"][key]) != int(engine[key]):
            raise ValueError(f"generation engine {key} differs from preregistration")
    if float(metadata["engine"]["gpu_memory_utilization"]) != float(
        engine["gpu_memory_utilization"]
    ):
        raise ValueError("generation GPU memory utilization differs from preregistration")
    if metadata["engine"]["cudagraph_capture_sizes"] != engine["cudagraph_capture_sizes"]:
        raise ValueError("generation CUDA-graph geometry differs from preregistration")
    if bool(metadata["engine"]["enable_prefix_caching"]) != bool(
        engine["prefix_caching"]
    ):
        raise ValueError("generation prefix caching differs from preregistration")
    if bool(metadata["engine_args"]["async_scheduling"]) != bool(
        engine["async_scheduling"]
    ):
        raise ValueError("generation async scheduling differs from preregistration")
    if metadata["runtime"]["git_dirty"]:
        raise ValueError("generation ran from a dirty worktree")
    lock_path = EXP.parents[1] / "requirements-vllm.lock.txt"
    if metadata["runtime"]["environment_lock"]["sha256"] != hashlib.sha256(
        lock_path.read_bytes()
    ).hexdigest():
        raise ValueError("generation environment lock differs from the repository pin")
    frozen = args.arm == "frozen_action"
    if frozen:
        if (
            args.training_seed is not None
            or metadata["model_override"] is not None
            or args.adapter_gate_receipt is not None
        ):
            raise ValueError("frozen arm must use the base model without a training seed")
    else:
        if args.training_seed not in set(config["training"]["staged_seeds"].values()):
            raise ValueError("adapter arm lacks a preregistered training seed")
        if metadata["model_override"] is None:
            raise ValueError("trained arm did not use a receipt-bound merged checkpoint")
        if args.adapter_gate_receipt is None:
            raise ValueError("trained arm lacks its adapter ON/OFF gate receipt")
        adapter_gate = json.loads(args.adapter_gate_receipt.read_text())
        training_arm = {
            "reflection_correct_action": "reflection_correct",
            "reflection_shuffled_action": "reflection_shuffled",
            "auxiliary_plan_label_correct_action": "auxiliary_plan_label_correct",
            "direct_plan_answer_positive_control_action": "direct_plan_answer_positive_control",
        }[args.arm]
        if (
            adapter_gate.get("pass") is not True
            or adapter_gate.get("experiment_id") != config["experiment_id"]
            or adapter_gate.get("config_sha256")
            != hashlib.sha256((EXP / "configs" / "default.yaml").read_bytes()).hexdigest()
            or adapter_gate.get("arm") != training_arm
            or adapter_gate.get("seed") != args.training_seed
            or adapter_gate.get("model_override") != metadata["model_override"]
        ):
            raise ValueError("adapter ON/OFF gate does not bind this arm/seed/merged model")
        if (
            args.arm == "direct_plan_answer_positive_control_action"
            and args.training_seed != config["training"]["staged_seeds"]["screen"]
        ):
            raise ValueError("positive-control replication seed is not authorized")
    counts = tuple(
        sorted(
            {
                int(evaluation["primary_candidate_count"]),
                *[int(value) for value in evaluation["descriptive_candidate_counts"]],
            }
        )
    )
    scored = score_generation_rows(
        _read_jsonl(args.generated),
        _read_jsonl(args.labels),
        arm=args.arm,
        candidate_counts=counts,
        answer_max_tokens=int(evaluation["answer_max_tokens"]),
        loop_detector=evaluation["loop_detector"],
    )
    metadata_sha256 = hashlib.sha256(args.metadata.read_bytes()).hexdigest()
    for row in scored:
        row["training_seed"] = args.training_seed
        row["generated_sha256"] = generated_sha256
        row["metadata_sha256"] = metadata_sha256
        row["adapter_gate_receipt_sha256"] = (
            None
            if args.adapter_gate_receipt is None
            else hashlib.sha256(args.adapter_gate_receipt.read_bytes()).hexdigest()
        )
    digest = _write_exclusive(args.output, scored)
    print(json.dumps({"rows": len(scored), "output_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
