#!/usr/bin/env python3
"""Append each sampled literal reflection exactly and build its action continuation."""

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

from taskgen import ACTION_QUESTION, REFLECTION_QUESTION, build_corpus  # noqa: E402


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
    parser.add_argument("--split", choices=("qualification",), required=True)
    parser.add_argument("--reflection-generated", type=Path, required=True)
    parser.add_argument("--reflection-metadata", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.receipt.resolve():
        raise ValueError("--output and --receipt must be distinct paths")
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    if config["authorization"]["evaluation"] is not True:
        raise SystemExit("evaluation is not authorized")
    reflection_metadata = json.loads(args.reflection_metadata.read_text())
    input_receipt = json.loads(args.input_receipt.read_text())
    reflection_generated_sha256 = hashlib.sha256(
        args.reflection_generated.read_bytes()
    ).hexdigest()
    diagnostic = config["evaluation"]["literal_reflection_diagnostic"]
    runner_sha256 = hashlib.sha256((EXP / "src" / "vllm_runner.py").read_bytes()).hexdigest()
    if (
        input_receipt.get("split") != args.split
        or reflection_metadata["input"]["sha256"] != input_receipt["prompt_sha256"]
        or reflection_metadata["runner_sha256"] != runner_sha256
        or reflection_metadata["base_model"] != config["model"]["id"]
        or reflection_metadata["model_revision"] != config["model"]["revision"]
        or reflection_metadata["model_override"] is not None
        or reflection_metadata["runtime"]["git_dirty"]
        or reflection_metadata.get("output", {}).get("sha256")
        != reflection_generated_sha256
        or int(reflection_metadata.get("output", {}).get("rows", -1))
        != int(input_receipt["rows"])
    ):
        raise ValueError("literal reflection generation provenance is invalid")
    engine = config["evaluation"]["engine"]
    for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        if int(reflection_metadata["engine"][key]) != int(engine[key]):
            raise ValueError(f"literal reflection engine {key} differs")
    if float(reflection_metadata["engine"]["gpu_memory_utilization"]) != float(
        engine["gpu_memory_utilization"]
    ):
        raise ValueError("literal reflection GPU memory utilization differs")
    if (
        reflection_metadata["engine"]["cudagraph_capture_sizes"]
        != engine["cudagraph_capture_sizes"]
    ):
        raise ValueError("literal reflection CUDA-graph geometry differs")
    if bool(reflection_metadata["engine"]["enable_prefix_caching"]) != bool(
        engine["prefix_caching"]
    ):
        raise ValueError("literal reflection prefix caching differs")
    if bool(reflection_metadata["engine_args"]["async_scheduling"]) != bool(
        engine["async_scheduling"]
    ):
        raise ValueError("literal reflection async scheduling differs")
    lock_path = EXP.parents[1] / "requirements-vllm.lock.txt"
    if reflection_metadata["runtime"]["environment_lock"]["sha256"] != hashlib.sha256(
        lock_path.read_bytes()
    ).hexdigest():
        raise ValueError("literal reflection environment lock differs")
    reflection_sampling = reflection_metadata["sampling"]
    expected_reflection = {
        "thinking": "off",
        "n": int(diagnostic["candidate_count"]),
        "max_tokens": int(diagnostic["reflection_max_tokens"]),
        "temperature": float(diagnostic["reflection_temperature"]),
        "top_p": float(diagnostic["reflection_top_p"]),
        "top_k": int(diagnostic["reflection_top_k"]),
        "run_seed": int(diagnostic["reflection_seed"]),
    }
    if any(reflection_sampling[key] != value for key, value in expected_reflection.items()):
        raise ValueError("literal reflection sampling differs from preregistration")
    construction = config["construction"]
    counts = {
        split: int(construction["per_family"][split])
        for split in ("train", "calibration", "qualification", "confirmation")
    }
    tasks = build_corpus(counts, int(construction["seed"]))[args.split]
    task_by_id = {task["task_id"]: task for task in tasks}
    reflected = _read_jsonl(args.reflection_generated)
    if {row["id"] for row in reflected} != set(task_by_id):
        raise ValueError("literal reflection rows do not match the qualification tasks")
    candidate_count = int(diagnostic["candidate_count"])
    prompts = []
    for row in reflected:
        task = task_by_id[row["id"]]
        if len(row["outputs"]) != candidate_count:
            raise ValueError("literal reflection candidate count differs from config")
        for sample_index, output in enumerate(row["outputs"]):
            prompts.append(
                {
                    "id": f"{task['task_id']}::literal::{sample_index}",
                    "messages": [
                        *task["common_messages"],
                        {"role": "user", "content": REFLECTION_QUESTION},
                        {"role": "assistant", "content": str(output["text"])},
                        {"role": "user", "content": ACTION_QUESTION},
                    ],
                    "meta": {
                        "split": args.split,
                        "family": task["family"],
                        "depth": task["depth"],
                        "parent_task_id": task["task_id"],
                        "sample_index": sample_index,
                        "reflection_text_sha256": hashlib.sha256(
                            str(output["text"]).encode()
                        ).hexdigest(),
                    },
                }
            )
    digest = _write_exclusive(args.output, prompts)
    receipt = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "split": args.split,
        "rows": len(prompts),
        "prompt_sha256": digest,
        "source_reflection_generated_sha256": reflection_generated_sha256,
        "source_reflection_metadata_sha256": hashlib.sha256(
            args.reflection_metadata.read_bytes()
        ).hexdigest(),
    }
    _write_exclusive(args.receipt, [receipt])
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
