#!/usr/bin/env python3
"""Score literal reflection and its shortest token-matched frozen base prefix."""

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
from scoring import score_literal_reflection_diagnostic  # noqa: E402

install_benchmark_firewall(EXP.parents[1])


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reflection-generated", type=Path, required=True)
    parser.add_argument("--reflection-metadata", type=Path, required=True)
    parser.add_argument("--action-generated", type=Path, required=True)
    parser.add_argument("--action-metadata", type=Path, required=True)
    parser.add_argument("--base-generated", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--action-input-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    diagnostic = config["evaluation"]["literal_reflection_diagnostic"]
    evaluation = config["evaluation"]
    reflection_meta = json.loads(args.reflection_metadata.read_text())
    action_meta = json.loads(args.action_metadata.read_text())
    base_meta = json.loads(args.base_metadata.read_text())
    input_receipt = json.loads(args.input_receipt.read_text())
    action_input_receipt = json.loads(args.action_input_receipt.read_text())
    runner_sha256 = hashlib.sha256((EXP / "src" / "vllm_runner.py").read_bytes()).hexdigest()
    generated_paths = (
        args.reflection_generated,
        args.action_generated,
        args.base_generated,
    )
    expected_rows = (
        int(input_receipt["rows"]),
        int(action_input_receipt["rows"]),
        int(input_receipt["rows"]),
    )
    engine = evaluation["engine"]
    lock_sha256 = hashlib.sha256(
        (EXP.parents[1] / "requirements-vllm.lock.txt").read_bytes()
    ).hexdigest()
    for metadata, generated_path, row_count in zip(
        (reflection_meta, action_meta, base_meta), generated_paths, expected_rows
    ):
        if (
            metadata["runner_sha256"] != runner_sha256
            or metadata["base_model"] != config["model"]["id"]
            or metadata["model_revision"] != config["model"]["revision"]
            or metadata["model_override"] is not None
            or metadata["runtime"]["git_dirty"]
            or metadata.get("output", {}).get("sha256")
            != hashlib.sha256(generated_path.read_bytes()).hexdigest()
            or int(metadata.get("output", {}).get("rows", -1)) != row_count
            or metadata["runtime"]["environment_lock"]["sha256"] != lock_sha256
        ):
            raise ValueError("literal diagnostic generation provenance is invalid")
        for key in ("max_model_len", "max_num_seqs", "max_num_batched_tokens"):
            if int(metadata["engine"][key]) != int(engine[key]):
                raise ValueError(f"literal diagnostic engine {key} differs")
        if float(metadata["engine"]["gpu_memory_utilization"]) != float(
            engine["gpu_memory_utilization"]
        ):
            raise ValueError("literal diagnostic GPU memory utilization differs")
        if metadata["engine"]["cudagraph_capture_sizes"] != engine["cudagraph_capture_sizes"]:
            raise ValueError("literal diagnostic CUDA-graph geometry differs")
        if bool(metadata["engine"]["enable_prefix_caching"]) != bool(
            engine["prefix_caching"]
        ):
            raise ValueError("literal diagnostic prefix caching differs")
        if bool(metadata["engine_args"]["async_scheduling"]) != bool(
            engine["async_scheduling"]
        ):
            raise ValueError("literal diagnostic async scheduling differs")
    if (
        reflection_meta["input"]["sha256"] != input_receipt["prompt_sha256"]
        or base_meta["input"]["sha256"] != input_receipt["prompt_sha256"]
        or action_meta["input"]["sha256"] != action_input_receipt["prompt_sha256"]
        or hashlib.sha256(args.labels.read_bytes()).hexdigest() != input_receipt["label_sha256"]
    ):
        raise ValueError("literal diagnostic input or labels differ from receipts")
    if (
        action_input_receipt["source_reflection_generated_sha256"]
        != hashlib.sha256(args.reflection_generated.read_bytes()).hexdigest()
        or action_input_receipt["source_reflection_metadata_sha256"]
        != hashlib.sha256(args.reflection_metadata.read_bytes()).hexdigest()
    ):
        raise ValueError("literal action branch is not bound to this reflection generation")
    reflection_expected = {
        "thinking": "off",
        "n": int(diagnostic["candidate_count"]),
        "max_tokens": int(diagnostic["reflection_max_tokens"]),
        "temperature": float(diagnostic["reflection_temperature"]),
        "top_p": float(diagnostic["reflection_top_p"]),
        "top_k": int(diagnostic["reflection_top_k"]),
        "run_seed": int(diagnostic["reflection_seed"]),
    }
    if any(reflection_meta["sampling"][key] != value for key, value in reflection_expected.items()):
        raise ValueError("literal reflection sampling differs")
    action_expected = {
        "thinking": "budget",
        "thinking_budget": int(evaluation["thinking_budget"]),
        "answer_max_tokens": int(evaluation["answer_max_tokens"]),
        "n": 1,
        "temperature": float(evaluation["temperature"]),
        "top_p": float(evaluation["top_p"]),
        "top_k": int(evaluation["top_k"]),
        "run_seed": int(diagnostic["action_seed"]),
    }
    if any(action_meta["sampling"][key] != value for key, value in action_expected.items()):
        raise ValueError("literal action sampling differs")
    base_expected = {
        **{key: action_expected[key] for key in (
            "thinking", "thinking_budget", "answer_max_tokens", "temperature", "top_p", "top_k"
        )},
        "n": int(diagnostic["matched_frozen_reserve_candidates"]),
        "run_seed": int(evaluation["sample_seeds"]["qualification"]),
    }
    if any(base_meta["sampling"][key] != value for key, value in base_expected.items()):
        raise ValueError("literal matched-base reserve sampling differs")
    base = _read(args.base_generated)
    if any(len(row["outputs"]) != int(diagnostic["matched_frozen_reserve_candidates"]) for row in base):
        raise ValueError("base reserve candidate count differs from preregistration")
    scored = score_literal_reflection_diagnostic(
        _read(args.reflection_generated),
        _read(args.action_generated),
        base,
        _read(args.labels),
        literal_candidate_count=int(diagnostic["candidate_count"]),
    )
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in scored
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(
        json.dumps(
            {
                "rows": len(scored),
                "output_sha256": hashlib.sha256(payload).hexdigest(),
                "reflection_generated_sha256": hashlib.sha256(
                    args.reflection_generated.read_bytes()
                ).hexdigest(),
                "action_generated_sha256": hashlib.sha256(
                    args.action_generated.read_bytes()
                ).hexdigest(),
                "base_generated_sha256": hashlib.sha256(
                    args.base_generated.read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
