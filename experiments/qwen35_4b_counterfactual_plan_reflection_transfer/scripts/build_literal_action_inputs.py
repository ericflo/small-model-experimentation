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

from eval_inputs import (  # noqa: E402
    literal_action_prompts,
    literal_action_receipt,
    reflection_receipt,
)
from provenance import validate_generation_protocol, validate_sampling  # noqa: E402
from runtime_contract import require_detached_execution_worktree  # noqa: E402
from vllm_runner import SamplingConfig  # noqa: E402


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
    require_detached_execution_worktree(EXP.parents[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("qualification",), required=True)
    parser.add_argument("--reflection-generated", type=Path, required=True)
    parser.add_argument("--reflection-metadata", type=Path, required=True)
    parser.add_argument("--reflection-input-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.receipt.resolve():
        raise ValueError("--output and --receipt must be distinct paths")
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    if config["authorization"]["evaluation"] is not True:
        raise SystemExit("evaluation is not authorized")
    reflection_metadata = json.loads(args.reflection_metadata.read_text())
    input_receipt = json.loads(args.reflection_input_receipt.read_text())
    reflection_generated_sha256 = hashlib.sha256(
        args.reflection_generated.read_bytes()
    ).hexdigest()
    diagnostic = config["evaluation"]["literal_reflection_diagnostic"]
    config_path = EXP / "configs" / "default.yaml"
    expected_input_receipt = reflection_receipt(
        config, hashlib.sha256(config_path.read_bytes()).hexdigest(), args.split
    )
    if input_receipt != expected_input_receipt:
        raise ValueError("literal reflection input receipt differs from sealed reconstruction")
    if reflection_metadata["input"]["sha256"] != expected_input_receipt["prompt_sha256"]:
        raise ValueError("literal reflection generation used the wrong sealed prompts")
    validate_sampling(
        reflection_metadata,
        SamplingConfig(
            thinking="off",
            n=int(diagnostic["candidate_count"]),
            max_tokens=int(diagnostic["reflection_max_tokens"]),
            temperature=float(diagnostic["reflection_temperature"]),
            top_p=float(diagnostic["reflection_top_p"]),
            top_k=int(diagnostic["reflection_top_k"]),
            run_seed=int(diagnostic["reflection_seed"]),
        ),
    )
    validate_generation_protocol(
        metadata=reflection_metadata,
        config=config,
        experiment_root=EXP,
        generated_path=args.reflection_generated,
        expected_rows=int(expected_input_receipt["rows"]),
        expect_merged=False,
        expected_stage="screen_training",
        expected_split=args.split,
        expected_input_kind="literal_reflection",
        expected_source_seed=None,
    )
    reflected = _read_jsonl(args.reflection_generated)
    prompts = literal_action_prompts(
        config, args.split, reflected, int(diagnostic["candidate_count"])
    )
    digest = _write_exclusive(args.output, prompts)
    receipt = literal_action_receipt(
        config=config,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        split=args.split,
        prompts=prompts,
        source_reflection_generated_sha256=reflection_generated_sha256,
        source_reflection_metadata_sha256=hashlib.sha256(
            args.reflection_metadata.read_bytes()
        ).hexdigest(),
        source_reflection_input_receipt_sha256=hashlib.sha256(
            args.reflection_input_receipt.read_bytes()
        ).hexdigest(),
    )
    if receipt["prompt_sha256"] != digest:
        raise RuntimeError("literal action prompt serialization changed")
    _write_exclusive(args.receipt, [receipt])
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
