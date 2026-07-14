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
from eval_inputs import (  # noqa: E402
    literal_action_prompts,
    literal_action_receipt,
    reflection_receipt,
)
from provenance import (  # noqa: E402
    validate_action_inputs,
    validate_generation_protocol,
    validate_sampling,
)
from scoring import score_literal_reflection_diagnostic  # noqa: E402
from vllm_runner import SamplingConfig  # noqa: E402

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
    parser.add_argument("--base-input-receipt", type=Path, required=True)
    parser.add_argument("--reflection-input-receipt", type=Path, required=True)
    parser.add_argument("--literal-action-input-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    diagnostic = config["evaluation"]["literal_reflection_diagnostic"]
    evaluation = config["evaluation"]
    reflection_meta = json.loads(args.reflection_metadata.read_text())
    action_meta = json.loads(args.action_metadata.read_text())
    base_meta = json.loads(args.base_metadata.read_text())
    base_input_receipt = json.loads(args.base_input_receipt.read_text())
    reflection_input_receipt = json.loads(args.reflection_input_receipt.read_text())
    action_input_receipt = json.loads(args.literal_action_input_receipt.read_text())
    config_path = EXP / "configs" / "default.yaml"
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    split, expected_task_metadata, sealed_action = validate_action_inputs(
        config=config,
        config_path=config_path,
        receipt_path=args.base_input_receipt,
        labels_path=args.labels,
        expected_split="qualification",
    )
    expected_reflection_receipt = reflection_receipt(config, config_sha256, split)
    if reflection_input_receipt != expected_reflection_receipt:
        raise ValueError("literal reflection receipt differs from sealed reconstruction")
    reflected_rows = _read(args.reflection_generated)
    reconstructed_action_prompts = literal_action_prompts(
        config,
        split,
        reflected_rows,
        int(diagnostic["candidate_count"]),
    )
    expected_action_receipt = literal_action_receipt(
        config=config,
        config_sha256=config_sha256,
        split=split,
        prompts=reconstructed_action_prompts,
        source_reflection_generated_sha256=hashlib.sha256(
            args.reflection_generated.read_bytes()
        ).hexdigest(),
        source_reflection_metadata_sha256=hashlib.sha256(
            args.reflection_metadata.read_bytes()
        ).hexdigest(),
        source_reflection_input_receipt_sha256=hashlib.sha256(
            args.reflection_input_receipt.read_bytes()
        ).hexdigest(),
    )
    if action_input_receipt != expected_action_receipt:
        raise ValueError("literal action receipt differs from exact reconstruction")
    if (
        reflection_meta["input"]["sha256"] != expected_reflection_receipt["prompt_sha256"]
        or base_meta["input"]["sha256"] != sealed_action["prompt_sha256"]
        or action_meta["input"]["sha256"] != action_input_receipt["prompt_sha256"]
    ):
        raise ValueError("literal diagnostic input differs from sealed reconstruction")
    validate_sampling(
        reflection_meta,
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
    action_sampling = SamplingConfig(
        thinking="budget",
        thinking_budget=int(evaluation["thinking_budget"]),
        n=1,
        answer_max_tokens=int(evaluation["answer_max_tokens"]),
        temperature=float(evaluation["temperature"]),
        top_p=float(evaluation["top_p"]),
        top_k=int(evaluation["top_k"]),
        run_seed=int(diagnostic["action_seed"]),
    )
    validate_sampling(action_meta, action_sampling)
    validate_sampling(
        base_meta,
        SamplingConfig(
            **{
                **action_sampling.__dict__,
                "n": int(diagnostic["matched_frozen_reserve_candidates"]),
                "run_seed": int(evaluation["sample_seeds"]["qualification"]),
            }
        ),
    )
    protocols = {
        validate_generation_protocol(
            metadata=reflection_meta,
            config=config,
            experiment_root=EXP,
            generated_path=args.reflection_generated,
            expected_rows=int(expected_reflection_receipt["rows"]),
            expect_merged=False,
            expected_stage="screen_training",
            expected_split=split,
            expected_input_kind="literal_reflection",
            expected_source_seed=None,
        ),
        validate_generation_protocol(
            metadata=action_meta,
            config=config,
            experiment_root=EXP,
            generated_path=args.action_generated,
            expected_rows=int(action_input_receipt["rows"]),
            expect_merged=False,
            expected_stage="screen_training",
            expected_split=split,
            expected_input_kind="literal_action",
            expected_source_seed=None,
        ),
        validate_generation_protocol(
            metadata=base_meta,
            config=config,
            experiment_root=EXP,
            generated_path=args.base_generated,
            expected_rows=int(base_input_receipt["rows"]),
            expect_merged=False,
            expected_stage="screen_training",
            expected_split=split,
            expected_input_kind="action",
            expected_source_seed=None,
        ),
    }
    if len(protocols) != 1:
        raise ValueError("literal and matched-base runs used different runtime protocols")
    runtime_protocol_sha256 = protocols.pop()
    base = _read(args.base_generated)
    if any(len(row["outputs"]) != int(diagnostic["matched_frozen_reserve_candidates"]) for row in base):
        raise ValueError("base reserve candidate count differs from preregistration")
    scored = score_literal_reflection_diagnostic(
        reflected_rows,
        _read(args.action_generated),
        base,
        _read(args.labels),
        literal_candidate_count=int(diagnostic["candidate_count"]),
    )
    for row in scored:
        family, depth = expected_task_metadata[row["task_id"]]
        if row["family"] != family:
            raise ValueError("literal scored family differs from sealed reconstruction")
        row["depth"] = depth
        row["runtime_protocol_sha256"] = runtime_protocol_sha256
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
