#!/usr/bin/env python3
"""Model-free smoke entry point; the live path remains sealed."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protocol import (  # noqa: E402
    HF_MODEL_EOS_ID,
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_EOS_ID,
    boundary_pair_smoke_cases,
    smoke_cases,
    validate_boundary_pair_smoke_cases,
    validate_smoke_cases,
)


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def run_smoke() -> dict[str, object]:
    cases = smoke_cases()
    validate_smoke_cases(cases)
    pair_cases = boundary_pair_smoke_cases()
    validate_boundary_pair_smoke_cases(pair_cases)
    payload: dict[str, object] = {
        "schema_version": 2,
        "stage": "model_free_protocol_smoke",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_calls": 0,
        "sampled_outputs": 0,
        "answer_stage_stop_ids": {
            "hf_model_eos_control": HF_MODEL_EOS_ID,
            "tokenizer_eos_candidate": TOKENIZER_EOS_ID,
        },
        "thought_stage_policy_changed": False,
        "cases": {
            label: dataclasses.asdict(result) for label, result in cases.items()
        },
        "boundary_pair_cases": {
            label: dataclasses.asdict(result)
            for label, result in pair_cases.items()
        },
        "boundary_pair_policy": "all_pairs_through_earliest_stop_or_cap",
        "decision": "MODEL_FREE_COMMIT_PROTOCOL_VALID",
    }
    payload["content_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    # Preserve the design-scaffold receipt at runs/smoke/summary.json.
    output = ROOT / "runs" / "protocol_smoke" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="run the model-free protocol smoke"
    )
    args = parser.parse_args()
    if not args.smoke:
        parser.error("live execution is sealed pending construction and design review")
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
