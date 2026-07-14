#!/usr/bin/env python3
"""Model-free scaffold smoke for the fresh temporal-replay successor."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PARENT = ROOT / "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial"
SMOKE = EXP / "runs/smoke/summary.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_canonical(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != payload:
        raise RuntimeError("smoke receipt changed")
    path.write_text(payload)


def smoke() -> dict[str, object]:
    required = (
        EXP / "README.md",
        EXP / "idea_intake.md",
        EXP / "configs/default.yaml",
        EXP / "reports/preregistration.md",
        EXP / "reports/design_review.md",
        PARENT / "runs/mechanics/failure.json",
        PARENT / "reports/mechanics_failure_review.md",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required design files missing: {missing}")
    config = (EXP / "configs/default.yaml").read_text()
    for literal in (
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "tokenizer-eos-residual-mechanics-fresh-replay-v1",
        "forbid_parent_sampled_bundles: true",
        "model_calls_authorized: false",
        "hidden_read_authorized: false",
    ):
        if literal not in config:
            raise RuntimeError(f"model-free boundary missing: {literal}")
    return {
        "benchmark_files_read": [],
        "design_review_status": "PENDING_INDEPENDENT_REVIEW",
        "freshness_controls_declared": 7,
        "hidden_files_read": [],
        "lifecycle_controls_declared": 8,
        "model": "Qwen/Qwen3.5-4B",
        "model_calls": 0,
        "parent_failure_sha256": sha256_file(PARENT / "runs/mechanics/failure.json"),
        "parent_sampled_bundles_read": [],
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "sampled_model_outputs": 0,
        "schema_version": 1,
        "stage": "model_free_scaffold_smoke",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="check the scaffold without running the full experiment")
    args = parser.parse_args()

    if args.smoke:
        value = smoke()
        write_canonical(SMOKE, value)
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    parser.error("implement the full experiment run before using this command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
