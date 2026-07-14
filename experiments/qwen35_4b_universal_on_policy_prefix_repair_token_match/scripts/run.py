#!/usr/bin/env python3
"""Fail-closed intake smoke for the on-policy prefix-repair experiment."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


EXPERIMENT_ID = "qwen35_4b_universal_on_policy_prefix_repair_token_match"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PARENT_WEIGHTS_SHA256 = "16e9dc75a0e33e182e916600ff6e1d75fc46dfa45e870216e2c149a41253c179"
PARENT_CONFIG_SHA256 = "de953bd57502ff728a12d1627d5aacab6284b045428ec7b83026388afd8c47ff"
SEEDS = (77113, 66113, 47, 88009, 78139)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
PARENT = (
    REPOSITORY_ROOT
    / "large_artifacts"
    / "qwen35_4b_universal_close_weight_token_match"
    / "adapters"
    / "close_xi"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoke() -> None:
    config = (EXPERIMENT_ROOT / "configs" / "default.yaml").read_text()
    intake = (EXPERIMENT_ROOT / "idea_intake.md").read_text()
    report = (EXPERIMENT_ROOT / "reports" / "report.md").read_text()

    required_config = (
        f"experiment_id: {EXPERIMENT_ID}",
        f"model_id: {MODEL_ID}",
        f"model_revision: {MODEL_REVISION}",
        f"parent_weights_sha256: {PARENT_WEIGHTS_SHA256}",
        f"parent_config_sha256: {PARENT_CONFIG_SHA256}",
        "status: intake_only_design_review_required",
    )
    missing = [entry for entry in required_config if entry not in config]
    if missing:
        raise RuntimeError(f"intake config missing frozen entries: {missing}")
    for seed in SEEDS:
        if str(seed) not in config or str(seed) not in intake:
            raise RuntimeError(f"reserved seed {seed} is not recorded in both intake and config")
    if len(SEEDS) != len(set(SEEDS)):
        raise RuntimeError("reserved seeds must be distinct")
    for placeholder in ("YYYY-MM-DD", "What specific uncertainty", "Fill this"):
        if placeholder in intake or placeholder in report:
            raise RuntimeError(f"unfinished intake placeholder remains: {placeholder!r}")

    weights = PARENT / "adapter_model.safetensors"
    adapter_config = PARENT / "adapter_config.json"
    if not weights.is_file() or not adapter_config.is_file():
        raise RuntimeError(f"authenticated parent adapter is missing: {PARENT}")
    observed = (sha256_file(weights), sha256_file(adapter_config))
    expected = (PARENT_WEIGHTS_SHA256, PARENT_CONFIG_SHA256)
    if observed != expected:
        raise RuntimeError(f"parent identity mismatch: expected {expected}, observed {observed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--smoke", action="store_true", help="authenticate intake files and proposed parent"
    )
    args = parser.parse_args()
    if not args.smoke:
        parser.error("intake exposes no rollout, training, evaluation, merge, or benchmark stage")
    smoke()
    print(
        "intake smoke passed: one Qwen3.5-4B parent authenticated; "
        "five fresh identities reserved; no scientific stage authorized"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
