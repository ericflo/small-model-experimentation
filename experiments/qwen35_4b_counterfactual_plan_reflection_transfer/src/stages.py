"""Strict stage-receipt schema and transitive prerequisite validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


STAGES = {
    "calibration_generation",
    "screen_training",
    "replication_training",
    "confirmation",
    "final",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def require_clean_worktree() -> None:
    if subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout:
        raise ValueError("stage consumption requires a clean worktree")


def _validate_claim_schema(claim: dict[str, Any]) -> None:
    schemas = {
        "calibration_gate": {"kind", "sha256", "pass"},
        "decision": {
            "kind",
            "sha256",
            "block",
            "seed",
            "capability_pass",
            "reflection_specific_pass",
            "positive_control_pass",
        },
        "retention": {"kind", "sha256", "seed", "arm", "pass"},
    }
    kind = claim.get("kind")
    if kind not in schemas or set(claim) != schemas[kind]:
        raise ValueError("stage prerequisite claim schema changed")
    digest = claim.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("stage prerequisite lacks a SHA-256 identity")
    if kind == "decision" and (
        not isinstance(claim["capability_pass"], bool)
        or not isinstance(claim["reflection_specific_pass"], bool)
        or claim["positive_control_pass"] not in {None, True, False}
    ):
        raise ValueError("stage decision claim has invalid pass values")


def validate_stage_receipt(
    receipt: dict[str, Any],
    *,
    config: dict[str, Any],
    config_path: Path,
    expected_stage: str,
) -> None:
    if expected_stage not in STAGES:
        raise ValueError(f"unknown expected stage: {expected_stage}")
    require_clean_worktree()
    required = {
        "schema_version",
        "experiment_id",
        "authorized_stage",
        "config_sha256",
        "issuer_git_commit",
        "issuer_script_sha256",
        "prerequisites",
    }
    authorize_script = config_path.parents[1] / "scripts" / "authorize_stage.py"
    if set(receipt) != required:
        raise ValueError("stage receipt top-level schema changed")
    if (
        receipt["schema_version"] != 2
        or receipt["experiment_id"] != config["experiment_id"]
        or receipt["authorized_stage"] != expected_stage
        or receipt["config_sha256"] != sha256_file(config_path)
        or receipt["issuer_git_commit"] != git_commit()
        or receipt["issuer_script_sha256"] != sha256_file(authorize_script)
        or not isinstance(receipt["prerequisites"], list)
    ):
        raise ValueError("stage receipt identity differs from current frozen implementation")
    claims = receipt["prerequisites"]
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("stage prerequisite is not an object")
        _validate_claim_schema(claim)
    screen = int(config["training"]["staged_seeds"]["screen"])
    replication = int(config["training"]["staged_seeds"]["replication"])
    if expected_stage == "calibration_generation":
        if claims:
            raise ValueError("calibration generation stage must have no prerequisites")
        return
    if expected_stage == "screen_training":
        if (
            len(claims) != 1
            or claims[0]["kind"] != "calibration_gate"
            or claims[0]["pass"] is not True
        ):
            raise ValueError("screen stage lacks its exact calibration prerequisite")
        return
    decisions = [claim for claim in claims if claim["kind"] == "decision"]
    retentions = [claim for claim in claims if claim["kind"] == "retention"]
    if expected_stage == "replication_training":
        if len(claims) != 2 or len(decisions) != 1 or len(retentions) != 1:
            raise ValueError("replication stage prerequisite cardinality changed")
        decision, retention = decisions[0], retentions[0]
        if (
            decision["block"] != "qualification"
            or decision["seed"] != screen
            or decision["capability_pass"] is not True
            or decision["positive_control_pass"] is not True
            or retention["seed"] != screen
            or retention["arm"] != "reflection_correct_action"
            or retention["pass"] is not True
        ):
            raise ValueError("replication stage lacks exact screen prerequisites")
        return
    if expected_stage == "confirmation":
        if len(claims) != 4 or len(decisions) != 2 or len(retentions) != 2:
            raise ValueError("confirmation stage prerequisite cardinality changed")
        by_seed = {int(claim["seed"]): claim for claim in decisions}
        retention_by_seed = {int(claim["seed"]): claim for claim in retentions}
        if set(by_seed) != {screen, replication} or set(retention_by_seed) != {
            screen,
            replication,
        }:
            raise ValueError("confirmation stage lacks both frozen seeds")
        for seed in (screen, replication):
            if (
                by_seed[seed]["block"] != "qualification"
                or by_seed[seed]["capability_pass"] is not True
                or retention_by_seed[seed]["arm"] != "reflection_correct_action"
                or retention_by_seed[seed]["pass"] is not True
            ):
                raise ValueError("confirmation stage has a failing prerequisite")
        if by_seed[screen]["positive_control_pass"] is not True:
            raise ValueError("confirmation stage lacks screen positive-control success")
        if by_seed[replication]["positive_control_pass"] is not None:
            raise ValueError("confirmation stage contains an unauthorized replication positive control")
        return
    if len(claims) != 2 or len(decisions) != 2 or retentions:
        raise ValueError("final stage prerequisite cardinality changed")
    by_seed = {int(claim["seed"]): claim for claim in decisions}
    if set(by_seed) != {screen, replication} or any(
        claim["block"] != "confirmation" or claim["capability_pass"] is not True
        for claim in by_seed.values()
    ):
        raise ValueError("final stage lacks both passing confirmation decisions")
    if any(claim["positive_control_pass"] is not None for claim in by_seed.values()):
        raise ValueError("final stage contains confirmation positive-control evidence")


def read_and_validate_stage_receipt(
    path: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    expected_stage: str,
) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    validate_stage_receipt(
        receipt,
        config=config,
        config_path=config_path,
        expected_stage=expected_stage,
    )
    return receipt
