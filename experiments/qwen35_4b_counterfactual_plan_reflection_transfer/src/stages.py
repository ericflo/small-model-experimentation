"""Strict stage receipts with replayed, transitive prerequisite validation."""

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


def _validate_claim_schema(claim: dict[str, Any]) -> Path:
    kinds = {"calibration_gate", "decision", "retention"}
    if claim.get("kind") not in kinds or set(claim) != {"kind", "path", "sha256"}:
        raise ValueError("stage prerequisite claim schema changed")
    digest = claim.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("stage prerequisite lacks a SHA-256 identity")
    path_value = claim.get("path")
    if not isinstance(path_value, str):
        raise ValueError("stage prerequisite lacks an absolute artifact path")
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file() or sha256_file(path) != digest:
        raise ValueError("stage prerequisite artifact is absent or differs from its claim")
    return path


def _resolved_claim(
    claim: dict[str, Any],
    *,
    config: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    from gate_artifacts import validate_gate_artifact

    return validate_gate_artifact(
        _validate_claim_schema(claim),
        kind=claim["kind"],
        config=config,
        config_path=config_path,
        experiment_root=config_path.parents[1],
    )


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
        receipt["schema_version"] != 3
        or receipt["experiment_id"] != config["experiment_id"]
        or receipt["authorized_stage"] != expected_stage
        or receipt["config_sha256"] != sha256_file(config_path)
        or receipt["issuer_git_commit"] != git_commit()
        or receipt["issuer_script_sha256"] != sha256_file(authorize_script)
        or not isinstance(receipt["prerequisites"], list)
    ):
        raise ValueError("stage receipt identity differs from current frozen implementation")
    claims = receipt["prerequisites"]
    resolved: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("stage prerequisite is not an object")
        resolved.append(
            _resolved_claim(claim, config=config, config_path=config_path)
        )
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
            or resolved[0].get("gate", {}).get("pass") is not True
        ):
            raise ValueError("screen stage lacks its exact calibration prerequisite")
        return
    decisions = [
        value for claim, value in zip(claims, resolved) if claim["kind"] == "decision"
    ]
    retentions = [
        value for claim, value in zip(claims, resolved) if claim["kind"] == "retention"
    ]
    if expected_stage == "replication_training":
        if len(claims) != 2 or len(decisions) != 1 or len(retentions) != 1:
            raise ValueError("replication stage prerequisite cardinality changed")
        decision, retention = decisions[0], retentions[0]
        if (
            decision["block"] != "qualification"
            or decision["seed"] != screen
            or decision.get("capability", {}).get("capability_pass") is not True
            or decision.get("positive_control", {}).get("pass") is not True
            or retention["seed"] != screen
            or retention["arm"] != "reflection_correct_action"
            or retention.get("gate", {}).get("pass") is not True
        ):
            raise ValueError("replication stage lacks exact screen prerequisites")
        return
    if expected_stage == "confirmation":
        if len(claims) != 4 or len(decisions) != 2 or len(retentions) != 2:
            raise ValueError("confirmation stage prerequisite cardinality changed")
        by_seed = {int(value["seed"]): value for value in decisions}
        retention_by_seed = {int(value["seed"]): value for value in retentions}
        if set(by_seed) != {screen, replication} or set(retention_by_seed) != {
            screen,
            replication,
        }:
            raise ValueError("confirmation stage lacks both frozen seeds")
        for seed in (screen, replication):
            if (
                by_seed[seed]["block"] != "qualification"
                or by_seed[seed].get("capability", {}).get("capability_pass") is not True
                or retention_by_seed[seed]["arm"] != "reflection_correct_action"
                or retention_by_seed[seed].get("gate", {}).get("pass") is not True
            ):
                raise ValueError("confirmation stage has a failing prerequisite")
        if by_seed[screen].get("positive_control", {}).get("pass") is not True:
            raise ValueError("confirmation stage lacks screen positive-control success")
        if "positive_control" in by_seed[replication]:
            raise ValueError("confirmation stage contains an unauthorized replication positive control")
        return
    if len(claims) != 2 or len(decisions) != 2 or retentions:
        raise ValueError("final stage prerequisite cardinality changed")
    by_seed = {int(value["seed"]): value for value in decisions}
    if set(by_seed) != {screen, replication} or any(
        value["block"] != "confirmation"
        or value.get("capability", {}).get("capability_pass") is not True
        for value in by_seed.values()
    ):
        raise ValueError("final stage lacks both passing confirmation decisions")
    if any("positive_control" in value for value in by_seed.values()):
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
