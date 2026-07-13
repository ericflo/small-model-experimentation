#!/usr/bin/env python3
"""Reduce aggregate-only paired Menagerie CLI events under the frozen gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import yaml

from downstream_common import (
    EXP,
    ROOT,
    fail,
    finite_rate,
    harness,
    is_sha256,
    sha256_file,
)


EVENT_KEYS = {
    "schema_version",
    "tier",
    "seed",
    "arms",
    "firewall_storage",
    "delta",
    "provenance",
}
ARM_KEYS = {
    "model_path",
    "model_weight_sha256",
    "model_config_sha256",
    "generation_config_sha256",
    "merge_receipt_sha256",
    "tokenizer_files",
    "tokenizer_manifest_sha256",
    "tokenizer_compatibility_sha256",
    "aggregate",
    "per_family",
    "within_budget",
    "wall_seconds",
}
CHECKPOINT_FINGERPRINT_KEYS = {
    "model_path",
    "model_weight_sha256",
    "model_config_sha256",
    "generation_config_sha256",
    "merge_receipt_sha256",
    "tokenizer_files",
    "tokenizer_manifest_sha256",
    "tokenizer_compatibility_sha256",
}
PROVENANCE_KEYS = {
    "config_path",
    "config_sha256",
    "bench_path",
    "bench_sha256",
    "analyzer_path",
    "analyzer_sha256",
    "design_lock_path",
    "design_lock_sha256",
    "design_commit",
    "authorization_path",
    "authorization_sha256",
    "public_menagerie_git_tree",
}
PUBLIC_TREE_KEYS = {"repository_path", "git_tree_oid"}
COMPLETE_RESERVATION_KEYS = {
    "schema_version",
    "tier",
    "seed",
    "status",
    "provenance",
    "checkpoint_fingerprints",
    "event_sha256",
}
MENAGERIE_REPOSITORY_PATH = str(Path("benchmarks") / "menagerie")
FORBIDDEN_DETAIL_KEYS = {
    "task",
    "tasks",
    "item",
    "items",
    "result",
    "results",
    "row",
    "rows",
    "case",
    "cases",
    "dyad",
    "dyads",
    "trajectory",
    "trajectories",
    "transcript",
    "transcripts",
    "prompt",
    "prompts",
    "completion",
    "completions",
    "output",
    "outputs",
    "answer",
    "answers",
}
AUTHORIZATION_KEYS = {
    "schema_version",
    "stage",
    "gate",
    "all_whitebox_gates_passed",
    "menagerie_authorized",
    "candidate_model_weight_sha256",
    "incumbent_model_weight_sha256",
    "selected_answer_max_tokens",
    "gate_receipts",
}


def canonical_json_sha256(value: object) -> str:
    """Hash an aggregate receipt without ever touching benchmark internals."""
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        fail(f"Menagerie event is not canonical JSON: {exc}")
    return hashlib.sha256(encoded).hexdigest()


def _is_git_oid(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _git_output(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        fail(
            "cannot resolve public Menagerie Git identity: "
            + (exc.stderr.strip() or str(exc))
        )


def public_menagerie_git_tree_identity(design_commit: str) -> dict:
    """Return the public benchmark tree OID using Git metadata only.

    No file below ``benchmarks/`` is opened here.  Requiring the current tree to
    equal the design-commit tree prevents a resumed paired run from silently
    mixing benchmark implementations or changed backend defaults.
    """
    if not _is_git_oid(design_commit):
        fail("design lock has no immutable commit for Menagerie provenance")
    dirty = _git_output("status", "--short", "--", MENAGERIE_REPOSITORY_PATH)
    if dirty:
        fail("public Menagerie worktree differs from committed Git metadata")
    registered = _git_output(
        "rev-parse", f"{design_commit}:{MENAGERIE_REPOSITORY_PATH}"
    )
    current = _git_output("rev-parse", f"HEAD:{MENAGERIE_REPOSITORY_PATH}")
    if not _is_git_oid(registered) or registered != current:
        fail("public Menagerie Git tree changed after the design lock")
    return {
        "repository_path": MENAGERIE_REPOSITORY_PATH,
        "git_tree_oid": registered,
    }


def load_registered_config(path: Path) -> dict:
    default = (EXP / "configs" / "default.yaml").resolve()
    if Path(path).resolve() != default:
        fail("Menagerie analysis requires the exact frozen default config path")
    try:
        payload = yaml.safe_load(default.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        fail(f"cannot read frozen Menagerie config: {exc}")
    if not isinstance(payload, dict):
        fail("frozen Menagerie config is not an object")
    return payload


def checkpoint_fingerprint(model: Path) -> dict:
    """Re-hash every deployed checkpoint surface consumed by Menagerie."""
    model = Path(model).resolve()
    required = {
        "model.safetensors": model / "model.safetensors",
        "config.json": model / "config.json",
        "generation_config.json": model / "generation_config.json",
        "merge_receipt.json": model / "merge_receipt.json",
    }
    if not model.is_dir() or any(not path.is_file() for path in required.values()):
        fail(f"Menagerie checkpoint is incomplete: {model}")
    try:
        tokenizer = harness.tokenizer_provenance(model)
    except (OSError, ValueError) as exc:
        fail(f"Menagerie checkpoint tokenizer is invalid at {model}: {exc}")
    return {
        "model_path": str(model),
        "model_weight_sha256": sha256_file(required["model.safetensors"]),
        "model_config_sha256": sha256_file(required["config.json"]),
        "generation_config_sha256": sha256_file(
            required["generation_config.json"]
        ),
        "merge_receipt_sha256": sha256_file(required["merge_receipt.json"]),
        "tokenizer_files": tokenizer["tokenizer_files"],
        "tokenizer_manifest_sha256": tokenizer["tokenizer_manifest_sha256"],
        "tokenizer_compatibility_sha256": tokenizer[
            "tokenizer_compatibility_sha256"
        ],
    }


def registered_event_provenance(
    authorization: dict,
    design_lock_path: Path | None = None,
) -> dict:
    """Build the sole accepted provenance record for a Menagerie event."""
    canonical_lock = (EXP / "runs" / "preregistration_receipt.json").resolve()
    lock_path = (
        canonical_lock if design_lock_path is None else Path(design_lock_path).resolve()
    )
    if lock_path != canonical_lock or not lock_path.is_file():
        fail("Menagerie provenance requires the canonical design-lock receipt")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read Menagerie design lock: {exc}")
    design_commit = lock.get("design_commit")
    frozen = lock.get("frozen_files")
    required_frozen = {
        "configs/default.yaml": EXP / "configs" / "default.yaml",
        "scripts/bench.py": EXP / "scripts" / "bench.py",
        "scripts/analyze_menagerie.py": Path(__file__).resolve(),
    }
    if (
        lock.get("schema_version") != 1
        or lock.get("status") != "locked"
        or lock.get("experiment_id") != EXP.name
        or lock.get("model_output_precedes_lock") is not False
        or not isinstance(frozen, dict)
        or not _is_git_oid(design_commit)
    ):
        fail("Menagerie provenance design lock is malformed")
    for relative, file_path in required_frozen.items():
        if not file_path.is_file() or frozen.get(relative) != sha256_file(file_path):
            fail(f"Menagerie provenance drifted from the design lock: {relative}")
    expected_authorization_path = (
        EXP / "analysis" / "whitebox_authorization.json"
    ).resolve()
    if (
        Path(authorization.get("authorization_path", "")).resolve()
        != expected_authorization_path
        or authorization.get("authorization_sha256")
        != sha256_file(expected_authorization_path)
    ):
        fail("Menagerie provenance has a stale white-box authorization")
    return {
        "config_path": str(required_frozen["configs/default.yaml"].resolve()),
        "config_sha256": sha256_file(required_frozen["configs/default.yaml"]),
        "bench_path": str(required_frozen["scripts/bench.py"].resolve()),
        "bench_sha256": sha256_file(required_frozen["scripts/bench.py"]),
        "analyzer_path": str(required_frozen["scripts/analyze_menagerie.py"]),
        "analyzer_sha256": sha256_file(
            required_frozen["scripts/analyze_menagerie.py"]
        ),
        "design_lock_path": str(lock_path),
        "design_lock_sha256": sha256_file(lock_path),
        "design_commit": design_commit,
        "authorization_path": str(expected_authorization_path),
        "authorization_sha256": authorization["authorization_sha256"],
        "public_menagerie_git_tree": public_menagerie_git_tree_identity(
            design_commit
        ),
    }


def checkpoint_fingerprints_from_event(event: dict) -> dict:
    return {
        role: {
            key: event["arms"][role][key]
            for key in CHECKPOINT_FINGERPRINT_KEYS
        }
        for role in ("incumbent", "candidate")
    }


def reservation_path(tier: str, seed: int) -> Path:
    return (
        EXP
        / "runs"
        / "menagerie_reservations"
        / f"{tier}_seed{seed}.json"
    )


def validate_completed_reservation(
    raw_event: dict,
    event: dict,
    expected_provenance: dict,
    *,
    source: str,
) -> dict:
    path = reservation_path(event["tier"], event["seed"])
    try:
        reservation = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Menagerie event is unreserved in {source}: {exc}")
    if (
        not isinstance(reservation, dict)
        or set(reservation) != COMPLETE_RESERVATION_KEYS
        or reservation.get("schema_version") != 1
        or reservation.get("tier") != event["tier"]
        or reservation.get("seed") != event["seed"]
        or reservation.get("status") != "aggregate_event_recorded"
        or reservation.get("provenance") != expected_provenance
        or reservation.get("checkpoint_fingerprints")
        != checkpoint_fingerprints_from_event(event)
        or reservation.get("event_sha256") != canonical_json_sha256(raw_event)
    ):
        fail(f"Menagerie event has no matching completed reservation in {source}")
    return reservation


def _assert_no_detail_keys(value: object, location: str = "event") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_DETAIL_KEYS:
                fail(f"Menagerie aggregate firewall rejected detail key {location}.{key}")
            _assert_no_detail_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        # The aggregate event contract has no list-valued field. Rejecting lists
        # also prevents task-level records hidden under an unfamiliar key.
        fail(f"Menagerie aggregate firewall rejected list-valued field at {location}")


def validate_event(event: dict, cfg: dict, *, source: str) -> dict:
    _assert_no_detail_keys(event)
    if set(event) != EVENT_KEYS:
        fail(f"Menagerie event has non-aggregate keys in {source}: {sorted(set(event) - EVENT_KEYS)}")
    if event.get("schema_version") != 1:
        fail(f"unsupported Menagerie event schema in {source}")
    if event.get("firewall_storage") != "aggregate_and_per_family_only":
        fail(f"Menagerie event lacks the aggregate-only firewall marker in {source}")
    tier = event.get("tier")
    if tier not in cfg["tiers"]:
        fail(f"unregistered Menagerie tier in {source}: {tier!r}")
    if not isinstance(event.get("seed"), int):
        fail(f"Menagerie event seed is not an integer in {source}")
    provenance = event.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
        fail(f"Menagerie event provenance is incomplete in {source}")
    for key in (
        "config_sha256",
        "bench_sha256",
        "analyzer_sha256",
        "design_lock_sha256",
        "authorization_sha256",
    ):
        if not is_sha256(provenance.get(key)):
            fail(f"Menagerie event provenance has an invalid {key} in {source}")
    for key in (
        "config_path",
        "bench_path",
        "analyzer_path",
        "design_lock_path",
        "authorization_path",
    ):
        value = provenance.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            fail(f"Menagerie event provenance has a non-canonical {key} in {source}")
    if not _is_git_oid(provenance.get("design_commit")):
        fail(f"Menagerie event provenance has a moving design ref in {source}")
    public_tree = provenance.get("public_menagerie_git_tree")
    if (
        not isinstance(public_tree, dict)
        or set(public_tree) != PUBLIC_TREE_KEYS
        or public_tree.get("repository_path") != MENAGERIE_REPOSITORY_PATH
        or not _is_git_oid(public_tree.get("git_tree_oid"))
    ):
        fail(f"Menagerie public Git-tree provenance is malformed in {source}")
    arms = event.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"incumbent", "candidate"}:
        fail(f"Menagerie event does not contain exactly the paired arms in {source}")
    normalized_arms = {}
    for arm_name, arm in arms.items():
        if not isinstance(arm, dict) or set(arm) != ARM_KEYS:
            fail(f"Menagerie {arm_name} has non-aggregate fields in {source}")
        if (
            not isinstance(arm.get("model_path"), str)
            or not Path(arm["model_path"]).is_absolute()
        ):
            fail(f"Menagerie {arm_name} model path is not canonical in {source}")
        if not is_sha256(arm.get("model_weight_sha256")):
            fail(f"Menagerie {arm_name} model hash is missing in {source}")
        for key in (
            "model_config_sha256", "generation_config_sha256",
            "merge_receipt_sha256",
            "tokenizer_manifest_sha256", "tokenizer_compatibility_sha256",
        ):
            if not is_sha256(arm.get(key)):
                fail(f"Menagerie {arm_name} {key} is missing in {source}")
        tokenizer_files = arm.get("tokenizer_files")
        if (
            not isinstance(tokenizer_files, dict)
            or set(tokenizer_files) != set(harness.TOKENIZER_FILE_NAMES)
            or any(not is_sha256(value) for value in tokenizer_files.values())
        ):
            fail(f"Menagerie {arm_name} tokenizer file manifest is invalid in {source}")
        expected_manifest = cfg[
            "_anchor_tokenizer_manifest_sha256"
            if arm_name == "incumbent"
            else "_start_tokenizer_manifest_sha256"
        ]
        expected_generation_config = cfg[
            "_anchor_generation_config_sha256"
            if arm_name == "incumbent"
            else "_start_generation_config_sha256"
        ]
        if (
            arm["tokenizer_manifest_sha256"] != expected_manifest
            or arm["generation_config_sha256"] != expected_generation_config
            or arm["tokenizer_compatibility_sha256"]
            != cfg["_tokenizer_compatibility_sha256"]
        ):
            fail(f"Menagerie {arm_name} tokenizer is unregistered in {source}")
        aggregate = finite_rate(arm.get("aggregate"), f"{source}.{arm_name}.aggregate")
        per_family = arm.get("per_family")
        if not isinstance(per_family, dict) or not per_family:
            fail(f"Menagerie {arm_name} per-family aggregate is missing in {source}")
        normalized_family = {
            str(name): finite_rate(value, f"{source}.{arm_name}.per_family.{name}")
            for name, value in per_family.items()
        }
        if arm.get("within_budget") is not True:
            fail(f"Menagerie {arm_name} was not within the registered budget in {source}")
        wall_seconds = arm.get("wall_seconds")
        if (
            not isinstance(wall_seconds, (int, float))
            or isinstance(wall_seconds, bool)
            or not math.isfinite(float(wall_seconds))
            or float(wall_seconds) < 0.0
        ):
            fail(f"Menagerie {arm_name} wall time is malformed in {source}")
        normalized_arms[arm_name] = {
            "model_path": arm["model_path"],
            "model_weight_sha256": arm["model_weight_sha256"],
            "model_config_sha256": arm["model_config_sha256"],
            "generation_config_sha256": arm["generation_config_sha256"],
            "merge_receipt_sha256": arm["merge_receipt_sha256"],
            "tokenizer_files": dict(sorted(tokenizer_files.items())),
            "tokenizer_manifest_sha256": arm["tokenizer_manifest_sha256"],
            "tokenizer_compatibility_sha256": arm[
                "tokenizer_compatibility_sha256"
            ],
            "aggregate": aggregate,
            "per_family": normalized_family,
            "within_budget": True,
            "wall_seconds": float(wall_seconds),
        }
    if set(normalized_arms["candidate"]["per_family"]) != set(
        normalized_arms["incumbent"]["per_family"]
    ):
        fail(f"Menagerie paired arms expose different family aggregates in {source}")
    if normalized_arms["incumbent"]["model_weight_sha256"] != cfg[
        "_anchor_weight_sha256"
    ]:
        fail(f"Menagerie incumbent does not match the registered anchor in {source}")
    observed_delta = normalized_arms["candidate"]["aggregate"] - normalized_arms[
        "incumbent"
    ]["aggregate"]
    if not math.isclose(
        float(event.get("delta")), observed_delta, rel_tol=0.0, abs_tol=1e-12
    ):
        fail(f"Menagerie event delta does not match its paired aggregates in {source}")
    return {
        "schema_version": 1,
        "tier": tier,
        "seed": event["seed"],
        "arms": normalized_arms,
        "firewall_storage": "aggregate_and_per_family_only",
        "delta": observed_delta,
        "provenance": provenance,
    }


def validate_authorization(path: Path, full_cfg: dict) -> dict:
    if path.resolve() != (EXP / "analysis" / "whitebox_authorization.json").resolve():
        fail("white-box authorization is not at the registered path")
    try:
        authorization = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read white-box authorization receipt {path}: {exc}")
    if not isinstance(authorization, dict) or set(authorization) != AUTHORIZATION_KEYS:
        fail("white-box authorization receipt has the wrong contract")
    if (
        authorization.get("schema_version") != 1
        or authorization.get("stage") != "whitebox_authorization"
        or authorization.get("gate") != {
            "passed": True, "verdict": "WHITEBOX_PASS"
        }
        or authorization.get("all_whitebox_gates_passed") is not True
        or authorization.get("menagerie_authorized") is not True
    ):
        fail("white-box authorization did not open the Menagerie seal")
    candidate_hash = authorization.get("candidate_model_weight_sha256")
    incumbent_hash = authorization.get("incumbent_model_weight_sha256")
    if not is_sha256(candidate_hash) or not is_sha256(incumbent_hash):
        fail("white-box authorization omits candidate/incumbent hashes")
    if incumbent_hash != full_cfg["model"]["anchor_weight_sha256"]:
        fail("white-box authorization names the wrong incumbent weights")
    if candidate_hash == incumbent_hash:
        fail("white-box authorization candidate and incumbent hashes are identical")
    if authorization.get("selected_answer_max_tokens") not in full_cfg["evaluation"][
        "interface_answer_rungs"
    ]:
        fail("white-box authorization names an unregistered answer allowance")
    receipt_map = authorization.get("gate_receipts")
    required_receipts = {
        "training_compute_gate",
        "locality_candidate_vs_anchor",
        "calibration_gate",
        "transfer_dev_gate",
        "transfer_confirm_gate",
        "retention_gate",
    }
    if not isinstance(receipt_map, dict) or set(receipt_map) != required_receipts:
        fail("white-box authorization does not name the exact frozen gate set")
    normalized_receipts = {}
    for name, receipt in receipt_map.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(receipt, dict)
            or set(receipt) != {"path", "sha256"}
            or not is_sha256(receipt.get("sha256"))
        ):
            fail(f"malformed white-box gate-receipt provenance: {name!r}")
        receipt_path = Path(receipt["path"])
        if not receipt_path.is_absolute():
            receipt_path = (path.parent / receipt_path).resolve()
        expected_path = EXP / "analysis" / f"{name}.json"
        if (
            receipt_path.resolve() != expected_path.resolve()
            or not receipt_path.is_file()
            or sha256_file(receipt_path) != receipt["sha256"]
        ):
            fail(f"white-box gate receipt changed or is unavailable: {name}")
        gate_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if gate_payload.get("gate", {}).get("passed") is not True:
            fail(f"white-box authorization includes a failed gate: {name}")
        if name == "locality_candidate_vs_anchor":
            if gate_payload.get("auditor_sha256") != sha256_file(
                EXP / "scripts" / "audit_locality.py"
            ):
                fail("white-box locality auditor provenance drifted")
        elif name == "training_compute_gate":
            training_receipts = gate_payload.get("training_receipts")
            expected_arms = set(full_cfg["training"]["arms"])
            expected_keys = {
                "schema_version", "stage", "issuer_sha256", "config_sha256",
                "design_lock_sha256", "serial_forward_tokens_per_epoch",
                "optimizer_steps", "merged_model_weight_sha256",
                "candidate_model_weight_sha256", "max_to_min_ratio",
                "registered_ratio_max", "training_receipts", "gate",
                "menagerie_authorized",
            }
            if (
                set(gate_payload) != expected_keys
                or gate_payload.get("schema_version") != 1
                or gate_payload.get("stage") != "training_compute"
                or gate_payload.get("issuer_sha256")
                != sha256_file(EXP / "scripts" / "run.py")
                or gate_payload.get("config_sha256")
                != sha256_file(EXP / "configs" / "default.yaml")
                or gate_payload.get("design_lock_sha256")
                != sha256_file(EXP / "runs" / "preregistration_receipt.json")
                or gate_payload.get("gate")
                != {"passed": True, "verdict": "TRAINING_COMPUTE_MATCHED"}
                or gate_payload.get("menagerie_authorized") is not False
                or not isinstance(training_receipts, dict)
                or set(training_receipts) != expected_arms
            ):
                fail("white-box training-compute gate provenance drifted")
            artifact_root = Path(full_cfg["artifacts"]["root"])
            if not artifact_root.is_absolute():
                artifact_root = ROOT / artifact_root
            observed_serial = {}
            observed_steps = {}
            observed_merged = {}
            for arm, registration in training_receipts.items():
                expected_training = (
                    artifact_root / "adapters" / arm
                    / "training_receipt.json"
                )
                if (
                    not isinstance(registration, dict)
                    or registration.get("path") != str(expected_training.resolve())
                    or not expected_training.is_file()
                    or registration.get("sha256") != sha256_file(expected_training)
                ):
                    fail(f"white-box training receipt changed: {arm}")
                training = json.loads(expected_training.read_text(encoding="utf-8"))
                if training.get("optimizer_steps") != training.get("max_steps"):
                    fail(f"white-box training stopped early: {arm}")
                observed_serial[arm] = training.get(
                    "serial_forward_tokens_per_epoch"
                )
                observed_steps[arm] = {
                    "planned": int(training.get("max_steps", -1)),
                    "actual": int(training.get("optimizer_steps", -2)),
                }
                merged = artifact_root / "merged" / arm
                observed_merged[arm] = checkpoint_fingerprint(merged)[
                    "model_weight_sha256"
                ]
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in observed_serial.values()
            ):
                fail("white-box training-compute tokens are malformed")
            ratio = max(observed_serial.values()) / min(observed_serial.values())
            registered_max = float(
                full_cfg["training"]["serial_token_compute_ratio_max"]
            )
            if (
                gate_payload.get("serial_forward_tokens_per_epoch")
                != observed_serial
                or gate_payload.get("optimizer_steps") != observed_steps
                or gate_payload.get("merged_model_weight_sha256")
                != observed_merged
                or gate_payload.get("candidate_model_weight_sha256")
                != observed_merged.get("evidence_binding")
                or not math.isclose(
                    float(gate_payload.get("max_to_min_ratio", float("nan"))),
                    ratio, rel_tol=0.0, abs_tol=1e-12,
                )
                or gate_payload.get("registered_ratio_max") != registered_max
                or ratio > registered_max
            ):
                fail("white-box training-compute receipt does not recompute")
        else:
            analyzer = {
                "calibration_gate": "analyze_calibration.py",
                "transfer_dev_gate": "analyze_transfer.py",
                "transfer_confirm_gate": "analyze_transfer.py",
                "retention_gate": "analyze_retention.py",
            }[name]
            if (
                gate_payload.get("analyzer_sha256")
                != sha256_file(EXP / "scripts" / analyzer)
                or gate_payload.get("config_sha256")
                != sha256_file(EXP / "configs" / "default.yaml")
            ):
                fail(f"white-box gate analyzer/config provenance drifted: {name}")
        expected_stage = {
            "calibration_gate": "trained_calibration",
            "transfer_dev_gate": "transfer",
            "transfer_confirm_gate": "transfer",
            "retention_gate": "legacy_retention",
        }.get(name)
        if expected_stage is not None and gate_payload.get("stage") != expected_stage:
            fail(f"white-box gate has the wrong stage: {name}")
        if name == "locality_candidate_vs_anchor":
            if (
                gate_payload.get("after_model_weight_sha256") != candidate_hash
                or gate_payload.get("before_model_weight_sha256") != incumbent_hash
            ):
                fail("white-box locality gate names different checkpoints")
        elif name == "training_compute_gate":
            if gate_payload.get("candidate_model_weight_sha256") != candidate_hash:
                fail("white-box training-compute gate names a different candidate")
        else:
            if gate_payload.get("candidate_model_weight_sha256") != candidate_hash:
                fail(f"white-box gate names a different candidate: {name}")
        if name in {
            "calibration_gate", "transfer_dev_gate", "transfer_confirm_gate",
            "retention_gate",
        }:
            if gate_payload.get("answer_max_tokens") != authorization[
                "selected_answer_max_tokens"
            ]:
                fail(f"white-box gate uses a different answer allowance: {name}")
        if name.startswith("transfer_"):
            expected_block = name.removesuffix("_gate")
            if gate_payload.get("block") != expected_block:
                fail(f"white-box transfer gate has the wrong block: {name}")
        normalized_receipts[name] = {
            "path": str(receipt_path.resolve()),
            "sha256": receipt["sha256"],
        }
    return {
        **authorization,
        "gate_receipts": normalized_receipts,
        "authorization_path": str(path.resolve()),
        "authorization_sha256": sha256_file(path),
    }


def validate_live_event_checkpoints(event: dict, authorization: dict, *, source: str) -> None:
    locality_path = Path(
        authorization["gate_receipts"]["locality_candidate_vs_anchor"]["path"]
    )
    locality = json.loads(locality_path.read_text(encoding="utf-8"))
    expected_weights = {
        "incumbent": authorization["incumbent_model_weight_sha256"],
        "candidate": authorization["candidate_model_weight_sha256"],
    }
    for role, prefix in (("incumbent", "before"), ("candidate", "after")):
        recorded = {
            key: event["arms"][role][key] for key in CHECKPOINT_FINGERPRINT_KEYS
        }
        observed = checkpoint_fingerprint(Path(recorded["model_path"]))
        if observed != recorded:
            fail(f"Menagerie {role} checkpoint changed after its event in {source}")
        if (
            recorded["model_path"] != locality.get(f"{prefix}_model")
            or recorded["model_weight_sha256"] != expected_weights[role]
            or recorded["model_config_sha256"]
            != locality.get(f"{prefix}_model_config_sha256")
            or recorded["generation_config_sha256"]
            != locality.get(f"{prefix}_model_generation_config_sha256")
            or recorded["merge_receipt_sha256"]
            != locality.get(f"{prefix}_merge_receipt_sha256")
            or recorded["tokenizer_manifest_sha256"]
            != locality.get(f"{prefix}_tokenizer_manifest_sha256")
            or recorded["tokenizer_compatibility_sha256"]
            != locality.get(f"{prefix}_tokenizer_compatibility_sha256")
        ):
            fail(f"Menagerie {role} checkpoint is stale or unauthorized in {source}")


def _menagerie_config(full_cfg: dict) -> dict:
    cfg = dict(full_cfg["menagerie"])
    cfg["_anchor_weight_sha256"] = full_cfg["model"]["anchor_weight_sha256"]
    cfg["_start_generation_config_sha256"] = full_cfg["model"][
        "start_generation_config_sha256"
    ]
    cfg["_anchor_generation_config_sha256"] = full_cfg["model"][
        "anchor_generation_config_sha256"
    ]
    cfg["_start_tokenizer_manifest_sha256"] = full_cfg["model"][
        "start_tokenizer_manifest_sha256"
    ]
    cfg["_anchor_tokenizer_manifest_sha256"] = full_cfg["model"][
        "anchor_tokenizer_manifest_sha256"
    ]
    cfg["_tokenizer_compatibility_sha256"] = full_cfg["model"][
        "tokenizer_compatibility_sha256"
    ]
    return cfg


def validate_registered_event(
    raw_event: dict,
    full_cfg: dict,
    authorization: dict,
    *,
    design_lock_path: Path | None = None,
    source: str,
) -> dict:
    """Authenticate one event for analysis or an orchestrator resume-skip."""
    cfg = _menagerie_config(full_cfg)
    event = validate_event(raw_event, cfg, source=source)
    expected_seed = cfg.get("paired_seeds", {}).get(event["tier"])
    if expected_seed is None or event["seed"] != int(expected_seed):
        fail(f"Menagerie event does not use its frozen tier seed in {source}")
    expected_provenance = registered_event_provenance(
        authorization, design_lock_path
    )
    if event["provenance"] != expected_provenance:
        fail(f"Menagerie event provenance is stale or preseeded in {source}")
    validate_completed_reservation(
        raw_event,
        event,
        expected_provenance,
        source=source,
    )
    validate_live_event_checkpoints(event, authorization, source=source)
    return event


def validate_event_for_resume(
    raw_event: dict,
    full_cfg: dict,
    authorization_path: Path,
    design_lock_path: Path,
    *,
    source: str = "Menagerie resume log",
) -> dict:
    """Public hook for ``run.py`` before it skips an existing paired event."""
    authorization = validate_authorization(authorization_path, full_cfg)
    return validate_registered_event(
        raw_event,
        full_cfg,
        authorization,
        design_lock_path=design_lock_path,
        source=source,
    )


def analyze(cfg: dict, events: list[dict], authorization: dict) -> dict:
    expected = {name: int(seed) for name, seed in cfg["paired_seeds"].items()}
    if set(expected) != set(cfg["tiers"]):
        fail("Menagerie tier and paired-seed registries disagree")
    if len(events) != len(expected) or {
        (row["tier"], row["seed"]) for row in events
    } != set(expected.items()):
        fail("Menagerie log contains an unexpected, missing, or duplicate event")
    selected = {}
    for tier, seed in expected.items():
        matches = [row for row in events if row["tier"] == tier and row["seed"] == seed]
        if len(matches) != 1:
            fail(f"expected exactly one preregistered Menagerie event for {tier}/{seed}")
        selected[tier] = matches[0]
    candidate_hashes = {
        row["arms"]["candidate"]["model_weight_sha256"] for row in selected.values()
    }
    if len(candidate_hashes) != 1:
        fail("Menagerie tiers evaluated different candidate checkpoints")
    candidate_hash = next(iter(candidate_hashes))
    if candidate_hash != authorization["candidate_model_weight_sha256"]:
        fail("Menagerie events do not use the white-box-authorized candidate")
    incumbent_hashes = {
        row["arms"]["incumbent"]["model_weight_sha256"] for row in selected.values()
    }
    if incumbent_hashes != {authorization["incumbent_model_weight_sha256"]}:
        fail("Menagerie events do not use the white-box-authorized incumbent")
    locality_path = Path(
        authorization["gate_receipts"]["locality_candidate_vs_anchor"]["path"]
    )
    locality = json.loads(locality_path.read_text(encoding="utf-8"))
    for row in selected.values():
        for arm, prefix in (("incumbent", "before"), ("candidate", "after")):
            event_arm = row["arms"][arm]
            if (
                event_arm["model_config_sha256"]
                != locality.get(f"{prefix}_model_config_sha256")
                or event_arm["generation_config_sha256"]
                != locality.get(f"{prefix}_model_generation_config_sha256")
                or event_arm["merge_receipt_sha256"]
                != locality.get(f"{prefix}_merge_receipt_sha256")
                or event_arm["tokenizer_manifest_sha256"]
                != locality.get(f"{prefix}_tokenizer_manifest_sha256")
                or event_arm["tokenizer_compatibility_sha256"]
                != locality.get(f"{prefix}_tokenizer_compatibility_sha256")
            ):
                fail(f"Menagerie {arm} checkpoint differs from white-box locality")
    deltas = {tier: float(row["delta"]) for tier, row in selected.items()}
    provenance_digests = {
        canonical_json_sha256(row["provenance"]) for row in selected.values()
    }
    checks = {
        "one_positive_tier": max(deltas.values())
        >= float(cfg["minimum_positive_tier_delta"]),
        "no_tier_regression": min(deltas.values())
        >= -float(cfg["maximum_tier_regression"]),
        "aggregate_only_storage": all(
            row["firewall_storage"] == "aggregate_and_per_family_only"
            for row in selected.values()
        ),
        "paired_model_identity": len(candidate_hashes) == 1,
        "identical_authenticated_provenance": len(provenance_digests) == 1,
        "whitebox_authorization": (
            authorization.get("gate", {}).get("passed") is True
            and authorization.get("all_whitebox_gates_passed") is True
            and authorization.get("menagerie_authorized") is True
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "stage": "menagerie",
        "seeds": expected,
        "candidate_model_weight_sha256": candidate_hash,
        "incumbent_model_weight_sha256": authorization[
            "incumbent_model_weight_sha256"
        ],
        "authorization": {
            "path": authorization["authorization_path"],
            "sha256": authorization["authorization_sha256"],
            "selected_answer_max_tokens": authorization[
                "selected_answer_max_tokens"
            ],
            "gate_receipts": authorization["gate_receipts"],
        },
        "deltas": deltas,
        "events": selected,
        "provenance": next(iter(selected.values()))["provenance"],
        "checks": checks,
        "gate": {
            "passed": passed,
            "verdict": "MENAGERIE_PASS" if passed else "MENAGERIE_FAIL",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    full_cfg = load_registered_config(args.config)
    menagerie_cfg = _menagerie_config(full_cfg)
    expected_log = (EXP / "runs" / "menagerie_log.jsonl").resolve()
    expected_out = (EXP / "analysis" / "menagerie_gate.json").resolve()
    if args.log.resolve() != expected_log:
        fail("Menagerie analysis requires the registered aggregate log path")
    if args.out.resolve() != expected_out:
        fail("Menagerie analysis requires the registered final gate path")
    authorization = validate_authorization(args.authorization, full_cfg)
    events = []
    for line_number, line in enumerate(args.log.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"invalid Menagerie JSONL at line {line_number}: {exc}")
        if not isinstance(event, dict):
            fail(f"Menagerie JSONL line {line_number} is not an object")
        events.append(
            validate_registered_event(
                event,
                full_cfg,
                authorization,
                source=f"{args.log}:line {line_number}",
            )
        )
    if not events:
        fail("Menagerie aggregate log is empty")
    result = analyze(menagerie_cfg, events, authorization)
    result["log"] = {"path": str(args.log.resolve()), "sha256": sha256_file(args.log)}
    result["analyzer_sha256"] = sha256_file(Path(__file__).resolve())
    result["config_sha256"] = sha256_file(EXP / "configs" / "default.yaml")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
