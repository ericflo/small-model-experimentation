#!/usr/bin/env python3
"""Run one firewall-clean paired Menagerie incumbent/candidate event."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
MENAGERIE = ROOT / "benchmarks" / "menagerie" / "run.py"
LOG = EXP / "runs" / "menagerie_log.jsonl"
BASELINE_SEED = 31337
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

import harness  # noqa: E402
from analyze_menagerie import (  # noqa: E402
    canonical_json_sha256,
    checkpoint_fingerprint,
    registered_event_provenance,
    validate_authorization,
    validate_registered_event,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def used_seeds() -> set[int]:
    seeds = {BASELINE_SEED}
    for path in (ROOT / "experiments").glob("*/runs/menagerie_log.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                payload = json.loads(line)
                if "seed" in payload:
                    seeds.add(int(payload["seed"]))
    for path in (ROOT / "experiments").glob(
        "*/runs/menagerie_reservations/*.json"
    ):
        payload = json.loads(path.read_text())
        if "seed" in payload:
            seeds.add(int(payload["seed"]))
    return seeds


def run_arm(
    tier: str,
    seed: int,
    arm: str,
    model: Path,
    expected_fingerprint: dict,
) -> dict:
    aggregate_path = EXP / "runs" / "menagerie" / f"{tier}_seed{seed}_{arm}.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_fingerprint(model) != expected_fingerprint:
        raise SystemExit(f"Menagerie {arm} checkpoint changed before public execution")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="menagerie_raw_") as directory:
        raw = Path(directory) / "raw.json"
        completed = subprocess.run(
            [
                str(PYTHON), str(MENAGERIE), "--tier", tier, "--seed", str(seed),
                "--model-id", str(model.resolve()), "--out", str(raw),
            ],
            cwd=MENAGERIE.parent,
            env={
                **os.environ,
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            check=False,
        )
        if completed.returncode:
            raise SystemExit(
                f"Menagerie {arm} failed with exit {completed.returncode}"
            )
        # The task-level CLI payload exists only in this auto-cleaned temporary
        # directory and is reduced before any experiment-local write.
        payload = json.loads(raw.read_text())
    per_family = {}
    for family, stats in payload["per_family"].items():
        value = stats.get("score", stats.get("mean")) if isinstance(stats, dict) else stats
        if value is None:
            raise SystemExit(f"unrecognized aggregate entry for {family}")
        per_family[family] = float(value)
    observed_fingerprint = checkpoint_fingerprint(model)
    if observed_fingerprint != expected_fingerprint:
        raise SystemExit(f"Menagerie {arm} checkpoint changed during public execution")
    aggregate_only = {
        **observed_fingerprint,
        "aggregate": float(payload["aggregate"]),
        "per_family": per_family,
        "within_budget": payload.get("within_budget"),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    temporary = aggregate_path.with_suffix(".aggregate.tmp")
    temporary.write_text(json.dumps(aggregate_only, indent=2, sort_keys=True) + "\n")
    temporary.replace(aggregate_path)
    return aggregate_only


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["quick", "medium"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--design-lock", type=Path, required=True)
    args = parser.parse_args()
    try:
        harness.validate_model_execution_lock(EXP, args.design_lock, "scripts/bench.py")
    except ValueError as exc:
        raise SystemExit(f"Menagerie is not design-locked: {exc}") from exc
    full_cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    try:
        registered_checkpoints = {
            "incumbent": harness.validate_registered_checkpoint(
                EXP, args.incumbent, full_cfg, args.design_lock, "anchor"
            ),
            "candidate": harness.validate_registered_checkpoint(
                EXP,
                args.candidate,
                full_cfg,
                args.design_lock,
                "evidence_binding",
            ),
        }
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Menagerie checkpoint is not registered: {exc}") from exc
    authorization = validate_authorization(args.authorization, full_cfg)
    expected_seed = int(full_cfg["menagerie"]["paired_seeds"][args.tier])
    if args.seed != expected_seed:
        raise SystemExit(
            f"Menagerie {args.tier} requires frozen seed {expected_seed}, not {args.seed}"
        )
    if args.seed in used_seeds():
        raise SystemExit(f"Menagerie seed {args.seed} is already in the public log union")
    if (
        authorization.get("incumbent_model_weight_sha256")
        != registered_checkpoints["incumbent"]["model_weight_sha256"]
        or authorization.get("candidate_model_weight_sha256")
        != registered_checkpoints["candidate"]["model_weight_sha256"]
    ):
        raise SystemExit("white-box authorization names different model weights")
    locality_registration = authorization["gate_receipts"][
        "locality_candidate_vs_anchor"
    ]
    locality = json.loads(Path(locality_registration["path"]).read_text())
    for role, path, prefix, expected_manifest, expected_generation_config in (
        (
            "incumbent", args.incumbent, "before",
            full_cfg["model"]["anchor_tokenizer_manifest_sha256"],
            full_cfg["model"]["anchor_generation_config_sha256"],
        ),
        (
            "candidate", args.candidate, "after",
            full_cfg["model"]["start_tokenizer_manifest_sha256"],
            full_cfg["model"]["start_generation_config_sha256"],
        ),
    ):
        tokenizer = harness.tokenizer_provenance(path)
        if (
            locality.get(f"{prefix}_model") != str(path.resolve())
            or locality.get(f"{prefix}_model_config_sha256")
            != sha256_file(path / "config.json")
            or locality.get(f"{prefix}_merge_receipt_sha256")
            != sha256_file(path / "merge_receipt.json")
            or locality.get(f"{prefix}_model_generation_config_sha256")
            != sha256_file(path / "generation_config.json")
            or sha256_file(path / "generation_config.json")
            != expected_generation_config
            or locality.get(f"{prefix}_tokenizer_manifest_sha256")
            != tokenizer["tokenizer_manifest_sha256"]
            or tokenizer["tokenizer_manifest_sha256"] != expected_manifest
            or tokenizer["tokenizer_compatibility_sha256"]
            != full_cfg["model"]["tokenizer_compatibility_sha256"]
        ):
            raise SystemExit(f"white-box authorization names a stale {role} checkpoint")
    fingerprints = {
        "incumbent": checkpoint_fingerprint(args.incumbent),
        "candidate": checkpoint_fingerprint(args.candidate),
    }
    provenance = registered_event_provenance(authorization, args.design_lock)
    reservation = (
        EXP / "runs" / "menagerie_reservations"
        / f"{args.tier}_seed{args.seed}.json"
    )
    reservation.parent.mkdir(parents=True, exist_ok=True)
    reservation_payload = {
        "schema_version": 1,
        "tier": args.tier,
        "seed": args.seed,
        "provenance": provenance,
        "checkpoint_fingerprints": fingerprints,
        "status": "reserved_before_first_public_call",
    }
    try:
        with reservation.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(reservation_payload, indent=2, sort_keys=True) + "\n"
            )
    except FileExistsError as exc:
        raise SystemExit(
            f"Menagerie seed reservation already exists: {reservation}"
        ) from exc
    event = {
        "schema_version": 1,
        "tier": args.tier,
        "seed": args.seed,
        "arms": {
            "incumbent": run_arm(
                args.tier,
                args.seed,
                "incumbent",
                args.incumbent,
                fingerprints["incumbent"],
            ),
            "candidate": run_arm(
                args.tier,
                args.seed,
                "candidate",
                args.candidate,
                fingerprints["candidate"],
            ),
        },
        "firewall_storage": "aggregate_and_per_family_only",
        "provenance": provenance,
    }
    event["delta"] = (
        event["arms"]["candidate"]["aggregate"]
        - event["arms"]["incumbent"]["aggregate"]
    )
    reservation_payload["status"] = "aggregate_event_recorded"
    reservation_payload["event_sha256"] = canonical_json_sha256(event)
    reservation_temporary = reservation.with_suffix(".complete.tmp")
    reservation_temporary.write_text(
        json.dumps(reservation_payload, indent=2, sort_keys=True) + "\n"
    )
    reservation_temporary.replace(reservation)
    # Authenticate the exact aggregate event before making it resumable in the
    # shared append-only log.  This performs only local receipt/Git checks.
    validate_registered_event(
        event,
        full_cfg,
        authorization,
        design_lock_path=args.design_lock,
        source="fresh Menagerie aggregate event",
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    print(json.dumps(event, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
