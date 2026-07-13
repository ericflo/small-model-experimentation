#!/usr/bin/env python3
"""Run one authorized aggregate-only Menagerie event through the firewall."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
PY = REPO / ".venv" / "bin" / "python"
GATEWAY = REPO / "scripts" / "run_benchmark_aggregate.py"
MENAGERIE = REPO / "benchmarks" / "menagerie" / "run.py"
AUTHORIZATION = EXP / "analysis" / "benchmark_authorization.json"
CONFIRMATION = EXP / "analysis" / "confirmation.json"
PREREGISTRATION = EXP / "runs" / "preregistration_receipt.json"
ANALYZER = EXP / "scripts" / "analyze_benchmark.py"
CONFIRMATION_ANALYZER = EXP / "scripts" / "analyze_confirmation.py"
CONFIRMATION_EVALUATOR = EXP / "scripts" / "eval_policy.py"
CONTROL_REMATCH = EXP / "src" / "control_rematch.py"
AUTHORIZER = EXP / "scripts" / "authorize_benchmark.py"
BACKEND = "qwen_vllm"
BENCHMARK_EVENT_COUNT = 33
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from run_benchmark_aggregate import (  # noqa: E402
    PUBLIC_FAMILY_KEYS,
    benchmark_source_inventory,
)
from io_utils import (  # noqa: E402
    canonical_hash,
    confirmation_evaluator_source_inventory,
    load_config,
    resolve_repo_path,
    sha256_file,
    write_json,
)


GATEWAY_KEYS = frozenset(
    {
        "schema_version",
        "stage",
        "tier",
        "seed",
        "backend",
        "model",
        "model_merge_receipt_sha256",
        "benchmark_runner_sha256",
        "benchmark_source_inventory_sha256",
        "benchmark_source_file_count",
        "aggregate",
        "per_family",
        "within_budget",
        "wall_seconds",
    }
)
EVENT_KEYS = frozenset(
    {
        "schema_version",
        "stage",
        "config",
        "config_sha256",
        "authorization",
        "authorization_sha256",
        "confirmation_sha256",
        "tier",
        "seed",
        "label",
        "backend",
        "model",
        "model_merge_receipt_sha256",
        "model_weight_inventory_sha256",
        "model_config_sha256",
        "model_inference_inventory_sha256",
        "aggregate_gateway_sha256",
        "benchmark_runner_sha256",
        "benchmark_source_inventory_sha256",
        "benchmark_source_file_count",
        "aggregate",
        "per_family",
        "within_budget",
        "wall_seconds",
    }
)
EVENT_BINDING_KEYS = frozenset(
    {
        "tier",
        "seed",
        "label",
        "model",
        "model_merge_receipt_sha256",
        "model_weight_inventory_sha256",
        "model_config_sha256",
        "model_inference_inventory_sha256",
    }
)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def model_provenance(model: Path) -> dict[str, str]:
    """Verify and hash weights plus every root-level inference file."""

    model = model.resolve()
    receipt_path = model / "merge_receipt.json"
    config_path = model / "config.json"
    if not receipt_path.is_file() or not config_path.is_file():
        raise ValueError("merged checkpoint provenance files are missing")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("merged checkpoint receipt is invalid") from exc
    rows = receipt.get("weight_files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("merged checkpoint receipt lacks a weight inventory")
    inventory = []
    names = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "sha256"}:
            raise ValueError("merged checkpoint weight inventory is malformed")
        name, expected = row["name"], row["sha256"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise ValueError("merged checkpoint weight inventory entry is invalid")
        artifact = model / name
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError("merged checkpoint weight hash mismatch")
        names.append(name)
        inventory.append({"name": name, "sha256": expected})
    if len(names) != len(set(names)):
        raise ValueError("merged checkpoint weight inventory contains duplicates")
    actual_names = sorted(path.name for path in model.glob("*.safetensors"))
    if sorted(names) != actual_names:
        raise ValueError("merged checkpoint weight inventory is incomplete")
    inference_inventory = [
        {
            "path": path.relative_to(model).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(model.rglob("*"))
        if path.is_file()
    ]
    if not inference_inventory:
        raise ValueError("merged checkpoint inference inventory is empty")
    return {
        "model": str(model),
        "model_merge_receipt_sha256": sha256_file(receipt_path),
        "model_weight_inventory_sha256": canonical_hash(
            sorted(inventory, key=lambda row: row["name"])
        ),
        "model_config_sha256": sha256_file(config_path),
        "model_inference_inventory_sha256": canonical_hash(inference_inventory),
    }


def validate_authorization_artifacts(payload: dict) -> None:
    """Rehash the procedural evidence inputs bound by preauthorization."""

    rows = payload.get("evidence_artifacts")
    integrations = payload.get("integration_receipts")
    controls = payload.get("controls_receipt")
    confirmation = payload.get("confirmation_artifacts")
    if (
        not isinstance(rows, list)
        or not rows
        or not isinstance(integrations, list)
        or not integrations
        or not isinstance(controls, dict)
        or not isinstance(confirmation, list)
        or not confirmation
        or rows
        != sorted(
            [*integrations, controls, *confirmation],
            key=lambda row: row.get("path", "") if isinstance(row, dict) else "",
        )
    ):
        raise ValueError("authorization evidence inventory is missing")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("authorization evidence binding is malformed")
        try:
            path = Path(row["path"]).resolve()
        except (TypeError, ValueError) as exc:
            raise ValueError("authorization evidence path is invalid") from exc
        expected = row["sha256"]
        if (
            path in seen
            or not isinstance(expected, str)
            or len(expected) != 64
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise ValueError("authorization evidence hash is stale")
        seen.add(path)


def _tier_seeds(config: dict) -> dict[str, list[int]]:
    first = int(config["benchmark"]["first_seed"])
    quick_n = int(config["benchmark"]["quick_events"])
    medium_n = int(config["benchmark"]["medium_events"])
    return {
        "quick": list(range(first, first + quick_n)),
        "medium": list(range(first + quick_n, first + quick_n + medium_n)),
    }


def _expected_models(config: dict, tier: str) -> dict[str, Path]:
    root = resolve_repo_path(config["model"]["artifacts_root"])
    training_seed = int(config["seeds"]["integration_training"][0])
    final_round = int(config["mopd"]["rounds"]) - 1
    primary = (
        root / "merged" / "primary" / f"seed_{training_seed}" / f"round_{final_round}"
    )
    return {
        "primary": primary.resolve(),
        "soup": resolve_repo_path(config["model"]["student_checkpoint"]).resolve(),
        "visible": resolve_repo_path(
            config["model"]["quick_teacher" if tier == "quick" else "deep_teacher"]
        ).resolve(),
    }


def _authorization(config: dict, config_path: Path) -> tuple[dict, dict]:
    if not AUTHORIZATION.is_file():
        raise SystemExit("independent benchmark authorization receipt is missing")
    try:
        payload = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
        confirmation = json.loads(CONFIRMATION.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("benchmark authorization chain is unreadable") from exc
    source_inventory = benchmark_source_inventory(MENAGERIE.parent)
    evaluator_source = confirmation_evaluator_source_inventory()
    required = {
        "schema_version": 2,
        "stage": "benchmark_aggregate_authorization",
        "config_sha256": sha256_file(config_path),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "confirmation_sha256": sha256_file(CONFIRMATION),
        "aggregate_gateway_sha256": sha256_file(GATEWAY),
        "benchmark_runner_sha256": sha256_file(MENAGERIE),
        "benchmark_source_inventory_sha256": source_inventory["sha256"],
        "benchmark_source_file_count": source_inventory["file_count"],
        "bench_sha256": sha256_file(Path(__file__)),
        "analyzer_sha256": sha256_file(ANALYZER),
        "confirmation_analyzer_sha256": sha256_file(CONFIRMATION_ANALYZER),
        "confirmation_evaluator_sha256": sha256_file(CONFIRMATION_EVALUATOR),
        "confirmation_evaluator_source_inventory_sha256": evaluator_source["sha256"],
        "confirmation_evaluator_source_file_count": evaluator_source["file_count"],
        "control_rematch_sha256": sha256_file(CONTROL_REMATCH),
        "authorizer_sha256": sha256_file(AUTHORIZER),
        "backend": BACKEND,
        "downstream_authorization": "aggregate_only_benchmark_cli",
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise SystemExit("independent benchmark authorization receipt is stale")
    if not payload.get("gate", {}).get("passed"):
        raise SystemExit("independent benchmark provenance audit did not pass")
    try:
        validate_authorization_artifacts(payload)
    except ValueError as exc:
        raise SystemExit("independent benchmark evidence bindings are stale") from exc
    if (
        confirmation.get("stage") != "two_block_same_prefix_advantage_confirmation"
        or not confirmation.get("gate", {}).get("passed")
        or confirmation.get("downstream_authorization") != "benchmark_cli"
        or confirmation.get("config_sha256") != sha256_file(config_path)
        or payload.get("confirmation_manifest_sha256")
        != confirmation.get("manifest_sha256")
    ):
        raise SystemExit("procedural confirmation does not authorize benchmark CLI")
    try:
        confirmation_manifest = Path(confirmation["manifest"]).resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("procedural confirmation manifest path is invalid") from exc
    if (
        not confirmation_manifest.is_file()
        or sha256_file(confirmation_manifest)
        != confirmation.get("manifest_sha256")
    ):
        raise SystemExit("procedural confirmation manifest provenance is stale")

    rows = payload.get("events")
    if not isinstance(rows, list):
        raise SystemExit("benchmark authorization lacks an event inventory")
    keyed: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != EVENT_BINDING_KEYS:
            raise SystemExit("benchmark authorization event binding is malformed")
        try:
            key = (str(row["tier"]), int(row["seed"]), str(row["label"]))
        except (TypeError, ValueError) as exc:
            raise SystemExit("benchmark authorization event key is malformed") from exc
        if key in keyed:
            raise SystemExit("benchmark authorization contains duplicate events")
        keyed[key] = row
    tier_seeds = _tier_seeds(config)
    expected_keys = {
        (tier, seed, label)
        for tier, seeds in tier_seeds.items()
        for seed in seeds
        for label in ("primary", "soup", "visible")
    }
    if len(expected_keys) != BENCHMARK_EVENT_COUNT or set(keyed) != expected_keys:
        raise SystemExit("benchmark authorization event inventory is stale")

    for key, row in keyed.items():
        tier, _, label = key
        expected_model = _expected_models(config, tier)[label]
        try:
            authorized_model = Path(row["model"]).resolve()
        except (TypeError, ValueError) as exc:
            raise SystemExit("benchmark authorization model path is invalid") from exc
        if authorized_model != expected_model:
            raise SystemExit("benchmark authorization model mapping is invalid")
    return payload, keyed


def _validate_gateway(
    payload: Any,
    *,
    tier: str,
    seed: int,
    model: Path,
    model_receipt_sha256: str,
) -> dict:
    if not isinstance(payload, dict) or set(payload) != GATEWAY_KEYS:
        raise ValueError("aggregate gateway schema mismatch")
    if (
        payload.get("schema_version") != 1
        or payload.get("stage") != "menagerie_aggregate_gateway"
        or payload.get("tier") != tier
        or int(payload.get("seed", -1)) != seed
        or payload.get("backend") != BACKEND
        or payload.get("model") != str(model.resolve())
        or payload.get("model_merge_receipt_sha256") != model_receipt_sha256
        or payload.get("benchmark_runner_sha256") != sha256_file(MENAGERIE)
        or payload.get("benchmark_source_inventory_sha256")
        != benchmark_source_inventory(MENAGERIE.parent)["sha256"]
        or int(payload.get("benchmark_source_file_count", -1))
        != benchmark_source_inventory(MENAGERIE.parent)["file_count"]
        or not _finite_number(payload.get("aggregate"))
        or not _finite_number(payload.get("wall_seconds"))
        or float(payload["wall_seconds"]) < 0.0
    ):
        raise ValueError("aggregate gateway provenance mismatch")
    families = payload.get("per_family")
    if (
        not isinstance(families, dict)
        or set(families) != PUBLIC_FAMILY_KEYS
        or not all(
            isinstance(name, str) and name and _finite_number(value)
            for name, value in families.items()
        )
        or payload.get("within_budget") is not True
    ):
        raise ValueError("aggregate gateway score schema mismatch")
    return payload


def _validate_event(
    payload: Any,
    *,
    config_path: Path,
    authorization_sha256: str,
    confirmation_sha256: str,
    binding: dict,
) -> None:
    expected = {
        "schema_version": 2,
        "stage": "aggregate_only_menagerie_event",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "authorization": str(AUTHORIZATION),
        "authorization_sha256": authorization_sha256,
        "confirmation_sha256": confirmation_sha256,
        "tier": binding["tier"],
        "seed": int(binding["seed"]),
        "label": binding["label"],
        "backend": BACKEND,
        "model": binding["model"],
        "model_merge_receipt_sha256": binding["model_merge_receipt_sha256"],
        "model_weight_inventory_sha256": binding[
            "model_weight_inventory_sha256"
        ],
        "model_config_sha256": binding["model_config_sha256"],
        "model_inference_inventory_sha256": binding[
            "model_inference_inventory_sha256"
        ],
        "aggregate_gateway_sha256": sha256_file(GATEWAY),
        "benchmark_runner_sha256": sha256_file(MENAGERIE),
        "benchmark_source_inventory_sha256": benchmark_source_inventory(
            MENAGERIE.parent
        )["sha256"],
        "benchmark_source_file_count": benchmark_source_inventory(
            MENAGERIE.parent
        )["file_count"],
    }
    if not isinstance(payload, dict) or set(payload) != EVENT_KEYS:
        raise ValueError("benchmark event schema mismatch")
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("benchmark event provenance mismatch")
    if (
        not _finite_number(payload.get("aggregate"))
        or not _finite_number(payload.get("wall_seconds"))
        or float(payload["wall_seconds"]) < 0.0
        or payload.get("within_budget") is not True
    ):
        raise ValueError("benchmark event aggregate is invalid")
    families = payload.get("per_family")
    if (
        not isinstance(families, dict)
        or set(families) != PUBLIC_FAMILY_KEYS
        or not all(
            isinstance(name, str) and name and _finite_number(value)
            for name, value in families.items()
        )
    ):
        raise ValueError("benchmark event per-family aggregates are invalid")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("quick", "medium"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", choices=("primary", "soup", "visible"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config()
    authorization, keyed = _authorization(config, config_path)
    key = (args.tier, args.seed, args.label)
    if key not in keyed:
        raise SystemExit("event is outside the authorized benchmark inventory")
    binding = keyed[key]
    model = args.model.resolve()
    if model != Path(binding["model"]).resolve():
        raise SystemExit("requested model does not match benchmark authorization")
    try:
        observed_model = model_provenance(model)
    except ValueError as exc:
        raise SystemExit("requested model failed checkpoint provenance audit") from exc
    if any(binding.get(field) != observed_model[field] for field in observed_model):
        raise SystemExit("requested model provenance changed after authorization")
    authorization_sha256 = sha256_file(AUTHORIZATION)
    confirmation_sha256 = sha256_file(CONFIRMATION)

    if args.out.is_file():
        try:
            existing = json.loads(args.out.read_text(encoding="utf-8"))
            _validate_event(
                existing,
                config_path=config_path,
                authorization_sha256=authorization_sha256,
                confirmation_sha256=confirmation_sha256,
                binding=binding,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"stale benchmark receipt: {args.out}") from exc
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
    if args.out.exists():
        raise SystemExit(f"benchmark output path is not a file: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="deep_mopd_aggregate_summary_") as tmp:
        summary_path = Path(tmp) / "aggregate.json"
        completed = subprocess.run(
            [
                str(PY),
                str(GATEWAY),
                "--tier",
                args.tier,
                "--seed",
                str(args.seed),
                "--model",
                str(model),
                "--out",
                str(summary_path),
            ],
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"aggregate benchmark gateway failed with exit code "
                f"{completed.returncode}; private output suppressed"
            )
        try:
            gateway = _validate_gateway(
                json.loads(summary_path.read_text(encoding="utf-8")),
                tier=args.tier,
                seed=args.seed,
                model=model,
                model_receipt_sha256=binding["model_merge_receipt_sha256"],
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise SystemExit("aggregate benchmark gateway returned an invalid summary") from exc

    try:
        postrun_model = model_provenance(model)
    except ValueError as exc:
        raise SystemExit("benchmark model changed or became unreadable during the event") from exc
    if postrun_model != observed_model:
        raise SystemExit("benchmark model provenance changed during the event")

    receipt = {
        "schema_version": 2,
        "stage": "aggregate_only_menagerie_event",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "authorization": str(AUTHORIZATION),
        "authorization_sha256": authorization_sha256,
        "confirmation_sha256": confirmation_sha256,
        "tier": args.tier,
        "seed": args.seed,
        "label": args.label,
        "backend": BACKEND,
        "model": str(model),
        "model_merge_receipt_sha256": binding["model_merge_receipt_sha256"],
        "model_weight_inventory_sha256": binding[
            "model_weight_inventory_sha256"
        ],
        "model_config_sha256": binding["model_config_sha256"],
        "model_inference_inventory_sha256": binding[
            "model_inference_inventory_sha256"
        ],
        "aggregate_gateway_sha256": authorization["aggregate_gateway_sha256"],
        "benchmark_runner_sha256": authorization["benchmark_runner_sha256"],
        "benchmark_source_inventory_sha256": authorization[
            "benchmark_source_inventory_sha256"
        ],
        "benchmark_source_file_count": authorization[
            "benchmark_source_file_count"
        ],
        "aggregate": gateway["aggregate"],
        "per_family": gateway["per_family"],
        "within_budget": gateway["within_budget"],
        "wall_seconds": time.perf_counter() - started,
    }
    _validate_event(
        receipt,
        config_path=config_path,
        authorization_sha256=authorization_sha256,
        confirmation_sha256=confirmation_sha256,
        binding=binding,
    )
    write_json(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
