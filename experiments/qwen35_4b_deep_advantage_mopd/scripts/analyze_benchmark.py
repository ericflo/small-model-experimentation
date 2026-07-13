#!/usr/bin/env python3
"""Analyze the frozen, authorization-bound aggregate benchmark events."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
AUTHORIZATION = EXP / "analysis" / "benchmark_authorization.json"
CONFIRMATION = EXP / "analysis" / "confirmation.json"
PREREGISTRATION = EXP / "runs" / "preregistration_receipt.json"
GATEWAY = REPO / "scripts" / "run_benchmark_aggregate.py"
MENAGERIE = REPO / "benchmarks" / "menagerie" / "run.py"
BENCH = EXP / "scripts" / "bench.py"
AUTHORIZER = EXP / "scripts" / "authorize_benchmark.py"
CONFIRMATION_ANALYZER = EXP / "scripts" / "analyze_confirmation.py"
CONFIRMATION_EVALUATOR = EXP / "scripts" / "eval_policy.py"
CONTROL_REMATCH = EXP / "src" / "control_rematch.py"
BACKEND = "qwen_vllm"
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

from bench import (  # noqa: E402
    BENCHMARK_EVENT_COUNT,
    EVENT_BINDING_KEYS,
    EVENT_KEYS,
    PUBLIC_FAMILY_KEYS,
    benchmark_source_inventory,
    model_provenance,
    validate_authorization_artifacts,
)
from io_utils import (  # noqa: E402
    confirmation_evaluator_source_inventory,
    load_config,
    sha256_file,
    write_json,
)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON receipt: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"receipt is not an object: {path}")
    return value


def _tier_seeds(config: dict) -> dict[str, list[int]]:
    first = int(config["benchmark"]["first_seed"])
    quick_n = int(config["benchmark"]["quick_events"])
    medium_n = int(config["benchmark"]["medium_events"])
    return {
        "quick": list(range(first, first + quick_n)),
        "medium": list(range(first + quick_n, first + quick_n + medium_n)),
    }


def _authorization(
    path: Path,
    confirmation_path: Path,
    config_path: Path,
    expected: set[tuple[str, int, str]],
) -> tuple[dict, dict[tuple[str, int, str], dict], str, str]:
    payload = _load(path)
    confirmation = _load(confirmation_path)
    source_inventory = benchmark_source_inventory(MENAGERIE.parent)
    evaluator_source = confirmation_evaluator_source_inventory()
    required = {
        "schema_version": 2,
        "stage": "benchmark_aggregate_authorization",
        "config_sha256": sha256_file(config_path),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "confirmation_sha256": sha256_file(confirmation_path),
        "aggregate_gateway_sha256": sha256_file(GATEWAY),
        "benchmark_runner_sha256": sha256_file(MENAGERIE),
        "benchmark_source_inventory_sha256": source_inventory["sha256"],
        "benchmark_source_file_count": source_inventory["file_count"],
        "bench_sha256": sha256_file(BENCH),
        "analyzer_sha256": sha256_file(Path(__file__)),
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
        raise ValueError("benchmark authorization receipt is stale")
    if not payload.get("gate", {}).get("passed"):
        raise ValueError("benchmark authorization gate did not pass")
    validate_authorization_artifacts(payload)
    if (
        confirmation.get("stage")
        != "two_block_same_prefix_advantage_confirmation"
        or confirmation.get("config_sha256") != sha256_file(config_path)
        or not confirmation.get("gate", {}).get("passed")
        or confirmation.get("downstream_authorization") != "benchmark_cli"
        or confirmation.get("manifest_sha256")
        != payload.get("confirmation_manifest_sha256")
    ):
        raise ValueError("procedural confirmation authorization is stale")
    try:
        manifest = Path(confirmation["manifest"]).resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("procedural confirmation manifest path is invalid") from exc
    if not manifest.is_file() or sha256_file(manifest) != confirmation.get(
        "manifest_sha256"
    ):
        raise ValueError("procedural confirmation manifest hash mismatch")

    rows = payload.get("events")
    if not isinstance(rows, list):
        raise ValueError("benchmark authorization lacks events")
    keyed: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != EVENT_BINDING_KEYS:
            raise ValueError("benchmark authorization event schema mismatch")
        key = (str(row["tier"]), int(row["seed"]), str(row["label"]))
        if key in keyed:
            raise ValueError("duplicate benchmark authorization event")
        keyed[key] = row
    if set(keyed) != expected:
        raise ValueError("benchmark authorization event inventory mismatch")

    observed_models: dict[Path, dict[str, str]] = {}
    for row in keyed.values():
        model = Path(row["model"]).resolve()
        if model not in observed_models:
            observed_models[model] = model_provenance(model)
        provenance = observed_models[model]
        if any(row.get(field) != provenance[field] for field in provenance):
            raise ValueError("authorized benchmark model provenance changed")
    return payload, keyed, sha256_file(path), sha256_file(confirmation_path)


def _event_protocol(
    row: dict,
    *,
    binding: dict,
    config_path: Path,
    authorization_path: Path,
    authorization_sha256: str,
    confirmation_sha256: str,
    authorization: dict,
) -> bool:
    expected = {
        "schema_version": 2,
        "stage": "aggregate_only_menagerie_event",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "authorization": str(authorization_path.resolve()),
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
        "aggregate_gateway_sha256": authorization[
            "aggregate_gateway_sha256"
        ],
        "benchmark_runner_sha256": authorization["benchmark_runner_sha256"],
        "benchmark_source_inventory_sha256": authorization[
            "benchmark_source_inventory_sha256"
        ],
        "benchmark_source_file_count": authorization[
            "benchmark_source_file_count"
        ],
    }
    families = row.get("per_family")
    return (
        set(row) == EVENT_KEYS
        and all(row.get(key) == value for key, value in expected.items())
        and _finite_number(row.get("aggregate"))
        and _finite_number(row.get("wall_seconds"))
        and float(row["wall_seconds"]) >= 0.0
        and row.get("within_budget") is True
        and isinstance(families, dict)
        and set(families) == PUBLIC_FAMILY_KEYS
        and all(
            isinstance(name, str) and name and _finite_number(value)
            for name, value in families.items()
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", action="append", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION)
    parser.add_argument("--confirmation", type=Path, default=CONFIRMATION)
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "benchmark.json"
    )
    args = parser.parse_args()
    config, config_path = load_config()
    tier_seeds = _tier_seeds(config)
    expected = {
        (tier, seed, label)
        for tier, seeds in tier_seeds.items()
        for seed in seeds
        for label in ("primary", "soup", "visible")
    }
    if len(expected) != BENCHMARK_EVENT_COUNT:
        raise ValueError("frozen benchmark inventory is not exactly 33 events")
    authorization, bindings, authorization_sha256, confirmation_sha256 = (
        _authorization(
            args.authorization.resolve(),
            args.confirmation.resolve(),
            config_path,
            expected,
        )
    )
    rows = [_load(path) for path in args.event]
    keyed = {(row["tier"], int(row["seed"]), row["label"]): row for row in rows}
    protocol = {
        "authorization_current": True,
        "exact_event_inventory": set(keyed) == expected and len(rows) == len(expected),
        "aggregate_only_schema_and_provenance": (
            set(keyed) == expected
            and all(
                _event_protocol(
                    row,
                    binding=bindings[key],
                    config_path=config_path,
                    authorization_path=args.authorization.resolve(),
                    authorization_sha256=authorization_sha256,
                    confirmation_sha256=confirmation_sha256,
                    authorization=authorization,
                )
                for key, row in keyed.items()
            )
        ),
    }
    comparisons = {}
    for tier, seeds in tier_seeds.items():
        comparisons[tier] = {}
        for comparator in ("soup", "visible"):
            deltas = [
                float(keyed[(tier, seed, "primary")]["aggregate"])
                - float(keyed[(tier, seed, comparator)]["aggregate"])
                for seed in seeds
            ]
            comparisons[tier][comparator] = {
                "n": len(deltas),
                "deltas": deltas,
                "mean_delta": sum(deltas) / len(deltas),
                "all_events_positive": all(value > 0.0 for value in deltas),
            }
    checks = {
        "protocol": all(protocol.values()),
        "every_event_within_budget": all(
            row.get("within_budget") is True for row in rows
        ),
        "primary_beats_soup_and_visible_every_event": all(
            row["mean_delta"] > 0.0 and row["all_events_positive"]
            for tier in comparisons.values()
            for row in tier.values()
        ),
    }
    result = {
        "schema_version": 2,
        "stage": "aggregate_only_menagerie_confirmation",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "authorization": str(args.authorization.resolve()),
        "authorization_sha256": authorization_sha256,
        "confirmation_sha256": confirmation_sha256,
        "expected_seeds": tier_seeds,
        "event_artifacts": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.event
        ],
        "protocol_checks": protocol,
        "comparisons": comparisons,
        "checks": checks,
        "gate": {"passed": all(checks.values())},
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
