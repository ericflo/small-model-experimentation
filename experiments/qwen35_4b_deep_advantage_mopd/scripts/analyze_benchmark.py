#!/usr/bin/env python3
"""Analyze the frozen, authorization-bound aggregate benchmark events."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
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
CONTROL_RECEIPTS = EXP / "src" / "control_receipts.py"
BACKEND = "qwen_vllm"
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

from bench import (  # noqa: E402
    BENCHMARK_EVENT_COUNT,
    EVENT_BINDING_KEYS,
    EVENT_KEYS,
    PUBLIC_FAMILY_KEYS,
    _canonical_benchmark_event_path,
    _lexical_absolute,
    _publication_start_state,
    _publish_json_no_clobber,
    _reject_symlink_components,
    _validated_benchmark_event_path,
    benchmark_source_inventory,
    model_provenance,
    validate_authorization_artifacts,
)
from confirmation_artifacts import (  # noqa: E402
    configured_confirmation_raw_root,
    validate_confirmation_campaign_tree,
)
from io_utils import (  # noqa: E402
    confirmation_evaluator_source_inventory,
    load_config,
    sha256_file,
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


def _load_event_snapshot(path: Path) -> tuple[dict, str]:
    """Parse and hash one event from the exact same immutable byte snapshot."""

    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON event receipt: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"event receipt is not an object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _tier_seeds(config: dict) -> dict[str, list[int]]:
    first = int(config["benchmark"]["first_seed"])
    quick_n = int(config["benchmark"]["quick_events"])
    medium_n = int(config["benchmark"]["medium_events"])
    return {
        "quick": list(range(first, first + quick_n)),
        "medium": list(range(first + quick_n, first + quick_n + medium_n)),
    }


def _event_order(
    tier_seeds: dict[str, list[int]],
) -> list[tuple[str, int, str]]:
    return [
        (tier, seed, label)
        for tier, seeds in tier_seeds.items()
        for seed in seeds
        for label in ("primary", "soup", "visible")
    ]


def _validated_event_path_inventory(
    paths: list[Path], tier_seeds: dict[str, list[int]]
) -> dict[tuple[str, int, str], Path]:
    """Bind every authorized event key to its one canonical, symlink-free path."""

    order = _event_order(tier_seeds)
    expected = {
        _canonical_benchmark_event_path(tier, seed, label): (tier, seed, label)
        for tier, seed, label in order
    }
    if len(expected) != BENCHMARK_EVENT_COUNT:
        raise ValueError("frozen benchmark path inventory is not exactly 33 events")

    keyed: dict[tuple[str, int, str], Path] = {}
    for supplied in paths:
        lexical = _lexical_absolute(supplied)
        _reject_symlink_components(lexical, label="benchmark event input")
        key = expected.get(lexical)
        if key is None:
            raise ValueError(
                f"benchmark event input is not an exact canonical path: {lexical}"
            )
        if key in keyed:
            raise ValueError(f"duplicate benchmark event path: {lexical}")
        tier, seed, label = key
        keyed[key] = _validated_benchmark_event_path(
            lexical, tier=tier, seed=seed, label=label
        )
    if len(paths) != len(expected) or set(keyed) != set(expected.values()):
        raise ValueError("benchmark event path inventory mismatch")
    return keyed


def _safe_directory_entries(path: Path, *, label: str) -> dict[str, tuple[Path, int]]:
    """Inventory one directory without following any entry or ancestor symlink."""

    path = _lexical_absolute(path)
    _reject_symlink_components(path, label=label)
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} is not a directory: {path}")
    try:
        entries = list(path.iterdir())
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    by_name: dict[str, tuple[Path, int]] = {}
    for entry in entries:
        try:
            entry_metadata = entry.lstat()
        except OSError as exc:
            raise ValueError(f"{label} entry is unreadable: {entry}") from exc
        if stat.S_ISLNK(entry_metadata.st_mode):
            raise ValueError(f"{label} contains a symlink: {entry}")
        if entry.name in by_name:
            raise ValueError(f"{label} contains duplicate entries: {entry.name}")
        by_name[entry.name] = (entry, entry_metadata.st_mode)
    return by_name


def _validate_benchmark_campaign_tree(
    event_paths: dict[tuple[str, int, str], Path],
    tier_seeds: dict[str, list[int]],
    event_sha256: dict[Path, str],
) -> None:
    """Require exactly the frozen benchmark directory scaffold and 33 event leaves."""

    root = _lexical_absolute(EXP / "runs" / "benchmark")
    root_entries = _safe_directory_entries(root, label="benchmark campaign root")
    expected_tiers = set(tier_seeds)
    if set(root_entries) != expected_tiers:
        raise ValueError("benchmark campaign root contains unregistered entries")

    observed: set[Path] = set()
    for tier, seeds in tier_seeds.items():
        tier_path, tier_mode = root_entries[tier]
        if not stat.S_ISDIR(tier_mode):
            raise ValueError(f"benchmark tier is not a directory: {tier_path}")
        tier_entries = _safe_directory_entries(
            tier_path, label=f"benchmark {tier} tier"
        )
        expected_seed_names = {f"seed_{seed}" for seed in seeds}
        if set(tier_entries) != expected_seed_names:
            raise ValueError(f"benchmark {tier} tier contains unregistered entries")

        for seed in seeds:
            seed_name = f"seed_{seed}"
            seed_path, seed_mode = tier_entries[seed_name]
            if not stat.S_ISDIR(seed_mode):
                raise ValueError(
                    f"benchmark seed is not a directory: {seed_path}"
                )
            seed_entries = _safe_directory_entries(
                seed_path, label=f"benchmark {tier}/{seed_name} directory"
            )
            expected_names = {
                "primary.json",
                "soup.json",
                "visible.json",
            }
            if set(seed_entries) != expected_names:
                raise ValueError(
                    f"benchmark {tier}/{seed_name} contains unregistered entries"
                )
            for label in ("primary", "soup", "visible"):
                event_path, event_mode = seed_entries[f"{label}.json"]
                if not stat.S_ISREG(event_mode):
                    raise ValueError(
                        f"benchmark event leaf is not a regular file: {event_path}"
                    )
                expected_path = event_paths[(tier, seed, label)]
                if event_path != expected_path:
                    raise ValueError(
                        f"benchmark event leaf is not canonical: {event_path}"
                    )
                if sha256_file(event_path) != event_sha256.get(expected_path):
                    raise ValueError(
                        f"benchmark event changed during analysis: {event_path}"
                    )
                observed.add(event_path)

    if observed != set(event_paths.values()) or set(event_sha256) != observed:
        raise ValueError("benchmark campaign leaf inventory mismatch")


def _validate_terminal_confirmation_campaign(
    confirmation_path: Path,
    config: dict,
    *,
    expected_confirmation_sha256: str,
) -> None:
    """Revalidate the exhaustive admitted confirmation tree at publication time."""

    if sha256_file(confirmation_path) != expected_confirmation_sha256:
        raise ValueError("procedural confirmation changed during benchmark analysis")
    confirmation = _load(confirmation_path)
    admission = confirmation.get("confirmation_admission")
    if not isinstance(admission, dict):
        raise ValueError("procedural confirmation admission is missing")
    admission_value = admission.get("path")
    admission_sha256 = admission.get("sha256")
    if (
        not isinstance(admission_value, str)
        or not admission_value
        or not isinstance(admission_sha256, str)
        or len(admission_sha256) != 64
    ):
        raise ValueError("procedural confirmation admission binding is malformed")
    admission_path = Path(admission_value)
    if not admission_path.is_absolute():
        raise ValueError("procedural confirmation admission path is not absolute")
    try:
        admission_metadata = admission_path.lstat()
    except FileNotFoundError as exc:
        raise ValueError("procedural confirmation admission is missing") from exc
    if (
        not stat.S_ISREG(admission_metadata.st_mode)
        or sha256_file(admission_path) != admission_sha256
    ):
        raise ValueError("procedural confirmation admission binding is stale")
    states = validate_confirmation_campaign_tree(
        admission_path,
        raw_root=configured_confirmation_raw_root(config),
        terminal=True,
        require_manifest=True,
    )
    if (
        not isinstance(states, dict)
        or not states
        or set(states.values()) != {"COMMITTED"}
    ):
        raise ValueError(
            "procedural confirmation campaign is not terminal and complete"
        )
    if (
        sha256_file(admission_path) != admission_sha256
        or sha256_file(confirmation_path) != expected_confirmation_sha256
    ):
        raise ValueError("procedural confirmation campaign changed during validation")


def _analysis_publication_start(path: Path) -> tuple[Path, bool]:
    return _publication_start_state(
        path,
        expected=EXP / "analysis" / "benchmark.json",
        label="benchmark analysis output",
    )


def _publish_analysis_no_clobber(
    path: Path, result: dict, *, existed_at_start: bool
) -> bool:
    return _publish_json_no_clobber(
        path,
        result,
        expected=EXP / "analysis" / "benchmark.json",
        label="benchmark analysis output",
        existed_at_start=existed_at_start,
    )


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
        "control_receipts_sha256": sha256_file(CONTROL_RECEIPTS),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", action="append", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION)
    parser.add_argument("--confirmation", type=Path, default=CONFIRMATION)
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "benchmark.json"
    )
    args = parser.parse_args(argv)
    out, output_existed_at_start = _analysis_publication_start(args.out)
    config, config_path = load_config()
    tier_seeds = _tier_seeds(config)
    order = _event_order(tier_seeds)
    expected = set(order)
    if len(expected) != BENCHMARK_EVENT_COUNT:
        raise ValueError("frozen benchmark inventory is not exactly 33 events")
    event_paths = _validated_event_path_inventory(args.event, tier_seeds)
    authorization, bindings, authorization_sha256, confirmation_sha256 = (
        _authorization(
            args.authorization.resolve(),
            args.confirmation.resolve(),
            config_path,
            expected,
        )
    )
    event_snapshots = {
        key: _load_event_snapshot(event_paths[key]) for key in order
    }
    keyed = {key: event_snapshots[key][0] for key in order}
    rows = [keyed[key] for key in order]
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
    event_artifacts = [
        {"path": str(event_paths[key]), "sha256": event_snapshots[key][1]}
        for key in order
    ]
    result = {
        "schema_version": 2,
        "stage": "aggregate_only_menagerie_confirmation",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "authorization": str(args.authorization.resolve()),
        "authorization_sha256": authorization_sha256,
        "confirmation_sha256": confirmation_sha256,
        "expected_seeds": tier_seeds,
        "event_artifacts": event_artifacts,
        "protocol_checks": protocol,
        "comparisons": comparisons,
        "checks": checks,
        "gate": {"passed": all(checks.values())},
    }
    _validate_terminal_confirmation_campaign(
        args.confirmation.resolve(),
        config,
        expected_confirmation_sha256=confirmation_sha256,
    )
    _validate_benchmark_campaign_tree(
        event_paths,
        tier_seeds,
        {Path(row["path"]): row["sha256"] for row in event_artifacts},
    )
    _publish_analysis_no_clobber(
        out,
        result,
        existed_at_start=output_existed_at_start,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
