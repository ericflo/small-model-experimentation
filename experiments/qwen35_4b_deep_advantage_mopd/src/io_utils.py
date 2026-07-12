"""Deterministic config, hashing, and JSONL helpers for the curriculum."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]


def load_config(path: Path | None = None) -> tuple[dict[str, Any], Path]:
    path = (path or EXP / "configs" / "default.yaml").resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXP.name:
        raise ValueError(f"config experiment_id mismatch: {config.get('experiment_id')!r}")
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise ValueError("one-model rule violation")
    if config["model"]["revision"] != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a":
        raise ValueError("pinned model revision mismatch")
    trained = set(config["strata"]["trained_families"])
    transfer = set(config["strata"]["transfer_families"])
    if not trained or not transfer or trained & transfer:
        raise ValueError("trained and transfer families must be non-empty and disjoint")
    if float(config["decision"]["route_advantage_threshold"]) != 0.0:
        raise ValueError("successor forbids an arbitrary positive route-advantage bar")
    if float(config["decision"]["final_joint_delta_threshold"]) != 0.0:
        raise ValueError("successor forbids an arbitrary positive final-gain bar")
    route = config["route"]
    if int(route["selection_branches_per_policy"]) < 1:
        raise ValueError("route selection requires at least one branch")
    if int(route["audit_branches_per_policy"]) < 1:
        raise ValueError("route audit requires at least one disjoint branch")
    if not bool(route["ties_abstain"]):
        raise ValueError("route ties must abstain")
    if not bool(route["choose_only_if_strictly_above_student"]):
        raise ValueError("a routed teacher must beat the current student")
    if not bool(route["choose_only_if_strictly_above_alternate_teacher"]):
        raise ValueError("a routed teacher must beat the alternate teacher")
    if route.get("qualified_teacher") != "deep":
        raise ValueError("this successor qualifies only the frozen deep route")
    if int(route["minimum_deep_routed_states_per_block"]) < 1:
        raise ValueError("deep qualification requires routed support in every block")
    mopd = config["mopd"]
    if mopd.get("capability_teacher") != "deep":
        raise ValueError("this successor trains only on deep-advantage capability units")
    capability = float(mopd["capability_fraction"])
    anchor = float(mopd["anchor_fraction"])
    if abs(capability + anchor - 1.0) > 1e-12:
        raise ValueError("MOPD capability and anchor fractions must sum to one")
    if not 0.0 < capability < 1.0 or not 0.0 < anchor < 1.0:
        raise ValueError("MOPD capability and anchor fractions must both be interior")
    capability_units = int(mopd["capability_units_per_round"])
    anchor_units = int(mopd["anchor_units_per_round"])
    if capability_units <= 0 or anchor_units <= 0:
        raise ValueError("MOPD round quotas must be positive")
    total_units = capability_units + anchor_units
    if abs(capability_units / total_units - capability) > 1e-12:
        raise ValueError("MOPD unit quotas must realize the frozen capability fraction")
    if abs(anchor_units / total_units - anchor) > 1e-12:
        raise ValueError("MOPD unit quotas must realize the frozen anchor fraction")
    if int(mopd["rounds"]) != len(config["seeds"]["rollout_rounds"]):
        raise ValueError("every MOPD round requires exactly one frozen rollout seed")
    if int(config["controls"]["matched_total_updates"]) != (
        int(mopd["rounds"]) * int(mopd["updates_per_round"])
    ):
        raise ValueError("controls must match the primary arm's total update count")
    required_match_order = ("exact_cell", "family_kind", "kind_level", "kind")
    if tuple(config["controls"]["non_advantage_route_match_order"]) != required_match_order:
        raise ValueError("non-advantage controls must use the frozen no-cross-kind match order")
    student_sha = str(config["model"]["student_model_sha256"])
    if len(student_sha) != 64 or any(ch not in "0123456789abcdef" for ch in student_sha):
        raise ValueError("student checkpoint requires an exact lowercase sha256")
    benchmark = config["benchmark"]
    if int(benchmark["first_seed"]) != int(config["seeds"]["benchmark_first"]):
        raise ValueError("benchmark seed aliases disagree")
    if int(benchmark["quick_events"]) != 3 or int(benchmark["medium_events"]) < 8:
        raise ValueError("benchmark event counts violate the frozen evidence floor")
    for key in (
        "require_every_event_reported",
        "require_paired_positive_delta_each_tier",
        "require_each_event_positive",
    ):
        if not bool(benchmark[key]):
            raise ValueError(f"benchmark safeguard disabled: {key}")
    return config, path


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def training_seed(config: dict[str, Any], index: int = 0) -> int:
    """Return one frozen integration seed."""
    seeds = config["seeds"]["integration_training"]
    if isinstance(seeds, list):
        if not seeds:
            raise ValueError("seeds.training must not be empty")
        return int(seeds[index % len(seeds)])
    return int(seeds)


def all_families(config: dict[str, Any]) -> list[str]:
    return list(config["strata"]["trained_families"]) + list(
        config["strata"]["transfer_families"]
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_specs(
    families: Iterable[str],
    per_level: dict[int | str, int],
    seed_base: int,
) -> list[tuple[str, int, int]]:
    """Create disjoint deterministic episode specs from a seed namespace."""
    normalized = {int(level): int(count) for level, count in per_level.items()}
    specs: list[tuple[str, int, int]] = []
    for family_index, family in enumerate(families):
        for level, count in sorted(normalized.items()):
            for item_index in range(count):
                seed = int(seed_base) + family_index * 100_000 + level * 1_000 + item_index
                specs.append((str(family), int(level), seed))
    if len(specs) != len(set(specs)):
        raise ValueError("duplicate episode specs")
    return specs


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_receipt(config: dict[str, Any], specs: list[tuple[str, int, int]]) -> dict[str, Any]:
    payload = {
        "train_families": list(config["strata"]["trained_families"]),
        "transfer_families": list(config["strata"]["transfer_families"]),
        "specs": specs,
    }
    return {"payload": payload, "sha256": canonical_hash(payload)}
