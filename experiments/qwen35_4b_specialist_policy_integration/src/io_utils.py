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
    return config, path


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def training_seed(config: dict[str, Any], index: int = 0) -> int:
    """Return one frozen training seed while accepting the preregistered list."""
    seeds = config["seeds"]["training"]
    if isinstance(seeds, list):
        if not seeds:
            raise ValueError("seeds.training must not be empty")
        return int(seeds[index % len(seeds)])
    return int(seeds)


def domain_families(config: dict[str, Any], domain: str) -> list[str]:
    """Resolve one specialist domain, or the frozen joint training mixture."""
    if domain == "joint":
        return list(config["split"]["train_families"])
    domains = config.get("domains", {})
    if domain not in domains:
        raise ValueError(
            f"unknown domain {domain!r}; expected one of {sorted(domains)} or 'joint'"
        )
    families = [str(value) for value in domains[domain]]
    if not families:
        raise ValueError(f"domain {domain!r} has no families")
    outside = set(families) - set(config["split"]["train_families"])
    if outside:
        raise ValueError(f"domain {domain!r} contains non-training families: {sorted(outside)}")
    return families


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
        "train_families": list(config["split"]["train_families"]),
        "transfer_families": list(config["split"]["transfer_families"]),
        "specs": specs,
    }
    return {"payload": payload, "sha256": canonical_hash(payload)}
