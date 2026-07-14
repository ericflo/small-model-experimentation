#!/usr/bin/env python3
"""Materialize exact, nested replay-anchored doses from frozen parent corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable


EXP = Path(__file__).resolve().parents[1]
DESIGNED = EXP / "data" / "sft_universal_fast.jsonl"
REPLAY = EXP / "data" / "sft_blend.jsonl"
DESIGNED_SHA256 = "4a0833756b5497fcbccf278476ece8c98bcbfce80e900ecc2489150e496b27c4"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
SEED = 77102


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_rows(path: Path, expected_sha256: str) -> list[tuple[str, dict]]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"frozen source changed: {path}")
    rows: list[tuple[str, dict]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        rows.append((line, row))
    return rows


def allocation(group_sizes: dict[tuple[str, ...], int], total: int) -> dict[tuple[str, ...], int]:
    population = sum(group_sizes.values())
    if not 0 <= total <= population:
        raise ValueError("invalid stratified target")
    quotas = {
        key: size * total // population for key, size in group_sizes.items()
    }
    remainder_order = sorted(
        group_sizes,
        key=lambda key: (-(group_sizes[key] * total % population), key),
    )
    for key in remainder_order[: total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def deterministic_rank(namespace: str, index: int, line: str) -> bytes:
    return hashlib.sha256(f"{SEED}:{namespace}:{index}:".encode() + line.encode()).digest()


def select_stratified(
    rows: list[tuple[str, dict]],
    total: int,
    group_fn: Callable[[dict], tuple[str, ...]],
    namespace: str,
) -> tuple[list[tuple[str, dict]], list[int]]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, (_, row) in enumerate(rows):
        groups[group_fn(row)].append(index)
    quotas = allocation({key: len(indices) for key, indices in groups.items()}, total)
    selected: list[int] = []
    for key, indices in sorted(groups.items()):
        ranked = sorted(
            indices,
            key=lambda index: deterministic_rank(namespace, index, rows[index][0]),
        )
        selected.extend(ranked[: quotas[key]])
    selected.sort()
    return [rows[index] for index in selected], selected


def encoded(rows: list[tuple[str, dict]]) -> bytes:
    return ("\n".join(line for line, _ in rows) + "\n").encode("utf-8")


def summary(value: bytes) -> dict:
    parsed = [json.loads(line) for line in value.decode("utf-8").splitlines() if line]
    return {
        "rows": len(parsed),
        "sha256": sha256_bytes(value),
        "kinds": dict(sorted(Counter(row.get("kind", "missing") for row in parsed).items())),
        "families": dict(sorted(Counter(row.get("family", "missing") for row in parsed).items())),
    }


def build_outputs() -> tuple[dict[str, bytes], dict]:
    designed = load_rows(DESIGNED, DESIGNED_SHA256)
    replay = load_rows(REPLAY, REPLAY_SHA256)
    if len(designed) != 800 or len(replay) != 2240:
        raise ValueError("frozen parent row count changed")

    designed_half, designed_indices = select_stratified(
        designed,
        400,
        lambda row: (str(row.get("kind")),),
        "designed-half",
    )
    replay_shared, shared_indices = select_stratified(
        replay,
        1120,
        lambda row: (str(row.get("family")), str(row.get("kind"))),
        "replay-shared",
    )
    shared_set = set(shared_indices)
    replay_remainder = [row for index, row in enumerate(replay) if index not in shared_set]
    replay_extra, extra_relative_indices = select_stratified(
        replay_remainder,
        400,
        lambda row: (str(row.get("family")), str(row.get("kind"))),
        "replay-extra",
    )

    outputs = {
        "warm_union.jsonl": encoded(designed_half + replay_shared),
        "replay_refresh.jsonl": encoded(replay_shared + replay_extra),
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "seed": SEED,
        "sources": {
            "designed": {"path": str(DESIGNED.relative_to(EXP)), "rows": 800, "sha256": DESIGNED_SHA256},
            "replay": {"path": str(REPLAY.relative_to(EXP)), "rows": 2240, "sha256": REPLAY_SHA256},
        },
        "selection": {
            "designed_rows": len(designed_indices),
            "shared_replay_rows": len(shared_indices),
            "extra_control_replay_rows": len(extra_relative_indices),
            "candidate_total_rows": 1520,
            "control_total_rows": 1520,
            "nested_replay_control": True,
        },
        "outputs": {name: summary(value) for name, value in sorted(outputs.items())},
    }
    return outputs, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify existing bytes without writing")
    args = parser.parse_args()
    outputs, manifest = build_outputs()
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    targets = {EXP / "data" / name: value for name, value in outputs.items()}
    targets[EXP / "data" / "dose_manifest.json"] = manifest_bytes
    if args.check:
        for path, expected in targets.items():
            if not path.is_file() or path.read_bytes() != expected:
                raise SystemExit(f"derived dose is absent or changed: {path}")
        print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    conflicts = [path for path in targets if path.exists()]
    if conflicts:
        parser.error(f"refusing to overwrite existing dose: {conflicts[0]}")
    for path, value in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
