#!/usr/bin/env python3
"""Build nested 0/160/240-row designed doses with exact token matching."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable


EXP = Path(__file__).resolve().parents[1]
DESIGNED = EXP / "data" / "sft_universal_fast.jsonl"
REPLAY = EXP / "data" / "sft_blend.jsonl"
SOURCE_TOKENS = EXP / "data" / "source_token_lengths.json"
DESIGNED_SHA256 = "4a0833756b5497fcbccf278476ece8c98bcbfce80e900ecc2489150e496b27c4"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SEED = 77109
MATCH_ABS_LIMIT = 0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_rows(path: Path, expected_sha256: str, expected_rows: int) -> list[tuple[str, dict]]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"frozen source changed: {path}")
    rows = [(line, json.loads(line)) for line in raw.decode("utf-8").splitlines() if line]
    if len(rows) != expected_rows or not all(isinstance(row, dict) for _, row in rows):
        raise ValueError(f"unexpected source rows: {path}")
    return rows


def load_lengths() -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    payload = json.loads(SOURCE_TOKENS.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_id") != EXP.name
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("max_length") != 4096
    ):
        raise ValueError("source token receipt identity changed")
    result = []
    for name, expected_path, expected_sha, expected_rows in (
        ("designed", DESIGNED, DESIGNED_SHA256, 800),
        ("replay", REPLAY, REPLAY_SHA256, 2240),
    ):
        source = payload.get("sources", {}).get(name, {})
        lengths = source.get("lengths", [])
        if (
            source.get("path") != expected_path.relative_to(EXP).as_posix()
            or source.get("sha256") != expected_sha
            or source.get("rows") != expected_rows
            or len(lengths) != expected_rows
            or not all(set(item) == {"forward", "prompt", "think_target", "answer_target"} for item in lengths)
        ):
            raise ValueError(f"source token receipt changed for {name}")
        result.append(lengths)
    return result[0], result[1]


def allocation(group_sizes: dict[tuple[str, ...], int], total: int) -> dict[tuple[str, ...], int]:
    population = sum(group_sizes.values())
    if not 0 <= total <= population:
        raise ValueError("invalid stratified target")
    quotas = {key: size * total // population for key, size in group_sizes.items()}
    order = sorted(
        group_sizes,
        key=lambda key: (-(group_sizes[key] * total % population), key),
    )
    for key in order[: total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def deterministic_rank(namespace: str, index: int, line: str) -> bytes:
    return hashlib.sha256(f"{SEED}:{namespace}:{index}:".encode() + line.encode()).digest()


def select_stratified(
    rows: list[tuple[str, dict]],
    indices: list[int],
    total: int,
    group_fn: Callable[[dict], tuple[str, ...]],
    namespace: str,
) -> list[int]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in indices:
        groups[group_fn(rows[index][1])].append(index)
    quotas = allocation({key: len(value) for key, value in groups.items()}, total)
    selected = []
    for key, members in sorted(groups.items()):
        ranked = sorted(
            members,
            key=lambda index: deterministic_rank(namespace, index, rows[index][0]),
        )
        selected.extend(ranked[: quotas[key]])
    return sorted(selected)


def token_sum(indices: list[int], lengths: list[dict[str, int]], field: str = "forward") -> int:
    return sum(lengths[index][field] for index in indices)


def select_token_match(
    rows: list[tuple[str, dict]],
    lengths: list[dict[str, int]],
    available: list[int],
    count: int,
    target: int,
    namespace: str,
) -> list[int]:
    target_average = target / count
    ranked = sorted(
        available,
        key=lambda index: (
            abs(lengths[index]["forward"] - target_average),
            deterministic_rank(namespace, index, rows[index][0]),
        ),
    )
    selected = set(ranked[:count])
    current = token_sum(sorted(selected), lengths)
    while current != target:
        current_error = abs(current - target)
        best: tuple[int, bytes, int, int, int] | None = None
        outside = [index for index in available if index not in selected]
        for old in sorted(selected):
            old_length = lengths[old]["forward"]
            for new in outside:
                proposed = current - old_length + lengths[new]["forward"]
                error = abs(proposed - target)
                if error >= current_error:
                    continue
                tie = deterministic_rank(f"{namespace}-swap-{old}", new, rows[new][0])
                candidate = (error, tie, old, new, proposed)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        _, _, old, new, current = best
        selected.remove(old)
        selected.add(new)
    if current != target:
        outside = sorted(index for index in available if index not in selected)
        outside_by_length: dict[int, list[int]] = defaultdict(list)
        for index in outside:
            outside_by_length[lengths[index]["forward"]].append(index)
        correction: tuple[int, int, int, int] | None = None
        for old_a, old_b in combinations(sorted(selected), 2):
            needed = (
                target
                - current
                + lengths[old_a]["forward"]
                + lengths[old_b]["forward"]
            )
            for new_a in outside:
                candidates = outside_by_length.get(
                    needed - lengths[new_a]["forward"], []
                )
                new_b = next((value for value in candidates if value != new_a), None)
                if new_b is not None:
                    correction = (old_a, old_b, new_a, new_b)
                    break
            if correction is not None:
                break
        if correction is not None:
            old_a, old_b, new_a, new_b = correction
            selected.difference_update((old_a, old_b))
            selected.update((new_a, new_b))
            current = token_sum(sorted(selected), lengths)
    if current != target:
        outside = sorted(index for index in available if index not in selected)
        pair_by_sum: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for new_a, new_b in combinations(outside, 2):
            total = lengths[new_a]["forward"] + lengths[new_b]["forward"]
            if len(pair_by_sum[total]) < 8:
                pair_by_sum[total].append((new_a, new_b))
        needed_to_old: dict[int, tuple[int, int, int]] = {}
        for old_a, old_b, old_c in combinations(sorted(selected), 3):
            needed = (
                target
                - current
                + lengths[old_a]["forward"]
                + lengths[old_b]["forward"]
                + lengths[old_c]["forward"]
            )
            needed_to_old.setdefault(needed, (old_a, old_b, old_c))
        correction3: tuple[int, int, int, int, int, int] | None = None
        for needed, old_values in sorted(needed_to_old.items()):
            for new_a in outside:
                pair_total = needed - lengths[new_a]["forward"]
                pair = next(
                    (
                        values for values in pair_by_sum.get(pair_total, [])
                        if new_a not in values
                    ),
                    None,
                )
                if pair is not None:
                    correction3 = (*old_values, new_a, *pair)
                    break
            if correction3 is not None:
                break
        if correction3 is not None:
            old_a, old_b, old_c, new_a, new_b, new_c = correction3
            selected.difference_update((old_a, old_b, old_c))
            selected.update((new_a, new_b, new_c))
            current = token_sum(sorted(selected), lengths)
    result = sorted(selected)
    error = abs(token_sum(result, lengths) - target)
    if error > MATCH_ABS_LIMIT:
        raise ValueError(f"{namespace} token-match error {error} exceeds {MATCH_ABS_LIMIT}")
    return result


def encode_slots(slots: list[tuple[str, dict]]) -> bytes:
    order = sorted(
        range(len(slots)),
        key=lambda index: deterministic_rank("shared-slot-order", index, str(index)),
    )
    return ("\n".join(slots[index][0] for index in order) + "\n").encode("utf-8")


def summary(value: bytes) -> dict:
    parsed = [json.loads(line) for line in value.decode("utf-8").splitlines() if line]
    return {
        "rows": len(parsed),
        "sha256": sha256_bytes(value),
        "kinds": dict(sorted(Counter(row.get("kind", "missing") for row in parsed).items())),
        "families": dict(sorted(Counter(row.get("family", "missing") for row in parsed).items())),
    }


def build_outputs() -> tuple[dict[str, bytes], dict]:
    designed = load_rows(DESIGNED, DESIGNED_SHA256, 800)
    replay = load_rows(REPLAY, REPLAY_SHA256, 2240)
    designed_lengths, replay_lengths = load_lengths()

    designed240 = select_stratified(
        designed, list(range(len(designed))), 240,
        lambda row: (str(row.get("kind")),), "designed-240",
    )
    remaining = list(designed240)
    designed_blocks: list[list[int]] = []
    for label in ("a", "b"):
        block = select_stratified(
            designed, remaining, 80,
            lambda row: (str(row.get("kind")),), f"designed-{label}",
        )
        designed_blocks.append(block)
        block_set = set(block)
        remaining = [index for index in remaining if index not in block_set]
    designed_blocks.append(sorted(remaining))
    if len(designed_blocks) != 3 or any(len(block) != 80 for block in designed_blocks):
        raise ValueError("designed blocks are not three exact 80-row partitions")
    expected_kinds = {row[1]["kind"] for row in designed}
    for label, block in zip(("a", "b", "c"), designed_blocks, strict=True):
        observed = {designed[index][1]["kind"] for index in block}
        if observed != expected_kinds:
            raise ValueError(f"designed block {label} does not cover every skill")

    available = list(range(len(replay)))
    replay_blocks: list[list[int]] = []
    for label, designed_block in zip(
        ("a", "b", "c"), designed_blocks, strict=True
    ):
        replay_block = select_token_match(
            replay, replay_lengths, available, 80,
            token_sum(designed_block, designed_lengths), f"replay-{label}",
        )
        replay_blocks.append(replay_block)
        replay_set = set(replay_block)
        available = [index for index in available if index not in replay_set]
    extra_set = set().union(*(set(block) for block in replay_blocks))
    core = select_stratified(
        replay,
        [index for index in range(len(replay)) if index not in extra_set],
        1280,
        lambda row: (str(row.get("family")), str(row.get("kind"))),
        "replay-core",
    )

    core_rows = [replay[index] for index in core]
    designed_rows = [
        [designed[index] for index in block] for block in designed_blocks
    ]
    replay_rows = [
        [replay[index] for index in block] for block in replay_blocks
    ]
    outputs = {
        "replay_repeat.jsonl": encode_slots(core_rows + sum(replay_rows, [])),
        "designed160.jsonl": encode_slots(
            core_rows + designed_rows[0] + designed_rows[1] + replay_rows[2]
        ),
        "designed240.jsonl": encode_slots(
            core_rows + designed_rows[0] + designed_rows[1] + designed_rows[2]
        ),
    }
    core_tokens = token_sum(core, replay_lengths)
    estimates = {
        "replay_repeat": core_tokens
        + sum(token_sum(block, replay_lengths) for block in replay_blocks),
        "designed160": core_tokens
        + sum(token_sum(block, designed_lengths) for block in designed_blocks[:2])
        + sum(token_sum(block, replay_lengths) for block in replay_blocks[2:]),
        "designed240": core_tokens
        + sum(token_sum(block, designed_lengths) for block in designed_blocks[:3])
    }
    labels = ("a", "b", "c")
    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "seed": SEED,
        "source_token_receipt": {
            "path": SOURCE_TOKENS.relative_to(EXP).as_posix(),
            "sha256": sha256_file(SOURCE_TOKENS),
        },
        "sources": {
            "designed": {"path": DESIGNED.relative_to(EXP).as_posix(), "rows": 800, "sha256": DESIGNED_SHA256},
            "replay": {"path": REPLAY.relative_to(EXP).as_posix(), "rows": 2240, "sha256": REPLAY_SHA256},
        },
        "selection": {
            "shared_replay_rows": len(core),
            "designed_block_rows": {
                label: len(block) for label, block in zip(labels, designed_blocks, strict=True)
            },
            "matched_replay_block_rows": {
                label: len(block) for label, block in zip(labels, replay_blocks, strict=True)
            },
            "rows_per_arm": 1520,
            "nested_slot_replacement": True,
            "token_match_abs_limit_per_80_row_block": MATCH_ABS_LIMIT,
            "block_forward_tokens": {
                label: {
                    "designed": token_sum(designed_block, designed_lengths),
                    "replay": token_sum(replay_block, replay_lengths),
                }
                for label, designed_block, replay_block in zip(
                    labels, designed_blocks, replay_blocks, strict=True
                )
            },
            "estimated_arm_forward_tokens": estimates,
            "max_estimated_arm_token_delta": max(estimates.values()) - min(estimates.values()),
        },
        "outputs": {name: summary(value) for name, value in sorted(outputs.items())},
    }
    return outputs, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify existing bytes without writing")
    args = parser.parse_args()
    outputs, manifest = build_outputs()
    targets = {EXP / "data" / name: value for name, value in outputs.items()}
    targets[EXP / "data" / "dose_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
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
