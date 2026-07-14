#!/usr/bin/env python3
"""Build byte-identical standard/treatment data and an exact-token replay control."""

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
SOURCE_TOKEN_SHA256 = "064a1cce4bded46c2fa9184f71e9efe8b2f649f65f3e1284e74b1a70751f542f"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SEED = 77110
PARENT_SELECTION_SEED = 77109
PARENT_D160_INDEX_SHA256 = "6228daa8f5c7defbedaddc952af13f468a98dbe10f835717bcfebfeea17c25af"
TARGET_COUNTS = {"u_execute": 40, "u_induct": 40}
COMMON_REPLAY_ROWS = 200
REPLAY_FILLER_ROWS = 40
REPLAY_CONTROL_ROWS = 120
MATCH_ABS_LIMIT = 0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_index_sha256(indices: list[int]) -> str:
    value = (json.dumps(sorted(indices), separators=(",", ":")) + "\n").encode()
    return sha256_bytes(value)


def load_rows(path: Path, expected_sha256: str, expected_rows: int) -> list[tuple[str, dict]]:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"frozen source changed: {path}")
    rows = [(line, json.loads(line)) for line in raw.decode("utf-8").splitlines() if line]
    if len(rows) != expected_rows or not all(isinstance(row, dict) for _, row in rows):
        raise ValueError(f"unexpected source rows: {path}")
    return rows


def load_lengths() -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    if sha256_file(SOURCE_TOKENS) != SOURCE_TOKEN_SHA256:
        raise ValueError("source token receipt bytes changed")
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
    fields = {"forward", "prompt", "think_target", "close_target", "answer_target"}
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
            or not all(set(item) == fields for item in lengths)
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


def deterministic_rank(seed: int, namespace: str, index: int, line: str) -> bytes:
    return hashlib.sha256(f"{seed}:{namespace}:{index}:".encode() + line.encode()).digest()


def select_stratified(
    rows: list[tuple[str, dict]],
    indices: list[int],
    total: int,
    group_fn: Callable[[dict], tuple[str, ...]],
    namespace: str,
    *,
    seed: int,
) -> list[int]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in indices:
        groups[group_fn(rows[index][1])].append(index)
    quotas = allocation({key: len(value) for key, value in groups.items()}, total)
    selected = []
    for key, members in sorted(groups.items()):
        ranked = sorted(
            members,
            key=lambda index: deterministic_rank(seed, namespace, index, rows[index][0]),
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
    """Deterministically find a fixed-cardinality exact token sum."""
    target_average = target / count
    ranked = sorted(
        available,
        key=lambda index: (
            abs(lengths[index]["forward"] - target_average),
            deterministic_rank(SEED, namespace, index, rows[index][0]),
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
                tie = deterministic_rank(SEED, f"{namespace}-swap-{old}", new, rows[new][0])
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
            needed = target - current + lengths[old_a]["forward"] + lengths[old_b]["forward"]
            for new_a in outside:
                candidates = outside_by_length.get(needed - lengths[new_a]["forward"], [])
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
        correction3: tuple[int, int, int, int, int, int] | None = None
        for old_a, old_b, old_c in combinations(sorted(selected), 3):
            needed = (
                target - current + lengths[old_a]["forward"]
                + lengths[old_b]["forward"] + lengths[old_c]["forward"]
            )
            for new_a in outside:
                pair = next(
                    (
                        values for values in pair_by_sum.get(
                            needed - lengths[new_a]["forward"], []
                        )
                        if new_a not in values
                    ),
                    None,
                )
                if pair is not None:
                    correction3 = (old_a, old_b, old_c, new_a, *pair)
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
    if len(result) != count or error > MATCH_ABS_LIMIT:
        raise ValueError(
            f"{namespace} cardinality={len(result)} token-match error={error} "
            f"exceeds ({count}, {MATCH_ABS_LIMIT})"
        )
    return result


def reconstruct_parent_d160(designed: list[tuple[str, dict]]) -> list[int]:
    designed240 = select_stratified(
        designed, list(range(len(designed))), 240,
        lambda row: (str(row.get("kind")),), "designed-240",
        seed=PARENT_SELECTION_SEED,
    )
    remaining = list(designed240)
    blocks = []
    for label in ("a", "b"):
        block = select_stratified(
            designed, remaining, 80,
            lambda row: (str(row.get("kind")),), f"designed-{label}",
            seed=PARENT_SELECTION_SEED,
        )
        blocks.extend(block)
        selected = set(block)
        remaining = [index for index in remaining if index not in selected]
    result = sorted(blocks)
    if len(result) != 160 or canonical_index_sha256(result) != PARENT_D160_INDEX_SHA256:
        raise ValueError("failed to authenticate the published designed160 source exclusion")
    return result


def encode_slots(slots: list[tuple[str, dict]]) -> bytes:
    order = sorted(
        range(len(slots)),
        key=lambda index: deterministic_rank(SEED, "shared-slot-order", index, str(index)),
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
    parent_d160 = reconstruct_parent_d160(designed)
    parent_set = set(parent_d160)

    target_indices = []
    target_by_kind = {}
    for kind, count in TARGET_COUNTS.items():
        available = [
            index for index, (_, row) in enumerate(designed)
            if row.get("kind") == kind and index not in parent_set
        ]
        selected = select_stratified(
            designed, available, count,
            lambda row: (str(row.get("surface")), str(row.get("level"))),
            f"target-{kind}", seed=SEED,
        )
        target_by_kind[kind] = selected
        target_indices.extend(selected)
    target_indices = sorted(target_indices)
    if len(target_indices) != sum(TARGET_COUNTS.values()) or parent_set.intersection(target_indices):
        raise ValueError("target selection is not the frozen fresh 80-row block")

    replay_filler = select_stratified(
        replay, list(range(len(replay))), REPLAY_FILLER_ROWS,
        lambda row: (str(row.get("family")), str(row.get("kind"))),
        "replay-filler", seed=SEED,
    )
    variable_tokens = token_sum(target_indices, designed_lengths) + token_sum(
        replay_filler, replay_lengths
    )
    filler_set = set(replay_filler)
    replay_control = select_token_match(
        replay, replay_lengths,
        [index for index in range(len(replay)) if index not in filler_set],
        REPLAY_CONTROL_ROWS, variable_tokens, "replay-control",
    )
    excluded = filler_set.union(replay_control)
    replay_core = select_stratified(
        replay,
        [index for index in range(len(replay)) if index not in excluded],
        COMMON_REPLAY_ROWS,
        lambda row: (str(row.get("family")), str(row.get("kind"))),
        "replay-core", seed=SEED,
    )
    if len(set(replay_core).union(replay_filler, replay_control)) != (
        COMMON_REPLAY_ROWS + REPLAY_FILLER_ROWS + REPLAY_CONTROL_ROWS
    ):
        raise ValueError("replay core/filler/control selections overlap")

    core_rows = [replay[index] for index in replay_core]
    targeted_variable = [designed[index] for index in target_indices] + [
        replay[index] for index in replay_filler
    ]
    control_variable = [replay[index] for index in replay_control]
    outputs = {
        "replay_repeat.jsonl": encode_slots(core_rows + control_variable),
        "targeted_standard.jsonl": encode_slots(core_rows + targeted_variable),
    }
    common_tokens = token_sum(replay_core, replay_lengths)
    estimates = {
        "replay_repeat": common_tokens + token_sum(replay_control, replay_lengths),
        "targeted_standard": common_tokens + variable_tokens,
    }
    if len(set(estimates.values())) != 1:
        raise ValueError(f"forward-token match failed: {estimates}")

    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "seed": SEED,
        "source_token_receipt": {
            "path": SOURCE_TOKENS.relative_to(EXP).as_posix(),
            "sha256": sha256_file(SOURCE_TOKENS),
        },
        "sources": {
            "designed": {
                "path": DESIGNED.relative_to(EXP).as_posix(), "rows": 800,
                "sha256": DESIGNED_SHA256,
            },
            "replay": {
                "path": REPLAY.relative_to(EXP).as_posix(), "rows": 2240,
                "sha256": REPLAY_SHA256,
            },
        },
        "parent_exclusion": {
            "experiment": "qwen35_4b_universal_mid_density_token_match",
            "arm": "designed160",
            "selection_seed": PARENT_SELECTION_SEED,
            "designed_source_indices": parent_d160,
            "index_sha256": canonical_index_sha256(parent_d160),
            "new_target_overlap_rows": 0,
        },
        "selection": {
            "shared_replay_rows": len(replay_core),
            "target_rows_by_kind": {
                kind: len(indices) for kind, indices in target_by_kind.items()
            },
            "replay_filler_rows": len(replay_filler),
            "replay_control_rows": len(replay_control),
            "rows_per_arm": COMMON_REPLAY_ROWS + REPLAY_CONTROL_ROWS,
            "position_aligned_slots": True,
            "standard_and_close_training_bytes_identical": True,
            "token_match_abs_limit": MATCH_ABS_LIMIT,
            "block_forward_tokens": {
                "shared_replay": common_tokens,
                "target_designed": token_sum(target_indices, designed_lengths),
                "target_replay_filler": token_sum(replay_filler, replay_lengths),
                "replay_control": token_sum(replay_control, replay_lengths),
            },
            "estimated_arm_forward_tokens": estimates,
            "max_estimated_arm_token_delta": max(estimates.values()) - min(estimates.values()),
            "target_designed_source_indices": target_indices,
            "replay_core_source_indices": replay_core,
            "replay_filler_source_indices": replay_filler,
            "replay_control_source_indices": replay_control,
        },
        "loss_contrast": {
            "ordinary_close_weight": 0.2,
            "treatment_close_weight": 1.0,
            "treatment_kinds": sorted(TARGET_COUNTS),
            "all_other_assigned_token_weights_identical": True,
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
    targets[EXP / "data" / "stream_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if args.check:
        for path, expected in targets.items():
            if not path.is_file() or path.read_bytes() != expected:
                raise SystemExit(f"derived stream is absent or changed: {path}")
        print(json.dumps({
            "experiment_id": EXP.name,
            "rows_per_arm": manifest["selection"]["rows_per_arm"],
            "forward_tokens": manifest["selection"]["estimated_arm_forward_tokens"],
            "outputs": {
                name: value["sha256"] for name, value in manifest["outputs"].items()
            },
        }, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    conflicts = [path for path in targets if path.exists()]
    if conflicts:
        parser.error(f"refusing to overwrite existing stream: {conflicts[0]}")
    for path, value in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
