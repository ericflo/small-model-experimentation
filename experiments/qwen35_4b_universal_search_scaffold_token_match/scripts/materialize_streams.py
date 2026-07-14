#!/usr/bin/env python3
"""Build exact-token replay and staged-search continuations from a frozen partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SCAFFOLD = EXP / "data" / "search_scaffold_source.jsonl"
REPLAY = EXP / "data" / "sft_blend.jsonl"
SOURCE_TOKENS = EXP / "data" / "source_token_lengths.json"
PREDECESSOR_MANIFEST = EXP / "data" / "predecessor_stream_manifest.json"
SCAFFOLD_SHA256 = "5854c218479a500f969bf2dbcfdbc30cd8a6095fa38aeaa652a220219b50a093"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
SOURCE_TOKEN_SHA256 = "2d48f5e2340da555e6bebeb65f26dc4d9148095f829fff09d22ed6e507028200"
PREDECESSOR_MANIFEST_SHA256 = "abf8b5055e68c0fb2bb6e32a29f7be3b3677a0dd179e77397647777a2aa0966f"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SEED = 77111
COMMON_ROWS = 200
SCAFFOLD_ROWS = 80
FILLER_ROWS = 40
CONTROL_ROWS = 120
COMMON_TOKENS = 199360
VARIABLE_TOKENS = 87454
ARM_TOKENS = 286814
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
        ("scaffold", SCAFFOLD, SCAFFOLD_SHA256, SCAFFOLD_ROWS),
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


def token_sum(indices: list[int], lengths: list[dict[str, int]]) -> int:
    return sum(lengths[index]["forward"] for index in indices)


def deterministic_rank(namespace: str, index: int, line: str) -> bytes:
    return hashlib.sha256(f"{SEED}:{namespace}:{index}:".encode() + line.encode()).digest()


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
            for new in outside:
                proposed = current - lengths[old]["forward"] + lengths[new]["forward"]
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
        correction = None
        for old_a, old_b in combinations(sorted(selected), 2):
            needed = target - current + lengths[old_a]["forward"] + lengths[old_b]["forward"]
            for new_a in outside:
                new_b = next(
                    (
                        value
                        for value in outside_by_length.get(needed - lengths[new_a]["forward"], [])
                        if value != new_a
                    ),
                    None,
                )
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
        correction = None
        for old_a, old_b, old_c in combinations(sorted(selected), 3):
            needed = (
                target - current + lengths[old_a]["forward"]
                + lengths[old_b]["forward"] + lengths[old_c]["forward"]
            )
            for new_a in outside:
                pair = next(
                    (
                        values
                        for values in pair_by_sum.get(needed - lengths[new_a]["forward"], [])
                        if new_a not in values
                    ),
                    None,
                )
                if pair is not None:
                    correction = (old_a, old_b, old_c, new_a, *pair)
                    break
            if correction is not None:
                break
        if correction is not None:
            old_a, old_b, old_c, new_a, new_b, new_c = correction
            selected.difference_update((old_a, old_b, old_c))
            selected.update((new_a, new_b, new_c))

    result = sorted(selected)
    error = abs(token_sum(result, lengths) - target)
    if len(result) != count or error > MATCH_ABS_LIMIT:
        raise ValueError(
            f"{namespace} cardinality={len(result)} token-match error={error} "
            f"exceeds ({count}, {MATCH_ABS_LIMIT})"
        )
    return result


def load_predecessor_partition(replay_lengths: list[dict[str, int]]) -> tuple[list[int], list[int]]:
    if sha256_file(PREDECESSOR_MANIFEST) != PREDECESSOR_MANIFEST_SHA256:
        raise ValueError("predecessor stream manifest changed")
    payload = json.loads(PREDECESSOR_MANIFEST.read_text(encoding="utf-8"))
    selection = payload.get("selection", {})
    core = selection.get("replay_core_source_indices", [])
    control = selection.get("replay_control_source_indices", [])
    if (
        payload.get("experiment_id") != "qwen35_4b_universal_close_weight_token_match"
        or len(core) != COMMON_ROWS
        or len(control) != CONTROL_ROWS
        or len(set(core).union(control)) != COMMON_ROWS + CONTROL_ROWS
        or token_sum(core, replay_lengths) != COMMON_TOKENS
        or token_sum(control, replay_lengths) != VARIABLE_TOKENS
    ):
        raise ValueError("predecessor replay partition identity changed")
    return core, control


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
    scaffold = load_rows(SCAFFOLD, SCAFFOLD_SHA256, SCAFFOLD_ROWS)
    replay = load_rows(REPLAY, REPLAY_SHA256, 2240)
    scaffold_lengths, replay_lengths = load_lengths()
    core, control = load_predecessor_partition(replay_lengths)
    scaffold_tokens = token_sum(list(range(SCAFFOLD_ROWS)), scaffold_lengths)
    filler_target = VARIABLE_TOKENS - scaffold_tokens
    excluded = set(core).union(control)
    filler = select_token_match(
        replay,
        replay_lengths,
        [index for index in range(len(replay)) if index not in excluded],
        FILLER_ROWS,
        filler_target,
        "scaffold-replay-filler",
    )
    if excluded.intersection(filler):
        raise ValueError("filler overlaps inherited replay partition")

    core_rows = [replay[index] for index in core]
    outputs = {
        "replay_after_close.jsonl": encode_slots(core_rows + [replay[index] for index in control]),
        "scaffold_after_close.jsonl": encode_slots(core_rows + scaffold + [replay[index] for index in filler]),
    }
    estimates = {
        "replay_after_close": token_sum(core, replay_lengths) + token_sum(control, replay_lengths),
        "scaffold_after_close": (
            token_sum(core, replay_lengths)
            + scaffold_tokens
            + token_sum(filler, replay_lengths)
        ),
    }
    if set(estimates.values()) != {ARM_TOKENS}:
        raise ValueError(f"forward-token match failed: {estimates}")

    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "seed": SEED,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parent": {
            "experiment": "qwen35_4b_universal_close_weight_token_match",
            "arm": "close_xi",
            "weights_sha256": "16e9dc75a0e33e182e916600ff6e1d75fc46dfa45e870216e2c149a41253c179",
            "config_sha256": "de953bd57502ff728a12d1627d5aacab6284b045428ec7b83026388afd8c47ff",
        },
        "source_token_receipt": {
            "path": SOURCE_TOKENS.relative_to(EXP).as_posix(),
            "sha256": SOURCE_TOKEN_SHA256,
        },
        "predecessor_partition": {
            "path": PREDECESSOR_MANIFEST.relative_to(EXP).as_posix(),
            "sha256": PREDECESSOR_MANIFEST_SHA256,
        },
        "sources": {
            "scaffold": {
                "path": SCAFFOLD.relative_to(EXP).as_posix(),
                "rows": SCAFFOLD_ROWS,
                "sha256": SCAFFOLD_SHA256,
            },
            "replay": {
                "path": REPLAY.relative_to(EXP).as_posix(),
                "rows": len(replay),
                "sha256": REPLAY_SHA256,
            },
        },
        "selection": {
            "shared_replay_rows": len(core),
            "scaffold_rows": len(scaffold),
            "scaffold_rows_by_stage": dict(sorted(Counter(
                row[1]["kind"] for row in scaffold
            ).items())),
            "candidate_replay_filler_rows": len(filler),
            "replay_control_rows": len(control),
            "rows_per_arm": COMMON_ROWS + CONTROL_ROWS,
            "position_aligned_slots": True,
            "token_match_abs_limit": MATCH_ABS_LIMIT,
            "block_forward_tokens": {
                "shared_replay": token_sum(core, replay_lengths),
                "scaffold": scaffold_tokens,
                "candidate_replay_filler": token_sum(filler, replay_lengths),
                "replay_control": token_sum(control, replay_lengths),
            },
            "estimated_arm_forward_tokens": estimates,
            "max_estimated_arm_token_delta": max(estimates.values()) - min(estimates.values()),
            "replay_core_source_indices": core,
            "candidate_replay_filler_source_indices": filler,
            "replay_control_source_indices": control,
        },
        "training": {
            "rows_per_arm": 320,
            "forward_tokens_per_arm": ARM_TOKENS,
            "optimizer_steps": 40,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "epochs": 1,
            "learning_rate": 1e-5,
            "seed": 45,
            "max_length": 4096,
            "thought_weight": 0.2,
            "close_weight": 0.2,
        },
        "outputs": {name: summary(value) for name, value in sorted(outputs.items())},
    }
    return outputs, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
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
    else:
        conflicts = [path for path in targets if path.exists()]
        if conflicts:
            parser.error(f"refusing to overwrite existing stream: {conflicts[0]}")
        for path, value in targets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(json.dumps({
        "experiment_id": EXP.name,
        "rows_per_arm": manifest["selection"]["rows_per_arm"],
        "forward_tokens": manifest["selection"]["estimated_arm_forward_tokens"],
        "blocks": manifest["selection"]["block_forward_tokens"],
        "outputs": {name: value["sha256"] for name, value in manifest["outputs"].items()},
    }, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
