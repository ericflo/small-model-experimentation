"""Deterministic natural-boundary checkpoint and pivot selection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def natural_checkpoint_indices(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    max_checkpoints: int,
) -> list[int]:
    """Return approximately even sentence/newline boundaries, including end."""
    if max_checkpoints < 2:
        raise ValueError("max_checkpoints must be at least two")
    if not token_ids:
        raise ValueError("cannot checkpoint an empty trace")
    markers = ("\n", "\n\n", ".", "?", "!", ";", "。", "！", "？")
    boundary_ids = {
        int(value)
        for marker in markers
        for value in tokenizer.encode(marker, add_special_tokens=False)
    }
    minimum = min(32, max(1, len(token_ids) // 4))
    candidates = [
        index + 1
        for index, token_id in enumerate(token_ids)
        if int(token_id) in boundary_ids and index + 1 >= minimum
    ]
    if len(token_ids) not in candidates:
        candidates.append(len(token_ids))
    candidates = sorted(set(candidates))
    if len(candidates) <= max_checkpoints:
        return candidates

    chosen: set[int] = {len(token_ids)}
    for slot in range(1, max_checkpoints):
        target = len(token_ids) * slot / max_checkpoints
        chosen.add(min(candidates, key=lambda value: (abs(value - target), value)))
    if len(chosen) < max_checkpoints:
        for candidate in sorted(
            candidates,
            key=lambda value: min(abs(value - prior) for prior in chosen),
            reverse=True,
        ):
            chosen.add(candidate)
            if len(chosen) == max_checkpoints:
                break
    return sorted(chosen)


def choose_pivot(
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    minimum_positive_jump: float,
    fallback_fraction: float,
    full_length: int,
) -> dict[str, Any]:
    """Choose the boundary before the largest registered joint-gain jump."""
    if not checkpoints:
        raise ValueError("checkpoint list may not be empty")
    ordered = sorted(checkpoints, key=lambda row: int(row["token_index"]))
    if int(ordered[-1]["token_index"]) != full_length:
        raise ValueError("the full natural trace must be the last checkpoint")
    previous_index = 0
    previous_gain = 0.0
    jumps: list[dict[str, Any]] = []
    for row in ordered:
        gain = float(row["joint_gain_per_answer_token"])
        jumps.append(
            {
                "from_token_index": previous_index,
                "to_token_index": int(row["token_index"]),
                "jump": gain - previous_gain,
            }
        )
        previous_index = int(row["token_index"])
        previous_gain = gain
    winner = max(jumps, key=lambda row: (float(row["jump"]), -int(row["to_token_index"])))
    if float(winner["jump"]) >= minimum_positive_jump:
        pivot_index = int(winner["from_token_index"])
        reason = "before_largest_positive_jump"
    else:
        target = fallback_fraction * full_length
        nonterminal = [row for row in ordered if int(row["token_index"]) < full_length]
        choices = nonterminal or ordered
        pivot_index = int(
            min(choices, key=lambda row: (abs(int(row["token_index"]) - target), int(row["token_index"]))) [
                "token_index"
            ]
        )
        reason = "nearest_fallback_fraction"
    return {
        "pivot_token_index": pivot_index,
        "pivot_reason": reason,
        "largest_jump": winner,
        "jumps": jumps,
    }
