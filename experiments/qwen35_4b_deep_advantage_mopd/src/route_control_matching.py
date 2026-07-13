"""Canonical deterministic matching for non-advantage route controls."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


MATCH_TIERS = ("exact_cell", "family_kind", "kind_level", "kind")


def match_key(row: Mapping[str, Any], tier: str) -> tuple[Any, ...]:
    """Return the frozen no-cross-kind geometry key for one matching tier."""

    values = {
        "exact_cell": (str(row["family"]), str(row["kind"]), int(row["level"])),
        "family_kind": (str(row["family"]), str(row["kind"])),
        "kind_level": (str(row["kind"]), int(row["level"])),
        "kind": (str(row["kind"]),),
    }
    if tier not in values:
        raise ValueError(f"unknown non-advantage-route matching tier: {tier}")
    return values[tier]


def matched_non_advantage_route_units(
    selected: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    match_order: Sequence[str],
) -> list[dict[str, Any]]:
    """Greedily match failed non-deep states to the exact primary geometry.

    Primary states and candidates are both ordered by stable state identity.
    The first available candidate at the first permitted tier is consumed.
    This is the canonical implementation shared by control rematching and,
    after the active integration process exits, round assembly.
    """

    order = tuple(str(value) for value in match_order)
    if order != MATCH_TIERS:
        raise ValueError(
            f"non-advantage-route match order changed: {order} != {MATCH_TIERS}"
        )
    remaining = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda row: str(row["state_id"]),
    )
    matched: list[dict[str, Any]] = []
    for source in sorted(selected, key=lambda row: str(row["state_id"])):
        chosen_index = None
        chosen_tier = None
        for tier in order:
            source_key = match_key(source, tier)
            chosen_index = next(
                (
                    index
                    for index, candidate in enumerate(remaining)
                    if match_key(candidate, tier) == source_key
                ),
                None,
            )
            if chosen_index is not None:
                chosen_tier = tier
                break
        if chosen_index is None or chosen_tier is None:
            raise ValueError(
                f"no kind-preserving non-advantage-route match for {source['state_id']}"
            )
        candidate = dict(remaining.pop(chosen_index))
        candidate["observed_route"] = candidate.get("primary_teacher") or "abstain"
        candidate["primary_teacher"] = "deep"
        candidate["role"] = "route_control"
        candidate["offpolicy_target"] = None
        candidate["matched_primary_state_id"] = str(source["state_id"])
        candidate["match_tier"] = chosen_tier
        matched.append(candidate)
    return matched
