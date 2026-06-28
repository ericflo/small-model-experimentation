from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any


def prefix_candidates(record: dict[str, Any], budget: int) -> list[dict[str, Any]]:
    return [candidate for candidate in record["candidates"] if int(candidate["order"]) < budget]


def visible_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if candidate.get("visible_all_pass")]


def coverage(candidates: list[dict[str, Any]]) -> bool:
    return any(candidate.get("full_pass") for candidate in candidates)


def visible_coverage(candidates: list[dict[str, Any]]) -> bool:
    return any(candidate.get("visible_all_pass") and candidate.get("full_pass") for candidate in candidates)


def select_candidate(
    candidates: list[dict[str, Any]],
    policy: str,
    rng: random.Random,
    scores: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    visible = visible_candidates(candidates)
    if not visible:
        return None
    if policy == "first_visible":
        return sorted(visible, key=lambda item: item["order"])[0]
    if policy == "shortest_visible":
        return sorted(visible, key=lambda item: (len(item.get("code", "")), item["order"]))[0]
    if policy == "random_visible":
        return rng.choice(visible)
    if policy == "public_signature_majority":
        counts = Counter(candidate.get("public_signature", "") for candidate in candidates)
        return sorted(visible, key=lambda item: (-counts[item.get("public_signature", "")], item["order"]))[0]
    if policy == "oracle_coverage":
        positives = [candidate for candidate in visible if candidate.get("full_pass")]
        return sorted(positives, key=lambda item: item["order"])[0] if positives else sorted(visible, key=lambda item: item["order"])[0]
    if policy == "model":
        assert scores is not None
        return sorted(visible, key=lambda item: (-scores.get(item["candidate_id"], -math.inf), item["order"]))[0]
    raise ValueError(f"unknown policy: {policy}")


def model_state_for_prefix(
    record: dict[str, Any],
    budget: int,
    scores: dict[str, float],
    rng: random.Random,
) -> dict[str, Any]:
    candidates = prefix_candidates(record, budget)
    selected = select_candidate(candidates, "model", rng, scores=scores)
    visible = visible_candidates(candidates)
    visible_scores = sorted([scores.get(candidate["candidate_id"], -math.inf) for candidate in visible], reverse=True)
    top = visible_scores[0] if visible_scores else 0.0
    margin = (visible_scores[0] - visible_scores[1]) if len(visible_scores) >= 2 else 0.0
    return {
        "budget": budget,
        "visible_count": len(visible),
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_source": selected["source"] if selected else "none",
        "selected_hidden_all": bool(selected and selected.get("full_pass")),
        "selected_public_status": "PASS" if selected else "NONE",
        "selected_code": selected.get("code", "") if selected else "",
        "top_score": float(top),
        "score_margin": float(margin),
        "prefix_coverage": coverage(candidates),
        "prefix_visible_coverage": visible_coverage(candidates),
    }

