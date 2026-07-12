"""Split-branch continuation-advantage routing and qualification statistics.

The selection split chooses a teacher.  Only the disjoint audit split estimates
that choice's value.  Continuations are repeated measurements of one state, so
all inference operates on state-level branch means.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


POLICIES = ("quick", "deep", "student")
TEACHERS = ("quick", "deep")
CONTRASTS = ("student", "alternate")


def _finite_mean(values: Sequence[float], *, label: str) -> float:
    if not values:
        raise ValueError(f"{label} must not be empty")
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{label} contains a non-finite score")
    return sum(converted) / len(converted)


def select_teacher(selection: Mapping[str, Sequence[float]]) -> str | None:
    """Return a strictly dominating teacher, otherwise abstain.

    There is deliberately no magnitude threshold.  A teacher must be the
    unique empirical winner and must be above the current student.
    """
    if set(selection) != set(POLICIES):
        raise ValueError(f"selection policies must be exactly {POLICIES}")
    means = {
        policy: _finite_mean(selection[policy], label=f"selection.{policy}")
        for policy in POLICIES
    }
    quick_wins = means["quick"] > means["deep"] and means["quick"] > means["student"]
    deep_wins = means["deep"] > means["quick"] and means["deep"] > means["student"]
    if quick_wins == deep_wins:
        return None
    return "quick" if quick_wins else "deep"


def score_state(
    row: Mapping[str, Any],
    *,
    selection_branches: int,
    audit_branches: int,
) -> dict[str, Any]:
    """Validate one state and compute audit-only contrasts."""
    for field in ("state_id", "block", "family", "kind", "level"):
        if field not in row:
            raise ValueError(f"state row missing {field}")
    selection = row.get("selection")
    audit = row.get("audit")
    if not isinstance(selection, Mapping) or not isinstance(audit, Mapping):
        raise ValueError("state row needs selection and audit mappings")
    if set(selection) != set(POLICIES) or set(audit) != set(POLICIES):
        raise ValueError(f"branch policies must be exactly {POLICIES}")
    for policy in POLICIES:
        if len(selection[policy]) != selection_branches:
            raise ValueError(
                f"{row['state_id']} {policy} selection branch count "
                f"{len(selection[policy])} != {selection_branches}"
            )
        if len(audit[policy]) != audit_branches:
            raise ValueError(
                f"{row['state_id']} {policy} audit branch count "
                f"{len(audit[policy])} != {audit_branches}"
            )
    selection_means = {
        policy: _finite_mean(selection[policy], label=f"selection.{policy}")
        for policy in POLICIES
    }
    audit_means = {
        policy: _finite_mean(audit[policy], label=f"audit.{policy}")
        for policy in POLICIES
    }
    teacher = select_teacher(selection)
    result = {
        "state_id": str(row["state_id"]),
        "block": int(row["block"]),
        "family": str(row["family"]),
        "kind": str(row["kind"]),
        "level": int(row["level"]),
        "selection_means": selection_means,
        "audit_means": audit_means,
        "selected_teacher": teacher,
    }
    if teacher is None:
        result.update(
            {
                "alternate_teacher": None,
                "selected_minus_student": None,
                "selected_minus_alternate": None,
            }
        )
        return result
    alternate = "deep" if teacher == "quick" else "quick"
    result.update(
        {
            "alternate_teacher": alternate,
            "selected_minus_student": audit_means[teacher] - audit_means["student"],
            "selected_minus_alternate": audit_means[teacher] - audit_means[alternate],
        }
    )
    return result


def _cell_key(row: Mapping[str, Any], *, include_block: bool) -> tuple[Any, ...]:
    key: tuple[Any, ...] = (
        str(row["family"]), str(row["kind"]), int(row["level"])
    )
    return (int(row["block"]), *key) if include_block else key


def macro_mean(
    rows: Iterable[Mapping[str, Any]],
    value_key: str,
    *,
    include_block: bool,
) -> float:
    cells: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = row[value_key]
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"invalid {value_key} value")
        cells[_cell_key(row, include_block=include_block)].append(float(value))
    if not cells:
        raise ValueError("cannot compute a macro mean over zero states")
    return sum(sum(values) / len(values) for values in cells.values()) / len(cells)


def stratified_bootstrap_lcb(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> float:
    """One-sided LCB with states resampled inside block/family/kind/level."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    cells: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = row[value_key]
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"invalid {value_key} value")
        cells[_cell_key(row, include_block=True)].append(float(value))
    if not cells:
        raise ValueError("cannot bootstrap zero states")
    ordered = [cells[key] for key in sorted(cells)]
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        cell_means = []
        for values in ordered:
            resampled = [values[rng.randrange(len(values))] for _ in values]
            cell_means.append(sum(resampled) / len(resampled))
        draws.append(sum(cell_means) / len(cell_means))
    draws.sort()
    # Linear interpolation is unnecessary at the preregistered 20k draws; the
    # conservative lower order statistic is deterministic and auditable.
    index = max(0, min(len(draws) - 1, math.floor((1.0 - confidence) * len(draws))))
    return float(draws[index])


def _summarize_subset(
    rows: Sequence[dict[str, Any]],
    *,
    blocks: Sequence[int],
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(rows), "by_contrast": {}}
    for contrast_index, contrast in enumerate(CONTRASTS):
        value_key = f"selected_minus_{contrast}"
        block_means = {}
        for block in blocks:
            block_rows = [row for row in rows if int(row["block"]) == int(block)]
            block_means[str(block)] = (
                macro_mean(block_rows, value_key, include_block=False)
                if block_rows
                else None
            )
        pooled = macro_mean(rows, value_key, include_block=True)
        lcb = stratified_bootstrap_lcb(
            rows,
            value_key,
            samples=samples,
            confidence=confidence,
            seed=seed + contrast_index,
        )
        result["by_contrast"][contrast] = {
            "block_macro_means": block_means,
            "pooled_macro_mean": pooled,
            "one_sided_lcb": lcb,
            "all_block_means_positive": all(
                value is not None and value > 0.0 for value in block_means.values()
            ),
            "lcb_above_zero": lcb > 0.0,
        }
    return result


def analyze_route_blocks(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    selection_branches: int,
    audit_branches: int,
    qualified_teacher: str,
    minimum_per_qualified_teacher_per_block: int,
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int = 71701,
) -> dict[str, Any]:
    """Analyze the full route, gating only the preregistered source teacher."""
    if qualified_teacher not in TEACHERS:
        raise ValueError(f"qualified_teacher must be one of {TEACHERS}")
    if len({str(row.get("state_id")) for row in raw_rows}) != len(raw_rows):
        raise ValueError("state_id values must be globally unique")
    scored = [
        score_state(
            row,
            selection_branches=selection_branches,
            audit_branches=audit_branches,
        )
        for row in raw_rows
    ]
    blocks = sorted({int(row["block"]) for row in scored})
    if len(blocks) != 2:
        raise ValueError(f"exactly two qualification blocks required, got {blocks}")
    routed = [row for row in scored if row["selected_teacher"] is not None]
    support: dict[str, dict[str, int]] = {}
    teacher_results = {}
    for teacher_index, teacher in enumerate(TEACHERS):
        subset = [row for row in routed if row["selected_teacher"] == teacher]
        counts = {
            str(block): sum(int(row["block"]) == block for row in subset)
            for block in blocks
        }
        support[teacher] = counts
        this_support = all(
            count >= minimum_per_qualified_teacher_per_block
            for count in counts.values()
        )
        if subset:
            summary = _summarize_subset(
                subset,
                blocks=blocks,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=bootstrap_seed + 100 * teacher_index,
            )
            this_effect = all(
                values["all_block_means_positive"] and values["lcb_above_zero"]
                for values in summary["by_contrast"].values()
            )
        else:
            summary = {"n": 0, "by_contrast": {}}
            this_effect = False
        summary["support_passed"] = this_support
        summary["effect_passed"] = this_effect
        summary["passed"] = this_support and this_effect
        teacher_results[teacher] = summary

    if routed:
        combined = _summarize_subset(
            routed,
            blocks=blocks,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=bootstrap_seed + 900,
        )
        combined_passed = all(
            values["all_block_means_positive"] and values["lcb_above_zero"]
            for values in combined["by_contrast"].values()
        )
    else:
        combined = {"n": 0, "by_contrast": {}}
        combined_passed = False
    combined["passed"] = combined_passed
    passed = bool(teacher_results[qualified_teacher]["passed"])
    return {
        "schema_version": 1,
        "stage": "split_branch_route_qualification",
        "estimand": "audit continuation value after selection-only teacher routing",
        "blocks": blocks,
        "state_count": len(scored),
        "routed_count": len(routed),
        "abstained_count": len(scored) - len(routed),
        "route_rate": len(routed) / len(scored) if scored else 0.0,
        "qualified_teacher": qualified_teacher,
        "minimum_per_qualified_teacher_per_block": (
            minimum_per_qualified_teacher_per_block
        ),
        "support": support,
        "by_teacher": teacher_results,
        "combined": combined,
        "gate": {"passed": passed},
        "downstream_authorization": "locality_pilot" if passed else "stop_before_mopd",
        "states": scored,
    }
