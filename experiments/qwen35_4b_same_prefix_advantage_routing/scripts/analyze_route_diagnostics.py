#!/usr/bin/env python3
"""Exploratory diagnostics for a completed split-branch route receipt.

These summaries do not alter or re-evaluate the preregistered gate.  They use
only already generated selection/audit branches to explain a terminal result.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, sha256_file, write_json  # noqa: E402
from advantage_routing import select_teacher  # noqa: E402


def _cell(row: dict) -> str:
    return f"{row['family']}/{row['kind']}/L{int(row['level'])}"


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.mean(left)
    mean_right = statistics.mean(right)
    numerator = sum(
        (x - mean_left) * (y - mean_right) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - mean_left) ** 2 for x in left)
        * sum((y - mean_right) ** 2 for y in right)
    )
    return numerator / denominator if denominator > 0.0 else None


def _cell_macro(rows: list[dict], value: Callable[[dict], float]) -> float:
    cells: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        converted = float(value(row))
        if not math.isfinite(converted):
            raise ValueError("route diagnostic encountered a non-finite value")
        cells[_cell(row)].append(converted)
    if not cells:
        raise ValueError("cannot macro-average zero route rows")
    return statistics.mean(statistics.mean(values) for values in cells.values())


def _policy_reliability(rows: list[dict], policy: str) -> dict:
    selection = [float(row["selection_means"][policy]) for row in rows]
    audit = [float(row["audit_means"][policy]) for row in rows]
    return {
        "n": len(rows),
        "selection_mean": statistics.mean(selection),
        "audit_mean": statistics.mean(audit),
        "selection_audit_pearson": _pearson(selection, audit),
        "selection_audit_mae": statistics.mean(
            abs(left - right) for left, right in zip(selection, audit)
        ),
    }


def _strict_teacher_choice(means: dict[str, float]) -> str | None:
    """Return the unique teacher winner, or abstain for student/tied winners."""

    policies = ("quick", "deep", "student")
    if set(means) != set(policies):
        raise ValueError("teacher choice requires quick, deep, and student means")
    best = max(policies, key=lambda policy: means[policy])
    if best == "student":
        return None
    if not all(means[best] > means[other] for other in policies if other != best):
        return None
    return best


def _cross_block_group_router(
    rows: list[dict], group_fields: tuple[str, ...], *, require_half_agreement: bool
) -> dict:
    """Fit a coarse route on one block and describe it on the other block.

    This is deliberately exploratory.  Each direction uses disjoint states,
    while the symmetric two-direction summary reuses each block once for fit
    and once for evaluation and therefore is not an independent confirmation.
    """

    def group_key(row: dict) -> tuple:
        return tuple(row[field] for field in group_fields)

    def group_means(group: list[dict], half: str) -> dict[str, float]:
        field = f"{half}_means"
        return {
            policy: statistics.mean(float(row[field][policy]) for row in group)
            for policy in ("quick", "deep", "student")
        }

    blocks = sorted({int(row["block"]) for row in rows})
    if blocks != [0, 1]:
        raise ValueError("cross-block route requires exact blocks 0 and 1")
    directions = {}
    for fit_block, eval_block in ((0, 1), (1, 0)):
        fit_groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            if int(row["block"]) == fit_block:
                fit_groups[group_key(row)].append(row)
        choices = {}
        for key, group in fit_groups.items():
            selection_choice = _strict_teacher_choice(
                group_means(group, "selection")
            )
            audit_choice = _strict_teacher_choice(group_means(group, "audit"))
            if require_half_agreement:
                choices[key] = (
                    audit_choice if audit_choice == selection_choice else None
                )
            else:
                choices[key] = audit_choice

        selected = []
        for row in rows:
            if int(row["block"]) != eval_block:
                continue
            teacher = choices.get(group_key(row))
            if teacher is None:
                continue
            alternate = "deep" if teacher == "quick" else "quick"
            selected.append(
                {
                    "row": row,
                    "teacher": teacher,
                    "student": float(row["audit_means"][teacher])
                    - float(row["audit_means"]["student"]),
                    "alternate": float(row["audit_means"][teacher])
                    - float(row["audit_means"][alternate]),
                }
            )

        cell_values: dict[str, list[dict]] = defaultdict(list)
        for item in selected:
            cell_values[_cell(item["row"])].append(item)
        contrasts = {}
        for contrast in ("student", "alternate"):
            contrasts[contrast] = {
                "state_mean": (
                    statistics.mean(item[contrast] for item in selected)
                    if selected
                    else None
                ),
                "cell_macro_mean": (
                    statistics.mean(
                        statistics.mean(item[contrast] for item in cell)
                        for cell in cell_values.values()
                    )
                    if cell_values
                    else None
                ),
            }
        directions[f"fit_{fit_block}_evaluate_{eval_block}"] = {
            "fit_block": fit_block,
            "evaluation_block": eval_block,
            "selected_states": len(selected),
            "selected_cells": len(cell_values),
            "teacher_counts": dict(
                sorted(Counter(item["teacher"] for item in selected).items())
            ),
            "kind_counts": dict(
                sorted(Counter(str(item["row"]["kind"]) for item in selected).items())
            ),
            "contrasts": contrasts,
        }
    return {
        "group_fields": list(group_fields),
        "fit_rule": (
            "same unique teacher strictly wins group means on selection and audit halves"
            if require_half_agreement
            else "unique teacher strictly wins group audit means"
        ),
        "directions": directions,
    }


def _teacher_block(rows: list[dict], teacher: str) -> dict:
    routed = [row for row in rows if row["selected_teacher"] == teacher]
    if not routed:
        return {"n": 0}
    alternate = "deep" if teacher == "quick" else "quick"
    selection_student = [
        float(row["selection_means"][teacher])
        - float(row["selection_means"]["student"])
        for row in routed
    ]
    selection_alternate = [
        float(row["selection_means"][teacher])
        - float(row["selection_means"][alternate])
        for row in routed
    ]
    audit_student = [float(row["selected_minus_student"]) for row in routed]
    audit_alternate = [float(row["selected_minus_alternate"]) for row in routed]
    family_counts = Counter(str(row["family"]) for row in routed)
    kind_counts = Counter(str(row["kind"]) for row in routed)
    cells: dict[str, list[dict]] = defaultdict(list)
    for row in routed:
        cells[_cell(row)].append(row)
    cell_table = {}
    for key, values in sorted(cells.items()):
        cell_table[key] = {
            "n": len(values),
            "selected_minus_student": statistics.mean(
                float(row["selected_minus_student"]) for row in values
            ),
            "selected_minus_alternate": statistics.mean(
                float(row["selected_minus_alternate"]) for row in values
            ),
            "teacher_audit_mean": statistics.mean(
                float(row["audit_means"][teacher]) for row in values
            ),
            "student_audit_mean": statistics.mean(
                float(row["audit_means"]["student"]) for row in values
            ),
        }
    audit_dominator = [
        float(row["audit_means"][teacher]) > float(row["audit_means"]["student"])
        and float(row["audit_means"][teacher]) > float(row["audit_means"][alternate])
        for row in routed
    ]
    margin_sensitivity = {}
    for threshold in (0.0, 0.1, 0.25, 0.5):
        subset = []
        for row in routed:
            minimum_margin = min(
                float(row["selection_means"][teacher])
                - float(row["selection_means"]["student"]),
                float(row["selection_means"][teacher])
                - float(row["selection_means"][alternate]),
            )
            if minimum_margin >= threshold:
                subset.append(row)
        key = f"{threshold:.2f}"
        if not subset:
            margin_sensitivity[key] = {"n": 0}
            continue
        dominators = [
            float(row["audit_means"][teacher])
            > float(row["audit_means"]["student"])
            and float(row["audit_means"][teacher])
            > float(row["audit_means"][alternate])
            for row in subset
        ]
        margin_sensitivity[key] = {
            "n": len(subset),
            "audit_state_mean_student": statistics.mean(
                float(row["selected_minus_student"]) for row in subset
            ),
            "audit_dominator_precision": sum(dominators) / len(dominators),
        }
    return {
        "n": len(routed),
        "kind_counts": dict(sorted(kind_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "maximum_family_share": max(family_counts.values()) / len(routed),
        "unique_cells": len(cells),
        "selection_margin": {
            "student_mean": statistics.mean(selection_student),
            "alternate_mean": statistics.mean(selection_alternate),
        },
        "audit_state_mean": {
            "student": statistics.mean(audit_student),
            "alternate": statistics.mean(audit_alternate),
        },
        "audit_cell_macro": {
            "student": _cell_macro(
                routed, lambda row: float(row["selected_minus_student"])
            ),
            "alternate": _cell_macro(
                routed, lambda row: float(row["selected_minus_alternate"])
            ),
        },
        "absolute_audit_cell_macro": {
            policy: _cell_macro(
                routed, lambda row, policy=policy: float(row["audit_means"][policy])
            )
            for policy in (teacher, alternate, "student")
        },
        "selection_optimism": {
            "student": statistics.mean(selection_student) - statistics.mean(audit_student),
            "alternate": statistics.mean(selection_alternate)
            - statistics.mean(audit_alternate),
        },
        "margin_audit_pearson": {
            "student": _pearson(selection_student, audit_student),
            "alternate": _pearson(selection_alternate, audit_alternate),
        },
        "audit_positive_fraction": {
            "student": sum(value > 0.0 for value in audit_student) / len(routed),
            "alternate": sum(value > 0.0 for value in audit_alternate) / len(routed),
            "both": sum(
                student > 0.0 and alternate_value > 0.0
                for student, alternate_value in zip(audit_student, audit_alternate)
            )
            / len(routed),
        },
        "audit_dominator_precision": sum(audit_dominator) / len(audit_dominator),
        "posthoc_minimum_selection_margin_sensitivity": margin_sensitivity,
        "cells": cell_table,
    }


def _split_label_stability(rows: list[dict]) -> dict:
    labels = ("quick", "deep", "abstain")
    confusion = {left: {right: 0 for right in labels} for left in labels}
    first_sets = {teacher: set() for teacher in ("quick", "deep")}
    second_sets = {teacher: set() for teacher in ("quick", "deep")}
    exact = 0
    for row in rows:
        first = select_teacher(row["selection"])
        second = select_teacher(row["audit"])
        left = first or "abstain"
        right = second or "abstain"
        confusion[left][right] += 1
        exact += left == right
        if first is not None:
            first_sets[first].add(str(row["state_id"]))
        if second is not None:
            second_sets[second].add(str(row["state_id"]))
    by_teacher = {}
    for teacher in ("quick", "deep"):
        left, right = first_sets[teacher], second_sets[teacher]
        union = left | right
        by_teacher[teacher] = {
            "first_half_routes": len(left),
            "second_half_routes": len(right),
            "same_teacher_intersection": len(left & right),
            "jaccard": len(left & right) / len(union) if union else None,
            "first_half_precision_for_same_second_half_teacher": (
                len(left & right) / len(left) if left else None
            ),
        }
    return {
        "n": len(rows),
        "exact_three_way_agreement": exact / len(rows),
        "confusion_first_half_to_second_half": confusion,
        "by_teacher": by_teacher,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--route",
        type=Path,
        default=EXP / "analysis" / "route_qualification.json",
    )
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "route_diagnostics.json"
    )
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    route = json.loads(args.route.read_text(encoding="utf-8"))
    if route.get("stage") != "split_branch_route_qualification":
        raise SystemExit("diagnostics require a completed route-qualification receipt")
    if route.get("config_sha256") != sha256_file(config_path):
        raise SystemExit("route/config checksum mismatch")
    rows = list(route.get("states") or [])
    if len(rows) != 384 or len({row["state_id"] for row in rows}) != len(rows):
        raise SystemExit("route diagnostics require the exact two-block state ledger")
    blocks = sorted({int(row["block"]) for row in rows})
    if blocks != [0, 1]:
        raise SystemExit(f"unexpected route blocks: {blocks}")

    by_block = {}
    teacher_cells: dict[str, dict[int, set[str]]] = {
        teacher: {} for teacher in ("quick", "deep")
    }
    for block in blocks:
        block_rows = [row for row in rows if int(row["block"]) == block]
        routed = [row for row in block_rows if row["selected_teacher"] is not None]
        reliability = {
            policy: _policy_reliability(block_rows, policy)
            for policy in ("quick", "deep", "student")
        }
        by_block[str(block)] = {
            "states": len(block_rows),
            "routed": len(routed),
            "abstained": len(block_rows) - len(routed),
            "policy_reliability": reliability,
            "unconditional_audit_delta_vs_student": {
                teacher: reliability[teacher]["audit_mean"]
                - reliability["student"]["audit_mean"]
                for teacher in ("quick", "deep")
            },
            "teachers": {
                teacher: _teacher_block(block_rows, teacher)
                for teacher in ("quick", "deep")
            },
        }
        for teacher in ("quick", "deep"):
            teacher_cells[teacher][block] = {
                _cell(row)
                for row in block_rows
                if row["selected_teacher"] == teacher
            }

    assembled_by_block = {}
    for artifact in route.get("block_artifacts") or []:
        path = Path(artifact["path"])
        if artifact.get("sha256") != sha256_file(path):
            raise SystemExit(f"assembled block checksum mismatch: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        block = int(payload["block"])
        assembled_by_block[str(block)] = _split_label_stability(payload["rows"])
    if set(assembled_by_block) != {"0", "1"}:
        raise SystemExit("route diagnostics could not recover both assembled blocks")

    kind_totals = Counter(str(row["kind"]) for row in rows)
    kind_routed = Counter(
        str(row["kind"]) for row in rows if row["selected_teacher"] is not None
    )
    cell_overlap = {}
    for teacher in ("quick", "deep"):
        left, right = teacher_cells[teacher][0], teacher_cells[teacher][1]
        union = left | right
        cell_overlap[teacher] = {
            "block_0_cells": len(left),
            "block_1_cells": len(right),
            "intersection": len(left & right),
            "jaccard": len(left & right) / len(union) if union else None,
        }

    quick_result = route["by_teacher"]["quick"]["by_contrast"]["student"]
    flags = {
        "terminal_gate_failed": not bool(route.get("gate", {}).get("passed")),
        "deep_teacher_passed": bool(route["by_teacher"]["deep"]["passed"]),
        "quick_teacher_failed": not bool(route["by_teacher"]["quick"]["passed"]),
        "combined_router_passed": bool(route["combined"]["passed"]),
        "quick_student_block_sign_reversal": (
            float(quick_result["block_macro_means"]["0"]) > 0.0
            and float(quick_result["block_macro_means"]["1"]) < 0.0
        ),
        "pooled_only_rule_would_have_missed_quick_failure": (
            float(quick_result["pooled_macro_mean"]) > 0.0
            and float(quick_result["one_sided_lcb"]) > 0.0
            and not bool(quick_result["all_block_means_positive"])
        ),
        "quick_block_1_audit_dominator_precision_below_half": (
            by_block["1"]["teachers"]["quick"]["audit_dominator_precision"] < 0.5
        ),
        "point_10_margin_would_not_rescue_quick_block_1": (
            by_block["1"]["teachers"]["quick"]
            ["posthoc_minimum_selection_margin_sensitivity"]["0.10"]
            ["audit_state_mean_student"]
            <= 0.0
        ),
    }
    result = {
        "schema_version": 1,
        "stage": "exploratory_route_diagnostics",
        "inferential_status": (
            "descriptive_only; does not alter the preregistered terminal gate"
        ),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "route": str(args.route.resolve()),
        "route_sha256": sha256_file(args.route),
        "route_gate": route["gate"],
        "by_block": by_block,
        "selection_audit_label_stability": assembled_by_block,
        "route_by_kind": {
            kind: {
                "states": kind_totals[kind],
                "routed": kind_routed[kind],
                "route_rate": kind_routed[kind] / kind_totals[kind],
            }
            for kind in sorted(kind_totals)
        },
        "selected_cell_overlap": cell_overlap,
        "cross_block_group_router_sensitivity": {
            granularity: {
                "audit_only_fit": _cross_block_group_router(
                    rows, fields, require_half_agreement=False
                ),
                "selection_audit_agreement_fit": _cross_block_group_router(
                    rows, fields, require_half_agreement=True
                ),
            }
            for granularity, fields in {
                "exact_cell": ("family", "kind", "level"),
                "family_kind": ("family", "kind"),
                "family": ("family",),
                "kind_level": ("kind", "level"),
            }.items()
        },
        "interpretive_flags": flags,
    }
    write_json(args.out, result)
    print(
        json.dumps(
            {
                "route_gate": result["route_gate"],
                "route_by_kind": result["route_by_kind"],
                "selected_cell_overlap": result["selected_cell_overlap"],
                "interpretive_flags": result["interpretive_flags"],
                "quick_block_1": result["by_block"]["1"]["teachers"]["quick"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
