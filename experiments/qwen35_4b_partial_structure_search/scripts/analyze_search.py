#!/usr/bin/env python3
"""Paired depth-5 search analysis and preregistered frontier-advance verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
import experiment_common as C  # noqa: E402
import run_search as R  # noqa: E402
import stats as S  # noqa: E402


METHODS = (
    "thinking",
    "thinking_shuffled",
    "nothink",
    "nextop",
    "uniform_seeded",
    "surface",
    "budget_truncated_brute",
    "oracle_live",
    "direct_sample_more",
    "direct_sample_more_total",
)


def _load(method: str, suffix: str) -> dict[str, Any]:
    path = EXP / "runs" / f"search_{method}{suffix}.json"
    if not path.exists():
        raise RuntimeError(f"missing search arm: {path}")
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_arm_integrity(
    method: str,
    result: dict[str, Any],
    *,
    canonical_ids: list[str],
    receipt: dict[str, Any],
    expected_beam: int,
    expected_fill_cap: int,
) -> None:
    if result.get("method") != method:
        raise RuntimeError(f"{method}: declared method mismatch")
    if result.get("input_sha256") != receipt.get("input_sha256"):
        raise RuntimeError(f"{method}: input fingerprints do not match completion receipt")
    if result.get("task_ids") != canonical_ids:
        raise RuntimeError(f"{method}: frozen task order is missing or stale")
    settings = result.get("settings", {})
    if settings != receipt.get("settings"):
        raise RuntimeError(f"{method}: settings do not match completion receipt")
    if int(settings.get("parameter_fill_cap_per_task", -1)) != expected_fill_cap:
        raise RuntimeError(f"{method}: fill-cap setting mismatch")
    rows = result.get("rows", [])
    row_ids = [str(row.get("task_id")) for row in rows]
    if row_ids != canonical_ids or len(set(row_ids)) != len(canonical_ids):
        raise RuntimeError(f"{method}: rows are not an exact ordered one-to-one task match")
    for index, row in enumerate(rows):
        if int(row.get("task_index", -1)) != index or int(row.get("shard", -1)) != index % 2:
            raise RuntimeError(f"{method}: task index/shard assignment changed at row {index}")
        if row.get("method") != method:
            raise RuntimeError(f"{method}: row method mismatch at {canonical_ids[index]}")
        if int(row.get("fill_cap", -1)) != expected_fill_cap:
            raise RuntimeError(f"{method}: row fill cap mismatch at {canonical_ids[index]}")
    beam_methods = {
        "thinking",
        "thinking_shuffled",
        "nothink",
        "nextop",
        "uniform_seeded",
        "surface",
        "oracle_live",
    }
    if method in beam_methods and int(result.get("beam_width", -1)) != expected_beam:
        raise RuntimeError(f"{method}: beam-width mismatch")
    expected_basis = {
        "direct_sample_more": "sampled_tokens",
        "direct_sample_more_total": "total_model_tokens",
    }.get(method)
    if expected_basis is not None:
        if result.get("match_basis") != expected_basis:
            raise RuntimeError(f"{method}: match basis must be {expected_basis}")
        for row in rows:
            if row.get("match_basis") != expected_basis:
                raise RuntimeError(f"{method}: row-level match basis mismatch")


def _bootstrap_difference(
    left: dict[str, bool], right: dict[str, bool], reps: int, seed: int
) -> dict[str, Any]:
    values = {task: float(left[task]) - float(right[task]) for task in sorted(set(left) & set(right))}
    return S.cluster_bootstrap(values, lambda xs: sum(xs) / len(xs), reps=reps, seed=seed)


def _arm_summary(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["rows"]
    n = len(rows)
    summary = {
        "n": n,
        "selected_hidden_success": sum(bool(row["selected_hidden_success"]) for row in rows) / n,
        "pool_hidden_coverage": sum(bool(row["pool_hidden_coverage"]) for row in rows) / n,
        "mean_completed_skeletons": sum(int(row["completed_skeletons"]) for row in rows) / n,
        "mean_fill_cap_used": sum(int(row["fill_cap_used"]) for row in rows) / n,
        "mean_visible_candidates": sum(int(row["visible_passing_concrete_candidates"]) for row in rows) / n,
        "by_shard": {},
    }
    if all("expanded_prefix_nodes" in row for row in rows):
        summary["mean_expanded_prefix_nodes"] = sum(
            int(row["expanded_prefix_nodes"]) for row in rows
        ) / n
    elif all("layers" in row for row in rows):
        summary["mean_expanded_prefix_nodes"] = sum(
            sum(int(layer["expanded"]) for layer in row["layers"]) for row in rows
        ) / n
    if "wall_seconds" in result:
        summary["arm_wall_seconds"] = float(result["wall_seconds"])
    for shard in (0, 1):
        shard_rows = [row for row in rows if int(row["shard"]) == shard]
        summary["by_shard"][str(shard)] = {
            "n": len(shard_rows),
            "selected_hidden_success": sum(bool(row["selected_hidden_success"]) for row in shard_rows)
            / len(shard_rows),
        }
    if all("model_accounting" in row for row in rows):
        summary["mean_sampled_model_tokens"] = sum(
            int(row["model_accounting"].get("sampled_tokens", 0)) for row in rows
        ) / n
        summary["mean_prefill_model_tokens"] = sum(
            int(row["model_accounting"].get("prefill_tokens", 0)) for row in rows
        ) / n
        summary["mean_model_requests"] = sum(
            int(row["model_accounting"].get("requests", 0)) for row in rows
        ) / n
        summary["mean_model_completions"] = sum(
            int(row["model_accounting"].get("completions", 0)) for row in rows
        ) / n
        summary["mean_total_model_tokens"] = sum(
            int(row["model_accounting"].get("total_model_tokens", 0)) for row in rows
        ) / n
    if result["method"] in {"direct_sample_more", "direct_sample_more_total"}:
        summary["match_basis"] = result["match_basis"]
        summary["mean_matched_tokens"] = sum(int(row["matched_tokens"]) for row in rows) / n
        summary["mean_sampled_model_tokens"] = sum(
            int(row["retained_model_accounting"]["sampled_tokens"]) for row in rows
        ) / n
        summary["mean_prefill_model_tokens"] = sum(
            int(row["retained_model_accounting"]["prefill_tokens"]) for row in rows
        ) / n
        summary["mean_total_model_tokens"] = sum(
            int(row["retained_model_accounting"]["total_model_tokens"]) for row in rows
        ) / n
        summary["mean_model_requests"] = sum(
            int(row["retained_model_accounting"]["requests"]) for row in rows
        ) / n
        summary["gross_pool_mean_sampled_tokens"] = sum(
            int(row["gross_pool_model_accounting"]["sampled_tokens"]) for row in rows
        ) / n
        summary["gross_pool_mean_prefill_tokens"] = sum(
            int(row["gross_pool_model_accounting"]["prefill_tokens"]) for row in rows
        ) / n
        summary["gross_pool_mean_total_model_tokens"] = sum(
            int(row["gross_pool_model_accounting"]["total_model_tokens"])
            for row in rows
        ) / n
        summary["gross_pool_mean_requests"] = sum(
            int(row["gross_pool_model_accounting"]["requests"]) for row in rows
        ) / n
        summary["mean_matched_completions"] = sum(int(row["matched_completions"]) for row in rows) / n
        summary["mean_parse_rate"] = sum(float(row["parse_rate"]) for row in rows) / n
        summary["pool_exhaustion_count"] = sum(
            bool(row["pool_exhausted_before_cap"]) for row in rows
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    suffix = "_smoke" if args.smoke else ""
    cfg = C.load_config()
    reps = 500 if args.smoke else int(cfg["search"]["bootstrap_reps"])
    receipt_path = EXP / "runs" / f"search_model_receipt{suffix}.json"
    if not receipt_path.exists():
        raise RuntimeError("validated search completion receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != 2:
        raise RuntimeError("search completion receipt schema is not current")
    current_inputs = R._fingerprints(suffix)
    if receipt.get("input_sha256") != current_inputs:
        raise RuntimeError("search inputs changed after result generation")
    if not R._valid_complete_cache(suffix, current_inputs):
        raise RuntimeError("search completion cache is not valid")
    task_path = EXP / "data" / f"primary_tasks{suffix}.jsonl"
    canonical_ids = [str(row["task_id"]) for row in C.load_jsonl(task_path)]
    expected_beam = 2 if args.smoke else int(cfg["search"]["beam_width"])
    expected_fill_cap = (
        256 if args.smoke else int(cfg["search"]["parameter_fill_cap_per_task"])
    )
    results = {method: _load(method, suffix) for method in METHODS}
    for method, result in results.items():
        result_path = EXP / "runs" / f"search_{method}{suffix}.json"
        if receipt.get("result_sha256", {}).get(method) != _sha256(result_path):
            raise RuntimeError(f"{method}: result hash does not match completion receipt")
        _validate_arm_integrity(
            method,
            result,
            canonical_ids=canonical_ids,
            receipt=receipt,
            expected_beam=expected_beam,
            expected_fill_cap=expected_fill_cap,
        )
    summaries = {method: _arm_summary(result) for method, result in results.items()}
    outcomes = {
        method: {str(row["task_id"]): bool(row["selected_hidden_success"]) for row in result["rows"]}
        for method, result in results.items()
    }
    comparisons = {}
    for index, baseline in enumerate(method for method in METHODS if method != "thinking"):
        comparisons[baseline] = {
            "risk_difference": _bootstrap_difference(
                outcomes["thinking"], outcomes[baseline], reps, 200 + index
            ),
            "mcnemar": S.mcnemar_exact(
                [outcomes["thinking"][task] for task in sorted(outcomes["thinking"])],
                [outcomes[baseline][task] for task in sorted(outcomes["thinking"])],
            ),
        }
    direct_differences = {
        method: comparisons[method]["risk_difference"]
        for method in ("direct_sample_more", "direct_sample_more_total")
    }
    shuffled_diff = comparisons["thinking_shuffled"]["risk_difference"]
    nextop_diff = comparisons["nextop"]["risk_difference"]
    shard_directions = {
        method: {
            shard: (
                summaries["thinking"]["by_shard"][shard]["selected_hidden_success"]
                - summaries[method]["by_shard"][shard]["selected_hidden_success"]
            )
            for shard in ("0", "1")
        }
        for method in ("direct_sample_more", "direct_sample_more_total")
    }
    token_parity_violations = []
    think_rows = {str(row["task_id"]): row for row in results["thinking"]["rows"]}
    for method in ("direct_sample_more", "direct_sample_more_total"):
        basis = str(results[method]["match_basis"])
        expected_pool_k = 256 if args.smoke else int(cfg["search"]["direct_sample_pool_k"])
        if int(results[method].get("pool_k", -1)) != expected_pool_k:
            raise RuntimeError(f"{method}: frozen pool size changed")
        for row in results[method]["rows"]:
            task_id = str(row["task_id"])
            cap = int(think_rows[task_id]["model_accounting"][basis])
            spent = int(row["matched_tokens"])
            next_cost = row["next_excluded_sample_cost"]
            reasons = []
            if int(row["token_cap_from_thinking_search"]) != cap:
                reasons.append("serialized_cap_mismatch")
            if int(row.get("sample_pool_k", -1)) != expected_pool_k:
                reasons.append("row_pool_k_mismatch")
            if int(row.get("pool_completions", -1)) != expected_pool_k:
                reasons.append("incomplete_pool")
            if spent > cap:
                reasons.append("overspent")
            if int(row.get("unspent_token_budget", -1)) != cap - spent:
                reasons.append("slack_mismatch")
            if int(row["retained_model_accounting"][basis]) != spent:
                reasons.append("retained_accounting_mismatch")
            if int(row.get("pool_capacity_tokens", -1)) != int(
                row["gross_pool_model_accounting"][basis]
            ):
                reasons.append("pool_capacity_mismatch")
            if bool(row["pool_exhausted_before_cap"]):
                reasons.append("pool_exhausted")
            if next_cost is not None and spent + int(next_cost) <= cap:
                reasons.append("nonmaximal_prefix")
            if reasons:
                token_parity_violations.append(
                    {"method": method, "task_id": task_id, "reasons": reasons}
                )
    sampled_direct = {
        str(row["task_id"]): row for row in results["direct_sample_more"]["rows"]
    }
    for row in results["direct_sample_more_total"]["rows"]:
        task_id = str(row["task_id"])
        if row["gross_pool_model_accounting"] != sampled_direct[task_id][
            "gross_pool_model_accounting"
        ]:
            token_parity_violations.append(
                {
                    "method": "direct_pool_cross_arm",
                    "task_id": task_id,
                    "reasons": ["gross_pool_accounting_mismatch"],
                }
            )
    shuffled_token_differences = {
        str(row["task_id"]): int(think_rows[str(row["task_id"])]["model_accounting"]["sampled_tokens"])
        - int(row["model_accounting"]["sampled_tokens"])
        for row in results["thinking_shuffled"]["rows"]
    }
    threshold = float(cfg["search"]["meaningful_accuracy_lift"])
    direct_gate = all(
        direct_differences[method]["estimate"] >= threshold
        and direct_differences[method]["ci_low"] > 0
        and all(value > 0 for value in shard_directions[method].values())
        for method in direct_differences
    )
    passed = bool(
        not token_parity_violations
        and direct_gate
        and shuffled_diff["estimate"] > 0
        and nextop_diff["estimate"] > 0
    )
    if passed:
        classification = "G4_frontier_advance"
    elif token_parity_violations:
        classification = "invalid_compute_match"
    elif shuffled_diff["estimate"] > 0:
        classification = "G3_model_contribution_without_frontier_advance"
    else:
        classification = "G2_recognition_signal_not_actionable_in_search"
    full_brute_path = EXP / "runs" / f"full_brute{suffix}.json"
    if not full_brute_path.exists():
        raise RuntimeError("mandatory full-brute reference is missing")
    full_brute = json.loads(full_brute_path.read_text())
    if not (
        full_brute.get("exact") is True
        and int(full_brute.get("task_count", -1)) == len(canonical_ids)
        and full_brute.get("task_source_sha256") == _sha256(task_path)
    ):
        raise RuntimeError("full-brute receipt is non-exact or stale for canonical tasks")
    verdict = {
        "schema_version": 2,
        "gate": "depth5_frontier_advance",
        "passed": passed,
        "classification": classification,
        "n_tasks": len(outcomes["thinking"]),
        "meaningful_lift_threshold": threshold,
        "arm_summaries": summaries,
        "thinking_vs_baselines": comparisons,
        "direct_matched_differences": direct_differences,
        "shard_directions_vs_direct": shard_directions,
        "sampled_token_parity_violations": token_parity_violations,
        "thinking_minus_shuffled_sampled_tokens_by_task": shuffled_token_differences,
        "search_receipt_sha256": _sha256(receipt_path),
        "full_brute": {
            "path_coverage_rate": full_brute["path_coverage_rate"],
            "selected_hidden_success_rate": full_brute["selected_hidden_success_rate"],
            "parallel_wall_seconds": full_brute["parallel_wall_seconds"],
            "logical_type_skeleton_leaves": full_brute["logical_accounting"]["type_skeleton_leaves"],
            "aggregate_accounting": full_brute["aggregate_accounting"],
        },
    }
    out = EXP / "runs" / f"search_verdict{suffix}.json"
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n")
    temporary.replace(out)
    print(
        json.dumps(
            {
                "passed": passed,
                "classification": classification,
                "thinking_accuracy": summaries["thinking"]["selected_hidden_success"],
                "direct_accuracy": summaries["direct_sample_more"]["selected_hidden_success"],
                "direct_differences": direct_differences,
                "shuffle_difference": shuffled_diff,
                "nextop_difference": nextop_diff,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
