#!/usr/bin/env python3
"""Select non-outcome-peeking sample-pool prefixes at actual compute match points."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from downstream_common import (
    read_json,
    sha256_file,
    validate_behavior_receipt,
)

EXP = Path(__file__).resolve().parents[1]


def load(path: Path, expected_mode: str) -> dict:
    value = json.loads(path.read_text())
    if value.get("mode") != expected_mode:
        raise SystemExit(f"{path} is not mode={expected_mode}")
    return value


def first_prefix(costs: list[dict], target_sampled: int, target_logical: int) -> dict:
    sampled = 0
    logical = 0
    selected = []
    first_sampled = None
    first_logical = None
    first_dual = None
    for index, row in enumerate(sorted(costs, key=lambda item: item["trajectory"]), 1):
        sampled += int(row["sampled_tokens"])
        logical += int(row["logical_model_tokens"])
        selected.append(row)
        if first_sampled is None and sampled >= target_sampled:
            first_sampled = index
        if first_logical is None and logical >= target_logical:
            first_logical = index
        if first_dual is None and sampled >= target_sampled and logical >= target_logical:
            first_dual = index
    def summarize(k: int | None) -> dict | None:
        if k is None:
            return None
        prefix = selected[:k]
        return {
            "trajectories": k,
            "sampled_tokens": sum(int(row["sampled_tokens"]) for row in prefix),
            "logical_model_tokens": sum(
                int(row["logical_model_tokens"]) for row in prefix
            ),
            "sampled_margin": sum(
                int(row["sampled_tokens"]) for row in prefix
            ) - target_sampled,
            "logical_margin": sum(
                int(row["logical_model_tokens"]) for row in prefix
            ) - target_logical,
            "workspace_success": any(row["workspace_success"] for row in prefix),
            "preverifier_member_success": any(
                row["preverifier_member_success"] for row in prefix
            ),
        }
    under_k = max(0, (first_dual or 1) - 1)
    return {
        "target_sampled_tokens": target_sampled,
        "target_logical_model_tokens": target_logical,
        "sampled_overmatch": summarize(first_sampled),
        "logical_overmatch": summarize(first_logical),
        "dual_overmatch": summarize(first_dual),
        "dual_undermatch": summarize(under_k) if under_k else {
            "trajectories": 0,
            "sampled_tokens": 0,
            "logical_model_tokens": 0,
            "sampled_margin": -target_sampled,
            "logical_margin": -target_logical,
            "workspace_success": False,
            "preverifier_member_success": False,
        },
        "full_pool_oracle": summarize(len(selected)),
        "pool_exhausted": first_dual is None,
    }


def rate(rows: list[dict], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--target-deep", type=Path, required=True)
    parser.add_argument("--sample-pool", type=Path, required=True)
    parser.add_argument("--expected-target-arm", required=True)
    parser.add_argument("--expected-target-weight-sha256", required=True)
    parser.add_argument("--expected-pool-arm", required=True)
    parser.add_argument("--expected-pool-weight-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cfg["__analysis_config_path__"] = str(args.config.resolve())
    target_header = read_json(args.target_deep)
    block = target_header.get("block")
    if block not in ("transfer_dev", "transfer_confirm"):
        raise SystemExit(f"sample matcher is restricted to transfer blocks: {block}")
    target = validate_behavior_receipt(
        args.target_deep,
        cfg,
        block=block,
        contract="inferred",
        scenario_set="acquisition",
        mode="deep",
        arm=args.expected_target_arm,
        expected_weight_sha256=args.expected_target_weight_sha256,
        scaffold=False,
    )
    pool = validate_behavior_receipt(
        args.sample_pool,
        cfg,
        block=block,
        contract="inferred",
        scenario_set="acquisition",
        mode="sample_pool",
        arm=args.expected_pool_arm,
        expected_weight_sha256=args.expected_pool_weight_sha256,
        scaffold=False,
    )
    if (
        target.get("answer_max_tokens") != pool.get("answer_max_tokens")
        or target.get("think_budget") != pool.get("think_budget")
    ):
        raise SystemExit("target/pool answer or thinking allowance differs")
    identity_keys = (
        "block", "contract", "scenario_set", "task_manifest_sha256",
        "task_content_manifest_sha256", "pair_static_manifest_sha256",
        "history_policy",
    )
    for key in identity_keys:
        if target.get(key) != pool.get(key):
            raise SystemExit(f"target/pool mismatch at {key}")
    target_cases = {
        row["case_id"]: row for row in target["aggregate"]["cases"]
    }
    pool_cases = {row["case_id"]: row for row in pool["aggregate"]["cases"]}
    if set(target_cases) != set(pool_cases):
        raise SystemExit("target/pool case IDs differ")
    cases = []
    for case_id in sorted(target_cases):
        target_case = target_cases[case_id]
        pool_case = pool_cases[case_id]
        match = first_prefix(
            pool_case["trajectory_costs"],
            int(target_case["sampled_tokens"]),
            int(target_case["logical_model_tokens"]),
        )
        cases.append({
            "case_id": case_id,
            "pair_id": target_case["pair_id"],
            "branch": target_case["branch"],
            "family": target_case["family"],
            "scenario": target_case["scenario"],
            "match": match,
        })
    if any(row["match"]["pool_exhausted"] for row in cases):
        exhausted = [
            row["case_id"] for row in cases if row["match"]["pool_exhausted"]
        ]
        result = {
            "schema_version": 1,
            "status": "FAIL",
            "reason": "sample_pool_compute_infeasible",
            "analyzer_sha256": sha256_file(Path(__file__).resolve()),
            "target": str(args.target_deep.resolve()),
            "sample_pool": str(args.sample_pool.resolve()),
            "target_receipt_sha256": sha256_file(args.target_deep),
            "sample_pool_receipt_sha256": sha256_file(args.sample_pool),
            "target_arm": args.expected_target_arm,
            "pool_arm": args.expected_pool_arm,
            "target_model_weight_sha256": target["model_weight_sha256"],
            "pool_model_weight_sha256": pool["model_weight_sha256"],
            "exhausted_case_ids": exhausted,
            "cases": cases,
            "selection_policy": (
                "trajectory-index prefix; first prefix meeting both actual token "
                "budgets; outcomes never enter prefix selection"
            ),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({
            "status": result["status"],
            "reason": result["reason"],
            "exhausted_case_ids": exhausted,
        }, indent=2))
        return 4

    summaries = {}
    for match_name in (
        "sampled_overmatch", "logical_overmatch", "dual_overmatch",
        "dual_undermatch", "full_pool_oracle",
    ):
        member_rows = []
        for row in cases:
            value = row["match"][match_name]
            member_rows.append({
                **{key: row[key] for key in ("case_id", "pair_id", "branch", "family", "scenario")},
                **value,
            })
        by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in member_rows:
            by_pair[(row["pair_id"], row["scenario"])].append(row)
        dyads = []
        for (pair_id, scenario), members in sorted(by_pair.items()):
            if len(members) != 2 or {row["branch"] for row in members} != {0, 1}:
                raise SystemExit(f"incomplete dyad at {pair_id} {scenario}")
            dyads.append({
                "pair_id": pair_id,
                "scenario": scenario,
                "family": members[0]["family"],
                "paired_workspace_success": all(
                    row["workspace_success"] for row in members
                ),
                "paired_preverifier_success": all(
                    row["preverifier_member_success"] for row in members
                ),
            })
        summaries[match_name] = {
            "n_cases": len(member_rows),
            "n_dyads": len(dyads),
            "workspace_success": rate(member_rows, "workspace_success"),
            "preverifier_member_success": rate(
                member_rows, "preverifier_member_success"
            ),
            "paired_workspace_success": rate(dyads, "paired_workspace_success"),
            "paired_preverifier_success": rate(
                dyads, "paired_preverifier_success"
            ),
            "mean_trajectories": sum(row["trajectories"] for row in member_rows)
            / len(member_rows),
            "mean_sampled_margin": sum(row["sampled_margin"] for row in member_rows)
            / len(member_rows),
            "mean_logical_margin": sum(row["logical_margin"] for row in member_rows)
            / len(member_rows),
            "members": member_rows,
            "dyads": dyads,
        }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "selection_policy": (
            "trajectory-index prefix; first prefix meeting each actual token budget; "
            "outcomes never enter prefix selection"
        ),
        "target": str(args.target_deep.resolve()),
        "sample_pool": str(args.sample_pool.resolve()),
        "target_receipt_sha256": sha256_file(args.target_deep),
        "sample_pool_receipt_sha256": sha256_file(args.sample_pool),
        "target_arm": args.expected_target_arm,
        "pool_arm": args.expected_pool_arm,
        "target_model_weight_sha256": target["model_weight_sha256"],
        "pool_model_weight_sha256": pool["model_weight_sha256"],
        "primary_control": "dual_overmatch",
        "full_pool_is_oracle_only": True,
        "summaries": summaries,
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "primary_control": result["primary_control"],
        "summaries": {
            name: {key: value for key, value in row.items() if key not in ("members", "dyads")}
            for name, row in summaries.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
