#!/usr/bin/env python3
"""Apply frozen public-only selectors to paired action/candidate branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_public(candidate: dict, action: dict) -> str:
    if action["final_visible_pass"] and not candidate["final_visible_pass"]:
        return "action"
    return "candidate"


def choose_action_default(candidate: dict, action: dict) -> str:
    if candidate["final_visible_pass"] and not action["final_visible_pass"]:
        return "candidate"
    return "action"


def choose_random(case_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{case_id}".encode()).digest()
    return "candidate" if digest[0] % 2 == 0 else "action"


def selection_aggregate(cases: list[dict], selector: str, answer_cap: int) -> dict:
    selected = [case[case[selector]] for case in cases]
    success = [bool(row["workspace_success"]) for row in selected]
    total_turns = sum(int(row["turns"]) for row in selected)
    steps = [step for row in selected for step in row["steps"]]
    per_family = {}
    for family in sorted({row["family"] for row in selected}):
        subset = [row for row in selected if row["family"] == family]
        per_family[family] = {
            "n": len(subset),
            "success": sum(bool(row["workspace_success"]) for row in subset) / len(subset),
        }
    per_scenario = {}
    for scenario in ("failed_test", "rejected_patch"):
        subset = [row for row in selected if row["scenario"] == scenario]
        transition_key = (
            "failed_test_changed_patch_within_two"
            if scenario == "failed_test"
            else "rejected_patch_valid_changed_within_two"
        )
        per_scenario[scenario] = {
            "n": len(subset),
            "success": sum(bool(row["workspace_success"]) for row in subset) / len(subset),
            "changed_patch_within_two": sum(bool(row[transition_key]) for row in subset)
            / len(subset),
        }
    return {
        "n": len(selected),
        "success": sum(success) / len(success),
        "successes": sum(success),
        "verified_rate": sum(bool(row["final_visible_pass"]) for row in selected) / len(selected),
        "invalid_action_rate_per_turn": sum(int(row["invalid_actions"]) for row in selected)
        / max(total_turns, 1),
        "answer_cap_hit_rate_per_turn": sum(
            int(step["n_answer_tokens"]) >= answer_cap for step in steps
        )
        / max(total_turns, 1),
        "mean_selected_sampled_tokens": statistics.mean(
            int(row["sampled_tokens"]) for row in selected
        ),
        "per_family": per_family,
        "per_scenario": per_scenario,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    candidate_payload = load(args.candidate)
    action_payload = load(args.action)
    candidate = {row["case_id"]: row for row in candidate_payload["trajectories"]}
    action = {row["case_id"]: row for row in action_payload["trajectories"]}
    if candidate.keys() != action.keys():
        raise SystemExit("candidate/action case IDs differ")
    if candidate_payload["task_manifest_sha256"] != action_payload["task_manifest_sha256"]:
        raise SystemExit("candidate/action task manifests differ")
    if candidate_payload["block"] != action_payload["block"]:
        raise SystemExit("candidate/action blocks differ")

    random_seed = int(cfg["selector"]["random_control_seed"])
    cases = []
    selected_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for case_id in sorted(candidate):
        c_row = candidate[case_id]
        a_row = action[case_id]
        public_arm = choose_public(c_row, a_row)
        action_default_arm = choose_action_default(c_row, a_row)
        random_arm = choose_random(case_id, random_seed)
        selected_counts["public"][public_arm] += 1
        selected_counts["action_default"][action_default_arm] += 1
        selected_counts["random"][random_arm] += 1
        cases.append({
            "case_id": case_id,
            "family": c_row["family"],
            "scenario": c_row["scenario"],
            "candidate_public_pass": bool(c_row["final_visible_pass"]),
            "action_public_pass": bool(a_row["final_visible_pass"]),
            "candidate_success": bool(c_row["workspace_success"]),
            "action_success": bool(a_row["workspace_success"]),
            "public_arm": public_arm,
            "action_default_arm": action_default_arm,
            "random_arm": random_arm,
            "candidate": c_row,
            "action": a_row,
        })

    answer_cap = int(cfg["evaluation"]["answer_max_tokens"])
    public = selection_aggregate(cases, "public_arm", answer_cap)
    action_default = selection_aggregate(cases, "action_default_arm", answer_cap)
    random = selection_aggregate(cases, "random_arm", answer_cap)
    union_successes = sum(
        case["candidate_success"] or case["action_success"] for case in cases
    )
    expected_random_success = statistics.mean(
        (int(case["candidate_success"]) + int(case["action_success"])) / 2
        for case in cases
    )
    public_success_ids = {
        case["case_id"] for case in cases if case[case["public_arm"]]["workspace_success"]
    }
    union_success_ids = {
        case["case_id"]
        for case in cases
        if case["candidate_success"] or case["action_success"]
    }
    compact_cases = []
    for case in cases:
        compact_cases.append({key: value for key, value in case.items()
                              if key not in ("candidate", "action")})
    result = {
        "schema_version": 1,
        "block": candidate_payload["block"],
        "selector_rule": (
            "choose action iff action final_visible_pass and candidate not; "
            "otherwise choose candidate"
        ),
        "task_manifest_sha256": candidate_payload["task_manifest_sha256"],
        "reserved_sampled_tokens_per_case": (
            int(candidate_payload["reserved_sampled_tokens_per_case"])
            + int(action_payload["reserved_sampled_tokens_per_case"])
        ),
        "mean_total_branch_sampled_tokens": statistics.mean(
            int(case["candidate"]["sampled_tokens"]) + int(case["action"]["sampled_tokens"])
            for case in cases
        ),
        "selection_counts": {
            name: dict(sorted(counts.items())) for name, counts in selected_counts.items()
        },
        "public": public,
        "action_default": action_default,
        "random": random,
        "expected_random": {"success": expected_random_success},
        "oracle_union": {
            "success": union_successes / len(cases),
            "successes": union_successes,
            "public_capture": len(public_success_ids & union_success_ids)
            / max(len(union_success_ids), 1),
        },
        "cases": compact_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
