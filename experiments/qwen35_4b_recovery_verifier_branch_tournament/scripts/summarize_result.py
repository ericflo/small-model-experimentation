#!/usr/bin/env python3
"""Build the compact committed receipt for the stopped prospective tournament."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
ARTIFACTS = ROOT / "large_artifacts" / EXP.name / "eval"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(payload: dict) -> dict:
    aggregate = payload["aggregate"]
    return {
        "success": aggregate["success"],
        "submit_rate": aggregate["submit_rate"],
        "invalid_action_rate_per_turn": aggregate["invalid_action_rate_per_turn"],
        "answer_cap_hit_rate_per_turn": aggregate["answer_cap_hit_rate_per_turn"],
        "mean_sampled_tokens": aggregate["mean_sampled_tokens"],
        "mean_turns": aggregate["mean_turns"],
        "failed_test_success": aggregate["per_scenario"]["failed_test"]["success"],
        "failed_test_changed_patch_within_two": aggregate["per_scenario"]["failed_test"][
            "changed_patch_within_two"
        ],
        "rejected_patch_success": aggregate["per_scenario"]["rejected_patch"]["success"],
        "rejected_patch_changed_patch_within_two": aggregate["per_scenario"][
            "rejected_patch"
        ]["valid_changed_patch_within_two"],
        "per_family": aggregate["per_family"],
    }


def paired_forensics(candidate: dict, action: dict) -> dict:
    c = {row["case_id"]: row for row in candidate["trajectories"]}
    a = {row["case_id"]: row for row in action["trajectories"]}
    if c.keys() != a.keys():
        raise ValueError("candidate/action case IDs differ")
    counts: Counter[str] = Counter()
    union = 0
    shared_fail = 0
    shared_fail_families: Counter[str] = Counter()
    for case_id in c:
        c_success = bool(c[case_id]["workspace_success"])
        a_success = bool(a[case_id]["workspace_success"])
        union += c_success or a_success
        if c_success and not a_success:
            counts["candidate_only"] += 1
        elif a_success and not c_success:
            counts["action_only"] += 1
        elif c_success and a_success:
            counts["both_success"] += 1
        else:
            shared_fail += 1
            shared_fail_families[c[case_id]["family"]] += 1
    return {
        **dict(counts),
        "shared_fail": shared_fail,
        "shared_fail_by_family": dict(sorted(shared_fail_families.items())),
        "union_success": union / len(c),
    }


def compact_retrospective(payload: dict) -> dict:
    return {
        "block": payload["block"],
        "public_success": payload["public"]["success"],
        "expected_random_success": payload["expected_random"]["success"],
        "oracle_union_success": payload["oracle_union"]["success"],
        "oracle_union_capture": payload["oracle_union"]["public_capture"],
        "selection_counts": payload["selection_counts"]["public"],
    }


def main() -> int:
    paths = {
        "base_deep": ARTIFACTS / "prospective_dev_recovery_base_deep.json",
        "candidate_deep": ARTIFACTS / "prospective_dev_recovery_candidate_deep.json",
        "action_deep": ARTIFACTS / "prospective_dev_recovery_action_deep.json",
        "candidate_sample_more": ARTIFACTS
        / "prospective_dev_recovery_candidate_sample_more.json",
        "action_sample_more": ARTIFACTS
        / "prospective_dev_recovery_action_sample_more.json",
    }
    payloads = {name: load(path) for name, path in paths.items()}
    feasibility_path = EXP / "analysis" / "prospective_dev_feasibility.json"
    feasibility = load(feasibility_path)
    paired = paired_forensics(payloads["candidate_deep"], payloads["action_deep"])
    result = {
        "schema_version": 1,
        "experiment": EXP.name,
        "verdict": "PROSPECTIVE_DEV_INFEASIBLE",
        "stopped_at": "prospective_dev_feasibility",
        "selector_scored_prospectively": False,
        "confirmation_exposed": False,
        "winner_bank_authorized": False,
        "menagerie_exposed": False,
        "headline": {
            "candidate_success": payloads["candidate_deep"]["aggregate"]["success"],
            "action_success": payloads["action_deep"]["aggregate"]["success"],
            "mixed_policy_union_ceiling": paired["union_success"],
            "best_sample_more": max(
                payloads["candidate_sample_more"]["aggregate"]["success"],
                payloads["action_sample_more"]["aggregate"]["success"],
            ),
            "shared_failure_cases": paired["shared_fail"],
        },
        "prospective_dev": {name: compact(payload) for name, payload in payloads.items()},
        "paired_source_forensics": paired,
        "feasibility": feasibility,
        "retrospective_qualification": {
            name: compact_retrospective(
                load(EXP / "analysis" / f"selector_{name}.json")
            )
            for name in ("transfer_dev", "transfer_confirm")
        },
        "source_sha256": {
            "evaluations": {name: sha256(path) for name, path in paths.items()},
            "feasibility": sha256(feasibility_path),
        },
    }
    output = EXP / "reports" / "result_receipt.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "verdict": result["verdict"],
        "headline": result["headline"],
        "paired": paired,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
