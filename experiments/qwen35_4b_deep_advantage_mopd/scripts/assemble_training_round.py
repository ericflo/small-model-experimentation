#!/usr/bin/env python3
"""Freeze one online round's routed capability and soup-anchor units."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from advantage_routing import select_teacher  # noqa: E402
from io_utils import load_config, read_jsonl, sha256_file, write_json  # noqa: E402


def _balanced_take(rows: list[dict], count: int) -> list[dict]:
    buckets: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["family"]), str(row["kind"]), int(row["level"]))].append(row)
    for values in buckets.values():
        values.sort(key=lambda value: value["state_id"])
    keys = sorted(key for key, values in buckets.items() if values)
    cursors = {key: 0 for key in keys}
    selected = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            cursor = cursors[key]
            if cursor >= len(buckets[key]):
                continue
            selected.append(buckets[key][cursor])
            cursors[key] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break
    return selected


def _match_key(row: dict, tier: str) -> tuple:
    values = {
        "exact_cell": (str(row["family"]), str(row["kind"]), int(row["level"])),
        "family_kind": (str(row["family"]), str(row["kind"])),
        "kind_level": (str(row["kind"]), int(row["level"])),
        "kind": (str(row["kind"]),),
    }
    if tier not in values:
        raise ValueError(f"unknown non-advantage-route matching tier: {tier}")
    return values[tier]


def _matched_non_advantage_route_units(
    selected: list[dict], candidates: list[dict], match_order: list[str]
) -> list[dict]:
    """Match non-deep-selected failed states to the exact primary geometry."""

    remaining = sorted(candidates, key=lambda row: str(row["state_id"]))
    matched = []
    for source in sorted(selected, key=lambda row: str(row["state_id"])):
        chosen_index = None
        chosen_tier = None
        for tier in match_order:
            source_key = _match_key(source, tier)
            chosen_index = next(
                (
                    index
                    for index, candidate in enumerate(remaining)
                    if _match_key(candidate, tier) == source_key
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


def _best_offpolicy(branches: list[dict]) -> dict:
    best = max(branches, key=lambda row: (float(row["score"]), -int(row["branch_index"])))
    if best["kind"] == "atom":
        completion = [int(value) for value in best["output"]["token_ids"]]
        injected = [int(value) for value in best["output"].get("injected_token_ids") or []]
    else:
        if not best.get("turns"):
            raise ValueError(f"episode branch {best['state_id']} has no action turns")
        completion = [int(value) for value in best["turns"][0]["token_ids"]]
        injected = [
            int(value) for value in best["turns"][0].get("injected_token_ids") or []
        ]
    return {
        "policy": best["policy"],
        "branch_index": int(best["branch_index"]),
        "terminal_score": float(best["score"]),
        "completion_ids": completion,
        "injected_token_ids": injected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--states", action="append", type=Path, required=True)
    parser.add_argument("--anchors", action="append", type=Path, required=True)
    parser.add_argument("--quick", action="append", type=Path, required=True)
    parser.add_argument("--deep", action="append", type=Path, required=True)
    parser.add_argument("--student", action="append", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    count = len(args.states)
    if not (len(args.anchors) == len(args.quick) == len(args.deep) == len(args.student) == count):
        raise SystemExit("training batch artifact lists must have identical lengths")
    maximum = int(config["mopd"]["candidate_batches_per_round_maximum"])
    if not 1 <= count <= maximum:
        raise SystemExit(f"candidate batch count must be in [1, {maximum}]")
    if not 0 <= args.round < int(config["mopd"]["rounds"]):
        raise SystemExit("invalid round")
    selection_n = int(config["route"]["selection_branches_per_policy"])
    state_map: dict[str, dict] = {}
    anchor_map: dict[str, dict] = {}
    artifacts = []
    branch_maps: dict[str, dict[tuple[str, int], dict]] = {
        "quick": {}, "deep": {}, "student": {}
    }
    for batch_index in range(count):
        states = read_jsonl(args.states[batch_index])
        anchors = read_jsonl(args.anchors[batch_index])
        for row in states:
            if row["state_id"] in state_map or row["state_id"] in anchor_map:
                raise SystemExit(f"duplicate state across candidate batches: {row['state_id']}")
            state_map[row["state_id"]] = row
        for row in anchors:
            if row["state_id"] in state_map or row["state_id"] in anchor_map:
                raise SystemExit(f"duplicate anchor across candidate batches: {row['state_id']}")
            anchor_map[row["state_id"]] = row
        record = {
            "batch": batch_index,
            "states": {"path": str(args.states[batch_index].resolve()), "sha256": sha256_file(args.states[batch_index])},
            "anchors": {"path": str(args.anchors[batch_index].resolve()), "sha256": sha256_file(args.anchors[batch_index])},
            "branches": {},
        }
        for policy, paths in (("quick", args.quick), ("deep", args.deep), ("student", args.student)):
            rows = read_jsonl(paths[batch_index])
            expected = {(state["state_id"], branch) for state in states for branch in range(selection_n)}
            observed = {(row["state_id"], int(row["branch_index"])) for row in rows}
            if observed != expected or len(observed) != len(rows):
                raise SystemExit(f"{policy} branch mismatch in batch {batch_index}")
            if any(row.get("policy") != policy for row in rows):
                raise SystemExit(f"{policy} branch file has wrong policy tag")
            branch_maps[policy].update(
                {(row["state_id"], int(row["branch_index"])): row for row in rows}
            )
            record["branches"][policy] = {
                "path": str(paths[batch_index].resolve()), "sha256": sha256_file(paths[batch_index])
            }
        artifacts.append(record)

    routed: dict[str, list[dict]] = {"quick": [], "deep": []}
    abstained = []
    for state_id, state in sorted(state_map.items()):
        scores = {
            policy: [
                float(branch_maps[policy][(state_id, branch)]["score"])
                for branch in range(selection_n)
            ]
            for policy in ("quick", "deep", "student")
        }
        teacher = select_teacher(scores)
        row = {
            "state_id": state_id,
            "family": state["family"],
            "kind": state["kind"],
            "level": int(state["level"]),
            "state": state,
            "selection_scores": scores,
            "selection_means": {
                policy: sum(values) / len(values) for policy, values in scores.items()
            },
            "primary_teacher": teacher,
        }
        if teacher is None:
            abstained.append(row)
        else:
            teacher_branches = [
                branch_maps[teacher][(state_id, branch)] for branch in range(selection_n)
            ]
            row["offpolicy_target"] = _best_offpolicy(teacher_branches)
            routed[teacher].append(row)

    micro_steps = int(config["mopd"]["updates_per_round"]) * int(config["mopd"]["grad_accum"])
    capability_n = int(config["mopd"]["capability_units_per_round"])
    anchor_n = int(config["mopd"]["anchor_units_per_round"])
    if capability_n + anchor_n != micro_steps:
        raise SystemExit("deep capability/anchor quotas do not match micro-step count")
    supply = {teacher: len(rows) for teacher, rows in routed.items()}
    supply["anchors"] = len(anchor_map)
    supply["non_deep_failed"] = len(routed["quick"]) + len(abstained)
    enough = (
        supply["deep"] >= capability_n
        and supply["anchors"] >= anchor_n
        and supply["non_deep_failed"] >= capability_n
    )
    if not enough:
        status = {
            "status": "insufficient_route_supply",
            "round": args.round,
            "candidate_batches": count,
            "required": {
                "deep": capability_n,
                "anchors": anchor_n,
                "non_deep_failed": capability_n,
            },
            "available": supply,
            "may_add_batch": count < maximum,
        }
        write_json(args.out.with_suffix(args.out.suffix + ".supply.json"), status)
        print(json.dumps(status, indent=2))
        return 3
    capability_units = []
    for row in _balanced_take(routed["deep"], capability_n):
        row = dict(row)
        row["role"] = "capability"
        capability_units.append(row)
    try:
        control_units = _matched_non_advantage_route_units(
            capability_units,
            [*routed["quick"], *abstained],
            [str(value) for value in config["controls"]["non_advantage_route_match_order"]],
        )
    except ValueError as error:
        status = {
            "status": "insufficient_non_advantage_route_match_supply",
            "round": args.round,
            "candidate_batches": count,
            "required": {"matched_non_deep_failed": capability_n},
            "available": supply,
            "match_order": list(config["controls"]["non_advantage_route_match_order"]),
            "error": str(error),
            "may_add_batch": count < maximum,
        }
        write_json(args.out.with_suffix(args.out.suffix + ".supply.json"), status)
        print(json.dumps(status, indent=2))
        return 3
    anchor_rows = [
        {
            "state_id": state_id,
            "family": state["family"],
            "kind": state["kind"],
            "level": int(state["level"]),
            "state": state,
            "role": "anchor",
            "primary_teacher": "soup",
            "selection_scores": None,
            "selection_means": None,
            "offpolicy_target": None,
        }
        for state_id, state in sorted(anchor_map.items())
    ]
    anchor_units = _balanced_take(anchor_rows, anchor_n)
    units = sorted(capability_units + anchor_units, key=lambda row: row["state_id"])
    if len(units) != micro_steps or len({row["state_id"] for row in units}) != micro_steps:
        raise SystemExit("training unit count or consume-once identity failed")
    if len(control_units) != capability_n:
        raise SystemExit("non-advantage-route control count failed")
    all_ids = [row["state_id"] for row in [*units, *control_units]]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("primary and non-advantage-route control states overlap")
    payload = {
        "schema_version": 2,
        "stage": "online_advantage_training_round",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "round": args.round,
        "candidate_batches": count,
        "artifacts": artifacts,
        "candidate_counts": {
            "failed_states": len(state_map),
            "successful_anchors": len(anchor_map),
            "quick_routed": len(routed["quick"]),
            "deep_routed": len(routed["deep"]),
            "abstained": len(abstained),
        },
        "unit_counts": {
            "total": len(units),
            "deep": sum(row["primary_teacher"] == "deep" for row in units),
            "soup_anchor": sum(row["primary_teacher"] == "soup" for row in units),
            "non_advantage_route_control": len(control_units),
            "non_advantage_route_match_tiers": dict(
                sorted(
                    (
                        tier,
                        sum(row["match_tier"] == tier for row in control_units),
                    )
                    for tier in config["controls"]["non_advantage_route_match_order"]
                )
            ),
        },
        "units": units,
        "control_units": sorted(control_units, key=lambda row: row["state_id"]),
    }
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"units", "control_units"}
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
