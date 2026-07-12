#!/usr/bin/env python3
"""Assemble one fixed state block without interpreting its route effect."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, read_jsonl, sha256_file, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--quick", type=Path, required=True)
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    states = {row["state_id"]: row for row in read_jsonl(args.states)}
    if not states:
        raise SystemExit("empty state set")
    route = config["route"]
    selection_n = int(route["selection_branches_per_policy"])
    audit_n = int(route["audit_branches_per_policy"])
    total = selection_n + audit_n
    paths = {"quick": args.quick, "deep": args.deep, "student": args.student}
    branches = {}
    for policy, path in paths.items():
        rows = read_jsonl(path)
        index = {(row["state_id"], int(row["branch_index"])): row for row in rows}
        if len(index) != len(rows):
            raise SystemExit(f"duplicate {policy} branch identities")
        expected = {(state_id, branch) for state_id in states for branch in range(total)}
        if set(index) != expected:
            missing = sorted(expected - set(index))[:5]
            extra = sorted(set(index) - expected)[:5]
            raise SystemExit(f"{policy} branch mismatch missing={missing} extra={extra}")
        if any(row.get("policy") != policy for row in rows):
            raise SystemExit(f"{policy} file contains another policy tag")
        branches[policy] = index
    assembled = []
    for state_id, state in sorted(states.items()):
        row = {
            "state_id": state_id,
            "block": args.block,
            "family": state["family"],
            "kind": state["kind"],
            "level": int(state["level"]),
            "selection": {},
            "audit": {},
            "branch_ids": {},
        }
        for policy in ("quick", "deep", "student"):
            values = [branches[policy][(state_id, index)] for index in range(total)]
            row["selection"][policy] = [float(value["score"]) for value in values[:selection_n]]
            row["audit"][policy] = [float(value["score"]) for value in values[selection_n:]]
            row["branch_ids"][policy] = [
                f"{state_id}::{policy}::{index}" for index in range(total)
            ]
        assembled.append(row)
    payload = {
        "schema_version": 1,
        "stage": "assembled_route_block",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "block": args.block,
        "states": str(args.states.resolve()),
        "states_sha256": sha256_file(args.states),
        "branch_artifacts": {
            policy: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for policy, path in paths.items()
        },
        "selection_branch_indices": list(range(selection_n)),
        "audit_branch_indices": list(range(selection_n, total)),
        "rows": assembled,
    }
    write_json(args.out, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
