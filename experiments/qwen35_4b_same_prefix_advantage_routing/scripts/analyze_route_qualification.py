#!/usr/bin/env python3
"""Apply the frozen split-branch qualification rule to both complete blocks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from advantage_routing import analyze_route_blocks  # noqa: E402
from io_utils import load_config, sha256_file, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--block", action="append", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "route_qualification.json"
    )
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    if len(args.block) != 2:
        raise SystemExit("exactly two route blocks are required")
    rows = []
    manifests = []
    for expected_block, path in enumerate(args.block):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("stage") != "assembled_route_block":
            raise SystemExit(f"invalid route block artifact: {path}")
        if int(payload.get("block", -1)) != expected_block:
            raise SystemExit(f"route block order mismatch at {path}")
        if payload.get("config_sha256") != sha256_file(config_path):
            raise SystemExit(f"route block config mismatch: {path}")
        rows.extend(payload["rows"])
        manifests.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    route = config["route"]
    result = analyze_route_blocks(
        rows,
        selection_branches=int(route["selection_branches_per_policy"]),
        audit_branches=int(route["audit_branches_per_policy"]),
        minimum_per_teacher_per_block=int(
            route["minimum_routed_states_per_teacher_per_block"]
        ),
        bootstrap_samples=int(route["bootstrap_samples"]),
        confidence=float(route["confidence"]),
        bootstrap_seed=71701,
    )
    result.update(
        {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "block_artifacts": manifests,
            "selection_audit_independence": (
                "teacher chosen only by branch indices 0-3; inference only by 4-7"
            ),
        }
    )
    write_json(args.out, result)
    print(json.dumps({key: value for key, value in result.items() if key != "states"}, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())

