#!/usr/bin/env python3
"""Gate the repaired agent interface before any scored qualification block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from analyze_qualification import EXP, load, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--unassisted", type=Path, required=True)
    parser.add_argument("--injected", type=Path, required=True)
    parser.add_argument("--control-search", type=Path, required=True)
    parser.add_argument("--explicit", type=Path, required=True)
    parser.add_argument("--answer-max-tokens", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    gates = cfg["qualification_gates"]
    unassisted = load(
        args.unassisted, cfg, "interface_preflight", "inferred", "acquisition",
        args.answer_max_tokens,
    )
    injected = load(
        args.injected, cfg, "interface_preflight", "inferred", "injected",
        args.answer_max_tokens,
    )
    control_search = load(
        args.control_search, cfg, "interface_preflight", "inferred", "random",
        args.answer_max_tokens,
    )
    explicit = load(
        args.explicit, cfg, "interface_preflight", "explicit", "acquisition",
        args.answer_max_tokens,
    )
    for comparator in (injected, control_search):
        for key in (
        "task_manifest_sha256", "task_content_manifest_sha256",
        "pair_static_manifest_sha256", "composed_mapping_manifest",
        ):
            if unassisted.get(key) != comparator.get(key):
                raise SystemExit(f"preflight inferred-arm mismatch at {key}")
    payloads = (unassisted, injected, control_search, explicit)
    checks = {
        "invalid_action_rate": all(
            row["aggregate"]["invalid_action_rate_per_turn"]
            <= float(gates["invalid_action_rate_per_turn_max"])
            for row in payloads
        ),
        "answer_limit_contact_rate": all(
            row["aggregate"]["answer_cap_hit_rate_per_turn"]
            <= float(gates["answer_cap_hit_rate_per_turn_max"])
            for row in payloads
        ),
        "fixed_search_exposes_evidence": (
            injected["aggregate"]["evidence_acquired_before_first_patch"] == 1.0
        ),
        "all_channels_present": (
            set(injected["aggregate"]["per_channel"])
            == set(gates["required_channels"])
        ),
        "all_query_skins_present": (
            set(injected["aggregate"]["per_query_skin"])
            == set(gates["required_qualification_query_skins"])
        ),
        "runner_identity_consistent": len({
            summary["runner_sha256"]
            for row in payloads for summary in row["runner_summaries"]
        }) == 1,
    }
    passed = all(checks.values())
    result = {
        "schema_version": 1,
        "stage": "interface_preflight",
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(args.config),
        "answer_max_tokens": args.answer_max_tokens,
        "checks": checks,
        "gate": {
            "passed": passed,
            "verdict": "INTERFACE_PASS" if passed else "INSTRUMENT_FAIL",
        },
        "qualification_authorized": passed,
        "training_authorized": False,
        "menagerie_authorized": False,
        "receipts": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in {
                "unassisted": args.unassisted,
                "injected": args.injected,
                "nondiscriminating_search": args.control_search,
                "explicit": args.explicit,
            }.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
