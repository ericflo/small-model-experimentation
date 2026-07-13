#!/usr/bin/env python3
"""Identity-only scaffold smoke; no model import, load, or call."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SUMMARY = EXP / "runs" / "scaffold" / "summary.json"


def receipt() -> dict[str, object]:
    return {
        "benchmark_files_read": [],
        "decision": "SCAFFOLD_IDENTITY_RESERVED_NO_MODEL_AUTHORIZATION",
        "experiment_id": (
            "qwen35_4b_materialized_residual_sibling_search_fresh_replication"
        ),
        "fresh_sampling_seed_domains": 1,
        "fresh_seed_block": list(range(2026072700, 2026072710)),
        "fresh_task_seed_domains": 1,
        "model_calls": 0,
        "model_loads": 0,
        "parent_experiment": "qwen35_4b_materialized_residual_sibling_search",
        "parent_incident_sha256": (
            "48ae3f49addc43b435fd5c2b121d57b9498223488a5251999910f56c50e9d4d2"
        ),
        "parent_terminal_started_sha256": (
            "f6aa447b1936fac397a353fc13183f008e31884b5006ed7fc50ac78deed3387a"
        ),
        "request_identity_namespace": "materialized-residual-fresh-replication-v1",
        "schema_version": 1,
    }


def run_smoke() -> None:
    expected = receipt()
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    if SUMMARY.exists():
        observed = json.loads(SUMMARY.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError("existing scaffold identity receipt changed")
    else:
        with SUMMARY.open("x", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(expected, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="write/verify the identity receipt"
    )
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
        return 0
    parser.error(
        "full execution is sealed until fresh construction, adversarial review, "
        "implementation publication, and a separate lock"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
