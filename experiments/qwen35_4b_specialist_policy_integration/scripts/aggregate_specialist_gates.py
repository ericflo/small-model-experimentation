#!/usr/bin/env python3
"""Aggregate four independent specialist qualification receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import write_json  # noqa: E402


DOMAINS = ("discover", "control", "tools", "compose")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=EXP / "analysis" / "specialists"
    )
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "specialist_gate.json"
    )
    args = parser.parse_args()
    receipts = {}
    for domain in DOMAINS:
        path = args.directory / f"{domain}.json"
        if not path.exists():
            raise SystemExit(f"missing specialist receipt: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("domain") != domain:
            raise SystemExit(f"domain mismatch in {path}")
        receipts[domain] = {
            "path": str(path.resolve()),
            "passed": bool(payload.get("gate", {}).get("passed")),
            "checks": payload.get("gate", {}).get("checks", {}),
        }
    passed_domains = [domain for domain in DOMAINS if receipts[domain]["passed"]]
    result = {
        "stage": "specialist_gate",
        "required_domains": list(DOMAINS),
        "passed_domains": passed_domains,
        "receipts": receipts,
        "gate": {
            "passed": len(passed_domains) == len(DOMAINS),
            "required_pass_count": len(DOMAINS),
            "observed_pass_count": len(passed_domains),
        },
        "downstream_authorization": (
            "teacher_audit" if len(passed_domains) == len(DOMAINS) else "stop_before_teacher_audit"
        ),
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
