#!/usr/bin/env python3
"""Apply the preregistered synthetic installability gate to a local-eval receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ABSTENTION_ANSWERS = {
    "", "ABSTAIN", "INSUFFICIENT", "N/A", "NO ANSWER", "NONE", "NULL",
    "UNDECIDABLE", "UNKNOWN",
}


def is_abstention(value: object) -> bool:
    return value is None or str(value).strip().upper() in ABSTENTION_ANSWERS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--candidate", default="warm_union")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text(encoding="utf-8"))
    summaries = payload.get("summaries", {})
    if args.candidate not in summaries:
        parser.error("candidate is absent from local receipt")
    candidate = summaries[args.candidate]
    rows = [row for row in payload.get("rows", []) if row.get("adapter") == args.candidate]
    route_abstentions = sum(
        row.get("kind") == "u_route" and is_abstention(row.get("parsed"))
        for row in rows
    )
    checks = {
        "parse_rate_at_least_0_90": candidate.get("parse_rate", 0.0) >= 0.90,
        "accuracy_at_least_0_65": candidate.get("accuracy", 0.0) >= 0.65,
        "cap_contacts_at_most_2": candidate.get("cap_contacts", 10**9) <= 2,
        "no_repeated_feasible_route_abstention": route_abstentions < 2,
    }
    result = {
        "schema_version": 1,
        "candidate": args.candidate,
        "seed": payload.get("seed"),
        "mix": payload.get("mix"),
        "summary": candidate,
        "route_abstentions": route_abstentions,
        "checks": checks,
        "passes": all(checks.values()),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        if args.out.exists():
            parser.error("refusing to overwrite local gate receipt")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
