#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.operator_tasks import FULL_OPERATOR_NAMES, OPERATORS, TEMPLATES, build_records  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-per-family", type=int, default=10)
    parser.add_argument("--visible-count", type=int, default=6)
    parser.add_argument("--hidden-count", type=int, default=18)
    parser.add_argument("--query-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "operator_inventory_eval.jsonl")
    args = parser.parse_args()

    rows = build_records(
        records_per_family=args.records_per_family,
        visible_count=args.visible_count,
        hidden_count=args.hidden_count,
        query_count=args.query_count,
        seed=args.seed,
    )
    write_jsonl(args.output, rows)

    manifest = {
        "experiment": "qwen35_4b_operator_inventory_search_pilot",
        "records": len(rows),
        "records_per_family": args.records_per_family,
        "visible_count": args.visible_count,
        "hidden_count": args.hidden_count,
        "query_count": args.query_count,
        "seed": args.seed,
        "data_file": str(args.output.relative_to(ROOT)),
        "operators": [op.__dict__ for op in OPERATORS],
        "closed_vocab_operators": ["sum", "first", "last"],
        "full_inventory_operators": FULL_OPERATOR_NAMES,
        "templates": [template.__dict__ for template in TEMPLATES],
        "by_operator_status": dict(sorted(Counter(row["operator_status"] for row in rows).items())),
        "by_template": dict(sorted(Counter(row["template"] for row in rows).items())),
        "purpose": "No-training pilot for type-colliding operator identification: closed-vocabulary search versus full inventory operator-hole search.",
    }
    manifest_path = ROOT / "data" / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

