#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.shifted_tasks import build_records  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-per-family", type=int, default=12)
    parser.add_argument("--visible-count", type=int, default=6)
    parser.add_argument("--hidden-count", type=int, default=18)
    parser.add_argument("--query-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "shifted_coverage_eval.jsonl")
    args = parser.parse_args()

    records = build_records(
        records_per_family=args.records_per_family,
        visible_count=args.visible_count,
        hidden_count=args.hidden_count,
        query_count=args.query_count,
        seed=args.seed,
    )
    write_jsonl(args.output, records)

    by_shift = Counter(row["shift_type"] for row in records)
    by_family = Counter(row["family"] for row in records)
    manifest = {
        "experiment": "qwen35_4b_sketch_coverage_shift_probe",
        "records": len(records),
        "records_per_family": args.records_per_family,
        "visible_count": args.visible_count,
        "hidden_count": args.hidden_count,
        "query_count": args.query_count,
        "seed": args.seed,
        "data_file": str(args.output.relative_to(ROOT)),
        "by_shift_type": dict(sorted(by_shift.items())),
        "by_family": dict(sorted(by_family.items())),
        "sketch_conditions": ["auto", "manual", "erased"],
        "purpose": "Falsify whether typed-sketch verified completion retains target coverage under name and primitive shift before adding another selector.",
    }
    manifest_path = ROOT / "data" / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

