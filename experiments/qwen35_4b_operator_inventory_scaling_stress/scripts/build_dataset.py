#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tasks import TEMPLATES, build_records  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-sizes", default="8,16,32,64,128,256,512")
    parser.add_argument("--records-per-template", type=int, default=12)
    parser.add_argument("--visible-count", type=int, default=6)
    parser.add_argument("--hidden-count", type=int, default=18)
    parser.add_argument("--query-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "operator_scaling_eval.jsonl")
    args = parser.parse_args()

    library_sizes = [int(item) for item in args.library_sizes.split(",") if item.strip()]
    rows, library = build_records(
        library_sizes=library_sizes,
        records_per_template=args.records_per_template,
        visible_count=args.visible_count,
        hidden_count=args.hidden_count,
        query_count=args.query_count,
        seed=args.seed,
        max_library_size=max(library_sizes),
    )
    write_jsonl(args.output, rows)

    manifest = {
        "experiment": "qwen35_4b_operator_inventory_scaling_stress",
        "records": len(rows),
        "records_per_template": args.records_per_template,
        "visible_count": args.visible_count,
        "hidden_count": args.hidden_count,
        "query_count": args.query_count,
        "seed": args.seed,
        "data_file": str(args.output.relative_to(ROOT)),
        "library_sizes": library_sizes,
        "operator_signature": "list[int] -> int",
        "operator_count": len(library),
        "operator_names": [operator.name for operator in library],
        "operator_families": dict(sorted(Counter(operator.family for operator in library).items())),
        "templates": [template.__dict__ for template in TEMPLATES],
        "by_library_size": dict(sorted(Counter(row["library_size"] for row in rows).items())),
        "by_template": dict(sorted(Counter(row["template"] for row in rows).items())),
        "purpose": "No-training scaling stress test for same-signature operator inventory search.",
    }
    manifest_path = ROOT / "data" / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

