#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.foofah_gate import load_cases, run_case, summarize  # noqa: E402


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/workspace/large_artifacts/external_sources/foofah_benchmarks"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cases = load_cases(args.source)
    if args.limit:
        cases = cases[: args.limit]
    records = []
    for idx, case in enumerate(cases):
        records.append(run_case(case))
        if (idx + 1) % 50 == 0:
            print(f"evaluated {idx + 1}/{len(cases)}", flush=True)
    summary = summarize(records)
    write_jsonl(ROOT / "data" / "case_records.jsonl", records)
    write_json(ROOT / "reports" / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
