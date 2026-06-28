#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.foofah import family_name, load_cases, prompt_for_case  # noqa: E402


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    source = Path("/workspace/large_artifacts/external_sources/foofah_benchmarks")
    cases = load_cases(source)
    for row in cases:
        row["family"] = family_name(row["file"], row["test_name"])
        row["prompt"] = prompt_for_case(row)
    write_jsonl(ROOT / "data" / "cases.jsonl", cases)
    summary = {
        "n": len(cases),
        "families": len(set(row["family"] for row in cases)),
        "by_num_samples": {str(k): sum(row["num_samples"] == k for row in cases) for k in sorted(set(row["num_samples"] for row in cases))},
    }
    write_json(ROOT / "data" / "case_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
