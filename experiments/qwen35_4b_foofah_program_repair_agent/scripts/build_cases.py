#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.foofah import direct_prompt, family_name, initial_program_prompt, load_cases  # noqa: E402


SOURCE = Path("/workspace/large_artifacts/external_sources/foofah_benchmarks")


def main() -> None:
    rows = load_cases(SOURCE)
    for row in rows:
        row["family"] = family_name(row["file"], row["test_name"])
        row["direct_prompt"] = direct_prompt(row)
        row["initial_program_prompt"] = initial_program_prompt(row)
    out = ROOT / "data" / "cases.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    summary = {
        "n": len(rows),
        "families": len({row["family"] for row in rows}),
        "by_num_samples": dict(sorted(Counter(row["num_samples"] for row in rows).items())),
    }
    (ROOT / "data" / "case_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
