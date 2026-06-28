#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_selective_fallback import family_summary, summarize  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-records", type=Path, default=ROOT / "reports" / "program_probe_records.jsonl")
    parser.add_argument("--model-probe-records", type=Path, default=ROOT / "reports" / "model_probe_visible_disagree_records.jsonl")
    parser.add_argument("--records-out", type=Path, default=ROOT / "reports" / "final_records.jsonl")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "reports" / "final_summary.json")
    parser.add_argument("--family-out", type=Path, default=ROOT / "reports" / "final_family_summary.json")
    args = parser.parse_args()

    base = load_jsonl(args.base_records)
    replacements = {row["file"]: row for row in load_jsonl(args.model_probe_records)}
    merged = []
    for row in base:
        if row["file"] in replacements:
            replacement = replacements[row["file"]]
            row["probes"] = replacement["probes"]
            row["probe_features"] = replacement["probe_features"]
        merged.append(row)

    args.records_out.parent.mkdir(parents=True, exist_ok=True)
    with args.records_out.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    summary = summarize(merged)
    summary["model_probe_replacements"] = len(replacements)
    write_json(args.summary_out, summary)
    write_json(args.family_out, family_summary(merged))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

