#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coverage_utils import add_usage, empty_usage, recompute_record_metrics, summarize_records, write_manifest  # noqa: E402
from src.jsonl import load_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.records)
    for row in rows:
        recompute_record_metrics(row)
    usage = empty_usage()
    for row in rows:
        usage = add_usage(usage, row.get("token_usage", {}))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest["records"] = summarize_records(rows)
    manifest["token_usage"] = usage
    write_manifest(args.manifest, manifest)
    print(json.dumps({"manifest": str(args.manifest), "records": manifest["records"], "token_usage": usage}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

