#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diversity_utils import (  # noqa: E402
    EXPERIMENT,
    add_usage,
    dedupe_candidates,
    empty_usage,
    recompute_record_metrics,
    summarize_records,
    write_manifest,
)
from src.jsonl import load_jsonl, write_jsonl  # noqa: E402


def load_manifest_for(path: Path) -> dict:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round-name", type=str, required=True)
    parser.add_argument("--base-records", type=Path, default=None)
    args = parser.parse_args()

    by_id: dict[str, dict] = {}
    usage = empty_usage()
    source_manifests = []
    for path in args.inputs:
        records = load_jsonl(path)
        manifest = load_manifest_for(path)
        source_manifests.append({"path": str(path), "manifest": manifest})
        usage = add_usage(usage, manifest.get("token_usage", {}))
        for record in records:
            rid = record["record_id"]
            if rid not in by_id:
                by_id[rid] = deepcopy(record)
                by_id[rid]["candidates"] = []
            by_id[rid]["candidates"].extend(deepcopy(record.get("candidates", [])))

    merged = []
    for record in by_id.values():
        record["round_name"] = args.round_name
        record["candidates"] = dedupe_candidates(record.get("candidates", []))
        recompute_record_metrics(record)
        merged.append(record)
    merged.sort(key=lambda r: r.get("task_id", r.get("record_id", "")))

    base_records = load_jsonl(args.base_records) if args.base_records else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, merged)
    manifest = {
        "experiment": EXPERIMENT,
        "round_name": args.round_name,
        "inputs": [str(path) for path in args.inputs],
        "base_records": str(args.base_records) if args.base_records else None,
        "token_usage": usage,
        "records": summarize_records(merged, base_records=base_records),
        "source_manifests": source_manifests,
        "path": str(args.out),
    }
    write_manifest(args.out.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
