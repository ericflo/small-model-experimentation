#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coverage_utils import EXPERIMENT, candidate_from_completion, load_mbpp_records  # noqa: E402
from src.jsonl import write_json, write_jsonl  # noqa: E402


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|\\d+", text.lower())


def summarize_code(code: str, max_chars: int = 1200) -> str:
    code = code.strip().replace("\r\n", "\n")
    return code[:max_chars]


def build_entry(record: dict[str, Any], order: int) -> dict[str, Any]:
    candidate = candidate_from_completion(
        record["reference_code"],
        record,
        source="verified_reference",
        order=0,
        prompt_tokens=0,
        completion_tokens=0,
    )
    text = " ".join(
        [
            record["task_text"],
            record["entry_point"],
            " ".join(case["assert_src"] for case in record.get("public_cases", [])),
            candidate.get("code", ""),
        ]
    )
    return {
        "library_id": f"alg_{order:04d}",
        "task_id": record["task_id"],
        "record_id": record["record_id"],
        "task_text": record["task_text"],
        "entry_point": record["entry_point"],
        "public_tests": [case["assert_src"] for case in record.get("public_cases", [])],
        "code": summarize_code(candidate.get("code", "") or record["reference_code"]),
        "verified": bool(candidate.get("full_pass")),
        "visible_all_pass": bool(candidate.get("visible_all_pass")),
        "parse_status": candidate.get("parse_status"),
        "functional_signature": candidate.get("functional_signature"),
        "retrieval_tokens": tokenize(text),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["mbpp_train", "mbpp_test"], default="mbpp_train")
    parser.add_argument("--count", type=int, default=374)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--visible-tests", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    records = load_mbpp_records(args.split, args.count, args.offset, args.visible_tests, args.timeout_s)
    rows = [build_entry(record, index) for index, record in enumerate(records)]
    verified = [row for row in rows if row["verified"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, verified)
    summary = {
        "experiment": EXPERIMENT,
        "split": args.split,
        "count_requested": args.count,
        "offset": args.offset,
        "records_loaded": len(records),
        "library_entries": len(verified),
        "dropped_unverified": len(rows) - len(verified),
        "path": str(args.out),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
