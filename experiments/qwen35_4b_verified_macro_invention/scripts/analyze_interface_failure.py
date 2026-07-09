#!/usr/bin/env python3
"""Regenerate the exact failure taxonomy for interface attempt 3."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

import macro_domain as domain  # noqa: E402
import model_harness as harness  # noqa: E402


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def analyze() -> dict[str, Any]:
    rows = _read_jsonl(EXP / "runs" / "interface_v3_failed" / "designed_ceiling.jsonl")
    libraries = _read_json(EXP / "data" / "libraries.json")["libraries"]
    macro_rows = libraries["designed_ceiling"]["macros"]
    macro_map = {
        str(macro["token"]): tuple(str(token) for token in macro["expansion"])
        for macro in macro_rows
    }
    parsed = harness.parse_program_outputs(
        rows,
        allowed_tokens=set(domain.PRIMITIVES) | set(macro_map),
        max_surface_calls=5,
    )
    row_by_id = {str(row["id"]): row for row in rows}
    output_by_key = {
        (str(row["id"]), int(output["sample_index"])): output
        for row in rows
        for output in row["outputs"]
    }
    samples: list[dict[str, Any]] = []
    successful_records: set[str] = set()
    for completion in parsed:
        row = row_by_id[completion.record_id]
        target = tuple(str(token) for token in row["meta"]["target_program"])
        record_index = int(completion.record_id.split("-")[2].split("::")[0])
        designated_alias = f"M{record_index}"
        output = output_by_key[(completion.record_id, completion.sample_index)]
        program = completion.program
        expanded = domain.expand_program(program, macro_map) if program is not None else None
        macro_tokens = [token for token in program or () if token in macro_map]
        exact = expanded == target
        optimal = bool(
            exact
            and program is not None
            and len(program) == int(row["meta"]["optimal_surface_calls"])
            and macro_tokens
        )
        if optimal:
            successful_records.add(completion.record_id)
        samples.append(
            {
                "record_id": completion.record_id,
                "sample_index": completion.sample_index,
                "parse_error": completion.parse_error,
                "program": list(program) if program is not None else None,
                "expanded_program": list(expanded) if expanded is not None else None,
                "expanded_depth": len(expanded) if expanded is not None else None,
                "macro_tokens": macro_tokens,
                "designated_alias": designated_alias,
                "contains_designated_alias": designated_alias in macro_tokens,
                "exact_expansion": exact,
                "optimal_success": optimal,
                "truncated": bool(output.get("truncated")),
            }
        )

    failures = [sample for sample in samples if not sample["optimal_success"]]
    depth_counts = Counter(
        int(sample["expanded_depth"])
        for sample in failures
        if sample["expanded_depth"] is not None
    )
    result = {
        "schema_version": 1,
        "experiment_id": "qwen35_4b_verified_macro_invention",
        "interface_attempt": 3,
        "source": "runs/interface_v3_failed/designed_ceiling.jsonl",
        "metrics": {
            "records": len(rows),
            "samples": len(samples),
            "strictly_parsed_samples": sum(sample["parse_error"] is None for sample in samples),
            "truncated_samples": sum(sample["truncated"] for sample in samples),
            "macro_using_samples": sum(bool(sample["macro_tokens"]) for sample in samples),
            "exact_expansion_samples": sum(sample["exact_expansion"] for sample in samples),
            "optimal_success_samples": sum(sample["optimal_success"] for sample in samples),
            "successful_records": len(successful_records),
            "successful_record_ids": sorted(successful_records),
            "failed_samples": len(failures),
            "failed_samples_with_multiple_macros": sum(
                len(sample["macro_tokens"]) > 1 for sample in failures
            ),
            "failed_samples_over_expanded_depth_five": sum(
                int(sample["expanded_depth"] or 0) > 5 for sample in failures
            ),
            "failed_samples_with_designated_alias": sum(
                sample["contains_designated_alias"] for sample in failures
            ),
            "failed_samples_without_designated_alias": sum(
                not sample["contains_designated_alias"] for sample in failures
            ),
            "failed_expanded_depth_counts": {
                str(depth): count for depth, count in sorted(depth_counts.items())
            },
        },
        "samples": samples,
    }
    expected = {
        "strictly_parsed_samples": 16,
        "truncated_samples": 0,
        "macro_using_samples": 16,
        "exact_expansion_samples": 3,
        "optimal_success_samples": 3,
        "successful_records": 1,
        "failed_samples": 13,
        "failed_samples_with_multiple_macros": 13,
        "failed_samples_over_expanded_depth_five": 13,
        "failed_samples_with_designated_alias": 10,
        "failed_samples_without_designated_alias": 3,
    }
    for key, value in expected.items():
        if result["metrics"][key] != value:
            raise RuntimeError(
                f"interface-v3 audit drift for {key}: {result['metrics'][key]} != {value}"
            )
    return result


def main() -> int:
    result = analyze()
    path = EXP / "analysis" / "interface_v3_audit.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
