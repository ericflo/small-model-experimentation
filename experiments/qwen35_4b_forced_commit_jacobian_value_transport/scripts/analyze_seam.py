#!/usr/bin/env python3
"""Bind terminal forced-commit metrics and parser-robustness diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml


EXP = Path(__file__).resolve().parents[1]
SUMMARY = EXP / "runs" / "seam_selection.json"
POLICY = EXP / "runs" / "seam_selection_policy_rows.jsonl"
TRACES = EXP / "runs" / "seam_selection_traces.jsonl"
TASKS = EXP / "data" / "procedural" / "seam_selection.jsonl"
OUTPUT = EXP / "analysis" / "seam_metrics.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> int:
    summary = json.loads(SUMMARY.read_text())
    rows = read_jsonl(POLICY)
    traces = read_jsonl(TRACES)
    tasks = read_jsonl(TASKS)
    if sha256_file(POLICY) != summary["policy_rows_sha256"]:
        raise RuntimeError("policy rows changed after terminal decision")
    if sha256_file(TRACES) != summary["trace_rows_sha256"]:
        raise RuntimeError("trace rows changed after terminal decision")
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    aliases = dict(config["data"]["operation_aliases"])
    alias_values = set(aliases.values())
    gold = {task["task_id"]: aliases[task["first_op"]] for task in tasks}
    pattern = re.compile(r"First:\s*`?([A-Za-z_]+)", flags=re.IGNORECASE)
    robust_by_cap = []
    for cap in config["generation"]["cap_rungs"]:
        cap_rows = [row for row in rows if int(row["cap"]) == int(cap)]
        robust_aliases = []
        for row in cap_rows:
            match = pattern.search(row["answer_text"])
            candidate = match.group(1).lower() if match else None
            robust_aliases.append(candidate if candidate in alias_values else None)
        robust_by_cap.append(
            {
                "cap": int(cap),
                "robust_parse_count": sum(value is not None for value in robust_aliases),
                "robust_parse_rate": sum(value is not None for value in robust_aliases) / len(cap_rows),
                "robust_correct_count": sum(
                    value == gold[row["task_id"]]
                    for row, value in zip(cap_rows, robust_aliases, strict=True)
                ),
                "frozen_parse_count": sum(bool(row["parseable"]) for row in cap_rows),
                "frozen_correct_count": sum(bool(row["correct"]) for row in cap_rows),
            }
        )
    result = {
        "schema_version": 1,
        "decision": summary["decision"],
        "passed": summary["passed"],
        "selected_cap": summary["selected_cap"],
        "counterfactual_policy": summary["counterfactual_policy"],
        "traces": len(traces),
        "policy_rows": len(rows),
        "metrics_by_cap": summary["metrics_by_cap"],
        "robust_parser_diagnostic_by_cap": robust_by_cap,
        "trace_rows_sha256": summary["trace_rows_sha256"],
        "policy_rows_sha256": summary["policy_rows_sha256"],
        "trace_sampled_tokens": summary["trace_sampled_tokens"],
        "policy_answer_tokens": summary["policy_answer_tokens"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "interpretation": (
            "A regex diagnostic that tolerates special tokens attached to the alias raises parse "
            "slightly but leaves every cap far below frozen parse, answer-termination, success, and "
            "mixed-task gates. It cannot change the preregistered decision."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
