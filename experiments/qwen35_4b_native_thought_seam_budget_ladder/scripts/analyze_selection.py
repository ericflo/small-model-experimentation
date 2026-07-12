#!/usr/bin/env python3
"""Derive termination diagnostics from the completed selection rows."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
ROWS = EXP / "runs" / "budget_selection_rows.jsonl"
SUMMARY = EXP / "runs" / "budget_selection.json"
OUTPUT = EXP / "analysis" / "selection_metrics.json"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def maximum_ngram_count(tokens: list[int], n: int) -> int:
    if len(tokens) < n:
        return 0
    return max(Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)).values())


def exact_tail_period(tokens: list[int], *, tail: int = 256, max_period: int = 32) -> int | None:
    values = tokens[-tail:]
    if len(values) < tail:
        return None
    for period in range(1, max_period + 1):
        if all(values[index] == values[index - period] for index in range(period, len(values))):
            return period
    return None


def main() -> int:
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line]
    summary: dict[str, Any] = json.loads(SUMMARY.read_text())
    if sha256_file(ROWS) != summary["rows_sha256"]:
        raise RuntimeError("selection row hash does not match the frozen summary")
    unique_ratios = [len(set(row["generated_token_ids"])) / len(row["generated_token_ids"]) for row in rows]
    trigram_counts = [maximum_ngram_count(row["generated_token_ids"], 3) for row in rows]
    periods = [exact_tail_period(row["generated_token_ids"]) for row in rows]
    result = {
        "schema_version": 1,
        "source_rows": str(ROWS.relative_to(EXP)),
        "source_rows_sha256": summary["rows_sha256"],
        "decision": summary["decision"],
        "metrics_by_cap": [
            {
                **metrics,
                "cap_contact_rate": metrics["cap_contacts"] / metrics["traces"],
            }
            for metrics in summary["metrics_by_cap"]
        ],
        "traces": len(rows),
        "think_tokens": {
            "minimum": min(int(row["think_tokens"]) for row in rows),
            "median": statistics.median(int(row["think_tokens"]) for row in rows),
            "maximum": max(int(row["think_tokens"]) for row in rows),
        },
        "natural_close_count": sum(bool(row["natural_close"]) for row in rows),
        "cache_contract_pass_count": sum(bool(row["cache_contract_pass"]) for row in rows),
        "unique_token_ratio": {
            "minimum": min(unique_ratios),
            "median": statistics.median(unique_ratios),
            "maximum": max(unique_ratios),
        },
        "maximum_trigram_count": {
            "minimum": min(trigram_counts),
            "median": statistics.median(trigram_counts),
            "maximum": max(trigram_counts),
        },
        "exact_periodic_256_token_tails": sum(period is not None for period in periods),
        "tail_periods": [period for period in periods if period is not None],
        "interpretation": (
            "The close failure is universal at the frozen ceiling. Exact short-period tail loops are "
            "reported separately; their absence cannot prove semantic progress."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
