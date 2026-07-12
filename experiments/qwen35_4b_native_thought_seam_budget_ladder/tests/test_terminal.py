from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_terminal_selection_is_complete_and_hash_bound() -> None:
    summary = json.loads((EXP / "runs" / "budget_selection.json").read_text())
    rows_path = EXP / "runs" / "budget_selection_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    assert summary["decision"] == "NO_BUDGET_SELECTED"
    assert summary["passed"] is False
    assert summary["selected_cap"] is None
    assert summary["traces"] == len(rows) == 48
    assert summary["rows_sha256"] == _sha256(rows_path)
    assert all(row["cache_contract_pass"] for row in rows)
    assert all(row["natural_close"] is False for row in rows)
    assert all(row["think_tokens"] == 1024 for row in rows)
    assert all(row["stopped_by"] == "think_cap_without_close" for row in rows)
    assert summary["sampled_tokens"] == summary["forward_calls"] == 49_152
    for metrics in summary["metrics_by_cap"]:
        assert metrics["gate_pass"] is False
        assert metrics["natural_closes"] == 0
        assert metrics["cap_contacts"] == 48
        assert metrics["parseable"] == metrics["usable_traces"] == 0


def test_post_decision_diagnostics_are_bound_to_terminal_rows() -> None:
    analysis = json.loads((EXP / "analysis" / "selection_metrics.json").read_text())
    assert analysis["decision"] == "NO_BUDGET_SELECTED"
    assert analysis["traces"] == 48
    assert analysis["natural_close_count"] == 0
    assert analysis["cache_contract_pass_count"] == 48
    assert analysis["think_tokens"] == {"minimum": 1024, "median": 1024.0, "maximum": 1024}
    assert analysis["exact_periodic_256_token_tails"] == 0
    assert analysis["source_rows_sha256"] == (
        "17e3b107154079ecd857af45544c92c2e11b13cd495edfeb6eb24dcf97f5d39c"
    )
