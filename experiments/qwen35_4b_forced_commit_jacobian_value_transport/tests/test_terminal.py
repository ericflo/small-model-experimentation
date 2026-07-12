from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_terminal_forced_seam_is_complete_and_hash_bound() -> None:
    summary = json.loads((EXP / "runs" / "seam_selection.json").read_text())
    traces_path = EXP / "runs" / "seam_selection_traces.jsonl"
    policy_path = EXP / "runs" / "seam_selection_policy_rows.jsonl"
    traces = [json.loads(line) for line in traces_path.read_text().splitlines() if line]
    policy = [json.loads(line) for line in policy_path.read_text().splitlines() if line]
    assert summary["decision"] == "FORCED_COMMIT_SEAM_FAIL"
    assert summary["passed"] is False
    assert summary["selected_cap"] is None
    assert summary["counterfactual_policy"] is True
    assert len(traces) == summary["traces"] == 48
    assert len(policy) == 144
    assert _sha(traces_path) == summary["trace_rows_sha256"]
    assert _sha(policy_path) == summary["policy_rows_sha256"]
    assert all(row["cache_contract_pass"] for row in traces)
    assert all(row["answer_cache_contract_pass"] for row in policy)
    assert all(row["commit_mode"] == "forced" for row in policy)
    assert [row["policy_successes"] for row in summary["metrics_by_cap"]] == [1, 1, 1]
    assert [row["mixed_policy_tasks"] for row in summary["metrics_by_cap"]] == [1, 1, 1]
    assert all(row["gate_pass"] is False for row in summary["metrics_by_cap"])


def test_parser_diagnostic_cannot_rescue_terminal_gate() -> None:
    analysis = json.loads((EXP / "analysis" / "seam_metrics.json").read_text())
    assert analysis["decision"] == "FORCED_COMMIT_SEAM_FAIL"
    assert analysis["traces"] == 48
    assert analysis["policy_rows"] == 144
    assert [row["robust_parse_count"] for row in analysis["robust_parser_diagnostic_by_cap"]] == [7, 11, 10]
    assert [row["robust_correct_count"] for row in analysis["robust_parser_diagnostic_by_cap"]] == [1, 2, 2]
    assert max(row["robust_parse_rate"] for row in analysis["robust_parser_diagnostic_by_cap"]) < 0.25
