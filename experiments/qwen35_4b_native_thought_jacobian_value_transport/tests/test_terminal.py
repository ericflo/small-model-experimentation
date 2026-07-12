from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_terminal_seam_receipt_is_complete_and_cap_bound() -> None:
    summary_path = EXP / "runs" / "seam_calibration.json"
    rows_path = EXP / "runs" / "seam_calibration_rows.jsonl"
    summary = json.loads(summary_path.read_text())
    rows = _jsonl(rows_path)
    assert summary["decision"] == "NO_NATURAL_SEAM"
    assert not summary["passed"]
    assert len(rows) == 48
    assert len({(row["task_id"], row["trace_index"]) for row in rows}) == 48
    assert len({row["task_id"] for row in rows}) == 16
    assert all(row["stopped_by"] == "think_cap_without_close" for row in rows)
    assert all(row["think_tokens"] == 160 for row in rows)
    assert not any(row["natural_close"] or row["parseable"] or row["correct"] for row in rows)
    assert sum(row["forward_calls"] for row in rows) == 7632
    metrics = json.loads((EXP / "analysis" / "seam_metrics.json").read_text())
    assert metrics["source_summary_sha256"] == hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    assert metrics["source_rows_sha256"] == hashlib.sha256(
        rows_path.read_bytes()
    ).hexdigest()
