#!/usr/bin/env python3
"""Derive compact terminal metrics from the committed donor/confirmation rows."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "analysis" / "metrics.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    donor = load(RUNS / "donor_gate.json")
    confirmation = load(RUNS / "confirmation.json")
    rows = [
        json.loads(line)
        for line in (RUNS / "confirmation_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    failures = [
        {
            "item_id": row["item_id"],
            "prompt_kind": row["prompt_kind"],
            "relative_error": row["norm_match_max_relative_error"],
        }
        for row in rows
        if row["condition"] == "random_norm_orthogonal" and not row["norm_match_passed"]
    ]
    metrics = {
        "schema_version": 1,
        "decision": confirmation["decision"],
        "selected_band": confirmation["band"],
        "confirmation_n": confirmation["confirmation_n"],
        "condition_summaries": confirmation["summaries"],
        "gate_metrics": confirmation["gate_metrics"],
        "control_audit": confirmation["control_audit"],
        "failed_norm_rows": failures,
        "donor_candidates": donor["candidates"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}; failed norm rows={len(failures)}")


if __name__ == "__main__":
    main()
