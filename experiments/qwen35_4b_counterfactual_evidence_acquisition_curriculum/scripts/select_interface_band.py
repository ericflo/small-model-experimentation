#!/usr/bin/env python3
"""Select the smallest registered nonbinding answer rung using mechanics only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--rung-receipt", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    registered = [int(value) for value in cfg["evaluation"]["interface_answer_rungs"]]
    rows = [json.loads(path.read_text()) for path in args.rung_receipt]
    expected_analyzer = sha256_file(EXP / "scripts" / "analyze_interface_preflight.py")
    expected_config = sha256_file(args.config)
    for path, row in zip(args.rung_receipt, rows, strict=True):
        if (
            row.get("schema_version") != 1
            or row.get("stage") != "interface_preflight"
            or row.get("analyzer_sha256") != expected_analyzer
            or row.get("config_sha256") != expected_config
            or row.get("training_authorized") is not False
            or row.get("menagerie_authorized") is not False
        ):
            raise SystemExit(f"unregistered interface rung receipt: {path}")
        receipts = row.get("receipts")
        if not isinstance(receipts, dict) or set(receipts) != {
            "unassisted", "injected", "nondiscriminating_search", "explicit"
        }:
            raise SystemExit(f"interface rung has the wrong raw receipt set: {path}")
        for registration in receipts.values():
            if not isinstance(registration, dict) or set(registration) != {"path", "sha256"}:
                raise SystemExit(f"malformed interface raw receipt: {path}")
            raw_path = Path(registration["path"])
            if not raw_path.is_file() or sha256_file(raw_path) != registration["sha256"]:
                raise SystemExit(f"stale interface raw receipt: {raw_path}")
    observed = [int(row["answer_max_tokens"]) for row in rows]
    if observed != registered[:len(observed)]:
        raise SystemExit(
            f"interface rungs must be evaluated in registered order: {observed}"
        )
    passed = [row for row in rows if row.get("gate", {}).get("passed")]
    if passed and passed[0] is not rows[-1]:
        raise SystemExit("interface evaluation continued after the first passing rung")
    selected = int(passed[0]["answer_max_tokens"]) if passed else None
    exhausted = selected is None and observed == registered
    gate_passed = selected is not None
    result = {
        "schema_version": 1,
        "stage": "interface_answer_band_selection",
        "selector_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": expected_config,
        "selection_rule": (
            "smallest registered rung passing invalid-action and every-answer-limit-contact "
            "mechanics gates; no correctness metric enters selection"
        ),
        "registered_rungs": registered,
        "observed_rungs": observed,
        "selected_answer_max_tokens": selected,
        "band_exhausted": exhausted,
        "rungs": rows,
        "rung_receipts": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.rung_receipt
        ],
        "gate": {
            "passed": gate_passed,
            "verdict": "INTERFACE_PASS" if gate_passed else (
                "INSTRUMENT_FAIL" if exhausted else "MORE_RUNGS_REQUIRED"
            ),
        },
        "qualification_authorized": gate_passed,
        "training_authorized": False,
        "menagerie_authorized": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if gate_passed:
        return 0
    return 4 if exhausted else 3


if __name__ == "__main__":
    raise SystemExit(main())
