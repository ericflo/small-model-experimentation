"""Shared fail-closed validation for durable gate lineage receipts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def validate_gate_lineage(
    lineage: Any, *, checkpoint_phase: str
) -> dict[str, dict[str, Any]]:
    expected = {
        "model_smoke": {"status": "MODEL_SMOKE_PASS", "phase": "g0"},
    }
    if checkpoint_phase == "full":
        expected["pilot_promotion"] = {
            "status": "PILOT_PROMOTION_READY",
            "phase": "pilot",
        }
    elif checkpoint_phase != "pilot":
        raise RuntimeError(f"unsupported checkpoint phase for gate lineage: {checkpoint_phase}")
    if not isinstance(lineage, Mapping) or set(lineage) != set(expected):
        raise RuntimeError(
            f"checkpoint gate lineage must contain exactly {sorted(expected)}"
        )
    validated: dict[str, dict[str, Any]] = {}
    for name, required in expected.items():
        record = lineage.get(name)
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "sha256",
            "receipt_identity_sha256",
            "status",
            "phase",
        }:
            raise RuntimeError(f"checkpoint {name} gate lineage is malformed")
        if not isinstance(record["path"], str) or not record["path"]:
            raise RuntimeError(f"checkpoint {name} gate lineage has no path")
        for digest_key in ("sha256", "receipt_identity_sha256"):
            if not isinstance(record[digest_key], str) or re.fullmatch(
                r"[0-9a-f]{64}", record[digest_key]
            ) is None:
                raise RuntimeError(
                    f"checkpoint {name} gate lineage has invalid {digest_key}"
                )
        if record["status"] != required["status"] or record["phase"] != required["phase"]:
            raise RuntimeError(f"checkpoint {name} gate lineage status/phase mismatch")
        validated[name] = dict(record)
    return validated
