#!/usr/bin/env python3
"""Archive full raw calibration traces externally and leave analysis-complete compact rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_row(tag: str, row: dict[str, Any]) -> dict[str, Any]:
    common = {
        "id": row["id"],
        "accounting": row.get("accounting", {}),
    }
    if tag == "nextop":
        return {**common, "choice_probabilities": row["choice_probabilities"]}
    compact = {
        **common,
        "p_viable": row["p_viable"],
        "forced_close": bool(row.get("forced_close", False)),
    }
    if tag == "task_shuffled":
        meta = row.get("meta", {})
        compact["meta"] = {
            "original_id": meta["original_id"],
            "visible_donor_task": meta.get("visible_donor_task"),
        }
    return compact


def _archive_one(tag: str, source: Path, external: Path) -> dict[str, Any]:
    if not source.is_file():
        raise RuntimeError(f"missing calibration output: {source}")
    if external.exists():
        raise RuntimeError(f"external archive already exists: {external}")
    raw_size = source.stat().st_size
    raw_sha = _sha256(source)
    external.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, external)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{source.name}.", suffix=".tmp", dir=source.parent
    )
    count = 0
    try:
        with external.open("r", encoding="utf-8") as input_handle, os.fdopen(
            descriptor, "w", encoding="utf-8"
        ) as output_handle:
            for line_number, line in enumerate(input_handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid raw JSONL at {external}:{line_number}: {exc}"
                    ) from exc
                output_handle.write(json.dumps(_compact_row(tag, row), sort_keys=True) + "\n")
                count += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary_name, source)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        if not source.exists() and external.exists():
            os.replace(external, source)
        raise
    return {
        "tag": tag,
        "compact_path": str(source.relative_to(EXP)),
        "compact_size_bytes": source.stat().st_size,
        "compact_sha256": _sha256(source),
        "external_raw_path": str(external.relative_to(ROOT)),
        "raw_size_bytes": raw_size,
        "raw_sha256": raw_sha,
        "row_count": count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-dir",
        type=Path,
        default=ROOT / "large_artifacts" / EXP.name / "calibration",
    )
    args = parser.parse_args()
    receipt = EXP / "runs" / "calibration_model_receipt.json"
    if not receipt.exists():
        raise RuntimeError("full calibration receipt must exist before compaction")
    rows = []
    for tag in ("thinking", "nothink", "nextop", "task_shuffled"):
        source = EXP / "runs" / f"calibration_{tag}.jsonl"
        external = args.external_dir.resolve() / f"calibration_{tag}.raw.jsonl"
        rows.append(_archive_one(tag, source, external))
    result = {
        "schema_version": 1,
        "operation": "external_raw_archive_with_analysis_complete_compact_replacement",
        "receipt_before_compaction_sha256": _sha256(receipt),
        "artifacts": rows,
    }
    output = EXP / "analysis" / "calibration_compaction.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "artifacts": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
