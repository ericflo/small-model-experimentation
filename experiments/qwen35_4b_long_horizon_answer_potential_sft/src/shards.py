"""Atomic compressed shard storage and reproducibility receipts."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Write one gzip JSONL shard atomically and return its receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    count = 0
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                count += 1
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(path),
        "rows": count,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def valid_receipt(receipt: dict[str, Any]) -> bool:
    path = Path(str(receipt.get("path", "")))
    return (
        path.is_file()
        and path.stat().st_size == int(receipt.get("bytes", -1))
        and sha256_file(path) == receipt.get("sha256")
    )
