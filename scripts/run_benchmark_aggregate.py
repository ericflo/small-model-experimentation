#!/usr/bin/env python3
"""Run Menagerie behind a process boundary and emit aggregate fields only.

This module is infrastructure, not experiment code.  The child benchmark owns
its raw output inside a private temporary directory.  Callers receive only the
small, explicitly whitelisted summary written to ``--out``; child stdout and
stderr are never forwarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "bin" / "python"
MENAGERIE = REPO / "benchmarks" / "menagerie" / "run.py"
BACKEND = "qwen_vllm"
PUBLIC_FAMILY_KEYS = frozenset(
    {
        "chronicle",
        "lockpick",
        "menders",
        "mirage",
        "rites",
        "siftstack",
        "sirens",
        "stockade",
        "toolsmith",
        "warren",
    }
)
OUTPUT_KEYS = frozenset(
    {
        "schema_version",
        "stage",
        "tier",
        "seed",
        "backend",
        "model",
        "model_merge_receipt_sha256",
        "benchmark_runner_sha256",
        "benchmark_source_inventory_sha256",
        "benchmark_source_file_count",
        "aggregate",
        "per_family",
        "within_budget",
        "wall_seconds",
    }
)


class RunnerFailure(RuntimeError):
    """A safe benchmark-runner failure with no captured child output."""

    def __init__(self, returncode: int):
        super().__init__(
            f"benchmark runner failed with exit code {returncode}; "
            "raw stdout/stderr suppressed"
        )
        self.returncode = int(returncode)


class AggregateFailure(RuntimeError):
    """The private runner output could not be reduced to the public schema."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_source_inventory(root: Path) -> dict[str, Any]:
    """Hash executable suite inputs without parsing or returning their contents."""

    root = root.resolve()
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(
            part in {"results", "__pycache__", ".pytest_cache"}
            for part in relative.parts
        ):
            continue
        rows.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise AggregateFailure("benchmark source inventory is empty")
    canonical = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "file_count": len(rows),
    }


def _finite_score(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise AggregateFailure(f"{field} is not a numeric score")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AggregateFailure(f"{field} is not a numeric score") from exc
    if not math.isfinite(result):
        raise AggregateFailure(f"{field} is not finite")
    return result


def _sanitize(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AggregateFailure("benchmark output is not an object")
    families = payload.get("per_family")
    if not isinstance(families, dict) or set(families) != PUBLIC_FAMILY_KEYS:
        raise AggregateFailure("benchmark output lacks per-family aggregates")
    per_family: dict[str, float] = {}
    for family, value in families.items():
        if not isinstance(family, str) or not family:
            raise AggregateFailure("benchmark family key is invalid")
        if isinstance(value, dict):
            if "score" in value:
                value = value["score"]
            elif "mean" in value:
                value = value["mean"]
            else:
                raise AggregateFailure("benchmark family aggregate is missing")
        per_family[family] = _finite_score(value, "per_family")
    within_budget = payload.get("within_budget")
    if within_budget is not True:
        raise AggregateFailure("benchmark event exceeded or omitted its budget gate")
    return {
        "aggregate": _finite_score(payload.get("aggregate"), "aggregate"),
        "per_family": per_family,
        "within_budget": within_budget,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AggregateFailure("aggregate output path already exists")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_event(
    *,
    tier: str,
    seed: int,
    model: Path,
    out: Path,
    runner: Path = MENAGERIE,
    python: Path = PY,
) -> dict[str, Any]:
    """Run one event while keeping all non-aggregate material in this process."""

    if tier not in {"quick", "medium"}:
        raise AggregateFailure("unsupported benchmark tier")
    runner = runner.resolve()
    model = model.resolve()
    merge_receipt = model / "merge_receipt.json"
    if not runner.is_file():
        raise AggregateFailure("benchmark runner is missing")
    if not merge_receipt.is_file():
        raise AggregateFailure("model merge receipt is missing")
    if out.exists():
        raise AggregateFailure("aggregate output path already exists")

    source_before = benchmark_source_inventory(runner.parent)
    started = time.perf_counter()
    old_umask = os.umask(0o077)
    try:
        with tempfile.TemporaryDirectory(prefix="benchmark_aggregate_gateway_") as tmp:
            raw = Path(tmp) / "private_runner_output.json"
            command = [
                str(python),
                str(runner),
                "--tier",
                tier,
                "--backend",
                BACKEND,
                "--seed",
                str(int(seed)),
                "--model-id",
                str(model),
                "--out",
                str(raw),
            ]
            completed = subprocess.run(
                command,
                cwd=runner.parent,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if completed.returncode != 0:
                raise RunnerFailure(completed.returncode)
            try:
                private_payload = json.loads(raw.read_text(encoding="utf-8"))
                aggregate = _sanitize(private_payload)
            except AggregateFailure:
                raise
            except Exception as exc:  # noqa: BLE001 - never expose private data
                raise AggregateFailure("private benchmark output was invalid") from exc
    finally:
        os.umask(old_umask)

    source_after = benchmark_source_inventory(runner.parent)
    if source_after != source_before:
        raise AggregateFailure("benchmark source inventory changed during the event")
    result = {
        "schema_version": 1,
        "stage": "menagerie_aggregate_gateway",
        "tier": tier,
        "seed": int(seed),
        "backend": BACKEND,
        "model": str(model),
        "model_merge_receipt_sha256": sha256_file(merge_receipt),
        "benchmark_runner_sha256": sha256_file(runner),
        "benchmark_source_inventory_sha256": source_after["sha256"],
        "benchmark_source_file_count": source_after["file_count"],
        **aggregate,
        "wall_seconds": time.perf_counter() - started,
    }
    if set(result) != OUTPUT_KEYS:
        raise AssertionError("aggregate gateway schema drift")
    _write_json_atomic(out, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("quick", "medium"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_event(tier=args.tier, seed=args.seed, model=args.model, out=args.out)
    except RunnerFailure as exc:
        print(str(exc), file=sys.stderr)
        return exc.returncode or 1
    except AggregateFailure:
        print(
            "aggregate benchmark gateway failed; private output suppressed",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
