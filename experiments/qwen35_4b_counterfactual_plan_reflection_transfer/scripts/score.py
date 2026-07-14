#!/usr/bin/env python3
"""Score one immutable vLLM generation bundle into task-level gate rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from firewall import install_benchmark_firewall  # noqa: E402

install_benchmark_firewall(EXP.parents[1])

from scoring import score_generation_rows  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_exclusive(path: Path, rows: list[dict]) -> str:
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    evaluation = config["evaluation"]
    counts = tuple(
        sorted(
            {
                int(evaluation["primary_candidate_count"]),
                *[int(value) for value in evaluation["descriptive_candidate_counts"]],
            }
        )
    )
    scored = score_generation_rows(
        _read_jsonl(args.generated),
        _read_jsonl(args.labels),
        arm=args.arm,
        candidate_counts=counts,
        answer_max_tokens=int(evaluation["answer_max_tokens"]),
        loop_detector=evaluation["loop_detector"],
    )
    digest = _write_exclusive(args.output, scored)
    print(json.dumps({"rows": len(scored), "output_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
