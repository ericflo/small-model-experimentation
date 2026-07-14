#!/usr/bin/env python3
"""Build disjoint action-prompt and oracle-label bundles for one frozen split."""

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

from taskgen import ACTION_QUESTION, build_corpus, build_retention_corpus  # noqa: E402


def _payload(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("calibration", "qualification", "confirmation", "retention"),
        required=True,
    )
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if len({args.prompts.resolve(), args.labels.resolve(), args.receipt.resolve()}) != 3:
        raise SystemExit("prompt, label, and receipt paths must differ")
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    if config["authorization"]["cpu_construction"] is not True:
        raise SystemExit("CPU construction is not authorized")
    construction = config["construction"]
    counts = {
        split: int(construction["per_family"][split])
        for split in ("train", "calibration", "qualification", "confirmation")
    }
    if args.split == "retention":
        tasks = build_retention_corpus(
            int(construction["per_family"]["retention_per_family_per_depth"]),
            int(construction["retention_seed"]),
        )
    else:
        tasks = build_corpus(counts, int(construction["seed"]))[args.split]
    prompts = [
        {
            "id": task["task_id"],
            "messages": [
                *task["common_messages"],
                {"role": "user", "content": ACTION_QUESTION},
            ],
            "meta": {
                "split": args.split,
                "family": task["family"],
                "depth": task["depth"],
            },
        }
        for task in tasks
    ]
    labels = [
        {
            "id": task["task_id"],
            "split": args.split,
            "family": task["family"],
            "depth": task["depth"],
            "answers": task["answers"],
        }
        for task in tasks
    ]
    prompt_payload = _payload(prompts)
    label_payload = _payload(labels)
    receipt = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "split": args.split,
        "rows": len(tasks),
        "prompt_sha256": hashlib.sha256(prompt_payload).hexdigest(),
        "label_sha256": hashlib.sha256(label_payload).hexdigest(),
        "model_calls": 0,
        "gpu_events": 0,
        "benchmark_reads": 0,
    }
    _write_exclusive(args.prompts, prompt_payload)
    _write_exclusive(args.labels, label_payload)
    _write_exclusive(
        args.receipt,
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
