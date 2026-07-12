#!/usr/bin/env python3
"""Run one aggregate-only Menagerie arm after procedural authorization."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
PY = REPO / ".venv" / "bin" / "python"
MENAGERIE = REPO / "benchmarks" / "menagerie" / "run.py"
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, sha256_file, write_json  # noqa: E402


def _authorized() -> dict:
    path = EXP / "analysis" / "confirmation.json"
    if not path.is_file():
        raise SystemExit("procedural confirmation receipt is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not payload.get("gate", {}).get("passed")
        or payload.get("downstream_authorization") != "benchmark_cli"
    ):
        raise SystemExit("procedural confirmation did not authorize benchmark CLI")
    return payload


def _sanitize(payload: dict) -> dict:
    if "aggregate" not in payload or "per_family" not in payload:
        raise ValueError("benchmark output lacks aggregate score fields")
    per_family = {}
    for family, value in payload["per_family"].items():
        if isinstance(value, dict):
            if "score" in value:
                value = value["score"]
            elif "mean" in value:
                value = value["mean"]
            else:
                raise ValueError(f"family {family} lacks score/mean")
        per_family[str(family)] = float(value)
    return {
        "aggregate": float(payload["aggregate"]),
        "per_family": per_family,
        "within_budget": payload.get("within_budget"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("quick", "medium"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", choices=("primary", "soup", "visible"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config()
    confirmation = _authorized()
    first = int(config["benchmark"]["first_seed"])
    quick_n = int(config["benchmark"]["quick_events"])
    medium_n = int(config["benchmark"]["medium_events"])
    allowed = {
        "quick": set(range(first, first + quick_n)),
        "medium": set(range(first + quick_n, first + quick_n + medium_n)),
    }
    if args.seed not in allowed[args.tier]:
        raise SystemExit(f"seed {args.seed} is outside frozen {args.tier} namespace")
    model = args.model.resolve()
    if not (model / "merge_receipt.json").is_file():
        raise SystemExit("benchmark model is not an explicit merged checkpoint")
    if args.out.is_file():
        existing = json.loads(args.out.read_text(encoding="utf-8"))
        expected = (args.tier, args.seed, args.label, str(model))
        observed = (
            existing.get("tier"),
            int(existing.get("seed", -1)),
            existing.get("label"),
            existing.get("model"),
        )
        if observed != expected:
            raise SystemExit(f"stale benchmark receipt: {args.out}")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
    if args.out.exists():
        raise SystemExit(f"benchmark output path is not a file: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # Item-level benchmark output may exist only inside this ephemeral directory.
    # The experiment persists aggregate and per-family measurements, never tasks,
    # transcripts, or item-level result details.
    with tempfile.TemporaryDirectory(prefix="deep_mopd_benchmark_") as tmp:
        raw = Path(tmp) / "raw.json"
        command = [
            str(PY),
            str(MENAGERIE),
            "--tier",
            args.tier,
            "--seed",
            str(args.seed),
            "--model-id",
            str(model),
            "--out",
            str(raw),
        ]
        completed = subprocess.run(
            command,
            cwd=MENAGERIE.parent,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            sys.stderr.write(completed.stdout[-4000:] + "\n" + completed.stderr[-4000:] + "\n")
            raise SystemExit(completed.returncode)
        sanitized = _sanitize(json.loads(raw.read_text(encoding="utf-8")))
    receipt = {
        "schema_version": 1,
        "stage": "aggregate_only_menagerie_event",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "confirmation_sha256": sha256_file(EXP / "analysis" / "confirmation.json"),
        "confirmation_gate": confirmation["gate"],
        "tier": args.tier,
        "seed": args.seed,
        "label": args.label,
        "model": str(model),
        "model_merge_receipt_sha256": sha256_file(model / "merge_receipt.json"),
        **sanitized,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
