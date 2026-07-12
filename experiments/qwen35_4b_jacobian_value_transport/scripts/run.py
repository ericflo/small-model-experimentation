#!/usr/bin/env python3
"""Restartable, immutable-design-gated Jacobian value-transport orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from io_utils import sha256_file, write_json  # noqa: E402
from task_data import build_splits  # noqa: E402

CONFIG_PATH = EXP / "configs" / "default.yaml"
DATA_DIR = EXP / "data" / "procedural"
RUNS_DIR = EXP / "runs"


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text())
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def design_boundary_receipt(config: dict[str, Any]) -> dict[str, Any]:
    boundary = config["design_boundary"]
    commit = boundary.get("commit")
    expected_prereg = boundary.get("preregistration_sha256")
    expected_readme = boundary.get("readme_sha256")
    if not commit or not expected_prereg or not expected_readme:
        raise RuntimeError("design boundary has not been anchored to a commit")
    head = _git(["rev-parse", "HEAD"]).stdout.strip()
    ancestor = _git(["merge-base", "--is-ancestor", str(commit), head], check=False).returncode == 0
    paths = {
        "preregistration": "experiments/qwen35_4b_jacobian_value_transport/reports/preregistration.md",
        "readme": "experiments/qwen35_4b_jacobian_value_transport/README.md",
    }
    observed = {}
    for name, path in paths.items():
        payload = _git(["show", f"{commit}:{path}"]).stdout.encode()
        observed[name] = hashlib.sha256(payload).hexdigest()
    passed = bool(
        ancestor
        and observed["preregistration"] == expected_prereg
        and observed["readme"] == expected_readme
    )
    receipt = {
        "schema_version": 1,
        "passed": passed,
        "design_commit": commit,
        "head": head,
        "design_is_ancestor": ancestor,
        "observed_sha256": observed,
        "expected_sha256": {
            "preregistration": expected_prereg,
            "readme": expected_readme,
        },
    }
    write_json(RUNS_DIR / "design_boundary_receipt.json", receipt)
    if not passed:
        raise RuntimeError(f"immutable design boundary failed: {receipt}")
    return receipt


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    manifest = build_splits(DATA_DIR, config)
    required = {
        "lens_fit", "positive_control", "value_calibration", "iid_eval",
        "held_string_eval", "held_register_eval", "hard_eval",
    }
    passed = required == set(manifest["counts"]) and all(manifest["counts"][name] > 0 for name in required)
    receipt = {
        "schema_version": 1,
        "scientific_evidence": False,
        "passed": passed,
        "manifest_sha256": sha256_file(DATA_DIR / "manifest.json"),
        "counts": manifest["counts"],
        "benchmark_content_used": manifest["firewall"]["benchmark_content_used"],
    }
    write_json(RUNS_DIR / "smoke" / "data_receipt.json", receipt)
    if not passed:
        raise RuntimeError(f"data smoke failed: {receipt}")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def unavailable_stage(stage: str) -> None:
    raise RuntimeError(
        f"stage {stage!r} is not implemented yet; refusing to emit a placeholder result"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "model-smoke", "fit-lens", "positive-control", "prefix-value", "causal-patch", "full"),
        default="smoke",
    )
    args = parser.parse_args()
    config = load_config()
    if args.stage == "smoke":
        run_smoke(config)
        return 0
    design_boundary_receipt(config)
    unavailable_stage(args.stage)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
