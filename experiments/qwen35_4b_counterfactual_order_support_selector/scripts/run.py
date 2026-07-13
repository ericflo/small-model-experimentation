#!/usr/bin/env python3
"""Run the frozen counterfactual order-support selector stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

from selector import analyze, load_jsonl, sha256_file, validate_and_group  # noqa: E402


CONFIG_PATH = EXP / "configs" / "default.yaml"
SELECTOR_PATH = EXP / "src" / "selector.py"
RUNNER_PATH = EXP / "scripts" / "run.py"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _implementation_hashes() -> dict[str, str]:
    return {
        "config_sha256": sha256_file(CONFIG_PATH),
        "selector_sha256": sha256_file(SELECTOR_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
    }


def _paths(split: str) -> tuple[Path, Path]:
    directory = EXP / "data" / split
    return directory / "real.jsonl", directory / "shuffled.jsonl"


def _validate_split(split: str, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    real_path, shuffled_path = _paths(split)
    if not real_path.is_file() or not shuffled_path.is_file():
        raise RuntimeError(f"{split} artifacts are absent")
    expected = config["source"][split]
    observed_hashes = {
        "real_sha256": sha256_file(real_path),
        "shuffled_sha256": sha256_file(shuffled_path),
    }
    if observed_hashes != expected:
        raise RuntimeError(f"{split} source hash mismatch")
    grouped, receipt = validate_and_group(
        load_jsonl(real_path),
        load_jsonl(shuffled_path),
        aliases=list(config["source"]["aliases"]),
        expected_tasks=int(config["source"]["expected_tasks"]),
        traces_per_task=int(config["source"]["traces_per_task"]),
    )
    receipt.update(observed_hashes)
    return grouped, receipt


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=REPO)


def _confirmation_authorized() -> dict[str, Any]:
    qualification_path = EXP / "runs" / "qualification.json"
    boundary_path = EXP / "runs" / "confirmation_boundary.json"
    if not qualification_path.is_file() or not boundary_path.is_file():
        raise RuntimeError("confirmation requires qualification and boundary receipts")
    qualification = json.loads(qualification_path.read_text())
    boundary = json.loads(boundary_path.read_text())
    if qualification.get("passed") is not True or qualification.get("decision") != "ORDER_SUPPORT_QUALIFIED":
        raise RuntimeError("qualification did not authorize confirmation")
    qualification_sha = hashlib.sha256(qualification_path.read_bytes()).hexdigest()
    if boundary.get("qualification_sha256") != qualification_sha:
        raise RuntimeError("confirmation boundary does not lock current qualification")
    if boundary.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("confirmation implementation changed after boundary")
    commit = str(boundary.get("anchored_commit", ""))
    if not commit:
        raise RuntimeError("confirmation boundary has no anchored commit")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO
    )
    relative_qualification = qualification_path.relative_to(REPO).as_posix()
    committed_qualification = _git("show", f"{commit}:{relative_qualification}")
    if hashlib.sha256(committed_qualification).hexdigest() != qualification_sha:
        raise RuntimeError("anchored commit does not contain locked qualification")
    for path, key in (
        (CONFIG_PATH, "config_sha256"),
        (SELECTOR_PATH, "selector_sha256"),
        (RUNNER_PATH, "runner_sha256"),
    ):
        relative = path.relative_to(REPO).as_posix()
        committed_bytes = _git("show", f"{commit}:{relative}")
        if hashlib.sha256(committed_bytes).hexdigest() != boundary["implementation_hashes"][key]:
            raise RuntimeError(f"anchored implementation hash mismatch for {key}")
    relative_boundary = boundary_path.relative_to(REPO).as_posix()
    if _git("show", f"HEAD:{relative_boundary}") != boundary_path.read_bytes():
        raise RuntimeError("confirmation boundary is not committed at HEAD")
    return boundary


def run_smoke(config: dict[str, Any]) -> None:
    _grouped, receipt = _validate_split("qualification", config)
    confirmation_present = any(path.exists() for path in _paths("confirmation"))
    qualification_path = EXP / "runs" / "qualification.json"
    if confirmation_present and not qualification_path.exists():
        raise RuntimeError("confirmation appeared before qualification")
    result = {
        "schema_version": 1,
        "stage": "smoke",
        "passed": True,
        "qualification": receipt,
        "confirmation_present": confirmation_present,
        "confirmation_opened": False,
        "model_loaded": False,
        "outcome_metrics_computed": False,
        "implementation_hashes": _implementation_hashes(),
    }
    _write_json(EXP / "runs" / "smoke.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def run_analysis(split: str, config: dict[str, Any]) -> None:
    if split == "qualification":
        if any(path.exists() for path in _paths("confirmation")):
            raise RuntimeError("qualification requires confirmation artifacts to remain absent")
        boundary = None
    else:
        boundary = _confirmation_authorized()
    grouped, receipt = _validate_split(split, config)
    result = analyze(
        grouped,
        aliases=list(config["source"]["aliases"]),
        gates=config["gates"],
        bootstrap_resamples=int(config["statistics"]["bootstrap_resamples"]),
        seed=int(config["statistics"]["seed"]),
        split=split,
    )
    task_rows = result.pop("task_rows")
    result.update(
        {
            "schema_version": 1,
            "source_receipt": receipt,
            "implementation_hashes": _implementation_hashes(),
            "confirmation_opened": split == "confirmation",
            "confirmation_boundary": boundary,
            "model_loaded": False,
            "matched_compute_capability_claim": False,
        }
    )
    _write_json(EXP / "runs" / f"{split}.json", result)
    _write_jsonl(EXP / "analysis" / f"{split}_task_predictions.jsonl", task_rows)
    print(json.dumps(result, indent=2, sort_keys=True))


def anchor_confirmation(config: dict[str, Any], commit: str) -> None:
    if any(path.exists() for path in _paths("confirmation")):
        raise RuntimeError("remove confirmation artifacts before anchoring")
    qualification_path = EXP / "runs" / "qualification.json"
    if not qualification_path.is_file():
        raise RuntimeError("qualification receipt is absent")
    qualification = json.loads(qualification_path.read_text())
    if qualification.get("passed") is not True or qualification.get("decision") != "ORDER_SUPPORT_QUALIFIED":
        raise RuntimeError("qualification did not pass")
    resolved = _git("rev-parse", commit).decode().strip()
    head = _git("rev-parse", "HEAD").decode().strip()
    if resolved != head:
        raise RuntimeError("confirmation must anchor the current pushed HEAD")
    relative = qualification_path.relative_to(REPO).as_posix()
    committed = _git("show", f"{resolved}:{relative}")
    qualification_sha = hashlib.sha256(qualification_path.read_bytes()).hexdigest()
    if hashlib.sha256(committed).hexdigest() != qualification_sha:
        raise RuntimeError("qualification is not committed at the anchor")
    receipt = {
        "schema_version": 1,
        "anchored_commit": resolved,
        "qualification_sha256": qualification_sha,
        "implementation_hashes": _implementation_hashes(),
        "expected_confirmation_hashes": config["source"]["confirmation"],
        "confirmation_artifacts_present": False,
        "authorization": "frozen_retrospective_confirmation_only",
    }
    _write_json(EXP / "runs" / "confirmation_boundary.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("smoke", "qualification", "anchor-confirmation", "confirmation"),
    )
    parser.add_argument("--commit", default="HEAD")
    args = parser.parse_args()
    config = _config()
    if args.stage == "smoke":
        run_smoke(config)
    elif args.stage == "qualification":
        run_analysis("qualification", config)
    elif args.stage == "anchor-confirmation":
        anchor_confirmation(config, args.commit)
    else:
        run_analysis("confirmation", config)


if __name__ == "__main__":
    main()
