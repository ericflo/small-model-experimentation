#!/usr/bin/env python3
"""Run the locked no-training semantic-policy headroom tournament."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import repo_tasks  # noqa: E402

FROZEN_FILES = (
    "configs/default.yaml",
    "idea_intake.md",
    "reports/preregistration.md",
    "reports/design_review.md",
    "src/repo_tasks.py",
    "src/repo_agent.py",
    "scripts/eval_repo_agent.py",
    "scripts/analyze.py",
    "scripts/run.py",
)


def config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(argv: list[str], allowed: tuple[int, ...] = (0,)) -> int:
    print("[run] " + " ".join(argv), flush=True)
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if completed.returncode not in allowed:
        raise subprocess.CalledProcessError(completed.returncode, argv)
    return completed.returncode


def verify_design_lock() -> dict:
    path = EXP / "runs" / "preregistration_receipt.json"
    if not path.is_file():
        raise SystemExit("preregistration receipt missing; model evaluation is illegal")
    payload = json.loads(path.read_text())
    if payload.get("status") != "locked" or tuple(payload.get("frozen_file_order", ())) != FROZEN_FILES:
        raise SystemExit("invalid preregistration receipt")
    for relative, expected in payload["frozen_files"].items():
        observed = sha256_file(EXP / relative)
        if observed != expected:
            raise SystemExit(f"frozen design changed: {relative}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", payload["design_commit"], "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise SystemExit("design commit is not an ancestor of HEAD")
    return payload


def write_design_lock(commit: str) -> None:
    if subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT, check=False
    ).returncode:
        raise SystemExit(f"unknown design commit: {commit}")
    paths = [str(EXP / relative) for relative in FROZEN_FILES]
    status = subprocess.run(
        ["git", "status", "--short", "--", *paths],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if status:
        raise SystemExit(f"frozen files are not committed:\n{status}")
    receipt = {
        "schema_version": 1,
        "status": "locked",
        "experiment_id": EXP.name,
        "design_commit": commit,
        "frozen_file_order": list(FROZEN_FILES),
        "frozen_files": {relative: sha256_file(EXP / relative) for relative in FROZEN_FILES},
        "model_output_precedes_lock": False,
        "note": "No Qwen output or training exists for this qualification before lock.",
    }
    output = EXP / "runs" / "preregistration_receipt.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


def cpu_smoke() -> dict:
    cfg = config()
    registered = tuple(cfg["families"]["all_headroom"])
    if registered != repo_tasks.HEADROOM_FAMILIES:
        raise AssertionError("headroom family registry mismatch")
    explicit = tuple(cfg["families"]["explicit_controls"])
    if explicit != repo_tasks.HEADROOM_EXPLICIT_FAMILIES:
        raise AssertionError("explicit control registry mismatch")
    for axis, families in repo_tasks.HEADROOM_AXES.items():
        if tuple(cfg["families"][f"inferred_{axis}"]) != families:
            raise AssertionError(f"axis registry mismatch: {axis}")

    tasks = repo_tasks.make_tasks(registered, 1, 88100, "smoke")
    for task in tasks:
        for state, expected in (
            ("initial", (False, False)),
            ("partial", (False, False)),
            ("oracle", (True, True)),
        ):
            env = repo_tasks.RepoEnv(task)
            try:
                if state == "partial":
                    env.apply_partial()
                elif state == "oracle":
                    env.apply_oracle()
                observed = env.visible_pass(), env.hidden_pass()
                if observed != expected:
                    raise AssertionError((task.task_id, state, observed))
            finally:
                env.close()
    bank.assert_firewall_clean([task.public_manifest() for task in tasks], tasks)

    blocks = {}
    for name in ("headroom_a", "headroom_b"):
        block = cfg["evaluation"]["blocks"][name]
        generated = repo_tasks.make_tasks(
            registered,
            int(block["tasks_per_family"]),
            int(block["seed"]),
            name,
        )
        digests = [repo_tasks.content_digest(task) for task in generated]
        if len(digests) != len(set(digests)):
            raise AssertionError(f"{name} contains duplicate public content")
        blocks[name] = digests
    if set(blocks["headroom_a"]) & set(blocks["headroom_b"]):
        raise AssertionError("qualification blocks overlap")
    return {
        "schema_version": 1,
        "status": "PASS",
        "families": len(registered),
        "axes": sorted(repo_tasks.HEADROOM_AXES),
        "representations": ["bundle", "record", "tuple"],
        "tasks_selftested": len(tasks),
        "initial_partial_fail_oracle_pass": True,
        "firewall_clean": True,
        "block_content_counts": {name: len(rows) for name, rows in blocks.items()},
        "blocks_content_disjoint": True,
        "model_output_generated": False,
        "menagerie_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--lock-design", metavar="COMMIT")
    args = parser.parse_args()
    if sum((args.smoke, args.gpu_smoke, args.full, bool(args.lock_design))) != 1:
        parser.error("choose exactly one mode")
    if args.smoke:
        receipt = cpu_smoke()
        output = EXP / "reports" / "smoke_receipt.json"
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.lock_design:
        write_design_lock(args.lock_design)
        return 0

    verify_design_lock()
    cfg = config()
    model = resolve(cfg["model"]["checkpoint"])
    observed_hash = sha256_file(model / "model.safetensors")
    if observed_hash != cfg["model"]["weight_sha256"]:
        raise SystemExit(f"model weight hash mismatch: {observed_hash}")
    vpy = str(ROOT / ".venv-vllm" / "bin" / "python")
    py = str(ROOT / ".venv" / "bin" / "python")
    artifacts = resolve(cfg["artifacts"]["root"])

    if args.gpu_smoke:
        output = artifacts / "smoke" / "eval.json"
        if not output.exists():
            command([
                vpy,
                str(EXP / "scripts" / "eval_repo_agent.py"),
                "--arm", "parent_smoke",
                "--model", str(model),
                "--block", "headroom_a",
                "--scenario-set", "recovery",
                "--mode", "deep",
                "--tasks-per-family", "1",
                "--output", str(output),
            ])
        if json.loads(output.read_text())["aggregate"]["n_cases"] != 24:
            raise SystemExit("GPU smoke did not cover all family/state cells")
        return 0

    paths = {}
    for block in ("headroom_a", "headroom_b"):
        output = artifacts / "eval" / f"{block}_parent_deep.json"
        if not output.exists():
            command([
                vpy,
                str(EXP / "scripts" / "eval_repo_agent.py"),
                "--arm", "parent",
                "--model", str(model),
                "--block", block,
                "--scenario-set", "recovery",
                "--mode", "deep",
                "--output", str(output),
            ])
        paths[block] = output
    analysis = EXP / "analysis" / "qualification.json"
    code = command([
        py,
        str(EXP / "scripts" / "analyze.py"),
        "--block-a", str(paths["headroom_a"]),
        "--block-b", str(paths["headroom_b"]),
        "--out", str(analysis),
    ], allowed=(0, 4))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
