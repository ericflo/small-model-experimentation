#!/usr/bin/env python3
"""Checkpointed harness for the retention-screen calibration study.

Pure eval calibration cell: FIVE inherited, externally published composites
are re-measured on FOUR fresh retention-only screens (5 x 4 = 20
authenticated engine events). This experiment trains nothing, merges
nothing, and promotes nothing; there is NO benchmark stage and NO aggregate
seed. The only stage is ``local`` and the consolidated calibration receipt
is the terminal artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
EXP_REL = f"experiments/{EXP.name}"
LOCAL_SEEDS = (88022, 88023, 88024, 88025)
LOCAL_ROWS = 104
LOCAL_LABELS = (
    "axis160_direct",
    "axis160_r64",
    "clean_parent",
    "hygiene_explore_direct",
    "replay_clean",
)
PROTOCOLS = ("single_screen", "pooled_k2", "pooled_k3")
# Every arm is an inherited, externally published composite: this experiment
# trains nothing and merges nothing. The committed external merge receipts
# are the smoke-verified prerequisites (fail-closed hash pins).
EXTERNAL_MERGE_RECEIPTS = {
    "experiments/qwen35_4b_dose_diversity_mechanism_cell/runs/merges/axis160_direct.json": (
        "7b878bb357e044c58a5ba27f34365906059259237e657ca77c2ad2e8fb77ea39"
    ),
    "experiments/qwen35_4b_rank_capacity_vehicle_cell/runs/merges/axis160_r64.json": (
        "bf0032ea7e9d11c819812f9a54025fce8b23c0921032b186098fdc83a77c5e40"
    ),
    "experiments/qwen35_4b_universal_fresh_surface_budget_commit_target_match/runs/merges/designed_fresh.json": (
        "ab3f20cc93d3fe21ead7a1d573edbca2903d59d6f9fe3d2af0c93e823676acc2"
    ),
    "experiments/qwen35_4b_hygiene_explore_destack_medium/runs/merges/hygiene_explore.json": (
        "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a"
    ),
    "experiments/qwen35_4b_hygiene_explore_destack_medium/runs/merges/replay_clean.json": (
        "24367084da5415ec8c3f922202a2028cc2930c4b99d87ea5e32b93e5c3b90332"
    ),
}
LOCAL_DESIGN = EXP / "data" / "local_design_receipt.json"
LOCAL_REVIEW = EXP / "reports" / "local_design_review.md"
LOCAL_RECEIPT = EXP / "runs" / "local" / "calibration.json"
READOUT_RECEIPT = EXP / "runs" / "local" / "calibration_readout.json"
LOCAL_VERDICT = "**Verdict:** `PASS_LOCAL_EVENT`."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"not a JSON object: {path}")
    return payload


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def output(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def require_pushed_checkpoint(relative_path: str) -> None:
    path = ROOT / relative_path
    status = output(["git", "status", "--short"])
    branch = output(["git", "branch", "--show-current"])
    head = output(["git", "rev-parse", "HEAD"])
    origin = output(["git", "rev-parse", "origin/main"])
    if status or branch != "main" or head != origin:
        raise SystemExit("stage requires a clean pushed main checkpoint")
    run(["git", "cat-file", "-e", f"HEAD:{relative_path}"])
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if not path.is_file() or path.read_bytes() != committed:
        raise SystemExit(f"stage prerequisite differs from HEAD: {relative_path}")


def require_verdict(path: Path, verdict: str, description: str) -> None:
    if not path.is_file() or verdict not in path.read_text(encoding="utf-8"):
        raise SystemExit(f"{description} has not been authorized: {path}")


def smoke_local_receipts() -> None:
    if not LOCAL_RECEIPT.exists():
        return
    local = load_json(LOCAL_RECEIPT)
    rows = local.get("rows")
    if (
        local.get("experiment_id") != EXP.name
        or local.get("seeds") != list(LOCAL_SEEDS)
        or local.get("rows_per_arm_per_screen") != LOCAL_ROWS
        or local.get("labels") != list(LOCAL_LABELS)
        or not isinstance(rows, list)
        or len(rows) != LOCAL_ROWS * len(LOCAL_LABELS) * len(LOCAL_SEEDS)
        or local.get("benchmark_data_read") is not False
        or not LOCAL_DESIGN.is_file()
        or local.get("design_receipt_sha256") != sha256_file(LOCAL_DESIGN)
    ):
        raise SystemExit("published local receipt failed smoke authentication")
    for key, artifact in sorted((local.get("raw_artifacts") or {}).items()):
        for name in ("output", "metadata", "log"):
            expected = artifact.get(f"{name}_sha256")
            if expected is None:
                continue
            path = Path(artifact.get(name, ""))
            if not path.is_file() or sha256_file(path) != expected:
                raise SystemExit(f"published local artifact changed for {key}: {path}")
    if READOUT_RECEIPT.exists():
        readout = load_json(READOUT_RECEIPT)
        readings = readout.get("readings") or {}
        flags = readings.get("stability_flags")
        if (
            readout.get("seeds") != list(LOCAL_SEEDS)
            or readout.get("local_receipt_sha256") != sha256_file(LOCAL_RECEIPT)
            or readout.get("benchmark_data_read") is not False
            or readout.get("aggregate_seed") is not None
            or readout.get("aggregate_seed_open") is not False
            or readout.get("promoted") is not None
            or readout.get("outcome") != "CALIBRATION_READ_COMPLETE"
            or readout.get("adjudication_protocol") not in PROTOCOLS
            or readings.get("adjudication_protocol")
            != readout.get("adjudication_protocol")
            or not isinstance(readings.get("screen_sd_pooled"), (int, float))
            or not isinstance(readings.get("recommended_band"), int)
            or readings.get("recommended_band") < 5
            or not isinstance(flags, list)
            or not flags
            or any(not isinstance(flag.get("inside"), bool) for flag in flags)
        ):
            raise SystemExit(
                "published calibration readout receipt failed smoke authentication"
            )


def smoke() -> None:
    for relative, expected in sorted(EXTERNAL_MERGE_RECEIPTS.items()):
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"published external merge receipt changed: {path}")
    if LOCAL_DESIGN.exists():
        # Recomputes all five inherited composite tree manifests against the
        # frozen pins in addition to regenerating the four screens
        # byte-identically.
        run([sys.executable, "-B", str(SCRIPTS / "gen_local_gate.py"), "--check"])
        if LOCAL_REVIEW.exists():
            require_verdict(LOCAL_REVIEW, LOCAL_VERDICT, "local adversarial review")
    smoke_local_receipts()
    with tempfile.TemporaryDirectory() as scratch:
        for path in sorted(SCRIPTS.glob("*.py")):
            try:
                py_compile.compile(
                    str(path),
                    cfile=str(Path(scratch) / (path.name + "c")),
                    doraise=True,
                )
            except py_compile.PyCompileError as error:
                raise SystemExit(f"compile check failed: {error}")
    tests_dir = EXP / "tests"
    if tests_dir.is_dir():
        run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(tests_dir), "-v"])
    print(
        "PASS: eval-only calibration design, five inherited-composite pins, "
        "four frozen retention screens, and the frozen twenty-run "
        "calibration-event contracts"
    )


def local_stage() -> None:
    for relative in (
        f"{EXP_REL}/data/local_design_receipt.json",
        f"{EXP_REL}/reports/local_design_review.md",
        *sorted(EXTERNAL_MERGE_RECEIPTS),
    ):
        require_pushed_checkpoint(relative)
    require_verdict(LOCAL_REVIEW, LOCAL_VERDICT, "local adversarial review")
    run([sys.executable, "-B", str(SCRIPTS / "eval_local_vllm.py")])
    if not LOCAL_RECEIPT.is_file():
        raise SystemExit("local evaluation did not produce the frozen local receipt")
    if READOUT_RECEIPT.exists():
        run([sys.executable, "-B", str(SCRIPTS / "check_local.py"), str(LOCAL_RECEIPT)])
    else:
        run([
            sys.executable,
            "-B",
            str(SCRIPTS / "check_local.py"),
            str(LOCAL_RECEIPT),
            "--out",
            str(READOUT_RECEIPT),
        ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--stage", choices=("local",))
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return 0
    if args.stage == "local":
        local_stage()
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
