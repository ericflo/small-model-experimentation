#!/usr/bin/env python3
"""Checkpointed harness for the repair-verifier 2AFC feasibility probe.

Lifecycle 21, pure eval-only cell: ONE inherited, externally published
composite (hygiene_explore, tree 9eb653d7...) is measured on ONE frozen
200-item 2AFC verification instrument under TWO decode configs (``think``
natural / ``nothink`` off) — 400 judgments, two sequential authenticated
engine events. This experiment trains nothing, merges nothing, and
promotes nothing; there is NO benchmark stage and NO aggregate seed. The
only stage is ``local`` and the ordered SIGNAL_PRESENT / SIGNAL_ABSENT
readout receipt is the terminal artifact. ``--smoke`` additionally
authenticates the standalone lineage package
(``rebuild_lineage.py --verify-inputs``).
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
SEED = 77160
ROWS = 200
LABELS = ("think", "nothink")
VERDICTS = ("SIGNAL_PRESENT", "SIGNAL_ABSENT")
# The one evaluated model is an inherited, externally published composite:
# this experiment trains nothing and merges nothing. The committed external
# merge receipt is the smoke-verified prerequisite (fail-closed hash pin).
EXTERNAL_MERGE_RECEIPTS = {
    "experiments/qwen35_4b_hygiene_explore_destack_medium/runs/merges/hygiene_explore.json": (
        "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a"
    ),
}
LOCAL_DESIGN = EXP / "data" / "local_design_receipt.json"
LOCAL_REVIEW = EXP / "reports" / "local_design_review.md"
LOCAL_RECEIPT = EXP / "runs" / "local" / "probe.json"
READOUT_RECEIPT = EXP / "runs" / "local" / "probe_readout.json"
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
        or local.get("seed") != SEED
        or local.get("rows_per_arm") != ROWS
        or local.get("labels") != list(LABELS)
        or not isinstance(rows, list)
        or len(rows) != ROWS * len(LABELS)
        or local.get("benchmark_data_read") is not False
        or not LOCAL_DESIGN.is_file()
        or local.get("design_receipt_sha256") != sha256_file(LOCAL_DESIGN)
    ):
        raise SystemExit("published local receipt failed smoke authentication")
    raw_artifacts = local.get("raw_artifacts") or {}
    if set(raw_artifacts) != {f"{label}_seed{SEED}" for label in LABELS}:
        raise SystemExit("published local raw-artifact key set changed")
    for key, artifact in sorted(raw_artifacts.items()):
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
        consequence = readings.get("consequence") or {}
        per_arm = readings.get("per_arm") or {}
        if (
            readout.get("seed") != SEED
            or readout.get("local_receipt_sha256") != sha256_file(LOCAL_RECEIPT)
            or readout.get("benchmark_data_read") is not False
            or readout.get("aggregate_seed") is not None
            or readout.get("aggregate_seed_open") is not False
            or readout.get("promoted") is not None
            or readout.get("outcome") != "PROBE_READ_COMPLETE"
            or readout.get("gating_arm") != "think"
            or consequence.get("verdict") not in VERDICTS
            or consequence.get("ordered_total_no_third_state") is not True
            or set(per_arm) != set(LABELS)
            or any(
                not isinstance(
                    (per_arm.get(label) or {}).get("2afc_accuracy"), (int, float)
                )
                or not isinstance((per_arm.get(label) or {}).get("ci95_exact"), list)
                for label in LABELS
            )
        ):
            raise SystemExit(
                "published probe readout receipt failed smoke authentication"
            )


def smoke() -> None:
    for relative, expected in sorted(EXTERNAL_MERGE_RECEIPTS.items()):
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"published external merge receipt changed: {path}")
    if LOCAL_DESIGN.exists():
        # Recomputes the inherited composite's tree manifest against the
        # frozen pins in addition to regenerating the 200-item probe set,
        # its 2AFC construction re-verification, and the overlap receipts
        # byte-identically.
        run([sys.executable, "-B", str(SCRIPTS / "gen_local_gate.py"), "--check"])
        if LOCAL_REVIEW.exists():
            require_verdict(LOCAL_REVIEW, LOCAL_VERDICT, "local adversarial review")
    smoke_local_receipts()
    # Standalone-reproducibility gate: authenticate the lineage package
    # (manifest schema, warm-start chaining, the six copied stage datasets,
    # the trainer and merger copies, the vendored frozen root adapter)
    # without touching any model. Fast, no GPU.
    run([sys.executable, "-B", str(SCRIPTS / "rebuild_lineage.py"), "--verify-inputs"])
    with tempfile.TemporaryDirectory() as scratch:
        for path in sorted(SCRIPTS.glob("*.py")) + sorted(
            (SCRIPTS / "lineage_trainers").glob("*.py")
        ):
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
        "PASS: eval-only feasibility-probe design, one inherited-composite "
        "pin, the frozen 200-item position-balanced SYMMETRIC 2AFC "
        "instrument (no repair history; marker-token and self-reference "
        "audited; per-item fix re-execution; overlap receipts), the ordered "
        "two-state consequence partition with the preregistered cap-contact "
        "scope, and the standalone lineage package"
    )


def local_stage() -> None:
    """Run the frozen two-run probe event and read the ordered consequence.

    eval_local_vllm.py refuses to start if ANY local artifact already exists
    and, on success, always writes BOTH the local receipt and the readout
    receipt (via check_local's shared finalize_probe writer). The
    independent check_local invocation below therefore only ever VERIFIES
    an existing pair. POST-CRASH RECOVERY (the one window where probe.json
    exists without its readout: a crash between the eval's two writes):
    audit the preserved artifacts, then recover the readout with
    ``check_local.py runs/local/probe.json --out
    runs/local/probe_readout.json`` — the shared writer keeps the recovered
    receipt schema-identical — and re-run --smoke.
    """
    for relative in (
        f"{EXP_REL}/data/local_design_receipt.json",
        f"{EXP_REL}/reports/local_design_review.md",
        *sorted(EXTERNAL_MERGE_RECEIPTS),
    ):
        require_pushed_checkpoint(relative)
    require_verdict(LOCAL_REVIEW, LOCAL_VERDICT, "local adversarial review")
    run([sys.executable, "-B", str(SCRIPTS / "eval_local_vllm.py")])
    if not LOCAL_RECEIPT.is_file():
        raise SystemExit(
            "local evaluation did not produce the frozen local receipt; "
            "see local_stage's docstring for the post-crash recovery path"
        )
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
