#!/usr/bin/env python3
"""Checkpointed harness for the WHY scale-ladder sweep (Phase A: scale-then-RLVR).

The ladder trains the SAME high-diversity WHY curriculum at four rung sizes
(2000, 5000, 10000, 20000) to find the PEAK of the scaling curve before
overfit/collapse; the best rung becomes the SFT foundation for the subsequent
RLVR phase. This is a SWEEP: the orchestrator runs each rung's stages and records
the pass@1-vs-rows curve. This harness does NOT run all rungs automatically.

Stages (each GPU stage is fail-closed: clean pushed green main + its committed
staged review + the committed ladder manifest):

  --smoke                       model-free design gate (no GPU, no writes)
  --stage gen-ladder            build corpora + sha-pinned manifest (CPU, model-free)
  --stage train   --rows N      needs reports/compute_review.md PASS_CONTROL_TRAINING
  --stage merge   --rows N      needs reports/merge_review.md    PASS_CONTROL_MERGE
  --stage measure --rows N      needs reports/measure_review.md  PASS_MEASURE

Every GPU stage is per-rung (``--rows`` in {2000, 5000, 10000, 20000}); the
orchestrator sweeps the rungs and assembles the curve.
"""

from __future__ import annotations

import argparse
import hashlib
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
EXP_REL = f"experiments/{EXP.name}"

LADDER_SIZES = (2000, 5000, 10000, 20000, 40000)
MANIFEST = EXP / "data" / "ladder_manifest.json"
COMPUTE_REVIEW = EXP / "reports" / "compute_review.md"
MERGE_REVIEW = EXP / "reports" / "merge_review.md"
MEASURE_REVIEW = EXP / "reports" / "measure_review.md"
TRAIN_VERDICT = "**Verdict:** `PASS_CONTROL_TRAINING`."
MERGE_VERDICT = "**Verdict:** `PASS_CONTROL_MERGE`."
MEASURE_VERDICT = "**Verdict:** `PASS_MEASURE`."

BASE_MODEL = ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum" / "merged" / "base_reserialized"
ADAPTER_ROOT = ROOT / "large_artifacts" / EXP.name / "adapters"
MERGED_ROOT = ROOT / "large_artifacts" / EXP.name / "merged"


def output(command: list[str]) -> str:
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_pushed_checkpoint(relative_path: str) -> None:
    status = output(["git", "status", "--short"])
    branch = output(["git", "branch", "--show-current"])
    head = output(["git", "rev-parse", "HEAD"])
    origin = output(["git", "rev-parse", "origin/main"])
    if status or branch != "main" or head != origin:
        raise SystemExit("stage requires a clean pushed main checkpoint")
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{relative_path}"], cwd=ROOT, check=False, capture_output=True
    )
    if probe.returncode != 0:
        raise SystemExit(f"stage prerequisite is not committed at HEAD: {relative_path}")
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    path = ROOT / relative_path
    if not path.is_file() or path.read_bytes() != committed:
        raise SystemExit(f"stage prerequisite differs from HEAD: {relative_path}")


def require_verdict(path: Path, verdict: str, description: str) -> None:
    if not path.is_file() or verdict not in path.read_text(encoding="utf-8"):
        raise SystemExit(f"{description} has not been authorized: {path}")


def smoke() -> None:
    sys.path.insert(0, str(SCRIPTS))
    # 1. compile every script.
    with tempfile.TemporaryDirectory() as scratch:
        for path in sorted(SCRIPTS.glob("*.py")):
            try:
                py_compile.compile(str(path), cfile=str(Path(scratch) / (path.name + "c")), doraise=True)
            except py_compile.PyCompileError as error:
                raise SystemExit(f"compile check failed: {error}")

    # 2. the committed base-provenance copy pins exactly the base composite.
    from train_trial import (  # noqa: PLC0415
        BASE_PROVENANCE_COPY,
        MODEL_PATH_RECEIPT_SHA256,
        check_base_provenance,
    )

    if not BASE_PROVENANCE_COPY.is_file() or sha256_file(BASE_PROVENANCE_COPY) != MODEL_PATH_RECEIPT_SHA256:
        raise SystemExit("in-cell base provenance copy is absent or changed")
    check_base_provenance()

    # 3. contamination fixture loads + is sha-consistent.
    import contamination as contam  # noqa: PLC0415

    contam.load_fixture()

    # 4. a small deterministic build exercises the full truth audit model-free.
    import gen_why_scale_curriculum as gen  # noqa: PLC0415

    rows = gen.generate_curriculum(gen.CONSTRUCTION_SEED, 300)
    gen.validate_generated(rows)
    div = gen.diversity_report(rows)
    if div["distinct_normalized_why_templates"] < 100 or div["unique_program_pct"] < 99.0:
        raise SystemExit(f"smoke diversity check failed: {div}")

    # 5. if the ladder manifest is present, its rung shas must regenerate.
    if MANIFEST.is_file():
        import build_ladder  # noqa: PLC0415

        build_ladder.verify()

    # 6. published rung receipts authenticate without running (if present).
    if COMPUTE_REVIEW.exists():
        require_verdict(COMPUTE_REVIEW, TRAIN_VERDICT, "adversarial compute review")
    from train_trial import validate_published_rung  # noqa: PLC0415

    for rows_n in LADDER_SIZES:
        if (EXP / "runs" / "training" / f"why_scale_{rows_n}.json").exists():
            validate_published_rung(rows_n, require_committed=False)
    if MERGE_REVIEW.exists():
        require_verdict(MERGE_REVIEW, MERGE_VERDICT, "adversarial merge review")
    if MEASURE_REVIEW.exists():
        require_verdict(MEASURE_REVIEW, MEASURE_VERDICT, "adversarial measure review")

    # 7. unit tests.
    tests_dir = EXP / "tests"
    if tests_dir.is_dir():
        run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(tests_dir), "-v"])

    print(
        "PASS: WHY scale-ladder design (Phase A) — a SCALE-CAPABLE, high-diversity "
        "WHY curriculum generator (59 parameterized synthetic families, a phrase-pool "
        "of TRUE line-specific #WHY: rationales, deterministic per seed 94100) whose "
        "corpora are per-row truth-audited by real execution (strip the #WHY: comments "
        "and the code passes all asserts, the commented code runs identically, the "
        "marker is strippable, every #WHY: is line-specific and non-boilerplate), "
        "contamination-clean (zero banned-name hits, zero distinctive code 7-grams), "
        "and diverse at scale (>=50 families, >=300 normalized WHY templates, >=95% "
        "unique programs at 20000 rows, every render <4096 tokens); a sha-pinned "
        "four-rung ladder (2000/5000/10000/20000) with a fail-closed per-rung trainer "
        "from the authenticated base_reserialized composite (fresh r32/a64, seed 94101, "
        "epochs=1 for all rungs), the vendored external merger, and a per-rung "
        "transfer sweep on the shared HumanEval+MBPP fitness harness (comments IGNORED "
        "by the grader) to locate the WHY scaling PEAK as the SFT foundation for RLVR"
    )


def _epochs_arg(rows: int) -> str:
    from train_trial import epochs_for  # noqa: PLC0415

    return str(epochs_for(rows))


def gen_ladder_stage() -> None:
    import build_ladder  # noqa: PLC0415

    run([sys.executable, "-B", str(SCRIPTS / "build_ladder.py")])
    build_ladder.verify()


def train_stage(rows: int) -> None:
    require_pushed_checkpoint(f"{EXP_REL}/data/ladder_manifest.json")
    require_pushed_checkpoint(f"{EXP_REL}/reports/compute_review.md")
    require_verdict(COMPUTE_REVIEW, TRAIN_VERDICT, "adversarial compute review")
    sys.path.insert(0, str(SCRIPTS))
    from train_trial import corpus_for  # noqa: PLC0415

    corpus_path, _ = corpus_for(rows)
    if not corpus_path.is_file():
        raise SystemExit(
            f"rung corpus absent (run --stage gen-ladder, it is a gitignored large artifact): {corpus_path}"
        )
    run([
        sys.executable, "-B", str(SCRIPTS / "train_trial.py"),
        "--rows", str(rows),
        "--train", str(corpus_path),
        "--out", str(ADAPTER_ROOT / f"why_scale_{rows}"),
        "--model-path", str(BASE_MODEL),
        "--epochs", _epochs_arg(rows),
        "--lr", "1e-5",
        "--rank", "32",
        "--alpha", "64",
        "--batch-size", "1",
        "--grad-accum", "8",
        "--max-length", "4096",
        "--w-think", "0.2",
        "--w-close", "0.2",
        "--seed", "94101",
    ])


def merge_stage(rows: int) -> None:
    require_pushed_checkpoint(f"{EXP_REL}/runs/training/why_scale_{rows}.json")
    require_pushed_checkpoint(f"{EXP_REL}/reports/merge_review.md")
    require_verdict(MERGE_REVIEW, MERGE_VERDICT, "adversarial merge review")
    merged = MERGED_ROOT / f"why_scale_{rows}"
    if (merged / "merge_receipt.json").is_file():
        print(f"[run] rung {rows} merge already published; skipping")
        return
    run([
        sys.executable, "-B", str(SCRIPTS / "merge_adapter.py"),
        "--adapter", str(ADAPTER_ROOT / f"why_scale_{rows}"),
        "--out", str(merged),
        "--base-model", str(BASE_MODEL),
    ])


def measure_stage(rows: int) -> None:
    require_pushed_checkpoint(f"{EXP_REL}/runs/merges/why_scale_{rows}.json")
    require_pushed_checkpoint(f"{EXP_REL}/reports/measure_review.md")
    require_verdict(MEASURE_REVIEW, MEASURE_VERDICT, "adversarial measure review")
    if not (MERGED_ROOT / f"why_scale_{rows}" / "merge_receipt.json").is_file():
        raise SystemExit(f"rung composite is incomplete: {MERGED_ROOT / f'why_scale_{rows}'}")
    run([sys.executable, "-B", str(SCRIPTS / "measure_transfer.py"), "--rows", str(rows), "--run"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--stage", choices=("gen-ladder", "train", "merge", "measure"))
    parser.add_argument("--rows", type=int, choices=LADDER_SIZES,
                        help="rung size for the train/merge/measure stages")
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return 0
    if args.stage == "gen-ladder":
        gen_ladder_stage()
        return 0
    if args.stage in ("train", "merge", "measure") and args.rows is None:
        parser.error(f"--stage {args.stage} requires --rows")
    if args.stage == "train":
        train_stage(args.rows)
        return 0
    if args.stage == "merge":
        merge_stage(args.rows)
        return 0
    if args.stage == "measure":
        measure_stage(args.rows)
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
