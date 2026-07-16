#!/usr/bin/env python3
"""Checkpointed harness for the zero-root lineage rebuild measurement.

Lifecycle 22 — the provenance question elevated to a measurement: the
six documented contamination-free stages of the hygiene_explore
composite, replayed from a FRESH zero-initialized adapter on the
official base (removing the undocumented C53-era gym-line blend root),
then measured ONCE at medium/tb1024 on the single sealed fresh seed
78159 against the original composite and the untouched base. This
experiment promotes nothing; there is no local gate and no pass bar —
the frozen readout with its ordered consequence partition
(ZERO_ROOT_COMPARABLE / ZERO_ROOT_DEGRADED) is the terminal artifact.

``--smoke`` (no GPU, no writes) verifies every pin: the copied lineage
package via ``rebuild_zero_root.py --verify-inputs`` (byte-pinned
manifest, datasets, trainers, merger; blend root NOT vendored), the
trusted gateway, the design receipt + code pins via
``gen_design_receipt.py --check`` when the receipt exists, any published
zero-root stage/merge receipts (schema, manifest agreement, warm-start
chain continuity, on-disk adapter shas when the adapters are present),
any published benchmark event artifacts against the single-seed
write-ahead ledger (including requiring the terminal readout once the
seed closes), compiles every script, and runs the unit tests. It does
NOT require the reviews to exist.

``--stage rebuild`` requires the committed-at-HEAD design receipt and a
``reports/compute_review.md`` carrying the literal PASS_REBUILD verdict
on clean pushed main, then replays the six stages + the merge
(~2.5-3h GPU) writing per-stage receipts to ``runs/lineage/``. Commit
the receipts, fill the three TODO-pins in scripts/run_benchmark.py from
runs/lineage/merge.json, and seek the benchmark review.

``--stage benchmark`` requires the committed-at-HEAD design receipt,
the committed rebuild receipts (six stages + merge), and a
``reports/benchmark_design_review.md`` carrying the literal
PASS_BENCHMARK_EVENT verdict on clean pushed main, then runs the frozen
single-seed three-arm event. ``--resume`` is forwarded to
run_benchmark.py only when explicitly passed by the operator.
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
FROZEN_NAME = "zero_root"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
FROZEN_SEED = 78159
MODEL_ORDER = ("base", "hygiene_explore_original", "zero_root_hygiene_explore")
STAGE_DIRNAMES = (
    "stage01_replay_refresh",
    "stage02_designed160",
    "stage03_close_xi",
    "stage04_replay_after_close",
    "stage05_designed_fresh",
    "stage06_hygiene_explore",
)
MANIFEST_SHA256 = "1f49cd8b8706c8db858d30af1bf14fe09403971256514919bc24e7b6c47ff121"
# The original composite's committed external merge receipt is a
# smoke-verified prerequisite (fail-closed hash pin). Base's reserialization
# receipt lives inside the composite and is pinned by the design receipt.
EXTERNAL_MERGE_RECEIPTS = {
    "experiments/qwen35_4b_hygiene_explore_destack_medium/runs/merges/hygiene_explore.json": (
        "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a"
    ),
}
GATEWAY = ROOT / "scripts" / "run_benchmark_aggregate.py"
GATEWAY_SHA256 = "53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17"
DESIGN_RECEIPT = EXP / "data" / "design_receipt.json"
COMPUTE_REVIEW = EXP / "reports" / "compute_review.md"
REBUILD_VERDICT = "**Verdict:** `PASS_REBUILD`."
BENCH_REVIEW = EXP / "reports" / "benchmark_design_review.md"
BENCH_VERDICT = "**Verdict:** `PASS_BENCHMARK_EVENT`."
LINEAGE_RECEIPTS = EXP / "runs" / "lineage"
MERGE_RECEIPT = LINEAGE_RECEIPTS / "merge.json"
ADAPTER_ROOT = ROOT / "large_artifacts" / EXP.name / "adapters"
MERGED_OUT = ROOT / "large_artifacts" / EXP.name / "merged" / "zero_root_hygiene_explore"
EVENT_DIR = (
    EXP / "runs" / "benchmark"
    / f"{FROZEN_TIER}_tb{FROZEN_THINK_BUDGET}_seed{FROZEN_SEED}_{FROZEN_NAME}"
)
READOUT = EXP / "runs" / "benchmark" / "zero_root_readout.json"
LEDGER = EXP / "runs" / "benchmark_events.jsonl"


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


def smoke_lineage_receipts() -> None:
    """Authenticate any published zero-root stage/merge receipts.

    Receipts must appear as a contiguous prefix of the six-stage chain
    (a receipt for stage N without every earlier stage is corrupt), each
    must restate the manifest recipe with the zero-root rewiring (stage 1
    fresh, stages 2-6 chained by adapter sha), and when the adapters are
    present on disk their hashes must match. The merge receipt requires
    the complete chain and must pin the stage-6 adapter. The manifest's
    ORIGINAL per-stage hashes are contrast fields and are never compared
    against the produced adapters.
    """
    manifest = load_json(EXP / "data" / "lineage" / "lineage_manifest.json")
    published = [
        name
        for name in STAGE_DIRNAMES
        if (LINEAGE_RECEIPTS / f"{name}.json").is_file()
    ]
    expected_prefix = list(STAGE_DIRNAMES[: len(published)])
    if published != expected_prefix:
        raise SystemExit(
            "published stage receipts are not a contiguous prefix of the "
            f"six-stage chain: {published}"
        )
    previous_produced = None
    for index, name in enumerate(published, start=1):
        receipt = load_json(LINEAGE_RECEIPTS / f"{name}.json")
        row = manifest["stages"][index - 1]
        produced = receipt.get("produced", {})
        if (
            receipt.get("schema_version") != 1
            or receipt.get("experiment_id") != EXP.name
            or receipt.get("lineage") != "zero_root"
            or receipt.get("manifest_sha256") != MANIFEST_SHA256
            or receipt.get("stage") != index
            or receipt.get("name") != row["name"]
            or receipt.get("seed") != row["seed"]
            or receipt.get("hyperparameters") != row["hyperparameters"]
            or receipt.get("targeted_close_overrides")
            != row.get("targeted_close_overrides")
            or receipt.get("trainer")
            != {"path": row["trainer"], "sha256": row["trainer_sha256"]}
            or receipt.get("dataset", {}).get("sha256") != row["dataset"]["sha256"]
            or receipt.get("fresh_init") is not (index == 1)
            or not isinstance(produced.get("adapter_weights_sha256"), str)
            or not isinstance(produced.get("adapter_config_sha256"), str)
        ):
            raise SystemExit(
                f"published stage receipt failed smoke authentication: {name}"
            )
        warm = receipt.get("warm_start")
        if index == 1:
            if warm != "fresh_zero_init":
                raise SystemExit(
                    "stage 1 must be the fresh zero-init stage (no warm start)"
                )
        else:
            if (
                not isinstance(warm, dict)
                or warm.get("source") != f"stage {index - 1}"
                or warm.get("adapter_weights_sha256")
                != previous_produced["adapter_weights_sha256"]
                or warm.get("adapter_config_sha256")
                != previous_produced["adapter_config_sha256"]
            ):
                raise SystemExit(
                    f"published stage receipt breaks the zero-root warm-start "
                    f"chain: {name}"
                )
        adapter_dir = ADAPTER_ROOT / name
        if adapter_dir.exists():
            weights = adapter_dir / "adapter_model.safetensors"
            config = adapter_dir / "adapter_config.json"
            if (
                not weights.is_file()
                or not config.is_file()
                or sha256_file(weights) != produced["adapter_weights_sha256"]
                or sha256_file(config) != produced["adapter_config_sha256"]
            ):
                raise SystemExit(
                    f"on-disk adapter does not match its published receipt: {name}"
                )
        previous_produced = produced
    if MERGE_RECEIPT.is_file():
        if len(published) != len(STAGE_DIRNAMES):
            raise SystemExit(
                "merge receipt exists before all six stage receipts"
            )
        receipt = load_json(MERGE_RECEIPT)
        if (
            receipt.get("schema_version") != 1
            or receipt.get("experiment_id") != EXP.name
            or receipt.get("stage") != "merge"
            or receipt.get("name") != "zero_root_hygiene_explore"
            or receipt.get("manifest_sha256") != MANIFEST_SHA256
            or receipt.get("merger", {}).get("sha256")
            != manifest["merger"]["sha256"]
            or receipt.get("adapter", {}).get("adapter_weights_sha256")
            != previous_produced["adapter_weights_sha256"]
            or receipt.get("adapter", {}).get("adapter_config_sha256")
            != previous_produced["adapter_config_sha256"]
            or Path(receipt.get("merged", "")) != MERGED_OUT
            or receipt.get("weights_size_bytes") != 9_078_620_536
        ):
            raise SystemExit("published merge receipt failed smoke authentication")
        if MERGED_OUT.exists():
            weights = MERGED_OUT / "model.safetensors"
            if (
                not weights.is_file()
                or weights.stat().st_size != receipt["weights_size_bytes"]
            ):
                raise SystemExit(
                    "merged composite on disk disagrees with the merge receipt "
                    "(full tree recompute happens at event time)"
                )


def opened_record() -> dict:
    return {
        "name": FROZEN_NAME,
        "phase": "opened",
        "seed": FROZEN_SEED,
        "think_budget": FROZEN_THINK_BUDGET,
        "tier": FROZEN_TIER,
    }


def authenticate_closed_event(closed: dict) -> None:
    """The closed event: summary AND all three receipts must match the pins.

    The closed record sha-pins every consequence input at close time;
    smoke recomputes each file's sha256 from disk and requires exact
    record equality — a swapped receipt or summary fails here, not just
    at the readout.
    """
    summary_path = EVENT_DIR / "summary.json"
    if not summary_path.is_file():
        raise SystemExit("published summary is absent for the closed event")
    receipt_shas = {}
    for label in MODEL_ORDER:
        receipt = EVENT_DIR / f"{label}.json"
        if not receipt.is_file():
            raise SystemExit(
                f"published gateway receipt is absent for arm {label}"
            )
        receipt_shas[label] = sha256_file(receipt)
    if closed != {
        "name": FROZEN_NAME,
        "phase": "closed",
        "tier": FROZEN_TIER,
        "think_budget": FROZEN_THINK_BUDGET,
        "seed": FROZEN_SEED,
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "receipts": receipt_shas,
    }:
        raise SystemExit(
            "published benchmark ledger failed smoke authentication: the "
            "on-disk summary/receipt sha256s do not match the closed "
            "record's pins"
        )
    summary = load_json(summary_path)
    if (
        summary.get("name") != FROZEN_NAME
        or summary.get("tier") != FROZEN_TIER
        or summary.get("think_budget") != FROZEN_THINK_BUDGET
        or summary.get("seed") != FROZEN_SEED
        or summary.get("model_order") != list(MODEL_ORDER)
        or set(summary.get("scores", {})) != set(MODEL_ORDER)
        or set(summary.get("budget", {})) != set(MODEL_ORDER)
        or summary.get("promoted") is not None
        or summary.get("benchmark_data_read") is not False
        or summary.get("gateway_sha256") != GATEWAY_SHA256
        or not MERGE_RECEIPT.is_file()
        or summary.get("zero_root_merge_receipt_sha256") != sha256_file(MERGE_RECEIPT)
        or not DESIGN_RECEIPT.is_file()
        or summary.get("design_receipt_sha256") != sha256_file(DESIGN_RECEIPT)
    ):
        raise SystemExit("published benchmark summary failed smoke authentication")


def smoke_event_receipts() -> None:
    """Authenticate any published event artifacts without running anything."""
    if not LEDGER.exists():
        if READOUT.exists() or EVENT_DIR.exists():
            raise SystemExit("published event artifacts exist without a ledger")
        return
    rows = [
        json.loads(line)
        for line in LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or rows[0] != opened_record():
        raise SystemExit(
            "published benchmark ledger row 1 does not match the frozen "
            "opened record"
        )
    if len(rows) == 1:
        raise SystemExit(
            "benchmark ledger has a crashed opened event; audit the preserved "
            "receipts and resume before smoke can pass"
        )
    if len(rows) != 2:
        raise SystemExit("published benchmark ledger has unexpected trailing rows")
    authenticate_closed_event(rows[1])
    if not READOUT.is_file():
        raise SystemExit(
            "published benchmark event lacks its terminal zero-root readout"
        )
    # Recomputes the readout byte-identically from the three receipts
    # (verify mode; writes nothing).
    run([sys.executable, "-B", str(SCRIPTS / "check_benchmark.py")])


def smoke() -> None:
    for relative, expected in sorted(EXTERNAL_MERGE_RECEIPTS.items()):
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"published external merge receipt changed: {path}")
    if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
        raise SystemExit("trusted gateway is absent or changed")
    # Standalone-reproducibility gate: authenticate the copied lineage
    # package (byte-pinned manifest, six datasets, three trainers, merger)
    # and the deliberate ABSENCE of the blend root. Fast, no GPU, no writes.
    run([sys.executable, "-B", str(SCRIPTS / "rebuild_zero_root.py"), "--verify-inputs"])
    if DESIGN_RECEIPT.exists():
        # Re-verifies every design pin (seed-freshness audit, package pins,
        # arm pins, code hashes, run_benchmark substring contracts) and the
        # receipt bytes.
        run([sys.executable, "-B", str(SCRIPTS / "gen_design_receipt.py"), "--check"])
        if COMPUTE_REVIEW.exists():
            require_verdict(COMPUTE_REVIEW, REBUILD_VERDICT, "rebuild compute review")
        if BENCH_REVIEW.exists():
            require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
    smoke_lineage_receipts()
    smoke_event_receipts()
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
        "PASS: zero-root lineage-rebuild design — byte-identical copied "
        "stage package (blend root deliberately absent), fresh zero-init "
        "stage-1 rewiring, inherited stage seeds 42/43/44/47/51/55, sealed "
        "fresh benchmark seed 78159, single-seed write-ahead ledger, and "
        "the frozen medium/tb1024 three-arm contracts"
    )


def rebuild_stage() -> None:
    require_pushed_checkpoint(f"{EXP_REL}/data/design_receipt.json")
    require_pushed_checkpoint(f"{EXP_REL}/reports/compute_review.md")
    require_verdict(COMPUTE_REVIEW, REBUILD_VERDICT, "rebuild compute review")
    run([sys.executable, "-B", str(SCRIPTS / "gen_design_receipt.py"), "--check"])
    run([sys.executable, "-B", str(SCRIPTS / "rebuild_zero_root.py")])
    print(
        "[run] rebuild complete; commit the runs/lineage receipts, fill the "
        "three TODO-pins in scripts/run_benchmark.py from "
        "runs/lineage/merge.json, and seek the PASS_BENCHMARK_EVENT review"
    )


def benchmark_stage(resume: bool) -> None:
    for relative in (
        f"{EXP_REL}/data/design_receipt.json",
        f"{EXP_REL}/reports/benchmark_design_review.md",
        *(f"{EXP_REL}/runs/lineage/{name}.json" for name in STAGE_DIRNAMES),
        f"{EXP_REL}/runs/lineage/merge.json",
        *sorted(EXTERNAL_MERGE_RECEIPTS),
    ):
        require_pushed_checkpoint(relative)
    require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
    run([sys.executable, "-B", str(SCRIPTS / "gen_design_receipt.py"), "--check"])
    sys.path.insert(0, str(SCRIPTS))
    from run_benchmark import FROZEN_MODEL_PATHS  # noqa: PLC0415

    command = [
        sys.executable,
        "-B",
        str(SCRIPTS / "run_benchmark.py"),
        "--name",
        FROZEN_NAME,
        "--tier",
        FROZEN_TIER,
        "--think-budget",
        str(FROZEN_THINK_BUDGET),
        "--seed",
        str(FROZEN_SEED),
    ]
    for label in MODEL_ORDER:
        command.extend(("--model", f"{label}={FROZEN_MODEL_PATHS[label]}"))
    # The audit-before-resume interlock lives in run_benchmark.py; only an
    # explicit operator --resume may be forwarded, never an automatic one.
    if resume:
        command.append("--resume")
    run(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--stage", choices=("rebuild", "benchmark"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a crashed benchmark event after auditing its "
        "preserved receipts (forwarded to run_benchmark.py)",
    )
    args = parser.parse_args()
    if args.smoke:
        if args.resume:
            parser.error("--resume only applies to --stage benchmark")
        smoke()
        return 0
    if args.stage == "rebuild":
        if args.resume:
            parser.error("--resume only applies to --stage benchmark")
        rebuild_stage()
        return 0
    if args.stage == "benchmark":
        benchmark_stage(args.resume)
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
