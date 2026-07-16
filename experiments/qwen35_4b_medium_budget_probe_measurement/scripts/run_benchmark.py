#!/usr/bin/env python3
"""Run the one four-model medium-tier BUDGET-PROBE event through the trusted gateway.

MEASUREMENT INTAKE, not a promotion: there is no candidate, no local gate,
and no pass bar anywhere in this file. The same four inherited published
composites as the seed-78150 medium event (base, designed_fresh,
replay_repeat, hygiene_explore) are evaluated sequentially on the sealed
fresh seed 78152 at tier medium, think budget 8192 (the probe lever; the
only intentional difference from the reference event), in that frozen
order, and the process exits 0 on any complete event. Discipline copied
from the hardened seed-78150 measurement runner:

- the PASS_BENCHMARK_EVENT review verdict and the design receipt's code
  pins are enforced HERE, at the seed-consuming boundary (gen_design_receipt
  --check re-runs as a subprocess), not only in the harness: a direct
  invocation cannot consume the seed with unreviewed or drifted code;
- clean pushed main plus the committed-at-HEAD design receipt,
  preregistration, benchmark design review, the three committed
  treated-arm merge receipts, AND the committed tb1024 contrast-source
  summary (sha256-pinned) are hard prerequisites;
- every arm authenticates by recomputing its full on-disk tree sha256
  (which covers the 9GB weights) against the design-time pin, base
  additionally against its frozen reserialized-weights hash;
- one-seed WRITE-AHEAD ledger: an ``opened`` record is appended before the
  first gateway call and a ``closed`` record after the summary. Any closed
  record refuses forever; a crashed event leaves a permanent opened record
  that forces recovery through ``--resume`` with the preserved receipts —
  deleting the event directory cannot silently re-consume the seed;
- gateway failures leave a safe failure receipt with the sanitized
  diagnostic only; child stdout/stderr never surface here;
- every score in a gateway receipt must be a finite float in [0, 1]; a NaN
  can never silently drop a family from the strict-win partition;
- NO local wall-time cap: at tb8192 each arm may take far longer than the
  reference event's 136-230s. The gateway owns budget policy; this runner
  records ``within_budget`` and ``wall_seconds`` exactly as returned (a
  strict bool / finite non-negative number) and never gates on them —
  the budget_integrity reading in check_benchmark.py scopes the paired
  comparison instead.

After the summary is written and the ledger appended, check_benchmark.py is
invoked to write the preregistered measurement readout. The benchmark suite
directory is never read; only scripts/run_benchmark_aggregate.py runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SCRIPTS = EXP / "scripts"
GATEWAY = ROOT / "scripts" / "run_benchmark_aggregate.py"
GATEWAY_SHA256 = "53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17"
DESIGN_RECEIPT = EXP / "data" / "design_receipt.json"
PREREGISTRATION = EXP / "reports" / "preregistration.md"
BENCH_REVIEW = EXP / "reports" / "benchmark_design_review.md"
BENCH_VERDICT = "**Verdict:** `PASS_BENCHMARK_EVENT`."
PUBLIC_FAMILIES = {
    "chronicle", "lockpick", "menders", "mirage", "rites", "siftstack",
    "sirens", "stockade", "toolsmith", "warren",
}
GATEWAY_KEYS = {
    "schema_version", "stage", "tier", "think_budget", "seed", "backend", "model",
    "model_merge_receipt_sha256", "benchmark_runner_sha256",
    "benchmark_source_inventory_sha256", "benchmark_source_file_count",
    "aggregate", "per_family", "within_budget", "wall_seconds",
}
FROZEN_NAME = "measurement"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 8192
FROZEN_SEED = 78152
MODEL_ORDER = ("base", "designed_fresh", "replay_repeat", "hygiene_explore")
FROZEN_MODEL_PATHS = {
    "base": (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    ),
    "designed_fresh": (
        ROOT / "large_artifacts"
        / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
        / "merged" / "designed_fresh"
    ),
    "replay_repeat": (
        ROOT / "large_artifacts" / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "merged" / "replay_repeat"
    ),
    "hygiene_explore": (
        ROOT / "large_artifacts" / "qwen35_4b_hygiene_explore_destack_medium"
        / "merged" / "hygiene_explore"
    ),
}
# Full on-disk tree hashes, recomputed at event time; the tree manifest
# covers every file including model.safetensors, so a tree match implies a
# weights match. Identical to the seed-78150 reference event's pins.
FROZEN_TREE_SHA256 = {
    "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
    "designed_fresh": (
        "93433aa2d5f3f0d6d4540126579c09feee1d8502df702c1563bae28eb7f60255"
    ),
    "replay_repeat": (
        "4c4f3561efbcafe1b9f777f4bd21bf4949ff89177f77946d0fa0f88cafafacd7"
    ),
    "hygiene_explore": (
        "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"
    ),
}
FROZEN_WEIGHTS_SHA256 = {
    "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    "designed_fresh": (
        "0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979"
    ),
    "replay_repeat": (
        "3df45004fcf42519ce28cdcfedcbb39b0907662f8ecfb8a87b13b416087d0072"
    ),
    "hygiene_explore": (
        "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"
    ),
}
WEIGHTS_SIZE_BYTES = 9_078_620_536
# The three treated arms carry committed merge receipts at their source
# experiments; base's reserialization receipt lives inside the composite.
COMMITTED_MERGE_RECEIPTS = {
    "designed_fresh": (
        "experiments/qwen35_4b_universal_fresh_surface_budget_commit_target_match"
        "/runs/merges/designed_fresh.json",
        "ab3f20cc93d3fe21ead7a1d573edbca2903d59d6f9fe3d2af0c93e823676acc2",
    ),
    "replay_repeat": (
        "experiments/qwen35_4b_goal_gap_axis_curriculum_target_match"
        "/runs/merges/replay_repeat.json",
        "22384463d7825ec2a0b95faeaeb273264d7331f4584f8b7e9e58a60545398af1",
    ),
    "hygiene_explore": (
        "experiments/qwen35_4b_hygiene_explore_destack_medium"
        "/runs/merges/hygiene_explore.json",
        "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a",
    ),
}
BASE_MERGE_RECEIPT_SHA256 = (
    "25aee794cfffe4d58110defc61177edef1f5324e47deb28fbd3cb7ccd61ae54f"
)
# The committed tb1024 seed-78150 medium event this probe contrasts against
# (budget_contrast reading; cross-seed confounded by construction). Pinned
# here so a drifted contrast source fails BEFORE the seed is consumed.
TB1024_SUMMARY = (
    "experiments/qwen35_4b_universal_medium_tier_measurement"
    "/runs/benchmark/medium_tb1024_seed78150_measurement/summary.json"
)
TB1024_SUMMARY_SHA256 = (
    "a927fc838ca8b1eaa3083d6034ba09ad0659c21a2a13b22c525487cf95a6fb43"
)
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MERGED_FILE_NAMES = frozenset(
    {
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "merge_receipt.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
LEDGER = EXP / "runs" / "benchmark_events.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def require_clean_pushed_main(paths: list[Path]) -> None:
    if git_output(["git", "status", "--short"]):
        raise ValueError("benchmark event requires a clean worktree")
    if git_output(["git", "branch", "--show-current"]) != "main":
        raise ValueError("benchmark event requires branch main")
    if git_output(["git", "rev-parse", "HEAD"]) != git_output(["git", "rev-parse", "origin/main"]):
        raise ValueError("benchmark event requires HEAD == origin/main")
    for path in paths:
        relative = path.resolve().relative_to(ROOT).as_posix()
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if not path.is_file() or path.read_bytes() != committed:
            raise ValueError(f"benchmark prerequisite differs from HEAD: {relative}")


def merged_tree_manifest(output: Path) -> list[dict]:
    """Hash the complete, flat merged-composite tree and reject surprises."""
    if not output.is_dir() or output.is_symlink():
        raise ValueError(f"merged composite is not a real directory: {output}")
    children = sorted(output.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise ValueError("merged composite contains a symlink or nested/non-file entry")
    names = {path.name for path in children}
    if names != MERGED_FILE_NAMES:
        raise ValueError(
            "merged composite file set changed: "
            f"missing={sorted(MERGED_FILE_NAMES - names)}, "
            f"unexpected={sorted(names - MERGED_FILE_NAMES)}"
        )
    return [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in children
    ]


def tree_manifest_sha256(manifest: list[dict]) -> str:
    rendered = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(rendered).hexdigest()


def require_verdict(path: Path, verdict: str, description: str) -> None:
    if not path.is_file() or verdict not in path.read_text(encoding="utf-8"):
        raise ValueError(f"{description} has not been authorized: {path}")


def require_tb1024_reference() -> None:
    """Fail closed if the pinned contrast-source summary drifted."""
    summary = ROOT / TB1024_SUMMARY
    if not summary.is_file() or sha256_file(summary) != TB1024_SUMMARY_SHA256:
        raise ValueError(
            f"pinned tb1024 contrast-source summary is absent or changed: {summary}"
        )


def ledger_rows(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    return [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require_unconsumed_ledger(ledger: Path, opened_record: dict, resume: bool) -> None:
    """Write-ahead one-seed budget: closed refuses forever, opened needs --resume.

    Any record that is not a well-formed ``opened`` record counts as closed
    (fail closed on legacy or malformed rows). A lone matching opened record
    is a crashed event: it may only continue under an explicit ``--resume``,
    never restart silently.
    """
    rows = ledger_rows(ledger)
    if not rows:
        return
    if any(row.get("phase") != "opened" for row in rows):
        raise ValueError(
            "benchmark ledger already has a closed entry; the one-event budget is spent"
        )
    if len(rows) != 1 or rows[0] != opened_record:
        raise ValueError(
            "benchmark ledger opened record does not match the frozen event"
        )
    if not resume:
        raise ValueError(
            "benchmark ledger has an opened (crashed) event; audit the preserved "
            "receipts and use --resume"
        )


def _valid_score(value: object) -> bool:
    """A gateway score must be a finite float in [0, 1]; NaN never passes."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _valid_wall_seconds(value: object) -> bool:
    """Wall time must be a finite non-negative number (recorded, never gated)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )


def authenticate_model_tree(label: str, model: Path) -> dict:
    """Bind the arm to its published bytes: recompute the full tree hash."""
    manifest = merged_tree_manifest(model)
    observed_tree = tree_manifest_sha256(manifest)
    if observed_tree != FROZEN_TREE_SHA256[label]:
        raise ValueError(f"benchmark arm tree changed for {label}: {observed_tree}")
    files = {row["name"]: row for row in manifest}
    weights = files["model.safetensors"]
    if (
        weights["sha256"] != FROZEN_WEIGHTS_SHA256[label]
        or weights["size"] != WEIGHTS_SIZE_BYTES
    ):
        raise ValueError(f"benchmark arm weights changed for {label}")
    if label in COMMITTED_MERGE_RECEIPTS:
        relative, expected = COMMITTED_MERGE_RECEIPTS[label]
        receipt_path = ROOT / relative
        if not receipt_path.is_file() or sha256_file(receipt_path) != expected:
            raise ValueError(f"committed merge receipt is absent or changed: {relative}")
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            payload.get("name") != label
            or payload.get("model_id") != MODEL_ID
            or payload.get("model_revision") != MODEL_REVISION
            or Path(payload.get("merged", "")).resolve() != model.resolve()
            or payload.get("output_tree_sha256") != FROZEN_TREE_SHA256[label]
            or {row.get("name"): row.get("sha256") for row in payload.get("weight_files", [])}
            != {"model.safetensors": FROZEN_WEIGHTS_SHA256[label]}
        ):
            raise ValueError(f"merge receipt does not describe this composite: {label}")
    else:
        if files["merge_receipt.json"]["sha256"] != BASE_MERGE_RECEIPT_SHA256:
            raise ValueError("base reserialization receipt changed")
    return {"tree_sha256": observed_tree, "weights_sha256": weights["sha256"]}


def load_event(path: Path, model: Path) -> dict:
    """Authenticate one aggregate-gateway receipt against the frozen event.

    ``within_budget`` must be a strict bool but is RECORDED, never required
    to be true: an over-budget arm keeps its scores and is scoped by the
    budget_integrity reading (paired_comparison_valid: false) instead of
    being rejected here. The gateway owns budget policy.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        set(payload) != GATEWAY_KEYS
        or payload.get("schema_version") != 1
        or payload.get("stage") != "menagerie_aggregate_gateway"
        or payload.get("tier") != FROZEN_TIER
        or payload.get("think_budget") != FROZEN_THINK_BUDGET
        or payload.get("seed") != FROZEN_SEED
        or payload.get("backend") != "qwen_vllm"
        or Path(payload.get("model", "")).resolve() != model.resolve()
        or not isinstance(payload.get("within_budget"), bool)
        or not _valid_wall_seconds(payload.get("wall_seconds"))
        or set(payload.get("per_family", {})) != PUBLIC_FAMILIES
        or not _valid_score(payload.get("aggregate"))
        or any(
            not _valid_score(value)
            for value in payload.get("per_family", {}).values()
        )
        or payload.get("model_merge_receipt_sha256")
        != sha256_file(model / "merge_receipt.json")
    ):
        raise ValueError(f"aggregate gateway event failed authentication: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tier", choices=("quick", "medium"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--think-budget", type=int)
    parser.add_argument("--model", action="append", required=True, help="label=/merged/model")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.name != FROZEN_NAME
        or args.tier != FROZEN_TIER
        or args.seed != FROZEN_SEED
        or args.think_budget != FROZEN_THINK_BUDGET
    ):
        parser.error("benchmark event differs from the preregistered measurement")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.name):
        parser.error("unsafe event name")
    models = {}
    for specification in args.model:
        label, separator, raw_path = specification.partition("=")
        path = Path(raw_path).resolve()
        if not separator or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", label):
            parser.error(f"invalid model specification: {specification}")
        if label in models or not (path / "merge_receipt.json").is_file():
            parser.error(f"duplicate label or missing merge receipt: {specification}")
        if label not in FROZEN_MODEL_PATHS or path != FROZEN_MODEL_PATHS[label].resolve():
            parser.error(f"model path differs from the frozen arm: {specification}")
        models[label] = path
    if set(models) != set(MODEL_ORDER) or len(models) != len(MODEL_ORDER):
        parser.error(
            "models must be exactly base, designed_fresh, replay_repeat, "
            "and hygiene_explore"
        )
    if len(set(models.values())) != len(models):
        parser.error("every benchmark arm must name a distinct merged model")
    opened_record = {
        "name": args.name,
        "phase": "opened",
        "seed": args.seed,
        "think_budget": args.think_budget,
        "tier": args.tier,
    }
    try:
        if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
            raise ValueError("trusted gateway is absent or changed")
        require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
        require_tb1024_reference()
        require_clean_pushed_main(
            [DESIGN_RECEIPT, PREREGISTRATION, BENCH_REVIEW, ROOT / TB1024_SUMMARY]
            + [ROOT / relative for relative, _ in COMMITTED_MERGE_RECEIPTS.values()]
        )
        # Re-verify the design receipt at the seed-consuming boundary: its
        # code pins cover this script and check_benchmark.py, so a committed
        # drift of either fails here before any gateway call.
        subprocess.run(
            [str(PYTHON), "-B", str(SCRIPTS / "gen_design_receipt.py"), "--check"],
            cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.DEVNULL,
            check=True,
        )
        require_unconsumed_ledger(LEDGER, opened_record, args.resume)
        model_hashes = {
            label: authenticate_model_tree(label, models[label])
            for label in MODEL_ORDER
        }
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

    budget_label = str(args.think_budget) if args.think_budget is not None else "native"
    output_dir = (
        EXP / "runs" / "benchmark"
        / f"{args.tier}_tb{budget_label}_seed{args.seed}_{args.name}"
    )
    if output_dir.exists() and not args.resume:
        parser.error("partial event exists; use --resume after auditing it")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write-ahead record: the seed is spent the moment the first gateway
    # call can start, so a mid-event crash leaves a permanent trace.
    if not ledger_rows(LEDGER):
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(opened_record, sort_keys=True) + "\n")

    events = {}
    for label in MODEL_ORDER:
        model = models[label]
        output = output_dir / f"{label}.json"
        failure = output_dir / f"{label}.failure.json"
        if failure.exists():
            parser.error(f"preserved failure exists for {label}; audit before retrying")
        if not output.exists():
            # No local wall-time cap: tb8192 arms may run far past the
            # reference event's wall times; the gateway alone owns budget
            # policy.
            command = [
                str(PYTHON), str(GATEWAY), "--tier", args.tier, "--seed", str(args.seed),
                "--model", str(model), "--out", str(output),
                "--think-budget", str(args.think_budget),
            ]
            completed = subprocess.run(
                command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, check=False,
            )
            if completed.returncode != 0:
                diagnostic = "unclassified_gateway_failure"
                safe_stderr = completed.stderr.strip()
                aggregate_match = re.fullmatch(
                    r"aggregate benchmark gateway failed; category=([a-z_]+); "
                    r"private output suppressed",
                    safe_stderr,
                )
                runner_match = re.fullmatch(
                    r"benchmark runner failed with exit code \d+; private aggregate "
                    r"state=([a-z_]+); raw stdout/stderr suppressed",
                    safe_stderr,
                )
                if aggregate_match:
                    diagnostic = aggregate_match.group(1)
                elif runner_match:
                    diagnostic = f"runner_failure_{runner_match.group(1)}"
                failure.write_text(
                    json.dumps({
                        "schema_version": 1,
                        "name": args.name,
                        "tier": args.tier,
                        "think_budget": args.think_budget,
                        "seed": args.seed,
                        "arm": label,
                        "model": str(model),
                        "model_merge_receipt_sha256": sha256_file(model / "merge_receipt.json"),
                        "gateway_exit_code": completed.returncode,
                        "safe_diagnostic": diagnostic,
                        "score_emitted": False,
                        "raw_streams_exposed": False,
                        "benchmark_output_exposed": False,
                    }, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                raise SystemExit(
                    f"aggregate gateway failed for {label} with exit {completed.returncode} "
                    f"({diagnostic}); "
                    "private output remained suppressed"
                )
        events[label] = load_event(output, model)

    signatures = {
        (
            event["benchmark_runner_sha256"],
            event["benchmark_source_inventory_sha256"],
            event["benchmark_source_file_count"],
        )
        for event in events.values()
    }
    if len(signatures) != 1:
        raise ValueError("benchmark implementation changed between paired arms")
    runner_sha, inventory_sha, file_count = next(iter(signatures))

    payload = {
        "schema_version": 1,
        "name": args.name,
        "tier": args.tier,
        "think_budget": args.think_budget,
        "seed": args.seed,
        "gateway_sha256": GATEWAY_SHA256,
        "design_receipt_sha256": sha256_file(DESIGN_RECEIPT),
        "tb1024_reference_summary_sha256": TB1024_SUMMARY_SHA256,
        "model_order": list(MODEL_ORDER),
        "models": {label: str(path) for label, path in models.items()},
        "model_tree_sha256s": {
            label: hashes["tree_sha256"] for label, hashes in model_hashes.items()
        },
        "model_weight_sha256s": {
            label: hashes["weights_sha256"] for label, hashes in model_hashes.items()
        },
        "benchmark_implementation": {
            "runner_sha256": runner_sha,
            "source_inventory_sha256": inventory_sha,
            "source_file_count": file_count,
        },
        "scores": {
            label: {"aggregate": event["aggregate"], "per_family": event["per_family"]}
            for label, event in events.items()
        },
        "budget": {
            label: {
                "within_budget": event["within_budget"],
                "wall_seconds": event["wall_seconds"],
            }
            for label, event in events.items()
        },
        "promoted": None,
        "benchmark_data_read": False,
    }
    result = output_dir / "summary.json"
    if result.exists():
        parser.error("refusing to overwrite event summary")
    result.write_text(
        json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "name": args.name, "phase": "closed", "tier": args.tier,
            "think_budget": args.think_budget, "seed": args.seed,
            "summary": str(result), "summary_sha256": sha256_file(result),
        }, sort_keys=True) + "\n")

    readout = output_dir / "measurement_readout.json"
    check_command = [
        str(PYTHON), "-B", str(SCRIPTS / "check_benchmark.py"),
    ]
    if not readout.exists():
        check_command.extend(("--out", str(readout)))
    subprocess.run(
        check_command, cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True,
    )
    print(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
