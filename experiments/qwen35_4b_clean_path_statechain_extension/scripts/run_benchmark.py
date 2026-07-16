#!/usr/bin/env python3
"""Run the one conditional four-model medium pilot through the trusted gateway.

CONDITIONAL on local promotion: this event exists only if ``statechain_clean``
passed the frozen seed-88041 local gate. Four composites (base, the
ZERO-ROOT parent — lifecycle 22's fully documented rebuild — the
replay_ctl4 control, and the promoted candidate) are evaluated sequentially
on the same sealed fresh seed 78160 at tier medium, think budget 1024, in
that frozen order. Discipline copied from the hardened medium-tier
measurement runner, plus lifecycle 22's NORMALIZED-HASH code pin: this file
is frozen by check_design.py's normalized hash — exactly the six trained-arm
pin VALUE slots are canonicalized to a fixed placeholder before hashing, so
EVERY OTHER BYTE of this runner (every guard call site included) is
byte-frozen pre- and post-fill, and any drift fails ``check_design --check``
at the seed-consuming boundary:

- the PASS_BENCHMARK_EVENT review verdict and the frozen design receipts are
  enforced HERE, at the seed-consuming boundary (check_design --check and
  gen_local_gate --check re-run as subprocesses), not only in the harness: a
  direct invocation cannot consume the seed with unreviewed or drifted code;
- clean pushed main plus the committed-at-HEAD design receipt, local design
  receipt, benchmark design review, promotion receipt, local receipt, and
  every arm's committed merge receipt are hard prerequisites;
- every arm authenticates by recomputing its full on-disk tree sha256 (which
  covers the 9GB weights) against its pin — the two trained arms' six pins
  (tree, weights, committed merge receipt each) are orchestrator-filled
  TODO-PIN slots that abort while None; the zero-root parent additionally
  authenticates against lifecycle 22's committed lineage merge receipt;
- one-seed WRITE-AHEAD ledger: an ``opened`` record is appended before the
  first gateway call and a ``closed`` record after the summary. Any closed
  record refuses forever; a crashed event forces recovery through
  ``--resume`` — deleting the event directory cannot silently re-consume the
  seed;
- gateway failures leave a safe failure receipt with the sanitized
  diagnostic only; child stdout/stderr never surface here;
- every score in a gateway receipt must be a finite float in [0, 1]; a NaN
  can never silently drop a family from the strict-win partition.

PILOT GATE (recorded in the summary): candidate aggregate strictly > base
AND > replay_ctl4 AND > zero_root_parent. The GOAL GATE (all ten families
strictly > base) and the RITES-CONVERSION reading (candidate rites vs
parent/replay rites, paired) are recorded from the same event either way,
with the frozen power statement. The benchmark suite directory is never
read; only scripts/run_benchmark_aggregate.py runs.
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
LOCAL_DESIGN_RECEIPT = EXP / "data" / "local_design_receipt.json"
BENCH_REVIEW = EXP / "reports" / "benchmark_design_review.md"
BENCH_VERDICT = "**Verdict:** `PASS_BENCHMARK_EVENT`."
LOCAL_SEED = 88041
LOCAL_RECEIPT = EXP / "runs" / "local" / f"seed{LOCAL_SEED}.json"
PROMOTION_RECEIPT = EXP / "runs" / "local" / f"seed{LOCAL_SEED}_promotion.json"
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
FROZEN_NAME = "pilot"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
FROZEN_SEED = 78160
FROZEN_PARENT = "zero_root_parent"
FROZEN_REPLAY_CONTROL = "replay_ctl4"
FROZEN_CANDIDATES = ("statechain_clean",)
MODEL_ORDER = ("base", FROZEN_PARENT, FROZEN_REPLAY_CONTROL, "statechain_clean")
TREATED_ARMS = (FROZEN_PARENT, FROZEN_REPLAY_CONTROL, "statechain_clean")
FROZEN_MODEL_PATHS = {
    "base": (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    ),
    FROZEN_PARENT: (
        ROOT / "large_artifacts" / "qwen35_4b_zero_root_lineage_rebuild"
        / "merged" / "zero_root_hygiene_explore"
    ),
    FROZEN_REPLAY_CONTROL: (
        ROOT / "large_artifacts" / EXP.name / "merged" / "replay_ctl4"
    ),
    "statechain_clean": ROOT / "large_artifacts" / EXP.name / "merged" / "statechain_clean",
}
# Full on-disk tree hashes, recomputed at event time; the tree manifest
# covers every file including model.safetensors, so a tree match implies a
# weights match. Inherited arms are pinned at design time from committed
# receipts; the two trained arms are TODO-PINs the orchestrator fills after
# their merges publish. A None pin aborts the event.
FROZEN_TREE_SHA256: dict[str, str | None] = {
    "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
    FROZEN_PARENT: (
        "414f582950bf60fed2fe462cd141ab98d0f772087b4f9c6bc5aa12f03f379e7d"
    ),
    # TODO-PIN(post-merge): runs/merges/replay_ctl4.json -> output_tree_sha256.
    # Fill by replacing None with the quoted 64-hex ON THIS LINE; the design
    # receipt's normalized-hash pin canonicalizes only this value.
    "replay_ctl4": "68d1489a78ac8f6369429c185f9c12ac49aa8279e42acd07fe4d6c9dc28f13bc",
    # TODO-PIN(post-merge): runs/merges/statechain_clean.json ->
    # output_tree_sha256. Fill ON THIS LINE only.
    "statechain_clean": "9f1a279a1c6120263013a6ecac0701cd73fa3af86f5269ed66b71199d164f450",
}
FROZEN_WEIGHTS_SHA256: dict[str, str | None] = {
    "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    FROZEN_PARENT: (
        "6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932"
    ),
    # TODO-PIN(post-merge): runs/merges/replay_ctl4.json weight sha256.
    # Fill by replacing None with the quoted 64-hex ON THIS LINE.
    "replay_ctl4": "7ab64488f5dd88200c615283d264032df521853ff097dbcfe50f63f9b7191c1a",
    # TODO-PIN(post-merge): runs/merges/statechain_clean.json weight sha256.
    # Fill by replacing None with the quoted 64-hex ON THIS LINE.
    "statechain_clean": "1b436a0e39f825ae12bf7ac327b3a0d4adf286596f13799f10de2db3422ec614",
}
WEIGHTS_SIZE_BYTES = 9_078_620_536
# TODO-PIN(post-merge): sha256 of the COMMITTED runs/merges/replay_ctl4.json.
# Fill by replacing None with the quoted 64-hex ON THIS LINE; the design
# receipt's normalized-hash pin canonicalizes only this value.
REPLAY_CTL4_MERGE_RECEIPT_SHA256 = "6844abf90b6ca85cb462f177c6405638751dafa9c330d78ee34bb8b28a3ad0a0"
# TODO-PIN(post-merge): sha256 of the COMMITTED
# runs/merges/statechain_clean.json. Fill ON THIS LINE only.
STATECHAIN_CLEAN_MERGE_RECEIPT_SHA256 = "f575af3be4257f3e1d129aea3dc213fcd2b1e06c978123b7c40e1ee6e8669d26"
# The two trained arms carry committed receipts in this experiment (their
# shas are the two slot constants above); the zero-root parent is
# authenticated against lifecycle 22's committed lineage merge receipt via
# require_zero_root_parent_provenance; base's reserialization receipt lives
# inside the composite.
COMMITTED_MERGE_RECEIPTS: dict[str, tuple[str, object]] = {
    FROZEN_REPLAY_CONTROL: (
        f"experiments/{EXP.name}/runs/merges/replay_ctl4.json",
        REPLAY_CTL4_MERGE_RECEIPT_SHA256,
    ),
    "statechain_clean": (
        f"experiments/{EXP.name}/runs/merges/statechain_clean.json",
        STATECHAIN_CLEAN_MERGE_RECEIPT_SHA256,
    ),
}
EXPECTED_MERGE_RECEIPT_NAMES = {
    FROZEN_REPLAY_CONTROL: "replay_ctl4",
    "statechain_clean": "statechain_clean",
}
# Lifecycle 22's committed zero-root lineage merge receipt: the parent arm's
# provenance anchor (fixed at design time, NOT a fill slot). This cell also
# carries a byte-identical copy at data/lineage/provenance/merge.json.
ZERO_ROOT_PARENT_MERGE_RECEIPT = (
    ROOT / "experiments" / "qwen35_4b_zero_root_lineage_rebuild"
    / "runs" / "lineage" / "merge.json"
)
ZERO_ROOT_PARENT_MERGE_RECEIPT_SHA256 = (
    "e906caea7c4b86f4a3eacb96affb7cc2fa9b7cc11e11b634b651cabc5dd01d2b"
)
BASE_MERGE_RECEIPT_SHA256 = (
    "25aee794cfffe4d58110defc61177edef1f5324e47deb28fbd3cb7ccd61ae54f"
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
POWER_STATEMENT = (
    "menders is closed (three pedagogies, the budget lever, and the 10x "
    "dose-scale cell all failed it), so the winnable ceiling is 9/10; the "
    "readings of consequence are (a) whether the statechain install "
    "CONVERTS to the rites family ON THE CLEAN LINEAGE (candidate rites vs "
    "parent/replay rites, paired on this seed) and (b) the fully-documented "
    "model's per-family profile; any 10/10 would be a menders draw and "
    "feeds a fresh confirmation cell before any claim"
)


def require_pin(value, name: str):
    if value is None:
        raise ValueError(
            f"frozen constant {name} is unpinned (TODO-PIN); the orchestrator "
            "must fill it from the committed merge receipts before this event"
        )
    return value


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


def require_todo_pins_filled() -> None:
    """Every trained-arm pin must be a filled sha256; None refuses fail-closed."""
    sha_re = re.compile(r"[0-9a-f]{64}")
    for label, value in (
        ("replay_ctl4 tree", FROZEN_TREE_SHA256[FROZEN_REPLAY_CONTROL]),
        ("replay_ctl4 weights", FROZEN_WEIGHTS_SHA256[FROZEN_REPLAY_CONTROL]),
        ("replay_ctl4 merge receipt", REPLAY_CTL4_MERGE_RECEIPT_SHA256),
        ("statechain_clean tree", FROZEN_TREE_SHA256["statechain_clean"]),
        ("statechain_clean weights", FROZEN_WEIGHTS_SHA256["statechain_clean"]),
        ("statechain_clean merge receipt", STATECHAIN_CLEAN_MERGE_RECEIPT_SHA256),
    ):
        if not isinstance(value, str) or sha_re.fullmatch(value) is None:
            raise ValueError(
                f"{label} pin is unfilled (TODO-PIN): merge the trained arms, "
                "commit their merge receipts, fill the six pins from them, and "
                "re-seek review before any seed can be consumed"
            )


def require_zero_root_parent_provenance(model: Path) -> None:
    """The committed lifecycle-22 receipt must pin exactly the parent arm."""
    if (
        not ZERO_ROOT_PARENT_MERGE_RECEIPT.is_file()
        or sha256_file(ZERO_ROOT_PARENT_MERGE_RECEIPT)
        != ZERO_ROOT_PARENT_MERGE_RECEIPT_SHA256
    ):
        raise ValueError(
            "committed zero-root parent merge receipt is absent or changed: "
            f"{ZERO_ROOT_PARENT_MERGE_RECEIPT}"
        )
    payload = json.loads(
        ZERO_ROOT_PARENT_MERGE_RECEIPT.read_text(encoding="utf-8")
    )
    if (
        payload.get("experiment_id") != "qwen35_4b_zero_root_lineage_rebuild"
        or payload.get("stage") != "merge"
        or payload.get("name") != "zero_root_hygiene_explore"
        or payload.get("base_model", {}).get("id") != MODEL_ID
        or payload.get("base_model", {}).get("revision") != MODEL_REVISION
        or Path(payload.get("merged", "")).resolve() != model.resolve()
        or payload.get("output_tree_sha256") != FROZEN_TREE_SHA256[FROZEN_PARENT]
        or payload.get("weights_sha256") != FROZEN_WEIGHTS_SHA256[FROZEN_PARENT]
        or payload.get("weights_size_bytes") != WEIGHTS_SIZE_BYTES
    ):
        raise ValueError(
            "zero-root parent merge receipt does not describe the frozen parent arm"
        )


def authenticate_model_tree(label: str, model: Path) -> dict:
    """Bind the arm to its published bytes: recompute the full tree hash."""
    expected_tree = require_pin(
        FROZEN_TREE_SHA256[label], f"FROZEN_TREE_SHA256[{label!r}]"
    )
    expected_weights = require_pin(
        FROZEN_WEIGHTS_SHA256[label], f"FROZEN_WEIGHTS_SHA256[{label!r}]"
    )
    manifest = merged_tree_manifest(model)
    observed_tree = tree_manifest_sha256(manifest)
    if observed_tree != expected_tree:
        raise ValueError(f"benchmark arm tree changed for {label}: {observed_tree}")
    files = {row["name"]: row for row in manifest}
    weights = files["model.safetensors"]
    if (
        weights["sha256"] != expected_weights
        or weights["size"] != WEIGHTS_SIZE_BYTES
    ):
        raise ValueError(f"benchmark arm weights changed for {label}")
    if label in COMMITTED_MERGE_RECEIPTS:
        relative, expected = COMMITTED_MERGE_RECEIPTS[label]
        expected = require_pin(expected, f"COMMITTED_MERGE_RECEIPTS[{label!r}]")
        receipt_path = ROOT / relative
        if not receipt_path.is_file() or sha256_file(receipt_path) != expected:
            raise ValueError(f"committed merge receipt is absent or changed: {relative}")
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            payload.get("name") != EXPECTED_MERGE_RECEIPT_NAMES[label]
            or payload.get("model_id") != MODEL_ID
            or payload.get("model_revision") != MODEL_REVISION
            or Path(payload.get("merged", "")).resolve() != model.resolve()
            or payload.get("output_tree_sha256") != expected_tree
            or {row.get("name"): row.get("sha256") for row in payload.get("weight_files", [])}
            != {"model.safetensors": expected_weights}
        ):
            raise ValueError(f"merge receipt does not describe this composite: {label}")
    elif label == FROZEN_PARENT:
        require_zero_root_parent_provenance(model)
        receipt = json.loads(
            ZERO_ROOT_PARENT_MERGE_RECEIPT.read_text(encoding="utf-8")
        )
        if files["merge_receipt.json"]["sha256"] != receipt.get(
            "inner_merge_receipt_sha256"
        ):
            raise ValueError(
                "zero-root parent composite's inner merge receipt does not "
                "match the committed lineage merge receipt"
            )
    else:
        if files["merge_receipt.json"]["sha256"] != BASE_MERGE_RECEIPT_SHA256:
            raise ValueError("base reserialization receipt changed")
    return {"tree_sha256": observed_tree, "weights_sha256": weights["sha256"]}


def authenticate_local_promotion(candidate: str) -> dict:
    if not PROMOTION_RECEIPT.is_file():
        raise ValueError("local promotion receipt is absent; benchmark stays sealed")
    payload = json.loads(PROMOTION_RECEIPT.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("seed") != LOCAL_SEED
        or payload.get("aggregate_seed") != FROZEN_SEED
        or payload.get("aggregate_seed_open") is not True
        or payload.get("benchmark_data_read") is not False
        or payload.get("promoted") != candidate
        or not LOCAL_RECEIPT.is_file()
        or payload.get("local_receipt_sha256") != sha256_file(LOCAL_RECEIPT)
        or payload.get("design_receipt_sha256") != sha256_file(LOCAL_DESIGN_RECEIPT)
    ):
        raise ValueError(
            "local promotion receipt does not authorize this candidate benchmark"
        )
    return payload


def load_event(path: Path, model: Path) -> dict:
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
        or payload.get("within_budget") is not True
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


def pilot_gate(events: dict[str, dict], candidate: str) -> dict:
    """Frozen pilot promotion reading: strict aggregate wins over all three."""
    scores = {label: events[label]["aggregate"] for label in MODEL_ORDER}
    gate = {
        "strictly_beats_base_aggregate": scores[candidate] > scores["base"],
        "strictly_beats_replay_control_aggregate": (
            scores[candidate] > scores[FROZEN_REPLAY_CONTROL]
        ),
        "strictly_beats_immediate_parent_aggregate": (
            scores[candidate] > scores[FROZEN_PARENT]
        ),
    }
    gate["passes_pilot_gate"] = all(gate.values())
    return gate


def goal_gate_reading(events: dict[str, dict]) -> dict:
    """Strict wins / ties / losses vs base per treated arm; pass = ten wins."""
    base = events["base"]["per_family"]
    table = {}
    for arm in TREATED_ARMS:
        values = events[arm]["per_family"]
        wins = sorted(f for f in PUBLIC_FAMILIES if values[f] > base[f])
        losses = sorted(f for f in PUBLIC_FAMILIES if values[f] < base[f])
        ties = sorted(f for f in PUBLIC_FAMILIES if values[f] == base[f])
        table[arm] = {
            "strict_wins": len(wins),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "goal_gate_pass": len(wins) == len(PUBLIC_FAMILIES),
        }
    return {
        "per_arm": table,
        "included_in_pilot_gate": False,
        "recorded_either_way": True,
        "power_statement": POWER_STATEMENT,
    }


def rites_conversion_reading(events: dict[str, dict], candidate: str) -> dict:
    """Reading (a): does the statechain install convert to the rites family
    ON THE CLEAN LINEAGE — candidate rites vs parent/replay rites, paired on
    this seed. Recorded either way; never part of the pilot gate."""
    rites = {label: events[label]["per_family"]["rites"] for label in MODEL_ORDER}
    return {
        "family": "rites",
        "per_arm": rites,
        "candidate_minus_parent": rites[candidate] - rites[FROZEN_PARENT],
        "candidate_minus_replay": rites[candidate] - rites[FROZEN_REPLAY_CONTROL],
        "candidate_minus_base": rites[candidate] - rites["base"],
        "converts_on_clean_lineage": (
            rites[candidate] > rites[FROZEN_PARENT]
            and rites[candidate] > rites[FROZEN_REPLAY_CONTROL]
        ),
        "included_in_pilot_gate": False,
        "recorded_either_way": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tier", choices=("quick", "medium"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--think-budget", type=int)
    parser.add_argument("--model", action="append", required=True, help="label=/merged/model")
    parser.add_argument(
        "--candidate", choices=FROZEN_CANDIDATES, required=True,
        help="the single locally promoted candidate label",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.name != FROZEN_NAME
        or args.tier != FROZEN_TIER
        or args.seed != FROZEN_SEED
        or args.think_budget != FROZEN_THINK_BUDGET
    ):
        parser.error("benchmark event differs from the preregistered pilot")
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
            "models must be exactly base, zero_root_parent, replay_ctl4, "
            "and statechain_clean"
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
        require_todo_pins_filled()
        if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
            raise ValueError("trusted gateway is absent or changed")
        require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
        promotion = authenticate_local_promotion(args.candidate)
        require_clean_pushed_main(
            [
                DESIGN_RECEIPT,
                LOCAL_DESIGN_RECEIPT,
                BENCH_REVIEW,
                PROMOTION_RECEIPT,
                LOCAL_RECEIPT,
                ZERO_ROOT_PARENT_MERGE_RECEIPT,
            ]
            + [
                ROOT / relative
                for relative, _ in COMMITTED_MERGE_RECEIPTS.values()
            ]
        )
        # Re-verify the frozen design at the seed-consuming boundary: the
        # local design receipt pins the gate/harness/runner code set and the
        # design receipt pins the construction, so a committed drift of
        # either fails here before any gateway call.
        for check_script in ("check_design.py", "gen_local_gate.py"):
            subprocess.run(
                [str(PYTHON), "-B", str(SCRIPTS / check_script), "--check"],
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
                    }, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
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

    base = events["base"]
    comparisons = {}
    for label in TREATED_ARMS:
        event = events[label]
        family_delta = {
            family: event["per_family"][family] - base["per_family"][family]
            for family in sorted(PUBLIC_FAMILIES)
        }
        comparisons[f"{label}_minus_base"] = {
            "aggregate_delta": event["aggregate"] - base["aggregate"],
            "per_family_delta": family_delta,
            "positive_families": sum(value > 0 for value in family_delta.values()),
            "nonnegative_families": sum(value >= 0 for value in family_delta.values()),
            "minimum_family_delta": min(family_delta.values()),
        }
    candidate = events[args.candidate]
    candidate_comparisons = {args.candidate: {}}
    for control_label in (FROZEN_PARENT, FROZEN_REPLAY_CONTROL):
        control = events[control_label]
        candidate_comparisons[args.candidate][f"{args.candidate}_minus_{control_label}"] = {
            "aggregate_delta": candidate["aggregate"] - control["aggregate"],
            "per_family_delta": {
                family: candidate["per_family"][family] - control["per_family"][family]
                for family in sorted(PUBLIC_FAMILIES)
            },
        }
    promotion_gate = pilot_gate(events, args.candidate)
    goal_gate = goal_gate_reading(events)
    rites_conversion = rites_conversion_reading(events, args.candidate)
    payload = {
        "schema_version": 1,
        "name": args.name,
        "tier": args.tier,
        "think_budget": args.think_budget,
        "seed": args.seed,
        "gateway_sha256": GATEWAY_SHA256,
        "design_receipt_sha256": sha256_file(DESIGN_RECEIPT),
        "local_design_receipt_sha256": sha256_file(LOCAL_DESIGN_RECEIPT),
        "zero_root_parent_merge_receipt_sha256": (
            ZERO_ROOT_PARENT_MERGE_RECEIPT_SHA256
        ),
        "candidate": args.candidate,
        "local_promotion_receipt": str(PROMOTION_RECEIPT.resolve()),
        "local_promotion_receipt_sha256": sha256_file(PROMOTION_RECEIPT),
        "local_promotion_promoted": promotion["promoted"],
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
        "comparisons": comparisons,
        "candidate_comparisons": candidate_comparisons,
        "promotions": {args.candidate: promotion_gate},
        "goal_gate": goal_gate,
        "rites_conversion": rites_conversion,
        "benchmark_data_read": False,
    }
    result = output_dir / "summary.json"
    if result.exists():
        parser.error("refusing to overwrite event summary")
    result.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "name": args.name, "phase": "closed", "tier": args.tier,
            "think_budget": args.think_budget, "seed": args.seed,
            "summary": str(result), "summary_sha256": sha256_file(result),
        }, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
