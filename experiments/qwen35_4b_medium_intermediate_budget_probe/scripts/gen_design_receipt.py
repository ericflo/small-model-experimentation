#!/usr/bin/env python3
"""Generate (or re-verify) the frozen medium intermediate budget-probe design receipt.

Model-free construction artifact: pins everything the single benchmark
event depends on BEFORE any model event, so the event stage can only run
the design that was reviewed. Pinned here:

- the sealed fresh seed 78153 plus a repo-wide grep-freshness audit (no
  file under experiments/, knowledge/, or research_programs/ may name
  78153 in a seed context except this experiment's own declarations).
  Raw-substring hits of ``78153`` inside floats and 10-digit seeds are
  expected across the repo and are excluded by the seed-context regex's
  word-boundary guards ((?<![0-9]) / (?![0-9]));
- tier medium, think budget 4096 (the intermediate probe lever; the LAST
  budget probe after the tb8192 BUDGET_GATE_STOP), the frozen four-arm
  order;
- the four composite paths with their on-disk tree sha256s (recomputed
  from disk in write mode; the event stage recomputes them again), their
  weights sha256s, and their merge receipts by path + sha256;
- the committed tb1024 seed-78150 medium event summary (the
  budget_contrast source, labeled cross_seed_confound) and the trusted
  gateway script, each by sha256;
- code_sha256 pins of every script in this experiment.

``--check`` recomputes the receipt byte-identically (re-running the seed
audit and every cheap pin check; the 9GB-per-arm tree recompute is
reserved for write mode and the event stage). Write mode refuses to
overwrite an existing receipt: a changed design is a new experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
OUT = EXP / "data" / "design_receipt.json"

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
FROZEN_NAME = "measurement"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 4096
FROZEN_SEED = 78153
MODEL_ORDER = ("base", "designed_fresh", "replay_repeat", "hygiene_explore")
TREATED_ARMS = ("designed_fresh", "replay_repeat", "hygiene_explore")
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
GATEWAY = ROOT / "scripts" / "run_benchmark_aggregate.py"
GATEWAY_SHA256 = "53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17"
TB1024_SUMMARY = (
    "experiments/qwen35_4b_universal_medium_tier_measurement"
    "/runs/benchmark/medium_tb1024_seed78150_measurement/summary.json"
)
TB1024_SUMMARY_SHA256 = (
    "a927fc838ca8b1eaa3083d6034ba09ad0659c21a2a13b22c525487cf95a6fb43"
)
TB1024_SEED = 78150
TB1024_THINK_BUDGET = 1024
PUBLIC_FAMILIES = (
    "chronicle", "lockpick", "menders", "mirage", "rites",
    "siftstack", "sirens", "stockade", "toolsmith", "warren",
)
BUDGET_FAMILIES = ("menders", "rites")
CODE_FILES = {
    "gen_design_receipt": SCRIPTS / "gen_design_receipt.py",
    "run_benchmark": SCRIPTS / "run_benchmark.py",
    "check_benchmark": SCRIPTS / "check_benchmark.py",
    "harness": SCRIPTS / "run.py",
}
# Word-boundary seed-context pattern: the digit guards exclude the expected
# raw-substring hits (78153 inside floats like 0.4781536 and inside 10-digit
# seeds) while still failing closed on any true seed-context use.
AUDIT_PATTERN = r"seed[^0-9]{0,3}78153(?![0-9])|(?<![0-9])78153[^0-9]{0,3}seed"
AUDIT_ROOTS = ("experiments", "knowledge", "research_programs")
AUDIT_SELF_WINDOW_LINES = 3


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_score(value: object) -> bool:
    """A pinned score must be a finite float in [0, 1]; NaN never passes."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


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
    return sha256_bytes(rendered)


def seed_freshness_audit() -> dict:
    """Prove seed 78153 has never been used before in a seed context.

    Line-based scan of the three knowledge-bearing roots. The raw substring
    ``78153`` occurs across the repo inside floats and 10-digit seeds; those
    hits are expected and are excluded because the seed-context pattern
    requires ``seed`` within three non-digit characters AND the number to
    stand alone (no adjacent digits). A matching line is a self-reference
    (allowed) when the file lives inside this experiment or the experiment
    id appears within a few lines of the match (generated knowledge files
    quote this experiment's design with the id on a neighbouring line).
    Anything else fails closed.
    """
    pattern = re.compile(AUDIT_PATTERN)
    needle = b"78153"
    self_prefix = f"experiments/{EXP.name}/"
    disallowed = []
    for root in AUDIT_ROOTS:
        for path in sorted((ROOT / root).rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith(self_prefix):
                continue
            raw = path.read_bytes()
            if needle not in raw:
                continue
            lines = raw.decode("utf-8", errors="replace").splitlines()
            for index, line in enumerate(lines):
                if not pattern.search(line):
                    continue
                window = lines[
                    max(0, index - AUDIT_SELF_WINDOW_LINES):
                    index + AUDIT_SELF_WINDOW_LINES + 1
                ]
                if any(EXP.name in nearby for nearby in window):
                    continue
                disallowed.append(f"{relative}:{index + 1}")
    if disallowed:
        raise ValueError(
            f"seed {FROZEN_SEED} is not fresh; seed-context matches: {disallowed}"
        )
    return {
        "seed": FROZEN_SEED,
        "pattern": AUDIT_PATTERN,
        "word_boundary_guards": True,
        "substring_hits_in_floats_and_long_seeds_excluded": True,
        "roots": list(AUDIT_ROOTS),
        "self_directory_excluded": f"experiments/{EXP.name}",
        "self_reference_line_window": AUDIT_SELF_WINDOW_LINES,
        "disallowed_matches": [],
        "fresh": True,
    }


def verify_tb1024_reference() -> None:
    """Pin the budget-contrast source: the committed tb1024 seed-78150 event."""
    summary = ROOT / TB1024_SUMMARY
    if not summary.is_file() or sha256_file(summary) != TB1024_SUMMARY_SHA256:
        raise ValueError(
            f"pinned tb1024 contrast-source summary is absent or changed: {summary}"
        )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    scores = payload.get("scores", {})
    if (
        payload.get("seed") != TB1024_SEED
        or payload.get("tier") != FROZEN_TIER
        or payload.get("think_budget") != TB1024_THINK_BUDGET
        or payload.get("model_order") != list(MODEL_ORDER)
        or set(scores) != set(MODEL_ORDER)
    ):
        raise ValueError("tb1024 contrast source is not the frozen seed-78150 event")
    for label in MODEL_ORDER:
        row = scores[label]
        if (
            set(row.get("per_family", {})) != set(PUBLIC_FAMILIES)
            or not _valid_score(row.get("aggregate"))
            or any(not _valid_score(value) for value in row["per_family"].values())
        ):
            raise ValueError(f"tb1024 contrast source violates the score shape: {label}")


def verify_composite(label: str, deep: bool) -> None:
    model = FROZEN_MODEL_PATHS[label]
    weights = model / "model.safetensors"
    if (
        not model.is_dir()
        or {child.name for child in model.iterdir()} != MERGED_FILE_NAMES
        or not weights.is_file()
        or weights.stat().st_size != WEIGHTS_SIZE_BYTES
    ):
        raise ValueError(f"published composite is absent or reshaped: {label}")
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
        receipt_path = model / "merge_receipt.json"
        if sha256_file(receipt_path) != BASE_MERGE_RECEIPT_SHA256:
            raise ValueError("base reserialization receipt changed")
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            payload.get("method") != "pinned_base_composite_reserialization"
            or payload.get("model_lineage") != MODEL_ID
            or payload.get("model_revision") != MODEL_REVISION
            or {row.get("name"): row.get("sha256") for row in payload.get("weight_files", [])}
            != {"model.safetensors": FROZEN_WEIGHTS_SHA256[label]}
        ):
            raise ValueError("base reserialization receipt violates pins")
    if deep:
        manifest = merged_tree_manifest(model)
        files = {row["name"]: row for row in manifest}
        if (
            tree_manifest_sha256(manifest) != FROZEN_TREE_SHA256[label]
            or files["model.safetensors"]["sha256"] != FROZEN_WEIGHTS_SHA256[label]
        ):
            raise ValueError(f"published composite tree changed: {label}")


def verify_pins(deep: bool) -> None:
    if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
        raise ValueError("trusted gateway is absent or changed")
    verify_tb1024_reference()
    for label in MODEL_ORDER:
        verify_composite(label, deep)
    for name, path in CODE_FILES.items():
        if not path.is_file():
            raise ValueError(f"pinned experiment script is absent: {name}")


def merge_receipt_pin(label: str) -> dict:
    if label in COMMITTED_MERGE_RECEIPTS:
        relative, expected = COMMITTED_MERGE_RECEIPTS[label]
        return {"path": relative, "sha256": expected, "committed": True}
    receipt = FROZEN_MODEL_PATHS[label] / "merge_receipt.json"
    return {
        "path": receipt.relative_to(ROOT).as_posix(),
        "sha256": BASE_MERGE_RECEIPT_SHA256,
        "committed": False,
    }


def build_receipt(deep: bool) -> dict:
    verify_pins(deep)
    audit = seed_freshness_audit()
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "medium_intermediate_budget_probe_design",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "loaded": False, "calls": 0},
        "event": {
            "name": FROZEN_NAME,
            "tier": FROZEN_TIER,
            "think_budget": FROZEN_THINK_BUDGET,
            "seed": FROZEN_SEED,
            "model_order": list(MODEL_ORDER),
            "sequential_same_seed_runs": True,
            "one_seed_ledger": "runs/benchmark_events.jsonl",
            "ledger_refuses_any_prior_entry": True,
            "single_event_only": True,
            "budget_probe_lever": {
                "reference_think_budget": TB1024_THINK_BUDGET,
                "probe_think_budget": FROZEN_THINK_BUDGET,
                "identical_for_all_arms": True,
                "local_wall_time_cap_added": False,
                "budget_policy_owner": "gateway",
            },
        },
        "seed_freshness_audit": audit,
        "models": {
            label: {
                "path": FROZEN_MODEL_PATHS[label].relative_to(ROOT).as_posix(),
                "deployment": "explicit_merged_composite",
                "runtime_lora_forbidden": True,
                "tree_sha256": FROZEN_TREE_SHA256[label],
                "tree_recomputed_at_event_time": True,
                "weights_sha256": FROZEN_WEIGHTS_SHA256[label],
                "weights_size_bytes": WEIGHTS_SIZE_BYTES,
                "merge_receipt": merge_receipt_pin(label),
            }
            for label in MODEL_ORDER
        },
        "tb1024_reference": {
            "seed": TB1024_SEED,
            "tier": FROZEN_TIER,
            "think_budget": TB1024_THINK_BUDGET,
            "summary": TB1024_SUMMARY,
            "summary_sha256": TB1024_SUMMARY_SHA256,
            "cross_seed_confound": True,
        },
        "readings": {
            "budget_movement": (
                "per (arm, family) over menders and rites: moved fires only "
                "when the arm's pinned tb1024 value for that family is "
                "exactly zero AND its tb4096 value is above zero (premise "
                "from the pinned contrast source: menders 0.0 for all four "
                "arms at tb1024/78150; rites 0.0 for base, replay_repeat, "
                "and hygiene_explore, and 0.1 for designed_fresh, whose "
                "rites is excluded from the booleans and reported as "
                "already_nonzero_at_tb1024 — a status-quo repeat can never "
                "fire any_arm_moved); plus the full per-family table"
            ),
            "budget_contrast": (
                "per arm per family: delta versus the committed tb1024 event "
                "at seed 78150 (summary sha256-pinned, fail closed on "
                "change); additionally fails closed unless both events "
                "share the same benchmark implementation (runner sha256, "
                "source inventory sha256, file count), with both "
                "signatures surfaced in the block; labeled "
                "cross_seed_confound: true — remaining confounds are seed "
                "AND budget, a movement reading, not a causal isolation"
            ),
            "goal_gate": (
                "per treated arm: strict wins/ties/losses vs base across the "
                "ten public families; pass = ten strict wins; recorded, "
                "never gated on"
            ),
            "budget_integrity": (
                "per arm: the gateway receipt's within_budget flag and "
                "wall_seconds; if any arm has within_budget false the "
                "readout sets paired_comparison_valid: false with the "
                "reason; scores are still recorded"
            ),
        },
        "public_families": list(PUBLIC_FAMILIES),
        "budget_families": list(BUDGET_FAMILIES),
        "gateway": {
            "path": GATEWAY.relative_to(ROOT).as_posix(),
            "sha256": GATEWAY_SHA256,
        },
        "measurement_intake": {
            "training": None,
            "promotion": None,
            "local_gate": None,
            "adapters": None,
            "exit_zero_on_any_complete_event": True,
        },
        "checkpoint_policy": {
            "next_authorized_stage": "benchmark",
            "one_stage_per_invocation": True,
            "clean_pushed_main_required": True,
            "design_receipt_committed_at_head_required": True,
            "benchmark_review_verdict_required": "PASS_BENCHMARK_EVENT",
        },
        "firewall": {
            "benchmark_data_read": False,
            "gateway_receipts_only": True,
        },
        "code": {
            f"{name}_sha256": sha256_file(path) for name, path in CODE_FILES.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        receipt = build_receipt(deep=not args.check)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    value = (
        json.dumps(receipt, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != value:
            parser.error("design receipt is absent or changed")
    else:
        if args.out.exists():
            parser.error("refusing to overwrite design receipt")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(value)
    print(json.dumps({"out": str(args.out), "sha256": sha256_bytes(value)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
