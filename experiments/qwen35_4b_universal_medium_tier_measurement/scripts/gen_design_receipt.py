#!/usr/bin/env python3
"""Generate (or re-verify) the frozen medium-tier measurement design receipt.

Model-free construction artifact: pins everything the single benchmark
event depends on BEFORE any model event, so the event stage can only run
the design that was reviewed. Pinned here:

- the sealed fresh seed 78150 plus a repo-wide grep-freshness audit (no
  file under experiments/, knowledge/, or research_programs/ may name
  78150 in a seed context except this experiment's own declarations);
- tier medium, think budget 1024, the frozen four-arm order;
- the four composite paths with their on-disk tree sha256s (recomputed
  from disk in write mode; the event stage recomputes them again), their
  weights sha256s, and their merge receipts by path + sha256;
- the tier-forensics constants analysis (the base sanity envelope source),
  the trusted gateway script, and the frozen quick-tier reference summary,
  each by sha256;
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
FROZEN_THINK_BUDGET = 1024
FROZEN_SEED = 78150
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
FORENSICS_ANALYSIS = (
    ROOT / "experiments" / "qwen35_4b_menders_sirens_tier_forensics"
    / "runs" / "constants_analysis.json"
)
FORENSICS_ANALYSIS_SHA256 = (
    "62aaa80cf71ebfb0510b5a7c892d4bcf04d6b81b6fa559c401f2ee82a9d8868f"
)
QUICK_SUMMARY = (
    "experiments/qwen35_4b_goal_gap_axis_curriculum_target_match"
    "/runs/benchmark/quick_tb1024_seed78144_pilot/summary.json"
)
QUICK_SUMMARY_SHA256 = (
    "4e28ba21a0c25e7bf46cabd42152a011fc86f3c0f4ba24c23ec1bf18beb78f23"
)
QUICK_SEED = 78144
QUICK_LABELS = {
    "base": "base",
    "designed_fresh": "designed_fresh_parent",
    "hygiene_explore": None,
    "replay_repeat": "replay_repeat",
}
QUICK_AGGREGATES = {
    "base": 0.10851063829787233,
    "designed_fresh": 0.46443720816263595,
    "hygiene_explore": None,
    "replay_repeat": 0.508134008201943,
}
PUBLIC_FAMILIES = (
    "chronicle", "lockpick", "menders", "mirage", "rites",
    "siftstack", "sirens", "stockade", "toolsmith", "warren",
)
CODE_FILES = {
    "gen_design_receipt": SCRIPTS / "gen_design_receipt.py",
    "run_benchmark": SCRIPTS / "run_benchmark.py",
    "check_benchmark": SCRIPTS / "check_benchmark.py",
    "harness": SCRIPTS / "run.py",
}
AUDIT_PATTERN = r"seed[^0-9]{0,3}78150|78150[^0-9]{0,3}seed"
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
    """Prove seed 78150 has never been used before in a seed context.

    Line-based scan of the three knowledge-bearing roots. A matching line
    is a self-reference (allowed) when the file lives inside this
    experiment or the experiment id appears within a few lines of the
    match (generated knowledge files quote this experiment's design with
    the id on a neighbouring line). Anything else fails closed.
    """
    pattern = re.compile(AUDIT_PATTERN)
    needle = b"78150"
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
        "roots": list(AUDIT_ROOTS),
        "self_directory_excluded": f"experiments/{EXP.name}",
        "self_reference_line_window": AUDIT_SELF_WINDOW_LINES,
        "disallowed_matches": [],
        "fresh": True,
    }


def verify_quick_reference() -> None:
    summary = ROOT / QUICK_SUMMARY
    if not summary.is_file() or sha256_file(summary) != QUICK_SUMMARY_SHA256:
        raise ValueError(f"pinned quick reference summary is absent or changed: {summary}")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    scores = payload.get("scores", {})
    if (
        payload.get("seed") != QUICK_SEED
        or payload.get("tier") != "quick"
        or payload.get("think_budget") != FROZEN_THINK_BUDGET
    ):
        raise ValueError("quick reference summary is not the frozen quick event")
    for label in MODEL_ORDER:
        quick_label = QUICK_LABELS[label]
        expected = QUICK_AGGREGATES[label]
        if quick_label is None:
            if expected is not None or label in scores:
                raise ValueError(f"quick reference unexpectedly measured {label}")
            continue
        if scores.get(quick_label, {}).get("aggregate") != expected:
            raise ValueError(f"frozen quick aggregate changed for {label}")


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
    if (
        not FORENSICS_ANALYSIS.is_file()
        or sha256_file(FORENSICS_ANALYSIS) != FORENSICS_ANALYSIS_SHA256
    ):
        raise ValueError("pinned forensics analysis is absent or changed")
    verify_quick_reference()
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
        "stage": "medium_tier_measurement_design",
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
        "frozen_quick_reference": {
            "seed": QUICK_SEED,
            "tier": "quick",
            "think_budget": FROZEN_THINK_BUDGET,
            "summary": QUICK_SUMMARY,
            "summary_sha256": QUICK_SUMMARY_SHA256,
            "quick_labels": dict(QUICK_LABELS),
            "aggregates": dict(QUICK_AGGREGATES),
        },
        "forensics_envelope": {
            "path": FORENSICS_ANALYSIS.relative_to(ROOT).as_posix(),
            "sha256": FORENSICS_ANALYSIS_SHA256,
            "field": "base_profile.medium.families.<family>.{min,max}",
            "inside_is_inclusive": True,
        },
        "readings": {
            "aggregate_ordering_vs_quick": (
                "medium aggregate per arm; ranking compared against the frozen "
                "quick ordering on the three quick-measured arms; "
                "hygiene_explore quick aggregate recorded as null"
            ),
            "recorded_goal_gate": (
                "per treated arm: strict wins/ties/losses vs base across the "
                "ten public families; pass = ten strict wins; recorded, "
                "never gated on"
            ),
            "base_sanity_envelope": (
                "per family, base medium score inside the historical base "
                "[min, max] from the pinned forensics analysis"
            ),
            "blocking_families": "per arm, the families not strictly won",
        },
        "public_families": list(PUBLIC_FAMILIES),
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
