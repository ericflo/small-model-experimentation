#!/usr/bin/env python3
"""Generate (or re-verify) the frozen goal-gate confirmation design receipt.

Model-free construction artifact: pins everything the three-seed
confirmation event depends on BEFORE any model event, so the event stage
can only run the design that was reviewed. Pinned here:

- the THREE sealed fresh seeds 78155, 78156, 78157, each with its own
  repo-wide grep-freshness audit (no file under experiments/, knowledge/,
  or research_programs/ may name the seed in a seed context except this
  experiment's own declarations). Raw-substring hits inside floats and
  10-digit seeds are expected across the repo and are excluded by each
  seed-context regex's word-boundary guards ((?<![0-9]) / (?![0-9]));
- tier medium, think budget 1024 (identical to the discovery event), the
  frozen two-arm order base then hygiene_explore, seed-major across the
  three per-seed event directories, and the k-seed write-ahead ledger
  contract (per-seed opened/closed records; a closed seed is never
  re-run; the overall event completes only when all three seeds close);
- the two composite paths with their on-disk tree sha256s (recomputed
  from disk in write mode; the event stage recomputes them again), their
  weights sha256s, and their merge receipts by path + sha256;
- the committed discovery-seed summary (seed 78154, the recorded 10/10
  goal-gate pass this cell replicates — REPORTED alongside the verdict,
  NEVER counted), authenticated by sha256 AND content (arm identity via
  tree/weights hashes, the recorded pass re-derived from the pinned
  scores), and its benchmark-implementation signature that all six new
  receipts must match fail-closed;
- the trusted gateway script by sha256;
- the STANDALONE LINEAGE PACKAGE (docs/quality_gates.md gate, owner
  directive 2026-07-15): the six copied stage datasets, the three trainer
  copies plus the merger copy, the vendored frozen root adapter (hard
  provenance boundary — no committed creation receipt exists), the fixed
  per-stage training seeds, and the lineage manifest whose recipe must
  agree with all of those pins and whose final-merge target IS the pinned
  hygiene_explore composite;
- code_sha256 pins of every script in this experiment.

``--check`` recomputes the receipt byte-identically (re-running the seed
audits and every cheap pin check; the 9GB-per-arm tree recompute is
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
FROZEN_NAME = "confirmation"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
SEED_ORDER = (78155, 78156, 78157)
MODEL_ORDER = ("base", "hygiene_explore")
TREATED_ARM = "hygiene_explore"
FROZEN_MODEL_PATHS = {
    "base": (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    ),
    "hygiene_explore": (
        ROOT / "large_artifacts" / "qwen35_4b_hygiene_explore_destack_medium"
        / "merged" / "hygiene_explore"
    ),
}
FROZEN_TREE_SHA256 = {
    "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
    "hygiene_explore": (
        "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"
    ),
}
FROZEN_WEIGHTS_SHA256 = {
    "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    "hygiene_explore": (
        "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"
    ),
}
WEIGHTS_SIZE_BYTES = 9_078_620_536
COMMITTED_MERGE_RECEIPTS = {
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
DISCOVERY_SUMMARY = (
    "experiments/qwen35_4b_statechain_only_dose"
    "/runs/benchmark/medium_tb1024_seed78154_pilot/summary.json"
)
DISCOVERY_SUMMARY_SHA256 = (
    "6b1a43869f013e24a048a45a04e5603b45fe59488912194eb3e76a43679255fa"
)
DISCOVERY_SEED = 78154
DISCOVERY_TREATED_ARM = "hygiene_explore_parent"
DISCOVERY_MODEL_ORDER = (
    "base", "hygiene_explore_parent", "replay_ctl2", "statechain_only"
)
DISCOVERY_IMPLEMENTATION = {
    "runner_sha256": (
        "a3beecd8b5c89ccfd99a172a6d85321d39b9feb6c29d12f10b2f4d7499e273cb"
    ),
    "source_inventory_sha256": (
        "218b8615a95f24da962c931e9cd2dba58d853a7bdcd2847cd8e2c42fc2c05f42"
    ),
    "source_file_count": 56,
}
PUBLIC_FAMILIES = (
    "chronicle", "lockpick", "menders", "mirage", "rites",
    "siftstack", "sirens", "stockade", "toolsmith", "warren",
)
FRAGILITY_FAMILIES = ("menders", "warren")
CODE_FILES = {
    "gen_design_receipt": SCRIPTS / "gen_design_receipt.py",
    "run_benchmark": SCRIPTS / "run_benchmark.py",
    "check_benchmark": SCRIPTS / "check_benchmark.py",
    "harness": SCRIPTS / "run.py",
    "rebuild_lineage": SCRIPTS / "rebuild_lineage.py",
}
# Standalone-reproducibility gate (docs/quality_gates.md, owner directive
# 2026-07-15): the treated composite's complete model-reproduction package
# lives in THIS cell. Every copy below is pinned by sha256 and fails the
# receipt closed on any drift.
LINEAGE_DIR = EXP / "data" / "lineage"
LINEAGE_MANIFEST = LINEAGE_DIR / "lineage_manifest.json"
LINEAGE_DATASETS = {
    "stage01_replay_refresh.jsonl": (
        "5d5d7c4b8a4b0a4f270fe8b2ecaebe356c771948d71b0f7bbeead6bfc04308b6",
        1520,
    ),
    "stage02_designed160.jsonl": (
        "5159cf41b6474bdc8640cdb2a4a168587b59232ca8171c7b7057fc6bfe1b40c8",
        1520,
    ),
    "stage03_close_xi__targeted_standard.jsonl": (
        "12fc613bb31a46bcea9acd49b26467656704aa3b3418dab8d920adf057d14f00",
        320,
    ),
    "stage04_replay_after_close.jsonl": (
        "541805df2d817707c1e76213e50c8f08fd9caff10d0a3887e1196424b6820be6",
        320,
    ),
    "stage05_designed_fresh.jsonl": (
        "6d4dc303bc159c19a1ffd0c60ca7d08ea64b02909366701b345d888482d67f3f",
        1520,
    ),
    "stage06_hygiene_explore.jsonl": (
        "82aa1a78c0a429a48c3db6b94ac84397cea001b041477e7b137b38c21354112f",
        1520,
    ),
}
LINEAGE_TRAINERS = {
    "lineage_trainers/train_think_stage12.py": (
        "400e4b856a5899db8805510c9f6d8d11da76b35cf30af16856b46f41abdb4e54"
    ),
    "lineage_trainers/train_think_close_stage3.py": (
        "10b4914cdc6c24858a77750f4ab64074fae5cbf36d98f3e4fc26b8958068fb14"
    ),
    "lineage_trainers/train_think_stage456.py": (
        "0cfb126feae6d73238c02362066229fff0b6a846625eed167d7681edde322cc4"
    ),
    "merge_adapter.py": (
        "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672"
    ),
}
LINEAGE_TRAINING_SEEDS = (42, 43, 44, 47, 51, 55)
LINEAGE_ROOT = (
    ROOT / "large_artifacts" / "qwen35_4b_goal_gate_confirmation"
    / "lineage_root" / "blend"
)
LINEAGE_ROOT_FILES = {
    "README.md": (
        "aeb83ceb33aa84802245cc3bc4abe24160d7d08550ca8278baff84d8b24e9ba4",
        5184,
    ),
    "adapter_config.json": (
        "cd764ae869b8a55526e283dd133e1940896b428839c6abf2e55d6ae2a0b32635",
        1094,
    ),
    "adapter_model.safetensors": (
        "ad2ef4fae785debedf5e50932a79bda97869d3efc212f53d48ccb04c59e25d21",
        169903320,
    ),
    "chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        7756,
    ),
    "tokenizer.json": (
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        19989325,
    ),
    "tokenizer_config.json": (
        "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
        1125,
    ),
}
LINEAGE_ROOT_PROVENANCE = (
    "HARD BOUNDARY: the frozen 'blend' root adapter has no committed "
    "creation receipt anywhere in the repository; it is vendored by bytes "
    "into this cell's own artifact storage and is NOT reconstructable from "
    "committed receipts. Every later stage IS reconstructable from this "
    "cell alone (data/lineage/ + scripts/lineage_trainers/ + "
    "scripts/merge_adapter.py + the pinned HF base revision)"
)
AUDIT_ROOTS = ("experiments", "knowledge", "research_programs")
AUDIT_SELF_WINDOW_LINES = 3


def audit_pattern(seed: int) -> str:
    """Word-boundary seed-context pattern for one seed.

    The digit guards exclude the expected raw-substring hits (the seed's
    digits inside floats and 10-digit seeds) while still failing closed
    on any true seed-context use.
    """
    return (
        rf"seed[^0-9]{{0,3}}{seed}(?![0-9])|(?<![0-9]){seed}[^0-9]{{0,3}}seed"
    )


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


def seed_freshness_audits() -> dict:
    """Prove each of the three seeds has never been used in a seed context.

    One line-based scan of the three knowledge-bearing roots covering all
    three seed patterns. Raw substrings occur across the repo inside
    floats and 10-digit seeds; those hits are expected and are excluded
    because each seed-context pattern requires ``seed`` within three
    non-digit characters AND the number to stand alone (no adjacent
    digits). A matching line is a self-reference (allowed) when the file
    lives inside this experiment or the experiment id appears within a
    few lines of the match (generated knowledge files quote this
    experiment's design with the id on a neighbouring line). Anything
    else fails closed.
    """
    patterns = {seed: re.compile(audit_pattern(seed)) for seed in SEED_ORDER}
    needles = {seed: str(seed).encode() for seed in SEED_ORDER}
    disallowed = {seed: [] for seed in SEED_ORDER}
    self_prefix = f"experiments/{EXP.name}/"
    for root in AUDIT_ROOTS:
        for path in sorted((ROOT / root).rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith(self_prefix):
                continue
            raw = path.read_bytes()
            active = [seed for seed in SEED_ORDER if needles[seed] in raw]
            if not active:
                continue
            lines = raw.decode("utf-8", errors="replace").splitlines()
            for index, line in enumerate(lines):
                hit_seeds = [
                    seed for seed in active if patterns[seed].search(line)
                ]
                if not hit_seeds:
                    continue
                window = lines[
                    max(0, index - AUDIT_SELF_WINDOW_LINES):
                    index + AUDIT_SELF_WINDOW_LINES + 1
                ]
                if any(EXP.name in nearby for nearby in window):
                    continue
                for seed in hit_seeds:
                    disallowed[seed].append(f"{relative}:{index + 1}")
    stale = {seed: hits for seed, hits in disallowed.items() if hits}
    if stale:
        raise ValueError(f"seeds are not fresh; seed-context matches: {stale}")
    return {
        str(seed): {
            "seed": seed,
            "pattern": audit_pattern(seed),
            "word_boundary_guards": True,
            "substring_hits_in_floats_and_long_seeds_excluded": True,
            "roots": list(AUDIT_ROOTS),
            "self_directory_excluded": f"experiments/{EXP.name}",
            "self_reference_line_window": AUDIT_SELF_WINDOW_LINES,
            "disallowed_matches": [],
            "fresh": True,
        }
        for seed in SEED_ORDER
    }


def verify_discovery_reference() -> None:
    """Pin the discovery source: the committed seed-78154 pilot summary.

    Byte pin plus content authentication: the frozen seed/tier/budget and
    four-arm order, the same base and hygiene_explore composites this
    cell runs (tree AND weights hashes; the treated arm was labeled
    ``hygiene_explore_parent`` there), the pinned benchmark-implementation
    signature, valid score shapes for both arms of interest, and the
    recorded 10/10 goal-gate pass re-derived from the pinned scores.
    """
    summary = ROOT / DISCOVERY_SUMMARY
    if not summary.is_file() or sha256_file(summary) != DISCOVERY_SUMMARY_SHA256:
        raise ValueError(
            f"pinned discovery-seed summary is absent or changed: {summary}"
        )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    scores = payload.get("scores", {})
    trees = payload.get("model_tree_sha256s", {})
    weights = payload.get("model_weight_sha256s", {})
    if (
        payload.get("seed") != DISCOVERY_SEED
        or payload.get("tier") != FROZEN_TIER
        or payload.get("think_budget") != FROZEN_THINK_BUDGET
        or payload.get("model_order") != list(DISCOVERY_MODEL_ORDER)
        or payload.get("benchmark_implementation") != DISCOVERY_IMPLEMENTATION
        or trees.get("base") != FROZEN_TREE_SHA256["base"]
        or trees.get(DISCOVERY_TREATED_ARM) != FROZEN_TREE_SHA256[TREATED_ARM]
        or weights.get("base") != FROZEN_WEIGHTS_SHA256["base"]
        or weights.get(DISCOVERY_TREATED_ARM) != FROZEN_WEIGHTS_SHA256[TREATED_ARM]
    ):
        raise ValueError("discovery summary is not the frozen seed-78154 event")
    for label in ("base", DISCOVERY_TREATED_ARM):
        row = scores.get(label, {})
        if (
            set(row.get("per_family", {})) != set(PUBLIC_FAMILIES)
            or not _valid_score(row.get("aggregate"))
            or any(not _valid_score(value) for value in row["per_family"].values())
        ):
            raise ValueError(f"discovery summary violates the score shape: {label}")
    base_family = scores["base"]["per_family"]
    treated_family = scores[DISCOVERY_TREATED_ARM]["per_family"]
    strict_wins = [
        family for family in PUBLIC_FAMILIES
        if treated_family[family] > base_family[family]
    ]
    recorded = (
        payload.get("goal_gate", {})
        .get("per_arm", {})
        .get(DISCOVERY_TREATED_ARM, {})
    )
    if (
        len(strict_wins) != len(PUBLIC_FAMILIES)
        or recorded.get("goal_gate_pass") is not True
        or recorded.get("strict_wins") != len(PUBLIC_FAMILIES)
        or recorded.get("wins") != strict_wins
    ):
        raise ValueError(
            "discovery summary does not carry the recorded 10/10 goal-gate pass"
        )


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


def verify_lineage_package() -> None:
    """Fail closed unless the standalone lineage package matches every pin.

    Covers the six copied datasets (sha256), the three trainer copies plus
    the merger copy (sha256), the vendored root adapter (exact file set,
    sha256 and size per file), and the manifest — whose recorded recipe
    must agree with these pins (dataset files/shas/rows in stage order,
    per-stage trainer shas, distinct fixed training seeds, root adapter
    hashes, and a final merge whose expected output IS the pinned
    hygiene_explore composite).
    """
    for name, (digest, _rows) in sorted(LINEAGE_DATASETS.items()):
        path = LINEAGE_DIR / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"lineage dataset copy is absent or changed: {path}")
    for relative, digest in sorted(LINEAGE_TRAINERS.items()):
        path = SCRIPTS / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"lineage trainer copy is absent or changed: {path}")
    if not LINEAGE_ROOT.is_dir() or LINEAGE_ROOT.is_symlink():
        raise ValueError(f"vendored lineage root adapter is absent: {LINEAGE_ROOT}")
    names = {child.name for child in LINEAGE_ROOT.iterdir()}
    if names != set(LINEAGE_ROOT_FILES):
        raise ValueError(
            "vendored lineage root file set changed: "
            f"missing={sorted(set(LINEAGE_ROOT_FILES) - names)}, "
            f"unexpected={sorted(names - set(LINEAGE_ROOT_FILES))}"
        )
    for name, (digest, size) in sorted(LINEAGE_ROOT_FILES.items()):
        path = LINEAGE_ROOT / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            raise ValueError(f"vendored lineage root file changed: {path}")
    if not LINEAGE_MANIFEST.is_file():
        raise ValueError(f"lineage manifest is absent: {LINEAGE_MANIFEST}")
    manifest = json.loads(LINEAGE_MANIFEST.read_text(encoding="utf-8"))
    stages = manifest.get("stages", [])
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("model", {}).get("id") != MODEL_ID
        or manifest.get("model", {}).get("revision") != MODEL_REVISION
        or len(stages) != len(LINEAGE_DATASETS)
    ):
        raise ValueError("lineage manifest disagrees with the frozen design")
    dataset_order = sorted(LINEAGE_DATASETS)
    trainer_shas = set(LINEAGE_TRAINERS.values())
    for index, row in enumerate(stages, start=1):
        name = dataset_order[index - 1]
        digest, rows = LINEAGE_DATASETS[name]
        if (
            row.get("stage") != index
            or row.get("dataset", {}).get("file") != f"data/lineage/{name}"
            or row.get("dataset", {}).get("sha256") != digest
            or row.get("dataset", {}).get("rows") != rows
            or row.get("trainer_sha256") not in trainer_shas
            or row.get("seed") != LINEAGE_TRAINING_SEEDS[index - 1]
            or row.get("warm_start")
            != ("root_adapter" if index == 1 else f"stage {index - 1}")
        ):
            raise ValueError(f"lineage manifest stage {index} disagrees with the pins")
    root = manifest.get("root_adapter", {})
    final = manifest.get("final_merge", {}).get("expected_output", {})
    if (
        root.get("vendored_path")
        != LINEAGE_ROOT.relative_to(ROOT).as_posix()
        or root.get("weights_sha256")
        != LINEAGE_ROOT_FILES["adapter_model.safetensors"][0]
        or root.get("config_sha256") != LINEAGE_ROOT_FILES["adapter_config.json"][0]
        or manifest.get("merger", {}).get("sha256")
        != LINEAGE_TRAINERS["merge_adapter.py"]
        or final.get("weights_sha256") != FROZEN_WEIGHTS_SHA256[TREATED_ARM]
        or final.get("published_tree_sha256") != FROZEN_TREE_SHA256[TREATED_ARM]
    ):
        raise ValueError(
            "lineage manifest root/merge pins disagree with the frozen design"
        )


def verify_pins(deep: bool) -> None:
    if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
        raise ValueError("trusted gateway is absent or changed")
    verify_discovery_reference()
    verify_lineage_package()
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
    audits = seed_freshness_audits()
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "goal_gate_confirmation_design",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "loaded": False, "calls": 0},
        "event": {
            "name": FROZEN_NAME,
            "tier": FROZEN_TIER,
            "think_budget": FROZEN_THINK_BUDGET,
            "seeds": list(SEED_ORDER),
            "seed_major_frozen_order": True,
            "model_order": list(MODEL_ORDER),
            "sequential_same_seed_runs": True,
            "gateway_runs_total": len(SEED_ORDER) * len(MODEL_ORDER),
            "per_seed_event_dirs": {
                str(seed): (
                    f"runs/benchmark/{FROZEN_TIER}_tb{FROZEN_THINK_BUDGET}"
                    f"_seed{seed}_{FROZEN_NAME}"
                )
                for seed in SEED_ORDER
            },
            "k_seed_ledger": "runs/benchmark_events.jsonl",
            "ledger_write_ahead_per_seed": True,
            "closed_records_pin_summary_and_receipt_sha256s": True,
            "readout_requires_complete_ledger": True,
            "closed_seed_refused_forever": True,
            "completed_seeds_never_rerun": True,
            "unopened_seed_requires_clean_slate": True,
            "crashed_summary_recovery": (
                "byte-identical deterministic regeneration; divergence "
                "refuses with both digests"
            ),
            "overall_completes_only_when_all_seeds_close": True,
            "terminal_readout": "runs/benchmark/confirmation_readout.json",
        },
        "seed_freshness_audits": audits,
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
        "discovery_reference": {
            "seed": DISCOVERY_SEED,
            "tier": FROZEN_TIER,
            "think_budget": FROZEN_THINK_BUDGET,
            "summary": DISCOVERY_SUMMARY,
            "summary_sha256": DISCOVERY_SUMMARY_SHA256,
            "treated_arm_label": DISCOVERY_TREATED_ARM,
            "recorded_goal_gate_pass": True,
            "reported_never_counted": True,
            "benchmark_implementation": dict(DISCOVERY_IMPLEMENTATION),
            "all_six_receipts_must_match_implementation": True,
        },
        "lineage_package": {
            "directive": (
                "standalone-reproducibility gate (AGENTS.md non-negotiables; "
                "docs/quality_gates.md, owner directive 2026-07-15)"
            ),
            "manifest": LINEAGE_MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(LINEAGE_MANIFEST),
            "datasets": {
                name: {"sha256": digest, "rows": rows}
                for name, (digest, rows) in sorted(LINEAGE_DATASETS.items())
            },
            "trainers": {
                relative: digest
                for relative, digest in sorted(LINEAGE_TRAINERS.items())
            },
            "training_seeds": list(LINEAGE_TRAINING_SEEDS),
            "root_adapter": {
                "vendored_path": LINEAGE_ROOT.relative_to(ROOT).as_posix(),
                "files": {
                    name: {"sha256": digest, "size": size}
                    for name, (digest, size) in sorted(LINEAGE_ROOT_FILES.items())
                },
                "provenance_boundary": LINEAGE_ROOT_PROVENANCE,
            },
            "rebuild": "scripts/rebuild_lineage.py",
            "fast_verification": "scripts/rebuild_lineage.py --verify-inputs",
            "verified_at_receipt_time": True,
        },
        "readings": {
            "per_seed": (
                "per seed: both aggregates, the full per-family table, and "
                "the goal gate — strict wins/ties/losses of hygiene_explore "
                "vs base over the ten public families, byte-identical to the "
                "tier forensics FAMILIES; pass = ten strict wins"
            ),
            "confirmation_verdict": (
                "ordered total partition: CONFIRMED iff hygiene_explore's "
                "aggregate strictly beats base on ALL THREE seeds AND the "
                "goal gate passes on AT LEAST TWO of three; AGGREGATE_ONLY "
                "iff the aggregate strictly beats base on all three seeds "
                "but the goal-gate majority fails; NOT_REPLICATED otherwise; "
                "exact ties are never strict wins; the discovery seed 78154 "
                "is reported alongside from the sha-pinned committed summary "
                "and NEVER counted in the verdict"
            ),
            "fragility": (
                "per seed: the menders and warren margins (hygiene_explore "
                "minus base — the two single-item margins that carried the "
                "discovery pass, 0.0167 and 0.050) and which families block "
                "any seed that does not pass the goal gate"
            ),
            "budget_integrity": (
                "per arm per seed: the gateway receipt's within_budget flag "
                "and wall_seconds; if any arm at any seed has within_budget "
                "false the readout sets paired_comparison_valid: false with "
                "the reason; scores are still recorded"
            ),
        },
        "public_families": list(PUBLIC_FAMILIES),
        "fragility_families": list(FRAGILITY_FAMILIES),
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
