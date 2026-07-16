#!/usr/bin/env python3
"""Generate (or re-verify) the frozen zero-root lineage-rebuild design receipt.

Model-free construction artifact: pins everything the two GPU stages
(the six-stage zero-root replay + merge, then the ONE sealed single-seed
three-arm benchmark event) depend on BEFORE any model event, so the
stages can only run the design that was reviewed. Pinned here:

- the COPIED LINEAGE PACKAGE, byte-identical to
  qwen35_4b_goal_gate_confirmation's committed standalone package: the
  manifest (sha256), the six stage datasets (sha256 + rows), the three
  trainer copies plus the merger copy (sha256). The blend ROOT ADAPTER
  IS DELIBERATELY OMITTED — training without it is the design — and the
  receipt fails closed if a copy appears under this cell's artifact
  storage;
- the STAGE PLAN: the manifest's exact per-stage recipe (dataset, fixed
  seed, trainer variant, hyperparameters, stage 3's targeted close
  overrides, stage 1's no-w_close) with the warm-start chain REWIRED to
  the zero root: stage 1 trains FRESH (no --warm-start; the trainer's
  default fresh path zero-initializes the LoRA delta), stages 2-6
  warm-start from the previous zero-root stage. The manifest's recorded
  per-stage adapter hashes are carried as CONTRAST fields only;
- the six training seeds 42/43/44/47/51/55 as INHERITED STAGE CONSTANTS
  (deliberately reused — they are part of what "same recipe" means —
  not fresh draws);
- the ONE fresh sealed benchmark seed 78159 with a repo-wide
  grep-freshness audit (word-boundary seed-context pattern; raw
  substring hits inside floats and hashes are expected and excluded);
- tier medium, think budget 1024, the frozen three-arm order base,
  hygiene_explore_original, zero_root_hygiene_explore, the single-seed
  write-ahead ledger contract, and the frozen readings + ordered total
  consequence partition (ZERO_ROOT_COMPARABLE / ZERO_ROOT_DEGRADED);
- the base and original-composite arms by path, tree sha256 (recomputed
  from disk in write mode; the event recomputes again), weights sha256,
  and merge receipts. The zero-root arm does not exist at design time:
  its pins are fail-closed TODO-PINs inside scripts/run_benchmark.py,
  filled post-merge from the committed runs/lineage/merge.json —
  run_benchmark.py is therefore pinned by a NORMALIZED HASH: exactly the
  three TODO-pin VALUE slots (None pre-fill, the quoted 64-hex
  post-fill) are canonicalized to a fixed placeholder by a deterministic
  regex, the canonicalized bytes are sha256-hashed, and the digest plus
  the normalization rule are frozen here. Every byte of the runner
  OUTSIDE the three value slots — every guard call site included — is
  thereby byte-frozen pre- and post-fill; a handful of call-site
  substring contracts remain as belt-and-braces only;
- the trusted gateway by sha256, the reference benchmark-implementation
  signature (the discovery/confirmation events under which the original
  recorded its two 10/10 sweeps), and those two recorded events by
  sha256 (reported context, never counted);
- code_sha256 pins of every non-deferred script in this experiment.

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
FROZEN_NAME = "zero_root"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
FROZEN_SEED = 78159
MODEL_ORDER = ("base", "hygiene_explore_original", "zero_root_hygiene_explore")
ORIGINAL_ARM = "hygiene_explore_original"
ZERO_ROOT_ARM = "zero_root_hygiene_explore"
FROZEN_MODEL_PATHS = {
    "base": (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    ),
    "hygiene_explore_original": (
        ROOT / "large_artifacts" / "qwen35_4b_hygiene_explore_destack_medium"
        / "merged" / "hygiene_explore"
    ),
    "zero_root_hygiene_explore": (
        ROOT / "large_artifacts" / "qwen35_4b_zero_root_lineage_rebuild"
        / "merged" / "zero_root_hygiene_explore"
    ),
}
FROZEN_TREE_SHA256 = {
    "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
    "hygiene_explore_original": (
        "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"
    ),
}
FROZEN_WEIGHTS_SHA256 = {
    "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    "hygiene_explore_original": (
        "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"
    ),
}
WEIGHTS_SIZE_BYTES = 9_078_620_536
COMMITTED_MERGE_RECEIPTS = {
    "hygiene_explore_original": (
        "experiments/qwen35_4b_hygiene_explore_destack_medium"
        "/runs/merges/hygiene_explore.json",
        "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a",
        "hygiene_explore",
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
REFERENCE_IMPLEMENTATION = {
    "runner_sha256": (
        "a3beecd8b5c89ccfd99a172a6d85321d39b9feb6c29d12f10b2f4d7499e273cb"
    ),
    "source_inventory_sha256": (
        "218b8615a95f24da962c931e9cd2dba58d853a7bdcd2847cd8e2c42fc2c05f42"
    ),
    "source_file_count": 56,
}
# The original composite's recorded position (reported context, NEVER
# counted in this cell's consequence): the seed-78154 discovery 10/10 pass
# and the three-seed confirmation readout (AGGREGATE_ONLY; second 10/10
# sweep at seed 78157; aggregate strict wins 4/4 across the four seeds).
ORIGINAL_POSITION_REFERENCES = {
    "discovery_summary": (
        "experiments/qwen35_4b_statechain_only_dose"
        "/runs/benchmark/medium_tb1024_seed78154_pilot/summary.json",
        "6b1a43869f013e24a048a45a04e5603b45fe59488912194eb3e76a43679255fa",
    ),
    "confirmation_readout": (
        "experiments/qwen35_4b_goal_gate_confirmation"
        "/runs/benchmark/confirmation_readout.json",
        "7eb1a4c540a0f15d48c08eb6e992b38f24f21ac0231e8a26827e9b3e1198e246",
    ),
}
PUBLIC_FAMILIES = (
    "chronicle", "lockpick", "menders", "mirage", "rites",
    "siftstack", "sirens", "stockade", "toolsmith", "warren",
)
MARGIN_FAMILIES = ("menders", "rites", "warren")
CONSEQUENCES = ("ZERO_ROOT_COMPARABLE", "ZERO_ROOT_DEGRADED")
CONSEQUENCE_RULE = (
    "ordered total partition: ZERO_ROOT_COMPARABLE iff the zero-root "
    "composite's aggregate strictly beats base AND its goal-gate strict "
    "wins over the ten public families are at least the original "
    "composite's strict wins on this seed minus one; ZERO_ROOT_DEGRADED "
    "otherwise; exact ties are never strict wins"
)
CONSEQUENCE_STATEMENTS = {
    "ZERO_ROOT_COMPARABLE": (
        "the documented stages alone carry the demonstrated position; the "
        "headline model is contamination-clean end-to-end"
    ),
    "ZERO_ROOT_DEGRADED": (
        "the undocumented prefix is load-bearing at medium; its "
        "contribution is the recorded contrast"
    ),
}
PREFIX_CONTRIBUTION_FRAMING = (
    "the gym-era root's contribution at medium, one seed, cross-arm "
    "same-seed paired"
)
# Standalone lineage package (byte-identical copy of the source cell's
# committed package; the manifest byte pin alone implies every recorded
# recipe field, and the per-file pins below hold the copies to it).
SOURCE_EXPERIMENT = "qwen35_4b_goal_gate_confirmation"
LINEAGE_DIR = EXP / "data" / "lineage"
LINEAGE_MANIFEST = LINEAGE_DIR / "lineage_manifest.json"
LINEAGE_MANIFEST_SHA256 = (
    "1f49cd8b8706c8db858d30af1bf14fe09403971256514919bc24e7b6c47ff121"
)
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
FORBIDDEN_ROOT_DIR = ROOT / "large_artifacts" / EXP.name / "lineage_root"
# The undocumented root being removed (recorded for documentation/contrast
# only; these bytes are deliberately NOT vendored into this cell).
ORIGINAL_ROOT_CONTRAST = {
    "name": "blend",
    "adapter_config_sha256": (
        "cd764ae869b8a55526e283dd133e1940896b428839c6abf2e55d6ae2a0b32635"
    ),
    "adapter_weights_sha256": (
        "ad2ef4fae785debedf5e50932a79bda97869d3efc212f53d48ccb04c59e25d21"
    ),
    "source_vendored_copy": (
        "large_artifacts/qwen35_4b_goal_gate_confirmation/lineage_root/blend"
    ),
}
CODE_FILES = {
    "gen_design_receipt": SCRIPTS / "gen_design_receipt.py",
    "check_benchmark": SCRIPTS / "check_benchmark.py",
    "harness": SCRIPTS / "run.py",
    "rebuild_zero_root": SCRIPTS / "rebuild_zero_root.py",
}
# run_benchmark.py carries the three fail-closed zero-root TODO-PINs that
# are filled AFTER --stage rebuild commits runs/lineage/merge.json; a raw
# hash pin would break on the fill, so the file is pinned by a NORMALIZED
# HASH instead: the deterministic slot patterns below canonicalize exactly
# the three pin VALUE slots (None pre-fill, the quoted 64-hex post-fill) to
# PIN_PLACEHOLDER, and the sha256 of the canonicalized bytes is frozen as
# RUN_BENCHMARK_NORMALIZED_SHA256. Every byte outside the three value slots
# — every guard call site included — is byte-frozen pre- and post-fill.
RUN_BENCHMARK = SCRIPTS / "run_benchmark.py"
PIN_PLACEHOLDER = "__ZERO_ROOT_TODO_PIN__"
# (name, multiline regex, required match count). Group 2 is the ONLY
# mutable region of the file; group 1 (and group 3 where present) are kept
# verbatim so indentation, key, and trailing comma stay frozen.
PIN_SLOT_PATTERNS = (
    (
        "tree_and_weights_dict_entries",
        r'^(    "zero_root_hygiene_explore": )(None|"[0-9a-f]{64}")(,)$',
        2,
    ),
    (
        "merge_receipt_constant",
        r'^(ZERO_ROOT_MERGE_RECEIPT_SHA256 = )(None|"[0-9a-f]{64}")$',
        1,
    ),
)
# Frozen normalized hash of run_benchmark.py (recompute and re-freeze only
# under a new review); --check fails closed on one byte of drift anywhere
# outside the three canonicalized pin value slots.
RUN_BENCHMARK_NORMALIZED_SHA256 = (
    "a2d87408efe346a9456bffc7b9933725e7c395ff720491420d45daf6f28209cc"
)
# Belt-and-braces call-site contracts (the normalized hash is the
# load-bearing control; these give readable diagnostics for the six guard
# call sites a drifted runner would most plausibly lose).
RUN_BENCHMARK_CALL_SITE_CONTRACTS = (
    "        require_todo_pins_filled()",
    '        require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")',
    "        require_clean_pushed_main(",
    "        plan = ledger_plan(ledger_rows(LEDGER), args.resume)",
    "        append_ledger(opened_record())",
    "        require_zero_root_provenance()",
)
AUDIT_ROOTS = ("experiments", "knowledge", "research_programs")
AUDIT_SELF_WINDOW_LINES = 3


def audit_pattern(seed: int) -> str:
    """Word-boundary seed-context pattern for the one sealed seed.

    The digit guards exclude the expected raw-substring hits (the seed's
    digits inside floats, hashes, and longer numbers) while still failing
    closed on any true seed-context use.
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
    """Prove the sealed seed 78159 has never been used in a seed context.

    One line-based scan of the three knowledge-bearing roots. Raw
    substrings occur across the repo inside floats, hashes, and longer
    numbers; those hits are expected and excluded because the pattern
    requires ``seed`` within three non-digit characters AND the number to
    stand alone (no adjacent digits). A matching line is a self-reference
    (allowed) when the file lives inside this experiment or the
    experiment id appears within a few lines of the match. Anything else
    fails closed.
    """
    pattern = re.compile(audit_pattern(FROZEN_SEED))
    needle = str(FROZEN_SEED).encode()
    disallowed = []
    self_prefix = f"experiments/{EXP.name}/"
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
        "pattern": audit_pattern(FROZEN_SEED),
        "word_boundary_guards": True,
        "substring_hits_in_floats_and_hashes_excluded": True,
        "roots": list(AUDIT_ROOTS),
        "self_directory_excluded": f"experiments/{EXP.name}",
        "self_reference_line_window": AUDIT_SELF_WINDOW_LINES,
        "disallowed_matches": [],
        "fresh": True,
    }


def verify_original_position_references() -> None:
    """Byte-pin the two recorded original-position events (context only)."""
    for name, (relative, digest) in sorted(ORIGINAL_POSITION_REFERENCES.items()):
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(
                f"pinned original-position reference is absent or changed: "
                f"{name} ({relative})"
            )
    readout = json.loads(
        (ROOT / ORIGINAL_POSITION_REFERENCES["confirmation_readout"][0]).read_text(
            encoding="utf-8"
        )
    )
    if (
        readout.get("experiment_id") != SOURCE_EXPERIMENT
        or readout.get("verdict") not in ("CONFIRMED", "AGGREGATE_ONLY", "NOT_REPLICATED")
        or readout.get("seeds") != [78155, 78156, 78157]
    ):
        raise ValueError("confirmation readout is not the recorded original event")


def verify_lineage_package() -> None:
    """Fail closed unless the copied package matches every pin.

    Covers the byte-pinned manifest, the six copied datasets (sha256 +
    rows), the three trainer copies plus the merger copy (sha256), and the
    manifest's recorded recipe (source experiment id, stage order,
    dataset/trainer/seed agreement, recorded warm-start chain). The blend
    root must NOT be vendored here: its omission is the design.
    """
    if FORBIDDEN_ROOT_DIR.exists():
        raise ValueError(
            "the blend root adapter must NOT be vendored into this cell "
            f"(training without it is the design): {FORBIDDEN_ROOT_DIR}"
        )
    if (
        not LINEAGE_MANIFEST.is_file()
        or sha256_file(LINEAGE_MANIFEST) != LINEAGE_MANIFEST_SHA256
    ):
        raise ValueError(
            f"copied lineage manifest is absent or changed: {LINEAGE_MANIFEST}"
        )
    for name, (digest, rows) in sorted(LINEAGE_DATASETS.items()):
        path = LINEAGE_DIR / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"lineage dataset copy is absent or changed: {path}")
        observed = sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        if observed != rows:
            raise ValueError(f"lineage dataset row count changed: {path}")
    for relative, digest in sorted(LINEAGE_TRAINERS.items()):
        path = SCRIPTS / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"lineage trainer copy is absent or changed: {path}")
    manifest = json.loads(LINEAGE_MANIFEST.read_text(encoding="utf-8"))
    stages = manifest.get("stages", [])
    if (
        manifest.get("experiment_id") != SOURCE_EXPERIMENT
        or manifest.get("model", {}).get("id") != MODEL_ID
        or manifest.get("model", {}).get("revision") != MODEL_REVISION
        or manifest.get("merger", {}).get("sha256")
        != LINEAGE_TRAINERS["merge_adapter.py"]
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
    if (
        root.get("weights_sha256") != ORIGINAL_ROOT_CONTRAST["adapter_weights_sha256"]
        or root.get("config_sha256") != ORIGINAL_ROOT_CONTRAST["adapter_config_sha256"]
        or root.get("name") != ORIGINAL_ROOT_CONTRAST["name"]
    ):
        raise ValueError(
            "lineage manifest root contrast disagrees with the frozen design"
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
        relative, expected, receipt_name = COMMITTED_MERGE_RECEIPTS[label]
        receipt_path = ROOT / relative
        if not receipt_path.is_file() or sha256_file(receipt_path) != expected:
            raise ValueError(f"committed merge receipt is absent or changed: {relative}")
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            payload.get("name") != receipt_name
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


def normalize_run_benchmark_source(text: str) -> str:
    """Canonicalize exactly the three TODO-pin value slots to the placeholder.

    Deterministic: each slot pattern must match its required count (None
    pre-fill or the quoted 64-hex post-fill) or normalization fails
    closed — a file whose pin slots drifted cannot even be hashed. Only
    the value group is replaced; every other byte passes through
    verbatim, so the normalized bytes are identical pre- and post-fill.
    """
    for name, pattern, count in PIN_SLOT_PATTERNS:
        compiled = re.compile(pattern, re.MULTILINE)
        matches = compiled.findall(text)
        if len(matches) != count:
            raise ValueError(
                f"run_benchmark.py pin slot '{name}' matched "
                f"{len(matches)} times, expected {count}; the "
                "normalized-hash pin cannot be computed on a drifted file"
            )
        text = compiled.sub(
            lambda m: m.group(1)
            + PIN_PLACEHOLDER
            + (m.group(3) if m.re.groups >= 3 else ""),
            text,
        )
    return text


def normalized_runner_sha256(text: str) -> str:
    return sha256_bytes(normalize_run_benchmark_source(text).encode("utf-8"))


def verify_runner_pin() -> None:
    """The load-bearing runner control: the NORMALIZED-HASH pin.

    Everything in run_benchmark.py except the three canonicalized pin
    value slots is byte-frozen against RUN_BENCHMARK_NORMALIZED_SHA256 —
    deleting or reordering ANY guard call site (require_todo_pins_filled,
    require_verdict, require_clean_pushed_main, ledger_plan,
    append_ledger(opened_record()), require_zero_root_provenance, …)
    changes the normalized hash and fails --check, including the re-run
    at the seed-consuming boundary. The call-site substring contracts are
    belt-and-braces diagnostics only.
    """
    if not RUN_BENCHMARK.is_file():
        raise ValueError("normalized-hash-pinned run_benchmark.py is absent")
    text = RUN_BENCHMARK.read_text(encoding="utf-8")
    digest = normalized_runner_sha256(text)
    if digest != RUN_BENCHMARK_NORMALIZED_SHA256:
        raise ValueError(
            "run_benchmark.py drifted outside the three TODO-pin value "
            f"slots: normalized sha256 {digest} != frozen "
            f"{RUN_BENCHMARK_NORMALIZED_SHA256}"
        )
    missing = [
        value for value in RUN_BENCHMARK_CALL_SITE_CONTRACTS if value not in text
    ]
    if missing:
        raise ValueError(
            f"frozen run_benchmark.py guard call sites missing: {missing}"
        )
    forbidden = "benchmarks" + "/"
    if forbidden in text:
        raise ValueError("benchmark content leaked into the event runner")


def verify_pins(deep: bool) -> None:
    if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
        raise ValueError("trusted gateway is absent or changed")
    verify_original_position_references()
    verify_lineage_package()
    verify_runner_pin()
    for label in ("base", ORIGINAL_ARM):
        verify_composite(label, deep)
    for name, path in CODE_FILES.items():
        if not path.is_file():
            raise ValueError(f"pinned experiment script is absent: {name}")


def stage_plan(manifest: dict) -> list[dict]:
    """The manifest recipe with the warm-start chain rewired to zero root."""
    plan = []
    for row in manifest["stages"]:
        index = row["stage"]
        entry = {
            "stage": index,
            "name": row["name"],
            "dataset": {
                "file": row["dataset"]["file"],
                "sha256": row["dataset"]["sha256"],
                "rows": row["dataset"]["rows"],
            },
            "seed": row["seed"],
            "seed_is_inherited_stage_constant": True,
            "trainer": {"path": row["trainer"], "sha256": row["trainer_sha256"]},
            "hyperparameters": dict(row["hyperparameters"]),
            "warm_start": "fresh_zero_init" if index == 1 else f"stage {index - 1}",
            "original_warm_start": row["warm_start"],
            "adapter_out": (
                f"large_artifacts/{EXP.name}/adapters/"
                f"stage{index:02d}_{row['name']}"
            ),
            "receipt": f"runs/lineage/stage{index:02d}_{row['name']}.json",
            "original_produced_contrast": {
                "adapter_config_sha256": row["produced"]["adapter_config_sha256"],
                "adapter_weights_sha256": row["produced"]["adapter_weights_sha256"],
                "contrast_only": True,
                "note": (
                    "the recorded hashes belong to the blend-rooted chain; a "
                    "different root means different bytes, so they are never "
                    "used as verification"
                ),
            },
        }
        if "targeted_close_overrides" in row:
            entry["targeted_close_overrides"] = dict(row["targeted_close_overrides"])
        plan.append(entry)
    return plan


def build_receipt(deep: bool) -> dict:
    verify_pins(deep)
    audit = seed_freshness_audit()
    manifest = json.loads(LINEAGE_MANIFEST.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "zero_root_lineage_rebuild_design",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "loaded": False, "calls": 0},
        "question": (
            "how load-bearing is the undocumented C53-era gym-line blend "
            "root adapter at medium: the six documented contamination-free "
            "stages, replayed from a FRESH zero-initialized adapter on the "
            "official base, measured once against the original blend-rooted "
            "composite and the untouched base"
        ),
        "lineage_package": {
            "directive": (
                "standalone-reproducibility gate (AGENTS.md non-negotiables; "
                "docs/quality_gates.md): the complete stage-replay package is "
                "carried byte-identically in THIS cell"
            ),
            "source_experiment": SOURCE_EXPERIMENT,
            "manifest": LINEAGE_MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_sha256": LINEAGE_MANIFEST_SHA256,
            "datasets": {
                name: {"sha256": digest, "rows": rows}
                for name, (digest, rows) in sorted(LINEAGE_DATASETS.items())
            },
            "trainers": {
                relative: digest
                for relative, digest in sorted(LINEAGE_TRAINERS.items())
            },
            "verified_at_receipt_time": True,
        },
        "root_omission": {
            "design": (
                "the undocumented C53-era gym-line 'blend' root adapter is "
                "deliberately NOT vendored and NOT used anywhere in this "
                "cell; stage 1 trains a FRESH rank-32/alpha-64 adapter (the "
                "trainer's default fresh path; LoRA-B zero-initialized, so "
                "the delta starts at zero) — removing the root IS the "
                "treatment"
            ),
            "original_root_contrast": dict(ORIGINAL_ROOT_CONTRAST),
            "vendored_here": False,
            "must_not_exist": FORBIDDEN_ROOT_DIR.relative_to(ROOT).as_posix(),
        },
        "stage_plan": stage_plan(manifest),
        "training_seeds": {
            "values": list(LINEAGE_TRAINING_SEEDS),
            "inherited_stage_constants": True,
            "note": (
                "42/43/44/47/51/55 are reused deliberately — the fixed "
                "per-stage seeds are part of what 'same recipe' means — not "
                "fresh draws; the only fresh sealed seed in this cell is the "
                "benchmark seed 78159"
            ),
        },
        "final_merge": {
            "merger": "scripts/merge_adapter.py",
            "merger_sha256": LINEAGE_TRAINERS["merge_adapter.py"],
            "base_model": f"{MODEL_ID} @ {MODEL_REVISION} (raw HF; the merger default)",
            "adapter": "stage 6 output (hygiene_explore), zero-root chain",
            "output": (
                f"large_artifacts/{EXP.name}/merged/zero_root_hygiene_explore"
            ),
            "receipt": "runs/lineage/merge.json",
            "original_contrast": {
                "published_tree_sha256": FROZEN_TREE_SHA256[ORIGINAL_ARM],
                "weights_sha256": FROZEN_WEIGHTS_SHA256[ORIGINAL_ARM],
                "contrast_only": True,
            },
        },
        "event": {
            "name": FROZEN_NAME,
            "tier": FROZEN_TIER,
            "think_budget": FROZEN_THINK_BUDGET,
            "seed": FROZEN_SEED,
            "model_order": list(MODEL_ORDER),
            "sequential_same_seed_runs": True,
            "gateway_runs_total": len(MODEL_ORDER),
            "event_dir": (
                f"runs/benchmark/{FROZEN_TIER}_tb{FROZEN_THINK_BUDGET}"
                f"_seed{FROZEN_SEED}_{FROZEN_NAME}"
            ),
            "ledger": "runs/benchmark_events.jsonl",
            "ledger_write_ahead": True,
            "closed_record_pins_summary_and_receipt_sha256s": True,
            "readout_requires_complete_ledger": True,
            "closed_seed_refused_forever": True,
            "unopened_seed_requires_clean_slate": True,
            "crashed_summary_recovery": (
                "byte-identical deterministic regeneration; divergence "
                "refuses with both digests"
            ),
            "terminal_readout": "runs/benchmark/zero_root_readout.json",
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
            }
            for label in ("base", ORIGINAL_ARM)
        }
        | {
            ZERO_ROOT_ARM: {
                "path": FROZEN_MODEL_PATHS[ZERO_ROOT_ARM].relative_to(ROOT).as_posix(),
                "deployment": "explicit_merged_composite",
                "runtime_lora_forbidden": True,
                "pins": (
                    "TODO-PINs in scripts/run_benchmark.py (tree sha256, "
                    "weights sha256, committed merge-receipt sha256), filled "
                    "post-merge from runs/lineage/merge.json; every unfilled "
                    "pin refuses fail-closed and the committed merge receipt "
                    "is re-authenticated at event time"
                ),
                "weights_size_bytes": WEIGHTS_SIZE_BYTES,
                "anchored_to_committed_rebuild_receipts": True,
            }
        },
        "original_position_references": {
            name: {"path": relative, "sha256": digest, "counted_in_consequence": False}
            for name, (relative, digest) in sorted(
                ORIGINAL_POSITION_REFERENCES.items()
            )
        },
        "reference_implementation": {
            "signature": dict(REFERENCE_IMPLEMENTATION),
            "all_three_receipts_must_match": True,
            "source": (
                "the discovery (78154) and confirmation (78155-78157) events "
                "under which the original composite recorded its two 10/10 "
                "sweeps"
            ),
        },
        "readings": {
            "per_family": (
                "aggregates and the full per-family table for all three arms"
            ),
            "goal_gate": (
                "strict wins/ties/losses versus base for BOTH composites over "
                "the ten public families, byte-identical to the tier "
                "forensics FAMILIES and strict-win logic; pass = ten strict "
                "wins; exact ties are never strict wins"
            ),
            "prefix_contribution": (
                "zero-root minus original per family and aggregate; framing: "
                + PREFIX_CONTRIBUTION_FRAMING
            ),
            "budget_integrity": (
                "per arm: the gateway receipt's within_budget flag and "
                "wall_seconds; any false sets paired_comparison_valid false "
                "with the reason; scores are still recorded"
            ),
            "margins": (
                "menders/rites/warren margins versus base for both "
                "composites; the statechain-to-rites conversion question does "
                "NOT apply (no statechain stage), but rites/warren margins "
                "matter for the sweep reading"
            ),
        },
        "consequence": {
            "values": list(CONSEQUENCES),
            "rule": CONSEQUENCE_RULE,
            "statements": dict(CONSEQUENCE_STATEMENTS),
            "no_promotion": True,
        },
        "public_families": list(PUBLIC_FAMILIES),
        "margin_families": list(MARGIN_FAMILIES),
        "gateway": {
            "path": GATEWAY.relative_to(ROOT).as_posix(),
            "sha256": GATEWAY_SHA256,
        },
        "measurement_intake": {
            "training": "six-stage zero-root replay (the treatment is the removed root)",
            "promotion": None,
            "local_gate": None,
            "exit_zero_on_any_complete_event": True,
        },
        "checkpoint_policy": {
            "stages": ["rebuild", "benchmark"],
            "one_stage_per_invocation": True,
            "clean_pushed_main_required": True,
            "design_receipt_committed_at_head_required": True,
            "rebuild_review_verdict_required": "PASS_REBUILD",
            "benchmark_review_verdict_required": "PASS_BENCHMARK_EVENT",
            "benchmark_requires_committed_rebuild_receipts": True,
        },
        "firewall": {
            "benchmark_data_read": False,
            "gateway_receipts_only": True,
        },
        "code": {
            f"{name}_sha256": sha256_file(path)
            for name, path in sorted(CODE_FILES.items())
        },
        "run_benchmark_normalized_pin": {
            "file": "scripts/run_benchmark.py",
            "mechanism": (
                "NORMALIZED-HASH code pin: run_benchmark.py carries the "
                "three fail-closed zero-root TODO-PINs filled post-merge, so "
                "a raw hash would break on the fill; instead exactly the "
                "three pin VALUE slots (None pre-fill, the quoted 64-hex "
                "post-fill) are canonicalized to the fixed placeholder and "
                "the sha256 of the canonicalized bytes is frozen — every "
                "byte outside those slots, every guard call site included, "
                "is byte-frozen pre- and post-fill and re-checked on every "
                "--check, including at the seed-consuming boundary"
            ),
            "run_benchmark_normalized_sha256": RUN_BENCHMARK_NORMALIZED_SHA256,
            "normalization_rule": {
                "placeholder": PIN_PLACEHOLDER,
                "value_group": 2,
                "slots": [
                    {
                        "name": name,
                        "pattern": pattern,
                        "required_matches": count,
                    }
                    for name, pattern, count in PIN_SLOT_PATTERNS
                ],
                "fail_closed_on_slot_count_mismatch": True,
            },
            "call_site_contracts_retained": list(
                RUN_BENCHMARK_CALL_SITE_CONTRACTS
            ),
            "call_site_contracts_are_belt_and_braces_only": True,
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
