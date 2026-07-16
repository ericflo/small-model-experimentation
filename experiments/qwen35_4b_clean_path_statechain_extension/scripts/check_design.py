#!/usr/bin/env python3
"""Recompute the model-free clean-path statechain-extension design receipt.

Lifecycle 23: the proven statechain converter dose (lifecycle 18) applied
to the ZERO-ROOT composite (lifecycle 22), producing a single installed
model whose ENTIRE lineage is documented and contamination-free
end-to-end. This receipt pins, before any model event:

- the BYTE-COPIED treatment corpus (fresh instances would change the
  treatment; the byte-identical copy is the controlled choice) and the
  replay pool, both held to the source cell's committed sha256 pins;
- the ZERO-ROOT parent identity: path, tree sha256 (414f5829...),
  weights sha256 (6e9aad25...), authenticated against lifecycle 22's
  committed lineage merge receipt (e906caea...) AND this cell's
  byte-identical provenance copy of it;
- the STANDALONE CLEAN-CHAIN PACKAGE: the six zero-root stage datasets,
  the three lineage trainer copies plus the merger copy, lifecycle 22's
  six stage receipts + merge receipt copied as provenance documents, and
  this cell's clean-chain lineage manifest recording the dose as STAGE 7
  — with a fail-closed guarantee that NO blend root is vendored
  anywhere in this cell (the clean chain is the point);
- every seed: namespace 55,150, training 73, gate 88,041, retention
  screens 88,042/88,044/88,045 (88,043 SKIPPED — taken by
  qwen35_4b_counterfactual_plan_reflection_transfer), sealed aggregate
  78,160;
- scripts/run_benchmark.py by a NORMALIZED HASH (lifecycle 22's
  mechanism): exactly the six trained-arm TODO-pin VALUE slots (None
  pre-fill, the quoted 64-hex post-fill) are canonicalized to a fixed
  placeholder by deterministic regexes with fail-closed match counts,
  and the sha256 of the canonicalized bytes is frozen here — every byte
  of the runner OUTSIDE those slots, every guard call site included, is
  byte-frozen pre- and post-fill;
- substring lifecycle contracts on the training/merge/harness/gate
  scripts (belt-and-braces; the normalized hash is the load-bearing
  runner control).

``--check`` recomputes the receipt byte-identically; write mode refuses
to overwrite. run_benchmark.py re-runs this check as a subprocess at the
seed-consuming boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(EXP / "scripts"))

import gen_statechain_curriculum as statechain  # noqa: E402
from check_local import ANSWER_NORMALIZATION  # noqa: E402


OUT = EXP / "data" / "design_receipt.json"
MANIFEST = EXP / "data" / "corpus_manifest.json"
MANIFEST_SHA256 = "b8ceeb20157f4839f0566cf7bcf971128b0a3439ff488287fef1942fb1751667"
TREATMENT_PATH = EXP / "data" / "sft_statechain_only.jsonl"
TREATMENT_SHA256 = "ab6c78458eb8a41f42ebee25e79354d42beb558f78bb3d348dd7902ca3a9bad3"
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
# The SOURCE construction seed is INHERITED with the byte-copied corpus, not
# a fresh draw of this cell.
SOURCE_CONSTRUCTION_SEED = 77140
SOURCE_STATECHAIN_EXP = ROOT / "experiments" / "qwen35_4b_statechain_only_dose"
PARENT_EXP = ROOT / "experiments" / "qwen35_4b_zero_root_lineage_rebuild"
PARENT = (
    ROOT / "large_artifacts" / PARENT_EXP.name / "merged" / "zero_root_hygiene_explore"
)
PARENT_RECEIPT = PARENT_EXP / "runs" / "lineage" / "merge.json"
PARENT_RECEIPT_SHA256 = "e906caea7c4b86f4a3eacb96affb7cc2fa9b7cc11e11b634b651cabc5dd01d2b"
PARENT_INNER_RECEIPT_SHA256 = "f8981f4638d901471eb41aff0ffd0bfac88aebd6e3e4d4db1e1c733be16709c0"
PARENT_TREE_SHA256 = "414f582950bf60fed2fe462cd141ab98d0f772087b4f9c6bc5aa12f03f379e7d"
PARENT_WEIGHTS_SHA256 = "6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932"
PARENT_WEIGHTS_SIZE_BYTES = 9_078_620_536
# The stage-6 adapter that PRODUCED the zero-root composite (recorded inside
# lifecycle 22's committed merge receipt). No warm start exists in this cell
# — both fresh rank-32 adapters train on the composite itself.
PARENT_LINEAGE_ADAPTER_CONFIG_SHA256 = "ffd5ef97871e98190b2474dabeb7ef865d13d0615d50d6f130e8a0aacd1c919a"
PARENT_LINEAGE_ADAPTER_WEIGHTS_SHA256 = "627c8df24937214afa51306d1435bf419d11d5c3406ef5782e2ed5be4fee7926"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ARMS = ("replay_ctl4", "statechain_clean")
CANDIDATE_ARMS = ("statechain_clean",)
PARENT_EVAL_LABEL = "zero_root_parent"
GATE_ARMS = ("zero_root_parent", "replay_ctl4", "statechain_clean")
TREATMENT_KIND = "u_statechain"
LOCAL_SEED = 88041
SCREEN_SEEDS = (88042, 88044, 88045)
SKIPPED_SEED = 88043
AGGREGATE_SEED = 78160
NAMESPACE_SEED = 55150
TRAINING_SEED = 73

# --- Standalone clean-chain package pins -----------------------------------
LINEAGE_DIR = EXP / "data" / "lineage"
LINEAGE_MANIFEST = LINEAGE_DIR / "lineage_manifest.json"
LINEAGE_MANIFEST_SHA256 = "e0589542b0047eae002a2650a81ad76a1ddedc10033ecc1eead2006e9ebc8839"
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
# Lifecycle 22's committed receipts, copied byte-identically into this cell
# as provenance documents (copy path -> (committed original, sha256)).
PROVENANCE_DIR = LINEAGE_DIR / "provenance"
PROVENANCE_RECEIPTS = {
    "stage01_replay_refresh.json": (
        "9fcb4521c2cd7f90a3325a09000ae0766f3a5b7e6f5042cf3ba177a4f6816b53"
    ),
    "stage02_designed160.json": (
        "fac2a348dd1608f82657bc3b0b01491da6a5c7938aa466fa6e9e7979c446cd74"
    ),
    "stage03_close_xi.json": (
        "0bb04493a11c6db879448c61b36f53add0774f4277b9051e3d997cd61f1b0854"
    ),
    "stage04_replay_after_close.json": (
        "94eac7ac9cf0295515a182fe5c4004a5c24fe798bfab252f70b00ae4ead4d6bd"
    ),
    "stage05_designed_fresh.json": (
        "6a7c94dfa0423af9ac15f52a4954885bfbe171bce707d76186ce33e3b7d73f05"
    ),
    "stage06_hygiene_explore.json": (
        "c5d85cf46bfb99b42eaa6d6a965c6e983512c7544da40de4a15c1159ff6fda48"
    ),
    "merge.json": PARENT_RECEIPT_SHA256,
}
# The clean chain is the point: NO blend root may exist anywhere in this
# cell. Fail closed on the artifact-storage directory AND on any manifest
# reference to the undocumented root adapter's bytes.
FORBIDDEN_ROOT_DIR = ROOT / "large_artifacts" / EXP.name / "lineage_root"
BLEND_ROOT_WEIGHTS_SHA256 = (
    "ad2ef4fae785debedf5e50932a79bda97869d3efc212f53d48ccb04c59e25d21"
)
BLEND_ROOT_CONFIG_SHA256 = (
    "cd764ae869b8a55526e283dd133e1940896b428839c6abf2e55d6ae2a0b32635"
)

# --- Normalized-hash pin of scripts/run_benchmark.py ------------------------
# run_benchmark.py carries SIX fail-closed trained-arm TODO-PINs (tree,
# weights, committed merge receipt for each of replay_ctl4 and
# statechain_clean) that are filled AFTER the merges publish; a raw hash pin
# would break on the fill, so the file is pinned by a NORMALIZED HASH: the
# deterministic slot patterns below canonicalize exactly the six pin VALUE
# slots (None pre-fill, the quoted 64-hex post-fill) to PIN_PLACEHOLDER, and
# the sha256 of the canonicalized bytes is frozen as
# RUN_BENCHMARK_NORMALIZED_SHA256. Every byte outside the six value slots —
# every guard call site included — is byte-frozen pre- and post-fill.
RUN_BENCHMARK = SCRIPTS / "run_benchmark.py"
PIN_PLACEHOLDER = "__CLEAN_PATH_TODO_PIN__"
# (name, multiline regex, required match count). Group 2 is the ONLY mutable
# region of the file; group 1 (and group 3 where present) are kept verbatim
# so indentation, key, and trailing comma stay frozen.
PIN_SLOT_PATTERNS = (
    (
        "replay_ctl4_tree_and_weights_dict_entries",
        r'^(    "replay_ctl4": )(None|"[0-9a-f]{64}")(,)$',
        2,
    ),
    (
        "statechain_clean_tree_and_weights_dict_entries",
        r'^(    "statechain_clean": )(None|"[0-9a-f]{64}")(,)$',
        2,
    ),
    (
        "replay_ctl4_merge_receipt_constant",
        r'^(REPLAY_CTL4_MERGE_RECEIPT_SHA256 = )(None|"[0-9a-f]{64}")$',
        1,
    ),
    (
        "statechain_clean_merge_receipt_constant",
        r'^(STATECHAIN_CLEAN_MERGE_RECEIPT_SHA256 = )(None|"[0-9a-f]{64}")$',
        1,
    ),
)
# Frozen normalized hash of run_benchmark.py (recompute and re-freeze only
# under a new review); --check fails closed on one byte of drift anywhere
# outside the six canonicalized pin value slots.
RUN_BENCHMARK_NORMALIZED_SHA256 = (
    "41c22c545a4c2c5c989dff6fe642b3d339cacc0f4f0f92134e1f298e64c68682"
)
# Belt-and-braces call-site contracts (the normalized hash is the
# load-bearing control; these give readable diagnostics for the guard call
# sites a drifted runner would most plausibly lose).
RUN_BENCHMARK_CALL_SITE_CONTRACTS = (
    "        require_todo_pins_filled()",
    '        require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")',
    "        promotion = authenticate_local_promotion(args.candidate)",
    "        require_clean_pushed_main(",
    "        require_unconsumed_ledger(LEDGER, opened_record, args.resume)",
    "        require_zero_root_parent_provenance(model)",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_frozen_corpora() -> dict:
    if MANIFEST_SHA256 is None:
        raise ValueError("corpus manifest pin is unfilled (TODO-PIN)")
    for path, expected in (
        (MANIFEST, MANIFEST_SHA256),
        (TREATMENT_PATH, TREATMENT_SHA256),
        (REPLAY_PATH, REPLAY_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen corpus artifact is absent or changed: {path}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    replay_rows = len(REPLAY_PATH.read_text(encoding="utf-8").splitlines())
    expected_kinds = {TREATMENT_KIND: 160}
    expected_surfaces = {
        formalism: 40 for formalism in statechain.STATECHAIN_FORMALISMS
    }
    provenance = manifest.get("treatment_provenance", {})
    if (
        manifest.get("experiment_id") != EXP.name
        or provenance.get("policy") != "byte_identical_copy_of_the_proven_dose"
        or provenance.get("source_experiment") != SOURCE_STATECHAIN_EXP.name
        or provenance.get("source_construction_seed") != SOURCE_CONSTRUCTION_SEED
        or provenance.get("regenerated_byte_identically_via_copied_generator")
        is not True
        or manifest.get("banned_vocabulary_checked") is not True
        or manifest.get("corpus", {}).get("rows") != 160
        or manifest.get("corpus", {}).get("sha256") != TREATMENT_SHA256
        or manifest.get("corpus", {}).get("kinds") != expected_kinds
        or manifest.get("corpus", {}).get("surfaces") != expected_surfaces
        or manifest.get("replay", {}).get("sha256") != REPLAY_SHA256
        or replay_rows != REPLAY_ROWS
        # The inherited fresh-surface and row-overlap audits live in the
        # SOURCE cell's committed manifest; the copy is only frozen when the
        # inherited totals are zero-hit / zero-overlap.
        or manifest.get("treatment_provenance", {})
        .get("inherited_audits", {})
        .get("surface_freshness_total_hits")
        != 0
        or manifest.get("treatment_provenance", {})
        .get("inherited_audits", {})
        .get("row_overlap_total")
        != 0
    ):
        raise ValueError("corpus manifest violates the frozen byte-copy contract")
    return manifest


def check_parent_identity() -> None:
    """Authenticate the zero-root parent against lifecycle 22's receipt."""
    parent_receipt = json.loads(PARENT_RECEIPT.read_text(encoding="utf-8"))
    provenance_copy = PROVENANCE_DIR / "merge.json"
    inner = PARENT / "merge_receipt.json"
    weights = PARENT / "model.safetensors"
    if (
        sha256_file(PARENT_RECEIPT) != PARENT_RECEIPT_SHA256
        or not provenance_copy.is_file()
        or provenance_copy.read_bytes() != PARENT_RECEIPT.read_bytes()
        or parent_receipt.get("experiment_id") != PARENT_EXP.name
        or parent_receipt.get("stage") != "merge"
        or parent_receipt.get("name") != "zero_root_hygiene_explore"
        or parent_receipt.get("base_model", {}).get("id") != MODEL_ID
        or parent_receipt.get("base_model", {}).get("revision") != MODEL_REVISION
        or Path(parent_receipt.get("merged", "")).resolve() != PARENT.resolve()
        or parent_receipt.get("output_tree_sha256") != PARENT_TREE_SHA256
        or parent_receipt.get("weights_sha256") != PARENT_WEIGHTS_SHA256
        or parent_receipt.get("weights_size_bytes") != PARENT_WEIGHTS_SIZE_BYTES
        or parent_receipt.get("inner_merge_receipt_sha256")
        != PARENT_INNER_RECEIPT_SHA256
        or parent_receipt.get("adapter", {}).get("adapter_config_sha256")
        != PARENT_LINEAGE_ADAPTER_CONFIG_SHA256
        or parent_receipt.get("adapter", {}).get("adapter_weights_sha256")
        != PARENT_LINEAGE_ADAPTER_WEIGHTS_SHA256
        or not inner.is_file()
        or sha256_file(inner) != PARENT_INNER_RECEIPT_SHA256
        or not weights.is_file()
        or weights.stat().st_size != PARENT_WEIGHTS_SIZE_BYTES
    ):
        raise ValueError("authenticated zero-root-parent identity changed")


def check_clean_chain_package() -> None:
    """The standalone clean-chain package, with NO blend root anywhere."""
    if FORBIDDEN_ROOT_DIR.exists():
        raise ValueError(
            "a lineage_root directory must NOT exist in this cell's artifact "
            f"storage (the clean chain is the point): {FORBIDDEN_ROOT_DIR}"
        )
    if LINEAGE_MANIFEST_SHA256 is None:
        raise ValueError("clean-chain lineage manifest pin is unfilled (TODO-PIN)")
    if (
        not LINEAGE_MANIFEST.is_file()
        or sha256_file(LINEAGE_MANIFEST) != LINEAGE_MANIFEST_SHA256
    ):
        raise ValueError(
            f"clean-chain lineage manifest is absent or changed: {LINEAGE_MANIFEST}"
        )
    manifest_text = LINEAGE_MANIFEST.read_text(encoding="utf-8")
    for token in (BLEND_ROOT_WEIGHTS_SHA256, BLEND_ROOT_CONFIG_SHA256, "root_adapter"):
        if token in manifest_text:
            raise ValueError(
                "the clean-chain lineage manifest references the undocumented "
                f"blend root: {token}"
            )
    manifest = json.loads(manifest_text)
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("stage") != "clean_chain_lineage_manifest"
        or manifest.get("framing") != "clean_chain"
        or len(manifest.get("stages", [])) != 7
        or manifest.get("stages", [{}])[0].get("warm_start") != "fresh_zero_init"
    ):
        raise ValueError("clean-chain lineage manifest violates the frozen design")
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
    for name, digest in sorted(PROVENANCE_RECEIPTS.items()):
        copy = PROVENANCE_DIR / name
        original = PARENT_EXP / "runs" / "lineage" / name
        if (
            not copy.is_file()
            or sha256_file(copy) != digest
            or not original.is_file()
            or copy.read_bytes() != original.read_bytes()
        ):
            raise ValueError(
                "lifecycle-22 provenance receipt copy is absent, changed, or "
                f"diverged from the committed original: {name}"
            )


def normalize_run_benchmark_source(text: str) -> str:
    """Canonicalize exactly the six TODO-pin value slots to the placeholder.

    Deterministic: each slot pattern must match its required count (None
    pre-fill or the quoted 64-hex post-fill) or normalization fails closed
    — a file whose pin slots drifted cannot even be hashed. Only the value
    group is replaced; every other byte passes through verbatim, so the
    normalized bytes are identical pre- and post-fill.
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

    Everything in run_benchmark.py except the six canonicalized pin value
    slots is byte-frozen against RUN_BENCHMARK_NORMALIZED_SHA256 — deleting
    or reordering ANY guard call site (require_todo_pins_filled,
    require_verdict, authenticate_local_promotion,
    require_clean_pushed_main, require_unconsumed_ledger,
    require_zero_root_parent_provenance, ...) changes the normalized hash
    and fails --check, including the re-run at the seed-consuming boundary.
    The call-site substring contracts are belt-and-braces diagnostics only.
    """
    if RUN_BENCHMARK_NORMALIZED_SHA256 is None:
        raise ValueError("run_benchmark normalized-hash pin is unfilled (TODO-PIN)")
    if not RUN_BENCHMARK.is_file():
        raise ValueError("normalized-hash-pinned run_benchmark.py is absent")
    text = RUN_BENCHMARK.read_text(encoding="utf-8")
    digest = normalized_runner_sha256(text)
    if digest != RUN_BENCHMARK_NORMALIZED_SHA256:
        raise ValueError(
            "run_benchmark.py drifted outside the six TODO-pin value "
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


def check_lifecycle_contracts() -> None:
    """Substring contracts survive later TODO-PIN fills; hashes would not."""
    trial = (EXP / "scripts" / "train_trial.py").read_text(encoding="utf-8")
    merge = (EXP / "scripts" / "merge_trained_arm.py").read_text(encoding="utf-8")
    harness = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
    gate = (EXP / "scripts" / "check_local.py").read_text(encoding="utf-8")
    bench = RUN_BENCHMARK.read_text(encoding="utf-8")
    required_trial = (
        "STREAM_TOKEN_RECEIPT_SHA256",
        "EXPECTED_ROWS = 1520",
        '"optimizer_steps": 190,',
        '"seed": 73,',
        "LORA_RANK = 32",
        "LORA_ALPHA = 64",
        'MODEL_PATH_WEIGHTS_SHA256 = "6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932"',
        'MODEL_PATH_TREE_SHA256 = "414f582950bf60fed2fe462cd141ab98d0f772087b4f9c6bc5aa12f03f379e7d"',
        '"fresh_adapter": True,',
        "--model-path",
        "PASS_CONTROL_TRAINING",
        'ARM_PREREQUISITES = {\n    "replay_ctl4": (),\n    "statechain_clean": ("replay_ctl4",),\n}',
        "PUBLISHED_ARM_HASHES",
    )
    required_merge = (
        'MERGER_SHA256 = "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672"',
        'payload.get("applied_lora_modules") != 128',
        "--base-model",
        'BASE_COMPOSITE_RECEIPT_SHA256 = "f8981f4638d901471eb41aff0ffd0bfac88aebd6e3e4d4db1e1c733be16709c0"',
        "PASS_CONTROL_MERGE",
        "LOCAL_SEED = 88041",
        "LOCAL_ROWS = 352",
    )
    required_harness = (
        "def require_pushed_checkpoint",
        '"train-control"',
        '"train-candidate"',
        '"merge-arms"',
        '"local"',
        '"benchmark"',
        "LOCAL_SEED = 88041",
        "SCREEN_SEEDS = (88042, 88044, 88045)",
        "AGGREGATE_SEED = 78160",
        "rebuild_clean_chain.py",
        "PASS_CONTROL_TRAINING",
        "PASS_CONTROL_MERGE",
        "PASS_LOCAL_EVENT",
        "PASS_BENCHMARK_EVENT",
    )
    required_gate = (
        "SEED = 88041",
        "SCREEN_SEEDS = (88042, 88044, 88045)",
        "AGGREGATE_SEED = 78160",
        "RETENTION_CORRECT_BAND = 5",
        "RETENTION_CAP_BAND = 3",
        "RETENTION_PARSED_BAND = 3",
        "def normalize_answer",
        "def pooled_sd",
        "def evaluate_promotion",
        "def pooled_band_checks",
        '"single_kind_dose_no_per_kind_split"',
        '"pooled_k3"',
        "no_absolute_per_kind_floors",
    )
    required_bench = (
        'FROZEN_NAME = "pilot"',
        'FROZEN_TIER = "medium"',
        "FROZEN_THINK_BUDGET = 1024",
        "FROZEN_SEED = 78160",
        "def require_unconsumed_ledger",
        "def require_todo_pins_filled",
        "def require_zero_root_parent_provenance",
        "def _valid_score",
        "def pilot_gate",
        "def goal_gate_reading",
        "def rites_conversion_reading",
        "PASS_BENCHMARK_EVENT",
        "POWER_STATEMENT",
        'BASE_MERGE_RECEIPT_SHA256 = (\n    "25aee794cfffe4d58110defc61177edef1f5324e47deb28fbd3cb7ccd61ae54f"\n)',
    )
    if any(value not in trial for value in required_trial):
        raise ValueError("frozen training wrapper contract changed")
    # The fresh-adapter contract: no warm-start pathway may exist anywhere in
    # the training or merge wrapper of this cell (prose mentions excluded).
    for token in ("--warm-start", "warm_start", "warm-start"):
        if token in trial or token in merge:
            raise ValueError("a warm-start path leaked into the fresh-adapter cell")
    if any(value not in merge for value in required_merge):
        raise ValueError("frozen merge wrapper contract changed")
    if any(value not in harness for value in required_harness):
        raise ValueError("frozen stage-harness contract changed")
    if any(value not in gate for value in required_gate):
        raise ValueError("frozen promotion-gate contract changed")
    if any(value not in bench for value in required_bench):
        raise ValueError("frozen benchmark-runner contract changed")
    forbidden = "benchmarks" + "/"
    if any(forbidden in text for text in (trial, merge, harness, gate, bench)):
        raise ValueError("benchmark content leaked into the lifecycle scripts")


def build_receipt() -> dict:
    manifest = check_frozen_corpora()
    check_parent_identity()
    check_clean_chain_package()
    verify_runner_pin()
    check_lifecycle_contracts()
    banned = tuple(statechain.BANNED_PROMPT_TOKENS)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "clean_path_statechain_extension_design",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "loaded": False, "calls": 0},
        "parent": {
            "experiment": PARENT_EXP.name,
            "arm": "zero_root_hygiene_explore",
            "eval_label": PARENT_EVAL_LABEL,
            "deployment": "explicit_merged_composite",
            "training_base_via_model_path": True,
            "committed_lineage_merge_receipt_sha256": PARENT_RECEIPT_SHA256,
            "provenance_copy": "data/lineage/provenance/merge.json",
            "inner_merge_receipt_sha256": PARENT_INNER_RECEIPT_SHA256,
            "tree_sha256": PARENT_TREE_SHA256,
            "weights_sha256": PARENT_WEIGHTS_SHA256,
            "weights_size_bytes": PARENT_WEIGHTS_SIZE_BYTES,
            "runtime_lora_forbidden": True,
            "lineage_adapter_receipt_pins": {
                "config_sha256": PARENT_LINEAGE_ADAPTER_CONFIG_SHA256,
                "weights_sha256": PARENT_LINEAGE_ADAPTER_WEIGHTS_SHA256,
            },
            "zero_root_position": (
                "sealed 78159: aggregate 0.3462, 7/10 strict wins vs base, "
                "zero losses (lifecycle 22's committed readout)"
            ),
        },
        "clean_chain": {
            "framing": (
                "the mission's cleanest artifact: every training stage of the "
                "final model — six zero-root stages + this cell's statechain "
                "dose as stage 7 — is documented, receipted, and "
                "contamination-free; NO undocumented root exists anywhere in "
                "the lineage"
            ),
            "package": {
                "manifest": LINEAGE_MANIFEST.relative_to(ROOT).as_posix(),
                "manifest_sha256": LINEAGE_MANIFEST_SHA256,
                "datasets": {
                    name: {"sha256": digest, "rows": rows}
                    for name, (digest, rows) in sorted(LINEAGE_DATASETS.items())
                },
                "trainers": dict(sorted(LINEAGE_TRAINERS.items())),
                "provenance_receipts": dict(sorted(PROVENANCE_RECEIPTS.items())),
                "rebuild_script": "scripts/rebuild_clean_chain.py",
            },
            "blend_root_forbidden": {
                "must_not_exist": FORBIDDEN_ROOT_DIR.relative_to(ROOT).as_posix(),
                "manifest_must_not_reference": [
                    BLEND_ROOT_WEIGHTS_SHA256,
                    BLEND_ROOT_CONFIG_SHA256,
                ],
                "verified_at_receipt_time": True,
            },
        },
        "seeds": {
            "source_construction_inherited": SOURCE_CONSTRUCTION_SEED,
            "stream_slot_matching_namespace": NAMESPACE_SEED,
            "training": TRAINING_SEED,
            "local_gate": LOCAL_SEED,
            "retention_screens": list(SCREEN_SEEDS),
            "seed_freshness": {
                "pinned": [
                    NAMESPACE_SEED,
                    TRAINING_SEED,
                    LOCAL_SEED,
                    *SCREEN_SEEDS,
                    AGGREGATE_SEED,
                ],
                "substitution_required": False,
                "skipped": {
                    str(SKIPPED_SEED): (
                        "taken as the retention seed of "
                        "qwen35_4b_counterfactual_plan_reflection_transfer; the "
                        "screen sequence skips it (88042, 88044, 88045)"
                    )
                },
                "verified": (
                    "every pinned seed verified grep-fresh in seed contexts "
                    "across experiments/, knowledge/, research_programs/, "
                    "scripts/, and configs/ at design time; 55150/88041/88042/"
                    "88044/88045/78160 have zero seed-context hits anywhere; "
                    "training seed 73 has zero seed-context hits in this "
                    "program's cells (paired-trial training-seed sequence 67, "
                    "71, 73) — its only repo occurrences are run seeds of two "
                    "pre-program, non-lineage experiments "
                    "(qwen_action_conditioned_vm_echo_policy_iteration, "
                    "qwen_onpolicy_repair_compiler) and per-row data fields, "
                    "recorded here and excluded exactly like the inherited "
                    "42/43/44 stage constants that recur repo-wide"
                ),
                "rule_if_collision": "next free integer, recorded here",
            },
            "aggregate_sealed": AGGREGATE_SEED,
            "source_construction_note": (
                "77140 is the SOURCE cell's construction seed, inherited with "
                "the byte-copied treatment; this cell draws no construction "
                "seed of its own"
            ),
        },
        "corpora": {
            "manifest_sha256": MANIFEST_SHA256,
            "treatment": {
                "path": TREATMENT_PATH.relative_to(EXP).as_posix(),
                "rows": 160,
                "sha256": TREATMENT_SHA256,
                "kinds": manifest["corpus"]["kinds"],
                "surfaces": manifest["corpus"]["surfaces"],
                "balance": manifest["balance"],
                "provenance": {
                    "policy": "byte_identical_copy_of_the_proven_dose",
                    "source_experiment": SOURCE_STATECHAIN_EXP.name,
                    "source_construction_seed": SOURCE_CONSTRUCTION_SEED,
                    "builder": "scripts/build_corpus.py",
                    "regenerated_byte_identically_via_copied_generator": True,
                },
            },
            "replay": {
                "path": REPLAY_PATH.relative_to(EXP).as_posix(),
                "rows": REPLAY_ROWS,
                "sha256": REPLAY_SHA256,
                "inheritance": {
                    "byte_identical_to_every_predecessor_copy": True,
                },
            },
            "banned_vocabulary": {
                "tokens": len(banned),
                "sha256": sha256_bytes("\n".join(banned).encode("utf-8")),
                "checked_by_generator": True,
            },
        },
        "arms": {
            "trained_control": "replay_ctl4",
            "trained_candidate": "statechain_clean",
            "control_trains_first": True,
            "parent_eval_label": PARENT_EVAL_LABEL,
            "gate_arms": list(GATE_ARMS),
        },
        "stream_geometry": {
            "rows_per_arm": 1520,
            "shared_core_replay_rows": 1280,
            "core_stratified_by": ["family", "kind"],
            "variable_block_rows": 240,
            "blocks": {
                "replay_ctl4": {"replay_rows": 240},
                "statechain_clean": {"treatment_rows": 160, "replay_filler_rows": 80},
            },
            "blocks_and_core_pairwise_disjoint_replay_indices": True,
            "slot_order_seed": NAMESPACE_SEED,
            "exact_match_axes": ["forward", "nonzero_target", "absolute_loss_mass_x5"],
            "match_limit": 0,
            "solver": {
                "method": "scipy.optimize.milp",
                "backend": "highs",
                "mip_rel_gap": 0,
                "time_limit_seconds": 600,
                "filler_and_control_solved_jointly": True,
            },
        },
        "training_plan": {
            "training_authorized": False,
            "epochs": 1,
            "batch_size": 1,
            "grad_accum": 8,
            "optimizer_steps": 190,
            "lr": 1e-5,
            "rank": 32,
            "alpha": 64,
            "w_think": 0.2,
            "w_close": 0.2,
            "max_length": 4096,
            "seed": TRAINING_SEED,
            "rows_per_arm": 1520,
            "zero_skipped_rows_required": True,
            "fresh_adapter": True,
            "warm_start": None,
            "model_path_base_composite": {
                "path": PARENT.relative_to(ROOT).as_posix(),
                "tree_sha256": PARENT_TREE_SHA256,
                "weights_sha256": PARENT_WEIGHTS_SHA256,
            },
            "required_before_training": [
                "committed exact three-axis stream token receipt",
                "second adversarial compute review with PASS_CONTROL_TRAINING",
            ],
        },
        "local_gate": {
            "seed": LOCAL_SEED,
            "screen_seeds": list(SCREEN_SEEDS),
            "rows_per_arm": 352,
            "instruments": {
                "axis_holdout": {
                    "rows": 40,
                    "kind": TREATMENT_KIND,
                    "per_surface": 10,
                    "surfaces": list(statechain.STATECHAIN_FORMALISMS),
                    "generator": "scripts/gen_statechain_curriculum.py",
                    "fresh_instances_from_the_copied_generator": True,
                },
                "retention": {
                    "rows_per_screen": 104,
                    "per_kind": 8,
                    "skills": 13,
                    "screens": len(SCREEN_SEEDS),
                    "generator": "scripts/gen_curriculum.py",
                    "adjudication": "pooled_k3",
                },
            },
            "arms": list(GATE_ARMS),
            "answer_normalization": ANSWER_NORMALIZATION,
            "promotion": {
                "axis_total_strictly_beats_parent_and_replay": True,
                "single_kind_dose_no_per_kind_split": True,
                "per_surface_reported_not_gated": True,
                "retention_pooled_correct_band": 5,
                "retention_pooled_cap_contact_band": 3,
                "retention_pooled_parsed_band": 3,
                "bands_on_pooled_means_not_per_screen": True,
                "bands_evaluated_on_pooled_sums_times_screens": True,
                "no_absolute_per_kind_floors": True,
            },
        },
        "benchmark_plan": {
            "conditional_on_local_promotion": True,
            "name": "pilot",
            "tier": "medium",
            "think_budget": 1024,
            "aggregate_seed_sealed": AGGREGATE_SEED,
            "model_order": [
                "base",
                PARENT_EVAL_LABEL,
                "replay_ctl4",
                "statechain_clean",
            ],
            "pilot_gate": (
                "candidate aggregate strictly > base AND > replay_ctl4 AND > "
                "zero_root_parent"
            ),
            "goal_gate": (
                "all ten public families strictly > base; recorded from the "
                "same event either way, never part of the pilot pass"
            ),
            "power_statement": (
                "menders is closed (three pedagogies, the budget lever, and "
                "the 10x dose-scale cell all failed it), so the winnable "
                "ceiling is 9/10; the readings of consequence are (a) whether "
                "the statechain install CONVERTS to the rites family ON THE "
                "CLEAN LINEAGE (candidate rites vs parent/replay rites, "
                "paired) and (b) the fully-documented model's per-family "
                "profile; any 10/10 would be a menders draw and feeds a fresh "
                "confirmation cell before any claim"
            ),
            "trained_arm_pins": (
                "six fail-closed TODO-PIN slots in scripts/run_benchmark.py "
                "(tree, weights, committed merge receipt per trained arm), "
                "filled post-merge and covered by the normalized-hash pin"
            ),
        },
        "run_benchmark_normalized_pin": {
            "file": "scripts/run_benchmark.py",
            "mechanism": (
                "NORMALIZED-HASH code pin (lifecycle 22's mechanism): the six "
                "trained-arm pin VALUE slots (None pre-fill, the quoted "
                "64-hex post-fill) are canonicalized to the fixed placeholder "
                "and the sha256 of the canonicalized bytes is frozen — every "
                "byte outside those slots, every guard call site included, is "
                "byte-frozen pre- and post-fill and re-checked on every "
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
        "checkpoint_policy": {
            "next_authorized_stage": "train-control",
            "one_stage_per_invocation": True,
            "clean_pushed_main_required": True,
            "preceding_receipt_committed_at_head": True,
            "full_check_rebase_push_two_workflow_gate_between_expensive_stages": True,
        },
        "firewall": {
            "benchmark_data_read": False,
            "benchmark_gateway_exposed": False,
            "aggregate_seed_sealed": True,
        },
        "code": {
            "curriculum_generator_sha256": sha256_file(EXP / "scripts" / "gen_curriculum.py"),
            "statechain_curriculum_generator_sha256": sha256_file(
                EXP / "scripts" / "gen_statechain_curriculum.py"
            ),
            "corpus_builder_sha256": sha256_file(EXP / "scripts" / "build_corpus.py"),
            "trainer_sha256": sha256_file(EXP / "scripts" / "train_think.py"),
            "rebuild_script_sha256": sha256_file(
                EXP / "scripts" / "rebuild_clean_chain.py"
            ),
            "vendored_merger_sha256": sha256_file(
                EXP / "scripts" / "merge_adapter.py"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        receipt = build_receipt()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    value = (json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != value:
            parser.error("design receipt is absent or changed")
    else:
        if args.out.exists():
            parser.error("refusing to overwrite design receipt")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(value)
    print(json.dumps({"out": str(args.out), "sha256": sha256_bytes(value)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
