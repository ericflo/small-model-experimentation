#!/usr/bin/env python3
"""Recompute the model-free count-don't-walk-protocol design receipt.

Lifecycle 27 — the evidence-backed successor to the enumerative-repair
cell (lifecycle 26), changing ONLY the expression pedagogy. That cell
proved the enumeration discipline INSTALLS (9/40 canonical-next versus
both controls at 0/40) but expresses as a verbose linear walk whose
token cost grows with the tried-list depth — 20 of 21 unparseable gate
rows were 1,024-token cap truncations caught mid-CORRECT walk (its
committed truncation forensics). This dose teaches COUNT, DON'T WALK:
the tried list has k entries in canonical order, so the target is entry
k+1 of the frozen order — index arithmetic and direct lookup over
ranges rendered in the prompt, constant token cost in k. This receipt
pins, before any model event:

- the FRESH 160-row SINGLE-KIND ``u_count_walk`` treatment corpus (one
  kind per dose at full concentration — the design rule hardened by the
  gym-mix cell) at construction seed 77,191, 20 rows per formalism
  across the eight machine formalisms REUSED from the menders
  dose-scale cell via a byte-identical machinery copy; the corpus
  manifest carrying the per-row canonical-next re-derivation contract,
  the rendered-range-vs-enumeration equality contract, the frozen think
  length budget (five-line fixed shape; real-tokenizer cap enforced by
  measure_source_tokens.py), and the full row-overlap audit (the
  load-bearing freshness receipt — every surface token is inherited by
  design; row overlap checked against EVERYTHING including the
  reference cell's corpus, streams, and gates);
- the ZERO-ROOT parent identity: path, tree sha256 (414f5829...),
  weights sha256 (6e9aad25...), authenticated against lifecycle 22's
  committed lineage merge receipt (e906caea...) AND this cell's
  byte-identical provenance copy of it;
- the STANDALONE CLEAN-CHAIN PACKAGE: the six zero-root stage datasets,
  the three lineage trainer copies plus the merger copy, lifecycle 22's
  six stage receipts + merge receipt copied as provenance documents, and
  this cell's clean-chain lineage manifest recording the dose as STAGE 7
  — with a fail-closed guarantee that NO blend root is vendored
  anywhere in this cell;
- every seed: construction 77,191, namespace 55,171, training 85, gate
  88,056, retention screens 88,057/88,058/88,059, sealed aggregate
  78,163 — all verified grep-fresh in seed contexts (known-taken:
  88,043/88,047/88,049 and everything <= 88,055 including the reference
  cell's 88,052-88,055; benchmark seeds spent through 78,162; ONE
  next-free substitution recorded — training seed 84 is taken by
  qwen35_4b_hypothesize_verify_wall, so 85);
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

import gen_count_walk_curriculum as cw_mod  # noqa: E402
import gen_feedloop_curriculum as feedloop  # noqa: E402
from check_local import ANSWER_NORMALIZATION  # noqa: E402


OUT = EXP / "data" / "design_receipt.json"
MANIFEST = EXP / "data" / "corpus_manifest.json"
MANIFEST_SHA256 = "4343251285ca9f4688cde77a98c956376e2ca38e169cfd5d33ec5a5d34dd3820"
TREATMENT_PATH = EXP / "data" / "sft_count_walk.jsonl"
TREATMENT_SHA256 = "21e6f5cb705f447f7a4dfc9bff24673f798f48df312b99a6cf686505855ee096"
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
CONSTRUCTION_SEED = 77191
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
ARMS = ("replay_ctl7", "count_walk")
CANDIDATE_ARMS = ("count_walk",)
PARENT_EVAL_LABEL = "zero_root_parent"
GATE_ARMS = ("zero_root_parent", "replay_ctl7", "count_walk")
TREATMENT_KINDS = {"u_count_walk": 160}
TREATMENT_SURFACES = {formalism: 20 for formalism in cw_mod.ENUM_FORMALISMS}
LOCAL_SEED = 88056
SCREEN_SEEDS = (88057, 88058, 88059)
AGGREGATE_SEED = 78163
NAMESPACE_SEED = 55171
TRAINING_SEED = 85

# --- Standalone clean-chain package pins -----------------------------------
LINEAGE_DIR = EXP / "data" / "lineage"
LINEAGE_MANIFEST = LINEAGE_DIR / "lineage_manifest.json"
LINEAGE_MANIFEST_SHA256 = "31174f0c845f081ba226ceb69ce1e5546b6421ff664903c0fc5ae6cfd32af0d7"
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
# as provenance documents (copy path -> sha256).
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
# weights, committed merge receipt for each of replay_ctl7 and
# count_walk) that are filled AFTER the merges publish; a raw hash pin
# would break on the fill, so the file is pinned by a NORMALIZED HASH: the
# deterministic slot patterns below canonicalize exactly the six pin VALUE
# slots (None pre-fill, the quoted 64-hex post-fill) to PIN_PLACEHOLDER, and
# the sha256 of the canonicalized bytes is frozen as
# RUN_BENCHMARK_NORMALIZED_SHA256. Every byte outside the six value slots —
# every guard call site included — is byte-frozen pre- and post-fill.
RUN_BENCHMARK = SCRIPTS / "run_benchmark.py"
PIN_PLACEHOLDER = "__COUNT_WALK_TODO_PIN__"
# (name, multiline regex, required match count). Group 2 is the ONLY mutable
# region of the file; group 1 (and group 3 where present) are kept verbatim
# so indentation, key, and trailing comma stay frozen.
PIN_SLOT_PATTERNS = (
    (
        "replay_ctl7_tree_and_weights_dict_entries",
        r'^(    "replay_ctl7": )(None|"[0-9a-f]{64}")(,)$',
        2,
    ),
    (
        "count_walk_tree_and_weights_dict_entries",
        r'^(    "count_walk": )(None|"[0-9a-f]{64}")(,)$',
        2,
    ),
    (
        "replay_ctl7_merge_receipt_constant",
        r'^(REPLAY_CTL7_MERGE_RECEIPT_SHA256 = )(None|"[0-9a-f]{64}")$',
        1,
    ),
    (
        "count_walk_merge_receipt_constant",
        r'^(COUNT_WALK_MERGE_RECEIPT_SHA256 = )(None|"[0-9a-f]{64}")$',
        1,
    ),
)
# Frozen normalized hash of run_benchmark.py (recompute and re-freeze only
# under a new review); --check fails closed on one byte of drift anywhere
# outside the six canonicalized pin value slots.
RUN_BENCHMARK_NORMALIZED_SHA256 = (
    "bc2b4129f782c168e41f4af00f9f13e5a5273b3aeb5d877b93e8690e45e56616"
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
    surface_audit = manifest.get("surface_freshness_audit", {})
    overlap_audit = manifest.get("row_overlap_audit", {})
    contract = manifest.get("count_walk_contract", {})
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("construction_seed") != CONSTRUCTION_SEED
        or manifest.get("generator") != "scripts/gen_count_walk_curriculum.py"
        or manifest.get("banned_vocabulary_checked") is not True
        or manifest.get("banned_vocabulary", {}).get("case_insensitive") is not True
        or manifest.get("machinery", {}).get("sha256")
        != sha256_file(SCRIPTS / "gen_feedloop_curriculum.py")
        or manifest.get("corpus", {}).get("rows") != 160
        or manifest.get("corpus", {}).get("sha256") != TREATMENT_SHA256
        or manifest.get("corpus", {}).get("kinds") != TREATMENT_KINDS
        or manifest.get("corpus", {}).get("surfaces") != TREATMENT_SURFACES
        or manifest.get("replay", {}).get("sha256") != REPLAY_SHA256
        or replay_rows != REPLAY_ROWS
        or contract.get("one_kind_per_dose_at_full_concentration") is not True
        or contract.get("canonical_order_statement")
        != cw_mod.CANONICAL_ORDER_STATEMENT
        or contract.get("canonical_order_statement_identical_in_every_row")
        is not True
        or contract.get("action_list_rendered_in_every_prompt") is not True
        or contract.get("canonical_order_rule_text_byte_identical_to_reference_cell")
        is not True
        or contract.get("range_statement_rendered_in_every_prompt") is not True
        or contract.get("range_statement_verified_against_enumeration_exactly")
        is not True
        or contract.get("k_cycle") != list(cw_mod.K_CYCLE)
        or contract.get("answer_format") != "STEP <k>: <corrected step>"
        or contract.get("think_budget", {}).get("token_cap")
        != cw_mod.THINK_TOKEN_CAP
        or contract.get("think_budget", {}).get("char_cap")
        != cw_mod.THINK_CHAR_CAP
        or contract.get("think_budget", {}).get("line_count")
        != cw_mod.THINK_LINE_COUNT
        or contract.get("think_budget", {}).get("line_patterns")
        != list(cw_mod.THINK_LINE_PATTERNS)
        or contract.get("think_budget", {}).get("constant_in_k") is not True
        or not isinstance(
            contract.get("think_budget", {}).get("max_estimated_think_tokens"),
            int,
        )
        or contract["think_budget"]["max_estimated_think_tokens"]
        > cw_mod.THINK_TOKEN_CAP
        or surface_audit.get("tokens") != []
        or not surface_audit.get("inherited_tokens", {}).get("tokens")
        or not overlap_audit.get("sources")
        or any(
            source.get("overlap") != 0
            for source in overlap_audit.get("sources", {}).values()
        )
    ):
        raise ValueError("corpus manifest violates the frozen fresh-corpus contract")
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
        or manifest.get("stages", [{}])[-1].get("name") != "count_walk"
        or manifest.get("stages", [{}])[-1].get("seed") != TRAINING_SEED
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
        '"seed": 85,',
        "LORA_RANK = 32",
        "LORA_ALPHA = 64",
        'MODEL_PATH_WEIGHTS_SHA256 = "6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932"',
        'MODEL_PATH_TREE_SHA256 = "414f582950bf60fed2fe462cd141ab98d0f772087b4f9c6bc5aa12f03f379e7d"',
        '"fresh_adapter": True,',
        "--model-path",
        "PASS_CONTROL_TRAINING",
        'ARM_PREREQUISITES = {\n    "replay_ctl7": (),\n    "count_walk": ("replay_ctl7",),\n}',
        "PUBLISHED_ARM_HASHES",
    )
    required_merge = (
        'MERGER_SHA256 = "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672"',
        'payload.get("applied_lora_modules") != 128',
        "--base-model",
        'BASE_COMPOSITE_RECEIPT_SHA256 = "f8981f4638d901471eb41aff0ffd0bfac88aebd6e3e4d4db1e1c733be16709c0"',
        "PASS_CONTROL_MERGE",
        "LOCAL_SEED = 88056",
        "LOCAL_ROWS = 352",
    )
    required_harness = (
        "def require_pushed_checkpoint",
        '"train-control"',
        '"train-candidate"',
        '"merge-arms"',
        '"local"',
        '"benchmark"',
        "LOCAL_SEED = 88056",
        "SCREEN_SEEDS = (88057, 88058, 88059)",
        "AGGREGATE_SEED = 78163",
        "rebuild_clean_chain.py",
        "PASS_CONTROL_TRAINING",
        "PASS_CONTROL_MERGE",
        "PASS_LOCAL_EVENT",
        "PASS_BENCHMARK_EVENT",
    )
    required_gate = (
        "SEED = 88056",
        "SCREEN_SEEDS = (88057, 88058, 88059)",
        "AGGREGATE_SEED = 78163",
        "RETENTION_CORRECT_BAND = 5",
        "RETENTION_CAP_BAND = 3",
        "RETENTION_PARSED_BAND = 3",
        "def normalize_answer",
        "def pooled_sd",
        "def evaluate_promotion",
        "def pooled_band_checks",
        "def fidelity_summary",
        "def expression_cost_summary",
        "expression_cost_per_arm",
        '"pooled_k3"',
        "no_absolute_per_kind_floors",
        "single_kind_gate",
        "mechanism_reading",
        "reported_not_gated",
    )
    required_bench = (
        'FROZEN_NAME = "pilot"',
        'FROZEN_TIER = "medium"',
        "FROZEN_THINK_BUDGET = 1024",
        "FROZEN_SEED = 78163",
        'MENDERS_FAMILY = "menders"',
        "def require_unconsumed_ledger",
        "def require_todo_pins_filled",
        "def require_zero_root_parent_provenance",
        "def _valid_score",
        "def pilot_gate",
        "def goal_gate_reading",
        "def menders_reading",
        "def fidelity_precondition",
        '"TURN_BUDGET_SCOPED"',
        '"FAILED_ON_ITS_OWN_TERMS"',
        '"MECHANISM_ANSWER"',
        "EPISODE_SUCCESS_SIMULATION",
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
    banned = tuple(cw_mod.BANNED_PROMPT_TOKENS)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "count_walk_protocol_design",
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
        },
        "mechanism_statement": (
            "lifecycle 26 (the enumerative-repair cell) proved the "
            "enumeration DISCIPLINE installs: 9/40 canonical-next on the "
            "axis holdout versus BOTH controls at exactly 0/40 — the "
            "program's starkest mechanism contrast. Its committed "
            "truncation forensics showed the failure was EXPRESSION COST, "
            "not discipline: 20 of 21 unparseable gate rows were "
            "1,024-token cap truncations caught mid-CORRECT walk, a token "
            "cost that grows with the tried-list depth k. This cell "
            "changes ONLY the expression pedagogy: COUNT, DON'T WALK — the "
            "tried list has k entries in canonical order, so the target is "
            "entry k+1 of the frozen order; the prompt renders the "
            "per-step candidate ranges and the think target is a "
            "fixed-shape five-line index computation (count -> k+1 -> "
            "range lookup -> slot offset -> emit), constant token cost in "
            "k, under a frozen think length budget verified with the real "
            "tokenizer. Same machinery, same invariants, same gate, same "
            "frozen menders consequences — one designed delta"
        ),
        "clean_chain": {
            "framing": (
                "the single-kind count-don't-walk dose trained onto the "
                "CLEAN lineage: every training stage of the final model — "
                "six zero-root stages + this cell's dose as stage 7 — is "
                "documented, receipted, and contamination-free; NO "
                "undocumented root exists anywhere in the lineage"
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
            "construction": CONSTRUCTION_SEED,
            "stream_slot_matching_namespace": NAMESPACE_SEED,
            "training": TRAINING_SEED,
            "local_gate": LOCAL_SEED,
            "retention_screens": list(SCREEN_SEEDS),
            "seed_freshness": {
                "pinned": [
                    CONSTRUCTION_SEED,
                    NAMESPACE_SEED,
                    TRAINING_SEED,
                    LOCAL_SEED,
                    *SCREEN_SEEDS,
                    AGGREGATE_SEED,
                ],
                "substitution_required": False,
                "known_taken_nearby": {
                    "88043": (
                        "taken as the retention seed of "
                        "qwen35_4b_counterfactual_plan_reflection_transfer "
                        "(documented since lifecycle 23)"
                    ),
                    "88047": (
                        "taken as the reflection seed of "
                        "qwen35_4b_counterfactual_plan_reflection_transfer"
                    ),
                    "88049": (
                        "taken as the action seed of "
                        "qwen35_4b_counterfactual_plan_reflection_transfer"
                    ),
                },
                "verified": (
                    "77191/55171/88056/88057/88058/88059/78163 verified "
                    "grep-fresh in seed contexts across experiments/, "
                    "knowledge/, research_programs/, scripts/, configs/, "
                    "and docs/ at design time (every raw numeric hit is a "
                    "sha256 or float substring, never a seed); the training "
                    "seed required ONE next-free substitution: 84 (the next "
                    "integer after the reference cell's 83) is TAKEN as a "
                    "task seed of qwen35_4b_hypothesize_verify_wall (eval "
                    "string d3) and appears in "
                    "qwen35_4b_meta_induction per-row data fields, so the "
                    "frozen training seed is 85 — zero seed-context hits "
                    "anywhere in the repo (paired-trial training-seed "
                    "sequence 67, 71, 73, 79, 83, 85)"
                ),
                "rule_if_collision": "next free integer, recorded here",
            },
            "aggregate_sealed": AGGREGATE_SEED,
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
                    "policy": "fresh_designed_single_kind_count_walk_corpus",
                    "construction_seed": CONSTRUCTION_SEED,
                    "generator": "scripts/gen_count_walk_curriculum.py",
                    "builder": "scripts/build_corpus.py",
                    "machinery": (
                        "byte-identical copy of the menders dose-scale "
                        "cell's reviewed feedloop generator (machinery "
                        "imported, never forked); the LESSON changes "
                        "completely — systematic enumeration, not "
                        "eliminative inference"
                    ),
                    "per_row_verification": manifest["count_walk_contract"][
                        "per_row_verification"
                    ],
                    "canonical_order_statement": (
                        cw_mod.CANONICAL_ORDER_STATEMENT
                    ),
                    "k_cycle": list(cw_mod.K_CYCLE),
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
                "base_inventory": "gen_feedloop_curriculum.BANNED_PROMPT_TOKENS",
                "feedloop_inventory_tokens": len(feedloop.BANNED_PROMPT_TOKENS),
                "case_insensitive": True,
                "checked_by_generator": True,
            },
        },
        "arms": {
            "trained_control": "replay_ctl7",
            "trained_candidate": "count_walk",
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
                "replay_ctl7": {"replay_rows": 240},
                "count_walk": {"treatment_rows": 160, "replay_filler_rows": 80},
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
                    "kinds": {"u_count_walk": 40},
                    "per_formalism": 5,
                    "generator": "scripts/gen_count_walk_curriculum.py",
                    "fresh_instances_from_this_cells_generator": True,
                    "same_invariants_as_the_treatment": True,
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
                "ties_fail": True,
                "single_kind_no_per_kind_split": True,
                "per_surface_reported_not_gated": True,
                "retention_pooled_correct_band": 5,
                "retention_pooled_cap_contact_band": 3,
                "retention_pooled_parsed_band": 3,
                "bands_on_pooled_means_not_per_screen": True,
                "bands_evaluated_on_pooled_sums_times_screens": True,
                "no_absolute_per_kind_floors": True,
            },
            "mechanism_readings": {
                "episode_success_simulation": (
                    "for each holdout row's underlying broken machine the "
                    "generator computes ANALYTICALLY (no model) the number "
                    "of turns a PERFECT canonical enumerator needs — the "
                    "canonical index of the unique both-trials fix plus one "
                    "— and the local design receipt records the "
                    "distribution; never gated"
                ),
                "enumeration_fidelity": (
                    "at eval time every axis row records three booleans "
                    "about the model's proposed candidate — legal, untried, "
                    "canonical-next — a mechanism decomposition beyond raw "
                    "correctness, summarized per arm; never gated"
                ),
                "expression_cost": (
                    "NEW, the reading this lineage owes after the "
                    "reference cell's truncation forensics: per-arm "
                    "think-token-length distribution over the 40 axis rows "
                    "(min/median/mean/max of n_thinking_tokens) plus the "
                    "truncation count, summarized by check_local.py; a "
                    "count-don't-walk install should show short, "
                    "k-independent thinking and zero truncations; never "
                    "gated"
                ),
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
                "replay_ctl7",
                "count_walk",
            ],
            "pilot_gate": (
                "candidate aggregate strictly > base AND > replay_ctl7 AND > "
                "zero_root_parent"
            ),
            "goal_gate": (
                "all ten public families strictly > base; recorded from the "
                "same event either way, never part of the pilot pass"
            ),
            "menders_reading": (
                "candidate vs base and vs parent on menders specifically — "
                "the frozen question: does taught enumeration convert to "
                "the family with live rerun feedback? Recorded either way, "
                "never part of the pilot gate"
            ),
            "episode_success_simulation_quoted": {
                "holdout_from_scratch": {
                    "mean": 27.1,
                    "median": 20.5,
                    "min": 4,
                    "max": 78,
                    "share_needing_more_than_10_turns": 0.8,
                },
                "treatment_from_scratch": {
                    "mean": 32.64375,
                    "median": 22.0,
                    "min": 2,
                    "max": 125,
                    "share_needing_more_than_10_turns": 0.86875,
                },
                "consequence_stated_plainly": (
                    "the family's episode budget is publicly known only as "
                    "'bounded'; if it is materially shorter than these "
                    "needs, a perfectly-installed enumerator converts few "
                    "or no episodes — the zero draw is ambiguous WITHOUT "
                    "the fidelity precondition below"
                ),
            },
            "menders_zero_scoping": {
                "fidelity_precondition": (
                    "F = the candidate's canonical-next rate on the 40-row "
                    "axis holdout (from the local promotion receipt's "
                    "mechanism_reading); the precondition HOLDS iff the "
                    "candidate promoted locally AND F >= 0.50 AND F "
                    "strictly exceeds both controls' canonical-next rates "
                    "(integer-exact comparisons)"
                ),
                "frozen_consequence_order": [
                    "MECHANISM_ANSWER",
                    "TURN_BUDGET_SCOPED",
                    "FAILED_ON_ITS_OWN_TERMS",
                ],
                "rules": {
                    "1_mechanism_answer_takes_precedence": (
                        "ANY candidate menders > 0 where the controls sit "
                        "at 0 is the mechanism answer"
                    ),
                    "2_turn_budget_scoped": (
                        "a menders reading of 0 WITH the fidelity "
                        "precondition met is TURN_BUDGET_SCOPED — "
                        "enumeration installed with high fidelity but did "
                        "not convert within the family's episode budget; "
                        "the protocol-install mechanism is NOT refuted; "
                        "what closes is the pure-enumeration route at the "
                        "family's actual budget"
                    ),
                    "3_failed_on_its_own_terms": (
                        "a menders 0 WITHOUT that precondition reads as "
                        "the install/conversion failing on its own terms"
                    ),
                },
                "no_third_state_for_the_zero_draw": True,
                "added_pre_freeze_by_review_amendment": True,
            },
            "power_statement": (
                "menders has been 0-margin on most draws across the "
                "program's sealed events, and the analytic "
                "perfect-enumerator simulation says the zero draw is "
                "ambiguous: holdout episodes need mean 27.1 turns from "
                "scratch (median 20.5, max 78; 80.0% need more than 10; "
                "treatment corpus mean 32.6, median 22, max 125; 86.9% > "
                "10) against a family budget publicly known only as "
                "'bounded'. Frozen ordered consequences, positive first: "
                "(1) any candidate menders > 0 where the controls sit at 0 "
                "is the mechanism answer; (2) a menders 0 WITH the "
                "fidelity precondition met (promoted AND holdout "
                "canonical-next rate >= 0.50 AND strictly above both "
                "controls) is TURN_BUDGET_SCOPED — the protocol-install "
                "mechanism is NOT refuted, only the pure-enumeration route "
                "at the family's actual budget closes; (3) a menders 0 "
                "WITHOUT it fails on its own terms. No third state for the "
                "zero draw; a 10/10 goal gate is a menders draw and feeds "
                "a fresh confirmation cell before any claim"
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
            "feedloop_machinery_sha256": sha256_file(
                EXP / "scripts" / "gen_feedloop_curriculum.py"
            ),
            "count_walk_curriculum_generator_sha256": sha256_file(
                EXP / "scripts" / "gen_count_walk_curriculum.py"
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
