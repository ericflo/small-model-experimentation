#!/usr/bin/env python3
"""Freeze the retention-only, three-input-file local gate and its model inputs.

There is DELIBERATELY no axis instrument in this cell: the stage-9 treatment
installs a NEW single kind (``u_state_track``), but the local gate holds out
NO axis instrument — the transfer/aggregate question is priced only by the
sealed benchmark event on held-out families, per the install-by-transfer
doctrine. The local gate is a pure RETENTION NON-DRIFT screen between
exactly two arms — the ``count_walk`` parent composite and the
``state_track`` candidate. The frozen gate comprises THREE model-facing
input files per arm:

- THREE RETENTION SCREENS (pooled_k3 protocol): 104 rows each, 8 per each
  of the 13 original skills, from the canonical gen_curriculum.py at seeds
  88063/88064/88065; ids ``ret<seed>_*``. No substitution was required for
  the gate seeds: every pinned seed was verified grep-fresh in seed
  contexts (known-taken: everything <= 88062 including the replay-compound
  reference cell's 88060-88062; the frozen sequence starts at the next free
  integer 88063).

Grading applies the frozen answer normalization documented in this receipt
(``answer_normalization``) identically to both arms and every input file.
Freshness is enforced fail-closed: the three input files are duplicate-free
internally and pairwise, and carry zero canonical-user-message overlap with
every frozen in-cell corpus (the 160-row state_track curriculum this cell
TRAINS on, the stage-8 replay pool, the stage-7 production inputs, and the
six zero-root lineage datasets), with the three reference cells' ELEVEN
frozen gate files (seeds 88052-88062, sha-pinned IN-CELL copies under
``data/predecessor_gates/``; the committed sibling originals are
verification aids — byte-identical when present, skipped when absent), and
with regenerated prior local seeds 88000-88062.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_curriculum as curriculum  # noqa: E402
from check_local import (  # noqa: E402
    AGGREGATE_SEED,
    ANSWER_NORMALIZATION,
    ARMS,
    CANDIDATES,
    RETENTION_CAP_BAND,
    RETENTION_CORRECT_BAND,
    RETENTION_PARSED_BAND,
    SCREEN_SEEDS,
)


RETENTION_MIX = ",".join(f"{name}=8" for name in curriculum.SKILLS)
RETENTION_ROWS = 104
RETENTION_PER_SKILL = 8
ROWS_PER_ARM = RETENTION_ROWS * len(SCREEN_SEEDS)
INPUT_SEEDS = tuple(SCREEN_SEEDS)
SOURCES = {seed: EXP / "data" / f"local_tasks_seed{seed}.jsonl" for seed in INPUT_SEEDS}
RUNNER_INPUTS = {
    seed: EXP / "data" / f"local_input_seed{seed}.jsonl" for seed in INPUT_SEEDS
}
RECEIPT = EXP / "data" / "local_design_receipt.json"
# Frozen in-cell corpora at design freeze (fail-closed hash pins): this
# cell's OWN stage-9 state_track training curriculum, the stage-8 replay
# pool, the stage-7 production inputs, and the six ordered zero-root lineage
# datasets.
FROZEN_SOURCES = (
    (EXP / "data" / "sft_state_track.jsonl",
     "66a8d5bec184a8a9cba20c2ea088e0216ac4cdbd0820541ee310170eb386e3ab", 160),
    (EXP / "data" / "sft_blend.jsonl",
     "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2", 2240),
    (EXP / "data" / "sft_count_walk.jsonl",
     "21e6f5cb705f447f7a4dfc9bff24673f798f48df312b99a6cf686505855ee096", 160),
    (EXP / "data" / "count_walk.jsonl",
     "71291542c3c901caccf9586543efb02da319b371244728ecfd1a0fc7cb92ed26", 1520),
    (EXP / "data" / "replay_ctl7.jsonl",
     "94e8259ec03800d0a4dcbf8075252c5180a668e2da74569fcf62497cf0f9de5a", 1520),
    (EXP / "data" / "lineage" / "stage01_replay_refresh.jsonl",
     "5d5d7c4b8a4b0a4f270fe8b2ecaebe356c771948d71b0f7bbeead6bfc04308b6", 1520),
    (EXP / "data" / "lineage" / "stage02_designed160.jsonl",
     "5159cf41b6474bdc8640cdb2a4a168587b59232ca8171c7b7057fc6bfe1b40c8", 1520),
    (EXP / "data" / "lineage" / "stage03_close_xi__targeted_standard.jsonl",
     "12fc613bb31a46bcea9acd49b26467656704aa3b3418dab8d920adf057d14f00", 320),
    (EXP / "data" / "lineage" / "stage04_replay_after_close.jsonl",
     "541805df2d817707c1e76213e50c8f08fd9caff10d0a3887e1196424b6820be6", 320),
    (EXP / "data" / "lineage" / "stage05_designed_fresh.jsonl",
     "6d4dc303bc159c19a1ffd0c60ca7d08ea64b02909366701b345d888482d67f3f", 1520),
    (EXP / "data" / "lineage" / "stage06_hygiene_explore.jsonl",
     "82aa1a78c0a429a48c3db6b94ac84397cea001b041477e7b137b38c21354112f", 1520),
)
# The three predecessor cells' eleven frozen gate files: the enumerative-
# repair cell's seeds 88052-88055, the count-don't-walk cell's seeds
# 88056-88059, and the replay-compound cell's seeds 88060-88062. The IN-CELL
# copies under data/predecessor_gates/ are the sha-pinned fail-closed inputs
# of the overlap audit (the audit runs identically without any sibling); the
# committed sibling originals are VERIFICATION AIDS only — byte-identical
# when present (divergence fails loudly), skipped when absent — never a
# reproduction dependency.
PREDECESSOR_GATE_COPY_DIR = EXP / "data" / "predecessor_gates"
PREDECESSOR_GATE_SIBLING_POLICY = (
    "verification aid: byte-identical when present, absent -> in-cell "
    "sha-pinned copy authoritative"
)
_PREDECESSOR_GATE_CELLS = (
    ("qwen35_4b_enumerative_repair_protocol", (
        (88052, "7390c384f479f90918c3f4236be5415fb9a44ccd129fb95f5086f369cd88e549", 40),
        (88053, "0b0a904af5a743f008bef6f14fca43c49a30ea86599847b7131cdcea22e1e7e6", 104),
        (88054, "bb177ae8948c52c72103a9c050cfd2df248aaf7f43d9f4d051296dfe26f6f9b2", 104),
        (88055, "bb83c2bff9b23f45c2a311a7895fbedefc4a1c6f90d70c83c382204ea625f699", 104),
    )),
    ("qwen35_4b_count_dont_walk_enumeration", (
        (88056, "66d6a16be8752784bd85d9589342975204fef7303be5cf0c150262a791fd9135", 40),
        (88057, "56a71bc1223b9a87266de53f33881a5225dbb86f72ee669268398bcd06dda63d", 104),
        (88058, "8de8e65171418f1662bbf8b44cdf8d2de45f1801c14a773ddc7e328d0007fc0b", 104),
        (88059, "26e524ec480e55107414a54f3a921c3bd4e08fe0fd949944705ade025aecbafc", 104),
    )),
    ("qwen35_4b_count_walk_replay_compound", (
        (88060, "836c971bbd019470c69ff66cc463dea5714864422922980a9af4c08839b5e461", 104),
        (88061, "4149e399df8773970f4137a215cee9c9c6d175f79e29aa2b952bf5193a0c1d75", 104),
        (88062, "7a143b41706e519917a54c09b5594f09b273b6da50ccc4857c5eefc8a6ebe187", 104),
    )),
)
# (sibling original, in-cell copy, sha256, rows) per predecessor gate file.
PREDECESSOR_GATES = tuple(
    (
        ROOT / "experiments" / cell / "data" / f"local_tasks_seed{seed}.jsonl",
        PREDECESSOR_GATE_COPY_DIR / f"local_tasks_seed{seed}.jsonl",
        sha,
        rows,
    )
    for cell, entries in _PREDECESSOR_GATE_CELLS
    for seed, sha, rows in entries
)
PRIOR_LOCAL_SEEDS = tuple(range(88000, 88063))
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PARENT_LABEL = "count_walk"
PARENT_MERGE_RECEIPT = (
    ROOT
    / "experiments"
    / "qwen35_4b_count_dont_walk_enumeration"
    / "runs"
    / "merges"
    / "count_walk.json"
)
PARENT_PROVENANCE_COPY = EXP / "data" / "provenance" / "count_walk_merge.json"
PARENT_MERGED = (
    ROOT
    / "large_artifacts"
    / "qwen35_4b_count_dont_walk_enumeration"
    / "merged"
    / "count_walk"
)
EXPECTED_PARENT_MERGE_RECEIPT_SHA256 = (
    "840edca0638b9e291bb34fde28b4b530df8743faf9b7b18b7f2358ce55ec4c36"
)
EXPECTED_PARENT_TREE_SHA256 = (
    "d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1"
)
EXPECTED_PARENT_WEIGHTS_SHA256 = (
    "ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3"
)
EXPECTED_PARENT_FILES = [
    {
        "name": "chat_template.jinja",
        "sha256": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        "size": 7756,
    },
    {
        "name": "config.json",
        "sha256": "a1c80f0efa6f83f631eaa9c25ffa166e3b1f9db395cc3b14374dfc0962261f60",
        "size": 2829,
    },
    {
        "name": "generation_config.json",
        "sha256": "0c46d8aa4f0ae5e611c961f70b87c83fb696043c1e319337708e96f882180de1",
        "size": 116,
    },
    {
        "name": "merge_receipt.json",
        "sha256": "3c432f110fe96a508d6a75ab34e4a649671a3d7b2d942f3346cab609bef437d7",
        "size": 954,
    },
    {
        "name": "model.safetensors",
        "sha256": "ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3",
        "size": 9078620536,
    },
    {
        "name": "tokenizer.json",
        "sha256": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "size": 19989325,
    },
    {
        "name": "tokenizer_config.json",
        "sha256": "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
        "size": 1123,
    },
]
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
# Files pinned by hash into the frozen design receipt. eval_local_vllm.py,
# run_benchmark.py, and train_trial.py are deliberately NOT pinned here:
# they carry orchestrator-filled TODO-PIN constants that change after this
# receipt freezes; each is authenticated by hash inside its own run receipt,
# and all three are additionally frozen by check_design.py's NORMALIZED
# hashes (their pin value slots canonicalized to a fixed placeholder), which
# ARE pinned here via check_design.py's raw hash.
CODE_FILES = {
    "generator": Path(__file__),
    "curriculum": EXP / "scripts" / "gen_curriculum.py",
    "gate": EXP / "scripts" / "check_local.py",
    "harness": EXP / "scripts" / "run.py",
    "runner": EXP / "src" / "vllm_runner.py",
    # merge_trained_arm.py carries no orchestrator-filled constants, so it is
    # pinned here and demands this exact pin back (code_sha256.merge) before
    # any composite is produced.
    "merge": EXP / "scripts" / "merge_trained_arm.py",
    # the vendored byte-identical merger copy (part of the standalone
    # clean-chain package; merge_trained_arm.py points here too).
    "external_merger": EXP / "scripts" / "merge_adapter.py",
    # the design checker carrying run_benchmark.py's normalized-hash pin.
    "design_checker": EXP / "scripts" / "check_design.py",
    # the standalone lineage rebuilder (no fill slots; byte-stable).
    "lineage": EXP / "scripts" / "rebuild_lineage.py",
}
CODE_PINS_DEFERRED = [
    "scripts/eval_local_vllm.py",
    "scripts/run_benchmark.py",
    "scripts/train_trial.py",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode()


def message_bytes(row: dict) -> bytes:
    return json.dumps(
        row["messages"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def load_frozen(path: Path, expected_sha256: str | None, expected_rows: int) -> list[dict]:
    if expected_sha256 is None:
        raise ValueError(f"frozen source pin is unfilled (TODO-PIN): {path}")
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"frozen source changed: {path}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if len(rows) != expected_rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"unexpected frozen source rows: {path}")
    return rows


def load_predecessor_gate(
    sibling_path: Path, copy_path: Path, expected_sha256: str, expected_rows: int
) -> list[dict]:
    """Load one predecessor gate file from its IN-CELL sha-pinned copy.

    The in-cell copy is hard-required and sha-pinned (the overlap audit runs
    identically without any sibling checkout). The committed sibling
    original is a verification aid: when present it must be byte-identical
    to the in-cell copy (divergence fails loudly as tamper evidence); when
    absent it is skipped — the in-cell pin is authoritative.
    """
    rows = load_frozen(copy_path, expected_sha256, expected_rows)
    if sibling_path.is_file() and (
        sibling_path.read_bytes() != copy_path.read_bytes()
    ):
        raise ValueError(
            "committed predecessor-gate sibling original diverged from the "
            f"in-cell copy: {sibling_path}"
        )
    return rows


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


def build_screen(seed: int) -> tuple[list[dict], list[dict]]:
    retention_rows = curriculum.generate_curriculum(RETENTION_MIX, seed)
    for row in retention_rows:
        row["task_id"] = f"ret{seed}_{row['task_id']}"
    retention_summary = curriculum.validate_generated(retention_rows)
    expected_retention_kinds = {
        f"u_{name}": RETENTION_PER_SKILL for name in curriculum.SKILLS
    }
    if (
        retention_summary["rows"] != RETENTION_ROWS
        or retention_summary["kinds"] != expected_retention_kinds
    ):
        raise ValueError(
            f"retention screen {seed} no longer has eight rows per registered skill"
        )
    runner_rows = [
        {
            "id": row["task_id"],
            "messages": row["messages"],
            "meta": {
                "kind": row["kind"],
                "surface": row["surface"],
                "seed": seed,
                "instrument": "retention",
            },
        }
        for row in retention_rows
    ]
    if any(set(row) != {"id", "messages", "meta"} for row in runner_rows):
        raise ValueError("local runner input schema leaked hidden fields")
    return retention_rows, runner_rows


def overlap_receipt(source_rows_by_seed: dict[int, list[dict]]) -> dict:
    messages_by_seed: dict[int, set[bytes]] = {}
    for seed, rows in source_rows_by_seed.items():
        messages = {message_bytes(row) for row in rows}
        if len(messages) != RETENTION_ROWS:
            raise ValueError(f"input file {seed} collides on canonical user messages")
        messages_by_seed[seed] = messages
    local_messages: set[bytes] = set()
    for seed in INPUT_SEEDS:
        local_messages |= messages_by_seed[seed]
    if len(local_messages) != ROWS_PER_ARM:
        raise ValueError("gate instruments collide on canonical user messages")
    cross_screen: dict[str, int] = {}
    for index, first in enumerate(INPUT_SEEDS):
        for second in INPUT_SEEDS[index + 1 :]:
            overlap = len(messages_by_seed[first] & messages_by_seed[second])
            if overlap:
                raise ValueError(
                    f"input files {first} and {second} share canonical user messages"
                )
            cross_screen[f"{first}x{second}"] = overlap
    frozen_sources: dict[str, dict] = {}
    for path, expected_sha256, expected_rows in FROZEN_SOURCES:
        messages = {
            message_bytes(row)
            for row in load_frozen(path, expected_sha256, expected_rows)
            if row.get("messages")
        }
        overlap = len(local_messages & messages)
        if overlap:
            raise ValueError(f"fresh local prompts overlap frozen corpus: {path}")
        frozen_sources[path.relative_to(EXP).as_posix()] = {
            "sha256": expected_sha256,
            "messages_compared": len(messages),
            "overlap": overlap,
        }
    predecessor_gates: dict[str, dict] = {}
    for sibling_path, copy_path, gate_sha256, gate_rows in PREDECESSOR_GATES:
        predecessor_messages = {
            message_bytes(row)
            for row in load_predecessor_gate(
                sibling_path, copy_path, gate_sha256, gate_rows
            )
            if row.get("messages")
        }
        predecessor_overlap = len(local_messages & predecessor_messages)
        if predecessor_overlap:
            raise ValueError(
                f"fresh local prompts overlap a predecessor's frozen gate: {copy_path}"
            )
        predecessor_gates[sibling_path.relative_to(ROOT).as_posix()] = {
            "sha256": gate_sha256,
            "in_cell_copy": copy_path.relative_to(EXP).as_posix(),
            "messages_compared": len(predecessor_messages),
            "overlap": predecessor_overlap,
            "sibling_original": PREDECESSOR_GATE_SIBLING_POLICY,
        }
    prior_messages = {
        message_bytes(row)
        for seed in PRIOR_LOCAL_SEEDS
        for row in curriculum.generate_curriculum(curriculum.SMOKE_MIX, seed)
    }
    prior_local_overlap = len(local_messages & prior_messages)
    if prior_local_overlap:
        raise ValueError("fresh local prompts overlap prior local seeds")
    return {
        "message_sha256s": sorted(sha256_bytes(value) for value in local_messages),
        "unique_local_messages": len(local_messages),
        "unique_local_messages_per_input": {
            str(seed): len(messages_by_seed[seed]) for seed in INPUT_SEEDS
        },
        "cross_screen_overlap": cross_screen,
        "frozen_sources": frozen_sources,
        "predecessor_gates": predecessor_gates,
        "prior_local_seeds_compared": list(PRIOR_LOCAL_SEEDS),
        "prior_local_mix": curriculum.SMOKE_MIX,
        "prior_local_messages_compared": len(prior_messages),
        "prior_local_overlap": prior_local_overlap,
        "training_pool_note": (
            "the stage-9 training curriculum (data/sft_state_track.jsonl) is "
            "checked above under frozen_sources; the gate instruments carry "
            "zero canonical-user-message overlap with everything this cell "
            "trains on"
        ),
    }


def build_outputs(*, authenticate_parent: bool = True) -> dict[Path, bytes]:
    for path in CODE_FILES.values():
        if not path.is_file():
            raise ValueError(f"required local-design input is absent: {path}")
    # The IN-CELL sha-pinned provenance copy is the hard fail-closed gate;
    # the committed lifecycle-27 sibling original is a verification aid —
    # byte-identical when present (divergence fails loudly), skipped when
    # absent (in-cell pin authoritative).
    if (
        not PARENT_PROVENANCE_COPY.is_file()
        or sha256_file(PARENT_PROVENANCE_COPY)
        != EXPECTED_PARENT_MERGE_RECEIPT_SHA256
    ):
        raise ValueError(
            f"in-cell parent provenance copy is absent or changed: "
            f"{PARENT_PROVENANCE_COPY}"
        )
    if PARENT_MERGE_RECEIPT.is_file() and (
        PARENT_MERGE_RECEIPT.read_bytes() != PARENT_PROVENANCE_COPY.read_bytes()
    ):
        raise ValueError(
            "committed lifecycle-27 sibling merge receipt diverged from the "
            f"in-cell provenance pin: {PARENT_MERGE_RECEIPT}"
        )
    parent_receipt = json.loads(PARENT_PROVENANCE_COPY.read_text(encoding="utf-8"))
    if (
        parent_receipt.get("name") != "count_walk"
        or parent_receipt.get("experiment_id") != "qwen35_4b_count_dont_walk_enumeration"
        or parent_receipt.get("model_id") != MODEL_ID
        or parent_receipt.get("model_revision") != MODEL_REVISION
        or Path(parent_receipt.get("merged", "")).resolve() != PARENT_MERGED.resolve()
        or parent_receipt.get("output_tree_sha256") != EXPECTED_PARENT_TREE_SHA256
        or parent_receipt.get("output_files") != EXPECTED_PARENT_FILES
        or {
            row.get("name"): row.get("sha256")
            for row in parent_receipt.get("weight_files", [])
        }
        != {"model.safetensors": EXPECTED_PARENT_WEIGHTS_SHA256}
    ):
        raise ValueError("published count_walk parent merge receipt violates pins")
    if authenticate_parent:
        parent_files = merged_tree_manifest(PARENT_MERGED)
        if (
            parent_files != EXPECTED_PARENT_FILES
            or tree_manifest_sha256(parent_files) != EXPECTED_PARENT_TREE_SHA256
        ):
            raise ValueError("published count_walk parent composite tree changed")
    source_rows_by_seed: dict[int, list[dict]] = {}
    runner_rows_by_seed: dict[int, list[dict]] = {}
    for seed in INPUT_SEEDS:
        source_rows_by_seed[seed], runner_rows_by_seed[seed] = build_screen(seed)
    sources = {seed: jsonl_bytes(source_rows_by_seed[seed]) for seed in INPUT_SEEDS}
    runner_inputs = {
        seed: jsonl_bytes(runner_rows_by_seed[seed]) for seed in INPUT_SEEDS
    }
    receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "state_track_local_gate_design",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "loaded": False,
            "calls": 0,
        },
        "screen_seeds": list(SCREEN_SEEDS),
        "seed_freshness": {
            "input_seeds": list(INPUT_SEEDS),
            "substitution_required": False,
            "known_taken_nearby": {
                "88052_to_88059": (
                    "taken as the gate and retention-screen seeds of the "
                    "enumerative-repair and count-don't-walk reference cells"
                ),
                "88060_to_88062": (
                    "taken as the retention-screen seeds of the replay-"
                    "compound reference cell "
                    "(qwen35_4b_count_walk_replay_compound); everything "
                    "<= 88062 is known-taken"
                ),
            },
            "verified": (
                "88063/88064/88065 verified grep-fresh in seed contexts "
                "across experiments/, knowledge/, research_programs/, "
                "scripts/, configs/, and docs/ at design time (every raw "
                "numeric hit is a sha256 or float substring, never a "
                "seed); the frozen gate sequence starts at the next free "
                "integer after the known-taken block and needed no "
                "substitution"
            ),
            "rule_if_collision": "next free integer, recorded here",
        },
        "aggregate_seed": AGGREGATE_SEED,
        "rows_per_arm": ROWS_PER_ARM,
        "instruments": {
            "retention": {
                "generator": "scripts/gen_curriculum.py",
                "mix": RETENTION_MIX,
                "rows_per_screen": RETENTION_ROWS,
                "per_kind": RETENTION_PER_SKILL,
                "screens": len(SCREEN_SEEDS),
                "id_prefixes": [f"ret{seed}_" for seed in SCREEN_SEEDS],
                "adjudication": "pooled_k3",
            },
        },
        "no_axis_instrument": (
            "the stage-9 treatment installs a NEW single kind (u_state_track), "
            "but the local gate deliberately holds out no axis instrument; the "
            "transfer/aggregate question is priced only by the sealed benchmark "
            "event on held-out families, so the local gate is a pure retention "
            "non-drift screen"
        ),
        "kinds": {
            str(seed): dict(
                sorted(
                    Counter(row["kind"] for row in source_rows_by_seed[seed]).items()
                )
            )
            for seed in INPUT_SEEDS
        },
        "sources": {
            str(seed): {
                "path": SOURCES[seed].relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(sources[seed]),
                "contains_executable_truth": True,
            }
            for seed in INPUT_SEEDS
        },
        "runner_inputs": {
            str(seed): {
                "path": RUNNER_INPUTS[seed].relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(runner_inputs[seed]),
                "schema": ["id", "messages", "meta"],
                "contains_answer": False,
                "contains_oracle": False,
            }
            for seed in INPUT_SEEDS
        },
        "answer_normalization": ANSWER_NORMALIZATION,
        "freshness": overlap_receipt(source_rows_by_seed),
        "backend": {
            "name": "vllm_merged_composite",
            "thinking": "natural",
            "greedy": True,
            "samples_per_task": 1,
            "max_tokens": 1024,
            "max_model_len": 4096,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 8192,
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
            "same_runner_and_geometry_for_every_arm": True,
            "runtime_lora_forbidden": True,
        },
        "arms": list(ARMS),
        "candidates": list(CANDIDATES),
        "run_order": {
            "policy": (
                "arm-major: for each arm in the frozen order (parent first), "
                "the three input files ascending by seed — 6 authenticated "
                "engine events"
            ),
            "sequence": [
                f"{label}_seed{seed}" for label in ARMS for seed in INPUT_SEEDS
            ],
        },
        "gates": {
            "retention_adjudication": "pooled_k3_pooled_mean_over_three_screens",
            "bands_two_sided": True,
            "retention_correct_band": RETENTION_CORRECT_BAND,
            "retention_cap_contact_band": RETENTION_CAP_BAND,
            "retention_parsed_band": RETENTION_PARSED_BAND,
            "bands_apply_to_pooled_means_not_per_screen": True,
            "bands_evaluated_on_pooled_sums_times_screens": True,
            "no_absolute_per_kind_floors": True,
            "no_passing_candidate_keeps_aggregate_seed_sealed": True,
            "no_axis_gate_exists": True,
        },
        "prerequisites": {
            "parent_merge_receipt": (
                PARENT_MERGE_RECEIPT.relative_to(ROOT).as_posix()
            ),
            "parent_merge_receipt_sibling": (
                "verification aid: byte-identical when present, absent -> "
                "in-cell provenance copy authoritative"
            ),
            "parent_merge_receipt_sha256": EXPECTED_PARENT_MERGE_RECEIPT_SHA256,
            "parent_provenance_copy": (
                PARENT_PROVENANCE_COPY.relative_to(EXP).as_posix()
            ),
            "parent_merge_files": EXPECTED_PARENT_FILES,
            "parent_merge_tree_sha256": EXPECTED_PARENT_TREE_SHA256,
            "parent_weights_sha256": EXPECTED_PARENT_WEIGHTS_SHA256,
            "parent_eval_label": PARENT_LABEL,
            "arm_training_and_merge_receipts": (
                "deferred: authenticated at merge/eval time via "
                "merge_trained_arm.validate_published_merge"
            ),
        },
        "code_sha256": {
            name: sha256_file(path) for name, path in sorted(CODE_FILES.items())
        },
        "code_pins_deferred": {
            "reason": (
                "these files carry orchestrator-filled TODO-PIN constants; each "
                "is authenticated by hash inside its own run receipt, and all "
                "three are additionally frozen by check_design.py's NORMALIZED "
                "hashes (pin value slots canonicalized; pinned here via "
                "design_checker)"
            ),
            "files": CODE_PINS_DEFERRED,
        },
        "firewall": {
            "benchmark_data_read": False,
            "benchmark_gateway_exposed": False,
            "aggregate_seed_sealed": True,
        },
        "next_authorized_stage": "train",
    }
    rendered_receipt = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    outputs: dict[Path, bytes] = {}
    for seed in INPUT_SEEDS:
        outputs[SOURCES[seed]] = sources[seed]
        outputs[RUNNER_INPUTS[seed]] = runner_inputs[seed]
    outputs[RECEIPT] = rendered_receipt
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build_outputs(authenticate_parent=not args.check)
    if args.check:
        changed = [
            path
            for path, value in outputs.items()
            if not path.is_file() or path.read_bytes() != value
        ]
        if changed:
            parser.error(
                "local-gate artifacts are absent or changed: "
                + ", ".join(map(str, changed))
            )
    else:
        existing = [path for path in outputs if path.exists()]
        if existing:
            parser.error(
                "refusing to overwrite local-gate artifacts: "
                + ", ".join(map(str, existing))
            )
        for path, value in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(
        json.dumps(
            {
                "rows_per_arm": ROWS_PER_ARM,
                "sources_sha256": {
                    str(seed): sha256_bytes(outputs[SOURCES[seed]])
                    for seed in INPUT_SEEDS
                },
                "runner_inputs_sha256": {
                    str(seed): sha256_bytes(outputs[RUNNER_INPUTS[seed]])
                    for seed in INPUT_SEEDS
                },
                "receipt_sha256": sha256_bytes(outputs[RECEIPT]),
                # Runtime sibling-original status (the receipt records the
                # POLICY; presence is environment state and lives here only).
                "sibling_originals": {
                    "parent_merge_receipt": (
                        "present"
                        if PARENT_MERGE_RECEIPT.is_file()
                        else "absent, in-cell pin authoritative"
                    ),
                    "predecessor_gates": {
                        sibling.relative_to(ROOT).as_posix(): (
                            "present"
                            if sibling.is_file()
                            else "absent, in-cell pin authoritative"
                        )
                        for sibling, _, _, _ in PREDECESSOR_GATES
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
