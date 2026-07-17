#!/usr/bin/env python3
"""Freeze the two-instrument, four-input-file local gate and its model inputs.

The frozen gate comprises FOUR model-facing input files per arm:

- AXIS HOLDOUT (installability): 40 rows — all ``u_count_walk``, 5 per
  formalism across all eight, one per K_CYCLE value per formalism — from
  gen_count_walk_curriculum.generate_curriculum(HOLDOUT_MIX, 88056);
  ids ``axis88056_*``. All rows are FRESH instances from this cell's
  generator under the SAME per-row invariants as the treatment
  (exhaustive canonical-next re-derivation, unique both-trials fix,
  verified tried prefixes).
- THREE RETENTION SCREENS (pooled_k3 protocol): 104 rows each, 8 per
  each of the 13 original skills, from the canonical gen_curriculum.py
  at seeds 88057/88058/88059; ids ``ret<seed>_*``. No substitution was
  required for the gate seeds: every pinned seed was verified grep-fresh
  in seed contexts (known-taken: 88043/88047/88049 and everything
  <= 88055 including the reference cell's 88052-88055; the frozen
  sequence starts at the next free integer 88056).

PREREGISTERED NON-GATING MECHANISM READING recorded in this receipt: the
``episode_success_simulation`` — for each holdout row's underlying
broken machine, the number of turns a PERFECT canonical enumerator would
need (computed analytically by the generator, no model): the canonical
index of the unique both-trials fix plus one, plus the remaining-turn
count after the rendered tried prefix. The paired eval-time reading (the
per-row enumeration-fidelity booleans) is produced by eval_local_vllm.py
and summarized by check_local.py; neither ever feeds promotion.

Grading applies the frozen answer normalization documented in this receipt
(``answer_normalization``) identically to every arm and every input file.
Freshness is enforced fail-closed: the four input files are duplicate-free
internally and pairwise, and carry zero canonical-user-message overlap with
every pinned predecessor corpus and training stream (build_corpus.py's
single shared inventory, which includes the menders dose-scale cell — the
formalism-sharing predecessor — the lifecycle 18/23 statechain cells, the
clean gym-mix cell, the ENUMERATIVE-REPAIR reference cell's corpus and
streams, and the six zero-root lineage datasets), every prior local gate
(seeds 88013-88055, frozen files, including the reference cell's four),
regenerated prior local seeds, the regenerated count-walk treatment
corpus, and this cell's own materialized training streams.
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
import gen_count_walk_curriculum as cw_mod  # noqa: E402
from build_corpus import (  # noqa: E402
    PREDECESSOR_GATES,
    PREDECESSOR_SOURCES,
)
from check_local import (  # noqa: E402
    ANSWER_NORMALIZATION,
    ARMS,
    AXIS_KIND_COUNTS,
    AXIS_PER_SURFACE,
    AXIS_SURFACES,
    CANDIDATES,
    RETENTION_CAP_BAND,
    RETENTION_CORRECT_BAND,
    RETENTION_PARSED_BAND,
    SCREEN_SEEDS,
    SEED,
)


AGGREGATE_SEED = 78163
CONSTRUCTION_SEED = 77191
AXIS_MIX = cw_mod.HOLDOUT_MIX
RETENTION_MIX = ",".join(f"{name}=8" for name in curriculum.SKILLS)
AXIS_ROWS = 40
AXIS_SURFACE_COUNTS = {surface: AXIS_PER_SURFACE for surface in AXIS_SURFACES}
RETENTION_ROWS = 104
RETENTION_PER_SKILL = 8
ROWS_PER_ARM = AXIS_ROWS + RETENTION_ROWS * len(SCREEN_SEEDS)
INPUT_SEEDS = (SEED, *SCREEN_SEEDS)
SOURCES = {seed: EXP / "data" / f"local_tasks_seed{seed}.jsonl" for seed in INPUT_SEEDS}
RUNNER_INPUTS = {
    seed: EXP / "data" / f"local_input_seed{seed}.jsonl" for seed in INPUT_SEEDS
}
RECEIPT = EXP / "data" / "local_design_receipt.json"
# Frozen corpora that exist at design freeze (fail-closed hash pins).
FROZEN_SOURCES = (
    (EXP / "data" / "sft_blend.jsonl",
     "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2", 2240),
    (EXP / "data" / "sft_count_walk.jsonl",
     "21e6f5cb705f447f7a4dfc9bff24673f798f48df312b99a6cf686505855ee096", 160),
)
# Training streams materialize independently of this design; the zero-overlap
# check against them runs lazily whenever the files exist (generation and
# --check both call it), and their content is covered by construction: every
# stream row comes from sft_blend.jsonl or the count-walk treatment corpus.
TRAINING_STREAMS = (
    EXP / "data" / "replay_ctl7.jsonl",
    EXP / "data" / "count_walk.jsonl",
)
PRIOR_LOCAL_SEEDS = tuple(range(88000, 88056))
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PARENT_LABEL = "zero_root_parent"
PARENT_MERGE_RECEIPT = (
    ROOT
    / "experiments"
    / "qwen35_4b_zero_root_lineage_rebuild"
    / "runs"
    / "lineage"
    / "merge.json"
)
PARENT_MERGED = (
    ROOT
    / "large_artifacts"
    / "qwen35_4b_zero_root_lineage_rebuild"
    / "merged"
    / "zero_root_hygiene_explore"
)
EXPECTED_PARENT_MERGE_RECEIPT_SHA256 = (
    "e906caea7c4b86f4a3eacb96affb7cc2fa9b7cc11e11b634b651cabc5dd01d2b"
)
EXPECTED_PARENT_TREE_SHA256 = (
    "414f582950bf60fed2fe462cd141ab98d0f772087b4f9c6bc5aa12f03f379e7d"
)
EXPECTED_PARENT_WEIGHTS_SHA256 = (
    "6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932"
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
        "sha256": "f8981f4638d901471eb41aff0ffd0bfac88aebd6e3e4d4db1e1c733be16709c0",
        "size": 880,
    },
    {
        "name": "model.safetensors",
        "sha256": "6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932",
        "size": 9078620536,
    },
    {
        "name": "tokenizer.json",
        "sha256": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "size": 19989325,
    },
    {
        "name": "tokenizer_config.json",
        "sha256": "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
        "size": 1125,
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
# run_benchmark.py, materialize_streams.py, and train_trial.py are
# deliberately NOT pinned here: they carry orchestrator-filled TODO-PIN
# constants that change after this receipt freezes; each is authenticated by
# hash inside its own run receipt instead.
CODE_FILES = {
    "generator": Path(__file__),
    "curriculum": EXP / "scripts" / "gen_curriculum.py",
    "feedloop_machinery": EXP / "scripts" / "gen_feedloop_curriculum.py",
    "count_walk_curriculum": EXP / "scripts" / "gen_count_walk_curriculum.py",
    "corpus_builder": EXP / "scripts" / "build_corpus.py",
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
}
CODE_PINS_DEFERRED = [
    "scripts/eval_local_vllm.py",
    "scripts/run_benchmark.py",
    "scripts/materialize_streams.py",
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


def episode_success_simulation(axis_rows: list[dict]) -> dict:
    """The preregistered NON-GATING analytic reading: turns a PERFECT
    canonical enumerator needs per holdout episode (no model anywhere).
    Re-derived exhaustively per row (never trusted from the audit)."""
    from_scratch: list[int] = []
    remaining: list[int] = []
    for row in axis_rows:
        audit = row["_audit"]
        candidates, machine = cw_mod.rederive_candidates(audit)
        turns = machine["success_index"] + 1
        if turns != audit["episode_success_turns"]:
            raise ValueError("episode-success simulation disagrees with the audit")
        from_scratch.append(turns)
        remaining.append(machine["success_index"] - audit["k_tried"] + 1)
    return {
        "definition": (
            "a PERFECT canonical enumerator proposes candidates in the "
            "frozen order and stops at the unique both-trials fix; "
            "from_scratch = canonical index of that fix + 1; "
            "remaining_after_tried subtracts the rendered tried prefix"
        ),
        "computed_by": "generator re-derivation, model-free",
        "per_row_from_scratch": from_scratch,
        "per_row_remaining_after_tried": remaining,
        "from_scratch_distribution": {
            "min": min(from_scratch),
            "max": max(from_scratch),
            "mean": sum(from_scratch) / len(from_scratch),
            "counts": {
                str(key): value
                for key, value in sorted(Counter(from_scratch).items())
            },
        },
        "remaining_after_tried_distribution": {
            "min": min(remaining),
            "max": max(remaining),
            "mean": sum(remaining) / len(remaining),
            "counts": {
                str(key): value
                for key, value in sorted(Counter(remaining).items())
            },
        },
        "reported_not_gated": True,
    }


def build_axis_rows() -> tuple[list[dict], list[dict]]:
    axis_rows = cw_mod.generate_curriculum(AXIS_MIX, SEED)
    for row in axis_rows:
        row["task_id"] = f"axis{SEED}_{row['task_id']}"
    axis_summary = cw_mod.validate_generated(axis_rows)
    cw_mod.check_banned_vocabulary(axis_rows)
    cw_mod.check_corpus_balance(axis_rows)
    if (
        axis_summary["rows"] != AXIS_ROWS
        or axis_summary["kinds"] != AXIS_KIND_COUNTS
        or axis_summary["surfaces"] != AXIS_SURFACE_COUNTS
    ):
        raise ValueError("axis holdout no longer carries the frozen split")
    runner_rows = [
        {
            "id": row["task_id"],
            "messages": row["messages"],
            "meta": {
                "kind": row["kind"],
                "surface": row["surface"],
                "seed": SEED,
                "instrument": "axis_holdout",
            },
        }
        for row in axis_rows
    ]
    if any(set(row) != {"id", "messages", "meta"} for row in runner_rows):
        raise ValueError("local runner input schema leaked hidden fields")
    return axis_rows, runner_rows


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


def check_training_streams(local_messages: set[bytes]) -> list[str]:
    """Lazily enforce zero prompt overlap against every existing stream file."""
    present: list[str] = []
    for path in TRAINING_STREAMS:
        if not path.is_file():
            continue
        stream_messages = {
            message_bytes(row) for row in load_jsonl(path) if row.get("messages")
        }
        if local_messages & stream_messages:
            raise ValueError(f"fresh local prompts overlap a training stream: {path}")
        present.append(path.relative_to(EXP).as_posix())
    return present


def overlap_receipt(source_rows_by_seed: dict[int, list[dict]]) -> dict:
    messages_by_seed: dict[int, set[bytes]] = {}
    for seed, rows in source_rows_by_seed.items():
        expected = AXIS_ROWS if seed == SEED else RETENTION_ROWS
        messages = {message_bytes(row) for row in rows}
        if len(messages) != expected:
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
    predecessor_sources: dict[str, dict] = {}
    for path, expected_sha256, expected_rows in PREDECESSOR_SOURCES:
        messages = {
            message_bytes(row)
            for row in load_frozen(path, expected_sha256, expected_rows)
            if row.get("messages")
        }
        overlap = len(local_messages & messages)
        if overlap:
            raise ValueError(
                f"fresh local prompts overlap an inherited corpus or stream: {path}"
            )
        predecessor_sources[path.relative_to(ROOT).as_posix()] = {
            "sha256": expected_sha256,
            "messages_compared": len(messages),
            "overlap": overlap,
        }
    # Regenerated count-walk training rows at the frozen construction seed:
    # covers the treatment corpus by construction and pins the generator
    # behavior.
    enum_training_messages = {
        message_bytes(row)
        for row in cw_mod.generate_curriculum(cw_mod.ARM_MIX, CONSTRUCTION_SEED)
    }
    enum_training_overlap = len(local_messages & enum_training_messages)
    if enum_training_overlap:
        raise ValueError(
            "fresh local prompts overlap regenerated count-walk training rows"
        )
    prior_messages = {
        message_bytes(row)
        for seed in PRIOR_LOCAL_SEEDS
        for row in curriculum.generate_curriculum(curriculum.SMOKE_MIX, seed)
    }
    prior_local_overlap = len(local_messages & prior_messages)
    if prior_local_overlap:
        raise ValueError("fresh local prompts overlap prior local seeds")
    predecessor_gates: dict[str, dict] = {}
    for gate_path, gate_sha256, gate_rows in PREDECESSOR_GATES:
        predecessor_messages = {
            message_bytes(row)
            for row in load_frozen(gate_path, gate_sha256, gate_rows)
            if row.get("messages")
        }
        predecessor_overlap = len(local_messages & predecessor_messages)
        if predecessor_overlap:
            raise ValueError(
                f"fresh local prompts overlap a predecessor's frozen gate: {gate_path}"
            )
        predecessor_gates[gate_path.relative_to(ROOT).as_posix()] = {
            "sha256": gate_sha256,
            "messages_compared": len(predecessor_messages),
            "overlap": predecessor_overlap,
        }
    streams_present = check_training_streams(local_messages)
    return {
        "message_sha256s": sorted(sha256_bytes(value) for value in local_messages),
        "unique_local_messages": len(local_messages),
        "unique_local_messages_per_input": {
            str(seed): len(messages_by_seed[seed]) for seed in INPUT_SEEDS
        },
        "cross_screen_overlap": cross_screen,
        "frozen_sources": frozen_sources,
        "predecessor_sources": predecessor_sources,
        "regenerated_count_walk_training": {
            "mix": cw_mod.ARM_MIX,
            "construction_seed": CONSTRUCTION_SEED,
            "messages_compared": len(enum_training_messages),
            "overlap": enum_training_overlap,
        },
        "prior_local_seeds_compared": list(PRIOR_LOCAL_SEEDS),
        "prior_local_mix": curriculum.SMOKE_MIX,
        "prior_local_messages_compared": len(prior_messages),
        "prior_local_overlap": prior_local_overlap,
        "predecessor_gates": predecessor_gates,
        "training_streams": {
            "paths": [path.relative_to(EXP).as_posix() for path in TRAINING_STREAMS],
            "present_at_generation_or_check": streams_present,
            "checked_lazily_when_present": True,
            "policy": (
                "zero canonical user-message overlap enforced against every "
                "stream file that exists, at generation time and at --check time"
            ),
            "covered_by_construction": (
                "streams are composed solely of sft_blend.jsonl replay rows and "
                "the count-walk treatment corpus, both checked above"
            ),
        },
    }


def build_outputs(*, authenticate_parent: bool = True) -> dict[Path, bytes]:
    for path in CODE_FILES.values():
        if not path.is_file():
            raise ValueError(f"required local-design input is absent: {path}")
    if (
        not PARENT_MERGE_RECEIPT.is_file()
        or sha256_file(PARENT_MERGE_RECEIPT) != EXPECTED_PARENT_MERGE_RECEIPT_SHA256
    ):
        raise ValueError(f"published prerequisite changed: {PARENT_MERGE_RECEIPT}")
    parent_receipt = json.loads(PARENT_MERGE_RECEIPT.read_text(encoding="utf-8"))
    if (
        parent_receipt.get("name") != "zero_root_hygiene_explore"
        or parent_receipt.get("stage") != "merge"
        or parent_receipt.get("experiment_id") != "qwen35_4b_zero_root_lineage_rebuild"
        or parent_receipt.get("base_model", {}).get("id") != MODEL_ID
        or parent_receipt.get("base_model", {}).get("revision") != MODEL_REVISION
        or Path(parent_receipt.get("merged", "")).resolve() != PARENT_MERGED.resolve()
        or parent_receipt.get("output_tree_sha256") != EXPECTED_PARENT_TREE_SHA256
        or parent_receipt.get("output_files") != EXPECTED_PARENT_FILES
        or parent_receipt.get("weights_sha256") != EXPECTED_PARENT_WEIGHTS_SHA256
    ):
        raise ValueError("published zero-root-parent merge receipt violates pins")
    if authenticate_parent:
        parent_files = merged_tree_manifest(PARENT_MERGED)
        if (
            parent_files != EXPECTED_PARENT_FILES
            or tree_manifest_sha256(parent_files) != EXPECTED_PARENT_TREE_SHA256
        ):
            raise ValueError("published zero-root-parent composite tree changed")
    source_rows_by_seed: dict[int, list[dict]] = {}
    runner_rows_by_seed: dict[int, list[dict]] = {}
    source_rows_by_seed[SEED], runner_rows_by_seed[SEED] = build_axis_rows()
    for seed in SCREEN_SEEDS:
        source_rows_by_seed[seed], runner_rows_by_seed[seed] = build_screen(seed)
    sources = {seed: jsonl_bytes(source_rows_by_seed[seed]) for seed in INPUT_SEEDS}
    runner_inputs = {
        seed: jsonl_bytes(runner_rows_by_seed[seed]) for seed in INPUT_SEEDS
    }
    receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "count_walk_local_gate_design",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "loaded": False,
            "calls": 0,
        },
        "seed": SEED,
        "screen_seeds": list(SCREEN_SEEDS),
        "seed_freshness": {
            "input_seeds": list(INPUT_SEEDS),
            "substitution_required": False,
            "known_taken_nearby": {
                "88043": (
                    "taken as the retention seed of "
                    "qwen35_4b_counterfactual_plan_reflection_transfer"
                ),
                "88047": (
                    "taken as the reflection seed of "
                    "qwen35_4b_counterfactual_plan_reflection_transfer"
                ),
                "88049": (
                    "taken as the action seed of "
                    "qwen35_4b_counterfactual_plan_reflection_transfer"
                ),
                "88052_to_88055": (
                    "taken as the gate and retention-screen seeds of the "
                    "enumerative-repair reference cell "
                    "(qwen35_4b_enumerative_repair_protocol); everything "
                    "<= 88055 is known-taken"
                ),
            },
            "verified": (
                "88056/88057/88058/88059 verified grep-fresh in seed "
                "contexts across experiments/, knowledge/, "
                "research_programs/, scripts/, configs/, and docs/ at "
                "design time (every raw numeric hit is a sha256 or float "
                "substring, never a seed); the frozen gate sequence starts "
                "at the next free integer after the known-taken block and "
                "needed no substitution"
            ),
            "rule_if_collision": "next free integer, recorded here",
        },
        "aggregate_seed": AGGREGATE_SEED,
        "rows_per_arm": ROWS_PER_ARM,
        "instruments": {
            "axis_holdout": {
                "generator": "scripts/gen_count_walk_curriculum.py",
                "mix": AXIS_MIX,
                "rows": AXIS_ROWS,
                "kinds": dict(sorted(AXIS_KIND_COUNTS.items())),
                "surfaces": dict(sorted(AXIS_SURFACE_COUNTS.items())),
                "k_cycle": list(cw_mod.K_CYCLE),
                "id_prefix": f"axis{SEED}_",
                "seed": SEED,
                "fresh_instances_from_this_cells_generator": True,
                "same_invariants_as_the_treatment": True,
            },
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
        "episode_success_simulation": episode_success_simulation(
            source_rows_by_seed[SEED]
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
                "arm-major: for each arm in the frozen order, the four input "
                "files ascending by seed — 12 authenticated engine events"
            ),
            "sequence": [
                f"{label}_seed{seed}" for label in ARMS for seed in INPUT_SEEDS
            ],
        },
        "gates": {
            "axis_total_strictly_beats_parent_and_replay": True,
            "ties_fail": True,
            "single_kind_gate": {
                "kind": "u_count_walk",
                "no_per_kind_split_exists": True,
                "per_surface_reported_not_gated": True,
            },
            "mechanism_readings_reported_not_gated": {
                "episode_success_simulation": "this receipt (analytic, model-free)",
                "enumeration_fidelity": (
                    "eval-time per-axis-row booleans (legal / untried / "
                    "canonical-next), summarized per arm by check_local.py"
                ),
                "expression_cost": (
                    "NEW — the reading this lineage owes after the "
                    "reference cell's truncation forensics: eval-time "
                    "per-arm think-token-length distribution over the 40 "
                    "axis rows plus the truncation count, summarized per "
                    "arm by check_local.py from every row's "
                    "n_thinking_tokens and cap_contact"
                ),
            },
            "retention_adjudication": "pooled_k3_pooled_mean_over_three_screens",
            "retention_correct_band": RETENTION_CORRECT_BAND,
            "retention_cap_contact_band": RETENTION_CAP_BAND,
            "retention_parsed_band": RETENTION_PARSED_BAND,
            "bands_apply_to_pooled_means_not_per_screen": True,
            "bands_evaluated_on_pooled_sums_times_screens": True,
            "no_absolute_per_kind_floors": True,
            "no_passing_candidate_keeps_aggregate_seed_sealed": True,
        },
        "prerequisites": {
            "parent_merge_receipt": (
                PARENT_MERGE_RECEIPT.relative_to(ROOT).as_posix()
            ),
            "parent_merge_receipt_sha256": EXPECTED_PARENT_MERGE_RECEIPT_SHA256,
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
                "is authenticated by hash inside its own run receipt"
            ),
            "files": CODE_PINS_DEFERRED,
        },
        "firewall": {
            "benchmark_data_read": False,
            "benchmark_gateway_exposed": False,
            "aggregate_seed_sealed": True,
        },
        "next_authorized_stage": "train-control",
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
    outputs = build_outputs()
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
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
