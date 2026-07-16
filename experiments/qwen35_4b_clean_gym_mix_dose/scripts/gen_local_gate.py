#!/usr/bin/env python3
"""Freeze the two-instrument, four-input-file local gate and its model inputs.

The frozen gate comprises FOUR model-facing input files per arm:

- AXIS HOLDOUT (installability): 40 rows — 14 ``u_siren_episode`` + 13
  ``u_statechain`` + 13 ``u_mirage_abstain`` — from
  gen_gym_mix_curriculum.generate_curriculum(HOLDOUT_MIX, 88046); ids
  ``axis88046_*``. All rows are FRESH instances from this cell's generator
  (the statechain rows come through the byte-copied PROVEN generator).
- THREE RETENTION SCREENS (pooled_k3 protocol): 104 rows each, 8 per each of
  the 13 original skills, from the canonical gen_curriculum.py at seeds
  88048/88050/88051; ids ``ret<seed>_*``. Seed substitutions are recorded
  fail-closed: the frozen design named 88047/88048/88049, but 88047 and
  88049 are TAKEN (the reflection and action seeds of
  qwen35_4b_counterfactual_plan_reflection_transfer, whose retention seed
  also took 88043), so the screens advance to the next free integers —
  88048, 88050, 88051 — each verified grep-fresh in seed contexts.

Grading applies the frozen answer normalization documented in this receipt
(``answer_normalization``) identically to every arm and every input file.
Freshness is enforced fail-closed: the four input files are duplicate-free
internally and pairwise, and carry zero canonical-user-message overlap with
every pinned predecessor corpus and training stream (build_corpus.py's
single shared inventory, which includes the lifecycle 18/23 statechain
cells' corpora, streams, and gates, the menders dose-scale cell's, and the
six zero-root lineage datasets), every prior local gate (seeds
88013-88045, frozen files), regenerated prior local seeds, the regenerated
gym-mix treatment corpus, and this cell's own materialized training
streams.
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
import gen_gym_mix_curriculum as gym  # noqa: E402
from build_corpus import (  # noqa: E402
    PREDECESSOR_GATES,
    PREDECESSOR_SOURCES,
)
from check_local import (  # noqa: E402
    ANSWER_NORMALIZATION,
    ARMS,
    AXIS_KIND_COUNTS,
    CANDIDATES,
    KINDS_REQUIRED_TO_WIN,
    RETENTION_CAP_BAND,
    RETENTION_CORRECT_BAND,
    RETENTION_PARSED_BAND,
    SCREEN_SEEDS,
    SEED,
)


AGGREGATE_SEED = 78161
CONSTRUCTION_SEED = 77180
AXIS_MIX = gym.HOLDOUT_MIX
RETENTION_MIX = ",".join(f"{name}=8" for name in curriculum.SKILLS)
AXIS_ROWS = 40
AXIS_SURFACE_COUNTS = {
    "stillroom": 14,
    "counterhouse": 13,
    "brewvat": 4,
    "courierloft": 3,
    "peatstove": 3,
    "muletrack": 3,
}
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
    (EXP / "data" / "sft_gym_mix.jsonl",
     "6295011622096992e889b58a1a004fee26f4f9787bd952d348c0bf8593564a89", 160),
)
# Training streams materialize independently of this design; the zero-overlap
# check against them runs lazily whenever the files exist (generation and
# --check both call it), and their content is covered by construction: every
# stream row comes from sft_blend.jsonl or the gym-mix treatment corpus.
TRAINING_STREAMS = (
    EXP / "data" / "replay_ctl5.jsonl",
    EXP / "data" / "gym_mix.jsonl",
)
PRIOR_LOCAL_SEEDS = tuple(range(88000, 88046))
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
    "statechain_curriculum": EXP / "scripts" / "gen_statechain_curriculum.py",
    "gym_mix_curriculum": EXP / "scripts" / "gen_gym_mix_curriculum.py",
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


def build_axis_rows() -> tuple[list[dict], list[dict]]:
    axis_rows = gym.generate_curriculum(AXIS_MIX, SEED)
    for row in axis_rows:
        row["task_id"] = f"axis{SEED}_{row['task_id']}"
    axis_summary = gym.validate_generated(axis_rows)
    gym.check_banned_vocabulary(axis_rows)
    gym.check_corpus_balance(axis_rows)
    if (
        axis_summary["rows"] != AXIS_ROWS
        or axis_summary["kinds"] != AXIS_KIND_COUNTS
        or axis_summary["surfaces"] != AXIS_SURFACE_COUNTS
    ):
        raise ValueError("axis holdout no longer carries the frozen kind split")
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
    # Regenerated gym-mix training rows at the frozen construction seed:
    # covers the treatment corpus by construction and pins the generator
    # behavior.
    gym_training_messages = {
        message_bytes(row)
        for row in gym.generate_curriculum(gym.ARM_MIX, CONSTRUCTION_SEED)
    }
    gym_training_overlap = len(local_messages & gym_training_messages)
    if gym_training_overlap:
        raise ValueError(
            "fresh local prompts overlap regenerated gym-mix training rows"
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
        "regenerated_gym_mix_training": {
            "mix": gym.ARM_MIX,
            "construction_seed": CONSTRUCTION_SEED,
            "messages_compared": len(gym_training_messages),
            "overlap": gym_training_overlap,
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
                "the gym-mix treatment corpus, both checked above"
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
        "stage": "clean_gym_mix_local_gate_design",
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
            "substitution_required": True,
            "substitutions": {
                "88047": (
                    "taken as the reflection seed of "
                    "qwen35_4b_counterfactual_plan_reflection_transfer; the "
                    "screen sequence advances to the next free integer"
                ),
                "88049": (
                    "taken as the action seed of "
                    "qwen35_4b_counterfactual_plan_reflection_transfer; the "
                    "screen sequence advances to the next free integer"
                ),
            },
            "verified": (
                "88046/88048/88050/88051 verified grep-fresh in seed contexts "
                "across experiments/, knowledge/, research_programs/, scripts/, "
                "and configs/ at design time (every raw numeric hit is a "
                "sha256 or float substring, never a seed); 88043 remains "
                "taken as the retention seed of "
                "qwen35_4b_counterfactual_plan_reflection_transfer and "
                "88047/88049 as its reflection/action seeds — the screen "
                "sequence is therefore 88048, 88050, 88051"
            ),
            "rule_if_collision": "next free integer, recorded here",
        },
        "aggregate_seed": AGGREGATE_SEED,
        "rows_per_arm": ROWS_PER_ARM,
        "instruments": {
            "axis_holdout": {
                "generator": "scripts/gen_gym_mix_curriculum.py",
                "mix": AXIS_MIX,
                "rows": AXIS_ROWS,
                "kinds": dict(sorted(AXIS_KIND_COUNTS.items())),
                "surfaces": dict(sorted(AXIS_SURFACE_COUNTS.items())),
                "id_prefix": f"axis{SEED}_",
                "seed": SEED,
                "fresh_instances_from_this_cells_generator": True,
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
            "kind_breadth_gate": {
                "kinds_required": KINDS_REQUIRED_TO_WIN,
                "rule": (
                    "at least TWO of the three kinds individually strict over "
                    "BOTH controls; a tie on either control fails that kind"
                ),
            },
            "per_surface_reported_not_gated": True,
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
