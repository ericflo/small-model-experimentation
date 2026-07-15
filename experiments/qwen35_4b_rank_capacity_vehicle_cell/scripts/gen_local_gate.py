#!/usr/bin/env python3
"""Freeze the two-instrument benchmark-free MECHANISM gate and its model input.

This is a mechanism cell: one freshly trained composite (``axis160_r64``)
is read against two already-published composites on a fresh instrument, and
nothing promotes — there is no aggregate seed. The frozen gate input is the
concatenation of two instruments generated at the fresh gate seed:

- AXIS HOLDOUT (installability): 40 rows, 10 per v1 axis kind, from
  gen_axis_curriculum.generate_curriculum(HOLDOUT_MIX, SEED); ids ``axis88021_*``.
- RETENTION SCREEN: 104 rows, 8 per each of the 13 original skills, from
  gen_curriculum.generate_curriculum at the same seed; ids ``ret88021_*``.

Grading applies the frozen answer normalization documented in this receipt
(``answer_normalization``) identically to every arm and both instruments.
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

import gen_axis_curriculum as axis  # noqa: E402
import gen_curriculum as curriculum  # noqa: E402
from check_local import ANSWER_NORMALIZATION  # noqa: E402


SEED = 88021
CONSTRUCTION_SEED = 77117
AXIS_MIX = axis.HOLDOUT_MIX
RETENTION_MIX = ",".join(f"{name}=8" for name in curriculum.SKILLS)
AXIS_ROWS = 40
AXIS_PER_KIND = 10
RETENTION_ROWS = 104
RETENTION_PER_SKILL = 8
ROWS = AXIS_ROWS + RETENTION_ROWS
SOURCE = EXP / "data" / f"local_tasks_seed{SEED}.jsonl"
RUNNER_INPUT = EXP / "data" / f"local_input_seed{SEED}.jsonl"
RECEIPT = EXP / "data" / "local_design_receipt.json"
# Frozen corpora that exist at design freeze (fail-closed hash pins): the
# byte-identically inherited treatment corpus and the byte-identical inherited
# replay blend.
FROZEN_SOURCES = (
    (EXP / "data" / "sft_blend.jsonl",
     "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2", 2240),
    (EXP / "data" / "sft_axis160.jsonl",
     "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e", 160),
)
# Every inherited corpus and every materialized training stream in the
# lineages of the three gate arms (fail-closed hash pins): the goal-gap
# donor's corpora and both of its streams (the axis160 corpus lineage), and
# the dose-diversity cell's corpora and its one stream (the axis160_direct
# rank-32 composite lineage).
PREDECESSOR_SOURCES = (
    (
        ROOT
        / "experiments"
        / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "data"
        / "sft_axis160.jsonl",
        "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e",
        160,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "data"
        / "sft_blend.jsonl",
        "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
        2240,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "data"
        / "replay_repeat.jsonl",
        "878f501f3c6665982b7b3e9ffb2de675382d6b366b62a61d0a13e6709841c468",
        1520,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "data"
        / "axis_curriculum.jsonl",
        "e2561485ae28c9c374b663af4a002ab67e02cc75ec8994d1d89770e6b198e98a",
        1520,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_dose_diversity_mechanism_cell"
        / "data"
        / "sft_blend.jsonl",
        "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
        2240,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_dose_diversity_mechanism_cell"
        / "data"
        / "sft_axis160.jsonl",
        "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e",
        160,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_dose_diversity_mechanism_cell"
        / "data"
        / "axis160_direct.jsonl",
        "0cbf3abf04781f3217189dd7b2bdf173d83772b0200b283a7c608dc3d386c1ee",
        1520,
    ),
)
# ALL EIGHT predecessor experiments' frozen local gates (fail-closed hash
# pins): the fresh-surface 104-task seed-88013 gate, the goal-gap 144-task
# seed-88014 gate, the stack trial's 144-task seed-88015 gate, the
# re-adjudication's 144-task seed-88016 gate, the axis-v2 staged-repair
# 154-task seed-88017 gate, the de-stack trial's 124-task seed-88018 gate,
# the interleaved-dose trial's 124-task seed-88019 gate, and the
# dose-diversity mechanism cell's 144-task seed-88020 gate.
PREDECESSOR_GATES = (
    (
        ROOT
        / "experiments"
        / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
        / "data"
        / "local_tasks_seed88013.jsonl",
        "be817bd09f80a1fdf045bfb7638634f5f0e0e34ac6a404a34a98283e49dc5c2b",
        104,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "data"
        / "local_tasks_seed88014.jsonl",
        "4753a277bdcfcab515a8bb5ece541498729107de6002f4be011b57b2eed34873",
        144,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_axis_replay_stack_medium_target_match"
        / "data"
        / "local_tasks_seed88015.jsonl",
        "1c019d70751a789a17ef555d51862c5a017f0a3c2bb1ebbce41a1a93496fefd1",
        144,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_axis_stack_readjudication_medium_pilot"
        / "data"
        / "local_tasks_seed88016.jsonl",
        "13788cc2e6fc237cfb709d2b22ae24dc6263a91f97c4822b856d098f816020b4",
        144,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_axis_corpus_v2_staged_repair"
        / "data"
        / "local_tasks_seed88017.jsonl",
        "746cf1c4efae1cdd6614827f6fe8abb0eb7fb8cf5c2dc0c2c7dac4a6d930db81",
        154,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_hygiene_explore_destack_medium"
        / "data"
        / "local_tasks_seed88018.jsonl",
        "597f10a44674cc12e5f499be8de6804bb040985019b18aacc5527339a26857eb",
        124,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_interleaved_replay_dose_medium"
        / "data"
        / "local_tasks_seed88019.jsonl",
        "6e927f591f9ae9d2edad6e263be3f7c0262b39de4854314a64802e656b98c15b",
        124,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_dose_diversity_mechanism_cell"
        / "data"
        / "local_tasks_seed88020.jsonl",
        "8e4dd8b21febf64048fef5cf15541356d4a5808e45caf5d6f19525e154168822",
        144,
    ),
)
# The training stream materializes independently of this design; the
# zero-overlap check against it runs lazily whenever the file exists
# (generation and --check both call it), and its content is covered by
# construction: every stream row comes from sft_blend.jsonl or the inherited
# treatment corpus above.
TRAINING_STREAMS = (
    EXP / "data" / "axis160_r64.jsonl",
)
PRIOR_LOCAL_SEEDS = tuple(range(88000, 88021))
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ARMS = (
    "clean_parent",
    "axis160_direct",
    "axis160_r64",
)
CANDIDATES = ("axis160_r64",)
INHERITED_ARMS = ("clean_parent", "axis160_direct")
# The two inherited arms are externally published composites; every one of
# their pins is filled at design freeze. Only the candidate's own merge is
# deferred (authenticated at merge/eval time via
# merge_trained_arm.validate_published_merge and the eval tree TODO-PIN).
PARENT_EXP_NAME = "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
DOSE_EXP_NAME = "qwen35_4b_dose_diversity_mechanism_cell"
COMPOSITE_RECEIPTS = {
    "clean_parent": (
        ROOT / "experiments" / PARENT_EXP_NAME / "runs" / "merges"
        / "designed_fresh.json"
    ),
    "axis160_direct": (
        ROOT / "experiments" / DOSE_EXP_NAME / "runs" / "merges"
        / "axis160_direct.json"
    ),
}
MERGED = {
    "clean_parent": (
        ROOT / "large_artifacts" / PARENT_EXP_NAME / "merged" / "designed_fresh"
    ),
    "axis160_direct": (
        ROOT / "large_artifacts" / DOSE_EXP_NAME / "merged" / "axis160_direct"
    ),
}
EXPECTED_RECEIPT_SHA256 = {
    "clean_parent": "ab3f20cc93d3fe21ead7a1d573edbca2903d59d6f9fe3d2af0c93e823676acc2",
    "axis160_direct": "7b878bb357e044c58a5ba27f34365906059259237e657ca77c2ad2e8fb77ea39",
}
EXPECTED_COMPOSITE_NAMES = {
    "clean_parent": "designed_fresh",
    "axis160_direct": "axis160_direct",
}
EXPECTED_TREE_SHA256 = {
    "clean_parent": "93433aa2d5f3f0d6d4540126579c09feee1d8502df702c1563bae28eb7f60255",
    "axis160_direct": "b773fd439eca753e70d3b16862497ee0ba9783cdb024ddff84b4595d10b7da61",
}
EXPECTED_WEIGHTS_SHA256 = {
    "clean_parent": "0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979",
    "axis160_direct": "3cd4272ee386dc3b8e878ef3b2ac9ebf94dd8d47ef7d9d480f921e7e45b37279",
}
EXPECTED_FILES = {
    "clean_parent": [
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
            "sha256": "10b0de69bbda6edf8e216f45667b15f73557102f0661461fec620786e8a9170f",
            "size": 897,
        },
        {
            "name": "model.safetensors",
            "sha256": "0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979",
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
    ],
    "axis160_direct": [
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
            "sha256": "a32f0d334611321f2ff4c3a6252a35fd1142da0c472c1af895ca9dd96b0f48f1",
            "size": 875,
        },
        {
            "name": "model.safetensors",
            "sha256": "3cd4272ee386dc3b8e878ef3b2ac9ebf94dd8d47ef7d9d480f921e7e45b37279",
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
    ],
}
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
# Files pinned by hash into the frozen design receipt. eval_local_vllm.py is
# deliberately NOT pinned here: it carries an orchestrator-filled TODO-PIN
# constant (the candidate tree) that changes after this receipt freezes; it is
# authenticated by hash inside its own run receipt instead.
CODE_FILES = {
    "generator": Path(__file__),
    "curriculum": EXP / "scripts" / "gen_curriculum.py",
    "axis_curriculum": EXP / "scripts" / "gen_axis_curriculum.py",
    "gate": EXP / "scripts" / "check_local.py",
    "harness": EXP / "scripts" / "run.py",
    "runner": EXP / "src" / "vllm_runner.py",
    # merge_trained_arm.py carries no orchestrator-filled constants, so it is
    # pinned here and demands this exact pin back (code_sha256.merge) before
    # any composite is produced.
    "merge": EXP / "scripts" / "merge_trained_arm.py",
    "external_merger": (
        ROOT
        / "experiments"
        / "qwen35_4b_same_prefix_advantage_routing"
        / "scripts"
        / "merge_adapter.py"
    ),
}
CODE_PINS_DEFERRED = [
    "scripts/eval_local_vllm.py",
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


def load_frozen(path: Path, expected_sha256: str, expected_rows: int) -> list[dict]:
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


def authenticate_composite_receipt(label: str) -> dict:
    """Authenticate one inherited composite's committed merge receipt."""
    receipt_path = COMPOSITE_RECEIPTS[label]
    if (
        not receipt_path.is_file()
        or sha256_file(receipt_path) != EXPECTED_RECEIPT_SHA256[label]
    ):
        raise ValueError(f"published prerequisite receipt changed: {receipt_path}")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        payload.get("name") != EXPECTED_COMPOSITE_NAMES[label]
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or Path(payload.get("merged", "")).resolve() != MERGED[label].resolve()
        or payload.get("output_tree_sha256") != EXPECTED_TREE_SHA256[label]
        or payload.get("output_files") != EXPECTED_FILES[label]
        or payload.get("weight_files")
        != [{"name": "model.safetensors", "sha256": EXPECTED_WEIGHTS_SHA256[label]}]
    ):
        raise ValueError(f"published prerequisite receipt violates pins: {label}")
    return payload


def authenticate_composite_tree(label: str) -> None:
    files = merged_tree_manifest(MERGED[label])
    if (
        files != EXPECTED_FILES[label]
        or tree_manifest_sha256(files) != EXPECTED_TREE_SHA256[label]
    ):
        raise ValueError(f"published composite tree changed: {label}")


def build_rows() -> tuple[list[dict], list[dict]]:
    axis_rows = axis.generate_curriculum(AXIS_MIX, SEED)
    for row in axis_rows:
        row["task_id"] = f"axis{SEED}_{row['task_id']}"
    axis_summary = axis.validate_generated(axis_rows)
    axis.check_banned_vocabulary(axis_rows)
    expected_axis_kinds = {f"u_{name}": AXIS_PER_KIND for name in axis.SKILLS}
    if axis_summary["rows"] != AXIS_ROWS or axis_summary["kinds"] != expected_axis_kinds:
        raise ValueError("axis holdout no longer has ten rows per axis kind")

    retention_rows = curriculum.generate_curriculum(RETENTION_MIX, SEED)
    for row in retention_rows:
        row["task_id"] = f"ret{SEED}_{row['task_id']}"
    retention_summary = curriculum.validate_generated(retention_rows)
    expected_retention_kinds = {
        f"u_{name}": RETENTION_PER_SKILL for name in curriculum.SKILLS
    }
    if (
        retention_summary["rows"] != RETENTION_ROWS
        or retention_summary["kinds"] != expected_retention_kinds
    ):
        raise ValueError(
            "retention screen no longer has eight rows per registered skill"
        )

    source_rows = axis_rows + retention_rows
    if len({row["task_id"] for row in source_rows}) != ROWS:
        raise ValueError("gate instruments collide on task ids")
    runner_rows = [
        {
            "id": row["task_id"],
            "messages": row["messages"],
            "meta": {
                "kind": row["kind"],
                "surface": row["surface"],
                "seed": SEED,
                "instrument": (
                    "axis_holdout" if index < AXIS_ROWS else "retention"
                ),
            },
        }
        for index, row in enumerate(source_rows)
    ]
    if any(set(row) != {"id", "messages", "meta"} for row in runner_rows):
        raise ValueError("local runner input schema leaked hidden fields")
    return source_rows, runner_rows


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


def overlap_receipt(source_rows: list[dict]) -> dict:
    local_messages = {message_bytes(row) for row in source_rows}
    if len(local_messages) != ROWS:
        raise ValueError("gate instruments collide on canonical user messages")
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
                f"fresh local prompts overlap a donor corpus or stream: {path}"
            )
        predecessor_sources[path.relative_to(ROOT).as_posix()] = {
            "sha256": expected_sha256,
            "messages_compared": len(messages),
            "overlap": overlap,
        }
    # Regenerated axis training rows at the frozen construction seed: covers
    # the treatment corpus by construction and pins the generator behavior.
    axis_training_messages = {
        message_bytes(row)
        for row in axis.generate_curriculum(axis.ARM_MIX, CONSTRUCTION_SEED)
    }
    axis_training_overlap = len(local_messages & axis_training_messages)
    if axis_training_overlap:
        raise ValueError("fresh local prompts overlap regenerated axis training rows")
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
    check_training_streams(local_messages)
    return {
        "message_sha256s": sorted(sha256_bytes(value) for value in local_messages),
        "unique_local_messages": len(local_messages),
        "frozen_sources": frozen_sources,
        "predecessor_sources": predecessor_sources,
        "regenerated_axis_training": {
            "mix": axis.ARM_MIX,
            "construction_seed": CONSTRUCTION_SEED,
            "messages_compared": len(axis_training_messages),
            "overlap": axis_training_overlap,
        },
        "prior_local_seeds_compared": list(PRIOR_LOCAL_SEEDS),
        "prior_local_mix": curriculum.SMOKE_MIX,
        "prior_local_messages_compared": len(prior_messages),
        "prior_local_overlap": prior_local_overlap,
        "predecessor_gates": predecessor_gates,
        "training_streams": {
            "paths": [path.relative_to(EXP).as_posix() for path in TRAINING_STREAMS],
            "checked_lazily_when_present": True,
            "policy": (
                "zero canonical user-message overlap enforced against every "
                "stream file that exists, at generation time and at --check time"
            ),
            "covered_by_construction": (
                "the stream is composed solely of sft_blend.jsonl replay rows "
                "and the inherited axis treatment corpus, both checked above"
            ),
        },
    }


def build_outputs(*, authenticate_composites: bool = True) -> dict[Path, bytes]:
    for path in CODE_FILES.values():
        if not path.is_file():
            raise ValueError(f"required local-design input is absent: {path}")
    composite_receipts = {
        label: authenticate_composite_receipt(label) for label in INHERITED_ARMS
    }
    if authenticate_composites:
        for label in INHERITED_ARMS:
            authenticate_composite_tree(label)
    source_rows, runner_rows = build_rows()
    source = jsonl_bytes(source_rows)
    runner_input = jsonl_bytes(runner_rows)
    receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "rank_capacity_vehicle_local_gate_design",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "loaded": False,
            "calls": 0,
        },
        "seed": SEED,
        "rows": ROWS,
        "mechanism_cell": True,
        "instruments": {
            "axis_holdout": {
                "generator": "scripts/gen_axis_curriculum.py",
                "mix": AXIS_MIX,
                "rows": AXIS_ROWS,
                "per_kind": AXIS_PER_KIND,
                "id_prefix": f"axis{SEED}_",
            },
            "retention": {
                "generator": "scripts/gen_curriculum.py",
                "mix": RETENTION_MIX,
                "rows": RETENTION_ROWS,
                "per_kind": RETENTION_PER_SKILL,
                "id_prefix": f"ret{SEED}_",
            },
        },
        "kinds": dict(sorted(Counter(row["kind"] for row in source_rows).items())),
        "source": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(source),
            "contains_executable_truth": True,
        },
        "runner_input": {
            "path": RUNNER_INPUT.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(runner_input),
            "schema": ["id", "messages", "meta"],
            "contains_answer": False,
            "contains_oracle": False,
        },
        "answer_normalization": ANSWER_NORMALIZATION,
        "freshness": overlap_receipt(source_rows),
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
        "readings": {
            "no_promotion": True,
            "retention_delta_r64": (
                "axis160_r64 retention correct minus clean_parent retention "
                "correct on the fresh 104-row screen"
            ),
            "retention_delta_r32": (
                "axis160_direct retention correct minus clean_parent "
                "retention correct on the same fresh screen (seed-88020 "
                "measured -9; re-measured fresh here)"
            ),
            "capacity_mechanism_verdict": {
                "SCREEN_INSTABILITY": "retention_delta_r32 >= -5",
                "CAPACITY_SUPPORTED": (
                    "retention_delta_r32 <= -6 and retention_delta_r64 >= -5"
                ),
                "CAPACITY_REFUTED": (
                    "retention_delta_r32 <= -6 and retention_delta_r64 <= -6"
                ),
            },
            "install_preserved_flag": (
                "axis160_r64 axis total correct >= axis160_direct axis total "
                "correct on this same screen"
            ),
            "complete_event_always_exits_zero": True,
            "no_aggregate_seed_exists": True,
        },
        "prerequisites": {
            "inherited_arms": {
                label: {
                    "merge_receipt": (
                        COMPOSITE_RECEIPTS[label].relative_to(ROOT).as_posix()
                    ),
                    "merge_receipt_sha256": EXPECTED_RECEIPT_SHA256[label],
                    "composite": MERGED[label].relative_to(ROOT).as_posix(),
                    "composite_name": EXPECTED_COMPOSITE_NAMES[label],
                    "tree_sha256": EXPECTED_TREE_SHA256[label],
                    "weights_sha256": EXPECTED_WEIGHTS_SHA256[label],
                    "files": EXPECTED_FILES[label],
                    "receipt_authenticated": composite_receipts[label]["name"]
                    == EXPECTED_COMPOSITE_NAMES[label],
                }
                for label in INHERITED_ARMS
            },
            "candidate_training_and_merge_receipts": (
                "deferred: authenticated at merge/eval time via "
                "merge_trained_arm.validate_published_merge"
            ),
        },
        "code_sha256": {
            name: sha256_file(path) for name, path in sorted(CODE_FILES.items())
        },
        "code_pins_deferred": {
            "reason": (
                "these files carry orchestrator-filled TODO-PIN constants or do "
                "not exist at design freeze; each is authenticated by hash "
                "inside its own run receipt"
            ),
            "files": CODE_PINS_DEFERRED,
        },
        "firewall": {
            "benchmark_data_read": False,
            "benchmark_gateway_exposed": False,
            "no_aggregate_seed_exists": True,
        },
        "next_authorized_stage": "train-candidate",
    }
    rendered_receipt = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    return {SOURCE: source, RUNNER_INPUT: runner_input, RECEIPT: rendered_receipt}


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
                "rows": ROWS,
                "source_sha256": sha256_bytes(outputs[SOURCE]),
                "runner_input_sha256": sha256_bytes(outputs[RUNNER_INPUT]),
                "receipt_sha256": sha256_bytes(outputs[RECEIPT]),
                "training_streams_present_now": [
                    path.relative_to(EXP).as_posix()
                    for path in TRAINING_STREAMS
                    if path.is_file()
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
