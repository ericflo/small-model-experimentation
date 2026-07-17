#!/usr/bin/env python3
"""Replay the ENTIRE lineage of this cell's evaluated composites, in-cell.

Standalone-reproducibility gate (AGENTS.md non-negotiables;
docs/quality_gates.md): this cell trains and evaluates composites whose
lineage runs through the whole zero-root chain, so it carries the complete
model-reproduction package in its OWN directory. The chain, documented by
``data/lineage/lineage_manifest.json`` (lifecycle 27's clean-chain manifest,
carried through lifecycle 28's ``stage7_confirmation_arms`` extension and
EXTENDED here with the ``stage8_replay_compound`` block):

    stages 1-6: the six documented, contamination-free zero-root stages
    replayed from a FRESH zero-initialized rank-32/alpha-64 adapter on
    the raw pinned HF base (stage 1 has NO warm start; stages 2-6
    warm-start from the previous stage), exactly as lifecycle 22
    recorded; the stage-6 adapter merges onto the raw HF base to
    reproduce the zero-root parent composite (tree 414f5829...,
    weights 6e9aad25...);

    stage 7: BOTH lifecycle-27 arms — ``replay_ctl7`` first, then
    ``count_walk`` — each a FRESH rank-32/alpha-64 adapter trained ON
    the zero-root composite via the trainer's ``--model-path`` at the
    fixed training seed 85, then merged onto that composite via
    ``scripts/merge_adapter.py`` ``--base-model``; the rebuilt
    ``count_walk`` composite (tree d5fdc55c..., weights ddd7bc4b...) is
    this cell's PARENT eval arm and the stage-8 training base;

    stage 8: this cell's ``replay_compound`` arm — a FRESH
    rank-32/alpha-64 adapter trained ON the count_walk composite via
    ``--model-path`` at the fixed fresh training seed 86 over the FULL
    2,240-row replay pool ``data/sft_blend.jsonl``, then merged onto
    that composite — this cell's candidate eval arm.

The ``base`` eval arm is a weight-identical reserialization of the raw
pinned HF revision and needs no training chain. Cross-experiment SHAs
(the copied provenance receipts, the committed lifecycle-27 merge
receipts) remain VERIFICATION AIDS ONLY; the reproduction path is the
in-cell package this script replays. NO undocumented root exists
anywhere in the chain and none may ever be vendored into this cell.

Modes:

- ``--verify-inputs`` (fast, no GPU, wired into ``run.py --smoke``):
  authenticates the complete package — the extended manifest bytes
  (sha256-pinned) and internal chaining, every copied dataset (sha256 +
  row count) including both stage-7 arm streams, the stage-8 replay
  pool, and their materialization inputs, the stream token receipt,
  every trainer/merger/wrapper copy (sha256), the in-cell provenance
  receipt copies (sha-pinned; lifecycle 22's committed sibling originals
  are verification aids — byte-identical when present, skipped with a
  recorded note when absent), and equality of the composite pins with
  the eval runner's frozen constants — and asserts no blend root is
  vendored. Writes nothing.
- default (GPU, ~4h): replays stages 1-8 into
  ``large_artifacts/<exp>/rebuild/`` with a receipt per stage under
  ``runs/lineage_rebuild/``. The benchmark event does NOT run this: it
  evaluates the published composites, authenticated fail-closed by full
  tree+weights sha256. On this repo's pinned stack the stage 1-6 replay
  must reproduce the zero-root composite's weights byte-identically
  (the merge fails closed otherwise); the stage-7 and stage-8 composites
  record their tree/weights hashes against the frozen eval pins as
  verification aids (on a different stack equivalence must be
  re-established behaviorally through the trusted gateway).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
MANIFEST = EXP / "data" / "lineage" / "lineage_manifest.json"
# Byte pin of this cell's extended clean-chain manifest (stage-8 extension).
MANIFEST_SHA256 = "45d1a0d9b9262dad00eb5576fff5a0427aab7f505e01232ae584fb3b65636d1c"
SOURCE_ZERO_ROOT_EXP = ROOT / "experiments" / "qwen35_4b_zero_root_lineage_rebuild"
PROVENANCE_DIR = EXP / "data" / "lineage" / "provenance"
LINEAGE_RECEIPTS = EXP / "runs" / "lineage_rebuild"
REBUILD_ROOT = ROOT / "large_artifacts" / EXP.name / "rebuild"
ADAPTER_ROOT = REBUILD_ROOT / "adapters"
ZERO_ROOT_MERGED_OUT = REBUILD_ROOT / "merged" / "zero_root_hygiene_explore"
ZERO_ROOT_MERGE_RECEIPT = LINEAGE_RECEIPTS / "merge_zero_root.json"
# The clean chain is the point: no copy of any undocumented root adapter may
# ever exist inside this cell's artifact storage.
FORBIDDEN_ROOT_DIRS = (
    ROOT / "large_artifacts" / EXP.name / "lineage_root",
    REBUILD_ROOT / "lineage_root",
)
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
STAGE_COUNT = 7
ZERO_ROOT_STAGES = 6
FRESH_INIT = "fresh_zero_init"
STAGE7_WARM = "fresh_adapter_on_zero_root_composite_via_model_path"
STAGE8_WARM = "fresh_adapter_on_count_walk_composite_via_model_path"
ARM_ORDER = ("replay_ctl7", "count_walk")
TRAINING_SEED = 85
STAGE8_ARM = "replay_compound"
STAGE8_SEED = 86
STAGE8_DATASET_SHA256 = (
    "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
)
STAGE8_DATASET_ROWS = 2240
SHA_RE = re.compile(r"[0-9a-f]{64}")
MANIFEST_KEYS = {
    "schema_version", "experiment_id", "stage", "framing", "directive", "model",
    "carrier", "base_reserialized_note", "clean_chain", "trainers",
    "merger", "stages", "final_merge", "determinism", "verification",
    "stage7_confirmation_arms", "stage8_replay_compound",
}
STAGE_KEYS = {
    "stage", "name", "dataset", "warm_start", "trainer", "trainer_sha256",
    "seed", "hyperparameters", "produced_zero_root",
}
STAGE7_EXTRA_KEYS = {"stage7_base", "receipt"}
HYPER_KEYS = {
    "epochs", "lr", "rank", "alpha", "batch_size", "grad_accum",
    "max_length", "w_think",
}
# Stage seeds: 42/43/44/47/51/55 are INHERITED STAGE CONSTANTS from the
# documented recipe; 85 is lifecycle 27's fixed paired-trial training seed,
# shared by BOTH stage-7 arms; 86 is this cell's fresh stage-8 seed.
STAGE_SEEDS = (42, 43, 44, 47, 51, 55, TRAINING_SEED)
# The zero-root composite this chain must reproduce before stage 7 lands
# (fail closed on this stack; verification aid on any other).
ZERO_ROOT_TREE_SHA256 = (
    "414f582950bf60fed2fe462cd141ab98d0f772087b4f9c6bc5aa12f03f379e7d"
)
ZERO_ROOT_WEIGHTS_SHA256 = (
    "6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932"
)
# The manifest's composite pins must equal the eval runner's frozen
# constants — the lineage package reproduces exactly what this cell measures
# (count_walk is the parent eval arm; replay_ctl7 is carried for the
# complete stage-7 record).
ARM_COMPOSITE_PINS = {
    "replay_ctl7": {
        "tree_sha256": (
            "044a4599ac5264e00256f66f65215ea497d3631d8aebd3467b698253648e484a"
        ),
        "weights_sha256": (
            "c5035b4db47e4da582a805ca009747a5618ef5badc35d960ca216e586dd3ab9d"
        ),
    },
    "count_walk": {
        "tree_sha256": (
            "d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1"
        ),
        "weights_sha256": (
            "ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3"
        ),
    },
}
ARM_HYPERPARAMETERS = {
    "alpha": 64,
    "batch_size": 1,
    "epochs": 1.0,
    "grad_accum": 8,
    "lr": 1e-05,
    "max_length": 4096,
    "rank": 32,
    "w_close": 0.2,
    "w_think": 0.2,
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _sha_or_null(value: object) -> bool:
    return value is None or _sha(value)


def stage_dirname(row: dict) -> str:
    return f"stage{row['stage']:02d}_{row['name']}"


def load_manifest() -> dict:
    """Load and authenticate the extended clean-chain lineage manifest."""
    if not MANIFEST.is_file():
        raise ValueError(f"clean-chain lineage manifest is absent: {MANIFEST}")
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise ValueError(f"clean-chain lineage manifest changed: {MANIFEST}")
    text = MANIFEST.read_text(encoding="utf-8")
    if "root_adapter" in text:
        raise ValueError(
            "the clean-chain manifest must not reference any root adapter"
        )
    manifest = json.loads(text)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != MANIFEST_KEYS
        or manifest.get("schema_version") != 1
        or manifest.get("experiment_id") != "qwen35_4b_count_dont_walk_enumeration"
        or manifest.get("stage") != "clean_chain_lineage_manifest"
        or manifest.get("framing") != "clean_chain"
        or manifest.get("model", {}).get("id") != MODEL_ID
        or manifest.get("model", {}).get("revision") != MODEL_REVISION
    ):
        raise ValueError("clean-chain lineage manifest failed schema authentication")
    stages = manifest.get("stages")
    if not isinstance(stages, list) or len(stages) != STAGE_COUNT:
        raise ValueError("clean-chain manifest must carry exactly seven stages")
    for index, row in enumerate(stages, start=1):
        keys = set(row) - {"targeted_close_overrides"}
        expected_keys = STAGE_KEYS | (STAGE7_EXTRA_KEYS if index == 7 else set())
        if (
            keys != expected_keys
            or row.get("stage") != index
            or not _sha(row.get("trainer_sha256"))
            or not isinstance(row.get("seed"), int)
            or row.get("seed") != STAGE_SEEDS[index - 1]
            or not _sha(row.get("dataset", {}).get("sha256"))
            or not isinstance(row.get("dataset", {}).get("rows"), int)
            or not set(row.get("hyperparameters", {})) <= HYPER_KEYS | {"w_close"}
            or not set(row.get("hyperparameters", {})) >= HYPER_KEYS
        ):
            raise ValueError(f"clean-chain manifest stage {index} failed authentication")
        if index <= ZERO_ROOT_STAGES:
            produced = row.get("produced_zero_root", {})
            if not _sha(produced.get("adapter_weights_sha256")) or not _sha(
                produced.get("adapter_config_sha256")
            ):
                raise ValueError(
                    f"clean-chain manifest stage {index} lacks zero-root "
                    "verification aids"
                )
        expected_warm = (
            FRESH_INIT
            if index == 1
            else (STAGE7_WARM if index == 7 else f"stage {index - 1}")
        )
        if row.get("warm_start") != expected_warm:
            raise ValueError(
                f"clean-chain manifest stage {index} breaks the recorded chain: "
                f"{row.get('warm_start')!r} != {expected_warm!r}"
            )
        trainer = row.get("trainer", "")
        trainers = manifest.get("trainers", {})
        if trainer not in trainers or trainers[trainer].get("sha256") != row["trainer_sha256"]:
            raise ValueError(
                f"clean-chain manifest stage {index} names an unregistered trainer"
            )
        if index not in trainers[trainer].get("stages", []):
            raise ValueError(
                f"clean-chain manifest stage {index} disagrees with the trainer's "
                "stage registration"
            )
    stage7 = stages[6]
    base = stage7.get("stage7_base", {})
    if (
        base.get("tree_sha256") != ZERO_ROOT_TREE_SHA256
        or base.get("weights_sha256") != ZERO_ROOT_WEIGHTS_SHA256
    ):
        raise ValueError("clean-chain manifest stage-7 base disagrees with the pins")
    merger = manifest.get("merger", {})
    if merger.get("path") != "scripts/merge_adapter.py" or not _sha(merger.get("sha256")):
        raise ValueError("clean-chain manifest merger block failed authentication")
    chain = manifest.get("clean_chain", {})
    zero_root = chain.get("zero_root_composite", {})
    if (
        zero_root.get("output_tree_sha256") != ZERO_ROOT_TREE_SHA256
        or zero_root.get("weights_sha256") != ZERO_ROOT_WEIGHTS_SHA256
        or not isinstance(chain.get("provenance_receipts"), dict)
        or len(chain["provenance_receipts"]) != ZERO_ROOT_STAGES + 1
    ):
        raise ValueError("clean-chain manifest zero-root block failed authentication")
    validate_confirmation_arms(manifest, stage7)
    validate_stage8(manifest, stage7)
    return manifest


def validate_confirmation_arms(manifest: dict, stage7: dict) -> None:
    """The lifecycle-28 extension: both arms, one recipe, frozen pins."""
    arms_block = manifest.get("stage7_confirmation_arms", {})
    recipe = arms_block.get("recipe", {})
    trainer = arms_block.get("trainer", {})
    merger = arms_block.get("merger", {})
    if (
        arms_block.get("extended_by") != "qwen35_4b_count_walk_menders_confirmation"
        or arms_block.get("training_seed") != TRAINING_SEED
        or recipe.get("hyperparameters") != ARM_HYPERPARAMETERS
        or list(recipe.get("order", ())) != list(ARM_ORDER)
        or recipe.get("warm_start") != STAGE7_WARM
        or trainer.get("path") != "scripts/train_think.py"
        or trainer.get("sha256") != stage7.get("trainer_sha256")
        or merger.get("path") != "scripts/merge_adapter.py"
        or merger.get("sha256") != manifest["merger"].get("sha256")
    ):
        raise ValueError(
            "stage7_confirmation_arms block failed recipe/trainer/merger "
            "authentication"
        )
    arms = arms_block.get("arms", {})
    if set(arms) != set(ARM_ORDER):
        raise ValueError("stage7_confirmation_arms must carry exactly the two arms")
    for name in ARM_ORDER:
        arm = arms[name]
        stream = arm.get("stream", {})
        composite = arm.get("composite", {})
        if (
            not _sha(stream.get("sha256"))
            or not isinstance(stream.get("rows"), int)
            or composite.get("tree_sha256")
            != ARM_COMPOSITE_PINS[name]["tree_sha256"]
            or composite.get("weights_sha256")
            != ARM_COMPOSITE_PINS[name]["weights_sha256"]
            or not _sha(composite.get("committed_merge_receipt_sha256"))
        ):
            raise ValueError(
                f"stage7_confirmation_arms arm {name} disagrees with the frozen "
                "eval pins"
            )
    # The count_walk arm stream IS the stage-7 dataset of the base manifest.
    if (
        arms["count_walk"]["stream"]["sha256"] != stage7["dataset"]["sha256"]
        or arms["count_walk"]["stream"]["rows"] != stage7["dataset"]["rows"]
    ):
        raise ValueError(
            "stage7_confirmation_arms count_walk stream diverges from the "
            "manifest's stage-7 dataset"
        )
    for section in ("production_wrappers", "documentation_copies"):
        for relative, block in sorted(arms_block.get(section, {}).items()):
            if not _sha(block.get("sha256")):
                raise ValueError(f"{section} entry lacks a sha pin: {relative}")
    if not _sha(
        arms_block.get("stream_token_receipt", {}).get("sha256")
    ) or not all(
        _sha(block.get("sha256"))
        for block in arms_block.get("materialization_inputs", {}).values()
    ):
        raise ValueError(
            "stage7_confirmation_arms token receipt / materialization inputs "
            "lack sha pins"
        )


def validate_stage8(manifest: dict, stage7: dict) -> None:
    """This cell's extension: one arm, the replay pool, the count_walk parent."""
    block = manifest.get("stage8_replay_compound", {})
    dataset = block.get("dataset", {})
    parent = block.get("parent_composite", {})
    recipe = block.get("recipe", {})
    trainer = block.get("trainer", {})
    merger = block.get("merger", {})
    composite = block.get("composite", {})
    deferred = block.get("production_wrappers_pins_deferred", {})
    if (
        block.get("extended_by") != EXP.name
        or block.get("arm") != STAGE8_ARM
        or block.get("training_seed") != STAGE8_SEED
        or dataset.get("file") != "data/sft_blend.jsonl"
        or dataset.get("sha256") != STAGE8_DATASET_SHA256
        or dataset.get("rows") != STAGE8_DATASET_ROWS
        or parent.get("tree_sha256") != ARM_COMPOSITE_PINS["count_walk"]["tree_sha256"]
        or parent.get("weights_sha256")
        != ARM_COMPOSITE_PINS["count_walk"]["weights_sha256"]
        or not _sha(parent.get("committed_merge_receipt_sha256"))
        or parent.get("provenance_copy") != "data/provenance/count_walk_merge.json"
        or recipe.get("hyperparameters") != ARM_HYPERPARAMETERS
        or recipe.get("optimizer_steps") != 280
        or recipe.get("warm_start") != STAGE8_WARM
        or trainer.get("path") != "scripts/train_think.py"
        or trainer.get("sha256") != stage7.get("trainer_sha256")
        or merger.get("path") != "scripts/merge_adapter.py"
        or merger.get("sha256") != manifest["merger"].get("sha256")
        or sorted(deferred.get("files", []))
        != ["scripts/merge_trained_arm.py", "scripts/train_trial.py"]
        or not _sha_or_null(composite.get("tree_sha256"))
        or not _sha_or_null(composite.get("weights_sha256"))
        or not _sha_or_null(composite.get("committed_merge_receipt_sha256"))
    ):
        raise ValueError(
            "stage8_replay_compound block failed recipe/parent/pin authentication"
        )


def require_root_not_vendored() -> None:
    """The clean chain is the design; fail closed on any vendored root."""
    for path in FORBIDDEN_ROOT_DIRS:
        if path.exists():
            raise ValueError(
                "no root adapter may ever be vendored into this cell (the "
                f"clean chain is the point): {path}"
            )


def verify_provenance_receipts(manifest: dict) -> dict:
    """Lifecycle 22's receipts, carried in-cell under sha256 pins.

    The IN-CELL copies are the hard fail-closed gate (sha-pinned by the
    manifest); the committed lifecycle-22 sibling originals are
    VERIFICATION AIDS only — byte-identical when present (divergence fails
    loudly as tamper evidence), skipped with a recorded note when absent.
    The reproduction path is this cell's own copies plus this script,
    never the sibling.
    """
    checked = 0
    siblings = {"present": 0, "absent": 0}
    for name, digest in sorted(
        manifest["clean_chain"]["provenance_receipts"].items()
    ):
        copy = PROVENANCE_DIR / name
        if not _sha(digest) or not copy.is_file() or sha256_file(copy) != digest:
            raise ValueError(
                f"in-cell provenance receipt copy is absent or changed: {name}"
            )
        original = SOURCE_ZERO_ROOT_EXP / "runs" / "lineage" / name
        if original.is_file():
            if copy.read_bytes() != original.read_bytes():
                raise ValueError(
                    "committed lifecycle-22 sibling original diverged from "
                    f"the in-cell provenance pin: {name}"
                )
            siblings["present"] += 1
        else:
            siblings["absent"] += 1
        checked += 1
    return {
        "checked": checked,
        "sibling_originals": {
            **siblings,
            "note": (
                "sibling originals are verification aids: byte-identical "
                "when present, absent -> in-cell pin authoritative"
            ),
        },
    }


def _verify_pinned_file(path: Path, expected_sha: str, rows: int | None = None) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f"lineage package file is absent or changed: {path}")
    if rows is not None:
        observed = sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        if observed != rows:
            raise ValueError(
                f"lineage dataset row count changed: {path} has {observed}, "
                f"manifest says {rows}"
            )


def verify_inputs(manifest: dict) -> dict:
    """Fast, model-free package authentication (wired into run.py --smoke)."""
    require_root_not_vendored()
    checked = {
        "datasets": 0,
        "arm_streams": 0,
        "stage8_dataset": 0,
        "materialization_inputs": 0,
        "token_receipt": 0,
        "trainers": 0,
        "merger": 0,
        "wrappers": 0,
        "documentation_copies": 0,
        "provenance_receipts": 0,
        "root_vendored": False,
    }
    for row in manifest["stages"]:
        _verify_pinned_file(
            EXP / row["dataset"]["file"],
            row["dataset"]["sha256"],
            row["dataset"]["rows"],
        )
        checked["datasets"] += 1
    arms_block = manifest["stage7_confirmation_arms"]
    for name in ARM_ORDER:
        stream = arms_block["arms"][name]["stream"]
        _verify_pinned_file(EXP / stream["file"], stream["sha256"], stream["rows"])
        checked["arm_streams"] += 1
    stage8 = manifest["stage8_replay_compound"]
    _verify_pinned_file(
        EXP / stage8["dataset"]["file"],
        stage8["dataset"]["sha256"],
        stage8["dataset"]["rows"],
    )
    checked["stage8_dataset"] += 1
    parent_copy = EXP / stage8["parent_composite"]["provenance_copy"]
    if (
        not parent_copy.is_file()
        or sha256_file(parent_copy)
        != stage8["parent_composite"]["committed_merge_receipt_sha256"]
    ):
        raise ValueError(
            f"stage-8 parent provenance copy is absent or changed: {parent_copy}"
        )
    for relative, block in sorted(arms_block["materialization_inputs"].items()):
        _verify_pinned_file(EXP / relative, block["sha256"])
        checked["materialization_inputs"] += 1
    receipt = arms_block["stream_token_receipt"]
    _verify_pinned_file(EXP / receipt["file"], receipt["sha256"])
    checked["token_receipt"] += 1
    for relative, block in sorted(manifest["trainers"].items()):
        _verify_pinned_file(EXP / relative, block["sha256"])
        checked["trainers"] += 1
    _verify_pinned_file(
        EXP / manifest["merger"]["path"], manifest["merger"]["sha256"]
    )
    checked["merger"] += 1
    for relative, block in sorted(arms_block["production_wrappers"].items()):
        _verify_pinned_file(EXP / relative, block["sha256"])
        checked["wrappers"] += 1
    for relative in sorted(stage8["production_wrappers_pins_deferred"]["files"]):
        if not (EXP / relative).is_file():
            raise ValueError(f"stage-8 production wrapper is absent: {relative}")
    for relative, block in sorted(arms_block["documentation_copies"].items()):
        _verify_pinned_file(EXP / relative, block["sha256"])
        checked["documentation_copies"] += 1
    provenance = verify_provenance_receipts(manifest)
    checked["provenance_receipts"] = provenance["checked"]
    checked["provenance_sibling_originals"] = provenance["sibling_originals"]
    return checked


def stage_command(row: dict, dataset: Path, out_dir: Path, warm_start: Path | None) -> list[str]:
    """The exact recorded per-stage recipe for zero-root stages 1-6."""
    hypers = row["hyperparameters"]
    command = [
        sys.executable, "-B", str(EXP / row["trainer"]),
        "--train", str(dataset),
        "--out", str(out_dir),
        "--epochs", str(hypers["epochs"]),
        "--lr", str(hypers["lr"]),
        "--rank", str(hypers["rank"]),
        "--alpha", str(hypers["alpha"]),
        "--batch-size", str(hypers["batch_size"]),
        "--grad-accum", str(hypers["grad_accum"]),
        "--max-length", str(hypers["max_length"]),
    ]
    if warm_start is not None:
        command.extend(("--warm" + "-start", str(warm_start)))
    command.extend(("--w-think", str(hypers["w_think"])))
    command.extend(("--seed", str(row["seed"])))
    if "w_close" in hypers:
        command.extend(("--w-close", str(hypers["w_close"])))
    targeted = row.get("targeted_close_overrides")
    if targeted is not None:
        for kind in targeted["target_close_kinds"]:
            command.extend(("--target-close-kind", kind))
        command.extend(("--target-w-close", str(targeted["target_w_close"])))
    return command


def composite_arm_command(
    hypers: dict, stream: Path, out_dir: Path, base_composite: Path, seed: int
) -> list[str]:
    """The frozen composite-parent arm recipe (stage 7 and stage 8 alike).

    A FRESH rank-32/alpha-64 adapter on a rebuilt merged composite via
    ``--model-path`` at the block's fixed seed.
    """
    return [
        sys.executable, "-B", str(EXP / "scripts" / "train_think.py"),
        "--train", str(stream),
        "--out", str(out_dir),
        "--epochs", str(hypers["epochs"]),
        "--lr", str(hypers["lr"]),
        "--rank", str(hypers["rank"]),
        "--alpha", str(hypers["alpha"]),
        "--batch-size", str(hypers["batch_size"]),
        "--grad-accum", str(hypers["grad_accum"]),
        "--max-length", str(hypers["max_length"]),
        "--model-path", str(base_composite),
        "--w-think", str(hypers["w_think"]),
        "--seed", str(seed),
        "--w-close", str(hypers["w_close"]),
    ]


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
    return hashlib.sha256(rendered).hexdigest()


def adapter_receipt_block(out_dir: Path) -> dict:
    weights = out_dir / "adapter_model.safetensors"
    config = out_dir / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise ValueError(f"stage output is incomplete: {out_dir}")
    return {
        "adapter_config_sha256": sha256_file(config),
        "adapter_weights_sha256": sha256_file(weights),
        "adapter_weights_size": weights.stat().st_size,
    }


def receipt_matches_disk(receipt: dict, out_dir: Path) -> bool:
    weights = out_dir / "adapter_model.safetensors"
    config = out_dir / "adapter_config.json"
    produced = receipt.get("produced", {})
    return (
        weights.is_file()
        and config.is_file()
        and sha256_file(weights) == produced.get("adapter_weights_sha256")
        and sha256_file(config) == produced.get("adapter_config_sha256")
    )


def write_receipt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_stage(
    row: dict, out_dir: Path, receipt_path: Path, warm_start: Path | None
) -> None:
    dataset = EXP / row["dataset"]["file"]
    label = f"stage {row['stage']} ({row['name']})"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not receipt_matches_disk(receipt, out_dir):
            raise ValueError(
                f"{label} receipt exists but its adapter does not verify on "
                f"disk: {out_dir}; audit and delete both explicitly before "
                "rebuilding"
            )
        print(f"[lineage] {label}: receipt verified, skipping", flush=True)
        return
    if out_dir.exists():
        raise ValueError(
            f"{label} output exists without a receipt: {out_dir}; delete it "
            "explicitly before rebuilding"
        )
    command = stage_command(row, dataset, out_dir, warm_start)
    print(f"[lineage] {label}: training seed {row['seed']}", flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=ROOT, check=True)
    wall_seconds = time.monotonic() - started
    warm_block: object
    if warm_start is None:
        warm_block = FRESH_INIT
    else:
        warm_block = {
            "source": f"stage {row['stage'] - 1}",
            "adapter": str(warm_start),
            "adapter_config_sha256": sha256_file(warm_start / "adapter_config.json"),
            "adapter_weights_sha256": sha256_file(
                warm_start / "adapter_model.safetensors"
            ),
        }
    write_receipt(receipt_path, {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": row["stage"],
        "name": row["name"],
        "lineage": "clean_chain",
        "manifest_sha256": MANIFEST_SHA256,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "dataset": dict(row["dataset"]),
        "trainer": {"path": row["trainer"], "sha256": row["trainer_sha256"]},
        "seed": row["seed"],
        "hyperparameters": dict(row["hyperparameters"]),
        "targeted_close_overrides": row.get("targeted_close_overrides"),
        "warm_start": warm_block,
        "fresh_init": row["stage"] == 1,
        "command": command,
        "adapter": str(out_dir),
        "produced": adapter_receipt_block(out_dir),
        "zero_root_verification_aid": row.get("produced_zero_root"),
        "wall_seconds": wall_seconds,
    })
    print(f"[lineage] {label}: receipt written", flush=True)


def merge_zero_root(manifest: dict, adapter: Path) -> None:
    """Merge the stage-6 adapter onto the raw HF base; must reproduce the pins."""
    if ZERO_ROOT_MERGE_RECEIPT.is_file():
        receipt = json.loads(ZERO_ROOT_MERGE_RECEIPT.read_text(encoding="utf-8"))
        tree = merged_tree_manifest(ZERO_ROOT_MERGED_OUT)
        if receipt.get("output_tree_sha256") != tree_manifest_sha256(tree):
            raise ValueError(
                "published zero-root rebuild merge receipt does not match disk"
            )
        print("[lineage] zero-root merge: receipt verified, skipping", flush=True)
        return
    if ZERO_ROOT_MERGED_OUT.exists():
        raise ValueError(
            f"merged output exists without a receipt: {ZERO_ROOT_MERGED_OUT}"
        )
    started = time.monotonic()
    subprocess.run(
        [
            sys.executable, "-B", str(EXP / manifest["merger"]["path"]),
            "--adapter", str(adapter),
            "--out", str(ZERO_ROOT_MERGED_OUT),
        ],
        cwd=ROOT, check=True,
    )
    wall_seconds = time.monotonic() - started
    tree = merged_tree_manifest(ZERO_ROOT_MERGED_OUT)
    files = {row["name"]: row for row in tree}
    if files["model.safetensors"]["sha256"] != ZERO_ROOT_WEIGHTS_SHA256:
        raise ValueError(
            "rebuilt zero-root composite weights do not reproduce lifecycle "
            "22's pinned bytes; on a different stack equivalence must be "
            "re-established behaviorally through the trusted gateway"
        )
    write_receipt(ZERO_ROOT_MERGE_RECEIPT, {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "merge_zero_root",
        "name": "zero_root_hygiene_explore",
        "lineage": "clean_chain",
        "manifest_sha256": MANIFEST_SHA256,
        "merger": dict(manifest["merger"]),
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "adapter": {
            "path": str(adapter),
            "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
            "adapter_weights_sha256": sha256_file(
                adapter / "adapter_model.safetensors"
            ),
        },
        "merged": str(ZERO_ROOT_MERGED_OUT),
        "output_files": tree,
        "output_tree_sha256": tree_manifest_sha256(tree),
        "weights_sha256": files["model.safetensors"]["sha256"],
        "weights_size_bytes": files["model.safetensors"]["size"],
        "reproduces_lifecycle22_weights": True,
        "wall_seconds": wall_seconds,
    })


def rebuild_composite_arm(
    manifest: dict,
    *,
    name: str,
    stage_label: str,
    stream: Path,
    dataset_block: dict,
    base_composite: Path,
    base_weights_sha256: str,
    seed: int,
    warm: str,
    pins: dict | None,
    stage_number: int,
) -> None:
    """Train one composite-parent arm (stage 7 or stage 8), then merge."""
    out_dir = ADAPTER_ROOT / f"stage{stage_number:02d}_{name}"
    receipt_path = LINEAGE_RECEIPTS / f"stage{stage_number:02d}_{name}.json"
    merged_out = REBUILD_ROOT / "merged" / name
    merge_receipt_path = LINEAGE_RECEIPTS / f"merge_{name}.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not receipt_matches_disk(receipt, out_dir):
            raise ValueError(
                f"{stage_label} receipt exists but its adapter does not verify "
                f"on disk: {out_dir}; audit and delete both explicitly before "
                "rebuilding"
            )
        print(f"[lineage] {stage_label}: receipt verified, skipping", flush=True)
    else:
        if out_dir.exists():
            raise ValueError(
                f"{stage_label} output exists without a receipt: {out_dir}; "
                "delete it explicitly before rebuilding"
            )
        command = composite_arm_command(
            ARM_HYPERPARAMETERS, stream, out_dir, base_composite, seed
        )
        print(f"[lineage] {stage_label}: training seed {seed}", flush=True)
        started = time.monotonic()
        subprocess.run(command, cwd=ROOT, check=True)
        wall_seconds = time.monotonic() - started
        write_receipt(receipt_path, {
            "schema_version": 1,
            "experiment_id": EXP.name,
            "stage": stage_number,
            "name": name,
            "lineage": "clean_chain",
            "manifest_sha256": MANIFEST_SHA256,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "dataset": dict(dataset_block),
            "trainer": {
                "path": "scripts/train_think.py",
                "sha256": sha256_file(EXP / "scripts" / "train_think.py"),
            },
            "seed": seed,
            "hyperparameters": dict(ARM_HYPERPARAMETERS),
            "warm_start": warm,
            "fresh_init": True,
            "command": command,
            "adapter": str(out_dir),
            "produced": adapter_receipt_block(out_dir),
            "wall_seconds": wall_seconds,
        })
        print(f"[lineage] {stage_label}: receipt written", flush=True)
    if merge_receipt_path.is_file():
        receipt = json.loads(merge_receipt_path.read_text(encoding="utf-8"))
        tree = merged_tree_manifest(merged_out)
        if receipt.get("output_tree_sha256") != tree_manifest_sha256(tree):
            raise ValueError(
                f"published {name} rebuild merge receipt does not match disk"
            )
        print(f"[lineage] {name} merge: receipt verified, skipping", flush=True)
        return
    if merged_out.exists():
        raise ValueError(f"merged output exists without a receipt: {merged_out}")
    started = time.monotonic()
    subprocess.run(
        [
            sys.executable, "-B", str(EXP / manifest["merger"]["path"]),
            "--adapter", str(out_dir),
            "--out", str(merged_out),
            "--base-model", str(base_composite),
        ],
        cwd=ROOT, check=True,
    )
    wall_seconds = time.monotonic() - started
    tree = merged_tree_manifest(merged_out)
    files = {row["name"]: row for row in tree}
    receipt_payload = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": f"merge_{name}",
        "name": name,
        "lineage": "clean_chain",
        "manifest_sha256": MANIFEST_SHA256,
        "merger": dict(manifest["merger"]),
        "base_composite": {
            "path": str(base_composite),
            "weights_sha256": base_weights_sha256,
        },
        "adapter": {
            "path": str(out_dir),
            "adapter_config_sha256": sha256_file(out_dir / "adapter_config.json"),
            "adapter_weights_sha256": sha256_file(
                out_dir / "adapter_model.safetensors"
            ),
        },
        "merged": str(merged_out),
        "output_files": tree,
        "output_tree_sha256": tree_manifest_sha256(tree),
        "weights_sha256": files["model.safetensors"]["sha256"],
        "weights_size_bytes": files["model.safetensors"]["size"],
        "wall_seconds": wall_seconds,
    }
    if pins is not None and pins.get("tree_sha256") is not None:
        receipt_payload["published_pins"] = dict(pins)
        # Verification aid (recorded, never gated here): same repo stack
        # should reproduce byte-identically; a different stack must
        # re-establish equivalence behaviorally through the gateway.
        receipt_payload["matches_published_tree"] = (
            tree_manifest_sha256(tree) == pins["tree_sha256"]
        )
        receipt_payload["matches_published_weights"] = (
            files["model.safetensors"]["sha256"] == pins["weights_sha256"]
        )
    else:
        receipt_payload["published_pins"] = None
        receipt_payload["published_pins_note"] = (
            "the published composite pins were still TODO-PIN (null) at "
            "rebuild time; fill them from the committed merge receipt and "
            "compare out-of-band"
        )
    write_receipt(merge_receipt_path, receipt_payload)


def rebuild(manifest: dict) -> None:
    """GPU path: six zero-root replays, the merge, both stage-7 arms, stage 8."""
    require_root_not_vendored()
    warm_start: Path | None = None
    stage6_adapter: Path | None = None
    for row in manifest["stages"][:ZERO_ROOT_STAGES]:
        out_dir = ADAPTER_ROOT / stage_dirname(row)
        receipt_path = LINEAGE_RECEIPTS / f"{stage_dirname(row)}.json"
        run_stage(row, out_dir, receipt_path, warm_start)
        if row["stage"] == ZERO_ROOT_STAGES:
            stage6_adapter = out_dir
        warm_start = out_dir
    merge_zero_root(manifest, stage6_adapter)
    arms_block = manifest["stage7_confirmation_arms"]
    for name in ARM_ORDER:
        stream_block = arms_block["arms"][name]["stream"]
        rebuild_composite_arm(
            manifest,
            name=name,
            stage_label=f"stage 7 arm ({name})",
            stream=EXP / stream_block["file"],
            dataset_block=stream_block,
            base_composite=ZERO_ROOT_MERGED_OUT,
            base_weights_sha256=ZERO_ROOT_WEIGHTS_SHA256,
            seed=TRAINING_SEED,
            warm=STAGE7_WARM,
            pins=ARM_COMPOSITE_PINS[name],
            stage_number=7,
        )
    stage8 = manifest["stage8_replay_compound"]
    count_walk_rebuilt = REBUILD_ROOT / "merged" / "count_walk"
    rebuild_composite_arm(
        manifest,
        name=STAGE8_ARM,
        stage_label=f"stage 8 arm ({STAGE8_ARM})",
        stream=EXP / stage8["dataset"]["file"],
        dataset_block=stage8["dataset"],
        base_composite=count_walk_rebuilt,
        base_weights_sha256=ARM_COMPOSITE_PINS["count_walk"]["weights_sha256"],
        seed=STAGE8_SEED,
        warm=STAGE8_WARM,
        pins=stage8["composite"],
        stage_number=8,
    )
    print(
        json.dumps(
            {
                "zero_root_merge_receipt": str(ZERO_ROOT_MERGE_RECEIPT),
                "arm_merge_receipts": {
                    name: str(LINEAGE_RECEIPTS / f"merge_{name}.json")
                    for name in (*ARM_ORDER, STAGE8_ARM)
                },
                "rebuilt_composites": {
                    "zero_root_parent": str(ZERO_ROOT_MERGED_OUT),
                    **{
                        name: str(REBUILD_ROOT / "merged" / name)
                        for name in (*ARM_ORDER, STAGE8_ARM)
                    },
                },
            },
            indent=1,
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--verify-inputs",
        action="store_true",
        help="authenticate the lineage package only (fast, no GPU, no writes)",
    )
    args = parser.parse_args()
    try:
        manifest = load_manifest()
        checked = verify_inputs(manifest)
        if not args.verify_inputs:
            rebuild(manifest)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST),
                "manifest_sha256": MANIFEST_SHA256,
                "mode": "verify_inputs" if args.verify_inputs else "rebuild",
                "checked": checked,
                "root_adapter_vendored": False,
                "ok": True,
            },
            indent=1,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
