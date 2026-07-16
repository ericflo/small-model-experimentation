#!/usr/bin/env python3
"""Replay the ENTIRE clean chain from nothing: six zero-root stages + the dose.

Lifecycle 23 — the standalone reproduction path for the mission's
cleanest artifact. The final installed model's complete lineage is:

    stages 1-6: the six documented, contamination-free training stages
    of the hygiene_explore recipe, replayed from a FRESH zero-initialized
    rank-32/alpha-64 adapter on the raw pinned HF base (stage 1 has NO
    warm start; stages 2-6 warm-start from the previous stage), exactly
    as lifecycle 22 (``qwen35_4b_zero_root_lineage_rebuild``) recorded —
    the stage-6 adapter merges onto the raw HF base to produce the
    zero-root composite (tree 414f5829..., weights 6e9aad25...);

    stage 7: this cell's gym-mix dose — a FRESH rank-32/alpha-64
    adapter trained ON the zero-root composite via the trainer's
    ``--model-path`` over the exposure-matched ``gym_mix``
    stream (the FRESH 160-row three-kind treatment + replay), then
    merged onto that composite.

NO undocumented root exists anywhere in the chain and none may ever be
vendored into this cell: this script fails closed if a ``lineage_root``
directory appears under this cell's artifact storage. Lifecycle 22's six
stage receipts and merge receipt are carried byte-identically in
``data/lineage/provenance/`` as the provenance documents for stages 1-6;
this cell's own run receipts document stage 7.

Modes:

- ``--verify-inputs`` (fast, no GPU, wired into ``run.py --smoke``):
  authenticates the complete package — the clean-chain manifest bytes
  (sha256-pinned) and internal chaining, every copied dataset (sha256 +
  row count) including the stage-7 stream and the fresh treatment
  corpus, every trainer/merger copy (sha256), and the provenance
  receipts (byte-identical to lifecycle 22's committed originals) — and
  asserts no blend root is vendored. Writes nothing.
- default (GPU, ~3h): replays stages 1-7 into
  ``large_artifacts/<exp>/rebuild/`` with a receipt per stage under
  ``runs/lineage_rebuild/``. The PRIMARY experiment path does NOT run
  this: it trains on lifecycle 22's published composite via run.py's
  stage DAG. A stage whose receipt AND on-disk output both verify is
  skipped (crash-safe resume); an unreceipted or mismatching output
  fails closed. On this repo's pinned stack the stage 1-6 replay must
  reproduce the zero-root composite's weights byte-identically (the
  merge fails closed otherwise); on a different stack the recorded
  hashes are verification aids and equivalence must be re-established
  behaviorally through the trusted gateway.
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
# Byte pin of this cell's frozen clean-chain manifest.
MANIFEST_SHA256 = "e996b7f350b86d98964b1885461a19589f57a17eca4de940c39e4754748b39c7"
SOURCE_ZERO_ROOT_EXP = ROOT / "experiments" / "qwen35_4b_zero_root_lineage_rebuild"
PROVENANCE_DIR = EXP / "data" / "lineage" / "provenance"
LINEAGE_RECEIPTS = EXP / "runs" / "lineage_rebuild"
REBUILD_ROOT = ROOT / "large_artifacts" / EXP.name / "rebuild"
ADAPTER_ROOT = REBUILD_ROOT / "adapters"
ZERO_ROOT_MERGED_OUT = REBUILD_ROOT / "merged" / "zero_root_hygiene_explore"
FINAL_MERGED_OUT = REBUILD_ROOT / "merged" / "gym_mix"
ZERO_ROOT_MERGE_RECEIPT = LINEAGE_RECEIPTS / "merge_zero_root.json"
FINAL_MERGE_RECEIPT = LINEAGE_RECEIPTS / "merge_gym_mix.json"
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
SHA_RE = re.compile(r"[0-9a-f]{64}")
MANIFEST_KEYS = {
    "schema_version", "experiment_id", "stage", "framing", "directive", "model",
    "carrier", "base_reserialized_note", "clean_chain", "trainers",
    "merger", "stages", "final_merge", "determinism", "verification",
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
# documented recipe; 79 is this cell's fresh paired-trial training seed.
STAGE_SEEDS = (42, 43, 44, 47, 51, 55, 79)
# The zero-root composite this chain must reproduce before stage 7 lands
# (fail closed on this stack; verification aid on any other).
ZERO_ROOT_TREE_SHA256 = (
    "414f582950bf60fed2fe462cd141ab98d0f772087b4f9c6bc5aa12f03f379e7d"
)
ZERO_ROOT_WEIGHTS_SHA256 = (
    "6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932"
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def stage_dirname(row: dict) -> str:
    return f"stage{row['stage']:02d}_{row['name']}"


def load_manifest() -> dict:
    """Load and authenticate the frozen clean-chain lineage manifest."""
    if MANIFEST_SHA256 is None:
        raise ValueError("clean-chain manifest pin is unfilled (TODO-PIN)")
    if not MANIFEST.is_file():
        raise ValueError(f"clean-chain lineage manifest is absent: {MANIFEST}")
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise ValueError(
            f"clean-chain lineage manifest changed: {MANIFEST}"
        )
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
        or manifest.get("experiment_id") != EXP.name
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
    final = manifest.get("final_merge", {})
    if merger["sha256"] != final.get("merger_sha256"):
        raise ValueError("clean-chain manifest final merge block failed authentication")
    chain = manifest.get("clean_chain", {})
    zero_root = chain.get("zero_root_composite", {})
    if (
        zero_root.get("output_tree_sha256") != ZERO_ROOT_TREE_SHA256
        or zero_root.get("weights_sha256") != ZERO_ROOT_WEIGHTS_SHA256
        or not isinstance(chain.get("provenance_receipts"), dict)
        or len(chain["provenance_receipts"]) != ZERO_ROOT_STAGES + 1
    ):
        raise ValueError("clean-chain manifest zero-root block failed authentication")
    return manifest


def require_root_not_vendored() -> None:
    """The clean chain is the design; fail closed on any vendored root."""
    for path in FORBIDDEN_ROOT_DIRS:
        if path.exists():
            raise ValueError(
                "no root adapter may ever be vendored into this cell (the "
                f"clean chain is the point): {path}"
            )


def verify_provenance_receipts(manifest: dict) -> int:
    """Lifecycle 22's committed receipts, carried byte-identically here."""
    checked = 0
    for name, digest in sorted(
        manifest["clean_chain"]["provenance_receipts"].items()
    ):
        copy = PROVENANCE_DIR / name
        original = SOURCE_ZERO_ROOT_EXP / "runs" / "lineage" / name
        if (
            not _sha(digest)
            or not copy.is_file()
            or sha256_file(copy) != digest
            or not original.is_file()
            or copy.read_bytes() != original.read_bytes()
        ):
            raise ValueError(
                "provenance receipt copy is absent, changed, or diverged from "
                f"the committed lifecycle-22 original: {name}"
            )
        checked += 1
    return checked


def verify_inputs(manifest: dict) -> dict:
    """Fast, model-free package authentication (wired into run.py --smoke)."""
    require_root_not_vendored()
    checked = {
        "datasets": 0,
        "trainers": 0,
        "merger": 0,
        "provenance_receipts": 0,
        "root_vendored": False,
    }
    for row in manifest["stages"]:
        dataset = EXP / row["dataset"]["file"]
        if not dataset.is_file() or sha256_file(dataset) != row["dataset"]["sha256"]:
            raise ValueError(f"lineage dataset is absent or changed: {dataset}")
        rows = sum(
            1 for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        if rows != row["dataset"]["rows"]:
            raise ValueError(
                f"lineage dataset row count changed: {dataset} has {rows}, "
                f"manifest says {row['dataset']['rows']}"
            )
        checked["datasets"] += 1
    for relative, block in sorted(manifest["trainers"].items()):
        trainer = EXP / relative
        if not trainer.is_file() or sha256_file(trainer) != block["sha256"]:
            raise ValueError(f"lineage trainer copy is absent or changed: {trainer}")
        checked["trainers"] += 1
    merger = EXP / manifest["merger"]["path"]
    if not merger.is_file() or sha256_file(merger) != manifest["merger"]["sha256"]:
        raise ValueError(f"lineage merger copy is absent or changed: {merger}")
    checked["merger"] += 1
    # The FRESH gym-mix treatment corpus feeding the stage-7 stream.
    treatment = EXP / "data" / "sft_gym_mix.jsonl"
    if (
        not treatment.is_file()
        or sha256_file(treatment)
        != "6295011622096992e889b58a1a004fee26f4f9787bd952d348c0bf8593564a89"
    ):
        raise ValueError(f"gym-mix treatment corpus is absent or changed: {treatment}")
    checked["treatment_corpus"] = 1
    checked["provenance_receipts"] = verify_provenance_receipts(manifest)
    return checked


def stage_command(row: dict, dataset: Path, out_dir: Path, warm_start: Path | None) -> list[str]:
    """The exact recorded per-stage recipe.

    Stages 1-6 are byte-for-byte lifecycle 22's command construction:
    stage 1 omits ``--warm-start`` entirely (the trainer's default fresh
    path builds a new rank-32/alpha-64 LoRA whose B matrices are
    zero-initialized) and stages 2-6 receive the previous stage's adapter.
    Stage 7 uses this cell's trainer with ``--model-path`` pointing at the
    rebuilt zero-root composite (fresh adapter; never a warm start).
    """
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
    if row["stage"] == 7:
        command.extend(("--model-path", str(ZERO_ROOT_MERGED_OUT)))
    elif warm_start is not None:
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


def build_stage_receipt(
    row: dict,
    dataset: Path,
    out_dir: Path,
    warm_start: Path | None,
    command: list[str],
    wall_seconds: float,
) -> dict:
    weights = out_dir / "adapter_model.safetensors"
    config = out_dir / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise ValueError(f"stage output is incomplete: {out_dir}")
    if row["stage"] == 7:
        warm_block = STAGE7_WARM
    elif warm_start is None:
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
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": row["stage"],
        "name": row["name"],
        "lineage": "clean_chain",
        "manifest_sha256": MANIFEST_SHA256,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "dataset": {
            "file": row["dataset"]["file"],
            "sha256": row["dataset"]["sha256"],
            "rows": row["dataset"]["rows"],
        },
        "trainer": {"path": row["trainer"], "sha256": row["trainer_sha256"]},
        "seed": row["seed"],
        "hyperparameters": dict(row["hyperparameters"]),
        "targeted_close_overrides": row.get("targeted_close_overrides"),
        "warm_start": warm_block,
        "fresh_init": row["stage"] in (1, 7),
        "command": command,
        "adapter": str(out_dir),
        "produced": {
            "adapter_config_sha256": sha256_file(config),
            "adapter_weights_sha256": sha256_file(weights),
            "adapter_weights_size": weights.stat().st_size,
        },
        "zero_root_verification_aid": row.get("produced_zero_root"),
        "wall_seconds": wall_seconds,
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


def merge_zero_root(manifest: dict, adapter: Path) -> None:
    """Merge the stage-6 adapter onto the raw HF base; must reproduce the pins."""
    if ZERO_ROOT_MERGE_RECEIPT.is_file():
        receipt = json.loads(ZERO_ROOT_MERGE_RECEIPT.read_text(encoding="utf-8"))
        tree = merged_tree_manifest(ZERO_ROOT_MERGED_OUT)
        if receipt.get("output_tree_sha256") != tree_manifest_sha256(tree):
            raise ValueError(
                "published zero-root rebuild merge receipt does not match disk"
            )
        print("[clean-chain] zero-root merge: receipt verified, skipping", flush=True)
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


def merge_stage7(manifest: dict, adapter: Path) -> None:
    """Merge the stage-7 dose adapter onto the rebuilt zero-root composite."""
    if FINAL_MERGE_RECEIPT.is_file():
        receipt = json.loads(FINAL_MERGE_RECEIPT.read_text(encoding="utf-8"))
        tree = merged_tree_manifest(FINAL_MERGED_OUT)
        if receipt.get("output_tree_sha256") != tree_manifest_sha256(tree):
            raise ValueError(
                "published stage-7 rebuild merge receipt does not match disk"
            )
        print("[clean-chain] stage-7 merge: receipt verified, skipping", flush=True)
        return
    if FINAL_MERGED_OUT.exists():
        raise ValueError(
            f"merged output exists without a receipt: {FINAL_MERGED_OUT}"
        )
    started = time.monotonic()
    subprocess.run(
        [
            sys.executable, "-B", str(EXP / manifest["merger"]["path"]),
            "--adapter", str(adapter),
            "--out", str(FINAL_MERGED_OUT),
            "--base-model", str(ZERO_ROOT_MERGED_OUT),
        ],
        cwd=ROOT, check=True,
    )
    wall_seconds = time.monotonic() - started
    tree = merged_tree_manifest(FINAL_MERGED_OUT)
    files = {row["name"]: row for row in tree}
    write_receipt(FINAL_MERGE_RECEIPT, {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "merge_gym_mix",
        "name": "gym_mix",
        "lineage": "clean_chain",
        "manifest_sha256": MANIFEST_SHA256,
        "merger": dict(manifest["merger"]),
        "base_composite": {
            "path": str(ZERO_ROOT_MERGED_OUT),
            "weights_sha256": ZERO_ROOT_WEIGHTS_SHA256,
        },
        "adapter": {
            "path": str(adapter),
            "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
            "adapter_weights_sha256": sha256_file(
                adapter / "adapter_model.safetensors"
            ),
        },
        "merged": str(FINAL_MERGED_OUT),
        "output_files": tree,
        "output_tree_sha256": tree_manifest_sha256(tree),
        "weights_sha256": files["model.safetensors"]["sha256"],
        "weights_size_bytes": files["model.safetensors"]["size"],
        "wall_seconds": wall_seconds,
    })


def rebuild(manifest: dict) -> None:
    """GPU path: six zero-root stage replays, the zero-root merge, then stage 7."""
    require_root_not_vendored()
    warm_start: Path | None = None
    stage6_adapter: Path | None = None
    stage7_adapter: Path | None = None
    for row in manifest["stages"]:
        out_dir = ADAPTER_ROOT / stage_dirname(row)
        receipt_path = LINEAGE_RECEIPTS / f"{stage_dirname(row)}.json"
        dataset = EXP / row["dataset"]["file"]
        label = f"stage {row['stage']} ({row['name']})"
        if row["stage"] == 7:
            # Stage 7 trains ON the rebuilt zero-root composite.
            merge_zero_root(manifest, stage6_adapter)
            warm_start = None
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not receipt_matches_disk(receipt, out_dir):
                raise ValueError(
                    f"{label} receipt exists but its adapter does not verify "
                    f"on disk: {out_dir}; audit and delete both explicitly "
                    "before rebuilding"
                )
            print(f"[clean-chain] {label}: receipt verified, skipping", flush=True)
        else:
            if out_dir.exists():
                raise ValueError(
                    f"{label} output exists without a receipt: {out_dir}; "
                    "delete it explicitly before rebuilding"
                )
            command = stage_command(row, dataset, out_dir, warm_start)
            print(f"[clean-chain] {label}: training seed {row['seed']}", flush=True)
            started = time.monotonic()
            subprocess.run(command, cwd=ROOT, check=True)
            wall_seconds = time.monotonic() - started
            receipt = build_stage_receipt(
                row, dataset, out_dir, warm_start, command, wall_seconds
            )
            write_receipt(receipt_path, receipt)
            print(f"[clean-chain] {label}: receipt written", flush=True)
        if row["stage"] == 6:
            stage6_adapter = out_dir
        if row["stage"] == 7:
            stage7_adapter = out_dir
        warm_start = out_dir
    merge_stage7(manifest, stage7_adapter)
    print(
        json.dumps(
            {
                "zero_root_merge_receipt": str(ZERO_ROOT_MERGE_RECEIPT),
                "final_merge_receipt": str(FINAL_MERGE_RECEIPT),
                "final_merged": str(FINAL_MERGED_OUT),
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
        help="authenticate the clean-chain package only (fast, no GPU, no writes)",
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
