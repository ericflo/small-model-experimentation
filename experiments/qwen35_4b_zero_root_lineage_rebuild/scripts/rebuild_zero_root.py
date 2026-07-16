#!/usr/bin/env python3
"""Replay the six documented lineage stages from a FRESH zero-initialized root.

Lifecycle 22 — the provenance question: the hygiene_explore composite's
recorded lineage is carried by ONE warm-started LoRA adapter whose root
(the undocumented C53-era gym-line 'blend' adapter) has NO committed
creation receipt anywhere in the repository. This cell replays the SIX
documented, contamination-free training stages EXACTLY as recorded in
the byte-identical copied lineage manifest — same datasets, same fixed
seeds, same trainer variants, same per-stage hyperparameters, every
stage trained on the raw pinned HF base — with exactly ONE change:

    the warm-start chain is rewired to start from NOTHING. Stage 1
    trains a FRESH rank-32/alpha-64 adapter (the trainer's default
    fresh path: LoRA-B zero-initialized, so the delta starts at zero)
    instead of warm-starting from the vendored blend root. Stages 2-6
    warm-start from the PREVIOUS zero-root stage's adapter.

The blend root is deliberately NOT vendored into this cell — its
omission IS the design — and this script fails closed if a copy ever
appears under this cell's artifact storage. The manifest's recorded
per-stage adapter hashes belong to the blend-rooted chain; they CANNOT
be expected here and are written into every stage receipt as CONTRAST
fields only, never as verification.

Modes:

- ``--verify-inputs`` (fast, no GPU, wired into ``run.py --smoke``):
  authenticates the copied package — manifest bytes (sha256-pinned) and
  internal chaining, every copied dataset (sha256 + row count), every
  trainer and merger copy (sha256) — and asserts the root adapter is
  NOT vendored. Writes nothing.
- default (GPU, ~2.5-3h): replays stages 1→6 (stage 1 fresh, no
  ``--warm-start``; stages 2-6 warm-started from the previous zero-root
  output), writes a receipt per stage to ``runs/lineage/``, then merges
  the stage-6 adapter onto the raw pinned HF base via the copied merger
  and writes the merge receipt. A stage whose receipt AND on-disk
  adapter both verify is skipped (crash-safe resume); an unreceipted or
  mismatching output fails closed.
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
# Byte pin of the copied manifest: byte-identical to
# experiments/qwen35_4b_goal_gate_confirmation/data/lineage/lineage_manifest.json.
MANIFEST_SHA256 = "1f49cd8b8706c8db858d30af1bf14fe09403971256514919bc24e7b6c47ff121"
SOURCE_EXPERIMENT = "qwen35_4b_goal_gate_confirmation"
LINEAGE_RECEIPTS = EXP / "runs" / "lineage"
ADAPTER_ROOT = ROOT / "large_artifacts" / EXP.name / "adapters"
MERGED_OUT = ROOT / "large_artifacts" / EXP.name / "merged" / "zero_root_hygiene_explore"
MERGE_RECEIPT = LINEAGE_RECEIPTS / "merge.json"
# The design is the OMISSION of the root: no copy of the blend adapter may
# ever exist inside this cell's artifact storage.
FORBIDDEN_ROOT_DIR = ROOT / "large_artifacts" / EXP.name / "lineage_root"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
STAGE_COUNT = 6
FRESH_INIT = "fresh_zero_init"
SHA_RE = re.compile(r"[0-9a-f]{64}")
MANIFEST_KEYS = {
    "schema_version", "experiment_id", "stage", "directive", "model",
    "carrier", "base_reserialized_note", "root_adapter", "trainers",
    "merger", "stages", "final_merge", "determinism", "verification",
}
STAGE_KEYS = {
    "stage", "name", "dataset", "warm_start", "trainer", "trainer_sha256",
    "seed", "hyperparameters", "output", "produced",
}
HYPER_KEYS = {
    "epochs", "lr", "rank", "alpha", "batch_size", "grad_accum",
    "max_length", "w_think",
}
# The six per-stage training seeds are INHERITED STAGE CONSTANTS: they are
# part of what "same recipe" means and are reused deliberately, not fresh.
LINEAGE_TRAINING_SEEDS = (42, 43, 44, 47, 51, 55)
# The original blend-rooted composite this rebuild contrasts against
# (CONTRAST ONLY, never verification: a different root means different bytes).
ORIGINAL_MERGED_TREE_SHA256 = (
    "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"
)
ORIGINAL_MERGED_WEIGHTS_SHA256 = (
    "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"
)
CONTRAST_NOTE = (
    "CONTRAST ONLY: these are the ORIGINAL blend-rooted chain's recorded "
    "hashes; the zero-root chain trains from a different root, so its "
    "bytes are EXPECTED to differ and these values are never used as "
    "verification"
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
    """Load and authenticate the byte-identical copied lineage manifest."""
    if not MANIFEST.is_file():
        raise ValueError(f"copied lineage manifest is absent: {MANIFEST}")
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise ValueError(
            "copied lineage manifest is not byte-identical to the pinned "
            f"source package copy: {MANIFEST}"
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or set(manifest) != MANIFEST_KEYS
        or manifest.get("schema_version") != 1
        or manifest.get("experiment_id") != SOURCE_EXPERIMENT
        or manifest.get("stage") != "lineage_manifest"
        or manifest.get("model", {}).get("id") != MODEL_ID
        or manifest.get("model", {}).get("revision") != MODEL_REVISION
    ):
        raise ValueError("copied lineage manifest failed schema authentication")
    stages = manifest.get("stages")
    if not isinstance(stages, list) or len(stages) != STAGE_COUNT:
        raise ValueError("lineage manifest must carry exactly six stages")
    for index, row in enumerate(stages, start=1):
        keys = set(row) - {"targeted_close_overrides"}
        if (
            keys != STAGE_KEYS
            or row.get("stage") != index
            or not _sha(row.get("trainer_sha256"))
            or not isinstance(row.get("seed"), int)
            or row.get("seed") != LINEAGE_TRAINING_SEEDS[index - 1]
            or not _sha(row.get("dataset", {}).get("sha256"))
            or not isinstance(row.get("dataset", {}).get("rows"), int)
            or not _sha(row.get("produced", {}).get("adapter_weights_sha256"))
            or not _sha(row.get("produced", {}).get("adapter_config_sha256"))
            or not set(row.get("hyperparameters", {})) <= HYPER_KEYS | {"w_close"}
            or not set(row.get("hyperparameters", {})) >= HYPER_KEYS
        ):
            raise ValueError(f"lineage manifest stage {index} failed authentication")
        expected_warm = "root_adapter" if index == 1 else f"stage {index - 1}"
        if row.get("warm_start") != expected_warm:
            raise ValueError(
                f"lineage manifest stage {index} breaks the recorded warm-start "
                f"chain: {row.get('warm_start')!r} != {expected_warm!r}"
            )
        trainer = row.get("trainer", "")
        trainers = manifest.get("trainers", {})
        if trainer not in trainers or trainers[trainer].get("sha256") != row["trainer_sha256"]:
            raise ValueError(
                f"lineage manifest stage {index} names an unregistered trainer"
            )
        if index not in trainers[trainer].get("stages", []):
            raise ValueError(
                f"lineage manifest stage {index} disagrees with the trainer's "
                "stage registration"
            )
    merger = manifest.get("merger", {})
    if merger.get("path") != "scripts/merge_adapter.py" or not _sha(merger.get("sha256")):
        raise ValueError("lineage manifest merger block failed authentication")
    final = manifest.get("final_merge", {})
    if (
        merger["sha256"] != final.get("merger_sha256")
        or final.get("expected_output", {}).get("weights_sha256")
        != ORIGINAL_MERGED_WEIGHTS_SHA256
        or final.get("expected_output", {}).get("published_tree_sha256")
        != ORIGINAL_MERGED_TREE_SHA256
    ):
        raise ValueError("lineage manifest final merge block failed authentication")
    return manifest


def require_root_not_vendored() -> None:
    """The omission of the blend root IS the design; fail closed on a copy."""
    if FORBIDDEN_ROOT_DIR.exists():
        raise ValueError(
            "the blend root adapter must NOT be vendored into this cell "
            f"(training without it is the design): {FORBIDDEN_ROOT_DIR}"
        )


def verify_inputs(manifest: dict) -> dict:
    """Fast, model-free package authentication (wired into run.py --smoke)."""
    require_root_not_vendored()
    checked = {"datasets": 0, "trainers": 0, "merger": 0, "root_vendored": False}
    for row in manifest["stages"]:
        dataset = EXP / "data" / "lineage" / Path(row["dataset"]["file"]).name
        if not dataset.is_file() or sha256_file(dataset) != row["dataset"]["sha256"]:
            raise ValueError(f"lineage dataset copy is absent or changed: {dataset}")
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
    return checked


def stage_command(row: dict, dataset: Path, out_dir: Path, warm_start: Path | None) -> list[str]:
    """The exact recorded per-stage recipe, warm-start rewired to zero root.

    Byte-for-byte the source cell's command construction except the
    ``--warm-start`` argument: stage 1 omits it entirely (the trainer's
    default fresh path builds a new rank-32/alpha-64 LoRA whose B matrices
    are zero-initialized), and stages 2-6 receive the PREVIOUS zero-root
    stage's adapter instead of the blend-rooted chain.
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
    if warm_start is not None:
        command.extend(("--warm-start", str(warm_start)))
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


def build_stage_receipt(
    row: dict,
    dataset: Path,
    out_dir: Path,
    warm_start: Path | None,
    command: list[str],
    wall_seconds: float,
) -> dict:
    """One committed per-stage receipt; original hashes are CONTRAST only."""
    weights = out_dir / "adapter_model.safetensors"
    config = out_dir / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise ValueError(f"stage output is incomplete: {out_dir}")
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
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": row["stage"],
        "name": row["name"],
        "lineage": "zero_root",
        "manifest_sha256": MANIFEST_SHA256,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "dataset": {
            "file": f"data/lineage/{dataset.name}",
            "sha256": row["dataset"]["sha256"],
            "rows": row["dataset"]["rows"],
        },
        "trainer": {"path": row["trainer"], "sha256": row["trainer_sha256"]},
        "seed": row["seed"],
        "seed_is_inherited_stage_constant": True,
        "hyperparameters": dict(row["hyperparameters"]),
        "targeted_close_overrides": row.get("targeted_close_overrides"),
        "warm_start": warm_block,
        "fresh_init": warm_start is None,
        "command": command,
        "adapter": str(out_dir),
        "produced": {
            "adapter_config_sha256": sha256_file(config),
            "adapter_weights_sha256": sha256_file(weights),
            "adapter_weights_size": weights.stat().st_size,
        },
        "original_produced_contrast": {
            "adapter_config_sha256": row["produced"]["adapter_config_sha256"],
            "adapter_weights_sha256": row["produced"]["adapter_weights_sha256"],
            "note": CONTRAST_NOTE,
        },
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


def authenticate_stage_receipt(
    receipt: dict, row: dict, out_dir: Path, warm_start: Path | None
) -> None:
    """A published stage receipt must restate the manifest recipe exactly."""
    expected_warm = (
        FRESH_INIT
        if warm_start is None
        else {
            "source": f"stage {row['stage'] - 1}",
            "adapter": str(warm_start),
            "adapter_config_sha256": receipt.get("warm_start", {}).get(
                "adapter_config_sha256"
            ),
            "adapter_weights_sha256": receipt.get("warm_start", {}).get(
                "adapter_weights_sha256"
            ),
        }
    )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("experiment_id") != EXP.name
        or receipt.get("lineage") != "zero_root"
        or receipt.get("manifest_sha256") != MANIFEST_SHA256
        or receipt.get("stage") != row["stage"]
        or receipt.get("name") != row["name"]
        or receipt.get("seed") != row["seed"]
        or receipt.get("hyperparameters") != row["hyperparameters"]
        or receipt.get("targeted_close_overrides")
        != row.get("targeted_close_overrides")
        or receipt.get("trainer")
        != {"path": row["trainer"], "sha256": row["trainer_sha256"]}
        or receipt.get("dataset", {}).get("sha256") != row["dataset"]["sha256"]
        or receipt.get("fresh_init") is not (warm_start is None)
        or receipt.get("warm_start") != expected_warm
        or receipt.get("adapter") != str(out_dir)
        or receipt.get("original_produced_contrast", {}).get("note") != CONTRAST_NOTE
        or not _sha(receipt.get("produced", {}).get("adapter_weights_sha256"))
        or not _sha(receipt.get("produced", {}).get("adapter_config_sha256"))
    ):
        raise ValueError(
            f"published stage receipt disagrees with the manifest recipe: "
            f"stage {row['stage']} ({row['name']})"
        )
    if warm_start is not None:
        warm = receipt["warm_start"]
        if sha256_file(
            warm_start / "adapter_model.safetensors"
        ) != warm["adapter_weights_sha256"] or sha256_file(
            warm_start / "adapter_config.json"
        ) != warm["adapter_config_sha256"]:
            raise ValueError(
                f"stage {row['stage']} warm-start adapter on disk does not "
                "match the receipt's recorded chain"
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
    return hashlib.sha256(rendered).hexdigest()


def build_merge_receipt(
    manifest: dict, adapter: Path, wall_seconds: float
) -> dict:
    tree = merged_tree_manifest(MERGED_OUT)
    files = {row["name"]: row for row in tree}
    weights = files["model.safetensors"]
    if weights["size"] <= 0:
        raise ValueError("merged weights are empty")
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "merge",
        "name": "zero_root_hygiene_explore",
        "manifest_sha256": MANIFEST_SHA256,
        "merger": {
            "path": manifest["merger"]["path"],
            "sha256": manifest["merger"]["sha256"],
        },
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "adapter": {
            "path": str(adapter),
            "source": "stage 6 (hygiene_explore), zero-root chain",
            "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
            "adapter_weights_sha256": sha256_file(
                adapter / "adapter_model.safetensors"
            ),
        },
        "merged": str(MERGED_OUT),
        "output_files": tree,
        "output_tree_sha256": tree_manifest_sha256(tree),
        "weights_sha256": weights["sha256"],
        "weights_size_bytes": weights["size"],
        "inner_merge_receipt_sha256": files["merge_receipt.json"]["sha256"],
        "original_contrast": {
            "published_tree_sha256": ORIGINAL_MERGED_TREE_SHA256,
            "weights_sha256": ORIGINAL_MERGED_WEIGHTS_SHA256,
            "note": CONTRAST_NOTE,
        },
        "wall_seconds": wall_seconds,
    }


def write_receipt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def rebuild(manifest: dict) -> None:
    """GPU path: six zero-root stage replays, then the final merge."""
    require_root_not_vendored()
    warm_start: Path | None = None
    for row in manifest["stages"]:
        out_dir = ADAPTER_ROOT / stage_dirname(row)
        receipt_path = LINEAGE_RECEIPTS / f"{stage_dirname(row)}.json"
        dataset = EXP / "data" / "lineage" / Path(row["dataset"]["file"]).name
        label = f"stage {row['stage']} ({row['name']})"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            authenticate_stage_receipt(receipt, row, out_dir, warm_start)
            if not receipt_matches_disk(receipt, out_dir):
                raise ValueError(
                    f"{label} receipt exists but its adapter does not verify "
                    f"on disk: {out_dir}; audit and delete both explicitly "
                    "before rebuilding"
                )
            print(f"[zero-root] {label}: receipt verified, skipping", flush=True)
        else:
            if out_dir.exists():
                raise ValueError(
                    f"{label} output exists without a receipt: {out_dir}; "
                    "delete it explicitly before rebuilding"
                )
            command = stage_command(row, dataset, out_dir, warm_start)
            mode = "FRESH zero-init" if warm_start is None else f"warm from {warm_start.name}"
            print(
                f"[zero-root] {label}: training seed {row['seed']} ({mode})",
                flush=True,
            )
            started = time.monotonic()
            subprocess.run(command, cwd=ROOT, check=True)
            wall_seconds = time.monotonic() - started
            receipt = build_stage_receipt(
                row, dataset, out_dir, warm_start, command, wall_seconds
            )
            write_receipt(receipt_path, receipt)
            print(
                f"[zero-root] {label}: receipt written to {receipt_path}",
                flush=True,
            )
        warm_start = out_dir
    adapter = warm_start
    if MERGE_RECEIPT.is_file():
        receipt = json.loads(MERGE_RECEIPT.read_text(encoding="utf-8"))
        tree = merged_tree_manifest(MERGED_OUT)
        if (
            receipt.get("output_tree_sha256") != tree_manifest_sha256(tree)
            or receipt.get("adapter", {}).get("path") != str(adapter)
        ):
            raise ValueError(
                "published merge receipt does not match the merged composite "
                "on disk; audit and delete both explicitly before remerging"
            )
        print("[zero-root] merge: receipt verified, skipping", flush=True)
        return
    if MERGED_OUT.exists():
        raise ValueError(
            f"merged output exists without a receipt: {MERGED_OUT}; delete "
            "it explicitly before remerging"
        )
    print("[zero-root] merging stage-6 adapter onto the raw pinned HF base", flush=True)
    started = time.monotonic()
    subprocess.run(
        [
            sys.executable, "-B", str(EXP / manifest["merger"]["path"]),
            "--adapter", str(adapter),
            "--out", str(MERGED_OUT),
        ],
        cwd=ROOT, check=True,
    )
    wall_seconds = time.monotonic() - started
    receipt = build_merge_receipt(manifest, adapter, wall_seconds)
    write_receipt(MERGE_RECEIPT, receipt)
    print(
        json.dumps(
            {
                "merge_receipt": str(MERGE_RECEIPT),
                "merged": str(MERGED_OUT),
                "output_tree_sha256": receipt["output_tree_sha256"],
                "weights_sha256": receipt["weights_sha256"],
                "next": (
                    "commit the runs/lineage receipts, fill the three "
                    "TODO-pins in scripts/run_benchmark.py from merge.json, "
                    "then seek the PASS_BENCHMARK_EVENT review"
                ),
            },
            indent=1,
            sort_keys=True,
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--verify-inputs",
        action="store_true",
        help="authenticate the copied lineage package only (fast, no GPU, no writes)",
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
