#!/usr/bin/env python3
"""Rebuild the hygiene_explore composite from THIS cell alone (standalone gate).

Standalone-reproducibility gate (docs/quality_gates.md, owner directive
2026-07-15): this cell evaluates a non-base checkpoint, so it carries the
complete model-reproduction package. This script is the reproduction
path; cross-experiment SHAs are verification aids only. Inputs, all
inside this experiment's own directory tree:

- ``data/lineage/lineage_manifest.json`` — the complete fixed-seed
  recipe (six warm-start QLoRA stages plus the final merge);
- ``data/lineage/stage01_… … stage06_….jsonl`` — byte-identical copies
  of the six ordered SFT datasets;
- ``scripts/lineage_trainers/`` — byte-identical copies of the three
  trainer variants; ``scripts/merge_adapter.py`` — the merger copy;
- ``large_artifacts/qwen35_4b_goal_gate_confirmation/lineage_root/blend``
  — the vendored frozen root adapter (HARD provenance boundary: no
  committed creation receipt exists; it is a frozen input, not
  reconstructable from committed receipts).

Every stage trains from the raw pinned HF base ``Qwen/Qwen3.5-4B @
851bf6e8…`` plus the previous stage's adapter (``--warm-start``); the
final merge applies the stage-6 adapter to the same raw base (the
merger's default). ``base_reserialized`` is never an input.

Modes:

- ``--verify-inputs`` (fast, no GPU, wired into ``run.py --smoke``):
  authenticates the whole package — manifest schema and internal
  chaining, every copied dataset (sha256 + row count), every trainer and
  merger copy (sha256), every vendored root-adapter file (sha256 +
  size) — and writes nothing.
- default (GPU): replays stages 1→6 deterministically (correct trainer
  variant, dataset from ``data/lineage/``, fixed seed, warm start from
  the previous stage's output), verifies each produced adapter's
  weights/config sha256 against the manifest, then performs the final
  merge and verifies the merged content files (``model.safetensors``
  sha e2112344… is the hard check; the published tree sha covers a
  merge receipt whose embedded absolute adapter path is machine-local,
  as the manifest documents). A stage whose output directory already
  verifies is skipped; a mismatching output fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
MANIFEST = EXP / "data" / "lineage" / "lineage_manifest.json"
LINEAGE_RUNS = EXP / "runs" / "lineage"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
STAGE_COUNT = 6
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def load_manifest() -> dict:
    """Load and schema-check the lineage manifest; fail closed on drift."""
    if not MANIFEST.is_file():
        raise ValueError(f"lineage manifest is absent: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or set(manifest) != MANIFEST_KEYS
        or manifest.get("schema_version") != 1
        or manifest.get("experiment_id") != EXP.name
        or manifest.get("stage") != "lineage_manifest"
        or manifest.get("model", {}).get("id") != MODEL_ID
        or manifest.get("model", {}).get("revision") != MODEL_REVISION
    ):
        raise ValueError("lineage manifest failed schema authentication")
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
                f"lineage manifest stage {index} breaks the warm-start chain: "
                f"{row.get('warm_start')!r} != {expected_warm!r}"
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
    seeds = [row["seed"] for row in stages]
    if len(set(seeds)) != STAGE_COUNT:
        raise ValueError("lineage manifest stage seeds must be pairwise distinct")
    root = manifest.get("root_adapter", {})
    if (
        not _sha(root.get("weights_sha256"))
        or not _sha(root.get("config_sha256"))
        or not isinstance(root.get("files"), dict)
        or not root.get("provenance_boundary")
        or root.get("files", {}).get("adapter_model.safetensors", {}).get("sha256")
        != root.get("weights_sha256")
        or root.get("files", {}).get("adapter_config.json", {}).get("sha256")
        != root.get("config_sha256")
    ):
        raise ValueError("lineage manifest root adapter block failed authentication")
    merger = manifest.get("merger", {})
    if merger.get("path") != "scripts/merge_adapter.py" or not _sha(merger.get("sha256")):
        raise ValueError("lineage manifest merger block failed authentication")
    final = manifest.get("final_merge", {})
    if (
        merger["sha256"] != final.get("merger_sha256")
        or not _sha(final.get("expected_output", {}).get("weights_sha256"))
        or not isinstance(final.get("expected_output", {}).get("content_files_sha256"), dict)
        or final["expected_output"]["content_files_sha256"].get("model.safetensors")
        != final["expected_output"]["weights_sha256"]
        or not _sha(final.get("expected_output", {}).get("published_tree_sha256"))
    ):
        raise ValueError("lineage manifest final merge block failed authentication")
    return manifest


def verify_inputs(manifest: dict) -> dict:
    """Fast, model-free package authentication (wired into run.py --smoke)."""
    checked = {"datasets": 0, "trainers": 0, "root_files": 0, "merger": 0}
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
    root_dir = ROOT / manifest["root_adapter"]["vendored_path"]
    if not root_dir.is_dir() or root_dir.is_symlink():
        raise ValueError(f"vendored root adapter is absent: {root_dir}")
    expected_files = manifest["root_adapter"]["files"]
    names = {child.name for child in root_dir.iterdir()}
    if names != set(expected_files):
        raise ValueError(
            "vendored root adapter file set changed: "
            f"missing={sorted(set(expected_files) - names)}, "
            f"unexpected={sorted(names - set(expected_files))}"
        )
    for name, block in sorted(expected_files.items()):
        path = root_dir / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != block["size"]
            or sha256_file(path) != block["sha256"]
        ):
            raise ValueError(f"vendored root adapter file changed: {path}")
        checked["root_files"] += 1
    return checked


def adapter_matches(out_dir: Path, produced: dict) -> bool:
    weights = out_dir / "adapter_model.safetensors"
    config = out_dir / "adapter_config.json"
    return (
        weights.is_file()
        and config.is_file()
        and sha256_file(weights) == produced["adapter_weights_sha256"]
        and sha256_file(config) == produced["adapter_config_sha256"]
    )


def require_adapter(out_dir: Path, produced: dict, label: str) -> None:
    if not adapter_matches(out_dir, produced):
        raise ValueError(
            f"rebuilt {label} adapter does not match the manifest verification "
            f"aids at {out_dir}; the recorded lineage did not reproduce on this "
            "stack (see the manifest's determinism block)"
        )


def stage_command(row: dict, dataset: Path, out_dir: Path, warm_start: Path) -> list[str]:
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
        "--warm-start", str(warm_start),
        "--w-think", str(hypers["w_think"]),
        "--seed", str(row["seed"]),
    ]
    if "w_close" in hypers:
        command.extend(("--w-close", str(hypers["w_close"])))
    targeted = row.get("targeted_close_overrides")
    if targeted is not None:
        for kind in targeted["target_close_kinds"]:
            command.extend(("--target-close-kind", kind))
        command.extend(("--target-w-close", str(targeted["target_w_close"])))
    return command


def rebuild(manifest: dict) -> None:
    """GPU path: replay the six stages, then the final merge, verifying each."""
    warm_start = ROOT / manifest["root_adapter"]["vendored_path"]
    for row in manifest["stages"]:
        out_dir = EXP / row["output"]
        dataset = EXP / "data" / "lineage" / Path(row["dataset"]["file"]).name
        label = f"stage {row['stage']} ({row['name']})"
        if adapter_matches(out_dir, row["produced"]):
            print(f"[rebuild] {label}: already verified, skipping", flush=True)
        else:
            if out_dir.exists():
                raise ValueError(
                    f"{label} output exists but does not verify: {out_dir}; "
                    "delete it explicitly before rebuilding"
                )
            print(f"[rebuild] {label}: training (seed {row['seed']})", flush=True)
            subprocess.run(
                stage_command(row, dataset, out_dir, warm_start),
                cwd=ROOT, check=True,
            )
            require_adapter(out_dir, row["produced"], label)
        warm_start = out_dir
    final = manifest["final_merge"]
    merged = EXP / final["output"]
    expected = final["expected_output"]["content_files_sha256"]
    if not merged.exists():
        print("[rebuild] final merge onto the raw pinned HF base", flush=True)
        subprocess.run(
            [
                sys.executable, "-B", str(EXP / final["merger"]),
                "--adapter", str(warm_start),
                "--out", str(merged),
            ],
            cwd=ROOT, check=True,
        )
    mismatched = [
        name
        for name, digest in sorted(expected.items())
        if not (merged / name).is_file() or sha256_file(merged / name) != digest
    ]
    if mismatched:
        raise ValueError(
            f"rebuilt merge content files do not match the manifest: {mismatched}"
        )
    print(
        "[rebuild] merged content files verified "
        f"(weights {expected['model.safetensors'][:12]}...); the published "
        "tree sha additionally covers a merge receipt whose embedded adapter "
        "path is machine-local (see the manifest's tree_sha_note)",
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
                "manifest_sha256": sha256_file(MANIFEST),
                "mode": "verify_inputs" if args.verify_inputs else "rebuild",
                "checked": checked,
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
