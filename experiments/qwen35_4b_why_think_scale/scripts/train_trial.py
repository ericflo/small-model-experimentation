#!/usr/bin/env python3
"""Fail-closed trainer wrapper for a single WHY scale-ladder RUNG.

The ladder is a SWEEP: the same WHY curriculum is trained at four rung sizes
(2000, 5000, 10000, 20000 rows) to find the peak of the scaling curve. Each rung
trains a FRESH rank-32/alpha-64 QLoRA adapter (NO warm start) from the pinned
``base_reserialized`` composite (tree ``26d8ee48…``, weights ``b654e033…``) via
the vendored trainer's ``--model-path``, over the rung's corpus from the committed
ladder manifest (``data/ladder_manifest.json``, sha-pinned).

Recipe (frozen, identical across rungs except epochs): lr 1e-5, rank 32, alpha 64,
batch 1, grad-accum 8, max-length 4096, w_think 0.2, w_close 0.2, training seed
95201. EPOCH SCHEDULE: ALWAYS 1 EPOCH (owner directive) - with unlimited unique data, vary data volume not epochs; every step sees fresh data (no memorization, no epoch confound). Was:
``max(1, round(8000/rows))`` (RETIRED); now epochs_for()==1 for all rungs at
20000 — which holds total sample exposures roughly comparable (8k / 10k / 10k /
20k) while letting the model see each rung enough. Optimizer steps = rows * epochs
/ (batch 1 x grad-accum 8).

The base composite authenticates FAIL-CLOSED pre-training (in-cell provenance copy
+ per-file receipt/tokenizer checks + full-tree manifest sha + full 9 GB weights
hash), identical to the sibling why_comment cell. The rung corpus is pinned to the
committed ladder manifest, whose generator sha ties it to the exact in-cell
generator, so the training data provenance is standalone and reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
TRAINER = EXP / "scripts" / "train_think.py"
TRAINER_SHA256 = "e0eca2a230dae5d109d418dcb4cc19af05882a770af14350ffd741a8d5e90f01"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
COMPUTE_REVIEW = EXP / "reports" / "compute_review.md"
MANIFEST = EXP / "data" / "ladder_manifest.json"
GENERATOR = EXP / "scripts" / "gen_why_think_curriculum.py"
FIXTURE = EXP / "data" / "contamination" / "banned_function_names.json"

LADDER_SIZES = (2000, 5000, 10000, 20000, 40000)

# Training base: the reserialized BASE composite (a directory of full bf16
# weights), NOT an adapter. Each fresh rank-32 adapter trains on top of it.
MODEL_PATH = (
    ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum" / "merged" / "base_reserialized"
)
MODEL_PATH_TREE_SHA256 = "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677"
MODEL_PATH_WEIGHTS_SHA256 = "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db"
MODEL_PATH_WEIGHTS_SIZE_BYTES = 9_078_620_536
MODEL_PATH_RECEIPT_SHA256 = "25aee794cfffe4d58110defc61177edef1f5324e47deb28fbd3cb7ccd61ae54f"
MODEL_PATH_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
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
MERGED_FILE_SHA256 = {
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "config.json": "a1c80f0efa6f83f631eaa9c25ffa166e3b1f9db395cc3b14374dfc0962261f60",
    "generation_config.json": "0c46d8aa4f0ae5e611c961f70b87c83fb696043c1e319337708e96f882180de1",
    "merge_receipt.json": "25aee794cfffe4d58110defc61177edef1f5324e47deb28fbd3cb7ccd61ae54f",
    "model.safetensors": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
}
BASE_PROVENANCE_COPY = EXP / "data" / "provenance" / "base_reserialized.json"

LORA_RANK = 32
LORA_ALPHA = 64
LORA_TARGET_MODULES = (
    "down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj",
)
TRAINING_SEED = 95201
BATCH_SIZE = 1
GRAD_ACCUM = 8

# TODO-PIN per rung after training: the SINGLE-LINE sorted-key dict of the four
# published sha256 values {"adapter_config","adapter_weights","log","receipt"}.
PUBLISHED_RUNG_HASHES: dict[int, dict | None] = {2000: {"adapter_config": "3e286a2ee95fb30f52a822c5bae935b67d833e8ba39611b9d497f3298a393fe8", "adapter_weights": "1a0334c94573f4e4dc408ee361f21168c26f868c72464ad54998f82230707cd1", "log": "045c10080818896fa16bc861329e5fcb1767e57a789b19a07f955507d4b57aff", "receipt": "a8046edad13658d77cac67c3225373523e18cc70ab86d9a3fcd2c63b6e9954ec"}, 5000: {"adapter_config": "74312805b05fe7b73a46d0fd8b296c248cf47c3024547a0281358e0cc8ce24d8", "adapter_weights": "56dd32b227bf7c48c56f0cf440cd83ff7e996ccb1be3adf1836fb8f6fefdec2b", "log": "3a1f820414ef6081768e21e30c0bff669a48f546bc63f023fe3ff4221baffae3", "receipt": "54ba3c9837517a82974d33e919cb13cb032f9bc49b867287bd5682747522ca46"}, 10000: None, 20000: None, 40000: None}
ADAPTER_ROOT = ROOT / "large_artifacts" / EXP.name / "adapters"


def epochs_for(rows: int) -> int:
    """Larger corpora need fewer epochs; keep total exposures roughly comparable."""
    return 1  # 1 epoch everywhere: unlimited unique data -> vary data volume, never repeat (no memorization, no epoch confound)


def optimizer_steps_for(rows: int) -> int:
    return rows * epochs_for(rows) // (BATCH_SIZE * GRAD_ACCUM)


def arm_name(rows: int) -> str:
    return f"why_think_{rows}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"not a JSON object: {path}")
    return payload


def run_text(command: list[str]) -> str:
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def committed_at_head(path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=False, capture_output=True
    )
    return committed.returncode == 0 and committed.stdout == path.read_bytes()


def normalize_log(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def rung_from_manifest(rows: int) -> dict:
    """The manifest entry for ``rows`` (fail-closed on generator/fixture drift)."""
    if not MANIFEST.is_file():
        raise ValueError(f"ladder manifest absent: {MANIFEST} (run scripts/run.py --stage gen-ladder)")
    manifest = load_json(MANIFEST)
    if manifest.get("generator_sha256") != sha256_file(GENERATOR):
        raise ValueError("ladder manifest generator_sha256 disagrees with the in-cell generator")
    if manifest.get("contamination_fixture_sha256") != sha256_file(FIXTURE):
        raise ValueError("ladder manifest contamination_fixture_sha256 disagrees with the in-cell fixture")
    for entry in manifest.get("rungs", []):
        if entry.get("rows") == rows:
            return entry
    raise ValueError(f"rung {rows} absent from the ladder manifest")


def corpus_for(rows: int) -> tuple[Path, str]:
    entry = rung_from_manifest(rows)
    path = ROOT / entry["path"]
    return path, entry["corpus_sha256"]


def expected_hyperparameters(rows: int) -> dict:
    return {
        "epochs": float(epochs_for(rows)),
        "lr": 1e-5,
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "max_length": 4096,
        "w_think": 0.2,
        "w_close": 0.2,
        "seed": TRAINING_SEED,
        "optimizer_steps": optimizer_steps_for(rows),
    }


def merged_tree_manifest(output: Path) -> list[dict]:
    if not output.is_dir() or output.is_symlink():
        raise ValueError(f"base composite is not a real directory: {output}")
    children = sorted(output.iterdir(), key=lambda p: p.name)
    if any(p.is_symlink() or not p.is_file() for p in children):
        raise ValueError("base composite contains a symlink or non-file entry")
    names = {p.name for p in children}
    if names != MERGED_FILE_NAMES:
        raise ValueError(
            f"base composite file set changed: missing={sorted(MERGED_FILE_NAMES - names)}, "
            f"unexpected={sorted(names - MERGED_FILE_NAMES)}"
        )
    return [{"name": p.name, "size": p.stat().st_size, "sha256": sha256_file(p)} for p in children]


def tree_manifest_sha256(manifest: list[dict]) -> str:
    rendered = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(rendered).hexdigest()


def check_base_provenance() -> None:
    if (
        not BASE_PROVENANCE_COPY.is_file()
        or sha256_file(BASE_PROVENANCE_COPY) != MODEL_PATH_RECEIPT_SHA256
    ):
        raise ValueError(f"in-cell base provenance copy is absent or changed: {BASE_PROVENANCE_COPY}")
    payload = load_json(BASE_PROVENANCE_COPY)
    weight_files = {row.get("name"): (row.get("sha256"), row.get("size_bytes")) for row in payload.get("weight_files", [])}
    if (
        payload.get("method") != "pinned_base_composite_reserialization"
        or payload.get("model_lineage") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("tokenizer_sha256") != MODEL_PATH_TOKENIZER_SHA256
        or weight_files.get("model.safetensors") != (MODEL_PATH_WEIGHTS_SHA256, MODEL_PATH_WEIGHTS_SIZE_BYTES)
    ):
        raise ValueError("base provenance copy does not describe the frozen base_reserialized composite")


def check_base_composite_cheap() -> None:
    for name, expected in MERGED_FILE_SHA256.items():
        path = MODEL_PATH / name
        if not path.is_file():
            raise ValueError(f"base composite missing {name}")
        if name == "model.safetensors":
            if path.stat().st_size != MODEL_PATH_WEIGHTS_SIZE_BYTES:
                raise ValueError("base composite weights size changed")
            continue
        if sha256_file(path) != expected:
            raise ValueError(f"base composite {name} changed")


def authenticate_base_fail_closed(*, full_weights: bool) -> dict:
    check_base_provenance()
    check_base_composite_cheap()
    manifest = merged_tree_manifest(MODEL_PATH)
    observed_tree = tree_manifest_sha256(manifest)
    if observed_tree != MODEL_PATH_TREE_SHA256:
        raise ValueError(f"base composite tree changed: {observed_tree}")
    if full_weights:
        if sha256_file(MODEL_PATH / "model.safetensors") != MODEL_PATH_WEIGHTS_SHA256:
            raise ValueError("base composite weights hash changed")
    return {
        "path": str(MODEL_PATH.resolve()),
        "tree_sha256": observed_tree,
        "weights_sha256": MODEL_PATH_WEIGHTS_SHA256,
        "receipt_sha256": MODEL_PATH_RECEIPT_SHA256,
        "provenance_copy_sha256": MODEL_PATH_RECEIPT_SHA256,
    }


def validate_adapter_config(config: dict) -> bool:
    return (
        config.get("r") == LORA_RANK
        and config.get("lora_alpha") == LORA_ALPHA
        and config.get("base_model_name_or_path") == str(MODEL_PATH.resolve())
        and sorted(config.get("target_modules") or ()) == list(LORA_TARGET_MODULES)
    )


def validate_published_rung(rows: int, *, require_committed: bool = True) -> dict:
    if rows not in LADDER_SIZES:
        raise ValueError(f"unknown rung: {rows}")
    name = arm_name(rows)
    receipt_path = EXP / "runs" / "training" / f"{name}.json"
    log_path = EXP / "runs" / "training" / f"{name}.log"
    adapter = ADAPTER_ROOT / name
    if (
        not receipt_path.is_file()
        or not log_path.is_file()
        or (require_committed and not committed_at_head(receipt_path))
        or (require_committed and not committed_at_head(log_path))
    ):
        raise ValueError(f"published {name} receipt/log is absent from HEAD")
    payload = load_json(receipt_path)
    published = PUBLISHED_RUNG_HASHES.get(rows)
    if published is None and require_committed:
        raise ValueError(
            f"PUBLISHED_RUNG_HASHES[{rows}] is unfilled (TODO-PIN); pin the published "
            "receipt/log/adapter hashes before the next stage"
        )
    data_path, data_hash = corpus_for(rows)
    dataset = payload.get("dataset", {})
    base = payload.get("base_composite", {})
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    if (
        payload.get("experiment_id") != EXP.name
        or payload.get("name") != name
        or payload.get("rows") != rows
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("returncode") != 0
        or payload.get("adapter_complete") is not True
        or payload.get("trainer_sha256") != TRAINER_SHA256
        or payload.get("train_rows") != rows
        or payload.get("skipped_rows") != 0
        or payload.get("preflight_git_status") != ""
        or payload.get("hyperparameters") != expected_hyperparameters(rows)
        or dataset.get("path") != str(data_path.resolve())
        or dataset.get("sha256") != data_hash
        or dataset.get("rows") != rows
        or base.get("path") != str(MODEL_PATH.resolve())
        or base.get("tree_sha256") != MODEL_PATH_TREE_SHA256
        or base.get("weights_sha256") != MODEL_PATH_WEIGHTS_SHA256
        or base.get("receipt_sha256") != MODEL_PATH_RECEIPT_SHA256
        or payload.get("fresh_adapter") is not True
        or payload.get("log_sha256") != sha256_file(log_path)
        or Path(payload.get("adapter", "")).resolve() != adapter.resolve()
        or not config.is_file()
        or not weights.is_file()
        or payload.get("adapter_config_sha256") != sha256_file(config)
        or payload.get("adapter_weights_sha256") != sha256_file(weights)
        or not validate_adapter_config(load_json(config))
        or (
            published is not None
            and (
                sha256_file(receipt_path) != published["receipt"]
                or sha256_file(log_path) != published["log"]
                or sha256_file(config) != published["adapter_config"]
                or sha256_file(weights) != published["adapter_weights"]
            )
        )
    ):
        raise ValueError(f"published {name} violates the frozen training contract")
    return {
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "adapter": str(adapter.resolve()),
        "adapter_config_sha256": payload["adapter_config_sha256"],
        "adapter_weights_sha256": payload["adapter_weights_sha256"],
        "training_git_head": payload["preflight_git_head"],
    }


def preserve_failure(path: Path, common: dict, *, reason: str, returncode: int) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps({**common, "returncode": returncode, "failure_reason": reason},
                       indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--rows", type=int, choices=LADDER_SIZES, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=LORA_RANK)
    parser.add_argument("--alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--w-think", type=float, default=0.2)
    parser.add_argument("--w-close", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    args = parser.parse_args()

    rows = args.rows
    name = arm_name(rows)
    expected_data, expected_data_hash = corpus_for(rows)
    expected_out = ADAPTER_ROOT / name
    if args.train.resolve() != expected_data.resolve():
        parser.error("training data does not match the manifest-pinned rung corpus")
    if args.out.resolve() != expected_out.resolve():
        parser.error("adapter output does not match the frozen large-artifact path")
    if args.model_path.resolve() != MODEL_PATH.resolve():
        parser.error("model path does not match the frozen base_reserialized composite")
    observed = {
        "epochs": args.epochs, "lr": args.lr, "rank": args.rank, "alpha": args.alpha,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "max_length": args.max_length, "w_think": args.w_think, "w_close": args.w_close,
        "seed": args.seed, "optimizer_steps": optimizer_steps_for(rows),
    }
    if observed != expected_hyperparameters(rows):
        parser.error("training hyperparameters differ from the frozen rung recipe")
    if not expected_data.is_file():
        parser.error(f"rung corpus absent; run scripts/run.py --stage gen-ladder first: {expected_data}")
    actual_rows = sum(1 for line in expected_data.read_text(encoding="utf-8").splitlines() if line.strip())
    if (
        sha256_file(expected_data) != expected_data_hash
        or actual_rows != rows
        or not TRAINER.is_file()
        or sha256_file(TRAINER) != TRAINER_SHA256
        or not COMPUTE_REVIEW.is_file()
        or "**Verdict:** `PASS_CONTROL_TRAINING`." not in COMPUTE_REVIEW.read_text(encoding="utf-8")
    ):
        parser.error("rung corpus, trainer, or compute-review authorization changed")
    try:
        base_auth = authenticate_base_fail_closed(full_weights=True)
    except ValueError as error:
        parser.error(str(error))

    log_path = EXP / "runs" / "training" / f"{name}.log"
    receipt_path = EXP / "runs" / "training" / f"{name}.json"
    failure_path = EXP / "runs" / "training" / f"{name}.failure.json"
    if any(p.exists() for p in (expected_out, log_path, receipt_path, failure_path)):
        parser.error("refusing to overwrite an adapter, log, or receipt")
    branch = run_text(["git", "branch", "--show-current"])
    head = run_text(["git", "rev-parse", "HEAD"])
    origin = run_text(["git", "rev-parse", "origin/main"])
    status = run_text(["git", "status", "--short"])
    if branch != "main" or head != origin or status:
        parser.error("training requires a clean pushed main checkpoint")

    command = [
        str(PYTHON), "-B", str(TRAINER),
        "--train", str(expected_data.resolve()),
        "--out", str(expected_out.resolve()),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--rank", str(args.rank),
        "--alpha", str(args.alpha),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--max-length", str(args.max_length),
        "--w-think", str(args.w_think),
        "--w-close", str(args.w_close),
        "--seed", str(args.seed),
        "--model-path", str(MODEL_PATH.resolve()),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()
    elapsed = time.perf_counter() - started
    normalize_log(log_path)
    common = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "name": name,
        "rows": rows,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": {"path": str(expected_data.resolve()), "sha256": expected_data_hash, "rows": rows},
        "train_rows": rows,
        "trainer_sha256": TRAINER_SHA256,
        "base_composite": base_auth,
        "fresh_adapter": True,
        "hyperparameters": observed,
        "command": command,
        "wall_seconds": elapsed,
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "preflight_git_head": head,
        "preflight_git_branch": branch,
        "preflight_origin_main": origin,
        "preflight_git_status": status,
        "post_training_git_status": run_text(["git", "status", "--short"]),
    }
    if returncode != 0:
        preserve_failure(failure_path, common, reason="trainer_nonzero_exit", returncode=returncode)
        raise SystemExit(f"trainer failed with exit {returncode}; preserved log/failure receipt")
    log_text = log_path.read_text(encoding="utf-8")
    encoded = re.search(r"\[train_think\] (\d+) examples \((\d+) skipped", log_text)
    loss_matches = re.findall(r"'train_loss':\s*'?([0-9.eE+-]+)'?", log_text)
    config = expected_out / "adapter_config.json"
    weights = expected_out / "adapter_model.safetensors"
    if (
        not encoded
        or int(encoded.group(1)) != rows
        or int(encoded.group(2)) != 0
        or not config.is_file()
        or not weights.is_file()
    ):
        preserve_failure(failure_path, common, reason="zero_skip_or_adapter_contract_failed", returncode=0)
        raise SystemExit("trainer output failed zero-skip or complete-adapter validation")
    if not validate_adapter_config(load_json(config)):
        preserve_failure(failure_path, common, reason="adapter_rank_alpha_or_base_identity_mismatch", returncode=0)
        raise SystemExit("adapter rank/alpha/base identity mismatch")
    packages = {
        package: importlib.metadata.version(package)
        for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")
    }
    receipt = {
        **common,
        "returncode": 0,
        "skipped_rows": 0,
        "train_loss": float(loss_matches[-1]) if loss_matches else None,
        "adapter": str(expected_out.resolve()),
        "adapter_complete": True,
        "adapter_config_sha256": sha256_file(config),
        "adapter_weights_sha256": sha256_file(weights),
        "adapter_size_bytes": weights.stat().st_size,
        "packages": packages,
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
