#!/usr/bin/env python3
"""Explicitly merge one authenticated trained arm for same-backend vLLM use."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
MERGER = (
    ROOT
    / "experiments"
    / "qwen35_4b_same_prefix_advantage_routing"
    / "scripts"
    / "merge_adapter.py"
)
MERGER_SHA256 = "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LARGE = ROOT / "large_artifacts" / EXP.name
DESIGN_RECEIPT = EXP / "data" / "local_design_receipt.json"
DESIGN_REVIEW = EXP / "reports" / "local_design_review.md"
LOCAL_SEED = 88018
LOCAL_ROWS = 124
FROZEN_ADAPTERS = {
    "replay_clean": LARGE / "adapters" / "replay_clean",
    "hygiene_explore": LARGE / "adapters" / "hygiene_explore",
}
# The merge-arms stage merges the two arms sequentially inside one pushed
# checkpoint, so later arms legitimately observe the earlier arms' freshly
# written (still-uncommitted) merge receipts and logs — and nothing else.
MERGE_STATUS_PREFIX = f"experiments/{EXP.name}/runs/merges/"
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def merge_scoped_status(status: object) -> bool:
    """True when git status is empty or only this experiment's merge artifacts."""
    if not isinstance(status, str):
        return False
    return all(
        line.startswith("?? ") and line[3:].startswith(MERGE_STATUS_PREFIX)
        for line in status.splitlines()
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
    return sha256_bytes(rendered)


def validate_merged_config(output: Path) -> None:
    config = load_json(output / "config.json")
    text_config = config.get("text_config") or {}
    fingerprint = (
        config.get("model_type"),
        tuple(config.get("architectures") or ()),
        config.get("dtype"),
        config.get("auto_map"),
        text_config.get("model_type"),
        text_config.get("vocab_size"),
        text_config.get("hidden_size"),
        text_config.get("num_hidden_layers"),
        text_config.get("dtype"),
    )
    expected = (
        "qwen3_5",
        ("Qwen3_5ForConditionalGeneration",),
        "bfloat16",
        None,
        "qwen3_5_text",
        248320,
        2560,
        32,
        "bfloat16",
    )
    tokenizer = load_json(output / "tokenizer_config.json")
    if fingerprint != expected or (
        tokenizer.get("tokenizer_class"),
        tokenizer.get("model_max_length"),
        tokenizer.get("eos_token"),
    ) != ("Qwen2Tokenizer", 262144, "<|im_end|>"):
        raise ValueError("merged composite config/tokenizer fingerprint changed")


def committed_at_head(path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return committed.returncode == 0 and committed.stdout == path.read_bytes()


def authenticate_local_design(*, require_committed: bool = True) -> dict:
    if (
        not DESIGN_RECEIPT.is_file()
        or not DESIGN_REVIEW.is_file()
        or (require_committed and not committed_at_head(DESIGN_RECEIPT))
        or (require_committed and not committed_at_head(DESIGN_REVIEW))
    ):
        raise ValueError("fresh-local design/review is absent from HEAD")
    receipt = load_json(DESIGN_RECEIPT)
    review = DESIGN_REVIEW.read_text(encoding="utf-8")
    if (
        receipt.get("experiment_id") != EXP.name
        or receipt.get("stage") != "hygiene_explore_local_gate_design"
        or receipt.get("seed") != LOCAL_SEED
        or receipt.get("rows") != LOCAL_ROWS
        or receipt.get("code_sha256", {}).get("merge") != sha256_file(Path(__file__))
        or "**Verdict:** `PASS_CONTROL_MERGE`." not in review
    ):
        raise ValueError("fresh-local design/review violates the frozen merge gate")
    return receipt


def authenticate_adapter(name: str, *, require_committed: bool = True) -> dict:
    if name not in FROZEN_ADAPTERS:
        raise ValueError("merge name is not a preregistered trained arm")
    sys.path.insert(0, str(EXP / "scripts"))
    from train_trial import validate_published_arm  # noqa: PLC0415

    checkpoint = validate_published_arm(name, require_committed=require_committed)
    if Path(checkpoint["adapter"]).resolve() != FROZEN_ADAPTERS[name].resolve():
        raise ValueError("authenticated adapter path changed")
    return checkpoint


def validate_external_merge(
    output: Path, adapter: Path, output_files: list[dict] | None = None
) -> dict:
    output_files = output_files or merged_tree_manifest(output)
    files_by_name = {row["name"]: row for row in output_files}
    receipt_path = output / "merge_receipt.json"
    payload = load_json(receipt_path)
    delta_sum = payload.get("delta_frobenius_norm_sum")
    delta_max = payload.get("delta_frobenius_norm_max")
    merge_device = payload.get("merge_device")
    if (
        payload.get("method") != "explicit_composite_lora_merge"
        or payload.get("base_model") != MODEL_ID
        or payload.get("base_revision") != MODEL_REVISION
        or Path(payload.get("adapter", "")).resolve() != adapter.resolve()
        or payload.get("adapter_config_sha256")
        != sha256_file(adapter / "adapter_config.json")
        or payload.get("adapter_weights_sha256")
        != sha256_file(adapter / "adapter_model.safetensors")
        or payload.get("applied_lora_modules") != 128
        or payload.get("nonzero_lora_modules") != 128
        or merge_device not in {"cpu", "cuda"}
        or payload.get("scale") != 2.0
        or not isinstance(delta_sum, (int, float))
        or isinstance(delta_sum, bool)
        or not math.isfinite(delta_sum)
        or delta_sum <= 0
        or not isinstance(delta_max, (int, float))
        or isinstance(delta_max, bool)
        or not math.isfinite(delta_max)
        or delta_max <= 0
        or delta_max > delta_sum
        or payload.get("fp32_tf32_allowed")
        != (False if merge_device == "cuda" else None)
    ):
        raise ValueError("merge receipt failed lineage/application checks")
    files = payload.get("weight_files", [])
    if len(files) != 1 or files[0].get("name") != "model.safetensors":
        raise ValueError("merge receipt has unexpected full-weight layout")
    for row in files:
        path = output / row["name"]
        if (
            not path.is_file()
            or files_by_name.get(row["name"], {}).get("sha256") != row["sha256"]
        ):
            raise ValueError(f"merged weight changed: {path}")
    validate_merged_config(output)
    return payload


def validate_published_merge(name: str, *, require_committed: bool = True) -> dict:
    design = authenticate_local_design(require_committed=require_committed)
    checkpoint = authenticate_adapter(name, require_committed=require_committed)
    adapter = FROZEN_ADAPTERS[name].resolve()
    output = (LARGE / "merged" / name).resolve()
    run_receipt = EXP / "runs" / "merges" / f"{name}.json"
    log_path = EXP / "runs" / "merges" / f"{name}.log"
    if (
        not run_receipt.is_file()
        or not log_path.is_file()
        or (require_committed and not committed_at_head(run_receipt))
        or (require_committed and not committed_at_head(log_path))
    ):
        raise ValueError("trained-arm merge receipt/log is absent from HEAD")
    payload = load_json(run_receipt)
    output_files = merged_tree_manifest(output)
    external = validate_external_merge(output, adapter, output_files)
    output_tree_sha256 = tree_manifest_sha256(output_files)
    if (
        payload.get("schema_version") != 2
        or payload.get("experiment_id") != EXP.name
        or payload.get("name") != name
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or Path(payload.get("adapter", "")).resolve() != adapter
        or payload.get("adapter_config_sha256")
        != checkpoint["adapter_config_sha256"]
        or payload.get("adapter_weights_sha256")
        != checkpoint["adapter_weights_sha256"]
        or payload.get("training_receipt_sha256") != checkpoint["receipt_sha256"]
        or Path(payload.get("merged", "")).resolve() != output
        or payload.get("merge_receipt_sha256")
        != sha256_file(output / "merge_receipt.json")
        or payload.get("weight_files") != external["weight_files"]
        or payload.get("output_files") != output_files
        or payload.get("output_tree_sha256") != output_tree_sha256
        or payload.get("applied_lora_modules") != 128
        or payload.get("nonzero_lora_modules") != 128
        or payload.get("merger_sha256") != MERGER_SHA256
        or payload.get("local_design_receipt_sha256")
        != sha256_file(DESIGN_RECEIPT)
        or payload.get("local_design_receipt_sha256")
        != sha256_bytes(
            json.dumps(
                design, indent=2, sort_keys=True, ensure_ascii=False
            ).encode()
            + b"\n"
        )
        or not merge_scoped_status(payload.get("preflight_git_status"))
        or payload.get("preflight_git_branch") != "main"
        or payload.get("preflight_git_head") != payload.get("preflight_origin_main")
        or payload.get("log_sha256") != sha256_file(log_path)
    ):
        raise ValueError("published trained-arm merge violates its frozen contract")
    return payload


def normalize_log(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def run_text(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def preserve_failure(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--name", choices=tuple(FROZEN_ADAPTERS), required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.name):
        parser.error("unsafe merge name")
    adapter = args.adapter.resolve()
    output = args.out.resolve()
    if adapter != FROZEN_ADAPTERS[args.name].resolve():
        parser.error("adapter path does not match the frozen merge arm")
    if output != (LARGE / "merged" / args.name).resolve():
        parser.error("merged output does not match the frozen arm path")
    if not MERGER.is_file() or sha256_file(MERGER) != MERGER_SHA256:
        parser.error("explicit composite merger changed")
    try:
        authenticate_local_design()
        checkpoint = authenticate_adapter(args.name)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    run_receipt = EXP / "runs" / "merges" / f"{args.name}.json"
    log_path = EXP / "runs" / "merges" / f"{args.name}.log"
    failure_path = EXP / "runs" / "merges" / f"{args.name}.failure.json"
    if any(path.exists() for path in (output, run_receipt, log_path, failure_path)):
        parser.error("refusing to overwrite a trained-arm merge artifact")
    branch = run_text(["git", "branch", "--show-current"])
    head = run_text(["git", "rev-parse", "HEAD"])
    origin = run_text(["git", "rev-parse", "origin/main"])
    status = run_text(["git", "status", "--short"])
    if branch != "main" or head != origin or not merge_scoped_status(status):
        parser.error(
            "merge requires a pushed main checkpoint that is clean apart from "
            "this experiment's earlier same-stage merge receipts"
        )

    command = [
        str(PYTHON),
        "-B",
        str(MERGER),
        "--adapter",
        str(adapter),
        "--out",
        str(output),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            returncode = process.wait()
    except OSError as error:
        if log_path.is_file():
            normalize_log(log_path)
        preserve_failure(
            failure_path,
            {
                "schema_version": 1,
                "experiment_id": EXP.name,
                "name": args.name,
                "failure_stage": "merger_launch",
                "error": str(error),
                "preflight_git_head": head,
                "preflight_origin_main": origin,
                "log_sha256": sha256_file(log_path) if log_path.is_file() else None,
            },
        )
        raise SystemExit(f"merger launch failed: {error}")
    normalize_log(log_path)
    if returncode != 0:
        preserve_failure(
            failure_path,
            {
                "schema_version": 1,
                "experiment_id": EXP.name,
                "name": args.name,
                "returncode": returncode,
                "preflight_git_head": head,
                "preflight_origin_main": origin,
                "log_sha256": sha256_file(log_path),
            },
        )
        raise SystemExit(f"merge failed with exit {returncode}; failure preserved")
    try:
        output_files = merged_tree_manifest(output)
        merge = validate_external_merge(output, adapter, output_files)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        preserve_failure(
            failure_path,
            {
                "schema_version": 1,
                "experiment_id": EXP.name,
                "name": args.name,
                "failure_stage": "post_merge_validation",
                "error": str(error),
                "preflight_git_head": head,
                "preflight_origin_main": origin,
                "log_sha256": sha256_file(log_path),
            },
        )
        raise SystemExit(f"merged composite validation failed: {error}")
    receipt = {
        "schema_version": 2,
        "experiment_id": EXP.name,
        "name": args.name,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter": str(adapter),
        "adapter_config_sha256": checkpoint["adapter_config_sha256"],
        "adapter_weights_sha256": checkpoint["adapter_weights_sha256"],
        "training_receipt": checkpoint["receipt"],
        "training_receipt_sha256": checkpoint["receipt_sha256"],
        "merged": str(output),
        "merge_receipt_sha256": sha256_file(output / "merge_receipt.json"),
        "weight_files": merge["weight_files"],
        "output_files": output_files,
        "output_tree_sha256": tree_manifest_sha256(output_files),
        "applied_lora_modules": merge["applied_lora_modules"],
        "nonzero_lora_modules": merge["nonzero_lora_modules"],
        "merger": str(MERGER),
        "merger_sha256": MERGER_SHA256,
        "local_design_receipt": str(DESIGN_RECEIPT.resolve()),
        "local_design_receipt_sha256": sha256_file(DESIGN_RECEIPT),
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "preflight_git_head": head,
        "preflight_git_branch": branch,
        "preflight_origin_main": origin,
        "preflight_git_status": status,
        "command": command,
        "benchmark_data_read": False,
        "aggregate_seed_open": False,
    }
    run_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
