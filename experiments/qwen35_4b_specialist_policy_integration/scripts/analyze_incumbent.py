#!/usr/bin/env python3
"""Validate the regenerated common-root incumbent before task calibration."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, resolve_repo_path, sha256_file, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--encoding-audit", type=Path, required=True)
    parser.add_argument("--install-gate", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "incumbent_gate.json"
    )
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    training_path = args.adapter / "training_receipt.json"
    merge_path = args.merged / "merge_receipt.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    audit = json.loads(args.encoding_audit.read_text(encoding="utf-8"))
    install = json.loads(args.install_gate.read_text(encoding="utf-8"))
    train_cfg = config["incumbent_train"]
    source_data = resolve_repo_path(config["model"]["incumbent_data"])
    receipt_train_files = training.get("train_files", [])
    checks = {
        "pinned_source_model": (
            training.get("source_model") == config["model"]["id"]
            and training.get("source_model_revision") == config["model"]["revision"]
        ),
        "source_data_hash": (
            len(receipt_train_files) == 1
            and receipt_train_files[0].get("sha256") == sha256_file(source_data)
            and audit.get("train_files", [{}])[0].get("sha256") == sha256_file(source_data)
        ),
        "encoding_counts_match": (
            int(training.get("input_rows", -1)) == int(audit.get("input_rows", -2))
            and int(training.get("encoded_rows", -1)) == int(audit.get("encoded_rows", -2))
            and int(training.get("skipped_rows", -1)) == int(audit.get("skipped_rows", -2))
        ),
        "skip_rate_below_cap": float(audit.get("skip_rate", 1.0)) <= 0.15,
        "frozen_hyperparameters": (
            math.isclose(float(training.get("epochs", -1)), float(train_cfg["epochs"]))
            and math.isclose(
                float(training.get("learning_rate", -1)),
                float(train_cfg["learning_rate"]),
            )
            and int(training.get("rank", -1)) == int(train_cfg["rank"])
            and int(training.get("alpha", -1)) == int(train_cfg["alpha"])
            and int(training.get("batch_size", -1)) == int(train_cfg["batch_size"])
            and int(training.get("grad_accum", -1)) == int(train_cfg["grad_accum"])
            and int(training.get("max_length", -1)) == int(train_cfg["max_length"])
        ),
        "optimizer_ran": int(training.get("optimizer_steps", 0)) > 0,
        "nonzero_explicit_merge": (
            int(merge.get("applied_lora_modules", 0)) > 0
            and int(merge.get("nonzero_lora_modules", 0))
            == int(merge.get("applied_lora_modules", -1))
            and merge.get("merge_device") == "cuda"
            and merge.get("fp32_tf32_allowed") is False
        ),
        "behavioral_installation": bool(install.get("gate", {}).get("passed")),
        "install_targets_this_merge": (
            install.get("merge_receipt_sha256") == sha256_file(merge_path)
        ),
    }
    result = {
        "stage": "incumbent_gate",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "training_receipt": str(training_path.resolve()),
        "training_receipt_sha256": sha256_file(training_path),
        "merge_receipt": str(merge_path.resolve()),
        "merge_receipt_sha256": sha256_file(merge_path),
        "encoding_audit": str(args.encoding_audit.resolve()),
        "encoding_audit_sha256": sha256_file(args.encoding_audit),
        "install_gate": str(args.install_gate.resolve()),
        "install_gate_sha256": sha256_file(args.install_gate),
        "input_rows": audit.get("input_rows"),
        "encoded_rows": audit.get("encoded_rows"),
        "skipped_rows": audit.get("skipped_rows"),
        "skip_rate": audit.get("skip_rate"),
        "gate": {"passed": all(checks.values()), "checks": checks},
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
