#!/usr/bin/env python3
"""Prove that a merged incumbent is loaded and behaviorally nonzero."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import read_jsonl, sha256_file, write_json  # noqa: E402


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "incumbent_install_gate.json"
    )
    args = parser.parse_args()
    base_rows = {row["id"]: row for row in read_jsonl(args.base)}
    candidate_rows = {row["id"]: row for row in read_jsonl(args.candidate)}
    if set(base_rows) != set(candidate_rows):
        raise SystemExit("canary row mismatch")
    changed = []
    prompt_equal = []
    for key in sorted(base_rows):
        left, right = base_rows[key], candidate_rows[key]
        prompt_equal.append(left["prompt_sha256"] == right["prompt_sha256"])
        changed.append(
            left["outputs"][0]["token_ids"] != right["outputs"][0]["token_ids"]
        )
    base_meta_path = args.base.with_name(args.base.name + ".meta.json")
    candidate_meta_path = args.candidate.with_name(args.candidate.name + ".meta.json")
    base_meta = json.loads(base_meta_path.read_text(encoding="utf-8"))
    candidate_meta = json.loads(candidate_meta_path.read_text(encoding="utf-8"))
    merge_path = args.merged / "merge_receipt.json"
    merge = json.loads(merge_path.read_text(encoding="utf-8"))
    base_engine = dict(base_meta.get("engine", {}))
    candidate_engine = dict(candidate_meta.get("engine", {}))
    base_engine.pop("model_override", None)
    candidate_engine.pop("model_override", None)
    checks = {
        "paired_prompts": all(prompt_equal),
        "base_is_pinned": (
            base_meta.get("model") == MODEL_ID
            and base_meta.get("model_revision") == MODEL_REVISION
        ),
        "candidate_is_local_composite": (
            candidate_meta.get("model") == str(args.merged.resolve())
            and candidate_meta.get("model_revision") is None
            and bool(candidate_meta.get("model_config_sha256"))
        ),
        "nonzero_mapped_delta": int(merge.get("nonzero_lora_modules", 0)) > 0,
        "behavior_changed": any(changed),
        "runner_parity": base_meta.get("runner_sha256") == candidate_meta.get("runner_sha256"),
        "engine_parity": base_engine == candidate_engine,
        "sampling_parity": (
            base_meta.get("sampling") == candidate_meta.get("sampling")
            and base_meta.get("resolved_sampling") == candidate_meta.get("resolved_sampling")
        ),
        "cudagraph_parity": (
            base_meta.get("resolved_cudagraph")
            == candidate_meta.get("resolved_cudagraph")
        ),
        "environment_lock_parity": (
            base_meta.get("runtime", {}).get("environment_lock")
            == candidate_meta.get("runtime", {}).get("environment_lock")
        ),
    }
    result = {
        "stage": "incumbent_installation_gate",
        "base_output": str(args.base.resolve()),
        "candidate_output": str(args.candidate.resolve()),
        "merged": str(args.merged.resolve()),
        "base_metadata_sha256": sha256_file(base_meta_path),
        "candidate_metadata_sha256": sha256_file(candidate_meta_path),
        "merge_receipt_sha256": sha256_file(merge_path),
        "paired_canaries": len(changed),
        "changed_canaries": sum(changed),
        "changed_by_id": {
            key: value for key, value in zip(sorted(base_rows), changed)
        },
        "gate": {"passed": all(checks.values()), "checks": checks},
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
