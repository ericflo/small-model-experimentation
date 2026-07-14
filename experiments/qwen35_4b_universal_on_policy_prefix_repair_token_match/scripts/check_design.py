#!/usr/bin/env python3
"""Recompute the model-free on-policy collection design receipt."""

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

import gen_rollout_tasks as tasks  # noqa: E402


OUT = EXP / "data" / "design_receipt.json"
PARENT = (
    ROOT
    / "large_artifacts"
    / "qwen35_4b_universal_close_weight_token_match"
    / "adapters"
    / "close_xi"
)
REPLAY = EXP / "data" / "sft_blend.jsonl"
PREDECESSOR_MANIFEST = EXP / "data" / "predecessor_stream_manifest.json"
PARENT_WEIGHTS_SHA256 = "16e9dc75a0e33e182e916600ff6e1d75fc46dfa45e870216e2c149a41253c179"
PARENT_CONFIG_SHA256 = "de953bd57502ff728a12d1627d5aacab6284b045428ec7b83026388afd8c47ff"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
PREDECESSOR_MANIFEST_SHA256 = "abf8b5055e68c0fb2bb6e32a29f7be3b3677a0dd179e77397647777a2aa0966f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_receipt() -> dict:
    expected_outputs = tasks.build_outputs()
    for path, expected in expected_outputs.items():
        if not path.is_file() or path.read_bytes() != expected:
            raise ValueError(f"rollout task artifact changed: {path}")
    source_rows = [
        json.loads(line)
        for line in tasks.SOURCE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    tasks.validate(source_rows)
    public_rows = [
        json.loads(line)
        for line in tasks.RUNNER_INPUT.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if any(set(row) != {"id", "messages", "meta"} for row in public_rows):
        raise ValueError("model-facing rollout input exposes hidden fields")
    if (
        sha256_file(PARENT / "adapter_model.safetensors") != PARENT_WEIGHTS_SHA256
        or sha256_file(PARENT / "adapter_config.json") != PARENT_CONFIG_SHA256
        or sha256_file(REPLAY) != REPLAY_SHA256
        or sha256_file(PREDECESSOR_MANIFEST) != PREDECESSOR_MANIFEST_SHA256
    ):
        raise ValueError("parent or inherited clean replay identity changed")

    trainer = (EXP / "scripts" / "train_think.py").read_text(encoding="utf-8")
    miner = (EXP / "scripts" / "mine_prefix_repairs.py").read_text(encoding="utf-8")
    runner = (EXP / "src" / "vllm_runner.py").read_text(encoding="utf-8")
    harness = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
    required_trainer_contract = (
        'rec.get("assistant_prefix_token_ids")',
        'rec.get("prefix_loss_masked") is not True',
        '[0.0] * (len(pid) + len(prefix_ids))',
    )
    if any(value not in trainer for value in required_trainer_contract):
        raise ValueError("masked-prefix trainer contract changed")
    if (
        "QUOTA_PER_CLASS = 10" not in miner
        or "COMMIT_THINK_LIMIT = 32" not in miner
        or '"prefix_loss_masked": True' not in miner
    ):
        raise ValueError("failure mining or masking freeze changed")
    if (
        "model_override: Path | None" not in runner
        or '"model_override is not a merged Qwen/Qwen3.5-4B checkpoint' not in runner
        or 'parser.add_argument(\n        "--model-override"' not in runner
    ):
        raise ValueError("merged-composite vLLM gate changed")
    if "benchmarks/" in harness or "run_benchmark" in harness:
        raise ValueError("benchmark access leaked into the pretraining harness")

    counts = Counter(row["failure_class"] for row in source_rows)
    quota = 10
    reachability = {
        name: {
            "available_tasks": counts[name],
            "required_reachable_failures": quota,
            "structurally_reachable": counts[name] >= quota,
        }
        for name in tasks.FAILURE_CLASSES
    }
    if not all(value["structurally_reachable"] for value in reachability.values()):
        raise ValueError("a failure quota is structurally unreachable")
    manifest = json.loads(tasks.MANIFEST.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "model_free_on_policy_collection_design",
        "model": {
            "id": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "loaded": False,
            "calls": 0,
        },
        "parent": {
            "experiment": "qwen35_4b_universal_close_weight_token_match",
            "arm": "close_xi",
            "weights_sha256": PARENT_WEIGHTS_SHA256,
            "config_sha256": PARENT_CONFIG_SHA256,
            "deployment": "explicit_composite_merge_required",
            "runtime_lora_forbidden": True,
        },
        "seeds": {
            "construction": 77113,
            "parent_rollout": 66113,
            "training": 47,
            "local": 88009,
            "conditional_aggregate": 78139,
        },
        "rollout_tasks": {
            "rows": len(source_rows),
            "rows_per_failure_class": 48,
            "failure_classes": list(tasks.FAILURE_CLASSES),
            "counts": dict(sorted(counts.items())),
            "source_sha256": sha256_file(tasks.SOURCE),
            "runner_input_sha256": sha256_file(tasks.RUNNER_INPUT),
            "manifest_sha256": sha256_file(tasks.MANIFEST),
            "runner_input_excludes_hidden_oracle_fields": manifest[
                "runner_input_excludes_hidden_oracle_fields"
            ],
            "truth_audits_present": True,
            "fresh_local_seed_materialized": False,
        },
        "failure_selection": {
            "quota_per_class": quota,
            "selected_rows_if_reachable": quota * len(tasks.FAILURE_CLASSES),
            "reachability": reachability,
            "failure_only": True,
            "masked_parent_prefix": True,
            "commit_think_limit": 32,
            "insufficient_quota_fails_closed": True,
        },
        "rollout": {
            "backend": "vllm_merged_composite",
            "thinking": "natural",
            "greedy": True,
            "samples_per_task": 1,
            "max_tokens": 1024,
            "all_tasks_one_batched_runner_event": True,
            "metadata_sidecar_required": True,
        },
        "future_compute_freeze": {
            "training_authorized": False,
            "reason": "actual prefix lengths and class reachability are unobserved",
            "required_before_training": [
                "committed parent rollout receipt",
                "committed failure inventory and correction source",
                "exact forward-token matched replay streams",
                "zero-skip tokenizer receipt",
                "second adversarial compute review",
            ],
            "planned_rows_per_arm": 320,
            "planned_shared_position_aligned_replay_rows": 200,
            "planned_optimizer_steps": 40,
        },
        "sources": {
            "clean_replay_sha256": REPLAY_SHA256,
            "predecessor_partition_sha256": PREDECESSOR_MANIFEST_SHA256,
            "runner_sha256": sha256_file(EXP / "src" / "vllm_runner.py"),
            "trainer_sha256": sha256_file(EXP / "scripts" / "train_think.py"),
        },
        "checkpoint_policy": {
            "next_authorized_stage": "merge-parent",
            "one_stage_per_invocation": True,
            "clean_worktree_required": True,
            "preceding_receipt_committed_at_head": True,
            "merge_then_rollout_then_mine": True,
        },
        "firewall": {
            "benchmark_data_read": False,
            "benchmark_gateway_exposed": False,
            "aggregate_seed_sealed": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = (
        json.dumps(build_receipt(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != value:
            parser.error("design receipt is absent or changed")
    else:
        if args.out.exists():
            parser.error("refusing to overwrite design receipt")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(value)
    print(
        json.dumps(
            {"out": str(args.out), "sha256": hashlib.sha256(value).hexdigest()},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
