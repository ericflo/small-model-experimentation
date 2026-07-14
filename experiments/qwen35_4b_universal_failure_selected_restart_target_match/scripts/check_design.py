#!/usr/bin/env python3
"""Recompute the model-free counterfactual-restart design receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_rollout_tasks as tasks  # noqa: E402


OUT = EXP / "data" / "design_receipt.json"
PARENT_EXP = ROOT / "experiments" / "qwen35_4b_universal_on_policy_prefix_repair_token_match"
PARENT = ROOT / "large_artifacts" / PARENT_EXP.name / "merged" / "replay_after_close"
PARENT_RECEIPT = PARENT_EXP / "runs" / "merges" / "replay_after_close.json"
PARENT_RECEIPT_SHA256 = "bc78f33218afb99b4ebd5b173f1f24aa628b20fad82d627b00529cabf911d550"
PARENT_EXTERNAL_RECEIPT_SHA256 = "aa763255cb3b05599e765948d3a3db1787d5813b1cfafbdc7e1c21653ae745a3"
PARENT_WEIGHTS_SHA256 = "7ab4c419f70135d3fe058dba6e79e3a9a61c6661d43e6acb9662f331efe36e2e"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


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
            raise ValueError(f"fresh rollout substrate changed: {path}")
    parent_receipt = json.loads(PARENT_RECEIPT.read_text(encoding="utf-8"))
    external = PARENT / "merge_receipt.json"
    weight = PARENT / "model.safetensors"
    if (
        sha256_file(PARENT_RECEIPT) != PARENT_RECEIPT_SHA256
        or parent_receipt.get("model_id") != MODEL_ID
        or parent_receipt.get("model_revision") != MODEL_REVISION
        or parent_receipt.get("name") != "replay_after_close"
        or parent_receipt.get("merge_receipt_sha256") != PARENT_EXTERNAL_RECEIPT_SHA256
        or parent_receipt.get("weight_files", [{}])[0].get("sha256") != PARENT_WEIGHTS_SHA256
        or not external.is_file()
        or sha256_file(external) != PARENT_EXTERNAL_RECEIPT_SHA256
        or not weight.is_file()
        or weight.stat().st_size != 9_078_620_536
    ):
        raise ValueError("authenticated replay-parent identity changed")
    manifest = json.loads(tasks.MANIFEST.read_text(encoding="utf-8"))
    runner = (EXP / "src" / "vllm_runner.py").read_text(encoding="utf-8")
    miner = (EXP / "scripts" / "mine_restarts.py").read_text(encoding="utf-8")
    collector = (EXP / "scripts" / "collect_parent_rollouts.py").read_text(encoding="utf-8")
    harness = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
    required_runner = (
        "model_override: Path | None",
        "adapter and model_override are mutually exclusive",
        "model_override is not a merged Qwen/Qwen3.5-4B checkpoint",
        '"--model-override"',
    )
    required_miner = (
        "QUOTA_PER_SKILL = 4",
        "THINK_BUDGET = 128",
        '"parent_prefix_in_training_context": False',
        '"target_exposure_match_pending": bool(selected)',
    )
    if any(value not in runner for value in required_runner):
        raise ValueError("merged-composite runner contract changed")
    if any(value not in miner for value in required_miner):
        raise ValueError("failure-selection or clean-restart contract changed")
    if (
        '"--model-override", str(PARENT)' not in collector
        or '"--greedy"' not in collector
        or '"--max-tokens", str(MAX_TOKENS)' not in collector
    ):
        raise ValueError("parent rollout geometry changed")
    if "benchmarks/" in harness or "run_benchmark" in harness:
        raise ValueError("benchmark gateway leaked into the pretraining harness")
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "model_free_failure_selected_restart_design",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "loaded": False, "calls": 0},
        "parent": {
            "experiment": PARENT_EXP.name,
            "arm": "replay_after_close",
            "deployment": "explicit_merged_composite",
            "tracked_receipt_sha256": PARENT_RECEIPT_SHA256,
            "external_receipt_sha256": PARENT_EXTERNAL_RECEIPT_SHA256,
            "weights_sha256": PARENT_WEIGHTS_SHA256,
            "weights_size_bytes": weight.stat().st_size,
            "runtime_lora_forbidden": True,
        },
        "seeds": {
            "construction": 77114,
            "parent_rollout": 66114,
            "failure_selection": 55114,
            "training": 48,
            "local": 88010,
            "conditional_aggregate": 78140,
        },
        "rollout_substrate": {
            "rows": manifest["rows"],
            "rows_per_skill": manifest["rows_per_skill"],
            "skills": manifest["skills"],
            "source_sha256": manifest["source"]["sha256"],
            "runner_input_sha256": manifest["runner_input"]["sha256"],
            "manifest_sha256": sha256_file(tasks.MANIFEST),
            "truth_audited": True,
            "runner_input_excludes_oracle_fields": True,
            "freshness": manifest["freshness"],
            "fresh_local_seed_materialized": False,
        },
        "rollout": {
            "backend": "vllm_explicit_merged_composite",
            "thinking": "natural",
            "greedy": True,
            "samples_per_task": 1,
            "max_tokens": 1024,
            "max_model_len": 4096,
            "all_tasks_one_runner_event": True,
        },
        "failure_selection": {
            "skills": manifest["skills"],
            "quota_per_skill": 4,
            "selected_rows_if_reachable": 52,
            "think_budget_tokens": 128,
            "eligibility": ["cap_contact", "missing_answer", "wrong_answer", "over_think_budget"],
            "hard_failures_rank_before_budget_only_failures": True,
            "deterministic_selection_seed": 55114,
            "insufficient_quota_fails_closed": True,
        },
        "intervention": {
            "name": "failure_selected_counterfactual_restart",
            "selection_is_on_policy_at_task_level": True,
            "training_context_starts_at_original_prompt": True,
            "parent_failure_prefix_in_training_context": False,
            "target_is_full_truth_audited_concise_restart": True,
            "causal_contrast": "clean restart before error versus same-parent replay continuation",
        },
        "future_compute_freeze": {
            "training_authorized": False,
            "planned_rows_per_arm": 320,
            "planned_shared_position_aligned_replay_rows": 200,
            "planned_candidate_restart_rows": 52,
            "planned_candidate_replay_filler_rows": 68,
            "planned_control_variable_replay_rows": 120,
            "planned_optimizer_steps": 40,
            "exact_match_axes": ["forward_tokens", "loss_bearing_target_tokens", "absolute_loss_mass"],
            "zero_tokenizer_skips_required": True,
            "required_before_training": [
                "committed authenticated parent rollout",
                "committed balanced failure inventory and restart source",
                "self-contained clean replay artifact",
                "exact three-axis stream feasibility receipt",
                "second adversarial compute review",
            ],
        },
        "checkpoint_policy": {
            "next_authorized_stage": "collect-parent",
            "one_stage_per_invocation": True,
            "clean_pushed_main_required": True,
            "preceding_receipt_committed_at_head": True,
            "full_check_rebase_push_two_workflow_gate_between_expensive_stages": True,
        },
        "firewall": {
            "benchmark_data_read": False,
            "benchmark_gateway_exposed": False,
            "aggregate_seed_sealed": True,
        },
        "code": {
            "task_generator_sha256": sha256_file(EXP / "scripts" / "gen_rollout_tasks.py"),
            "curriculum_generator_sha256": sha256_file(EXP / "scripts" / "gen_curriculum.py"),
            "miner_sha256": sha256_file(EXP / "scripts" / "mine_restarts.py"),
            "collector_sha256": sha256_file(EXP / "scripts" / "collect_parent_rollouts.py"),
            "runner_sha256": sha256_file(EXP / "src" / "vllm_runner.py"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = (json.dumps(build_receipt(), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != value:
            parser.error("design receipt is absent or changed")
    else:
        if args.out.exists():
            parser.error("refusing to overwrite design receipt")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(value)
    print(json.dumps({"out": str(args.out), "sha256": hashlib.sha256(value).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
