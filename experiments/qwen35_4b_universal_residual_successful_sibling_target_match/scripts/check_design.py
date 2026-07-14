#!/usr/bin/env python3
"""Recompute the residual successful-sibling model-free design receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import prepare_residual_input as substrate  # noqa: E402
import sibling_policy as policy  # noqa: E402


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
    expected_outputs = substrate.build_outputs()
    for path, expected in expected_outputs.items():
        if not path.is_file() or path.read_bytes() != expected:
            raise ValueError(f"residual collection substrate changed: {path}")
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
    manifest = json.loads(substrate.MANIFEST.read_text(encoding="utf-8"))
    runner = (EXP / "src" / "vllm_runner.py").read_text(encoding="utf-8")
    collector = (EXP / "scripts" / "collect_siblings.py").read_text(encoding="utf-8")
    miner = (EXP / "scripts" / "mine_successful_siblings.py").read_text(encoding="utf-8")
    harness = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
    required_runner = (
        "model_override: Path | None",
        "adapter and model_override are mutually exclusive",
        "model_override is not a merged Qwen/Qwen3.5-4B checkpoint",
        '"--model-override"',
    )
    if any(value not in runner for value in required_runner):
        raise ValueError("merged-composite runner contract changed")
    if (
        tuple(policy.EXPECTED_SKILLS)
        != ("induct", "execute", "trace", "verify", "repair", "optimize", "abstain", "state", "order", "probe")
        or policy.QUOTA_PER_SKILL != 4
        or policy.SAMPLES_PER_FAILURE != 16
        or policy.MAX_SIBLING_THINKING_TOKENS != 768
        or policy.SELECTION_SEED != 55116
    ):
        raise ValueError("residual sibling policy changed")
    if (
        '"--n", str(SAMPLES)' not in collector
        or '"--temperature", "0.6"' not in collector
        or '"--top-p", "0.95"' not in collector
        or '"--top-k", "20"' not in collector
        or '"hand_authored_oracle_targets": False' not in miner
    ):
        raise ValueError("residual sibling collection or oracle-fallback contract changed")
    if "benchmarks/" in harness or "run_benchmark" in harness:
        raise ValueError("benchmark gateway leaked into the pretraining harness")
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "model_free_residual_successful_sibling_design",
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
        "inherited_collection": manifest["origin"],
        "residual_policy": manifest["residual_policy"],
        "retention_policy": manifest["retention_policy"],
        "seeds": {
            "inherited_construction": 77115,
            "inherited_greedy": 66115,
            "sibling_collection": 66117,
            "selection": 55116,
            "training": 50,
            "local": 88012,
            "conditional_aggregate": 78142,
        },
        "sibling_collection": {
            **manifest["sibling_collection"],
            "backend": "vllm_explicit_merged_composite",
            "thinking": "natural",
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 1024,
            "max_model_len": 4096,
            "all_residual_failures_one_runner_event": True,
        },
        "availability_gate": {
            "skills": list(policy.EXPECTED_SKILLS),
            "quota_per_skill": 4,
            "selected_rows_if_reachable": 40,
            "qualification": [
                "same authenticated parent",
                "verifier-correct exact answer",
                "natural stop and closed thinking",
                "canonical answer tail",
                "thinking tokens <= 768",
            ],
            "select_shortest_qualified_per_task_then_shortest_tasks_per_skill": True,
            "deterministic_selection_seed": 55116,
            "insufficient_quota_fails_closed": True,
            "oracle_trace_fallback_forbidden": True,
        },
        "intervention": {
            "name": "residual_policy_supported_successful_sibling_distillation",
            "training_context_starts_at_original_prompt": True,
            "teacher_trajectory_sampled_from_same_parent": True,
            "residual_skills_only": True,
            "saturated_or_thin_skills_receive_replay_not_manufactured_failures": True,
            "hand_authored_oracle_trajectory": False,
        },
        "future_compute_freeze": {
            "training_authorized": False,
            "planned_rows_per_arm": 320,
            "planned_shared_position_aligned_replay_rows": 200,
            "planned_candidate_sibling_rows": 40,
            "planned_candidate_replay_filler_rows": 80,
            "planned_control_variable_replay_rows": 120,
            "planned_optimizer_steps": 40,
            "exact_match_axes": ["forward_tokens", "loss_bearing_target_tokens", "absolute_loss_mass"],
            "zero_tokenizer_skips_required": True,
            "required_before_training": [
                "committed authenticated residual sibling collection",
                "committed balanced successful-sibling source",
                "self-contained clean replay artifact",
                "exact three-axis stream feasibility receipt",
                "second adversarial compute review",
            ],
        },
        "promotion_contract": {
            "local_seed": 88012,
            "all_thirteen_skills_remain_in_local_gate": True,
            "local_gate_unchanged_from_terminal_predecessors": True,
            "candidate_must_strictly_beat_parent_and_replay_overall_and_on_execute_induct_probe": True,
            "aggregate_seed": 78142,
            "aggregate_requires_strict_lift_and_no_negative_public_family": True,
            "matched_compute_sample_more_mandatory_before_universal_claim": True,
        },
        "checkpoint_policy": {
            "next_authorized_stage": "collect-siblings",
            "one_model_or_training_stage_per_invocation": True,
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
            "substrate_sha256": sha256_file(EXP / "scripts" / "prepare_residual_input.py"),
            "policy_sha256": sha256_file(EXP / "scripts" / "sibling_policy.py"),
            "collector_sha256": sha256_file(EXP / "scripts" / "collect_siblings.py"),
            "miner_sha256": sha256_file(EXP / "scripts" / "mine_successful_siblings.py"),
            "runner_sha256": sha256_file(EXP / "src" / "vllm_runner.py"),
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
    print(json.dumps({"out": str(args.out), "sha256": hashlib.sha256(value).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
