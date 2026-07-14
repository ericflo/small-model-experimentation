#!/usr/bin/env python3
"""Recompute the model-free successful-sibling design receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_collection_tasks as tasks  # noqa: E402
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
    expected_outputs = tasks.build_outputs()
    for path, expected in expected_outputs.items():
        if not path.is_file() or path.read_bytes() != expected:
            raise ValueError(f"fresh collection substrate changed: {path}")
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
    greedy = (EXP / "scripts" / "collect_greedy.py").read_text(encoding="utf-8")
    prepare = (EXP / "scripts" / "prepare_sibling_sampling.py").read_text(encoding="utf-8")
    sibling = (EXP / "scripts" / "collect_siblings.py").read_text(encoding="utf-8")
    miner = (EXP / "scripts" / "mine_successful_siblings.py").read_text(encoding="utf-8")
    harness = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
    required_runner = (
        "model_override: Path | None",
        "adapter and model_override are mutually exclusive",
        "model_override is not a merged Qwen/Qwen3.5-4B checkpoint",
        '"--model-override"',
    )
    required_policy = (
        policy.QUOTA_PER_SKILL == 4,
        policy.SAMPLES_PER_FAILURE == 16,
        policy.MAX_SIBLING_THINKING_TOKENS == 768,
        policy.SELECTION_SEED == 55115,
    )
    if any(value not in runner for value in required_runner):
        raise ValueError("merged-composite runner contract changed")
    if not all(required_policy):
        raise ValueError("successful-sibling selection policy changed")
    if (
        '"--greedy"' not in greedy
        or '"--n", "1"' not in greedy
        or '"--n", str(SAMPLES)' not in sibling
        or '"--temperature", "0.6"' not in sibling
        or '"--top-p", "0.95"' not in sibling
        or '"--top-k", "20"' not in sibling
        or '"oracle_fields_in_sibling_input": False' not in prepare
        or '"hand_authored_oracle_targets": False' not in miner
    ):
        raise ValueError("two-event collection or oracle-fallback prohibition changed")
    if "benchmarks/" in harness or "run_benchmark" in harness:
        raise ValueError("benchmark gateway leaked into the pretraining harness")
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "model_free_policy_supported_successful_sibling_design",
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
            "construction": 77115,
            "greedy_collection": 66115,
            "sibling_collection": 66116,
            "selection": 55115,
            "training": 49,
            "local": 88011,
            "conditional_aggregate": 78141,
        },
        "collection_substrate": {
            "rows": manifest["rows"],
            "rows_per_skill": manifest["rows_per_skill"],
            "skills": manifest["skills"],
            "source_sha256": manifest["source"]["sha256"],
            "greedy_input_sha256": manifest["runner_input"]["sha256"],
            "manifest_sha256": sha256_file(tasks.MANIFEST),
            "truth_audited": True,
            "model_input_excludes_oracle_fields": True,
            "freshness": manifest["freshness"],
            "fresh_local_seed_materialized": False,
        },
        "collection_events": {
            "greedy": {
                "backend": "vllm_explicit_merged_composite",
                "thinking": "natural",
                "n": 1,
                "greedy": True,
                "max_tokens": 1024,
                "max_model_len": 4096,
                "purpose": "identify hard failures only",
            },
            "siblings": {
                "conditional_on_published_greedy_failures": True,
                "backend": "vllm_explicit_merged_composite",
                "thinking": "natural",
                "n": policy.SAMPLES_PER_FAILURE,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "max_tokens": 1024,
                "max_model_len": 4096,
                "model_input_excludes_oracle_fields": True,
            },
        },
        "availability_gate": {
            "skills": list(policy.EXPECTED_SKILLS),
            "quota_per_skill": policy.QUOTA_PER_SKILL,
            "selected_rows_if_reachable": len(policy.EXPECTED_SKILLS) * policy.QUOTA_PER_SKILL,
            "greedy_eligibility": ["cap_contact", "missing_answer", "wrong_answer"],
            "sibling_eligibility": [
                "same authenticated parent",
                "verifier-correct exact answer",
                "natural stop and closed thinking",
                "canonical answer tail",
                f"thinking tokens <= {policy.MAX_SIBLING_THINKING_TOKENS}",
            ],
            "select_shortest_qualified_per_task_then_shortest_tasks_per_skill": True,
            "deterministic_selection_seed": policy.SELECTION_SEED,
            "insufficient_quota_fails_closed": True,
            "oracle_trace_fallback_forbidden": True,
        },
        "intervention": {
            "name": "policy_supported_successful_sibling_distillation",
            "selection_requires_greedy_failure": True,
            "training_context_starts_at_original_prompt": True,
            "teacher_trajectory_sampled_from_same_parent": True,
            "hand_authored_oracle_trajectory": False,
            "causal_contrast": "successful sampled sibling trajectories versus same-parent exact-exposure replay",
        },
        "future_compute_freeze": {
            "training_authorized": False,
            "planned_rows_per_arm": 320,
            "planned_shared_position_aligned_replay_rows": 200,
            "planned_candidate_sibling_rows": 52,
            "planned_candidate_replay_filler_rows": 68,
            "planned_control_variable_replay_rows": 120,
            "planned_optimizer_steps": 40,
            "exact_match_axes": ["forward_tokens", "loss_bearing_target_tokens", "absolute_loss_mass"],
            "zero_tokenizer_skips_required": True,
            "required_before_training": [
                "committed authenticated greedy collection",
                "committed oracle-free sibling sampling input",
                "committed authenticated sibling collection",
                "committed balanced successful-sibling source",
                "self-contained clean replay artifact",
                "exact three-axis stream feasibility receipt",
                "second adversarial compute review",
            ],
        },
        "promotion_contract": {
            "local_seed": 88011,
            "local_gate_unchanged_from_terminal_predecessor": True,
            "candidate_must_strictly_beat_parent_and_replay_overall_and_on_execute_induct_probe": True,
            "aggregate_seed": 78141,
            "aggregate_requires_strict_lift_and_no_negative_public_family": True,
            "successful_pilot_requires_result_separated_higher_tier_confirmation": True,
            "matched_compute_sample_more_is_mandatory_before_universal_claim": True,
        },
        "checkpoint_policy": {
            "next_authorized_stage": "collect-greedy",
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
            "task_generator_sha256": sha256_file(EXP / "scripts" / "gen_collection_tasks.py"),
            "curriculum_generator_sha256": sha256_file(EXP / "scripts" / "gen_curriculum.py"),
            "policy_sha256": sha256_file(EXP / "scripts" / "sibling_policy.py"),
            "greedy_collector_sha256": sha256_file(EXP / "scripts" / "collect_greedy.py"),
            "failure_preparer_sha256": sha256_file(EXP / "scripts" / "prepare_sibling_sampling.py"),
            "sibling_collector_sha256": sha256_file(EXP / "scripts" / "collect_siblings.py"),
            "sibling_miner_sha256": sha256_file(EXP / "scripts" / "mine_successful_siblings.py"),
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
