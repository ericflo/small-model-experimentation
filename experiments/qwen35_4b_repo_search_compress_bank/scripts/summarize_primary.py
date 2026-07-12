#!/usr/bin/env python3
"""Build the compact, committed receipt for the necessary-gate result."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
DEFAULT_ARTIFACT_ROOT = ROOT / "large_artifacts" / "qwen35_4b_repo_search_compress_bank"
CANONICAL_SEQUENCE = ("INSPECT", "PATCH", "VERIFY", "COMMIT")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transition_diagnostics(trajectories: list[dict]) -> dict:
    after_failed_test: Counter[str] = Counter()
    after_passed_test: Counter[str] = Counter()
    after_failed_patch: Counter[str] = Counter()
    repeated_failed_patch_tasks = 0
    canonical = 0
    for trajectory in trajectories:
        operators = trajectory["operator_sequence"]
        canonical += tuple(operators) == CANONICAL_SEQUENCE
        failed_patch_actions: list[str] = []
        for index, step in enumerate(trajectory["steps"]):
            observation = str(step.get("observation", ""))
            if observation.startswith("ERROR: old text"):
                failed_patch_actions.append(
                    json.dumps(step.get("action"), sort_keys=True, separators=(",", ":"))
                )
            if index + 1 >= len(operators):
                continue
            next_operator = operators[index + 1]
            if observation.startswith("FAIL"):
                after_failed_test[next_operator] += 1
            elif observation.startswith("PASS"):
                after_passed_test[next_operator] += 1
            elif observation.startswith("ERROR: old text"):
                after_failed_patch[next_operator] += 1
        if len(failed_patch_actions) >= 2 and len(set(failed_patch_actions)) == 1:
            repeated_failed_patch_tasks += 1

    failed_test_total = sum(after_failed_test.values())
    passed_test_total = sum(after_passed_test.values())
    return {
        "n_trajectories": len(trajectories),
        "canonical_inspect_patch_verify_commit": {
            "count": canonical,
            "rate": canonical / len(trajectories) if trajectories else 0.0,
        },
        "after_failed_test": {
            "total": failed_test_total,
            "next_operator_counts": dict(sorted(after_failed_test.items())),
            "next_patch_rate": (
                after_failed_test["PATCH"] / failed_test_total if failed_test_total else None
            ),
        },
        "after_passed_test": {
            "total": passed_test_total,
            "next_operator_counts": dict(sorted(after_passed_test.items())),
            "next_commit_rate": (
                after_passed_test["COMMIT"] / passed_test_total if passed_test_total else None
            ),
        },
        "after_failed_patch": {
            "total": sum(after_failed_patch.values()),
            "next_operator_counts": dict(sorted(after_failed_patch.items())),
        },
        "tasks_repeating_identical_failed_patch": repeated_failed_patch_tasks,
    }


def compact_metrics(payload: dict) -> dict:
    aggregate = payload["aggregate"]
    return {
        key: aggregate[key]
        for key in (
            "n_tasks",
            "success",
            "submit_rate",
            "verified_given_success",
            "commit_given_verified",
            "invalid_action_rate_per_turn",
            "mean_sampled_tokens",
            "mean_turns",
            "per_family",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--primary-gate", type=Path, default=EXP / "analysis" / "repo_primary_gate.json")
    parser.add_argument("--out", type=Path, default=EXP / "reports" / "result_receipt.json")
    args = parser.parse_args()

    files = {
        "apex_training": args.artifact_root / "adapters" / "apex_replay" / "training_receipt.json",
        "compact_training": args.artifact_root / "adapters" / "compact" / "training_receipt.json",
        "apex_merge": args.artifact_root / "merged" / "apex_replay" / "merge_receipt.json",
        "compact_merge": args.artifact_root / "merged" / "compact" / "merge_receipt.json",
        "trained_apex": args.artifact_root / "eval" / "trained_dev_apex_replay_deep.json",
        "trained_compact": args.artifact_root / "eval" / "trained_dev_compact_deep.json",
        "transfer_apex": args.artifact_root / "eval" / "transfer_dev_apex_replay_deep.json",
        "transfer_compact": args.artifact_root / "eval" / "transfer_dev_compact_deep.json",
        "sample_more": args.artifact_root / "eval" / "transfer_dev_apex_replay_sample_more.json",
        "locality": args.artifact_root / "eval" / "locality.json",
        "primary_gate": args.primary_gate,
    }
    provenance_files = {
        "evaluation_config": EXP / "configs" / "default.yaml",
        "evaluation_script": EXP / "scripts" / "eval_repo_agent.py",
        "merge_script": EXP / "scripts" / "merge_adapter.py",
        "training_script": EXP / "scripts" / "train.py",
        "vllm_runner": EXP / "src" / "vllm_runner.py",
    }
    missing = [
        str(path)
        for path in (*files.values(), *provenance_files.values())
        if not path.exists()
    ]
    if missing:
        raise SystemExit(f"missing necessary-gate artifact(s): {missing}")

    payloads = {name: load(path) for name, path in files.items()}
    apex_training = payloads["apex_training"]
    compact_training = payloads["compact_training"]
    apex_merge = payloads["apex_merge"]
    compact_merge = payloads["compact_merge"]
    primary = payloads["primary_gate"]
    locality = payloads["locality"]
    config = yaml.safe_load(provenance_files["evaluation_config"].read_text())
    receipt = {
        "schema_version": 1,
        "status": "PRIMARY_GATE_PASSED" if primary["gate"]["passed"] else "PRIMARY_GATE_FAILED",
        "model": apex_training["model"],
        "revision": apex_training["revision"],
        "decision": primary["downstream_authorization"],
        "runtime": {
            "gpu": "NVIDIA RTX 6000 Ada Generation",
            "driver": "550.127.08",
            "gpu_memory_mib": 49140,
            "python": "3.12.3",
            "training_packages": {
                "bitsandbytes": "0.49.2",
                "peft": "0.19.1",
                "torch": "2.11.0+cu129",
                "transformers": "5.13.0",
            },
            "inference_backend": {
                "name": "vllm",
                "version": "0.24.0",
                "torch": "2.11.0+cu129",
                "transformers": "5.13.0",
                "runner_schema_version": 4,
                "same_backend_all_arms": True,
                "runtime_lora": False,
                "merged_composite_checkpoints": True,
                "engine": config["engine"],
                "evaluation": config["evaluation"],
            },
        },
        "training": {
            "apex_replay": {
                key: apex_training[key]
                for key in (
                    "optimizer_steps", "effective_batch_size", "effective_dataset_epochs",
                    "training_loss", "peak_cuda_bytes", "wall_seconds", "loss_implementation",
                )
            },
            "compact": {
                key: compact_training[key]
                for key in (
                    "optimizer_steps", "effective_batch_size", "effective_dataset_epochs",
                    "training_loss", "peak_cuda_bytes", "wall_seconds", "loss_implementation",
                    "repository_rows", "repository_tasks", "repository_operator_rows",
                )
            },
        },
        "merge": {
            "apex_replay": {
                "applied_lora_modules": apex_merge["applied_lora_modules"],
                "nonzero_lora_modules": apex_merge["nonzero_lora_modules"],
                "adapter_weights_sha256": apex_merge["adapter_weights_sha256"],
                "merged_weights_sha256": apex_merge["weight_files"][0]["sha256"],
            },
            "compact": {
                "applied_lora_modules": compact_merge["applied_lora_modules"],
                "nonzero_lora_modules": compact_merge["nonzero_lora_modules"],
                "adapter_weights_sha256": compact_merge["adapter_weights_sha256"],
                "merged_weights_sha256": compact_merge["weight_files"][0]["sha256"],
            },
        },
        "evaluation": {
            "trained_dev": {
                "apex_replay": compact_metrics(payloads["trained_apex"]),
                "compact": compact_metrics(payloads["trained_compact"]),
            },
            "transfer_dev": {
                "apex_replay": compact_metrics(payloads["transfer_apex"]),
                "compact": compact_metrics(payloads["transfer_compact"]),
                "apex_sample_more": compact_metrics(payloads["sample_more"]),
            },
            "contrasts": primary["contrasts"],
            "checks": primary["checks"],
            "locality": {
                "median_non_target_centered_logit_drift": locality["median_non_target_centered_logit_drift"],
                "ceiling": locality["ceiling"],
                "mean_entropy_delta": locality["mean_entropy_delta"],
                "passed": locality["gate"]["passed"],
            },
        },
        "policy_transition_diagnostics": {
            "trained_compact": transition_diagnostics(payloads["trained_compact"]["trajectories"]),
            "transfer_apex_replay": transition_diagnostics(payloads["transfer_apex"]["trajectories"]),
            "transfer_compact": transition_diagnostics(payloads["transfer_compact"]["trajectories"]),
        },
        "downstream": {
            "action_only_trained": (args.artifact_root / "adapters" / "action_only" / "training_receipt.json").exists(),
            "confirmation_run": any((args.artifact_root / "eval").glob("transfer_confirm_*.json")),
            "menagerie_authorized": False,
            "menagerie_seeds_consumed": 0,
        },
        "input_sha256": {
            name: sha256_file(path)
            for name, path in sorted({**files, **provenance_files}.items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
