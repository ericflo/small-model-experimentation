from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import eval_inputs as E  # noqa: E402
import provenance as P  # noqa: E402
import adapter_gate_artifacts as G  # noqa: E402
from vllm_runner import EngineConfig, SamplingConfig  # noqa: E402


class EvalInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = EXP / "configs" / "default.yaml"
        cls.config = yaml.safe_load(cls.config_path.read_text())
        cls.config_sha = hashlib.sha256(cls.config_path.read_bytes()).hexdigest()

    def test_action_receipt_reconstructs_exact_prompt_labels_and_mapping(self) -> None:
        prompts, labels = E.action_bundles(self.config, "qualification")
        receipt = E.action_receipt(self.config, self.config_sha, "qualification")
        self.assertEqual(receipt["rows"], 144)
        self.assertEqual(
            receipt["prompt_sha256"], hashlib.sha256(E.jsonl_payload(prompts)).hexdigest()
        )
        self.assertEqual(
            receipt["label_sha256"], hashlib.sha256(E.jsonl_payload(labels)).hexdigest()
        )
        mapping = E.task_metadata(self.config, "qualification")
        self.assertEqual(
            {family for family, _ in mapping.values()}, {"list", "string", "register"}
        )
        self.assertEqual({depth for _, depth in mapping.values()}, {3})

    def test_reflection_bundle_is_distinct_and_literal_action_is_reconstructible(self) -> None:
        action, _ = E.action_bundles(self.config, "qualification")
        reflection = E.reflection_prompts(self.config, "qualification")
        self.assertNotEqual(
            hashlib.sha256(E.jsonl_payload(action)).hexdigest(),
            hashlib.sha256(E.jsonl_payload(reflection)).hexdigest(),
        )
        generated = [
            {"id": row["id"], "outputs": [{"text": f"PLAN: {index}"} for index in range(4)]}
            for row in reflection
        ]
        first = E.literal_action_prompts(self.config, "qualification", generated, 4)
        second = E.literal_action_prompts(self.config, "qualification", generated, 4)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 576)
        self.assertEqual(first[0]["meta"]["input_kind"], "literal_action")

    def test_complete_sampling_dictionary_rejects_unregistered_penalties(self) -> None:
        expected = SamplingConfig(
            thinking="budget",
            thinking_budget=1024,
            n=16,
            answer_max_tokens=128,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            run_seed=88031,
        )
        sampling, resolved = P.expected_sampling(expected)
        metadata = {"sampling": sampling, "resolved_sampling": resolved}
        P.validate_sampling(metadata, expected)
        altered = json.loads(json.dumps(metadata))
        altered["sampling"]["presence_penalty"] = 1.5
        altered["resolved_sampling"]["presence_penalty"] = 1.5
        with self.assertRaisesRegex(ValueError, "sampling dictionary"):
            P.validate_sampling(altered, expected)
        altered = json.loads(json.dumps(metadata))
        altered["sampling"]["allow_custom_prompts"] = True
        with self.assertRaisesRegex(ValueError, "sampling dictionary"):
            P.validate_sampling(altered, expected)

    def test_self_consistent_forged_label_receipt_is_rejected(self) -> None:
        receipt = E.action_receipt(self.config, self.config_sha, "qualification")
        _, labels = E.action_bundles(self.config, "qualification")
        forged = [
            {**row, "answers": ["forged", "forged", "forged"]} for row in labels
        ]
        forged_payload = E.jsonl_payload(forged)
        forged_receipt = {
            **receipt,
            "label_sha256": hashlib.sha256(forged_payload).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt_path = root / "receipt.json"
            labels_path = root / "labels.jsonl"
            receipt_path.write_text(json.dumps(forged_receipt))
            labels_path.write_bytes(forged_payload)
            with self.assertRaisesRegex(ValueError, "sealed reconstruction"):
                P.validate_action_inputs(
                    config=self.config,
                    config_path=self.config_path,
                    receipt_path=receipt_path,
                    labels_path=labels_path,
                    expected_split="qualification",
                )

    def test_direct_url_vllm_pin_is_parsed_and_enforced(self) -> None:
        lock = EXP.parents[1] / "requirements-vllm.lock.txt"
        versions = P._locked_versions(lock)
        self.assertEqual(versions["vllm"], "0.24.0+cu129")
        runtime = {
            "environment_lock": {"sha256": hashlib.sha256(lock.read_bytes()).hexdigest()},
            "packages": versions,
        }
        P.validate_runtime_packages(runtime, lock)
        runtime["packages"] = {**versions, "vllm": "999.0-forged"}
        with self.assertRaisesRegex(ValueError, "installed packages"):
            P.validate_runtime_packages(runtime, lock)

    def test_missing_or_non_wheel_vllm_direct_pin_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "lock.txt"
            lock.write_text("other==1.0\n")
            with self.assertRaisesRegex(ValueError, "exact 0.24.0"):
                P._locked_versions(lock)
            lock.write_text("vllm @ https://example.invalid/vllm-source.tar.gz\n")
            with self.assertRaisesRegex(ValueError, "exact wheel pin"):
                P._locked_versions(lock)

    def test_standalone_adapter_pass_receipt_lacks_replayable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adapter-gate.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "experiment_id": self.config["experiment_id"],
                        "arm": "reflection_correct",
                        "seed": 47,
                        "pass": True,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "replayable invocation"):
                G.validate_adapter_gate_artifact(
                    path,
                    config=self.config,
                    config_path=self.config_path,
                    experiment_root=EXP,
                )

    def test_false_remote_code_runner_metadata_passes_full_protocol_validation(self) -> None:
        frozen = self.config["evaluation"]["engine"]
        engine = P.canonical(
            EngineConfig(
                max_model_len=int(frozen["max_model_len"]),
                gpu_memory_utilization=float(frozen["gpu_memory_utilization"]),
                max_num_seqs=int(frozen["max_num_seqs"]),
                max_num_batched_tokens=int(frozen["max_num_batched_tokens"]),
                cudagraph_capture_sizes=tuple(frozen["cudagraph_capture_sizes"]),
                enable_prefix_caching=False,
                enforce_eager=False,
            )
        )
        engine_args = {
            "model": self.config["model"]["id"],
            "tokenizer": self.config["model"]["id"],
            "tokenizer_revision": self.config["model"]["revision"],
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_model_len": int(frozen["max_model_len"]),
            "gpu_memory_utilization": float(frozen["gpu_memory_utilization"]),
            "max_num_seqs": int(frozen["max_num_seqs"]),
            "max_num_batched_tokens": int(frozen["max_num_batched_tokens"]),
            "language_model_only": True,
            "enable_prefix_caching": False,
            "mamba_cache_mode": "none",
            "enforce_eager": False,
            "generation_config": "vllm",
            "max_logprobs": 20,
            "seed": 0,
            "async_scheduling": False,
            "cudagraph_capture_sizes": list(frozen["cudagraph_capture_sizes"]),
            "max_cudagraph_capture_size": max(frozen["cudagraph_capture_sizes"]),
            "revision": self.config["model"]["revision"],
        }
        lock = EXP.parents[1] / "requirements-vllm.lock.txt"
        commit = P.subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = root / "generated.jsonl"
            generated.write_text('{"id":"synthetic"}\n')
            stage = root / "stage.json"
            stage.write_text("{}\n")
            exact_base = {"synthetic": "base"}
            exact_tokenizer = {"synthetic": "tokenizer"}
            metadata = {
                "output": {
                    "description": "generated JSONL",
                    "sha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
                    "rows": 1,
                },
                "runner_sha256": hashlib.sha256(
                    (EXP / "src" / "vllm_runner.py").read_bytes()
                ).hexdigest(),
                "base_model": self.config["model"]["id"],
                "model_revision": self.config["model"]["revision"],
                "model_override": None,
                "adapter": None,
                "base_snapshot": exact_base,
                "tokenizer_snapshot": exact_tokenizer,
                "post_load_integrity": {
                    "base_snapshot": exact_base,
                    "tokenizer_snapshot": exact_tokenizer,
                    "model_override": None,
                    "decision": "POST_ENGINE_BYTES_MATCH_PRELOAD_COMMITMENTS",
                },
                "engine": engine,
                "engine_args": engine_args,
                "runtime": {
                    "git_dirty": False,
                    "git_commit": commit,
                    "git_root": str(EXP.parents[1].resolve()),
                    "cwd": str(EXP.parents[1].resolve()),
                    "git_head_mode": "detached",
                    "environment_lock": {
                        "sha256": hashlib.sha256(lock.read_bytes()).hexdigest()
                    },
                    "packages": P._locked_versions(lock),
                },
                "generation_stage": {
                    "authorized_stage": "calibration_generation",
                    "stage_receipt_path": str(stage.resolve()),
                    "stage_receipt_sha256": hashlib.sha256(stage.read_bytes()).hexdigest(),
                    "config_sha256": self.config_sha,
                    "issuer_git_commit": commit,
                    "split": "calibration",
                    "input_kind": "action",
                    "source_seed": None,
                },
                "capacity_preflight": {
                    "decision": "LIVE_KV_CAPACITY_PASS",
                    "engine": engine,
                    "live_cache": {"synthetic": True},
                },
                "resolved_cudagraph": {"synthetic": True},
            }
            with mock.patch(
                "stages.read_and_validate_stage_receipt", return_value={}
            ), mock.patch(
                "merge_replay.base_snapshot_commitment", return_value=exact_base
            ), mock.patch(
                "tokenizer_lineage.authenticate_tokenizer_snapshot",
                return_value=exact_tokenizer,
            ):
                protocol = P.validate_generation_protocol(
                    metadata=metadata,
                    config=self.config,
                    experiment_root=EXP,
                    generated_path=generated,
                    expected_rows=1,
                    expect_merged=False,
                    expected_stage="calibration_generation",
                    expected_split="calibration",
                    expected_input_kind="action",
                    expected_source_seed=None,
                )
            self.assertEqual(len(protocol), 64)
            metadata["engine_args"]["trust_remote_code"] = True
            with mock.patch(
                "merge_replay.base_snapshot_commitment", return_value=exact_base
            ), mock.patch(
                "tokenizer_lineage.authenticate_tokenizer_snapshot",
                return_value=exact_tokenizer,
            ), self.assertRaisesRegex(ValueError, "engine arguments"):
                P.validate_generation_protocol(
                    metadata=metadata,
                    config=self.config,
                    experiment_root=EXP,
                    generated_path=generated,
                    expected_rows=1,
                    expect_merged=False,
                    expected_stage="calibration_generation",
                    expected_split="calibration",
                    expected_input_kind="action",
                    expected_source_seed=None,
                )


if __name__ == "__main__":
    unittest.main()
