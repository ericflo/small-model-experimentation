from __future__ import annotations

import copy
import dataclasses
import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from calibration_stage import (  # noqa: E402
    CALIBRATION_RELATIVE_READS,
    INVOCATION_ORDER,
    BoundaryAuthenticationError,
    RUNTIME_METADATA_KEYS,
    authenticate_bundle_engine_preflight,
    authenticate_pair_thought_reuse,
    authenticate_thought_bundle,
    calibration_registrations,
    engine_config,
    load_analysis_tokenizer,
    load_calibration_inputs,
    sampling_configs,
    score_calibration_bundles,
)
from transactions import canonical_sha256, sha256_file  # noqa: E402
from vllm_runner import _stable_seed  # noqa: E402


class CalibrationStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_calibration_inputs()
        cls.tokenizer = load_analysis_tokenizer(cls.inputs)

    def test_loader_reads_only_registered_calibration_data(self) -> None:
        self.assertEqual(tuple(self.inputs.read_receipt), CALIBRATION_RELATIVE_READS)
        self.assertEqual(len(self.inputs.records), 48)
        self.assertEqual(
            {row["meta"]["arity"] for row in self.inputs.records}, {2, 3}
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in CALIBRATION_RELATIVE_READS:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(EXP / relative, destination)
            runner = root / "src/vllm_runner.py"
            runner.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(EXP / "src/vllm_runner.py", runner)
            loaded = load_calibration_inputs(root)
        self.assertEqual(loaded.records, self.inputs.records)
        self.assertEqual(loaded.read_receipt, self.inputs.read_receipt)

    def test_engine_sampling_and_invocation_geometry_are_frozen(self) -> None:
        engine = engine_config(self.inputs)
        self.assertEqual(engine.max_model_len, 4096)
        self.assertEqual(engine.max_num_seqs, 64)
        self.assertFalse(engine.enable_prefix_caching)
        plan = sampling_configs(self.inputs)
        self.assertEqual(tuple(plan), INVOCATION_ORDER)
        for name, sampling in plan.items():
            self.assertEqual(sampling.n, 1)
            self.assertEqual(sampling.answer_max_tokens, 24)
            self.assertTrue(sampling.paired_answer_seed)
            self.assertEqual(sampling.temperature, 0.6)
            self.assertEqual(sampling.top_p, 0.95)
            self.assertEqual(sampling.top_k, 20)
            if name.startswith("think512_") or name == "calibration_thoughts":
                self.assertEqual(sampling.thinking, "budget")
                self.assertEqual(sampling.thinking_budget, 512)
                self.assertTrue(sampling.force_answer_seam)
            else:
                self.assertEqual(sampling.thinking, "off")
        self.assertEqual(
            plan["no_think_program_slot_pairs"].answer_prefix, "PROGRAM:"
        )
        self.assertEqual(plan["no_think_freeform_pairs"].answer_prefix, "")

    def test_transaction_registrations_bind_all_authorization_files(self) -> None:
        registrations = calibration_registrations(inputs=self.inputs)
        self.assertEqual(tuple(registrations), INVOCATION_ORDER)
        for value in registrations.values():
            self.assertEqual(value["expected_rows"], 48)
            self.assertEqual(
                set(value["authorization_paths"]),
                {"config", "preoutcome", "tokenizer_receipt"},
            )

    def thought_bundle(self) -> dict:
        sampling = sampling_configs(self.inputs)["calibration_thoughts"]
        rows = []
        total_prompt = 0
        for record in self.inputs.records:
            prompt = self.inputs.tokenizer_receipt["calibration_prompt_token_ids"][
                record["id"]
            ]["think512"]
            sampled = [50, 248044]
            retained = [50]
            seed = _stable_seed(sampling.run_seed, record["id"], -1, "thought")
            rows.append(
                {
                    "id": record["id"],
                    "meta": record["meta"],
                    "prompt_sha256": prompt["prompt_text_sha256"],
                    "effective_prompt_sha256": hashlib.sha256(
                        b"".join(value.to_bytes(4, "big") for value in prompt["token_ids"])
                    ).hexdigest(),
                    "n_prompt_tokens": len(prompt["token_ids"]),
                    "n_original_prompt_tokens": len(prompt["token_ids"]),
                    "prompt_channel": "thinking",
                    "answer_prefix_token_ids": [],
                    "prompt_logprobs": None,
                    "outputs": [
                        {
                            "sample_index": 0,
                            "stage1_parent_seed": seed,
                            "seed_stage1": seed,
                            "seed_stage2": None,
                            "seed_domain_stage1": "thought",
                            "seed_domain_stage2": None,
                            "text": self.tokenizer.decode(
                                retained, skip_special_tokens=False
                            ),
                            "token_ids": retained,
                            "stage1_token_ids": sampled,
                            "retained_thinking_token_ids": retained,
                            "answer_prefix_token_ids": [],
                            "injected_token_ids": [],
                            "stage2_token_ids": [],
                            "n_thinking_tokens": 1,
                            "n_answer_tokens": 0,
                            "n_sampled_tokens": 2,
                            "n_injected_tokens": 0,
                            "n_completion_tokens": 1,
                            "n_terminal_tokens_trimmed": 1,
                            "n_tokens_discarded_after_close": 0,
                            "n_stage1_prompt_tokens": len(prompt["token_ids"]),
                            "n_stage2_prompt_tokens": 0,
                            "thinking_closed": False,
                            "forced_close": False,
                            "finish_reason": "stop",
                            "stop_reason": 248044,
                            "stage1_finish_reason": "stop",
                            "stage1_stop_reason": 248044,
                            "truncated": False,
                        }
                    ],
                }
            )
            total_prompt += len(prompt["token_ids"])
        sampled_total = 2 * len(rows)
        metadata = {
            "schema_version": 6,
            "generation_mode": "shared_thought_prefixes",
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "runner_sha256": sha256_file(EXP / "src/vllm_runner.py"),
            "sampling": dataclasses.asdict(sampling),
            "termination": {"hf_model_eos_token_id": 248044},
            "counts": {
                "requests": 48,
                "completions": 48,
                "unique_input_prompt_tokens": total_prompt,
                "stage1_logical_prompt_tokens": total_prompt,
                "stage2_logical_prompt_tokens": 0,
                "logical_model_input_tokens": total_prompt,
                "logical_prompt_tokens": total_prompt,
                "physical_prompt_tokens": total_prompt,
                "reused_prompt_tokens": 0,
                "sampled_tokens": sampled_total,
                "physical_sampled_tokens": sampled_total,
                "reused_sampled_tokens": 0,
                "logical_model_tokens": total_prompt + sampled_total,
                "physical_model_tokens": total_prompt + sampled_total,
                "reused_model_tokens": 0,
                "injected_tokens": 0,
            },
        }
        return {
            "schema_version": 1,
            "invocation": "calibration_thoughts",
            "rows": rows,
            "runner_metadata": metadata,
        }

    def test_thought_authenticator_checks_tokens_seeds_text_and_cost(self) -> None:
        bundle = self.thought_bundle()
        receipt = authenticate_thought_bundle(
            bundle, inputs=self.inputs, tokenizer=self.tokenizer
        )
        self.assertEqual(
            receipt["decision"], "SHARED_THOUGHT_TRANSACTION_AUTHENTICATED"
        )
        expected_source = hashlib.sha256(
            __import__("json").dumps(
                {
                    "rows": bundle["rows"],
                    "runner_metadata": bundle["runner_metadata"],
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        self.assertEqual(receipt["source_sha256"], expected_source)
        mutations = []
        bad = copy.deepcopy(bundle)
        bad["rows"][0]["outputs"][0]["retained_thinking_token_ids"] = []
        mutations.append((bad, "token/seed"))
        bad = copy.deepcopy(bundle)
        bad["rows"][0]["outputs"][0]["seed_stage1"] += 1
        mutations.append((bad, "token/seed"))
        bad = copy.deepcopy(bundle)
        bad["rows"][0]["outputs"][0]["text"] = "mutated"
        mutations.append((bad, "token/seed"))
        bad = copy.deepcopy(bundle)
        bad["runner_metadata"]["counts"]["sampled_tokens"] += 1
        mutations.append((bad, "cost summary"))
        for bad, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(
                BoundaryAuthenticationError, message
            ):
                authenticate_thought_bundle(
                    bad, inputs=self.inputs, tokenizer=self.tokenizer
                )

    def test_pair_outputs_are_bound_directly_to_persisted_thoughts(self) -> None:
        thought = self.thought_bundle()
        source_sha = hashlib.sha256(
            __import__("json").dumps(
                {
                    "rows": thought["rows"],
                    "runner_metadata": thought["runner_metadata"],
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        pair = {
            "rows": [
                {
                    "id": row["id"],
                    "meta": row["meta"],
                    "outputs": [
                        copy.deepcopy(row["outputs"][0]),
                        copy.deepcopy(row["outputs"][0]),
                    ],
                }
                for row in thought["rows"]
            ],
            "runner_metadata": {"thought_source_sha256": source_sha},
        }
        authenticate_pair_thought_reuse(pair, thought)
        for value in pair["rows"][0]["outputs"]:
            value["stage1_token_ids"] = [99, 248044]
            value["retained_thinking_token_ids"] = [99]
        with self.assertRaisesRegex(
            BoundaryAuthenticationError, "differs from persisted thought"
        ):
            authenticate_pair_thought_reuse(pair, thought)

    def test_scoring_refuses_an_incomplete_bundle_inventory(self) -> None:
        with self.assertRaisesRegex(
            BoundaryAuthenticationError, "bundle inventory changed"
        ):
            score_calibration_bundles(
                {"calibration_thoughts": self.thought_bundle()},
                inputs=self.inputs,
                tokenizer=self.tokenizer,
                live_preflight={},
            )

    def test_bundle_engine_is_bound_absolutely_to_live_preflight(self) -> None:
        runtime = {key: f"value-{key}" for key in RUNTIME_METADATA_KEYS}
        runtime["git_dirty"] = False
        engine = {"max_model_len": 4096}
        engine_args = {"model": "Qwen/Qwen3.5-4B", "seed": 0}
        resolved = {"cudagraph_capture_sizes": [1, 2]}
        preflight = {
            "engine": engine,
            "engine_args_sha256": canonical_sha256(engine_args),
            "resolved_cudagraph": resolved,
            "resolved_logprobs_mode": "raw_logprobs",
            "adapter": None,
            "rng_isolation": {
                "engine_seed": 0,
                "caller_global_rng_state_restored": True,
            },
            "runtime": runtime,
        }
        bundle = {
            "runner_metadata": {
                "engine": copy.deepcopy(engine),
                "engine_args": copy.deepcopy(engine_args),
                "resolved_cudagraph": copy.deepcopy(resolved),
                "resolved_logprobs_mode": "raw_logprobs",
                "adapter": None,
                "rng_isolation": {
                    "engine_seed": 0,
                    "caller_global_rng_state_restored": True,
                },
                "runtime": {**copy.deepcopy(runtime), "git_dirty": True},
            }
        }
        authenticate_bundle_engine_preflight(bundle, preflight)
        bad = copy.deepcopy(bundle)
        bad["runner_metadata"]["engine"]["max_model_len"] = 8192
        with self.assertRaisesRegex(BoundaryAuthenticationError, "live engine"):
            authenticate_bundle_engine_preflight(bad, preflight)
        bad = copy.deepcopy(bundle)
        bad["runner_metadata"]["runtime"]["gpu"] = "different"
        with self.assertRaisesRegex(BoundaryAuthenticationError, "live runtime"):
            authenticate_bundle_engine_preflight(bad, preflight)
        bad = copy.deepcopy(bundle)
        bad["runner_metadata"]["runtime"]["git_dirty"] = False
        with self.assertRaisesRegex(BoundaryAuthenticationError, "live runtime"):
            authenticate_bundle_engine_preflight(bad, preflight)
        for field, value in (
            ("adapter", {"path": "forged"}),
            (
                "rng_isolation",
                {"engine_seed": 1, "caller_global_rng_state_restored": False},
            ),
        ):
            bad = copy.deepcopy(bundle)
            bad["runner_metadata"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                BoundaryAuthenticationError, "live engine"
            ):
                authenticate_bundle_engine_preflight(bad, preflight)


if __name__ == "__main__":
    unittest.main()
