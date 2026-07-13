from __future__ import annotations

import copy
import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

from analyze_confirmation import (  # noqa: E402
    _cross_score_protocol_checks,
    _require_policy_score_protocol,
)
import analyze_confirmation  # noqa: E402
import authorize_benchmark  # noqa: E402
import bench  # noqa: E402
import harness  # noqa: E402
import io_utils  # noqa: E402
from confirmation_protocol import (  # noqa: E402
    ConfirmationRunner,
    PINNED_CUDA_TOOLKIT,
    PINNED_GPU,
    PINNED_LOCK,
    PINNED_LOCK_SHA256,
    PINNED_PLATFORM,
    PINNED_PYTHON,
    RUNNER,
    canonical_backend_protocol,
    capacity_receipt,
    expected_confirmation_sampling_protocol,
    hybrid_blocks,
    live_cache_capacity,
    validate_capacity_receipt,
    validate_confirmation_sampling_protocol,
    validate_live_cache_capacity,
)
from io_utils import (  # noqa: E402
    canonical_hash,
    confirmation_evaluator_source_inventory,
    sha256_file,
)


ENGINE = {
    "max_model_len": 16384,
    "gpu_memory_utilization": 0.85,
    "max_num_seqs": 48,
    "max_num_batched_tokens": 16384,
    "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48],
}
MODEL_A = "/models/arm-a"
MODEL_B = "/models/arm-b"
SAMPLING_CONFIG = {
    "generation": {"thinking_budget": 1024, "answer_max_tokens": 96},
    "confirmation": {
        "sample_more_temperature": 0.7,
        "sample_more_top_p": 0.9,
        "sample_more_top_k": 20,
    },
    "controls": {"sample_more_k": 8},
}


def sampling_settings(*, greedy: bool, n: int, seed: int) -> dict:
    return {
        "thinking": "budget",
        "thinking_budget": 1024,
        "n": n,
        "max_tokens": 512,
        "answer_max_tokens": 96,
        "greedy": greedy,
        "temperature": None if greedy else 0.7,
        "top_p": None if greedy else 0.9,
        "top_k": None if greedy else 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "run_seed": seed,
        "shuffle_thinking": False,
        "logprobs": None,
        "prompt_logprobs": None,
        "logprob_token_ids": [],
        "allow_custom_prompts": False,
    }


def resolved_sampling(*, greedy: bool) -> dict:
    return {
        "temperature": 0.0 if greedy else 0.7,
        "top_p": 1.0 if greedy else 0.9,
        "top_k": 0 if greedy else 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    }


def summary() -> dict:
    invariant_args = {
        "model": MODEL_A,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "language_model_only": True,
        "enable_prefix_caching": False,
        "mamba_cache_mode": "none",
        "enforce_eager": False,
        "generation_config": "vllm",
        "max_logprobs": 20,
        "seed": 0,
        "async_scheduling": False,
        **ENGINE,
        "max_cudagraph_capture_size": 48,
    }
    return {
        "schema_version": 4,
        "model": MODEL_A,
        "model_revision": None,
        "model_config_sha256": "a" * 64,
        "runner_sha256": sha256_file(RUNNER),
        "engine": {
            **ENGINE,
            "enable_prefix_caching": False,
            "enforce_eager": False,
            "adapter": None,
            "model_override": MODEL_A,
        },
        "engine_args": invariant_args,
        "resolved_cudagraph": {
            "cudagraph_capture_sizes": list(ENGINE["cudagraph_capture_sizes"]),
            "max_cudagraph_capture_size": 48,
            "mode": "FULL_AND_PIECEWISE",
            "decode_mode": "FULL",
            "mixed_mode": "PIECEWISE",
            "has_full_cudagraphs": True,
        },
        "think_token_ids": {
            "open": 248068,
            "close": 248069,
            "forced_close_sequence": [248069, 271],
            "thinking_prompt_suffix": [248045, 74455, 198, 248068, 198],
            "no_thinking_prompt_suffix": [248045, 74455, 198, 248068, 271, 248069, 271],
        },
        "termination": {
            "hf_model_eos_token_id": 248044,
            "vllm_tokenizer_eos_ignored": 248046,
        },
        "sampling": sampling_settings(greedy=True, n=1, seed=98700),
        "resolved_sampling": resolved_sampling(greedy=True),
        "adapter": None,
        "rng_isolation": {
            "engine_seed": 0,
            "caller_global_rng_state_restored": True,
        },
        "counts": {"requests": 1, "completions": 1, "sampled_tokens": 2},
        "timing": {"generation_seconds": 1.0},
        "runtime": {
            "python": "3.12.3",
            "python_executable": str(PINNED_PYTHON),
            "platform": PINNED_PLATFORM,
            "packages": {
                "vllm": "0.24.0+cu129",
                "torch": "2.11.0+cu129",
                "transformers": "5.13.0",
                "example-extra": "1",
            },
            "environment_lock": {
                "path": str(PINNED_LOCK),
                "sha256": PINNED_LOCK_SHA256,
            },
            "uv": "uv 0.9.0",
            "cuda_toolkit": PINNED_CUDA_TOOLKIT,
            "gpu": PINNED_GPU,
            "vllm_enable_v1_multiprocessing": "0",
            "git_commit": "old",
            "git_dirty": False,
        },
    }


def static_capacity() -> dict:
    blocks = 48 * hybrid_blocks(16384)
    concurrency = blocks / hybrid_blocks(16384)
    return {
        "formula": {
            "attention_block_tokens": 528,
            "mamba_block_tokens": 16384,
            "full_attention_layers": 8,
            "linear_attention_layers": 24,
            "mamba_groups": 3,
            "enable_prefix_caching": False,
            "mamba_cache_mode": "none",
        },
        "max_model_len": 16384,
        "max_num_seqs": 48,
        "num_gpu_blocks": blocks,
        "block_size": 528,
        "kv_cache_size_tokens": int(concurrency * 16384),
        "kv_cache_max_concurrency": concurrency,
        "blocks_per_full_request": 35,
        "forced_close_tokens": 2,
    }


class ConfirmationProtocolTests(unittest.TestCase):
    def test_request_proof_binds_messages_prepared_ids_and_return_order(self):
        records = [
            {"id": "a", "messages": [{"role": "user", "content": "one"}]},
            {"id": "b", "messages": [{"role": "user", "content": "two"}]},
        ]
        prepared = [
            SimpleNamespace(
                record_id="a",
                prompt_text="rendered-one",
                prompt_token_ids=[1, 2],
                prompt_channel="messages",
            ),
            SimpleNamespace(
                record_id="b",
                prompt_text="rendered-two",
                prompt_token_ids=[3, 4, 5],
                prompt_channel="messages",
            ),
        ]
        evidence = ConfirmationRunner._request_evidence(records, prepared)
        self.assertNotEqual(
            evidence[0]["record_sha256"],
            ConfirmationRunner._request_evidence(
                [
                    {
                        "id": "a",
                        "messages": [
                            {"role": "user", "content": "different"}
                        ],
                    }
                ],
                prepared[:1],
            )[0]["record_sha256"],
        )
        self.assertEqual(
            evidence[1]["prompt_token_ids_sha256"], canonical_hash([3, 4, 5])
        )
        rows = [
            {
                "id": row["id"],
                "prompt_sha256": row["prompt_sha256"],
                "n_prompt_tokens": row["n_prompt_tokens"],
                "prompt_channel": row["prompt_channel"],
            }
            for row in evidence
        ]
        ConfirmationRunner._validate_returned_request_rows(rows, evidence)
        with self.assertRaisesRegex(ValueError, "out of order"):
            ConfirmationRunner._validate_returned_request_rows(
                list(reversed(rows)), evidence
            )
        with self.assertRaisesRegex(ValueError, "malformed request row"):
            ConfirmationRunner._validate_returned_request_rows(
                [None, rows[1]], evidence
            )
        drifted = copy.deepcopy(rows)
        drifted[0]["prompt_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "prepared request"):
            ConfirmationRunner._validate_returned_request_rows(drifted, evidence)

    def test_invalid_return_order_is_journaled_before_request_validation(self):
        records = [
            {"id": "a", "messages": [{"role": "user", "content": "one"}]},
            {"id": "b", "messages": [{"role": "user", "content": "two"}]},
        ]
        prepared = [
            SimpleNamespace(
                record_id=record["id"],
                prompt_text=f"rendered-{record['id']}",
                prompt_token_ids=[index + 1],
                prompt_channel="messages",
            )
            for index, record in enumerate(records)
        ]
        evidence = ConfirmationRunner._request_evidence(records, prepared)
        returned = [
            {
                "id": row["id"],
                "prompt_sha256": row["prompt_sha256"],
                "n_prompt_tokens": row["n_prompt_tokens"],
                "prompt_channel": row["prompt_channel"],
                "outputs": [],
            }
            for row in reversed(evidence)
        ]

        class FakeRunner:
            def prepare(self, *_args):
                return prepared

            def generate(self, *_args):
                return returned, {"sampling": {}}

        journal = []
        proxy = object.__new__(ConfirmationRunner)
        proxy._runner = FakeRunner()
        proxy._journal = journal.append
        proxy.eval_summaries = []
        proxy._sample_evidence = {}
        proxy.capacity = static_capacity()
        sampling = SimpleNamespace(
            n=1,
            thinking="budget",
            thinking_budget=32,
            answer_max_tokens=16,
            max_tokens=48,
            allow_custom_prompts=False,
        )
        with self.assertRaisesRegex(ValueError, "out of order"):
            proxy.generate(records, sampling)
        self.assertEqual(len(journal), 1)
        self.assertEqual(
            [row["id"] for row in journal[0]["rows"]], ["b", "a"]
        )

    def test_hybrid_formula_boundaries(self):
        expected = {
            1: 4,
            528: 4,
            529: 5,
            4096: 11,
            4097: 11,
            4224: 11,
            4225: 12,
            16384: 35,
        }
        self.assertEqual({value: hybrid_blocks(value) for value in expected}, expected)
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                hybrid_blocks(invalid)  # type: ignore[arg-type]
        formula = static_capacity()["formula"]
        for key, value in (
            ("attention_block_tokens", 528.0),
            ("enable_prefix_caching", 0),
        ):
            with self.subTest(formula_field=key):
                mutation = copy.deepcopy(formula)
                mutation[key] = value
                with self.assertRaisesRegex(ValueError, "formula drifted"):
                    hybrid_blocks(16384, formula=mutation)

    def test_live_capacity_derives_full_context_mamba_geometry(self):
        static = static_capacity()
        cache = SimpleNamespace(
            num_gpu_blocks=static["num_gpu_blocks"],
            block_size=528,
            kv_cache_size_tokens=static["kv_cache_size_tokens"],
            kv_cache_max_concurrency=static["kv_cache_max_concurrency"],
            mamba_block_size=16384,
            mamba_cache_mode="none",
            enable_prefix_caching=False,
        )
        model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(
                layer_types=[
                    *("linear_attention" for _ in range(24)),
                    *("full_attention" for _ in range(8)),
                ]
            )
        )
        runner = SimpleNamespace(
            config=SimpleNamespace(max_model_len=16384, max_num_seqs=48),
            llm=SimpleNamespace(
                llm_engine=SimpleNamespace(
                    vllm_config=SimpleNamespace(
                        cache_config=cache, model_config=model_config
                    )
                )
            ),
        )
        receipt = live_cache_capacity(runner)
        self.assertEqual(receipt["blocks_per_full_request"], 35)
        self.assertEqual(receipt["formula"]["mamba_block_tokens"], 16384)
        self.assertEqual(receipt["formula"]["mamba_groups"], 3)
        validate_live_cache_capacity(receipt)
        receipt_mutations = (
            ("formula", "full_attention_layers", 8.0),
            ("formula", "enable_prefix_caching", 0),
            (None, "schema_version", True),
            (None, "kv_cache_max_concurrency", 48),
            ("checks", "aligned_attention_block_is_528", 1),
        )
        for section, key, value in receipt_mutations:
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(receipt)
                target = changed if section is None else changed[section]
                target[key] = value
                with self.assertRaises(ValueError):
                    validate_live_cache_capacity(changed)
        cache.mamba_block_size = 4096
        with self.assertRaisesRegex(ValueError, "formula drifted"):
            live_cache_capacity(runner)
        cache.mamba_block_size = 16384
        cache.enable_prefix_caching = 0
        with self.assertRaisesRegex(ValueError, "formula drifted"):
            live_cache_capacity(runner)
        cache.enable_prefix_caching = False
        cache.kv_cache_max_concurrency = 48
        with self.assertRaises(ValueError):
            live_cache_capacity(runner)

    def test_capacity_counts_sample8_and_accepts_exact_boundary(self):
        sampling = SimpleNamespace(
            n=8,
            thinking="budget",
            thinking_budget=1024,
            answer_max_tokens=96,
            max_tokens=1,
        )
        receipt = capacity_receipt(
            static_capacity(), prompt_token_lengths=[100] * 7, sampling=sampling
        )
        self.assertEqual(receipt["logical_sequences"], 56)
        self.assertEqual(receipt["active_sequences"], 48)
        validate_capacity_receipt(receipt)
        tampered = {**receipt, "required_blocks": receipt["required_blocks"] + 1}
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_capacity_receipt(tampered)
        static = static_capacity()
        static["formula"]["mamba_groups"] = 3.0
        with self.assertRaisesRegex(ValueError, "formula drifted"):
            capacity_receipt(
                static, prompt_token_lengths=[100] * 7, sampling=sampling
            )
        receipt_mutations = (
            ("formula", "mamba_groups", 3.0),
            ("formula", "enable_prefix_caching", 0),
            (None, "schema_version", True),
            (None, "logical_sequences", float(receipt["logical_sequences"])),
            ("checks", "context_fits", 1),
        )
        for section, key, value in receipt_mutations:
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(receipt)
                target = changed if section is None else changed[section]
                target[key] = value
                with self.assertRaises(ValueError):
                    validate_capacity_receipt(changed)

    def test_backend_fingerprint_ignores_arm_and_timing_noise(self):
        first = summary()
        protocol, fingerprint = canonical_backend_protocol(
            [first], expected_engine=ENGINE, expected_model=MODEL_A
        )
        noisy = copy.deepcopy(first)
        noisy.update(
            model=MODEL_B,
            model_config_sha256="b" * 64,
            sampling={"greedy": False, "n": 8},
            counts={"requests": 7, "completions": 56, "sampled_tokens": 99},
            timing={"generation_seconds": 12.0},
        )
        noisy["engine"]["model_override"] = MODEL_B
        noisy["engine_args"]["model"] = MODEL_B
        noisy["runtime"]["git_commit"] = "new"
        noisy["runtime"]["git_dirty"] = True
        self.assertEqual(
            canonical_backend_protocol(
                [noisy], expected_engine=ENGINE, expected_model=MODEL_B
            ),
            (protocol, fingerprint),
        )

    def test_backend_binds_admitted_model_identity_and_rng_isolation(self):
        admitted_mismatch = summary()
        admitted_mismatch["model"] = MODEL_B
        admitted_mismatch["engine"]["model_override"] = MODEL_B
        admitted_mismatch["engine_args"]["model"] = MODEL_B
        mutations = (
            ("admitted model", admitted_mismatch),
            (
                "model revision",
                {**summary(), "model_revision": "unexpected-revision"},
            ),
            ("top-level adapter", {**summary(), "adapter": "/adapter"}),
        )
        for label, row in mutations:
            with self.subTest(field=label):
                with self.assertRaises(ValueError):
                    canonical_backend_protocol(
                        [row], expected_engine=ENGINE, expected_model=MODEL_A
                    )

        nested_mutations = (
            ("engine", "model_override", MODEL_B),
            ("engine", "adapter", "/adapter"),
            ("engine_args", "model", MODEL_B),
            ("rng_isolation", "engine_seed", 1),
            ("rng_isolation", "caller_global_rng_state_restored", False),
        )
        for section, key, value in nested_mutations:
            with self.subTest(section=section, field=key):
                row = summary()
                row[section][key] = value
                with self.assertRaises(ValueError):
                    canonical_backend_protocol(
                        [row], expected_engine=ENGINE, expected_model=MODEL_A
                    )

        extra_rng_field = summary()
        extra_rng_field["rng_isolation"]["unregistered"] = True
        with self.assertRaises(ValueError):
            canonical_backend_protocol(
                [extra_rng_field], expected_engine=ENGINE, expected_model=MODEL_A
            )

    def test_backend_rejects_pinned_runtime_and_engine_drift(self):
        mutations = (
            ("runtime", "python", "3.12.4"),
            ("runtime", "gpu", "different"),
            ("runtime", "cuda_toolkit", "different"),
            ("engine_args", "async_scheduling", True),
            ("engine_args", "dtype", "float16"),
            ("resolved_cudagraph", "decode_mode", "NONE"),
        )
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                row = summary()
                row[section][key] = value
                with self.assertRaises(ValueError):
                    canonical_backend_protocol([row], expected_engine=ENGINE)

        type_mutations = (
            ("engine_args", "trust_remote_code", 1),
            ("engine_args", "tensor_parallel_size", True),
            ("engine_args", "seed", False),
            ("engine_args", "async_scheduling", 0),
            ("engine_args", "max_logprobs", 20.0),
            ("engine_args", "max_model_len", 16384.0),
            ("engine_args", "max_cudagraph_capture_size", 48.0),
            ("resolved_cudagraph", "max_cudagraph_capture_size", 48.0),
            ("think_token_ids", "open", 248068.0),
            ("termination", "hf_model_eos_token_id", 248044.0),
        )
        for section, key, value in type_mutations:
            with self.subTest(type_section=section, type_key=key):
                row = summary()
                row[section][key] = value
                with self.assertRaises(ValueError):
                    canonical_backend_protocol([row], expected_engine=ENGINE)

        list_type_mutations = (
            ("engine_args", "cudagraph_capture_sizes"),
            ("resolved_cudagraph", "cudagraph_capture_sizes"),
            ("think_token_ids", "forced_close_sequence"),
        )
        for section, key in list_type_mutations:
            with self.subTest(list_section=section, list_key=key):
                row = summary()
                row[section][key] = list(row[section][key])
                row[section][key][0] = True
                with self.assertRaises(ValueError):
                    canonical_backend_protocol([row], expected_engine=ENGINE)

        registered_engine = copy.deepcopy(ENGINE)
        registered_engine["max_num_seqs"] = 48.0
        with self.assertRaises(ValueError):
            canonical_backend_protocol(
                [summary()], expected_engine=registered_engine
            )

    def test_sampling_protocol_binds_every_raw_and_resolved_setting(self):
        atom = summary()
        episode = copy.deepcopy(atom)
        expected = expected_confirmation_sampling_protocol(
            SAMPLING_CONFIG, decode="greedy", block_seed=98700
        )
        self.assertEqual(
            validate_confirmation_sampling_protocol([atom, episode], expected),
            expected,
        )
        mutations = (
            ("sampling", "run_seed", 98701),
            ("sampling", "n", 8),
            ("sampling", "thinking_budget", 1023),
            ("sampling", "presence_penalty", 0.1),
            ("resolved_sampling", "top_p", 0.9),
        )
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                drifted = copy.deepcopy(episode)
                drifted[section][key] = value
                with self.assertRaisesRegex(ValueError, "registration|different"):
                    validate_confirmation_sampling_protocol(
                        [atom, drifted], expected
                    )

        sampled_atom = summary()
        sampled_episode = copy.deepcopy(sampled_atom)
        sampled_atom["sampling"] = sampling_settings(
            greedy=False, n=8, seed=98700
        )
        sampled_episode["sampling"] = sampling_settings(
            greedy=False, n=1, seed=98700
        )
        sampled_atom["resolved_sampling"] = resolved_sampling(greedy=False)
        sampled_episode["resolved_sampling"] = resolved_sampling(greedy=False)
        sampled_expected = expected_confirmation_sampling_protocol(
            SAMPLING_CONFIG, decode="sample8", block_seed=98700
        )
        validate_confirmation_sampling_protocol(
            [sampled_atom, sampled_episode], sampled_expected
        )

    def test_registered_sampling_exactly_matches_harness_dataclasses(self):
        for decode, greedy, k in (("greedy", True, 1), ("sample8", False, 8)):
            with self.subTest(decode=decode):
                expected = expected_confirmation_sampling_protocol(
                    SAMPLING_CONFIG, decode=decode, block_seed=98700
                )
                common = {
                    "think_budget": 1024,
                    "answer_max_tokens": 96,
                    "run_seed": 98700,
                    "greedy": greedy,
                    "temperature": None if greedy else 0.7,
                    "top_p": None if greedy else 0.9,
                    "top_k": None if greedy else 20,
                }
                for role, n in (("atom", k), ("episode", 1)):
                    sampling = harness._sampling(n=n, **common)
                    raw = dataclasses.asdict(sampling)
                    raw["logprob_token_ids"] = list(raw["logprob_token_ids"])
                    self.assertEqual(expected[role]["sampling"], raw)
                    self.assertEqual(
                        expected[role]["resolved_sampling"],
                        sampling.resolved_sampling(),
                    )

    def test_analysis_rejects_nonpolicy_or_empty_engine_protocol(self):
        rows = [summary(), summary()]
        expected_sampling = expected_confirmation_sampling_protocol(
            SAMPLING_CONFIG, decode="greedy", block_seed=98700
        )
        payload = {
            "stage": "policy_eval",
            "decode": "greedy",
            "runner_summary": rows,
            "sampling_protocol": expected_sampling,
            "engine_protocol": {"authenticated": True},
        }
        _require_policy_score_protocol(
            payload, config=SAMPLING_CONFIG, block_seed=98700
        )
        for mutation in (
            {**payload, "stage": "unit_test"},
            {**payload, "engine_protocol": {}},
            {**payload, "sampling_protocol": {}},
        ):
            with self.assertRaises(ValueError):
                _require_policy_score_protocol(
                    mutation, config=SAMPLING_CONFIG, block_seed=98700
                )

    def test_campaign_recomputes_digest_and_pairs_exact_tasks(self):
        base_summary = summary()
        protocol, fingerprint = canonical_backend_protocol([base_summary], expected_engine=ENGINE)
        def score(task: str, plan: str, row: dict) -> dict:
            return {
                "model": row["model"],
                "runner_summary": [row],
                "backend_protocol": protocol,
                "backend_fingerprint": fingerprint,
                "task_manifest_sha256": task,
                "ordered_plan_sha256": plan,
            }
        arms = {
            "a": [score("t0", "p0", base_summary), score("t1", "p1", base_summary)],
            "b": [score("t0", "p0", copy.deepcopy(base_summary)), score("t1", "p1", copy.deepcopy(base_summary))],
        }
        self.assertTrue(all(_cross_score_protocol_checks(arms, block_count=2, engine=ENGINE).values()))
        arms["b"][1]["runner_summary"][0]["runtime"]["packages"]["example-extra"] = "2"
        checks = _cross_score_protocol_checks(arms, block_count=2, engine=ENGINE)
        self.assertFalse(checks["one_exact_backend_across_all_scores"])
        self.assertFalse(checks["all_backend_fingerprints_recomputed"])

    def test_confirmation_analysis_seal_is_canonical_and_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "experiment"
            output = experiment / "analysis" / "confirmation.json"
            result = {"stage": "confirmation", "gate": {"passed": True}}
            with mock.patch.object(analyze_confirmation, "EXP", experiment):
                output, existed = analyze_confirmation._analysis_publication_start(
                    output
                )
                self.assertTrue(
                    analyze_confirmation._publish_analysis_no_clobber(
                        output, result, existed_at_start=existed
                    )
                )
                original_bytes = output.read_bytes()
                original_mtime = output.stat().st_mtime_ns

                output, existed = analyze_confirmation._analysis_publication_start(
                    output
                )
                self.assertFalse(
                    analyze_confirmation._publish_analysis_no_clobber(
                        output, result, existed_at_start=existed
                    )
                )
                self.assertEqual(output.read_bytes(), original_bytes)
                self.assertEqual(output.stat().st_mtime_ns, original_mtime)

                stale_bytes = b'{"stale":true}\n'
                output.write_bytes(stale_bytes)
                with self.assertRaisesRegex(ValueError, "refusing to overwrite stale"):
                    analyze_confirmation._publish_analysis_no_clobber(
                        output, result, existed_at_start=True
                    )
                self.assertEqual(output.read_bytes(), stale_bytes)

                with self.assertRaisesRegex(ValueError, "not the exact canonical path"):
                    analyze_confirmation._analysis_publication_start(
                        experiment / "analysis" / "alternate.json"
                    )

    def test_confirmation_analysis_seal_rejects_symlinks_and_races(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            real_analysis = root / "redirected-analysis"
            real_analysis.mkdir()
            experiment.mkdir()
            (experiment / "analysis").symlink_to(
                real_analysis, target_is_directory=True
            )
            output = experiment / "analysis" / "confirmation.json"
            with mock.patch.object(
                analyze_confirmation, "EXP", experiment
            ), self.assertRaisesRegex(ValueError, "symlink"):
                analyze_confirmation._analysis_publication_start(output)

            (experiment / "analysis").unlink()
            output, existed = None, None

            def lose_race(_source, target):
                Path(target).write_bytes(b"winner")
                raise FileExistsError

            with mock.patch.object(analyze_confirmation, "EXP", experiment):
                output, existed = analyze_confirmation._analysis_publication_start(
                    experiment / "analysis" / "confirmation.json"
                )
                with mock.patch.object(
                    bench.os, "link", side_effect=lose_race
                ), self.assertRaisesRegex(ValueError, "lost a race"):
                    analyze_confirmation._publish_analysis_no_clobber(
                        output, {"ours": True}, existed_at_start=existed
                    )
            self.assertEqual(output.read_bytes(), b"winner")

    def test_benchmark_authorizer_requires_canonical_safe_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            analysis = experiment / "analysis"
            analysis.mkdir(parents=True)
            canonical = analysis / "confirmation.json"
            canonical.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(authorize_benchmark, "EXP", experiment):
                self.assertEqual(
                    authorize_benchmark._validated_confirmation_analysis_file(
                        canonical
                    ),
                    canonical,
                )
                self.assertEqual(
                    authorize_benchmark._confirmation_analysis_binding(
                        canonical, {}
                    ),
                    {"path": str(canonical), "sha256": sha256_file(canonical)},
                )
                with self.assertRaisesRegex(ValueError, "changed"):
                    authorize_benchmark._confirmation_analysis_binding(
                        canonical, {"stale": True}
                    )
                canonical.write_text('{"value":0}\n', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed"):
                    authorize_benchmark._confirmation_analysis_binding(
                        canonical, {"value": False}
                    )
                canonical.write_text('{"value":2}\n', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed"):
                    authorize_benchmark._confirmation_analysis_binding(
                        canonical, {"value": 2.0}
                    )
                with self.assertRaisesRegex(ValueError, "not the exact canonical path"):
                    authorize_benchmark._validated_confirmation_analysis_file(
                        analysis / "other.json"
                    )

                canonical.unlink()
                outside = root / "outside.json"
                outside.write_text("{}\n", encoding="utf-8")
                canonical.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    authorize_benchmark._validated_confirmation_analysis_file(
                        canonical
                    )

    def test_evaluator_source_inventory_rejects_symlink_leaf_and_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            relative_sources = (
                "scripts/eval_policy.py",
                "src/confirmation_artifacts.py",
                "src/confirmation_protocol.py",
                "src/control_code_inventory.py",
                "src/harness.py",
                "src/io_utils.py",
                "src/state_replay.py",
                "src/vllm_runner.py",
                "src/gym/safe.py",
            )
            for relative in relative_sources:
                path = experiment / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {relative}\n", encoding="utf-8")

            with mock.patch.object(io_utils, "EXP", experiment):
                inventory = confirmation_evaluator_source_inventory()
                self.assertEqual(inventory["file_count"], len(relative_sources))

                safe = experiment / "src" / "gym" / "safe.py"
                safe.unlink()
                forbidden = root / "benchmarks" / "forbidden" / "source.py"
                forbidden.parent.mkdir(parents=True)
                forbidden.write_text("# forbidden-like source\n", encoding="utf-8")
                safe.symlink_to(forbidden)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    confirmation_evaluator_source_inventory()

                safe.unlink()
                safe.write_text("# restored\n", encoding="utf-8")
                gym = experiment / "src" / "gym"
                real_gym = experiment / "src" / "gym-real"
                gym.rename(real_gym)
                gym.symlink_to(real_gym, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    confirmation_evaluator_source_inventory()


if __name__ == "__main__":
    unittest.main()
