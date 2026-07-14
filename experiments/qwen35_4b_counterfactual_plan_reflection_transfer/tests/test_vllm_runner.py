from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
import yaml
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


RUNNER_PATH = Path(__file__).resolve().parents[1] / "src" / "vllm_runner.py"
sys.path.insert(0, str(RUNNER_PATH.parent))
import stages as stage_module  # noqa: E402

SPEC = importlib.util.spec_from_file_location("template_vllm_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class _FakeCudagraphMode:
    def __init__(self, name: str, decode: str, mixed: str, has_full: bool):
        self.name = name
        self._decode = decode
        self._mixed = mixed
        self._has_full = has_full

    def decode_mode(self) -> SimpleNamespace:
        return SimpleNamespace(name=self._decode)

    def mixed_mode(self) -> SimpleNamespace:
        return SimpleNamespace(name=self._mixed)

    def has_full_cudagraphs(self) -> bool:
        return self._has_full


def _compilation_config(
    sizes: tuple[int, ...] = (1, 2, 4, 8, 15),
    *,
    maximum: int = 15,
    mode: str = "FULL_DECODE_ONLY",
    decode: str = "FULL",
    mixed: str = "NONE",
    has_full: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        cudagraph_capture_sizes=list(sizes),
        max_cudagraph_capture_size=maximum,
        cudagraph_mode=_FakeCudagraphMode(mode, decode, mixed, has_full),
    )


class EngineConfigCaptureGeometryTests(unittest.TestCase):
    @staticmethod
    def live_cache(num_blocks: int = 1100) -> dict:
        concurrency = num_blocks / 11
        return {
            "num_gpu_blocks": num_blocks,
            "block_size": 528,
            "kv_cache_size_tokens": int(concurrency * 4096),
            "kv_cache_max_concurrency": concurrency,
            "enable_prefix_caching": False,
            "mamba_cache_mode": "none",
            "mamba_block_size": 4096,
        }

    def test_live_hybrid_capacity_geometry_and_invocation_preflight(self) -> None:
        config = runner.EngineConfig(
            max_model_len=4096,
            max_num_seqs=64,
            max_num_batched_tokens=16384,
            cudagraph_capture_sizes=(1, 2, 4, 8, 16, 32, 64),
        )
        cache = self.live_cache()
        shape = runner._validate_live_cache_geometry(cache, config)
        self.assertEqual(shape["blocks_per_max_request"], 11)
        live = {
            "live_model": {"max_model_len": 4096, "dtype": "torch.bfloat16"},
            "live_scheduler": {
                "max_num_seqs": 64,
                "max_num_batched_tokens": 16384,
                "async_scheduling": False,
            },
            "live_parallel": {
                "world_size": 1,
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
            },
            "live_cache": cache,
            "cache_shape": shape,
        }
        receipt = runner._capacity_preflight(
            live=live,
            config=config,
            prompt_lengths=[500] * 144,
            sampling=runner.SamplingConfig(
                thinking="budget", thinking_budget=1024, answer_max_tokens=128, n=16
            ),
            close_tokens=2,
        )
        self.assertEqual(receipt["decision"], "LIVE_KV_CAPACITY_PASS")
        self.assertEqual(receipt["invocation"]["active_sequences"], 64)
        self.assertGreater(receipt["invocation"]["remaining_cache_blocks"], 0)

    def test_live_capacity_rejects_changed_geometry_and_overcommit(self) -> None:
        config = runner.EngineConfig(max_model_len=4096, max_num_seqs=64)
        changed = self.live_cache()
        changed["block_size"] = 512
        with self.assertRaisesRegex(RuntimeError, "geometry changed"):
            runner._validate_live_cache_geometry(changed, config)

        cache = self.live_cache(704)
        shape = runner._validate_live_cache_geometry(cache, config)
        live = {"live_cache": cache, "cache_shape": shape}
        with self.assertRaisesRegex(RuntimeError, "cannot fit"):
            runner._capacity_preflight(
                live=live,
                config=config,
                prompt_lengths=[3000] * 64,
                sampling=runner.SamplingConfig(
                    thinking="budget", thinking_budget=1024, answer_max_tokens=128, n=1
                ),
                close_tokens=2,
            )

    def test_merged_model_override_is_existing_and_mutually_exclusive(self) -> None:
        runner.EngineConfig(model_override=RUNNER_PATH.parent).validate()
        with self.assertRaisesRegex(ValueError, "existing merged-checkpoint"):
            runner.EngineConfig(model_override=RUNNER_PATH.parent / "missing").validate()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            runner.EngineConfig(
                model_override=RUNNER_PATH.parent,
                adapter=RUNNER_PATH.parent,
            ).validate()
        with self.assertRaisesRegex(ValueError, "runtime LoRA adapters are forbidden"):
            runner.EngineConfig(adapter=RUNNER_PATH.parent).validate()

    def test_merged_override_receipt_binds_full_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights.safetensors").write_bytes(b"weights")
            lineage = root / "source_lineage"
            lineage.mkdir()
            config_sha = runner._sha256_file(
                RUNNER_PATH.parents[1] / "configs" / "default.yaml"
            )
            git_commit = runner._run_text(["git", "rev-parse", "HEAD"])
            stage_receipt = {
                "schema_version": 2,
                "experiment_id": "qwen35_4b_counterfactual_plan_reflection_transfer",
                "authorized_stage": "screen_training",
                "config_sha256": config_sha,
                "issuer_git_commit": git_commit,
                "issuer_script_sha256": runner._sha256_file(
                    RUNNER_PATH.parents[1] / "scripts" / "authorize_stage.py"
                ),
                "prerequisites": [
                    {"kind": "calibration_gate", "sha256": "c" * 64, "pass": True}
                ],
            }
            (lineage / "stage_receipt.json").write_text(json.dumps(stage_receipt))
            tokenizer_receipt = {
                "experiment_id": "qwen35_4b_counterfactual_plan_reflection_transfer",
                "model_id": runner.MODEL_ID,
                "model_revision": runner.MODEL_REVISION,
                "tokenizer_eos_token_id": 248046,
            }
            (lineage / "tokenizer_receipt.json").write_text(json.dumps(tokenizer_receipt))
            adapter_config = {
                "r": 32,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            }
            (lineage / "adapter_config.json").write_text(json.dumps(adapter_config))
            trainer_sha = runner._sha256_file(
                RUNNER_PATH.parents[1] / "scripts" / "train.py"
            )
            training_receipt = {
                "schema_version": 2,
                "experiment_id": "qwen35_4b_counterfactual_plan_reflection_transfer",
                "config_sha256": config_sha,
                "arm": "reflection_correct",
                "seed": 47,
                "model_id": runner.MODEL_ID,
                "model_revision": runner.MODEL_REVISION,
                "optimizer_steps": 36,
                "train_loss": 0.1,
                "trainer_sha256": trainer_sha,
                "trainer_git_commit": git_commit,
                "recipe_sha256": runner._sha256_bytes(
                    json.dumps(
                        yaml.safe_load(
                            (RUNNER_PATH.parents[1] / "configs" / "default.yaml").read_text()
                        )["training"]["recipe"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ),
                "stage_receipt_sha256": runner._sha256_file(lineage / "stage_receipt.json"),
                "tokenizer_receipt_sha256": runner._sha256_file(
                    lineage / "tokenizer_receipt.json"
                ),
                "copied_stage_receipt_sha256": runner._sha256_file(
                    lineage / "stage_receipt.json"
                ),
                "copied_tokenizer_receipt_sha256": runner._sha256_file(
                    lineage / "tokenizer_receipt.json"
                ),
                "record_receipt_sha256": "1" * 64,
                "parity_sha256": "2" * 64,
                "adapter_tree_excluding_training_receipt_sha256": "b" * 64,
            }
            (lineage / "training_receipt.json").write_text(json.dumps(training_receipt))
            tree_hash = runner._sha256_tree(root)
            receipt = {
                "schema_version": 2,
                "experiment_id": "qwen35_4b_counterfactual_plan_reflection_transfer",
                "config_sha256": config_sha,
                "model_id": runner.MODEL_ID,
                "model_revision": runner.MODEL_REVISION,
                "applied_lora_modules": 7,
                "merged_tree_sha256": tree_hash,
                "source_training_receipt_sha256": runner._sha256_file(
                    lineage / "training_receipt.json"
                ),
                "source_stage_receipt_sha256": runner._sha256_file(
                    lineage / "stage_receipt.json"
                ),
                "source_tokenizer_receipt_sha256": runner._sha256_file(
                    lineage / "tokenizer_receipt.json"
                ),
                "source_trainer_sha256": trainer_sha,
                "source_trainer_git_commit": git_commit,
                "source_recipe_sha256": training_receipt["recipe_sha256"],
                "source_adapter_tree_sha256": "b" * 64,
                "source_adapter_sha256": "f" * 64,
                "source_adapter_config_sha256": runner._sha256_file(
                    lineage / "adapter_config.json"
                ),
                "source_arm": "reflection_correct",
                "source_seed": 47,
            }
            (root / "merge_receipt.json").write_text(json.dumps(receipt))
            with mock.patch.object(stage_module, "require_clean_worktree"):
                validated = runner._validate_model_override(root)
            self.assertEqual(validated["merged_tree_sha256"], tree_hash)
            (root / "weights.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "tree hash differs"):
                with mock.patch.object(stage_module, "require_clean_worktree"):
                    runner._validate_model_override(root)

    def test_explicit_capture_list_requires_strict_positive_tied_geometry(self) -> None:
        runner.EngineConfig(
            max_num_seqs=19,
            cudagraph_capture_sizes=(1, 2, 4, 8, 16, 19),
        ).validate()

        invalid = (
            ((), "positive integers"),
            ((0, 19), "positive integers"),
            ((True, 19), "positive integers"),
            ((1, 4, 2, 19), "strictly increasing"),
            ((1, 2, 2, 19), "strictly increasing"),
            ((1, 2, 4, 16), "largest cudagraph capture size"),
        )
        for sizes, message in invalid:
            with self.subTest(sizes=sizes), self.assertRaisesRegex(
                ValueError, message
            ):
                runner.EngineConfig(
                    max_num_seqs=19,
                    cudagraph_capture_sizes=sizes,
                ).validate()

        with self.assertRaisesRegex(ValueError, "incompatible with enforce_eager"):
            runner.EngineConfig(
                max_num_seqs=19,
                cudagraph_capture_sizes=(1, 2, 4, 8, 16, 19),
                enforce_eager=True,
            ).validate()

    def test_mamba_clamp_from_19_to_15_is_deterministic_and_tied(self) -> None:
        requested = (1, 2, 4, 8, 16, 19)
        expected = (1, 2, 4, 8, 15)
        self.assertEqual(runner._clamp_cudagraph_capture_sizes(requested, 15), expected)
        self.assertEqual(runner._clamp_cudagraph_capture_sizes(requested, 15), expected)
        self.assertEqual(expected[-1], 15)
        self.assertEqual(tuple(sorted(set(expected))), expected)

    def test_mamba_clamp_rejects_invalid_target_or_empty_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            runner._clamp_cudagraph_capture_sizes((1, 2, 4), 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            runner._clamp_cudagraph_capture_sizes((), 15)


class PackageInventoryTests(unittest.TestCase):
    def test_atomic_json_writer_returns_exact_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "rows.jsonl"
            digest = runner._write_json_atomic(
                output, [{"id": "one"}, {"id": "two"}], jsonl=True
            )

            self.assertEqual(digest, runner._sha256_file(output))
            self.assertEqual(len(output.read_text().splitlines()), 2)

    def test_vendored_duplicate_cannot_override_real_distribution(self) -> None:
        real = SimpleNamespace(
            metadata={"Name": "packaging"},
            version="26.2",
        )
        vendored = SimpleNamespace(
            metadata={"Name": "packaging"},
            version="26.0",
        )
        with mock.patch.object(
            runner.importlib.metadata,
            "distributions",
            return_value=[real, vendored],
        ), mock.patch.object(
            runner.importlib.metadata,
            "version",
            return_value="26.2",
        ) as resolve:
            inventory = runner._installed_packages()

        self.assertEqual(inventory, {"packaging": "26.2"})
        resolve.assert_called_once_with("packaging")


class TerminationSemanticsTests(unittest.TestCase):
    def test_model_and_tokenizer_eos_ids_remain_distinct(self) -> None:
        runner._validate_termination_ids(248044, 248046)
        for model_eos, tokenizer_eos in ((248046, 248046), (248044, 248044)):
            with self.subTest(
                model_eos=model_eos, tokenizer_eos=tokenizer_eos
            ), self.assertRaisesRegex(RuntimeError, "termination IDs changed"):
                runner._validate_termination_ids(model_eos, tokenizer_eos)

    def test_sampling_ignores_tokenizer_eos_and_stops_on_model_eos(self) -> None:
        captured: dict[str, object] = {}

        class FakeSamplingParams:
            def __init__(self, **kwargs: object):
                captured.update(kwargs)

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = FakeSamplingParams
        instance = object.__new__(runner.VLLMRunner)
        instance.hf_eos_id = 248044
        with mock.patch.dict(sys.modules, {"vllm": fake_vllm}):
            instance._params(
                runner.SamplingConfig(thinking="off", greedy=True),
                max_tokens=8,
                seed=17,
                n=1,
            )

        self.assertIs(captured["ignore_eos"], True)
        self.assertEqual(captured["stop_token_ids"], [248044])

    def test_trimming_preserves_tokenizer_eos_and_removes_model_eos(self) -> None:
        instance = object.__new__(runner.VLLMRunner)
        instance.hf_eos_id = 248044
        self.assertEqual(
            instance._trim_hf_eos([1, 248046, 2, 248044, 3]),
            [1, 248046, 2],
        )


class CliRewriteTests(unittest.TestCase):
    def test_rewrite_replaces_every_split_and_equals_spelling(self) -> None:
        argv = [
            "vllm_runner.py",
            "--smoke",
            "1",
            "--max-num-seqs",
            "64",
            "--cudagraph-capture-size=1",
            "--max-num-seqs=32",
            "--cudagraph-capture-size",
            "2",
            "--max-num-seqs",
            "19",
            "--cudagraph-capture-size=19",
            "--output",
            "out.jsonl",
        ]
        capture_sizes = (1, 2, 4, 8, 15)

        rewritten = runner._rewrite_max_num_seqs_argv(argv, 15, capture_sizes)

        self.assertEqual(
            rewritten,
            [
                "vllm_runner.py",
                "--smoke",
                "1",
                "--output",
                "out.jsonl",
                "--max-num-seqs",
                "15",
                "--cudagraph-capture-size",
                "1",
                "--cudagraph-capture-size",
                "2",
                "--cudagraph-capture-size",
                "4",
                "--cudagraph-capture-size",
                "8",
                "--cudagraph-capture-size",
                "15",
            ],
        )
        parsed = runner._parse_args(rewritten[1:])
        self.assertEqual(parsed.max_num_seqs, 15)
        self.assertEqual(parsed.cudagraph_capture_size, list(capture_sizes))

    def test_rewrite_leaves_explicit_graph_flags_alone_when_not_replacing_them(self) -> None:
        argv = [
            "vllm_runner.py",
            "--max-num-seqs=19",
            "--cudagraph-capture-size=19",
        ]
        self.assertEqual(
            runner._rewrite_max_num_seqs_argv(argv, 15),
            [
                "vllm_runner.py",
                "--cudagraph-capture-size=19",
                "--max-num-seqs",
                "15",
            ],
        )

    def test_cli_disables_long_option_abbreviation(self) -> None:
        for abbreviated in ("--max-num-seq", "--cudagraph-capture-siz"):
            with self.subTest(argument=abbreviated), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit) as raised:
                runner._parse_args(
                    ["--smoke", "1", "--output", "out.jsonl", abbreviated, "15"]
                )
            self.assertEqual(raised.exception.code, 2)


class MambaReexecGuardTests(unittest.TestCase):
    def test_guard_parser_accepts_consistent_original_geometry(self) -> None:
        sizes = (1, 2, 4, 8, 16, 19)
        self.assertEqual(
            runner._parse_mamba_reexec_geometry("19", json.dumps(sizes)),
            (19, sizes),
        )
        self.assertEqual(runner._parse_mamba_reexec_geometry(None, None), (None, None))

    def test_guard_parser_rejects_orphaned_or_malformed_provenance(self) -> None:
        invalid = (
            (None, "[1,2,4,8,16,19]", "orphaned"),
            ("zero", None, "invalid"),
            ("0", None, "invalid"),
            ("19", "not-json", "invalid"),
            ("19", "[]", "invalid"),
            ("19", "[1,2,true,19]", "invalid"),
            ("19", "[1,4,2,19]", "inconsistent"),
            ("19", "[1,2,2,19]", "inconsistent"),
            ("19", "[1,2,4,15]", "inconsistent"),
        )
        for max_num_seqs, sizes, message in invalid:
            with self.subTest(
                max_num_seqs=max_num_seqs, sizes=sizes
            ), self.assertRaisesRegex(RuntimeError, message):
                runner._parse_mamba_reexec_geometry(max_num_seqs, sizes)

    def test_stale_guard_is_rejected_before_model_import(self) -> None:
        sizes = (1, 2, 4, 8, 16, 19)
        environment = {
            runner._MAMBA_CACHE_REEXEC_ENV: "19",
            runner._MAMBA_CACHE_REEXEC_CUDAGRAPH_ENV: json.dumps(sizes),
        }
        with mock.patch.dict(os.environ, environment), self.assertRaisesRegex(
            RuntimeError, "did not lower max_num_seqs"
        ):
            runner.VLLMRunner(
                runner.EngineConfig(
                    max_num_seqs=19,
                    cudagraph_capture_sizes=sizes,
                )
            )

    def test_changed_explicitness_is_rejected_before_model_import(self) -> None:
        with mock.patch.dict(
            os.environ,
            {runner._MAMBA_CACHE_REEXEC_ENV: "19"},
        ):
            os.environ.pop(runner._MAMBA_CACHE_REEXEC_CUDAGRAPH_ENV, None)
            with self.assertRaisesRegex(RuntimeError, "changed whether"):
                runner.VLLMRunner(
                    runner.EngineConfig(
                        max_num_seqs=15,
                        cudagraph_capture_sizes=(1, 2, 4, 8, 15),
                    )
                )


class ResolvedCudagraphTests(unittest.TestCase):
    def test_supported_full_decode_modes_are_accepted(self) -> None:
        requested = (1, 2, 4, 8, 15)
        modes = (
            ("FULL", "FULL", "FULL"),
            ("FULL_DECODE_ONLY", "FULL", "NONE"),
            ("FULL_AND_PIECEWISE", "FULL", "PIECEWISE"),
        )
        for mode, decode, mixed in modes:
            with self.subTest(mode=mode):
                resolved = runner._resolved_cudagraph_metadata(
                    _compilation_config(mode=mode, decode=decode, mixed=mixed)
                )
                runner._validate_explicit_cudagraph_resolution(requested, resolved)

    def test_none_piecewise_only_and_truncated_resolutions_are_rejected(self) -> None:
        requested = (1, 2, 4, 8, 15)
        rejected = (
            _compilation_config(
                mode="NONE", decode="NONE", mixed="NONE", has_full=False
            ),
            _compilation_config(
                mode="PIECEWISE",
                decode="PIECEWISE",
                mixed="PIECEWISE",
                has_full=False,
            ),
            _compilation_config((1, 2, 4, 8), maximum=8),
            _compilation_config(maximum=8),
        )
        for compilation_config in rejected:
            resolved = runner._resolved_cudagraph_metadata(compilation_config)
            with self.subTest(resolved=resolved), self.assertRaisesRegex(
                RuntimeError, "did not honor"
            ):
                runner._validate_explicit_cudagraph_resolution(requested, resolved)

    def test_reexec_metadata_preserves_original_and_effective_geometry(self) -> None:
        requested = (1, 2, 4, 8, 16, 19)
        effective = (1, 2, 4, 8, 15)
        captured_engine_args: dict[str, object] = {}

        class FakeTokenizer:
            eos_token_id = 248046
            eos_token = "<|im_end|>"

            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                values = {
                    "<|endoftext|>": [248044],
                    "<|im_end|>": [248046],
                    "<think>": [248068],
                    "</think>": [248069],
                    "</think>\n\n": [248069, 198],
                    "<|im_start|>assistant\n<think>\n": [1, 2, 3],
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n": [1, 2, 4],
                }
                return values[text]

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> FakeTokenizer:
                return FakeTokenizer()

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(text_config=SimpleNamespace(eos_token_id=248044))

        class FakeLLM:
            def __init__(self, **kwargs: object):
                captured_engine_args.update(kwargs)
                cache_blocks = 1100
                self.llm_engine = SimpleNamespace(
                    vllm_config=SimpleNamespace(
                        compilation_config=_compilation_config(),
                        cache_config=SimpleNamespace(
                            num_gpu_blocks=cache_blocks,
                            block_size=528,
                            kv_cache_size_tokens=int((cache_blocks / 11) * 4096),
                            kv_cache_max_concurrency=cache_blocks / 11,
                            enable_prefix_caching=False,
                            mamba_cache_mode="none",
                            mamba_block_size=4096,
                        ),
                        scheduler_config=SimpleNamespace(
                            max_num_seqs=15,
                            max_num_batched_tokens=32768,
                            async_scheduling=False,
                        ),
                        model_config=SimpleNamespace(
                            max_model_len=4096,
                            dtype="torch.bfloat16",
                        ),
                        parallel_config=SimpleNamespace(
                            world_size=1,
                            tensor_parallel_size=1,
                            data_parallel_size=1,
                        ),
                    ),
                    engine_core=SimpleNamespace(shutdown=lambda: None),
                )

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeAutoTokenizer
        fake_transformers.AutoConfig = FakeAutoConfig
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        environment = {
            runner._MAMBA_CACHE_REEXEC_ENV: "19",
            runner._MAMBA_CACHE_REEXEC_CUDAGRAPH_ENV: json.dumps(requested),
        }

        with mock.patch.dict(os.environ, environment), mock.patch.dict(
            sys.modules,
            {"transformers": fake_transformers, "vllm": fake_vllm},
        ):
            instance = runner.VLLMRunner(
                runner.EngineConfig(
                    max_model_len=4096,
                    max_num_seqs=15,
                    cudagraph_capture_sizes=effective,
                )
            )

        self.assertEqual(captured_engine_args["max_num_seqs"], 15)
        self.assertEqual(captured_engine_args["cudagraph_capture_sizes"], list(effective))
        self.assertEqual(captured_engine_args["max_cudagraph_capture_size"], 15)
        self.assertEqual(instance.engine_args["requested_max_num_seqs"], 19)
        self.assertEqual(instance.engine_args["effective_max_num_seqs"], 15)
        self.assertEqual(
            instance.engine_args["requested_cudagraph_capture_sizes"], list(requested)
        )
        self.assertEqual(
            instance.engine_args["effective_cudagraph_capture_sizes"], list(effective)
        )
        self.assertEqual(
            instance.resolved_cudagraph["cudagraph_capture_sizes"], list(effective)
        )
        instance.close()


if __name__ == "__main__":
    unittest.main()
