from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


RUNNER_PATH = Path(__file__).resolve().parents[1] / "src" / "vllm_runner.py"
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
        instance.tokenizer_eos_id = 248046
        with mock.patch.dict(sys.modules, {"vllm": fake_vllm}):
            instance._params(
                runner.SamplingConfig(thinking="off", greedy=True),
                max_tokens=8,
                seed=17,
                n=1,
            )

        self.assertIs(captured["ignore_eos"], True)
        self.assertEqual(captured["stop_token_ids"], [248044])

    def test_sampling_can_register_only_the_pinned_tokenizer_eos(self) -> None:
        captured: dict[str, object] = {}

        class FakeSamplingParams:
            def __init__(self, **kwargs: object):
                captured.update(kwargs)

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = FakeSamplingParams
        instance = object.__new__(runner.VLLMRunner)
        instance.hf_eos_id = 248044
        instance.tokenizer_eos_id = 248046
        with mock.patch.dict(sys.modules, {"vllm": fake_vllm}):
            instance._params(
                runner.SamplingConfig(thinking="off", greedy=True),
                max_tokens=8,
                seed=17,
                n=1,
                stop_token_id=248046,
            )
            with self.assertRaisesRegex(ValueError, "pinned HF model EOS"):
                instance._params(
                    runner.SamplingConfig(thinking="off", greedy=True),
                    max_tokens=8,
                    seed=17,
                    n=1,
                    stop_token_id=1,
                )

        self.assertIs(captured["ignore_eos"], True)
        self.assertEqual(captured["stop_token_ids"], [248046])

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
    def test_reexec_preserves_isolation_and_ignores_injected_python_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = root / "probe.py"
            probe.write_text(
                "import json, os, sys\n"
                "try:\n"
                " import injected_sentinel\n"
                " injected = True\n"
                "except ModuleNotFoundError:\n"
                " injected = False\n"
                "print(json.dumps({'isolated': sys.flags.isolated, "
                "'safe_path': sys.flags.safe_path, 'dont_write': "
                "sys.flags.dont_write_bytecode, 'injected': injected, "
                "'path': os.environ['PATH']}))\n"
            )
            (root / "injected_sentinel.py").write_text(
                "raise RuntimeError('PYTHONPATH sentinel executed')\n"
            )
            with mock.patch.dict(
                os.environ,
                {
                    "PYTHONPATH": str(root),
                    "PATH": str(root),
                    "LD_PRELOAD": "/tmp/forged.so",
                    "GIT_DIR": "/tmp/forged-git",
                },
            ):
                command = runner._python_reexec_command(
                    [str(probe)], isolated=True, dont_write_bytecode=True
                )
                environment = runner._reexec_environment(isolated=True)
            self.assertNotIn("LD_PRELOAD", environment)
            self.assertNotIn("GIT_DIR", environment)
            self.assertEqual(
                environment["LD_LIBRARY_PATH"], "/usr/local/cuda/lib64"
            )
            receipt = json.loads(
                subprocess.check_output(command, env=environment, text=True)
            )
        self.assertEqual(command[1:3], ["-I", "-B"])
        self.assertEqual(
            receipt,
            {
                "isolated": 1,
                "safe_path": True,
                "dont_write": 1,
                "injected": False,
                "path": runner._PINNED_EXECUTABLE_PATH,
            },
        )

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
                    "</think>\n\n": [248069, 271],
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
                self.llm_engine = SimpleNamespace(
                    vllm_config=SimpleNamespace(
                        compilation_config=_compilation_config(),
                        model_config=SimpleNamespace(logprobs_mode="raw_logprobs"),
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
                    max_num_seqs=15,
                    cudagraph_capture_sizes=effective,
                )
            )

        self.assertEqual(captured_engine_args["max_num_seqs"], 15)
        self.assertEqual(captured_engine_args["max_logprobs"], 24)
        self.assertEqual(captured_engine_args["logprobs_mode"], "raw_logprobs")
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
        self.assertEqual(instance.resolved_logprobs_mode, "raw_logprobs")
        instance.close()


class AnswerSeamTests(unittest.TestCase):
    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            if add_special_tokens:
                raise AssertionError("tests require exact no-special-token encoding")
            return {
                "": [],
                "prompt": [7],
                "PROGRAM:": [90, 91],
            }[text]

        def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
            if skip_special_tokens:
                raise AssertionError("special-token-preserving decode is required")
            pieces = {
                50: "reason",
                248069: "</think>",
                271: "\n\n",
                90: "PRO",
                91: "GRAM:",
                101: " A",
                102: " |",
                103: " B",
                248044: "<|endoftext|>",
                248046: "<|im_end|>",
            }
            return "".join(pieces[value] for value in token_ids)

    class FakeLLM:
        def __init__(self, calls: list[list[SimpleNamespace]]):
            self.calls = list(calls)
            self.prompts: list[list[dict[str, list[int]]]] = []
            self.params: list[list[SimpleNamespace]] = []

        def generate(self, prompts, params, **kwargs):
            self.prompts.append(prompts)
            self.params.append(params)
            completions = self.calls.pop(0)
            if len(completions) != len(prompts):
                raise AssertionError("fake completion geometry differs from prompts")
            return [
                SimpleNamespace(prompt_logprobs=None, outputs=[completion])
                for completion in completions
            ]

    @staticmethod
    def completion(
        token_ids: list[int], *, finish_reason: str = "stop", stop_reason=None
    ) -> SimpleNamespace:
        return SimpleNamespace(
            index=0,
            token_ids=token_ids,
            finish_reason=finish_reason,
            stop_reason=(
                (248044 if stop_reason is None else stop_reason)
                if finish_reason == "stop"
                else None
            ),
            cumulative_logprob=-1.0,
            logprobs=None,
        )

    def instance(self, calls: list[list[SimpleNamespace]]) -> runner.VLLMRunner:
        value = object.__new__(runner.VLLMRunner)
        value.config = runner.EngineConfig(max_model_len=1024)
        value.tokenizer = self.FakeTokenizer()
        value.hf_eos_id = 248044
        value.tokenizer_eos_id = 248046
        value.think_open_id = 248068
        value.think_close_id = 248069
        value.close_ids = [248069, 271]
        value.thinking_prompt_suffix_ids = [1, 2, 3]
        value.no_thinking_prompt_suffix_ids = [1, 2, 4]
        value.llm = self.FakeLLM(calls)
        value.lora_request = None
        value.adapter_info = None
        value.load_seconds = 0.0
        value.engine_args = {"seed": 0}
        value.resolved_cudagraph = {}
        value.resolved_logprobs_mode = "raw_logprobs"
        return value

    def run_generate(
        self,
        value: runner.VLLMRunner,
        sampling: runner.SamplingConfig,
    ):
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        with mock.patch.dict(sys.modules, {"vllm": fake_vllm}), mock.patch.object(
            runner.VLLMRunner,
            "runtime_metadata",
            return_value={"test": True},
        ):
            return value.generate(
                [{"id": "paired-record", "prompt": "prompt"}], sampling
            )

    def run_shared(
        self,
        value: runner.VLLMRunner,
        action,
    ):
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        with mock.patch.dict(sys.modules, {"vllm": fake_vllm}), mock.patch.object(
            runner.VLLMRunner,
            "runtime_metadata",
            return_value={"test": True},
        ):
            return action()

    def test_sampling_contract_rejects_unpaired_or_ambiguous_slot_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "natural thinking"):
            runner.SamplingConfig(
                thinking="natural", answer_prefix="PROGRAM:"
            ).validate()
        with self.assertRaisesRegex(ValueError, "requires force_answer_seam"):
            runner.SamplingConfig(
                thinking="budget",
                thinking_budget=8,
                answer_prefix="PROGRAM:",
            ).validate()
        with self.assertRaisesRegex(ValueError, "requires n=1"):
            runner.SamplingConfig(
                thinking="off", n=2, paired_answer_seed=True
            ).validate()

    def test_no_think_slot_is_injected_prefill_and_not_sampled_answer(self) -> None:
        value = self.instance(
            [[self.completion([101, 102, 103, 248044])]]
        )
        rows, metadata = self.run_generate(
            value,
            runner.SamplingConfig(
                thinking="off",
                max_tokens=24,
                answer_max_tokens=24,
                greedy=True,
                allow_custom_prompts=True,
                answer_prefix="PROGRAM:",
                paired_answer_seed=True,
                run_seed=19,
            ),
        )

        self.assertEqual(value.llm.prompts, [[{"prompt_token_ids": [7, 90, 91]}]])
        output = rows[0]["outputs"][0]
        self.assertEqual(output["text"], "PROGRAM: A | B")
        self.assertEqual(output["token_ids"], [90, 91, 101, 102, 103])
        self.assertEqual(output["stage1_token_ids"], [101, 102, 103, 248044])
        self.assertEqual(output["answer_prefix_token_ids"], [90, 91])
        self.assertEqual(output["n_answer_tokens"], 3)
        self.assertEqual(output["n_injected_tokens"], 2)
        self.assertEqual(output["n_stage1_prompt_tokens"], 3)
        self.assertEqual(output["seed_domain_stage1"], "answer")
        self.assertEqual(metadata["counts"]["sampled_tokens"], 4)
        self.assertEqual(metadata["counts"]["injected_tokens"], 2)

    def test_think_slot_forces_registered_prefix_after_any_natural_close(self) -> None:
        value = self.instance(
            [
                [self.completion([50, 248069, 77, 248044])],
                [self.completion([101, 102, 103, 248044])],
            ]
        )
        rows, _metadata = self.run_generate(
            value,
            runner.SamplingConfig(
                thinking="budget",
                thinking_budget=8,
                max_tokens=8,
                answer_max_tokens=24,
                greedy=True,
                allow_custom_prompts=True,
                answer_prefix="PROGRAM:",
                force_answer_seam=True,
                paired_answer_seed=True,
                run_seed=19,
            ),
        )

        self.assertEqual(
            value.llm.prompts,
            [
                [{"prompt_token_ids": [7]}],
                [
                    {
                        "prompt_token_ids": [
                            7,
                            50,
                            248069,
                            271,
                            90,
                            91,
                        ]
                    }
                ],
            ],
        )
        output = rows[0]["outputs"][0]
        self.assertEqual(output["text"], "reason</think>\n\nPROGRAM: A | B")
        self.assertEqual(output["retained_thinking_token_ids"], [50])
        self.assertEqual(
            output["injected_token_ids"], [248069, 271, 90, 91]
        )
        self.assertEqual(output["n_thinking_tokens"], 1)
        self.assertEqual(output["n_answer_tokens"], 3)
        self.assertEqual(output["n_injected_tokens"], 4)
        self.assertEqual(output["seed_domain_stage1"], "thought")
        self.assertEqual(output["seed_domain_stage2"], "answer")
        self.assertTrue(output["forced_close"])

    def test_answer_seed_is_paired_across_no_think_and_think_slot(self) -> None:
        no_think = self.instance(
            [[self.completion([101, 102, 103, 248044])]]
        )
        no_rows, _ = self.run_generate(
            no_think,
            runner.SamplingConfig(
                thinking="off",
                max_tokens=24,
                greedy=True,
                allow_custom_prompts=True,
                answer_prefix="PROGRAM:",
                paired_answer_seed=True,
                run_seed=23,
            ),
        )
        thinking = self.instance(
            [
                [self.completion([50], finish_reason="length")],
                [self.completion([101, 102, 103, 248044])],
            ]
        )
        think_rows, _ = self.run_generate(
            thinking,
            runner.SamplingConfig(
                thinking="budget",
                thinking_budget=1,
                max_tokens=1,
                answer_max_tokens=24,
                greedy=True,
                allow_custom_prompts=True,
                answer_prefix="PROGRAM:",
                force_answer_seam=True,
                paired_answer_seed=True,
                run_seed=23,
            ),
        )
        self.assertEqual(
            no_rows[0]["outputs"][0]["seed_stage1"],
            think_rows[0]["outputs"][0]["seed_stage2"],
        )

    def test_thinking_arms_fork_from_one_exact_persistable_token_prefix(self) -> None:
        value = self.instance(
            [
                [self.completion([50, 248069, 77, 248044])],
                [self.completion([101, 102, 103, 248044])],
                [self.completion([101, 102, 103, 248044])],
            ]
        )
        records = [{"id": "paired-record", "prompt": "prompt"}]
        source_sampling = runner.SamplingConfig(
            thinking="budget",
            thinking_budget=8,
            max_tokens=8,
            answer_max_tokens=24,
            greedy=True,
            allow_custom_prompts=True,
            force_answer_seam=True,
            paired_answer_seed=True,
            run_seed=29,
        )

        def execute():
            thought_rows, thought_metadata = value.generate_thought_prefixes(
                records, source_sampling
            )
            persisted_rows = json.loads(json.dumps(thought_rows))
            persisted_metadata = json.loads(json.dumps(thought_metadata))
            free_rows, free_metadata = value.generate_from_thought_prefixes(
                records, persisted_rows, persisted_metadata, source_sampling
            )
            slot_rows, slot_metadata = value.generate_from_thought_prefixes(
                records,
                persisted_rows,
                persisted_metadata,
                dataclasses.replace(source_sampling, answer_prefix="PROGRAM:"),
            )
            return (
                thought_rows,
                thought_metadata,
                free_rows,
                free_metadata,
                slot_rows,
                slot_metadata,
            )

        (
            thought_rows,
            thought_metadata,
            free_rows,
            free_metadata,
            slot_rows,
            slot_metadata,
        ) = self.run_shared(value, execute)

        self.assertEqual(
            value.llm.prompts,
            [
                [{"prompt_token_ids": [7]}],
                [{"prompt_token_ids": [7, 50, 248069, 271]}],
                [{"prompt_token_ids": [7, 50, 248069, 271, 90, 91]}],
            ],
        )
        source = thought_rows[0]["outputs"][0]
        free = free_rows[0]["outputs"][0]
        slot = slot_rows[0]["outputs"][0]
        self.assertEqual(source["stage1_token_ids"], [50, 248069, 77, 248044])
        self.assertEqual(source["retained_thinking_token_ids"], [50])
        self.assertEqual(source["n_tokens_discarded_after_close"], 2)
        self.assertEqual(free["stage1_token_ids"], source["stage1_token_ids"])
        self.assertEqual(slot["stage1_token_ids"], source["stage1_token_ids"])
        self.assertEqual(free["retained_thinking_token_ids"], [50])
        self.assertEqual(slot["retained_thinking_token_ids"], [50])
        self.assertEqual(free["seed_stage2"], slot["seed_stage2"])
        self.assertEqual(free["n_stage2_prompt_tokens"], 4)
        self.assertEqual(slot["n_stage2_prompt_tokens"], 6)
        self.assertEqual(thought_metadata["generation_mode"], "shared_thought_prefixes")
        self.assertEqual(free_metadata["counts"]["physical_sampled_tokens"], 4)
        self.assertEqual(free_metadata["counts"]["reused_sampled_tokens"], 4)
        self.assertEqual(free_metadata["counts"]["logical_prompt_tokens"], 5)
        self.assertEqual(free_metadata["counts"]["physical_prompt_tokens"], 4)
        self.assertEqual(free_metadata["counts"]["reused_prompt_tokens"], 1)
        self.assertEqual(free_metadata["counts"]["logical_model_tokens"], 13)
        self.assertEqual(free_metadata["counts"]["physical_model_tokens"], 8)
        self.assertEqual(free_metadata["counts"]["reused_model_tokens"], 5)
        self.assertEqual(slot_metadata["counts"]["physical_sampled_tokens"], 4)
        self.assertEqual(slot_metadata["counts"]["reused_sampled_tokens"], 4)
        self.assertEqual(slot_metadata["counts"]["logical_prompt_tokens"], 7)
        self.assertEqual(slot_metadata["counts"]["physical_prompt_tokens"], 6)
        self.assertEqual(slot_metadata["counts"]["reused_prompt_tokens"], 1)
        self.assertEqual(slot_metadata["counts"]["logical_model_tokens"], 15)
        self.assertEqual(slot_metadata["counts"]["physical_model_tokens"], 10)
        self.assertEqual(slot_metadata["counts"]["reused_model_tokens"], 5)

    def test_no_think_boundary_pair_is_adjacent_seed_paired_and_untrimmed(self) -> None:
        value = self.instance(
            [
                [
                    self.completion([101, 248046], stop_reason=248046),
                    self.completion([101, 102, 248044], stop_reason=248044),
                ]
            ]
        )
        sampling = runner.SamplingConfig(
            thinking="off",
            max_tokens=24,
            answer_max_tokens=24,
            greedy=True,
            allow_custom_prompts=True,
            answer_prefix="PROGRAM:",
            paired_answer_seed=True,
            run_seed=37,
        )
        rows, metadata = self.run_shared(
            value,
            lambda: value.generate_boundary_pairs(
                [{"id": "paired-record", "prompt": "prompt"}], sampling
            ),
        )

        self.assertEqual(
            value.llm.prompts,
            [
                [
                    {"prompt_token_ids": [7, 90, 91]},
                    {"prompt_token_ids": [7, 90, 91]},
                ]
            ],
        )
        params = value.llm.params[0]
        self.assertEqual([item.stop_token_ids for item in params], [[248046], [248044]])
        self.assertEqual(params[0].seed, params[1].seed)
        tokenizer_output, hf_output = rows[0]["outputs"]
        self.assertEqual(tokenizer_output["batch_position"], 0)
        self.assertEqual(hf_output["batch_position"], 1)
        self.assertEqual(tokenizer_output["boundary"], "tokenizer_eos")
        self.assertEqual(hf_output["boundary"], "hf_model_eos")
        self.assertEqual(
            tokenizer_output["raw_answer_token_ids"], [101, 248046]
        )
        self.assertEqual(hf_output["raw_answer_token_ids"], [101, 102, 248044])
        self.assertEqual(
            tokenizer_output["token_ids"], [90, 91, 101, 248046]
        )
        self.assertEqual(tokenizer_output["text"], "PROGRAM: A<|im_end|>")
        self.assertEqual(tokenizer_output["n_terminal_tokens_trimmed"], 0)
        self.assertEqual(rows[0]["effective_prompt_token_ids"], [7, 90, 91])
        self.assertEqual(metadata["generation_mode"], "answer_boundary_pairs")
        self.assertEqual(metadata["counts"]["completions"], 2)
        self.assertEqual(metadata["counts"]["physical_sampled_tokens"], 5)
        self.assertEqual(
            metadata["boundary_pairing"]["registered_stop_token_ids"],
            [248046, 248044],
        )

    def test_thinking_boundary_pair_reuses_one_authenticated_thought(self) -> None:
        value = self.instance(
            [
                [self.completion([50, 248069, 77, 248044])],
                [
                    self.completion([101, 248046], stop_reason=248046),
                    self.completion([101, 103, 248044], stop_reason=248044),
                ],
            ]
        )
        records = [{"id": "paired-record", "prompt": "prompt"}]
        source_sampling = runner.SamplingConfig(
            thinking="budget",
            thinking_budget=8,
            max_tokens=8,
            answer_max_tokens=24,
            greedy=True,
            allow_custom_prompts=True,
            force_answer_seam=True,
            paired_answer_seed=True,
            run_seed=41,
        )

        def execute():
            thought_rows, thought_metadata = value.generate_thought_prefixes(
                records, source_sampling
            )
            pair_rows, pair_metadata = value.generate_boundary_pairs(
                records,
                dataclasses.replace(source_sampling, answer_prefix="PROGRAM:"),
                thought_rows=json.loads(json.dumps(thought_rows)),
                thought_metadata=json.loads(json.dumps(thought_metadata)),
            )
            return thought_rows, pair_rows, pair_metadata

        thought_rows, rows, metadata = self.run_shared(value, execute)
        expected_prompt = [7, 50, 248069, 271, 90, 91]
        self.assertEqual(
            value.llm.prompts,
            [
                [{"prompt_token_ids": [7]}],
                [
                    {"prompt_token_ids": expected_prompt},
                    {"prompt_token_ids": expected_prompt},
                ],
            ],
        )
        self.assertEqual(
            thought_rows[0]["outputs"][0]["stage1_token_ids"],
            [50, 248069, 77, 248044],
        )
        tokenizer_output, hf_output = rows[0]["outputs"]
        self.assertEqual(tokenizer_output["retained_thinking_token_ids"], [50])
        self.assertEqual(hf_output["retained_thinking_token_ids"], [50])
        self.assertEqual(tokenizer_output["stage1_token_ids"], hf_output["stage1_token_ids"])
        self.assertEqual(tokenizer_output["answer_seed"], hf_output["answer_seed"])
        self.assertEqual(
            tokenizer_output["token_ids"],
            [50, 248069, 271, 90, 91, 101, 248046],
        )
        self.assertEqual(metadata["generation_mode"], "shared_thought_boundary_pairs")
        self.assertEqual(metadata["counts"]["physical_sampled_tokens"], 5)
        self.assertEqual(metadata["counts"]["reused_sampled_tokens"], 8)
        self.assertIn("thought_source_sha256", metadata)

    def test_boundary_pair_contract_rejects_wrong_cap_and_thought_geometry(self) -> None:
        value = self.instance([])
        records = [{"id": "paired-record", "prompt": "prompt"}]
        with self.assertRaisesRegex(ValueError, "answer_max_tokens=24"):
            self.run_shared(
                value,
                lambda: value.generate_boundary_pairs(
                    records,
                    runner.SamplingConfig(
                        thinking="off",
                        max_tokens=24,
                        answer_max_tokens=23,
                        greedy=True,
                        allow_custom_prompts=True,
                        paired_answer_seed=True,
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "require persisted thought"):
            self.run_shared(
                value,
                lambda: value.generate_boundary_pairs(
                    records,
                    runner.SamplingConfig(
                        thinking="budget",
                        thinking_budget=8,
                        max_tokens=8,
                        answer_max_tokens=24,
                        greedy=True,
                        allow_custom_prompts=True,
                        force_answer_seam=True,
                        paired_answer_seed=True,
                    ),
                ),
            )

    def test_shared_thought_authentication_rejects_identity_token_and_runner_drift(self) -> None:
        value = self.instance(
            [[self.completion([50], finish_reason="length")]]
        )
        records = [{"id": "paired-record", "prompt": "prompt"}]
        sampling = runner.SamplingConfig(
            thinking="budget",
            thinking_budget=1,
            max_tokens=1,
            answer_max_tokens=24,
            greedy=True,
            allow_custom_prompts=True,
            force_answer_seam=True,
            paired_answer_seed=True,
            run_seed=31,
        )
        thought_rows, thought_metadata = self.run_shared(
            value,
            lambda: value.generate_thought_prefixes(records, sampling),
        )
        self.assertEqual(
            thought_rows[0]["outputs"][0]["retained_thinking_token_ids"], [50]
        )

        mutations = []
        bad_id = copy.deepcopy(thought_rows)
        bad_id[0]["id"] = "other"
        mutations.append((bad_id, thought_metadata, "identity"))
        bad_token = copy.deepcopy(thought_rows)
        bad_token[0]["outputs"][0]["retained_thinking_token_ids"] = []
        mutations.append((bad_token, thought_metadata, "token/seed"))
        bad_metadata = copy.deepcopy(thought_metadata)
        bad_metadata["runner_sha256"] = "0" * 64
        mutations.append((thought_rows, bad_metadata, "metadata"))

        for rows, metadata, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(
                RuntimeError, message
            ):
                self.run_shared(
                    value,
                    lambda rows=rows, metadata=metadata: value.generate_from_thought_prefixes(
                        records, rows, metadata, sampling
                    ),
                )
        self.assertEqual(len(value.llm.prompts), 1)


if __name__ == "__main__":
    unittest.main()
