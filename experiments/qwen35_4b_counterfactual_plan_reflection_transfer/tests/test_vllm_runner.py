from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
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
    def test_merged_model_override_is_existing_and_mutually_exclusive(self) -> None:
        runner.EngineConfig(model_override=RUNNER_PATH.parent).validate()
        with self.assertRaisesRegex(ValueError, "existing merged-checkpoint"):
            runner.EngineConfig(model_override=RUNNER_PATH.parent / "missing").validate()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            runner.EngineConfig(
                model_override=RUNNER_PATH.parent,
                adapter=RUNNER_PATH.parent,
            ).validate()

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
                self.llm_engine = SimpleNamespace(
                    vllm_config=SimpleNamespace(
                        compilation_config=_compilation_config()
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
