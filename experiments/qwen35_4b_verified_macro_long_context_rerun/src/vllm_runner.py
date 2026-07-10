#!/usr/bin/env python3
"""Single-file, experiment-local vLLM runner for Qwen/Qwen3.5-4B.

Copy this file into an experiment's ``src/`` directory and either import
``VLLMRunner`` or execute it as a JSONL command-line tool.  It deliberately
has no repository-local imports.

Input JSONL rows have a unique ``id`` and exactly one of:

    {"id": "p1", "prompt": "an already rendered prompt"}
    {"id": "p2", "messages": [{"role": "user", "content": "..."}]}

An optional JSON-valued ``meta`` field is copied to the output.  The CLI emits
one row per input with an ``outputs`` list and writes a ``.meta.json`` sidecar.

Scientific invariants:

* The only accepted model is Qwen/Qwen3.5-4B at the pinned repository revision.
* Chat messages are rendered here, then exact prompt token IDs go to vLLM.
* Sampling parameters are always explicit; Hugging Face generation defaults
  are never inherited.
* Budgeted thinking defaults to the repository's historical two-stage
  force-close protocol, not vLLM's semantically different native budget.
* Seeds are stable under input reordering and are recorded per sample/stage.
* vLLM asynchronous scheduling is disabled because mixed termination lengths can
  otherwise change later requests' sampled RNG trajectories despite fixed seeds.
* vLLM and Transformers samples are not RNG-identical.  Never mix backends
  between experimental arms or matched-compute baselines.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


# vLLM's reproducibility guide requires the in-process V1 engine for offline
# deterministic scheduling.  This must be set before importing vllm.
_V1_MULTIPROCESSING = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
if _V1_MULTIPROCESSING not in (None, "0"):
    raise RuntimeError(
        "VLLM_ENABLE_V1_MULTIPROCESSING must be unset or 0 for reproducible offline runs; "
        f"got {_V1_MULTIPROCESSING!r}"
    )
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Calling ``.venv-vllm/bin/python`` directly does not activate the venv, so
# console tools installed beside it (notably ninja, used by FlashInfer JIT)
# are otherwise invisible.  Keep direct invocation as reliable as activation.
_PYTHON_BIN = str(Path(sys.prefix) / "bin")
if _PYTHON_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _PYTHON_BIN + os.pathsep + os.environ.get("PATH", "")

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RUNNER_SCHEMA_VERSION = 3

THINK_TEMPERATURE = 0.6
THINK_TOP_P = 0.95
THINK_TOP_K = 20
NO_THINK_TEMPERATURE = 0.7
NO_THINK_TOP_P = 0.8
NO_THINK_TOP_K = 20
MIN_NONZERO_TEMPERATURE = 0.01
MAX_TEMPERATURE = 2.0
MAX_N = 16_384


def _stable_seed(run_seed: int, record_id: str, sample_index: int, stage: str) -> int:
    payload = f"{run_seed}\0{record_id}\0{sample_index}\0{stage}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_text(command: Sequence[str]) -> str:
    try:
        return subprocess.run(
            list(command), check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _installed_packages() -> dict[str, str]:
    """Return the full distribution inventory needed to reproduce kernel/runtime state."""
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower().replace("_", "-")] = distribution.version
    return dict(sorted(packages.items()))


# Importing vLLM adds setuptools' vendored distributions to sys.path. Snapshot
# the actual environment first so the sidecar matches the uv lock instead of
# reporting those implementation-detail copies as separately installed wheels.
_INITIAL_PACKAGES = _installed_packages()


def _environment_lock_metadata() -> dict[str, str] | None:
    """Find the repository lock even after this file is copied into an experiment."""
    starts = (Path(__file__).resolve().parent, Path.cwd().resolve())
    visited: set[Path] = set()
    for start in starts:
        for directory in (start, *start.parents):
            if directory in visited:
                continue
            visited.add(directory)
            lock = directory / "requirements-vllm.lock.txt"
            if lock.is_file():
                return {
                    "path": str(lock),
                    "sha256": _sha256_file(lock),
                }
    return None


def _jsonable_logprobs(value: Any) -> Any:
    """Convert vLLM's nested Logprob objects without importing private types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable_logprobs(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_logprobs(item) for item in value]
    if hasattr(value, "logprob"):
        return {
            "logprob": float(value.logprob),
            "rank": getattr(value, "rank", None),
            "decoded_token": getattr(value, "decoded_token", None),
        }
    raise TypeError(f"cannot serialize logprob object of type {type(value).__name__}")


@dataclasses.dataclass(frozen=True)
class EngineConfig:
    """Engine settings chosen once, before the model is loaded."""

    max_model_len: int = 16_384
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 32_768
    enable_prefix_caching: bool = False
    enforce_eager: bool = False
    adapter: Path | None = None

    def validate(self) -> None:
        if self.max_model_len < 256:
            raise ValueError("max_model_len must be at least 256")
        if not 0.1 <= self.gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory_utilization must be in [0.1, 1.0)")
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1:
            raise ValueError("max_num_seqs and max_num_batched_tokens must be positive")


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
    """Explicit generation settings shared by a run."""

    thinking: str = "off"  # off | natural | budget
    thinking_budget: int | None = None
    n: int = 1
    max_tokens: int = 512
    answer_max_tokens: int = 512
    greedy: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    run_seed: int = 0
    shuffle_thinking: bool = False
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    logprob_token_ids: tuple[int, ...] = ()
    allow_custom_prompts: bool = False

    def validate(self) -> None:
        if self.thinking not in {"off", "natural", "budget"}:
            raise ValueError("thinking must be one of: off, natural, budget")
        if self.thinking == "budget":
            if self.thinking_budget is None or self.thinking_budget < 1:
                raise ValueError("budget thinking requires thinking_budget >= 1")
        elif self.thinking_budget is not None:
            raise ValueError("thinking_budget is only valid with thinking='budget'")
        if self.n < 1:
            raise ValueError("n must be positive")
        if self.n > MAX_N:
            raise ValueError(f"n cannot exceed this runner's vLLM limit of {MAX_N}")
        if self.greedy and self.n != 1:
            raise ValueError("greedy generation requires n=1")
        if self.max_tokens < 1 or self.answer_max_tokens < 1:
            raise ValueError("token caps must be positive")
        if self.shuffle_thinking and self.thinking != "budget":
            raise ValueError("shuffle_thinking is only valid for budget thinking")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError("logprobs must be non-negative")
        if self.logprobs is not None and self.logprobs > 20:
            raise ValueError("logprobs cannot exceed this runner's vLLM max_logprobs=20")
        if self.prompt_logprobs is not None and self.prompt_logprobs < 0:
            raise ValueError("prompt_logprobs must be non-negative")
        if self.prompt_logprobs is not None and self.prompt_logprobs > 20:
            raise ValueError(
                "prompt_logprobs cannot exceed this runner's vLLM max_logprobs=20"
            )
        if self.prompt_logprobs is not None and self.thinking == "budget":
            raise ValueError(
                "prompt_logprobs with two-stage budget thinking are not yet supported; "
                "score the completed sequences in a separate pass"
            )
        if self.logprob_token_ids and self.logprobs != len(self.logprob_token_ids):
            raise ValueError(
                "vLLM requires logprobs to equal the number of logprob_token_ids; "
                f"got logprobs={self.logprobs!r} and {len(self.logprob_token_ids)} IDs"
            )
        resolved = self.resolved_sampling()
        finite_values = {
            "temperature": resolved["temperature"],
            "top_p": resolved["top_p"],
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
        }
        for name, value in finite_values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(resolved["temperature"]) < 0:
            raise ValueError("temperature must be non-negative")
        if 0 < float(resolved["temperature"]) < MIN_NONZERO_TEMPERATURE:
            raise ValueError(
                f"nonzero temperature must be at least {MIN_NONZERO_TEMPERATURE}; "
                "vLLM silently clamps smaller values"
            )
        if float(resolved["temperature"]) > MAX_TEMPERATURE:
            raise ValueError(f"temperature must not exceed {MAX_TEMPERATURE}")
        if not 0 < float(resolved["top_p"]) <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if int(resolved["top_k"]) < -1:
            raise ValueError("top_k must be 0/-1 (disabled) or positive")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")
        if not -2 <= self.presence_penalty <= 2:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if not -2 <= self.frequency_penalty <= 2:
            raise ValueError("frequency_penalty must be in [-2, 2]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if float(resolved["temperature"]) == 0 and self.n != 1:
            raise ValueError("effective greedy generation (temperature=0) requires n=1")

    def resolved_sampling(self) -> dict[str, float | int]:
        if self.greedy or self.temperature == 0:
            return {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "repetition_penalty": self.repetition_penalty,
            }
        thinking = self.thinking != "off"
        return {
            "temperature": self.temperature
            if self.temperature is not None
            else (THINK_TEMPERATURE if thinking else NO_THINK_TEMPERATURE),
            "top_p": self.top_p
            if self.top_p is not None
            else (THINK_TOP_P if thinking else NO_THINK_TOP_P),
            "top_k": self.top_k
            if self.top_k is not None
            else (THINK_TOP_K if thinking else NO_THINK_TOP_K),
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
        }


@dataclasses.dataclass(frozen=True)
class _PreparedRecord:
    record_id: str
    meta: Any
    prompt_text: str
    prompt_token_ids: list[int]
    prompt_channel: str


def _validate_adapter(adapter: Path | None) -> dict[str, Any] | None:
    if adapter is None:
        return None
    adapter = adapter.expanduser().resolve()
    config_path = adapter / "adapter_config.json"
    weights = sorted(adapter.glob("*.safetensors"))
    if not config_path.is_file():
        raise ValueError(f"adapter is missing adapter_config.json: {adapter}")
    if not weights:
        raise ValueError(f"adapter has no .safetensors weights: {adapter}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("only PEFT LoRA adapters are supported")
    base = str(config.get("base_model_name_or_path", ""))
    accepted_base = (
        not base
        or base == MODEL_ID
        or "models--Qwen--Qwen3.5-4B" in base
        or MODEL_REVISION in base
    )
    if not accepted_base:
        raise ValueError(f"adapter targets a different base model: {base!r}")
    if config.get("use_dora"):
        raise ValueError("DoRA adapters are not supported by this runner")
    if config.get("modules_to_save"):
        raise ValueError("adapters with modules_to_save are not supported")
    if str(config.get("bias", "none")).lower() != "none":
        raise ValueError("LoRA bias weights are not supported")
    if config.get("rank_pattern"):
        raise ValueError("per-module rank_pattern adapters are not supported")
    if config.get("alpha_pattern"):
        raise ValueError("per-module alpha_pattern adapters are not supported")
    rank = int(config.get("r", 0))
    if rank < 1:
        raise ValueError("adapter rank r must be positive")
    if rank not in {1, 8, 16, 32, 64, 128, 256, 320, 512}:
        raise ValueError(f"adapter rank {rank} is not supported by vLLM 0.24")
    digest = hashlib.sha256()
    for path in weights:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return {
        "path": str(adapter),
        "rank": rank,
        "base_model_name_or_path": base,
        "target_modules": config.get("target_modules"),
        "config_sha256": _sha256_file(config_path),
        "weights_sha256": digest.hexdigest(),
    }


class VLLMRunner:
    """Importable high-throughput runner for text generation and PEFT LoRA."""

    def __init__(self, config: EngineConfig = EngineConfig()):
        config.validate()
        self.config = config
        self.adapter_info = _validate_adapter(config.adapter)
        self._closed = False

        # In-process vLLM intentionally seeds Python, NumPy, and Torch. Capture
        # the caller's streams before any model/tokenizer setup so constructing
        # this importable runner cannot alter later procedural task generation.
        python_rng_state = random.getstate()
        import numpy as np
        import torch

        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        torch_initial_seed = torch.initial_seed()
        cuda_was_initialized = torch.cuda.is_initialized()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if cuda_was_initialized else None
        )

        # Tokenize ourselves so the exact input IDs are auditable and do not
        # depend on vLLM's chat endpoint or reasoning parser.
        from transformers import AutoConfig, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            use_fast=True,
        )
        model_config = AutoConfig.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
        )
        self.hf_eos_id = int(model_config.text_config.eos_token_id)
        if self.hf_eos_id != 248044:
            raise RuntimeError(
                f"Qwen3.5 model EOS changed from 248044 to {self.hf_eos_id}; audit termination"
            )
        self.think_open_id = self._single_token_id("<think>")
        self.think_close_id = self._single_token_id("</think>")
        self.close_ids = self.tokenizer.encode(
            "</think>\n\n", add_special_tokens=False
        )
        if not self.close_ids or self.close_ids[0] != self.think_close_id:
            raise RuntimeError("unexpected Qwen3.5 </think> tokenization")
        if (self.think_open_id, self.think_close_id) != (248068, 248069):
            raise RuntimeError(
                "Qwen3.5 think-token IDs changed; audit prompts before continuing: "
                f"{self.think_open_id}, {self.think_close_id}"
            )

        self.thinking_prompt_suffix_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n<think>\n", add_special_tokens=False
        )
        self.no_thinking_prompt_suffix_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )

        engine_args: dict[str, Any] = {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_num_seqs": config.max_num_seqs,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "max_cudagraph_capture_size": config.max_num_seqs,
            "language_model_only": True,
            "enable_prefix_caching": config.enable_prefix_caching,
            "mamba_cache_mode": "align" if config.enable_prefix_caching else "none",
            "enforce_eager": config.enforce_eager,
            "generation_config": "vllm",
            "max_logprobs": 20,
            "seed": 0,
            # vLLM 0.24 auto-enables async scheduling.  With more logical
            # sequences than max_num_seqs, early termination in the first wave
            # can then alter the sampled trajectories of later requests even
            # under explicit per-request seeds.  Offline research favors stable
            # request-local RNG over that throughput optimization.
            "async_scheduling": False,
        }
        if self.adapter_info is not None:
            engine_args.update(
                enable_lora=True,
                max_loras=1,
                max_cpu_loras=1,
                max_lora_rank=self.adapter_info["rank"],
            )

        try:
            from vllm import LLM

            started = time.perf_counter()
            self.llm = LLM(**engine_args)
            self.load_seconds = time.perf_counter() - started
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            elif torch.cuda.is_initialized():
                torch.cuda.manual_seed_all(torch_initial_seed)
        self.engine_args = engine_args

        self.lora_request = None
        if self.adapter_info is not None:
            from vllm.lora.request import LoRARequest

            name_hash = hashlib.sha256(
                self.adapter_info["path"].encode("utf-8")
            ).hexdigest()[:12]
            self.lora_request = LoRARequest(
                f"experiment-{name_hash}", 1, self.adapter_info["path"]
            )

    def close(self) -> None:
        """Release vLLM's engine resources and distributed process group."""
        if self._closed:
            return
        self._closed = True
        llm = getattr(self, "llm", None)
        engine = getattr(llm, "llm_engine", None)
        client = getattr(engine, "engine_core", None)
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def __enter__(self) -> "VLLMRunner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _single_token_id(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"expected one token for {text!r}, got {ids}")
        return int(ids[0])

    def _render_messages(self, messages: Any, enable_thinking: bool) -> str:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("each message must be an object")
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"unsupported message role: {message.get('role')!r}")
            if not isinstance(message.get("content"), str):
                raise ValueError("this text-only runner requires string message content")
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError as exc:
            raise RuntimeError(
                "the pinned Qwen3.5 chat template rejected enable_thinking; "
                "refusing a silent prompt-format fallback"
            ) from exc
        if not isinstance(rendered, str):
            raise RuntimeError("chat template did not return rendered text")
        return rendered

    def _prompt_channel(self, token_ids: Sequence[int]) -> str:
        """Classify the exact generation suffix produced by Qwen's chat template."""
        if (
            list(token_ids[-len(self.no_thinking_prompt_suffix_ids) :])
            == self.no_thinking_prompt_suffix_ids
        ):
            return "off"
        if (
            list(token_ids[-len(self.thinking_prompt_suffix_ids) :])
            == self.thinking_prompt_suffix_ids
        ):
            return "thinking"
        return "custom"

    def prepare(
        self,
        records: Sequence[dict[str, Any]],
        thinking: str,
        allow_custom_prompts: bool = False,
    ) -> list[_PreparedRecord]:
        seen: set[str] = set()
        prepared: list[_PreparedRecord] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"input row {index} is not a JSON object")
            record_id = str(record.get("id", ""))
            if not record_id:
                raise ValueError(f"input row {index} has no non-empty id")
            if record_id in seen:
                raise ValueError(f"duplicate input id: {record_id!r}")
            seen.add(record_id)
            has_prompt = "prompt" in record
            has_messages = "messages" in record
            if has_prompt == has_messages:
                raise ValueError(
                    f"input {record_id!r} must contain exactly one of prompt or messages"
                )
            if has_prompt:
                prompt = record["prompt"]
                if not isinstance(prompt, str):
                    raise ValueError(f"input {record_id!r} prompt must be a string")
            else:
                prompt = self._render_messages(record["messages"], thinking != "off")
            token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if not token_ids:
                raise ValueError(f"input {record_id!r} produced an empty prompt")
            prompt_channel = self._prompt_channel(token_ids)
            expected_channel = "off" if thinking == "off" else "thinking"
            if prompt_channel != expected_channel and not allow_custom_prompts:
                raise ValueError(
                    f"input {record_id!r} has {prompt_channel!r} prompt channel, but "
                    f"thinking={thinking!r} requires {expected_channel!r}; rerender it or use "
                    "allow_custom_prompts=True/--allow-custom-prompts for an audited custom format"
                )
            prepared.append(
                _PreparedRecord(
                    record_id,
                    record.get("meta"),
                    prompt,
                    list(token_ids),
                    prompt_channel,
                )
            )
        return prepared

    def _check_context(self, records: Sequence[_PreparedRecord], sampling: SamplingConfig) -> None:
        if sampling.thinking == "budget":
            reserve = (
                int(sampling.thinking_budget)
                + len(self.close_ids)
                + sampling.answer_max_tokens
            )
        else:
            reserve = sampling.max_tokens
        too_long = [
            (record.record_id, len(record.prompt_token_ids) + reserve)
            for record in records
            if len(record.prompt_token_ids) + reserve > self.config.max_model_len
        ]
        if too_long:
            detail = ", ".join(f"{rid}={length}" for rid, length in too_long[:5])
            raise ValueError(
                f"prompt + generation cap exceeds max_model_len={self.config.max_model_len}: "
                + detail
            )

    def _params(self, sampling: SamplingConfig, *, max_tokens: int, seed: int, n: int):
        from vllm import SamplingParams

        resolved = sampling.resolved_sampling()
        return SamplingParams(
            n=n,
            temperature=float(resolved["temperature"]),
            top_p=float(resolved["top_p"]),
            top_k=int(resolved["top_k"]),
            min_p=float(resolved["min_p"]),
            presence_penalty=float(resolved["presence_penalty"]),
            frequency_penalty=float(resolved["frequency_penalty"]),
            repetition_penalty=float(resolved["repetition_penalty"]),
            seed=seed,
            max_tokens=max_tokens,
            # The tokenizer marks <|im_end|> as EOS, while the pinned HF model
            # generation config stops one token later at <|endoftext|>.  Ignore
            # vLLM's tokenizer EOS and use the model EOS so migrations preserve
            # the established HF answer boundary and token accounting.
            ignore_eos=True,
            stop_token_ids=[self.hf_eos_id],
            logprobs=sampling.logprobs,
            prompt_logprobs=sampling.prompt_logprobs,
            logprob_token_ids=list(sampling.logprob_token_ids) or None,
            detokenize=True,
            skip_special_tokens=False,
        )

    def _decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def _trim_hf_eos(self, token_ids: Sequence[int]) -> list[int]:
        ids = list(token_ids)
        return ids[: ids.index(self.hf_eos_id)] if self.hf_eos_id in ids else ids

    def _ordinary_output(
        self,
        record: _PreparedRecord,
        completion: Any,
        sample_index: int,
        seed: int,
        thinking: str,
    ) -> dict[str, Any]:
        sampled_ids = list(completion.token_ids)
        token_ids = self._trim_hf_eos(sampled_ids)
        close_index = (
            token_ids.index(self.think_close_id)
            if self.think_close_id in token_ids
            else None
        )
        if thinking == "off":
            n_thinking = 0
            n_answer = len(token_ids)
        elif close_index is None:
            n_thinking = len(token_ids)
            n_answer = 0
        else:
            n_thinking = close_index
            n_answer = len(token_ids) - close_index - 1
        return {
            "sample_index": sample_index,
            "stage1_parent_seed": seed,
            "seed_stage1": seed + sample_index,
            "seed_stage2": None,
            "text": self._decode(token_ids),
            "token_ids": token_ids,
            "stage1_token_ids": sampled_ids,
            "injected_token_ids": [],
            "stage2_token_ids": [],
            "n_thinking_tokens": n_thinking,
            "n_answer_tokens": n_answer,
            "n_sampled_tokens": len(sampled_ids),
            "n_injected_tokens": 0,
            "n_completion_tokens": len(token_ids),
            "n_terminal_tokens_trimmed": len(sampled_ids) - len(token_ids),
            "n_stage1_prompt_tokens": len(record.prompt_token_ids),
            "n_stage2_prompt_tokens": 0,
            "thinking_closed": close_index is not None,
            "forced_close": False,
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
            "stage1_finish_reason": completion.finish_reason,
            "stage1_stop_reason": completion.stop_reason,
            "truncated": completion.finish_reason == "length",
            "stage1_cumulative_logprob": completion.cumulative_logprob,
            "stage2_cumulative_logprob": None,
            "sampled_cumulative_logprob": completion.cumulative_logprob,
            "stage1_logprobs": _jsonable_logprobs(completion.logprobs),
            "stage2_logprobs": None,
        }

    def generate(
        self,
        records: Sequence[dict[str, Any]],
        sampling: SamplingConfig = SamplingConfig(),
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        sampling.validate()
        prepared = self.prepare(
            records, sampling.thinking, sampling.allow_custom_prompts
        )
        if not prepared:
            raise ValueError("input is empty")
        self._check_context(prepared, sampling)
        prompts = [{"prompt_token_ids": record.prompt_token_ids} for record in prepared]
        seeds = [
            _stable_seed(sampling.run_seed, record.record_id, -1, "stage1")
            for record in prepared
        ]
        first_cap = (
            int(sampling.thinking_budget)
            if sampling.thinking == "budget"
            else sampling.max_tokens
        )
        params = [
            self._params(sampling, max_tokens=first_cap, seed=seed, n=sampling.n)
            for seed in seeds
        ]

        started = time.perf_counter()
        first_outputs = self.llm.generate(
            prompts,
            params,
            use_tqdm=False,
            lora_request=self.lora_request,
        )

        rows: list[dict[str, Any]] = []
        continuation_prompts: list[dict[str, list[int]]] = []
        continuation_params: list[Any] = []
        continuation_meta: list[tuple[int, int, list[int], list[int], bool, int, Any]] = []
        # (row index, output index, original stage1, retained thinking,
        #  forced_close, stage2 seed, stage1 completion)

        for row_index, (record, request_output, seed) in enumerate(
            zip(prepared, first_outputs, seeds)
        ):
            row = {
                "id": record.record_id,
                "meta": record.meta,
                "prompt_sha256": _sha256_bytes(record.prompt_text.encode("utf-8")),
                "n_prompt_tokens": len(record.prompt_token_ids),
                "prompt_channel": record.prompt_channel,
                "prompt_logprobs": _jsonable_logprobs(request_output.prompt_logprobs),
                "outputs": [None] * len(request_output.outputs),
            }
            rows.append(row)
            for completion in request_output.outputs:
                sample_index = int(completion.index)
                if sampling.thinking != "budget":
                    row["outputs"][sample_index] = self._ordinary_output(
                        record, completion, sample_index, seed, sampling.thinking
                    )
                    continue

                stage1_sampled = list(completion.token_ids)
                stage1 = self._trim_hf_eos(stage1_sampled)
                close_index = (
                    stage1.index(self.think_close_id)
                    if self.think_close_id in stage1
                    else None
                )
                naturally_finished = close_index is not None and completion.finish_reason == "stop"
                if naturally_finished:
                    row["outputs"][sample_index] = self._ordinary_output(
                        record, completion, sample_index, seed, "budget"
                    )
                    continue

                retained = stage1[:close_index] if close_index is not None else list(stage1)
                forced_close = close_index is None
                stage2_seed = _stable_seed(
                    sampling.run_seed, record.record_id, sample_index, "stage2"
                )
                if sampling.shuffle_thinking and retained:
                    random.Random(
                        _stable_seed(
                            sampling.run_seed, record.record_id, sample_index, "shuffle"
                        )
                    ).shuffle(retained)
                continuation_prompts.append(
                    {"prompt_token_ids": record.prompt_token_ids + retained + self.close_ids}
                )
                continuation_params.append(
                    self._params(
                        sampling,
                        max_tokens=sampling.answer_max_tokens,
                        seed=stage2_seed,
                        n=1,
                    )
                )
                continuation_meta.append(
                    (
                        row_index,
                        sample_index,
                        stage1_sampled,
                        retained,
                        forced_close,
                        stage2_seed,
                        completion,
                    )
                )

        if continuation_prompts:
            second_outputs = self.llm.generate(
                continuation_prompts,
                continuation_params,
                use_tqdm=False,
                lora_request=self.lora_request,
            )
            for meta, request_output in zip(continuation_meta, second_outputs):
                (
                    row_index,
                    sample_index,
                    stage1_sampled,
                    retained,
                    forced_close,
                    stage2_seed,
                    first_completion,
                ) = meta
                completion = request_output.outputs[0]
                stage2_sampled = list(completion.token_ids)
                stage2 = self._trim_hf_eos(stage2_sampled)
                final_ids = retained + self.close_ids + stage2
                stage2_prompt_tokens = (
                    len(prepared[row_index].prompt_token_ids)
                    + len(retained)
                    + len(self.close_ids)
                )
                stage1_cumulative = first_completion.cumulative_logprob
                stage2_cumulative = completion.cumulative_logprob
                sampled_cumulative = (
                    stage1_cumulative + stage2_cumulative
                    if stage1_cumulative is not None and stage2_cumulative is not None
                    else None
                )
                rows[row_index]["outputs"][sample_index] = {
                    "sample_index": sample_index,
                    "stage1_parent_seed": seeds[row_index],
                    "seed_stage1": seeds[row_index] + sample_index,
                    "seed_stage2": stage2_seed,
                    "text": self._decode(final_ids),
                    "token_ids": final_ids,
                    "stage1_token_ids": stage1_sampled,
                    "retained_thinking_token_ids": retained,
                    "injected_token_ids": list(self.close_ids),
                    "stage2_token_ids": stage2_sampled,
                    "n_thinking_tokens": len(retained),
                    "n_answer_tokens": len(stage2),
                    "n_sampled_tokens": len(stage1_sampled) + len(stage2_sampled),
                    "n_injected_tokens": len(self.close_ids),
                    "n_completion_tokens": len(final_ids),
                    "n_terminal_tokens_trimmed": (
                        len(stage1_sampled) - len(self._trim_hf_eos(stage1_sampled))
                    )
                    + (len(stage2_sampled) - len(stage2)),
                    "n_stage1_prompt_tokens": len(
                        prepared[row_index].prompt_token_ids
                    ),
                    "n_stage2_prompt_tokens": stage2_prompt_tokens,
                    "thinking_closed": True,
                    "forced_close": forced_close,
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                    "stage1_finish_reason": first_completion.finish_reason,
                    "stage1_stop_reason": first_completion.stop_reason,
                    "truncated": completion.finish_reason == "length",
                    "stage1_cumulative_logprob": stage1_cumulative,
                    "stage2_cumulative_logprob": stage2_cumulative,
                    "sampled_cumulative_logprob": sampled_cumulative,
                    "stage1_logprobs": _jsonable_logprobs(first_completion.logprobs),
                    "stage2_logprobs": _jsonable_logprobs(completion.logprobs),
                }

        generation_seconds = time.perf_counter() - started
        for row in rows:
            if any(output is None for output in row["outputs"]):
                raise RuntimeError(f"internal error: missing output for {row['id']!r}")

        unique_input_prompt = sum(row["n_prompt_tokens"] for row in rows)
        stage1_logical_prompt = sum(
            output["n_stage1_prompt_tokens"]
            for row in rows
            for output in row["outputs"]
        )
        stage2_logical_prompt = sum(
            output["n_stage2_prompt_tokens"]
            for row in rows
            for output in row["outputs"]
        )
        total_sampled = sum(
            output["n_sampled_tokens"] for row in rows for output in row["outputs"]
        )
        total_injected = sum(
            output["n_injected_tokens"] for row in rows for output in row["outputs"]
        )
        summary = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "engine": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in dataclasses.asdict(self.config).items()
            },
            "engine_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in self.engine_args.items()
            },
            "sampling": dataclasses.asdict(sampling),
            "resolved_sampling": sampling.resolved_sampling(),
            "adapter": self.adapter_info,
            "think_token_ids": {
                "open": self.think_open_id,
                "close": self.think_close_id,
                "forced_close_sequence": self.close_ids,
                "thinking_prompt_suffix": self.thinking_prompt_suffix_ids,
                "no_thinking_prompt_suffix": self.no_thinking_prompt_suffix_ids,
            },
            "termination": {
                "hf_model_eos_token_id": self.hf_eos_id,
                "vllm_tokenizer_eos_ignored": self.tokenizer.eos_token_id,
            },
            "rng_isolation": {
                "engine_seed": self.engine_args["seed"],
                "caller_global_rng_state_restored": True,
            },
            "counts": {
                "requests": len(rows),
                "completions": sum(len(row["outputs"]) for row in rows),
                "unique_input_prompt_tokens": unique_input_prompt,
                "stage1_logical_prompt_tokens": stage1_logical_prompt,
                "stage2_logical_prompt_tokens": stage2_logical_prompt,
                "logical_model_input_tokens": (
                    stage1_logical_prompt + stage2_logical_prompt
                ),
                "sampled_tokens": total_sampled,
                "injected_tokens": total_injected,
            },
            "timing": {
                "model_load_seconds": self.load_seconds,
                "generation_seconds": generation_seconds,
                "sampled_tokens_per_second": total_sampled / generation_seconds,
            },
            "runtime": self.runtime_metadata(),
        }
        return rows, summary

    @staticmethod
    def runtime_metadata() -> dict[str, Any]:
        git_root = _run_text(["git", "rev-parse", "--show-toplevel"])
        git_commit = _run_text(["git", "rev-parse", "HEAD"]) if git_root else ""
        git_status = _run_text(["git", "status", "--short"]) if git_root else ""
        gpu = _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        return {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": dict(_INITIAL_PACKAGES),
            "environment_lock": _environment_lock_metadata(),
            "uv": _run_text(["uv", "--version"]),
            "cuda_toolkit": _run_text(["nvcc", "--version"]),
            "gpu": gpu,
            "vllm_enable_v1_multiprocessing": os.environ.get(
                "VLLM_ENABLE_V1_MULTIPROCESSING"
            ),
            "git_commit": git_commit,
            "git_dirty": bool(git_status),
        }


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    data = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
    return records, _sha256_bytes(data)


def _smoke_records(count: int) -> list[dict[str, Any]]:
    tasks = [
        "Reply with exactly VLLM_OK.",
        "Return only the integer equal to 17 + 25.",
        "Write the reverse of the string abcdef, with no explanation.",
        "Return only a JSON array containing the first five positive odd integers.",
    ]
    return [
        {
            "id": f"smoke-{index:04d}",
            "messages": [{"role": "user", "content": tasks[index % len(tasks)]}],
            "meta": {"task_index": index % len(tasks)},
        }
        for index in range(count)
    ]


def _write_json_atomic(path: Path, value: Any, *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        if jsonl:
            for row in value:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        else:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="input JSONL")
    source.add_argument("--smoke", type=int, metavar="N", help="use N built-in smoke prompts")
    parser.add_argument("--output", type=Path, required=True, help="output JSONL")
    parser.add_argument("--metadata", type=Path, help="metadata JSON; default: OUTPUT.meta.json")
    parser.add_argument("--thinking", choices=["off", "natural", "budget"], default="off")
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-thinking", action="store_true")
    parser.add_argument("--logprobs", type=int)
    parser.add_argument("--prompt-logprobs", type=int)
    parser.add_argument("--logprob-token-id", type=int, action="append", default=[])
    parser.add_argument(
        "--allow-custom-prompts",
        action="store_true",
        help="allow raw prompts whose think-channel suffix does not match --thinking",
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-model-len", type=int, default=16_384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32_768)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--include-prompt-token-ids",
        action="store_true",
        help="include exact prompt IDs in output rows (metadata always records counts/hashes)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    metadata_path = args.metadata or args.output.with_name(args.output.name + ".meta.json")
    if args.output.resolve() == metadata_path.resolve():
        raise ValueError("--metadata must not be the same path as --output")
    if args.input is not None and args.input.resolve() in {
        args.output.resolve(),
        metadata_path.resolve(),
    }:
        raise ValueError("--input must not be overwritten by --output or --metadata")
    if args.input is not None:
        records, input_sha256 = _read_jsonl(args.input)
        input_description = str(args.input.resolve())
    else:
        if args.smoke < 1:
            raise ValueError("--smoke N requires N >= 1")
        records = _smoke_records(args.smoke)
        encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records).encode()
        input_sha256 = _sha256_bytes(encoded)
        input_description = f"built-in-smoke:{args.smoke}"

    engine = EngineConfig(
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        adapter=args.adapter,
    )
    sampling = SamplingConfig(
        thinking=args.thinking,
        thinking_budget=args.thinking_budget,
        n=args.n,
        max_tokens=args.max_tokens,
        answer_max_tokens=args.answer_max_tokens,
        greedy=args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        run_seed=args.seed,
        shuffle_thinking=args.shuffle_thinking,
        logprobs=args.logprobs,
        prompt_logprobs=args.prompt_logprobs,
        logprob_token_ids=tuple(args.logprob_token_id),
        allow_custom_prompts=args.allow_custom_prompts,
    )

    sampling.validate()  # fail before an expensive model load
    runner: VLLMRunner | None = None
    try:
        runner = VLLMRunner(engine)
        rows, summary = runner.generate(records, sampling)
        prepared = runner.prepare(
            records, sampling.thinking, sampling.allow_custom_prompts
        )
        if args.include_prompt_token_ids:
            for row, record in zip(rows, prepared):
                row["prompt_token_ids"] = record.prompt_token_ids
        summary["input"] = {
            "description": input_description,
            "sha256": input_sha256,
        }
        _write_json_atomic(args.output, rows, jsonl=True)
        _write_json_atomic(metadata_path, summary)
        timing = summary["timing"]
        counts = summary["counts"]
        print(
            f"wrote {counts['completions']} completions / {counts['sampled_tokens']} sampled tokens "
            f"at {timing['sampled_tokens_per_second']:.1f} tok/s to {args.output}",
            file=sys.stderr,
        )
    finally:
        if runner is not None:
            runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
