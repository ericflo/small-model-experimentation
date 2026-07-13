"""Sealed-confirmation runtime, cache, and sampled-token protocol.

This module is deliberately confirmation-only.  It wraps the frozen shared
runner without changing its behavior for acquisition or training jobs.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from io_utils import canonical_hash, sha256_file


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
RUNNER = EXP / "src" / "vllm_runner.py"
PINNED_PYTHON = (REPO / ".venv-vllm" / "bin" / "python").resolve()
PINNED_LOCK = (REPO / "requirements-vllm.lock.txt").resolve()
PINNED_LOCK_SHA256 = "64c3dde1e24f2b18a83cda3a3d84daf8701aad728e864c508854b10e3e3734c6"
PINNED_PYTHON_VERSION = "3.12.3"
PINNED_PLATFORM = "Linux-5.15.0-122-generic-x86_64-with-glibc2.39"
PINNED_GPU = "NVIDIA L40, 550.127.08, 46068"
PINNED_CUDA_TOOLKIT = (
    "nvcc: NVIDIA (R) Cuda compiler driver\n"
    "Copyright (c) 2005-2025 NVIDIA Corporation\n"
    "Built on Fri_Feb_21_20:23:50_PST_2025\n"
    "Cuda compilation tools, release 12.8, V12.8.93\n"
    "Build cuda_12.8.r12.8/compiler.35583870_0"
)
PINNED_CORE_PACKAGES = {
    "torch": "2.11.0+cu129",
    "transformers": "5.13.0",
    "vllm": "0.24.0+cu129",
}
PINNED_FORCED_CLOSE = [248069, 271]
PINNED_THINKING_SUFFIX = [248045, 74455, 198, 248068, 198]
PINNED_NO_THINKING_SUFFIX = [248045, 74455, 198, 248068, 271, 248069, 271]

# Qwen3.5-4B has 8 full-attention and 24 linear-attention layers.  With prefix
# caching disabled and mamba_cache_mode=none, pinned vLLM gives each linear
# group one full-context (16384-token) Mamba block.  The 24:8 layer ratio forms
# three Mamba groups alongside one 528-token attention-block stream.
ATTENTION_BLOCK_TOKENS = 528
MAMBA_BLOCK_TOKENS = 16384
FULL_ATTENTION_LAYERS = 8
LINEAR_ATTENTION_LAYERS = 24
MAMBA_GROUPS = 3
SAMPLING_FIELDS = {
    "thinking",
    "thinking_budget",
    "n",
    "max_tokens",
    "answer_max_tokens",
    "greedy",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "run_seed",
    "shuffle_thinking",
    "logprobs",
    "prompt_logprobs",
    "logprob_token_ids",
    "allow_custom_prompts",
}
RESOLVED_SAMPLING_FIELDS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
}


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"confirmation {label} is invalid")
    return value


def _plain_float(value: Any, label: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"confirmation {label} is invalid")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"confirmation {label} is missing or malformed")
    return value


def _strict_json_equal(left: Any, right: Any) -> bool:
    """Compare persisted protocol values without bool/int or int/float coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _strict_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _pinned_cache_formula() -> dict[str, Any]:
    return {
        "attention_block_tokens": ATTENTION_BLOCK_TOKENS,
        "mamba_block_tokens": MAMBA_BLOCK_TOKENS,
        "full_attention_layers": FULL_ATTENTION_LAYERS,
        "linear_attention_layers": LINEAR_ATTENTION_LAYERS,
        "mamba_groups": MAMBA_GROUPS,
        "enable_prefix_caching": False,
        "mamba_cache_mode": "none",
    }


def hybrid_blocks(
    reservation_tokens: int,
    *,
    max_model_len: int | None = None,
    formula: Mapping[str, Any] | None = None,
) -> int:
    """Return the pinned hybrid-cache blocks required by one live sequence."""

    reservation_tokens = _plain_int(
        reservation_tokens, "hybrid reservation", minimum=1
    )
    if max_model_len is not None and reservation_tokens > _plain_int(
        max_model_len, "maximum model length", minimum=1
    ):
        raise ValueError("confirmation hybrid reservation exceeds max_model_len")
    geometry = _pinned_cache_formula() if formula is None else _mapping(
        formula, "hybrid cache formula"
    )
    if not _strict_json_equal(dict(geometry), _pinned_cache_formula()):
        raise ValueError("confirmation hybrid cache formula drifted")
    return math.ceil(
        reservation_tokens / int(geometry["attention_block_tokens"])
    ) + int(geometry["mamba_groups"]) * math.ceil(
        reservation_tokens / int(geometry["mamba_block_tokens"])
    )


def live_cache_capacity(runner: Any) -> dict[str, Any]:
    """Authenticate vLLM's post-load hybrid-cache counters."""

    try:
        config = runner.config
        vllm_config = runner.llm.llm_engine.vllm_config
        cache = vllm_config.cache_config
        layer_types = list(vllm_config.model_config.hf_text_config.layer_types)
        maximum = _plain_int(config.max_model_len, "maximum model length", minimum=1)
        max_num_seqs = _plain_int(config.max_num_seqs, "maximum sequence count", minimum=1)
        num_gpu_blocks = _plain_int(cache.num_gpu_blocks, "GPU block count", minimum=1)
        block_size = _plain_int(cache.block_size, "cache block size", minimum=1)
        kv_tokens = _plain_int(
            cache.kv_cache_size_tokens, "KV cache token count", minimum=1
        )
        concurrency = _plain_float(
            cache.kv_cache_max_concurrency, "KV cache maximum concurrency"
        )
        mamba_block_size = _plain_int(
            cache.mamba_block_size, "Mamba block size", minimum=1
        )
        mamba_cache_mode = cache.mamba_cache_mode
        prefix_caching = cache.enable_prefix_caching
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("confirmation runner did not expose live cache capacity") from exc
    full_layers = layer_types.count("full_attention")
    linear_layers = layer_types.count("linear_attention")
    if (
        len(layer_types) != full_layers + linear_layers
        or full_layers < 1
        or linear_layers < 1
    ):
        raise ValueError("confirmation live hybrid layer geometry is malformed")
    formula = {
        "attention_block_tokens": block_size,
        "mamba_block_tokens": mamba_block_size,
        "full_attention_layers": full_layers,
        "linear_attention_layers": linear_layers,
        "mamba_groups": math.ceil(linear_layers / full_layers),
        "enable_prefix_caching": prefix_caching,
        "mamba_cache_mode": mamba_cache_mode,
    }
    if not _strict_json_equal(formula, _pinned_cache_formula()):
        raise ValueError("confirmation live hybrid cache formula drifted")
    full_blocks = hybrid_blocks(
        maximum, max_model_len=maximum, formula=formula
    )
    expected_concurrency = num_gpu_blocks / full_blocks
    checks = {
        "aligned_attention_block_is_528": block_size == ATTENTION_BLOCK_TOKENS,
        "mamba_block_is_full_context": mamba_block_size == maximum,
        "hybrid_layer_mix_is_8_full_24_linear": (
            full_layers == FULL_ATTENTION_LAYERS
            and linear_layers == LINEAR_ATTENTION_LAYERS
        ),
        "mamba_group_count_is_3": formula["mamba_groups"] == MAMBA_GROUPS,
        "prefix_cache_off_mamba_mode_none": (
            prefix_caching is False and mamba_cache_mode == "none"
        ),
        "max_concurrency_matches_hybrid_blocks": math.isclose(
            concurrency, expected_concurrency, rel_tol=0.0, abs_tol=1e-12
        ),
        "kv_tokens_match_floored_concurrency": kv_tokens
        == int(concurrency * maximum),
        "full_configured_concurrency_fits": num_gpu_blocks
        >= max_num_seqs * full_blocks,
    }
    if not all(checks.values()):
        raise ValueError(f"confirmation live hybrid-cache authentication failed: {checks}")
    return {
        "schema_version": 1,
        "formula": formula,
        "max_model_len": maximum,
        "max_num_seqs": max_num_seqs,
        "num_gpu_blocks": num_gpu_blocks,
        "block_size": block_size,
        "kv_cache_size_tokens": kv_tokens,
        "kv_cache_max_concurrency": concurrency,
        "blocks_per_full_request": full_blocks,
        "checks": checks,
    }


def validate_live_cache_capacity(receipt: Any) -> None:
    """Recompute a persisted post-load cache-capacity preflight receipt."""

    row = _mapping(receipt, "live capacity preflight")
    maximum = _plain_int(row.get("max_model_len"), "maximum model length", minimum=1)
    max_num_seqs = _plain_int(row.get("max_num_seqs"), "maximum sequence count", minimum=1)
    num_gpu_blocks = _plain_int(row.get("num_gpu_blocks"), "GPU block count", minimum=1)
    block_size = _plain_int(row.get("block_size"), "cache block size", minimum=1)
    kv_tokens = _plain_int(row.get("kv_cache_size_tokens"), "KV cache tokens", minimum=1)
    concurrency = _plain_float(
        row.get("kv_cache_max_concurrency"), "KV cache maximum concurrency"
    )
    formula = _mapping(row.get("formula"), "live capacity formula")
    if not _strict_json_equal(dict(formula), _pinned_cache_formula()):
        raise ValueError("confirmation live capacity formula drifted")
    full_blocks = hybrid_blocks(
        maximum, max_model_len=maximum, formula=formula
    )
    checks = {
        "aligned_attention_block_is_528": block_size == ATTENTION_BLOCK_TOKENS,
        "mamba_block_is_full_context": (
            formula["mamba_block_tokens"] == maximum
        ),
        "hybrid_layer_mix_is_8_full_24_linear": (
            formula["full_attention_layers"] == FULL_ATTENTION_LAYERS
            and formula["linear_attention_layers"] == LINEAR_ATTENTION_LAYERS
        ),
        "mamba_group_count_is_3": formula["mamba_groups"] == MAMBA_GROUPS,
        "prefix_cache_off_mamba_mode_none": (
            formula["enable_prefix_caching"] is False
            and formula["mamba_cache_mode"] == "none"
        ),
        "max_concurrency_matches_hybrid_blocks": math.isclose(
            concurrency, num_gpu_blocks / full_blocks, rel_tol=0.0, abs_tol=1e-12
        ),
        "kv_tokens_match_floored_concurrency": kv_tokens
        == int(concurrency * maximum),
        "full_configured_concurrency_fits": num_gpu_blocks
        >= max_num_seqs * full_blocks,
    }
    expected = {
        "schema_version": 1,
        "formula": dict(formula),
        "max_model_len": maximum,
        "max_num_seqs": max_num_seqs,
        "num_gpu_blocks": num_gpu_blocks,
        "block_size": block_size,
        "kv_cache_size_tokens": kv_tokens,
        "kv_cache_max_concurrency": concurrency,
        "blocks_per_full_request": full_blocks,
        "checks": checks,
    }
    if not _strict_json_equal(dict(row), expected) or not all(checks.values()):
        raise ValueError("confirmation live capacity preflight is stale or failed")


def capacity_receipt(
    static: Mapping[str, Any],
    *,
    prompt_token_lengths: Sequence[int],
    sampling: Any,
) -> dict[str, Any]:
    """Prove that one generate call cannot preempt for cache capacity."""

    if not prompt_token_lengths:
        raise ValueError("confirmation capacity receipt requires a non-empty call")
    lengths = [
        _plain_int(value, "prompt token length", minimum=1)
        for value in prompt_token_lengths
    ]
    n = _plain_int(sampling.n, "sampling multiplicity", minimum=1)
    if sampling.thinking == "budget":
        reserve = (
            _plain_int(sampling.thinking_budget, "thinking budget", minimum=1)
            + _plain_int(sampling.answer_max_tokens, "answer cap", minimum=1)
            + _plain_int(static.get("forced_close_tokens"), "forced-close length", minimum=1)
        )
    else:
        reserve = _plain_int(sampling.max_tokens, "generation cap", minimum=1)
    maximum = _plain_int(static.get("max_model_len"), "maximum model length", minimum=1)
    max_num_seqs = _plain_int(static.get("max_num_seqs"), "maximum sequence count", minimum=1)
    reservation = max(lengths) + reserve
    formula = _mapping(static.get("formula"), "capacity formula")
    if not _strict_json_equal(dict(formula), _pinned_cache_formula()):
        raise ValueError("confirmation capacity formula drifted")
    blocks_each = hybrid_blocks(
        reservation, max_model_len=maximum, formula=formula
    )
    logical = len(lengths) * n
    active = min(logical, max_num_seqs)
    rounded = math.ceil(reservation / ATTENTION_BLOCK_TOKENS) * ATTENTION_BLOCK_TOKENS
    required_tokens = active * rounded
    required_blocks = active * blocks_each
    available_tokens = _plain_int(
        static.get("kv_cache_size_tokens"), "KV cache token count", minimum=1
    )
    available_blocks = _plain_int(
        static.get("num_gpu_blocks"), "GPU block count", minimum=1
    )
    checks = {
        "context_fits": reservation <= maximum,
        "token_capacity_fits": required_tokens <= available_tokens,
        "hybrid_block_capacity_fits": required_blocks <= available_blocks,
        "capacity_bound_proves_no_preemption": (
            required_tokens <= available_tokens and required_blocks <= available_blocks
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"confirmation no-preemption capacity failed: {checks}")
    return {
        "schema_version": 1,
        "formula": dict(formula),
        "request_count": len(lengths),
        "sampling_n": n,
        "logical_sequences": logical,
        "active_sequences": active,
        "max_num_seqs": max_num_seqs,
        "max_model_len": maximum,
        "maximum_prompt_tokens": max(lengths),
        "generation_reserve_tokens": reserve,
        "reservation_tokens": reservation,
        "rounded_reservation_tokens": rounded,
        "blocks_per_request": blocks_each,
        "required_tokens": required_tokens,
        "available_tokens": available_tokens,
        "remaining_tokens": available_tokens - required_tokens,
        "required_blocks": required_blocks,
        "available_blocks": available_blocks,
        "remaining_blocks": available_blocks - required_blocks,
        "preemption_events_permitted": 0,
        "checks": checks,
    }


def validate_capacity_receipt(receipt: Any) -> None:
    """Recompute every persisted dynamic capacity field."""

    row = _mapping(receipt, "capacity receipt")
    formula = _mapping(row.get("formula"), "capacity formula")
    if not _strict_json_equal(dict(formula), _pinned_cache_formula()):
        raise ValueError("confirmation capacity formula drifted")
    request_count = _plain_int(row.get("request_count"), "request count", minimum=1)
    n = _plain_int(row.get("sampling_n"), "sampling multiplicity", minimum=1)
    logical = request_count * n
    max_num_seqs = _plain_int(row.get("max_num_seqs"), "maximum sequence count", minimum=1)
    maximum = _plain_int(row.get("max_model_len"), "maximum model length", minimum=1)
    active = min(logical, max_num_seqs)
    if row.get("logical_sequences") != logical or row.get("active_sequences") != active:
        raise ValueError("confirmation capacity sequence accounting drifted")
    prompt = _plain_int(row.get("maximum_prompt_tokens"), "maximum prompt", minimum=1)
    reserve = _plain_int(row.get("generation_reserve_tokens"), "generation reserve", minimum=1)
    reservation = prompt + reserve
    rounded = math.ceil(reservation / ATTENTION_BLOCK_TOKENS) * ATTENTION_BLOCK_TOKENS
    blocks_each = hybrid_blocks(
        reservation, max_model_len=maximum, formula=formula
    )
    available_tokens = _plain_int(row.get("available_tokens"), "available tokens", minimum=1)
    available_blocks = _plain_int(row.get("available_blocks"), "available blocks", minimum=1)
    expected = {
        "schema_version": 1,
        "formula": dict(formula),
        "request_count": request_count,
        "sampling_n": n,
        "logical_sequences": logical,
        "active_sequences": active,
        "max_num_seqs": max_num_seqs,
        "max_model_len": maximum,
        "maximum_prompt_tokens": prompt,
        "generation_reserve_tokens": reserve,
        "reservation_tokens": reservation,
        "rounded_reservation_tokens": rounded,
        "blocks_per_request": blocks_each,
        "required_tokens": active * rounded,
        "available_tokens": available_tokens,
        "remaining_tokens": available_tokens - active * rounded,
        "required_blocks": active * blocks_each,
        "available_blocks": available_blocks,
        "remaining_blocks": available_blocks - active * blocks_each,
        "preemption_events_permitted": 0,
        "checks": {
            "context_fits": True,
            "token_capacity_fits": active * rounded <= available_tokens,
            "hybrid_block_capacity_fits": active * blocks_each <= available_blocks,
            "capacity_bound_proves_no_preemption": (
                active * rounded <= available_tokens
                and active * blocks_each <= available_blocks
            ),
        },
    }
    if not _strict_json_equal(dict(row), expected) or not all(
        expected["checks"].values()
    ):
        raise ValueError("confirmation capacity receipt is stale or failed")


def expected_confirmation_sampling_protocol(
    config: Mapping[str, Any], *, decode: str, block_seed: int
) -> dict[str, Any]:
    """Return the exact registered atom/episode sampling settings for one arm."""

    if decode not in {"greedy", "sample8"}:
        raise ValueError("confirmation decode is not registered")
    seed = _plain_int(block_seed, "block seed")
    greedy = decode == "greedy"
    k = 1 if greedy else _plain_int(
        config["controls"]["sample_more_k"], "sample-more multiplicity", minimum=1
    )
    thinking_budget = _plain_int(
        config["generation"]["thinking_budget"], "thinking budget", minimum=1
    )
    answer_max_tokens = _plain_int(
        config["generation"]["answer_max_tokens"], "answer cap", minimum=1
    )
    temperature = (
        None
        if greedy
        else float(config["confirmation"]["sample_more_temperature"])
    )
    top_p = (
        None if greedy else float(config["confirmation"]["sample_more_top_p"])
    )
    top_k = (
        None
        if greedy
        else _plain_int(
            config["confirmation"]["sample_more_top_k"],
            "sample-more top-k",
            minimum=1,
        )
    )

    def settings(n: int) -> dict[str, Any]:
        return {
            "thinking": "budget",
            "thinking_budget": thinking_budget,
            "n": n,
            # SamplingConfig defaults that are inactive under budget thinking are
            # still pinned because they are part of the executed runner request.
            "max_tokens": 512,
            "answer_max_tokens": answer_max_tokens,
            "greedy": greedy,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
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

    resolved = {
        "temperature": (
            0.0
            if greedy
            else float(config["confirmation"]["sample_more_temperature"])
        ),
        "top_p": (
            1.0 if greedy else float(config["confirmation"]["sample_more_top_p"])
        ),
        "top_k": (
            0
            if greedy
            else _plain_int(
                config["confirmation"]["sample_more_top_k"],
                "sample-more top-k",
                minimum=1,
            )
        ),
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    return {
        "schema_version": 1,
        "atom": {"sampling": settings(k), "resolved_sampling": dict(resolved)},
        "episode": {
            "sampling": settings(1),
            "resolved_sampling": dict(resolved),
        },
    }


def _summary_sampling_pair(summary: Any) -> dict[str, Any]:
    row = _mapping(summary, "runner summary")
    sampling = _mapping(row.get("sampling"), "runner sampling")
    resolved = _mapping(row.get("resolved_sampling"), "resolved runner sampling")
    if set(sampling) != SAMPLING_FIELDS or set(resolved) != RESOLVED_SAMPLING_FIELDS:
        raise ValueError("confirmation runner sampling inventory is malformed")
    normalized_sampling = dict(sampling)
    token_ids = normalized_sampling.get("logprob_token_ids")
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if not isinstance(token_ids, list):
        raise ValueError("confirmation sampling logprob-token inventory is malformed")
    normalized_sampling["logprob_token_ids"] = token_ids
    return {
        "sampling": normalized_sampling,
        "resolved_sampling": dict(resolved),
    }


def canonical_confirmation_sampling_protocol(
    summaries: Any,
) -> dict[str, Any]:
    """Derive one exact atom protocol and one exact episode protocol."""

    if not isinstance(summaries, list) or len(summaries) < 2:
        raise ValueError(
            "confirmation sampling evidence must include atom and episode calls"
        )
    atom = _summary_sampling_pair(summaries[0])
    episode_rows = [_summary_sampling_pair(summary) for summary in summaries[1:]]
    if any(not _strict_json_equal(row, episode_rows[0]) for row in episode_rows[1:]):
        raise ValueError("confirmation episode calls used different sampling settings")
    return {
        "schema_version": 1,
        "atom": atom,
        "episode": episode_rows[0],
    }


def validate_confirmation_sampling_protocol(
    summaries: Any, expected: Any
) -> dict[str, Any]:
    """Authenticate persisted runner settings against the registered arm settings."""

    observed = canonical_confirmation_sampling_protocol(summaries)
    if not isinstance(expected, dict) or not _strict_json_equal(observed, expected):
        raise ValueError("confirmation sampling settings differ from registration")
    return observed


def _canonical_one_summary(
    summary: Any, *, expected_model: str | None = None
) -> dict[str, Any]:
    summary = _mapping(summary, "runner summary")
    engine = _mapping(summary.get("engine"), "runner engine")
    args = _mapping(summary.get("engine_args"), "runner engine arguments")
    rng_isolation = _mapping(
        summary.get("rng_isolation"), "runner RNG-isolation inventory"
    )
    resolved = _mapping(summary.get("resolved_cudagraph"), "resolved CUDA graph")
    runtime = _mapping(summary.get("runtime"), "runner runtime")
    packages = _mapping(runtime.get("packages"), "runtime packages")
    lock = _mapping(runtime.get("environment_lock"), "runtime lock")
    think = _mapping(summary.get("think_token_ids"), "think-token inventory")
    termination = _mapping(summary.get("termination"), "termination inventory")

    model = summary.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("confirmation runner model identity is malformed")
    if expected_model is not None and model != expected_model:
        raise ValueError("confirmation runner model differs from admission")
    if (
        "model_revision" not in summary
        or summary["model_revision"] is not None
        or "adapter" not in summary
        or summary["adapter"] is not None
        or "model_override" not in engine
        or engine["model_override"] != model
        or "adapter" not in engine
        or engine["adapter"] is not None
        or "model" not in args
        or args["model"] != model
    ):
        raise ValueError("confirmation runner model identity drifted")
    expected_rng_isolation = {
        "engine_seed": 0,
        "caller_global_rng_state_restored": True,
    }
    if not _strict_json_equal(dict(rng_isolation), expected_rng_isolation):
        raise ValueError("confirmation runner RNG isolation drifted")

    normalized_packages: dict[str, str] = {}
    for name, version in packages.items():
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise ValueError("confirmation runtime package inventory is malformed")
        canonical_name = name.lower().replace("_", "-")
        if canonical_name in normalized_packages:
            raise ValueError("confirmation runtime package inventory has aliases")
        normalized_packages[canonical_name] = version
    normalized_packages = dict(sorted(normalized_packages.items()))
    if any(normalized_packages.get(name) != version for name, version in PINNED_CORE_PACKAGES.items()):
        raise ValueError("confirmation core package versions are not pinned")

    capture = list(engine.get("cudagraph_capture_sizes") or [])
    if not capture or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in capture
    ):
        raise ValueError("confirmation CUDA-graph capture list is malformed")
    invariant_args = {
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
    }
    if any(
        not _strict_json_equal(args.get(key), value)
        for key, value in invariant_args.items()
    ):
        raise ValueError("confirmation invariant vLLM arguments drifted")
    geometry = {
        "max_model_len": _plain_int(engine.get("max_model_len"), "maximum model length", minimum=1),
        "gpu_memory_utilization": _plain_float(
            engine.get("gpu_memory_utilization"), "GPU memory utilization"
        ),
        "max_num_seqs": _plain_int(engine.get("max_num_seqs"), "maximum sequence count", minimum=1),
        "max_num_batched_tokens": _plain_int(
            engine.get("max_num_batched_tokens"), "maximum batched tokens", minimum=1
        ),
        "cudagraph_capture_sizes": capture,
    }
    if engine.get("enable_prefix_caching") is not False or engine.get("enforce_eager") is not False:
        raise ValueError("confirmation runner engine flags drifted")
    if any(
        not _strict_json_equal(args.get(key), value)
        for key, value in geometry.items()
    ):
        raise ValueError("confirmation requested vLLM geometry drifted")
    if not _strict_json_equal(args.get("max_cudagraph_capture_size"), capture[-1]):
        raise ValueError("confirmation requested CUDA-graph endpoint drifted")
    if (
        not _strict_json_equal(resolved.get("cudagraph_capture_sizes"), capture)
        or not _strict_json_equal(
            resolved.get("max_cudagraph_capture_size"), capture[-1]
        )
        or not _strict_json_equal(resolved.get("mode"), "FULL_AND_PIECEWISE")
        or not _strict_json_equal(resolved.get("decode_mode"), "FULL")
        or not _strict_json_equal(resolved.get("mixed_mode"), "PIECEWISE")
        or resolved.get("has_full_cudagraphs") is not True
    ):
        raise ValueError("confirmation resolved CUDA-graph geometry drifted")
    if (
        not _strict_json_equal(summary.get("schema_version"), 4)
        or not _strict_json_equal(summary.get("runner_sha256"), sha256_file(RUNNER))
        or not _strict_json_equal(think.get("open"), 248068)
        or not _strict_json_equal(think.get("close"), 248069)
        or not _strict_json_equal(
            think.get("forced_close_sequence"), PINNED_FORCED_CLOSE
        )
        or not _strict_json_equal(
            think.get("thinking_prompt_suffix"), PINNED_THINKING_SUFFIX
        )
        or not _strict_json_equal(
            think.get("no_thinking_prompt_suffix"), PINNED_NO_THINKING_SUFFIX
        )
        or not _strict_json_equal(
            termination.get("hf_model_eos_token_id"), 248044
        )
        or not _strict_json_equal(
            termination.get("vllm_tokenizer_eos_ignored"), 248046
        )
    ):
        raise ValueError("confirmation runner/token protocol drifted")
    python_executable = runtime.get("python_executable")
    if (
        not _strict_json_equal(runtime.get("python"), PINNED_PYTHON_VERSION)
        or not isinstance(python_executable, str)
        or Path(python_executable).resolve() != PINNED_PYTHON
        or not _strict_json_equal(runtime.get("platform"), PINNED_PLATFORM)
        or not _strict_json_equal(lock.get("path"), str(PINNED_LOCK))
        or not _strict_json_equal(lock.get("sha256"), PINNED_LOCK_SHA256)
        or not PINNED_LOCK.is_file()
        or sha256_file(PINNED_LOCK) != PINNED_LOCK_SHA256
        or not _strict_json_equal(runtime.get("gpu"), PINNED_GPU)
        or not _strict_json_equal(runtime.get("cuda_toolkit"), PINNED_CUDA_TOOLKIT)
        or not _strict_json_equal(
            runtime.get("vllm_enable_v1_multiprocessing"), "0"
        )
        or not _strict_json_equal(runtime.get("uv"), "uv 0.9.0")
    ):
        raise ValueError("confirmation pinned runtime drifted")
    protocol = {
        "schema_version": 1,
        "runner_schema_version": 4,
        "runner_sha256": summary["runner_sha256"],
        "engine": geometry,
        "engine_args": invariant_args,
        "resolved_cudagraph": {
            "cudagraph_capture_sizes": capture,
            "max_cudagraph_capture_size": capture[-1],
            "mode": resolved.get("mode"),
            "decode_mode": resolved.get("decode_mode"),
            "mixed_mode": resolved.get("mixed_mode"),
            "has_full_cudagraphs": True,
        },
        "think_token_ids": {
            "open": 248068,
            "close": 248069,
            "forced_close_sequence": list(PINNED_FORCED_CLOSE),
            "thinking_prompt_suffix": list(PINNED_THINKING_SUFFIX),
            "no_thinking_prompt_suffix": list(PINNED_NO_THINKING_SUFFIX),
        },
        "termination": {
            "hf_model_eos_token_id": 248044,
            "vllm_tokenizer_eos_ignored": 248046,
        },
        "runtime": {
            "python": runtime["python"],
            "python_executable": str(PINNED_PYTHON),
            "platform": runtime["platform"],
            "packages": normalized_packages,
            "environment_lock": dict(lock),
            "uv": runtime.get("uv"),
            "cuda_toolkit": runtime["cuda_toolkit"],
            "gpu": runtime["gpu"],
            "vllm_enable_v1_multiprocessing": "0",
        },
    }
    return protocol


def canonical_backend_protocol(
    summaries: Any,
    *,
    expected_engine: Mapping[str, Any] | None = None,
    expected_model: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Return the one authenticated backend protocol used by every call."""

    if not isinstance(summaries, list) or not summaries:
        raise ValueError("confirmation runner summaries are missing")
    if expected_model is not None and (
        not isinstance(expected_model, str) or not expected_model
    ):
        raise ValueError("confirmation admitted model identity is malformed")
    protocols = [
        _canonical_one_summary(summary, expected_model=expected_model)
        for summary in summaries
    ]
    hashes = {canonical_hash(protocol) for protocol in protocols}
    if len(hashes) != 1:
        raise ValueError("confirmation generation calls used different backends")
    protocol = protocols[0]
    if expected_engine is not None:
        expected = {
            "max_model_len": _plain_int(
                expected_engine["max_model_len"], "registered maximum model length", minimum=1
            ),
            "gpu_memory_utilization": _plain_float(
                expected_engine["gpu_memory_utilization"],
                "registered GPU memory utilization",
            ),
            "max_num_seqs": _plain_int(
                expected_engine["max_num_seqs"], "registered maximum sequence count", minimum=1
            ),
            "max_num_batched_tokens": _plain_int(
                expected_engine["max_num_batched_tokens"],
                "registered maximum batched tokens",
                minimum=1,
            ),
            "cudagraph_capture_sizes": [
                _plain_int(value, "registered CUDA-graph capture size", minimum=1)
                for value in expected_engine["cudagraph_capture_sizes"]
            ],
        }
        if not _strict_json_equal(protocol["engine"], expected):
            raise ValueError("confirmation backend does not match frozen engine config")
    fingerprint = next(iter(hashes))
    return protocol, fingerprint


class ConfirmationRunner:
    """Confirmation-only proxy adding capacity proof and durable call journaling."""

    def __init__(
        self,
        runner: Any,
        *,
        journal: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        self._runner = runner
        self._journal = journal
        self.eval_summaries: list[dict[str, Any]] = []
        self._sample_evidence: dict[
            tuple[str, int],
            tuple[list[int], list[int], int, str, str, str, str],
        ] = {}
        static = live_cache_capacity(runner)
        static["forced_close_tokens"] = len(runner.close_ids)
        self.capacity = static

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runner, name)

    def close(self) -> None:
        self._runner.close()

    @staticmethod
    def _request_evidence(
        records: Sequence[dict[str, Any]], prepared: Sequence[Any]
    ) -> list[dict[str, Any]]:
        if len(records) != len(prepared) or not records:
            raise ValueError("confirmation prepared-request geometry is invalid")
        evidence = []
        seen = set()
        for record, prompt in zip(records, prepared):
            if not isinstance(record, Mapping):
                raise ValueError("confirmation input record is malformed")
            record_id = record.get("id")
            messages = record.get("messages")
            if (
                not isinstance(record_id, str)
                or not record_id
                or record_id in seen
                or not isinstance(messages, list)
                or prompt.record_id != record_id
                or not isinstance(prompt.prompt_text, str)
                or not isinstance(prompt.prompt_channel, str)
                or not prompt.prompt_channel
            ):
                raise ValueError("confirmation input/prepared request identity drifted")
            prompt_ids = list(prompt.prompt_token_ids)
            if not prompt_ids or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in prompt_ids
            ):
                raise ValueError("confirmation prepared prompt IDs are malformed")
            seen.add(record_id)
            evidence.append(
                {
                    "schema_version": 1,
                    "id": record_id,
                    "record_sha256": canonical_hash(
                        {"id": record_id, "messages": messages}
                    ),
                    "prompt_token_ids_sha256": canonical_hash(prompt_ids),
                    "prompt_sha256": hashlib.sha256(
                        prompt.prompt_text.encode("utf-8")
                    ).hexdigest(),
                    "n_prompt_tokens": len(prompt_ids),
                    "prompt_channel": prompt.prompt_channel,
                }
            )
        return evidence

    @staticmethod
    def _validate_returned_request_rows(
        rows: Any, evidence: Sequence[Mapping[str, Any]]
    ) -> None:
        if not isinstance(rows, list) or len(rows) != len(evidence):
            raise ValueError("confirmation runner returned wrong request geometry")
        if any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("confirmation runner returned a malformed request row")
        if [row.get("id") for row in rows] != [row["id"] for row in evidence]:
            raise ValueError("confirmation runner returned requests out of order")
        for row, expected in zip(rows, evidence):
            if (
                not isinstance(row, Mapping)
                or row.get("prompt_sha256") != expected["prompt_sha256"]
                or row.get("n_prompt_tokens") != expected["n_prompt_tokens"]
                or row.get("prompt_channel") != expected["prompt_channel"]
            ):
                raise ValueError(
                    "confirmation returned prompt differs from prepared request"
                )

    def generate(self, records: Sequence[dict[str, Any]], sampling: Any):
        prepared = self._runner.prepare(
            records, sampling.thinking, sampling.allow_custom_prompts
        )
        request_evidence = self._request_evidence(records, prepared)
        receipt = capacity_receipt(
            self.capacity,
            prompt_token_lengths=[len(row.prompt_token_ids) for row in prepared],
            sampling=sampling,
        )
        rows, summary = self._runner.generate(records, sampling)
        summary = {**summary, "confirmation_capacity": receipt}
        # The return is durable before any confirmation-side interpretation.
        self._journal(
            {
                "rows": rows,
                "summary": summary,
                "request_evidence": request_evidence,
            }
        )
        self._validate_returned_request_rows(rows, request_evidence)
        for row, pre_call in zip(rows, request_evidence):
            record_id = str(row["id"])
            request_sha256 = canonical_hash(
                {key: value for key, value in row.items() if key != "outputs"}
            )
            for output in row["outputs"]:
                first = list(output.get("stage1_token_ids") or [])
                second = list(output.get("stage2_token_ids") or [])
                count = _plain_int(
                    output.get("n_sampled_tokens"), "runner sampled-token count"
                )
                if count != len(first) + len(second):
                    raise ValueError("confirmation runner sampled-token evidence disagrees")
                key = (record_id, int(output["sample_index"]))
                if key in self._sample_evidence:
                    raise ValueError("confirmation runner repeated a generation identity")
                self._sample_evidence[key] = (
                    first,
                    second,
                    count,
                    request_sha256,
                    canonical_hash(output),
                    str(pre_call["record_sha256"]),
                    str(pre_call["prompt_token_ids_sha256"]),
                )
        return rows, summary

    def annotate_raw_rows(
        self, atom_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]]
    ) -> None:
        consumed: set[tuple[str, int]] = set()
        for row in atom_rows:
            for output in row["outputs"]:
                key = (str(row["id"]), int(output["sample_index"]))
                (
                    first,
                    second,
                    count,
                    request_sha256,
                    output_sha256,
                    record_sha256,
                    prompt_token_ids_sha256,
                ) = (
                    self._sample_evidence[key]
                )
                output["stage1_sampled_token_ids"] = first
                output["stage2_sampled_token_ids"] = second
                output["generation_request_sha256"] = request_sha256
                output["generation_output_sha256"] = output_sha256
                output["generation_record_sha256"] = record_sha256
                output["generation_prompt_token_ids_sha256"] = (
                    prompt_token_ids_sha256
                )
                if int(output["n_sampled_tokens"]) != count:
                    raise ValueError("confirmation atom slim token count drifted")
                consumed.add(key)
        for row in episode_rows:
            rid = str(row["rid"])
            for turn in row["turns"]:
                key = (f"{rid}-t{int(turn['turn'])}", 0)
                (
                    first,
                    second,
                    count,
                    request_sha256,
                    output_sha256,
                    record_sha256,
                    prompt_token_ids_sha256,
                ) = (
                    self._sample_evidence[key]
                )
                turn["stage1_sampled_token_ids"] = first
                turn["stage2_sampled_token_ids"] = second
                turn["generation_request_sha256"] = request_sha256
                turn["generation_output_sha256"] = output_sha256
                turn["generation_record_sha256"] = record_sha256
                turn["generation_prompt_token_ids_sha256"] = (
                    prompt_token_ids_sha256
                )
                if int(turn["n_sampled_tokens"]) != count:
                    raise ValueError("confirmation episode slim token count drifted")
                consumed.add(key)
        if consumed != set(self._sample_evidence):
            raise ValueError("confirmation sampled-token evidence was not consumed exactly")
