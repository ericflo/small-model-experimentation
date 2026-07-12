"""Experiment-local factory for the pinned vLLM runner."""

from __future__ import annotations

from pathlib import Path

from vllm_runner import EngineConfig, VLLMRunner


def make_runner(
    engine_cfg: dict,
    *,
    adapter: str | None = None,
    model_override: str | None = None,
) -> VLLMRunner:
    return VLLMRunner(
        EngineConfig(
            max_model_len=int(engine_cfg.get("max_model_len", 16384)),
            gpu_memory_utilization=float(engine_cfg.get("gpu_memory_utilization", 0.85)),
            max_num_seqs=int(engine_cfg.get("max_num_seqs", 32)),
            max_num_batched_tokens=int(engine_cfg.get("max_num_batched_tokens", 16384)),
            adapter=Path(adapter) if adapter else None,
            model_override=Path(model_override) if model_override else None,
        )
    )
