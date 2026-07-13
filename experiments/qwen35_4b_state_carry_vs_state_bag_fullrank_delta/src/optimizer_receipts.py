"""CPU-testable fail-closed receipts for full-rank-delta optimizer state."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_is_finite(tensor: torch.Tensor) -> bool:
    flattened = tensor.detach().reshape(-1)
    return all(
        bool(torch.isfinite(chunk).all().item())
        for chunk in flattened.split(8 * 1024 * 1024)
    )


def optimizer_state_receipt(
    optimizer: torch.optim.Optimizer,
    *,
    delta_parameters: Sequence[torch.nn.Parameter],
) -> dict[str, Any]:
    """Audit complete FP32 Adam moments for every direct delta tensor."""
    bytes_by_dtype: dict[str, int] = {}
    tensors = 0
    for state in optimizer.state.values():
        for value in state.values():
            if not torch.is_tensor(value):
                continue
            tensors += 1
            dtype = str(value.dtype)
            bytes_by_dtype[dtype] = bytes_by_dtype.get(dtype, 0) + value.numel() * value.element_size()
    delta_moment_bytes = 0
    audited = []
    for index, parameter in enumerate(delta_parameters):
        if parameter.dtype != torch.float32:
            raise RuntimeError(
                f"full-rank delta parameter {index} is not FP32: {parameter.dtype}"
            )
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping):
            raise RuntimeError(f"full-rank delta parameter {index} has no Adam state")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = state.get(moment_name)
            if not torch.is_tensor(moment):
                raise RuntimeError(
                    f"full-rank delta parameter {index} lacks Adam {moment_name}"
                )
            if moment.dtype != torch.float32 or moment.shape != parameter.shape:
                raise RuntimeError(
                    f"full-rank delta parameter {index} has invalid {moment_name} "
                    f"dtype/shape: {moment.dtype} {tuple(moment.shape)}"
                )
            if not _tensor_is_finite(moment):
                raise RuntimeError(
                    f"full-rank delta parameter {index} has nonfinite {moment_name}"
                )
            delta_moment_bytes += moment.numel() * moment.element_size()
        step = state.get("step")
        if torch.is_tensor(step):
            if step.numel() != 1 or not _tensor_is_finite(step):
                raise RuntimeError(
                    f"full-rank delta parameter {index} has invalid Adam step"
                )
        elif not isinstance(step, (int, float)) or not math.isfinite(float(step)):
            raise RuntimeError(
                f"full-rank delta parameter {index} has invalid Adam step"
            )
        audited.append(
            {
                "index": index,
                "shape": list(parameter.shape),
                "parameter_dtype": str(parameter.dtype),
                "moment_dtype": "torch.float32",
                "moment_tensors": 2,
            }
        )
    return {
        "tensors": tensors,
        "bytes_by_dtype": bytes_by_dtype,
        "total_bytes": sum(bytes_by_dtype.values()),
        "total_gib": sum(bytes_by_dtype.values()) / (1024**3),
        "delta_parameters_audited": len(audited),
        "delta_moment_tensors": len(audited) * 2,
        "delta_moment_bytes": delta_moment_bytes,
        "delta_states_complete": len(audited) == len(delta_parameters),
        "delta_state_manifest_sha256": _canonical_sha256({"parameters": audited}),
    }
