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
    allowed_missing_parameters: Sequence[torch.nn.Parameter] = (),
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
    moment_cache: dict[int, int] = {}
    allowed_missing_ids = {id(parameter) for parameter in allowed_missing_parameters}
    if allowed_missing_ids & {id(parameter) for parameter in delta_parameters}:
        raise RuntimeError("adaptation parameters cannot be exempt from optimizer state")
    grouped_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    grouped_ids = [id(parameter) for parameter in grouped_parameters]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("optimizer parameter groups contain duplicate tensors")
    if not allowed_missing_ids <= set(grouped_ids):
        raise RuntimeError("optimizer-state exemption is not a registered optimizer parameter")
    if any(parameter in optimizer.state for parameter in allowed_missing_parameters):
        raise RuntimeError("registered missing-state exemption unexpectedly has Adam state")

    def audit_parameter(parameter: torch.nn.Parameter, label: str) -> int:
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping):
            raise RuntimeError(f"{label} has no Adam state")
        moment_bytes = 0
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = state.get(moment_name)
            if not torch.is_tensor(moment):
                raise RuntimeError(f"{label} lacks Adam {moment_name}")
            if moment.dtype != torch.float32 or moment.shape != parameter.shape:
                raise RuntimeError(
                    f"{label} has invalid {moment_name} "
                    f"dtype/shape: {moment.dtype} {tuple(moment.shape)}"
                )
            if not _tensor_is_finite(moment):
                raise RuntimeError(f"{label} has nonfinite {moment_name}")
            moment_bytes += moment.numel() * moment.element_size()
        step = state.get("step")
        if torch.is_tensor(step):
            if step.numel() != 1 or not _tensor_is_finite(step):
                raise RuntimeError(f"{label} has invalid Adam step")
        elif not isinstance(step, (int, float)) or not math.isfinite(float(step)):
            raise RuntimeError(f"{label} has invalid Adam step")
        moment_cache[id(parameter)] = moment_bytes
        return moment_bytes

    delta_moment_bytes = 0
    audited = []
    for index, parameter in enumerate(delta_parameters):
        if parameter.dtype != torch.float32:
            raise RuntimeError(
                f"full-rank delta parameter {index} is not FP32: {parameter.dtype}"
            )
        delta_moment_bytes += audit_parameter(
            parameter, f"full-rank delta parameter {index}"
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
    group_receipts = []
    for group_index, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("group_name", f"group_{group_index}"))
        parameters = list(group["params"])
        group_moment_bytes = 0
        manifest = []
        missing_exempt = 0
        for parameter_index, parameter in enumerate(parameters):
            if parameter.dtype != torch.float32:
                raise RuntimeError(
                    f"optimizer group {group_name} parameter {parameter_index} is not FP32"
                )
            if parameter not in optimizer.state and id(parameter) in allowed_missing_ids:
                missing_exempt += 1
                manifest.append(
                    {
                        "index": parameter_index,
                        "shape": list(parameter.shape),
                        "dtype": str(parameter.dtype),
                        "moment_tensors": 0,
                        "registered_missing_state_exemption": True,
                    }
                )
                continue
            group_moment_bytes += moment_cache.get(id(parameter)) or audit_parameter(
                parameter, f"optimizer group {group_name} parameter {parameter_index}"
            )
            manifest.append(
                {
                    "index": parameter_index,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "moment_tensors": 2,
                }
            )
        group_receipts.append(
            {
                "group_name": group_name,
                "parameters": len(parameters),
                "moment_tensors": 2 * (len(parameters) - missing_exempt),
                "moment_bytes": group_moment_bytes,
                "registered_missing_state_exemptions": missing_exempt,
                "state_manifest_sha256": _canonical_sha256(
                    {"parameters": manifest}
                ),
                "required_states_complete_and_finite": bool(parameters),
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
        "groups": group_receipts,
        "all_required_group_states_complete_and_finite": all(
            group["required_states_complete_and_finite"] for group in group_receipts
        ),
        "registered_missing_state_exemptions": len(allowed_missing_ids),
    }
