"""Matched LoRA and full-rank extra-call adaptation backends.

Both capacities use the same hook implementation and the same native-dropout
call site.  This is deliberate: resetting the CUDA RNG before every
microbatch then gives the two parameterizations the same dropout masks for a
given row and recurrence geometry.  The frozen base weights are never owned by
this module and the hooks are inert outside ``enabled()``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def capacity_seed(model_seed: int, capacity: str) -> int:
    payload = f"capacity-init-v1|{int(model_seed)}|{capacity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def microbatch_dropout_seed(model_seed: int, microbatch_index: int, row_id: str, k: int) -> int:
    payload = (
        f"adaptation-dropout-v1|{int(model_seed)}|{int(microbatch_index)}|{row_id}|{int(k)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


class AdaptationBank(nn.Module):
    """Common hook controller for rank-32 and direct full-shape deltas."""

    def __init__(
        self,
        base_model: nn.Module,
        target_names: list[str],
        *,
        capacity: str,
        model_seed: int,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if capacity not in {"lora", "fullrank"}:
            raise ValueError(f"unknown capacity: {capacity}")
        modules = dict(base_model.named_modules())
        self.capacity = capacity
        capacity_config = config[capacity]
        self.dropout = float(capacity_config["dropout"])
        self.scale = float(capacity_config["scale"])
        self.target_names = tuple(target_names)
        self._enabled_depth = 0
        self._suspended_depth = 0
        self._active_call_count = 0
        self._handles: list[Any] = []
        self._key_to_target: dict[str, str] = {}
        self._last_call_manifest: list[dict[str, Any]] = []
        self._capture_masks = False
        self._mask_digest: hashlib._Hash | None = None

        # Parameterization-specific construction cannot perturb the global RNG
        # used by shared state modules or training dropout.
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(capacity_seed(model_seed, capacity))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(capacity_seed(model_seed, capacity))
            if capacity == "lora":
                self.down = nn.ModuleDict()
                self.up = nn.ModuleDict()
                rank = int(capacity_config["rank"])
            else:
                self.deltas = nn.ModuleDict()

            for index, name in enumerate(self.target_names):
                target = modules.get(name)
                if not isinstance(target, nn.Linear):
                    raise RuntimeError(f"adaptation target is not nn.Linear: {name}")
                key = f"d{index:03d}"
                if capacity == "lora":
                    down = nn.Linear(
                        target.in_features,
                        rank,
                        bias=False,
                        device=target.weight.device,
                        dtype=torch.float32,
                    )
                    up = nn.Linear(
                        rank,
                        target.out_features,
                        bias=False,
                        device=target.weight.device,
                        dtype=torch.float32,
                    )
                    nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
                    nn.init.zeros_(up.weight)
                    self.down[key] = down
                    self.up[key] = up
                else:
                    delta = nn.Linear(
                        target.in_features,
                        target.out_features,
                        bias=False,
                        device=target.weight.device,
                        dtype=torch.float32,
                    )
                    nn.init.zeros_(delta.weight)
                    self.deltas[key] = delta
                self._key_to_target[key] = name
                self._handles.append(target.register_forward_hook(self._make_hook(key)))

    def _drop(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout == 0.0:
            return value
        dropped, mask = torch.ops.aten.native_dropout.default(value, self.dropout, True)
        if self._capture_masks:
            if self._mask_digest is None:
                raise RuntimeError("mask capture has no active digest")
            self._mask_digest.update(
                mask.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            )
        return dropped

    def _make_hook(self, key: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            del module
            if self._enabled_depth == 0 or self._suspended_depth > 0:
                return None
            if len(inputs) != 1:
                raise RuntimeError("adaptation target received unexpected arguments")
            value = inputs[0]
            self._active_call_count += 1
            self._last_call_manifest.append(
                {
                    "target": self._key_to_target[key],
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )
            if self.capacity == "lora":
                # Pinned PEFT casts to the adapter dtype *before* dropout.  Do
                # the same here; full rank uses the identical FP32 dropout
                # input so realized masks and stochastic arithmetic match.
                adapted_input = value.to(dtype=self.down[key].weight.dtype)
                dropped = self._drop(adapted_input)
                update = self.up[key](self.down[key](dropped))
            else:
                adapted_input = value.to(dtype=self.deltas[key].weight.dtype)
                dropped = self._drop(adapted_input)
                update = self.deltas[key](dropped)
            return (output + self.scale * update).to(dtype=output.dtype)

        return hook

    @contextlib.contextmanager
    def enabled(self, enabled: bool):
        activate = enabled and self._suspended_depth == 0
        if activate:
            self._enabled_depth += 1
        try:
            yield
        finally:
            if activate:
                self._enabled_depth -= 1
            if self._enabled_depth < 0:
                raise RuntimeError("adaptation context underflow")

    @contextlib.contextmanager
    def suspended(self):
        self._suspended_depth += 1
        try:
            yield
        finally:
            self._suspended_depth -= 1
            if self._suspended_depth < 0:
                raise RuntimeError("adaptation suspension underflow")

    def begin_microbatch(self, seed: int, *, capture_masks: bool = False) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("matched adaptation dropout requires CUDA")
        torch.cuda.manual_seed_all(int(seed))
        self._active_call_count = 0
        self._last_call_manifest = []
        self._capture_masks = bool(capture_masks)
        self._mask_digest = hashlib.sha256() if capture_masks else None

    def end_microbatch(self) -> dict[str, Any]:
        manifest_digest = hashlib.sha256(
            json.dumps(
                self._last_call_manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        target_count = len(self.target_names)
        cycles = [
            self._last_call_manifest[index : index + target_count]
            for index in range(0, len(self._last_call_manifest), target_count)
        ] if target_count else []
        cycle_digests = [
            hashlib.sha256(
                json.dumps(cycle, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for cycle in cycles
        ]
        expected_targets = set(self.target_names)
        cycles_complete = bool(cycles) and all(len(cycle) == target_count for cycle in cycles)
        each_cycle_exact_target_set = cycles_complete and all(
            len({item["target"] for item in cycle}) == target_count
            and {item["target"] for item in cycle} == expected_targets
            for cycle in cycles
        )
        receipt = {
            "calls": self._active_call_count,
            "call_manifest_sha256": manifest_digest,
            "cycles": len(cycles),
            "cycle_manifest_sha256s": cycle_digests,
            "cycle_order_identical": (
                cycles_complete and len(set(cycle_digests)) == 1
            ),
            "each_cycle_exact_target_set": each_cycle_exact_target_set,
            "mask_sha256": self._mask_digest.hexdigest() if self._mask_digest else None,
        }
        self._capture_masks = False
        self._mask_digest = None
        return receipt

    @property
    def is_enabled(self) -> bool:
        return self._enabled_depth > 0 and self._suspended_depth == 0

    def reset_call_count(self) -> None:
        self._active_call_count = 0
        self._last_call_manifest = []

    @property
    def active_call_count(self) -> int:
        return self._active_call_count

    def target_manifest(self) -> list[dict[str, Any]]:
        result = []
        for key in sorted(self._key_to_target):
            if self.capacity == "lora":
                parameters = self.down[key].weight.numel() + self.up[key].weight.numel()
                shapes = [list(self.down[key].weight.shape), list(self.up[key].weight.shape)]
            else:
                parameters = self.deltas[key].weight.numel()
                shapes = [list(self.deltas[key].weight.shape)]
            result.append(
                {
                    "key": key,
                    "target": self._key_to_target[key],
                    "shapes": shapes,
                    "dtype": "torch.float32",
                    "parameters": parameters,
                }
            )
        return result

    def zero_function_receipt(self) -> dict[str, Any]:
        tensors = self.up.values() if self.capacity == "lora" else self.deltas.values()
        nonzero = 0
        max_abs = 0.0
        for module in tensors:
            weight = module.weight.detach()
            nonzero += int(torch.count_nonzero(weight).cpu())
            max_abs = max(max_abs, float(weight.abs().max().cpu()))
        return {"nonzero_output_weights": nonzero, "max_abs_output_weight": max_abs}

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
