"""Pinned native-prefix loading and cache-free semantic-anchor patching."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from coordinates import pseudoinverse, read_coordinates, replace_coordinates


def _tensor_from_output(output: Any) -> torch.Tensor:
    return output if torch.is_tensor(output) else output[0]


def _with_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    return (tensor,) + tuple(output[1:])


@dataclass(frozen=True)
class ContextLens:
    concepts: tuple[str, ...]
    token_ids: tuple[int, ...]
    source_layers: tuple[int, ...]
    directions: dict[int, torch.Tensor]  # [d_model,n_concepts]
    n_prompts: int
    estimator: str = "mean_direct_logit_pullback_at_selected_token"

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "concepts": self.concepts,
            "token_ids": self.token_ids,
            "source_layers": self.source_layers,
            "directions": {
                layer: value.to(torch.float16).cpu()
                for layer, value in self.directions.items()
            },
            "n_prompts": self.n_prompts,
            "estimator": self.estimator,
        }

    @classmethod
    def load(cls, path: str) -> "ContextLens":
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            concepts=tuple(state["concepts"]),
            token_ids=tuple(int(value) for value in state["token_ids"]),
            source_layers=tuple(int(value) for value in state["source_layers"]),
            directions={int(layer): value.float() for layer, value in state["directions"].items()},
            n_prompts=int(state["n_prompts"]),
            estimator=str(state["estimator"]),
        )


class ActivationRecorder:
    """Capture selected block outputs and root autograd at the earliest source."""

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        at: Iterable[int],
        *,
        start_graph_at: int | None = None,
    ):
        self.layers = layers
        self.indices = sorted(set(at) | ({start_graph_at} if start_graph_at is not None else set()))
        self.start_graph_at = start_graph_at
        self.activations: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _hook(self, index: int):
        def hook(_module, _inputs, output):
            tensor = _tensor_from_output(output)
            if index == self.start_graph_at:
                tensor.requires_grad_(True)
            self.activations[index] = tensor

        return hook

    def __enter__(self):
        for index in self.indices:
            self.handles.append(self.layers[index].register_forward_hook(self._hook(index)))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class FullActivationPatcher:
    """Set one sequence position to a fixed clean donor activation by layer."""

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        position: int,
        desired_by_layer: dict[int, torch.Tensor],
    ):
        self.layers = layers
        self.position = int(position)
        self.desired_by_layer = {
            layer: value.detach().clone().cpu() for layer, value in desired_by_layer.items()
        }
        self.handles: list[Any] = []
        self.deltas: dict[int, torch.Tensor] = {}
        self.applications: dict[int, int] = {layer: 0 for layer in desired_by_layer}

    def _hook(self, layer: int):
        desired = self.desired_by_layer[layer]

        def hook(_module, _inputs, output):
            tensor = _tensor_from_output(output)
            if tensor.shape[0] != 1 or not 0 <= self.position < tensor.shape[1]:
                raise RuntimeError("scientific patching requires an in-range batch-one position")
            if self.applications[layer] != 0:
                raise RuntimeError("full donor patch repeated at one layer")
            patched = tensor.clone()
            current = patched[:, self.position, :]
            target = desired.to(device=tensor.device, dtype=tensor.dtype).reshape_as(current)
            before = current.float().clone()
            patched[:, self.position, :] = target
            self.deltas[layer] = (patched[:, self.position, :].float() - before).detach().cpu()
            self.applications[layer] += 1
            return _with_tensor(output, patched)

        return hook

    def __enter__(self):
        for layer in sorted(self.desired_by_layer):
            self.handles.append(self.layers[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class CoordinateClampPatcher:
    """Set selected coordinates to fixed donor values at one sequence position."""

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        position: int,
        directions_by_layer: dict[int, torch.Tensor],
        desired_by_layer: dict[int, torch.Tensor],
        *,
        rtol: float,
    ):
        if set(directions_by_layer) != set(desired_by_layer):
            raise ValueError("directions and desired coordinates must cover the same layers")
        self.layers = layers
        self.position = int(position)
        self.directions_by_layer = {
            layer: value.detach().clone().cpu() for layer, value in directions_by_layer.items()
        }
        self.desired_by_layer = {
            layer: value.detach().clone().cpu() for layer, value in desired_by_layer.items()
        }
        self.rtol = float(rtol)
        self.handles: list[Any] = []
        self.deltas: dict[int, torch.Tensor] = {}
        self.applications: dict[int, int] = {layer: 0 for layer in desired_by_layer}

    def _hook(self, layer: int):
        directions = self.directions_by_layer[layer]
        desired = self.desired_by_layer[layer]

        def hook(_module, _inputs, output):
            tensor = _tensor_from_output(output)
            if tensor.shape[0] != 1 or not 0 <= self.position < tensor.shape[1]:
                raise RuntimeError("scientific patching requires an in-range batch-one position")
            if self.applications[layer] != 0:
                raise RuntimeError("coordinate donor patch repeated at one layer")
            patched = tensor.clone()
            before = patched[:, self.position, :].float().clone()
            changed, delta = replace_coordinates(
                patched[:, self.position, :],
                directions.to(tensor.device),
                desired.to(tensor.device).reshape(1, -1),
                rtol=self.rtol,
            )
            patched[:, self.position, :] = changed
            del delta
            self.deltas[layer] = (patched[:, self.position, :].float() - before).detach().cpu()
            self.applications[layer] += 1
            return _with_tensor(output, patched)

        return hook

    def __enter__(self):
        for layer in sorted(self.directions_by_layer):
            self.handles.append(self.layers[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class AddDeltaPatcher:
    """Add fixed per-layer delta vectors at one batch-one sequence position."""

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        position: int,
        deltas_by_layer: dict[int, torch.Tensor],
    ):
        self.layers = layers
        self.position = int(position)
        self.deltas_by_layer = {
            layer: value.detach().clone().cpu() for layer, value in deltas_by_layer.items()
        }
        self.handles: list[Any] = []
        self.deltas: dict[int, torch.Tensor] = {}
        self.applications: dict[int, int] = {layer: 0 for layer in deltas_by_layer}

    def _hook(self, layer: int):
        fixed_delta = self.deltas_by_layer[layer]

        def hook(_module, _inputs, output):
            tensor = _tensor_from_output(output)
            if tensor.shape[0] != 1 or not 0 <= self.position < tensor.shape[1]:
                raise RuntimeError("scientific patching requires an in-range batch-one position")
            if self.applications[layer] != 0:
                raise RuntimeError("additive patch repeated at one layer")
            patched = tensor.clone()
            before = patched[:, self.position, :].float().clone()
            delta = fixed_delta.to(device=tensor.device, dtype=tensor.dtype).reshape(1, -1)
            patched[:, self.position, :] += delta
            self.deltas[layer] = (patched[:, self.position, :].float() - before).detach().cpu()
            self.applications[layer] += 1
            return _with_tensor(output, patched)

        return hook

    def __enter__(self):
        for layer in sorted(self.deltas_by_layer):
            self.handles.append(self.layers[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class NormMatchedDeltaPatcher:
    """Add span-orthogonal bases with bf16-realized norms matched in each hook."""

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        position: int,
        bases_by_layer: dict[int, torch.Tensor],
        target_norms_by_layer: dict[int, float],
        *,
        search_steps: int = 64,
    ):
        if set(bases_by_layer) != set(target_norms_by_layer):
            raise ValueError("control bases and target norms must cover the same layers")
        self.layers = layers
        self.position = int(position)
        self.bases_by_layer = bases_by_layer
        self.target_norms_by_layer = target_norms_by_layer
        self.search_steps = int(search_steps)
        self.handles: list[Any] = []
        self.deltas: dict[int, torch.Tensor] = {}
        self.relative_errors: dict[int, float] = {}
        self.scales: dict[int, float] = {}
        self.chosen_indices: dict[int, int] = {}

    @staticmethod
    def _candidate(
        current: torch.Tensor, base: torch.Tensor, scale: float
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        changed = current + (base * scale).to(dtype=current.dtype)
        actual = changed.float() - current.float()
        return changed, actual, float(actual.norm())

    def _hook(self, layer: int):
        base_cpu = self.bases_by_layer[layer]
        target_norm = float(self.target_norms_by_layer[layer])

        def hook(_module, _inputs, output):
            tensor = _tensor_from_output(output)
            if tensor.shape[0] != 1 or not 0 <= self.position < tensor.shape[1]:
                raise RuntimeError("scientific patching requires an in-range batch-one position")
            patched = tensor.clone()
            current = patched[:, self.position, :]
            bases = base_cpu.to(device=tensor.device, dtype=torch.float32)
            if bases.ndim == 1:
                bases = bases[None, :]
            if bases.ndim != 2 or bases.shape[1] != current.shape[1]:
                raise RuntimeError("control base candidates must have shape [draws,d_model]")
            if target_norm == 0.0:
                changed = current
                actual = torch.zeros_like(current, dtype=torch.float32)
                best_scale = 0.0
                best_error = 0.0
                best_index = 0
            else:
                best = None
                best_scale = 0.0
                best_error = float("inf")
                best_index = -1
                for base_index, base in enumerate(bases):
                    base = base.reshape_as(current)
                    low, high = 0.0, 1.0
                    high_candidate = self._candidate(current, base, high)
                    while high_candidate[2] < target_norm and high < 1024.0:
                        low, high = high, high * 2.0
                        high_candidate = self._candidate(current, base, high)
                    if high_candidate[2] < target_norm:
                        continue
                    local_best = high_candidate
                    local_scale = high
                    local_error = abs(high_candidate[2] - target_norm) / target_norm
                    low_candidate = self._candidate(current, base, low)
                    low_error = abs(low_candidate[2] - target_norm) / target_norm
                    if low_error < local_error:
                        local_best, local_scale, local_error = low_candidate, low, low_error
                    for _ in range(self.search_steps):
                        midpoint = (low + high) / 2.0
                        candidate = self._candidate(current, base, midpoint)
                        error = abs(candidate[2] - target_norm) / target_norm
                        if error < local_error:
                            local_best, local_scale, local_error = candidate, midpoint, error
                        if candidate[2] < target_norm:
                            low = midpoint
                        else:
                            high = midpoint
                    if local_error < best_error:
                        best = local_best
                        best_scale = local_scale
                        best_error = local_error
                        best_index = base_index
                if best is None:
                    raise RuntimeError("could not bracket target perturbation norm")
                changed, actual, _norm = best
            patched[:, self.position, :] = changed
            self.deltas[layer] = actual.detach().cpu()
            self.relative_errors[layer] = float(best_error)
            self.scales[layer] = float(best_scale)
            self.chosen_indices[layer] = int(best_index)
            return _with_tensor(output, patched)

        return hook

    def __enter__(self):
        for layer in sorted(self.bases_by_layer):
            self.handles.append(self.layers[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class QuantizationAwareOrthogonalPatcher:
    """Choose controls by post-bf16 norm and J-span geometry only."""

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        position: int,
        bases_by_layer: dict[int, torch.Tensor],
        directions_by_layer: dict[int, torch.Tensor],
        target_norms_by_layer: dict[int, float],
        *,
        rtol: float,
        norm_tolerance: float,
        projection_tolerance: float,
        correction_iterations: int,
        correction_damping: float,
        binary_search_steps: int,
        lattice_pair_steps: int,
        repair_safety_margin: float,
        check_interval: int = 16,
    ):
        keys = set(bases_by_layer)
        if keys != set(directions_by_layer) or keys != set(target_norms_by_layer):
            raise ValueError("control geometry must cover identical layers")
        self.layers = layers
        self.position = int(position)
        self.bases_by_layer = bases_by_layer
        self.target_norms_by_layer = target_norms_by_layer
        self.norm_tolerance = float(norm_tolerance)
        self.projection_tolerance = float(projection_tolerance)
        self.correction_iterations = int(correction_iterations)
        self.correction_damping = float(correction_damping)
        self.binary_search_steps = int(binary_search_steps)
        self.lattice_pair_steps_max = int(lattice_pair_steps)
        self.repair_safety_margin = float(repair_safety_margin)
        if not 0.0 < self.repair_safety_margin < 1.0:
            raise ValueError("repair safety margin must be between zero and one")
        self.check_interval = int(check_interval)
        self.geometry = {
            layer: pseudoinverse(directions_by_layer[layer], rtol=rtol)
            for layer in keys
        }
        self.handles: list[Any] = []
        self.deltas: dict[int, torch.Tensor] = {}
        self.norm_errors: dict[int, float] = {}
        self.projection_fractions: dict[int, float] = {}
        self.chosen_indices: dict[int, int] = {}
        self.iterations_used: dict[int, int] = {}
        self.lattice_pair_steps: dict[int, int] = {}
        self.passed_by_layer: dict[int, bool] = {}
        self.input_activations: dict[int, torch.Tensor] = {}
        self.applications: dict[int, int] = {layer: 0 for layer in keys}

    @staticmethod
    def _actual(current: torch.Tensor, requested: torch.Tensor) -> torch.Tensor:
        changed = current + requested.to(dtype=current.dtype)
        return changed.float() - current.float()

    @staticmethod
    def _project(
        vectors: torch.Tensor, dictionary: torch.Tensor, inverse: torch.Tensor
    ) -> torch.Tensor:
        return (vectors.float() @ inverse.T) @ dictionary.T

    def _geometry_match(
        self,
        current: torch.Tensor,
        candidates: torch.Tensor,
        target_norm: float,
        dictionary: torch.Tensor,
        inverse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scale-search while retaining the best joint realized geometry.

        Bfloat16 scale plateaus can contain several deltas with essentially the
        same norm but materially different J-span leakage.  Selecting solely by
        closest norm silently discards a jointly feasible state.  The frozen
        decision constraints therefore define the numeric objective at every
        visited binary-search scale.
        """
        draws = candidates.shape[0]
        if target_norm == 0.0:
            actual = torch.zeros_like(candidates)
            zeros = torch.zeros(draws, device=candidates.device)
            return actual, zeros, zeros, zeros
        low = torch.zeros(draws, device=candidates.device)
        high = torch.ones(draws, device=candidates.device)
        for _ in range(12):
            actual = self._actual(current, candidates * high[:, None])
            below = actual.norm(dim=-1) < target_norm
            if not bool(below.any()):
                break
            high = torch.where(below, high * 2.0, high)
        best_actual = self._actual(current, candidates * high[:, None])
        best_norm = best_actual.norm(dim=-1)
        best_error = (best_norm - target_norm).abs() / target_norm
        best_projection = self._project(
            best_actual, dictionary, inverse
        ).norm(dim=-1) / best_norm.clamp_min(1e-12)
        best_objective = torch.maximum(
            best_error / self.norm_tolerance,
            best_projection / self.projection_tolerance,
        )
        best_scale = high.clone()
        for _ in range(self.binary_search_steps):
            midpoint = (low + high) / 2.0
            actual = self._actual(current, candidates * midpoint[:, None])
            norms = actual.norm(dim=-1)
            errors = (norms - target_norm).abs() / target_norm
            projection = self._project(
                actual, dictionary, inverse
            ).norm(dim=-1) / norms.clamp_min(1e-12)
            objective = torch.maximum(
                errors / self.norm_tolerance,
                projection / self.projection_tolerance,
            )
            better = objective < best_objective
            best_objective = torch.where(better, objective, best_objective)
            best_error = torch.where(better, errors, best_error)
            best_projection = torch.where(better, projection, best_projection)
            best_scale = torch.where(better, midpoint, best_scale)
            low = torch.where(norms < target_norm, midpoint, low)
            high = torch.where(norms < target_norm, high, midpoint)
        best_actual = self._actual(current, candidates * best_scale[:, None])
        return best_actual, best_error, best_projection, best_scale

    def _lattice_pair_repair(
        self,
        current: torch.Tensor,
        initial_actual: torch.Tensor,
        target_norm: float,
        dictionary: torch.Tensor,
        inverse: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, int]:
        """Improve joint geometry with exact pairs of neighboring bf16 moves.

        For an orthogonal projector P, changing coordinate i by s changes
        ||P delta||^2 by 2*s*(P delta)_i + s^2*P_ii.  This lets us score every
        distinct pair of one-ULP bf16 moves exactly, in bounded blocks, without
        another model evaluation or any outcome signal.  Pairing is important:
        one move generally cannot preserve the 1e-5 norm tolerance.
        """
        if target_norm == 0.0:
            return torch.zeros_like(initial_actual), 0.0, 0.0, 0
        current_flat = current.reshape(-1)
        changed = (
            current_flat.float() + initial_actual.reshape(-1).float()
        ).to(current.dtype)
        projection_matrix = dictionary @ inverse
        diagonal = projection_matrix.diagonal()
        width = current_flat.numel()
        coordinate = torch.arange(width, device=current.device).repeat(2)
        last = None
        for step in range(self.lattice_pair_steps_max + 1):
            delta = changed.float() - current_flat.float()
            projection = self._project(delta, dictionary, inverse).reshape(-1)
            norm = delta.norm()
            projection_fraction = projection.norm() / norm.clamp_min(1e-12)
            norm_error = (norm - target_norm).abs() / target_norm
            objective = torch.maximum(
                norm_error / self.norm_tolerance,
                projection_fraction / self.projection_tolerance,
            )
            last = (
                delta.reshape_as(initial_actual),
                float(norm_error),
                float(projection_fraction),
                step,
            )
            if (
                float(objective) <= self.repair_safety_margin
                or step == self.lattice_pair_steps_max
            ):
                return last
            upper = torch.nextafter(
                changed, torch.full_like(changed, float("inf"))
            )
            lower = torch.nextafter(
                changed, torch.full_like(changed, float("-inf"))
            )
            move = torch.cat((
                upper.float() - changed.float(),
                lower.float() - changed.float(),
            ))
            norm_change = 2.0 * move * delta[coordinate] + move.square()
            projection_change = (
                2.0 * move * projection[coordinate]
                + move.square() * diagonal[coordinate]
            )
            best_value = float(objective)
            best_pair: tuple[int, int] | None = None
            block = 256
            for start in range(0, move.numel(), block):
                stop = min(move.numel(), start + block)
                first = torch.arange(start, stop, device=current.device)
                cross = (
                    2.0
                    * move[first, None]
                    * move[None, :]
                    * projection_matrix[
                        coordinate[first, None], coordinate[None, :]
                    ]
                )
                candidate_norm_sq = (
                    norm.square()
                    + norm_change[first, None]
                    + norm_change[None, :]
                ).clamp_min(0.0)
                candidate_projection_sq = (
                    projection.square().sum()
                    + projection_change[first, None]
                    + projection_change[None, :]
                    + cross
                ).clamp_min(0.0)
                candidate_norm = candidate_norm_sq.sqrt()
                candidate_error = (
                    candidate_norm - target_norm
                ).abs() / target_norm
                candidate_fraction = (
                    candidate_projection_sq.sqrt()
                    / candidate_norm.clamp_min(1e-12)
                )
                candidate_objective = torch.maximum(
                    candidate_error / self.norm_tolerance,
                    candidate_fraction / self.projection_tolerance,
                )
                candidate_objective = candidate_objective.masked_fill(
                    coordinate[first, None] == coordinate[None, :],
                    float("inf"),
                )
                local_value, local_flat = candidate_objective.flatten().min(dim=0)
                if float(local_value) < best_value:
                    local_row = int(local_flat) // move.numel()
                    local_column = int(local_flat) % move.numel()
                    best_value = float(local_value)
                    best_pair = (int(first[local_row]), local_column)
            if best_pair is None:
                return last
            first, second = best_pair
            first_coordinate = int(coordinate[first])
            second_coordinate = int(coordinate[second])
            changed[first_coordinate] = (
                upper if first < width else lower
            )[first_coordinate]
            changed[second_coordinate] = (
                upper if second < width else lower
            )[second_coordinate]
        raise AssertionError("bounded lattice repair did not return")

    def _hook(self, layer: int):
        bases_cpu = self.bases_by_layer[layer]
        target_norm = float(self.target_norms_by_layer[layer])
        dictionary_cpu, inverse_cpu = self.geometry[layer]

        def hook(_module, _inputs, output):
            tensor = _tensor_from_output(output)
            if tensor.shape[0] != 1 or not 0 <= self.position < tensor.shape[1]:
                raise RuntimeError("scientific patching requires an in-range batch-one position")
            if self.applications[layer] != 0:
                raise RuntimeError("orthogonal control patch repeated at one layer")
            patched = tensor.clone()
            current = patched[:, self.position, :]
            current_float = current.float().clone()
            self.input_activations[layer] = current_float.detach().cpu()
            candidates = bases_cpu.to(device=tensor.device, dtype=torch.float32)
            dictionary = dictionary_cpu.to(tensor.device)
            inverse = inverse_cpu.to(tensor.device)
            best_objective = torch.full(
                (candidates.shape[0],), float("inf"), device=tensor.device
            )
            best_actual = torch.zeros_like(candidates)
            best_norm_error = torch.full_like(best_objective, float("inf"))
            best_projection = torch.full_like(best_objective, float("inf"))
            best_iteration = torch.zeros(
                candidates.shape[0], dtype=torch.long, device=tensor.device
            )
            # The live bf16 map is discontinuous.  A correction trajectory can
            # briefly visit a better quantization cell between the relatively
            # expensive scale-search checkpoints below.  Retain, separately
            # for each of the 32 preregistered starts, the request whose raw
            # realized delta has the smallest span projection.  At the frozen
            # 512-iteration boundary we scale-search these retained requests as
            # well.  This is geometry-only bookkeeping: it neither creates new
            # random draws nor inspects logits or labels.
            best_requested = candidates.clone()
            initial_actual = self._actual(current, candidates)
            if target_norm == 0.0:
                best_requested_projection = torch.zeros(
                    candidates.shape[0], device=tensor.device
                )
            else:
                best_requested_projection = self._project(
                    initial_actual, dictionary, inverse
                ).norm(dim=-1) / initial_actual.norm(dim=-1).clamp_min(1e-12)
            best_requested_iteration = torch.zeros(
                candidates.shape[0], dtype=torch.long, device=tensor.device
            )
            completed = 0
            chosen = None
            while True:
                actual, norm_error, projection_fraction, _scales = (
                    self._geometry_match(
                        current,
                        candidates,
                        target_norm,
                        dictionary,
                        inverse,
                    )
                )
                objective = torch.maximum(
                    norm_error / self.norm_tolerance,
                    projection_fraction / self.projection_tolerance,
                )
                better = objective < best_objective
                best_objective = torch.where(better, objective, best_objective)
                best_actual = torch.where(better[:, None], actual, best_actual)
                best_norm_error = torch.where(better, norm_error, best_norm_error)
                best_projection = torch.where(better, projection_fraction, best_projection)
                best_iteration = torch.where(
                    better,
                    torch.full_like(best_iteration, completed),
                    best_iteration,
                )
                eligible = (
                    (norm_error <= self.norm_tolerance)
                    & (projection_fraction <= self.projection_tolerance)
                )
                passing = eligible.nonzero(as_tuple=False).flatten()
                if passing.numel():
                    chosen = int(passing[0].item())
                    chosen_actual = actual[chosen : chosen + 1]
                    chosen_norm_error = float(norm_error[chosen])
                    chosen_projection = float(projection_fraction[chosen])
                    chosen_iteration = completed
                    break
                if completed >= self.correction_iterations:
                    (
                        retained_actual,
                        retained_norm_error,
                        retained_projection,
                        _retained_scales,
                    ) = self._geometry_match(
                        current,
                        best_requested,
                        target_norm,
                        dictionary,
                        inverse,
                    )
                    retained_objective = torch.maximum(
                        retained_norm_error / self.norm_tolerance,
                        retained_projection / self.projection_tolerance,
                    )
                    retained_better = retained_objective < best_objective
                    best_objective = torch.where(
                        retained_better, retained_objective, best_objective
                    )
                    best_actual = torch.where(
                        retained_better[:, None], retained_actual, best_actual
                    )
                    best_norm_error = torch.where(
                        retained_better, retained_norm_error, best_norm_error
                    )
                    best_projection = torch.where(
                        retained_better, retained_projection, best_projection
                    )
                    best_iteration = torch.where(
                        retained_better,
                        best_requested_iteration,
                        best_iteration,
                    )
                    retained_eligible = (
                        (retained_norm_error <= self.norm_tolerance)
                        & (retained_projection <= self.projection_tolerance)
                    )
                    retained_passing = retained_eligible.nonzero(
                        as_tuple=False
                    ).flatten()
                    if retained_passing.numel():
                        chosen = int(retained_passing[0].item())
                        chosen_actual = retained_actual[chosen : chosen + 1]
                        chosen_norm_error = float(retained_norm_error[chosen])
                        chosen_projection = float(retained_projection[chosen])
                        chosen_iteration = int(best_requested_iteration[chosen])
                        break
                    chosen = int(torch.argmin(best_objective).item())
                    chosen_actual = best_actual[chosen : chosen + 1]
                    chosen_norm_error = float(best_norm_error[chosen])
                    chosen_projection = float(best_projection[chosen])
                    chosen_iteration = int(best_iteration[chosen])
                    break
                steps = min(
                    self.check_interval, self.correction_iterations - completed
                )
                for _ in range(steps):
                    actual = self._actual(current, candidates)
                    projection = self._project(actual, dictionary, inverse)
                    orthogonal = actual - projection
                    orthogonal_norm = orthogonal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    desired = orthogonal * (target_norm / orthogonal_norm)
                    candidates = candidates + self.correction_damping * (desired - actual)
                    completed += 1
                    corrected_actual = self._actual(current, candidates)
                    if target_norm == 0.0:
                        corrected_projection = torch.zeros(
                            candidates.shape[0], device=tensor.device
                        )
                    else:
                        corrected_projection = self._project(
                            corrected_actual, dictionary, inverse
                        ).norm(dim=-1) / corrected_actual.norm(dim=-1).clamp_min(1e-12)
                    request_better = (
                        corrected_projection < best_requested_projection
                    )
                    best_requested_projection = torch.where(
                        request_better,
                        corrected_projection,
                        best_requested_projection,
                    )
                    best_requested = torch.where(
                        request_better[:, None], candidates, best_requested
                    )
                    best_requested_iteration = torch.where(
                        request_better,
                        torch.full_like(best_requested_iteration, completed),
                        best_requested_iteration,
                    )
            lattice_steps = 0
            if not (
                chosen_norm_error <= self.norm_tolerance
                and chosen_projection <= self.projection_tolerance
            ):
                (
                    chosen_actual,
                    chosen_norm_error,
                    chosen_projection,
                    lattice_steps,
                ) = self._lattice_pair_repair(
                    current,
                    chosen_actual,
                    target_norm,
                    dictionary,
                    inverse,
                )
            patched[:, self.position, :] = (
                current_float + chosen_actual
            ).to(tensor.dtype)
            realized = patched[:, self.position, :].float() - current_float
            if target_norm == 0.0:
                chosen_norm_error = 0.0
                chosen_projection = 0.0
            else:
                chosen_norm_error = float(
                    (realized.norm() - target_norm).abs() / target_norm
                )
                chosen_projection = float(
                    self._project(realized, dictionary, inverse).norm()
                    / realized.norm().clamp_min(1e-12)
                )
            self.deltas[layer] = realized.detach().cpu()
            self.norm_errors[layer] = chosen_norm_error
            self.projection_fractions[layer] = chosen_projection
            self.chosen_indices[layer] = int(chosen)
            self.iterations_used[layer] = int(chosen_iteration)
            self.lattice_pair_steps[layer] = int(lattice_steps)
            self.passed_by_layer[layer] = bool(
                chosen_norm_error <= self.norm_tolerance
                and chosen_projection <= self.projection_tolerance
            )
            self.applications[layer] += 1
            return _with_tensor(output, patched)

        return hook

    def __enter__(self):
        for layer in sorted(self.bases_by_layer):
            self.handles.append(self.layers[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class QwenClampModel:
    def __init__(self, config: dict[str, Any]):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_config = config["model"]
        if model_config["id"] != "Qwen/Qwen3.5-4B":
            raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["id"], revision=model_config["revision"], trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["id"],
            revision=model_config["revision"],
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=model_config["device"],
            attn_implementation=model_config["attention"],
        ).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.text_model = self.model.model
        self.layers = self.text_model.layers
        self.lm_head = self.model.lm_head
        self.device = self.lm_head.weight.device
        text_config = self.model.config.get_text_config()
        self.n_layers = int(text_config.num_hidden_layers)
        self.d_model = int(text_config.hidden_size)
        self.vocab_size = int(text_config.vocab_size)
        self.load_seconds = time.perf_counter() - started

    def render(self, user: str, *, enable_thinking: bool = False) -> str:
        messages = [
            {"role": "system", "content": "Follow the requested output format exactly."},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def encode(self, text: str, *, max_length: int) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        ids = encoded.input_ids
        bos = self.tokenizer.bos_token_id
        if bos is not None and ids.shape[1] and int(ids[0, 0]) != int(bos):
            ids = torch.cat([torch.tensor([[bos]], dtype=ids.dtype), ids], dim=1)
            ids = ids[:, -max_length:]
        return ids.to(self.device)

    def concept_token_id(self, concept: str) -> int:
        ids = self.tokenizer(" " + concept, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"concept {concept!r} is not one leading-space token: {ids}")
        return int(ids[0])

    def bare_token_id(self, text: str) -> int:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"answer {text!r} is not one bare token: {ids}")
        return int(ids[0])

    def rendered_prefix(self, user: str, *, kind: str) -> str:
        response_prefix = {"direct": "Key:", "consequence": "Value: "}.get(kind)
        if response_prefix is None:
            raise ValueError(f"unknown prompt kind: {kind}")
        return self.render(user) + response_prefix

    def prepare_native_prompt(self, user: str, *, max_length: int) -> dict[str, Any]:
        rendered = self.render(user, enable_thinking=True)
        input_ids = self.tokenizer(rendered, return_tensors="pt").input_ids
        bos = self.tokenizer.bos_token_id
        if bos is not None and input_ids.shape[1] and int(input_ids[0, 0]) != int(bos):
            input_ids = torch.cat([torch.tensor([[bos]], dtype=input_ids.dtype), input_ids], dim=1)
        if input_ids.shape[1] > max_length:
            raise RuntimeError(
                f"native prompt has {input_ids.shape[1]} tokens, above frozen maximum {max_length}"
            )
        open_id = self.tokenizer.convert_tokens_to_ids("<think>")
        close_id = self.tokenizer.convert_tokens_to_ids("</think>")
        opens = (input_ids[0] == open_id).nonzero(as_tuple=False).flatten()
        if opens.numel() != 1 or bool((input_ids[0] == close_id).any()):
            raise RuntimeError("native prompt must have exactly one open and no close token")
        return {
            "rendered": rendered,
            "input_ids": input_ids.to(self.device),
            "sequence_tokens": int(input_ids.shape[1]),
            "think_open_position": int(opens[0]),
            "think_open_id": int(open_id),
            "think_close_id": int(close_id),
        }

    @torch.no_grad()
    def generate_native_prefix(
        self,
        prepared: dict[str, Any],
        *,
        seed: int,
        tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> dict[str, Any]:
        prompt_ids = prepared["input_ids"]
        prompt_tokens = int(prompt_ids.shape[1])
        devices = [self.device.index] if self.device.type == "cuda" else []
        started = time.perf_counter()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            generated = self.model.generate(
                input_ids=prompt_ids,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                max_new_tokens=int(tokens),
                eos_token_id=self.model.config.get_text_config().eos_token_id,
                pad_token_id=self.model.config.get_text_config().eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
            ).sequences[0, prompt_tokens:]
        close_id = int(prepared["think_close_id"])
        eos_id = int(self.model.config.get_text_config().eos_token_id)
        if generated.numel() != int(tokens):
            raise RuntimeError("native prefix terminated before the frozen token boundary")
        if bool(((generated == close_id) | (generated == eos_id)).any()):
            raise RuntimeError("native prefix contains close/EOS before the anchor")
        return {
            "token_ids": [int(value) for value in generated.tolist()],
            "text": self.tokenizer.decode(generated.tolist(), skip_special_tokens=False),
            "tokens": int(generated.numel()),
            "elapsed_seconds": time.perf_counter() - started,
            "close_present": False,
            "eos_present": False,
        }

    def leading_space_token_id(self, text: str) -> int:
        ids = self.tokenizer(" " + text, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"{text!r} is not one leading-space token: {ids}")
        return int(ids[0])

    def prepare_anchor_context(
        self,
        native: dict[str, Any],
        thought_token_ids: Sequence[int],
        *,
        prefix_text: str,
        anchor_alias: str,
        suffix_text: str,
        max_length: int,
    ) -> dict[str, Any]:
        prefix_ids = [
            int(value)
            for value in self.tokenizer(prefix_text, add_special_tokens=False).input_ids
        ]
        anchor_id = self.leading_space_token_id(anchor_alias)
        suffix_ids = [
            int(value)
            for value in self.tokenizer(suffix_text, add_special_tokens=False).input_ids
        ]
        manual_scaffold = prefix_ids + [anchor_id] + suffix_ids
        literal_scaffold = prefix_text + " " + anchor_alias + suffix_text
        whole_ids = [
            int(value)
            for value in self.tokenizer(literal_scaffold, add_special_tokens=False).input_ids
        ]
        if manual_scaffold != whole_ids:
            raise RuntimeError("piecewise anchor tokenization differs from literal scaffold")
        if self.tokenizer.decode(manual_scaffold, skip_special_tokens=False) != literal_scaffold:
            raise RuntimeError("piecewise anchor scaffold does not decode exactly")
        base = native["input_ids"][0].tolist() + [int(value) for value in thought_token_ids]
        position = len(base) + len(prefix_ids)
        full = base + prefix_ids + [anchor_id] + suffix_ids
        if len(full) > int(max_length):
            raise RuntimeError("anchor context exceeds frozen maximum length")
        input_ids = torch.tensor([full], dtype=native["input_ids"].dtype, device=self.device)
        if int(input_ids[0, position]) != anchor_id:
            raise AssertionError("anchor token position changed")
        open_id = int(native["think_open_id"])
        close_id = int(native["think_close_id"])
        eos_id = int(self.model.config.get_text_config().eos_token_id)
        opens = (input_ids[0] == open_id).nonzero(as_tuple=False).flatten()
        closes = (input_ids[0] == close_id).nonzero(as_tuple=False).flatten()
        if opens.numel() != 1 or closes.numel() != 1 or int(closes[0]) <= position:
            raise RuntimeError("anchor context requires one open and one later close token")
        if bool((input_ids[0] == eos_id).any()):
            raise RuntimeError("anchor context contains EOS")
        occurrences = (input_ids[0] == anchor_id).nonzero(as_tuple=False).flatten().tolist()
        return {
            "input_ids": input_ids,
            "position": int(position),
            "anchor_id": int(anchor_id),
            "anchor_alias": anchor_alias,
            "anchor_occurrences": occurrences,
            "sequence_tokens": len(full),
            "prefix_tokens": len(base),
            "forced_prefix_tokens": len(prefix_ids),
            "suffix_tokens": len(suffix_ids),
            "think_close_position": int(closes[0]),
            "whole_scaffold_tokenization_pass": True,
        }

    def prepare(self, user: str, *, kind: str, selected_concept: str, max_length: int) -> dict[str, Any]:
        rendered = self.rendered_prefix(user, kind=kind)
        input_ids = self.tokenizer(rendered, return_tensors="pt").input_ids
        bos = self.tokenizer.bos_token_id
        if bos is not None and input_ids.shape[1] and int(input_ids[0, 0]) != int(bos):
            input_ids = torch.cat([torch.tensor([[bos]], dtype=input_ids.dtype), input_ids], dim=1)
        if input_ids.shape[1] > max_length:
            raise RuntimeError(
                f"rendered prompt has {input_ids.shape[1]} tokens, above frozen maximum {max_length}"
            )
        input_ids = input_ids.to(self.device)
        token_id = self.concept_token_id(selected_concept)
        occurrences = (input_ids[0] == token_id).nonzero(as_tuple=False).flatten().tolist()
        if not occurrences:
            raise RuntimeError(f"selected concept token {selected_concept!r} is absent")
        return {
            "rendered": rendered,
            "input_ids": input_ids,
            "position": int(occurrences[-1]),
            "selected_token_id": token_id,
            "sequence_tokens": int(input_ids.shape[1]),
        }

    @torch.no_grad()
    def capture(
        self,
        prepared: dict[str, Any],
        *,
        layers: Sequence[int],
        retain_logits: bool = True,
    ) -> dict[str, Any]:
        with ActivationRecorder(self.layers, at=layers) as recorder:
            output = self.model(
                input_ids=prepared["input_ids"], use_cache=False, logits_to_keep=1
            )
        position = int(prepared["position"])
        result = {
            "activations": {
                layer: recorder.activations[layer][0, position].float().detach().cpu()
                for layer in layers
            },
            "position": position,
            "sequence_tokens": int(prepared["sequence_tokens"]),
        }
        if retain_logits:
            result["logits"] = output.logits[0, -1].float().detach().cpu()
        return result

    @torch.no_grad()
    def score(self, prepared: dict[str, Any], *, patcher: Any | None = None) -> dict[str, Any]:
        if patcher is None:
            output = self.model(
                input_ids=prepared["input_ids"], use_cache=False, logits_to_keep=1
            )
        else:
            with patcher:
                output = self.model(
                    input_ids=prepared["input_ids"], use_cache=False, logits_to_keep=1
                )
        logits = output.logits[0, -1].float().detach().cpu()
        return {
            "logits": logits,
            "top_id": int(torch.argmax(logits).item()),
            "sequence_tokens": int(prepared["sequence_tokens"]),
            "deltas": {} if patcher is None else dict(patcher.deltas),
        }

    @torch.no_grad()
    def apply_without_retaining_logits(
        self, prepared: dict[str, Any], *, patcher: Any
    ) -> dict[str, Any]:
        with patcher:
            self.model(
                input_ids=prepared["input_ids"], use_cache=False, logits_to_keep=1
            )
        return {
            "sequence_tokens": int(prepared["sequence_tokens"]),
            "deltas": dict(patcher.deltas),
            "logits_retained": False,
        }

    def fit_context_lens(
        self,
        prepared_prompts: Sequence[dict[str, Any]],
        concepts: Sequence[str],
        *,
        source_layers: Sequence[int],
        concept_batch: int,
    ) -> tuple[ContextLens, list[dict[str, Any]]]:
        token_ids = tuple(self.concept_token_id(concept) for concept in concepts)
        sums = {
            layer: torch.zeros(self.d_model, len(concepts), dtype=torch.float32)
            for layer in source_layers
        }
        receipts = []
        for prompt_index, prepared in enumerate(prepared_prompts):
            input_ids = prepared["input_ids"]
            position = int(prepared["position"])
            for start in range(0, len(concepts), concept_batch):
                stop = min(len(concepts), start + concept_batch)
                batch_ids = input_ids.expand(stop - start, -1)
                with ActivationRecorder(
                    self.layers,
                    at=source_layers,
                    start_graph_at=min(source_layers),
                ) as recorder, torch.enable_grad():
                    output = self.model(
                        input_ids=batch_ids, use_cache=False, logits_to_keep=1
                    )
                    row = torch.arange(stop - start, device=self.device)
                    chosen = output.logits[:, -1, :][row, list(token_ids[start:stop])]
                    sources = [recorder.activations[layer] for layer in source_layers]
                    gradients = torch.autograd.grad(chosen.sum(), sources, retain_graph=False)
                    for layer, gradient in zip(source_layers, gradients, strict=True):
                        sums[layer][:, start:stop] += gradient[:, position, :].float().T.detach().cpu()
                    del gradients, output, chosen
            receipts.append({
                "prompt_index": prompt_index,
                "position": position,
                "sequence_tokens": int(prepared["sequence_tokens"]),
            })
        directions = {layer: value / len(prepared_prompts) for layer, value in sums.items()}
        return ContextLens(
            concepts=tuple(concepts),
            token_ids=token_ids,
            source_layers=tuple(int(layer) for layer in source_layers),
            directions=directions,
            n_prompts=len(prepared_prompts),
        ), receipts

    def donor_coordinates(
        self,
        activations: dict[int, torch.Tensor],
        directions: dict[int, torch.Tensor],
        *,
        rtol: float,
    ) -> dict[int, torch.Tensor]:
        return {
            layer: read_coordinates(
                activations[layer].reshape(1, -1), directions[layer], rtol=rtol
            )[0].cpu()
            for layer in directions
        }
