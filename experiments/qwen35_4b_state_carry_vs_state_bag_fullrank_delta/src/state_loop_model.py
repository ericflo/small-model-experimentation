"""Qwen3.5 recurrence with direct full-rank deltas on extra R calls only.

This module intentionally requires the pinned Transformers training environment.
It does not provide a fallback architecture: a failed Qwen contract is a failed
experiment setup, not permission to substitute another model.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StateLoopOutput:
    answer_logits: torch.Tensor
    loss: torch.Tensor | None
    answer_loss: torch.Tensor | None
    state_loss: torch.Tensor | None
    fixed_point_loss: torch.Tensor | None
    states: tuple[torch.Tensor, ...]
    node_logits: torch.Tensor
    phase_logits: torch.Tensor
    checksum_logits: torch.Tensor
    diagnostics: dict[str, Any]


class SinusoidalStepEncoder(nn.Module):
    """Stationary step signal defined beyond the trained recurrence horizon."""

    def __init__(self, encoding_dim: int, hidden_size: int) -> None:
        super().__init__()
        if encoding_dim % 2:
            raise ValueError("step encoding dimension must be even")
        self.encoding_dim = encoding_dim
        self.project = nn.Linear(encoding_dim, hidden_size, bias=False)
        nn.init.zeros_(self.project.weight)

    def forward(self, step: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        half = self.encoding_dim // 2
        exponent = torch.arange(half, device=device, dtype=torch.float32)
        frequencies = torch.exp(-math.log(10_000.0) * exponent / max(half - 1, 1))
        phase = float(step) * frequencies
        encoding = torch.cat((phase.sin(), phase.cos()), dim=0).to(dtype=dtype)
        return self.project(encoding)


class LowRankStateAdapter(nn.Module):
    """Small trainable initializer used only when extra recurrence is requested."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state + self.up(F.silu(self.down(self.norm(state))))


class StateSufficiencyHeads(nn.Module):
    """Shared query-after-state decoders; the natural-language query is never an input."""

    def __init__(self, hidden_size: int, node_count: int, checksum_modulus: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.node = nn.Linear(hidden_size, node_count)
        self.phase = nn.Linear(hidden_size, 2)
        self.checksum = nn.Linear(hidden_size, checksum_modulus)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # states: [batch, step, slot, hidden]
        pooled = self.norm(states.mean(dim=2))
        return self.node(pooled), self.phase(pooled), self.checksum(pooled)


class FullRankDeltaBank(nn.Module):
    """Direct full-shape weight deltas attached to frozen base linears.

    Hooks are inert unless the surrounding recurrent block explicitly opens the
    extra-call context.  Target modules are deliberately kept as non-owning
    references so the frozen base model is not registered twice.
    """

    def __init__(
        self,
        base_model: nn.Module,
        target_names: list[str],
        *,
        dropout: float,
        scale: float,
    ) -> None:
        super().__init__()
        modules = dict(base_model.named_modules())
        self.dropout = float(dropout)
        self.scale = float(scale)
        self.deltas = nn.ModuleDict()
        self.target_names = tuple(target_names)
        self._enabled_depth = 0
        self._suspended_depth = 0
        self._active_call_count = 0
        self._handles: list[Any] = []
        self._key_to_target: dict[str, str] = {}
        for index, name in enumerate(self.target_names):
            target = modules.get(name)
            if not isinstance(target, nn.Linear):
                raise RuntimeError(f"full-rank target is not nn.Linear: {name}")
            key = f"d{index:03d}"
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
            # The handle owns only a callback.  Do not store target as a child.
            self._handles.append(target.register_forward_hook(self._make_hook(key)))

    def _make_hook(self, key: str):
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            del module
            if self._enabled_depth == 0 or self._suspended_depth > 0:
                return None
            if len(inputs) != 1:
                raise RuntimeError("full-rank delta target received unexpected arguments")
            self._active_call_count += 1
            dropped = F.dropout(inputs[0], p=self.dropout, training=self.training)
            return output + self.scale * self.deltas[key](dropped)

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
                raise RuntimeError("full-rank delta context underflow")

    @contextlib.contextmanager
    def suspended(self):
        self._suspended_depth += 1
        try:
            yield
        finally:
            self._suspended_depth -= 1
            if self._suspended_depth < 0:
                raise RuntimeError("full-rank delta suspension underflow")

    @property
    def is_enabled(self) -> bool:
        return self._enabled_depth > 0 and self._suspended_depth == 0

    def reset_call_count(self) -> None:
        self._active_call_count = 0

    @property
    def active_call_count(self) -> int:
        return self._active_call_count

    def target_manifest(self) -> list[dict[str, Any]]:
        manifest = []
        for key, delta in self.deltas.items():
            manifest.append(
                {
                    "key": key,
                    "target": self._key_to_target[key],
                    "shape": list(delta.weight.shape),
                    "dtype": str(delta.weight.dtype),
                    "parameters": delta.weight.numel(),
                }
            )
        return manifest

    def zero_receipt(self) -> dict[str, Any]:
        nonzero = 0
        max_abs = 0.0
        for delta in self.deltas.values():
            weight = delta.weight.detach()
            nonzero += int(torch.count_nonzero(weight).cpu())
            max_abs = max(max_abs, float(weight.abs().max().cpu()))
        return {"nonzero": nonzero, "max_abs": max_abs}


class StateLoopModel(nn.Module):
    """Manual Qwen text forward with a recurrent middle-block state bottleneck.

    The first P->R->C path has every full-rank delta disabled.  For K=1 it is
    therefore algebraically the original Qwen forward over the identical token
    sequence.  For K>1, only state-slot activations cross repeated R calls;
    non-state token activations are reset to their first-pass values.
    """

    def __init__(
        self, base_model: nn.Module, config: Mapping[str, Any], target_names: list[str]
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.experiment_config = config
        self.arch = config["architecture"]
        self.substrate = config["substrate"]
        self.training_config = config["training"]
        core = base_model
        # Keep non-owning shortcuts without registering the already-owned base
        # module two additional times in this wrapper.
        object.__setattr__(self, "_core_ref", core)
        object.__setattr__(self, "_text_ref", core.model)
        self.hidden_size = int(self.text.config.hidden_size)
        self.loop_start = int(self.arch["loop_start"])
        self.loop_end = int(self.arch["loop_end"])
        self.state_slots = int(self.arch["state_slots"])
        self.max_recurrence = int(self.arch["max_recurrence"])
        delta_config = self.arch["full_rank_delta"]
        self.delta_bank = FullRankDeltaBank(
            base_model,
            target_names,
            dropout=float(delta_config["dropout"]),
            scale=float(delta_config["scale"]),
        )

        self.state_initializer = LowRankStateAdapter(
            self.hidden_size, int(self.arch["state_adapter_rank"])
        )
        self.step_encoder = SinusoidalStepEncoder(
            int(self.arch["step_encoding_dim"]), self.hidden_size
        )
        damping = float(self.arch["damping_initial"])
        self.damping_logit = nn.Parameter(torch.tensor(math.log(damping / (1.0 - damping))))
        aggregate = float(self.arch["aggregate_last_initial"])
        self.aggregate_logit = nn.Parameter(torch.tensor(math.log(aggregate / (1.0 - aggregate))))
        self.sufficiency = StateSufficiencyHeads(
            self.hidden_size,
            int(self.substrate["node_count"]),
            int(self.substrate["checksum_modulus"]),
        )
        self.validate_model_contract()

    @property
    def core(self) -> nn.Module:
        return object.__getattribute__(self, "_core_ref")

    @property
    def text(self) -> nn.Module:
        return object.__getattribute__(self, "_text_ref")

    def validate_model_contract(self) -> None:
        layer_types = list(self.text.config.layer_types)
        expected_layers = int(self.arch["expected_num_layers"])
        if len(self.text.layers) != expected_layers or len(layer_types) != expected_layers:
            raise RuntimeError(
                f"expected exactly {expected_layers} Qwen text layers, got "
                f"{len(self.text.layers)} modules and {len(layer_types)} layer types"
            )
        pattern = list(self.arch["expected_layer_pattern"])
        expected = [pattern[index % len(pattern)] for index in range(expected_layers)]
        if layer_types != expected:
            raise RuntimeError(f"Qwen layer pattern changed: expected {expected}, got {layer_types}")
        if self.core.config.model_type not in {"qwen3_5", "qwen3_5_text"}:
            raise RuntimeError(
                "expected the pinned Qwen3.5 composite/text configuration, got "
                f"{self.core.config.model_type}"
            )

    def _prepare_geometry(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_module

        inputs_embeds = self.text.embed_tokens(input_ids)
        batch, sequence = input_ids.shape
        position_ids = torch.arange(sequence, device=input_ids.device).view(1, 1, -1)
        position_ids = position_ids.expand(4, batch, -1)
        text_position_ids = position_ids[0]
        rotary_position_ids = position_ids[1:]
        mask_kwargs = {
            "config": self.text.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": text_position_ids,
        }
        masks = {
            "full_attention": qwen_module.create_causal_mask(**mask_kwargs),
            "linear_attention": qwen_module.create_recurrent_attention_mask(**mask_kwargs),
        }
        position_embeddings = self.text.rotary_emb(inputs_embeds, rotary_position_ids)
        return inputs_embeds, masks, text_position_ids, position_embeddings

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        start: int,
        end: int,
        *,
        masks: Mapping[str, torch.Tensor],
        text_position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        deltas_enabled: bool,
    ) -> torch.Tensor:
        with self.delta_bank.enabled(deltas_enabled):
            for index in range(start, end):
                hidden_states = self.text.layers[index](
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=masks[self.text.config.layer_types[index]],
                    position_ids=text_position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
        return hidden_states

    def _gather_state(self, hidden: torch.Tensor, state_mask: torch.Tensor) -> torch.Tensor:
        batch, _, width = hidden.shape
        counts = state_mask.sum(dim=1)
        if not torch.all(counts == self.state_slots):
            raise RuntimeError(
                f"each row must contain {self.state_slots} state tokens; got {counts.tolist()}"
            )
        return hidden[state_mask].reshape(batch, self.state_slots, width)

    def _scatter_state(
        self, memory: torch.Tensor, state_mask: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        hidden = memory.clone()
        hidden[state_mask] = state.reshape(-1, state.shape[-1])
        return hidden

    def _semantic_echo(self, state: torch.Tensor) -> torch.Tensor:
        if self.arch["semantic_echo"]["mode"] != "continuous":
            raise RuntimeError("only the identity continuous-state interface is registered")
        return state

    def _state_losses(
        self,
        states: torch.Tensor,
        node_logits: torch.Tensor,
        phase_logits: torch.Tensor,
        checksum_logits: torch.Tensor,
        targets: Mapping[str, torch.Tensor] | None,
        depths: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if targets is None:
            return None, None
        node_loss = F.cross_entropy(
            node_logits.flatten(0, 1), targets["node"].flatten()
        )
        phase_loss = F.cross_entropy(
            phase_logits.flatten(0, 1), targets["phase"].flatten()
        )
        checksum_loss = F.cross_entropy(
            checksum_logits.flatten(0, 1), targets["checksum"].flatten()
        )
        state_loss = (node_loss + phase_loss + checksum_loss) / 3.0
        fixed_point = states.new_zeros(())
        if depths is not None and states.shape[1] > 1:
            deltas = (states[:, 1:] - states[:, :-1]).float().pow(2).mean(dim=(2, 3))
            steps = torch.arange(2, states.shape[1] + 1, device=states.device)[None, :]
            mask = steps > depths[:, None]
            if mask.any():
                fixed_point = deltas[mask].mean().to(dtype=states.dtype)
        return state_loss, fixed_point

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state_mask: torch.Tensor,
        answer_positions: torch.Tensor,
        k: int,
        mode: str,
        answer_targets: torch.Tensor | None = None,
        state_targets: Mapping[str, torch.Tensor] | None = None,
        depths: torch.Tensor | None = None,
        state_override: Mapping[int, torch.Tensor] | None = None,
    ) -> StateLoopOutput:
        if mode not in {"carry", "bag"}:
            raise ValueError("mode must be carry or bag")
        if not 1 <= k <= self.max_recurrence:
            raise ValueError(f"k must be in [1, {self.max_recurrence}]")
        embeds, masks, text_position_ids, position_embeddings = self._prepare_geometry(
            input_ids, attention_mask
        )
        # Prelude and the untouched first R application do not need gradients in
        # recurrent modes; they are a frozen source state.
        with torch.no_grad():
            hidden = self._run_layers(
                embeds,
                0,
                self.loop_start,
                masks=masks,
                text_position_ids=text_position_ids,
                position_embeddings=position_embeddings,
                deltas_enabled=False,
            )
            hidden = self._run_layers(
                hidden,
                self.loop_start,
                self.loop_end,
                masks=masks,
                text_position_ids=text_position_ids,
                position_embeddings=position_embeddings,
                deltas_enabled=False,
            )
        memory = hidden.detach()
        raw_first_state = self._gather_state(memory, state_mask)

        if k == 1:
            states = [raw_first_state]
            aggregated = raw_first_state
        else:
            first_state = self.state_initializer(raw_first_state)
            if state_override and 1 in state_override:
                first_state = state_override[1].to(first_state)
            states = [first_state]
            previous = first_state
            damping = torch.sigmoid(self.damping_logit).to(dtype=first_state.dtype)
            for step in range(2, k + 1):
                source = previous if mode == "carry" else first_state
                step_signal = self.step_encoder(
                    step, device=source.device, dtype=source.dtype
                ).view(1, 1, -1)
                recurrent_input = self._scatter_state(memory, state_mask, source + step_signal)
                candidate_hidden = self._run_layers(
                    recurrent_input,
                    self.loop_start,
                    self.loop_end,
                    masks=masks,
                    text_position_ids=text_position_ids,
                    position_embeddings=position_embeddings,
                    deltas_enabled=True,
                )
                candidate = self._gather_state(candidate_hidden, state_mask)
                updated = source + damping * (candidate - source)
                updated = self._semantic_echo(updated)
                if state_override and step in state_override:
                    updated = state_override[step].to(updated)
                states.append(updated)
                previous = updated
            stacked_for_aggregate = torch.stack(states, dim=1)
            last_weight = torch.sigmoid(self.aggregate_logit).to(dtype=first_state.dtype)
            aggregated = last_weight * stacked_for_aggregate[:, -1] + (
                1.0 - last_weight
            ) * stacked_for_aggregate.mean(dim=1)

        final_hidden = self._scatter_state(memory, state_mask, aggregated)
        final_hidden = self._run_layers(
            final_hidden,
            self.loop_end,
            len(self.text.layers),
            masks=masks,
            text_position_ids=text_position_ids,
            position_embeddings=position_embeddings,
            deltas_enabled=False,
        )
        final_hidden = self.text.norm(final_hidden)
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        answer_hidden = final_hidden[batch_indices, answer_positions]
        answer_logits = self.core.lm_head(answer_hidden)

        stacked_states = torch.stack(states, dim=1)
        node_logits, phase_logits, checksum_logits = self.sufficiency(stacked_states)
        state_loss, fixed_point_loss = self._state_losses(
            stacked_states,
            node_logits,
            phase_logits,
            checksum_logits,
            state_targets,
            depths,
        )
        answer_loss = (
            F.cross_entropy(answer_logits, answer_targets)
            if answer_targets is not None
            else None
        )
        loss = None
        if answer_loss is not None:
            loss = float(self.training_config["answer_loss_weight"]) * answer_loss
            if state_loss is not None:
                loss = loss + float(self.training_config["state_loss_weight"]) * state_loss
            if fixed_point_loss is not None:
                loss = loss + float(
                    self.training_config["fixed_point_loss_weight"]
                ) * fixed_point_loss

        return StateLoopOutput(
            answer_logits=answer_logits,
            loss=loss,
            answer_loss=answer_loss,
            state_loss=state_loss,
            fixed_point_loss=fixed_point_loss,
            states=tuple(states),
            node_logits=node_logits,
            phase_logits=phase_logits,
            checksum_logits=checksum_logits,
            diagnostics={
                "mode": mode,
                "k": k,
                "damping": float(torch.sigmoid(self.damping_logit).detach().cpu()),
                "aggregate_last_weight": float(
                    torch.sigmoid(self.aggregate_logit).detach().cpu()
                ),
                "state_interface": "continuous_identity",
            },
        )

    def extra_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        modules = {
            "state_initializer": self.state_initializer,
            "step_encoder": self.step_encoder,
            "sufficiency": self.sufficiency,
        }
        result = {name: module.state_dict() for name, module in modules.items()}
        result["scalars"] = {
            "damping_logit": self.damping_logit.detach(),
            "aggregate_logit": self.aggregate_logit.detach(),
        }
        return result

    def delta_state_dict(self) -> dict[str, torch.Tensor]:
        return self.delta_bank.state_dict()

    def load_extra_state_dict(self, payload: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        self.state_initializer.load_state_dict(payload["state_initializer"])
        self.step_encoder.load_state_dict(payload["step_encoder"])
        self.sufficiency.load_state_dict(payload["sufficiency"])
        with torch.no_grad():
            self.damping_logit.copy_(payload["scalars"]["damping_logit"])
            self.aggregate_logit.copy_(payload["scalars"]["aggregate_logit"])

    def load_delta_state_dict(self, payload: Mapping[str, torch.Tensor]) -> None:
        self.delta_bank.load_state_dict(payload, strict=True)
