"""Qwen3.5 carried-state recurrence with matched extra-call adaptation.

This module intentionally requires the pinned Transformers training environment.
It does not provide a fallback architecture: a failed Qwen contract is a failed
experiment setup, not permission to substitute another model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptation import AdaptationBank


@dataclass
class StateLoopOutput:
    answer_logits: torch.Tensor | None
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


class StateLoopModel(nn.Module):
    """Manual Qwen text forward with a recurrent middle-block state bottleneck.

    The first P->R->C path has every adaptation delta disabled.  For K=1 it is
    therefore algebraically the original Qwen forward over the identical token
    sequence.  For K>1, only state-slot activations cross repeated R calls;
    non-state token activations are reset to their first-pass values.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: Mapping[str, Any],
        target_names: list[str],
        *,
        capacity: str,
        model_seed: int,
        shared_state: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.experiment_config = config
        self.arch = config["architecture"]
        self.substrate = config["substrate"]
        self.training_config = config["training"]
        self.capacity = capacity
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
        self.adaptation = AdaptationBank(
            base_model,
            target_names,
            capacity=capacity,
            model_seed=model_seed,
            config=self.arch["adaptation"],
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
        self.load_extra_state_dict(shared_state)
        self.validate_model_contract()

    @property
    def delta_bank(self) -> AdaptationBank:
        """Compatibility alias used by the inherited runner during migration."""
        return self.adaptation

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
        with self.adaptation.enabled(deltas_enabled):
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
        compute_answer: bool,
        answer_targets: torch.Tensor | None = None,
        state_targets: Mapping[str, torch.Tensor] | None = None,
        depths: torch.Tensor | None = None,
        state_override: Mapping[int, torch.Tensor] | None = None,
    ) -> StateLoopOutput:
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
                source = previous
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

        answer_logits = None
        answer_loss = None
        if compute_answer:
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
            if answer_targets is not None:
                answer_loss = F.cross_entropy(answer_logits, answer_targets)

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
        return StateLoopOutput(
            answer_logits=answer_logits,
            # Objective composition is deliberately external so state-only
            # training omits the answer graph term rather than multiplying it
            # by zero inside this shared forward.
            loss=None,
            answer_loss=answer_loss,
            state_loss=state_loss,
            fixed_point_loss=fixed_point_loss,
            states=tuple(states),
            node_logits=node_logits,
            phase_logits=phase_logits,
            checksum_logits=checksum_logits,
            diagnostics={
                "k": k,
                "compute_answer": compute_answer,
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
        return self.adaptation.state_dict()

    def load_extra_state_dict(self, payload: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        self.state_initializer.load_state_dict(payload["state_initializer"])
        self.step_encoder.load_state_dict(payload["step_encoder"])
        self.sufficiency.load_state_dict(payload["sufficiency"])
        with torch.no_grad():
            self.damping_logit.copy_(payload["scalars"]["damping_logit"])
            self.aggregate_logit.copy_(payload["scalars"]["aggregate_logit"])

    def load_delta_state_dict(self, payload: Mapping[str, torch.Tensor]) -> None:
        self.adaptation.load_state_dict(payload, strict=True)
