"""Pinned Qwen loading, targeted Jacobian fitting, and cache-free interventions."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from jacobian import swap_coordinates, swap_coordinates_batched


@dataclass(frozen=True)
class TargetedLens:
    concepts: tuple[str, ...]
    token_ids: tuple[int, ...]
    source_layers: tuple[int, ...]
    target_layer: int
    directions: dict[int, torch.Tensor]
    n_prompts: int
    pair_weighting: str = "equal_valid_causal_source_target_pairs"

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "concepts": self.concepts,
            "token_ids": self.token_ids,
            "source_layers": self.source_layers,
            "target_layer": self.target_layer,
            "directions": {layer: value.to(torch.float16).cpu() for layer, value in self.directions.items()},
            "n_prompts": self.n_prompts,
            "pair_weighting": self.pair_weighting,
        }

    @classmethod
    def load(cls, path: str) -> "TargetedLens":
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            concepts=tuple(state["concepts"]),
            token_ids=tuple(int(value) for value in state["token_ids"]),
            source_layers=tuple(int(value) for value in state["source_layers"]),
            target_layer=int(state["target_layer"]),
            directions={int(layer): value.float() for layer, value in state["directions"].items()},
            n_prompts=int(state["n_prompts"]),
            pair_weighting=str(state["pair_weighting"]),
        )


class ActivationRecorder:
    """Capture selected block outputs and root autograd at the earliest source."""

    def __init__(self, layers: Sequence[torch.nn.Module], at: Iterable[int], *, start_graph_at: int | None = None):
        self.layers = layers
        self.indices = sorted(set(at) | ({start_graph_at} if start_graph_at is not None else set()))
        self.start_graph_at = start_graph_at
        self.activations: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _hook(self, index: int):
        def hook(_module, _inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
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


class CoordinatePatcher:
    """Clamp a source->target coordinate swap at selected sequence positions."""

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        layer_directions: dict[int, tuple[torch.Tensor, torch.Tensor]],
        position_mask: torch.Tensor,
        *,
        alpha: float,
    ):
        self.layers = layers
        self.layer_directions = layer_directions
        self.position_mask = position_mask
        self.alpha = alpha
        self.handles: list[Any] = []
        self.delta_norms: list[float] = []

    def _hook(self, layer: int):
        source, target = self.layer_directions[layer]

        def hook(_module, _inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            mask = self.position_mask.to(tensor.device)
            if mask.numel() != tensor.shape[1]:
                raise RuntimeError("patch position mask does not match sequence length")
            if not bool(mask.any()):
                return output
            patched = tensor.clone()
            selected = patched[:, mask, :]
            changed, delta = swap_coordinates(
                selected, source.to(tensor.device), target.to(tensor.device), alpha=self.alpha
            )
            patched[:, mask, :] = changed
            self.delta_norms.append(float(delta.float().norm(dim=-1).mean().detach().cpu()))
            if torch.is_tensor(output):
                return patched
            return (patched,) + tuple(output[1:])

        return hook

    def __enter__(self):
        for layer in sorted(self.layer_directions):
            self.handles.append(self.layers[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class BatchLastCoordinatePatcher:
    """Apply per-example coordinate swaps at the final sequence position."""

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        layer_directions: dict[int, tuple[torch.Tensor, torch.Tensor]],
        *,
        alpha: float,
    ):
        self.layers = layers
        self.layer_directions = layer_directions
        self.alpha = alpha
        self.handles: list[Any] = []
        self.delta_norms: list[float] = []

    def _hook(self, layer: int):
        source, target = self.layer_directions[layer]

        def hook(_module, _inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            patched = tensor.clone()
            changed, delta = swap_coordinates_batched(
                patched[:, -1:, :],
                source.to(tensor.device),
                target.to(tensor.device),
                alpha=self.alpha,
            )
            patched[:, -1:, :] = changed
            self.delta_norms.extend(delta.float().norm(dim=-1)[:, 0].detach().cpu().tolist())
            if torch.is_tensor(output):
                return patched
            return (patched,) + tuple(output[1:])

        return hook

    def __enter__(self):
        for layer in sorted(self.layer_directions):
            self.handles.append(self.layers[layer].register_forward_hook(self._hook(layer)))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class QwenTransportModel:
    def __init__(self, config: dict[str, Any]):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_config = config["model"]
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
        self.norm = self.text_model.norm
        self.lm_head = self.model.lm_head
        self.device = self.lm_head.weight.device
        self.think_open_id = int(model_config["think_open_id"])
        self.think_close_id = int(model_config["think_close_id"])
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = {int(eos)} if isinstance(eos, int) else {int(value) for value in eos}
        text_config = self.model.config.get_text_config()
        self.n_layers = int(text_config.num_hidden_layers)
        self.d_model = int(text_config.hidden_size)
        self.vocab_size = int(text_config.vocab_size)
        self.load_seconds = time.perf_counter() - started
        if self.tokenizer.convert_tokens_to_ids("<think>") != self.think_open_id:
            raise RuntimeError("configured think-open token does not match tokenizer")
        if self.tokenizer.convert_tokens_to_ids("</think>") != self.think_close_id:
            raise RuntimeError("configured think-close token does not match tokenizer")

    def render(self, user: str, *, enable_thinking: bool) -> str:
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
        if bos is not None and ids.shape[1] > 0 and int(ids[0, 0]) != int(bos):
            ids = torch.cat([torch.tensor([[bos]], dtype=ids.dtype), ids], dim=1)
            ids = ids[:, -max_length:]
        return ids.to(self.device)

    def concept_token_id(self, concept: str) -> int:
        ids = self.tokenizer(" " + concept, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"concept {concept!r} is not one leading-space token: {ids}")
        return int(ids[0])

    def bare_token_id(self, text: str) -> int:
        """Token after a fixed prefix that already ends in a literal space."""
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"answer {text!r} is not one bare token: {ids}")
        return int(ids[0])

    def audit_concepts(self, concepts: Sequence[str]) -> dict[str, int]:
        return {concept: self.concept_token_id(concept) for concept in concepts}

    def _forward_text(self, input_ids: torch.Tensor) -> Any:
        return self.text_model(input_ids=input_ids, use_cache=False)

    def fit_targeted_lens(
        self,
        prompts: Sequence[str],
        concepts: Sequence[str],
        *,
        source_layers: Sequence[int],
        target_layer: int,
        concept_batch: int,
        max_sequence_tokens: int,
        skip_first_positions: int,
    ) -> tuple[TargetedLens, list[dict[str, Any]]]:
        if max(source_layers) >= target_layer:
            raise ValueError("source layers must precede target layer")
        token_ids = tuple(self.concept_token_id(concept) for concept in concepts)
        sums = {
            layer: torch.zeros(len(concepts), self.d_model, dtype=torch.float32)
            for layer in source_layers
        }
        prompt_receipts = []
        for prompt_index, prompt in enumerate(prompts):
            input_ids = self.encode(prompt, max_length=max_sequence_tokens)
            sequence_length = int(input_ids.shape[1])
            valid = torch.arange(skip_first_positions, sequence_length - 1, device=self.device)
            if valid.numel() == 0:
                raise ValueError(f"lens prompt {prompt_index} is too short ({sequence_length})")
            pair_count = int(valid.numel() * (valid.numel() + 1) // 2)
            for start in range(0, len(concepts), concept_batch):
                stop = min(len(concepts), start + concept_batch)
                batch_ids = input_ids.expand(stop - start, -1)
                with ActivationRecorder(
                    self.layers,
                    at=[*source_layers, target_layer],
                    start_graph_at=min(source_layers),
                ) as recorder, torch.enable_grad():
                    self._forward_text(batch_ids)
                    target = recorder.activations[target_layer]
                    sources = [recorder.activations[layer] for layer in source_layers]
                    cotangent = torch.zeros_like(target)
                    covectors = self.lm_head.weight[list(token_ids[start:stop])].to(target.dtype)
                    cotangent[:, valid, :] = covectors[:, None, :]
                    gradients = torch.autograd.grad(
                        outputs=target,
                        inputs=sources,
                        grad_outputs=cotangent,
                        retain_graph=False,
                    )
                    for layer, gradient in zip(source_layers, gradients, strict=True):
                        # Each source gradient already sums over all causally reachable
                        # target positions. Summing sources and dividing by the number
                        # of valid causal pairs gives equal pair weighting.
                        directions = gradient[:, valid, :].float().sum(dim=1) / pair_count
                        sums[layer][start:stop] += directions.detach().cpu()
                    del gradients
            prompt_receipts.append({
                "prompt_index": prompt_index,
                "sequence_tokens": sequence_length,
                "valid_positions": int(valid.numel()),
                "causal_pairs": pair_count,
            })
        directions = {layer: value / len(prompts) for layer, value in sums.items()}
        lens = TargetedLens(
            concepts=tuple(concepts),
            token_ids=token_ids,
            source_layers=tuple(source_layers),
            target_layer=target_layer,
            directions=directions,
            n_prompts=len(prompts),
        )
        return lens, prompt_receipts

    @torch.no_grad()
    def score_next_token_batch(
        self,
        rendered_prefixes: Sequence[str],
        *,
        layer_directions: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        alpha: float = 1.0,
    ) -> dict[str, Any]:
        sequences = [self.encode(prefix, max_length=1024)[0] for prefix in rendered_prefixes]
        lengths = {int(sequence.numel()) for sequence in sequences}
        if len(lengths) != 1:
            raise ValueError(f"positive-control batch has unequal token lengths: {sorted(lengths)}")
        input_ids = torch.stack(sequences)
        if layer_directions:
            patcher: contextlib.AbstractContextManager = BatchLastCoordinatePatcher(
                self.layers, layer_directions, alpha=alpha
            )
        else:
            patcher = contextlib.nullcontext()
        with patcher:
            output = self.model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
            logits = output.logits[:, -1, :].float()
        top_ids = torch.argmax(logits, dim=-1)
        result = {
            "logits": logits.cpu(),
            "top_ids": top_ids.cpu(),
            "sequence_tokens": next(iter(lengths)),
            "forward_tokens": int(input_ids.numel()),
            "delta_norms": (
                list(patcher.delta_norms) if isinstance(patcher, BatchLastCoordinatePatcher) else [0.0] * len(sequences)
            ),
        }
        return result

    def think_position_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Mask positions from the last think-open through, stopping before close."""
        if token_ids.ndim != 1:
            raise ValueError("think mask expects one sequence")
        values = token_ids.tolist()
        opens = [index for index, value in enumerate(values) if value == self.think_open_id]
        mask = torch.zeros(len(values), dtype=torch.bool)
        if not opens:
            return mask
        start = opens[-1]
        closes = [index for index in range(start + 1, len(values)) if values[index] == self.think_close_id]
        stop = closes[0] if closes else len(values)
        mask[start:stop] = True
        return mask

    @torch.no_grad()
    def generate_full_recompute(
        self,
        rendered_prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        layer_directions: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
        alpha: float = 1.0,
    ) -> dict[str, Any]:
        ids = self.encode(rendered_prompt, max_length=16_384)[0]
        prompt_tokens = int(ids.numel())
        generator = torch.Generator(device=self.device).manual_seed(seed)
        forward_tokens = 0
        delta_norms: list[float] = []
        for _step in range(max_new_tokens):
            sequence = ids[None, :]
            forward_tokens += int(sequence.numel())
            patcher: contextlib.AbstractContextManager
            if layer_directions:
                patcher = CoordinatePatcher(
                    self.layers,
                    layer_directions,
                    self.think_position_mask(ids),
                    alpha=alpha,
                )
            else:
                patcher = contextlib.nullcontext()
            with patcher:
                output = self.model(input_ids=sequence, use_cache=False, logits_to_keep=1)
                logits = output.logits[0, -1].float()
            if isinstance(patcher, CoordinatePatcher):
                delta_norms.extend(patcher.delta_norms)
            if do_sample:
                logits = logits / temperature
                if top_k > 0:
                    threshold = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                    logits = logits.masked_fill(logits < threshold, -torch.inf)
                probabilities = torch.softmax(logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumulative - sorted_probs > top_p
                    sorted_probs = sorted_probs.masked_fill(remove, 0)
                    sorted_probs = sorted_probs / sorted_probs.sum()
                    choice = torch.multinomial(sorted_probs, 1, generator=generator)
                    next_id = sorted_indices[choice]
                else:
                    next_id = torch.multinomial(probabilities, 1, generator=generator)
            else:
                next_id = torch.argmax(logits)[None]
            ids = torch.cat([ids, next_id.to(ids.device)])
            if int(next_id.item()) in self.eos_ids:
                break
        generated = ids[prompt_tokens:].tolist()
        return {
            "token_ids": generated,
            "text": self.tokenizer.decode(generated, skip_special_tokens=False),
            "prompt_tokens": prompt_tokens,
            "sampled_tokens": len(generated),
            "forward_tokens": forward_tokens,
            "mean_patch_delta_norm": (
                sum(delta_norms) / len(delta_norms) if delta_norms else 0.0
            ),
        }
