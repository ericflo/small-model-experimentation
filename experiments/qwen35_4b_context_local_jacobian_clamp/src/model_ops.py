"""Pinned Qwen loading, context-local lens fitting, and cache-free patching."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from coordinates import read_coordinates, replace_coordinates


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
        self.desired_by_layer = desired_by_layer
        self.handles: list[Any] = []
        self.deltas: dict[int, torch.Tensor] = {}

    def _hook(self, layer: int):
        desired = self.desired_by_layer[layer]

        def hook(_module, _inputs, output):
            tensor = _tensor_from_output(output)
            if tensor.shape[0] != 1 or not 0 <= self.position < tensor.shape[1]:
                raise RuntimeError("scientific patching requires an in-range batch-one position")
            patched = tensor.clone()
            current = patched[:, self.position, :]
            target = desired.to(device=tensor.device, dtype=tensor.dtype).reshape_as(current)
            before = current.float()
            patched[:, self.position, :] = target
            self.deltas[layer] = (patched[:, self.position, :].float() - before).detach().cpu()
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
        self.directions_by_layer = directions_by_layer
        self.desired_by_layer = desired_by_layer
        self.rtol = float(rtol)
        self.handles: list[Any] = []
        self.deltas: dict[int, torch.Tensor] = {}

    def _hook(self, layer: int):
        directions = self.directions_by_layer[layer]
        desired = self.desired_by_layer[layer]

        def hook(_module, _inputs, output):
            tensor = _tensor_from_output(output)
            if tensor.shape[0] != 1 or not 0 <= self.position < tensor.shape[1]:
                raise RuntimeError("scientific patching requires an in-range batch-one position")
            patched = tensor.clone()
            before = patched[:, self.position, :].float()
            changed, delta = replace_coordinates(
                patched[:, self.position, :],
                directions.to(tensor.device),
                desired.to(tensor.device).reshape(1, -1),
                rtol=self.rtol,
            )
            patched[:, self.position, :] = changed
            del delta
            self.deltas[layer] = (patched[:, self.position, :].float() - before).detach().cpu()
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
        self.deltas_by_layer = deltas_by_layer
        self.handles: list[Any] = []
        self.deltas: dict[int, torch.Tensor] = {}

    def _hook(self, layer: int):
        fixed_delta = self.deltas_by_layer[layer]

        def hook(_module, _inputs, output):
            tensor = _tensor_from_output(output)
            if tensor.shape[0] != 1 or not 0 <= self.position < tensor.shape[1]:
                raise RuntimeError("scientific patching requires an in-range batch-one position")
            patched = tensor.clone()
            before = patched[:, self.position, :].float()
            delta = fixed_delta.to(device=tensor.device, dtype=tensor.dtype).reshape(1, -1)
            patched[:, self.position, :] += delta
            self.deltas[layer] = (patched[:, self.position, :].float() - before).detach().cpu()
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

    def render(self, user: str) -> str:
        messages = [
            {"role": "system", "content": "Follow the requested output format exactly."},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
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
    ) -> dict[str, Any]:
        with ActivationRecorder(self.layers, at=layers) as recorder:
            output = self.model(
                input_ids=prepared["input_ids"], use_cache=False, logits_to_keep=1
            )
        position = int(prepared["position"])
        return {
            "logits": output.logits[0, -1].float().detach().cpu(),
            "activations": {
                layer: recorder.activations[layer][0, position].float().detach().cpu()
                for layer in layers
            },
            "position": position,
            "sequence_tokens": int(prepared["sequence_tokens"]),
        }

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
