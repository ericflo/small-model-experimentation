"""Pinned native traces, close-only replay, and constrained commit-slot readout."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

def _tensor_from_output(output: Any) -> torch.Tensor:
    return output if torch.is_tensor(output) else output[0]


@dataclass(frozen=True)
class ContextLens:
    concepts: tuple[str, ...]
    token_ids: tuple[int, ...]
    source_layers: tuple[int, ...]
    directions: dict[int, torch.Tensor]
    n_prompts: int
    estimator: str

    @classmethod
    def load(cls, path: str) -> "ContextLens":
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            concepts=tuple(str(value) for value in state["concepts"]),
            token_ids=tuple(int(value) for value in state["token_ids"]),
            source_layers=tuple(int(value) for value in state["source_layers"]),
            directions={
                int(layer): value.float()
                for layer, value in state["directions"].items()
            },
            n_prompts=int(state["n_prompts"]),
            estimator=str(state["estimator"]),
        )


class ActivationRecorder:
    def __init__(self, layers: Sequence[torch.nn.Module], at: Sequence[int]):
        self.layers = layers
        self.indices = tuple(sorted(set(int(value) for value in at)))
        self.activations: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _hook(self, index: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            self.activations[index] = _tensor_from_output(output)

        return hook

    def __enter__(self):
        for index in self.indices:
            self.handles.append(
                self.layers[index].register_forward_hook(self._hook(index))
            )
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _with_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    return (tensor, *output[1:])


class FixedBranchPatcher:
    """Apply one precomputed branch delta per batch row at one sequence token."""

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        *,
        position: int,
        branches_by_layer: dict[int, torch.Tensor],
    ) -> None:
        self.layers = layers
        self.position = int(position)
        self.branches_by_layer = branches_by_layer
        widths = {int(value.shape[1]) for value in branches_by_layer.values()}
        if len(widths) != 1:
            raise ValueError("all branch layers must have one common width")
        self.width = next(iter(widths))
        self.handles: list[Any] = []
        self.requested: dict[int, torch.Tensor] = {}
        self.realized: dict[int, torch.Tensor] = {}
        self.input_activations: dict[int, torch.Tensor] = {}
        self.applications: dict[int, int] = {layer: 0 for layer in branches_by_layer}

    def _hook(self, layer: int):
        branches = self.branches_by_layer[layer]

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            tensor = _tensor_from_output(output)
            if tensor.shape[0] != self.width:
                raise RuntimeError("branch batch width changed")
            if not 0 <= self.position < tensor.shape[1]:
                raise RuntimeError("branch patch position is outside sequence")
            if self.applications[layer] != 0:
                raise RuntimeError("branch patch repeated at one layer")
            patched = tensor.clone()
            current = patched[:, self.position, :]
            current_float = current.float().clone()
            requested = branches.T.to(device=tensor.device, dtype=torch.float32)
            changed = (current_float + requested).to(tensor.dtype)
            patched[:, self.position, :] = changed
            self.input_activations[layer] = current_float.detach().cpu()
            self.requested[layer] = requested.detach().cpu()
            self.realized[layer] = (changed.float() - current_float).detach().cpu()
            self.applications[layer] += 1
            return _with_tensor(output, patched)

        return hook

    def __enter__(self):
        for layer in sorted(self.branches_by_layer):
            self.handles.append(
                self.layers[layer].register_forward_hook(self._hook(layer))
            )
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


class QuantizationAwareFixedNonJPatcher:
    """Repair fixed non-J branches against paired realized J norms after bf16."""

    def __init__(
        self,
        layers: Sequence[torch.nn.Module],
        *,
        position: int,
        branches_by_layer: dict[int, torch.Tensor],
        directions_by_layer: dict[int, torch.Tensor],
        target_norms_by_layer: dict[int, torch.Tensor],
        rtol: float,
        norm_tolerance: float,
        projection_tolerance: float,
        correction_iterations: int,
        correction_damping: float,
    ) -> None:
        keys = set(branches_by_layer)
        if keys != set(directions_by_layer) or keys != set(target_norms_by_layer):
            raise ValueError("quantized control layers differ")
        self.layers = layers
        self.position = int(position)
        self.branches_by_layer = branches_by_layer
        self.target_norms_by_layer = target_norms_by_layer
        self.norm_tolerance = float(norm_tolerance)
        self.projection_tolerance = float(projection_tolerance)
        self.correction_iterations = int(correction_iterations)
        self.correction_damping = float(correction_damping)
        self.geometry = {}
        for layer in keys:
            directions = directions_by_layer[layer].float()
            norms = directions.norm(dim=0, keepdim=True)
            dictionary = directions / norms
            self.geometry[layer] = (
                dictionary,
                torch.linalg.pinv(dictionary, rtol=float(rtol)),
            )
        widths = {int(value.shape[1]) for value in branches_by_layer.values()}
        if len(widths) != 1:
            raise ValueError("all control layers must have one width")
        self.width = next(iter(widths))
        self.handles: list[Any] = []
        self.requested: dict[int, torch.Tensor] = {}
        self.realized: dict[int, torch.Tensor] = {}
        self.input_activations: dict[int, torch.Tensor] = {}
        self.applications: dict[int, int] = {layer: 0 for layer in keys}
        self.norm_errors: dict[int, torch.Tensor] = {}
        self.projection_fractions: dict[int, torch.Tensor] = {}
        self.passed_rows: dict[int, torch.Tensor] = {}
        self.iterations_used: dict[int, int] = {}

    @staticmethod
    def _actual(current: torch.Tensor, requested: torch.Tensor) -> torch.Tensor:
        return (current + requested).to(torch.bfloat16).float() - current

    def _hook(self, layer: int):
        branches = self.branches_by_layer[layer]
        targets_cpu = self.target_norms_by_layer[layer]
        dictionary_cpu, inverse_cpu = self.geometry[layer]

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            tensor = _tensor_from_output(output)
            if tensor.shape[0] != self.width or not 0 <= self.position < tensor.shape[1]:
                raise RuntimeError("quantized control batch/position changed")
            if self.applications[layer] != 0:
                raise RuntimeError("quantized control repeated at one layer")
            patched = tensor.clone()
            current = patched[:, self.position, :].float().clone()
            requests = branches.T.to(device=tensor.device, dtype=torch.float32).clone()
            targets = targets_cpu.to(device=tensor.device, dtype=torch.float32)
            dictionary = dictionary_cpu.to(tensor.device)
            inverse = inverse_cpu.to(tensor.device)
            best_objective = torch.full((self.width,), float("inf"), device=tensor.device)
            best_actual = torch.zeros_like(requests)
            best_request = requests.clone()
            best_error = torch.full_like(best_objective, float("inf"))
            best_projection = torch.full_like(best_objective, float("inf"))
            used = 0
            for step in range(self.correction_iterations + 1):
                actual = self._actual(current, requests)
                norms = actual.norm(dim=-1)
                errors = (norms - targets).abs() / targets.clamp_min(1e-12)
                projection = (actual @ inverse.T) @ dictionary.T
                fractions = projection.norm(dim=-1) / norms.clamp_min(1e-12)
                objective = torch.maximum(
                    errors / self.norm_tolerance,
                    fractions / self.projection_tolerance,
                )
                better = objective < best_objective
                best_objective = torch.where(better, objective, best_objective)
                best_actual = torch.where(better[:, None], actual, best_actual)
                best_request = torch.where(better[:, None], requests, best_request)
                best_error = torch.where(better, errors, best_error)
                best_projection = torch.where(better, fractions, best_projection)
                passed = (best_error <= self.norm_tolerance) & (
                    best_projection <= self.projection_tolerance
                )
                used = step
                if bool(passed.all()) or step == self.correction_iterations:
                    break
                orthogonal = actual - projection
                desired = orthogonal * (
                    targets[:, None] / orthogonal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                )
                updated = requests + self.correction_damping * (desired - actual)
                requests = torch.where(passed[:, None], best_request, updated)
            changed = (current + best_actual).to(tensor.dtype)
            patched[:, self.position, :] = changed
            realized = changed.float() - current
            norms = realized.norm(dim=-1)
            projection = (realized @ inverse.T) @ dictionary.T
            norm_errors = (norms - targets).abs() / targets.clamp_min(1e-12)
            fractions = projection.norm(dim=-1) / norms.clamp_min(1e-12)
            self.input_activations[layer] = current.detach().cpu()
            self.requested[layer] = best_request.detach().cpu()
            self.realized[layer] = realized.detach().cpu()
            self.norm_errors[layer] = norm_errors.detach().cpu()
            self.projection_fractions[layer] = fractions.detach().cpu()
            self.passed_rows[layer] = (
                (norm_errors <= self.norm_tolerance)
                & (fractions <= self.projection_tolerance)
            ).detach().cpu()
            self.iterations_used[layer] = used
            self.applications[layer] += 1
            return _with_tensor(output, patched)

        return hook

    def __enter__(self):
        for layer in sorted(self.branches_by_layer):
            self.handles.append(
                self.layers[layer].register_forward_hook(self._hook(layer))
            )
        return self

    def __exit__(self, *_exc: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


class _TraceStopper:
    def __init__(
        self,
        *,
        prompt_tokens: int,
        close_id: int,
        eos_id: int,
        thought_cap: int,
        answer_cap: int,
    ) -> None:
        self.prompt_tokens = int(prompt_tokens)
        self.close_id = int(close_id)
        self.eos_id = int(eos_id)
        self.thought_cap = int(thought_cap)
        self.answer_cap = int(answer_cap)

    def __call__(self, input_ids: torch.Tensor, _scores: torch.Tensor, **_kwargs: Any) -> torch.BoolTensor:
        if input_ids.shape[0] != 1:
            raise RuntimeError("frozen generation requires batch one")
        generated = input_ids[0, self.prompt_tokens :]
        close = (generated == self.close_id).nonzero(as_tuple=False).flatten()
        if close.numel() == 0:
            stop = generated.numel() >= self.thought_cap
        else:
            answer_tokens = generated.numel() - int(close[0]) - 1
            stop = answer_tokens >= self.answer_cap or int(generated[-1]) == self.eos_id
        return torch.tensor([stop], device=input_ids.device, dtype=torch.bool)


class QwenCommitModel:
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
        self.layers = self.model.model.layers
        self.device = self.model.lm_head.weight.device
        text_config = self.model.config.get_text_config()
        self.n_layers = int(text_config.num_hidden_layers)
        self.d_model = int(text_config.hidden_size)
        self.vocab_size = int(text_config.vocab_size)
        self.think_open_id = int(model_config["think_open_id"])
        self.think_close_id = int(model_config["think_close_id"])
        self.eos_id = int(text_config.eos_token_id)
        observed = (
            self.tokenizer.convert_tokens_to_ids("<think>"),
            self.tokenizer.convert_tokens_to_ids("</think>"),
        )
        if observed != (self.think_open_id, self.think_close_id):
            raise RuntimeError(f"native-thinking token IDs changed: {observed}")
        self.load_seconds = time.perf_counter() - started

    def render(self, user: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Follow the requested output format exactly."},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def prepare(self, user: str, *, prompt_max_tokens: int) -> dict[str, Any]:
        rendered = self.render(user)
        ids = self.tokenizer(rendered, return_tensors="pt").input_ids
        bos = self.tokenizer.bos_token_id
        if bos is not None and ids.shape[1] and int(ids[0, 0]) != int(bos):
            ids = torch.cat([torch.tensor([[bos]], dtype=ids.dtype), ids], dim=1)
        if ids.shape[1] > int(prompt_max_tokens):
            raise RuntimeError(f"prompt has {ids.shape[1]} tokens above cap {prompt_max_tokens}")
        opens = (ids[0] == self.think_open_id).nonzero(as_tuple=False).flatten()
        if opens.numel() != 1 or (ids[0] == self.think_close_id).any():
            raise RuntimeError("prompt must contain one open and no close token")
        return {
            "rendered": rendered,
            "input_ids": ids.to(self.device),
            "prompt_tokens": int(ids.shape[1]),
            "think_open_position": int(opens[0]),
        }

    def leading_space_token_id(self, text: str) -> int:
        ids = self.tokenizer(" " + text, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"alias {text!r} is not one leading-space token: {ids}")
        return int(ids[0])

    def slot_token_ids(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if not ids or self.think_open_id in ids or self.think_close_id in ids:
            raise ValueError(f"invalid fixed slot text tokenization: {ids}")
        return [int(value) for value in ids]

    def _audit_hook(self, lengths: list[int]):
        def hook(_module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            ids = kwargs.get("input_ids")
            if ids is not None:
                lengths.append(int(ids.shape[1]))

        return hook

    def _seed_context(self, seed: int):
        devices = [self.device.index] if self.device.type == "cuda" else []
        return torch.random.fork_rng(devices=devices)

    @torch.no_grad()
    def generate_trace(
        self,
        input_ids: torch.Tensor,
        *,
        seed: int,
        thought_cap: int,
        answer_cap: int,
        total_max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> dict[str, Any]:
        from transformers import StoppingCriteriaList

        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens + thought_cap + answer_cap > total_max_tokens:
            raise RuntimeError("trace allowance exceeds total context cap")
        stopper = _TraceStopper(
            prompt_tokens=prompt_tokens,
            close_id=self.think_close_id,
            eos_id=self.eos_id,
            thought_cap=thought_cap,
            answer_cap=answer_cap,
        )
        lengths: list[int] = []
        handle = self.model.register_forward_pre_hook(self._audit_hook(lengths), with_kwargs=True)
        started = time.perf_counter()
        try:
            with self._seed_context(seed):
                torch.manual_seed(int(seed))
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(int(seed))
                output = self.model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    max_new_tokens=int(thought_cap + answer_cap),
                    stopping_criteria=StoppingCriteriaList([stopper]),
                    eos_token_id=self.eos_id,
                    pad_token_id=self.eos_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
        finally:
            handle.remove()
        generated = output.sequences[0, prompt_tokens:]
        closes = (generated == self.think_close_id).nonzero(as_tuple=False).flatten()
        natural_close = closes.numel() == 1
        close_index = int(closes[0]) if natural_close else None
        close_step = close_index + 1 if close_index is not None else None
        answer_ids = generated[close_index + 1 :] if close_index is not None else generated[:0]
        return {
            "generated_token_ids": generated.tolist(),
            "generated_text": self.tokenizer.decode(generated.tolist(), skip_special_tokens=False),
            "answer_text": self.tokenizer.decode(answer_ids.tolist(), skip_special_tokens=False),
            "natural_close": natural_close,
            "close_step": close_step,
            "think_tokens": close_index if close_index is not None else int(generated.numel()),
            "answer_tokens": int(answer_ids.numel()),
            "stopped_by": (
                "eos" if natural_close and answer_ids.numel() and int(answer_ids[-1]) == self.eos_id
                else "answer_cap" if natural_close
                else "eos_before_close" if generated.numel() and int(generated[-1]) == self.eos_id
                else "think_cap_without_close"
            ),
            "forward_input_lengths": lengths,
            "cache_contract_pass": bool(
                lengths and lengths[0] == prompt_tokens and all(length == 1 for length in lengths[1:])
            ),
            "forward_calls": len(lengths),
            "elapsed_seconds": time.perf_counter() - started,
        }

    @torch.no_grad()
    def force_commit(
        self,
        prompt_ids: torch.Tensor,
        thought_token_ids: list[int],
        *,
        seed: int,
        answer_cap: int,
        total_max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> dict[str, Any]:
        if self.think_close_id in thought_token_ids or self.eos_id in thought_token_ids:
            raise RuntimeError("forced prefix contains a natural close or EOS")
        suffix = torch.tensor(
            [thought_token_ids + [self.think_close_id]],
            device=self.device,
            dtype=prompt_ids.dtype,
        )
        forced_ids = torch.cat([prompt_ids.to(self.device), suffix], dim=1)
        prefill_tokens = int(forced_ids.shape[1])
        if prefill_tokens + answer_cap > total_max_tokens:
            raise RuntimeError("forced answer allowance exceeds total context cap")
        lengths: list[int] = []
        handle = self.model.register_forward_pre_hook(self._audit_hook(lengths), with_kwargs=True)
        started = time.perf_counter()
        try:
            with self._seed_context(seed):
                torch.manual_seed(int(seed))
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(int(seed))
                output = self.model.generate(
                    input_ids=forced_ids,
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    max_new_tokens=int(answer_cap),
                    eos_token_id=self.eos_id,
                    pad_token_id=self.eos_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
        finally:
            handle.remove()
        answer_ids = output.sequences[0, prefill_tokens:]
        return {
            "answer_token_ids": answer_ids.tolist(),
            "answer_text": self.tokenizer.decode(answer_ids.tolist(), skip_special_tokens=False),
            "answer_tokens": int(answer_ids.numel()),
            "stopped_by": (
                "eos" if answer_ids.numel() and int(answer_ids[-1]) == self.eos_id else "answer_cap"
            ),
            "forward_input_lengths": lengths,
            "cache_contract_pass": bool(
                lengths and lengths[0] == prefill_tokens and all(length == 1 for length in lengths[1:])
            ),
            "forward_calls": len(lengths),
            "elapsed_seconds": time.perf_counter() - started,
            "forced_close": True,
            "counterfactual_to_natural_close": True,
        }

    @torch.no_grad()
    def slot_readout(
        self,
        prompt_ids: torch.Tensor,
        thought_token_ids: list[int],
        *,
        slot_text: str,
        aliases: list[str],
        total_max_tokens: int,
    ) -> dict[str, Any]:
        if self.think_close_id in thought_token_ids or self.eos_id in thought_token_ids:
            raise RuntimeError("slot prefix contains a natural close or EOS")
        slot_ids = self.slot_token_ids(slot_text)
        suffix_ids = thought_token_ids + [self.think_close_id] + slot_ids
        suffix = torch.tensor([suffix_ids], device=self.device, dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids.to(self.device), suffix], dim=1)
        if full_ids.shape[1] + 1 > int(total_max_tokens):
            raise RuntimeError("slot readout exceeds total context cap")
        output = self.model(input_ids=full_ids, use_cache=False, logits_to_keep=1)
        logits = output.logits[0, -1].float()
        alias_ids = [self.leading_space_token_id(alias) for alias in aliases]
        constrained_logits = logits[alias_ids]
        probabilities = torch.softmax(constrained_logits, dim=-1)
        full_log_normalizer = torch.logsumexp(logits, dim=-1)
        alias_full_probabilities = torch.exp(constrained_logits - full_log_normalizer)
        order = torch.argsort(constrained_logits, descending=True)
        choice_index = int(order[0])
        margin = float(constrained_logits[order[0]] - constrained_logits[order[1]])
        entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
        full_top_id = int(torch.argmax(logits))
        return {
            "slot_text": slot_text,
            "slot_token_ids": slot_ids,
            "prefill_tokens": int(full_ids.shape[1]),
            "thought_tokens": len(thought_token_ids),
            "forced_close": True,
            "counterfactual_to_natural_close": True,
            "constrained_to_alias_tokens": True,
            "alias_token_ids": alias_ids,
            "alias_probabilities": {
                alias: float(probabilities[index]) for index, alias in enumerate(aliases)
            },
            "alias_full_vocab_probabilities": {
                alias: float(alias_full_probabilities[index])
                for index, alias in enumerate(aliases)
            },
            "full_vocab_alias_probability_mass": float(alias_full_probabilities.sum()),
            "chosen_alias": aliases[choice_index],
            "chosen_alias_token_id": alias_ids[choice_index],
            "constrained_margin": margin,
            "constrained_entropy": entropy,
            "full_vocab_top_token_id": full_top_id,
            "full_vocab_top_is_alias": full_top_id in set(alias_ids),
            "finite": bool(
                torch.isfinite(constrained_logits).all()
                and torch.isfinite(probabilities).all()
                and torch.isfinite(alias_full_probabilities).all()
            ),
        }

    @torch.no_grad()
    def capture_thought_prefix(
        self,
        prompt_ids: torch.Tensor,
        thought_token_ids: list[int],
        *,
        layers: Sequence[int],
        coordinate_inverses_by_layer: dict[int, torch.Tensor],
        total_max_tokens: int,
    ) -> dict[str, Any]:
        """Read J coordinates at the final token of a live thought prefix.

        The input contains no close token, slot, answer, or future thought.
        """
        if not thought_token_ids:
            raise RuntimeError("coordinate capture requires a nonempty thought")
        if self.think_close_id in thought_token_ids or self.eos_id in thought_token_ids:
            raise RuntimeError("coordinate prefix contains close or EOS")
        layer_tuple = tuple(int(layer) for layer in layers)
        if set(layer_tuple) != set(coordinate_inverses_by_layer):
            raise RuntimeError("capture layers and coordinate inverses differ")
        suffix = torch.tensor(
            [thought_token_ids], device=self.device, dtype=prompt_ids.dtype
        )
        full_ids = torch.cat([prompt_ids.to(self.device), suffix], dim=1)
        if int(full_ids.shape[1]) > int(total_max_tokens):
            raise RuntimeError("coordinate prefix exceeds total context cap")
        position = int(full_ids.shape[1]) - 1
        with ActivationRecorder(self.layers, at=layer_tuple) as recorder:
            self.model(input_ids=full_ids, use_cache=False, logits_to_keep=1)
        activations = {
            layer: recorder.activations[layer][0, position].float().detach().cpu()
            for layer in layer_tuple
        }
        coordinates = {
            layer: (
                activations[layer].reshape(1, -1).float()
                @ coordinate_inverses_by_layer[layer].float().T
            )[0].detach().cpu()
            for layer in layer_tuple
        }
        return {
            "sequence_tokens": int(full_ids.shape[1]),
            "position": position,
            "thought_tokens": len(thought_token_ids),
            "layers": list(layer_tuple),
            "activations": activations,
            "coordinates": coordinates,
            "finite": bool(
                all(torch.isfinite(value).all() for value in activations.values())
                and all(torch.isfinite(value).all() for value in coordinates.values())
            ),
            "close_present": bool((full_ids == self.think_close_id).any()),
            "slot_present": False,
        }

    @torch.no_grad()
    def branched_slot_readout(
        self,
        prompt_ids: torch.Tensor,
        thought_token_ids: list[int],
        *,
        slot_text: str,
        aliases: list[str],
        branches_by_layer: dict[int, torch.Tensor],
        total_max_tokens: int,
        quantization_control: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Patch the final live-thought token and read every branch at the slot."""

        if not thought_token_ids:
            raise RuntimeError("branching requires a nonempty thought prefix")
        if self.think_close_id in thought_token_ids or self.eos_id in thought_token_ids:
            raise RuntimeError("branch prefix contains close or EOS")
        widths = {int(value.shape[1]) for value in branches_by_layer.values()}
        if widths != {len(aliases)}:
            raise RuntimeError("branch width must equal public alias count")
        slot_ids = self.slot_token_ids(slot_text)
        suffix_ids = thought_token_ids + [self.think_close_id] + slot_ids
        suffix = torch.tensor([suffix_ids], device=self.device, dtype=prompt_ids.dtype)
        one = torch.cat([prompt_ids.to(self.device), suffix], dim=1)
        if int(one.shape[1]) + 1 > int(total_max_tokens):
            raise RuntimeError("branched slot readout exceeds context cap")
        full_ids = one.repeat(len(aliases), 1)
        position = int(prompt_ids.shape[1]) + len(thought_token_ids) - 1
        alias_ids = [self.leading_space_token_id(alias) for alias in aliases]
        patcher: Any
        if quantization_control is None:
            patcher = FixedBranchPatcher(
                self.layers, position=position, branches_by_layer=branches_by_layer
            )
        else:
            patcher = QuantizationAwareFixedNonJPatcher(
                self.layers,
                position=position,
                branches_by_layer=branches_by_layer,
                directions_by_layer=quantization_control["directions_by_layer"],
                target_norms_by_layer=quantization_control["target_norms_by_layer"],
                rtol=float(quantization_control["rtol"]),
                norm_tolerance=float(quantization_control["norm_tolerance"]),
                projection_tolerance=float(quantization_control["projection_tolerance"]),
                correction_iterations=int(quantization_control["correction_iterations"]),
                correction_damping=float(quantization_control["correction_damping"]),
            )
        with patcher:
            output = self.model(input_ids=full_ids, use_cache=False, logits_to_keep=1)
        logits = output.logits[:, -1].float()
        constrained = logits[:, alias_ids]
        probabilities = torch.softmax(constrained, dim=-1)
        choices = torch.argmax(constrained, dim=-1)
        return {
            "branches": len(aliases),
            "sequence_tokens": int(full_ids.shape[1]),
            "patched_position": position,
            "layers": sorted(int(layer) for layer in branches_by_layer),
            "alias_token_ids": alias_ids,
            "alias_probabilities": [
                {alias: float(probabilities[row, column]) for column, alias in enumerate(aliases)}
                for row in range(len(aliases))
            ],
            "chosen_aliases": [aliases[int(index)] for index in choices.tolist()],
            "finite": bool(torch.isfinite(constrained).all() and torch.isfinite(probabilities).all()),
            "requested_deltas": patcher.requested,
            "realized_deltas": patcher.realized,
            "input_activations": patcher.input_activations,
            "applications": patcher.applications,
            "control_norm_errors": getattr(patcher, "norm_errors", {}),
            "control_projection_fractions": getattr(patcher, "projection_fractions", {}),
            "control_passed_rows": getattr(patcher, "passed_rows", {}),
            "control_iterations_used": getattr(patcher, "iterations_used", {}),
        }
