"""Pinned cached native-thinking generation for the seam budget ladder."""

from __future__ import annotations

import time
from typing import Any

import torch


class _NaturalThinkStopper:
    """Stop at a natural answer allowance or a right-censoring thought cap."""

    def __init__(
        self,
        *,
        prompt_tokens: int,
        think_close_id: int,
        eos_id: int,
        max_think_steps: int,
        answer_max_tokens: int,
    ) -> None:
        self.prompt_tokens = int(prompt_tokens)
        self.think_close_id = int(think_close_id)
        self.eos_id = int(eos_id)
        self.max_think_steps = int(max_think_steps)
        self.answer_max_tokens = int(answer_max_tokens)

    def __call__(self, input_ids: torch.Tensor, _scores: torch.Tensor, **_kwargs: Any) -> torch.BoolTensor:
        if input_ids.shape[0] != 1:
            raise RuntimeError("the frozen generation contract requires batch one")
        generated = input_ids[0, self.prompt_tokens :]
        close = (generated == self.think_close_id).nonzero(as_tuple=False).flatten()
        if close.numel() == 0:
            stop = generated.numel() >= self.max_think_steps
        else:
            answer_tokens = generated.numel() - int(close[0]) - 1
            stop = answer_tokens >= self.answer_max_tokens or int(generated[-1]) == self.eos_id
        return torch.tensor([stop], dtype=torch.bool, device=input_ids.device)


class QwenCachedThinkModel:
    """Qwen3.5-4B generation with an audited batch-one KV-cache contract."""

    def __init__(self, config: dict[str, Any]):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_config = config["model"]
        if model_config["id"] != "Qwen/Qwen3.5-4B":
            raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["id"],
            revision=model_config["revision"],
            trust_remote_code=True,
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

    def render_thinking(self, user: str) -> str:
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
        rendered = self.render_thinking(user)
        input_ids = self.tokenizer(rendered, return_tensors="pt").input_ids
        bos = self.tokenizer.bos_token_id
        if bos is not None and input_ids.shape[1] and int(input_ids[0, 0]) != int(bos):
            input_ids = torch.cat(
                [torch.tensor([[bos]], dtype=input_ids.dtype), input_ids], dim=1
            )
        if input_ids.shape[1] > int(prompt_max_tokens):
            raise RuntimeError(
                f"rendered prompt has {input_ids.shape[1]} tokens, above frozen maximum {prompt_max_tokens}"
            )
        open_positions = (input_ids[0] == self.think_open_id).nonzero(as_tuple=False).flatten()
        if open_positions.numel() != 1:
            raise RuntimeError(f"prompt must contain exactly one <think>, got {open_positions.tolist()}")
        if (input_ids[0] == self.think_close_id).any():
            raise RuntimeError("prompt already contains a think-close token")
        return {
            "rendered": rendered,
            "input_ids": input_ids.to(self.device),
            "prompt_tokens": int(input_ids.shape[1]),
            "think_open_position": int(open_positions[0]),
        }

    def leading_space_token_id(self, text: str) -> int:
        ids = self.tokenizer(" " + text, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"alias {text!r} is not one leading-space token: {ids}")
        return int(ids[0])

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        seed: int,
        max_think_steps: int,
        answer_max_tokens: int,
        total_max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> dict[str, Any]:
        from transformers import StoppingCriteriaList

        if input_ids.shape[0] != 1:
            raise RuntimeError("the frozen generation contract requires batch one")
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens + max_think_steps + answer_max_tokens > total_max_tokens:
            raise RuntimeError("prompt plus frozen generation allowance exceeds total context cap")
        stopper = _NaturalThinkStopper(
            prompt_tokens=prompt_tokens,
            think_close_id=self.think_close_id,
            eos_id=self.eos_id,
            max_think_steps=max_think_steps,
            answer_max_tokens=answer_max_tokens,
        )
        input_lengths: list[int] = []

        def pre_hook(_module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            ids = kwargs.get("input_ids")
            if ids is not None:
                input_lengths.append(int(ids.shape[1]))

        handle = self.model.register_forward_pre_hook(pre_hook, with_kwargs=True)
        devices = [self.device.index] if self.device.type == "cuda" else []
        started = time.perf_counter()
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(int(seed))
                output = self.model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    max_new_tokens=int(max_think_steps + answer_max_tokens),
                    stopping_criteria=StoppingCriteriaList([stopper]),
                    eos_token_id=self.eos_id,
                    pad_token_id=self.eos_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
        finally:
            handle.remove()
        sequence = output.sequences[0]
        generated = sequence[prompt_tokens:]
        close_positions = (generated == self.think_close_id).nonzero(as_tuple=False).flatten()
        natural_close = close_positions.numel() == 1
        close_index = int(close_positions[0]) if natural_close else None
        close_step = close_index + 1 if close_index is not None else None
        think_tokens = close_index if close_index is not None else int(generated.numel())
        answer_ids = generated[close_index + 1 :] if close_index is not None else generated[:0]
        answer_text = self.tokenizer.decode(answer_ids.tolist(), skip_special_tokens=False)
        if not natural_close:
            stopped_by = (
                "eos_before_close" if generated.numel() and int(generated[-1]) == self.eos_id
                else "think_cap_without_close"
            )
        elif answer_ids.numel() and int(answer_ids[-1]) == self.eos_id:
            stopped_by = "eos"
        else:
            stopped_by = "answer_cap"
        return {
            "generated_token_ids": generated.tolist(),
            "generated_text": self.tokenizer.decode(generated.tolist(), skip_special_tokens=False),
            "answer_text": answer_text,
            "natural_close": natural_close,
            "close_step": close_step,
            "think_tokens": int(think_tokens),
            "answer_tokens": int(answer_ids.numel()),
            "stopped_by": stopped_by,
            "forward_calls": len(input_lengths),
            "forward_input_lengths": input_lengths,
            "cache_contract_pass": bool(
                input_lengths
                and input_lengths[0] == prompt_tokens
                and all(length == 1 for length in input_lengths[1:])
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
