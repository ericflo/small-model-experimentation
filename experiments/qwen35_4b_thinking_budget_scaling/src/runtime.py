"""Qwen3.5-4B runtime: prompts, batched generation, s1-style thinking-budget forcing.

Think tokens are model-specific and were verified empirically for Qwen3.5-4B:
  <think>  = 248068,  </think> = 248069   (vocab 248320; these differ from Qwen3-4B!)

Budget-forced thinking protocol (think@B):
  Stage 1: generate up to B tokens.
    - if </think> AND EOS both appear within B  -> done (model finished early).
    - else keep the thinking region (g[:idx] if </think> seen, else all B tokens),
      optionally scramble it (content control), append a forced "</think>", and
      Stage 2: regenerate the answer (cap answer_max) from prompt+thinking+</think>.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

MODEL_ID = "Qwen/Qwen3.5-4B"
THINK_OPEN = 248068
THINK_CLOSE = 248069

SYSTEM = ("You are a Python coding assistant. Provide a single correct Python function. "
          "Put the final function in one ```python ... ``` code block.")

THINK_SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
NOTHINK_SAMPLING = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
GREEDY = dict(do_sample=False)


@dataclass
class GenResult:
    text: str       # decoded generated region (contains </think> so the code extractor works)
    n_think: int    # thinking tokens emitted (== budget when forced)
    n_gen: int      # total generated tokens (think + answer)
    forced: bool    # whether </think> was force-injected
    finished: bool  # answer reached EOS naturally


class Runtime:
    def __init__(self, model_id: str = MODEL_ID, dtype=torch.bfloat16, device: str = "cuda:0"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=dtype, device_map=device,
            attn_implementation="sdpa")
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set([eos] if isinstance(eos, int) else list(eos))
        self.close_ids = self.tok("</think>\n\n", add_special_tokens=False).input_ids
        self.load_secs = time.time() - t0

    # -- prompts -----------------------------------------------------------
    def prompt(self, user: str, enable_thinking: bool) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # -- helpers -----------------------------------------------------------
    def _ids(self, text: str) -> list[int]:
        return self.tok(text, return_tensors=None).input_ids

    def _trim_eos(self, ids: list[int]) -> list[int]:
        for i, t in enumerate(ids):
            if t in self.eos_ids:
                return ids[:i]
        return ids

    def _has_eos(self, ids: list[int]) -> bool:
        return any(t in self.eos_ids for t in ids)

    @torch.no_grad()
    def _gen(self, prefix_ids: list[list[int]], max_new: int, sampling: dict) -> list[list[int]]:
        """Left-pad a list of token-id sequences, generate, return generated regions."""
        maxlen = max(len(p) for p in prefix_ids)
        pad = self.tok.pad_token_id
        input_ids = torch.tensor([[pad] * (maxlen - len(p)) + p for p in prefix_ids], device=self.device)
        attn = (input_ids != pad).long()
        gk = dict(max_new_tokens=max_new, pad_token_id=pad, **sampling)
        out = self.model.generate(input_ids=input_ids, attention_mask=attn, **gk)
        return out[:, maxlen:].tolist()

    def _finalize(self, gen_ids: list[int]) -> GenResult:
        trimmed = self._trim_eos(gen_ids)
        n_think = trimmed.index(THINK_CLOSE) if THINK_CLOSE in trimmed else 0
        return GenResult(text=self.tok.decode(trimmed, skip_special_tokens=False),
                         n_think=n_think, n_gen=len(trimmed),
                         forced=False, finished=self._has_eos(gen_ids))

    # -- main entry --------------------------------------------------------
    def generate(self, prompts: list[str], *, think: bool, budget: int | None,
                 greedy: bool = False, shuffle_think: bool = False,
                 answer_max: int = 512, think_cap: int = 4096, batch_size: int = 32) -> list[GenResult]:
        results: list[GenResult | None] = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))

        def process(idxs: list[int], bs: int):
            for s in range(0, len(idxs), bs):
                sub = idxs[s:s + bs]
                try:
                    outs = self._batch([prompts[i] for i in sub], think, budget, greedy,
                                       shuffle_think, answer_max, think_cap)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if len(sub) == 1:
                        raise
                    process(sub, max(1, len(sub) // 2))  # subdivide and retry
                    continue
                for r, i in zip(outs, sub):
                    results[i] = r

        process(order, batch_size)
        return results  # type: ignore[return-value]

    def _batch(self, prompts, think, budget, greedy, shuffle_think, answer_max, think_cap):
        sampling = GREEDY if greedy else (THINK_SAMPLING if think else NOTHINK_SAMPLING)
        prefix_ids = [self._ids(p) for p in prompts]

        # no-think, or natural thinking (no budget): single-stage to EOS
        if not think:
            return [self._finalize(g) for g in self._gen(prefix_ids, answer_max, sampling)]
        if budget is None:
            return [self._finalize(g) for g in self._gen(prefix_ids, think_cap, sampling)]

        # budget-forced thinking
        gen1 = self._gen(prefix_ids, budget, sampling)
        results: list[GenResult | None] = [None] * len(prompts)
        cont_prefix, cont_meta = [], []  # meta: (j, n_think, forced)
        for j, g in enumerate(gen1):
            has_close = THINK_CLOSE in g
            if has_close and self._has_eos(g[g.index(THINK_CLOSE):]):
                results[j] = self._finalize(g)
                continue
            if has_close:
                think_region = g[:g.index(THINK_CLOSE)]
                forced = False
            else:
                think_region = self._trim_eos(g)
                forced = True
            n_think = len(think_region)
            if shuffle_think and think_region:
                perm = torch.randperm(len(think_region)).tolist()
                think_region = [think_region[k] for k in perm]
            cont_prefix.append(prefix_ids[j] + think_region + self.close_ids)
            cont_meta.append((j, n_think, forced))

        if cont_prefix:
            gen2 = self._gen(cont_prefix, answer_max, sampling)
            for (j, n_think, forced), ans in zip(cont_meta, gen2):
                ans_t = self._trim_eos(ans)
                results[j] = GenResult(
                    text="</think>\n" + self.tok.decode(ans_t, skip_special_tokens=False),
                    n_think=n_think, n_gen=n_think + len(ans_t),
                    forced=forced, finished=self._has_eos(ans))
        return results  # type: ignore[return-value]
