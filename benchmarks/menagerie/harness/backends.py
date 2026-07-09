"""Policy backends and Qwen inference adapter for the Menagerie harness."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


class _RngCache:
    def __init__(self, seed: int):
        self.seed = seed
        self._cache: dict[tuple[str, str], random.Random] = {}

    def get(self, family: str, item_id: str) -> random.Random:
        key = (family, item_id)
        if key not in self._cache:
            digest = hashlib.sha256(f"{self.seed}|{family}|{item_id}".encode()).hexdigest()
            self._cache[key] = random.Random(int(digest[:16], 16))
        return self._cache[key]


class OracleBackend:
    """CPU backend that follows each family's oracle policy."""

    def __init__(self):
        self.stats: dict = {}

    def batch_act(self, contexts: list[dict]) -> list[str]:
        return [ctx["module"].oracle_policy(ctx["item"], ctx["history"]) for ctx in contexts]


class RandomBackend:
    """CPU backend that follows each family's random policy with stable RNGs."""

    def __init__(self, seed: int = 0):
        self.stats: dict = {}
        self._rngs = _RngCache(seed)

    def batch_act(self, contexts: list[dict]) -> list[str]:
        actions = []
        for ctx in contexts:
            rng = self._rngs.get(ctx["family"], ctx["item_id"])
            actions.append(ctx["module"].random_policy(ctx["item"], ctx["history"], rng))
        return actions


class NoisyOracleBackend:
    """CPU backend that mixes oracle and random actions per turn."""

    def __init__(self, eps: float, seed: int = 0):
        self.eps = eps
        self.stats: dict = {}
        self._rngs = _RngCache(seed)

    def batch_act(self, contexts: list[dict]) -> list[str]:
        actions = []
        for ctx in contexts:
            rng = self._rngs.get(ctx["family"], ctx["item_id"])
            if rng.random() < self.eps:
                actions.append(ctx["module"].random_policy(ctx["item"], ctx["history"], rng))
            else:
                actions.append(ctx["module"].oracle_policy(ctx["item"], ctx["history"]))
        return actions


class ConstBackend:
    """CPU backend that always emits a fixed string."""

    def __init__(self, text: str):
        self.text = text
        self.stats: dict = {}

    def batch_act(self, contexts: list[dict]) -> list[str]:
        return [self.text for _ in contexts]


def build_chat_messages(meta: dict, history: list, obs: str) -> list[dict]:
    """Build chat messages from public observation/action context only."""

    messages = [{"role": "system", "content": meta["action_format"]}]
    for turn in history:
        messages.append({"role": "user", "content": turn["obs"]})
        messages.append({"role": "assistant", "content": turn["action"]})
    messages.append({"role": "user", "content": obs})
    return messages


def pad_batch(prompts: list[list[int]], pad: int) -> tuple[list[list[int]], list[list[int]]]:
    """Left-pad prompts to a common length; mask is 1 for real tokens, 0 for padding, decided by LENGTH not token value."""

    maxlen = max(len(prompt) for prompt in prompts)
    rows = []
    mask = []
    for prompt in prompts:
        padding = maxlen - len(prompt)
        rows.append([pad] * padding + prompt)
        mask.append([0] * padding + [1] * len(prompt))
    return rows, mask


class QwenBackend:
    """Lazy torch/transformers backend for Qwen3.5-4B inference."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-4B",
        device: str = "cuda:0",
        think: bool = False,
        think_budget: int | dict[str, int] = 512,
        max_batch: int = 96,
        max_new_tokens: dict | None = None,
        adapter: str | None = None,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The qwen backend needs torch+transformers. Run under "
                "/home/ericflo/Development/small-model-experimentation/.venv/bin/python "
                "(see run.py --help). CPU backends (oracle/random/noisy/const) work under any python3."
            ) from exc

        self.torch = torch
        self.think = think
        self.think_budget = think_budget
        self.max_batch = max_batch
        self.max_new_tokens = max_new_tokens or {"atom": 64, "episode": 96}
        self.stats = {"calls": 0, "generated_tokens": 0, "forced_think_closes": 0}

        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="sdpa",
        ).eval()
        eos_ids: set[int] = set()
        for token_id in (tok.eos_token_id, tok.pad_token_id):
            if isinstance(token_id, int):
                eos_ids.add(token_id)
        cfg_eos = model.generation_config.eos_token_id
        if isinstance(cfg_eos, int):
            eos_ids.add(cfg_eos)
        elif cfg_eos:
            eos_ids.update(int(value) for value in cfg_eos)
        if adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "The qwen backend needs peft to load --adapter; peft is required in the HF venv "
                    "/home/ericflo/Development/small-model-experimentation/.venv and must not be auto-installed."
                ) from exc
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
        self.eos_ids = eos_ids
        self.THINK_CLOSE = 248069
        self.close_ids = tok("</think>\n\n", add_special_tokens=False).input_ids
        self.tok = tok
        self.model = model
        self.pad = tok.pad_token_id
        self._oom_errors = (torch.cuda.OutOfMemoryError, getattr(torch, "AcceleratorError", torch.cuda.OutOfMemoryError))

    def _prompt_ids(self, ctx: dict) -> list[int]:
        messages = build_chat_messages(ctx["meta"], ctx["history"], ctx["obs"])
        try:
            text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.think,
            )
        except TypeError:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tok(text, add_special_tokens=False).input_ids

    def _trim_at_eos(self, ids: list[int]) -> list[int]:
        for idx, token_id in enumerate(ids):
            if token_id in self.eos_ids:
                return ids[:idx]
        return ids

    def _decode(self, ids: list[int]) -> str:
        return self.tok.decode(self._trim_at_eos(ids), skip_special_tokens=True)

    def _generate_batch(self, prompts: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        torch = self.torch
        maxlen = max(len(prompt) for prompt in prompts)
        rows, mask = pad_batch(prompts, self.pad)
        inp = torch.tensor(rows, device=self.model.device)
        attention_mask = torch.tensor(mask, device=self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inp,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.pad,
                eos_token_id=sorted(self.eos_ids),
                do_sample=False,
            )
        generated = out[:, maxlen:].tolist()
        self.stats["calls"] += 1
        self.stats["generated_tokens"] += sum(len(self._trim_at_eos(ids)) for ids in generated)
        return generated

    def _generate_resilient(self, prompts: list[list[int]], max_new_tokens: int, chunk_size: int | None = None) -> list[list[int]]:
        chunk_size = chunk_size or len(prompts)
        outputs: list[list[int]] = []
        start = 0
        while start < len(prompts):
            chunk = prompts[start : start + chunk_size]
            try:
                outputs.extend(self._generate_batch(chunk, max_new_tokens))
                start += chunk_size
            except self._oom_errors:
                self.torch.cuda.empty_cache()
                if len(chunk) == 1:
                    raise
                half = max(1, len(chunk) // 2)
                outputs.extend(self._generate_resilient(chunk, max_new_tokens, half))
                start += len(chunk)
        return outputs

    def _run_chunk(self, records: list[tuple[int, dict, list[int]]], mode: str) -> list[tuple[int, str]]:
        prompts = [record[2] for record in records]
        if not self.think:
            generated = self._generate_resilient(prompts, self.max_new_tokens[mode])
            return [(records[idx][0], self._decode(ids)) for idx, ids in enumerate(generated)]

        think_budget = int(self.think_budget[mode]) if isinstance(self.think_budget, dict) else int(self.think_budget)
        thoughts = self._generate_resilient(prompts, think_budget)
        answers: list[tuple[int, str] | None] = [None] * len(records)
        continuations = []
        continuation_indices = []
        for idx, generated in enumerate(thoughts):
            original_index, _ctx, prompt = records[idx]
            if self.THINK_CLOSE in generated:
                close_idx = generated.index(self.THINK_CLOSE)
                thinking = generated[:close_idx]
                rest = generated[close_idx + 1 :]
                if any(token_id in self.eos_ids for token_id in rest):
                    answers[idx] = (original_index, self._decode(rest))
                    continue
            else:
                thinking = generated
                self.stats["forced_think_closes"] += 1

            thinking = self._trim_at_eos(thinking)
            continuations.append(prompt + thinking + self.close_ids)
            continuation_indices.append(idx)

        if continuations:
            generated_answers = self._generate_resilient(continuations, self.max_new_tokens[mode])
            for idx, answer_ids in zip(continuation_indices, generated_answers):
                answers[idx] = (records[idx][0], self._decode(answer_ids))

        return [answer for answer in answers if answer is not None]

    def batch_act(self, contexts: list[dict]) -> list[str]:
        results: list[str | None] = [None] * len(contexts)
        by_mode: dict[str, list[tuple[int, dict, list[int]]]] = defaultdict(list)
        for index, ctx in enumerate(contexts):
            public_ctx = {
                "meta": ctx["meta"],
                "history": ctx["history"],
                "obs": ctx["obs"],
            }
            by_mode[ctx["mode"]].append((index, public_ctx, self._prompt_ids(public_ctx)))

        for mode, records in by_mode.items():
            records.sort(key=lambda record: len(record[2]))
            for start in range(0, len(records), self.max_batch):
                for index, answer in self._run_chunk(records[start : start + self.max_batch], mode):
                    results[index] = answer

        return [result if result is not None else "" for result in results]


class QwenVllmBackend:
    """Opt-in vLLM backend: runs vLLM in an isolated .venv-vllm subprocess.

    The HF harness process never imports vllm/torch. Per batch_act call we send a
    JSON job (one line) to a long-lived subprocess and read one JSON response line.
    The subprocess loads Qwen3.5-4B once and enforces the SAME thinking budget as
    the HF `qwen` backend, so scores are comparable.
    """

    VENV_PYTHON = "/home/ericflo/Development/smx-menagerie/.venv-vllm/bin/python"

    def __init__(
        self,
        model_id="Qwen/Qwen3.5-4B",
        device="cuda:0",
        think=False,
        think_budget: int | dict[str, int] = 512,
        max_batch=96,
        max_new_tokens=None,
        adapter=None,
    ):
        self.think = think
        self.think_budget = think_budget
        self.max_new_tokens = max_new_tokens or {"atom": 64, "episode": 96}
        self.adapter = str(adapter) if adapter is not None else None
        self.stats = {
            "calls": 0,
            "generated_think_tokens": 0,
            "forced_think_closes": 0,
            "max_think_tokens": 0,
            "think_token_counts": [],
            "context_capped": 0,
            "max_prompt_tokens": 0,
        }
        runner = str(Path(__file__).resolve().parent / "vllm_runner.py")
        env = dict(os.environ)
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        env["MENAGERIE_VLLM_MODEL"] = model_id
        env.setdefault("MENAGERIE_VLLM_BUDGET", "two_phase")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if adapter is not None:
            env["MENAGERIE_VLLM_ADAPTER"] = str(adapter)
        if not Path(self.VENV_PYTHON).exists():
            raise RuntimeError(
                "vLLM venv not found at /home/ericflo/Development/smx-menagerie/.venv-vllm. Build it with: uv venv --python 3.12 .venv-vllm && uv pip sync --python .venv-vllm/bin/python --torch-backend=cu129 requirements-vllm.lock.txt (see docs/vllm_inference.md), or fall back to --backend qwen (HF backend, slower)."
            )
        self.proc = subprocess.Popen(
            [self.VENV_PYTHON, runner],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        ready = self._read_line()
        info = json.loads(ready)
        if not info.get("ready"):
            raise RuntimeError(f"vllm_runner did not report ready: {ready!r}")
        self.method = info.get("method")

    def _read_line(self):
        line = self.proc.stdout.readline()
        if line == "":
            code = self.proc.poll()
            raise RuntimeError(f"vllm_runner subprocess exited (code={code}) before responding; check its stderr above")
        return line

    def batch_act(self, contexts):
        prompts = []
        modes = []
        for ctx in contexts:
            prompts.append(build_chat_messages(ctx["meta"], ctx["history"], ctx["obs"]))
            modes.append(ctx["mode"])
        job = {
            "prompts": prompts,
            "modes": modes,
            "think": self.think,
            "think_budget": self.think_budget,
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.0,
        }
        if self.adapter is not None:
            job["adapter"] = self.adapter
        self.proc.stdin.write(json.dumps(job) + "\n")
        self.proc.stdin.flush()
        resp = json.loads(self._read_line())
        completions = resp["completions"]
        if len(completions) != len(contexts):
            raise ValueError(f"vllm_runner returned {len(completions)} completions for {len(contexts)} contexts")
        self.stats["calls"] += 1
        for comp in completions:
            tt = int(comp.get("think_tokens", 0))
            prompt_tokens = int(comp.get("prompt_tokens", 0))
            self.stats["generated_think_tokens"] += tt
            self.stats["max_think_tokens"] = max(self.stats["max_think_tokens"], tt)
            self.stats["max_prompt_tokens"] = max(self.stats["max_prompt_tokens"], prompt_tokens)
            if comp.get("forced_close"):
                self.stats["forced_think_closes"] += 1
            if comp.get("context_capped"):
                self.stats["context_capped"] += 1
            if self.think:
                self.stats["think_token_counts"].append(tt)
        return [comp["answer"] for comp in completions]

    def close(self):
        proc = getattr(self, "proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def make_backend(spec: str, *, seed: int = 0, qwen_opts: dict | None = None):
    """Construct a backend from a CLI/backend spec string."""

    if spec == "oracle":
        return OracleBackend()
    if spec == "random":
        return RandomBackend(seed)
    if spec.startswith("noisy:"):
        try:
            eps = float(spec.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("invalid noisy backend; expected noisy:EPS") from exc
        return NoisyOracleBackend(eps, seed)
    if spec.startswith("const:"):
        return ConstBackend(spec.split(":", 1)[1])
    if spec == "qwen":
        return QwenBackend(**(qwen_opts or {}))
    if spec == "qwen_vllm":
        return QwenVllmBackend(**(qwen_opts or {}))
    raise ValueError("unknown backend spec; valid specs are oracle, random, noisy:EPS, const:TEXT, qwen, qwen_vllm")
