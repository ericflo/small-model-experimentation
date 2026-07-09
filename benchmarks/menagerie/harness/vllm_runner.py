"""Isolated vLLM runner for the Menagerie Qwen backend.

- This script is run by /home/ericflo/Development/smx-menagerie/.venv-vllm/bin/python only.
- FlashInfer is enabled by default and works via the pinned stack's precompiled flashinfer-cubin (no nvcc needed); set MENAGERIE_VLLM_NO_FLASHINFER=1 to fall back to native torch sampling.
- This runner is the benchmark's PERSISTENT-SERVER sibling of the repo's reusable one-shot experiment runner templates/experiment/src/vllm_runner.py, sharing the same pinned venv (../../.venv-vllm from requirements-vllm.lock.txt) and the same two-phase thinking semantics; see docs/vllm_inference.md.
- vLLM V1 uses spawn on WSL, hence the __main__ guard.
- Model is Qwen3.5-4B, a hybrid model. Defaults are gpu_memory_utilization=0.85 and max_model_len=16384; more Mamba cache blocks exist at 0.85, so max_num_seqs stays 64 and is safely under the available block count.
- THINKING BUDGET method: budgets may differ by mode, and a per-prompt context guard caps them against MAX_MODEL_LEN after reserving the mode's answer budget and a margin. The default is TWO-PHASE (env MENAGERIE_VLLM_BUDGET=two_phase), which mirrors the HF `qwen` backend token-for-token (generate up to B think tokens; if </think> was not emitted, force it via close_ids "</think>\\n\\n"; then regenerate the answer with the mode's answer budget using prefix caching). This was chosen as the default because it reproduces the HF backend's scores EXACTLY (identical aggregate on the quick tier), which is required for cross-backend comparability.
- The vLLM-NATIVE budget (env MENAGERIE_VLLM_BUDGET=native, single pass) also works and was EMPIRICALLY VERIFIED on vllm 0.24.0 to cap think tokens at B (budget=32 -> exactly 31 sampled think tokens then a forced </think>; the uncapped baseline never closed within 512 tokens). Native requires reasoning_parser="qwen3" and is a faster single pass, but its post-forced-close answer conditioning differs slightly from the HF backend, so scores can diverge in low-budget/truncated regimes; use it for speed, two_phase for exact HF parity.
- MENAGERIE_VLLM_ADAPTER points at a PEFT LoRA adapter directory to load.
- Adapter validation is cribbed from templates/experiment/src/vllm_runner.py.
- max_lora_rank is sized from the adapter instead of vLLM's rank-16 default.
"""

import hashlib
import os
_no_flashinfer = os.environ.get("MENAGERIE_VLLM_NO_FLASHINFER")
if _no_flashinfer and _no_flashinfer.lower() not in {"0", "false", "no", "off"}:
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import json
import sys

try:
    from harness.adapter_spec import validate_adapter
except ImportError:
    from adapter_spec import validate_adapter


THINK_CLOSE = 248069  # id of "</think>"
MAX_MODEL_LEN = int(os.environ.get("MENAGERIE_VLLM_MAXLEN", "16384"))


def _engine_kwargs(method, adapter_info=None) -> dict:
    model_id = os.environ.get("MENAGERIE_VLLM_MODEL", "Qwen/Qwen3.5-4B")
    gmu = float(os.environ.get("MENAGERIE_VLLM_GMU", "0.85"))
    kwargs = dict(
        model=model_id,
        gpu_memory_utilization=gmu,
        language_model_only=True,
        # The old 8192 cap was an artifact of the conservative
        # gpu_memory_utilization=0.5, not a model limit; the model supports far
        # more. 16384 covers deep-tier episodes with full escalated think budgets.
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=64,
    )
    if method == "native":
        kwargs["reasoning_parser"] = "qwen3"
    if adapter_info is not None:
        kwargs["enable_lora"] = True
        kwargs["max_loras"] = 1
        kwargs["max_cpu_loras"] = 1
        kwargs["max_lora_rank"] = adapter_info["rank"]
    return kwargs


def _load_llm(method, adapter_info=None):
    from vllm import LLM

    kwargs = _engine_kwargs(method, adapter_info)
    try:
        return LLM(enforce_eager=False, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[vllm_runner] enforce_eager=False failed ({exc}); retrying enforce_eager=True",
            file=sys.stderr,
            flush=True,
        )
        try:
            return LLM(enforce_eager=True, **kwargs)
        except Exception:  # noqa: BLE001
            raise RuntimeError(
                "vLLM engine init failed. If the GPU is shared or memory-constrained, retry with "
                "MENAGERIE_VLLM_GMU=0.5 MENAGERIE_VLLM_MAXLEN=8192, or fall back to "
                "--backend qwen (HF)."
            ) from exc


def _job_lora_request(job, lora_request, adapter_path):
    if not job.get("adapter"):
        return None
    if adapter_path is None:
        raise ValueError("job requested an adapter but the runner was not started with MENAGERIE_VLLM_ADAPTER")
    if str(job["adapter"]) != adapter_path:
        raise ValueError(f"job adapter {job['adapter']!r} does not match runner adapter {adapter_path!r}")
    return lora_request


def _trim_eos(ids, eos_ids):
    for idx, token_id in enumerate(ids):
        if token_id in eos_ids:
            return ids[:idx]
    return ids


def _answer_budget(max_new, mode):
    if isinstance(max_new, dict):
        return int(max_new[mode])
    return int(max_new)


def _think_budget(think_budget, mode):
    if isinstance(think_budget, dict):
        return int(think_budget[mode])
    return int(think_budget)


def _run_no_think(llm, tok, prompt_ids, modes, max_new, temperature, eos_ids, lora_request=None):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    answer_budgets = [
        max(0, min(_answer_budget(max_new, modes[i]), MAX_MODEL_LEN - len(prompt_ids[i]) - 16))
        for i in range(len(prompt_ids))
    ]
    active = [i for i in range(len(prompt_ids)) if answer_budgets[i] > 0]
    sps = [
        SamplingParams(
            temperature=temperature,
            max_tokens=answer_budgets[i],
            skip_special_tokens=False,
        )
        for i in active
    ]
    outs = (
        llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids[i]) for i in active],
            sps,
            lora_request=lora_request,
        )
        if active
        else []
    )
    res = [
        {
            "answer": "",
            "think_tokens": 0,
            "forced_close": False,
            "prompt_tokens": len(prompt_ids[i]),
            "context_capped": False,
        }
        for i in range(len(prompt_ids))
    ]
    for i, o in zip(active, outs):
        ids = _trim_eos(list(o.outputs[0].token_ids), eos_ids)
        res[i]["answer"] = tok.decode(ids, skip_special_tokens=True)
    return res


def _run_native(llm, tok, prompt_ids, modes, max_new, think_budget, temperature, eos_ids, lora_request=None):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    budgets = [_think_budget(think_budget, mode) for mode in modes]
    answer_budgets = [_answer_budget(max_new, mode) for mode in modes]
    allowed_think = [
        max(0, min(budgets[i], MAX_MODEL_LEN - len(prompt_ids[i]) - answer_budgets[i] - 16))
        for i in range(len(prompt_ids))
    ]
    context_capped = [allowed_think[i] < budgets[i] for i in range(len(prompt_ids))]
    sps = [
        SamplingParams(
            temperature=temperature,
            max_tokens=allowed_think[i] + answer_budgets[i],
            thinking_token_budget=allowed_think[i],
            skip_special_tokens=False,
        )
        for i in range(len(prompt_ids))
    ]
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], sps, lora_request=lora_request)
    res = []
    for i, o in enumerate(outs):
        ids = list(o.outputs[0].token_ids)
        if THINK_CLOSE in ids:
            ci = ids.index(THINK_CLOSE)
            think_tokens = ci
            ans_ids = _trim_eos(ids[ci + 1 :], eos_ids)
            forced = ci >= allowed_think[i] - 1
        else:
            think_tokens = len(ids)
            ans_ids = []
            forced = True
        res.append(
            {
                "answer": tok.decode(ans_ids, skip_special_tokens=True),
                "think_tokens": think_tokens,
                "forced_close": bool(forced),
                "prompt_tokens": len(prompt_ids[i]),
                "context_capped": context_capped[i],
            }
        )
    return res


def _run_two_phase(llm, tok, prompt_ids, modes, max_new, think_budget, temperature, eos_ids, close_ids, lora_request=None):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    budgets = [_think_budget(think_budget, mode) for mode in modes]
    answer_budgets = [_answer_budget(max_new, mode) for mode in modes]
    allowed_think = [
        max(0, min(budgets[i], MAX_MODEL_LEN - len(prompt_ids[i]) - answer_budgets[i] - 16))
        for i in range(len(prompt_ids))
    ]
    context_capped = [allowed_think[i] < budgets[i] for i in range(len(prompt_ids))]
    active = [i for i in range(len(prompt_ids)) if allowed_think[i] > 0]
    sp1 = [
        SamplingParams(temperature=temperature, max_tokens=allowed_think[i], skip_special_tokens=False)
        for i in active
    ]
    outs1 = (
        llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids[i]) for i in active],
            sp1,
            lora_request=lora_request,
        )
        if active
        else []
    )
    outs_by_idx = dict(zip(active, outs1))
    n = len(prompt_ids)
    res = [None] * n
    think_tokens = [0] * n
    forced = [False] * n
    conts = []
    cont_idx = []
    for i in range(n):
        if allowed_think[i] == 0:
            thinking = []
            forced[i] = True
            conts.append(TokensPrompt(prompt_token_ids=prompt_ids[i] + close_ids))
            cont_idx.append(i)
            continue
        o = outs_by_idx[i]
        ids = list(o.outputs[0].token_ids)
        if THINK_CLOSE in ids:
            ci = ids.index(THINK_CLOSE)
            thinking = ids[:ci]
            rest = ids[ci + 1 :]
            think_tokens[i] = len(thinking)
            if any(t in eos_ids for t in rest):
                res[i] = {
                    "answer": tok.decode(_trim_eos(rest, eos_ids), skip_special_tokens=True),
                    "think_tokens": len(thinking),
                    "forced_close": False,
                    "prompt_tokens": len(prompt_ids[i]),
                    "context_capped": context_capped[i],
                }
                continue
        else:
            thinking = _trim_eos(ids, eos_ids)
            think_tokens[i] = len(thinking)
            forced[i] = True
        conts.append(TokensPrompt(prompt_token_ids=prompt_ids[i] + thinking + close_ids))
        cont_idx.append(i)
    if conts:
        sp2 = [
            SamplingParams(
                temperature=temperature,
                max_tokens=_answer_budget(max_new, modes[j]),
                skip_special_tokens=False,
            )
            for j in cont_idx
        ]
        outs2 = llm.generate(conts, sp2, lora_request=lora_request)
        for j, o in zip(cont_idx, outs2):
            aids = _trim_eos(list(o.outputs[0].token_ids), eos_ids)
            res[j] = {
                "answer": tok.decode(aids, skip_special_tokens=True),
                "think_tokens": think_tokens[j],
                "forced_close": forced[j],
                "prompt_tokens": len(prompt_ids[j]),
                "context_capped": context_capped[j],
            }
    return res


def main():
    # Reserve a clean protocol channel: duplicate the real stdout (fd 1) and then
    # point fd 1 at stderr. vLLM/HF write warnings and progress bars to stdout;
    # routing them to stderr keeps the parent's stdout pipe pure protocol JSON.
    # The spawned EngineCore subprocess inherits this redirected fd 1, so its noise
    # also goes to stderr. Responses are written to `proto` (the saved real stdout).
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    proto = os.fdopen(proto_fd, "w", buffering=1)
    sys.stdout = sys.stderr

    adapter_env = os.environ.get("MENAGERIE_VLLM_ADAPTER") or None
    adapter_info = validate_adapter(adapter_env) if adapter_env else None
    method = os.environ.get("MENAGERIE_VLLM_BUDGET", "two_phase")
    from vllm import SamplingParams  # noqa: F401
    from vllm.inputs import TokensPrompt  # noqa: F401

    llm = _load_llm(method, adapter_info)
    lora_request = None
    if adapter_info is not None:
        from vllm.lora.request import LoRARequest

        name_hash = hashlib.sha256(adapter_info["path"].encode("utf-8")).hexdigest()[:12]
        lora_request = LoRARequest(f"menagerie-{name_hash}", 1, adapter_info["path"])
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        os.environ.get("MENAGERIE_VLLM_MODEL", "Qwen/Qwen3.5-4B"),
        trust_remote_code=True,
    )
    close_ids = tok("</think>\n\n", add_special_tokens=False).input_ids
    eos_ids = set()
    if getattr(tok, "eos_token_id", None) is not None:
        eos_ids.add(tok.eos_token_id)
    proto.write(json.dumps({"ready": True, "method": method, "adapter": adapter_info["path"] if adapter_info else None}) + "\n")
    proto.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        job = json.loads(line)
        if job.get("cmd") == "shutdown":
            break
        prompts = job["prompts"]
        modes = job.get("modes") or ["atom"] * len(prompts)
        think = bool(job["think"])
        think_budget = job["think_budget"]
        temperature = float(job.get("temperature", 0.0))
        max_new = job["max_new_tokens"]
        job_lora_request = _job_lora_request(job, lora_request, adapter_info["path"] if adapter_info else None)
        prompt_ids = []
        for msgs in prompts:
            try:
                text = tok.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=think,
                )
            except TypeError:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompt_ids.append(tok(text, add_special_tokens=False).input_ids)
        if not think:
            completions = _run_no_think(llm, tok, prompt_ids, modes, max_new, temperature, eos_ids, job_lora_request)
        elif method == "native":
            completions = _run_native(llm, tok, prompt_ids, modes, max_new, think_budget, temperature, eos_ids, job_lora_request)
        else:
            completions = _run_two_phase(
                llm,
                tok,
                prompt_ids,
                modes,
                max_new,
                think_budget,
                temperature,
                eos_ids,
                close_ids,
                job_lora_request,
            )
        proto.write(json.dumps({"completions": completions}) + "\n")
        proto.flush()


if __name__ == "__main__":
    main()
