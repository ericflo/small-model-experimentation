"""Isolated vLLM runner for the Menagerie Qwen backend.

- This script is run by /home/ericflo/Development/smx-menagerie/.venv-vllm/bin/python only.
- FlashInfer is enabled by default and works via the pinned stack's precompiled flashinfer-cubin (no nvcc needed); set MENAGERIE_VLLM_NO_FLASHINFER=1 to fall back to native torch sampling.
- This runner is the benchmark's PERSISTENT-SERVER sibling of the repo's reusable one-shot experiment runner templates/experiment/src/vllm_runner.py, sharing the same pinned venv (../../.venv-vllm from requirements-vllm.lock.txt) and the same two-phase thinking semantics; see docs/vllm_inference.md.
- vLLM V1 uses spawn on WSL, hence the __main__ guard.
- Model is Qwen3.5-4B, a hybrid model: at gpu_memory_utilization=0.5 only about 93 Mamba cache blocks exist, so max_num_seqs MUST be capped (<=93) or CUDA-graph capture fails with "max_num_seqs exceeds available Mamba cache blocks". We use max_num_seqs=64.
- THINKING BUDGET method: default is TWO-PHASE (env MENAGERIE_VLLM_BUDGET=two_phase), which mirrors the HF `qwen` backend token-for-token (generate up to B think tokens; if </think> was not emitted, force it via close_ids "</think>\\n\\n"; then regenerate the answer with the mode's answer budget using prefix caching). This was chosen as the default because it reproduces the HF backend's scores EXACTLY (identical aggregate on the quick tier), which is required for cross-backend comparability.
- The vLLM-NATIVE budget (env MENAGERIE_VLLM_BUDGET=native, single pass) also works and was EMPIRICALLY VERIFIED on vllm 0.24.0 to cap think tokens at B (budget=32 -> exactly 31 sampled think tokens then a forced </think>; the uncapped baseline never closed within 512 tokens). Native requires reasoning_parser="qwen3" and is a faster single pass, but its post-forced-close answer conditioning differs slightly from the HF backend, so scores can diverge in low-budget/truncated regimes; use it for speed, two_phase for exact HF parity.
"""

import os
_no_flashinfer = os.environ.get("MENAGERIE_VLLM_NO_FLASHINFER")
if _no_flashinfer and _no_flashinfer.lower() not in {"0", "false", "no", "off"}:
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import json
import sys


THINK_CLOSE = 248069  # id of "</think>"


def _load_llm(method):
    from vllm import LLM

    model_id = os.environ.get("MENAGERIE_VLLM_MODEL", "Qwen/Qwen3.5-4B")
    gmu = float(os.environ.get("MENAGERIE_VLLM_GMU", "0.5"))
    kwargs = dict(
        model=model_id,
        gpu_memory_utilization=gmu,
        language_model_only=True,
        max_model_len=4096,
        max_num_seqs=64,
    )
    if method == "native":
        kwargs["reasoning_parser"] = "qwen3"
    try:
        return LLM(enforce_eager=False, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[vllm_runner] enforce_eager=False failed ({exc}); retrying enforce_eager=True",
            file=sys.stderr,
            flush=True,
        )
        return LLM(enforce_eager=True, **kwargs)


def _trim_eos(ids, eos_ids):
    for idx, token_id in enumerate(ids):
        if token_id in eos_ids:
            return ids[:idx]
    return ids


def _answer_budget(max_new, mode):
    if isinstance(max_new, dict):
        return int(max_new[mode])
    return int(max_new)


def _run_no_think(llm, tok, prompt_ids, modes, max_new, temperature, eos_ids):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sps = [
        SamplingParams(
            temperature=temperature,
            max_tokens=_answer_budget(max_new, modes[i]),
            skip_special_tokens=False,
        )
        for i in range(len(prompt_ids))
    ]
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], sps)
    res = []
    for o in outs:
        ids = _trim_eos(list(o.outputs[0].token_ids), eos_ids)
        res.append({"answer": tok.decode(ids, skip_special_tokens=True), "think_tokens": 0, "forced_close": False})
    return res


def _run_native(llm, tok, prompt_ids, modes, max_new, think_budget, temperature, eos_ids):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sps = [
        SamplingParams(
            temperature=temperature,
            max_tokens=think_budget + _answer_budget(max_new, modes[i]),
            thinking_token_budget=think_budget,
            skip_special_tokens=False,
        )
        for i in range(len(prompt_ids))
    ]
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], sps)
    res = []
    for o in outs:
        ids = list(o.outputs[0].token_ids)
        if THINK_CLOSE in ids:
            ci = ids.index(THINK_CLOSE)
            think_tokens = ci
            ans_ids = _trim_eos(ids[ci + 1 :], eos_ids)
            forced = ci >= think_budget - 1
        else:
            think_tokens = len(ids)
            ans_ids = []
            forced = True
        res.append(
            {
                "answer": tok.decode(ans_ids, skip_special_tokens=True),
                "think_tokens": think_tokens,
                "forced_close": bool(forced),
            }
        )
    return res


def _run_two_phase(llm, tok, prompt_ids, modes, max_new, think_budget, temperature, eos_ids, close_ids):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp1 = SamplingParams(temperature=temperature, max_tokens=think_budget, skip_special_tokens=False)
    outs1 = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompt_ids], sp1)
    n = len(prompt_ids)
    res = [None] * n
    think_tokens = [0] * n
    forced = [False] * n
    conts = []
    cont_idx = []
    for i, o in enumerate(outs1):
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
        outs2 = llm.generate(conts, sp2)
        for j, o in zip(cont_idx, outs2):
            aids = _trim_eos(list(o.outputs[0].token_ids), eos_ids)
            res[j] = {
                "answer": tok.decode(aids, skip_special_tokens=True),
                "think_tokens": think_tokens[j],
                "forced_close": forced[j],
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

    from vllm import SamplingParams  # noqa: F401
    from vllm.inputs import TokensPrompt  # noqa: F401

    method = os.environ.get("MENAGERIE_VLLM_BUDGET", "two_phase")
    llm = _load_llm(method)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        os.environ.get("MENAGERIE_VLLM_MODEL", "Qwen/Qwen3.5-4B"),
        trust_remote_code=True,
    )
    close_ids = tok("</think>\n\n", add_special_tokens=False).input_ids
    eos_ids = set()
    if getattr(tok, "eos_token_id", None) is not None:
        eos_ids.add(tok.eos_token_id)
    proto.write(json.dumps({"ready": True, "method": method}) + "\n")
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
        think_budget = int(job["think_budget"])
        temperature = float(job.get("temperature", 0.0))
        max_new = job["max_new_tokens"]
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
            completions = _run_no_think(llm, tok, prompt_ids, modes, max_new, temperature, eos_ids)
        elif method == "native":
            completions = _run_native(llm, tok, prompt_ids, modes, max_new, think_budget, temperature, eos_ids)
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
            )
        proto.write(json.dumps({"completions": completions}) + "\n")
        proto.flush()


if __name__ == "__main__":
    main()
