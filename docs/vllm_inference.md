# vLLM Inference

Use the experiment template at
`templates/experiment/src/vllm_runner.py` for high-throughput text generation with
`Qwen/Qwen3.5-4B`. New experiment scaffolds receive a copy automatically. For an existing
experiment, copy that one file into the experiment's `src/` directory; it has no repository-local
imports.

The runner is for bulk generation, including runtime PEFT LoRA inference. Transformers remains
necessary for training, hidden-state/activation work, arbitrary forward passes, and any probability
readout that has not passed a backend-parity check.

## Install on the current RunPod

Keep vLLM isolated from a Transformers training environment because vLLM pins its own Torch build.
The current box has an RTX 6000 Ada (48 GB), driver 550.127, Python 3.12, and CUDA 12.8 toolkit. The
driver exposes CUDA 12.4, but NVIDIA's CUDA 12 minor-version compatibility permits the pinned CUDA
12.9 runtime on this driver. A real CUDA allocation and full Qwen3.5 model load have passed here.

```bash
uv venv --python 3.12 .venv-vllm
uv pip sync --python .venv-vllm/bin/python --torch-backend=cu129 requirements-vllm.lock.txt
uv pip check --python .venv-vllm/bin/python
```

`requirements-vllm.txt` is the small human-maintained input; `requirements-vllm.lock.txt` pins the
complete Linux dependency graph. Regenerate the lock deliberately with the command in its header
after changing the input, then repeat every parity gate below.

The resolved, validated core stack is:

- vLLM `0.24.0+cu129`
- Torch `2.11.0+cu129`
- Transformers `5.13.0`
- Qwen/Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`

If this is moved to a host with a different driver or GPU, rerun the CUDA and model smoke instead of
assuming minor-version compatibility. Do not replace the pinned wheel with a floating nightly. If a
future Qwen3.5 fix requires nightly vLLM, pin the exact vLLM commit index in the requirements file.

## Smoke test

The first launch downloads the 8.68 GiB checkpoint and compiles/caches FlashInfer sampling kernels.
On this machine the one-time kernel compile took roughly a minute. Later engine starts load the cached
checkpoint and compiled graph in seconds.

```bash
.venv-vllm/bin/python templates/experiment/src/vllm_runner.py \
  --smoke 8 \
  --output /tmp/qwen_vllm_smoke.jsonl \
  --thinking off \
  --greedy \
  --max-tokens 64
```

The runner adds the venv's `bin/` directory to `PATH` itself. Direct invocation therefore finds the
venv-provided `ninja`; activating the environment is optional. When imported, runner construction
also restores the caller's Python, NumPy, and Torch RNG states after vLLM initializes its seeded
in-process engine, so later procedural generation is not silently perturbed.

Two warm validation workloads currently measure about 1,921 aggregate tok/s for 512 very short
no-thinking completions and 2,126 aggregate tok/s for 128 mixed-length think@512 completions. The
second workload sampled 29,675 tokens in 13.96 seconds, including its forced-close continuations.
An exact repeat of the first workload reproduced 512/512 raw and cleaned token sequences. See
[`docs/compute_environment.md`](compute_environment.md) for the full benchmark definitions and the
warning against comparing them directly with the previous machine's HF measurements.

A representative integration smoke also used four frozen C48 tasks with its real system prompt,
two-stage think@192 protocol, code extractor, and grader. All four prompt-token counts exactly matched
the committed HF harness, and the unchanged parser accepted vLLM outputs. Candidate samples differed,
as they must be expected to when the backend RNG and kernels change.

## JSONL interface

Each input row needs a unique `id` and exactly one of `messages` or an already rendered `prompt`:

```json
{"id":"task-1","messages":[{"role":"system","content":"Return only Python code."},{"role":"user","content":"Write add(a, b)."}],"meta":{"split":"smoke"}}
{"id":"task-2","prompt":"<|im_start|>user\nAn exact pre-rendered prompt<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"}
```

Raw prompts are checked for the Qwen chat template's thinking/no-thinking suffix. The runner rejects
a channel mismatch by default. For a deliberately non-chat substrate, use `--allow-custom-prompts`
and document why the model-facing format is valid; the detected channel is stored in each output row.

Run all prompts together so vLLM can continuously batch them. Use `--n K` rather than a Python loop
around single samples:

```bash
.venv-vllm/bin/python experiments/EXPERIMENT/src/vllm_runner.py \
  --input experiments/EXPERIMENT/data/prompts.jsonl \
  --output experiments/EXPERIMENT/runs/completions.jsonl \
  --thinking off \
  --n 12 \
  --max-tokens 512 \
  --seed 71
```

The output contains exact token IDs, finish reasons, parent and effective per-sample seeds, thinking
and answer counts, injected-token accounting, and separate logical stage-one/stage-two prompt-token
counts (important when `n > 1` or forced continuations re-prefill a prefix). `OUTPUT.meta.json`
records the runner hash, pinned model revision, adapter hashes, engine/sampling arguments, the full
installed-package inventory, environment-lock digest, GPU/driver, Git state, and load/generation
throughput.

## Thinking modes

- `--thinking off` renders Qwen's explicit no-thinking chat channel.
- `--thinking natural --max-tokens N` lets the model close its reasoning naturally within `N` total
  generated tokens.
- `--thinking budget --thinking-budget B --answer-max-tokens A` reproduces the repository's
  historical two-stage protocol: sample up to `B` reasoning tokens, inject the exact
  `</think>\n\n` token sequence when necessary, then generate an answer capped at `A` tokens.
- `--shuffle-thinking` deterministically shuffles the retained reasoning tokens before the forced
  close. It exists only for the established content-control experiment.

Do not substitute vLLM's native `thinking_token_budget` for an existing two-stage result. It has
different semantics and runner-version constraints. Treat it as a separate intervention if added.

## LoRA adapters

Pass a standard PEFT adapter directory at engine startup:

```bash
.venv-vllm/bin/python experiments/EXPERIMENT/src/vllm_runner.py \
  --input experiments/EXPERIMENT/data/prompts.jsonl \
  --output experiments/EXPERIMENT/runs/adapter_completions.jsonl \
  --adapter /external/path/to/lora_adapter \
  --thinking off \
  --n 12
```

The wrapper validates `adapter_config.json`, the Qwen3.5 base identity, LoRA rank, unsupported DoRA,
bias, `modules_to_save`, rank patterns, and alpha patterns before allocating the model. It sizes vLLM's
`max_lora_rank` from the adapter rather than using vLLM's insufficient rank-16 default. Existing
repository adapters are normally rank 32 and may be applied to the bf16 base without merging even
though they were trained with QLoRA.

Adapter execution must still pass a behavior/parity gate after an adapter is restored or regenerated
on a fresh machine. A successful base-model smoke does not prove that an arbitrary adapter's module
mapping is correct.

The current host has passed a plumbing-only LoRA test: a generated zero rank-8 adapter loaded through
vLLM and gave 4/4 token-identical greedy outputs versus the base model. This proves the text-backbone
module mapping and runtime request path load; it does not replace the required rank-32 behavioral
gate for a real repository adapter.

## Defaults and tuning

The template starts with conservative research defaults:

- bf16, tensor parallel 1, text-only `language_model_only=True`;
- max model length 16,384 rather than the native 262,144;
- GPU memory utilization 0.90;
- 128 concurrent sequences and 32,768 batched tokens;
- CUDA graph capture capped at `max_num_seqs`, avoiding the Qwen GDN cache-line assertion;
- MTP/speculative decoding off;
- prefix caching off.

Sampling validation runs before model allocation. It rejects values vLLM would otherwise fail on or
silently normalize (including nonzero temperatures below 0.01), and `resolved_sampling` records the
effective greedy/non-greedy parameters actually passed to vLLM.

Qwen3.5's hybrid GDN/Mamba prefix-cache `align` mode is experimental in vLLM 0.24. Enable
`--enable-prefix-caching` only when prompts actually share long prefixes, then re-run parity checks.
MTP is primarily a low-concurrency latency optimization and may reduce high-concurrency throughput;
it is intentionally absent from the initial wrapper.

Tune `--max-model-len`, `--max-num-seqs`, and `--max-num-batched-tokens` on the real workload. Record
the chosen values from the metadata sidecar. Throughput measurements must exclude model-load time but
include both stages and every sampled token for budgeted thinking.

## Scientific parity rules

- vLLM and Transformers use different kernels and RNG implementations. The same numeric seed does
  not imply identical samples. Run every arm and matched-compute baseline in an experiment through
  the same backend.
- Keep the chat channel used during adapter training. Switching a thinking-trained adapter to the
  no-thinking template (or the reverse) is a format confound.
- Preserve `</think>` and other special-token IDs until task parsing is complete. Do not rely only on
  detokenized text.
- Qwen3.5's tokenizer treats `<|im_end|>` as EOS, but its pinned HF generation config stops after the
  following newline at `<|endoftext|>`. The runner explicitly uses the latter stop ID, retains it in
  raw stage token IDs and sampled-token accounting, and removes only that terminal token from the
  semantic completion. This preserves the established Transformers answer boundary.
- Compare compute with `n_sampled_tokens`, not `n_completion_tokens`; the latter includes injected
  force-close tokens while the former includes all actually sampled stage-one and stage-two tokens.
- Before migrating a result-bearing harness, gate exact prompt-token equality, answer parsing,
  greedy task behavior, termination, and any requested log-probability readout. Sampled text is not
  expected to match Transformers byte-for-byte.
- A mixed HF/vLLM comparison is invalid even if temperatures and seeds have the same names.

Primary upstream references: [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B),
[vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/),
[Qwen3.5 recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md),
[LoRA support](https://docs.vllm.ai/en/stable/features/lora/), and
[reproducibility](https://docs.vllm.ai/en/latest/usage/reproducibility/).
