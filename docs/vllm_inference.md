# vLLM Inference

Use the experiment template at
`templates/experiment/src/vllm_runner.py` for high-throughput text generation with
`Qwen/Qwen3.5-4B`. New experiment scaffolds receive a copy automatically. For an existing
experiment, copy that one file into the experiment's `src/` directory; it has no repository-local
imports.

The runner is for bulk generation, including runtime PEFT LoRA inference. Transformers remains
necessary for training, hidden-state/activation work, arbitrary forward passes, and any probability
readout that has not passed a backend-parity check.

The Menagerie benchmark's persistent-server sibling runner lives at
`benchmarks/menagerie/harness/vllm_runner.py`; it shares this pinned venv and the same two-phase
thinking semantics.

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

Also validated: RTX 4090 24 GB, WSL2, CUDA-13 driver, and no CUDA toolkit. The same pinned lock works
there out of the box: FlashInfer sampling uses the precompiled `flashinfer-cubin` wheels, and the
remaining kernels are Triton-JIT, so `nvcc` is not required. First-launch JIT warnings are expected.

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
records the runner hash, pinned model revision, adapter hashes, requested engine/sampling arguments,
resolved CUDA-graph geometry and mode, the full installed-package inventory, environment-lock
digest, GPU/driver, Git state, and load/generation throughput.

## Thinking modes

- `--thinking off` renders Qwen's explicit no-thinking chat channel.
- `--thinking natural --max-tokens N` lets the model close its reasoning naturally within `N` total
  generated tokens.
- `--thinking budget --thinking-budget B --answer-max-tokens A` reproduces the repository's
  historical force-close protocol: the first call may naturally close reasoning and answer within
  `B` sampled tokens; otherwise the runner injects the exact `</think>\n\n` sequence and generates
  a continuation capped at `A` tokens.
- `--shuffle-thinking` deterministically shuffles the retained reasoning tokens before the forced
  close. It exists only for the established content-control experiment.

Runner schema 4 directly enforces `A` only on the forced continuation. A naturally closed first
call can contain `A` or more answer tokens while ending normally. Any experiment claiming a shared
answer allowance must conservatively classify `n_answer_tokens >= A` as an answer-limit contact (in
addition to a stage-two `length` finish) and reject/escalate that tier. Do not infer a nonbinding
answer allowance from `truncated=false` alone.

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

**WARNING — verified silent no-op (2026-07-10).** vLLM 0.24 runtime LoRA does
NOT apply Qwen3.5-4B adapters trained with PEFT on `AutoModelForCausalLM`: an
in-process probe (one engine, same prompts, greedy, `lora_request` on vs off)
produced token-identical outputs, and two *different* trained rank-32 adapters
produced byte-identical outputs across 1,200 eval generations. Mechanism: the
adapter names its tensors `base_model.model.model.layers.*` while the served
composite checkpoint keeps its text stack under `model.language_model.layers.*`;
vLLM's LoRA weight mapping matches nothing and raises no error. The earlier
"plumbing-only" test below could never catch this: a ZERO adapter is expected
to produce base-identical outputs whether or not it is applied. The required
gate for any adapter arm is therefore an ON-vs-OFF behavioral diff with a REAL
trained adapter (identical outputs = fail), e.g.
`experiments/qwen35_4b_gauntlet_breadth_round1`'s `lora_probe3` pattern.

Until the mapping is fixed, deploy installs as **merged composite
checkpoints**: `experiments/qwen35_4b_gauntlet_breadth_round1/scripts/merge_adapter.py`
merges LoRA deltas into the full composite by explicit name mapping
(W += B·A·α/r on `model.language_model.layers.*`), and the result loads via
`--model-id` (menagerie) or the experiment runner's `model_override`. Note a
text-only `merge_and_unload` checkpoint does NOT load (vLLM's Qwen3.5 class
requires the composite config); merge into the composite.

The current host has passed a plumbing-only LoRA test: a generated zero rank-8 adapter loaded through
vLLM and gave 4/4 token-identical greedy outputs versus the base model. This proves the runtime
request path loads; per the warning above it does NOT prove weight application, and the same host
subsequently failed the on-vs-off gate with a real adapter.

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

Sharing the GPU with a training run usually means lowering `--gpu-memory-utilization` (for example,
`0.5`) so vLLM leaves room on the card. Two knobs then bind on a 24 GB card (both verified on the
RTX 4090):

- `--max-model-len`: at utilization 0.5 the default 16,384 no longer fits the KV budget — engine
  init fails, reporting an estimated max around ~12k. Drop to 8,192 or what the workload needs.
- `--max-num-seqs`: Qwen3.5-4B is a hybrid model with Mamba/linear-attention cache blocks, and the
  available Mamba blocks scale roughly with the memory budget (≈93 at utilization 0.5 on 24 GB); if
  vLLM reports that `--max-num-seqs` exceeds the available Mamba cache blocks, the reusable runner
  logs a loud warning and auto-clamps `max_num_seqs` plus its tied CUDA-graph capture cap for the
  single retry.

Tune `--max-model-len`, `--max-num-seqs`, and `--max-num-batched-tokens` on the real workload. Record
the chosen values from the metadata sidecar. Throughput measurements must exclude model-load time but
include both stages and every sampled token for budgeted thinking.

For long-context batches, `max_num_seqs` must also fit the live KV-token capacity—not merely pass
engine initialization. After constructing the engine and before generation, read its resolved cache
token count and block size, round the largest prompt-plus-generation reservation up to that block
size, and require:

```text
min(number_of_logical_sequences, max_num_seqs) * rounded_sequence_reservation
    <= live_kv_cache_tokens
```

Record every term and the remaining margin in the preflight/receipt. vLLM 0.24 can otherwise admit
too many long sequences, then preempt a running request by freeing its cache blocks and resetting its
computed-token count. With prefix caching disabled, the evicted prefix is recomputed from zero. GPU
utilization can remain near 100% while aggregate sampled-token throughput and wall time deteriorate,
so “the card is busy” is not evidence that concurrency is efficient. Treat a concurrency change as
a new inference protocol on Ada and use the same capacity-fit value for every compared arm at a rung.

Cache fit and CUDA-graph coverage are separate gates. In vLLM 0.24's balanced mode, specifying only
an off-stride `max_cudagraph_capture_size` does **not** guarantee that endpoint is captured: the
implicit list is `1, 2, 4` and then multiples of eight. For example, requested maxima 15 and 19
resolve to 8 and 16, and larger decode batches fall back to no CUDA graph. For an off-stride
`max_num_seqs`, pass an explicit increasing `cudagraph_capture_sizes` list whose last value equals
the requested maximum (for example `[1, 2, 4, 8, 15]` or `[1, 2, 4, 8, 16, 19]`). After engine
construction, fail closed unless the resolved list and maximum match exactly, the resolved mode has
full decode CUDA graphs, and the live KV-capacity check still passes. Record the resolved mode and
sizes in the receipt; graph allocation can itself change available cache capacity.

Do not infer useful compute from sampled-token throughput alone when cache preemption is possible.
Recomputed prefixes consume forward work without increasing the emitted-token counter. A deliberately
overcommitted scheduler can therefore look faster in emitted tokens per second while doing more
unrecorded model work, which is unsuitable for a matched-compute comparison unless preemptions and
actual computed tokens are measured.

### Reproducibility boundary on Ada

On fixed hardware and software, explicit request seeds and `async_scheduling=False` make an
otherwise fixed vLLM call reproducible, but they do not make different batch shapes or token budgets
common-random-number continuations on the RTX 6000 Ada. In a long-context calibration, changing only
`max_tokens` changed 32/64 sampled prefixes after the first `max_num_seqs=32` scheduling wave; the
same boundary effect occurred with asynchronous scheduling both enabled and disabled. vLLM's true
batch-invariant mode requires NVIDIA compute capability 9.0 or newer, while Ada is 8.9.

Treat prompt order, shard membership, `n`, token budget, scheduler mode, and concurrency as part of
the inference protocol. Freeze and checksum a prespecified sharding/batch plan; do not treat rows
from different shapes as token-paired or replay-equivalent, and never combine them opportunistically.
A prespecified mixture of fixed shards remains one valid protocol, and a completed tier remains a
valid independent protocol; neither is a common-random-number continuation of another shape.

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
