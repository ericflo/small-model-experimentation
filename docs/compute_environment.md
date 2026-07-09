# Compute Environment & Throughput

How this repo runs on the current box, and how to keep inference fast. Keep this current when the
environment or model changes.

## Current box

- RunPod Linux container with one **RTX 6000 Ada (48 GB)**, driver **550.127.05** (CUDA 12.4
  interface), CUDA 12.8 toolkit, Python 3.12.3, and `uv 0.9.0`.
- The high-throughput environment is **`.venv-vllm`** (gitignored), created with `uv`. Its validated
  core stack is vLLM **0.24.0+cu129**, torch **2.11.0+cu129**, and transformers **5.13.0**. CUDA 12
  minor-version compatibility works on this driver: a CUDA allocation and full Qwen3.5 load/generate
  smoke both passed. Install the complete pinned graph from `requirements-vllm.lock.txt`.
- There is not yet a shared Transformers training `.venv` on this fresh pod. Create it separately
  from `.venv-vllm`; vLLM pins Torch and should not be allowed to rewrite the training stack.
- The Transformers throughput, OOM, and training measurements later in this document came from the
  previous single **RTX 4090 (24 GB), WSL** environment. They remain recovery/reference evidence, not
  measurements of this RunPod.

## Model

All experiments use **Qwen/Qwen3.5-4B** (`model_type: qwen3_5`) — never an older model. It is a hybrid
**linear-attention + multimodal** model; load text-only via
`AutoModelForCausalLM(trust_remote_code=True, dtype=torch.bfloat16)` (`Qwen3_5ForCausalLM`, ~8.4 GB
VRAM). Think-token ids: `<think>`=248068, `</think>`=248069 (differ from Qwen3-4B; verify per model).

## vLLM bulk-generation path

For ordinary text generation and runtime LoRA evaluation, use the single-file experiment template
at `templates/experiment/src/vllm_runner.py`; new experiment scaffolds copy it automatically. Setup,
CLI examples, thinking-budget semantics, parity gates, and backend-mixing rules live in
[`docs/vllm_inference.md`](vllm_inference.md).

Validated current behavior:

- pinned Qwen3.5 revision loads in text-only bf16 using about **8.0 GiB VRAM**;
- cached checkpoint weight loading is about **2 seconds**; a full cached engine startup is about
  **16 seconds** for the 4k/32-sequence smoke configuration;
- the first launch builds FlashInfer's sampling extension and may take roughly a minute; the result
  is cached under `/root/.cache/flashinfer`;
- the wrapper's no-thinking and legacy two-stage forced-thinking smoke paths both complete and write
  exact token/accounting metadata;
- runtime LoRA loading passed with a generated zero rank-8 adapter and produced 4/4 token-identical
  greedy outputs versus the base model; this validates plumbing, not a real adapter's behavior;
- invoke all prompts together and use `n` sampling so continuous batching can eliminate HF's
  slowest-sequence batch gate.

Warm measurements on this RTX 6000 Ada, excluding engine load but including all generation stages:

| workload | sampled completions | sampled tokens | wall time | aggregate tok/s |
| --- | ---: | ---: | ---: | ---: |
| no-think, 128 prompts × `n=4` (very short outputs; median 6 sampled tokens) | 512 | 4,480 | 2.33 s | 1,921 |
| think@512 legacy two-stage, 64 prompts × `n=2` (median 161 thinking tokens; 15 forced closes) | 128 | 29,675 | 13.96 s | 2,126 |

The no-think command was repeated and all **512/512 raw and cleaned token sequences were identical**.
On four frozen C48 smoke tasks, vLLM prompt-token counts matched the committed HF harness exactly
(4/4), and the unchanged C48 answer parser successfully consumed vLLM's two-stage outputs. As
expected, sampled candidates were not byte-identical across backends.
These are infrastructure benchmarks, not a fair head-to-head with the historical 4090 HF table below:
the GPU, prompts, and termination distribution differ. Benchmark the frozen prompts from a target
experiment before projecting its wall time.

The default runner uses a 16,384-token model limit, 128 sequences, 32,768 batched tokens, and 0.90
GPU memory utilization. It deliberately leaves MTP and experimental GDN/Mamba prefix caching off.
Tune against the real workload and retain the metadata sidecar. Base generation and synthetic LoRA
plumbing are validated; an actual repository rank-32 adapter still requires its separate behavioral
parity gate after restoration.

## Historical Transformers fast path (previous 4090 only)

The recipe below describes the previous WSL/torch-cu130 environment. Do not run it verbatim on the
current RunPod; there is no shared Transformers environment here yet, and its Torch/CUDA pins must be
chosen and validated separately when an internals-dependent experiment needs one.

The qwen3_5 fast path needs **both** `flash-linear-attention` (Triton; `is_flash_linear_attention_available()`)
and **`causal-conv1d`** (`is_causal_conv1d_available()`). If either is missing the model prints
*"The fast path is not available … Falling back to torch implementation"* and runs the linear-attention
layers in a slow pure-PyTorch path.

Verify:

```python
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
assert is_causal_conv1d_available() and is_flash_linear_attention_available()
```

Rebuild `causal-conv1d` from source if the venv is reset (no prebuilt wheel matches torch cu130; ~9 min compile):

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=8 \
  uv pip install --python .venv/bin/python --no-build-isolation --no-binary :all: causal-conv1d
```

## Historical Transformers throughput is batch-bound, not VRAM-bound

Per-sequence decode is ~12–13 tok/s **regardless of batch** — the hybrid arch has high per-token
kernel-dispatch overhead, and `causal-conv1d` mainly speeds prefill (only ~+16% on decode). The lever
is **batch size**: aggregate throughput scales nearly linearly and VRAM stays well under 24 GB
(measured, sdpa attention, fast path on):

| sequence length | batch | aggregate tok/s | peak VRAM |
| ---: | ---: | ---: | ---: |
| 512 | 32 | ~410 | 10.3 GB |
| 512 | 64 | ~780 | 12.3 GB |
| 512 | 96 | ~1180 | 14.2 GB |
| 512 | 128 | ~1460 | 15.7 GB |
| 2048 | 32 | ~410 | 12.0 GB |
| 2048 | 48 | ~670 | 13.7 GB |
| 2048 | 64 | regressed | 15.5 GB |

Recommended batch sizes (used in `experiments/qwen35_4b_thinking_budget_scaling/scripts/run.py`):
**no-think 96, ≤512-budget 64, ≤1024 48, ≥2048 40.** Always set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and use OOM-resilient batch subdivision (the
runtime auto-halves a batch that OOMs). Going much past bs≈48 for long (≥2048-token) generations gave
diminishing/negative returns, so cap long-sequence batches there.

## Historical OOM, CUDA corruption, and recovery (WSL2/4090)

Hard-won failure knowledge — read before launching anything training-scale:

- **An OOM during training can corrupt the CUDA context persistently** (observed: batch-16 ×
  maxlen-640 QLoRA OOM). Symptom: `RuntimeError: CUDA driver error: device not ready` that
  SURVIVES process kills and long waits. The corruption has a **size threshold**: small ops and
  model loading work, but any training-scale forward+loss (the large cross-entropy over the
  ~150K vocab) crashes. Recovery without a WSL restart: train *under* the threshold
  (per-device batch 2 + gradient accumulation — slow but works). The clean fix is
  `wsl --shutdown` (user action).
- **Do not launch a second GPU job while one is running.** A competing job is how the OOM →
  corruption sequence gets triggered; single-tenant the 4090.
- **OOM can masquerade as "device not ready"** under `expandable_segments`; to surface the
  real error, re-run without `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or with
  `CUDA_LAUNCH_BLOCKING=1`.
- **torch 2.12 raises generation OOM as `torch.AcceleratorError`**, NOT
  `torch.cuda.OutOfMemoryError` — any batch-halving/OOM-resilience `except` clause must catch
  BOTH (`(torch.cuda.OutOfMemoryError, getattr(torch, "AcceleratorError", ...))`), or the
  designed graceful degradation becomes a hard crash. Every experiment that copied the shared
  `gen_lib.py` before 2026-07-08 has the narrow catch.
- **Set the alloc config in the orchestrator, not the shell**: launching a pipeline without
  `expandable_segments:True` OOM'd a harvest that runs fine with it at the same batch size.
  `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` at the top of
  the experiment's run.py makes the requirement un-forgettable.
- **`cmd | grep | tee | tail` reports the LAST pipe stage's exit code** — a crashed run looks
  like exit 0 to the harness. Use `set -o pipefail` in every launch command.
- **`torch.cuda.empty_cache()` inside an OOM-recovery path can ITSELF raise
  `torch.AcceleratorError`** and turn designed batch-halving into a hard crash (observed
  2026-07-08 under memory pressure from long prompts × 1024-token thinking KV at batch 48). Wrap
  the CLEANUP call too — retry once after a pause and never let it propagate (see
  `_safe_empty_cache` in `experiments/qwen35_4b_hypothesize_verify_wall/src/gen_lib.py`), and
  drop think-eval batch to ~32 (16 at budget ≥2048) when prompts run long.

### Memory-safe large-vocab logits

The full-vocab logits tensor is the recurring OOM source (~150K vocab):

- **Teacher-forced sequence logprobs:** never `log_softmax` the whole sequence in float32
  (OOMs at batch 8 × ~800 tokens on 24 GB). Keep model logits bf16 and log-softmax in float32
  over **sequence chunks** (~128 tokens), gathering target-token logprobs per chunk — see
  `experiments/qwen35_4b_code_confidence/scripts/eval_code_conf.py::mean_logprobs`.
- **Scoring N candidate continuations** (ranking ops/answers): vectorize the logprob gather
  into one GPU sync — a loop with `.item()` per candidate is ~10× slower (32 syncs). And with
  a long prompt prefix, score a few candidates per forward: 32 ops × ~1200 tokens × 152K vocab
  × float is a 23 GB tensor.
- **Single-token P(True) judging can still OOM on long code prompts.** Even with
  `logits_to_keep=1`, long HumanEval candidate+judge prefixes at batch 16 OOMed on the 24 GB
  4090 during `qwen35_4b_code_confidence`; the same run succeeded with
  `--judge-batch-size 1`. Treat P(True) judge batch size as a memory knob distinct from
  generation batch size, and save an intermediate logprob-scored artifact before the judge pass.

## Historical training (QLoRA) throughput

- Reference recipe: QLoRA r32/α64 on short episodes at **batch 16, grad-accum 1, maxlen 384 →
  ~1 s/step** (a 4k-episode × 2-epoch run ≈ 8 min). The same run at batch 1 is ~2.8 h. After
  the corruption workaround (batch 2 + accumulation) expect ~6–7 s/step.
- **Verify a training launch before trusting it:** the edit actually applied (grep the
  script), GPU memory rising to training scale (~15 GB), and step time in the expected range.
- PeftModel (adapter) forward is ~1.5× slower than base — budget adapter-heavy eval sweeps
  accordingly; at n=80 a think-mode eval is ~55 min.
- Training time scales with dataset size; do **not** queue several long trains ahead of the
  cheap headline eval — interleave, or put the longest train last-and-optional.
- Adapters are ~180 MB each: keep them OUT of the working tree (the validator scans the
  filesystem, not just git, and GitHub hard-fails >100 MB files). Store externally and declare
  in `reports/artifact_manifest.yaml` (see `docs/artifact_policy.md`).

## Known limitations / future levers

- **HF batch-gating:** `model.generate` advances a whole batch until the *slowest* sequence finishes,
  so a few long-thinking items stall the batch. This was the dominant cost in the thinking-budget sweep.
- **torch.compile:** would target the per-token dispatch overhead, but the supported
  `cache_implementation="static"` + compile path **fails** on qwen3_5 with
  `KeyError('linear_attention')` — the static cache does not model the hybrid linear-attention layers.
  Adopting it would need a custom hybrid-cache-compatible compile path; not worth it for now.
- **vLLM coverage boundary:** bulk generation and the two-stage thinking protocol are now validated.
  Keep Transformers for activations, training, and arbitrary forward passes; validate LoRA and any
  log-probability readout separately before migrating a result-bearing harness.
