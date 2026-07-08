# Compute Environment & Throughput

How this repo runs on the current box, and how to keep inference fast. Keep this current when the
environment or model changes.

## Box

- Single **RTX 4090 (24 GB)**, **WSL**. CUDA toolkits at `/usr/local/cuda-13.2` (and `/usr/local/cuda` → 13.2).
- Standard project **`.venv`** (gitignored), created with `uv` (no sudo; system Python lacks `ensurepip`).
- Stack: **torch 2.12.1+cu130**, **transformers 5.12.1** (native `qwen3_5`), datasets, accelerate,
  bitsandbytes, matplotlib, pandas, **flash-linear-attention**, **causal-conv1d 1.6.2.post1**.

## Model

All experiments use **Qwen/Qwen3.5-4B** (`model_type: qwen3_5`) — never an older model. It is a hybrid
**linear-attention + multimodal** model; load text-only via
`AutoModelForCausalLM(trust_remote_code=True, dtype=torch.bfloat16)` (`Qwen3_5ForCausalLM`, ~8.4 GB
VRAM). Think-token ids: `<think>`=248068, `</think>`=248069 (differ from Qwen3-4B; verify per model).

## Fast path (REQUIRED — do not regress)

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

## Throughput is batch-bound, not VRAM-bound

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

## OOM, CUDA corruption, and recovery (WSL2)

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

## Training (QLoRA) throughput

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
- **vLLM:** continuous batching would remove the gating and add faster kernels — the biggest potential
  win — but vLLM pins its own torch and qwen3_5 (new hybrid arch) support is unverified. If pursued, do
  it in a **separate** venv so the working HF env (above) is not disturbed.
