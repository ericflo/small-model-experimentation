# Preregistration: verified-macro exact CUDA-graph vLLM rerun

Status: **frozen before this experiment's first GPU call**. No external artifact root for this
experiment has been created and no model output has been generated or imported.

## Question and scope

Can the frozen verified-macro v2 smoke reach an efficient, interpretable termination regime when
vLLM uses both live-KV-safe concurrency and explicit CUDA-graph capture sizes ending at the exact
active batch width?

This is an inference-validity follow-up. It can reject or select a termination rung and, only after
a complete eligible K=12 matrix, run the inherited semantic interface smoke. It cannot show that
macros beat matched-compute sampling.

## Frozen provenance boundary

The closest near-duplicate is `qwen35_4b_verified_macro_capacity_fit_rerun`. Exact SHA-256 copies
of `tasks.json`, `demonstrations.json`, `libraries.json`, `prompt_manifest.json`, `macro_domain.py`,
and `model_harness.py` are listed in `data/source_provenance.json`. No predecessor JSONL, metadata,
receipt, selection, decoded output, parse result, or score is imported or decision-eligible.

The 12 smoke-v2 prompt identities are frozen and previously prompted under other scheduler
protocols; they are not model-unseen. Each exact-geometry invocation samples them anew.

## Model and backend

- Sole model: `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Sole inference backend: experiment-local vLLM runner schema 4, SHA-256
  `3a98eb8da787054aded56a1ec3fd040ee2edaacc7d0694b4aec5a0309488774a`.
- Transformers inference, adapters, prefix caching, asynchronous scheduling, and mixed backends are
  forbidden.

## Exact CUDA-graph intervention

All non-capture inference settings are inherited from the capacity-fit follow-up:
`max_model_len=65536`, `gpu_memory_utilization=0.9`,
`max_num_batched_tokens=32768`, bf16, tensor parallel 1, and no prefix caching.

The registered rung mapping is:

| Thinking budget | `max_num_seqs` | Explicit capture sizes | Required resolved maximum |
| ---: | ---: | --- | ---: |
| 49,152 | 19 | 1, 2, 4, 8, 16, 19 | 19 |
| 61,440 | 15 | 1, 2, 4, 8, 15 | 15 |

After engine construction the runner reads
`llm_engine.vllm_config.compilation_config`. It must observe exactly the registered list and
maximum, a resolved `decode_mode` of `FULL`, and `has_full_cudagraphs=true`. The top-level mode may
be `FULL`, `FULL_DECODE_ONLY`, or `FULL_AND_PIECEWISE` only when those semantic predicates hold.
Any normalization, truncation, missing active-width coverage, eager/piecewise-only mode, or mismatch
aborts before generation and is an infrastructure failure.

The live KV gate independently rounds the worst prompt plus full generation reserve to the live
cache block size and requires
`min(logical_sequences,max_num_seqs) * rounded_reserve <= kv_cache_size_tokens`.

## Sampling and phase geometry

- Temperature 0.6, top-p 0.95, top-k 20, seed 2701.
- Thinking allowances: 49,152 then 61,440; forced-close answer allowance 512.
- Probe: fresh base K=4, termination-only and never promotable.
- Selectable matrix: fresh base K=12 plus fresh designed-ceiling K=12 at one rung.
- One invocation constructs one engine, runs one phase, writes one receipt last, and exits.

Start with the independent 49k probe. If it is inadequate, the independent 61k probe is authorized.
If a probe passes, it authorizes a new K=12 base arm at that same rung. Only an adequate K=12 base
authorizes the designed arm. An inadequate base or designed arm rejects the whole rung immediately;
no row crosses a rung.

## Content-blind decision rule

A probe or K=12 arm is termination-adequate only when:

- unresolved reasoning-boundary contact rate is strictly below 5%;
- answer-limit contact rate is strictly below 5%; and
- exact periodic-loop contact rate is at most 25%.

Selection may use token counts, finish metadata, and token IDs only for the frozen exact-period
detector. It may not decode text, parse answers, inspect correctness, or load hidden examples. The
first complete adequate rung is the only selectable matrix. Terminal 61k failure writes an explicit
`pass:false, selected:null` result.

## Semantic gate

Only after both selected K=12 receipts and the full lower-rung history re-verify may the analyzer
decode or grade. The inherited smoke gate requires parse rate at least 0.5 in each arm, valid macro
candidates on at least two reuse tasks, and designed reuse oracle coverage no lower than base.
Passing licenses a separately preregistered matched-compute capability experiment; it is not itself
a capability-gain claim.

## Throughput interpretation

Report sampled tokens divided by generation seconds, excluding model load and including both
thinking stages. A faster exact-geometry probe supports the operational compilation hypothesis but
is not a clean causal estimate: Ada scheduling can change sampled trajectories and termination
lengths across batch geometries. A tie or slowdown preserves the result and weakens the hypothesis.

## Artifact and stop rules

The only permitted root is
`/workspace/large_artifacts/qwen35_4b_verified_macro_exact_cudagraph_rerun/scientific_smoke_v1`
or a safe absolute override through `QWEN35_MACRO_EXACT_CUDAGRAPH_ARTIFACT_ROOT`. Both predecessor
roots are forbidden. Preflight-only is the sole resumable partial state; all other partial or unknown
files fail closed. No automatic budget or arm advance occurs.
