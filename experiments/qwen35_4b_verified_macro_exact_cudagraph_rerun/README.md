# Qwen3.5-4B verified-macro exact CUDA-graph vLLM rerun

**Status:** finished

Status: **terminal setup negative. Both fresh exact-capture K=4 probes are complete and
termination-inadequate. No K=12 matrix, semantic analysis, or macro result is eligible, and no
decoded or scored content has been inspected.**

## Research program

- Primary: `operator_and_skill_inventories`.
- Secondary: `structured_execution_and_compilers` and `test_time_reasoning_budget`.
- Closest near-duplicate: `qwen35_4b_verified_macro_capacity_fit_rerun`.

This is a separate experiment because explicit CUDA-graph capture shapes change the vLLM inference
protocol on Ada. No predecessor output may be imported, pooled, promoted, or used to skip a rung.

## Question

The capacity-fit rerun correctly limited concurrent near-65k contexts to the live KV cache, but its
runner supplied only `max_cudagraph_capture_size=max_num_seqs`. Pinned vLLM's implicit sparse list
resolved a requested maximum of 19 to 16 and 15 to 8. Does explicitly capturing through the exact
active widths remove that avoidable eager-dispatch region while preserving the scientific gates?

This experiment does not assume the answer is yes. CUDA-graph warmup, padding, memory, and changed
termination trajectories can erase or reverse a throughput benefit.

## Frozen protocol

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Backend: only the experiment-local vLLM runner, schema 4, SHA-256
  `3a98eb8da787054aded56a1ec3fd040ee2edaacc7d0694b4aec5a0309488774a`.
- 49k rung: thinking budget 49,152, `max_num_seqs=19`, capture sizes
  `[1,2,4,8,16,19]`.
- 61k rung: thinking budget 61,440, `max_num_seqs=15`, capture sizes
  `[1,2,4,8,15]`.
- Engine: `max_model_len=65536`, bf16, tensor parallel 1,
  `gpu_memory_utilization=0.9`, `max_num_batched_tokens=32768`, prefix caching off,
  asynchronous scheduling off.
- Sampling: temperature 0.6, top-p 0.95, top-k 20, answer allowance 512, seed 2701.
- Probe: fresh base K=4 and termination-only. Selectable matrix: fresh base and designed K=12 at
  the first adequate rung.

Each constructed engine must independently pass two checks before generation:

1. live block-rounded KV demand fits the exposed cache capacity; and
2. vLLM's resolved capture-size list and maximum exactly equal the registered values, with
   `decode_mode=FULL`, full CUDA graphs enabled, and the full active width covered.

## Setup with `uv`

From the repository root, create the pinned vLLM environment only if it is absent:

```bash
uv venv --python 3.12 .venv-vllm
uv pip sync --python .venv-vllm/bin/python --torch-backend=cu129 requirements-vllm.lock.txt
uv pip check --python .venv-vllm/bin/python
```

Model-free gates:

```bash
.venv-vllm/bin/python -m unittest discover \
  -s experiments/qwen35_4b_verified_macro_exact_cudagraph_rerun/tests -v
.venv-vllm/bin/python \
  experiments/qwen35_4b_verified_macro_exact_cudagraph_rerun/scripts/run.py --validate
```

## GPU runbook

Do not launch until the current GPU owner and an independent design reviewer give GO. Every command
below creates one engine, invokes one experiment phase, commits one bundle, and exits; it never
automatically advances.

Start with the fresh 49k probe:

```bash
.venv-vllm/bin/python \
  experiments/qwen35_4b_verified_macro_exact_cudagraph_rerun/scripts/run.py \
  --phase probe --budget 49152
```

If and only if its content-blind termination gate is adequate, run the new K=12 base arm:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_exact_cudagraph_rerun/scripts/run.py \
  --phase base --budget 49152
```

If and only if that K=12 base arm is also termination-adequate, run the designed arm:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_exact_cudagraph_rerun/scripts/run.py \
  --phase designed --budget 49152
```

The completed 49k probe authorized one separately fresh 61k probe, which was run as:

```bash
.venv-vllm/bin/python \
  experiments/qwen35_4b_verified_macro_exact_cudagraph_rerun/scripts/run.py \
  --phase probe --budget 61440
```

The 61k probe also failed, so the ladder is now terminal. The following analysis command was never
authorized because no K=12 matrix was selected:

```bash
.venv-vllm/bin/python \
  experiments/qwen35_4b_verified_macro_exact_cudagraph_rerun/scripts/analyze.py
```

## Decision rule

Termination selection uses counts and finish metadata plus token IDs only for the frozen periodic
loop detector. It never decodes or scores while selecting a rung. Adequacy requires unresolved cap
contacts below 5%, answer-limit contacts below 5%, and periodic-loop contacts at most 25%.

Semantic smoke remains an interface gate: parse rate at least 0.5 in each K=12 arm, valid macro
candidates on at least two reuse tasks, and designed reuse oracle coverage no lower than base. A
positive smoke does not beat matched-compute sampling and cannot support a macro capability claim.

## 49k probe result

The completed fresh K=4 probe passed both infrastructure gates. Its live engine exposed 996,864 KV
tokens in 528-token blocks; 19 block-rounded 50,688-token reservations required 963,072 tokens and
left a 33,792-token margin. vLLM resolved the registered full-decode graph list exactly as
`[1,2,4,8,16,19]`, including the active width 19.

All 48 samples contacted the frozen reasoning boundary and required force-close. The token-only
periodicity audit found 38 exact loops; 10 contacts remained unresolved and six answer stages
reached their limit. The corresponding rates---79.17%, 20.83%, and 12.50%---fail all three
registered thresholds, so 49k was rejected before decoding or scoring. At that checkpoint the
fresh exact-capture 61k probe was authorized; its terminal result is recorded below.

The probe sampled 2,363,163 tokens in 4,809.081 seconds (491.396 sampled tokens/s), including late
JIT warnings in elapsed generation time. That is 4.16% faster than the closest implicit-capture
capacity-fit probe and 16.21% slower than the invalidly overcommitted max-seqs-64 diagnostic. These
cross-protocol timings are descriptive rather than a causal benchmark. The result is termination
and inference evidence only, with no macro claim.

## 61k probe and terminal result

The separately fresh K=4 probe at 61,440 passed both infrastructure gates. Its live engine exposed
997,888 KV tokens in 528-token blocks; 15 block-rounded 63,360-token reservations required 950,400
tokens and left a 47,488-token margin. vLLM resolved full-decode CUDA graphs exactly as
`[1,2,4,8,15]`, covering the active width.

All 48 samples again contacted the frozen reasoning boundary and required force-close. The
content-blind token-ID audit found 40 exact periodic loops, eight unresolved contacts, and four
answer stages at their limit. Their rates---83.33%, 16.67%, and 8.33%---all fail the registered
thresholds. The run sampled 2,951,995 tokens in 7,422.886 generation seconds (397.688 sampled
tokens/s), after a 100.294-second load.

The terminal selection therefore records `pass=false` and `selected_thinking_budget=null`. No K=12
base or designed arm exists; decoding, scoring, and semantic analysis remain prohibited. This is a
preserved negative about the setup: exact CUDA-graph coverage and a 61k reasoning allowance did not
produce termination-adequate samples. It is not a negative macro-capability result.

## Artifacts

- `idea_intake.md`: novelty, closest duplicate, controls, and falsifier.
- `reports/preregistration.md`: frozen geometry, branch logic, and inspection boundary.
- `reports/design_review.md`: adversarial risks and required controls.
- `data/source_provenance.json`: exact copies, derived-runner hash, and noninheritance boundary.
- `src/vllm_runner.py`: single-file vLLM wrapper with explicit/resolved capture validation.
- `src/scientific_artifacts.py`: fail-closed external receipts, catalogs, and selection.
- `scripts/run.py`: one-engine/one-phase orchestrator and live preflight.
- `scripts/analyze.py`: three-pass termination-then-semantics analyzer.
- `reports/artifact_manifest.yaml`: fresh external namespace and regeneration contract.
- `analysis/scientific_smoke_49k_termination_audit.json`: content-blind capacity, graph,
  termination, and timing audit for the rejected 49k probe.
- `analysis/scientific_smoke_61k_termination_audit.json`: content-blind audit for the rejected
  terminal 61k probe and fail-closed selection.
