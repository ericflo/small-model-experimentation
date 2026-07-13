# Qwen3.5-4B verified-macro capacity-fit vLLM rerun

**Status:** finished

Status: **the capacity-fit 49,152-token K=4 probe is termination-inadequate. The fresh 61,440-token
K=4 probe passed its live-capacity preflight, then was manually stopped before any result after a
CUDA-graph capture-geometry problem was identified. Its preflight-only artifact is preserved; no
decoded or scored content has been inspected, and there is no macro claim.**

## Research program

- Primary program: `operator_and_skill_inventories`.
- Direct parent: `qwen35_4b_verified_macro_long_context_rerun`.
- Closest near-duplicate: the direct parent. This is a separate experiment because it uses a new
  scheduler geometry, fresh external namespace, and independent stop rule. Its bytes may never be
  pooled with or promoted from the parent's `max_num_seqs=64` diagnostics.

## Question

The preceding long-context attempt showed that a 768-token cap was plainly too small, then reached
another setup boundary at much larger contexts. Did those later observations reflect a
verified-macro limitation, or did `max_num_seqs=64` ask vLLM's scheduler to support more simultaneous
near-65k contexts than the live KV cache could hold?

This follow-up tests the latter possibility before drawing any semantic conclusion. It freezes
capacity-fit concurrency at 19 sequences for a 49,152-token thinking allowance and 15 for 61,440,
then verifies the actual vLLM cache capacity and block rounding after engine construction and before
generation. Only a complete, termination-adequate K=12 base/designed matrix may be decoded and
graded.

The 12 `smoke-v2` tasks are the **frozen, previously unscored v2 smoke matrix under an independent
capacity-fit protocol**. They are not newly generated or model-unseen: the stateless model received
these prompt identities during predecessor diagnostics. No predecessor output is imported, read,
pooled, or scored here.

## Frozen protocol

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Backend: only the experiment-local vLLM runner, SHA-256
  `fd9972bdcb3a9e8b9841b45ed8e2849017a6e80b601e924817cdaaa5144b8782`, byte-identical to
  the direct parent.
- Engine: `max_model_len=65536`, `gpu_memory_utilization=0.9`,
  `max_num_batched_tokens=32768`, prefix caching off, asynchronous scheduling off.
- Rungs: `(thinking_budget, max_num_seqs) = (49152, 19), (61440, 15)`.
- Sampling: temperature 0.6, top-p 0.95, top-k 20, answer allowance 512, run seed 2701.
- Probe: fresh base K=4, termination-only. Probe rows are never K=12 evidence.
- Selectable matrix: fresh base K=12 plus fresh designed-ceiling K=12 at one rung.
- Storage: a new fail-closed external root, receipts written last, and a tracked checksum catalog.

At the 995,328-token cache capacity measured on the predecessor engine, the longest frozen designed
prompt gives conservative block-rounded requirements of 963,984 tokens at 49k and 945,360 at 61k.
Those are planning numbers, not assumed facts: each invocation must pass the same calculation using
the newly constructed engine's live `kv_cache_size_tokens`, block size, model context, and rendered
prompt lengths.

## Setup with `uv`

From the repository root, create the pinned vLLM environment only if it is absent:

```bash
uv venv --python 3.12 .venv-vllm
uv pip sync --python .venv-vllm/bin/python --torch-backend=cu129 requirements-vllm.lock.txt
uv pip check --python .venv-vllm/bin/python
```

Run the model-free gates before reserving the GPU:

```bash
.venv-vllm/bin/python -m unittest discover \
  -s experiments/qwen35_4b_verified_macro_capacity_fit_rerun/tests -v
.venv-vllm/bin/python \
  experiments/qwen35_4b_verified_macro_capacity_fit_rerun/scripts/run.py --validate
```

## GPU runbook

Coordinate with the owner of any existing GPU process before launching. Each command below creates
exactly one vLLM engine, invokes `generate_vllm_batch` once for one experiment phase, commits one
bundle, and exits. The budget wrapper may make a stage-one vLLM call and a stage-two continuation
call for forced closures within that phase. The
default root is
`/workspace/large_artifacts/qwen35_4b_verified_macro_capacity_fit_rerun/scientific_smoke_v1`;
`QWEN35_MACRO_CAPACITY_FIT_ARTIFACT_ROOT` may point to another absolute, nonsymlinked root that does
not contain or overlap the predecessor root.

The commands below record the frozen protocol that produced this experiment's artifacts. **Do not
resume the interrupted 61k probe in this experiment.** Correcting the resolved CUDA-graph capture
geometry changes the inference protocol and therefore requires fresh rows in a separate follow-up
namespace rather than filling this preflight-only bundle.

Start only with:

```bash
.venv-vllm/bin/python \
  experiments/qwen35_4b_verified_macro_capacity_fit_rerun/scripts/run.py \
  --phase probe --budget 49152
```

If its returned termination audit is adequate, run the two **new** K=12 arms:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_capacity_fit_rerun/scripts/run.py \
  --phase base --budget 49152
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_capacity_fit_rerun/scripts/run.py \
  --phase designed --budget 49152
```

If the probe or either K=12 arm is rejected by the registered termination gate, move to the next
rung with a new probe; do not reuse any 49k row:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_capacity_fit_rerun/scripts/run.py \
  --phase probe --budget 61440
```

Run `base` and `designed` at 61k only if that probe passes, using the same command shape as above.
The runner rejects out-of-order phases, an already terminal history, partial/unknown external files,
runtime drift within a rung, and any capacity audit that does not fit.

After a passing K=12 matrix has been selected, and only then, run:

```bash
.venv-vllm/bin/python \
  experiments/qwen35_4b_verified_macro_capacity_fit_rerun/scripts/analyze.py
```

The analyzer re-verifies every receipt and recomputes the complete lower-rung termination history
before decoded text is parsed or inspected and before hidden task data is loaded. A terminal 61k
rejection writes `pass:false` with no selected bundle;
that history is a setup boundary, not a negative macro result.

## Capacity-fit 49k result

The fresh base-only K=4 probe at think@49,152 passed the live scheduler-capacity check before
generation: max-seqs 19 required 963,072 block-rounded tokens from 997,888 live KV-cache tokens,
leaving 34,816 tokens of headroom with 528-token blocks. Its receipt and tracked catalog verify
against the external bytes.

Termination nevertheless remained inadequate. All 48 samples ended by stage-one length, were
force-closed, and contacted the reasoning boundary. Token-ID-only periodicity classified 37 as
exact periodic loops and left 11 unresolved; 9 answer stages reached their limit. These rates fail
all three registered thresholds. The probe produced 2,364,643 sampled tokens in 5,012.451 seconds,
or 471.754 tokens/s. No decoded output, parser result, correctness value, or hidden example was
inspected.

The strictly capacity-fit run was 19.6% slower than the predecessor's max-seqs-64 diagnostic
(471.754 versus 586.471 sampled tokens/s), but that comparison is confounded by CUDA-graph geometry
and cannot isolate the cost of avoiding cache recomputation. The predecessor initially admitted all
48 logical sequences and had a captured width-48 decode graph. Its 995,328-token cache could hold
only about 20,736 total context tokens per sequence at that width, after which cache pressure could
preempt requests and recompute prefixes. The max-seqs-19 run bounded that cache demand, but vLLM
resolved the runner's requested graph maximum of 19 to `[1, 2, 4, 8, 16]`; decode widths 17--19
therefore ran without a CUDA graph. The runs differed by only 1,977 sampled tokens (less than 0.1%)
while wall time differed by 24.2%. Sampled-token throughput also excludes any recomputed prefix
tokens, so the faster overcommitted diagnostic does not establish lower actual model compute.

Thus cache-safe concurrency and capture-aware throughput tuning are separate requirements. The 49k
rung remains rejected by its termination gate, and its throughput comparison remains infrastructure
diagnosis only, not evidence for or against verified macros.

## Clean 61k interruption

The fresh base-only K=4 probe at think@61,440 started under the frozen max-seqs-15 protocol. Before
generation, its live preflight passed: 15 active sequences required 950,400 block-rounded tokens
from 997,888 live KV-cache tokens, leaving 47,488 tokens of headroom with 528-token blocks. The
7,053-byte preflight is preserved at
`smoke_budget_probes/think_61440/base.preflight.json`, SHA-256
`a2a3ef1f4ba9e68909374460030bc947712f10a488870a0db1bf081e368b8a5a`.

While the model call was running, a source-level audit found that the runner requested
`max_cudagraph_capture_size=15`, but vLLM 0.24 constructs default capture sizes at 1, 2, 4, and then
multiples of 8. It therefore resolved the effective maximum to 8, so decode widths 9--15 used no
CUDA graph. The process was manually stopped before rows, runner metadata, or a last-written receipt
existed. The tracked catalog consequently and correctly records this bundle as `preflight_only`.
There is no 61k termination count, timing result, decoded output, score, or scientific outcome to
infer from the interrupted call.

## Decision rule

Termination selection uses token counts, finish metadata, and token-ID periodicity, never decoded
answers, parser outcomes, correctness, or hidden examples. A probe/arm is adequate only when:

- unresolved reasoning-boundary contact rate is strictly below 5%;
- answer-limit contact rate is strictly below 5%; and
- detected periodic-loop contact rate is at most 25%.

The first rung with an adequate probe and both adequate K=12 arms is selected. Semantic smoke then
requires parse rate at least 0.5 in each arm, valid macro candidates on at least two reuse tasks, and
designed reuse oracle coverage no lower than base. Smoke is an interface gate, not a capability-gain
claim; matched-compute sampling remains mandatory for any later scientific claim.

## Artifacts

- `data/source_provenance.json`: exact copied-file and predecessor-boundary provenance.
- `data/prompt_manifest.json`: ordered content-blind prompt identities; no model output.
- `reports/preregistration.md`: frozen hypotheses, branch logic, and thresholds.
- `reports/design_review.md`: adversarial pre-launch review and required fixes.
- `analysis/scientific_smoke_49k_termination_audit.json`: receipt-bound, content-blind 49k
  capacity, termination, exact-period, and throughput audit.
- `analysis/scientific_smoke_artifact_catalog.json`: complete 49k bundle plus the preserved,
  checksum-bound 61k preflight-only interruption state.
- `src/scientific_artifacts.py`: external bundle, receipt, catalog, and selection validator.
- `scripts/run.py`: one-engine/one-phase vLLM runner.
- `scripts/analyze.py`: content-blind termination finalizer and post-gate semantic analyzer.
- `reports/artifact_manifest.yaml`: external-storage and regeneration contract.
