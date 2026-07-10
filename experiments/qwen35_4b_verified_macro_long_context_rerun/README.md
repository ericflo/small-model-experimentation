# Qwen3.5-4B Verified Macro Invention Long-Context Rerun

**Status:** ongoing anti-censoring rerun; no fresh task output has been scored. The corrected
plan-given interface passed 16/16 records at think@16,384, overturning the parent's apparent
low-budget interface limit. The fresh base-smoke arm then proved that 16,384 was still binding for
induction (131/144 unresolved contacts and 60/144 answer truncations), so those rows were excluded
and think@32,768 also rejected from content-blind finish/count metadata plus the preregistered
token-ID periodicity test: 81/144 exact loops, 63/144
unresolved contacts, and 37/144 answer-limit contacts. The max-seqs-64 K=4 probe at think@49,152
then finished with 34/48 exact loops, 14/48 unresolved contacts, and 13/48 answer-limit contacts,
but amendment 12 had already made it diagnostic-only from pre-result cache-capacity evidence. Fresh
long-rung selection moved to the capacity-fit follow-up, whose 61k attempt was stopped before a
receipt after an audit found that its implicit CUDA-graph list did not cover the active batch
width. A separate exact-capture follow-up then rejected fresh 49k and 61k probes even though both
live-KV and exact-graph gates passed. Its terminal selector records `pass=false` and no selected
budget, so no K=12 matrix, semantic analysis, or macro claim is authorized.

## Research program

- Primary: `operator_and_skill_inventories`
- Secondary: `structured_execution_and_compilers`, `benchmark_generalization`, and
  `test_time_reasoning_budget`
- Parent: [`qwen35_4b_verified_macro_invention`](../qwen35_4b_verified_macro_invention/)
  at commit `1c8c5bbb81d2a67618891597205ceb2f40f498d8`
- Intake: [idea_intake.md](idea_intake.md)
- Preregistration: [reports/preregistration.md](reports/preregistration.md)
- External scientific-smoke durability amendment:
  [reports/preregistration_amendment_10.md](reports/preregistration_amendment_10.md)
- Fail-early full-run audit amendment:
  [reports/preregistration_amendment_11.md](reports/preregistration_amendment_11.md)
- Capacity-fit scheduler amendment:
  [reports/preregistration_amendment_12.md](reports/preregistration_amendment_12.md)
- Adversarial design review: [reports/design_review.md](reports/design_review.md)
- Frozen-data provenance: [data/source_provenance.json](data/source_provenance.json)

## Question

With enough context and a nonbinding, explicitly calibrated reasoning allowance, can a verified
abstraction library built only from prior solved programs improve visible-only selected accuracy on
fresh behaviorally true-depth-5 programs beyond matched-compute sampling over the original
primitives?

## Why this rerun exists

The parent did not reach the scientific comparison. Its budgeted plan-given attempt gave every
completion only 768 thinking tokens and a 128-token answer: all 16 samples hit the thinking cap and
12/16 hit the answer cap. A later no-thinking transcription probe removed truncation but measured a
different low-compute interface. Neither path establishes that verified macros help or fail under a
properly provisioned induction protocol.

This follow-up changes the inference envelope, not the hypothesis. It keeps the parent's frozen
construction corpus, macro libraries, never-prompted v2 smoke, never-prompted full tasks, scientific
arms, K values, analyzer logic, hidden-label boundary, and confirmatory thresholds. It replaces the
single 768-token assumption with content-blind termination calibration and whole-matrix escalation.

## Setup

- **Only model:** `Qwen/Qwen3.5-4B` at repository-pinned revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **Only inference backend:** the experiment-local [src/vllm_runner.py](src/vllm_runner.py) under
  `.venv-vllm`; no Transformers inference and no backend mixing.
- **Engine envelope:** `max_model_len=65536`, with exact prompt-plus-generation preflight.
- **Scientific reasoning ladder:** budgeted thinking at 16,384, 32,768, 49,152, then 61,440
  tokens after amendments 3--5 separated exact loops and extended the setup envelope.
- **Answer allowance:** 512 tokens at every rung. Forced-close stage 2 is directly capped; a
  naturally closed stage-1 answer reaching 512 also rejects the rung under amendment 7.
- **Reasoning-cap accounting:** amendment 9 counts a forced intervention or a close in the final
  stage-1 slot as contact. An earlier natural `</think>` followed by a stage-1 length finish is an
  answer-restart diagnostic, not evidence that reasoning exhausted its allowance; the fresh answer
  still has to clear the unchanged 512-token rule.
- **Largest context guard:** the frozen train-only proposal prompt is 3,478 tokens. Its largest-rung
  total is 65,432, and the CPU regression guard preserves the remaining 104-token headroom.
- **Substrate:** the byte-identical contamination-free procedural corpus copied from the parent:
  800 construction programs, 12 unseen v2 smoke tasks, and 120 unseen full tasks (80 motif-reuse,
  40 primitive-multiset-matched no-reuse).
- **Sampling:** macro arms K=12; base K=24 for the matched-token sample-more curve.
- **Primary metric:** visible-only selected hidden-all accuracy on the reuse split.
- **Oracle-only metric:** whether any sampled candidate passes every hidden case.

The frozen `tasks.json` SHA-256 is
`82fbbd57e26fd392aa8f30ec6f26d370dc08dd78b3279bed6ee2e2174aea5073`; the frozen
`libraries.json` SHA-256 is
`a2ae3663753a3a0d0c9614a5d7c1d250506c74fd7879e11e99b66f5c1e43f865`.

## Anti-censoring gates

1. The original train-only calibration ladder used four deterministic plan-given records sampled
   16 times each. Amendment 4 selected think@16,384 using finish/count metadata plus the frozen
   token-ID periodicity test: below-5%
   unresolved-cap and answer-limit-contact rates, at most 80% p99 productive thinking use, and at
   most 50% p99 answer use. Generated text and correctness were unavailable to selection.
2. At that rung, a disjoint train-only plan-given gate ran 16 records with four samples each. It
   passed 16/16 record coverage; 63/64 samples were strict valid macro-using surfaces, all 12 cap
   contacts were classified by the frozen exact-token periodic-tail detector, none remained
   unresolved, and no answer truncated.
3. Fresh induction has its own workload-conditioned ladder. A completed inadequate arm can reject
   a lower rung immediately; every remaining arm is recorded as skipped, and all lower rows remain
   diagnostic and unscored. A rung is selectable only when the complete base/designed matrix is
   termination-adequate at one shared budget. If either completed K=12 arm rejects the 32,768
   matrix, higher rungs first use all 12 base prompts at non-scored K=4 to locate a viable
   allowance; the selected rung is then rerun from scratch at the original complete K=12 matrix.
4. Raw stage-1 length, forced intervention, final-slot boundary contact, earlier-close answer
   restart, and fresh-answer limit contact are reported separately. Runtime selection and offline
   analysis use the same amendment-9 definition.
5. Exhausting 61,440 is an inconclusive setup failure, never a capability result. Full generation
   remains blocked until an uncensored smoke matrix passes the original semantic gates.
6. Amendment 12 found before the 49k receipt that max-seqs 64 overcommits the measured 995,328-token
   KV cache at the long rungs. That completed probe is decision-ineligible regardless of its
   termination result. Fresh 49k/61k probes and any complete matrix move to the separate
   capacity-fit follow-up at max-seqs 19/15 respectively.

The full decision remains the parent's conjunction: `mined` must beat base sampling, the
non-callable `mined_hint`, and matched random composite libraries under the registered effect-size,
paired-interval, compute, macro-use, and reuse-specificity thresholds. The `qwen_ranked` verdict
remains separate.

## Run

All commands use the uv-managed vLLM environment.

CPU preparation and tests:

```bash
.venv-vllm/bin/python -m unittest discover -s experiments/qwen35_4b_verified_macro_long_context_rerun/tests -v
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_long_context_rerun/scripts/run.py --prepare
```

Before resuming a legacy repository-local scientific-smoke cache, stop the old runner and perform
the model-free staged migration. The first command validates a staging copy, writes receipts and a
deterministic catalog, atomically installs the external tree, and deliberately preserves the local
source for review. The second command revalidates byte identity before removing only the canonical
local tier/probe directories. Cleanup is resumable: any remaining local bundles need only be an
exact verified subset of an external inventory that may already have advanced:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_long_context_rerun/scripts/run.py --migrate-scientific-artifacts
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_long_context_rerun/scripts/run.py --migrate-scientific-artifacts --remove-local-scientific-artifacts
```

Scientific matrix and termination-probe bundles then live under
`/workspace/large_artifacts/qwen35_4b_verified_macro_long_context_rerun/scientific_smoke_v1/`.
`QWEN35_MACRO_SCIENTIFIC_ARTIFACT_ROOT` may select a byte-identical absolute copy on another host.
Do not hand-copy individual JSONL files: each bundle is valid only as a preflight alone or as the
exact preflight/rows/metadata triplet plus its last-written receipt. Unknown files, symlinks,
partials, or catalog drift fail closed before model allocation or row interpretation.
A narrow crash window is self-healing under the shared stage lock: when unchanged protocol bytes
and exact-valid external bundles prove that a cataloged preflight merely gained its last-written
receipt, the catalog is reconciled before model allocation. Source drift, unknown bundles, and
rows/metadata partials are never reconciled.

Calibrate the rung, run the heldout interface gate, and run the fresh smoke:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_long_context_rerun/scripts/run.py --smoke
```

Run the full matrix only after all gates pass:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_long_context_rerun/scripts/run.py --full
```

Full generation follows preregistration amendments 8 and 11 and does not create
`runs/full/*.jsonl`.
Canonical raw artifacts live at
`/workspace/large_artifacts/qwen35_4b_verified_macro_long_context_rerun/full/think_<budget>/<arm>/shard_<index>/`.
Base uses 20 six-task shards at K=24; every macro arm uses 10 twelve-task shards at K=12. Each
shard therefore contains exactly 144 completions and preserves the frozen 2:1 reuse mix. A final
shard is reusable only when its directory contains exactly `preflight.json`, `rows.jsonl`,
`runner.meta.json`, and the last-written `receipt.json`, and every ordered prompt, identity, byte
size, and SHA-256 validates. Re-running `--full` first scans and exact-validates every existing
final in the active rung, including downstream arms, and only then resumes missing work. `.tmp-*`
directories from interruptions are inventoried but never salvaged.

A persistent sibling `.full.lock` serializes scientific migration, smoke, and full processes. It
uses nonblocking `flock` and is acquired before frozen-data checks, smoke replay, proposal work,
model construction, or external-root/catalog mutation. Before vLLM loads, the runner
also rejects repository-local full raw rows, symlinks, unknown budgets/arms/shards, malformed
finals, partial Qwen ensembles, and stale protocol/data/catalog bindings. A malformed final
`shard_<index>` fails closed and is never overwritten. Inspect it, move the
whole directory to an explicitly named quarantine location outside the canonical arm directory,
then rerun the entire shard. Do not copy individual rows or samples back. At a successful selected
rung, `analysis/full_artifact_catalog.json` is the tracked logical pointer; no raw promotion copy
is made. Its schema-2 inventory is checkpointed from startup onward, including diagnostic receipts,
nonreusable temporary directories, rejected rungs, and a selected-null setup-inconclusive finish.
It binds the exact config, hidden-label-bearing tasks, post-proposal libraries, smoke gates,
selection, source files, plan, receipts, and protocol identities. Offline analysis revalidates that
binding and recomputes selected-arm termination adequacy before scoring any completion.

Analyze an existing full run without loading the model:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_verified_macro_long_context_rerun/scripts/analyze.py --run full
```

## Results

The inference repair has already changed the diagnosis:

- max-seqs-64 calibration at think@16,384: 3/64 unresolved contacts, 0 answer truncations,
  productive p99 11,629 tokens;
- independent plan-given interface: 16/16 records covered, 63/64 strict valid macro-using samples,
  0 unresolved contacts, and 0 answer truncations;
- fresh base smoke at the same allowance: 144/144 raw cap contacts, 13 exact periodic tails,
  131/144 unresolved contacts, and 60/144 answer-limit contacts;
- fresh base smoke at think@32,768: 144/144 forced interventions at the reasoning boundary,
  81 exact periodic tails, 63/144 unresolved contacts, 37/144 answer-limit contacts, and no
  earlier-close answer restarts.

The max-seqs-64 K=4 diagnostic at 49,152 also forced all 48 reasoning traces closed: 34 were exact
loops, 14 remained unresolved, and 13 answers hit 512 tokens. It sampled 2,366,620 tokens in
4,035.356 seconds (586.47 tokens/s). Amendment 12 had made it selection-ineligible before its
receipt existed because 48 admitted long sequences required 2,433,024 cache tokens against the
995,328-token engine cache with prefix caching off.

These are workload/scheduler-calibration failures, not macro results. The 16k arm consumed 2,391,698 sampled
tokens in 2,138.606 seconds (1,118.34 tokens/s); 32k consumed 4,739,527 in 5,971.182 seconds
(793.73 tokens/s). All three diagnostics were excluded before parsing or correctness inspection.
The capacity-fit follow-up then rejected a fresh 49k probe but exposed a second scheduler defect:
its requested capture maximums 19/15 resolved only through widths 16/8. Its 61k attempt was stopped
before a receipt, with no reusable rows. The separate exact-capture follow-up covered width 19
exactly at 49k while fitting 963,072 cache tokens into 996,864 live tokens, but still rejected the
probe content-blind: 38/48 exact loops, 10/48 unresolved contacts, and 6/48 answer-limit contacts.
It sampled 2,363,163 tokens in 4,809.081 seconds (491.396 tokens/s). At 61k, vLLM resolved
`[1, 2, 4, 8, 15]` exactly in FULL mode and fit 950,400 required cache tokens into 997,888 live
tokens with 47,488 of headroom. The fresh probe nevertheless failed all three thresholds: 40/48
exact loops, 8/48 unresolved contacts, and 4/48 answer-limit contacts, at 397.688 sampled tokens/s.
The terminal selection is `pass=false` with no selected budget. No decoded or scored content
informed either decision, and no K=12 matrix or macro comparison was authorized. The next
registered follow-up should stop increasing context and test a symmetric loop-control protocol in
a separate experiment while preserving the unresolved-contact and answer-limit gates.

## Artifacts

- `data/`: byte-identical frozen parent inputs plus explicit source provenance.
- `configs/`: the registered ladder, context envelope, sampling, gates, and scientific constants.
- `runs/`: train-only calibration and interface vLLM rows with exact runner metadata. Scientific
  smoke/probe rows use the canonical external root above; full rows use the external shard root.
- `analysis/`: machine-readable gates/verdicts, budget selections, and deterministic external
  checksum catalogs. Smoke and full per-task output is compact: hashes, aggregate counts/tokens,
  and the selected program/grades only; reasoning text and raw token arrays stay external.
- `reports/`: preregistration, design review, final report, and artifact manifest.
- `src/vllm_runner.py`: the single-file, pinned vLLM inference wrapper used by every arm.
