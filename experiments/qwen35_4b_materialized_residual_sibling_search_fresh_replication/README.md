# Qwen3.5-4B Materialized Residual Sibling Search Fresh Replication

**Status:** in-progress · since 2026-07-13 · corrected model-free construction passed; mechanics implementation, audit, publication, separate lock, and model run remain sealed

This separately registered recovery replication preserves the parent's frozen
materialized-residual science while regenerating tasks, request IDs, and
sampling seeds and hardening durable generation receipts. The fresh task
construction is frozen; no model call is authorized.

## Research Program

- Primary: `structured_execution_and_compilers`.
- Secondary targets after construction review: `evidence_conditioned_selection`,
  `interpretability_and_diagnostics`, and `test_time_reasoning_budget`.
- Scientific parent:
  `qwen35_4b_materialized_residual_sibling_search`.
- Immediate reason for a new experiment: the parent is sealed by a terminal
  `STARTED` transaction after 52 rows returned only in memory. Reusing its task
  identities or sampling seeds would resample a terminal invocation.

## Question

On fresh exact-depth-three tasks, can Qwen3.5-4B complete useful two-operation
residuals when an external interpreter materializes every candidate first
operation's public consequences, and can that all-sibling explorer beat
name-only, token-preserving semantic derangement, and taskwise matched-compute
ordinary sampling?

## Hypothesis

Concrete public state-to-target relations should reduce a depth-three inverse
problem to a depth-two suffix problem. If that mechanism is real, symmetric
all-24 materialized completion should improve proposal coverage and
visible/probe-only selected hidden accuracy beyond both representation controls
and conservative sampled-token/logical-token first-over sampling. Cheap
no-think top-four ranking remains secondary and cannot veto the all-24 test.

## Setup

- Model: only `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16.
- Backend: the pinned experiment-local vLLM runner for every model arm; no
  backend mixing and no training in this experiment.
- Scientific design: copy the parent's frozen arms, controls, stop gates, and
  matched-compute estimands without outcome-dependent relaxation.
- Freshness: regenerate every procedural task under seed block
  `2026072700`--`2026072709`, change every request-ID namespace, and use the new
  mechanics sampling seed `2026072702`. Paired arms share identities only
  within this fresh successor.
- Transaction repair: `STARTED -> generate -> durable GENERATED bundle ->
  re-read/authenticate -> COMPLETE`. Failed quarantine bytes remain sealed and
  downstream analysis requires `COMPLETE`.
- Termination receipt: independently lock model EOS `248044`, tokenizer
  `<|im_end|>` EOS `248046`, `ignore_eos=true`, and explicit stop token
  `[248044]` before any model request.
- Hidden boundary: no hidden output may influence prompts, stopping, resource
  matching, pool construction, selected IDs, or escalation. No benchmark
  content is read or used.

## Controls and metrics

- Primary treatment: one strict two-operation suffix from all 24 materialized
  sibling states.
- Mechanism controls: all-24 name-only siblings and all-24 task-hash-deranged
  state/target alignments with the same token multiset.
- Baseline: candidate-blind full-program sampling matched taskwise at the first
  conservative sampled-token and logical-model-token overtake points.
- Dominance references: exact public viability and exhaustive CPU enumeration.
- Primary metrics: hidden-correct proposal coverage and visible/probe-only
  selected hidden accuracy.
- Claim-grade bar: untouched confirmation must beat every registered primary
  comparator under the parent's paired McNemar/Holm family and bootstrap gate.

## Run

The historical identity-only scaffold receipt is preserved at
`runs/scaffold/summary.json`. Two independent reviews now authorize this
model-free construction command:

```bash
.venv/bin/python experiments/qwen35_4b_materialized_residual_sibling_search_fresh_replication/scripts/run.py --stage smoke
```

Full mechanics is intentionally absent and unauthorized until its crash-safe
implementation, request/seed-overlap proof, adversarial audit, pushed code, and
separately pushed clean lock all exist.

## Results

The corrected construction passed in 93.6 seconds and froze 264 tasks: 24
mechanics, 48 qualification, and 192 confirmation. It found 3,526 eligible
exact-depth-three behaviors, preserved the 88/88/44/44 live-sibling balance,
and achieved 0.952 simulated compound power at the registered alternative.
The authenticated parent comparison found zero shared task IDs, identity-free
public instances, all-mechanics prompts, or terminal materialized prompts.
Finite-DSL reuse is reported rather than hidden: 56 functions, 41 concrete
triples, and 181 suffixes. The real tokenizer receipt records model EOS 248044,
tokenizer EOS 248046, `ignore_eos=true`, and explicit stop `[248044]`. All 44
experiment tests pass. No benchmark was read, no model was loaded, and no model
call occurred. This is readiness/provenance evidence, not a capability result.
The manifest's locked runner hash exactly matches pushed construction commit
`e43c701e`; a post-construction guard now verifies the frozen manifest/summary
instead of rewriting them on rerun.

## Interpretation

The parent incident and this model-free construction change no belief about
residualization. The successor now has a fresh, auditable substrate on which to
obtain the first durable model test without replaying a terminal draw.
Scientific arms and outcome gates remain frozen.

## Knowledgebase Update

- Program evidence updated: no model evidence yet.
- Program backlog updated: fresh construction passed; mechanics safety is next.
- Claim ledger updated: no; no model result exists.

## Artifacts

- `configs/default.yaml`
- `data/procedural/manifest.json`
- `runs/scaffold/summary.json`
- `runs/smoke/summary.json`
- `runs/smoke/publication_receipt.json`
- `idea_intake.md`
- `reports/artifact_manifest.yaml`
- `reports/report.md`
- `src/`
- `scripts/`
- `tests/`
