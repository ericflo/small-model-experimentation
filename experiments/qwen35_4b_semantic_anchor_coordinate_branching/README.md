# Qwen3.5-4B Semantic-Anchor Coordinate Branching

**Status:** in-progress · since 2026-07-13 · adversarial design and model-free smoke precede model mechanics

This experiment tests the last mechanism-specific bridge left by the native
Jacobian line: put a named hypothesis token inside real reasoning and replace
its context-local J coordinates with each candidate donor state.

## Research Program

- Primary: `interpretability_and_diagnostics`.
- Secondary: `test_time_reasoning_budget`, `structured_execution_and_compilers`,
  and `evidence_conditioned_selection` only after their gates pass.
- Positive anchor: `qwen35_4b_jacobian_transport_control_replication`.
- Negative boundary: `qwen35_4b_jacobian_counterfactual_branching`.
- Closest near-duplicate: `qwen35_4b_context_local_jacobian_clamp`; it used an
  explicit selected token in a short synthetic lookup, not a hypothesis anchor
  after native thought or balanced candidate continuations.

## Question

Did arbitrary-position additive steering fail because J is not a native
reasoning control, or because the replicated effect specifically requires an
explicit semantic token and donor-coordinate replacement?

## Hypothesis

The fixed early J dictionary carries a context-local concept state only when a
token supplies a semantic workspace address. After 512 native thinking tokens,
all 24 J coordinates at a forced `Candidate first-operation alias:` token are
replaced with clean coordinates captured from another alias in the identical
context. This should transport a task-randomized, separately computed operation
consequence and then seed candidate-specific reasoning. Literal text anchors
are the deployable baseline; full donor activation is the causal upper bound.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Backend: Transformers bf16 SDPA for every arm. Activation mechanics use
  batch-one full recomputation; backend mixing is forbidden.
- Lens: byte-identical 24-concept context lens, layers 4--8, with no refit,
  layer sweep, or amplitude sweep.
- Fresh substrate: exact-depth-two procedural list transformations, split into
  four label-sealed mechanics, 24 qualification, and 48 untouched confirmation
  tasks, with complete fingerprint disjointness from ancestors.
- Prompt mapping: the 12 public one-token aliases are assigned to operations by
  a frozen task-balanced permutation, so token identity cannot stand in for an
  operation across tasks.
- Anchor bank: a task-ID-selected valid source alias is the surface recipient;
  every other public alias supplies a clean, context-local donor state.
- Primary mechanics endpoint: apply the supplied candidate's prompt-local
  operation to `[3,-1,2,0]` (`k=2`) and select a task-randomized one-token label
  for the resulting list. Task correctness is never loaded.
- Arms: source, literal target text, full donor activation, all-24 donor J,
  mean donor J, frozen additive J, two exact live-bf16 non-J controls, rotated
  wrong donor, and ordinary concept-logit-lens replacement.
- Capability baselines: literal text, clean shared-prefix continuations, and a
  frozen master pool of fully independent sample-more traces at sampled-token
  and total-forward-compute parity.
- Hidden-label boundary: mechanics sees only public task prompt fields and its
  diagnostic mapping. Qualification writes every output and resource receipt
  before grading. Confirmation stays absent until qualification passes.

## Run

Model-free smoke:

```bash
.venv/bin/python experiments/qwen35_4b_semantic_anchor_coordinate_branching/scripts/run.py --stage smoke
```

Model mechanics remains fail-closed until the reviewed implementation is
committed and its hashes are anchored in the configuration.

## Results

Model-free smoke passes. The frozen lens is byte-identical and rank 24 at every
layer 4--8; all 12 diagnostic operation results are distinct; and 76 fresh task
behaviors have zero overlap with 1,046 readable ancestor fingerprints. The
original smoke boundary had six tests; 15 now pass after implementation. No
model result exists. Mechanics must prove a
randomized computed consequence—not merely alias writing—with live numeric
controls before any correctness-scored continuation.

## Interpretation

Pending. The terminal decision will distinguish literal text control, full-state
transport, direct-only J writing, computed J consequence transport, and additive
anchor transport rather than collapsing them into one pass/fail label.

## Knowledgebase Update

- Program evidence: update after a model-stage decision.
- Program backlog: this is the one final context-local bridge already recorded.
- Claim ledger: no new claim while the repository claim re-grade is open.

## Artifacts

- `idea_intake.md`: closest-neighbor and novelty decision.
- `reports/preregistration.md`: immutable stage logic and gates.
- `reports/design_review.md`: adversarial review before any GPU run.
- `assets/context_lens.pt`: byte-identical frozen lens.
- `scripts/run.py`: model-free smoke; model stages are unavailable.
- `reports/artifact_manifest.yaml`: artifact policy.
