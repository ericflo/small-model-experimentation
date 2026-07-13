# Qwen3.5-4B Semantic-Anchor Coordinate Branching

**Status:** finished

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

The terminal mechanics run was:

```bash
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/python experiments/qwen35_4b_semantic_anchor_coordinate_branching/scripts/run.py --stage mechanics
```

## Results

Terminal `INVALID_MECHANICS_CONTROL` at the frozen parse gate. All 880 numeric
and 2,240 intervention rows reproduce the calibration after canonical identity
sorting, but unrestricted parse is only 56/880 (6.36%); no consequence row
parses. Text, full-donor, and donor-J direct constrained choices reach 43/44,
43/44, and 42/44, while their consequence choices reach only 6/44, 6/44, and
5/44. Donor-J consequence probability lift over source is `+0.00170` versus
the preregistered `+0.15` gate. No continuation or correctness stage opened.

The mandatory post-mechanics adversarial audit also found that the alias-to-
operation and operation-to-label Latin rotations cancel: the composed alias-to-
label mapping is identical in all four mechanics tasks. The advertised task-
randomized computation endpoint was therefore not realized.

## Interpretation

This exact one-token consequence interface is invalid and unreachable. The
conditional direct-choice pattern is hypothesis-generating evidence that an
explicit anchor can write a name, not passed evidence of reasoning transport.
Because literal text and full-state donors also fail the consequence interface,
and because the composed mapping was fixed, this run cannot isolate a general
J-space limitation. The experiment is frozen; any repair or continuation must
use a fresh directory and fresh data.

## Knowledgebase Update

- Program evidence and shared synthesis record the terminal invalid result and
  retire this late opaque-anchor interface.
- Program backlog redirects to a fresh deployable early-text hypothesis fork,
  not another J amplitude/layer sweep.
- Claim ledger: no new claim while the repository claim re-grade is open.

## Artifacts

- `idea_intake.md`: closest-neighbor and novelty decision.
- `reports/preregistration.md`: immutable stage logic and gates.
- `reports/design_review.md`: adversarial review before any GPU run.
- `assets/context_lens.pt`: byte-identical frozen lens.
- `scripts/run.py`: frozen staged runner and automatic decision logic.
- `reports/pre_mechanics_adversarial_audit.md`: authorization audit.
- `reports/post_mechanics_adversarial_audit.md`: terminal audit and discovered
  composition confound.
- `runs/mechanics/`: complete terminal mechanics receipts.
- `reports/artifact_manifest.yaml`: artifact policy.
