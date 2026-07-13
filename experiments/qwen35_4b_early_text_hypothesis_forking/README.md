# Qwen3.5-4B Early Text Hypothesis Forking

**Status:** in-progress · since 2026-07-13 · adversarial review and CPU smoke passed; model mechanics remains sealed.

This experiment tests whether supplying each fully bound first operation at the
start of Qwen3.5-4B reasoning changes complete two-step program proposals enough
to beat late hints and compute-matched sampling under visible-only selection.

## Research program

- Primary: `structured_execution_and_compilers`.
- Secondary: `evidence_conditioned_selection`, `test_time_reasoning_budget`,
  and `interpretability_and_diagnostics`.
- Immediate parent: `qwen35_4b_semantic_anchor_coordinate_branching`.
- Closest near-duplicate: `qwen35_4b_hypothesize_verify_wall`, which used one
  generic procedure scaffold rather than a trajectory for every bound first
  operation.

## Question

Does placing a concrete first-operation hypothesis at the beginning of native
thinking shift Qwen3.5-4B's full-program proposal distribution, and can one
frozen visible-only selector turn that shift into more correct programs than
late injection and compute-matched ordinary sampling?

## Why this follows the Jacobian work

The parent late-anchor experiment did not establish a valid consequence effect:
its unrestricted interface failed and its composed mappings accidentally
cancelled. Its constrained diagnostic nevertheless supplied one useful timing
clue—late state could write an operation name without evidence that subsequent
computation consumed it. This successor tests that clue with text, before
reasoning, without claiming J-space transport or internal certainty.

## Frozen setup

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Backend: one experiment-local pinned vLLM runner, bf16, for every generation
  arm. No backend mixing.
- Substrate: fresh exact-depth-two list transformations with 8 visible, 8
  hidden, and 16 unlabeled probe inputs; 48 qualification tasks in two frozen
  blocks and 96 sealed confirmation tasks.
- Bound bank: all 24 legal first operations—8 parameter-free, 6 additions, 3
  multiplications, 4 takes, and 3 rotations. `negate` remains a distractor but
  is not generated as the gold first step because of pervasive equivalent
  reorderings.
- Task admission: exhaustive enumeration of all 24² programs must prove no
  depth-one fit, unique identification of the bound first operation, and
  hidden/probe equivalence of every visible-consistent program.
- Output: a strict Python helper function with exactly two allowed assignments.
  An AST parser canonicalizes the calls and an interpreter executes them;
  generated code is never evaluated.
- Selector: canonical deduplication, exact visible pass, clustering by outputs
  on unlabeled probes, cluster support, then a frozen task hash. Gold is opened
  only after prompts, outputs, resources, and selected IDs are written.

## Arms

- `early_concrete_24`: one independently sampled trajectory per bound operation;
  the exact hypothesis tokens are inserted immediately inside open `<think>`.
- `late_equal_total_24`: an independent blind 512-token prefix per branch, the
  same hypothesis tokens, then 512 more thought tokens.
- `late_equal_post_24`: the same 512-token blind construction followed by 1,024
  post-hypothesis tokens, deliberately overmatching early's usable suffix.
- `early_duplicate_24`: 24 branches with one task-hash-selected operation.
- `early_placebo_24`: an exact-scaffold neutral hypothesis bank with token use
  charged.
- `neutral_sample_more_master` and `plain_sample_more_master`: independently
  frozen 48-sample pools. Prefixes selected before grading provide sampled-token
  and logical-model-token match points plus first-over-budget controls.
- `cpu_exhaustive`: all 576 two-step programs tested against visible examples.
  This is the symbolic scope ceiling, not a model oracle.

Every model arm uses the same decode distribution and final visible-only
selector. Every prompt, blind prefix, resumed-prefix prefill, injected token,
thought token, forced close, answer token, invalid output, and duplicate is
charged in resource receipts.

## Stages

1. CPU smoke proves inventory, task construction, ancestor disjointness, strict
   parse/execution, gold-mutation invariance, outcome-blind compute matching,
   branch permutation independence, and composed slot-to-behavior variation.
2. Label-free mechanics crosses four public diagnostic inputs with all 24 bound
   operations. It requires natural parse, correct computed lists, broad support,
   low cap contact, and adherence above deranged/duplicate/placebo controls.
3. Qualification runs 48 tasks only after mechanics passes. Early must beat
   both late controls and every token-matched duplicate/placebo/sample-more
   construction under paired task uncertainty and Holm correction.
4. Confirmation repeats the frozen protocol on 96 untouched tasks. Splits are
   never pooled. Training is not part of this experiment.

## Run

Model-free smoke:

```bash
.venv/bin/python experiments/qwen35_4b_early_text_hypothesis_forking/scripts/run.py --stage smoke
```

Model stages remain fail closed until their audited implementation receipt is
committed and pushed.

## Current result

`CPU_SMOKE_PASS`. The refreshed smoke freezes 144 fresh tasks with zero readable-
ancestor behavior collisions, exhausts all 576 two-step programs for every
task, verifies 24 distinct bound-operation consequences on each of four public
diagnostics, and records 144 distinct composed branch maps with balanced gold
positions. Strict parser, selector, gold-mutation, resource-matching, and exact-
token runner tests pass (31 tests plus 33 parameterized subtests). No model has
been loaded and no scientific capability outcome exists. The rejected 12-type
draft is not evidence.

## Claim boundary

A replicated pass would show externally stratified, verifier-assisted proposal
shaping on a small exhaustively searchable DSL. It would not show consciousness,
internal certainty, J-space transfer, weight installation, autonomous solving,
or superiority to exhaustive symbolic search. A higher-depth or installation
test must be a new experiment.

## Artifacts

- `idea_intake.md`: novelty and nearest-neighbor decision.
- `reports/design_review.md`: adversarial attacks and mandatory resolutions.
- `reports/preregistration.md`: frozen arms, gates, and stop logic.
- `configs/default.yaml`: exact model, data, compute, and thresholds.
- `runs/smoke/summary.json`: model-free validation receipt.
- `reports/artifact_manifest.yaml`: external/omitted artifact policy.
