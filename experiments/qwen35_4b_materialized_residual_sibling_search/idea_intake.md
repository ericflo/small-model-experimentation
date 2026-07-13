# Idea Intake

## Program Fit

- Program: structured_execution_and_compilers
- Existing or new program: existing
- Closest program scorecard reviewed: knowledge/program_scorecards.md
- Related future queue item: none; `supervision_causality_ablation` is the
  training follow-up only if this frozen inference interface works

## Prior Evidence

- Anchor 1: `qwen35_4b_early_text_hypothesis_forking` — immediate parent;
  concrete names controlled one-step execution but full programs reached 3/8.
- Anchor 2: `qwen35_4b_decompose_compose_frontier` — closest algorithmic
  near-duplicate; it recursively materialized current states and ranked the next
  primitive, but did not enumerate sibling first operations and generate one
  complete residual suffix under token-matched direct-sampling controls.
- Anchor 3: `qwen35_4b_partial_structure_search` — type-only viability was
  unreadable and its report explicitly leaves materialized residual states open.
- Additional anchors: `qwen35_4b_depth_wall_anatomy`,
  `qwen35_4b_latent_decomposition`, and `qwen35_4b_coverage_vs_selection`.
- Closest duplicate or near-duplicate: `qwen35_4b_decompose_compose_frontier`.

## Novelty Claim

This is the first Qwen3.5-4B test that symmetrically enumerates every concrete
first-operation sibling, exposes only that sibling's public post-intervention
state-to-target relation, generates one complete two-operation residual suffix
per sibling, and demands a gain over both sampled-token- and
logical-model-token-matched monolithic sampling on fresh exact-depth-three
functions. Cheap sibling ranking is a cost-reduction secondary, not the primary
explorer.

## Related Claims

- C35: Brute search dominates tested banked models through depth 4 and remains operationally cheap at depth 5; learned crossover remains open (Promising)
- C48: Hypothesize-and-verify SFT lifts taught depth 2 but not depth 3 at think@1024; cross-depth and cross-substrate transfer are null while the budget curve remains open (Promising)
- C25: 'Be your own tool-search': base first-move ranking is at chance; banking improves step-wise next-op guidance at lookahead distance (Promising)

## Mechanism

The parent establishes that a concrete early name can route local execution,
while C25/C26 show recognition strengthens nearer the goal. Materializing each
candidate's actual consequences should turn every public-live sibling from a
three-step inverse problem into a two-step residual problem. The mechanism is
false if materialized suffix proposals do not beat name-only, token-preserving
state/target derangement, and compute-matched direct proposals after
visible/probe-only selection.

## Control Plan

- Baseline: one frozen candidate-blind full-program sample master per task,
  grown outcome-blind to conservative sampled-token and logical-token
  first-over match points.
- Mechanism-falsifying control: all-24 name-only siblings and all-24
  token-preserving state/target derangements. Exact public viability and full
  CPU enumeration are explicit deployable dominance references.
- Shift or robustness check: two qualification shards followed by eight sealed
  confirmation blocks; no-think binary/listwise, random, and cross-fitted
  surface rankings are secondary cost controls.
- Hidden-label boundary: construction uses common-panel function signatures
  and visible public-live labels but never candidate hidden correctness or
  selector-probe agreement. Hidden outputs open only after pools, resource
  matches, probe-only selections, and IDs are frozen.

## Evidence Output

- Program evidence update: record whether a materialized residual relation is a
  readable/actionable search state or another oracle-useful interface.
- Claim ledger or synthesis update: synthesis only unless qualification and
  independent confirmation both beat every matched-sampling/control arm.
- Reusable artifact: exact-depth-three generator, sibling-state renderer,
  targeted ranking receipts, strict suffix parser, resource matcher, and CPU
  exhaustive reference.
- Stop or branch condition: suffix-interface failure seals the explorer;
  ranking failure seals only the top-four secondary; qualification failure
  seals confirmation. Training remains a separately justified experiment
  regardless, with a search pass supplying positive prior evidence.

## Decision

- Run experiment: yes, staged and fail-closed.
- Create program: no; `structured_execution_and_compilers` is the primary fit.
- Write synthesis only: no; the exact sibling-state intervention is untested.
- Defer: training, banking, and any J-space claim until this interface clears
  both recognition and matched-search gates.
