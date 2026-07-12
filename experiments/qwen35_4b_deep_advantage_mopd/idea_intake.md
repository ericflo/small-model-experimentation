# Idea Intake

## Program Fit

- Program: `agentic_breadth_installation` (primary) and
  `posttraining_and_adaptation` (mechanism owner).
- Existing or new program: existing.
- Closest program scorecard reviewed: `knowledge/program_scorecards.md`.
- Related future queue item: `posttraining_method_shared_substrate`; this is a
  narrower prerequisite test, not that broad method comparison.

## Prior Evidence

- Anchor 1: `qwen35_4b_same_prefix_advantage_routing`—deep passed the intended
  local-teacher gate; quick did not; MOPD was never run.
- Anchor 2: `qwen35_4b_pareto_policy_integration`—external tier rank did not
  transport to clean student states.
- Anchor 3: `qwen35_4b_gauntlet_frontier` / C54—the quick/deep sources occupy a
  measured single-checkpoint Pareto frontier, while the 40/60 soup is the
  strongest joint initialization.
- Closest duplicate: `qwen35_4b_same_prefix_advantage_routing`.

## Novelty Claim

This is the first experiment that is allowed to test MOPD after a named source
policy independently passed replicated same-prefix advantage; it asks whether
deep's local residual can improve the already joint soup without requiring the
failed quick route as a second training source.

## Mechanism

The soup retains quick behavior in its weights. Deep MOPD is applied only to
fresh current-student states where four selection branches put deep strictly
above both quick and the current student; frozen-soup anchors protect the joint
initialization. The explanation is false if deep fails fresh audit, the update
fails locality, matched non-advantage-state deep MOPD performs as well, quick
targets on the same states perform as well, or the learned checkpoint cannot
beat the source/router/sample-more baselines.

## Control Plan

- Baseline: immutable no-update 40/60 soup, both source checkpoints, visible
  tier routing, and execution-filtered soup best-of-8.
- Mechanism-falsifying controls: deep MOPD on one-to-one kind-preserving
  non-deep-selected states; quick MOPD on the exact deep-selected states;
  off-policy best-deep-continuation SFT.
- Shift or robustness check: two qualification blocks, two sealed procedural
  blocks, three fixed training seeds, retention cells, and two transfer-only
  families.
- Hidden-label boundary: verifier scores route training examples only and are
  absent from prompts, targets, checkpoint inference, and final deployment.

## Evidence Output

- Program evidence update: `agentic_breadth_installation` and
  `posttraining_and_adaptation` evidence/backlogs.
- Claim ledger or synthesis update: shared synthesis for any terminal result;
  claim promotion only after the repository's claim-review process.
- Reusable artifact: deep-only route analyzer, matched non-advantage control
  constructor, corrected sparse MOPD harness, and exact provenance receipts.
- Stop or branch condition: deep qualification, then locality, then all three
  primary seeds, controls, sealed confirmation, and only then benchmark CLI.

## Decision

- Run experiment: yes.
- Create program: no.
- Write synthesis only: no; MOPD remains empirically untested.
- Defer: two-teacher integration until a cross-fitted direct-advantage predictor
  independently qualifies quick on a third block.
