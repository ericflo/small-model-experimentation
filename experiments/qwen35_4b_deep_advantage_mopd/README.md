# Qwen3.5-4B Deep-Advantage MOPD

## Status

**Fresh deep qualification passes on both untouched blocks; exact-logit
locality is authorized and no MOPD update exists yet.** This is a new result-bearing successor to
`qwen35_4b_same_prefix_advantage_routing`, not an extension of its terminal
result.

## Research Program

- Primary program: `agentic_breadth_installation`.
- Supporting programs: `posttraining_and_adaptation`,
  `evidence_conditioned_selection`, `benchmark_generalization`, and
  `reliability_and_safety`.
- Closest duplicate: `qwen35_4b_same_prefix_advantage_routing`.
- Program question: can the first independently qualified same-prefix source
  signal—deep—be installed into the strongest joint 4B checkpoint without
  erasing its quick behavior or losing to deployment-time routing/sampling?

## Question

On fresh states from the immutable 40% quick / 60% deep soup, does the exact
strict three-policy rule again identify a replicated deep continuation
advantage? If so, can corrected top-50 MOPD on only those deep-selected states
produce one checkpoint that beats quick, deep, the soup, visible routing,
matched mechanism controls, and verifier-best soup best-of-8?

## Hypothesis

The predecessor isolated a real conditional deep advantage but could not test
MOPD because quick was a required second source. The joint soup already carries
quick behavior. Applying deep pressure only where deep strictly beats both
quick and the current student should add the missing local residual while the
25% frozen-soup anchor preserves the existing mixture. If routing is causal,
the update should beat both deep MOPD on matched non-advantage states and quick
MOPD on the exact selected states.

## Setup

- Model: only `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Initial student: the predecessor's immutable explicit 40/60 composite,
  SHA-256 `04610723f3f46d0a094ae0e5bc1a491bb6ad9e0fb6c8a84417dfe5e527f15b50`.
- Source policies: the same explicit `quick_blend` and `deep_apex` composites.
- Substrate: copied 14-family procedural gym; 12 families may supply updates,
  while `brinework` and `spindle` remain transfer-only.
- Qualification: two new 192-state blocks, four selection plus four disjoint
  audit branches for quick, deep, and student.
- Frozen route: deep is selected only when its selection mean is strictly above
  both quick and student. Ties and all other states abstain. There is no gain
  magnitude threshold.
- Gate: at least 16 deep routes per block; deep-minus-student and
  deep-minus-quick audit macros positive in both blocks; pooled one-sided 95%
  lower bounds above zero for both contrasts.
- Update: four online rounds, each with 60 consume-once deep capability units
  and 20 frozen-soup anchors; five updates must first pass exact-logit locality.
- Controls: matched non-advantage-state deep MOPD, wrong-teacher quick MOPD on
  the exact selected states, off-policy best-deep-continuation SFT, fixed
  parameter soups, source checkpoints, no-update soup, visible routing, and
  soup best-of-8.
- Hidden-label boundary: verifier outcomes select training states only. They
  are never rendered to the model or used at deployment.

## Run

CPU/scientific smoke:

```bash
python3 experiments/qwen35_4b_deep_advantage_mopd/scripts/run.py --smoke
```

Reached stages are explicit and resumable:

```bash
python3 experiments/qwen35_4b_deep_advantage_mopd/scripts/run.py --stage model-smoke
python3 experiments/qwen35_4b_deep_advantage_mopd/scripts/run.py --stage verify-student
python3 experiments/qwen35_4b_deep_advantage_mopd/scripts/run.py --stage route-qualify
python3 experiments/qwen35_4b_deep_advantage_mopd/scripts/run.py --stage locality
python3 experiments/qwen35_4b_deep_advantage_mopd/scripts/run.py --stage integrate --seed 42
python3 experiments/qwen35_4b_deep_advantage_mopd/scripts/run.py --stage integrate --seed 43
python3 experiments/qwen35_4b_deep_advantage_mopd/scripts/run.py --stage integrate --seed 44
python3 experiments/qwen35_4b_deep_advantage_mopd/scripts/run.py --stage controls
python3 experiments/qwen35_4b_deep_advantage_mopd/scripts/run.py --stage confirm
python3 experiments/qwen35_4b_deep_advantage_mopd/scripts/run.py --stage benchmark
```

Every command after smoke requires an immutable design receipt. A failed gate
forbids later stages.

## Decision Rule

The seed-42 final merged checkpoint must have a positive joint mean, positive
one-sided 95% lower bound, and positive means in both sealed blocks versus
quick, deep, soup, visible routing, every matched control, and every parameter
soup. Its quick and deep strata must each exceed the better source in each
block; seeds 43/44 must point positively versus both sources and soup; retention
and transfer regressions may not exceed 0.02; and greedy joint performance must
beat verifier-best soup best-of-8. Tiny replicated gains count. Large unstable
gains do not.

## Qualification Result

The strict selector routed 54/384 fresh states to deep (28 and 26 by block).
On disjoint audit branches, deep beat soup by `+0.1650` and `+0.1220` in the
two blocks (pooled `+0.1421`, one-sided 95% lower bound `+0.1230`) and beat
quick by `+0.2000` and `+0.1420` (pooled `+0.1691`, lower bound `+0.1534`).
Every frozen support/sign/uncertainty gate passed. Quick also independently
passed on 47 routed states in this fresh replication; that is retained as
future two-teacher evidence, but the locked treatment remains deep-only.

## Artifacts

- `idea_intake.md`: novelty and duplicate decision.
- `configs/default.yaml`: frozen seeds, geometry, gates, and controls.
- `reports/preregistration.md`: estimands and terminal decisions.
- `reports/design_review.md`: adversarial pre-output review.
- `reports/literature_review.md`: primary-paper and repository basis.
- `runs/preregistration_receipt.json`: immutable design hashes and commit.
- `analysis/`: machine-readable gates and final receipt.
- `reports/artifact_manifest.yaml`: external checkpoints and regeneration.

Benchmark files remain unread and unreachable unless the procedural
confirmation explicitly authorizes the run-only CLI.
