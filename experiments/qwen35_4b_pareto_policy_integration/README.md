# Qwen3.5-4B Pareto Policy Integration

Status: **stopped negative on 2026-07-12 before teacher audit**. This clean
successor removed the prior experiment's arbitrary `+0.10` teacher hurdle and
gave any replicated, statistically credible positive paired gain a path
forward. The regenerated C54 policies did not form the required complementary
pair on the clean procedural proxy: `blend` lost its intended quick comparison
in both blocks, while `apex` won deep capability but missed retention.

## Research Programs

- Primary: `agentic_breadth_installation`.
- Supporting: `posttraining_and_adaptation`, `test_time_reasoning_budget`,
  `benchmark_generalization`.
- Closest near-duplicate: `qwen35_4b_specialist_policy_integration`, preserved
  as the design-feasibility negative that motivated this correction.
- New anchor: C54 in `qwen35_4b_gauntlet_frontier`, which measured a
  non-convex quick/deep Pareto pair from the same pinned 4B base.

## Question and Hypothesis

Can one Qwen3.5-4B policy consolidate two same-origin policies that separately
win on short/quick and interactive/deep work, or is their tradeoff a genuine
shared-parameter capacity frontier?

The student starts from the quick policy and generates its own continuations.
At the exact visible student prefix, the quick teacher supplies retention
pressure on short atoms and the deep teacher supplies capability pressure on
long atoms and interactive states. Corrected top-50 MOPD should preserve dense
policy choices that data union and parameter interpolation lose.

## Corrected Gates

- No fixed absolute specialist delta.
- A complementary teacher advantage is `delta > 0`, both independent blocks
  positive, with a one-sided 95% stratified-bootstrap lower bound above zero.
- Saturated cells such as `ferrier` are retention anchors. Equality is fine;
  regression is not. They never veto another teacher or integration.
- The final one-checkpoint system—not each teacher—must beat matched-compute
  sampling.
- Integration seed 42 (the first frozen seed) is the deployable primary;
  seeds 43 and 44 are directional replications, never a checkpoint-selection
  pool.

Exact frozen rules are in [reports/preregistration.md](reports/preregistration.md),
with adversarial review in [reports/design_review.md](reports/design_review.md).

## Substrate

The copied C54 procedural gym contains 12 training families and two never-trained
transfer families (`brinework`, `spindle`). Quick evaluation is atoms L1-L2;
deep evaluation is atoms L3-L6 plus episodes L2/L3/L5. All generation seeds are
disjoint across calibration, qualification, rollouts, and confirmation.

The two specialist datasets are committed provenance-clean C54 artifacts:

- quick: `qwen35_4b_gauntlet_frontier/data/sft_blend.jsonl`;
- deep: `qwen35_4b_gauntlet_frontier/data/sft_apex.jsonl`.

Both policies are regenerated independently from the identical pinned base and
explicitly merged before evaluation.

## Firewall

- The only model is `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Nothing under `benchmarks/` is read, imported, or used for training.
- Programmatic state and scores never appear in prompts.
- Comparable evaluation arms use the same pinned vLLM backend and metadata.
- Runtime LoRA is forbidden; every evaluated adapter is explicitly merged and
  behavior-gated.

## Run

CPU scientific smoke:

```bash
python3 experiments/qwen35_4b_pareto_policy_integration/scripts/run.py --smoke
```

Reached model stages will be resumable:

```bash
python3 experiments/qwen35_4b_pareto_policy_integration/scripts/run.py --stage model-smoke
python3 experiments/qwen35_4b_pareto_policy_integration/scripts/run.py --stage specialists
python3 experiments/qwen35_4b_pareto_policy_integration/scripts/run.py --stage specialist-canary
python3 experiments/qwen35_4b_pareto_policy_integration/scripts/run.py --stage calibrate
python3 experiments/qwen35_4b_pareto_policy_integration/scripts/run.py --stage qualify
python3 experiments/qwen35_4b_pareto_policy_integration/scripts/run.py --stage teacher-audit
python3 experiments/qwen35_4b_pareto_policy_integration/scripts/run.py --stage locality
python3 experiments/qwen35_4b_pareto_policy_integration/scripts/run.py --stage integrate --seed 42
python3 experiments/qwen35_4b_pareto_policy_integration/scripts/run.py --stage controls
python3 experiments/qwen35_4b_pareto_policy_integration/scripts/run.py --stage confirm
```

The benchmark stage is intentionally unavailable until every procedural gate
passes.

The reached `qualify` command now exits nonzero by design after writing the
terminal receipt. All later commands fail closed on that receipt and must not
be run in this experiment.

## Current Evidence

- Prior result preserved: the old `+0.10` rule was impossible at a 0.994 tools
  baseline and did not test MOPD. This successor actually tested the corrected
  `delta > 0` prerequisite.
- Both specialists were independently regenerated, explicitly merged, and
  behavior-gated. Calibration and all four qualification arms passed exact
  model, engine, seed, scope, and pairing checks.
- On 768 pooled quick capability pairs, `blend - apex = -0.02241`; both block
  means were negative (`-0.00693`, `-0.03789`) and the one-sided 95% lower
  bound was `-0.04897`. The failure is the sign of the effect, not an
  arbitrary minimum magnitude.
- On 4,032 pooled deep capability pairs, `apex - blend = +0.04563`; both block
  means were positive (`+0.04254`, `+0.04871`) and the lower bound was
  `+0.03401`. However, six deep retention cells regressed by more than the
  frozen 0.02 allowance.
- Therefore the C54 quick/medium Pareto labeling did not transport into a
  clean quick/deep teacher crossover. This result says nothing about MOPD's
  efficacy: no teacher audit, locality pilot, MOPD update, control,
  confirmation, or benchmark invocation ran.

## Artifacts

- `configs/default.yaml`: frozen splits, seeds, statistics, and controls.
- `idea_intake.md`: relation to the stopped predecessor and C54.
- `reports/preregistration.md`: decision rules.
- `reports/design_review.md`: adversarial pre-run review.
- `reports/literature_review.md`: primary-paper map behind the social-post
  acronyms and the experiment's collapse safeguards.
- `analysis/specialist_qualification.json`: terminal paired gate receipt.
- `runs/policy_eval/*qualification*`: all four raw qualification arms and
  provenance metadata.
- `src/gym/`: contamination-safe procedural substrate.
- `src/mopd_loss.py`: corrected teacher-top-k reverse-KL objective.
- `reports/artifact_manifest.yaml`: external checkpoint policy.
