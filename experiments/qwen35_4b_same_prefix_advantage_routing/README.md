# Qwen3.5-4B Same-Prefix Advantage Routing

## Status

**Terminal negative at the preregistered split-branch route gate.** Deep had
replicated positive continuation advantage, but quick was worse than the soup
student in block 1 (`-0.0253` macro) after being strongly positive in block 0.
MOPD, locality, controls, confirmation, and benchmarks were therefore not run.

## Research Program

- Primary program: `agentic_breadth_installation`.
- Supporting programs: `posttraining_and_adaptation`,
  `evidence_conditioned_selection`, `benchmark_generalization`,
  `process_control_and_tool_use`, and `reliability_and_safety`.
- Program question: can training-only verification identify locally useful
  same-origin teachers and install their complementary behavior in one 4B
  checkpoint without the collapse seen in indiscriminate dense distillation?
- Closest anchors: `qwen35_4b_pareto_policy_integration`,
  `qwen35_4b_gauntlet_frontier`, and
  `qwen35_4b_opsd_pressure_locality_audit`.

## Question

On exact states visited by the strongest existing one-checkpoint student, can
independent verifier branches identify both C54 source policies as genuinely
better continuation teachers, and—only if that advantage replicates—can
positive-advantage-routed MOPD produce one checkpoint that beats the student,
both teachers, visible routing, matched controls, and sample-more?

## Hypothesis

Coarse quick/deep labels failed because endpoint benchmark rank is not a local
teacher label. The useful unit is a student-visible prefix. At each residual
state, independent continuations from the quick teacher, deep teacher, and
current student estimate continuation value. A teacher is used only when its
selection-split mean is strictly above both alternatives; tied or nonpositive
states abstain. If that rule retains positive advantage on fresh audit branches
and two independent state blocks, corrected top-k reverse KL can transfer the
locally better policy while a frozen-student anchor limits collateral drift.

## Setup

- Model: only `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Teachers: independently regenerated C54 `quick_blend` and `deep_apex`
  composites from the same pinned base—the pair used by C54's now directly
  measured tier router. The corrected large-n C54 estimate puts the apex
  medium gain at `+0.321 ± 0.017 SE`, straddling rather than decisively clearing
  `+0.32`; this run assumes no local teacher rank and measures it from scratch.
- Student: an independently regenerated 40% quick / 60% deep convex LoRA-delta
  soup, the strongest existing joint point rather than either endpoint.
- Substrate: the self-contained 14-family procedural gym copied from the prior
  experiment; the 12 trained families produce updates and `brinework` plus
  `spindle` remain transfer-only.
- State distribution: fresh failed student atoms at an exact mid-thought token
  prefix and failed interactive episodes immediately before the first invalid
  action (or the final action if every action was syntactically valid).
- Hidden-label boundary: scores select training states and teachers but are
  never rendered into a model prompt. Benchmark content is never read or
  imported; its CLI remains unreachable until every procedural gate passes.
- Primary metric: equal-weight quick/deep paired joint score on two sealed
  procedural blocks, with one-sided stratified-bootstrap lower bounds.

## The Split-Branch Gate

For every frozen state, each of the two teachers and the student receives four
route-selection continuations and four disjoint audit continuations. The route
uses selection outcomes only. It chooses a teacher iff that teacher's mean is
strictly greater than both the current student's and the alternate teacher's;
otherwise it abstains. No effect-size margin is imposed.

The rule qualifies only if both named teachers have adequate preregistered
support and, separately for each teacher, selected-teacher minus student and
selected-teacher minus alternate-teacher are positive in both state blocks and
have pooled one-sided 95% lower bounds above zero on the untouched audit
branches. This is the experiment's teacher-existence test. MOPD is forbidden if
it fails.

It failed exactly that test. Both teachers had ample support and the combined
router was positive, but quick's selected states did not retain positive audit
advantage over the current soup student in the independent block. The gate was
not weakened to a pooled-only rule.

## Conditional Training and Controls

If qualified, four rounds refresh exact current-student states and apply the
same frozen route. Seventy-five percent of consume-once units use corrected
teacher-top-50 reverse KL from equally represented quick/deep routed states;
25% use the frozen soup as a retention anchor. A five-update exact-logit pilot
must first keep centered non-target drift at or below 0.10 logits, entropy loss
within 10%, and corrected top-k loss within the registered ceiling.

Matched controls are shuffled routing, the old visible quick/deep coarse route,
off-policy best-teacher-continuation SFT, fixed-deep-teacher MOPD, the no-update
soup, and explicit parameter soups. The deployable visible router selects quick
for quick atoms and deep for deep atoms/episodes. The terminal sample-more arm
is execution-filtered best-of-8 from the soup under the identical vLLM runner.

## Run

CPU/scientific smoke:

```bash
python3 experiments/qwen35_4b_same_prefix_advantage_routing/scripts/run.py --smoke
```

Reached GPU stages are resumable and fail closed:

```bash
python3 experiments/qwen35_4b_same_prefix_advantage_routing/scripts/run.py --stage model-smoke
python3 experiments/qwen35_4b_same_prefix_advantage_routing/scripts/run.py --stage build-student
python3 experiments/qwen35_4b_same_prefix_advantage_routing/scripts/run.py --stage route-qualify
python3 experiments/qwen35_4b_same_prefix_advantage_routing/scripts/run.py --stage locality
python3 experiments/qwen35_4b_same_prefix_advantage_routing/scripts/run.py --stage integrate --seed 42
python3 experiments/qwen35_4b_same_prefix_advantage_routing/scripts/run.py --stage controls
python3 experiments/qwen35_4b_same_prefix_advantage_routing/scripts/run.py --stage confirm
```

There is deliberately no benchmark stage until a procedural confirmation
receipt explicitly authorizes it.

## Decision Rule

The final artifact is one merged 4B checkpoint with no deployment-time teacher
or verifier. It passes only if its joint lower bound is above zero versus each
teacher, the soup, the visible router, and every matched one-checkpoint control;
both quick and deep means beat the better source endpoint in both blocks;
three preregistered training seeds point in the same direction; retention and
transfer regress by no more than 0.02; and greedy performance beats the soup's
execution-filtered best-of-8. Tiny but replicated gains count. Large unstable
gains do not.

## Artifacts

- `idea_intake.md`: novelty and anti-duplication decision.
- `configs/default.yaml`: frozen states, branches, seeds, gates, and controls.
- `reports/preregistration.md`: estimands and terminal decision rule.
- `reports/design_review.md`: adversarial review before model output.
- `reports/literature_review.md`: primary-paper basis for the design.
- `runs/preregistration_receipt.json`: immutable design hashes and commit.
- `analysis/`: gate and final machine-readable receipts.
- `reports/artifact_manifest.yaml`: external checkpoints and regeneration.

If the procedural receipt opens the benchmark, the upstream C54 power
correction is binding: report three quick events and at least eight medium
events, all paired and without exclusion. A three-event medium mean cannot
support the terminal claim.
