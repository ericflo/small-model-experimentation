# Qwen3.5-4B Deep-Advantage MOPD

**Status:** in-progress · since 2026-07-12 · Fresh deep qualification, exact-logit locality, all three four-round integrations, and all controls pass; sealed confirmation remains.

## Status

**Fresh deep qualification passes on both untouched blocks, the frozen
five-update MOPD pilot passes exact-logit locality, and seeds 42, 43, and 44
each complete all four integration rounds. All trained and parameter controls
also pass their artifact/training gates; no procedural performance result
exists yet.** This is a new result-bearing successor to
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
forbids later stages. Model stages re-exec the orchestrator under the pinned
`.venv` before dependency-bearing validators load; a missing training runtime
therefore fails before any stage work rather than after producing an artifact.

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

## Locality Result

Three fixed candidate batches supplied 90 deep-qualified failed states. The
assembler selected exactly 60 deep capability units, 20 soup anchors, and 60
disjoint matched non-advantage controls; 57/60 controls matched the exact
family/kind/level cell and three matched family/kind. The cache bound 140
samples and 35,147 active positions to quick, deep, and soup top-50 targets.

The consume-once pilot completed all five updates with the frozen 15-deep/5-soup
mixture. Training mean corrected top-50 loss was `0.05242`; its held probe loss
fell from `0.04773` to `0.02947` and overlap rose from `0.84840` to `0.85163`.
On the preregistered batch-of-one exact probes, centered non-target logit drift
was `0.02760` (ceiling `0.10`), entropy fell `3.11%` (ceiling `10%`), and exact
target loss improved from `0.01293` to `0.01170`. Every locality check passed,
authorizing four-round MOPD. Exact drift was measured at one midpoint token for
each of the 20 consumed units, so this is a literal local-safety result, not a
claim of invariance over every trained token or of improved capability.

## Seed-42 Integration Result

All four full-dose rounds passed their frozen training gates. Each used three
fresh 192-state candidate batches, selected exactly 60 deep capability units
plus 20 soup anchors, completed 20 updates, and non-decreased held-probe top-50
overlap. Deep-route supply was 90, 81, 78, and 83 states; mean corrected loss
was `0.05669`, `0.04901`, `0.04855`, and `0.05404`. Held-probe loss changed
`0.08318→0.05112`, `0.03915→0.02020`, `0.03476→0.01893`, and
`0.04873→0.02793`. The final round-3 merge receipt is
`88512a57ebb190f0392118a30258eee5fb3bc58d5d34ae04e384afc8842f9122`.

No capability claim follows from these training gates. Full-round probe entropy
contracted `10.28%`, `12.33%`, `8.90%`, and `11.42%`; the first, second, and
fourth exceed the locality pilot's 10% caution line. That ceiling was not a
registered full-round stop, so the result is preserved as a collapse-risk
warning rather than post-hoc reclassified. Round 1's sole 131-token cut was a
cache-only route control and no consumed capability/anchor was truncated;
rounds 2/3 had zero cuts in every role.

## Matched Controls Result

All three trained controls completed four rounds of 20/20 consume-once updates
and passed their frozen round gates. Full-prefix non-advantage routing had mean
corrected losses `0.05393`/`0.05036`/`0.04990`/`0.04619`; held-probe loss fell
and top-50 overlap rose in every round. Its overlay reproduced the original
matched mapping exactly in rounds 0, 2, and 3; round 1 deterministically
replaced the sole original state incompatible with zero-truncation eligibility.

Wrong-teacher quick MOPD on the selected states had mean corrected losses
`0.07040`/`0.06537`/`0.06047`/`0.06949`, again with improving probe loss and
non-decreasing overlap in every round. Off-policy best-deep-continuation SFT
had mean cross-entropies `0.10926`/`0.11021`/`0.09851`/`0.09942`; every round
completed its registered update gate and reduced probe loss. The off-policy
gate deliberately has no top-k-overlap or CE threshold, so those values are
descriptive rather than borrowed MOPD gates.

The 25%/50%/75% deep parameter soups each applied 128/128 nonzero LoRA modules
and bind exhaustive inference-file inventories. Terminal merge-receipt hashes
for non-advantage, wrong-teacher, and off-policy controls are respectively
`99e4d3258f450173204466bd4a2b4f1dfadfc54d706008e6fc3944a5f7bd57f5`,
`90ba5ad70a6dede8e0181c1c05f80ffa9a0d9651b604a1cc27659a8da69df544`,
and `5f6b2c9c1d2a68001b7556c30324976c8312c3c4f170fe489496f0580853c435`.
The aggregate controls receipt is
`103ef4cc0b24d7c10666b6f0adfcd4dfae4720415c7fbbc76b681ab79162640b`;
independent canonical replay and model-byte authentication pass. This licenses
sealed comparison only. It does not show that any trained arm gained capability
or that advantage routing is causal.

The no-clobber semantic authorization then sealed the exact 13-arm map at hash
`709694b7d770b5cbb09afe8b932bba3891ab4fea39c54c625fc84c5da973072d`.
Its receipt hashes to
`f4a5456844adeafd39e2e4f2a8036ed9fff2c78830b2eab9d4a7bfa1300d2278`;
the complete control-code inventory was identical immediately before and after
publication. Global confirmation admission subsequently passed its independent
pre/post arm-byte checks and hashes to
`18c019e92fb6b7f7caed0b0f916b958d528b36b9a30607c2890e6b9385d0125d`.
That first admission is now archived: block-0 deep completed all generation but
failed closed before `scores.json` because the ordinary runner path omitted a
strictly required journal field that the scoring projection had represented as
an empty list. The complete 6,879-output transaction is quarantined and no
performance content was inspected. The failure receipt hashes to
`2e645322ead3fbbdf58760849fe17def81fd12b62cdfa4b6c58808e24612ed41`;
it authorizes only a schema-contract fix, fresh no-clobber authorization, and a
full rerun from an empty current confirmation tree.

The contract repair leaves the strict journal validator unchanged and adds the
missing field as the established empty-list value on ordinary outputs. It does
not alter sampled IDs, text, seeds, token budgets, scores, task geometry, or
backend settings. All 212 experiment tests pass, including a direct naturally
closed budget-output regression. The archived generation will not be reused;
fresh authorization and admission remain mandatory before rerun.

## NF4/BF16 Interpretation Diagnostic

The committed interpretation-only seed-42 diagnostic is valid over four fixed
6-deep/2-soup probes (32 units, 7,970 target positions), but it finds weak
training/deployment update parity. Mean NF4 objective gain was `+0.02191`, while
the explicit bf16 merges averaged `-0.000224`; unit gain-sign agreement was
`46.88%`, gain correlation was `-0.152`, and midpoint update cosine averaged
`0.407`. Endpoint predictions were often superficially close (31/32 top-1
agreement), which does not rescue the divergent update signal. The diagnostic
cannot stop, rescue, or reclassify the frozen experiment. It establishes that
NF4 probe improvement is weak evidence about deployed bf16 behavior and makes
the sealed same-vLLM confirmation—not trainer-side loss—the decisive test.

## Artifacts

- `idea_intake.md`: novelty and duplicate decision.
- `configs/default.yaml`: frozen seeds, geometry, gates, and controls.
- `reports/preregistration.md`: estimands and terminal decisions.
- `reports/design_review.md`: adversarial pre-output review.
- `reports/literature_review.md`: primary-paper and repository basis.
- `runs/preregistration_receipt.json`: immutable design hashes and commit.
- `analysis/`: machine-readable gates and final receipt.
- `reports/artifact_manifest.yaml`: external checkpoints and regeneration.

Confirmation keeps only the atomic `scores.json` commit marker under each
`runs/confirmation/` arm. The mirrored
`large_artifacts/qwen35_4b_deep_advantage_mopd/confirmation/` directory retains
`STARTED`, `GENERATED`, and `COMPLETE` receipts, gzip atom/episode rows, and one
durable gzip journal bundle per returned generation call. No partial is deleted
or resampled: a started-only/interrupted attempt is terminal, an authenticated
`GENERATED` or `COMPLETE` attempt may be finalized without generation, and a
caught failure is quarantined with hashes of every retained byte.
Before any arm may reserve `STARTED`, a no-clobber global `ADMISSION.json`
binds the exact semantic-controls authorization receipt, its complete stable
control-code inventory, every admitted model, both blocks, and the evaluator
source inventory. The same binding is required in every transaction and score,
so a score created before (or under different code than) that authorization
cannot be reused.

Every score authenticates sampled-token totals from stage-1/stage-2 sampled ID
arrays (including trimmed terminal IDs and excluding injected close tokens),
cross-checks runner request/completion/token totals, hashes each complete
returned request and output before scoring, and replays atom scoring plus every
episode action, transition, and terminal score from the journaled text. It binds
exact task-manifest and ordered-plan hashes, the registered raw and resolved
greedy/sample-8 settings (including seed, budgets, penalties, and multiplicity),
and one canonical fingerprint of the pinned vLLM/Python/package-lock/GPU/CUDA
engine protocol. A confirmation-only wrapper also proves each call fits the
live-derived hybrid cache with
`ceil(tokens/528) + 3*ceil(tokens/16384)` (35 blocks at full context) and
zero-preemption capacity. Resume,
analysis, and benchmark authorization recompute all of this, require one backend
across all 26 score sets and one exact task plan per block, and inventory every
score, marker, raw file, and call bundle. Benchmark authorization itself is an
exclusive no-clobber seal. Any later mutation stops benchmarking.

Benchmark files remain unread and unreachable unless the procedural
confirmation explicitly authorizes the run-only CLI.
