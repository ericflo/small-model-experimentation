# Adversarial Design Review

## Verdict

`sound_with_scope_and_resource_fixes_applied`.

The balanced funnel can answer whether potential-selected full traces beat serious controls on three
families. It cannot complete or rescue the parent's nine-family/pivot claim. The following fixes are part of
the frozen design.

## Findings And Applied Fixes

1. **A post-calibration rewrite would be invalid.** The work is a new experiment, declares every observed
   input, and leaves the parent preregistration untouched.
2. **The family subset could be cherry-picked.** The core is defined by complete leading blocks in the
   parent's pre-existing train order. No family score or correctness selected it. Claims remain three-family.
3. **Inherited data could mutate silently.** Import requires the frozen source-index SHA-256, exact aggregate
   counts, and every shard receipt. The new index records provenance and never rewrites inherited shards.
4. **Finishing on another batch shape could change Ada samples.** Generation remains one task per logical
   call, N=64, identical sampling, engine geometry, and stable task seeds. Existing tasks are never replayed
   or compared token-pairwise across shapes.
5. **Answer potential already lost to length in calibration.** `shortest_natural` is now mandatory and may
   win the experiment. A potential result must beat the strongest of shortest, random, and success-RFT.
6. **Dropping the empty arm weakens answer-only attribution.** The user-approved six-arm budget replaces it
   with the more threatening shortest control. The conclusion is selector-relative, not a complete
   decomposition of answer-seam learning.
7. **Fast scoring could silently change the metric.** New parity covers the entire joint target, empty
   baselines, and gains, not only canonical answer likelihood. Bulk scoring is blocked before model work if
   any registered discrepancy exceeds 0.15 mean nats/token.
8. **Canonical-only scoring could exploit a formatting accident.** The already-observed format stability
   (tau-b 0.841) is disclosed as a design input. All train candidates use one uniform canonical boundary.
9. **Joint score can be dominated by boundary length.** It is normalized by canonical-answer token count as
   preregistered in the parent, reported alongside raw boundary/answer sums, and gets its own treatment rather
   than replacing answer gain.
10. **Shortest has fewer supervised tokens.** Rows and optimizer steps match, but token dose intentionally
    does not. Actual thought, supervised, and forward tokens are mandatory; a shortest win supports an
    efficiency/compression hypothesis, not evidence for answer potential.
11. **Success-RFT may have sparse task coverage.** Report unique tasks/rows before oversampling and the exact
    common-task comparison. Oversampling cannot be described as more independent evidence.
12. **Natural-close survivorship can select easier tasks.** Eligibility is identical across arms and any
    deficient task is excluded symmetrically. Family/level close rates and selected-task coverage are reported.
13. **Task-shuffle may mismatch length.** Use a one-to-one within-family/level derangement minimizing total
    length mismatch; preserve both target and source task IDs in SFT records.
14. **Six long-trace adapters can consume the saved budget.** Hard-stop at 360 tasks, omit branches, use the
    parity-gated vLLM scorer, and make Stage B conditional. No evaluation shortcut changes Stage-A arms.
15. **Staging can become selective reporting.** Stage A always evaluates every arm on all three declared
    subsets. The trigger, optional arms, and terminal negative are frozen before training.
16. **The positive bar may be mathematically impossible on a saturated split.** A machine-readable [0,1]
    reachability check precedes Stage B. An impossible gate is recorded as a stop, never repaired in place.
17. **Matched-compute sample-more could disappear after a training win.** It is mandatory in Stage B and
    required for the mission verdict, using actual prompt plus sampled tokens.
18. **Runtime LoRA can no-op on Qwen3.5.** Merge into the full composite, require base/base determinism, and
    reject every trained arm with zero same-prompt token differences.
19. **Held-family tasks might leak through copied imports.** Training imports only the local train-family
    modules. Held-family registries are loaded exclusively by evaluation scoring after datasets are frozen.
20. **A null could be overgeneralized.** The report must say core-scope negative and preserve the unfinished
    broader parent. Nine-family and pivot conclusions are prohibited.

## Remaining Limits

- One model and one procedural gym.
- Three training families selected by saved ordering, not random family sampling.
- Calibration influenced the control set and compute schedule.
- One mandatory training seed unless Stage B triggers.
- Canonical answers are oracle-side at curation time.
- The physical 16,384-token context still censors some long searches.

## Expensive-Run Authorization

Authorized only after the design commit is pushed, the implementation test suite passes, inherited receipts
validate, and the exact long-row training stress receipt remains valid on the current environment. Train
likelihood scoring additionally requires either the 32-row joint parity gate to pass or a committed dated
instrument amendment after that gate fails. Selection additionally requires the committed post-score
deviation/evidence seal described below.

## Post-Freeze Instrument Audit — 2026-07-12

The candidate vLLM scorer did not earn authorization: its task-diverse 32-row maximum was 0.692447 against
the frozen 0.15 ceiling. No bulk score existed. The scientifically conservative repair is not a softer gate
or friendlier row set; it is uniform use of the single-context Transformers reference that defined the other
side of the comparison. This removes batching from the measurement and preserves every scientific selector,
control, and decision threshold. Bulk scoring is authorized only under that amended backend.

## Post-Score, Pre-Official-Selection Balance Deviation — 2026-07-13

After all 360 tasks had canonical scores, but before R1 was complete and before any official selection
dataset or adapter existed, a read-only CPU application of the registered helper exposed that it could return
only the best row when no second trace was within 0.25 nats/answer-token. The resulting 116-task subset and
its family imbalance were observed. This violates the preregistration's rule that candidate scores not be
observed before an amendment; a later seal cannot restore prospectivity. The implementation deviation keeps
the near-best rule when possible and otherwise uses the deterministic second-ranked trace from the same
frozen top-12. Hard gates require all 360 tasks, 40 tasks in every family/level cell, and 720 rows per arm.
No score value, rollout label, SFT outcome, or held-out outcome was used to choose the repair. Partial R1
labels were subsequently inspected for cost planning before commit. Claims about this repaired selector are
therefore explicitly post-score/partial-rollout exploratory within the otherwise frozen evaluation.

## Pre-Training Seed Audit — 2026-07-13

Before any adapter existed, the same preflight found that `TrainingArguments(seed=...)` was constructed after
`get_peft_model`, so the advertised seed did not control LoRA initialization. The training entry point now
sets the global seed before model creation and resets it immediately before adapter construction. A CPU
contract test enforces ordering and every training receipt exposes the global, adapter, Trainer, and data-seed
contract. Hyperparameters, data, and evaluation rules are unchanged.

## Pre-Training Provenance Audit — 2026-07-13

The same audit repaired three restart/accounting hazards before any adapter existed: one-pass receipt totals
despite two epochs, permissive partial-adapter merge acceptance, and path-bound probe/evaluation caches. The
training receipt now records actual two-epoch exposure, optimizer steps, lock digest, initial-state digest,
and final artifact hashes. Merge requires all 128 A/B pairs and fingerprints every deployed file. Probe and
evaluation receipts bind the checkpoint fingerprint plus their task, sampling, engine, and code contracts.
Stage-A analysis also verifies exact task identity and an explicit conservative strongest-baseline tie rule.

The shuffle-control implementation was upgraded from best cyclic rotation to an exact deterministic
minimum-cost assignment with same-task edges forbidden, matching the original wording. Length mismatch is
reported, and target/source selection metadata are namespaced so the trained source can be audited row by
row. Training receipts are written by atomic replacement and are never included in their own artifact map.
This changed no treatment trace, score, threshold, or observed evaluation outcome.

Applying the repaired selector in memory to the frozen score bank before writing official datasets showed
answer fallback on 5/360 tasks (median gap 3.409, maximum 7.378 nats/answer-token) and joint fallback on
244/360 tasks (median 0.893, mean 1.410, p90 2.954, maximum 12.128). The latter is scientifically material:
the joint arm is a mixed best-plus-near-best-diverse-or-second-ranked treatment, not a uniformly
near-best-diverse treatment. Selection receipts and any result interpretation must stratify it by mode and
gap.
