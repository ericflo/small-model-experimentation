# Adversarial Design Review

**Decision:** `DESIGN_GO` for the registered sequential experiment. Model-bearing execution remains
prohibited until the separate source-bound implementation review is `GO`, all tests/checks pass, the
registered inputs are committed and clean, and the canonical design receipt is created.

**Review date:** 2026-07-13

## Question actually answered

The experiment can answer a narrow but useful question: under this fixed recurrent Carry recipe,
does rank-32 LoRA form a readable terminal `(node, phase, checksum)` representation under the joint
answer-plus-state objective; and, after a valid miss, does replacing the factorized extra-call update
with a direct full-shape update practically relieve the failure? It cannot identify matrix rank in
isolation because full shape also changes parameter count, FLOPs, and optimizer geometry. It cannot
establish downstream causal use, answer capability, deployment value, or a sampling advantage.

That distinction is load-bearing. All affirmative full-shape labels say “direct-full-shape recipe,”
and a failure of both registered recipes leaves supervision/readout architecture versus optimization
unresolved.

## Primary adversarial findings and frozen resolutions

### 1. The prior full-rank negative did not adjudicate LoRA capacity

The parent LoRA and successor full-rank pilots used unmatched common-state construction and dropout
streams, only one short seed, and verdict ladders confounded by unrelated Carry/Bag and answer gates.
Comparing their raw scores would falsely attribute every difference to adaptation capacity.

Resolution: this successor uses one serialized common-state bundle per seed, capacity/objective-
independent row order and adaptation-dropout schedule, a common hook controller, three fixed-final
seeds, separate optimizer clipping groups, and exact cross-arm receipts. Prior scores are motivation,
not cells in this experiment.

### 2. LoRA must be tried first, but a negative cannot leave the concern hanging

Running full rank unconditionally would spend substantial compute and weaken the sequential estimand.
Stopping after a complete LoRA miss would fail the motivating question.

Resolution: all three LoRA joint cells run first. A pass terminates and prohibits later capacity
arms. Any complete absolute miss emits an identity-bound receipt that mandates both LoRA state-only
and full-rank joint. A full-rank trigger miss mandates full-rank state-only. After sealed access, an
all-cell LoRA pass stops first; otherwise a full-rank sealed absolute miss mandates state-only.
Missing, invalid, or mechanically failed evidence authorizes repair only, never a scientific branch.

### 3. A joint miss may reflect objective competition rather than representational capacity

Answer loss can compete with dense state supervision. A state-only-first experiment would no longer
test the original joint recipe, while omitting state-only after a miss would overinterpret failure.

Resolution: joint is always primary. State-only omits the answer graph entirely and is conditional.
The four registered Stage-C patterns provide descriptive signatures consistent with joint-objective
interaction. Because these conditional controls reuse branch-selecting trigger rows, they do not
causally identify answer-loss competition. If neither state-only arm passes, the terminal result
explicitly remains unresolved between
supervision/readout architecture and the registered optimizer/training recipe.

### 4. A pooled metric could hide seed- or depth-specific failure

Trajectory averages, component accuracy, shallow depths, or pooling across seeds could produce an
apparently strong score while the terminal joint state fails at the depths that matter.

Resolution: the primary event is exact terminal joint correctness at K equal to semantic depth. The
`0.40` gate applies to every seed × required depth separately: validation depths 2–4 and both deep
splits at depths 5–12. Depth 1 and all component/trajectory metrics are diagnostics only. Exact row
matrices and task identities must be complete before classification.

### 5. Conditional follow-up creates selection bias

Using the same held-out rows to trigger full rank and then claim a full-rank-minus-LoRA rescue would
condition the comparison on those rows.

Resolution: Stage A and Stage B use fresh trigger splits. Three counterpart contrast splits remain
sealed at the outcome level: 768 trained-depth rows at seed 73307 and 1,024 rows at each of the deep
and joint-shift seeds 73305 and 73306. This is necessary because deep-only contrasts could not test
whether a trigger `TRAINED_DEPTH_MISS` replicated. The LoRA-miss receipt licenses training but cannot
open them. A dedicated Stage-B seal binds every reached fixed final and trigger evaluation,
reproduces the LoRA miss/control
evidence, proves exact cross-arm matching, and checks an empty access ledger without decompressing
sealed rows. A full-rank trigger miss goes directly to Stage C and leaves contrast rows unopened.
Only a full-rank trigger pass authorizes the exact six capacity×seed fixed-checkpoint evaluation jobs;
each job scores all three sealed splits in intact and disabled modes. The sealed adjudication also
requires the LoRA absolute miss to replicate across the complete trained-depth, depth-extrapolation,
and joint-shift matrix. If LoRA passes every sealed cell, the terminal label says the trigger miss did
not replicate; full-rank superiority may not be relabeled as evidence that LoRA prevented formation.
If full rank passes all absolute cells but LoRA fails only in different sealed categories, rescue is
still prohibited: every trigger-failed category must fail again in its corresponding sealed domain,
while additional sealed failures are allowed. A mismatch emits
`LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_ON_SEALED_CONTRAST`. Full-rank absolute failure takes
priority over this rescue guard and opens the registered state-only control.

### 6. Deleting evidence could cosmetically restore a “clean” firewall

An append-only ledger is ineffective if a partial contrast run can be erased or rerun under a new
path.

Resolution: the ledger records the exact Stage-B receipt, cell, fixed checkpoint, and canonical
evaluation path before decompression. A retry may reuse that event only for the identical
cell/checkpoint/path after the incomplete output is preserved in the content-addressed failure
archive. Every replay must contribute exactly one newly tracked, content-validated
`FAILED_ATTEMPT_ARCHIVED` receipt; its archive/tree/source/design/lock lineage is reopened and then
appended to the event before decompression. Event and ledger identities are recomputed, and mutation
holds a separate stable lock inode across an atomic, temporary-file-and-parent-directory-fsynced
replacement. Initial access rejects an archive predating its event. A previously bound archive cannot
license another retry, multiple new archives fail closed, and a completed evaluation cannot be
archived as failed.
Conflicting or unexplained access burns these contrast rows for rescue inference and requires a fresh
successor.

### 7. Shared seeds do not imply shared initialization or stochastic realization

Capacity-specific construction can consume RNG differently. Ordinary dropout can also shift even if
the nominal seed matches.

Resolution: common state is built in an isolated CPU RNG fork and serialized once per result seed.
Every arm reopens the exact bundle and proves tensor equality. Adaptation dropout uses the exact
capacity/objective-free preimage
`adaptation-dropout-v1|model-seed|global-microbatch-index|row-id|K`; target order, call cycles, and
realized masks are receipted. Active unreceipted dropout is a setup failure.

### 8. The custom LoRA hook could silently differ from the parent PEFT recipe

Algebraic similarity is insufficient because dtype, dropout, scaling, and gradient behavior can
differ.

Resolution: G0 compares the actual hook against the pinned PEFT `Linear` implementation on copied
deterministic tensors in exact FP32/dropout-off/autocast-off and live-like
bf16-autocast/dropout-0.05 regimes, including outputs and A/B gradients. The tolerances are
`atol=1e-6`, `rtol=1e-5` and `atol=2e-3`, `rtol=1e-2`, respectively. The stochastic probe resets the
same device RNG seed immediately before each implementation's forward so both consume the
corresponding first native-dropout mask; the custom hook receipts its realized mask and one-call cycle.
This is a tensor reference, not another model or result arm. The live Qwen wrapper separately proves
target discovery, call ordering, K=1 bypass, and zero-function start.

### 9. A state readout could pass without depending on the learned adaptation

Shared initializer/heads might learn enough that the adaptation itself is unnecessary.

Resolution: every checkpoint is evaluated intact and adaptation-disabled on identical rows.
`ADAPTATION_REQUIRED` needs an intact absolute pass, disabled miss, positive effect in every seed, no
negative pooled depth, and positive crossed lower bounds on all three sealed splits. If disabled also
passes, the label is `ADAPTATION_NOT_REQUIRED_AT_INFERENCE`, not a capacity rescue. If intact misses
but disabled passes, report `ADAPTATION_DISABLED_REVERSAL`: branching remains intact-based, the case
cannot satisfy adaptation dependence, and both trigger and sealed analyses preserve the capacity-
specific reversal—including LoRA on the sealed matrix—rather than collapsing it into uncertainty.

### 10. Full-shape optimization is not mechanically comparable by default

The 892M-parameter update can dominate clipping, exhaust memory, or lack valid optimizer moments.

Resolution: adaptation and common-state tensors are disjoint AdamW/clipping groups. Every step is
receipted with both preclip norms and applied scales; final optimizer state must contain finite,
shape-matched FP32 moments. Per-seed G0 includes a real two-step gradient onset, ten-step timing,
worst-depth K=12 forward, destructive checkpoint reload, peak memory, and at least 4 GiB post-G0
headroom. OOM is a feasibility stop and cannot be converted into a LoRA result by shrinking the arm.

### 11. Setup controls could be mistaken for result evidence

A tiny overfit can validate mechanics but says little about held-out state formation.

Resolution: the 48-row factorial positive control is fresh, setup-only, seed-separated, and never
initializes a result arm. It must pass exact overfit and oracle-analysis gates for each reached
capacity/seed. Its only interpretation is that the state-supervision/readout path is mechanically
capable of fitting; held-out result cells still determine the verdict.

### 12. Mutable or external artifacts could detach analysis from training

Large checkpoints live outside git; copied status strings and ignored sidecars are insufficient.

Resolution: every artifact binds model/config/source/test/lock/design/data identities and reopens
upstream receipt bytes and canonical identities. External initialization and training artifacts have
tracked receipt/index mirrors. Checkpoints bind payload hashes, final metrics, every-step optimizer
receipts, setup lineage, row order, dropout realization, and fixed-final identity. An analyzer must
reopen all of them and reject path escape, stale bytes, missing rows, or mismatched branch ancestry.

## Residual limitations accepted by design

- The `0.40` threshold is a preregistered operational formation criterion, not a universal boundary
  for representation quality.
- Direct full shape is more expensive than LoRA; this experiment asks whether the practical recipe
  relieves failure, not whether it is compute-efficient.
- Readability does not prove the answer coda uses the state. A positive can motivate a separate
  utilization experiment but cannot silently open one here.
- No public benchmark is read or trained on, and no capability-gain claim is available. Consequently
  a matched-compute sample-more comparator is outside this measurement-only capacity adjudication.
- Three training seeds support the registered crossed analysis but do not license broad population
  claims beyond this model, substrate, and recipe.

## Final go/no-go conditions

The scientific design is coherent only if implementation enforces all of the following before a
model-bearing call: Qwen/Qwen3.5-4B and the pinned revision only; exact fresh split geometry; no
benchmark access; source/test/lock-bound artifacts; canonical clean-at-HEAD design freeze; exact
setup gates for every reached seed/capacity; fixed step-1500 checkpoints; no intermediate selection;
and fail-closed branch/firewall lineage. Failure of any condition is `NO_GO` for execution until a
mechanical repair passes review without changing this scientific contract.
