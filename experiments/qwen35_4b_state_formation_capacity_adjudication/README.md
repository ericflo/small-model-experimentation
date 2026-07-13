# State-Formation Capacity Adjudication

**Status:** in-progress · since 2026-07-13 · frozen design unchanged; revision-provenance correction is `GO`; corrected-source setup is complete; LoRA G0 retry is next

## Current status

This is the canonical fresh adjudication of the unresolved LoRA-capacity question from
`qwen35_4b_state_carry_vs_state_bag`. It is not a continuation of either prior checkpoint.
Preregistration, adversarial design review, implementation review, and the frozen scientific design
are complete. The first seed-7411 LoRA G0 loaded only the pinned Qwen snapshot, then stopped before
wrapper construction or any mechanics probe because the pinned runtime drops the outer commit hash
when it derives the Qwen text config. No sealed contrast was opened, no canonical G0 receipt was
written, and no positive control, result training, evaluation, or scientific analysis ran. The
source-bound correction records and commit-verifies every config/tokenizer/index/shard path through one
pinned snapshot, forces local-only safetensors loading, passes 133/133 tests, and has independent
implementation-review `GO`. The prior-source setup is preserved as invalidated history. CPU smoke,
all seven procedural splits, the empty contrast-access ledger, and all three common initialization
bundles have now been regenerated under source digest `9fd420f5…614fb` and independently reopened.
The next step is to retry the three LoRA G0 gates and their seed-matched setup-only positive controls.

## Research program and prior anchors

- Primary program:
  [`structured_execution_and_compilers`](../../research_programs/structured_execution_and_compilers/charter.md),
  because the scientific endpoint is whether a recurrent latent execution state forms.
- Secondary program:
  [`posttraining_and_adaptation`](../../research_programs/posttraining_and_adaptation/charter.md),
  because the experiment compares a factorized LoRA update with a direct full-shape update and tests
  how joint versus state-only posttraining changes that state. This is secondary program fit, not a
  second estimand or an additional verdict axis.
- Closest anchor: [`qwen35_4b_state_carry_vs_state_bag`](../qwen35_4b_state_carry_vs_state_bag/README.md),
  whose valid rank-32 LoRA pilot failed joint state formation after 300 steps.
- Mandatory capacity anchor:
  [`qwen35_4b_state_carry_vs_state_bag_fullrank_delta`](../qwen35_4b_state_carry_vs_state_bag_fullrank_delta/README.md),
  whose direct-delta pilot also formed almost no state but did not match cross-capacity shared
  initialization or dropout RNG and simultaneously failed non-capacity promotion gates.
- Earlier recurrent negative: [`qwen_fastweight_hook`](../qwen_fastweight_hook/README.md), whose
  256-dimensional answer-supervised hook showed no robust K scaling under larger retests.

The novelty is a fresh, three-seed, fixed-final capacity adjudication that changes only the
registered extra-call adaptation parameterization while explicitly matching shared loop-state
tensors, row order, and adaptation-dropout streams. It separates the original joint objective from a
state-only control and gives state formation its own verdict, independent of Bag, answer-gain,
edge-cut, swap, or sample-more gates.

## Question

Does rank-32 extra-call LoRA prevent the repeated Qwen block from forming the registered deep joint
`(node, phase, checksum)` representation under the original joint training objective? If LoRA misses,
does direct full-rank parameterization rescue formation under the same seeds and controlled
initialization/stochastic streams, or does the failure remain when answer competition is removed?

## Hypothesis

LoRA is plausible: its 62 rank-32 updates act throughout two complete repeated Qwen motifs, while the
carried state remains full width and receives dense state supervision. A valid three-seed LoRA joint
pass would therefore show that low rank does not prevent state formation in this design and would
prohibit the expensive full-rank branch.

If LoRA joint misses, the state-only LoRA control gives a descriptive pattern consistent with answer
loss competition without causally identifying it, while a mandatory full-rank joint arm tests the
practical rank/parameterization concern.
Full-rank relief, with an adaptation-dependent state gain, supports a practical LoRA limitation. If
both joint parameterizations miss and their setup controls are valid, full-rank relief was not
sufficient. If both state-only controls also miss, the registered recipe bottleneck remains
unresolved between supervision/readout architecture and optimization rather than justifying another
rank retry.

## Frozen design

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, through Transformers 5.13.0.
- Runtime: the parent's causal Carry loop over layers 12–19, with eight state-before-query slots,
  untouched non-state memory between extra calls, and exact base K=1 behavior.
- Data: fresh procedural pointer-world rows from the same substrate logic; no benchmark content and no
  reuse of the prior experiments' evaluation rows.
- Training: seeds 7411, 7412, and 7413; exactly 1,500 optimizer steps; fixed final checkpoint only;
  batch one with 16-way accumulation; no pilot, early stopping, seed replacement, or checkpoint
  selection.
- Joint objective: `1.0 * answer + 0.5 * state + 0.05 * fixed_point`.
- State-only control: `0.5 * state + 0.05 * fixed_point`; the answer term is absent from the graph.
- LoRA: rank 32, alpha 64, 0.05 adaptation dropout, active only on extra R applications.
- PEFT compatibility gate: the actual custom hook must match a pinned PEFT `Linear` reference in
  output and A/B gradients under both FP32/dropout-off/autocast-off (`atol=1e-6`, `rtol=1e-5`) and a
  live-like bf16-autocast/dropout-0.05 regime (`atol=2e-3`, `rtol=1e-2`). Both forwards receive
  copied base/A/B/input tensors and the same device RNG reset immediately before each forward, so the
  live-like comparison uses the matched realized mask protocol rather than merely equal nominal
  probabilities.
- Full rank: 62 zero-initialized direct FP32 deltas, 892,272,640 parameters, the same 0.05 adaptation
  dropout and scale 2, active only on the same extra R applications.
- Pairing: a common custom adaptation-hook path, one bit-identical shared loop-state initialization
  bundle per seed, capacity-specific construction RNG, deterministic row order, and a matched
  per-microbatch dropout schedule that excludes capacity and objective from its seed.
- Optimizer isolation: shared loop-state and adaptation parameters are clipped as separate groups, so
  the dense arm's gradient norm cannot rescale the common-module update.
- Evaluation: every checkpoint is scored both intact and with adaptation disabled while retaining its
  trained shared state modules and readout heads.

## Sequential firewall

1. Run the mechanically validated, setup-positive-control-qualified LoRA **joint** objective for all
   three seeds.
2. If LoRA clears every registered state-formation gate, emit
   `LORA_DOES_NOT_PREVENT_STATE_FORMATION` and prohibit all further capacity arms.
3. If a complete, valid LoRA joint result misses any formation gate, its identity-bound analysis
   receipt mandates three LoRA **state-only** controls and three full-rank **joint** runs on the same
   seeds.
4. If full-rank joint misses any trigger cell, run the three full-rank **state-only** controls without
   opening sealed data. If its trigger passes but a sealed absolute cell later misses, open the same
   state-only branch only after applying the all-cell LoRA-pass priority in step 5. These are the only
   paths to the maximum 12 result-bearing runs.
5. If full rank passes its trigger, a dedicated seal opens three fresh contrast splits for both joint
   arms: trained-depth validation, depth extrapolation, and joint shift. The evaluator still runs six
   capacity×seed jobs because each job scores all three splits in both intact and disabled modes.
   A LoRA pass there emits `LORA_TRIGGER_MISS_NOT_REPLICATED_ON_SEALED_CONTRAST` and prohibits a
   rescue claim regardless of full-rank score. Otherwise a full-rank absolute miss still mandates
   state-only. If full rank passes, every LoRA category that failed on trigger must fail again in its
   corresponding sealed domain; otherwise emit
   `LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_ON_SEALED_CONTRAST` and prohibit rescue. Extra
   sealed failures cannot substitute for a missing category replication.
6. Missing cells, mechanics failures, initialization/dropout mismatches, or positive-control failures
   authorize repair only. They never become a scientific terminal result and never authorize a seed
   substitution.

## Primary metric and scope

The primary event is exact terminal joint state correctness at K equal to semantic depth: node,
phase, and checksum must all be correct. A parameterization passes only when **every seed×depth cell**
reaches at least `0.40` on trained depths 2–4, unseen depths 5–12, and the depth-5–12 joint held-out
family-plus-surface split. Depth 1 is reported but excluded because K=1 bypasses adaptation. Terminal,
trajectory-step, component, trained-depth, unseen-depth, and joint-shift results remain separate;
none may be pooled to hide a failed depth.

Fresh trigger splits contain 1,024 rows each for validation, depth extrapolation, and joint shift.
After a LoRA miss, a 768-row trained-depth validation contrast at seed 73307 (depths 2–4, 256/depth)
and two 1,024-row deep contrasts at seeds 73305 and 73306 remain sealed until all
three LoRA-joint, three LoRA-state-only, and three direct-full-shape-joint fixed finals and their
intact/disabled trigger evaluations are complete. A dedicated identity-bound
`STAGE_B_CONTRAST_AUTHORIZED` receipt—not the earlier LoRA-miss receipt—must also prove that no
contrast scoring output or prior-open record exists. It is emitted only if direct full shape passes
every trigger cell; a trigger miss mandates state-only immediately and leaves the contrasts unopened.
Only the authorization receipt can open those rows, and only those sealed rows support a
cross-capacity rescue contrast.

An interrupted sealed evaluation may retry the same cell/checkpoint/canonical path only after its
incomplete output is moved to the content-addressed failure archive. Each retry must discover exactly
one newly tracked and content-validated `FAILED_ATTEMPT_ARCHIVED` receipt; that receipt is appended to
the existing access-ledger event and both event and ledger identities are recomputed. The ledger is
atomically and durably replaced while a separate stable lock inode is held, before decompression.
Initial access rejects an archive that predates its event; a completed evaluation cannot be archived
as failed, and an old archive cannot license another retry.

For every checkpoint, the analysis also reports intact minus adaptation-disabled state accuracy.
If intact misses formation while adaptation-disabled passes, it emits
`ADAPTATION_DISABLED_REVERSAL`: removing the trained adaptation improves the readable state enough to
reverse the absolute verdict. Branching still follows the preregistered intact checkpoint, and this
diagnostic prohibits an adaptation-required or direct-recipe-rescue interpretation.
The post-contrast receipt exposes `lora_sealed_contrast_adaptation` separately from the full-rank
sealed adaptation status, so a LoRA sealed reversal cannot be hidden by the cross-capacity terminal
label.
Any `DIRECT_FULLSHAPE_RECIPE_RESCUE` additionally requires preregistered positive
intact-minus-disabled and direct-full-shape-minus-LoRA effects on all three sealed splits, with positive
every-seed effects, no depth reversal, and crossed task-by-seed lower bounds above zero. This prevents
trained shared heads or conditional split reuse from manufacturing a rescue. Model-seed rows are
never treated as independent tasks.

The selection-safe replication rule is category-specific: trigger `trained`, `depth`, and `joint`
failures map to `contrast_validation`, `contrast_depth`, and `contrast_joint`. Every trigger-failed
category must fail again; additional sealed category failures are allowed. The category guard is
applied only after full rank passes its absolute trigger and sealed cells, because a full-rank miss
instead opens the registered state-only control.

This experiment can establish readable state formation and a practical direct-full-shape recipe
signature. It cannot identify mathematical rank alone, causal state use, answer improvement, serial
advantage over Bag, deployment capability, or a win over matched-compute sampling. Those require a
fresh successor.

## Run

The run is deliberately non-monolithic. Start with the non-model smoke:

```bash
python3 experiments/qwen35_4b_state_formation_capacity_adjudication/scripts/run.py --stage cpu-smoke
```

Then follow [`docs/gpu_runbook.md`](docs/gpu_runbook.md). Every model-bearing or branch stage must
reopen and verify the exact upstream receipt; a status string copied into another file is not
authorization.

## Expected artifacts

- `idea_intake.md`: duplicate search, novelty, and decision.
- `reports/preregistration.md`: frozen scientific contract and terminal taxonomy.
- `reports/design_review.md`: adversarial pre-run review; required before the design receipt.
- `reports/design_receipt.json`: canonical pre-model identity boundary once frozen.
- `docs/architecture.md`: common loop and adaptation-backend contract.
- `docs/gpu_runbook.md`: phase order, inspections, and recovery rules.
- `docs/research_handoff.md`: rationale and continuity.
- `reports/artifact_manifest.yaml`: tracked/external artifact policy.
- `runs/` and `analysis/`: runtime receipts and results after execution; no result exists yet.
