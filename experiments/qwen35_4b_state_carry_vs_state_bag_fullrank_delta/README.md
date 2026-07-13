# Full-Rank Extra-R Delta: State-Carry Versus State-Bag

**Status:** finished

**Authoritative terminal disposition: `PILOT_PROMOTION_BLOCKED`.** The preserved
analyzer receipt historically emitted `PILOT_STATE_FORMATION_MISS`, but that
classifier gave state failure priority over simultaneous non-capacity failures.
The frozen preregistration says those outcomes are mutually distinct. Because
Carry-minus-Bag and both registered answer strata also failed, the held-fixed
LoRA-rank capacity branch is **not closed**. Confirmation, edge cut, G3, and G4
remain unlicensed for these checkpoints.

## Pilot result and terminal disposition

The complete seed-7401 Carry/Bag pilot used identical initialization receipts,
data order, prompt tokens (`2,594,937` each), decoder-layer-token applications
(`145,316,472` each), fixed 300-step schedule, source/config/data identities,
and G0 lineage. Both final checkpoints reloaded with exact K=1 parity (`0.0`).

Full-rank Carry's macro task-mean joint node+phase+checksum step accuracy was
`0.0027686`, versus the frozen `0.40` sufficiency gate; node accuracy was
`0.06167`. The corresponding micro count was 7 jointly correct states over
2,176 registered steps (`0.0032169`). Carry minus Bag on the 256 matched primary
tasks was `-0.015625` (pilot 95% interval
`[-0.06640625, 0.0390625]`), with only three of eight depths positive. Unseen-K
scaling was `-0.0078125`
(`[-0.0625, 0.046875]`), and joint-holdout Carry minus Bag was `+0.01953125`
(`[-0.0234375, 0.0625]`). Under bidirectional swaps, donor following fell by
`0.0078125` (`[-0.0390625, 0.015625]`) and remained `0.078125` below recipient
preservation.

All registered cells were complete, the +0.05 answer gate was reachable, and
Carry's answer interface was valid (`0.97265625` full-top-is-answer). However,
the realized checks simultaneously had `joint_state_sufficient=false`,
`positive_carry_minus_bag=false`, and `query_kinds_positive=false`: node was
exactly `0.0` and checksum was `-0.03125`. The latter two are non-capacity
promotion failures under the frozen taxonomy, so the raw state miss is a useful
descriptive result but cannot license the registered capacity conclusion.

The cross-experiment comparison also did not preserve a bit-identical shared
state-module initialization or dropout RNG stream: constructing PEFT LoRA and
constructing then zeroing 892M direct parameters consume different random
streams before training. Direct deltas also change optimizer geometry and the
effect of global gradient clipping, even with the same nominal learning rate
and schedule. Thus this experiment establishes that one mechanically valid
direct-full-shape recipe also failed to learn the registered state; it does not
isolate LoRA rank as the cause or non-cause. A fresh, independently
preregistered capacity adjudication is mandatory. The current state was not
readable, so these checkpoints do not license the readable-but-unused interface
successor either.

## Program and question

- Primary research program: `structured_execution_and_compilers`.
- Parent/closest near-duplicate: `qwen35_4b_state_carry_vs_state_bag`.
- Trigger: the parent's complete, source-bound `PILOT_MECHANISM_MISS`, with a
  reachable answer-level gate and `joint_state_sufficient=false`.
- Question: holding the parent task, base path, supervision, recurrence, seeds,
  Carry/Bag control, and causal tests fixed, does removing LoRA's rank constraint
  permit the deeper joint representation to form?

The trigger is checked cryptographically and semantically at G0. A missing,
incomplete, infeasible, or wrong-parent identity-bound receipt aborts rather than silently
turning this into an unregistered architecture search.

## Registered intervention

The base is the pinned `Qwen/Qwen3.5-4B` revision. Layers 12–19 are still the
two complete repeated Qwen motifs. The parent LoRA is replaced by 62 direct,
zero-initialized, FP32 full-shape weight deltas totaling exactly 892,272,640
parameters. For a targeted linear on an **extra** R call only:

```text
y = W_base x + 2 * DeltaW * dropout(x, p=0.05)
```

`W_base` stays frozen. `DeltaW` is disabled for the prelude, first R call, and
coda. Thus K=1 is the exact original base-model path, not a merged or
approximately cancelled adaptation. Carry and Bag instantiate the identical
wrapper; only the source of state at calls 2..K differs within this successor.
That makes the successor's Carry/Bag comparison clean. It does not make the
comparison with the parent LoRA run a one-factor randomized contrast, because
parameterization-specific initialization, random-stream, and optimizer geometry
were not matched across experiments.

## G0 is a real feasibility gate

The direct deltas alone require 3.324 GiB for FP32 parameters, 3.324 GiB for
gradients, and 6.648 GiB for Adam moments: 13.296 GiB steady state before the
frozen model, activations, kernels, allocator slack, or optimizer temporaries.
G0 therefore runs an actual scheduled AdamW step, proves that every one of the
62 delta tensors has two finite, shape-matched FP32 Adam moments, records
allocated and reserved peaks plus reserved headroom, verifies finite K=12
recurrent logits, and round-trips both delta and loop-state files after
deliberately destroying the live tensors. It also independently proves
zero delta calls for Carry K=1 and Bag K=1, exact expected calls at K=4/K=12,
nonzero gradients in both arms, and no base-model gradient.

If G0 OOMs, stop. Reserved headroom is a diagnostic, not a post-hoc threshold.
Do not lower precision, switch models,
reduce targets, use a high-rank LoRA surrogate, or reinterpret the result as
scientific evidence.

The live gate passed on the 48GB RTX 6000 Ada. It discovered exactly 62 targets
and 892,272,640 FP32 delta parameters, produced nonzero delta gradients in both
arms, allocated both finite FP32 Adam moments for every delta, preserved exact
K=1 parity before and after the optimizer step, and produced finite K=12 logits.
Peak allocation was 24.49 GiB and peak reservation was 24.93 GiB, leaving 22.57
GiB of reserved headroom. A 3,571,392,174-byte delta-plus-loop checkpoint was
destroyed and reloaded with recurrent-logit error `0.0`. This establishes live
feasibility only; it is not a behavioral result.

## Data and evidence firewall

The generator is self-contained and never imports parent or benchmark code.
Nevertheless, every full split is bound to a parent-derived hash of canonical
decompressed rows. Regeneration checks row count, order, IDs, and every content
field for all 11 splits; when parent artifacts are present it also compares them
directly. Every model-bearing stage recomputes those canonical receipts from the
current artifacts and checks the full parity metadata rather than trusting a
copied pass flag.

The canonical preparation pass produced all 11 splits and 27,744 rows. It
matched both the frozen canonical-row contract and the locally available parent
artifacts exactly, with zero cross-split structural duplicates and zero
benchmark reads.

The seed-7401 pilot was a non-evidentiary promotion gate. Only a complete,
reachable pilot whose joint state specifically remains insufficient receives
`PILOT_STATE_FORMATION_MISS` and closes the held-fixed capacity branch.
Incomplete diagnostics or failures on another promotion requirement stop the
run as `PILOT_INCOMPLETE` or `PILOT_PROMOTION_BLOCKED` without licensing that
capacity conclusion. Only `PILOT_PROMOTION_READY` licenses seeds 7411–7413 and
the full causal ladder through G3. G4/sample-more is deliberately deferred: this
successor was designed to resolve the LoRA-capacity counterfactual but cannot
make a deployment claim. The realized cross-experiment controls were not strong
enough to identify rank, as detailed below.

Every pilot checkpoint embeds the exact G0 receipt path, file hash, identity
hash, status, and phase that licensed training. Full checkpoints additionally
embed the corresponding G1 promotion receipt lineage. Checkpoint loading and
analysis fail closed on missing or malformed lineage, so later evidence cannot
silently outlive the gate chain that authorized it.

The immutable analyzer emitted historical `PILOT_STATE_FORMATION_MISS`, but the
post-result audit applies the frozen taxonomy and authoritatively reclassifies
the run as `PILOT_PROMOTION_BLOCKED`, with `capacity_branch_closed=false`. Seeds
7411–7413, the same-checkpoint edge cut, G3, and G4 were correctly not run. The
pilot swaps are a one-seed diagnostic, not completed G3 causal identification.

A fresh adjudication must use fresh procedural evaluation rows, bit-identical
shared loop-state initialization and controlled CPU/CUDA/dropout RNG streams
across LoRA and direct-delta arms, an early held-out trained-depth state
positive control, and a full fixed-final multi-seed comparison. Representation
formation and answer/mechanism promotion must be separate verdict axes so a
simultaneous downstream miss cannot silently decide the capacity question.

See [reports/preregistration.md](reports/preregistration.md),
[reports/design_review.md](reports/design_review.md), and
[reports/terminal_science_audit.md](reports/terminal_science_audit.md) for the
authoritative disposition. Generated corpora and checkpoints are omitted under
[reports/artifact_manifest.yaml](reports/artifact_manifest.yaml).
