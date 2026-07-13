# Full-Rank Extra-R Delta: State-Carry Versus State-Bag

**Status:** in-progress · since 2026-07-13 · the preregistered matched seed-7401 pilot remains.

**G0 status: `MODEL_SMOKE_PASS`; scientific pilot not yet run.** This is the
preregistered capacity successor to `qwen35_4b_state_carry_vs_state_bag`. Its
valid rank-32 LoRA pilot formed almost no joint state (`0.00459` joint trajectory
accuracy), so it could not answer whether the serial-state mechanism failed or
the adaptation subspace was too restrictive.

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

## Single changed factor

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
wrapper; only the source of state at calls 2..K differs.

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

The seed-7401 pilot remains a non-evidentiary promotion gate. Only a complete,
reachable pilot whose joint state specifically remains insufficient receives
`PILOT_STATE_FORMATION_MISS` and closes the held-fixed capacity branch.
Incomplete diagnostics or failures on another promotion requirement stop the
run as `PILOT_INCOMPLETE` or `PILOT_PROMOTION_BLOCKED` without licensing that
capacity conclusion. Only `PILOT_PROMOTION_READY` licenses seeds 7411–7413 and
the full causal ladder through G3. G4/sample-more is deliberately deferred: this
successor can resolve the LoRA-capacity counterfactual but cannot make a
deployment claim.

Every pilot checkpoint embeds the exact G0 receipt path, file hash, identity
hash, status, and phase that licensed training. Full checkpoints additionally
embed the corresponding G1 promotion receipt lineage. Checkpoint loading and
analysis fail closed on missing or malformed lineage, so later evidence cannot
silently outlive the gate chain that authorized it.

See [reports/preregistration.md](reports/preregistration.md),
[reports/design_review.md](reports/design_review.md), and
[docs/gpu_runbook.md](docs/gpu_runbook.md). Generated corpora and checkpoints are
omitted under [reports/artifact_manifest.yaml](reports/artifact_manifest.yaml).
