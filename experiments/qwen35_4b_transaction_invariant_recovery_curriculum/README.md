# Transaction-invariant recovery curriculum

**Status:** finished

Teach a single Qwen3.5-4B coding policy to validate whole requests, copy state,
and commit atomically while preserving its verifier-conditioned recovery loop.

## Research Program

- Program: `agentic_breadth_installation`.
- Direct predecessor: `qwen35_4b_recovery_verifier_branch_tournament`.
- Prior anchors: C49 (merged-checkpoint deployment), C52 (token-local
  uncertainty steering is not context-local), C54 (current apex incumbent), and
  the conditional recovery line.

## Question

Can a low-dose, action-seam curriculum install the missing transactional coding
invariant—validate the complete request, copy state, then commit atomically—into
the locality-safe recovery-action checkpoint without deleting its general
tool-loop behavior, and does that transfer to unseen transaction families and
the Menagerie?

## Hypothesis

The predecessor localized all 20 shared deterministic failures to atomic
reservations. The agents continued inspecting, patching, testing, and revising,
but alternated between whole-request validation and input nonmutation instead of
producing their conjunction. This is proposal failure, so a selector, extra
sampling, or another loop-recovery update cannot create the missing program.

Twenty-four fresh procedural repositories provide executable full-conjunction
repairs. Mixing their seven transition-complete action targets with 24 frozen
recovery task blocks should install the semantic invariant while retaining the
conditional loop. A matched control trains on 48 recovery-only blocks with the
same rows, epochs, optimizer, transition counts, operator mass, and parent
checkpoint. The hypothesis fails if the control matches the transaction arm,
if the gain does not transfer to unseen transaction families, or if locality or
broad recovery regresses.

## Frozen Intervention

- Parent: the merged recovery-action checkpoint, exact weight hash
  `991d2d...aea`; this already contains the C54 apex and the locality-safe
  seven-transition recovery update.
- Primary bank: 24 programmatic transaction tasks (six families × four) plus
  24 deterministic task blocks from the frozen recovery bank.
- Matched control: 48 deterministic frozen recovery task blocks.
- Each task contributes exactly one row for each of seven state→action
  transitions. Think-block loss is zero; only the JSON tool-action seam is
  supervised.
- Both arms are calibrated to 38,248 weighted action tokens per operator per
  epoch and receive six epochs from the identical parent.
- Primary selection is fixed to `transaction_replay`; the replay-only arm is a
  mechanism control, not a model-selection candidate.

The train families cover inventory orders, ledger transfers, seat groups,
multidimensional claims, flag batches, and rename batches. Transfer uses atomic
reservations as the predecessor sentinel plus unseen debit, membership-move,
and document-patch families. Every initial and partial implementation fails
both executable suites; only the full repair passes. Hidden executables and
repair objects remain host-side.

## Evaluation and Gate Order

1. Verify hashes, exact seven-transition task blocks, equal weighted action
   mass, procedural replay, and benchmark-firewall cleanliness.
2. Train and merge both arms; compare the fixed primary against the C54 apex on
   48 fresh non-coding contexts. Require centered non-target drift ≤0.15 and
   mean entropy delta ≥−0.05; record varentropy without using it as token
   pressure or a selection label.
3. On trained transaction families, run the parent and replay-only control
   first, prove the frozen bars attainable, then require primary success ≥80%,
   +15 points over the parent, and +10 over replay-only while retaining both
   recovery transitions and interface validity.
4. On four transfer families, compare primary against parent, replay-only, and
   equal-reservation parent sample-more. Require +10/+5/+5 points respectively,
   a nonnegative paired-bootstrap lower bound versus parent, transition
   retention, and no family collapse. Repeat unchanged on an independent seed.
5. On the four older broad-recovery families, require recovery and normal-loop
   success within three points of the parent plus verification, commit,
   transition, invalid-action, and payload-cap retention.
6. Only an all-pass white-box battery authorizes fresh paired Menagerie `quick`
   and `medium` events through the public CLI. Compare the single primary
   checkpoint to the frozen C54 apex: at least one tier must gain two points and
   neither may lose more than three.

Exact thresholds and stop labels are frozen in
[`reports/preregistration.md`](reports/preregistration.md).

## Run

```bash
python experiments/qwen35_4b_transaction_invariant_recovery_curriculum/scripts/run.py --smoke
.venv/bin/python experiments/qwen35_4b_transaction_invariant_recovery_curriculum/scripts/run.py --gpu-smoke
.venv/bin/python experiments/qwen35_4b_transaction_invariant_recovery_curriculum/scripts/run.py --full
```

## Results

**Verdict: `TRANSACTION_DEV_FAIL`.** The primary passed locality (0.119 drift),
then installed the training families strongly: 81.7% versus parent 51.7% and
replay-only 38.3%, with perfect two-turn recovery and improved interface
validity. On unseen transaction dev it reached 71.9%, versus parent and
equal-compute sample-more 70.3% and replay-only 64.1%. It therefore missed the
registered +10/+5 bars and paired lower-bound gate. Confirmation, broad
retention, and Menagerie remained sealed.

The mechanistic result is more specific than “no transfer.” On all 16 atomic-
reservation cases, the first candidate patch newly contained whole-request
validation, copied state, and atomic commit—the intended proposal shift—but
omitted the distinct negative-amount `ValueError` rule. After visible failure,
all 16 overcorrected by raising on every insufficient request. The next
iteration should teach verifier-faithful validation-policy distinctions from
near-correct failed-test states, not add generic transaction dose. Full results
are in [`reports/report.md`](reports/report.md) and
[`reports/result_receipt.json`](reports/result_receipt.json).

## Knowledgebase Update

- Program evidence and backlog record proposal-structure installation without
  task-success transfer and queue the counterexample-policy successor.
- Shared synthesis and program scorecard now distinguish transaction structure
  from validation-policy fidelity.
- Claim ledger remains unchanged: the transfer gate failed and no Menagerie
  event ran.

## Artifacts

Committed code, frozen design, compact gates, and result receipts live here.
Banks, adapters, merged checkpoints, logits, and detailed trajectories live at
`large_artifacts/qwen35_4b_transaction_invariant_recovery_curriculum` under
[`reports/artifact_manifest.yaml`](reports/artifact_manifest.yaml).
