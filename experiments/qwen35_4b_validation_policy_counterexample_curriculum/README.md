# Validation-policy counterexample curriculum

**Status:** in-progress · since 2026-07-13 · design locked at `e0b19f5d`;
GPU smoke and staged model run remain.

Teach Qwen3.5-4B the semantic distinction left by the transaction curriculum:
negative quantities are malformed and raise `ValueError`, while unknown or
insufficient resources are ordinary `False` decisions.

## Research program

- Program: `agentic_breadth_installation`.
- Direct predecessor: `qwen35_4b_transaction_invariant_recovery_curriculum`.
- Supporting lineage: `qwen35_4b_recovery_verifier_branch_tournament`,
  `qwen35_4b_verifier_conditioned_recovery_bank`, and C54's apex incumbent.

## Question

Can a residual action-seam intervention teach verifier-faithful exception
versus rejection policy from near-correct failed-test states, transfer across
new repository schemas, and retain the transaction structure and conditional
tool loop already installed in the parent?

## Why this is the next test

The predecessor installed the intended proposal structure: on all 16 atomic-
reservation cases, the first changed patch copied state, checked the whole
request, committed atomically, and returned `False` for ordinary rejection.
Every patch nevertheless omitted the separate negative-amount exception. After
visible failure, every trajectory overcorrected by raising on all unavailable
or insufficient requests. Generic transaction dose is therefore the wrong
next lever; the remaining error is policy discrimination after verifier
feedback.

## Frozen intervention

- Parent: the local transaction candidate, exact weight SHA-256
  `1cf5fbca317808d6d00225f5cd533c82c7e1602b2b2e5e2da8f4307b01941ba3`.
- Control source: the predecessor's exact 48-task, 336-row
  transaction-plus-recovery bank, SHA-256
  `9c196d1e7e49881bbf151e1575c98811bcdca66e6ef38858c34f60f1256b9315`.
- Candidate bank: the same 48 complete task blocks, except the
  `diagnosis_to_changed_patch` row in each of 24 recovery blocks is replaced
  by one fresh near-correct validation-policy revision. The other 312 rows are
  retained from the prior bank.
- Matched control: all 336 prior rows receive the same extra update from the
  same parent.
- Both arms have zero think loss, identical transition/operator counts,
  38,248 weighted action tokens per operator per epoch, and three epochs at
  LR 2e-5, rank 32/alpha 64, batch 4 × seven transition-stratified accumulation
  steps.

Training uses six API skins spanning bundle mappings, record objects, and tuple
requests. Transfer uses three new skins across the same three representations.
`atomic_reservations` is a separately gated predecessor sentinel and is never
counted as unseen transfer.

## Evidence hygiene

Every generated initial and partial repository fails visible and hidden
executables; every oracle passes. Public-content hashes prove all 24 bank and
24 calibration repositories unique and disjoint, all 32 development and 32
confirmation repositories unique and mutually disjoint, and all official
manifests stable across Python hash seeds. Hidden tests and repair objects stay
host-side. Nothing under `benchmarks/` is read or imported; Menagerie is run
only through its public CLI and only aggregate/per-family scores are retained.

## Gate order

1. Lock the frozen design commit and verify exact hashes, bank composition,
   transition completeness, action-mass parity, executable replay, and
   firewall cleanliness.
2. Train and merge both fixed arms. Before behavior generation, compare the
   candidate directly with C54 apex on 48 fresh non-coding contexts; require
   centered non-target drift ≤0.15 and entropy delta ≥−0.05, while recording
   varentropy diagnostically.
3. Generate parent and matched-control calibration receipts first. If the
   bars are feasible, expose the candidate and require ≥80% success, +15
   points over parent, +10 over control, transition retention, and interface
   validity.
4. On one known sentinel plus three fresh transfer families, compare candidate
   with parent, matched extra-transaction training, and equal-reservation
   parent sample-more. Require +10/+5/+5 points, nonnegative paired-bootstrap
   lower bound versus parent, ≥50% sentinel success, nonnegative transfer on
   all three fresh families, and verify/commit/transition/interface retention.
   Repeat unchanged on a content-disjoint seed.
5. Require broad recovery and normal-loop retention on four older families.
6. Only an all-pass white-box battery runs frozen paired Menagerie quick seed
   71301 and medium seed 71302 against C54 apex. At least one tier must gain two
   points and neither may lose more than three.

Exact thresholds and stop labels are frozen in
[`reports/preregistration.md`](reports/preregistration.md).

## Run

```bash
python experiments/qwen35_4b_validation_policy_counterexample_curriculum/scripts/run.py --smoke
.venv/bin/python experiments/qwen35_4b_validation_policy_counterexample_curriculum/scripts/run.py --lock-design <design-commit>
.venv/bin/python experiments/qwen35_4b_validation_policy_counterexample_curriculum/scripts/run.py --gpu-smoke
.venv/bin/python experiments/qwen35_4b_validation_policy_counterexample_curriculum/scripts/run.py --full
```

## Current evidence

Deterministic CPU preflight passed. The candidate and control each contain 48
complete task blocks and 336 rows; exactly 24 candidate rows carry the new
semantic revision, 312 candidate rows retain prior behavior, and weighted
action mass is identical for every transition and operator. This is design
evidence only. No result or capability claim exists until the staged run
finishes.

## Artifacts

Committed design, code, tests, and compact receipts live here. Banks, adapters,
merged checkpoints, logits, and detailed trajectories live under
`large_artifacts/qwen35_4b_validation_policy_counterexample_curriculum`, as
documented in [`reports/artifact_manifest.yaml`](reports/artifact_manifest.yaml).
