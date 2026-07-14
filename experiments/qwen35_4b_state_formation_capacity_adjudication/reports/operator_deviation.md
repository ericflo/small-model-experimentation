# Operator Visibility Deviation

**Date:** 2026-07-14

**Disposition:** preserve the event; complete the already-fixed seed-7413 evaluation unchanged;
disclose imperfect operator blinding in every terminal interpretation.

## Event

After the source-v11 seed-7412 trigger evaluator exited successfully, the operator intended to print
only identity, access, row-count, and row-hash metadata from `summary.json`. The projection included
the complete `modes` mapping. That mapping contains per-split and per-depth scientific metrics, so
seed-7412 values were printed before seed 7413 and before the registered three-seed analyzer.

The exposure occurred after the evaluation receipt and both row files were atomically complete. It
did not affect generation, scoring, checkpoint selection, or receipt contents. The later corrected
projection selected only row counts/paths/hashes and lineage/access fields.

## Boundary assessment

This violates the intended and explicitly recorded within-stage rule that scientific values remain
uninspected until the fixed three-seed matrix is complete. It must not be described as blinded.

It does not change the frozen scientific computation:

- model, revision, backend, source v11, config, data, seeds, checkpoints, row matrices, thresholds,
  and analyzer were committed before the exposure;
- seed 7413 is a fixed mandatory cell, not an optional replication;
- no seed replacement, retry, early stop, checkpoint choice, code/design repair, analysis,
  classification, or conditional branch occurred after exposure;
- the source-bound analyzer refuses to classify until all three exact evaluation cells exist;
- sealed contrast remained unopened, with no access event or authorization.

Thus the only defensible continuation is to publish the deviation before seed 7413, require green
checks, execute that fixed cell exactly once, and then run the unchanged preregistered analyzer. The
terminal report must qualify the evidence as mechanically prospective but not perfectly
operator-blind. If any command, threshold, seed, or design is changed in response to the exposed
values, this experiment's confirmatory interpretation is invalid and the work must move to a fresh
successor.

## Non-authority

The exposed partial values authorize nothing. They cannot support a LoRA verdict, stop the last
seed, open a full-rank or state-only arm, access sealed contrast, or motivate an in-place repair.
Only the exact complete v11 analysis receipt retains branch authority.
