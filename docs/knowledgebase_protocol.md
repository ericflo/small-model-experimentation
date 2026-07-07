# Knowledgebase Protocol

## Pending: process the claim re-grade first

Before adding or promoting any claim, apply the outstanding corrections in
[`knowledge/claims/claim_review_todo.md`](../knowledge/claims/claim_review_todo.md)
(from the 2026-07-06 adversarial review). Hold `Confirmed` to a high bar
(replicated across seeds, controlled, ≥2 substrates); never state a 1–5-task
difference as "significant"/"decisively" without the actual test or an
eval-noise caveat; headline the **deployable** (no-think, single-shot) metric,
not the think-mode/oracle/cov@k one; and when a later claim refutes an earlier
one, update the earlier claim's status and cross-reference in the same pass.

## What Belongs In The Knowledgebase

Put cross-experiment claims in `knowledge/` only when they cite or link to specific experiments. The knowledgebase should make reuse easier without flattening uncertainty.

Good entries:

- a repeated bottleneck observed across experiments,
- a result that changes which direction should be tried next,
- a negative control that prevents a tempting wrong explanation,
- a reusable evaluation pattern.
- a program-level direction or stop condition.
- a decision record for a strategic pivot that affects more than one program or experiment.

Avoid entries that are only aspirations or generic advice.

## Evidence Labels

Use these labels in synthesis notes:

- `Confirmed`: directly supported by a result-bearing run.
- `Promising`: supported by a pilot or small run, needs scale or replication.
- `Negative`: tested and failed under the recorded setup.
- `Open`: plausible and not yet adequately tested.

## Updating Generated Files

Do not hand-edit generated files:

- `knowledge/claims/index.md`
- `knowledge/claims/index.csv`
- `knowledge/research_program_index.md`
- `knowledge/research_program_index.csv`
- `knowledge/experiment_catalog.md`
- `knowledge/experiment_catalog.csv`
- `knowledge/tag_index.md`
- `knowledge/artifact_index.md`
- `knowledge/source_tracks.md`
- `knowledge/source_tracks.csv`
- `knowledge/experiment_manifest.json`
- `knowledge/readme_coverage.md`

Regenerate them with:

```bash
make catalog
```

Before committing, run the full gate:

```bash
make check
```

## Updating Program Files

Hand-edit program files when a result changes direction:

- `research_programs/<program>/charter.md` for purpose or boundary changes,
- `research_programs/<program>/backlog.md` for next experiments,
- `research_programs/<program>/evidence.md` for seed evidence and current read.

Update `knowledge/program_scorecards.md` when the best next experiment, strongest anchors, or anti-duplication guidance changes.

Use `knowledge/decision_records/` for strategic pivots that affect multiple experiments or programs.

Update `knowledge/claims/claim_ledger.json` when a result changes a durable claim. The generated claim index validates claim ids, statuses, programs, and evidence references.
