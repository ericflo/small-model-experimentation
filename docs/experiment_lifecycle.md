# Experiment Lifecycle

## 0. Choose The Program

Every new experiment must advance an existing research program or justify a new one. Start from `research_programs/README.md` and `knowledge/research_program_index.md`.

If no current program fits, create one before adding the experiment:

```bash
make new-program PROGRAM=<program-id> TITLE="<Title>" FOCUS="<one-sentence focus>"
```

## 1. Claim The Question

Create the experiment scaffold before the expensive run:

```bash
make new-experiment EXPERIMENT=<id> PROGRAM=<program-id> TITLE="<Title>"
```

If the idea already appears in the future queue, scaffold from that item instead:

```bash
make from-queue PROPOSAL=<future_queue_id>
```

This creates the experiment, copies the queue item into `queue_proposal.json`, and pre-fills `idea_intake.md`. Candidate program lines remain context until they are promoted into `research_programs/registry.yaml`; the scaffold is attached to registered programs.

Keep the question narrow enough that a single result can move belief.

If the work is a follow-up benchmark, replication, ablation, or design variant of
an existing result-bearing experiment, create a new experiment directory and copy
the prior harness/artifacts into it before modifying. Do not append a new
substrate or follow-up result to the prior experiment.

The README should include:

- Research program.
- Question.
- Hypothesis.
- Baseline and controls.
- Dataset or task source.
- Smoke command.
- Full command.
- Expected primary metrics.
- Artifact plan.
- External or omitted artifact manifest plan.

## 2. Make The Smoke Path Real

Run a small version first. Save the smoke config and enough output to prove the path works. A smoke result is not evidence for the hypothesis, but it is evidence that the run path is alive.

## 3. Run With Controls

Prefer controls that test the mechanism:

- shuffled labels or shuffled retrieval queries,
- random retrieval or random acquisition,
- frozen-model versus trained-model comparisons,
- public-test versus hidden-test separation,
- oracle ceilings clearly labeled as non-deployable.

## 4. Analyze

Put derived outputs under `analysis/` and final narrative under `reports/`. Preserve raw run outputs when they are small enough or update `reports/artifact_manifest.yaml` for external or omitted artifacts.

## 5. Update Shared Knowledge

Run:

```bash
make check
```

Then update the human knowledge pages when the result changes strategy.

Also update the owning program's `evidence.md` and `backlog.md` when the result changes what should be tried next.

## 6. Publish To The Site (required)

Every experiment must reach the public site with a **plain-language practitioner
brief** — the friendly top-of-page summary (verdict, plain question/answer,
why-it-matters, KPI numbers, per-chart how-to-read). Charts and dates are handled
automatically (charts via your result data; dates via git), but the brief is
model-authored and is a **hard gate**: `make check` fails until it exists.

After a new experiment is committed, author its brief:

```bash
make site-dates          # git-fill the run date (deterministic)
# author the brief for the new experiment id(s):
#   run the workflow scripts/enrichment/enrich_briefs.workflow.js with args = the id(s),
#   then: python3 scripts/enrichment/merge_briefs.py --in <workflow-output.json>
make site-content        # confirm coverage is 100%
```

The brief must stay jargon-free (all the precise technical detail already lives in
the README/report below the fold). Full guide: [`site_maintenance.md`](site_maintenance.md).
`make check` will not pass until every experiment has a brief.
