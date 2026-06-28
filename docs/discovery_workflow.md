# Discovery Workflow

Use this workflow before turning an idea into a new run. It is designed to route new work toward the closest research program, find relevant prior experiments, and expose claims that the idea may support or contradict.

## Quick Search

```bash
make related QUERY="visible-only selector with abstention for candidate pools"
```

The command searches:

- `research_programs/registry.yaml` plus program charters, backlogs, and evidence,
- `knowledge/future_experiment_queue.csv`,
- `knowledge/claims/index.csv`,
- `knowledge/experiment_catalog.csv`.

It returns the closest programs, queued future work, claims, and experiments with matched terms.

## Intake Note

Create a prefilled intake note:

```bash
python scripts/find_related.py "typed bytecode compiler with corrupted-state controls" \
  --write-intake experiments/<id>/idea_intake.md
```

Then complete the missing mechanism, control, hidden-label, and evidence-output fields before running the experiment.

## How To Use Results

- If the top experiments already answer the uncertainty, write a synthesis update instead of a duplicate run.
- If a queued future-work item already matches, use it as the starting proposal and update the queue when the decision changes.
- If the top program fits, attach the experiment there.
- If no program fits after reading scorecards and related claims, create a new program.
- If a related claim would change status after the proposed result, update `knowledge/claims/claim_ledger.json`.
