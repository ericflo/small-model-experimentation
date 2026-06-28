# Idea Intake Protocol

Use this protocol before creating a new experiment. It is lighter than a proposal review, but strict enough to prevent repeated variants from displacing genuinely new evidence.

## When To Use It

- A new experiment idea is forming.
- A new research program might be needed.
- A result suggests a branch, stop condition, or major strategic pivot.
- An agent is about to run work that overlaps prior experiments.

## Intake Steps

1. Start from [program_scorecards.md](../knowledge/program_scorecards.md).
2. Check [future_experiment_queue.md](../knowledge/future_experiment_queue.md) for an existing queued probe or candidate program line.
3. Run `make related QUERY="<rough idea>"`.
4. Find the closest program and at least three prior anchors.
5. Write the novelty claim as one sentence.
6. Name the closest duplicate or near-duplicate experiment.
7. State the control that could falsify the mechanism.
8. Decide the evidence output before running anything.

## Outcomes

- Create a new experiment with `make new-experiment`.
- Create a new experiment from a queued proposal with `make from-queue PROPOSAL=<future_queue_id>`.
- Create a new program with `make new-program`.
- Add a synthesis note instead of running a duplicate experiment.
- Write a decision record if the idea changes portfolio direction.

## Required Artifact

Use [templates/idea_intake.md](../templates/idea_intake.md) for material ideas. Store completed intake notes inside the relevant experiment folder or decision record. The intentionally curated central queue is [future_experiment_queue.json](../knowledge/future_experiment_queue.json); loose proposals should either enter that schema or stay local to an experiment or decision record.

For a prefilled note, run:

```bash
python scripts/find_related.py "<rough idea>" --write-intake experiments/<id>/idea_intake.md
```
