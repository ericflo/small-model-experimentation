# Agent Operating Guide

This repository is a compounding research memory. Optimize for preserving experiment autonomy while making findings easier to compare, reuse, and challenge.

## First Pass

1. Read [README.md](README.md).
2. Read [knowledge/synthesis.md](knowledge/synthesis.md).
3. Use [knowledge/experiment_catalog.md](knowledge/experiment_catalog.md) and [knowledge/tag_index.md](knowledge/tag_index.md) to find relevant prior work.
4. Before adding or changing experiments, read [docs/experiment_lifecycle.md](docs/experiment_lifecycle.md) and [docs/knowledgebase_protocol.md](docs/knowledgebase_protocol.md).

## Non-Negotiables

- Keep experiments self-contained under `experiments/<id>/`.
- Update the knowledgebase when a result changes what future work should believe.
- Preserve negative results and failed controls; they are part of the map.
- Record enough detail that another agent can rerun, audit, or extend the experiment.
- Run `make catalog` and `make validate` before committing.

## When Starting a New Experiment

Copy `templates/experiment/` into `experiments/<new-id>/`, fill in the README, and make the smallest runnable smoke path first. Add controls early enough that a positive-looking result can be falsified.

## When Editing Imported Work

Prefer additive notes in `README.md`, `experiment_log.md`, `analysis/`, or `reports/`. Do not rewrite historical outputs unless you are regenerating them from code and can explain the change.

