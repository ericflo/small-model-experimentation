# Universal-Line Medium-Tier Measurement

The contamination-free universal line's first medium-tier paired benchmark event: four published composites, one fresh sealed seed, the goal gate recorded at the granularity where history says it is winnable.

**Status:** in-progress · since 2026-07-15 · model-free construction under way; the single benchmark event has not run

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the tier forensics (goal gate 9/94 at medium vs 1/84 at quick; base never at a family ceiling at medium; menders/sirens constants are quick artifacts); the goal-gap pilot (7 families up, 0 down at quick, gate failed on the two artifacts); replay_repeat 0.5081 best-ever quick aggregate.

## Question

Where does the line's Pareto set actually stand at medium: does the quick ordering hold, how close is each arm to the recorded all-ten-families goal gate, and which families block the next dose?

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Arms (explicit merges, tree-hash bound at event time): `base` (tree 26d8ee48…, weights b654e033…), `designed_fresh` (93433aa2…), `replay_repeat` (4c4f3561…), `hygiene_explore` (9eb653d7…).
- Event: tier medium, think budget 1,024, sealed fresh seed 78,150, trusted gateway only, one-seed ledger, sequential same-seed runs in frozen order.
- Readings (no promotion bars): medium aggregates + ordering vs quick; recorded goal gate (strict wins vs base per family); base sanity envelope vs the forensics' historical distribution; blocking families per arm.

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_medium_tier_measurement/scripts/run.py --smoke
.venv/bin/python -B experiments/qwen35_4b_universal_medium_tier_measurement/scripts/run.py --stage benchmark
```

## Results

The benchmark event has not run.

## Interpretation

None yet.

## Knowledgebase Update

- Program evidence updated: pending the event.
- Program backlog updated: this cell is the forensics' funded successor.
- Claim ledger updated: no.

## Artifacts

- `data/design_receipt.json`: seed/tier/budget/model/gateway/forensics pins.
- `reports/preregistration.md`, `reports/benchmark_design_review.md`: contract and authorization.
