# Axis Stack Re-adjudication with Medium Pilot

Re-judge the published axis-stack composites on a fresh instrument with the measured ceiling-tie flaw corrected prospectively — control-ceiling kinds excluded and reported as not-detectable, wins required on two-thirds of detectable kinds — then fund the medium-tier pilot conditionally. Training-free; both prior failures remain recorded and their seeds sealed.

**Status:** in-progress · since 2026-07-15 · model-free construction under way; no model event has run

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the axis install replicated across two parents (24/40 vs 18/15; hygiene 9/10 twice) and was blocked once by the aggregate pilot's replay comparison and once by a single breadth check whose protocol kind tied at the parent ceiling in both experiments.

## Question

Measured fairly on fresh tasks — with undetectable kinds excluded rather than silently tightening the quota — do the already-installed axis skills clear the program's relative bars, and does medium-tier granularity then convert them at the family level?

## Hypothesis

The install is real (replicated twice); the prior block was instrument noise (a systematic ceiling tie plus a control kind-fluke). The corrected bar removes exactly that noise without weakening any other condition, adds a fail-closed GATE_UNDETECTABLE outcome, and leaves retention bands unchanged.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Arms (all inherited published composites, weight- and tree-pinned): `replay_parent` (`3df45004...0072`), `replay_squared` (`e43b885c...069e`), candidate `axis_on_replay` (`7ebcad39...d0e4`). No training, merging, or exposure matching.
- Gate: fresh seed 88,016, the standard two instruments (40-task axis holdout, 104-task retention screen). Corrected promotion: axis total strictly above both controls; strict wins on at least two-thirds (rounded up) of DETECTABLE kinds (a kind is undetectable if either control scores ≥ 9/10; undetectable kinds are reported, not counted); retention non-inferiority bands unchanged; route abstentions ≤ 4; zero detectable kinds fails closed as `GATE_UNDETECTABLE`.
- Conditional pilot: sealed seed 78,146, MEDIUM tier, think budget 1,024, four weight-authenticated composites (base, both controls, candidate); candidate aggregate strictly above base, replay_squared, and replay_parent; the every-family-versus-base record is the goal gate at the tier where it has passed 8 of 92 events.
- Hidden boundary: `benchmarks/` unread.

## Run

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_axis_stack_readjudication_medium_pilot/scripts/run.py --smoke
```

Checkpointed stages:

```bash
.venv/bin/python -B experiments/qwen35_4b_axis_stack_readjudication_medium_pilot/scripts/run.py --stage local
.venv/bin/python -B experiments/qwen35_4b_axis_stack_readjudication_medium_pilot/scripts/run.py --stage benchmark
```

## Results

No model event has run.

## Interpretation

None yet.

## Knowledgebase Update

- Program evidence updated: pending the corrected-instrument reading.
- Program backlog updated: this trial claims the queued re-adjudication slot.
- Claim ledger updated: no.

## Artifacts

- `data/local_tasks_seed88016.jsonl`, `data/local_input_seed88016.jsonl`, `data/local_design_receipt.json`: frozen gate.
- `reports/preregistration.md`, `reports/design_review.md`: contract and authorization.
- `reports/artifact_manifest.yaml`: inherited composite pins.
