# Axis Stack Re-adjudication with Medium Pilot

Re-judge the published axis-stack composites on a fresh instrument with the measured ceiling-tie flaw corrected prospectively — control-ceiling kinds excluded and reported as not-detectable, wins required on two-thirds of detectable kinds — then fund the medium-tier pilot conditionally. Training-free; both prior failures remain recorded and their seeds sealed.

**Status:** finished · 2026-07-15 · not promoted under the corrected bar (2 of 3 required kind wins); the three-replication mechanism map is now crisp: hygiene/explore/termination install, tracefix/protocol do not; seed 78,146 permanently sealed

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

The corrected gate event ran at fresh seed 88,016 with all three inherited composites weight-authenticated. All four kinds were DETECTABLE (no control ceiling this time), so the corrected bar required 3 kind wins. Axis holdout of 40: candidate 22, parent 15, replay_squared 18 — the axis total win replicated for the THIRD time on a third fresh instrument. Per-kind candidate/parent/squared: explore 7/3/6 (win), hygiene 7/5/5 (win), protocol 7/7/5 (tie with parent — third consecutive event), tracefix 1/0/2 (loss). Retention: candidate 65/98/5 (correct/parsed/caps) vs 61/92/12 and 66/91/13 — best termination in every one of the three events. Two kind wins < 3 required: NOT_PROMOTED; seed 78,146 is permanently sealed; the medium pilot never ran.

## Interpretation

Across three preregistered fresh-instrument events the mechanism map is no longer noise: the hygiene and explore lessons install reliably (hygiene beat both controls in all three events; explore in two), the termination benefit is unconditional (caps roughly halved every time), and the axis TOTAL always wins — but the tracefix lesson never installed (4/10, 2/10, 1/10 — trending to chance) and the protocol lesson adds nothing the parent lacks (tied in all three events). The corrected bar did exactly its job: with no ceiling excuse available, it exposed that the corpus installed half its content. The correct successor is a content revision — replace the two dead blocks using this line's own raw failure outputs — not another measurement.

## Terminal Disposition

No later event is authorized here. Seed 78,146 is spent-by-sealing. Three published composites and three gate events' raw outputs (432 graded completions per arm family) are preserved for the successor's failure forensics.

## Knowledgebase Update

- Program evidence updated: the three-replication installable-lesson map recorded.
- Program backlog updated: axis corpus v2 (tracefix/protocol blocks redesigned from failure forensics) queued with calibration notes.
- Claim ledger updated: no.

## Artifacts

- `data/local_tasks_seed88016.jsonl`, `data/local_input_seed88016.jsonl`, `data/local_design_receipt.json`: frozen gate.
- `reports/preregistration.md`, `reports/design_review.md`: contract and authorization.
- `reports/artifact_manifest.yaml`: inherited composite pins.
