# Qwen35 4b State Track Confirmation

**Status:** in-progress · since 2026-07-17 · design-frozen eval-only six-seed confirmation of lifecycle 30's single-seed INSTALLED_TRANSFER; no seed consumed, benchmark event pending

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can a designed synthetic curriculum install a
  transferable skill that adds *durable* aggregate on a saturated parent,
  proven by transfer to held-out benchmark surfaces?
- Prior anchors: lifecycle 30 `qwen35_4b_state_track_install`
  (INSTALLED_TRANSFER at seed 78169, +0.0256 paired lift, single seed);
  lifecycle 28 `qwen35_4b_count_walk_menders_confirmation` (the eval-only
  multi-seed discipline that correctly retired a single-seed headline as
  seed noise); lifecycle 29 `qwen35_4b_count_walk_replay_compound`
  (replay compounding BOUNDED at stage 8).

## Question

Does the single-seed +0.0256 paired lift of `state_track` over its
`count_walk` parent (seed 78169) replicate across fresh sealed seeds, or
was it within the parent's own 0.30-0.36 seed-to-seed aggregate band?

## Hypothesis

The install-universal-features doctrine predicts a designed transferable
skill adds real, seed-stable aggregate. If so, the paired delta
`state_track - count_walk` should be positive on a majority of fresh sealed
seeds with a positive mean. The null is that the 78169 lift was seed noise,
in which case the paired delta is symmetric about zero.

## Setup

- Model: `Qwen/Qwen3.5-4B` (rev `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) only.
- Arms: two pre-existing committed composites - `count_walk` (parent) and
  `state_track` (candidate). No base arm.
- Eval source: the trusted aggregate gateway
  (`scripts/run_benchmark_aggregate.py`), public family scores only.
- Baseline: `count_walk` (the current program reference composite).
- Controls: the pairing itself (same-seed delta cancels the seed-variance
  that produced the parent's 0.30-0.36 swing); per-family tables and the
  candidate-vs-parent family gate are descriptive.
- Primary metric: the frozen three-state paired verdict (CONFIRMED /
  NOT_CONFIRMED / AMBIGUOUS) over six new seeds.
- Oracle-only metrics: none.
- Hidden-label boundary: the benchmark suite contents are never read as
  data; only the sha-pinned gateway runs.

## Run

Smoke (verifies pins, provenance copies, power arithmetic, the in-cell
stage 1-9 lineage package, any published ledger, compiles, runs tests):

```bash
.venv/bin/python -B scripts/run.py --smoke
```

Full (the only stage; requires clean pushed green main + the committed
preregistration and PASS_BENCHMARK_EVENT design review):

```bash
.venv/bin/python -B scripts/run.py --stage benchmark
```

The event is seed-major over 78170-78175 with a k-seed write-ahead ledger.
A mid-event crash preserves an `opened` record; recover by auditing the
preserved receipts and re-running with `--resume` (the summary regenerates
byte-identically). Never edit receipts by hand: delete the torn artifact
and let `--resume` rebuild it.

## Results

The benchmark event has not run (design freeze; no seed consumed). On
completion the terminal readout is
`runs/benchmark/confirmation_readout.json`. See `reports/report.md` for the
frozen design and consequences and `reports/preregistration.md` for the
frozen rule and power.

## Interpretation

CONFIRMED promotes `state_track` to the program reference composite and
validates the install-universal-features doctrine as a durable aggregate
mover. NOT_CONFIRMED (the decisive, high-value outcome under this liberal
directional rule) retires the single-seed reading as seed noise and keeps
`count_walk` as the reference. AMBIGUOUS demands a mechanism-differentiated
or larger-N design, never a re-roll of these seeds.

## Knowledgebase Update

- Program evidence updated: on event completion, the
  `agentic_breadth_installation` scorecard records whether the
  divergent-skill install is a durable aggregate mover.
- Program backlog updated: the next divergent-skill dose is gated on this
  verdict (compound on state_track if CONFIRMED, else re-anchor on
  count_walk).
- Claim ledger updated: design-only until the event runs; the verdict
  then supports or retires the durable-reference claim.

## Artifacts

- `src/` - the shared vLLM runner (unused at eval; the gateway is the instrument)
- `scripts/` - `run.py`, `run_benchmark.py`, `check_benchmark.py`,
  `power_analysis.py`, `rebuild_lineage.py`, and the copied lineage
  trainers/mergers/wrappers
- `configs/`
- `data/` - the in-cell stage 1-9 lineage package and the three provenance copies
- `runs/` - created at event time (ledger, per-seed summaries, readout)
- `reports/` - preregistration, benchmark design review, report, artifact manifest
- `reports/artifact_manifest.yaml`
