# Universal-Line Medium-Tier Measurement Experiment Log

## 2026-07-15 — Model-free design freeze

- Opened as the tier forensics' funded successor: the goal gate's venue
  moves to medium, where the universal line has never been measured.
- Frozen: four published composites (base / designed_fresh / replay_repeat
  / hygiene_explore, tree-hash bound), tier medium, think budget 1,024,
  sealed fresh seed 78,150, one-seed ledger, four preregistered readings
  (aggregate ordering, recorded goal gate, base sanity envelope, blocking
  families), no promotion logic anywhere.
- No model, GPU, or benchmark event has run; nothing trains in this cell.

## 2026-07-15 — Measurement pipeline built (still model-free)

- Implemented the four-script pipeline: `gen_design_receipt.py` (frozen
  pins + `--check` byte-identity + refuse-overwrite), `run_benchmark.py`
  (event-time tree recompute per arm, one-seed ledger that refuses ANY
  prior entry, trusted-gateway invocation in the frozen order, safe
  failure receipts), `check_benchmark.py` (the four preregistered
  readings from the gateway receipts only), and the `run.py` harness
  (`--smoke` and the single `--stage benchmark` behind
  `PASS_BENCHMARK_EVENT`).
- Correction while pinning: the scaffold docs labeled `b654e033…16db` as
  base's TREE hash; it is base's reserialized WEIGHTS hash (per the
  goal-gap pilot's frozen external-weights block). Both identities are
  now pinned and enforced: base tree `26d8ee48…b677` (recomputed from
  disk), base weights `b654e033…16db`.
- Design receipt generated after full on-disk verification (all four
  composite trees recomputed and matched) and re-verified byte-identically
  twice; sha256
  `e3dfc87434b9eea3173db3d7f5a2b0c2fb501154d94784b5a601adc51de51422`.
- Seed-freshness audit inside the receipt: no seed-context use of 78150
  anywhere under `experiments/`, `knowledge/`, or `research_programs/`
  outside this experiment's own declarations.
- Removed the scaffold's `src/vllm_runner.py` and its test: no engine
  code outside the trusted gateway exists in this cell (see
  `src/README.md`).
- 37 unit tests green (goal-gate counting, envelope, ordering, ledger
  refusal, receipt authentication, cross-module frozen constants, the
  forensics FAMILIES tuple byte-for-byte); smoke green; all three entry
  points verified to fail closed pre-commit.

## 2026-07-15 — Adversarial review fixes (two MAJOR, three minor)

- MAJOR 1 closed: `run_benchmark.py` itself now enforces the
  `PASS_BENCHMARK_EVENT` verdict, adds the review and preregistration to
  its committed-at-HEAD list, and re-runs `gen_design_receipt.py --check`
  (code pins, seed audit, quick pins) at the seed-consuming boundary — a
  direct invocation can no longer consume seed 78150 with unreviewed or
  drifted code. Verified live: direct invocation refuses at the review
  gate; a one-byte drift of `check_benchmark.py` fails the `--check`.
- MAJOR 2 closed: write-ahead one-seed ledger. An `opened` record is
  appended before the first gateway call and a `closed` record after the
  summary; any closed/legacy record refuses forever, and a crashed
  event's opened record forces recovery through explicit `--resume`
  (matching record required) — deleting the event directory can no
  longer silently re-consume the seed.
- Minors: `run.py` forwards `--resume` only on explicit operator request
  (never auto); both receipt loaders reject non-finite or out-of-[0,1]
  scores (a NaN can no longer silently drop a family from the strict-win
  partition); smoke now FAILS a published event whose terminal
  `measurement_readout.json` is missing.
- Design receipt regenerated after the code-pin change (full deep
  verification re-run) and re-verified byte-identically twice; new sha256
  `26034d8383a146cc3ddb3d8c67e564a53ee2262c6f99ac1269ce1f8482536cad`.
  Seed-freshness audit still clean.
- Tests now 46 (opened/closed ledger semantics, resume matching,
  finiteness guards in both loaders, resume-forwarding and
  seed-boundary-gate contracts); smoke green.

## 2026-07-15 — Adversarial review: seed-boundary hardened pre-freeze

- Three-lens review with adversarial verification confirmed two MAJOR
  findings, both fixed and drill-verified before commit: the
  seed-consuming runner now enforces the review verdict and the design
  receipt's code-pin re-check inside its own pre-event block (a one-byte
  drift of the readings evaluator demonstrably trips it), and the one-seed
  ledger writes an `opened` record before the first gateway call so a
  crashed event can never be silently re-consumed by deleting scratch
  artifacts.
- Minors fixed with it: operator-explicit `--resume`, finite-[0,1] score
  validation in both loaders, smoke requiring the terminal readout of any
  published event.
- Receipt regenerated (code pins changed) with full deep tree
  verification; 46 tests green; smoke green. Verdict:
  `PASS_BENCHMARK_EVENT`.
