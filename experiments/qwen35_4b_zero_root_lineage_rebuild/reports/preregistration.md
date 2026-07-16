# Preregistration: Zero-Root Lineage Rebuild

Frozen before any model event. A provenance measurement: no promotion,
one sealed seed, both outcomes durable.

## Frozen identities

- Experiment: `qwen35_4b_zero_root_lineage_rebuild`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- The treatment IS an omission: the six documented stages of the
  hygiene_explore lineage are replayed with their exact recorded recipes
  (datasets byte-identical from the copied package; fixed seeds
  42/43/44/47/51/55, deliberately reused — that is what "same recipe"
  means; the three recorded trainer variants; stage 3's targeted
  overrides) but stage 1 trains a FRESH zero-initialized rank-32/alpha-64
  adapter instead of warm-starting from the undocumented C53-era `blend`
  root (weights ad2ef4fa…). The root is not vendored and its appearance
  anywhere in this cell fails closed — the `root_omission` receipt block
  documents the removal as the design.
- Stage receipts record the original chain's per-stage hashes as
  CONTRAST fields only; they can never match (different root) and are
  never verification.
- Merge: the stage-6 zero-root adapter onto the raw base via the copied
  merger (cb9af8b4…) → `zero_root_hygiene_explore`, receipt-pinned.
- Benchmark: ONE medium event at sealed fresh seed `78159`, tb 1,024,
  three arms in frozen order — base (26d8ee48…/b654e033…), the original
  hygiene_explore (9eb653d7…/e2112344…), and the zero-root composite
  (fail-closed TODO-pins filled post-merge from the committed
  merge receipt) — behind the hardened single-seed runner (verdict +
  receipt code-pin checks at the boundary, write-ahead ledger with
  per-arm receipt shas in the closed record, provenance-anchored
  readout, implementation signature pinned to the discovery block).

## Preregistered readings (no promotion)

1. Per-family table and aggregates for all three arms.
2. The goal gate versus base for BOTH composites (ten public families,
   strict wins/ties/losses, forensics-identical counting).
3. The prefix-contribution contrast: zero-root minus original, per
   family and aggregate — "the gym-era root's contribution at medium,
   one seed, same-seed paired."
4. Budget integrity; menders/rites/warren margins recorded (no
   statechain stage exists in this chain — noted).

## Frozen consequence partition (ordered, total)

- `ZERO_ROOT_COMPARABLE` iff the zero-root aggregate strictly beats base
  AND its goal-gate strict wins ≥ the original's strict wins on this
  seed minus 1: the documented stages alone carry the demonstrated
  position — the headline model is contamination-clean end-to-end.
- `ZERO_ROOT_DEGRADED` otherwise: the undocumented prefix is
  load-bearing at medium; its contribution is the recorded contrast and
  scopes every prior reading involving this lineage.

## Mandatory checkpoint order

1. Model-free construction + review — committed, pushed, green.
2. `rebuild` (requires `PASS_REBUILD` in reports/compute_review.md;
   six sequential stage replays + the merge, ~2.5–3 h GPU; per-stage
   receipts committed after).
3. `benchmark` (requires committed rebuild receipts +
   `PASS_BENCHMARK_EVENT` in reports/benchmark_design_review.md + clean
   pushed green main).

## Interpretation limits

One seed bounds the contrast; GPU training is deterministic-given-stack,
not bitwise-portable — the zero-root composite is THE artifact measured,
its receipts the record. The original's two recorded sweeps came from
four-seed evidence; this cell's single seed reads the contrast, not the
sweep rate. Benchmark firewall unchanged.
