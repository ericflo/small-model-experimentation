# Qwen35 4b Zero Root Lineage Rebuild Report

## Summary

PENDING — design frozen and receipted (2026-07-16); the rebuild and
benchmark stages have not run. The question: how load-bearing is the
undocumented C53-era gym-line 'blend' root adapter at medium? The six
documented contamination-free stages of the hygiene_explore composite
are replayed from a FRESH zero-initialized adapter (same datasets, same
fixed seeds, same trainers, same hyperparameters, raw pinned HF base
everywhere) and measured ONCE at medium/tb1024 on the fresh sealed seed
78159 against the original composite and base.

## Research Program Fit

`agentic_breadth_installation` — the provenance question elevated to a
measurement. The goal-gate confirmation demonstrated the position
(AGGREGATE_ONLY; two 10/10 sweeps across four sealed seeds) but the
composite's lineage root has no committed creation receipt. If the
documented stages alone carry the position, the headline model is
contamination-clean end-to-end; if not, the recorded contrast IS the
root's contribution.

## Method

- Byte-identical copied lineage package (manifest `1f49cd8b…`, six
  datasets, three trainers, merger); the blend root deliberately NOT
  vendored — its omission is the design and fails closed if present.
- `rebuild_zero_root.py`: stage 1 fresh rank-32/alpha-64 (no
  `--warm-start`, LoRA-B zero-init) at seed 42 with the exact recorded
  stage-1 recipe; stages 2-6 warm-start from the previous zero-root
  stage with their recorded recipes (stage 3 targeted close overrides).
  Per-stage receipts committed to `runs/lineage/`; the original chain's
  adapter hashes recorded as CONTRAST fields only.
- Merge of the stage-6 zero-root adapter onto the raw base via the
  copied merger; the merge receipt pins the full output tree.
- ONE sealed benchmark event: medium, tb1024, seed 78159 (grep-fresh
  audited), three arms in frozen order (base, hygiene_explore_original,
  zero_root_hygiene_explore), single-seed write-ahead ledger whose
  closed record sha-pins the summary and all three gateway receipts,
  TODO-pinned zero-root arm filled post-merge, implementation signature
  anchored to the discovery/confirmation block.
- Frozen consequence (ordered, total): ZERO_ROOT_COMPARABLE iff the
  zero-root aggregate strictly beats base AND its goal-gate strict wins
  >= original's strict wins on this seed − 1; ZERO_ROOT_DEGRADED
  otherwise.

## Results

PENDING — terminal artifact will be `runs/benchmark/zero_root_readout.json`.

## Controls

- Same-seed cross-arm pairing: all three arms run the identical sealed
  seed, tier, and think budget in one event.
- The original composite arm is the exact published tree the goal-gate
  events measured (tree `9eb653d7…`, weights `e2112344…`).
- The base arm is the frozen reserialized base (tree `26d8ee48…`).
- Budget integrity is a reading, never a gate; any over-budget arm
  invalidates the paired comparison scope, scores still recorded.

## Oracle Versus Deployable Evidence

No oracle anywhere: the measurement is the deployable benchmark surface
through the trusted gateway (receipts only; nothing under `benchmarks/`
is ever read).

## Interpretation

PENDING the sealed event.

## Next Experiments

- If ZERO_ROOT_COMPARABLE: the zero-root composite becomes the
  contamination-clean headline lineage; the menders dose-scale intake
  proceeds on a clean base.
- If ZERO_ROOT_DEGRADED: the recorded prefix-contribution contrast
  becomes the program's measurement of the gym-era root; decide whether
  to reconstruct the root's function via documented curricula.

## Artifact Manifest

See `artifact_manifest.yaml` — adapters and the merged composite live
under `large_artifacts/` (receipts committed in `runs/lineage/`); the
complete stage-replay package is carried in-repo.
