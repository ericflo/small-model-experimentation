# Preregistration: Goal-Gate Confirmation

Frozen before any model event. Eval-only replication: no training, no
promotion; three sealed seeds are consumed once each and the cell closes
on the ordered verdict.

## Frozen identities

- Experiment: `qwen35_4b_goal_gate_confirmation`.
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Arms — two authenticated composites, tree-recomputed and bound at event
  time, frozen order per seed (base first): `base` (weights b654e033…,
  tree 26d8ee48…) and `hygiene_explore` (tree 9eb653d7…, weights
  e2112344…, committed merge receipt 22a22a68…).
- Seeds: `78155 / 78156 / 78157` — independent, fresh (word-boundary
  audits in the design receipt), each consumed exactly once under a
  per-seed write-ahead opened/closed ledger. Tier `medium`, think budget
  1,024, trusted gateway only, identical benchmark source inventory
  across all six runs AND matching the discovery event's pinned
  implementation signature (fail closed).

## Preregistered readings and the ordered verdict

1. Per seed: both aggregates, the full per-family table, and the goal
   gate — hygiene_explore's strict wins/ties/losses versus base over the
   ten public families; pass = ten strict wins.
2. `confirmation_verdict`, an ordered total partition:
   - `CONFIRMED` iff hygiene_explore aggregate strictly exceeds base on
     ALL THREE seeds AND the goal gate passes on AT LEAST TWO of three;
   - `AGGREGATE_ONLY` iff the aggregate condition holds on all three but
     the goal-gate majority fails;
   - `NOT_REPLICATED` otherwise.
   The discovery seed 78,154 is reported alongside from its sha-pinned
   committed summary and is NEVER counted in the verdict.
3. Fragility: per seed, the menders and warren margins and the blocking
   families of any non-pass.
4. Budget integrity per arm per seed.

Frozen consequence statements: `CONFIRMED` completes the program goal's
primary demonstration-plus-confirmation chain at the repository's highest
gateway-supported tier (medium), with the claim scoped to that
instrument; `AGGREGATE_ONLY` and `NOT_REPLICATED` price the discovery
honestly as a favorable draw, and the program continues from the 9/10 +
conversion-mechanism position with the menders dose-scale intake as the
queued bet. Either way every receipt is preserved.

## Instrument honesty (frozen)

The trusted gateway exposes quick|medium tiers with a fixed internal
decode, so a same-backend matched-compute sample-more arm cannot be
constructed at this instrument without violating the benchmark firewall;
replication across independent sealed seeds is the confirmation
mechanism. The 78,154 summary sha is pinned so the discovery context is
byte-anchored.

## Standalone reproduction package (owner directive, 2026-07-15)

This cell carries the measured composite's complete model-reproduction
package: `data/lineage/` holds byte-identical copies of all six training
streams in training order (stage01_replay_refresh … stage06_hygiene_explore,
shas pinned in `data/lineage/lineage_manifest.json`), the manifest records
each stage's warm-start source, trainer variant, fixed seed
(42/43/44/47/51/55), and complete hyperparameters, the three trainer
variants and the merger are copied into `scripts/`, and
`scripts/rebuild_lineage.py` replays the chain deterministically with
per-stage hash verification (its no-GPU `--verify-inputs` mode runs inside
smoke). The base for every stage and every merge is the raw pinned HF
revision — the official post-trained Qwen3.5-4B; nothing ever retrains the
base weights.

Provenance boundary, stated plainly: the adapter chain's ROOT is the
frozen C53-era `blend` adapter (weights ad2ef4fa…, config cd764ae8…),
vendored into this cell's own artifact storage
(`large_artifacts/qwen35_4b_goal_gate_confirmation/lineage_root/blend`,
~181 MiB, hash-pinned). No committed creation receipt exists for it: the
six documented stages reproduce from that vendored binary, not from a
zero-initialized adapter. The root predates the universal line and carries
gym-era training (in-repo training gyms; the benchmarks/ firewall was
never crossed). Any confirmation verdict from this cell is therefore a
claim about the measured composite AS BUILT — official base + the vendored
root + six receipted contamination-free doses — and a zero-root rebuild
(fresh adapter, same six stages) is a distinct, queued question about how
much the undocumented prefix contributes.

## Mandatory checkpoint order

1. Model-free construction + adversarial design review — committed,
   pushed, green.
2. `benchmark` (requires `PASS_BENCHMARK_EVENT` in
   `reports/benchmark_design_review.md`, the design receipt committed at
   HEAD, clean pushed green main). No other stage exists.

## Interpretation limits

Three seeds bound the replication read; single-item margins at menders
and warren remain single-item margins per seed and are reported as such.
Benchmark firewall unchanged: gateway aggregates and public family
scores only.
