# Benchmark Event Adversarial Review

An eval-only measurement intake: four published composites, one sealed
medium seed, no training, no promotion. Three independent review lenses
(contract consistency, seed-seal/fail-closed integrity, readings
correctness) ran with adversarial verification of every non-minor finding.

- TWO MAJOR findings confirmed, both fixed and drill-verified before any
  commit or model event:
  1. The seed-consuming runner did not itself enforce the review gate or
     re-verify the receipt's code pins — a direct invocation could consume
     the unrepeatable seed without authorization, and a committed drift of
     the readings evaluator would have computed the preregistered readings
     with unreviewed code. `run_benchmark.py` now requires the literal
     `PASS_BENCHMARK_EVENT` verdict, pins this file and the preregistration
     byte-identical at HEAD, and runs the design receipt's `--check`
     (which binds every script's sha256, the seed-freshness audit, and the
     quick-reference pins) inside its own pre-event fail-closed block. The
     drift drill confirms a single appended byte in `check_benchmark.py`
     trips the boundary.
  2. The one-seed ledger was written only on full success, so a mid-event
     crash left no consumption marker and deleting the scratch directory
     would have silently re-consumed the seed. The ledger now writes an
     `opened` record before the first gateway invocation; any closed or
     malformed record refuses forever; resuming a crashed event requires
     the explicit `--resume` flag plus an exact (name, tier, budget, seed)
     match against the lone opened record, and completed per-arm receipts
     are never re-run.
- Three minor findings also fixed: `--resume` is operator-explicit (no
  silent auto-forwarding past the audit interlock); every aggregate and
  per-family score is validated as a finite non-bool float in [0, 1] in
  both loaders (NaN cannot silently drop a family from the strict-win
  partition); smoke fails a published event whose terminal readout is
  missing.
- The readings lens returned no confirmed defects: strict-inequality
  goal-gate counting over the forensics' byte-identical FAMILIES tuple,
  inclusive historical envelope from the sha-pinned forensics analysis,
  frozen full-precision quick references, byte-identical verify mode.
- 46 unit tests green; smoke green; receipt regenerated post-fix
  (`26034d8383a146cc3ddb3d8c67e564a53ee2262c6f99ac1269ce1f8482536cad`),
  `--check` byte-identical twice; seed 78,150 grep-fresh; all four
  composite trees deep-verified against their pins.

**Verdict:** `PASS_BENCHMARK_EVENT`.
