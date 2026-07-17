# Qwen35 4b State Track Confirmation Experiment Log

## 2026-07-17 - Design freeze (lifecycle 31, eval-only; no seed consumed)

Frozen the six-seed eval-only confirmation of lifecycle 30's single-seed
INSTALLED_TRANSFER (seed 78169: `state_track` aggregate 0.3260 vs
`count_walk` 0.3004, paired lift +0.0256). No training, merging, corpus,
or promotion in this cell.

Cloned machinery from the two reference cells:

- lifecycle 28 `qwen35_4b_count_walk_menders_confirmation` (the eval-only
  multi-seed confirmation): the k-seed write-ahead ledger with byte-equal
  crash reconciliation, arm authentication by frozen tree/weights pins,
  gateway sha pin, implementation-signature equality across all receipts,
  the sha-pinned prior-event summary (never pooled), the power-analysis
  script, `check_benchmark.py`, and the standalone lineage package + tests.
- lifecycle 30 `qwen35_4b_state_track_install` (the two-arm aggregate
  install): the two-arm aggregate reading, the 1e-12 aggregate tie guard,
  the in-cell-authoritative provenance (siblings as verification aids), the
  per-file normalized provenance pins, and the full stage 1-9 lineage
  package (copied byte-identically; `rebuild_lineage.py` adapted so the
  stage-9 `extended_by` identity stays the source cell
  `qwen35_4b_state_track_install`).

Design decisions frozen:

- Two arms (count_walk, state_track); no base arm - the event is the
  parent-vs-candidate paired comparison.
- Six fresh sealed seeds 78170-78175, verified grep-fresh in seed contexts
  repo-wide at design time (zero seed-context hits; all raw numeric matches
  are float/sha256 substrings in per-row data files). The next six free
  integers after the prior 78169 event.
- Frozen PAIRED rule over the six new events: `d_i = state_track_aggregate
  - count_walk_aggregate`; `wins` = events with `d_i > 1e-12`;
  `mean_d` = mean. CONFIRMED iff `mean_d > 0` and `wins >= 4`;
  NOT_CONFIRMED iff `mean_d <= 0` (dominates the `wins >= 4` edge);
  AMBIGUOUS otherwise. No fourth state. 78169 is prior evidence, never
  pooled.

Preregistered power (exact/quadrature, `power_analysis.py --check`
enforced): under the pure null the paired majority rule is a LIBERAL
directional check - false-CONFIRMED joint = 0.311 (scale-free; bounded by
the exact P(wins>=4)=0.34375 and the independence product 0.17188). Under
the observed +0.0256 lift the test has 0.9028-0.9839 CONFIRMED power across
sigma_d in {0.02, 0.025, 0.03} (rho in {0.78, 0.65, 0.50}, via
`sigma_d = sigma_arm*sqrt(2(1-rho))`, sigma_arm=0.03). The stated,
high-value outcome is the decisive NOT_CONFIRMED (mean <= 0).

Adversarial review returned `PASS_BENCHMARK_EVENT` (see
`reports/benchmark_design_review.md`); mutation probes (tampered
tree/weights pins, tampered provenance sha, ledger re-open, spent budget,
off-list seed 78169, over-budget arm) refuse fail-closed; crash walks
reconcile byte-equal; the stage 1-9 lineage package verifies
(`rebuild_lineage.py --verify-inputs`).

Verification green at design freeze: `py_compile` all scripts; 109 unit
tests; `run.py --smoke`; `power_analysis.py --check`; `rebuild_lineage.py
--verify-inputs`.

Deviation from the intake brief: the count_walk merge-receipt file sha in
the intake was `4170b082...`, which is actually the `replay_compound`
receipt sha from a sibling cell; the correct count_walk merge-receipt file
sha (used by lifecycle 30 and matching the file on disk) is
`840edca0...`. The correct pin is used here.

## 2026-07-17 — Six-seed confirmation complete: CONFIRMED (directional; statistically soft)

- Events 78170-78175 (12 sealed runs, budget-clean, implementation
  signature identical across all receipts and the prior 78169 event).
  Paired deltas d_i = state_track_agg - count_walk_agg (same seed):
  [-0.0123, +0.0373, -0.0385, +0.0050, +0.0439, +0.0887]. mean_d =
  +0.0207, wins = 4/6 -> frozen verdict CONFIRMED (mean_d > 0 AND wins
  >= 4).
- HONEST EFFECT SIZE (descriptive, promised at freeze): SD 0.0453, SE
  0.0185, paired t = 1.12 on 5 df — NOT strictly significant (one-sided
  p ~ 0.16). The observed variance exceeds the preregistered sigma_d
  (0.02-0.03), exactly why the frozen rule was declared a LIBERAL
  directional check (alpha ~ 0.31), not a significance test. The mean
  lift (+0.0207) matches the single-seed 78169 observation (+0.0256);
  combined 7 seeds give mean +0.0214, 5/7 positive.
- READING: state_track is a real-but-small and noisy aggregate
  improvement over count_walk — durable enough to treat state_track as
  the current-best composite, but the edge is ~+0.02 with high
  seed variance, NOT a large or crisp gain. The install-universal-
  features doctrine is supported (a divergent skill added transferable
  aggregate and it replicated directionally); it is not proven at
  strict significance at n=6.
- Consequence: state_track adopted as the program reference composite
  for the next phase (the coding-harness measurement); the modest,
  noisy magnitude is carried forward honestly.
