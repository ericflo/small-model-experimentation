# Qwen35 4b Count Walk Menders Confirmation Experiment Log

## 2026-07-17 — pre-event review amendments (no seed consumed)

The adversarial review of the frozen design (frozen at `bd253e48`) returned
3 MAJOR + 4 minor findings; all were applied inside the legitimate pre-event
amendment window — the ledger does not exist, no gateway call has ever run,
and every change precedes the benchmark design review verdict. Full
provenance in `reports/preregistration.md`, "Review amendments" section.

- **A1+A2 — one coherent full-episode semantics.** Episode conversion moved
  from `round(10*score)` to FLOOR `int(10*score + 1e-9)` (a k/60-lattice
  partial-credit draw contributes ZERO episodes unless it crosses a full 0.1
  step; a new unit test sweeps all 61 lattice points via the float k/60
  representation and matches `int(k/6)` exactly). Hits redefined: an event is
  a hit only if the arm's FULL-EPISODE count is > 0 — partial-only events are
  recorded descriptively (`raw_positive` per event and per arm) but are
  neither hits nor episodes. The rule now coincides exactly with the priced
  model. Power arithmetic restated on the full-episode null: alpha 0.0450 at
  the headline p = 0.10 AND 0.0475 at the exact p = 3/29 (exact fraction
  11885589964581732052992/250246473680347348787521 = 0.04749553426180864,
  printed and `--check`-enforced); p = 5/29 (0.0947) retained strictly as a
  counterfactual ceiling; hits>=2 and REPLICATED power numbers verified
  unchanged (0.5248/0.6875/0.8735 and 0.4717/0.6289/0.8230).
- **B1 — standalone doctrine for an eval-only cell.** Copied byte-identically
  from lifecycle 27: the entire `data/lineage/` package (six stage datasets,
  manifest, seven provenance receipts), the stage-7 production inputs
  (`count_walk.jsonl`, `replay_ctl7.jsonl`, `sft_count_walk.jsonl`,
  `sft_blend.jsonl`, `stream_token_receipt.json`), and the production scripts
  (`lineage_trainers/` ×3, `train_think.py`, `merge_adapter.py`,
  `rebuild_clean_chain.py`, `train_trial.py`, `merge_trained_arm.py`).
  Extended the copied `lineage_manifest.json` with a
  `stage7_confirmation_arms` block: both arm streams (shas recomputed from
  the copies — replay_ctl7 `94e8259e…`, count_walk `71291542…`), training
  seed 85, trainer/merger shas, and the final composite tree/weights pins
  this cell authenticates. Added `scripts/rebuild_lineage.py` (stages 1-6
  rebuild the zero-root parent; stage 7 trains the two arms with the
  train_trial.py recipe at seed 85 and merges via the merge_trained_arm.py
  merge); its `--verify-inputs` checks every copied file against the manifest
  shas and runs green in smoke and a new unit test. Reproduction path is now
  IN-CELL; receipt copies remain verification aids.
- **Minor 1.** Design-time audit corrected to 9 recorded medium/tb1024 sealed
  events (78,150/78,154/78,155/78,156/78,157/78,159/78,160/78,162/78,163);
  the 29 arm-event count was already correct.
- **Minor 2.** The implementation-signature equality check now ALSO runs
  pre-consumption, before each seed's FIRST gateway arm (live signature via
  the trusted gateway's own hash-only inventory functions) — a drifted suite
  refuses before any GPU run or opened record; the post-arm check is kept.
- **Minor 3.** NOT_REPLICATED consequence text now carries "(at a true
  per-event hit rate of 0.3 this outcome retains probability ≈ 0.24 — the
  closure is a preregistered funding decision, not a nonexistence proof)"
  everywhere it is stated; 0.2401 is `--check`-enforced.
- **Minor 4.** Torn-ledger / partial-receipt manual recovery documented in
  the README ops section (delete the torn artifact; `--resume` regenerates
  byte-identically; never edit receipts by hand).
- Tests updated for the new semantics (lattice sweep, partial-only
  NOT_REPLICATED branch, raw-positive records, lineage package); full suite
  green; smoke green; `power_analysis.py --check` green;
  `rebuild_lineage.py --verify-inputs` green. No GPU stage run; no seed
  consumed.

## 2026-07-17 — design freeze (lifecycle 28, eval-only)

- Scaffolded as the mandatory confirmation cell for lifecycle 27's
  MECHANISM_ANSWER (count_walk menders 0.1 vs base / zero_root_parent /
  replay_ctl7 all 0.0 at sealed seed 78,163).
- Seed-freshness audit: 78,164 / 78,165 / 78,166 / 78,167 verified grep-fresh
  in seed contexts across the repo (every raw numeric hit is a float/sha
  substring in unrelated data files); benchmark seeds previously spent through
  78,163; no substitution required.
- Frozen the integer-exact two-directional replication rule (REPLICATED /
  NOT_REPLICATED / AMBIGUOUS, no fourth state) with all three claims worded in
  the preregistration, and the exact power arithmetic (false-REPLICATED 0.0450
  under the p=0.10 noise model, sensitivity 0.0947; power of hits_c >= 2:
  0.5248 / 0.6875 / 0.8735 at q = 0.4 / 0.5 / 0.65; full REPLICATED power
  0.4717 / 0.6289 / 0.8230), recomputed fail-closed by
  `scripts/power_analysis.py --check`.
- Cloned and adapted the hardened runner machinery: fail-closed tree+weights
  authentication of the four pre-existing composites (constants baked at design
  time, no TODO pins), lifecycle-27 merge receipt + lifecycle-22 zero-root
  provenance authentication, gateway sha pin, k-seed write-ahead opened/closed
  ledger with byte-equal crash reconciliation, implementation-signature
  equality against the pinned prior event, ledger-anchored terminal readout.
- Copied the four committed provenance documents byte-identically into
  `data/provenance/` as verification aids; composite reproduction remains
  lifecycle 27's / lifecycle 22's own standalone rebuild path (this cell
  produces no model); the measurement gateway stays shared per
  docs/quality_gates.md.
- Unit tests: replication-rule truth table (including the E_c tie branch),
  ledger open/close/reconcile/double-consume refusals, arm-authentication
  failure paths, frozen constants, readout schema, finiteness guards, power
  arithmetic. Smoke green; no GPU stage run; no seed consumed.
- Next checkpoint: adversarial benchmark design review
  (`reports/benchmark_design_review.md` with the literal
  PASS_BENCHMARK_EVENT verdict) before `--stage benchmark` can consume any
  seed.

## 2026-07-17 — Four-seed event complete: AMBIGUOUS; cell closed

- Events 78164-78167 (16 runs, all within budget, paired comparison
  valid, implementation signature identical across all sixteen receipts
  and the prior event). Menders per event: 78164 candidate 0.1 AND
  replay control 0.1 (one full episode each); 78165/78166 all arms 0.0;
  78167 candidate 0.0167 partial (recorded, never counted — the
  review's floor-semantics fix operating as designed).
- Frozen rule: hits_c = 1 (< 2) and episode totals candidate 1 vs
  replay 1 (tie = no dominance) → AMBIGUOUS. Frozen claim applies: no
  claim; further spending on this contrast requires a
  mechanism-differentiated NEW design, not more seeds of the same.
- Honest reading: the replay control's full episode at 78164 is the
  noise process the preregistration priced (background arm-event rate
  ~0.10); the 78163 MECHANISM_ANSWER is now best read as that
  coincidence. Menders remains without a confirmed mover. Descriptive:
  count_walk topped the aggregate at 78164 (0.398) and 78167 (0.392),
  lost narrowly at 78165 (parent 0.3492 vs 0.3373) and 78166 (replay
  0.3283 vs 0.3269) — single-seed readings, never gating.
