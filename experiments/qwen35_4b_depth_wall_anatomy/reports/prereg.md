# Pre-registration draft: What determines the compositional frontier — serial depth or information structure?

Drafted BEFORE the factorial battery runs (2026-07-02, 00:05). Motivating anomalies from EXISTING data only:

1. Naive independent-step law violated: depth-2 greedy 0.133 observed vs 0.444 predicted from r1=0.667
   (3.3x too steep); pass@6 0.333 vs 0.871. Composition failure is superadditive.
2. Post-hoc slice of M2 depth-2 by #information-destroying ops (filters/dedup/unique/take/drop/chunk/mod/
   abs/running_max — ops whose intermediate states are not recoverable from visible I/O):
   0 destructive: 7/9 = 0.78 (nearly depth-1 level!) | 1: 2/6 = 0.33 | 2: 0/5 = 0.0
   And depth-3 k=0: 2/3 in the tiny baseline sample. Cells are small — hence this battery.

## Hypotheses

- H-depth (serial-depth story, implicit in C11/C12 "depth-3 frontier"): solve rate is governed by
  composition LENGTH — falls ~geometrically in d at fixed destruction count k; k matters little.
- H-info (invertibility story): solve rate is governed by INFORMATION DESTRUCTION — falls sharply in k at
  fixed d; transparent-only (k=0) compositions stay solvable at depth 4-6; the "depth-3 wall" is an
  artifact of random deep compositions almost surely containing destructive ops.

## Pre-registered predictions (falsifiable, in advance)

P1 (headline, bold): k=0 compositions at depth 4-5 solve at >= 0.4 pass@6 — where the M1 baseline
    (random compositions) measured 0.067/0.0. If true, "the depth wall" was never about depth.
P2: at fixed d=3, solve rate is monotonically decreasing in k, with a drop of >= 0.3 absolute from k=0 to k=2.
P3 (two-parameter reliability model): p(solve) ~ c * rT^(nT) * rD^(nD) (nT/nD = # transparent/destructive
    ops) fits the full grid substantially better than depth-only c*r^d (AIC / held-out cell prediction).
P4 (planner slice): letter-logit FIRST-op ranking accuracy degrades with the number of destructive ops
    DOWNSTREAM of position 1 (they scramble the observable output), not with depth per se.
    Operational: top-3 hit rate at (d=3,k=0) > (d=2,k=1) despite greater depth.
P5 (selection/ambiguity slice): visible-pass-but-hidden-fail (false-pass) rate increases with k
    (destructive ops create spurious consistent hypotheses). k=0 false-pass ~ 0.
P6 (thinking budget, logged free): thinking length used correlates with d, but conditional on k;
    no strong prediction — exploratory.

## Design

- Controlled generator: compositions with exact (d, k, destructive-positions). Transparent set T (10):
  reverse, sort_asc, sort_desc, square, negate, add_k, mul_k, rotate_k, running_sum, adjacent_diff*
  Destructive set D (13): unique_stable, dedup_adjacent, abs_all, filter_even, filter_odd, keep_positive,
  filter_gt_k, filter_lt_k, take_k, drop_k, chunk_sum_k, mod_k, running_max.
  (*adjacent_diff shortens by 1 but is linear-invertible up to a constant -> classify T; abs/mod/running_max
  destroy sign/magnitude/order info -> D. Classification fixed here, before data.)
- Grid: d in {1,2,3,4,5} x k in {0..min(d,3)}, positions randomized (log position for k=1 analysis);
  n=25 tasks/cell (~16 cells, ~400 tasks). 10 visible + 8 hidden examples, same as C12.
- Measures per task: monolithic thinking greedy@1 + pass@6 (hidden-graded), n_think, per-candidate
  visible-pass vs hidden-pass (for P5), first-op letter-logit ranking (for P4; one forward/task, reuse
  decompose_lib). Budget 512.
- Analysis: cellwise solve rates + fitted models (P3) + the four sliced predictions. All predictions above
  committed before the run.

## Interpretation commitments

- If P1&P2 hold: the frontier is informational, not serial -> reframes C11/C12; banking/search should be
  re-targeted at invertibility (e.g., search only needed where inversion is impossible); C12's planner-wall
  becomes "you cannot rank what you cannot see through."
- If P1 fails (k=0 deep also collapses): serial depth is real for INDUCTION even when invertible ->
  strengthens the serial-compute story; points to I4 (thinking as serial workspace).
- Mixed: fit P3's two-parameter model and report which factor dominates; either way the law replaces
  "depth wall" with a quantitative, predictive account.

---

## Addendum (pre-Phase-1, post-Phase-0 — logged before the grid ran)

Phase 0 (behavioral min-depth audit of ALL existing M1/M2/C12 tasks) is COMPLETE and both predictions held:
- P0a CONFIRMED: 40% of nominal-d3 tasks collapse to min-depth <=2 (M1 6/15, C12 16/40; predicted >=20%).
- P0b CONFIRMED (strong form): monolithic true-depth-3 solves = 0 across the ENTIRE corpus; all recorded
  d3 solves were on collapsed tasks. C12's decompose search: 16/16 collapsed vs 4/24 (17%) true-d3.
- Destruction signal SURVIVES the collapse control (M2 true-depth-2: k=0 -> 6/8; k>=1 -> 0/8).

Design consequences (applied before Phase 1): the factorial generator REJECTS collapsed compositions
(exact BFS to depth min(d-1,3); d=5 cells may retain d4-equivalents — reported as a caveat).

## Phase 2 (discriminator) predictions — logged before Phase 2 runs

Conditions on the SAME verified tasks (subset, d in {2,3,4}, k in {0,2}): (a) BARE I/O (Phase-1 protocol);
(b) PLAN-GIVEN — prompt additionally states the exact op pipeline (names+params); model only translates to
code; (c) INTERMEDIATES-SHOWN — visible examples show full state chains input->s1->...->output, no op names.
- P7: plan-given ERASES both the depth and destruction effects (translation is ~depth/k-invariant):
  plan-given solve >= 0.7 in every cell, k=0 vs k=2 gap < 0.1. If instead plan-given stays low at d4,
  a genuine serial-execution deficit exists (falsifies the pure-identification account).
- P8: intermediates-shown rescues the k effect specifically (observability restored): k=2 solve rises to
  within 0.15 of k=0 at matched depth. If it does NOT rescue, destruction hurts via something other than
  observability (e.g., harder per-step inference even when states are visible).
- P9 (planner slice, from Phase-1 data): first-op letter-logit rank degrades with k (destroyed
  observability), not with d at fixed k. Operational: median rank at (d3,k0) better than (d2,k2).
