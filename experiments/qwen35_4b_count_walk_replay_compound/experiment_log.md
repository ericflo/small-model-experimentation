# Qwen35 4b Count Walk Replay Compound Experiment Log

## 2026-07-17 — design freeze (lifecycle 29; model-free, no seed consumed)

Stage 8 of the documented zero-root chain: REPLAY COMPOUNDING onto the
count_walk composite. Everything below is model-free construction; no GPU
stage has run, no review has been sought yet, and the sealed seed 78,168 is
unconsumed.

- **Treatment frozen.** One fresh rank-32/alpha-64 adapter
  (`replay_compound`) on the FULL 2,240-row replay pool
  (`data/sft_blend.jsonl`, sha `25a9595f…`, byte-identical lifecycle-27
  copy; max forward 3,193 tokens, zero skips enforced) from the count_walk
  composite parent via `--model-path`, with the chain's established
  replay-refresh recipe (epochs 1, lr 1e-5, bs 1, ga 8, maxlen 4096,
  w_think 0.2, w_close 0.2 — identical to stages 1/4 and the stage-7 arms)
  at the fixed FRESH training seed 86. Seed audit: 42/43/44/47/51/55/85 are
  the chain's taken training seeds; 86 verified grep-fresh in training-seed
  contexts repo-wide; no substitution required.
- **Parent authentication frozen.** Fail-closed pre-training and pre-merge:
  committed lifecycle-27 merge receipt (sha `840edca0…`), byte-identical
  in-cell provenance copy (`data/provenance/count_walk_merge.json`), inner
  receipt / tokenizer / size pins, then the full 9 GB weights hash
  (`ddd7bc4b…`). Cloned from the reference cell's train_trial.py /
  merge_trained_arm.py pattern, adapted to this parent.
- **Local gate frozen and generated.** Retention-only (no axis kind exists
  for a pool treatment), TWO arms (count_walk parent first, then the
  candidate), three pooled_k3 screens at fresh seeds 88060/88061/88062
  (104 rows each, 8 per each of 13 skills, canonical gen_curriculum.py).
  Seed audit: everything <= 88059 known-taken (reference cell holds
  88056-88059); 88060/88061/88062 verified grep-fresh; no substitution.
  TWO-SIDED bands on integer screen sums: correct ±15, parsed ±9, cap
  contacts ±9 (means ±5/±3/±3). Design receipt + task files + runner
  inputs generated model-free and pinned (`gen_local_gate.py --check`
  green); freshness audit: zero canonical-user-message overlap with every
  in-cell corpus (including the training pool), the reference cell's four
  frozen gate files, and regenerated prior local seeds 88000-88059.
- **Sealed event frozen.** Medium / tb1024 / fresh seed 78168 (benchmark
  seeds spent through 78,167; grep-fresh; no substitution), three arms in
  frozen order base → count_walk → replay_compound through the trusted
  gateway (`53cf6533…`). The candidate's tree/weights/committed-receipt
  pins are three fail-closed TODO slots; `run_benchmark.py` is frozen by
  `check_design.py`'s three-slot NORMALIZED hash (`d619d5df…`) — every
  byte outside the slots, every guard call site included, is byte-frozen
  pre- and post-fill. One-seed write-ahead ledger; byte-equal crash
  reconciliation (the summary payload is a pure function of the receipts;
  a preserved summary must reconcile byte-identically before the ledger
  closes).
- **Consequence frozen (two-directional, no third state).** COMPOUNDED iff
  candidate aggregate strictly > parent AND no family strictly below the
  parent by more than 0.1 (`candidate_family >= parent_family - 0.1 -
  1e-9`; exactly 0.1 below passes, 0.10000001 fails; unit-tested over the
  full k/10 and k/60 lattices) AND candidate aggregate strictly > base.
  BOUNDED otherwise. Frozen claims in the preregistration. Goal gate vs
  base (10/10 strict wins) recorded descriptively for both treated arms.
- **Honest priors frozen.** The stage-7 replay control is the exact
  move-class precedent (beat its parent 4/5 sealed draws, mean +0.018),
  but the family-slack clause historically binds (~4/5 draws show a dip
  beyond one episode): P(aggregate > parent) ≈ 0.5-0.6, P(COMPOUNDED) ≈
  0.25-0.40; BOUNDED is the believed-likelier verdict and is a finding
  about the law's boundary, not a failure. Menders is NOT re-litigated
  (lifecycle 28 closed that contrast); it appears only descriptively.
- **Standalone package extended.** Copied byte-identically: the full
  `data/lineage/` package (six stage datasets + seven provenance
  receipts), the stage-7 production inputs, `lineage_trainers/` ×3,
  `train_think.py`, `merge_adapter.py`, `rebuild_clean_chain.py`, and
  lifecycle 27's wrappers into `scripts/stage7_wrappers/` (shas
  `a83240a0…` / `b566c486…` unchanged). The manifest was extended with the
  `stage8_replay_compound` block (arm, pool sha, seed 86, trainer/merger
  shas, parent pins, recipe, three null post-merge TODO slots) and the
  stage-7 block's wrapper paths re-pointed; new byte pin `45d1a0d9…`.
  `rebuild_lineage.py` now replays stages 1-8; `--verify-inputs` green.
- **Verification at freeze.** `run.py --smoke` green end-to-end
  (check_design --check, rebuild_lineage --verify-inputs, gen_local_gate
  --check, 127 unit tests). Boundary drills refuse: every staged gate
  without its committed review verdict, the sealed runner with unfilled
  TODO pins, a tampered parent merge receipt in a scratch copy, fake and
  incomplete composite trees, NaN gateway scores, and ledger
  double-consume.

Next: commit + push, seek the adversarial compute review
(PASS_CONTROL_TRAINING) for `--stage train`.

## 2026-07-17 — review-driven design hardenings (pre-freeze; still model-free, no seed consumed)

Construction review surfaced two majors and six minors against the
still-uncommitted cell; all applied before the first commit. Frozen task
and corpus content is UNCHANGED (all six gate task/input files
regenerated byte-identically: sources `836c971b…`/`4149e399…`/
`7a143b41…`, runner inputs `122c631e…`/`e2b6acb4…`/`26b3761a…`); only
code, receipts, and docs moved.

- **MAJOR 1 — aggregate tie guard.** The strict aggregate comparisons in
  `run_benchmark.consequence_reading` were unguarded against ulp-level
  rendering of TRUE rational ties: distinct per-family multisets with
  exactly equal rational aggregates float-render 1 ulp apart
  (demonstrated: parent `[1.0,0.1,0.6,0.8,0.1,0.0,0.1,1.0,0.8,0.1]` vs
  candidate `[0.9,0.2,0.6,0.8,0.1,0.0,0.1,1.0,0.8,0.1]`, both exactly
  0.46, rendering `0.45999999999999996` vs `0.46000000000000008`),
  flipping BOUNDED to COMPOUNDED. Python 3.12 `sum()` is
  Neumaier-compensated (order is not the mechanism) and `math.fsum`
  does not fix it; the fix is the explicit `AGG_TIE_EPSILON = 1e-12`:
  strictly-above means `(candidate - other) > 1e-12`, `|delta| <= 1e-12`
  is a tie and ties are BOUNDED. Real aggregate differences are >=
  ~1.7e-3. Truth-table unit tests added over both demonstrated 1-ulp
  flip orderings (both must read BOUNDED) and a genuine +0.002 win
  (stays COMPOUNDED); the frozen-semantics sentence added to the
  preregistration.
- **MAJOR 2 — standalone reproduction path (owner's standalone
  directive: cross-experiment files are verification aids, NEVER the
  reproduction path).** Every sibling-original requirement became: the
  IN-CELL sha256 pin is the hard fail-closed gate; a PRESENT sibling
  must be byte-identical (divergence fails loudly as tamper evidence);
  an ABSENT sibling is skipped with a recorded note ("absent, in-cell
  pin authoritative"). Applied to `rebuild_lineage.py`
  (`verify_provenance_receipts`), `gen_local_gate.py` (parent receipt +
  predecessor gates), `train_trial.py` (`check_parent_provenance`, note
  recorded in the training receipt), `eval_local_vllm.py` (inherited-arm
  receipt now the in-cell copy), and `run_benchmark.py`
  (`require_count_walk_parent_provenance`; the sibling is HEAD-checked
  only when present). The four predecessor gate files gained sha-pinned
  in-cell copies under `data/predecessor_gates/` so the overlap audit
  runs identically without siblings. Drills: a sibling-free checkout
  passes `rebuild_lineage.py --verify-inputs` AND `gen_local_gate.py
  --check` (verified live by temporarily relocating the three sibling
  cells); divergent-present still fails (unit drills).
- **Minors.** (1) Slack-gloss wording corrected everywhere: every family
  independently gets at most one episode (0.1) of slack below the
  parent — the rule caps depth per family, not the number of families
  using slack. (2) Local write-ahead ledger
  (`runs/local/local_events.jsonl`): an `opened` record (arm, seed,
  seed list, design-receipt sha, monotonic index) appended BEFORE every
  engine event, a `receipts` record sha-pinning the raw artifacts after
  validation; a torn/discarded attempt refuses any new local pass;
  open/refuse/complete unit-tested. (3) `merge_trained_arm.py` preflight
  now hashes the full 9 GB merge-base weights against `ddd7bc4b…`
  (mirroring `train_trial.py`); README now matches reality. (4) Overlap
  audit extended with the enumerative-repair cell's four gate files
  (seeds 88052-88055), same load pattern, in-cell copies; local design
  receipt regenerated (`bd6a8f47…`). (5) `run.py
  require_pushed_checkpoint` catches the git cat-file probe and refuses
  one-line (`stage prerequisite is not committed at HEAD: <path>`); the
  three stage drills re-run green (one-line refusals, no tracebacks).
  (6) Pin symmetry: `check_design.py` now normalized-hash-pins ALL
  three fill-slot files — `run_benchmark.py` (three slots, re-frozen
  `11a6cc14…` after the tie guard), `train_trial.py`
  (`PUBLISHED_ARM_HASHES` single-line sorted-key dict slot,
  `97c06297…`), `eval_local_vllm.py` (`EXPECTED_TRAINED_TREE_SHA256`
  slot, `1b294792…`) — one-byte non-slot edits fail `--check`, legal
  slot fills do not (probed in tests).
- **Verification after the fixes.** `check_design.py --check` green
  twice; `gen_local_gate.py --check` green; task files byte-identical
  (shas above); unit tests 127 → 176, all green; `run.py --smoke` green
  end-to-end; `rebuild_lineage.py --verify-inputs` green (now reporting
  sibling-original status); `make check` green from repo root.

## 2026-07-17 — Sealed event 78168: BOUNDED; cell closed

- Three arms at 78168 (medium tb1024): base 0.1040, count_walk parent
  0.3626, replay_compound candidate 0.3420. The candidate's aggregate
  fell 0.0206 BELOW the parent (aggregate_strictly_beats_parent=False,
  tie guard inactive — a real loss, not a tie) AND warren dipped 0.15
  below parent (families_below_slack=[warren]). Either condition alone
  fires BOUNDED; both did.
- Frozen consequence: BOUNDED — "the replay-compounding law hits
  diminishing returns at stage 8 on this parent; the count_walk
  composite remains the reference; further aggregate pushes need a
  different move class." This is the first chain stage where replay
  compounding failed to add aggregate.
- Descriptive (never gating): the candidate still beats base by 0.238
  (goal gate vs base 8 strict wins, tie on menders, one loss on rites);
  per-family vs parent it is a net-negative reshuffle (lockpick +0.1,
  toolsmith +0.07; menders -0.1, mirage -0.1, warren -0.15, stockade
  -0.025). Replay on an already replay-saturated parent redistributes
  rather than adds.
