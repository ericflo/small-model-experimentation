# Preregistration: Count-Walk Replay Compound (Stage 8)

Frozen before any model event. Lifecycle 29 — stage 8 of the documented
zero-root chain: REPLAY COMPOUNDING onto the count_walk composite. One
fresh adapter trains, one merge publishes, one two-arm retention gate
runs locally, and ONE sealed seed is consumed under a frozen
two-directional consequence. A BOUNDED outcome is a preserved finding,
never permission to change this contract inside this experiment
directory.

## Frozen identities

- Experiment: `qwen35_4b_count_walk_replay_compound` (lifecycle 29).
- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Treatment: ONE fresh rank-32/alpha-64 QLoRA adapter
  (`replay_compound`) trained on the FULL 2,240-row replay pool
  `data/sft_blend.jsonl` (sha
  `25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2`,
  byte-identical copy of the lifecycle-27 committed original; max
  forward length 3,193 tokens < 4,096, zero skips enforced) from the
  count_walk composite parent via the trainer's `--model-path`,
  mirroring the chain's established replay-refresh recipe: epochs 1.0,
  lr 1e-5, rank 32, alpha 64, batch 1, grad-accum 8, max-length 4,096,
  w_think 0.2, w_close 0.2 — identical hyperparameters to the chain's
  replay stages (stage 1 `replay_refresh`, stage 4 `replay_after_close`)
  and to the stage-7 arms (the composite-parent `--model-path` form,
  fixed seed 85); this stage's fixed FRESH training seed is 86
  (verified grep-fresh in training-seed contexts repo-wide; the chain's
  taken training seeds are 42/43/44/47/51/55/85; no substitution
  required). 280 optimizer steps (2,240 / 8).
- Parent composite (training base, merge base, and eval arm):
  `large_artifacts/qwen35_4b_count_dont_walk_enumeration/merged/count_walk`,
  tree `d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1`,
  weights `ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3`,
  authenticated FAIL-CLOSED pre-training and pre-merge against the
  IN-CELL sha-pinned provenance copy at
  `data/provenance/count_walk_merge.json` (sha
  `840edca0638b9e291bb34fde28b4b530df8743faf9b7b18b7f2358ce55ec4c36`,
  payload equality on experiment/name/model/merged-path/tree/weights/
  inner-receipt; the committed lifecycle-27 sibling receipt is a
  VERIFICATION AID — byte-identical when present, skipped with a
  recorded note when absent), the composite's inner merge receipt
  (`3c432f11…`), the tokenizer pins, the weights size (9,078,620,536
  bytes), and the full 9 GB weights hash immediately before training
  AND immediately before the merge; the benchmark runner recomputes the
  full on-disk tree sha at the seed-consuming boundary.
- Merge: through the vendored external merger `scripts/merge_adapter.py`
  (sha `cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672`)
  with `--base-model` = the count_walk composite, into the explicit
  composite
  `large_artifacts/qwen35_4b_count_walk_replay_compound/merged/replay_compound`
  (scale 2.0, all 128 LoRA modules applied, full-tree receipt).
- Trainer: the vendored `scripts/train_think.py` (sha
  `e0eca2a230dae5d109d418dcb4cc19af05882a770af14350ffd741a8d5e90f01`,
  byte-identical to the chain's stage-7 trainer), wrapped by the
  fail-closed `scripts/train_trial.py`.

## Local gate (BEFORE the sealed event; retention-only, two arms)

There is NO axis instrument: the stage-8 treatment is the full replay
pool itself — no new kind exists to hold out — so the local gate is a
pure RETENTION NON-DRIFT screen and the aggregate question belongs
exclusively to the sealed event.

- Instruments: THREE pooled_k3 retention screens, 104 rows each (8 per
  each of the 13 original skills) from the vendored canonical
  `gen_curriculum.py`, at fresh seeds 88060 / 88061 / 88062 (verified
  grep-fresh in seed contexts repo-wide at design time; everything
  <= 88059 is known-taken, including the reference cell's 88056-88059;
  the frozen sequence starts at the next free integer; no substitution
  required). Model-free generation; the frozen design receipt pins the
  instruments, the freshness audit (zero canonical-user-message overlap
  with every in-cell corpus including the training pool, the two
  reference cells' EIGHT frozen gate files — seeds 88052-88059,
  sha-pinned IN-CELL copies under `data/predecessor_gates/`; the
  committed sibling originals are verification aids, byte-identical
  when present and skipped when absent — and regenerated prior local
  seeds 88000-88059), and the code set.
- TWO arms only (no axis kind exists): the `count_walk` parent
  composite and the `replay_compound` candidate, arm-major, parent
  first — 6 authenticated vLLM engine events.
- WRITE-AHEAD local ledger (`runs/local/local_events.jsonl`): an
  `opened` record (arm, seed, the frozen seed list, the design-receipt
  sha, a monotonic index) is appended BEFORE every engine event
  launches, and a matching `receipts` record sha-pinning the run's raw
  artifacts closes it after validation; a new local pass refuses while
  any opened record lacks its matching receipts or the pinned artifacts
  no longer verify on disk — a torn or discarded local attempt stays
  visible, never silently re-rollable.
- FROZEN PROMOTION RULE (two-sided pooled_k3 bands vs the parent, read
  on pooled means over the three screens, evaluated in exact integer
  arithmetic on the screen SUMS): candidate pooled correct within ±5 of
  the parent (±15 on sums); candidate pooled parsed within ±3 (±9 on
  sums); candidate pooled cap contacts within ±3 (±9 on sums). All
  three must hold; the bands are deliberately two-sided (a drift
  screen, not a win gate — only the sealed event may price aggregate
  movement). No absolute per-kind floors; per-kind counts and
  across-screen SDs are descriptive. No promotion = the aggregate seed
  stays sealed forever in this cell.

## Sealed benchmark event

- ONE event: tier `medium`, think budget 1,024, fresh seed 78168
  (verified grep-fresh in seed contexts repo-wide at design time;
  benchmark seeds are spent through 78,167; the frozen seed is the next
  free integer; no substitution required), THREE arms in frozen order:
  1. `base` —
     `large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized`,
     tree `26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677`,
     weights `b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db`,
     reserialization receipt `25aee794…` inside the composite.
  2. `count_walk` (parent) — pins above, authenticated against the
     committed lifecycle-27 merge receipt at the seed-consuming
     boundary.
  3. `replay_compound` (candidate) — tree / weights / committed merge
     receipt sha are THREE fail-closed TODO-PIN slots the orchestrator
     fills from the committed merge receipt after the merge publishes;
     the event refuses while any slot is None. `run_benchmark.py` is
     frozen by `check_design.py`'s NORMALIZED HASH
     (`11a6cc140da470c57f85f2da0215f851f4722acf3e1ee057e44e3cd1605066f5`):
     exactly the three pin VALUE slots are canonicalized to a fixed
     placeholder before hashing, so every other byte — every guard call
     site included — is byte-frozen pre- and post-fill. The same
     machinery pins `train_trial.py` (the `PUBLISHED_ARM_HASHES` value
     slot, normalized hash `97c06297…`) and `eval_local_vllm.py` (the
     `EXPECTED_TRAINED_TREE_SHA256` value slot, normalized hash
     `1b294792…`) symmetrically.
- Instrument: only the trusted aggregate gateway
  (`scripts/run_benchmark_aggregate.py`, sha
  `53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17`)
  ever runs; the benchmark suite's contents are never parsed or read as
  data.
- One-seed WRITE-AHEAD ledger: `opened` appended before the first
  gateway call, `closed` (sha-pinning the summary) after; any closed
  record refuses forever; a crashed event recovers only under an
  explicit `--resume`. BYTE-EQUAL CRASH RECONCILIATION: the summary
  payload is a pure function of the three gateway receipts and the
  committed prerequisites (no wall-clock anywhere in the summary), so a
  crash between summary write and ledger close reconciles by
  recomputing the payload and requiring byte equality with the
  preserved file before the closed record is appended; divergence
  refuses forever.

## The frozen two-directional consequence (no third state)

Implemented integer-exactly in `run_benchmark.consequence_reading` and
unit-tested over the truth table, both score lattices (k/10 and k/60),
and the demonstrated 1-ulp rational-tie rendering pairs:

- **COMPOUNDED** iff the candidate aggregate is STRICTLY above the
  parent aggregate AND no family sits strictly below the parent by more
  than 0.1 — every family independently gets at most one episode (0.1)
  of slack below the parent; the rule caps depth per family, not the
  number of families using slack; frozen a priori as the compounding
  tolerance; the comparison is frozen as
  `candidate_family >= parent_family - 0.1 - 1e-9`, so a family exactly
  0.1 below still passes and 0.10000001 below fails (the 1e-9 only
  absorbs float error at exact lattice multiples of 0.1) — AND the
  candidate aggregate is STRICTLY above the base aggregate. FROZEN
  AGGREGATE SEMANTICS: the aggregate comparison is on the
  gateway-reported float with a 1e-12 tie guard — strictly above means
  `(candidate - other) > 1e-12`, `|delta| <= 1e-12` is a tie, and a
  true rational tie (distinct per-family multisets whose exactly equal
  rational aggregates float-render one ulp apart) resolves as BOUNDED;
  real aggregate differences are >= ~1.7e-3, so the guard can never
  absorb a genuine win. Frozen claim: "replay compounding holds at
  stage 8; the composite becomes the program reference artifact and
  feeds the raised-floor confirmation."
- **BOUNDED** otherwise. Frozen claim: "the replay-compounding law hits
  diminishing returns at stage 8 on this parent; the count_walk
  composite remains the reference; further aggregate pushes need a
  different move class."

Also recorded, all DESCRIPTIVE, never gating: the goal gate vs base
(all ten families strictly above base — the 10/10 strict-wins reading)
for both treated arms, the full per-family tables, and the
candidate-minus-parent / candidate-minus-base deltas.

## Honest priors (computed before any event; sealed history only)

The chain's documented aggregate history (the vendored lineage package
records the chain; the committed sealed-event summaries record its
measured positions — all medium/tb1024):

| arm | 78163 | 78164 | 78165 | 78166 | 78167 | mean |
|---|---|---|---|---|---|---|
| base | 0.0753 | 0.0777 | 0.0911 | 0.0900 | 0.0897 | 0.0848 |
| zero_root_parent (stage 6) | 0.2950 | 0.3269 | 0.3492 | 0.3248 | 0.3304 | 0.3253 |
| replay_ctl7 (stage 7 replay) | 0.3298 | 0.3538 | 0.3358 | 0.3283 | 0.3682 | 0.3432 |
| count_walk (stage 7, THE PARENT) | 0.3312 | 0.3980 | 0.3373 | 0.3269 | 0.3916 | 0.3570 |

(Stage 6 itself measured 0.3462 over base 0.0713 at lifecycle 22's
sealed seed 78,159; stages 1-6 built the parent from base ≈0.07 to
≈0.33.)

Compounding evidence at every prior chain stage: stages 1-6 each added
aggregate per the documented chain; at stage 7 BOTH arms — including
`replay_ctl7`, the EXACT move class this stage repeats (a pure replay
stream trained onto the composite) — beat their stage-6 parent on 4 of
5 sealed draws (replay control deltas +0.0348 / +0.0269 / −0.0134 /
+0.0035 / +0.0378, mean +0.0179; count_walk deltas +0.0362 / +0.0711 /
−0.0119 / +0.0021 / +0.0612, mean +0.0317).

- P(candidate aggregate strictly > parent aggregate) ≈ **0.5-0.6**:
  the stage-7 empirical rate for the same move class is 4/5, but stage
  8 starts from a HIGHER parent (0.3570 vs 0.3253), the pool is now
  two stages stale relative to the composite's behavior, and
  diminishing returns is the believed alternative; the honest prior is
  discounted below the raw 0.8.
- P(no family below the parent by more than 0.1), the clause that
  historically BINDS: across the five sealed stage-7 draws the
  candidate-vs-parent per-family tables show dips beyond one episode on
  4 of 5 draws for count_walk (−0.2 siftstack at 78163/78166, −0.25
  warren at 78165, −0.1148 stockade at 78167) and 5 of 5 for the
  replay control. Under historical per-family noise this clause alone
  passes on roughly **1 of 5** draws. The frozen 0.1 tolerance is
  therefore a deliberately STRICT reading — the modal dip is one
  episode, larger dips are common — and it is frozen anyway because
  the COMPOUNDED claim is a REFERENCE-ARTIFACT promotion: a composite
  that trades a family away by two episodes is not a clean raise of
  the floor.
- Joint honest prior: **P(COMPOUNDED) ≈ 0.25-0.40**;
  P(BOUNDED) ≈ 0.60-0.75, with "aggregate up but one family dipped by
  more than an episode" the MODAL BOUNDED path. Stated plainly before
  the event: BOUNDED is the likelier verdict, it is a FINDING (the
  diminishing-returns boundary of the replay-compounding law), not a
  failure, and its frozen claim funds a different move class rather
  than a re-roll.
- P(candidate aggregate strictly > base) ≈ 1.0 given the ≈0.25
  separation; the clause exists to keep the rule total, not because it
  is expected to bind.

This cell never re-litigates menders: lifecycle 28's AMBIGUOUS closed
that contrast ("no more seeds of the same; a mechanism-differentiated
NEW design required"). Stage 8 asks the AGGREGATE compounding question
with a different treatment (the full replay pool), a different parent,
and a different consequence rule; menders appears only inside the
descriptive per-family tables.

## Standalone and provenance boundary (stated plainly)

This cell trains ON and evaluates non-base composites, so it carries
the complete model-reproduction package IN ITS OWN DIRECTORY per
AGENTS.md and `docs/quality_gates.md`:

- `data/lineage/` — the six ordered zero-root stage datasets
  (byte-identical copies), lifecycle 22's seven provenance receipts,
  and `lineage_manifest.json`: lifecycle 27's clean-chain manifest,
  carried through lifecycle 28's `stage7_confirmation_arms` extension
  (both stage-7 arm streams, seed 85, all pins unchanged; only the
  wrapper-copy paths were re-pointed to `scripts/stage7_wrappers/` and
  the verification commands updated to this cell) and EXTENDED with the
  `stage8_replay_compound` block: arm, training-data sha (25a9595f…),
  fixed fresh seed 86, trainer/merger shas, parent pins, the recipe,
  and three null TODO slots for the candidate composite pins (filled
  post-merge; verification aids, never the reproduction path). Manifest
  byte pin:
  `45d1a0d9b9262dad00eb5576fff5a0427aab7f505e01232ae584fb3b65636d1c`.
- Stage-7 production inputs, byte-identical copies:
  `data/count_walk.jsonl`, `data/replay_ctl7.jsonl`,
  `data/sft_count_walk.jsonl`, `data/sft_blend.jsonl` (ALSO this
  stage's training pool), `data/stream_token_receipt.json`.
- Production scripts: `scripts/lineage_trainers/` (three byte-identical
  stage trainers), `scripts/train_think.py`, `scripts/merge_adapter.py`,
  `scripts/rebuild_clean_chain.py` (documentation copy),
  `scripts/stage7_wrappers/` (lifecycle 27's wrappers, byte-identical),
  and this cell's adapted `scripts/train_trial.py` /
  `scripts/merge_trained_arm.py` (stage-8 wrappers; pinned by their own
  run receipts and the local design receipt).
- `scripts/rebuild_lineage.py` — the executable rebuild path: stages
  1-6 rebuild the zero-root parent (must reproduce its pinned weights
  on this stack), stage 7 trains both lifecycle-27 arms at seed 85 and
  merges them (the rebuilt `count_walk` is the stage-8 base), stage 8
  trains `replay_compound` at seed 86 over the copied pool and merges
  onto it. `--verify-inputs` authenticates every copied file against
  the extended manifest and runs inside smoke and the unit tests.

Cross-experiment files (the committed lifecycle-27 merge receipt, the
lifecycle-22 provenance receipts, the two reference cells' frozen gate
files in the freshness audit) are VERIFICATION AIDS ONLY and NEVER the
reproduction path: every guard gates on an IN-CELL sha-pinned copy
(`data/provenance/count_walk_merge.json`, `data/lineage/provenance/`,
`data/predecessor_gates/` — eight gate files, seeds 88052-88059) and
consults the sibling original only when it exists — byte-identical then
(divergence fails loudly as tamper evidence), skipped with a recorded
note when absent. A sibling-free checkout of this cell passes
`rebuild_lineage.py --verify-inputs` and `gen_local_gate.py --check`
unchanged. The measurement side stays shared per
`docs/quality_gates.md`; `benchmarks/` contents are never parsed or
read as data.

## Mandatory checkpoint order

1. Model-free construction (this contract, the gate design, the tests,
   the lineage package) — committed, pushed, green.
2. Adversarial compute review — committed `reports/compute_review.md`
   carrying the literal line `**Verdict:** `PASS_CONTROL_TRAINING`.`
   → `--stage train`.
3. Training receipt committed; local adversarial review — committed
   `reports/local_design_review.md` carrying
   `**Verdict:** `PASS_CONTROL_MERGE`.` → `--stage merge`; then the
   three TODO pins filled from the committed merge receipt, committed,
   and the same review carrying `**Verdict:** `PASS_LOCAL_EVENT`.`
   → `--stage local`.
4. Local receipts + promotion committed; adversarial benchmark design
   review — committed `reports/benchmark_design_review.md` carrying
   `**Verdict:** `PASS_BENCHMARK_EVENT`.` → `--stage benchmark` (the
   only seed-consuming stage).

## Interpretation limits

One sealed seed bounds the read; the verdict prices THIS candidate
against THIS parent at THIS instrument (medium/tb1024). A COMPOUNDED
verdict makes the new composite the program reference artifact and
FEEDS the raised-floor confirmation (a fresh multi-seed cell) — it does
not itself claim a confirmed floor raise. A BOUNDED verdict closes the
replay-compounding move class at stage 8 on this parent and does not
claim the pool is worthless elsewhere. Benchmark firewall unchanged:
gateway aggregates and public family scores only.
