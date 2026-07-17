# Preregistration: State-Track Installation (Stage 9)

Frozen before any model event. Lifecycle 30 — stage 9 of the documented
zero-root chain: a DIVERGENT single-kind installation dose of a NEW
transferable skill — STATE-TRACKING UNDER DECLARATIVE UPDATES — onto the
count_walk composite. One fresh adapter trains, one merge publishes, one
two-arm retention gate runs locally, and ONE sealed seed is consumed under a
frozen two-directional consequence. A BOUNDED outcome is a preserved
finding, never permission to change this contract inside this experiment
directory.

## Why this design (the divergent-skill bet)

Replay compounding just **BOUNDED at stage 8** (lifecycle 29): on a
replay-saturated parent, another dose of the accumulated replay pool
*redistributes* strength between families rather than adding aggregate (the
candidate fell 0.021 below the parent and one family tripped the slack
guard). But `count_walk` itself — an enumeration-SKILL dose (a NEW kind, not
replay) — beat its zero-root parent by **+0.036 aggregate** at stage 7. The
inference: a NON-OVERLAPPING transferable SKILL can add where more replay
cannot. This cell tests exactly that with a divergent skill chosen to share
no structure with the chain's existing curricula: state-tracking execution.
If the divergent-skill move class adds aggregate with retention on the same
replay-saturated parent, the install-by-skill lever is not bounded where
replay is.

## Frozen identities

- Experiment: `qwen35_4b_state_track_install` (lifecycle 30, chain stage 9).
- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Treatment: ONE fresh rank-32/alpha-64 QLoRA adapter (`state_track`)
  trained on the FRESH 160-row single-kind state-tracking curriculum
  `data/sft_state_track.jsonl` (sha
  `66a8d5bec184a8a9cba20c2ea088e0216ac4cdbd0820541ee310170eb386e3ab`, kind
  `u_state_track` at full concentration; max forward 775 tokens < 4,096,
  zero skips enforced) from the count_walk composite parent via the
  trainer's `--model-path`, with the chain's frozen QLoRA recipe: epochs
  1.0, lr 1e-5, rank 32, alpha 64, batch 1, grad-accum 8, max-length 4,096,
  w_think 0.2, w_close 0.2 — the ONLY designed delta from the chain's prior
  doses is the curriculum, not the recipe. This stage's fixed FRESH training
  seed is 87 (also the curriculum construction seed; verified grep-fresh in
  training-seed contexts repo-wide; the chain's taken training seeds are
  42/43/44/47/51/55/85/86; no substitution required). 20 optimizer steps
  (160 / 8).
- The designed curriculum (`scripts/gen_state_track_curriculum.py`, seed
  87): each row maintains a running ledger of 3-6 neutral, invented named
  registers through K ∈ {4..8} declarative UPDATE statements (increase /
  decrease / set / set-to-scaled / set-to-sum / move / double), expressed
  across four VARIED SURFACES (plain NL, terse imperative, semi-formal
  `X += 3`, narrated), then answers a final-state QUERY (a register's value,
  which register is largest/smallest, or a pairwise larger). The think
  target shows the ledger updated line-by-line as a running table, then the
  answer — EXECUTION of GIVEN updates (never INDUCTION of an unseen rule),
  which the program's execute-vs-induce law says is installable. Every
  ledger is re-derived by a SECOND independent interpreter and byte-compared;
  every answer is recomputed from that independent state. Contamination
  audit: a banned-vocabulary check rejects any collision with the ten
  benchmark family names (and obvious surface words) or the reference cell's
  universal-curriculum / machine-formalism inventory; ZERO
  canonical-user-message overlap with the stage-8 replay pool, the eleven
  predecessor gate files, and the fresh retention screens (unit-tested).
  Balanced 40/40/40/40 across query types and surfaces; 32 rows per chain
  length; 40 per register count. Think tokens (estimated): mean 168, max 265.
- Parent composite (training base, merge base, and eval arm):
  `large_artifacts/qwen35_4b_count_dont_walk_enumeration/merged/count_walk`,
  tree `d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1`,
  weights `ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3`,
  authenticated FAIL-CLOSED pre-training and pre-merge against the IN-CELL
  sha-pinned provenance copy at `data/provenance/count_walk_merge.json` (sha
  `840edca0638b9e291bb34fde28b4b530df8743faf9b7b18b7f2358ce55ec4c36`,
  payload equality on experiment/name/model/merged-path/tree/weights/
  inner-receipt; the committed lifecycle-27 sibling receipt is a
  VERIFICATION AID — byte-identical when present, skipped with a recorded
  note when absent), the composite's inner merge receipt (`3c432f11…`), the
  tokenizer pins, the weights size (9,078,620,536 bytes), and the full 9 GB
  weights hash immediately before training AND immediately before the merge;
  the benchmark runner recomputes the full on-disk tree sha at the
  seed-consuming boundary.
- Merge: through the vendored external merger `scripts/merge_adapter.py`
  (sha `cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672`)
  with `--base-model` = the count_walk composite, into the explicit
  composite
  `large_artifacts/qwen35_4b_state_track_install/merged/state_track` (scale
  2.0, all 128 LoRA modules applied, full-tree receipt).
- Trainer: the vendored `scripts/train_think.py` (sha
  `e0eca2a230dae5d109d418dcb4cc19af05882a770af14350ffd741a8d5e90f01`,
  byte-identical to the chain's stage-7/8 trainer), wrapped by the
  fail-closed `scripts/train_trial.py`.

## Local gate (BEFORE the sealed event; retention-only, two arms)

The stage-9 treatment DOES install a new kind (`u_state_track`), but the
local gate DELIBERATELY holds out no axis instrument — the transfer /
aggregate question is priced only by the sealed benchmark event on held-out
families (the install-by-transfer doctrine: training data looks nothing like
the eval; prove by transfer to held-out surfaces). The local gate is a pure
RETENTION NON-DRIFT screen.

- Instruments: THREE pooled_k3 retention screens, 104 rows each (8 per each
  of the 13 original skills) from the vendored canonical `gen_curriculum.py`,
  at fresh seeds 88063 / 88064 / 88065 (verified grep-fresh in seed contexts
  repo-wide at design time; everything <= 88062 is known-taken, including the
  replay-compound reference cell's 88060-88062; the frozen sequence starts at
  the next free integer; no substitution required). Model-free generation;
  the frozen design receipt pins the instruments, the freshness audit (zero
  canonical-user-message overlap with every in-cell corpus including the
  state_track curriculum this cell TRAINS on, the three reference cells'
  ELEVEN frozen gate files — seeds 88052-88062, sha-pinned IN-CELL copies
  under `data/predecessor_gates/`; the committed sibling originals are
  verification aids, byte-identical when present and skipped when absent —
  and regenerated prior local seeds 88000-88062), and the code set.
- TWO arms only: the `count_walk` parent composite and the `state_track`
  candidate, arm-major, parent first — 6 authenticated vLLM engine events.
- WRITE-AHEAD local ledger (`runs/local/local_events.jsonl`): an `opened`
  record is appended BEFORE every engine event launches, and a matching
  `receipts` record sha-pinning the run's raw artifacts closes it after
  validation; a new local pass refuses while any opened record lacks its
  matching receipts or the pinned artifacts no longer verify on disk.
- FROZEN PROMOTION RULE (two-sided pooled_k3 bands vs the parent, read on
  pooled means over the three screens, evaluated in exact integer arithmetic
  on the screen SUMS): candidate pooled correct within ±5 of the parent (±15
  on sums); candidate pooled parsed within ±3 (±9 on sums); candidate pooled
  cap contacts within ±3 (±9 on sums). All three must hold; the bands are
  deliberately two-sided (a drift screen, not a win gate — only the sealed
  event may price aggregate movement). No absolute per-kind floors. No
  promotion = the aggregate seed stays sealed forever in this cell.

## Sealed benchmark event

- ONE event: tier `medium`, think budget 1,024, fresh seed 78169 (verified
  grep-fresh; benchmark seeds are spent through 78,168 — the replay-compound
  cell's sealed seed; the frozen seed is the next free integer; no
  substitution required), THREE arms in frozen order:
  1. `base` —
     `large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized`,
     tree `26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677`,
     weights `b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db`.
  2. `count_walk` (parent) — pins above, authenticated against the committed
     lifecycle-27 merge receipt at the seed-consuming boundary.
  3. `state_track` (candidate) — tree / weights / committed merge receipt sha
     are THREE fail-closed TODO-PIN slots the orchestrator fills from the
     committed merge receipt after the merge publishes; the event refuses
     while any slot is None. `run_benchmark.py` is frozen by
     `check_design.py`'s NORMALIZED HASH
     (`8e2d54207590612a3000de30a452a67e036791675c723c082e1d628387854586`):
     exactly the three pin VALUE slots are canonicalized to a fixed
     placeholder before hashing, so every other byte — every guard call site
     included — is byte-frozen pre- and post-fill. The same machinery pins
     `train_trial.py` (`9396cff7…`) and `eval_local_vllm.py` (`8350c61a…`)
     symmetrically on their own fill slots.
- Instrument: only the trusted aggregate gateway
  (`scripts/run_benchmark_aggregate.py`, sha
  `53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17`) ever
  runs; the benchmark suite's contents are never parsed or read as data.
- One-seed WRITE-AHEAD ledger: `opened` before the first gateway call,
  `closed` (sha-pinning the summary) after; any closed record refuses
  forever; a crashed event recovers only under `--resume`. BYTE-EQUAL CRASH
  RECONCILIATION: the summary payload is a pure function of the three gateway
  receipts and the committed prerequisites (no wall-clock), so a crash
  between summary write and ledger close reconciles by recomputing the
  payload and requiring byte equality before the closed record is appended.

## The frozen two-directional consequence (no third state)

Implemented integer-exactly in `run_benchmark.consequence_reading` and
unit-tested over the truth table, both score lattices (k/10 and k/60), and
the demonstrated 1-ulp rational-tie rendering pairs:

- **INSTALLED_TRANSFER** iff the candidate aggregate is STRICTLY above the
  parent aggregate AND no family sits strictly below the parent by more than
  0.1 (every family independently gets at most one episode (0.1) of slack
  below the parent; the comparison is frozen as
  `candidate_family >= parent_family - 0.1 - 1e-9`) AND the candidate
  aggregate is STRICTLY above the base aggregate. FROZEN AGGREGATE SEMANTICS:
  the aggregate comparison is on the gateway-reported float with a 1e-12 tie
  guard — strictly above means `(candidate - other) > 1e-12`; a true rational
  tie (distinct per-family multisets whose exactly equal rational aggregates
  float-render one ulp apart) resolves as BOUNDED; real aggregate differences
  are >= ~1.7e-3. Frozen claim: "a divergent transferable skill installs and
  adds aggregate on the replay-saturated parent; state_track becomes the
  program reference; the divergent-skill move class is not bounded where
  replay is."
- **BOUNDED** otherwise. Frozen claim: "the divergent-skill dose does not add
  aggregate at this dose on this parent; count_walk remains the reference;
  the install-not-equal-convert law extends to this skill."

Also recorded, all DESCRIPTIVE, never gating: the goal gate vs base (all ten
families strictly above base — the 10/10 strict-wins reading) for both
treated arms, the full per-family tables, and the candidate-minus-parent /
candidate-minus-base deltas.

## Honest priors (computed before any event)

- Precedent FOR: the ONLY prior divergent-skill dose in the chain
  (`count_walk`, a fresh enumeration kind onto the stage-6 composite) beat
  its parent aggregate on 4 of 5 sealed draws, mean **+0.032**; it is the
  reason this move class is believed. Precedent AGAINST: this parent is now
  the chain's best (0.357-mean), higher than count_walk's own parent, and the
  per-family slack clause historically BINDS — across the chain's sealed
  history a candidate-vs-parent dip beyond one episode appeared on ~4 of 5
  draws. state_track is also a NARROW single kind trained at only 20
  optimizer steps; a 160-row dose may install the skill locally yet not move
  ten held-out families.
- P(candidate aggregate strictly > parent) ≈ **0.4-0.5**: a divergent skill
  can add, but from a higher parent with a narrow single-kind dose the honest
  read is a coin flip discounted below the count_walk precedent.
- P(no family below the parent by more than 0.1), the clause that
  historically BINDS: ≈ **0.4-0.6** for a divergent skill (less family
  churn than a full replay re-dose, but not free).
- Joint honest prior: **P(INSTALLED_TRANSFER) ≈ 0.30-0.40**;
  P(BOUNDED) ≈ 0.60-0.70. Stated plainly before the event: BOUNDED is the
  modestly likelier verdict; it is a FINDING (the install-not-equal-convert
  boundary — the skill installs locally but does not convert to held-out
  aggregate at this dose), not a failure, and its frozen claim funds a
  different dose/parent rather than a re-roll.
- P(candidate aggregate strictly > base) ≈ 1.0 given the ≈0.27 separation;
  the clause exists to keep the rule total, not because it is expected to
  bind.

## Standalone and provenance boundary (stated plainly)

This cell trains ON and evaluates non-base composites, so it carries the
complete model-reproduction package IN ITS OWN DIRECTORY per AGENTS.md and
`docs/quality_gates.md`:

- `data/lineage/` — the six ordered zero-root stage datasets (byte-identical
  copies), lifecycle 22's seven provenance receipts, and
  `lineage_manifest.json`: lifecycle 27's clean-chain manifest, carried
  through lifecycle 28's `stage7_confirmation_arms` extension and lifecycle
  29's `stage8_replay_compound` block (both unchanged, chain history), and
  EXTENDED with the `stage9_state_track_install` block: arm, curriculum sha
  (`66a8d5be…`), fixed fresh seed 87, trainer/merger shas, count_walk parent
  pins, the recipe (optimizer_steps 20), and three null TODO slots for the
  candidate composite pins (filled post-merge; verification aids, never the
  reproduction path). Manifest byte pin:
  `c05b0eb6c29d1e886f70795a26a1b2732c814f60f26b43fad0663a7060f53a89`.
- Stage-7/8 production inputs, byte-identical copies: `data/count_walk.jsonl`,
  `data/replay_ctl7.jsonl`, `data/sft_count_walk.jsonl`, `data/sft_blend.jsonl`,
  `data/stream_token_receipt.json`. The stage-9 training curriculum
  `data/sft_state_track.jsonl` and its in-cell token-exposure receipt
  `data/state_track_token_receipt.json` are this cell's own.
- Production scripts: `scripts/lineage_trainers/` (three byte-identical stage
  trainers), `scripts/train_think.py`, `scripts/merge_adapter.py`,
  `scripts/rebuild_clean_chain.py` (documentation copy),
  `scripts/stage7_wrappers/` (lifecycle 27's wrappers, byte-identical), and
  this cell's adapted `scripts/train_trial.py` /
  `scripts/merge_trained_arm.py` (stage-9 wrappers; pinned by their own run
  receipts and the local design receipt).
- `scripts/rebuild_lineage.py` — the executable rebuild path: stages 1-6
  rebuild the zero-root parent, stage 7 trains both lifecycle-27 arms at seed
  85 (the rebuilt `count_walk` is the stage-9 base), stage 8 trains
  `replay_compound` (history), and stage 9 trains `state_track` at seed 87
  over the copied curriculum and merges onto count_walk.
  `--verify-inputs` authenticates every copied file against the extended
  manifest and runs inside smoke and the unit tests.

Cross-experiment files (the committed lifecycle-27 merge receipt, the
lifecycle-22 provenance receipts, the three reference cells' frozen gate
files) are VERIFICATION AIDS ONLY and NEVER the reproduction path: every
guard gates on an IN-CELL sha-pinned copy and consults the sibling original
only when it exists — byte-identical then (divergence fails loudly), skipped
with a recorded note when absent. A sibling-free checkout of this cell passes
`rebuild_lineage.py --verify-inputs` and `gen_local_gate.py --check`
unchanged. `benchmarks/` contents are never parsed or read as data.

## Mandatory checkpoint order

1. Model-free construction (this contract, the gate design, the tests, the
   lineage package) — committed, pushed, green.
2. Adversarial compute review — committed `reports/compute_review.md`
   carrying `**Verdict:** `PASS_CONTROL_TRAINING`.` → `--stage train`.
3. Training receipt committed; local adversarial review — committed
   `reports/local_design_review.md` carrying
   `**Verdict:** `PASS_CONTROL_MERGE`.` → `--stage merge`; then the three
   TODO pins filled from the committed merge receipt, committed, and the same
   review carrying `**Verdict:** `PASS_LOCAL_EVENT`.` → `--stage local`.
4. Local receipts + promotion committed; adversarial benchmark design review
   — committed `reports/benchmark_design_review.md` carrying
   `**Verdict:** `PASS_BENCHMARK_EVENT`.` → `--stage benchmark` (the only
   seed-consuming stage).

## Interpretation limits

One sealed seed bounds the read; the verdict prices THIS candidate against
THIS parent at THIS instrument (medium/tb1024). An INSTALLED_TRANSFER verdict
makes the new composite the program reference artifact and shows the
divergent-skill move class adds where replay is bounded — it does not itself
claim a confirmed multi-seed floor raise. A BOUNDED verdict extends the
install-not-equal-convert law to this skill (installed locally, not converted
to held-out aggregate at this dose) and does not claim the skill is worthless
elsewhere. Benchmark firewall unchanged: gateway aggregates and public family
scores only.
