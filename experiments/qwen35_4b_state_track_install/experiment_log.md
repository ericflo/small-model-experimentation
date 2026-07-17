# Qwen35 4b State Track Install Experiment Log

## 2026-07-17 — design freeze (lifecycle 30; model-free, no seed consumed)

Stage 9 of the documented zero-root chain: a DIVERGENT single-kind installation
dose of a NEW transferable skill — STATE-TRACKING UNDER DECLARATIVE UPDATES —
onto the count_walk composite. Everything below is model-free construction; no
GPU stage has run, no review has been sought yet, and the sealed seed 78,169 is
unconsumed.

- **Why this design.** Replay compounding BOUNDED at stage 8 (lifecycle 29): on
  a replay-saturated parent, another replay dose redistributes rather than adds.
  But count_walk (a fresh enumeration-SKILL dose) beat its zero-root parent by
  +0.032 mean aggregate, so a NON-OVERLAPPING transferable skill can add where
  replay cannot. This cell tests that with a divergent skill (state-tracking
  execution) sharing no structure with any chain curriculum. Machinery cloned
  byte-identically from the just-finished lifecycle-29 cell
  (`qwen35_4b_count_walk_replay_compound`); adapted names count_walk→state_track,
  seeds, and — the one designed delta — the curriculum generator.
- **The curriculum (the designed delta).** New generator
  `scripts/gen_state_track_curriculum.py`, construction seed 87: 160 rows,
  single kind `u_state_track` at full concentration (dilution law). Each row
  tracks 3-6 neutral invented named registers through K∈{4..8} declarative
  updates (increase / decrease / set / set-to-scaled / set-to-sum / move /
  double) across four surfaces (plain / terse / `X += 3` / narrated), then
  answers a final-state query (value / largest / smallest / pairwise-larger).
  Think target shows the ledger line-by-line then the answer — EXECUTION of
  given updates (execute-vs-induce law). Fail-closed truth audit: every ledger
  re-derived by a SECOND independent interpreter and byte-compared; every answer
  recomputed from that independent state; constant-KIND asserted;
  banned-vocabulary audit rejects any collision with the ten benchmark families
  (+ obvious surface words) or the reference universal-curriculum /
  machine-formalism inventory. Corpus sha
  `66a8d5bec184a8a9cba20c2ea088e0216ac4cdbd0820541ee310170eb386e3ab`; balanced
  40/40/40/40 across query types and surfaces, 32 per chain length, 40 per
  register count; estimated think tokens mean 168 / max 265. Token exposure
  measured in-cell (`data/state_track_token_receipt.json`, composite tokenizer):
  forward 83,919/epoch, nonzero target 58,478, max forward 775 < 4,096 → zero
  skips guaranteed and enforced.
- **Contamination clean.** Banned-vocab audit passes (register names and content
  collide with nothing in the inventory). ZERO canonical-user-message overlap
  (unit-tested) with `data/sft_blend.jsonl` (the replay pool), the eleven
  predecessor gate files (seeds 88052-88062), and the fresh retention screens.
- **Treatment frozen.** One fresh rank-32/alpha-64 adapter (`state_track`) on
  the 160-row curriculum from the count_walk composite parent via `--model-path`,
  with the chain's frozen QLoRA recipe (epochs 1, lr 1e-5, bs 1, ga 8, maxlen
  4096, w_think 0.2, w_close 0.2) at the fixed FRESH training seed 87 (also the
  curriculum construction seed; grep-fresh; chain seeds 42/43/44/47/51/55/85/86
  taken). 20 optimizer steps.
- **Parent authentication frozen.** Fail-closed pre-training and pre-merge:
  committed lifecycle-27 merge receipt (`840edca0…`), byte-identical in-cell
  provenance copy, inner receipt / tokenizer / size pins, then the full 9 GB
  weights hash (`ddd7bc4b…`).
- **Local gate frozen and generated.** Retention-only, TWO arms (parent first,
  then candidate); the new kind is deliberately NOT held out locally — transfer
  is priced by the sealed event (install-by-transfer doctrine). Three pooled_k3
  screens at fresh seeds 88063/88064/88065 (104 rows each, 8 per each of 13
  skills, canonical gen_curriculum.py). Seed audit: everything <= 88062
  known-taken (the replay-compound cell holds 88060-88062); 88063/88064/88065
  verified grep-fresh; no substitution. TWO-SIDED bands on integer screen sums:
  correct ±15, parsed ±9, cap contacts ±9. Design receipt + task/input files
  generated model-free and pinned (`gen_local_gate.py --check` green, receipt
  sha `48c98bae…`); freshness audit clean over eleven predecessor gates and
  prior local seeds 88000-88062.
- **Sealed event frozen.** Medium / tb1024 / fresh seed 78169 (benchmark seeds
  spent through 78,168 — the replay-compound cell's seed; grep-fresh; no
  substitution), three arms in frozen order base → count_walk → state_track
  through the trusted gateway (`53cf6533…`). The candidate's
  tree/weights/committed-receipt pins are three fail-closed TODO slots (None);
  `run_benchmark.py` is frozen by `check_design.py`'s three-slot NORMALIZED hash
  (`8e2d5420…`) — every byte outside the slots, every guard call site included,
  byte-frozen pre- and post-fill. `train_trial.py` (`9396cff7…`) and
  `eval_local_vllm.py` (`8350c61a…`) pinned symmetrically. One-seed write-ahead
  ledger; byte-equal crash reconciliation.
- **Consequence frozen (two-directional, no third state).** INSTALLED_TRANSFER
  iff candidate aggregate strictly > parent AND no family strictly below parent
  by more than 0.1 (`candidate_family >= parent_family - 0.1 - 1e-9`; exactly
  0.1 below passes, 0.10000001 fails; unit-tested over k/10 and k/60 lattices)
  AND candidate aggregate strictly > base. BOUNDED otherwise. Frozen claims in
  the preregistration. Goal gate vs base recorded descriptively.
- **Honest priors frozen.** The only prior divergent-skill dose (count_walk)
  beat its parent 4/5 sealed draws (mean +0.032), but from a higher parent with
  a narrow single-kind dose at 20 steps, and the family-slack clause historically
  binds (~4/5 draws): P(aggregate > parent) ≈ 0.4-0.5, P(INSTALLED_TRANSFER) ≈
  0.30-0.40; BOUNDED is the modestly likelier verdict and is a finding about the
  install-not-equal-convert boundary, not a failure.
- **Standalone package extended.** Copied byte-identically: the full
  `data/lineage/` package (six stage datasets + seven provenance receipts), the
  stage-7/8 production inputs, `lineage_trainers/` ×3, `train_think.py`,
  `merge_adapter.py`, `rebuild_clean_chain.py`, and lifecycle 27's wrappers into
  `scripts/stage7_wrappers/`. The manifest was extended with the
  `stage9_state_track_install` block (arm, curriculum sha, seed 87,
  optimizer_steps 20, trainer/merger shas, count_walk parent pins, three null
  post-merge TODO slots) while carrying lifecycle 28/29's blocks unchanged as
  chain history; new byte pin `c05b0eb6…`. `rebuild_lineage.py` now replays
  stages 1-9; `--verify-inputs` green (stage9_dataset=1).
- **Verification at freeze.** `run.py --smoke` green end-to-end (check_design
  --check, rebuild_lineage --verify-inputs, gen_local_gate --check, 197 unit
  tests). Boundary drills refuse: every staged gate without its committed review
  verdict, the sealed runner with unfilled TODO pins, a tampered parent merge
  receipt in a scratch copy, fake and incomplete composite trees, NaN gateway
  scores, ledger double-consume, and — for the generator — a corrupted ledger
  re-derivation, a banned token, and a mutated kind.

Next: commit + push, seek the adversarial compute review
(PASS_CONTROL_TRAINING) for `--stage train`.

## 2026-07-17 — Sealed event 78169: INSTALLED_TRANSFER; cell closed

- Three arms at 78169 (medium tb1024): base 0.1675, count_walk parent
  0.3004, state_track candidate 0.3260. Candidate beats parent (+0.0256)
  AND base on aggregate, with NO family below the one-episode slack
  (families_below_slack=[]) -> frozen verdict INSTALLED_TRANSFER. First
  NEW aggregate-adding move class since replay bounded at stage 8;
  validates the core doctrine (install a universal skill via a designed
  synthetic curriculum that looks nothing like the eval, prove by
  transfer).
- Per-family vs parent: siftstack +0.2, lockpick +0.1, mirage +0.1;
  ties on chronicle/menders/rites/warren; sub-episode dips sirens -0.1,
  toolsmith -0.03, stockade -0.014 (all within slack). The gains are on
  agentic families plausibly reached by state-tracking transfer.
- HONEST SCOPE (recorded at closure): (1) SINGLE SEED. The parent's own
  aggregate swings 0.30-0.36 across sealed seeds (0.3004 here vs 0.3626
  at 78168), and state_track's 0.3260 sits inside that band — the
  +0.0256 could be partly seed noise. Per the confirmation doctrine
  (which correctly killed the menders reading in lifecycle 28), a fresh
  eval-only multi-seed confirmation on the same committed composites is
  required before crowning state_track the durable reference. (2) The
  ultimate /goal bar (every family strictly > base) is NOT met: warren
  0.200 < base 0.367 — a count_walk-inherited weakness state_track did
  not fix (goal gate vs base: 6 wins, 3 ties, 1 loss on warren).
- state_track composite: tree 45fd2925..., weights b4bafbb7.... Funded
  successor: eval-only multi-seed confirmation of the aggregate lift.
