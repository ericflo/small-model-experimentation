# Report: State-Track Installation (Stage 9) — design frozen

Lifecycle 30. Model-free construction is complete and frozen; no GPU stage has
run and no seed has been consumed. This report records the frozen design; the
results section fills only after the sealed event.

## What this cell asks

Stage 9 of the documented zero-root chain: does a DIVERGENT transferable SKILL
— one fresh rank-32/alpha-64 adapter trained on a fresh 160-row single-kind
STATE-TRACKING curriculum (`data/sft_state_track.jsonl`, sha `66a8d5be…`) ON
the count_walk composite parent (tree `d5fdc55c…`) at fresh seed 87, merged
back onto that composite — add held-out aggregate on the replay-saturated
parent, where another dose of replay just BOUNDED at stage 8?

## Why this design

Replay compounding BOUNDED at stage 8: on a replay-saturated parent it
redistributes rather than adds. But `count_walk` — a fresh enumeration-SKILL
dose — beat its own parent by +0.032 mean aggregate, so a NON-OVERLAPPING skill
can add where replay cannot. state_track is a divergent transferable skill
chosen to share no structure with any chain curriculum: it installs
state-tracking EXECUTION (execute-vs-induce law: execution is installable).

## The frozen design

- **Treatment.** The chain's frozen QLoRA recipe, unchanged: epochs 1.0, lr
  1e-5, rank 32/alpha 64, batch 1, grad-accum 8, max-length 4,096, w_think 0.2,
  w_close 0.2, via the vendored trainer (`train_think.py`, sha `e0eca2a2…`)
  with `--model-path` on the composite; 20 optimizer steps; zero row skips
  enforced (corpus max forward 775 < 4,096, measured in-cell). Fresh training
  seed 87 (also the curriculum construction seed; grep-fresh; chain seeds
  42/43/44/47/51/55/85/86 are taken). The ONLY designed delta is the
  curriculum.
- **The curriculum.** `scripts/gen_state_track_curriculum.py`, seed 87: 160
  rows, single kind `u_state_track` at full concentration. Each row tracks
  3-6 invented named registers through K in {4..8} declarative updates across
  four surfaces (plain / terse / `X += 3` / narrated), then answers a
  final-state query. Truth-audited by independent re-derivation (byte-matched)
  and answer recomputation; banned-vocabulary audit vs the ten families and
  the reference inventory; ZERO canonical-user-message overlap with the
  replay pool, the eleven predecessor gate files, and the retention screens
  (unit-tested). Balanced across surfaces / query types / chain lengths /
  register counts; think tokens mean 168, max 265.
- **Fail-closed authentication.** The parent authenticates pre-training and
  pre-merge against the committed lifecycle-27 merge receipt (sha `840edca0…`,
  byte-identical in-cell copy) plus tokenizer/size pins and the full 9 GB
  weights hash; the benchmark runner recomputes the full on-disk tree sha256
  of every arm at the seed-consuming boundary.
- **Local gate.** Retention-only, TWO arms (parent vs candidate; the new kind
  is deliberately NOT held out locally — transfer is priced by the sealed
  event): three pooled_k3 screens at fresh seeds 88063/88064/88065, 104 rows
  each; TWO-SIDED bands on integer screen sums (correct ±15, parsed ±9, cap
  contacts ±9). Promotion only if all three hold.
- **Sealed event.** Medium / tb1024 / fresh seed 78169, three arms in frozen
  order (base → count_walk → state_track) through the trusted gateway
  (`53cf6533…`). Three TODO-pin slots (candidate tree / weights / committed
  merge receipt) fail closed while unfilled; `run_benchmark.py` is frozen by a
  three-slot NORMALIZED hash (`8e2d5420…`) so every guard call site is
  byte-frozen pre- and post-fill. One-seed write-ahead ledger; byte-equal
  crash reconciliation.
- **Frozen consequence (no third state).** INSTALLED_TRANSFER iff candidate
  aggregate strictly > parent AND no family strictly below parent by more than
  0.1 (`candidate_family >= parent_family - 0.1 - 1e-9`, exact at both lattice
  boundaries) AND candidate aggregate strictly > base. BOUNDED otherwise. The
  frozen claims and the goal-gate-vs-base descriptive reading are in the
  preregistration.

## Honest priors (frozen before the event)

The only prior divergent-skill dose (count_walk) beat its parent on 4 of 5
sealed draws (mean +0.032) — the reason this move is believed. But the parent
is now the chain's best (0.357-mean), the dose is a narrow single kind at 20
steps, and the per-family slack clause historically binds (~4/5 draws). Frozen
priors: P(aggregate strictly > parent) ≈ 0.4-0.5; P(INSTALLED_TRANSFER) ≈
0.30-0.40; BOUNDED is the modestly likelier verdict and is a finding about the
install-not-equal-convert boundary (the skill installs locally but may not
convert to held-out aggregate at this dose), not a failure.

## Verification state at freeze

- `run.py --smoke` green: check_design (three normalized pins, gateway sha,
  five frozen corpora incl. the state_track curriculum, no-benchmark-reads
  audit), `rebuild_lineage.py --verify-inputs` (extended manifest `c05b0eb6…`;
  7 stage datasets + 2 arm streams + the stage-8 pool + the stage-9
  curriculum + 7 provenance receipts + trainers/merger/wrappers),
  `gen_local_gate.py --check` (instruments + freshness over eleven predecessor
  gates + code pins), and all 197 unit tests.
- Boundary drills refuse: every staged gate without its committed review
  verdict; the sealed runner with unfilled TODO pins; a tampered parent merge
  receipt in a scratch copy; fake/incomplete composite trees; NaN gateway
  scores; ledger double-consume; a corrupted ledger re-derivation, a banned
  token, and a mutated kind in the curriculum generator.

## Results

Pending the staged reviews and GPU stages. Terminal artifact:
`runs/benchmark/medium_tb1024_seed78169_install/summary.json` with the frozen
verdict, claims, per-family tables, and the descriptive goal gate.

## Interpretation

Design-frozen. INSTALLED_TRANSFER promotes the composite to program reference
artifact and shows the divergent-skill move class adds where replay is bounded;
BOUNDED extends the install-not-equal-convert law to this skill and redirects
to a different dose/parent. Either way the chain's stage-9 boundary — does a
divergent skill add where replay could not — becomes a measured fact.
