# Report: Count-Walk Replay Compound (Stage 8) — design frozen

Lifecycle 29. Model-free construction is complete and frozen; no GPU stage has
run and no seed has been consumed. This report records the frozen design; the
results section fills only after the sealed event.

## What this cell asks

Stage 8 of the documented zero-root chain: does REPLAY COMPOUNDING — one fresh
rank-32/alpha-64 adapter trained on the FULL 2,240-row replay pool
(`data/sft_blend.jsonl`, sha `25a9595f…`) ON the count_walk composite parent
(tree `d5fdc55c…`) at fresh seed 86, merged back onto that composite — still
add held-out aggregate, or does the replay-compounding law hit diminishing
returns on this parent?

## The frozen design

- **Treatment.** The chain's established replay-refresh recipe, unchanged:
  epochs 1.0, lr 1e-5, rank 32/alpha 64, batch 1, grad-accum 8, max-length
  4,096, w_think 0.2, w_close 0.2, via the vendored stage-7 trainer
  (`train_think.py`, sha `e0eca2a2…`) with `--model-path` on the composite;
  280 optimizer steps; zero row skips enforced (pool max forward 3,193 <
  4,096). Fresh training seed 86 (grep-fresh; chain seeds 42/43/44/47/51/55/85
  are taken).
- **Fail-closed authentication.** The parent authenticates pre-training and
  pre-merge against the committed lifecycle-27 merge receipt (sha `840edca0…`,
  byte-identical in-cell copy) plus tokenizer/size pins and the full 9 GB
  weights hash; the benchmark runner recomputes the full on-disk tree sha256
  of every arm at the seed-consuming boundary.
- **Local gate.** Retention-only, TWO arms (parent vs candidate; no axis kind
  exists for a pool treatment): three pooled_k3 screens at fresh seeds
  88060/88061/88062, 104 rows each; TWO-SIDED bands on integer screen sums
  (correct ±15, parsed ±9, cap contacts ±9). Promotion only if all three hold;
  the aggregate question belongs exclusively to the sealed event.
- **Sealed event.** Medium / tb1024 / fresh seed 78168, three arms in frozen
  order (base → count_walk → replay_compound) through the trusted gateway
  (`53cf6533…`). Three TODO-pin slots (candidate tree / weights / committed
  merge receipt) fail closed while unfilled; `run_benchmark.py` is frozen by a
  three-slot NORMALIZED hash (`d619d5df…`) so every guard call site is
  byte-frozen pre- and post-fill. One-seed write-ahead ledger; byte-equal
  crash reconciliation (the summary is a pure function of the receipts).
- **Frozen consequence (no third state).** COMPOUNDED iff candidate aggregate
  strictly > parent AND no family strictly below parent by more than 0.1
  (`candidate_family >= parent_family - 0.1 - 1e-9`, exact at both lattice
  boundaries) AND candidate aggregate strictly > base. BOUNDED otherwise. The
  frozen claims and the goal-gate-vs-base descriptive reading are in the
  preregistration.

## Honest priors (frozen before the event)

The stage-7 replay control is the exact move-class precedent: it beat its
parent on 4 of 5 sealed draws (mean +0.018). But the per-family slack clause
historically binds: dips beyond one episode appeared on ~4 of 5
candidate-vs-parent draws across the chain's sealed history. Frozen priors:
P(aggregate strictly > parent) ≈ 0.5-0.6; P(COMPOUNDED) ≈ 0.25-0.40; BOUNDED
is the believed-likelier verdict and is a finding about the law's boundary
(the modal path: aggregate up, one family down by two episodes), not a
failure.

## Verification state at freeze

- `run.py --smoke` green: check_design (normalized pin, gateway sha, frozen
  corpora, no-benchmark-reads audit), `rebuild_lineage.py --verify-inputs`
  (extended manifest `45d1a0d9…`; 7 stage datasets + 2 arm streams + the
  stage-8 pool + 7 provenance receipts + trainers/merger/wrappers),
  `gen_local_gate.py --check` (instruments + freshness + code pins), and all
  127 unit tests.
- Boundary drills refuse: every staged gate without its committed review
  verdict; the sealed runner with unfilled TODO pins; a tampered parent merge
  receipt in a scratch copy; fake/incomplete composite trees; NaN gateway
  scores; ledger double-consume.

## Results

Pending the staged reviews and GPU stages. Terminal artifact:
`runs/benchmark/medium_tb1024_seed78168_compound/summary.json` with the frozen
verdict, claims, per-family tables, and the descriptive goal gate.

## Interpretation

Pending. Both branches are priced in the preregistration: COMPOUNDED promotes
the composite to program reference artifact and feeds the raised-floor
confirmation; BOUNDED closes the replay-compounding move class at stage 8 on
this parent and redirects further aggregate pushes to a different move class.
