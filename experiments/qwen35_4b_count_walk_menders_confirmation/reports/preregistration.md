# Preregistration: Count-Walk Menders Confirmation

Frozen before any model event. Eval-only replication: no training, no
merging, no corpus, no promotion; four sealed seeds are consumed once
each and the cell closes on the frozen three-state verdict. A failed
outcome is a preserved result, never permission to change this contract
inside this experiment directory.

## Frozen identities

- Experiment: `qwen35_4b_count_walk_menders_confirmation` (lifecycle 28).
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Arms — four pre-existing committed composites, authenticated fail-closed
  at event time by recomputing the FULL on-disk tree sha256 (covering the
  9GB weights) against constants baked at design time (no TODO-PIN slot
  exists anywhere in this cell), in frozen per-seed order:
  1. `base` — `large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized`,
     tree `26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677`,
     weights `b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db`,
     reserialization receipt `25aee794…` inside the composite.
  2. `zero_root_parent` — `large_artifacts/qwen35_4b_zero_root_lineage_rebuild/merged/zero_root_hygiene_explore`,
     tree `414f582950bf60fed2fe462cd141ab98d0f772087b4f9c6bc5aa12f03f379e7d`,
     weights `6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932`,
     authenticated against lifecycle 22's committed lineage merge receipt
     (sha `e906caea7c4b86f4a3eacb96affb7cc2fa9b7cc11e11b634b651cabc5dd01d2b`,
     payload equality on experiment/stage/name/base-model/merged-path/tree/
     weights/size plus the composite's inner merge receipt hash).
  3. `replay_ctl7` — `large_artifacts/qwen35_4b_count_dont_walk_enumeration/merged/replay_ctl7`,
     tree `044a4599ac5264e00256f66f65215ea497d3631d8aebd3467b698253648e484a`,
     weights `c5035b4db47e4da582a805ca009747a5618ef5badc35d960ca216e586dd3ab9d`,
     committed lifecycle-27 merge receipt sha
     `3f65b4c6f4a8b0574a574a89d417c174c3762de6f93508bed8a5a987b91e224c`.
  4. `count_walk` — `large_artifacts/qwen35_4b_count_dont_walk_enumeration/merged/count_walk`,
     tree `d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1`,
     weights `ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3`,
     committed lifecycle-27 merge receipt sha
     `840edca0638b9e291bb34fde28b4b530df8743faf9b7b18b7f2358ce55ec4c36`.
- Event: FOUR fresh sealed seeds `78164 / 78165 / 78166 / 78167`, tier
  `medium`, think budget 1,024, four arms per seed, seed-major, sixteen
  gateway runs total, each seed consumable exactly once under a per-seed
  write-ahead opened/closed ledger whose closed records sha-pin the sealed
  summary AND all four per-arm gateway receipts; crash recovery only via
  `--resume` with byte-identical deterministic summary regeneration.
- Instrument: only the trusted aggregate gateway
  (`scripts/run_benchmark_aggregate.py`, sha
  `53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17`) ever
  runs; the benchmark suite directory is never read. Every one of the
  sixteen receipts must carry the identical benchmark-implementation
  signature AND match the prior event's pinned block (runner
  `a3beecd8b5c89ccfd99a172a6d85321d39b9feb6c29d12f10b2f4d7499e273cb`,
  inventory `218b8615a95f24da962c931e9cd2dba58d853a7bdcd2847cd8e2c42fc2c05f42`,
  56 files), fail closed — all sixteen readings are thereby anchored to
  the seed-78163 instrument.

## Seed-freshness audit (design time)

`78164 / 78165 / 78166 / 78167` were verified grep-fresh in seed contexts
across the repository at design time (patterns `seed[^0-9]{0,4}<n>` and
`<n>` with thousands separators, over every tracked file class): zero
seed-context hits for all four. Every raw numeric hit is a float/sha256
substring inside unrelated per-row data files (CSV metric tables,
inventory JSONL), never a seed field. Benchmark seeds are spent through
78,163 (78,150–78,163 appear in committed ledgers); NO substitution was
required — the frozen seeds are the next four free integers after the
prior event.

## The prior evidence (why this cell exists — never pooled)

Lifecycle 27's sealed seed-78163 medium/tb1024 event
(`experiments/qwen35_4b_count_dont_walk_enumeration/runs/benchmark/medium_tb1024_seed78163_pilot/summary.json`,
sha `a8c394758aeea8255389b1d7c2b6d7c3f37d6072d9ea226f1b4786a8eee191af`,
byte-identical verification copy at
`data/provenance/prior_event_seed78163_summary.json`) drew the
preregistered positive branch — the first MECHANISM_ANSWER in program
history:

| arm | menders | aggregate |
|---|---|---|
| base | 0.0 | 0.0753 |
| zero_root_parent | 0.0 | 0.2950 |
| replay_ctl7 | 0.0 | 0.3298 |
| count_walk | 0.1 | 0.3312 |

The honest caveats, recorded before this cell was funded: it is ONE
menders episode on ONE sealed seed; untreated arms have drawn a menders
episode before (the reference cell's replay control `replay_ctl6` drew
0.1 at seed 78,162); and lifecycle 27's own local reading refuted its
taught mechanism (the candidate still thinks to the 1,024-token cap;
canonical-next fidelity 7/40 vs the frozen 0.50 bar), so whatever
converted that episode is not the taught five-line arithmetic. This cell
asks only: does the PATTERN — candidate menders above zero while every
control sits at zero — replicate across fresh sealed seeds? The 78,163
event is PRIOR EVIDENCE: reported alongside the verdict, authenticated by
sha, and NEVER pooled into the rule below.

## The frozen replication rule (integer-exact, two-directional)

Over the FOUR NEW events only. Let `hits_c` = the number of new events
with candidate (`count_walk`) menders RAW score > 0. Let `E_c` = the
candidate's total menders episodes across the four events, where each
event contributes `round(10*score)` episodes (Python 3 `round`; the
menders family's observed full-episode granularity is 0.1, and the
program's two recorded partial-credit draws of 0.0167 each round to ZERO
episodes while still counting as raw hits). Let `E_j` likewise per
control arm `j` over base, zero_root_parent, replay_ctl7.

- **REPLICATED** iff `hits_c >= 2` AND `E_c > E_j` for EVERY control `j`
  (a tie is not dominance). Frozen claim: "the count_walk composite
  solves menders episodes at a rate no control matches; the first
  confirmed menders capability movement in the program."
- **NOT_REPLICATED** iff `hits_c == 0`. Frozen claim: "the 78163 reading
  closes as seed noise; the count-dont-walk dose did not durably move
  menders; the expression-cost law stands; the composite remains a
  documented artifact."
- **AMBIGUOUS** otherwise. Frozen claim: "no claim; further spending on
  this contrast requires a mechanism-differentiated NEW design, not more
  seeds of the same."

No fourth state. The rule is implemented in
`run_benchmark.replication_reading`, unit-tested over the full truth
table (including the `E_c` tie branch), and computed only from
ledger-pinned receipts.

Also recorded per event, all DESCRIPTIVE, never gating: the full
per-family tables, the four aggregates, the strict-win goal gates versus
base for each treated arm, and the candidate-versus-each-control deltas
(aggregate, per-family, and menders specifically). Budget integrity
(`within_budget` / `wall_seconds` per arm per seed) scopes the paired
comparison and never gates.

## Power and priors (honest, computed before any event)

Frozen noise model for the null: every arm-event independently draws at
least one FULL menders episode with probability p ≈ 0.10 — the program's
observed arm-event rate. Design-time audit over all 8 recorded
medium/tb1024 sealed events (29 arm-events): 3 arm-events drew a full
episode (3/29 = 0.1034: seed 78,157 hygiene_explore, seed 78,162
replay_ctl6, seed 78,163 count_walk — each exactly one episode, 0.1),
plus 2 partial-credit draws of 0.0167 (raw-positive rate 5/29 = 0.1724)
which contribute zero episodes under the frozen conversion.

Under the null at p = 0.10 with one episode per hitting arm-event, the
candidate's total E_c ~ Binomial(4, 0.1) and each control's total
E_j ~ Binomial(4, 0.1), independent:

- P(hits_c >= 2) = 1 − 0.9⁴ − 4·(0.1)·0.9³ = 1 − 0.6561 − 0.2916 = **0.0523**.
- P(false REPLICATED) = Σ_{k=2..4} P(E_c = k) · P(E_j < k)³
  = 0.0486·(0.9477)³ + 0.0036·(0.9963)³ + 0.0001·(0.9999)³
  = 0.04137 + 0.00356 + 0.00010 = **0.0450**.
- Sensitivity bound: promoting every raw-positive draw to a full episode
  (p = 5/29 = 0.1724 on all four arms) gives P(false REPLICATED) =
  **0.0947** — the pessimistic ceiling if partial-credit draws were
  episodes, which the frozen conversion does not allow.

Under a real effect where the candidate's per-event hit rate is q (one
episode per hit) and the controls stay at the null p = 0.10:

- P(hits_c >= 2) = 1 − (1−q)⁴ − 4q(1−q)³ = **0.5248** at q = 0.4,
  **0.6875** at q = 0.5, **0.8735** at q = 0.65.
- P(REPLICATED) (hits AND dominance over all three controls) =
  Σ_{k=2..4} P(Bin(4,q) = k) · P(Bin(4,0.1) < k)³ = **0.4717** at
  q = 0.4, **0.6289** at q = 0.5, **0.8230** at q = 0.65.

Every number recomputes exactly from `scripts/power_analysis.py`
(`--check` runs inside smoke and the unit tests). Stated plainly: a
REPLICATED verdict carries a ≈4.5% false-positive risk under the
observed noise rate, and the test has 47–82% power across the plausible
effect range — an AMBIGUOUS outcome is live and its frozen consequence
(no claim; a mechanism-differentiated NEW design, never more seeds of
the same) forbids seed-mining this contrast.

## Standalone and provenance boundary (stated plainly)

This cell produces NO model: its production side is this preregistration
plus the frozen runner; the measurement side stays shared per
`docs/quality_gates.md` (the trusted aggregate gateway is repo-level
infrastructure referenced in place; `benchmarks/` is never read).
Reproduction of the four evaluated composites is NOT this cell's path:
`count_walk` and `replay_ctl7` rebuild from lifecycle 27's own standalone
clean-chain package
(`experiments/qwen35_4b_count_dont_walk_enumeration/data/lineage/` +
`scripts/rebuild_clean_chain.py`, which also carries the zero-root
parent's six-stage chain), and the zero-root parent from lifecycle 22's
documented rebuild (`experiments/qwen35_4b_zero_root_lineage_rebuild`).
The four committed provenance documents are copied byte-identically into
`data/provenance/` as VERIFICATION AIDS (`replay_ctl7_merge.json`,
`count_walk_merge.json`, `zero_root_parent_merge.json`,
`prior_event_seed78163_summary.json`); the seed-consuming boundary and
smoke both enforce copy-equals-source-equals-pin, fail closed.

## Mandatory checkpoint order

1. Model-free construction (this contract, the runner, the tests, the
   provenance copies) — committed, pushed, green.
2. Adversarial benchmark design review — a committed
   `reports/benchmark_design_review.md` carrying the literal line
   `**Verdict:** `PASS_BENCHMARK_EVENT`.`
3. `benchmark` — the only stage; requires clean pushed green main with
   the preregistration, review, provenance copies, and committed source
   receipts byte-identical at HEAD. No other stage exists.

## Interpretation limits

Four seeds bound the replication read; menders episodes are single items
per seed and are reported as such. A REPLICATED verdict claims a
menders-rate difference for the count_walk composite AS BUILT against
these three controls at this instrument (medium/tb1024) — it does NOT
claim the taught count-don't-walk arithmetic is the mechanism (lifecycle
27's expression-cost reading already refuted that route locally).
Benchmark firewall unchanged: gateway aggregates and public family
scores only.
