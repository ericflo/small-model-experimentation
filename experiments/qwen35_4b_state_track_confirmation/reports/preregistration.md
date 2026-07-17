# Preregistration: State-Track Confirmation

Frozen before any model event. Eval-only replication: no training, no
merging, no corpus, no promotion; six sealed seeds are consumed once each
and the cell closes on the frozen three-state verdict. A failed outcome is
a preserved result, never permission to change this contract inside this
experiment directory.

## Frozen identities

- Experiment: `qwen35_4b_state_track_confirmation` (lifecycle 31).
- Model: only `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Arms — TWO pre-existing committed composites, authenticated fail-closed
  at event time by recomputing the FULL on-disk tree sha256 (covering the
  9GB weights) against constants baked at design time (no TODO-PIN slot
  exists anywhere in this cell), in frozen per-seed order:
  1. `count_walk` (the parent / baseline) —
     `large_artifacts/qwen35_4b_count_dont_walk_enumeration/merged/count_walk`,
     tree `d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1`,
     weights `ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3`,
     authenticated against this cell's IN-CELL provenance copy
     `data/provenance/count_walk_merge.json` (file sha
     `840edca0638b9e291bb34fde28b4b530df8743faf9b7b18b7f2358ce55ec4c36`,
     payload equality on experiment/name/model/merged-path/tree/weights plus
     the composite's inner merge-receipt hash
     `3c432f110fe96a508d6a75ab34e4a649671a3d7b2d942f3346cab609bef437d7`).
  2. `state_track` (the candidate) —
     `large_artifacts/qwen35_4b_state_track_install/merged/state_track`,
     tree `45fd2925e417c82e4848b2ca89907934df9e60503b6529af0bddbd8aa359be7e`,
     weights `b4bafbb7d3ff8dedd2fa216bc9c62997d960d43a6cac22a88976245bcc35d1c1`,
     authenticated against this cell's IN-CELL provenance copy
     `data/provenance/state_track_merge.json` (file sha
     `089f280eab1b6f4afd53e636a49f1b4fd92efd5fa1ee42a1a07e35e49a98c94e`,
     inner merge-receipt hash
     `d23862f70cdbb71b2b232bee0501e65f45a432cacd3e37189418194e27493a0d`).

  There is NO `base` arm: the two-arm event is the parent-versus-candidate
  paired comparison. For each arm the IN-CELL sha-pinned provenance copy is
  the authoritative fail-closed gate; the committed sibling original in its
  own cell is a VERIFICATION AID only — byte-identical when present
  (divergence fails loudly as tamper evidence), skipped with a recorded
  note when absent (owner's standalone directive: in-cell pins are
  authoritative, cross-experiment SHAs are verification aids, never the
  reproduction path).
- Event: SIX fresh sealed seeds `78170 / 78171 / 78172 / 78173 / 78174 /
  78175`, tier `medium`, think budget 1,024, two arms per seed, seed-major,
  twelve gateway runs total, each seed consumable exactly once under a
  per-seed write-ahead opened/closed ledger whose closed records sha-pin
  the sealed summary AND both per-arm gateway receipts; crash recovery only
  via `--resume` with byte-identical deterministic summary regeneration.
- Instrument: only the trusted aggregate gateway
  (`scripts/run_benchmark_aggregate.py`, sha
  `53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17`) ever
  runs; the benchmark suite's contents are never parsed or read as data.
  BEFORE each seed's first gateway call the LIVE implementation
  signature — computed through the sha-authenticated gateway's own
  hash-only inventory functions (suite bytes hashed, never parsed; the
  hidden-label firewall holds) — must equal the prior event's pinned
  block, so a drifted suite refuses pre-consumption. Afterwards every one
  of the twelve receipts must carry the identical benchmark-implementation
  signature AND match the same pinned block (runner
  `a3beecd8b5c89ccfd99a172a6d85321d39b9feb6c29d12f10b2f4d7499e273cb`,
  inventory `218b8615a95f24da962c931e9cd2dba58d853a7bdcd2847cd8e2c42fc2c05f42`,
  56 files), fail closed — all twelve readings are thereby anchored to the
  seed-78169 instrument.

## Seed-freshness audit (design time)

`78170 / 78171 / 78172 / 78173 / 78174 / 78175` were verified grep-fresh
in seed contexts across the repository at design time (patterns
`seed[_= "]?<n>`, `_seed<n>`, `"seed": <n>`, over every tracked file
class): zero seed-context hits for all six. Every raw numeric hit is a
float/sha256 substring inside unrelated per-row data files, never a seed
field. Benchmark seeds are spent through 78,169 (the prior event); NO
substitution was required — the frozen seeds are the next six free
integers after the prior event.

## The prior evidence (why this cell exists — never pooled)

Lifecycle 30's sealed seed-78169 medium/tb1024 event
(`experiments/qwen35_4b_state_track_install/runs/benchmark/medium_tb1024_seed78169_install/summary.json`,
sha `187cc3acfe81016899cb08a8bebf5f6045a6cabba9868edd5379c51708ec1192`,
byte-identical verification copy at
`data/provenance/prior_event_seed78169_summary.json`) drew the
preregistered positive branch — INSTALLED_TRANSFER, the first
aggregate-adding move class since replay compounding bounded at stage 8:

| arm | aggregate |
|---|---|
| base | 0.1675 |
| count_walk (parent) | 0.3004 |
| state_track (candidate) | 0.3260 |

The candidate beat the parent by a PAIRED lift of +0.0256 with no family
below the one-episode retention slack. The honest caveat recorded at that
time and the reason this cell was funded: it is ONE sealed seed, and the
parent's OWN aggregate swings 0.30–0.36 across sealed seeds (0.3004 at
78169 versus 0.3626 at 78168), so state_track's 0.3260 sits inside the
parent's own seed band. The +0.0256 lift could be partly or wholly seed
noise. This cell asks only: does the PAIRED lift — state_track's aggregate
above count_walk's on the SAME seed — replicate across fresh sealed seeds?
The 78,169 event is PRIOR EVIDENCE: reported alongside the verdict,
authenticated by sha, and NEVER pooled into the rule below.

## The frozen paired replication rule (two-directional)

Over the SIX NEW events only. For each event `i` compute the PAIRED
aggregate delta

    d_i = state_track_aggregate_i − count_walk_aggregate_i

on the SAME seed, so the large common per-seed benchmark-difficulty
variance cancels — this is the whole point of pairing. Let

- `wins` = number of events with `d_i > AGG_TIE_EPSILON` (the 1e-12 tie
  guard: a true rational tie whose two float renderings differ by one ulp
  is NOT a win; real aggregate differences per event are ≥ ~1.7e-3, nine
  orders of magnitude above the guard);
- `mean_d` = arithmetic mean of the six `d_i`, "strictly positive" iff
  `mean_d > AGG_TIE_EPSILON` (the mean of six lands on a 1/3,600 lattice,
  smallest nonzero |mean_d| ≈ 2.8e-4, far above the guard).

- **CONFIRMED** iff `mean_d` strictly positive AND `wins >= 4`
  (`ceil(2·6/3) = 4`, a strict two-thirds majority). Frozen claim: "the
  state_track aggregate lift replicates across sealed seeds; the
  divergent-skill install is a durable gain and state_track is the program
  reference composite; the install-universal-features doctrine yields real
  transferable aggregate."
- **NOT_CONFIRMED** iff `mean_d` is not strictly positive
  (`mean_d <= AGG_TIE_EPSILON`). The mean clause DOMINATES: even at
  `wins >= 4`, a non-positive mean closes NOT_CONFIRMED. Frozen claim: "the
  78169 lift does not replicate; it was within the parent's seed variance;
  count_walk remains the reference; the single-seed INSTALLED_TRANSFER is
  retired as seed noise."
- **AMBIGUOUS** otherwise (`mean_d` strictly positive but `wins < 4`).
  Frozen claim: "directional but not decisive; a mechanism-differentiated
  or larger-N design is required, not a re-roll of these seeds."

No fourth state. The rule is implemented in
`run_benchmark.paired_reading`, unit-tested over the full truth table
(including the `wins >= 4`-but-`mean <= 0` edge that resolves
NOT_CONFIRMED, the tie-guard branches, and a total-partition sweep), and
computed only from ledger-pinned receipts.

Also recorded per event, all DESCRIPTIVE, never gating: both aggregates,
the paired delta, the per-family delta table (candidate − parent), the
candidate-versus-parent per-family strict-win partition (the goal-gate
analog with no base arm), and the per-family within-slack retention flags.
Budget integrity (`within_budget` / `wall_seconds` per arm per seed)
scopes the paired comparison and never gates.

## Power and priors (honest, computed before any event)

The six paired deltas `d_i` are modelled as i.i.d. `Normal(mu, sigma_d)`.

sigma_d ARITHMETIC (shown). The marginal single-arm across-seed aggregate
SD is ~0.03 (`SIGMA_ARM`; the parent drew 0.3004 at 78169 and 0.3626 at
78168). For two arms measured on the SAME seed with correlation `rho`,
`Var(d) = 2·sigma_arm²·(1 − rho)`, i.e.
`sigma_d = sigma_arm·sqrt(2·(1 − rho))`. A moderate-to-high positive
correlation (the two arms share the seed's item difficulty) gives:
`rho = 0.778 → sigma_d = 0.020`; `rho = 0.653 → sigma_d = 0.025`
(HEADLINE); `rho = 0.500 → sigma_d = 0.030`. The range 0.02–0.03 is priced;
the headline is `sigma_d = 0.025`.

(a) FALSE CONFIRMED under the null `mu = 0`. Because the deltas are
symmetric about 0, `wins ~ Binomial(6, 0.5)` so
`P(wins >= 4) = 22/64 = 0.34375` exactly and `P(mean_d > 0) = 0.5` exactly.
`wins` and `mean_d` are POSITIVELY correlated (more positive deltas → larger
mean), so the joint `P(mean_d > 0 AND wins >= 4)` lies strictly between the
independence product 0.17188 and the marginal 0.34375. Under `mu = 0` the
joint is SCALE-FREE (both events are scale-invariant functions of six
i.i.d. symmetric normals). Computed by deterministic numerical convolution
of the sign-split sub-densities (no simulation), it is

    P(false CONFIRMED) = 0.3110.

STATED PLAINLY: this is a fairly LIBERAL directional replication check,
NOT a stringent test. Under the pure null (no true difference) the rule
still fires CONFIRMED ≈ 31% of the time, because "4 of 6 coin-flip wins
plus a positive mean" is a weak bar. This cell GATES the durable-reference
claim, and the honest reading is that a CONFIRMED verdict establishes a
DIRECTIONAL replication of the paired lift, not a low-alpha significance
result. The value is asymmetric: a NOT_CONFIRMED verdict (mean ≤ 0) is the
decisive, high-value outcome — it retires the single-seed headline as seed
noise and frees the budget — and that outcome is what the design is built
to force out honestly.

(b) POWER under the observed effect `mu = +0.0256`, at each `sigma_d`.
`wins ~ Binomial(6, Phi(mu/sigma_d))`; `mean_d ~ Normal(mu, sigma_d²/6)` so
`P(mean_d > 0) = Phi(sqrt(6)·mu/sigma_d)`. `P(CONFIRMED)` is the same
convolution joint at the shifted mean:

| sigma_d | rho | P(mean_d>0) | P(wins>=4) | **P(CONFIRMED)** | P(AMBIGUOUS) | P(NOT_CONFIRMED) |
|---|---|---|---|---|---|---|
| 0.020 | 0.778 | 0.9991 | 0.9840 | **0.9839** | 0.0152 | 0.0009 |
| 0.025 | 0.653 | 0.9939 | 0.9502 | **0.9494** | 0.0445 | 0.0061 |
| 0.030 | 0.500 | 0.9817 | 0.9051 | **0.9028** | 0.0789 | 0.0183 |

If the +0.0256 lift is real at these paired-noise levels the six-seed test
confirms it with 90–98% power. The `P(NOT_CONFIRMED)` column is the closure
risk quoted in the frozen NOT_CONFIRMED consequence: even a real modest
effect leaves a 0.1–1.8% chance the test closes the line. Every number
recomputes from `scripts/power_analysis.py` (`--check` runs inside smoke
and the unit tests and enforces every printed number within a 5e-4
deterministic-quadrature tolerance).

## Standalone and provenance boundary (stated plainly)

This cell produces NO model, but it EVALUATES two non-base composites and
therefore carries the complete model-reproduction package IN ITS OWN
DIRECTORY per `docs/quality_gates.md` (eval-only cells must comply) and the
AGENTS.md standalone non-negotiable. The reproduction path is IN-CELL and
identical to lifecycle 30's carried stage 1-9 clean-chain package:

- `data/lineage/` — the six ordered zero-root stage datasets (`stage01_…`
  through `stage06_…`, byte-identical copies), lifecycle 22's seven
  provenance receipts (`data/lineage/provenance/`), and
  `lineage_manifest.json` (sha
  `c05b0eb6c29d1e886f70795a26a1b2732c814f60f26b43fad0663a7060f53a89`) —
  lifecycle 27's clean-chain manifest carried through the stage-7
  confirmation arms, the stage-8 replay_compound block, and the stage-9
  `state_track_install` block (`extended_by = qwen35_4b_state_track_install`,
  the source cell — this confirmation cell carries it byte-identically).
- Production inputs, byte-identical copies: `data/count_walk.jsonl`,
  `data/replay_ctl7.jsonl`, `data/sft_count_walk.jsonl`,
  `data/sft_blend.jsonl`, `data/sft_state_track.jsonl`,
  `data/stream_token_receipt.json`.
- Production scripts, byte-identical copies: `scripts/lineage_trainers/`
  (all three stage trainers), `scripts/train_think.py`,
  `scripts/merge_adapter.py`, `scripts/train_trial.py`,
  `scripts/merge_trained_arm.py`, `scripts/stage7_wrappers/`, and
  `scripts/rebuild_clean_chain.py` (documentation copy).
- `scripts/rebuild_lineage.py` — this cell's executable rebuild path
  (stages 1-6 rebuild the zero-root parent, stage 7 the count_walk arm,
  stage 9 the state_track arm). Its `--verify-inputs` mode authenticates
  every copied file against the manifest shas and runs inside smoke and the
  unit tests. The two composites' full GPU reproduction is delegated to
  their own cells' standalone rebuild paths; the provenance copies here are
  verification aids (in-cell pins authoritative).

The measurement side stays shared (the trusted aggregate gateway is
repo-level infrastructure referenced in place; `benchmarks/` contents are
never parsed or read as data). The three provenance documents are copied
byte-identically into `data/provenance/` as VERIFICATION AIDS
(`count_walk_merge.json`, `state_track_merge.json`,
`prior_event_seed78169_summary.json`); the seed-consuming boundary and
smoke both enforce copy-equals-source-equals-pin, fail closed.

## Mandatory checkpoint order

1. Model-free construction (this contract, the runner, the tests, the
   provenance copies, the lineage package) — committed, pushed, green.
2. Adversarial benchmark design review — a committed
   `reports/benchmark_design_review.md` carrying the literal line
   `**Verdict:** `PASS_BENCHMARK_EVENT`.`
3. `benchmark` — the only stage; requires clean pushed green main with the
   preregistration, review, provenance copies, and committed source
   receipts byte-identical at HEAD. No other stage exists.

## Interpretation limits

Six seeds bound the replication read via a directional paired majority
rule. A CONFIRMED verdict claims a DURABLE paired aggregate lift for the
state_track composite AS BUILT against the count_walk parent at this
instrument (medium/tb1024) — it is a directional replication, not a
low-alpha significance result (α ≈ 0.31 under the pure null, stated
above), and it does NOT claim the every-family-beats-base bar is met
(warren remains below base, inherited from the parent). Benchmark firewall
unchanged: gateway aggregates and public family scores only.
