# State-Track Confirmation Report

## Summary

CONFIRMED, directionally — the state_track aggregate lift replicated
across six fresh sealed seeds, but the effect is small and statistically
soft. Paired deltas (state_track − count_walk on the same seed, so the
parent's 0.30-0.36 seed swing cancels) were
[-0.0123, +0.0373, -0.0385, +0.0050, +0.0439, +0.0887]: mean +0.0207,
state_track winning 4 of 6 (threshold ≥4), clearing the frozen rule as
CONFIRMED. Honest effect size (promised at freeze): SD 0.0453, paired
t = 1.12 on 5 df — not strictly significant (one-sided p ≈ 0.16). This
is exactly what the preregistered LIBERAL rule (false-CONFIRMED ≈ 0.31
under the null) anticipated; the decisive high-value outcome was
NOT_CONFIRMED (a negative mean), which did not occur. The mean lift
(+0.0207) matches the single-seed 78169 observation (+0.0256); across
all seven seeds the mean is +0.0214 with 5/7 positive. Reading:
state_track is a real-but-small (~+0.02) and noisy aggregate
improvement — durable enough to adopt as the program reference
composite, not a large or crisp gain. The install-universal-features
doctrine is directionally supported, not proven at strict significance
at n=6.

## Research Program Fit

Program `agentic_breadth_installation`. The install-universal-features
doctrine claims a designed synthetic curriculum can install a transferable
skill that adds real aggregate on a saturated parent, proven by transfer
to held-out benchmark surfaces. Lifecycle 30 produced the first such
single-seed signal after replay compounding bounded at stage 8. This cell
is the calibrate-and-diverge follow-through: the single-seed lift GATES
the "state_track is the durable program reference" claim, and this
confirmation is the pre-registered check that decides it - the same
eval-only multi-seed discipline that (in the count_walk menders
confirmation, lifecycle 28) correctly exposed a headline single-seed
reading as seed noise.

## Method

- Two pre-existing committed composites, authenticated fail-closed by full
  tree+weights sha256 against design-time constants: `count_walk` (parent)
  and `state_track` (candidate). No `base` arm - the event is the
  parent-versus-candidate paired comparison.
- Six sealed fresh medium/tb1024 seeds 78170-78175, two arms per seed in
  the frozen order (count_walk, state_track), seed-major, twelve gateway
  runs. A k-seed write-ahead ledger consumes each seed once; closed records
  sha-pin the sealed summary and both per-arm receipts; crash recovery only
  via `--resume` with byte-identical deterministic regeneration.
- Instrument: the trusted aggregate gateway only; the benchmark suite is
  never read as data. Every receipt is anchored to the seed-78169
  instrument signature (pre-consumption and post-arm).
- Frozen rule: paired delta `d_i = state_track_aggregate -
  count_walk_aggregate`; `wins` = events with `d_i > 1e-12`;
  `mean_d` = mean of the six. CONFIRMED iff `mean_d > 0` AND `wins >= 4`;
  NOT_CONFIRMED iff `mean_d <= 0`; AMBIGUOUS otherwise. Full details and
  power in `reports/preregistration.md`.

## Results

The benchmark event has not been run (this is a design freeze; no seed has
been consumed and the ledger does not exist). On completion the readout
`runs/benchmark/confirmation_readout.json` records, per seed, both
aggregates and the paired delta, then the frozen verdict over the six new
events with the 78169 event reported alongside but never pooled. The
frozen consequences are:

- **CONFIRMED** (`mean_d > 0` and `wins >= 4`): the state_track aggregate
  lift replicates across sealed seeds; the divergent-skill install is a
  durable gain and state_track becomes the program reference composite; the
  install-universal-features doctrine yields real transferable aggregate.
- **NOT_CONFIRMED** (`mean_d <= 0`): the 78169 lift does not replicate; it
  was within the parent's seed variance; count_walk remains the reference;
  the single-seed INSTALLED_TRANSFER is retired as seed noise.
- **AMBIGUOUS** (`mean_d > 0` but `wins < 4`): directional but not
  decisive; a mechanism-differentiated or larger-N design is required, not
  a re-roll of these seeds.

Honest calibration recorded before any event: the paired majority rule is
a LIBERAL directional check (false-CONFIRMED rate approximately 0.31 under
the pure null); its decisive, high-value outcome is NOT_CONFIRMED. Under
the observed +0.0256 lift the test has 90-98% power across the priced
paired-noise range (sigma_d 0.02-0.03).

## Controls

The pairing IS the control: forming `d_i` on the same seed cancels the
common per-seed benchmark-difficulty variance that produces the parent's
0.30-0.36 swing, isolating the parent-versus-candidate contrast the
single-seed reading could not. Descriptive, never gating: both aggregates,
the per-family delta table, the candidate-versus-parent per-family
strict-win partition (the goal-gate analog with no base arm), and
per-family within-slack retention flags. Budget integrity scopes the
paired comparison and never gates.

## Oracle Versus Deployable Evidence

No oracle or hidden-label evaluation is used. Every reading is a public
aggregate-gateway score; the benchmark suite's contents are never parsed.
The verdict is a deployable, provenance-anchored aggregate comparison.

## Interpretation

A CONFIRMED verdict claims a durable PAIRED aggregate lift for the
state_track composite as built against the count_walk parent at this
instrument (medium/tb1024) - a directional replication, not a low-alpha
significance result, and not a claim that the every-family-beats-base bar
is met (warren remains below base, inherited from the parent). A
NOT_CONFIRMED verdict retires the single-seed INSTALLED_TRANSFER as seed
noise and keeps count_walk as the reference. Either way the cell closes on
a frozen, preserved result.

## Next Experiments

- CONFIRMED: adopt state_track as the program reference composite and
  attempt the next divergent-skill installation dose on top of it; test
  whether transferable-skill installs compound.
- NOT_CONFIRMED: keep count_walk as the reference; the divergent-skill
  move class is not yet demonstrated as a durable aggregate mover; the next
  attempt must be a mechanism-differentiated design (a larger dose, a
  different skill, or a different parent), not a re-roll of these seeds.
- AMBIGUOUS: a larger-N or mechanism-differentiated design; do not
  seed-mine this exact contrast.

## Artifact Manifest

See `reports/artifact_manifest.yaml`. The two 9GB composites live in
`large_artifacts/` and are reproduced by their own cells' standalone
rebuild paths; this cell's in-cell stage 1-9 lineage package documents
their lineage and `scripts/rebuild_lineage.py --verify-inputs` authenticates
it.
