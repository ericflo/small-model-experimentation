# Count-Walk Menders Confirmation Report

## Summary

Design-frozen, events pending. Lifecycle 28 is the eval-only multi-seed
confirmation owed to lifecycle 27's MECHANISM_ANSWER: at sealed seed 78,163 the
count_walk composite drew menders 0.1 while base, zero_root_parent, and
replay_ctl7 all drew exactly 0.0 — but single-episode menders draws by untreated
arms have happened before (replay_ctl6 at seed 78,162), so one such event has
non-trivial probability under seed noise (observed arm-event rate ≈ 0.10). Four
fresh sealed medium/tb1024 seeds (78,164–78,167), four authenticated
pre-existing arms per seed, and one frozen integer-exact rule decide it:
REPLICATED iff the candidate hits menders on at least two of the four events AND
its episode total strictly exceeds every control's; NOT_REPLICATED iff it hits
none; AMBIGUOUS otherwise — no fourth state, with all three claims frozen
verbatim in the preregistration. Preregistered arithmetic: false-REPLICATED
0.0450 under the noise model; power 0.47–0.82 across candidate hit rates
0.4–0.65. The 78,163 event is prior evidence — reported, sha-pinned, never
pooled.

## Research Program Fit

The program's confirmation law: a favorable draw is priced by fresh sealed
seeds, never by re-reading the discovery. This cell does for the first menders
movement what the goal-gate confirmation did for the 10/10 sweep, at the same
instrument (medium/tb1024, trusted gateway only).

## Method

See `reports/preregistration.md` — frozen identities, the seed-freshness audit,
the replication rule, the power arithmetic, and the provenance boundary. The
runner (`scripts/run_benchmark.py`) enforces the review verdict, clean pushed
main, gateway sha, fail-closed tree/weights authentication of all four arms,
the k-seed write-ahead ledger with byte-equal crash reconciliation, and
implementation-signature equality against the pinned prior event.

## Results

Pending: no seed has been consumed. The terminal artifact will be
`runs/benchmark/confirmation_readout.json`; every verdict input is
provenance-anchored (receipt shas pinned in closed ledger records; the readout
refuses any break in the sealed chain).

## Controls

Three control arms per event (base, the zero-root parent, and the
exposure-matched replay control from the same lifecycle-27 training pair), all
authenticated by full tree+weights sha256 against design-time constants;
descriptive per-family tables, goal gates, and candidate-vs-control deltas are
recorded per event and never gate.

## Oracle Versus Deployable Evidence

Gateway aggregates and public family scores only; `benchmarks/` never read.
Menders episode counts derive from public family scores via the frozen
round(10*score) conversion.

## Interpretation

Deferred to the frozen consequence set: REPLICATED claims a menders-rate
difference for the composite as built (mechanism-agnostic — lifecycle 27 already
refuted its taught expression route locally); NOT_REPLICATED closes 78,163 as
seed noise and leaves the expression-cost law standing; AMBIGUOUS forbids
further seeds on this contrast in favor of a mechanism-differentiated new
design.

## Next Experiments

Determined by the verdict, per the frozen claims; no successor is funded from
this cell's design phase.

## Artifact Manifest

Four composite pins are external with committed receipts (verification copies
in `data/provenance/`); everything else is in-repo. See
`reports/artifact_manifest.yaml`.
