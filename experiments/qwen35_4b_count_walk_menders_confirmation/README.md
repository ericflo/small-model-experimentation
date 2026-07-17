# Count-Walk Menders Confirmation

The mandatory multi-seed confirmation of lifecycle 27's MECHANISM_ANSWER: at sealed seed 78,163 the count_walk composite drew menders 0.1 while base, zero_root_parent, and replay_ctl7 all drew exactly 0.0 — the preregistered positive branch, first in program history. Eval-only: four fresh sealed medium seeds, four authenticated pre-existing arms per seed, one frozen integer-exact replication rule, no training anywhere.

**Status:** in-progress · since 2026-07-17 · model-free construction frozen (runner, replication rule, power arithmetic, provenance copies, tests); the four-seed benchmark event awaits its adversarial review

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can DESIGNED synthetic curricula install universal, transferable
  agentic skills into the one 4B — provably, on fully documented lineage?
- Prior anchors: lifecycle 27 (`qwen35_4b_count_dont_walk_enumeration` — the prior
  event: count_walk menders 0.1 vs all controls 0.0 at seed 78,163, aggregates
  0.3312 / 0.3298 / 0.2950 / base 0.0753), lifecycle 26
  (`qwen35_4b_enumerative_repair_protocol` — its replay control drew menders 0.1 at
  seed 78,162, proving untreated arms can draw), and the goal-gate confirmation
  law (a favorable draw is priced by fresh sealed seeds, never by re-reading).

## Question

Does the seed-78163 menders pattern — candidate above zero while every control
sits at exactly zero — replicate across four fresh sealed seeds, or does it close
as seed noise? Single-episode menders draws by untreated arms have already
happened once in eight recorded medium events, so under the observed noise rate
one MECHANISM_ANSWER draw has non-trivial probability; only replication can
separate a real rate difference from a favorable roll.

## Hypothesis

If the count-dont-walk dose genuinely moved menders, the candidate hits episodes
at a per-event rate materially above the program's 0.10 arm-event noise rate and
accumulates more episodes than every control over four events. Honest prior: the
mechanism the dose TAUGHT was already refuted locally (the candidate still thinks
to the 1,024-token cap; fidelity 7/40 vs the 0.50 bar), so this cell tests the
capability movement itself, mechanism-agnostic, and a NOT_REPLICATED close is the
likelier branch under the noise model.

## Setup

- Model: Qwen/Qwen3.5-4B (revision `851bf6e8…`), always; nothing trains or merges.
- Arms (four pre-existing committed composites, full tree+weights sha256
  recomputed and matched fail-closed at event time against design-time constants —
  no TODO pins): `base` (tree `26d8ee48…`, weights `b654e033…`),
  `zero_root_parent` (tree `414f5829…`, weights `6e9aad25…`, lifecycle 22's
  committed lineage merge receipt `e906caea…`), `replay_ctl7` (tree `044a4599…`,
  weights `c5035b4d…`, committed receipt `3f65b4c6…`), `count_walk` (tree
  `d5fdc55c…`, weights `ddd7bc4b…`, committed receipt `840edca0…`).
- Event: four fresh sealed seeds 78,164 / 78,165 / 78,166 / 78,167 (grep-fresh in
  seed contexts at design time; audit in the preregistration), tier medium, think
  budget 1,024, four arms per seed in frozen order, seed-major; per-seed
  write-ahead opened/closed ledger, closed records sha-pin the summary and all
  four gateway receipts; `--resume` is the single recovery path with
  byte-identical deterministic summary regeneration; implementation signature of
  all sixteen receipts must equal the prior event's pinned block.
- FROZEN REPLICATION RULE (integer-exact, over the four NEW events only; 78,163
  is prior evidence, never pooled): hits_c = new events with candidate menders
  > 0; E per arm = sum of round(10*score) episodes. REPLICATED iff hits_c >= 2
  AND E_c strictly exceeds EVERY control's total; NOT_REPLICATED iff hits_c == 0;
  AMBIGUOUS otherwise. No fourth state. The three frozen consequences:
  1. REPLICATED — "the count_walk composite solves menders episodes at a rate no
     control matches; the first confirmed menders capability movement in the
     program."
  2. NOT_REPLICATED — "the 78163 reading closes as seed noise; the
     count-dont-walk dose did not durably move menders; the expression-cost law
     stands; the composite remains a documented artifact."
  3. AMBIGUOUS — "no claim; further spending on this contrast requires a
     mechanism-differentiated NEW design, not more seeds of the same."
- Honest priors (arithmetic in the preregistration; recomputed by
  `scripts/power_analysis.py --check` in smoke and tests): under the observed
  noise rate p ≈ 0.10 the false-REPLICATED probability is 0.0450 (sensitivity
  ceiling 0.0947 at the raw-positive rate 5/29); power of hits_c >= 2 is 0.5248 /
  0.6875 / 0.8735 at candidate hit rates 0.4 / 0.5 / 0.65, and full REPLICATED
  power (with dominance) is 0.4717 / 0.6289 / 0.8230.
- Standalone boundary: this cell produces no model — its production side is the
  preregistration + frozen runner; composite reproduction is lifecycle 27's /
  lifecycle 22's own standalone rebuild path, with the four committed provenance
  documents copied byte-identically into `data/provenance/` as verification aids;
  the measurement gateway stays shared per `docs/quality_gates.md`.
- Hidden-label boundary: the benchmark suite directory is never read; only the
  trusted aggregate gateway (sha `53cf6533…`) runs.

## Run

Smoke (no GPU, no writes):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_count_walk_menders_confirmation/scripts/run.py --smoke
```

Full (the only stage; requires the committed PASS_BENCHMARK_EVENT review and
clean pushed main):

```bash
.venv/bin/python -B experiments/qwen35_4b_count_walk_menders_confirmation/scripts/run.py --stage benchmark
```

## Results

Pending: the model-free construction is frozen; the four-seed sealed event runs
behind its review verdict. The terminal artifact will be
`runs/benchmark/confirmation_readout.json` with the frozen three-state verdict.

## Interpretation

Pending the sealed events. Whatever the draw, the frozen claims above are the
only sentences this cell may emit; a REPLICATED verdict speaks about the
composite as built, never about the refuted count-don't-walk expression
mechanism.

## Knowledgebase Update

- Program evidence updated: pending the readout.
- Program backlog updated: pending the readout.
- Claim ledger updated: pending the readout (design-only work manufactures no claim).

## Artifacts

- `scripts/run_benchmark.py` — the hardened four-seed sixteen-run event runner
  (k-seed write-ahead ledger, byte-equal crash reconciliation, fail-closed arm
  authentication, the frozen replication rule).
- `scripts/check_benchmark.py` — ledger-anchored readout writer/verifier.
- `scripts/power_analysis.py` — the exact preregistered power arithmetic.
- `scripts/run.py` — `--smoke` and `--stage benchmark` only.
- `data/provenance/` — byte-identical verification copies of the four committed
  provenance documents.
- `reports/preregistration.md` — the frozen contract.
- `reports/artifact_manifest.yaml`
