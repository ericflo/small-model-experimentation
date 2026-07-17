# Count-Walk Menders Confirmation

The mandatory multi-seed confirmation of lifecycle 27's MECHANISM_ANSWER: at sealed seed 78,163 the count_walk composite drew menders 0.1 while base, zero_root_parent, and replay_ctl7 all drew exactly 0.0 — the preregistered positive branch, first in program history. Eval-only: four fresh sealed medium seeds, four authenticated pre-existing arms per seed, one frozen integer-exact replication rule, no training anywhere.

**Status:** in-progress · since 2026-07-17 · model-free construction frozen, then amended pre-event per the adversarial review (A1+A2 full-episode rule semantics, B1 in-cell lineage package, four minors — recorded in the preregistration's amendment section; no seed consumed); the four-seed benchmark event awaits its PASS verdict

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
happened once in nine recorded medium events, so under the observed noise rate
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
  byte-identical deterministic summary regeneration; the implementation
  signature is checked LIVE before each seed's first gateway call
  (pre-consumption — a drifted suite refuses before any GPU run) and all
  sixteen receipts must equal the prior event's pinned block.
- FROZEN REPLICATION RULE (integer-exact, over the four NEW events only; 78,163
  is prior evidence, never pooled; review amendment A1+A2): an event counts as a
  hit only if it contains at least one FULL menders episode (score contributes
  int(10*s + 1e-9) episodes, FLOOR semantics — partial-credit draws on the k/60
  lattice are recorded but never counted, as hits or episodes); hits_c = new
  events with candidate FULL-EPISODE count > 0; E per arm = sum of
  int(10*score + 1e-9) episodes. REPLICATED iff hits_c >= 2 AND E_c strictly
  exceeds EVERY control's total; NOT_REPLICATED iff hits_c == 0; AMBIGUOUS
  otherwise. No fourth state. The three frozen consequences:
  1. REPLICATED — "the count_walk composite solves menders episodes at a rate no
     control matches; the first confirmed menders capability movement in the
     program."
  2. NOT_REPLICATED — "the 78163 reading closes as seed noise; the
     count-dont-walk dose did not durably move menders; the expression-cost law
     stands; the composite remains a documented artifact (at a true per-event
     hit rate of 0.3 this outcome retains probability ≈ 0.24 — the closure is a
     preregistered funding decision, not a nonexistence proof)."
  3. AMBIGUOUS — "no claim; further spending on this contrast requires a
     mechanism-differentiated NEW design, not more seeds of the same."
- Honest priors (arithmetic in the preregistration; recomputed by
  `scripts/power_analysis.py --check` in smoke and tests, which enforce every
  printed number): under the FULL-EPISODE null (design-time audit over all 9
  recorded medium/tb1024 sealed events, 29 arm-events, 3 full-episode draws;
  the 2 partial draws are rule-invisible, recorded-only) the false-REPLICATED
  probability is 0.0450 at the headline p = 0.10 and 0.0475 at the exact
  p = 3/29 (exact fraction printed by the script); the counterfactual ceiling —
  if every raw-positive draw were promoted to a full episode, which the frozen
  conversion forbids — is 0.0947 at p = 5/29. Power of hits_c >= 2 is 0.5248 /
  0.6875 / 0.8735 at candidate hit rates 0.4 / 0.5 / 0.65, and full REPLICATED
  power (with dominance) is 0.4717 / 0.6289 / 0.8230 (both unchanged by the
  amendment); NOT_REPLICATED retains probability 0.2401 at q = 0.3.
- Standalone boundary (review amendment B1): this cell produces no model but
  EVALUATES three non-base composites, so it carries the complete in-cell
  reproduction package per `docs/quality_gates.md` (including eval-only cells):
  `data/lineage/` (six ordered zero-root stage datasets + the extended
  `lineage_manifest.json` + lifecycle 22's provenance receipts), the stage-7
  production inputs (`data/count_walk.jsonl`, `data/replay_ctl7.jsonl`,
  `data/sft_count_walk.jsonl`, `data/sft_blend.jsonl`,
  `data/stream_token_receipt.json`), the byte-identical production scripts
  (`scripts/lineage_trainers/`, `scripts/train_think.py`,
  `scripts/merge_adapter.py`, `scripts/train_trial.py`,
  `scripts/merge_trained_arm.py`, `scripts/rebuild_clean_chain.py`), and
  `scripts/rebuild_lineage.py` (stages 1-6 rebuild the zero-root parent; stage 7
  trains both arms at fixed seed 85 and merges; `--verify-inputs` runs in smoke
  and tests). The four committed provenance documents remain copied
  byte-identically into `data/provenance/` as verification aids; the measurement
  gateway stays shared per `docs/quality_gates.md`.
- Hidden-label boundary: the benchmark suite's contents are never parsed or read
  as data; only the trusted aggregate gateway (sha `53cf6533…`) runs, and the
  pre-consumption implementation check hashes suite bytes exclusively through
  that gateway's own inventory functions.

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

Lineage package verification alone (no GPU, no writes; also runs inside smoke):

```bash
.venv/bin/python -B experiments/qwen35_4b_count_walk_menders_confirmation/scripts/rebuild_lineage.py --verify-inputs
```

### Ops: crash recovery for torn artifacts

A hard crash can tear (partially write) exactly one derived artifact — the
trailing ledger line, a per-seed `summary.json`, or the terminal
`confirmation_readout.json`. Each of these is a deterministic pure function of
the authenticated gateway receipts and the frozen pins, so the recovery is
always the same:

1. Audit the event directory, then DELETE only the torn artifact (never edit
   it in place).
2. Re-run with `--resume`: the artifact regenerates BYTE-IDENTICALLY from the
   preserved receipts, the byte-equality reconciliation re-anchors it, and the
   ledger close proceeds.

NEVER edit a receipt, summary, ledger line, or readout by hand — every one of
them is sha-pinned at close time and a hand edit permanently fails the chain.
Per-arm gateway receipts are written atomically by the trusted gateway
(temp-file + rename), so a torn receipt should not occur; if one somehow does,
audit it, delete it, and `--resume` re-runs only that arm through the gateway
(receipts are gateway outputs, not deterministic regenerations — the re-run
consumes no new seed because the seed's opened record already exists). A
preserved `<arm>.failure.json` always requires an explicit audit-and-delete
before any retry.

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
  authentication, pre-consumption implementation check, the frozen full-episode
  replication rule).
- `scripts/check_benchmark.py` — ledger-anchored readout writer/verifier.
- `scripts/power_analysis.py` — the exact preregistered power arithmetic (both
  alphas, the counterfactual ceiling, the NOT_REPLICATED retention).
- `scripts/rebuild_lineage.py` — the in-cell standalone rebuild path
  (stages 1-6 zero-root parent; stage 7 both arms at seed 85; `--verify-inputs`).
- `scripts/run.py` — `--smoke` and `--stage benchmark` only.
- `data/lineage/` — the copied ordered stage datasets, the extended
  `lineage_manifest.json`, and lifecycle 22's provenance receipts.
- `data/count_walk.jsonl`, `data/replay_ctl7.jsonl`, `data/sft_count_walk.jsonl`,
  `data/sft_blend.jsonl`, `data/stream_token_receipt.json` — the stage-7
  production inputs, byte-identical copies.
- `scripts/lineage_trainers/`, `scripts/train_think.py`,
  `scripts/merge_adapter.py`, `scripts/train_trial.py`,
  `scripts/merge_trained_arm.py`, `scripts/rebuild_clean_chain.py` — the
  byte-identical production script copies.
- `data/provenance/` — byte-identical verification copies of the four committed
  provenance documents.
- `reports/preregistration.md` — the frozen contract (with the recorded
  pre-event review amendments).
- `reports/artifact_manifest.yaml`
