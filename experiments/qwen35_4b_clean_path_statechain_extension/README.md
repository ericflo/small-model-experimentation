# Clean-Path Statechain Extension

**Status:** finished · 2026-07-16 · verdict PILOT_NOT_PROMOTED + CONVERSION_NOT_REPLICATED — the install held on its third parent (local 21/40 strict, retention in-band) and the clean model beat base (0.3333 vs 0.1234, 2.7×) and its replay control, but lost to its parent (−0.018) and the rites conversion did NOT replicate on the clean lineage (candidate 0.0 vs the original lineage's 0.300; converts_on_clean_lineage false) — the conversion is 1-for-2 and lineage-dependent; footnote: the candidate took a menders draw (0.017, a strict win) on the strongest base seed ever drawn (6/10 gates all around)

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can DESIGNED synthetic curricula install universal, transferable
  agentic skills into the one 4B — provably, on fully documented lineage?
- Prior anchors: lifecycle 18 (`qwen35_4b_statechain_only_dose` — the statechain dose
  installs, 21/40 axis strict over both controls, and CONVERTS to the rites family:
  0.300 vs 0.100/0.100 paired at sealed 78,154) and lifecycle 22
  (`qwen35_4b_zero_root_lineage_rebuild` — the six documented stages replayed from a
  fresh zero-initialized adapter produce the zero-root composite, tree `414f5829…`,
  weights `6e9aad25…`, 0.3462 aggregate with 7/10 strict wins and zero losses at
  sealed 78,159).

## Question

Lifecycle 23 — the mission's cleanest artifact. Does the PROVEN statechain converter
dose, applied byte-identically to the ZERO-ROOT composite, produce a single installed
model whose ENTIRE lineage is documented and contamination-free end-to-end — and does
the rites conversion replicate ON THE CLEAN LINEAGE?

## Hypothesis

The statechain install is a property of the dose, not of the blend-rooted parent it
was first proven on: the same 160 frozen rows at the same exposure-matched geometry
should clear the same calibrated gate from the zero-root composite, and the local
install should again convert to the rites family at medium.

## Setup

- Model: Qwen/Qwen3.5-4B (revision `851bf6e8…`), always.
- Parent and adapter base: the zero-root composite
  (`large_artifacts/qwen35_4b_zero_root_lineage_rebuild/merged/zero_root_hygiene_explore`,
  tree `414f5829…`, weights `6e9aad25…`), authenticated against lifecycle 22's
  committed merge receipt (`e906caea…`; byte-identical provenance copy in
  `data/lineage/provenance/merge.json`).
- Treatment: `data/sft_statechain_only.jsonl` — the source cell's frozen 160-row
  corpus copied BYTE-IDENTICALLY (`ab6c7845…`); fresh instances would change the
  treatment, so the byte-copy is the controlled choice. Replay pool `sft_blend.jsonl`
  (`25a9595f…`) byte-identical to every predecessor copy.
- Arms: `replay_ctl4` (control, trains FIRST) then `statechain_clean` (candidate);
  fresh rank-32/alpha-64 adapters, NO warm start, training seed 73, 1 epoch over
  1,520 rows (190 optimizer updates, LR 1e-5, batch 1×8, max length 4,096,
  w_think/w_close 0.2).
- Exposure: exact zero-delta three-axis MILP (forward / nonzero-target / absolute
  loss mass ×5) at the frozen geometry — 1,280-row shared stratified core + 240-row
  variable block (control: 240 replay; candidate: 160 treatment + 80 fillers),
  namespace seed 55,150.
- Local gate (three arms: parent + both trained): 40-row statechain axis holdout at
  seed 88,041 (10 per formalism, FRESH instances from the copied generator) + three
  104-row retention screens at 88,042/88,044/88,045 under pooled_k3 (88,043 is taken
  by `qwen35_4b_counterfactual_plan_reflection_transfer` — documented skip).
  Promotion: axis total strictly > parent AND > replay_ctl4; pooled retention bands
  on screen sums (correct −15, caps +9, parsed −9) vs BOTH controls.
- Conditional benchmark (only on promotion): ONE sealed medium tb1024 event at fresh
  seed 78,160, four arms in frozen order — base (`26d8ee48…`), zero_root_parent
  (`414f5829…`), replay_ctl4, statechain_clean. Trained-arm pins are six fail-closed
  TODO-PIN slots in `scripts/run_benchmark.py`, frozen by check_design's
  NORMALIZED-HASH pin (lifecycle 22's mechanism).
- Primary metric: local axis-holdout total (promotion), then pilot gate (candidate
  aggregate strictly > base AND > replay_ctl4 AND > zero_root_parent).
- Frozen framing: menders is closed, so the winnable ceiling is 9/10; the readings of
  consequence are (a) the rites conversion ON THE CLEAN LINEAGE (candidate rites vs
  parent/replay rites, paired) and (b) the fully documented model's per-family
  profile. Any 10/10 is a menders draw and feeds a fresh confirmation cell before any
  claim.
- Standalone: `data/lineage/` carries the complete clean-chain package — the six
  zero-root stage datasets, lifecycle 22's stage + merge receipts as provenance
  documents, the trainer/merger copies, and a clean-chain manifest recording this
  cell's dose as STAGE 7. NO blend root exists anywhere in this cell (fail-closed).
- Hidden-label boundary: gate answers live only in `data/local_tasks_seed*.jsonl`;
  the model-facing `local_input_seed*.jsonl` files carry id/messages/meta only. The
  benchmark suite directory is never read; only the trusted aggregate gateway runs.

## Run

Smoke (no GPU, no writes):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_clean_path_statechain_extension/scripts/run.py --smoke
```

Full (one stage per pushed checkpoint, each behind its review verdict):

```bash
.venv/bin/python -B experiments/qwen35_4b_clean_path_statechain_extension/scripts/run.py --stage train-control
# then: train-candidate, merge-arms, local, benchmark
```

Standalone lineage verification (no GPU) / full clean-chain rebuild (GPU):

```bash
.venv/bin/python -B experiments/qwen35_4b_clean_path_statechain_extension/scripts/rebuild_clean_chain.py --verify-inputs
```

## Results

Local gate: PROMOTED on all eight checks — the statechain install's third replication, on its third distinct parent (axis 21/40 strictly over parent 19 and replay 16; pooled retention 61.33 vs 62.33/63.0, deep inside the calibrated bands). Training-loss property recorded: the clean chain fits the replay surface at ~1.3 versus the original lineage's ~0.43 while performing within ~10% at the benchmark (loss-level ≠ capability, dramatically).

Sealed event at 78,160 (all arms authenticated; the six-slot normalized pin held through the fill):

| arm | aggregate | goal gate vs base | rites |
|---|---|---|---|
| base | 0.1234 (strongest base draw yet) | — | 0.100 |
| zero_root_parent | 0.3517 | 6/10 | 0.100 |
| statechain_clean | 0.3333 | 6/10 (incl. a strict MENDERS win, 0.017) | **0.000** |
| replay_ctl4 | 0.3119 | 6/10 | 0.000 |

Pilot: candidate > base ✓, > replay ✓, > parent ✗ (−0.018) — NOT promoted, the same shape as the original statechain cell. The frozen conversion reading: `converts_on_clean_lineage: false` — candidate rites 0.0 against the original-lineage conversion's 0.300.

## Interpretation

Three durable readings. (1) The statechain INSTALL is robust — three parents, three promotions, retention held each time under calibrated bands. (2) The CONVERSION is lineage-dependent: 1-for-2, expressed on the prefix lineage and absent on the clean one, and the pattern is legible — rites/sirens/mirage were precisely the C53 prefix's strengths, so the designed dose appears to convert only where the prefix's substrate already leans toward the family. The program's one proven data→family mechanism thus carries a substrate precondition, which scopes the conversion law honestly. (3) Per-seed goal gates swing on base's own draws: this seed's base took rites 0.1/warren 0.133/lockpick 0.1 and squeezed every treated arm to 6/10 — more evidence that per-seed sweep readings are rate measurements, never single-event claims. The clean lineage remains the mission's best-documented artifact: 2.7× base aggregate, fully receipted stages 1–7, zero contamination anywhere in its history.

## Knowledgebase Update

- Program evidence updated: pending results.
- Program backlog updated: pending results.
- Claim ledger updated: pending results.

## Artifacts

- `src/` — frozen vLLM runner (byte-identical to the source cell's).
- `scripts/` — staged harness, exposure pipeline, gate, benchmark runner, clean-chain
  rebuild script, vendored trainer/merger copies.
- `configs/` — frozen identity.
- `data/` — byte-copied corpora, exposure streams + receipts, gate files, clean-chain
  lineage package (`data/lineage/`).
- `runs/` — stage receipts (written by the staged GPU runs).
- `reports/artifact_manifest.yaml`
