# Clean-Path Statechain Extension

**Status:** in-progress · since 2026-07-16 · model-free design frozen; staged GPU runs pending review verdicts

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

No model event has run. Fill after the staged runs; separate deployable evidence from
oracle/hidden evaluation.

## Interpretation

Pending.

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
