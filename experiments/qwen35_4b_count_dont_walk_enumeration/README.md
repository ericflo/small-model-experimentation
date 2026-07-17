# Count-Don't-Walk Enumeration

**Status:** in-progress · since 2026-07-16 · model-free construction frozen (corpus, exposure streams, gate, receipts, clean-chain package); GPU stages pending review

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can DESIGNED synthetic curricula install universal, transferable
  agentic skills into the one 4B — provably, on fully documented lineage?
- Prior anchors: lifecycle 22 (`qwen35_4b_zero_root_lineage_rebuild` — the zero-root
  composite, tree `414f5829…`), lifecycle 25 (`qwen35_4b_clean_gym_mix_dose` — ONE KIND
  PER DOSE AT FULL CONCENTRATION), and lifecycle 26
  (`qwen35_4b_enumerative_repair_protocol` — the reference cell: the enumeration
  discipline INSTALLED, 9/40 canonical-next vs both controls at 0/40, but failed on its
  own terms at 22.5% fidelity).

## Question

Lifecycle 27 — the evidence-backed successor to the enumerative-repair cell, changing
ONLY the expression pedagogy. The reference cell's committed truncation forensics
(`experiments/qwen35_4b_enumerative_repair_protocol/analysis/truncation_forensics.md`)
showed its 22.5% fidelity UNDERCOUNTS the installed discipline: 20 of 21 unparseable
gate rows were 1,024-token cap truncations caught mid-CORRECT walk — the model
faithfully executed the frozen canonical walk and ran out of budget, a token cost that
grows with the tried-list depth k. Does teaching the SAME discipline with a compact,
constant-cost expression — COUNT, DON'T WALK — lift the canonical-next fidelity past
the frozen 0.50 precondition and (conditionally) convert to the menders family?

## Hypothesis

The discipline is already installable (proven by the reference cell's 9/40 vs 0/40).
The bottleneck is expression cost. The tried list has k entries in canonical order, so
the target is entry k+1 of the frozen order; with the per-step candidate ranges
RENDERED IN THE PROMPT, locating entry k+1 is pure index arithmetic (range lookup +
slot offset), constant in k. Training a fixed-shape five-line computation under a
frozen think length budget (<= 120 real tokens; measured max 105) should express the
installed discipline inside any bounded budget where the walker truncated.

## Setup

- Model: Qwen/Qwen3.5-4B (revision `851bf6e8…`), always.
- Parent and adapter base: the zero-root composite
  (`large_artifacts/qwen35_4b_zero_root_lineage_rebuild/merged/zero_root_hygiene_explore`,
  tree `414f5829…`, weights `6e9aad25…`), authenticated against lifecycle 22's
  committed merge receipt (`e906caea…`; byte-identical provenance copy in
  `data/lineage/provenance/merge.json`).
- Treatment: `data/sft_count_walk.jsonl` — 160 rows, ONE KIND `u_count_walk`, at
  construction seed 77,191 (sha `21e6f5cb…`), 20 rows per formalism across all eight
  machine formalisms REUSED from the menders dose-scale cell via a byte-identical
  machinery copy (`scripts/gen_feedloop_curriculum.py` — imported, never forked).
  The task shape is byte-equivalent to the reference cell's (partial enumeration
  episode; frozen canonical-order statement, byte-identical rule text; verified tried
  prefix, k over 0/1/3/6/10; unique both-trials fix; exact-match
  `STEP <k>: <corrected step>`). The ONLY designed deltas:
  1. THINK TARGETS teach COUNT-DON'T-WALK: a fixed-shape compact computation,
     identical five-line structure in every row — count the tried entries → k; the
     target is change number k+1; locate k+1 in the rendered ranges (explicit offset
     subtraction); resolve the offset to the action-list slot (skipping the written
     action); emit `STEP <n>: <action>`. Constant token cost in k, under a frozen
     THINK LENGTH BUDGET (five-line shape + caps in the generator, per row; the REAL
     tokenizer bound <= 120 tokens fail-closed in `measure_source_tokens.py`;
     measured max 105 / mean 95.8 real tokens).
  2. THE ORDER STATEMENT gains the rendered per-step candidate counts ("step 1 offers
     17 changes (numbers 1-17); step 2 offers 17 (numbers 18-34); …"), byte-identical
     rule text otherwise; per-step counts computed generically and the rendered
     ranges verified against the generator's own exhaustive enumeration exactly.
  3. NEW GATE READING (recorded, non-gating): `expression_cost` — per-arm
     think-token-length distribution on the axis rows + truncation count, the
     expression-cost reading this lineage owes.
- Surfaces: ALL inherited (no fresh-surface claim); freshness is at the ROW level —
  zero canonical-user-message overlap against 83 pinned predecessor corpora, streams,
  and gates, INCLUDING the reference cell's corpus/streams/gates (same formalisms,
  same episode shape — row-level freshness is the bar).
- Arms: `replay_ctl7` (control, trains FIRST) then `count_walk` (candidate); fresh
  rank-32/alpha-64 adapters, NO warm start, training seed 85 (84 is taken — next-free
  substitution recorded), standard recipe (1 epoch over 1,520 rows, 190 optimizer
  updates, LR 1e-5, batch 1×8, max length 4,096, w_think/w_close 0.2).
- Exposure: exact zero-delta three-axis MILP (forward / nonzero-target / absolute loss
  mass ×5) at the frozen geometry — 1,280-row shared stratified core + 240-row
  variable block (control: 240 replay; candidate: 160 treatment + 80 fillers),
  namespace seed 55,171. Matched exactly: 1,438,010 forward / 564,379 nonzero /
  621,239 mass×5 per arm; zero skips; 1,280 aligned core rows.
- Local gate (three arms: parent + both trained): 40-row axis holdout at seed 88,056
  (all `u_count_walk`, 5 per formalism, one per k-value; same invariants as the
  treatment) + three 104-row retention screens at 88,057/88,058/88,059 under
  pooled_k3. All seeds verified grep-fresh in seed contexts (known-taken:
  88,043/88,047/88,049 and everything <= 88,055). Promotion: axis total strictly >
  parent AND > replay_ctl7 (ties fail) — single kind, so NO per-kind split exists;
  pooled retention bands on screen sums (correct −15, caps +9, parsed −9) vs both
  controls.
- Preregistered NON-GATING mechanism readings: (a) `episode_success_simulation` —
  holdout from-scratch MEAN 27.1 turns (median 20.5, max 78; 80.0% of episodes need
  MORE than 10 turns); treatment corpus mean 32.6 (median 22, max 125; 86.9% > 10) —
  against a family episode budget publicly known only as "bounded". (b)
  `enumeration_fidelity` — per axis row: LEGAL, UNTRIED, CANONICAL-NEXT booleans; it
  feeds the frozen zero-draw scoping below. (c) `expression_cost` (NEW) — per-arm
  think-token-length distribution + truncation count on the axis rows.
- Conditional benchmark (only on promotion): ONE sealed medium tb1024 event at fresh
  seed 78,163, four arms in frozen order — base (`26d8ee48…`/`b654e033…`),
  zero_root_parent (`414f5829…`), replay_ctl7, count_walk. Trained-arm pins are six
  fail-closed TODO-PIN slots in `scripts/run_benchmark.py`, frozen by check_design's
  NORMALIZED-HASH pin. Pilot gate: candidate aggregate strictly > base AND >
  replay_ctl7 AND > zero_root_parent. Recorded either way: the goal gate, the
  per-family table, and THE MENDERS READING. FROZEN ORDERED CONSEQUENCES, positive
  first, no third state for the zero draw: (1) ANY candidate menders > 0 where the
  controls sit at 0 is the mechanism answer; (2) a menders 0 WITH the fidelity
  precondition met (promoted locally AND holdout canonical-next rate F >= 0.50 AND F
  strictly above both controls' rates) is TURN_BUDGET_SCOPED — the protocol-install
  mechanism is NOT refuted; what closes is the pure-enumeration route at the family's
  actual budget; (3) a menders 0 WITHOUT that precondition fails on its own terms.
- Standalone: `data/lineage/` carries the complete clean-chain package — the six
  zero-root stage datasets, lifecycle 22's stage + merge receipts as provenance
  documents, the trainer/merger copies, and a clean-chain manifest recording this
  cell's dose as STAGE 7 (`count_walk`, seed 85). NO blend root exists anywhere in
  this cell (fail-closed).
- Hidden-label boundary: gate answers and per-row audits live only in
  `data/local_tasks_seed*.jsonl`; the model-facing `local_input_seed*.jsonl` files
  carry id/messages/meta only. The benchmark suite directory is never read; only the
  trusted aggregate gateway runs.

## Run

Smoke (no GPU, no writes):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_count_dont_walk_enumeration/scripts/run.py --smoke
```

Full (one stage per pushed checkpoint, each behind its review verdict):

```bash
.venv/bin/python -B experiments/qwen35_4b_count_dont_walk_enumeration/scripts/run.py --stage train-control
# then: train-candidate, merge-arms, local, benchmark
```

Standalone lineage verification (no GPU) / full clean-chain rebuild (GPU):

```bash
.venv/bin/python -B experiments/qwen35_4b_count_dont_walk_enumeration/scripts/rebuild_clean_chain.py --verify-inputs
```

## Results

Pending: the model-free construction is frozen; GPU stages run behind their review
verdicts.

## Interpretation

Pending results. Separate deployable evidence from oracle/hidden evaluation.

## Knowledgebase Update

- Program evidence updated: pending results.
- Program backlog updated: pending results.
- Claim ledger updated: pending results.

## Artifacts

- `src/` — frozen vLLM runner (byte-identical to the reference cell's).
- `scripts/` — staged harness, the count-walk generator + the byte-copied menders
  machinery + canonical retention generator, corpus builder with audits, exposure
  pipeline (with the real-tokenizer think-budget certification), gate with the
  expression-cost reading, benchmark runner with the menders reading, clean-chain
  rebuild script, vendored trainer/merger copies.
- `configs/` — frozen identity.
- `data/` — treatment corpus + manifest, replay copy, exposure streams + receipts,
  gate files, design receipts, clean-chain lineage package (`data/lineage/`).
- `runs/` — stage receipts (written by the staged GPU runs).
- `reports/artifact_manifest.yaml`
