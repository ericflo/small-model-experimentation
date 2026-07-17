# Enumerative Repair Protocol

**Status:** finished · 2026-07-16 · verdict PILOT_NOT_PROMOTED + FAILED_ON_ITS_OWN_TERMS — the enumeration discipline INSTALLED (9/40 canonical-next vs both controls at 0/40, the program's starkest mechanism contrast) but at 22.5% fidelity (below the frozen 0.50 precondition) and did NOT convert (candidate menders 0.0 while the replay control drew 0.1; aggregate 0.3252 lost to both controls at 0.3502); per the preregistered ordered rule the pure-enumeration SFT route closes at this dose; seed 78,162 spent

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can DESIGNED synthetic curricula install universal, transferable
  agentic skills into the one 4B — provably, on fully documented lineage?
- Prior anchors: lifecycle 22 (`qwen35_4b_zero_root_lineage_rebuild` — the zero-root
  composite, tree `414f5829…`), lifecycle 24 (`qwen35_4b_menders_dose_scale` — the 10×
  dose closed the dose-scale mechanism class for the eliminative-inference lesson), and
  lifecycle 25 (`qwen35_4b_clean_gym_mix_dose` — mixture dilution hardened the design
  rule: ONE KIND PER DOSE AT FULL CONCENTRATION).

## Question

Lifecycle 26 — the new-mechanism attack on the last goal-gating family (menders).
Every failed menders dose taught the model to INFER the right fix (eliminative
inference — closed at every dose 80–800; even 2AFC verification sat at chance). Can a
dose that teaches SYSTEMATIC ENUMERATION instead — given failure evidence, propose the
legal single-step candidates one per turn in a frozen canonical order, let trial
feedback decide, stop at first success — install on the axis holdout and convert to
the menders family, which is a bounded multi-turn episode WITH rerun feedback?

## Hypothesis

The repair kill rules bind the INFERENCE mechanism, not this one. C34 says brute-force
search dominates the model's reasoning (a model-level law), and protocols are the
line's only reliably installable class (hygiene/explore/termination/statechain). An
enumerator converts turn budget into coverage without needing the walled inference
skill; a taught frozen-order enumeration protocol should therefore install where
taught inference did not, and any candidate menders > 0 where the controls sit at 0 is
the mechanism answer.

## Setup

- Model: Qwen/Qwen3.5-4B (revision `851bf6e8…`), always.
- Parent and adapter base: the zero-root composite
  (`large_artifacts/qwen35_4b_zero_root_lineage_rebuild/merged/zero_root_hygiene_explore`,
  tree `414f5829…`, weights `6e9aad25…`), authenticated against lifecycle 22's
  committed merge receipt (`e906caea…`; byte-identical provenance copy in
  `data/lineage/provenance/merge.json`).
- Treatment: `data/sft_enum_repair.jsonl` — 160 rows, ONE KIND `u_enum_repair`, at
  construction seed 77,190 (sha `c9b539bf…`), 20 rows per formalism across all eight
  machine formalisms REUSED from the menders dose-scale cell via a byte-identical
  machinery copy (`scripts/gen_feedloop_curriculum.py` — imported, never forked).
  Each row is one PARTIAL enumeration episode in ONE user message: the machine spec
  with legality clauses PLUS a numbered action list rendering the full bounded grammar
  in its frozen order; the broken written sequence; both trials' wanted+observed
  failure evidence; a frozen canonical-order statement (byte-identical in every row:
  step number ascending, then action-list position); the first `k` canonical
  candidates already tried, each with its observed two-trial outcome (all failures by
  construction; `k` cycles over 0/1/3/6/10); and the ask: name the NEXT untried legal
  candidate. The think target narrates enumerate → cross off → emit next. The answer
  is exact-match `STEP <k>: <corrected step>`. The generator verifies per row, by
  exhaustive re-derivation over the full candidate space, that the target IS the
  canonical-next untried legal candidate, that exactly ONE candidate repairs both
  trials, and that every tried entry is legal, canonically ordered, and genuinely
  failing (each re-simulated against both trials).
- Surfaces: ALL inherited from the menders cell by design (no fresh-surface claim);
  freshness is at the ROW level — zero canonical-user-message overlap against 76
  pinned predecessor corpora, streams, and gates (including the formalism-sharing
  menders corpus/holdout), plus fresh instances by construction.
- Arms: `replay_ctl6` (control, trains FIRST) then `enum_repair` (candidate); fresh
  rank-32/alpha-64 adapters, NO warm start, training seed 83, standard recipe (1 epoch
  over 1,520 rows, 190 optimizer updates, LR 1e-5, batch 1×8, max length 4,096,
  w_think/w_close 0.2).
- Exposure: exact zero-delta three-axis MILP (forward / nonzero-target / absolute loss
  mass ×5) at the frozen geometry — 1,280-row shared stratified core + 240-row
  variable block (control: 240 replay; candidate: 160 treatment + 80 fillers),
  namespace seed 55,170. Matched exactly: 1,436,178 forward / 572,724 nonzero /
  629,552 mass×5 per arm; zero skips; 1,280 aligned core rows.
- Local gate (three arms: parent + both trained): 40-row axis holdout at seed 88,052
  (all `u_enum_repair`, 5 per formalism, one per k-value; same invariants as the
  treatment) + three 104-row retention screens at 88,053/88,054/88,055 under
  pooled_k3. All seeds verified grep-fresh in seed contexts; no substitution was
  needed (88,043/88,047/88,049 remain taken by the counterfactual cell and are
  avoided). Promotion: axis total strictly > parent AND > replay_ctl6 (ties fail) —
  single kind, so NO per-kind split exists; pooled retention bands on screen sums
  (correct −15, caps +9, parsed −9) vs both controls.
- Preregistered NON-GATING mechanism readings: (a) `episode_success_simulation` — the
  analytic perfect-enumerator turn count per holdout episode (recorded in the local
  design receipt, model-free). THE NUMBERS, quoted up front because they scope the
  transfer bet: holdout from-scratch MEAN 30.4 turns (median 18.5, max 122; 87.5% of
  episodes need MORE than 10 turns); treatment corpus mean 31.4 (median 23, max 125;
  88.1% > 10) — against a family episode budget publicly known only as "bounded".
  Stated plainly: if the family's budget is materially shorter than these needs, a
  perfectly-installed enumerator converts few or no episodes. (b)
  `enumeration_fidelity` — per axis row at eval time, three booleans about the model's
  proposal: LEGAL, UNTRIED, CANONICAL-NEXT — a mechanism decomposition beyond raw
  correctness, summarized per arm; it feeds the frozen zero-draw scoping below.
- Conditional benchmark (only on promotion): ONE sealed medium tb1024 event at fresh
  seed 78,162, four arms in frozen order — base (`26d8ee48…`/`b654e033…`),
  zero_root_parent (`414f5829…`), replay_ctl6, enum_repair. Trained-arm pins are six
  fail-closed TODO-PIN slots in `scripts/run_benchmark.py`, frozen by check_design's
  NORMALIZED-HASH pin. Pilot gate: candidate aggregate strictly > base AND >
  replay_ctl6 AND > zero_root_parent. Recorded either way: the goal gate, the
  per-family table, and THE MENDERS READING — candidate vs base and vs parent on
  menders specifically (frozen question: does taught enumeration convert to the family
  with live rerun feedback?). FROZEN ORDERED CONSEQUENCES, positive first, no third
  state for the zero draw: (1) ANY candidate menders > 0 where the controls sit at 0
  is the mechanism answer; (2) a menders 0 WITH the fidelity precondition met
  (promoted locally AND holdout canonical-next rate F >= 0.50 AND F strictly above
  both controls' rates) is TURN_BUDGET_SCOPED — enumeration installed with high
  fidelity but did not convert within the family's episode budget; the
  protocol-install mechanism is NOT refuted; what closes is the pure-enumeration route
  at the family's actual budget; (3) a menders 0 WITHOUT that precondition reads as
  the install/conversion failing on its own terms. A 10/10 feeds a fresh confirmation
  cell. (The scoping and quoted simulation numbers were added pre-freeze by review
  amendment; no seed had been consumed.)
- Standalone: `data/lineage/` carries the complete clean-chain package — the six
  zero-root stage datasets, lifecycle 22's stage + merge receipts as provenance
  documents, the trainer/merger copies, and a clean-chain manifest recording this
  cell's dose as STAGE 7. NO blend root exists anywhere in this cell (fail-closed).
- Hidden-label boundary: gate answers and per-row audits live only in
  `data/local_tasks_seed*.jsonl`; the model-facing `local_input_seed*.jsonl` files
  carry id/messages/meta only. The benchmark suite directory is never read; only the
  trusted aggregate gateway runs.

## Run

Smoke (no GPU, no writes):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_enumerative_repair_protocol/scripts/run.py --smoke
```

Full (one stage per pushed checkpoint, each behind its review verdict):

```bash
.venv/bin/python -B experiments/qwen35_4b_enumerative_repair_protocol/scripts/run.py --stage train-control
# then: train-candidate, merge-arms, local, benchmark
```

Standalone lineage verification (no GPU) / full clean-chain rebuild (GPU):

```bash
.venv/bin/python -B experiments/qwen35_4b_enumerative_repair_protocol/scripts/rebuild_clean_chain.py --verify-inputs
```

## Results

Local gate: PROMOTED — the starkest mechanism contrast recorded: enum_repair 9/40 canonical-next on fresh instances versus BOTH controls at exactly 0/40 (fidelity cascade: parseable 19/40, legal 18, untried 16, canonical-next 9 — the bottleneck is answer formatting on long prompts; ordering discipline 9/16 = 56% once legal-untried); retention 57.67 pooled vs 59.33/59.67, deep in-band.

Sealed event at 78,162 (all arms authenticated; the normalized pin held through the fill):

| arm | aggregate | menders | goal gate |
|---|---|---|---|
| base | 0.0882 | 0.000 | — |
| zero_root_parent | 0.3502 | 0.000 | 8/10 (ties menders, rites) |
| replay_ctl6 | 0.3502 | **0.100** | 8/10 (ties rites, sirens) |
| enum_repair | 0.3252 | 0.000 | 7/10 (ties menders, sirens; loses rites) |

Pilot: candidate > base only — NOT promoted. The frozen menders rule: candidate_nonzero false, controls_all_zero false (the replay control drew an item), fidelity 22.5% < the 0.50 precondition → **FAILED_ON_ITS_OWN_TERMS**: the pure-enumeration SFT route closes at this dose, on its own preregistered terms.

## Interpretation

Three readings. (1) The INSTALL is genuine and unprecedented in contrast: untrained models score literal zero at canonical-next enumeration and the dose lifted it to 9/40 — protocols remain the installable class (now 5-for-5 on installs). (2) The CONVERSION failed on its own terms: 22.5% local fidelity was too low to earn the budget-scoped reading, the family drew 0 for the candidate while the replay control (which trains nothing) drew an item — re-confirming that menders movement at this granularity remains draw-dominated for everything except a genuinely reliable installed skill, which this dose did not reach. (3) The identified bottleneck is upstream of the discipline: half the holdout rows never parsed into the answer format (long-prompt formatting), and where a legal untried candidate emerged the ordering was right 56% of the time. A formatting-targeted variant is a marginal iteration on a mechanism that just failed its preregistered terms — per calibrate-and-diverge it needs new evidence before funding, and the frozen consequence stands.

## Knowledgebase Update

- Program evidence updated: pending results.
- Program backlog updated: pending results.
- Claim ledger updated: pending results.

## Artifacts

- `src/` — frozen vLLM runner (byte-identical to the lifecycle 25 cell's).
- `scripts/` — staged harness, the new enum-repair generator + the byte-copied menders
  machinery + canonical retention generator, corpus builder with audits, exposure
  pipeline, gate, benchmark runner with the menders reading, clean-chain rebuild
  script, vendored trainer/merger copies.
- `configs/` — frozen identity.
- `data/` — treatment corpus + manifest, replay copy, exposure streams + receipts,
  gate files, design receipts, clean-chain lineage package (`data/lineage/`).
- `runs/` — stage receipts (written by the staged GPU runs).
- `reports/artifact_manifest.yaml`
