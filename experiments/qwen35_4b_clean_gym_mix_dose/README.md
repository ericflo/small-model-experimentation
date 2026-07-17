# Clean Gym-Mix Dose

**Status:** finished · 2026-07-16 · verdict NOT_PROMOTED (mixture dilution) — the three-kind mix scored 15/40 on its own holdout, BELOW both the parent (17) and the replay control (19), with retention comfortably in-band: thin per-kind doses (50–60 rows) dilute below the proven 160-row single-kind concentration, the mirage kind ceilinged for untrained controls (replay 10/13), and the dose-diversity law is re-confirmed on clean ground; seed 78,161 permanently sealed

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can DESIGNED synthetic curricula install universal, transferable
  agentic skills into the one 4B — provably, on fully documented lineage?
- Prior anchors: lifecycle 22 (`qwen35_4b_zero_root_lineage_rebuild` — the zero-root
  composite, tree `414f5829…`, 0.3462 aggregate at sealed 78,159) and lifecycle 23
  (`qwen35_4b_clean_path_statechain_extension` — the statechain install replicates on
  the clean lineage but its rites CONVERSION does not: rites/sirens/mirage were
  precisely the retired C53 prefix's strengths, so the conversion law carries a
  substrate precondition).

## Question

Lifecycle 25 — owner-directed design. The zero-root composite's weak axes versus the
retired gym-era prefix (sealed 78,159 contrast) are exactly the prefix's strengths:
sirens (0.5 vs 0.7; 0.4-0.5 across seeds), rites (0.0-0.1), and mirage (0.5-0.6 vs
0.8). Can FRESH, documented, contamination-free gym-style content — designed from the
PUBLIC family descriptions only, never touching the prefix composite — recover those
margins on clean ground?

## Hypothesis

The prefix's family advantages came from trainable behaviors, not from its
undocumented substrate: a designed 160-row three-kind dose (goal fidelity under
embedded imperatives; hidden-state protocol compliance; calibrated abstention under
provable forcing/unforcing) at the standard exposure-matched geometry should install
on the axis holdout and move the sirens/rites/mirage readings toward the prefix's
margins at medium.

## Setup

- Model: Qwen/Qwen3.5-4B (revision `851bf6e8…`), always.
- Parent and adapter base: the zero-root composite
  (`large_artifacts/qwen35_4b_zero_root_lineage_rebuild/merged/zero_root_hygiene_explore`,
  tree `414f5829…`, weights `6e9aad25…`), authenticated against lifecycle 22's
  committed merge receipt (`e906caea…`; byte-identical provenance copy in
  `data/lineage/provenance/merge.json`).
- Treatment: `data/sft_gym_mix.jsonl` — 160 FRESH rows at construction seed 77,180
  (sha `62950116…`), all invented vocabulary, executable truth throughout:
  - 60 `u_siren_episode`: an invented multi-step book-retrieval transcript in ONE
    user message; 45 rows embed adversarial imperatives carrying a format-matched
    decoy (the reviewed u_hygiene mechanism in episode form; decoy ≠ truth always),
    15 clean; the think target narrates ignoring embedded orders as book content.
  - 50 `u_statechain`: FRESH instances from the byte-copied PROVEN lifecycle 18
    generator (brewvat/courierloft/peatstove/muletrack, 13/13/12/12), reviewed
    invariants intact.
  - 50 `u_mirage_abstain`: invented counter systems (3-5 entities, pairwise ties,
    domain 1-6) proven by exhaustive enumeration — 25 UNIQUELY FORCED (answer = the
    value) and 25 provably not forced (13 unsatisfiable + 12 undetermined; answer =
    the invented abstain token `NOWHERE`, chosen over the family's public
    `IMPOSSIBLE`, which is banned everywhere in this corpus). Generated in
    digit-only pairs: each forced/abstain pair shares one surface skeleton, and the
    two classes' alphabetic token sets are identical (audited).
  - Banned vocabulary: the proven statechain inventory EXTENDED with the
    sirens/mirage description nouns (injection, retrieval, document(s),
    directive(s), abstain/abstention, constraint(s), unsatisfiable, impossible),
    scanned case-insensitively; fresh-surface + row-overlap audits clear all 69
    pinned predecessor sources (corpora, streams, gates, lineage datasets).
- Arms: `replay_ctl5` (control, trains FIRST) then `gym_mix` (candidate); fresh
  rank-32/alpha-64 adapters, NO warm start, training seed 79, 1 epoch over 1,520
  rows (190 optimizer updates, LR 1e-5, batch 1×8, max length 4,096, w_think/w_close
  0.2).
- Exposure: exact zero-delta three-axis MILP (forward / nonzero-target / absolute
  loss mass ×5) at the frozen geometry — 1,280-row shared stratified core (replay
  `sft_blend.jsonl`, byte-identical `25a9595f…`) + 240-row variable block (control:
  240 replay; candidate: 160 treatment + 80 fillers), namespace seed 55,160. Matched
  exactly: 1,359,192 forward / 567,805 nonzero / 621,517 mass×5 per arm.
- Local gate (three arms: parent + both trained): 40-row axis holdout at seed 88,046
  (14 siren / 13 statechain / 13 mirage, FRESH instances) + three 104-row retention
  screens at 88,048/88,050/88,051 under pooled_k3 (the frozen triple 88,047/88,048/
  88,049 collided: 88,047 and 88,049 are the counterfactual cell's reflection/action
  seeds — next-free substitutions recorded; 88,043 was already taken there).
  Promotion: axis total strictly > parent AND > replay; AND at least TWO of the
  three kinds individually strict over BOTH controls (a tie fails a kind); pooled
  retention bands on screen sums (correct −15, caps +9, parsed −9) vs both controls.
- Conditional benchmark (only on promotion): ONE sealed medium tb1024 event at fresh
  seed 78,161, four arms in frozen order — base (`26d8ee48…`), zero_root_parent
  (`414f5829…`), replay_ctl5, gym_mix. Trained-arm pins are six fail-closed TODO-PIN
  slots in `scripts/run_benchmark.py`, frozen by check_design's NORMALIZED-HASH pin
  (lifecycle 22's mechanism).
- Primary metric: local axis-holdout total + kind breadth (promotion), then pilot
  gate (candidate aggregate strictly > base AND > replay_ctl5 AND >
  zero_root_parent).
- Frozen framing: menders remains closed, so the winnable ceiling is 9/10; the
  readings of consequence are the THREE AXIS READINGS — candidate vs parent on
  sirens, rites, and mirage specifically (does fresh gym-style content recover the
  retired prefix's margins on clean ground?) — and the per-family table, recorded
  either way. Any 10/10 is a menders draw and feeds a fresh confirmation cell.
- Contamination statement: the retired prefix composite is NEVER touched or
  referenced as a model input anywhere in this cell; only the public family
  descriptions (permitted metadata) informed the design.
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
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_clean_gym_mix_dose/scripts/run.py --smoke
```

Full (one stage per pushed checkpoint, each behind its review verdict):

```bash
.venv/bin/python -B experiments/qwen35_4b_clean_gym_mix_dose/scripts/run.py --stage train-control
# then: train-candidate, merge-arms, local, benchmark
```

Standalone lineage verification (no GPU) / full clean-chain rebuild (GPU):

```bash
.venv/bin/python -B experiments/qwen35_4b_clean_gym_mix_dose/scripts/rebuild_clean_chain.py --verify-inputs
```

## Results

Both arms trained clean (retention bands all passed — the dose was not destructive); the 12-run gate:

| arm | axis total (40) | siren_episode (14) | statechain (13) | mirage_abstain (13) | retention pooled |
|---|---|---|---|---|---|
| zero_root_parent | 17 | 4 | 5 | 8 | 62.67 |
| replay_ctl5 | **19** | 2 | 7 | **10** | 64.33 |
| gym_mix | 15 | 3 | 6 | 6 | 59.67 |

NOT promoted: the candidate lost the axis total to BOTH controls and won no kind. Seed 78,161 permanently sealed per contract.

## Interpretation

Three lessons. (1) MIXTURE DILUTION: 50–60 rows per kind installs nothing — the proven statechain effect needed its full 160-row concentration (its own cell won 21/40 with the same vehicle), and splitting the budget three ways landed below controls; this re-confirms the dose-diversity refutation on clean ground and hardens it into a design rule: one kind per dose at full concentration. (2) INSTRUMENT CEILING: the mirage-abstain kind was too easy untrained (replay 10/13) — a kind whose holdout the controls nearly ceiling cannot register installation; future abstention instruments need harder forced/abstain discrimination. (3) The siren-episode kind floors for everyone (2–4/14) — episode-form injection resistance likely needs its own concentrated cell to move at all. The path to the three families runs through three SEPARATE concentrated doses, not one mix.

## Knowledgebase Update

- Program evidence updated: pending results.
- Program backlog updated: pending results.
- Claim ledger updated: pending results.

## Artifacts

- `src/` — frozen vLLM runner (byte-identical to the lifecycle 23 cell's).
- `scripts/` — staged harness, generators (gym-mix + byte-copied proven statechain
  + canonical retention), corpus builder with audits, exposure pipeline, gate,
  benchmark runner, clean-chain rebuild script, vendored trainer/merger copies.
- `configs/` — frozen identity.
- `data/` — treatment corpus + manifest, replay copy, exposure streams + receipts,
  gate files, clean-chain lineage package (`data/lineage/`).
- `runs/` — stage receipts (written by the staged GPU runs).
- `reports/artifact_manifest.yaml`
