# Sweep-Rate Consolidation (Erratum: 2/6, not ~50%)

Terminal bookkeeping for the agentic-breadth program's headline sweep claim: the informal "~50% sweep rate" was computed over a favorable four-seed window (78154–78157); the full committed record holds SIX goal-gate readings of the hygiene_explore composite vs base at sealed medium/tb1024 seeds, and the honest all-events rate is **2/6** (exact Clopper-Pearson 95% CI [0.043, 0.777]; Beta(1,1) posterior mean 0.375, 95% CrI [0.099, 0.710]). Analysis-only: zero GPU, zero seeds, no model, `benchmarks/` never read.

**Status:** finished · 2026-07-16 · verdict CONSOLIDATED — over ALL six recorded goal-gate readings the sweep rate is 2/6 (exact 95% CI [0.04, 0.78]; Beta posterior mean 0.375), correcting the informal window figure (~50% over 2/4); menders blocked every miss at 0-margin; zero strict losses in sixty family comparisons; aggregate win 6/6; visible errata landed in all carrier documents

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the medium-tier measurement (78150, 8/10); the statechain dose's sealed sweep (78154, 10/10); the three-seed confirmation (78155–78157: 9/10, 8/10, 10/10 — AGGREGATE_ONLY); the zero-root rebuild's original reading (78159, 9/10).

## Question

What is the program's honest all-events goal-gate sweep rate, with calibrated uncertainty — and which documents carried the favorable-window figure that needs the erratum?

## Method

Byte-copy the six committed benchmark summaries into `data/source_summaries/` and hard-pin their sha256s; recompute every goal gate from `per_family` scores (strict wins/ties/losses vs base; FAMILIES byte-identical to the tier-forensics analyzer, enforced by test); cross-check against any goal-gate block the summary already records (78154's recorded block agrees; the others record none); then compute the rate with exact interval math (integer-shape beta quantiles via the binomial-tail identity, deterministic bisection, stdlib only — verified externally against scipy to 6 decimals). The harness re-derives both artifacts byte-identically.

## Results

| seed | source cell | strict wins | ties | losses |
|---|---|---|---|---|
| 78,150 | universal_medium_tier_measurement | 8/10 | menders, rites | none |
| 78,154 | statechain_only_dose | **10/10** | — | none |
| 78,155 | goal_gate_confirmation | 9/10 | menders (warren WON +0.267) | none |
| 78,156 | goal_gate_confirmation | 8/10 | menders, warren | none |
| 78,157 | goal_gate_confirmation | **10/10** | — | none |
| 78,159 | zero_root_lineage_rebuild | 9/10 | menders | none |

Sweep rate **2/6 = 0.333** (exact Clopper–Pearson 95% CI [0.043, 0.777]; Beta(1,1) posterior mean 0.375, 95% CrI [0.099, 0.710]). Menders blocks 4/4 misses, all 0-margin draws; rites and warren once each. Zero strict losses across all sixty family comparisons; aggregate strict win 6/6. Base draw distribution computed per family (base rites 0.0 on all six seeds — the intake's example corrected; chronicle drew >0 for base on exactly the two passing seeds).

## Interpretation

The terminal claim, on all the evidence: the reference model beats base on the aggregate always and on every family on one seed in three, with the entire distance between "one in three" and "always" being menders draws. The earlier "~50%" framing was a favorable window (2/4), later held as "a fifth data point" at 2/5 — corrected here with visible errata in every carrier document; no per-seed fact changed anywhere. The wide CI is the honest cost of six readings and is reported as such.

## Knowledgebase Update

- Program evidence updated: the corrected 2/6 rate with exact CI, the blocker table, and the erratum target list recorded in `runs/sweep_rate_analysis.json`.
- Program backlog updated: the documents on the erratum list need their figures amended to cite 2/6 (or to scope their window explicitly).
- Claim ledger updated: no new claim — this corrects the informal figure attached to existing entries.

## Artifacts

- `data/source_summaries/`: the six byte-identical summary copies (sha-pinned inputs).
- `runs/readings_table.json`: one verified row per seed — aggregates, per-family scores/deltas, wins/ties/losses, blockers, provenance sha, cross-check status.
- `runs/sweep_rate_analysis.json`: rate + exact CI + posterior, blocker table, base-draw distribution, and the ERRATUM block.
- `scripts/`: collect (fail-closed pins) + analyze (`--verify` byte-identity) + harness.
- `tests/`: 30 unittest cases — pinned expectations for all six readings, CI math at boundaries, provenance-drift negatives, FAMILIES byte-identity, erratum presence.
