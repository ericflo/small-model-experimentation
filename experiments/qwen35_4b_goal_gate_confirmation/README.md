# Goal-Gate Confirmation

The mandatory replication of the program's recorded 10/10: base versus the hygiene_explore composite on three independent fresh sealed medium seeds, with an ordered confirmation verdict and the discovery seed reported but never counted.

**Status:** finished · 2026-07-15 · verdict AGGREGATE_ONLY — the aggregate win replicated on ALL THREE fresh seeds (0.329–0.384 vs base 0.059–0.112; 4/4 all-time) and seed 78,157 swept the goal gate 10/10 (two full sweeps across four independent sealed seeds), but the 2/3 majority bar failed: 78,155 read 9/10 and 78,156 read 8/10 with ZERO losses, blocked purely by menders ties (0-margin both) and one warren tie — menders is definitively the single binding family

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the statechain dose's sealed event (hygiene_explore_parent goal_gate_pass TRUE at 78,154 — aggregate 0.3663 vs 0.0800, all ten families strictly above, menders 0.017 and warren 0.050 — 0.150 vs 0.100 — on single-item margins); the confirmation law; the tier forensics (menders was draw-dependent, never a wall).

## Question

Does the all-families pass replicate? CONFIRMED requires the aggregate to win on all three fresh seeds and the 10/10 gate on at least two.

## Setup

- Arms: `base` (b654e033…/26d8ee48…) and `hygiene_explore` (9eb653d7…/e2112344…), full trees recomputed once per runner invocation, before any gateway call (wording matches the code: authentication is per-invocation, not per-seed; closed seeds' receipts stay sha-pinned in the ledger).
- Event: three sealed seeds 78,155/78,156/78,157, medium, tb 1,024, per-seed write-ahead ledger whose closed records sha-pin the sealed summary AND both per-arm gateway receipts (the readout refuses anything the ledger did not pin), implementation signature anchored to the discovery event.
- Recovery: `--resume` is the single recovery path. A crashed (opened) seed reuses its preserved receipts; a crash between a seed's summary write and its closed-record append is recovered by deterministic byte-identical regeneration of the summary (divergence refuses loudly with both digests); an unopened seed refuses to run over pre-existing event files (clean slate).
- Verdict: CONFIRMED / AGGREGATE_ONLY / NOT_REPLICATED (ordered, total); fragility margins reported per seed; no promotion logic anywhere.
- Standalone lineage (owner directive 2026-07-15): the hygiene_explore composite's complete reproduction package lives in this cell — `data/lineage/` (six ordered SFT dataset copies + fixed-seed recipe manifest), `scripts/lineage_trainers/` + `scripts/merge_adapter.py` (byte-identical trainer/merger copies), the frozen root adapter vendored at `large_artifacts/qwen35_4b_goal_gate_confirmation/lineage_root/blend` (hard provenance boundary: no committed creation receipt), and `scripts/rebuild_lineage.py` (GPU replay + sha verification; `--verify-inputs` runs in smoke).

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_goal_gate_confirmation/scripts/run.py --smoke
.venv/bin/python -B experiments/qwen35_4b_goal_gate_confirmation/scripts/run.py --stage benchmark
```

## Results

All six runs clean (both arms authenticated per invocation, within budget, per-seed ledger opened/closed, readout provenance-anchored):

| seed | base | hygiene_explore | goal gate | blockers |
|---|---|---|---|---|
| 78,155 | 0.0586 | 0.3287 | 9/10 | menders tie (0.0 margin); warren WON +0.267 |
| 78,156 | 0.1122 | 0.3737 | 8/10 | menders + warren ties; zero losses |
| 78,157 | 0.0982 | **0.3837** | **PASS 10/10** | — |
| (78,154 discovery, not counted) | 0.0800 | 0.3663 | PASS 10/10 | — |

Verdict per the frozen ordered partition: **AGGREGATE_ONLY** — aggregate strict wins 3/3 (plus the discovery, 4/4 all-time), goal-gate passes 1/3 against the required 2/3.

## Interpretation

The replication sharpens rather than overturns the discovery. The aggregate transfer is unconditional — never close on any seed. The all-families sweep is real but draw-gated at exactly one family: every non-passing seed carried ZERO losses, and menders blocked with a 0.0 margin (both arms at zero) on both failing seeds while warren, the other discovery-day fragility, WON by +0.267 on one seed and tied once. Two full 10/10 sweeps in four independent sealed seeds is a model within one item-draw of the goal on every roll — and a preregistered bar honestly not met. The program position: the goal's primary condition is demonstrated but not confirmed at the frozen majority bar; menders is the single binding family, and the queued dose-scale intake (the one mechanism class the kill rules permit) is now aimed at a precisely-known target: any reliable nonzero menders yield completes the gate.

## Knowledgebase Update

- Program evidence updated: the AGGREGATE_ONLY verdict, the second sweep, and the menders 0-margin localization recorded.
- Program backlog updated: the menders dose-scale intake is the funded successor; the zero-root rebuild stays queued.
- Claim ledger updated: no confirmed claim; the demonstrated-not-confirmed position stated exactly.

## Artifacts

- `data/design_receipt.json`: seeds/tier/budget/models/gateway/discovery pins + the standalone lineage-package pins.
- `data/lineage/`: the six stage datasets and `lineage_manifest.json` (the complete fixed-seed recipe; produced shas are verification aids).
- `reports/preregistration.md`, `reports/benchmark_design_review.md`: contract and authorization.
