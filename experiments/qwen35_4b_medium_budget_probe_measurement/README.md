# Medium Budget-Probe Measurement

The last cheap lever on the binding constraint: the same four published composites as the seed-78,150 medium event, re-measured once at think budget 8,192 — does thinking room alone move menders (zero for every arm at tb1024) or rites (zero for all but designed_fresh) off their floors, i.e. is the goal-gate ceiling 9 or 10?

**Status:** finished · 2026-07-15 · verdict BUDGET_GATE_STOP — base failed the gateway's hard wall-budget gate at medium/tb8192 before any treated arm ran (safe diagnostic `budget_gate_failed`, no score emitted); the 8× thinking-budget lever is infeasible for paired events at this tier; seed 78,152 spent by the opened ledger record per the preregistered stop

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the two-tie install (repair kill rule extended to all tested SFT pedagogies — training paths cap at 9/10); the medium measurement (hygiene_explore at 8/10, ties at menders/rites); C44 (serial-compute limit; always give the 4B chain-of-thought); the budgets-maxed directive (caps throttle measured capability); all arms within wall budget at tb1024 (136–230 s).

## Question

At medium/tb8192 on a fresh sealed seed, does menders (universally zero at tb1024) or rites (zero for three of four arms) move off its floor — and does the goal gate's reachable ceiling change?

## Setup

- Arms (identical pins to the 78,150 event): `base`, `designed_fresh`, `replay_repeat`, `hygiene_explore`.
- Event: medium, think budget 8,192, sealed seed 78,152, hardened seed-boundary runner, one-seed write-ahead ledger.
- Readings (no promotion): budget movement (fires only for arm/family pairs at 0 in the pinned tb1024 event that turn positive; designed_fresh's already-nonzero rites reported descriptively); budget contrast vs the pinned 78,150 summary (implementation-signature verified equal fail-closed; seed+budget confounds labeled); recorded goal gate; within_budget integrity per arm.

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_medium_budget_probe_measurement/scripts/run.py --smoke
.venv/bin/python -B experiments/qwen35_4b_medium_budget_probe_measurement/scripts/run.py --stage benchmark
```

## Results

The preregistered stop outcome fired on the first arm: the gateway refused `base` at medium/tb8192 with safe diagnostic `budget_gate_failed` (exit 2, no score emitted, no raw output exposed). Per the frozen order (base first, precisely to minimize spend on this known risk) zero treated arms ran; the write-ahead ledger's opened record marks seed 78,152 spent; the failure receipt is preserved at `runs/benchmark/medium_tb8192_seed78152_measurement/base.failure.json`. Per the frozen contract there is no retry and no lower-budget re-run inside this directory.

## Interpretation

- The full 8× thinking-budget lever is closed at medium: the gateway's per-arm wall budget binds well below tb8192 for base (which passed comfortably at tb1024, 157 s). The budget-movement question (does serial-compute room move menders) remains open only at intermediate budgets.
- Pricing an intermediate successor honestly: base's tb1024 wall was 157 s and hygiene_explore's 230 s — the slowest arm, not base, may bind at higher budgets, so any successor must expect either to trip. One preregistered intermediate probe (tb4096 or tb2048, fresh seed, same stop contract) is the last believable shot at the lever; a second stop would close the budget lever entirely and fix the statechain successor's 9/10 ceiling as the program's honest position.

## Knowledgebase Update

- Program evidence updated: the stop, its minimal cost, and the lever's remaining scope recorded.
- Program backlog updated: one intermediate-budget probe queued as the lever's last believable test; the statechain-only dose remains the funded training branch.
- Claim ledger updated: no.

## Artifacts

- `data/design_receipt.json`: seed/tier/budget/model/gateway/contrast-source pins.
- `reports/preregistration.md`, `reports/benchmark_design_review.md`: contract and authorization.
