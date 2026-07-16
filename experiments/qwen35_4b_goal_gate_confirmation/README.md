# Goal-Gate Confirmation

The mandatory replication of the program's recorded 10/10: base versus the hygiene_explore composite on three independent fresh sealed medium seeds, with an ordered confirmation verdict and the discovery seed reported but never counted.

**Status:** in-progress · since 2026-07-15 · model-free construction under way; the three-seed event has not run

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

The three-seed event has not run.

## Interpretation

None yet.

## Knowledgebase Update

- Program evidence updated: pending the event.
- Program backlog updated: this cell outranks everything until it closes.
- Claim ledger updated: no claim until the verdict.

## Artifacts

- `data/design_receipt.json`: seeds/tier/budget/models/gateway/discovery pins + the standalone lineage-package pins.
- `data/lineage/`: the six stage datasets and `lineage_manifest.json` (the complete fixed-seed recipe; produced shas are verification aids).
- `reports/preregistration.md`, `reports/benchmark_design_review.md`: contract and authorization.
