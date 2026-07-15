# Feedback-Loop + State-Chain Install

The two-tie install: teach the episode protocol (using rerun feedback; tracking narrated hidden state) that the public metadata names for menders and rites — the only two families separating the hygiene_explore parent from the recorded all-families goal gate at medium.

**Status:** in-progress · since 2026-07-15 · model-free construction frozen; training, local gate, and the conditional medium event remain

## Research Program

- Program: `agentic_breadth_installation`.
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the medium measurement (parent at 8/10 strict wins, zero losses, ties only at menders/rites); the tier forensics (goal-gate venue is medium); the calibration cell (pooled_k3 retention protocol, first use here); the trace-repair kill rule (this dose's mechanism argument is the episode PROTOCOL, not repair content).

## Question

Does a single-dose episode-protocol curriculum — 80 act→observe→revise feedback-loop rows plus 80 narrated hidden-state chain rows on six invented formalisms — install the two missing multi-turn skills without breaking the parent's eight strict wins?

## Setup

- Parent and adapter base: the `hygiene_explore` composite (tree 9eb653d7…), fresh rank-32/alpha-64 adapters, no warm start.
- Corpus: `data/sft_feedloop_state.jsonl` (e6d45ed4…), 160 rows, seed 77,130, generator-verified invariants (≥2 LEGAL candidates after evidence round 1, exactly 1 after round 2, extended-grammar exclusion audit per the pre-freeze review amendment; ≥3 hidden-state updates with stateless/last-step distractors verified wrong), banned-vocab + 56-token fresh-surface audits clean.
- Exposure: exact zero-delta MILP vs `replay_ctl` (1,393,242 forward tokens, 584,414 targets, 640,286 mass×5 per arm).
- Local gate: axis holdout 88,026 (20+20) with strict per-kind bars; retention pooled over screens 88,027/88,028/88,030 under pooled_k3 bands.
- Conditional benchmark: medium, tb1024, sealed seed 78,151, four models, hardened runner; goal gate recorded either way.

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_feedback_loop_state_chain_install/scripts/run.py --smoke
# staged: --stage train-control | train-candidate | merge-arms | local | benchmark
```

## Results

No model event has run.

## Interpretation

None yet.

## Knowledgebase Update

- Program evidence updated: pending events.
- Program backlog updated: this cell is the medium measurement's funded successor.
- Claim ledger updated: no.

## Artifacts

- `data/`: frozen corpus + manifests, exposure streams + receipt, four gate input pairs + design receipts.
- `reports/preregistration.md`: the full frozen contract.
