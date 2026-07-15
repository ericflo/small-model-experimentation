# Feedback-Loop + State-Chain Install

The two-tie install: teach the episode protocol (using rerun feedback; tracking narrated hidden state) that the public metadata names for menders and rites — the only two families separating the hygiene_explore parent from the recorded all-families goal gate at medium.

**Status:** finished · 2026-07-15 · verdict NOT_PROMOTED (split install) — u_statechain installed (11/20, strict over both controls) but u_feedloop failed completely (0/20, below both controls at 1/20; the third failed repair pedagogy); retention failed the replay band by 0.67 under pooled_k3; the axis total tied replay 11–11; seed 78,151 permanently sealed

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

Both arms trained clean (1,520 rows each, zero skips, fresh rank-32 adapters); the 12-run local gate executed with full boundary authentication:

| arm | axis total (40) | u_feedloop (20) | u_statechain (20) | retention pooled (104) | caps pooled | parsed pooled |
|---|---|---|---|---|---|---|
| hygiene_explore_parent | 8 | 1 | 7 | 62.33 | 10.67 | 93.33 |
| replay_ctl | 11 | 1 | 10 | **65.00** | 11.00 | 93.33 |
| feedloop_state | 11 | **0** | **11** | 59.33 | 13.00 | 90.67 |

- Promotion failed on three of the frozen bars: axis total tied replay (11–11, ties fail); u_feedloop lost to both controls; pooled retention fell 5.67 below the replay control (band ±5) while passing the parent band (−3.0).
- The event's own measured noise (delta SD 4.08 across the three screens) matched the calibration study's 4.27 — the pooled_k3 instrument performed exactly as designed on its first use.
- Seed 78,151 was never opened and is permanently sealed per the frozen contract.

## Interpretation

- The split is the finding. The state-chain protocol lesson INSTALLED (+4 over parent, +1 over the strong replay control, on fresh instances) — narrated hidden-state tracking is teachable at a 80-row dose, the rites-relevant skill. The feedback-loop lesson did NOT install even in-domain: 80 training rows on the same four formalisms yielded 0/20 on fresh instances — worse than untrained controls. Repair-with-feedback is now the THIRD failed pedagogy at the menders-shaped skill (after asserted single-turn repair and demonstrated bounded search, both killed by rule), and this one fails at the instrument that its own training surface defines.
- Replay continuation strengthened again: the control gained +2.67 pooled retention over the parent and reached 10/20 on statechain untrained — replay remains the strongest single broad move, and half of the candidate's apparent statechain edge rides the replay core.
- The retention cost (−3.0 vs parent) sits inside the revised 1–4-point tax law; the failure against the replay band is the dose competing against replay's own gain, which the calibrated bands are designed to catch.

## Knowledgebase Update

- Program evidence updated: the split-install verdict, the third repair-pedagogy failure, and the pooled_k3 instrument validation recorded.
- Program backlog updated: the statechain-only successor (drop the dead feedloop rows) is the natural next branch; menders needs a mechanism argument no small designed dose has survived.
- Claim ledger updated: no new claim.

## Artifacts

- `data/`: frozen corpus + manifests, exposure streams + receipt, four gate input pairs + design receipts.
- `reports/preregistration.md`: the full frozen contract.
