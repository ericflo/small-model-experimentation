# Statechain-Only Dose

Dose ONLY the proven skill: lifecycle 15 split its verdict — u_statechain INSTALLED (11/20, strict over both controls) while u_feedloop died at 0/20 and dragged retention below the replay band. This cell re-runs the install with a 160-row statechain-only corpus (no dead feedloop rows) and asks whether the clean dose clears the calibrated gate the mixed dose failed.

**Status:** in-progress · since 2026-07-15 · pipeline + gate frozen; awaiting compute review before train-control

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can synthetic curricula install general capability that lifts the held-out aggregate without a negative family?
- Prior anchors: the reference cell `qwen35_4b_feedback_loop_state_chain_install` (lifecycle 15, NOT_PROMOTED split verdict: statechain installed, feedloop dead, retention failed the replay band by 0.67 under pooled_k3); the medium measurement (parent at 8/10 strict wins, ties only at menders/rites); the pooled_k3 calibration cell; menders closed for every believable arm (three pedagogies + the budget lever).

## Question

Does a 160-row statechain-only dose — brewvat and courierloft reused as fresh instances plus two new legality-bounded formalisms (peatstove, muletrack) — install narrated hidden-state tracking (axis total strictly over parent AND replay control) while holding the pooled_k3 retention bands that the mixed feedloop+statechain dose failed?

## Hypothesis

The statechain lesson already installed at 80 rows inside a mixed dose (11/20 vs replay's 10/20); doubling the dose to 160 rows and removing the dead feedloop rows (whose 0/20 surface consumed half the variable block) should widen the axis margin, and the freed loss mass no longer trains a failing skill, which is the mechanism argument for retention landing inside the replay band this time.

## Setup

- Parent and adapter base: the `hygiene_explore` composite (tree 9eb653d7…), fresh rank-32/alpha-64 adapters, no warm start, training seed 67.
- Corpus: `data/sft_statechain_only.jsonl` (ab6c7845…), 160 rows, construction seed 77,140, four formalisms x 40 (brewvat, courierloft, peatstove, muletrack); >=3 hidden updates per row; stateless and last-step-only distractors verified wrong; new formalisms' parameterized ops legality-bounded in the rendered spec text; banned vocabulary extended with the reference cell's retired feedloop pools; fresh-surface grep audit + zero row-overlap receipts vs every pinned predecessor corpus, stream, and gate (including the reference cell's).
- Arms: `replay_ctl2` (control, trains FIRST) and `statechain_only` (candidate).
- Exposure: exact zero-delta MILP vs `replay_ctl2` at namespace seed 55,131 (1,368,815 forward tokens, 574,630 targets, 628,314 mass x5 per arm; 1,280 position-aligned shared replay rows; zero encoder skips).
- Local gate: axis holdout 88,033 (40 u_statechain, 10 per formalism; strict TOTAL over both controls, no per-kind split — single-kind dose) + retention pooled over screens 88,034/88,035/88,036 under pooled_k3 bands on pooled sums (correct >= -15, caps <= +9, parsed >= -9 vs BOTH controls; i.e. +-5/3/3 on means).
- Conditional benchmark: medium, tb1024, sealed fresh seed 78,154, four models (base, parent, replay_ctl2, statechain_only), hardened runner; pilot gate = candidate aggregate strictly > base AND > replay_ctl2 AND > parent; goal gate recorded either way (max reachable 9/10 — menders closed; the reading of interest is rites conversion).

## Run

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_statechain_only_dose/scripts/run.py --smoke
# staged: --stage train-control | train-candidate | merge-arms | local | benchmark
```

## Results

Fill after the staged runs. Separate deployable evidence from oracle/hidden evaluation.

## Interpretation

Pending.

## Knowledgebase Update

- Program evidence updated: pending.
- Program backlog updated: pending.
- Claim ledger updated: pending.

## Artifacts

- `data/`: frozen corpus + manifests, exposure streams + receipt, four gate input pairs + design receipts.
- `scripts/`: full staged lifecycle (fail-closed TODO-pins for post-GPU hashes).
- `reports/artifact_manifest.yaml`
