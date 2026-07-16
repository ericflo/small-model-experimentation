# Statechain-Only Dose

Dose ONLY the proven skill: lifecycle 15 split its verdict — u_statechain INSTALLED (11/20, strict over both controls) while u_feedloop died at 0/20 and dragged retention below the replay band. This cell re-runs the install with a 160-row statechain-only corpus (no dead feedloop rows) and asks whether the clean dose clears the calibrated gate the mixed dose failed.

**Status:** finished · 2026-07-15 · verdict PILOT_NOT_PROMOTED + RITES_CONVERTED + PARENT_GOAL_GATE_PASS_RECORDED — the candidate beat base and replay but not its parent at medium; its rites 0.300 vs 0.100/0.100 is the program's first local-install→family conversion; and the hygiene_explore parent recorded the FIRST 10/10 all-families goal-gate pass in program history (aggregate 0.3663 vs base 0.0800, zero ties, zero losses) on sealed seed 78,154 — confirmation now owed per the frozen law

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

Local gate (seed 88,033 + screens 88,034–88,036): PROMOTED on all eight frozen checks — axis 21/40 strictly over replay_ctl2 (19) and the parent (17); pooled retention 64.67 vs 66.67/67.33, inside the calibrated ±5 bands; per-formalism brewvat 8/7/6, courierloft 5/3/2, muletrack 1/0/0, peatstove 7/9/9 (the candidate lost peatstove — recorded).

Medium event at sealed seed 78,154 (tb 1,024), all arms authenticated and within budget:

| arm | aggregate | goal gate vs base | notes |
|---|---|---|---|
| base | 0.0800 | — | inside every historical envelope |
| **hygiene_explore_parent** | **0.3663** | **PASS 10/10** — zero ties, zero losses | menders 0.017, warren 0.150 vs 0.100 |
| statechain_only | 0.3494 | 8/10 (ties menders, warren) | **rites 0.300** vs parent/replay 0.100 |
| replay_ctl2 | 0.3157 | 8/10 (tie menders; loses warren) | |

Pilot gates: candidate > base ✓, > replay ✓, > parent ✗ (−0.017) — NOT promoted per the frozen contract. The conversion reading: candidate rites 0.300 against 0.100 for BOTH the parent and the exposure-matched replay control on the same seed — the program's first demonstrated local-install→family transfer. The recorded goal gate: the parent passed all ten families strictly, the first such pass in program history by a contamination-free model.

## Interpretation

Three lessons, one owed action. (1) The statechain dose converts: teaching narrated hidden-state tracking on invented machines moved the protocol-compliance family threefold over matched controls — the axis→family under-conversion law has its first counterexample, with an end-to-end causal chain from designed data to benchmark family. (2) The dose still trades: −0.017 aggregate versus its own parent (lockpick/siftstack/sirens gave back what rites gained), so the parent remains the portfolio's best single model. (3) The parent's 10/10 is a recorded event fact on one seed with single-item margins at menders and warren; the "9/10 ceiling" was a draw-dependent floor-tie, exactly as the tier forensics predicted. The confirmation law — independent fresh seeds plus a same-backend matched-compute sample-more baseline — governs before any claim; confirmation is the immediate funded successor.

## Knowledgebase Update

- Program evidence updated: pending.
- Program backlog updated: pending.
- Claim ledger updated: pending.

## Artifacts

- `data/`: frozen corpus + manifests, exposure streams + receipt, four gate input pairs + design receipts.
- `scripts/`: full staged lifecycle (fail-closed TODO-pins for post-GPU hashes).
- `reports/artifact_manifest.yaml`
