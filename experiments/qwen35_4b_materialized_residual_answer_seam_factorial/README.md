# Qwen3.5-4B Materialized Residual Answer-Seam Factorial

**Status:** in-progress · since 2026-07-13 · scaffold reserved; construction, adversarial design review, and every model stage remain sealed

This fresh successor isolates the failure that made the durable materialized-
residual generation result uninterpretable: the model never entered a reliable
short answer ABI. It must qualify an answer seam on known-answer echo tasks
before any disjoint residual-mechanics task can be opened.

## Research Program

- Program: `structured_execution_and_compilers`
- Program question: can externalized intermediate state become a deployable
  compositional interface rather than only local semantic routing?
- Prior anchors: `qwen35_4b_materialized_residual_sibling_search_fresh_replication`,
  `qwen35_4b_commit_slot_semantic_power_replication`, and
  `qwen35_4b_early_text_hypothesis_forking`.

## Question

Can a separately calibrated, autonomously scored short answer seam make
materialized residual generation measurable, and if so does it beat name-only,
token-preserving shuffled, and matched-compute direct sampling controls?

## Hypothesis

The prior model could copy a structured echo on 20/24 non-cap rows but every
thought hit its cap. A no-think structured seam or a forced close plus literal
`PROGRAM:` slot should remove answer-mode ambiguity without supplying answer
identity. If one achieves >=90% exact echo/parse and <=5% cap contact on
calibration tasks, it may expose residual completion that the invalid free-form
interface hid.

## Setup

- Model: only `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16.
- Dataset/task source: fresh procedural exact-depth-three list transforms;
  never `benchmarks/`.
- Split: 48 known-answer interface-calibration tasks and 24 disjoint mechanics
  tasks, with new task/request/seed domains.
- Interface arms: current forced-close free-form control, no-think short
  structured emission, and forced-close `PROGRAM:` slot with autonomously
  generated program tokens.
- Baseline: taskwise matched-compute direct full-program sampling on the same
  backend and selected interface budget.
- Controls: name-only siblings, task-hash shuffled materialized states/targets,
  exact echo, candidate-blind direct sampling, and exhaustive CPU ceiling.
- Primary metric: hidden-correct all-sibling proposal coverage and a
  visible-only selector, gated behind the interface calibration.
- Oracle-only metrics: exact candidate viability and hidden program success;
  neither may affect interface choice, prompts, budgets, or selected IDs.
- Hidden-label boundary: mechanics remains inaccessible until one interface
  passes all three frozen calibration gates; qualification/confirmation and
  all benchmark content remain unread.

## Run

Smoke:

```bash
python experiments/qwen35_4b_materialized_residual_answer_seam_factorial/scripts/run.py --smoke
```

Full:

```bash
sealed pending adversarial design review
```

## Results

The model-free scaffold passes with three frozen interface arms, 48/24 disjoint
calibration/mechanics task counts, zero model loads/calls, and empty forbidden-
read receipts. This is registration evidence only.

## Interpretation

No scientific belief changes. The experiment ID and answer-seam decision
boundary are reserved before harness copying or model use.

## Knowledgebase Update

- Program evidence updated: no; no model result.
- Program backlog updated: successor identity reserved.
- Claim ledger updated: no.

## Artifacts

- `src/`
- `scripts/`
- `configs/`
- `runs/`
- `runs/smoke/summary.json`
- `reports/`
- `reports/artifact_manifest.yaml`
