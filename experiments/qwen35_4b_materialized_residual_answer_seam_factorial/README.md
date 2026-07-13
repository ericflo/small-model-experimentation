# Qwen3.5-4B Materialized Residual Answer-Seam Factorial

**Status:** in-progress · since 2026-07-13 · fresh construction complete; adversarial review holds every model stage sealed pending calibration firewall/lock implementation

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
thought hit its cap. A complete 2x2 crossing think@512/no-think with
freeform/literal-`PROGRAM:` prefill can isolate answer syntax from reasoning
policy without supplying answer identity. A separately calibrated policy may
expose residual completion that the invalid free-form interface hid.

## Setup

- Model: only `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16.
- Dataset/task source: fresh procedural exact-depth-three list transforms;
  never `benchmarks/`.
- Split: 48 known-answer interface-calibration tasks and 24 disjoint mechanics
  tasks, with new task/request/seed domains.
- Interface arms: `think512_freeform`, `think512_program_slot`,
  `no_think_freeform`, and `no_think_program_slot`; all answer aliases are
  sampled autonomously with the same 24-token tail cap.
- Baseline: taskwise matched-compute direct full-program sampling on the same
  backend and selected interface budget.
- Controls: name-only siblings, task-hash shuffled materialized states/targets,
  exact echo, candidate-blind direct sampling, and exhaustive CPU ceiling.
- Primary metric: hidden-correct all-sibling proposal coverage and a
  visible-only selector, gated behind the interface calibration.
- Oracle-only metrics: exact candidate viability and hidden program success;
  neither may affect interface choice, prompts, budgets, or selected IDs.
- Calibration gates: >=44/48 exact echoes, >=44/48 parses, <=2/48 answer-cap
  contacts, plus >=22/24 exact/parse and <=1/24 cap contacts in each arity.
- Winner: first qualifier in the fixed least-departure priority, never the
  best observed metric.
- Hidden-label boundary: mechanics remains inaccessible until a committed
  winner receipt and second lock; hidden scoring remains inaccessible until a
  committed visible-selection receipt. Qualification/confirmation and all
  benchmark content remain unread.

## Run

Smoke:

```bash
python experiments/qwen35_4b_materialized_residual_answer_seam_factorial/scripts/run.py --smoke
```

Fresh construction and append-only v2 smoke:

```bash
python experiments/qwen35_4b_materialized_residual_answer_seam_factorial/scripts/construct.py
python experiments/qwen35_4b_materialized_residual_answer_seam_factorial/scripts/run.py --design-smoke
```

Full:

```bash
sealed pending implementation review and published calibration lock
```

## Results

The immutable scaffold-v1 receipt still passes. V2 construction also passes
twice byte-identically: 72 unique public instances, zero overlap with 264
authenticated parent instances, exact live-sibling strata, 4,104 prepared rows,
2,952 unique canonical request IDs, balanced A-X calibration positions, zero
model calls, and empty forbidden-read receipts. Append-only real-tokenizer
receipts also authenticate all answer compositions, context fit, zero rendered
parent overlap, and the current shared-thought runner. Sixty-one model-free
tests pass, including adversarial durable transaction/recovery mutations and
exact persisted-token thought forking. A calibration-only loader is byte-
identical with every mechanics artifact absent, and a fake-runner integration
passes all five durable invocations plus analysis/restart. This remains
registration and implementation evidence only.

## Interpretation

No capability belief changes. Adversarial review materially strengthened the
design and still holds live execution: the calibration implementation lock,
live-engine preflight, independent implementation review, and committed
outcome boundaries remain mandatory.

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
