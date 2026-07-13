# Qwen3.5-4B Counterfactual Order-Support Selector

## Research Program

- Program: `evidence_conditioned_selection`
- Secondary: `test_time_reasoning_budget` and
  `interpretability_and_diagnostics`.
- Prior anchors: `qwen35_4b_commit_slot_semantic_power_replication`,
  `qwen35_4b_thinking_content_vs_compute`, and
  `qwen35_4b_confidence_guided_compute`.

## Question

Does the forward probability contribution of coherent thought order provide a
label-free answer selector? For each alias, average across three paths:

`P(alias | ordered thought) - P(alias | exact-token shuffle)`

and choose the largest delta.

## Hypothesis

The shuffled counterfactual holds the thought-token multiset, length, answer
syntax, and alias vocabulary fixed while destroying coherent order. Subtraction
should cancel identity and token-presence nuisance, isolating the part of the
answer state caused by coherent reasoning. The hypothesis is actionable only if
this vector beats every strong K=3 selector and a task-mismatched shuffle control
on qualification and separately opened confirmation.

## Setup

- Model: no new model call; source rows were produced only by
  `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` under Transformers bf16 SDPA.
- Dataset/task source: the parent's two disjoint 113-task, three-path,
  contamination-free procedural seam stages at cap 1,024.
- Train/eval split: qualification may open now. Confirmation files are absent
  locally and fail closed until a passing qualification is committed/pushed.
- Baselines: first trace, majority with mean-probability tie-break, mean ordered
  probability, max-confidence trace, and minimum-entropy trace.
- Controls: reverse delta and a gold-alias-balanced but task-mismatched shuffled
  distribution. The latter uses hidden labels only to make the control harder;
  it never contributes to the candidate.
- Primary metric: exact task accuracy and paired task-bootstrap differences.
- Oracle-only metrics: correct/chosen alias breadth and mismatch construction.
- Hidden-label boundary: prediction functions accept no correct alias or
  correctness field; mutation invariance is unit-tested.

Qualification and confirmation each require candidate accuracy in 15%--70%,
at least +3pp over every deployable baseline and the mismatch control, a
one-sided 95% paired-task lower bound above zero against every baseline/control,
at least eight chosen and eight successful aliases, and at least +10pp over the
reverse-delta anti-selector. No secondary score can rescue the primary rule.

This is a selector-signal qualification, not a matched-compute capability test.
Ordered+shuffle uses three additional full prefills. Even a two-stage pass only
licenses a fresh K=3 candidate versus K=6 actual-forward-token sample-more run.

## Run

Smoke (never opens confirmation):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/qwen35_4b_counterfactual_order_support_selector/scripts/run.py \
  --stage smoke
```

Qualification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  experiments/qwen35_4b_counterfactual_order_support_selector/scripts/run.py \
  --stage qualification
```

## Results

Pending. Confirmation and fresh GPU work are sealed.

## Interpretation

Pending.

## Knowledgebase Update

- Program evidence updated: pending terminal result.
- Program backlog updated: pending terminal result.
- Claim ledger updated: no new claim from retrospective rows.

## Artifacts

- `src/selector.py`: pure prediction, controls, statistics, and gates.
- `scripts/run.py`: qualification/confirmation firewall.
- `configs/default.yaml`: immutable hashes, aliases, gates, and seeds.
- `data/qualification/`: hash-identical copies of parent qualification rows.
- `data/confirmation/`: intentionally absent unless qualification passes.
- `runs/` and `analysis/`: gated summaries and task-level predictions.
- `reports/preregistration.md` and `reports/design_review.md`: frozen rules and
  adversarial review.
- `reports/pre_qualification_implementation_audit.md`: 36 outcome-blind data,
  firewall, selector, statistics, and anchoring assertions.
- `reports/artifact_manifest.yaml`
