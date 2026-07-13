# Qwen3.5-4B Counterfactual Order-Support Selector

**Status:** finished

Terminal `NO_ORDER_SUPPORT_SELECTOR`: ordered-minus-exact-shuffle probability
beats hard voting but not confidence/entropy robustly or the task-mismatch
control, so confirmation and a fresh matched-compute run remain sealed.

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

Terminal qualification decision: `NO_ORDER_SUPPORT_SELECTOR`. The registered
raw probability-delta rule reached 43/113 (38.05%), substantially above first
trace (31/113), majority (33/113), and mean ordered probability (37/113). It
nevertheless missed the conjunctive direct-control standard: minimum-entropy
selection reached 41/113 and max-confidence 40/113, so candidate gains were
only +1.77pp and +2.65pp with one-sided paired lower bounds -3.54pp and
-2.65pp. The oracle-balanced task-mismatched shuffle reached 44/113, one task
better than the candidate.

| selector/control | accuracy | candidate gap | paired lower |
| --- | ---: | ---: | ---: |
| order-support delta | 38.05% | — | — |
| first trace | 27.43% | +10.62pp | +4.42pp |
| majority | 29.20% | +8.85pp | +2.65pp |
| mean ordered probability | 32.74% | +5.31pp | -0.88pp |
| max-confidence trace | 35.40% | +2.65pp | -2.65pp |
| minimum-entropy trace | 36.28% | +1.77pp | -3.54pp |
| oracle-balanced mismatch | 38.94% | -0.88pp | -7.08pp |
| reverse delta | 7.08% | +30.97pp | diagnostic |

Accuracy, reachability, breadth, and reverse-delta gates passed; mandatory
point-gain and uncertainty gates failed. Predictions spanned 11 aliases and
successes 10 target aliases. Twenty-seven predictions were absent from the
three ordered argmax choices, and eight of those were correct, showing that the
delta sometimes extracts weak probability support beyond voting. That clue is
insufficient for qualification.

Confirmation artifacts remain absent. No fresh matched-compute run, causal
stage, or capability claim is licensed.

## Interpretation

The replicated ordered-thought group effect contains some per-task selection
signal, but raw exact-shuffle subtraction is not reliably better than cheap
confidence/entropy selection and is not specifically dependent on the matched
task's shuffled distribution. Retire this exact selector rather than tune its
transform on observed outcomes. The next capability mechanism must change the
proposal/continuation, not merely re-rank the same commit logits.

## Knowledgebase Update

- Program evidence updated: all three programs.
- Program backlog updated: exact-shuffle support selection retired.
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
- `reports/post_qualification_adversarial_audit.md`: 12 post-result objections,
  non-rescues, and the sealed-confirmation routing decision.
- `reports/artifact_manifest.yaml`
