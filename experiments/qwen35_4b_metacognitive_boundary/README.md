# Qwen3.5-4B: Does the Model Know When It Will Fail?

**Status:** finished

## Research Program
- Program: `benchmark_generalization` (first metacognition/calibration claim)
- Question: the arc pinned a VERIFIED competence boundary (C39). Does the model's own confidence/uncertainty track it -- does it know when it is guessing?

## Setup
- Format-equalized single-value task ('advance k in a cyclic order') across a verified competence spectrum. Two non-degenerate logit confidence signals (verbalized 0-100 is degenerate = constant 100): IMPLICIT P(answer) = softmax over the 10 digit tokens at 'Answer: ' (one forward pass); EXPLICIT P(True) = Kadavath self-verification. Clean test = WITHIN-condition item-level AUROC in the surface-matched familiar_induce cell vs an external surface-feature baseline.

## Run
`python scripts/eval_metacog.py --n 150` then `python scripts/analyze.py`.

## Results
Implicit P(answer) tracks accuracy (1.00/0.44/0.29/0.15 vs acc 1.00/0.40/0.19/0.10); explicit P(True) flat (~0.4). Within familiar_induce: P(answer) AUROC 0.95 (CI 0.90-0.99) >> surface 0.61 >> P(True) 0.46 (chance). Selective prediction lifts accuracy-on-attempted 0.23 -> ~1.0. See `reports/report.md`, `analysis/metacognitive_boundary.png`.

## Interpretation
The model knows when it will fail -- but only in its OUTPUT DISTRIBUTION, not in anything it can SAY. Deployable: read answer-token probability for a confidence/abstain signal; never trust explicit self-assessment.

## Knowledgebase Update
- Claim ledger: C40

## Artifacts
- `scripts/succ_family.py` (format-equalized substrate), `scripts/eval_metacog.py` (logit confidence signals), `scripts/analyze.py`
- `runs/metacog_records.json`, `runs/verdict.json`, `analysis/metacognitive_boundary.png`, `reports/{report,design_review}.md`
