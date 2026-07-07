# Qwen3.5-4B: Can the Model Localize Its Own Errors in Multi-Step Reasoning?

## Research Program
- Program: `benchmark_generalization`
- Question: extends C40 (single-step metacognition) to multi-step. Does per-step confidence pinpoint WHERE the model first slips?

## Setup
- Model advances k steps in a cyclic order over depth-4-7 chains via SCAFFOLDED decoding (force 'Step i: <digit>', read the digit distribution at each step = live per-step confidence). Ground truth = local correctness (m_i == successor of the model's OWN previous step). Familiar (natural) order -> genuine arithmetic slips ~31%/step. Make-or-break control: DE-TREND (confidence rises with position). Baselines: uniform, always-last, position-prior.

## Run
`python scripts/eval_localize.py --n-per-cond 150` then `python scripts/analyze.py`.

## Results
Per-step error prediction survives de-trending (AUROC 0.75 vs 0.73). The de-trended confidence dips EXACTLY at the first error (offset 0). Single-slip localization 0.56 (raw 0.64) >> position-prior 0.36 >> uniform 0.19. Targeted repair (redo from located step) fixes 0.56 at 3.8 steps vs redo-all 5.6. Caveat: multi-slip chains -- finds AN error 0.76 but the FIRST 0.27. See `reports/report.md`, `analysis/error_localization.png`.

## Interpretation
C40's implicit metacognition is STEP-RESOLVED: per-step confidence carries WHERE the model slipped. Deployable targeted repair; strongest when the model slips once.

## Knowledgebase Update
- Claim ledger: C42

## Artifacts
- `scripts/chain_family.py`, `scripts/eval_localize.py` (scaffolded decoding + per-step confidence), `scripts/analyze.py`
- `runs/localize.json`, `runs/verdict.json`, `analysis/error_localization.png`, `reports/{report,design_review}.md`
