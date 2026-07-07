# Qwen3.5-4B: Does the Structure-Proposal Wall Exist in Language?

## Research Program
- Program: `benchmark_generalization` / `structured_execution_and_compilers`
- Question (complement to C37): C37 showed the model SIMULATES multi-step reasoning in language. Does the C32/C36 structure-PROPOSAL (rule-INDUCTION) wall also persist in language?

## Setup
- Relational-composition INDUCTION: R=4 made-up relations (random bijections over ~16 made-up entities), hidden depth-D rule; give KB + examples + a query start NOT in examples -> infer which relations compose + apply. Min-depth-verified, uniqueness-pinned. Application-only control (rule GIVEN) = the ceiling. no-think + think (budget 4096, truncation-checked).

## Run
`python scripts/eval_proposal.py --render {ling,app,formal} --depths 1 2 3 4 [--think]` then `python scripts/analyze.py`.

## Results
Clean forward-pass dissociation at depth-1: application (execute given rule) 0.86 vs INDUCTION (infer rule) 0.00. Induction at chance no-think all depths; think only half-rescues (0.50 at d1). The model is an EXECUTOR, not an INDUCER, in language as in formal domains. See `reports/report.md`, `analysis/language_proposal_wall.png`.

## Interpretation
C37+C38: the compositional wall's two components dissociate by modality -- SIMULATION is modality-dependent (formal walls, language does not), PROPOSAL/INDUCTION is modality-general (hard in both).

## Knowledgebase Update
- Claim ledger: C38

## Artifacts
- `scripts/reasoning_proposal.py` (induction substrate, application-only control), `scripts/eval_proposal.py`, `scripts/analyze.py`, `scripts/reasoning_family.py`
- `runs/prop_*.json`, `runs/verdict.json`, `analysis/language_proposal_wall.png`, `reports/{report,design_review}.md`
