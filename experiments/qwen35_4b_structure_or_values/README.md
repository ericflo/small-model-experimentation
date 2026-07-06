# Qwen3.5-4B: Is the Wall Structure or Values? (skeleton-then-fill)

## Research Program
- Program: `structured_execution_and_compilers` / `interpretability_and_diagnostics`
- Question (C31 follow-up): when the model fails depth-3, is it a STRUCTURE error (wrong op-type sequence) or a VALUE error (right skeleton, wrong params)?

## Setup (pivoted after review + smoke)
- op-seq GENERATION fails (0.00 even depth-1, format handicap), so: model NATIVE Python coverage as baseline; a format-immune STRUCTURE signal (model program BEHAVIOR matched to the true op-type skeleton with any params); oracle-skeletonfill (true structure + value-search); random-skeletonfill (value-fungibility control). Min-depth-verified true-depth tasks, n=120/depth.

## Run
`python scripts/skeleton_fill.py --n-per-depth 120 --k 8 --randR 8 50 200 --depths 2 3` then `python scripts/analyze.py`.
(`scripts/gen_skeletons.py` documents the op-seq generation format failure.)

## Results
The wall is STRUCTURE. Value tax (structure-cov - concrete-cov) = +0.000 at depth-3; oracle-skeletonfill = 1.000 (values trivial given structure); random-skeletonfill low (R200=0.108 at depth-3, not value-fungible). See `reports/report.md`, `analysis/structure_or_values.png`.

## Interpretation
The compositional wall is a STRUCTURE-PROPOSAL problem: the model can't propose which ops in which order; once structure is known, values are free. Unifies C19/C25/C31; explains why tool-structure-seeds (C22) and banking were necessary.

## Knowledgebase Update
- Claim ledger: C32

## Artifacts
- `scripts/skeleton_fill.py` (monolithic Python + format-immune structure signal + oracle/random skeletonfill), `scripts/gen_skeletons.py` (op-seq gen format probe), `scripts/analyze.py`
- `runs/{skelfill_results,verdict}.json`, `analysis/structure_or_values.png`, `reports/{report,design_review}.md`
