# Qwen3.5-4B: Do the Structure Findings Generalize? (string/register)

## Research Program
- Program: `structured_execution_and_compilers` / `interpretability_and_diagnostics`
- Question: C16 tested the EARLY ladder (C13-15) cross-family. Do the RECENT findings -- C32 (wall-is-structure), C34 (brute-search dominates) -- hold on STRING and REGISTER, or are they list-DSL artifacts?

## Setup
- Family-generic replication on STRING (char edits, 13 prims), REGISTER (3-register machine, 12 prims), LIST (anchor, 16 prims), depth-3, min-depth-verified, n=100. Base model + search (no banking).
- Metrics: base greedy@1/cov@8 + format-immune structure-coverage; oracle-skeletonfill; random-skeletonfill@R; brute-full structure-search + value-fill + execution-consensus deploy.

## Run
`python scripts/cross_substrate.py --family {string,register,list} --n 100 --k 8 --randR 8 50 200` then `python scripts/analyze.py`.

## Results
C32 + C34 are MODEL-LEVEL LAWS: identical pattern on all three (base ~0, structure-cov = concrete-cov, oracle-skelfill 1.0, random low, brute-deploy ~1.0). See `reports/report.md`, `analysis/crosssubstrate_structure.png`.

## Interpretation
The fixed 4B is a value-computer, not a deep-structure-proposer, across substrates; the wall is structure-proposal everywhere; with an interpreter, brute-force structure-search dominates the weights outright everywhere. Establishes the compositional arc as model-level, not list-DSL-specific.

## Knowledgebase Update
- Claim ledger: C36

## Artifacts
- `scripts/cross_substrate.py` (family-generic C32+C34), `scripts/analyze.py`
- `runs/cs_{string,register,list}.json`, `runs/verdict.json`, `analysis/crosssubstrate_structure.png`, `reports/report.md`
