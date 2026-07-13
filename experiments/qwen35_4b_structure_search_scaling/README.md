# Qwen3.5-4B: When Does the Model's Structure Beat Brute Search? (depth-4)

**Status:** finished

## Research Program
- Program: `structured_execution_and_compilers` / `evidence_conditioned_selection`
- Question (C34 follow-up): C34 found brute-force structure-search dominates the model at deploy at depth-3 (enumerable space). Does the model's structure-pruning win at depth-4 (16^4=65536, too large to enumerate cheaply)?

## Setup
- banked_d4 (depth-4-banked) model + held-out depth-4 tasks (n=60, min-depth-verified).
- Gate: banked_d4 structure-coverage at depth-4 (bank_structure.py). Brute-full structure-search deploy (brute_only.py: enumerate 65536, value-fill, execution-consensus).

## Run
`python scripts/bank_structure.py --tag banked_d4 --adapter <banked_d4> --eval-file data/eval_frozen_d4.jsonl` (gate); `python scripts/brute_only.py` (brute-full deploy); `python scripts/analyze.py`.

## Results
The scissors WIDENS, never crosses: banked_d4 structure-cov 0.10 (vs 0.51 at depth-3) while brute-full deploys 0.967 (vs 0.975). Banking's structure collapses with depth faster than brute's exponential cost grows intractable. See `reports/report.md`, `analysis/structure_search_scaling.png`.

## Interpretation
The model never beats brute-force structure-search on this substrate. Closes the C32->C33->C34->C35 arc: the wall is structure; banking installs it into the forward pass (collapsing with depth); with an interpreter, brute structure-search dominates the weights outright.

## Knowledgebase Update
- Claim ledger: C35

## Artifacts
- `scripts/bank_structure.py` (gate), `scripts/bank_fill.py` (model-guided, superseded by brute), `scripts/brute_only.py` (brute-full deploy), `scripts/analyze.py`
- `data/eval_frozen_d4.jsonl`, `runs/{bank_banked_d4,brute_d4,verdict}.json`, `analysis/structure_search_scaling.png`, `reports/report.md`
- Uses external banked_d4 adapter (scratchpad).
