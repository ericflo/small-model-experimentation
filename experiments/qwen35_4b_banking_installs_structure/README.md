# Qwen3.5-4B: Does Banking Install STRUCTURE?

**Status:** finished

## Research Program
- Program: `structured_execution_and_compilers` / `posttraining_and_adaptation`
- Question (C32 follow-up): C32 showed the wall is structure-proposal. C22-24 showed banking crosses depth-3. Does banking install the STRUCTURE the base can't propose?

## Setup
- Run C32's format-immune structure-coverage (model program BEHAVIOR matches the true op-type skeleton with any params) on BASE vs BANKED_1280 (C24 adapter), on held-out depth-3 (banked's frozen eval, disjoint from training), min-depth-verified, no-think, n=80.

## Run
`python scripts/bank_structure.py --tag base` and `python scripts/bank_structure.py --tag banked --adapter <banked_1280>`, then `python scripts/analyze.py`.

## Results
Banking installs STRUCTURE: base structure-cov 0.000 -> banked 0.512 (held-out, generalizable). Banking converts the wall from structure-bound (base struct=concrete=0) to value-bound (banked struct 0.512 > concrete 0.362, value tax +0.15, fillable). See `reports/report.md`, `analysis/banking_installs_structure.png`.

## Interpretation
Mechanistic closure of C32: banking's entire lever is installing op-sequence structure; the residual value tax is small and fillable (oracle-skelfill=1.0). Unifies C22-24/C31/C32: banking = structure-installation.

## Knowledgebase Update
- Claim ledger: C33

## Artifacts
- `scripts/bank_structure.py`, `scripts/skeleton_fill.py` + `scripts/gen_skeletons.py` (C32 structure signal), `scripts/analyze.py`
- `data/eval_frozen_d3.jsonl` (banked_1280 held-out eval), `runs/bank_{base,banked}.json`, `runs/verdict.json`, `analysis/banking_installs_structure.png`, `reports/report.md`
- Uses external banked_1280 adapter (C24, scratchpad).
