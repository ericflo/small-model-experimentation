# Qwen3.5-4B: Does the Compositional Wall Exist in Language?

## Research Program
- Program: `benchmark_generalization` / `structured_execution_and_compilers`
- Question: all 36 prior claims are formal/procedural. The model is a LANGUAGE model -- does the compositional (mental-SIMULATION, C13) wall exist in its native linguistic domain?

## Setup
- Contamination-free successor-chain traversal over made-up pronounceable entities + distractor chains, shuffled. Same chain rendered linguistic-semantic / linguistic-symbolic('gorps', made-up relation control) / formal-dict. no-think PRIMARY (mental simulation), depths 1-6, n=80. Shortcut-hardened (answer interior/never sink, random start, recency baseline 0.04).

## Run
`python scripts/eval_reasoning.py --render {ling_sem,ling_sym,formal} --n-per-depth 80 [--think]` then `python scripts/analyze.py`.

## Results
NO depth-3 wall in language: linguistic-semantic d1-d4 = 0.99/1.00/0.99/0.94, made-up-relation control perfect through depth-3 (1.00). Stark contrast to the formal-composition wall (depth-3). The wall is formal-modality-specific. Formal-dict triggers code-mode (confounded). See `reports/report.md`, `analysis/language_reasoning_wall.png`.

## Interpretation
The compositional wall is a property of FORMAL composition, not the model's ability to reason multi-step. In language, mental simulation is intact (depth 4-5). Tests SIMULATION (C13), not the C32/C36 proposal wall.

## Knowledgebase Update
- Claim ledger: C37

## Artifacts
- `scripts/reasoning_family.py` (contamination-free substrate, 3 renderings), `scripts/eval_reasoning.py`, `scripts/analyze.py`
- `runs/reason_*.json`, `runs/verdict.json`, `analysis/language_reasoning_wall.png`, `reports/{report,design_review}.md`
