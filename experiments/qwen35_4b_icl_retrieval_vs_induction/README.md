# Qwen3.5-4B: Is In-Context Learning Retrieval or Induction?

**Status:** finished

## Research Program
- Program: `benchmark_generalization`
- Capstone question: if the model can't induce a novel rule (C38), what is in-context learning doing? Retrieval of familiar structure, or induction of novel structure?

## Setup
- Execution-safe single-value task: "advance k steps in a cyclic order" (output one digit). Crux (adversarial review): FAMILIAR order (natural 0-9, retrievable) vs NOVEL order (a STATED random cyclic order) at matched 1-param complexity, x EXECUTE (rule stated) vs INDUCE (few-shot). No-think (code-mode-free), chance 1/10.
- The first vehicle (letter ciphers) floored (4B can't apply even a given cipher, 0.20 -- char-assembly limit); pivoted to single-value.

## Run
`python scripts/eval_succ.py --n 60` (2x2), then `python scripts/analyze.py`.

## Results
EXECUTE: familiar 1.00, novel 0.97 (familiarity-independent). INDUCE: familiar 0.45, novel 0.12 (= chance). More examples do not rescue novel induction (0.15->0.05). The model EXECUTES a novel rule perfectly but cannot INDUCE it -> ICL = retrieval of familiar structure, not induction of novel structure. See `reports/report.md`, `analysis/icl_retrieval_vs_induction.png`.

## Interpretation
Unifies the arc: executor/retriever of pretrained structure (C37), not inducer of novel structure (C38, C32/C36). ICL is the retrieval half of reasoning, not the induction half.

## Knowledgebase Update
- Claim ledger: C39

## Artifacts
- `scripts/succ_family.py` (single-value successor substrate), `scripts/eval_succ.py`, `scripts/analyze.py`
- `scripts/ic_family.py` (original cipher substrate -- floored, kept for the methodological lesson), `scripts/eval_icl.py`
- `runs/succ_crux_nothink.json`, `runs/moreex.json`, `runs/verdict.json`, `analysis/icl_retrieval_vs_induction.png`, `reports/{report,design_review}.md`
