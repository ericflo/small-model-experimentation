# Qwen3.5-4B Latent Decomposition: be your own tool-search

**Status:** finished

## Research Program
- Program: `structured_execution_and_compilers`
- Question: can the FIXED model climb the depth wall by proposing+verifying one DSL op at a time (be its own
  tool-search)? WHERE does it break, and does banking (C24) fix it?
- Anchors: C12 (base decompose-guidance = efficiency not coverage), C24 (banking installs monolithic depth-3).

## Pivot
An adversarial workflow review (`reports/design_review.md`, verdict *flawed*) showed the original "is depth-3
latent" framing was unfair + partly redundant with C12. Pivoted to DISSECT the failure (per-step ranking
accuracy) and test whether banking installs transferable planning. Adopted all must-fixes: VISIBLE-only search
termination (hidden-graded), brute-force honesty bar, min-depth-verified true-depth-3 (all 80), pruning ablation.

## Setup
- Model: Qwen3.5-4B only. list 16-op DSL (32 op/param combos). 80 min-depth-verified true-depth-3 held-out.
- Model ranks the 32 ops by likelihood given current-lists->goal; interpreter applies+verifies; beam search.
- Guides: base / banked_640 / banked_1280 (reused C24 adapters). Controls: brute (all 32) / random.

## Run
`python scripts/run.py --tag base --search --with-controls`
`python scripts/run.py --tag banked1280 --adapter <C24 banked_1280> --search`
`python scripts/analyze.py`

## Results
- **Lookahead wall (base):** per-step next-op top-1 = 0.013/0.062/0.237 (step1/2/3; chance 0.031). Recognition, no lookahead. Base as its own guide is WORSE than random (0.013 vs 0.025 vs brute 0.287 @ matched budget).
- **Banking installs transferable LOOKAHEAD** (refuting my monolithic-compilation hypothesis): banked lifts every step dose-dependently (step1 0.013->0.125->0.138, step2 0.062->0.138->0.250, step3 0.237->0.463->0.550).
- **Upgrades the guide:** banked1280-guided coverage 0.013->0.225 (~17x), competitive with brute at low budget.
- **Control:** random+pruning only 0.037 (pruning alone isn't the solver); brute+pruning 0.487.
See `reports/report.md`, `analysis/decomposition.png`.

## Interpretation
The single-shot depth wall is a LOOKAHEAD/planning gap; the model can't be its own multi-step search heuristic.
But banking (self-training on verified solutions) installs TRANSFERABLE compositional planning -- not just a
monolithic input->output map -- upgrading the model's step-wise search-guidance dose-dependently.

## Knowledgebase Update
- Program evidence: `research_programs/structured_execution_and_compilers/evidence.md` (C25)
- Claim ledger: C25 added

## Artifacts
- `scripts/decompose.py` (search + vectorized op-scoring), `scripts/run.py`, `scripts/analyze.py`
- `data/eval_frozen_d3.jsonl` (80 true-depth-3, reused from C23/C24), `runs/rank_*.json`, `runs/search_*.json`, `runs/ablation.json`, `runs/verdict.json`, `analysis/decomposition.png`, `reports/{prereg,report,design_review}.md`
- Reuses C24 banked_640/banked_1280 adapters (out of repo).
