# Qwen3.5-4B: Beating Sample-More with the Model's Own Uncertainty

## Research Program
- Program: `benchmark_generalization`
- Question: turn C40's calibrated implicit confidence (answer-token probability) into a compute tool -- can it beat uniform sample-more?

## Setup
- A mix of C40 successor problems spanning easy (execute), coverage-limited (familiar_induce), capability-limited (novel_induce). Sample k=12/problem; read each sample's P(answer) (one forward pass). Compare selection/allocation/abstention policies at matched budget.

## Run
`python scripts/eval_sampling.py --n 80 --k 12` then `python scripts/analyze.py`.

## Results
Self-consistency (majority vote) is FLAT (~0.48); confidence-select (argmax P(answer), verification-free) RISES 0.47->0.62, beating majority at every budget (oracle 0.83). Max P(answer) predicts solvability (AUROC 0.83) -> abstain on low-confidence -> ~1.0 on the confident third. Allocation ~tied with uniform confidence-select. See `reports/report.md`, `analysis/confidence_guided_compute.png`.

## Interpretation
The fixed 4B's own logits tell you which sample to trust and when to stop -- no verifier, no execution. Beats the standard verifier-free method (self-consistency). Deployable use of C40.

## Knowledgebase Update
- Claim ledger: C41

## Artifacts
- `scripts/succ_family.py`, `scripts/eval_sampling.py` (greedy + k sampled, each with P(answer)), `scripts/analyze.py`
- `runs/sampling_records.json`, `runs/verdict.json`, `analysis/confidence_guided_compute.png`, `reports/report.md`
