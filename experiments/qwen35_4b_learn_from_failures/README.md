# Qwen3.5-4B: Learn from Your Own Failures (DPO)

## Research Program
- Program: `posttraining_and_adaptation` / `evidence_conditioned_selection`
- Question: does preference/contrastive training on the model's OWN (correct, incorrect) samples raise deployable greedy@1 beyond SFT-on-positives (closing the coverage->deployable gap)?
- Extends prior MBPP DPO work here (constrained_coverage_dpo, offline_hard_negative_coverage_dpo) to the controlled list-DSL depth-3 substrate, targeting the greedy@1 gap.

## Setup
- Harvest 174 same-task (chosen=verified-correct, rejected=verified-wrong) pairs from banked_1280's own no-think samples (disjoint from held-out).
- Arms (fresh QLoRA from base, matched correct data): base; SFT (positives, 3ep); SFT_2x (6ep, compute ctrl); DPO (cached-ref-SFT + NLL anchor, various epochs); DPO-shuffled (rejecteds deranged, loss-shape ctrl).
- Eval no-think greedy@1 (deployable) + coverage@16, frozen held-out depth-3, n=80.

## Run
`python scripts/harvest_pairs.py --adapter <banked_1280> --pool 500 --k 16` then `build_shuffled.py`, `train_lora.py` (SFT/SFT_2x), `train_dpo.py`, `eval_ladder.py`, `analyze.py`.

## Results
DPO does NOT close the gap: pre-DPO 2AFC=0.81 (strong latent verifier) but preference-optimizing it COLLAPSES generation (greedy 0.050@0.25ep -> 0.000@0.5ep -> 0.013@3ep; coverage crashes too). The lever is MORE SFT: SFT_2x triples greedy@1 (0.037->0.113). See `reports/report.md`, `analysis/learn_from_failures.png`.

## Interpretation
Strong latent sample-discrimination (2AFC 0.81) is READ-ONLY; it does not transfer to a generation gain via preference training (DPO destroys the model). Close the coverage->deployable gap with more SFT-on-positives, not preference-on-failures.

## Knowledgebase Update
- Claim ledger: C29

## Artifacts
- `scripts/harvest_pairs.py`, `scripts/build_shuffled.py`, `scripts/train_lora.py`, `scripts/train_dpo.py` (manual cached-ref DPO on QLoRA), `scripts/eval_ladder.py`, `scripts/analyze.py`
- `data/pairs.jsonl`, `data/pairs_shuffled.jsonl`, `data/sft.jsonl`, `runs/eval_*.json`, `runs/verdict.json`, `analysis/learn_from_failures.png`, `reports/{report,design_review}.md`
- Adapters (~180MB each) moved out of repo.
