# Qwen3.5-4B: Can SFT Install the Skill of Induction?

## Research Program
- Program: `posttraining_and_adaptation` / `benchmark_generalization`
- Question (mission-core, lift the wall): can QLoRA SFT install the general skill of inducing a hidden rule from examples (C38/C39: base can't)?

## Setup
- Each episode = a random SCRAMBLED digit order (stated) + hidden rule + 6 examples + query -> infer + apply. Families: shift (train), affine a in {3,7,9} (out-of-family). Answer-only QLoRA r32/a64. GATE: base EXECUTE ceiling per family (induction failure meaningful only if base can execute). Eval = forced `Answer:` argmax over the 10 digit tokens.

## Run
`python scripts/gen_data.py --n-train 8000`; `python scripts/train_lora.py --train data/train_shift.jsonl --out runs/lora_shift8k --epochs 2`; `python scripts/eval_induction.py --data data/heldout_shift.jsonl --mode induce --adapter runs/lora_shift8k`; `python scripts/analyze.py`.

## Results
Shift induce: base 0.087 -> SFT-4k 0.35 -> SFT-8k 0.40 (data-limited) but plateaus below execute ceiling 0.72. Affine (OOF) 0.21 -> 0.30 (shift-specific). Catastrophic forgetting: shift execute 0.72 -> 0.09. See `reports/report.md`, `analysis/meta_induction.png`.

## Interpretation
The induction wall is neither a hard architectural bound nor cleanly liftable: partial, procedure-specific install with catastrophic forgetting. Trained to induce, the fixed 4B learns a specific procedure, not the general skill -- an executor at heart.

## Knowledgebase Update
- Claim ledger: C43

## Artifacts
- `scripts/episode_gen.py`, `scripts/gen_data.py`, `scripts/train_lora.py` (answer-only QLoRA), `scripts/eval_induction.py` (forced-digit induce + execute ceiling), `scripts/analyze.py`
- `runs/lora_shift*`, `runs/eval_*.json`, `runs/verdict.json`, `analysis/meta_induction.png`, `reports/{report,design_review}.md`
