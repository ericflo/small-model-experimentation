# Experiment Log

## 2026-06-24

- Created standalone package `qwen35_4b_joint_shortlister_ladder`.
- Separated heavyweight model output under `/workspace/large_artifacts/qwen35_4b_joint_shortlister_ladder`.
- Implemented record-local alias maps so the model must read the inventory instead of relying on stable code meanings.
- Implemented joint pair output `LLL,RRR` with constrained decoding and exact recall@k.
- Implemented random and max-split designed observation prompts.

## Results

- Built full ladder dataset: 768 train records and 64 eval records.
- Trained Qwen3.5-4B QLoRA joint-pair adapter for 240 optimizer steps.
- Attempted beam-32 evaluation; trained 512-operator generation hit CUDA OOM after base evaluation, so exact model recall was rerun at beam 16 with dynamic recall reporting.
- Beam-16 model evaluation used 16 records: 2 per library-size/template cell.
- Joint pair recall@16 was 0.0% for base random, trained random, trained designed, and trained shuffled-inventory random.
- Marginal trained recall@16 showed small movement (LEFT 12.5%, RIGHT 18.8%) but did not cleanly separate from shuffled inventory (LEFT 12.5%, RIGHT 12.5%).
- Observation design remained useful on the executable side: average survivor count dropped from 2177.406 to 1191.25 over 32 diagnostic records, and selected-hidden-all moved from 50.0% to 56.25%.
