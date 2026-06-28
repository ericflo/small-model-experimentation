# Qwen3.5-4B Inventory Shortlister Training

Standalone Qwen3.5-4B QLoRA experiment for inventory-conditioned operator shortlisting.

The experiment trains the model to select operator aliases for LEFT and RIGHT slots in two-operator programs over a 512-operator `list[int] -> int` inventory. Evaluation forms candidate budgets by crossing top aliases for each slot:

- 1024 candidates = top-32 LEFT x top-32 RIGHT
- 4096 candidates = top-64 LEFT x top-64 RIGHT
- 16384 candidates = top-128 LEFT x top-128 RIGHT

Large artifacts are stored outside this directory:

`/workspace/large_artifacts/qwen35_4b_inventory_shortlister_training`

## Layout

- `configs/`: experiment configuration.
- `data/`: generated train/eval examples and manifests.
- `logs/`: chronological experiment log.
- `reports/`: result JSON, CSV summaries, figures, and final report.
- `run_logs/`: captured command output.
- `scripts/`: dataset, training, evaluation, and reporting entry points.
- `src/`: standalone operator library, task generator, prompts, and evaluators.

## Main Commands

```bash
python scripts/build_dataset.py
python scripts/train_shortlister.py --train data/train_slots.jsonl --output-dir /workspace/large_artifacts/qwen35_4b_inventory_shortlister_training/models/qwen35_lora
python scripts/eval_shortlister.py --adapter-dir /workspace/large_artifacts/qwen35_4b_inventory_shortlister_training/models/qwen35_lora
python scripts/make_report.py
```

