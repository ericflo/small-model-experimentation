# Experiment Log

## 2026-06-24

Initialized a standalone Qwen3.5-4B inventory shortlister training experiment.

Design commitments:

- Use Qwen3.5-4B from the local model cache.
- Train a QLoRA adapter, with all large model artifacts stored outside the experiment directory.
- Focus on 512-operator two-hole programs.
- Evaluate coverage at fixed candidate budgets: 1024, 4096, and 16384.
- Include base-model and shuffled-inventory controls.
- Include an observation-design diagnostic for the low-information comparison template.

Environment check:

- GPU: NVIDIA RTX 6000 Ada, 49140 MiB VRAM.
- Local model cache found at `/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B`.
- Qwen3.5-4B tokenizer and 4-bit model load succeeded.
- Installed stack included PyTorch, Transformers, PEFT, TRL, bitsandbytes, and Accelerate.

Initial implementation:

- Added a standalone 512-operator library with arbitrary aliases `op_000` through `op_511`.
- Added terse inventory descriptions so aliases cannot be solved by name alone.
- Added two templates:
  - `pair_affine_mod`
  - `pair_compare_gate`
- Added slot prompts where the model predicts the three digits after `op_` for either LEFT or RIGHT.
- Added QLoRA training script.
- Added constrained digit-beam evaluator.
- Added shuffled-inventory control by permuting descriptions relative to aliases at evaluation time.
- Added observation-design diagnostic using six max-split probes on comparison records.

## Smoke Validation

Commands:

```bash
python -m py_compile scripts/*.py src/*.py
python scripts/build_dataset.py --train-records 16 --eval-records 8 > run_logs/dataset_smoke_console.log 2>&1
python scripts/train_shortlister.py \
  --train data/train_slots.jsonl \
  --output-dir /workspace/large_artifacts/qwen35_4b_inventory_shortlister_training/models/_smoke_lora \
  --max-steps 1 \
  --grad-accum 1 \
  --limit 4 \
  > run_logs/train_smoke_console.log 2>&1
```

Result:

- Syntax check passed.
- Smoke dataset generation passed.
- One-step QLoRA smoke train passed and saved an adapter externally.
- Initial verbose inventory produced prompts around 7650 tokens and one optimizer step took about 30 seconds.

Iteration:

- Compressed inventory descriptions to terse formulas.
- Listed three-digit codes in the inventory instead of repeating `op_` on every line.
- Rebuilt the full dataset.
- New prompt length on sampled slot examples: min 4871, max 4910, average 4889.5 tokens.
- Set max sequence length to 5120.

Second smoke:

```bash
python scripts/train_shortlister.py \
  --train data/train_slots.jsonl \
  --output-dir /workspace/large_artifacts/qwen35_4b_inventory_shortlister_training/models/_smoke_lora_compact \
  --max-steps 1 \
  --limit 4 \
  > run_logs/train_smoke_compact_console.log 2>&1
```

Result:

- Compact one-step train passed.
- One compact optimizer step took about 27 seconds including model load and save overhead.
- Temporary smoke adapters were removed after the final run.

## Full Dataset

Command:

```bash
python scripts/build_dataset.py > run_logs/dataset_build_console.log 2>&1
```

Dataset:

- 768 train records.
- 96 eval records.
- 1536 train slot examples.
- 192 eval slot examples.
- 512 same-signature `list[int] -> int` operators.
- 6 visible cases, 18 hidden cases, and 48 query-pool cases per record.
- Train/eval templates are balanced across `pair_affine_mod` and `pair_compare_gate`.

## QLoRA Training

Command:

```bash
python scripts/train_shortlister.py \
  --train data/train_slots.jsonl \
  --output-dir /workspace/large_artifacts/qwen35_4b_inventory_shortlister_training/models/qwen35_lora \
  --max-steps 80 \
  > run_logs/train_shortlister_console.log 2>&1
```

Training configuration:

- Base model: Qwen3.5-4B from local cache.
- Quantization: 4-bit NF4 with bfloat16 compute.
- LoRA rank: 16.
- LoRA alpha: 32.
- Trainable parameters: 21,233,664.
- Max sequence length: 5120.
- Batch size: 1.
- Gradient accumulation: 1.
- Optimizer steps: 80.
- Train examples seen: 80 slot examples.

Loss observations:

- Step 10: 1.6914.
- Step 20: 1.6936.
- Step 30: 2.2485.
- Step 40: 1.6932.
- Step 50: 1.8529.
- Final step 80: 1.6272.

Interpretation:

- This is a real Qwen3.5-4B LoRA training run, but intentionally pilot-scale.
- The loss did not show a strong monotonic drop, so evaluation was expected to be difficult.

## Evaluation

Initial beam attempts:

- Beam 128 failed on the base-model pass with CUDA out-of-memory from generation cache expansion.
- Beam 64 failed on the base-model pass with a PyTorch 32-bit indexing limit in the model convolution path.
- Beam 32 ran successfully.

Completed command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/eval_shortlister.py \
  --adapter-dir /workspace/large_artifacts/qwen35_4b_inventory_shortlister_training/models/qwen35_lora \
  --max-records 16 \
  --probe-limit 16 \
  --beam-width 32 \
  --output reports/shortlister_results.json \
  > run_logs/eval_shortlister_console.log 2>&1
```

Scope:

- 16 eval records.
- 9 `pair_affine_mod` records.
- 7 `pair_compare_gate` records.
- Controls:
  - base model
  - trained model
  - trained model with shuffled inventory descriptions
- Exact fixed candidate budget measured: 1024 candidate pairs, from top-32 LEFT x top-32 RIGHT.
- Larger budget columns in result files are lower bounds from the same beam-32 outputs, not full top-64/top-128 evaluations.

Shortlister result:

| control | records | pair top-1 | exact 1024-candidate coverage |
| --- | ---: | ---: | ---: |
| base model | 16 | 0.0% | 0.0% |
| trained model | 16 | 0.0% | 0.0% |
| trained model, shuffled inventory | 16 | 0.0% | 0.0% |

Template split:

| control | template | records | exact 1024-candidate coverage |
| --- | --- | ---: | ---: |
| base model | `pair_affine_mod` | 9 | 0.0% |
| base model | `pair_compare_gate` | 7 | 0.0% |
| trained model | `pair_affine_mod` | 9 | 0.0% |
| trained model | `pair_compare_gate` | 7 | 0.0% |
| trained model, shuffled inventory | `pair_affine_mod` | 9 | 0.0% |
| trained model, shuffled inventory | `pair_compare_gate` | 7 | 0.0% |

Observation-design diagnostic on comparison records:

| observation set | avg visible-consistent candidates | selected hidden-all |
| --- | ---: | ---: |
| random six visible cases | 22396.286 | 28.6% |
| designed six max-split probes | 5656.286 | 42.9% |

Interpretation:

- The trained shortlister did not beat the base or shuffled controls at the exact 1024-candidate budget.
- The generated beams collapsed around a narrow alias range in the smoke inspection, consistent with weak task learning rather than inventory-conditioned semantic shortlisting.
- The observation-design lever produced a clear ambiguity reduction and modest selection lift.
- The next useful direction is not bigger blind beam search. It is either stronger structured supervision for the shortlister, a scoring interface that avoids beam-cache blowup, or semantic/typed search that uses examples to rank operator families before enumerating pairs.

Generated artifacts:

- Report: `reports/qwen35_4b_inventory_shortlister_training_report.md`
- Results: `reports/shortlister_results.json`
- Training losses: `reports/training_losses.json`
- CSV summaries: `reports/prediction_summary.csv`, `reports/prediction_template_summary.csv`, `reports/probe_rows.csv`
- Figures: `reports/figures/*.png`
- LoRA adapter: `/workspace/large_artifacts/qwen35_4b_inventory_shortlister_training/models/qwen35_lora`

## Final Audit

Commands/checks:

```bash
python -m py_compile scripts/*.py src/*.py
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -size +50M -print
du -sh . /workspace/large_artifacts/qwen35_4b_inventory_shortlister_training/models/qwen35_lora
```

Audit result:

- Final syntax check passed.
- No Python cache directories remain.
- No file larger than 50 MB is present in the experiment directory.
- Experiment directory size: 14 MB.
- Final external LoRA adapter size: 101 MB.
- A dynamic text scan against sibling experiment directory names found no references.
- No standalone-forbidden temporal references were found.
- PNG figures were opened and verified with PIL.
