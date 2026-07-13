#!/usr/bin/env bash
set -euo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_gauntlet_frontier
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
../../.venv/bin/python scripts/train_think.py \
  --train data/sft_induction.jsonl \
  --out ../../large_artifacts/qwen35_4b_gauntlet_frontier/adapters/induction1 \
  --epochs 2.0 --lr 2e-4 --rank 32 --alpha 64 \
  --batch-size 1 --grad-accum 16 --max-length 3072 --w-think 0.2 --seed 42
echo "===== induction training complete ====="
