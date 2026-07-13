#!/usr/bin/env bash
# B: train the generic multi-skill curriculum (co-train from base), then eval
# TRANSFER to the held-out menagerie (fast HF-adapter proxy, quick@1024, paired).
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation
mkdir -p experiments/qwen35_4b_universal_curriculum/runs
exec > >(tee experiments/qwen35_4b_universal_curriculum/runs/train.log) 2>&1   # self-log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ADIR=large_artifacts/qwen35_4b_universal_curriculum/adapters/universal1
echo "===== TRAIN universal curriculum ====="
.venv/bin/python experiments/qwen35_4b_gauntlet_frontier/scripts/train_think.py \
  --train experiments/qwen35_4b_universal_curriculum/data/sft_universal.jsonl \
  --out $ADIR --epochs 2.0 --rank 32 --alpha 64 --batch-size 2 --grad-accum 4 \
  --max-length 2560 --w-think 0.2 --seed 42
if ! grep -q 'saved adapter' experiments/qwen35_4b_universal_curriculum/runs/train.log 2>/dev/null && [ ! -f $ADIR/adapter_model.safetensors ]; then echo "TRAIN FAILED"; exit 1; fi
echo "===== EVAL transfer -> menagerie quick@1024 (base vs universal, paired) ====="
.venv/bin/python experiments/qwen35_4b_gauntlet_frontier/scripts/bench.py \
  --tier quick --seed 59001 --backend qwen --adapter $ADIR --arms base adapter \
  --think-budget 1024 --note "universal-curriculum transfer to menagerie"
echo "===== train_eval_chain complete ====="
