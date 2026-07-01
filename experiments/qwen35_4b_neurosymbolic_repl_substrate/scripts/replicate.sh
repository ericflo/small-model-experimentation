#!/usr/bin/env bash
# Training replication: fresh training data (seed 505), retrain, re-eval on the SAME held-out (seed 404).
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_neurosymbolic_repl_substrate
PY=../../.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
flt() { grep -avE "it/s\]$|Loading|Fetching|FutureWarning|_check_is_size|triton|make_block|UserWarning|warnings.warn"; }
cp data/train.jsonl data/train_seed202.jsonl   # preserve the committed seed-202 training set
echo "=== COLLECT seed 505 $(date +%T) ==="
$PY scripts/collect_solutions.py --train-depths 1 2 --per-depth 120 --k 6 --train-seed 505 2>&1 | flt | grep -aE "collected|solved|wrote"
cp data/train.jsonl data/train_seed505.jsonl
echo "seed505 pairs: $(wc -l < data/train_seed505.jsonl)"
echo "=== TRAIN adapter2 $(date +%T) ==="
$PY scripts/train_lora.py --train data/train_seed505.jsonl --out runs/lora_adapter2 --epochs 2 2>&1 | flt | grep -aE "training on|train_loss|saved"
echo "=== EVAL trained2 on held-out seed404 $(date +%T) ==="
$PY scripts/eval_lora.py --eval-tasks data/eval_tasks_big.jsonl --k 5 --adapter runs/lora_adapter2 --out runs/eval_trained2_big.json 2>&1 | flt | grep -aE "trained|think_|wrote"
cp data/train_seed202.jsonl data/train.jsonl   # restore committed training set
rm -rf runs/lora_adapter2
echo "=== REPLICATE_DONE $(date +%T) ==="
