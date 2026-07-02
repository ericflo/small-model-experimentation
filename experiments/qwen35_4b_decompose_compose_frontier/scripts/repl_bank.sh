#!/usr/bin/env bash
# Banking replication: fresh harvest seed 999 -> retrain -> re-eval on the SAME held-out (frontier_eval).
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_decompose_compose_frontier
PY=../../.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
flt(){ grep -avE "it/s\]$|Loading|Fetching|FutureWarning|_check_is_size|triton|make_block|UserWarning|warnings.warn"; }
echo "=== HARVEST seed 999 $(date +%T) ==="
$PY scripts/harvest.py --seed 999 --out data/frontier_train2.jsonl 2>&1 | flt | grep -aE "harvested"
echo "=== TRAIN2 $(date +%T) ==="
$PY scripts/train_lora.py --train data/frontier_train2.jsonl --out runs/frontier_adapter2 --epochs 2 2>&1 | flt | grep -aE "training on|train_loss|saved"
echo "=== EVAL TRAINED2 $(date +%T) ==="
$PY scripts/eval_lora.py --eval-tasks data/frontier_eval.jsonl --k 5 --adapter runs/frontier_adapter2 --out runs/eval_trained2.json 2>&1 | flt | grep -aE "trained|nothink|think_|wrote"
rm -rf runs/frontier_adapter2
echo "=== REPL_BANK_DONE $(date +%T) ==="
