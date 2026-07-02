#!/usr/bin/env bash
# Bank frontier-exceeding solutions: SFT the 4B on decompose-found (prompt->code) solutions, then eval
# monolithic depth-2/3 frozen vs trained -- does internalizing search-found depth-3 lift single-shot depth-3?
set -uo pipefail
cd /home/ericflo/Development/small-model-experimentation/experiments/qwen35_4b_decompose_compose_frontier
PY=../../.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
flt(){ grep -avE "it/s\]$|Loading|Fetching|FutureWarning|_check_is_size|triton|make_block|UserWarning|warnings.warn"; }
echo "=== TRAIN $(date +%T) ($(wc -l < data/frontier_train.jsonl) solutions) ==="
$PY scripts/train_lora.py --train data/frontier_train.jsonl --out runs/frontier_adapter --epochs 2 2>&1 | flt | grep -aE "training on|train_loss|saved"
echo "=== EVAL FROZEN $(date +%T) ==="
$PY scripts/eval_lora.py --eval-tasks data/frontier_eval.jsonl --k 5 --out runs/eval_frozen.json 2>&1 | flt | grep -aE "frozen|nothink|think_|wrote"
echo "=== EVAL TRAINED $(date +%T) ==="
$PY scripts/eval_lora.py --eval-tasks data/frontier_eval.jsonl --k 5 --adapter runs/frontier_adapter --out runs/eval_trained.json 2>&1 | flt | grep -aE "trained|nothink|think_|wrote"
rm -rf runs/frontier_adapter
echo "=== BANK_DONE $(date +%T) ==="
